from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.config import namespace_from_config
from training.molnextr_markush.src.markush_layout_labels import markush_layout_cells
from utils.MolNexTR.components import Encoder
from utils.MolNexTR.dataset import TrainDataset, bms_collate
from training.molnextr_markush.tools.calibrate_markush_layout_expert import (
    DEFAULT_THRESHOLDS,
    build_report as build_markush_confidence_report,
    parse_thresholds,
)

csv.field_size_limit(sys.maxsize)


IGNORE_INDEX = -100
BOX_IGNORE = -1.0
COUNT_BUCKETS = ["0", "1", "2", "3-4", "5-8", "9+"]
POSITIVE_COUNT_BUCKETS = COUNT_BUCKETS[1:]
QUERY_CLASS_NAMES = ["variable", "no_object"]
NO_OBJECT_CLASS_WEIGHT = 0.1
MATCH_COST_CLASS = 1.0
MATCH_COST_BBOX = 5.0
QUERY_CLASS_LOSS_WEIGHT = 1.0
BBOX_LOSS_WEIGHT = 5.0
CARDINALITY_LOSS_WEIGHT = 0.25
COUNT_BUCKET_LOSS_WEIGHT = 1.0
DEFAULT_COUNT_CORN_LOSS_WEIGHT = 0.0
DEFAULT_NEGATIVE_COUNT_ZERO_LOSS_WEIGHT = 0.5
RISK_LOSS_WEIGHT = 0.5
DEFAULT_CLASS_BALANCE_BETA = 0.999
DEFAULT_MAX_CLASS_WEIGHT = 20.0
RISK_TARGET_MODES = ["sidecar_emission_quality", "accept_correct"]


def parameter_count(module: nn.Module, *, trainable: bool | None = None) -> int:
    total = 0
    for parameter in module.parameters():
        if trainable is None or bool(parameter.requires_grad) is bool(trainable):
            total += int(parameter.numel())
    return total


def cuda_peak_memory_gb() -> list[float]:
    if not torch.cuda.is_available():
        return []
    values: list[float] = []
    for device_index in range(torch.cuda.device_count()):
        values.append(float(torch.cuda.max_memory_allocated(device_index) / (1024**3)))
    return values


def build_model_scale_report(
    *,
    encoder: nn.Module,
    expert: nn.Module,
    input_dim: int,
    input_size: int,
    args: argparse.Namespace,
    training_stage: str,
    debug_only: bool,
    ddp: dict[str, Any],
    amp_enabled: bool,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    encoder_total = parameter_count(encoder)
    expert_total = parameter_count(expert)
    trainable = parameter_count(encoder, trainable=True) + parameter_count(expert, trainable=True)
    total = encoder_total + expert_total
    frozen = total - trainable
    scale_blockers: list[str] = []
    if int(args.hidden_dim) < 256:
        scale_blockers.append("hidden_dim is below the current formal expert minimum of 256")
    if int(args.expert_depth) < 4:
        scale_blockers.append("expert_depth is below the current formal expert minimum of 4")
    if int(args.expert_heads) < 8:
        scale_blockers.append("expert_heads is below the current formal expert minimum of 8")
    if int(args.layout_queries) < 16:
        scale_blockers.append("layout_queries is below the current formal Markush minimum of 16")
    if int(args.patch_size) > 16:
        scale_blockers.append("patch_size is larger than the current formal maximum of 16")
    limited_rows = any(
        int(value) > 0
        for value in [
            args.max_markush_rows,
            args.max_negative_rows,
            args.max_eval_markush_rows,
            args.max_eval_negative_rows,
        ]
    )
    if limited_rows:
        scale_blockers.append("row limits are set; this run is debug/smoke evidence only")
    if bool(args.cpu):
        scale_blockers.append("CPU run cannot be measured/formal model-scale evidence")
    if training_stage == "micro_smoke":
        scale_blockers.append("micro_smoke stage is debug evidence only")
    world_size = int(ddp.get("world_size") or 1)
    if training_stage in {"measured_smoke", "formal"} and world_size < 2:
        scale_blockers.append("dual-GPU evidence is missing for measured/formal Markush run")
    grad_accum = max(1, int(args.gradient_accumulation_steps))
    runtime = runtime if isinstance(runtime, dict) else {}
    image_resolution = [int(input_size), int(input_size)] if int(input_size) > 0 else []
    patch_size = int(args.patch_size)
    image_token_count = int((int(input_size) // patch_size) ** 2) if int(input_size) > 0 and patch_size > 0 else 0
    return {
        "schema_version": "model_scale_report_v1",
        "architecture": "markush_layout_expert",
        "training_stage": str(training_stage),
        "debug_only": bool(debug_only or limited_rows or training_stage == "micro_smoke" or bool(args.cpu)),
        "allowed_for_measured_or_formal_evidence": not bool(debug_only or scale_blockers),
        "scale_blockers": scale_blockers,
        "parameters": {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "frozen_parameters": int(frozen),
            "trainable_ratio": float(trainable / max(1, total)),
            "encoder_total_parameters": int(encoder_total),
            "encoder_trainable_parameters": int(parameter_count(encoder, trainable=True)),
            "expert_total_parameters": int(expert_total),
            "expert_trainable_parameters": int(parameter_count(expert, trainable=True)),
        },
        "backbone": {
            "name": "MolNexTR",
            "checkpoint": str(args.base_checkpoint),
            "pretrained_backbone_used": True,
            "encoder_frozen": True,
            "decoder_loaded": False,
            "complete_path": "direct_original_molnextr",
            "frozen_layer_ranges": ["encoder:all"],
            "trainable_layer_ranges": ["markush_layout_expert:all"],
        },
        "routing": {
            "expert_branch": "markush_layout",
            "complete_molecules_enter_expert": False,
            "complete_path": "direct_original_molnextr",
            "requires_explicit_router_decision": True,
        },
        "scale_minimums": {
            "hidden_dim": 256,
            "fusion_transformer_depth": 4,
            "fusion_transformer_heads": 8,
            "layout_queries": 16,
            "patch_size_max": 16,
        },
        "scale_observed": {
            "hidden_dim": int(args.hidden_dim),
            "fusion_transformer_depth": int(args.expert_depth),
            "fusion_transformer_heads": int(args.expert_heads),
            "layout_queries": int(args.layout_queries),
            "patch_size": int(args.patch_size),
            "image_resolution": image_resolution,
            "image_token_count": int(image_token_count),
            "graph_feature_dim": int(input_dim),
            "tokenizer_vocab_or_feature_dim": int(input_dim),
        },
        "expert_branch": {
            "input_dim": int(input_dim),
            "hidden_dim": int(args.hidden_dim),
            "fusion_transformer_depth": int(args.expert_depth),
            "fusion_transformer_heads": int(args.expert_heads),
            "layout_queries": int(args.layout_queries),
            "dropout": float(args.dropout),
            "patch_size": int(args.patch_size),
            "set_prediction_decoder": True,
            "count_prediction_source": str(args.count_prediction_source),
            "negative_count_zero_loss_weight": float(args.negative_count_zero_loss_weight),
            "sidecar_risk_target": str(args.risk_target),
            "sidecar_risk_loss_weight": float(args.risk_loss_weight),
        },
        "training_scale": {
            "batch_size_per_process": int(args.batch_size),
            "gradient_accumulation_steps": int(grad_accum),
            "effective_batch_size": int(args.batch_size) * int(world_size) * int(grad_accum),
            "epochs": int(args.epochs),
            "mixed_precision": bool(amp_enabled),
            "ddp": bool(ddp.get("enabled")),
            "world_size": int(world_size),
            "gpu_count_requested": int(world_size) if not bool(args.cpu) else 0,
            "gpu_count_observed": int(torch.cuda.device_count()) if torch.cuda.is_available() and not bool(args.cpu) else 0,
            "distributed_strategy": "ddp" if bool(ddp.get("enabled")) else "single_process",
            "per_gpu_peak_memory_gb": runtime.get("per_gpu_peak_memory_gb", []),
            "throughput_samples_per_second": runtime.get("throughput_samples_per_second"),
            "single_gpu_rationale": "",
        },
    }


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR | None:
    if scheduler_name == "none":
        return None
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress_denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(progress_denominator)))
        if scheduler_name == "linear":
            return max(0.0, 1.0 - progress)
        if scheduler_name == "cosine":
            return 0.5 * (1.0 + float(np.cos(np.pi * progress)))
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def require_measured_runtime(args: argparse.Namespace, ddp: dict[str, Any]) -> None:
    if str(args.training_stage) not in {"measured_smoke", "formal"}:
        return
    if bool(args.cpu):
        raise ValueError("--cpu is allowed only for debug/research runs, not measured_smoke or formal training.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for measured_smoke and formal Markush expert training.")
    if int(ddp.get("world_size") or 1) < 2:
        raise ValueError(
            "measured_smoke and formal Markush expert training require torchrun/DDP with at least two GPUs."
        )
    if int(torch.cuda.device_count()) < 2:
        raise ValueError("measured_smoke and formal Markush expert training require two visible CUDA devices.")


def selected_threshold_score(confidence_report: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Rank calibration results for checkpoint selection without weakening gates.

    Deployable thresholds always outrank non-deployable curve points. For
    non-deployable runs, positive task quality is ranked before coverage so
    early stopping does not chase thresholds that accept many wrong positives.
    """
    selected = confidence_report.get("selected_threshold")
    if isinstance(selected, dict):
        score = (
            10_000.0
            + 100.0 * float(selected.get("positive_layout_precision") or 0.0)
            + 10.0 * float(selected.get("positive_layout_recall") or 0.0)
            + float(selected.get("coverage") or 0.0)
            - 1000.0 * float(selected.get("negative_accepted") or 0.0)
        )
        return score, {
            "kind": "deployable_selected_threshold",
            "selected_threshold": selected,
            "score": score,
        }
    curve = confidence_report.get("threshold_curve")
    if not isinstance(curve, list) or not curve:
        return -1.0, {"kind": "missing_threshold_curve", "score": -1.0}
    zero_negative = [row for row in curve if int(row.get("negative_accepted") or 0) == 0]
    candidates = zero_negative if zero_negative else curve
    best = max(
        candidates,
        key=lambda row: (
            float(row.get("positive_layout_precision") or 0.0),
            float(row.get("positive_layout_recall") or 0.0),
            float(row.get("coverage") or 0.0),
            -float(row.get("negative_accepted") or 0.0),
        ),
    )
    score = (
        100.0 * float(best.get("positive_layout_precision") or 0.0)
        + 10.0 * float(best.get("positive_layout_recall") or 0.0)
        + 0.01 * float(best.get("coverage") or 0.0)
        - 1000.0 * float(best.get("negative_accepted") or 0.0)
    )
    return score, {
        "kind": "best_non_deployable_threshold_curve_point",
        "selection_order": [
            "positive_layout_precision",
            "positive_layout_recall",
            "coverage",
            "negative_accepted",
        ],
        "zero_negative_required": bool(zero_negative),
        "threshold": best,
        "score": score,
    }


def save_markush_checkpoint(
    path: Path,
    *,
    markush_expert: MarkushLayoutExpert,
    config: dict[str, Any],
    input_dim: int,
    epoch: int,
    calibration_score: float | None,
) -> None:
    torch.save(
        {
            "model_state": markush_expert.state_dict(),
            "config": config,
            "input_dim": input_dim,
            "epoch": int(epoch),
            "calibration_score": calibration_score,
            "count_buckets": COUNT_BUCKETS,
            "positive_count_buckets": POSITIVE_COUNT_BUCKETS,
            "count_policy": "query_evidence_nominal_bucket_with_optional_positive_corn_ordinal_auxiliary",
            "query_classes": QUERY_CLASS_NAMES,
            "no_object_class_weight": NO_OBJECT_CLASS_WEIGHT,
        },
        path,
    )


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layer_dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(layer_dims[index], layer_dims[index + 1]) for index in range(num_layers)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index < len(self.layers) - 1:
                value = nn.functional.relu(value)
        return value


def load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_context() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed Markush expert training requires CUDA for one-process-per-GPU DDP.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=2))
    return {"enabled": enabled, "world_size": world_size, "rank": rank, "local_rank": local_rank}


def is_main_process(ddp: dict[str, Any]) -> bool:
    return int(ddp.get("rank", 0)) == 0


def cleanup_distributed(ddp: dict[str, Any]) -> None:
    if ddp.get("enabled") and dist.is_initialized():
        dist.destroy_process_group()


def ddp_barrier(ddp: dict[str, Any]) -> None:
    if ddp.get("enabled") and dist.is_initialized():
        dist.barrier(device_ids=[int(ddp.get("local_rank", 0))])


def reduce_mean(value: torch.Tensor, ddp: dict[str, Any]) -> torch.Tensor:
    if ddp.get("enabled") and dist.is_initialized():
        value = value.detach().clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= int(ddp["world_size"])
    return value


def all_gather_float(value: float, ddp: dict[str, Any], device: torch.device) -> list[float]:
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if ddp.get("enabled") and dist.is_initialized():
        gathered = [torch.zeros_like(tensor) for _ in range(int(ddp["world_size"]))]
        dist.all_gather(gathered, tensor)
        return [float(item.detach().cpu().item()) for item in gathered]
    return [float(value)]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        raw = str(row.get("file_path") or row.get("image_path") or "").strip()
        if not raw:
            continue
        image_path = Path(raw)
        if not image_path.is_absolute():
            row["file_path"] = str((csv_path.parent / image_path).resolve())
    return rows


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_render_quality(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("render_quality") or "").strip()
    if not text:
        return {}
    value = json.loads(text)
    return value if isinstance(value, dict) else {}


def count_bucket(value: int) -> int:
    if value <= 0:
        return 0
    if value == 1:
        return 1
    if value == 2:
        return 2
    if value <= 4:
        return 3
    if value <= 8:
        return 4
    return 5


def markush_cells(row: dict[str, Any]) -> list[dict[str, Any]]:
    return markush_layout_cells(row)


def load_frozen_encoder(base_checkpoint: str, config: dict[str, Any], device: torch.device) -> tuple[nn.Module, Any]:
    checkpoint = torch.load(base_checkpoint, map_location="cpu")
    args = namespace_from_config(
        {
            "output_dir": str(config["output_dir"]),
            "formats": [],
            "augment": bool(config.get("augment", True)),
            "coords_file": None,
            "predict_coords": False,
            "data_path": "",
        },
        checkpoint.get("args", {}),
    )
    args.save_path = str(config["output_dir"])
    args.data_path = ""
    encoder = Encoder(args, pretrained=False)
    state = {key.replace("module.", ""): value for key, value in checkpoint["encoder"].items()}
    encoder.load_state_dict(state, strict=False)
    return encoder.to(device), args


class MarkushLayoutExpert(nn.Module):
    """Frozen-encoder Markush layout expert that fuses image, OCSR, and layout-query evidence."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        dropout: float,
        depth: int,
        num_heads: int,
        patch_size: int,
        layout_queries: int,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.layout_queries = int(layout_queries)
        self.image_patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.image_token_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.molnextr_token_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.layout_query_embed = nn.Embedding(self.layout_queries, hidden_dim)
        self.molnextr_projection = nn.Sequential(
            nn.LayerNorm(input_dim + 2),
            nn.Linear(input_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layout_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.markush_presence = nn.Linear(hidden_dim, 2)
        self.query_class_embed = nn.Linear(hidden_dim, len(QUERY_CLASS_NAMES))
        self.layout_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        count_risk_dim = hidden_dim * 2 + 4 + (2 * self.layout_queries)
        self.count_norm = nn.LayerNorm(count_risk_dim)
        self.variable_count_bucket = MLP(count_risk_dim, hidden_dim, len(COUNT_BUCKETS), 3)
        self.variable_count_corn = MLP(count_risk_dim, hidden_dim, len(POSITIVE_COUNT_BUCKETS) - 1, 3)
        self.sidecar_risk_norm = nn.LayerNorm(count_risk_dim)
        self.sidecar_risk_embed = MLP(count_risk_dim, hidden_dim, 2, 3)

    def _tokens_with_coords(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            if features.shape[-1] == self.input_dim:
                spatial = features
            elif features.shape[1] == self.input_dim:
                spatial = features.permute(0, 2, 3, 1).contiguous()
            else:
                spatial = features
            batch, height, width, channels = spatial.shape
            tokens = spatial.reshape(batch, height * width, channels)
            y_coords = torch.linspace(0.0, 1.0, height, device=features.device, dtype=features.dtype)
            x_coords = torch.linspace(0.0, 1.0, width, device=features.device, dtype=features.dtype)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
            coords = torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2).expand(batch, -1, -1)
        elif features.ndim == 3:
            tokens = features if features.shape[-1] == self.input_dim else features.permute(0, 2, 1).contiguous()
            coords = torch.zeros(tokens.shape[0], tokens.shape[1], 2, device=features.device, dtype=features.dtype)
        else:
            tokens = features.flatten(start_dim=1).unsqueeze(1)
            coords = torch.zeros(tokens.shape[0], tokens.shape[1], 2, device=features.device, dtype=features.dtype)
        return torch.cat([tokens, coords], dim=-1)

    def _image_tokens(self, images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        patches = self.image_patch_embed(images.float()).to(dtype=dtype)
        batch, channels, height, width = patches.shape
        tokens = patches.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        y_coords = torch.linspace(0.0, 1.0, height, device=images.device, dtype=dtype)
        x_coords = torch.linspace(0.0, 1.0, width, device=images.device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        centers = torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2).expand(batch, -1, -1)
        pos_embed = torch.zeros(batch, height * width, self.hidden_dim, device=images.device, dtype=dtype)
        pos_embed[..., :2] = centers
        return tokens + pos_embed + self.image_token_type.to(dtype=dtype)

    def forward(self, features: torch.Tensor, images: torch.Tensor) -> dict[str, torch.Tensor]:
        molnextr_tokens = self.molnextr_projection(self._tokens_with_coords(features))
        batch_size = molnextr_tokens.shape[0]
        dtype = molnextr_tokens.dtype
        molnextr_tokens = molnextr_tokens + self.molnextr_token_type.to(dtype=dtype)
        image_tokens = self._image_tokens(images, dtype)
        cls_tokens = self.cls_token.to(dtype=dtype).expand(batch_size, -1, -1)
        fused_tokens = self.fusion_encoder(torch.cat([cls_tokens, image_tokens, molnextr_tokens], dim=1))
        global_hidden = self.global_norm(fused_tokens[:, 0])
        field_tokens = fused_tokens[:, 1:]
        queries = self.layout_query_embed.weight.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        query_contexts = self.layout_decoder(tgt=queries, memory=field_tokens)
        query_contexts = self.query_norm(query_contexts + queries)
        query_logits = self.query_class_embed(query_contexts)
        variable_probs = torch.softmax(query_logits, dim=-1)[..., 0]
        weighted_query = (query_contexts * variable_probs.unsqueeze(-1)).sum(dim=1) / (
            variable_probs.sum(dim=1, keepdim=True) + 1e-6
        )
        sorted_probs = torch.sort(variable_probs, dim=1, descending=True).values
        top_prob = sorted_probs[:, 0:1]
        second_prob = sorted_probs[:, 1:2] if sorted_probs.shape[1] > 1 else top_prob.new_zeros(top_prob.shape)
        cumulative_sorted_probs = torch.cumsum(sorted_probs, dim=1) / torch.arange(
            1,
            sorted_probs.shape[1] + 1,
            device=sorted_probs.device,
            dtype=sorted_probs.dtype,
        ).unsqueeze(0)
        objectness_stats = torch.cat(
            [
                variable_probs.mean(dim=1, keepdim=True),
                variable_probs.max(dim=1, keepdim=True).values,
                variable_probs.sum(dim=1, keepdim=True) / max(1, self.layout_queries),
                top_prob - second_prob,
            ],
            dim=1,
        )
        query_count_evidence = torch.cat([objectness_stats, sorted_probs, cumulative_sorted_probs], dim=1)
        risk_features = torch.cat([global_hidden, weighted_query, query_count_evidence.to(dtype=dtype)], dim=1)
        return {
            "markush_presence": self.markush_presence(global_hidden),
            "variable_count_bucket": self.variable_count_bucket(self.count_norm(risk_features)),
            "variable_count_corn": self.variable_count_corn(self.count_norm(risk_features)),
            "query_logits": query_logits,
            "layout_box": torch.sigmoid(self.layout_box_embed(query_contexts)),
            "sidecar_risk": self.sidecar_risk_embed(self.sidecar_risk_norm(risk_features)),
        }


def markush_positive_rows(path: str, *, max_rows: int, layout_queries: int) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(path):
        if not parse_bool(row.get("reliable_training_label")):
            continue
        if str(row.get("structure_type_bucket") or "").strip() != "markush_layout":
            continue
        cells = markush_cells(row)
        if not cells:
            continue
        boxes = [cell["box"] for cell in cells[:layout_queries]]
        objectness = [1.0] * len(boxes)
        while len(boxes) < layout_queries:
            boxes.append([BOX_IGNORE, BOX_IGNORE, BOX_IGNORE, BOX_IGNORE])
            objectness.append(0.0)
        rows.append(
            {
                "file_path": str(row.get("file_path") or ""),
                "SMILES": str(row.get("SMILES") or row.get("smiles") or ""),
                "source_id": str(row.get("source_id") or ""),
                "source_arrow": str(row.get("source_arrow") or "markush_positive"),
                "markush_presence_label": 1,
                "variable_count_bucket_label": count_bucket(len(cells)),
                "layout_boxes": json.dumps(boxes),
                "layout_objectness": json.dumps(objectness),
            }
        )
    if max_rows > 0 and len(rows) > max_rows:
        rows = random.sample(rows, max_rows)
    return rows


def negative_rows(paths: list[str], *, max_rows: int, layout_queries: int) -> list[dict[str, Any]]:
    rows = []
    empty_boxes = json.dumps([[BOX_IGNORE, BOX_IGNORE, BOX_IGNORE, BOX_IGNORE] for _ in range(layout_queries)])
    empty_objectness = json.dumps([0.0 for _ in range(layout_queries)])
    for path in paths:
        for row in read_csv(path):
            file_path = str(row.get("file_path") or "")
            if not file_path:
                continue
            bucket = str(row.get("structure_type_bucket") or "").strip()
            if bucket and bucket not in {"ordinary_structure", "attachment_fragment"}:
                continue
            rows.append(
                {
                    "file_path": file_path,
                    "SMILES": str(row.get("SMILES") or row.get("smiles") or ""),
                    "source_id": str(row.get("source_id") or ""),
                    "source_arrow": str(row.get("source_arrow") or "markush_negative"),
                    "markush_presence_label": 0,
                    "variable_count_bucket_label": 0,
                    "layout_boxes": empty_boxes,
                    "layout_objectness": empty_objectness,
                }
            )
    if max_rows > 0 and len(rows) > max_rows:
        rows = random.sample(rows, max_rows)
    return rows


def make_frame(
    *,
    markush_csv: str,
    negative_csvs: list[str],
    max_markush_rows: int,
    max_negative_rows: int,
    layout_queries: int,
) -> pd.DataFrame:
    rows = markush_positive_rows(markush_csv, max_rows=max_markush_rows, layout_queries=layout_queries)
    rows.extend(negative_rows(negative_csvs, max_rows=max_negative_rows, layout_queries=layout_queries))
    random.shuffle(rows)
    return pd.DataFrame(rows).reset_index(drop=True)


def audit_frame_images(df: pd.DataFrame, *, name: str, max_rows: int) -> dict[str, Any]:
    missing = []
    unreadable = []
    paths = [str(value) for value in df["file_path"].tolist()]
    sample_paths = paths if max_rows <= 0 else paths[:max_rows]
    for path_text in sample_paths:
        path = Path(path_text)
        if not path.is_absolute():
            raise ValueError(f"{name} image path is not absolute: {path_text}")
        if not path.exists():
            missing.append(path_text)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            unreadable.append({"path": path_text, "error": str(exc)})
    if missing or unreadable:
        raise ValueError(f"{name} image audit failed: missing={missing[:10]} unreadable={unreadable[:10]}")
    return {
        "name": name,
        "rows": int(len(paths)),
        "checked_rows": int(len(sample_paths)),
        "all_paths_absolute": True,
        "missing": 0,
        "unreadable": 0,
    }


def make_loader(
    model_args: Any,
    df: pd.DataFrame,
    *,
    batch_size: int,
    num_workers: int,
    weighted: bool,
    ddp: dict[str, Any] | None = None,
) -> DataLoader:
    dataset = TrainDataset(model_args, df, tokenizer={}, split="train", dynamic_indigo=False)
    sampler = None
    shuffle = True
    ddp = ddp or {"enabled": False, "rank": 0, "world_size": 1}
    if bool(ddp.get("enabled")):
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(ddp["world_size"]),
            rank=int(ddp["rank"]),
            shuffle=True,
            drop_last=False,
        )
        shuffle = False
    elif weighted:
        labels = [int(value) for value in df["markush_presence_label"].tolist()]
        counts = Counter(labels)
        total = float(sum(counts.values()))
        weights = [total / max(1.0, float(len(counts) * counts[int(label)])) for label in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=bms_collate,
        pin_memory=torch.cuda.is_available(),
    )


def labels_for(df: pd.DataFrame, row_ids: list[int], column: str, device: torch.device) -> torch.Tensor:
    return torch.tensor([int(df.iloc[int(row_id)][column]) for row_id in row_ids], dtype=torch.long, device=device)


def layout_targets_for(df: pd.DataFrame, row_ids: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    boxes = []
    objectness = []
    for row_id in row_ids:
        row = df.iloc[int(row_id)]
        boxes.append(json.loads(str(row["layout_boxes"])))
        objectness.append(json.loads(str(row["layout_objectness"])))
    return (
        torch.tensor(boxes, dtype=torch.float32, device=device),
        torch.tensor(objectness, dtype=torch.float32, device=device),
    )


def hungarian_box_match(cost: torch.Tensor) -> list[tuple[int, int]]:
    if cost.numel() == 0:
        return []
    query_indices, target_indices = linear_sum_assignment(cost.detach().cpu().numpy())
    return [(int(query_index), int(target_index)) for query_index, target_index in zip(query_indices, target_indices)]


def bucket_for_query_count(count: int) -> int:
    return count_bucket(int(count))


def query_variable_probs(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.softmax(outputs["query_logits"], dim=-1)[..., 0]


def query_count_bucket_and_confidence(variable_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_counts = (variable_probs >= 0.5).sum(dim=1).long()
    predicted_buckets = torch.tensor(
        [bucket_for_query_count(int(value)) for value in predicted_counts.detach().cpu().tolist()],
        dtype=torch.long,
        device=variable_probs.device,
    )
    if variable_probs.shape[1] == 0:
        margins = variable_probs.new_zeros((variable_probs.shape[0],))
    else:
        margins = torch.abs(variable_probs - 0.5).min(dim=1).values.clamp(0.0, 0.5) * 2.0
    return predicted_counts, predicted_buckets, margins


def count_bucket_and_confidence_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=1)
    confidence, buckets = probs.max(dim=1)
    return buckets, confidence


def positive_count_confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    positive_probs = probs[:, 1:]
    positive_confidence, _positive_bucket = positive_probs.max(dim=1)
    return positive_confidence


def corn_count_loss(logits: torch.Tensor, target: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    if logits.shape[1] != num_classes - 1:
        raise ValueError(f"CORN logits must have {num_classes - 1} columns, got {logits.shape[1]}")
    losses = []
    weights = []
    for task_index in range(num_classes - 1):
        mask = target > (task_index - 1)
        if not torch.any(mask):
            continue
        task_target = (target[mask] > task_index).to(dtype=logits.dtype)
        task_loss = nn.functional.binary_cross_entropy_with_logits(
            logits[mask, task_index],
            task_target,
            reduction="sum",
        )
        losses.append(task_loss)
        weights.append(int(task_target.numel()))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).sum() / max(1, sum(weights))


def corn_count_probabilities(logits: torch.Tensor) -> torch.Tensor:
    conditional = torch.sigmoid(logits)
    batch_size = conditional.shape[0]
    num_classes = conditional.shape[1] + 1
    probs = []
    previous = torch.ones(batch_size, device=logits.device, dtype=logits.dtype)
    for class_index in range(num_classes - 1):
        current = conditional[:, class_index]
        probs.append(previous * (1.0 - current))
        previous = previous * current
    probs.append(previous)
    return torch.stack(probs, dim=1).clamp_min(0.0)


def count_bucket_and_confidence_from_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    source: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    nominal_probs = torch.softmax(outputs["variable_count_bucket"], dim=1)
    nominal_confidence, nominal_buckets = nominal_probs.max(dim=1)
    nominal_positive_probs = nominal_probs[:, 1:]
    nominal_positive_mass = nominal_positive_probs.sum(dim=1).clamp(1e-6, 1.0)
    nominal_positive_normalized = nominal_positive_probs / nominal_positive_mass.unsqueeze(1)
    nominal_positive_confidence, nominal_positive_index = nominal_positive_probs.max(dim=1)
    nominal_positive_buckets = nominal_positive_index + 1

    corn_probs = corn_count_probabilities(outputs["variable_count_corn"])
    corn_confidence, corn_positive_index = corn_probs.max(dim=1)
    corn_positive_buckets = corn_positive_index + 1

    if source == "bucket":
        buckets = nominal_buckets
        confidence = torch.where(
            buckets == 0,
            nominal_confidence,
            nominal_positive_confidence,
        )
    elif source == "corn":
        buckets = torch.where(nominal_buckets == 0, nominal_buckets, corn_positive_buckets)
        confidence = torch.where(
            buckets == 0,
            nominal_confidence,
            corn_confidence * nominal_positive_mass,
        )
    elif source == "hybrid":
        hybrid_positive_probs = 0.5 * (nominal_positive_normalized + corn_probs)
        hybrid_confidence, hybrid_positive_index = hybrid_positive_probs.max(dim=1)
        hybrid_positive_buckets = hybrid_positive_index + 1
        buckets = torch.where(nominal_buckets == 0, nominal_buckets, hybrid_positive_buckets)
        confidence = torch.where(
            buckets == 0,
            nominal_confidence,
            hybrid_confidence * nominal_positive_mass,
        )
    else:
        raise ValueError(f"Unsupported count prediction source: {source}")
    return buckets, confidence, {
        "nominal_buckets": nominal_buckets,
        "nominal_positive_buckets": nominal_positive_buckets,
        "nominal_positive_confidence": nominal_positive_confidence,
        "corn_positive_buckets": corn_positive_buckets,
        "corn_confidence": corn_confidence,
        "nominal_positive_mass": nominal_positive_mass,
    }


def effective_class_weights(
    labels: list[int],
    num_classes: int,
    *,
    beta: float,
    max_weight: float,
    device: torch.device,
) -> torch.Tensor:
    counts = Counter(int(label) for label in labels if int(label) >= 0)
    weights = torch.ones(num_classes, dtype=torch.float32)
    if not counts:
        return weights.to(device)
    beta = min(max(float(beta), 0.0), 0.9999)
    raw: dict[int, float] = {}
    for class_id in range(num_classes):
        count = int(counts.get(class_id, 0))
        if count <= 0:
            raw[class_id] = 1.0
        elif beta == 0.0:
            raw[class_id] = 1.0 / float(count)
        else:
            raw[class_id] = (1.0 - beta) / max(1e-8, 1.0 - beta**count)
    present = [raw[class_id] for class_id in counts]
    normalizer = sum(present) / max(1, len(present))
    for class_id, weight in raw.items():
        weights[class_id] = min(float(max_weight), max(0.0, float(weight / max(1e-8, normalizer))))
    return weights.to(device)


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    ce_loss = nn.functional.cross_entropy(logits, target, weight=weight, reduction="none")
    probs = torch.softmax(logits, dim=1)
    pt = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
    return (((1.0 - pt) ** float(gamma)) * ce_loss).mean()


def matched_layout_errors(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_objectness: torch.Tensor,
) -> list[float]:
    valid_target_indices = torch.nonzero(target_objectness > 0, as_tuple=False).flatten()
    if valid_target_indices.numel() == 0:
        return []
    targets = target_boxes[valid_target_indices]
    cost = torch.cdist(pred_boxes, targets, p=1) / 4.0
    return [float(cost[q, t].detach().cpu()) for q, t in hungarian_box_match(cost)]


def set_prediction_targets(
    query_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_objectness: torch.Tensor,
    *,
    cost_class: float = MATCH_COST_CLASS,
    cost_bbox: float = MATCH_COST_BBOX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, query_count, _ = pred_boxes.shape
    no_object_class = 1
    class_target = torch.full(
        (batch_size, query_count),
        no_object_class,
        device=pred_boxes.device,
        dtype=torch.long,
    )
    matched_pred_boxes = []
    matched_target_boxes = []
    matched_indices = []
    for batch_index in range(batch_size):
        valid_target_indices = torch.nonzero(target_objectness[batch_index] > 0, as_tuple=False).flatten()
        if valid_target_indices.numel() == 0:
            continue
        targets = target_boxes[batch_index, valid_target_indices]
        variable_prob = torch.softmax(query_logits[batch_index], dim=-1)[:, 0]
        class_cost = -variable_prob[:, None].expand(-1, targets.shape[0])
        box_cost = torch.cdist(pred_boxes[batch_index], targets, p=1)
        cost = float(cost_class) * class_cost + float(cost_bbox) * box_cost
        for query_index, target_local_index in hungarian_box_match(cost):
            class_target[batch_index, query_index] = 0
            matched_pred_boxes.append(pred_boxes[batch_index, query_index])
            matched_target_boxes.append(targets[target_local_index])
            matched_indices.append((batch_index, query_index, int(valid_target_indices[target_local_index])))
    if matched_pred_boxes:
        return (
            class_target,
            torch.stack(matched_pred_boxes, dim=0),
            torch.stack(matched_target_boxes, dim=0),
            torch.tensor(matched_indices, dtype=torch.long, device=pred_boxes.device),
        )
    empty = pred_boxes.new_zeros((0, 4))
    empty_indices = torch.empty((0, 3), dtype=torch.long, device=pred_boxes.device)
    return class_target, empty, empty, empty_indices


def loss_for(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    class_weight_tensors: dict[str, torch.Tensor],
    *,
    presence_focal_gamma: float,
    count_focal_gamma: float,
    count_corn_loss_weight: float,
    negative_count_zero_loss_weight: float,
    count_prediction_source: str,
    risk_focal_gamma: float,
    risk_target: str,
    risk_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    presence_loss = focal_cross_entropy(
        outputs["markush_presence"],
        labels["markush_presence"],
        weight=class_weight_tensors.get("markush_presence"),
        gamma=float(presence_focal_gamma),
    )
    positive_mask = labels["markush_presence"] == 1
    negative_mask = labels["markush_presence"] == 0
    if torch.any(positive_mask):
        count_bucket_loss = focal_cross_entropy(
            outputs["variable_count_bucket"][positive_mask],
            labels["variable_count_bucket"][positive_mask],
            weight=class_weight_tensors.get("variable_count_bucket"),
            gamma=float(count_focal_gamma),
        )
        count_corn_loss = corn_count_loss(
            outputs["variable_count_corn"][positive_mask],
            labels["variable_count_bucket"][positive_mask] - 1,
            num_classes=len(POSITIVE_COUNT_BUCKETS),
        )
    else:
        count_bucket_loss = outputs["variable_count_bucket"].sum() * 0.0
        count_corn_loss = outputs["variable_count_corn"].sum() * 0.0
    if float(negative_count_zero_loss_weight) > 0.0 and torch.any(negative_mask):
        negative_count_zero_loss = focal_cross_entropy(
            outputs["variable_count_bucket"][negative_mask],
            labels["variable_count_bucket"][negative_mask],
            weight=None,
            gamma=float(count_focal_gamma),
        )
    else:
        negative_count_zero_loss = outputs["variable_count_bucket"].sum() * 0.0
    query_class_target, matched_pred_boxes, matched_target_boxes, _matched_indices = set_prediction_targets(
        outputs["query_logits"],
        outputs["layout_box"],
        labels["layout_boxes"],
        labels["layout_objectness"],
    )
    query_class_weight = outputs["query_logits"].new_tensor([1.0, NO_OBJECT_CLASS_WEIGHT])
    query_class_loss = nn.functional.cross_entropy(
        outputs["query_logits"].transpose(1, 2),
        query_class_target,
        weight=query_class_weight,
    )
    variable_probs = query_variable_probs(outputs)
    target_counts = labels["layout_objectness"].sum(dim=1)
    cardinality_loss = nn.functional.smooth_l1_loss(variable_probs.sum(dim=1), target_counts)
    if matched_pred_boxes.numel() > 0:
        box_loss = nn.functional.smooth_l1_loss(
            matched_pred_boxes,
            matched_target_boxes,
            reduction="sum",
        ) / max(1, int(matched_target_boxes.shape[0]) * 4)
    else:
        box_loss = outputs["layout_box"].sum() * 0.0
    with torch.no_grad():
        if risk_target == "sidecar_emission_quality":
            risk_labels = (labels["markush_presence"] == 1).long()
        elif risk_target == "accept_correct":
            pred_presence = outputs["markush_presence"].argmax(dim=1)
            pred_count, _count_confidence, _count_details = count_bucket_and_confidence_from_outputs(
                outputs,
                source=count_prediction_source,
            )
            layout_ok = torch.zeros_like(labels["markush_presence"], dtype=torch.bool)
            for batch_index in range(labels["markush_presence"].shape[0]):
                matched_costs = matched_layout_errors(
                    outputs["layout_box"][batch_index].detach(),
                    labels["layout_boxes"][batch_index],
                    labels["layout_objectness"][batch_index],
                )
                if not matched_costs:
                    layout_ok[batch_index] = True
                else:
                    layout_ok[batch_index] = max(matched_costs) <= 0.25
            negative_correct = (labels["markush_presence"] == 0) & (pred_presence == 0)
            positive_correct = (
                (labels["markush_presence"] == 1)
                & (pred_presence == 1)
                & (pred_count == labels["variable_count_bucket"])
                & layout_ok
            )
            risk_labels = (negative_correct | positive_correct).long()
        else:
            raise ValueError(f"Unsupported risk target: {risk_target}")
    risk_loss = focal_cross_entropy(
        outputs["sidecar_risk"],
        risk_labels,
        weight=class_weight_tensors.get("sidecar_risk"),
        gamma=float(risk_focal_gamma),
    )
    total = (
        presence_loss
        + QUERY_CLASS_LOSS_WEIGHT * query_class_loss
        + COUNT_BUCKET_LOSS_WEIGHT * count_bucket_loss
        + float(count_corn_loss_weight) * count_corn_loss
        + float(negative_count_zero_loss_weight) * negative_count_zero_loss
        + BBOX_LOSS_WEIGHT * box_loss
        + CARDINALITY_LOSS_WEIGHT * cardinality_loss
        + float(risk_loss_weight) * risk_loss
    )
    losses = {
        "markush_presence": float(presence_loss.detach().cpu()),
        "variable_count_bucket": float(count_bucket_loss.detach().cpu()),
        "variable_count_corn": float(count_corn_loss.detach().cpu()),
        "negative_count_zero": float(negative_count_zero_loss.detach().cpu()),
        "query_class": float(query_class_loss.detach().cpu()),
        "query_cardinality": float(cardinality_loss.detach().cpu()),
        "layout_box": float(box_loss.detach().cpu()),
        "matched_layout_boxes": float(matched_pred_boxes.shape[0]),
        "sidecar_risk": float(risk_loss.detach().cpu()),
        "loss_weights": {
            "query_class": QUERY_CLASS_LOSS_WEIGHT,
            "variable_count_bucket": COUNT_BUCKET_LOSS_WEIGHT,
            "variable_count_corn": float(count_corn_loss_weight),
            "negative_count_zero": float(negative_count_zero_loss_weight),
            "layout_box": BBOX_LOSS_WEIGHT,
            "query_cardinality": CARDINALITY_LOSS_WEIGHT,
            "sidecar_risk": float(risk_loss_weight),
            "class_balance_beta": DEFAULT_CLASS_BALANCE_BETA,
            "presence_focal_gamma": float(presence_focal_gamma),
            "count_focal_gamma": float(count_focal_gamma),
            "risk_focal_gamma": float(risk_focal_gamma),
            "risk_target": str(risk_target),
            "count_supervision": (
                "positive_markush_nominal_bucket_plus_optional_corn_ordinal_and_ordinary_negative_bucket_zero"
            ),
            "count_prediction_source": str(count_prediction_source),
            "count_confidence": "selected_positive_count_probability_times_nominal_positive_mass",
        },
    }
    return total, losses


@torch.no_grad()
def evaluate(
    *,
    encoder: nn.Module,
    markush_expert: MarkushLayoutExpert,
    model_args: Any,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    count_prediction_source: str,
    prediction_csv: Path | None = None,
) -> dict[str, Any]:
    encoder.eval()
    markush_expert.eval()
    loader = make_loader(model_args, df, batch_size=batch_size, num_workers=num_workers, weighted=False)
    counts = Counter()
    box_l1_sum = 0.0
    box_l1_count = 0
    prediction_rows: list[dict[str, Any]] = []
    for row_ids, images, _refs in loader:
        images = images.to(device, non_blocking=True)
        features, _ = encoder(images)
        outputs = markush_expert(features, images)
        presence_probs = torch.softmax(outputs["markush_presence"], dim=1)
        variable_probs = query_variable_probs(outputs)
        pred_query_variable_counts, pred_query_count, query_count_confidence_tensor = query_count_bucket_and_confidence(
            variable_probs
        )
        pred_count, count_confidence_tensor, count_details = count_bucket_and_confidence_from_outputs(
            outputs,
            source=count_prediction_source,
        )
        risk_probs = torch.softmax(outputs["sidecar_risk"], dim=1)
        pred_presence = outputs["markush_presence"].argmax(dim=1)
        gold_presence = labels_for(df, row_ids, "markush_presence_label", device)
        gold_count = labels_for(df, row_ids, "variable_count_bucket_label", device)
        boxes, objectness = layout_targets_for(df, row_ids, device)
        _query_class_target, matched_pred, matched_target, _matched_indices = set_prediction_targets(
            outputs["query_logits"],
            outputs["layout_box"],
            boxes,
            objectness,
        )
        if matched_pred.numel() > 0:
            box_l1_sum += float(torch.abs(matched_pred - matched_target).sum().detach().cpu())
            box_l1_count += int(matched_pred.shape[0]) * 4
        presence_conf = presence_probs[:, 1].detach().cpu().tolist()
        count_conf = count_confidence_tensor.detach().cpu().tolist()
        risk_conf = risk_probs[:, 1].detach().cpu().tolist()
        gold_objectness = objectness.detach().cpu().tolist()
        pred_presence_list = pred_presence.detach().cpu().tolist()
        pred_count_list = pred_count.detach().cpu().tolist()
        pred_nominal_count_list = count_details["nominal_buckets"].detach().cpu().tolist()
        pred_corn_count_list = count_details["corn_positive_buckets"].detach().cpu().tolist()
        nominal_positive_conf_list = count_details["nominal_positive_confidence"].detach().cpu().tolist()
        corn_conf_list = count_details["corn_confidence"].detach().cpu().tolist()
        nominal_positive_mass_list = count_details["nominal_positive_mass"].detach().cpu().tolist()
        pred_query_count_list = pred_query_count.detach().cpu().tolist()
        pred_query_variable_count_list = pred_query_variable_counts.detach().cpu().tolist()
        query_count_conf_list = query_count_confidence_tensor.detach().cpu().tolist()
        gold_presence_list = gold_presence.detach().cpu().tolist()
        gold_count_list = gold_count.detach().cpu().tolist()
        for index, row_id in enumerate(row_ids):
            source = df.iloc[int(row_id)]
            matched_costs = matched_layout_errors(
                outputs["layout_box"][index],
                boxes[index],
                objectness[index],
            )
            if matched_costs:
                box_error = max(matched_costs)
            else:
                box_error = 0.0
            count_correct = int(gold_presence_list[index] == 0 or pred_count_list[index] == gold_count_list[index])
            layout_correct = int(gold_presence_list[index] == 0 or box_error <= 0.25)
            accept_correct = int(
                (gold_presence_list[index] == 0 and pred_presence_list[index] == 0)
                or (
                    gold_presence_list[index] == 1
                    and pred_presence_list[index] == 1
                    and count_correct
                    and layout_correct
                )
            )
            sidecar_confidence = float(presence_conf[index]) * float(count_conf[index]) * float(risk_conf[index])
            prediction_rows.append(
                {
                    "row_id": int(row_id),
                    "source_id": str(source.get("source_id") or ""),
                    "source_arrow": str(source.get("source_arrow") or ""),
                    "smiles": str(source.get("SMILES") or ""),
                    "gold_markush_presence": int(gold_presence_list[index]),
                    "pred_markush_presence": int(pred_presence_list[index]),
                    "gold_count_bucket": COUNT_BUCKETS[int(gold_count_list[index])],
                    "pred_count_bucket": COUNT_BUCKETS[int(pred_count_list[index])],
                    "pred_nominal_count_bucket": COUNT_BUCKETS[int(pred_nominal_count_list[index])],
                    "pred_corn_count_bucket": COUNT_BUCKETS[int(pred_corn_count_list[index])],
                    "pred_query_count_bucket": COUNT_BUCKETS[int(pred_query_count_list[index])],
                    "pred_query_variable_count": int(pred_query_variable_count_list[index]),
                    "presence_confidence": float(presence_conf[index]),
                    "count_confidence": float(count_conf[index]),
                    "nominal_count_confidence": float(nominal_positive_conf_list[index]),
                    "corn_count_confidence": float(corn_conf_list[index]),
                    "nominal_positive_mass": float(nominal_positive_mass_list[index]),
                    "query_count_confidence": float(query_count_conf_list[index]),
                    "risk_confidence": float(risk_conf[index]),
                    "sidecar_confidence": sidecar_confidence,
                    "layout_box_l1": float(box_error),
                    "gold_variable_box_count": int(sum(1 for obj in gold_objectness[index] if float(obj) > 0.0)),
                    "accept_correct": int(accept_correct),
                }
            )
        counts["rows"] += int(gold_presence.numel())
        counts["presence_correct"] += int((pred_presence == gold_presence).sum().detach().cpu())
        counts["count_rows"] += int((gold_presence == 1).sum().detach().cpu())
        counts["count_correct"] += int(((pred_count == gold_count) & (gold_presence == 1)).sum().detach().cpu())
        counts["negative_false_accepts"] += int(((gold_presence == 0) & (pred_presence == 1)).sum().detach().cpu())
    if prediction_csv is not None:
        prediction_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prediction_rows).to_csv(prediction_csv, index=False)
    return {
        "rows": int(counts["rows"]),
        "presence_accuracy": counts["presence_correct"] / max(1, counts["rows"]),
        "positive_count_accuracy": counts["count_correct"] / max(1, counts["count_rows"]),
        "negative_false_accepts": int(counts["negative_false_accepts"]),
        "layout_box_l1": box_l1_sum / max(1, box_l1_count),
    }


def validate_training_gate(
    acceptance_report: str,
    readiness_report: str,
    formal_preflight_report: str,
    roadmap_constraints_report: str,
    training_stage: str,
) -> dict[str, Any]:
    if not acceptance_report:
        raise ValueError("--acceptance-report is required before Markush layout expert fitting.")
    roadmap_gate: dict[str, Any] = {"required": training_stage == "formal", "provided": bool(roadmap_constraints_report)}
    if roadmap_constraints_report:
        roadmap = load_json_object(roadmap_constraints_report)
        if roadmap.get("passed") is not True:
            raise ValueError(
                "Roadmap constraints report did not pass: "
                f"blockers={roadmap.get('blockers') if isinstance(roadmap.get('blockers'), list) else []}"
            )
        roadmap_policy = roadmap.get("policy") if isinstance(roadmap.get("policy"), dict) else {}
        for key in [
            "complete_molecules_stay_on_immutable_original_molnextr_path",
            "fragment_and_markush_rows_are_expert_routed_only",
            "generated_data_preservation_is_machine_checked",
            "formal_training_requires_all_gates_and_training_scale",
            "formal_training_must_remain_blocked_when_scale_or_contracts_are_missing",
        ]:
            if roadmap_policy.get(key) is not True:
                raise ValueError(f"Roadmap constraints report is missing policy.{key}=true")
        roadmap_gate = {
            "required": training_stage == "formal",
            "provided": True,
            "roadmap_constraints_report": str(roadmap_constraints_report),
            "passed": True,
        }
    elif training_stage == "formal":
        raise ValueError("--roadmap-constraints-report is required when --training-stage formal.")
    report = load_json_object(acceptance_report)
    smoke_allowed = report.get("measured_sidecar_smoke_allowed") is True
    formal_allowed = report.get("formal_training_allowed") is True
    blockers = report.get("formal_blockers") if training_stage == "formal" else report.get("smoke_blockers")
    if not isinstance(blockers, list):
        blockers = []
    required_key = "formal_training_allowed" if training_stage == "formal" else "measured_sidecar_smoke_allowed"
    allowed = formal_allowed if training_stage == "formal" else smoke_allowed
    if not allowed:
        raise ValueError(
            f"Acceptance report does not allow {training_stage}: {required_key}=false blockers={blockers}"
        )
    gates = report.get("manual_gates") if isinstance(report.get("manual_gates"), dict) else {}
    preflight: dict[str, Any] = {}
    preflight_acceptance_blockers: list[str] = []
    preflight_data_accepted = False
    if formal_preflight_report:
        preflight = load_json_object(formal_preflight_report)
        raw_acceptance_blockers = preflight.get("acceptance_blockers")
        if isinstance(raw_acceptance_blockers, list):
            preflight_acceptance_blockers = [str(item) for item in raw_acceptance_blockers]
        preflight_data_accepted = (
            preflight.get("accepted_for_formal_split") is True and not preflight_acceptance_blockers
        )
    required = [
        "markush_validation_trainable",
        "markush_pose_mapping_review_passed",
        "markush_source_leak_check_passed",
    ]
    missing = [key for key in required if gates.get(key) is not True]
    preflight_gates = preflight.get("gates") if isinstance(preflight.get("gates"), dict) else {}
    strict_visual_gate = preflight_gates.get("strict_machine_visual_acceptance")
    strict_visual_accepted = isinstance(strict_visual_gate, dict) and strict_visual_gate.get("passed") is True
    if training_stage != "formal" and gates.get("strict_machine_visual_acceptance") is True:
        strict_visual_accepted = True
    if not strict_visual_accepted and not preflight_data_accepted:
        missing.append("strict_machine_visual_acceptance_or_formal_preflight_accepted")
    markush_manifest_accepted = gates.get("markush_manifest_accepted") is True
    if not markush_manifest_accepted and not preflight_data_accepted:
        missing.append("markush_manifest_accepted_or_formal_preflight_accepted")
    if missing:
        raise ValueError(f"Markush acceptance gates are not green: {missing}")
    readiness_required = training_stage in {"measured_smoke", "formal"}
    readiness_gate: dict[str, Any] = {"required": readiness_required, "provided": bool(readiness_report)}
    if readiness_report:
        readiness = load_json_object(readiness_report)
        if training_stage == "formal":
            runtime_allowed = readiness.get("formal_training_start_allowed") is True
            runtime_blockers = readiness.get("formal_blockers")
            runtime_key = "formal_training_start_allowed"
        else:
            if "markush_measured_smoke_start_allowed" in readiness:
                runtime_allowed = readiness.get("markush_measured_smoke_start_allowed") is True
                runtime_blockers = readiness.get("markush_measured_blockers")
                runtime_key = "markush_measured_smoke_start_allowed"
            else:
                runtime_allowed = readiness.get("measured_smoke_start_allowed") is True
                runtime_blockers = readiness.get("blockers")
                runtime_key = "measured_smoke_start_allowed"
        if not isinstance(runtime_blockers, list):
            runtime_blockers = []
        if training_stage in {"measured_smoke", "formal"} and not runtime_allowed:
            raise ValueError(
                f"Readiness report does not allow {training_stage}: {runtime_key}=false blockers={runtime_blockers}"
            )
        readiness_gate = {
            "required": readiness_required,
            "provided": True,
            "readiness_report": str(readiness_report),
            "required_key": runtime_key,
            "accepted": runtime_allowed,
            "blockers": runtime_blockers,
            "formal_training_start_allowed": readiness.get("formal_training_start_allowed") is True,
            "measured_smoke_start_allowed": readiness.get("measured_smoke_start_allowed") is True,
            "markush_measured_smoke_start_allowed": readiness.get("markush_measured_smoke_start_allowed") is True,
            "fragment_measured_smoke_start_allowed": readiness.get("fragment_measured_smoke_start_allowed")
            is True,
        }
    elif readiness_required:
        raise ValueError("--readiness-report is required when --training-stage measured_smoke or formal.")
    preflight_gate: dict[str, Any] = {"required": training_stage == "formal", "provided": bool(formal_preflight_report)}
    if formal_preflight_report:
        preflight_gates = preflight.get("gates") if isinstance(preflight.get("gates"), dict) else {}
        red_gates = sorted(
            name
            for name, gate in preflight_gates.items()
            if not isinstance(gate, dict) or gate.get("passed") is not True
        )
        acceptance_blockers = preflight.get("acceptance_blockers")
        training_blockers = preflight.get("training_blockers")
        if not isinstance(acceptance_blockers, list):
            acceptance_blockers = []
        if not isinstance(training_blockers, list):
            training_blockers = []
        if training_stage == "formal":
            if preflight.get("accepted_for_formal_split") is not True:
                raise ValueError(
                    "Formal preflight does not accept the split: "
                    f"accepted_for_formal_split=false blockers={acceptance_blockers}"
                )
            if preflight.get("formal_training_start_allowed") is not True:
                raise ValueError(
                    "Formal preflight does not allow expert training: "
                    f"formal_training_start_allowed=false blockers={training_blockers}"
                )
        preflight_gate = {
            "required": training_stage == "formal",
            "provided": True,
            "formal_preflight_report": str(formal_preflight_report),
            "accepted_for_formal_split": preflight.get("accepted_for_formal_split") is True,
            "formal_training_start_allowed": preflight.get("formal_training_start_allowed") is True,
            "red_gates": red_gates,
            "acceptance_blockers": acceptance_blockers,
            "training_blockers": training_blockers,
            "blocking_for_current_stage": training_stage == "formal",
            "research_or_measured_evidence_only": training_stage != "formal",
            "markush_data_acceptance_authority": (
                "formal_preflight" if preflight_data_accepted and not markush_manifest_accepted else "acceptance_report"
            ),
        }
    elif training_stage == "formal":
        raise ValueError("--formal-preflight-report is required when --training-stage formal.")
    return {
        "enforced": True,
        "accepted": True,
        "training_stage": training_stage,
        "acceptance_report": str(acceptance_report),
        "roadmap_constraints_gate": roadmap_gate,
        "readiness_gate": readiness_gate,
        "formal_preflight_gate": preflight_gate,
        "required_key": required_key,
        "measured_sidecar_smoke_allowed": smoke_allowed,
        "formal_training_allowed": formal_allowed,
        "markush_manual_gates": {
            **{key: gates.get(key) for key in required},
            "strict_machine_visual_acceptance": strict_visual_accepted,
            "markush_manifest_accepted": markush_manifest_accepted,
            "formal_preflight_accepted_for_split": preflight_data_accepted,
        },
    }


def validate_count_head_training_policy(args: argparse.Namespace) -> None:
    if str(args.count_prediction_source) in {"corn", "hybrid"} and float(args.count_corn_loss_weight) <= 0.0:
        raise ValueError(
            "count_prediction_source=corn/hybrid requires --count-corn-loss-weight > 0 so inference confidence "
            "does not depend on an untrained ordinal count head."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen-base Markush layout expert.")
    parser.add_argument("--markush-label-csv", required=True)
    parser.add_argument("--negative-csv", action="append", default=[])
    parser.add_argument("--eval-markush-label-csv", default="")
    parser.add_argument("--eval-negative-csv", action="append", default=[])
    parser.add_argument("--acceptance-report", required=True)
    parser.add_argument("--readiness-report", default="")
    parser.add_argument("--formal-preflight-report", default="")
    parser.add_argument("--roadmap-constraints-report", default="")
    parser.add_argument("--input-projection-manifest", default="")
    parser.add_argument(
        "--training-stage",
        choices=["micro_smoke", "research_only", "measured_smoke", "formal"],
        default="measured_smoke",
    )
    parser.add_argument("--base-checkpoint", default="models/molnextr_best.pth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--expert-depth", type=int, default=4)
    parser.add_argument("--expert-heads", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--layout-queries", type=int, default=16)
    parser.add_argument("--max-markush-rows", type=int, default=0)
    parser.add_argument("--max-negative-rows", type=int, default=0)
    parser.add_argument("--max-eval-markush-rows", type=int, default=0)
    parser.add_argument("--max-eval-negative-rows", type=int, default=0)
    parser.add_argument("--image-audit-max-rows", type=int, default=2048)
    parser.add_argument("--confidence-thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--confidence-ece-bins", type=int, default=15)
    parser.add_argument("--confidence-max-negative-accepted", type=int, default=0)
    parser.add_argument("--confidence-min-positive-layout-precision", type=float, default=0.95)
    parser.add_argument("--confidence-min-positive-accepted", type=int, default=10)
    parser.add_argument("--confidence-min-positive-precision-wilson-lower", type=float, default=0.70)
    parser.add_argument("--confidence-error-limit", type=int, default=24)
    parser.add_argument("--class-balance-beta", type=float, default=DEFAULT_CLASS_BALANCE_BETA)
    parser.add_argument("--max-class-weight", type=float, default=DEFAULT_MAX_CLASS_WEIGHT)
    parser.add_argument("--presence-focal-gamma", type=float, default=0.0)
    parser.add_argument("--count-focal-gamma", type=float, default=2.0)
    parser.add_argument("--count-corn-loss-weight", type=float, default=DEFAULT_COUNT_CORN_LOSS_WEIGHT)
    parser.add_argument("--negative-count-zero-loss-weight", type=float, default=DEFAULT_NEGATIVE_COUNT_ZERO_LOSS_WEIGHT)
    parser.add_argument("--count-prediction-source", choices=["bucket", "corn", "hybrid"], default="bucket")
    parser.add_argument("--risk-focal-gamma", type=float, default=0.0)
    parser.add_argument(
        "--risk-target",
        choices=RISK_TARGET_MODES,
        default="sidecar_emission_quality",
        help=(
            "Training target for sidecar_risk. sidecar_emission_quality trains ordinary negatives as "
            "non-emittable and positive Markush rows as emittable; accept_correct is kept only for ablation."
        ),
    )
    parser.add_argument("--risk-loss-weight", type=float, default=RISK_LOSS_WEIGHT)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    validate_count_head_training_policy(args)

    ddp = distributed_context()
    require_measured_runtime(args, ddp)
    seed_all(int(args.seed) + int(ddp["rank"]))
    output_dir = Path(args.output_dir)
    if is_main_process(ddp):
        output_dir.mkdir(parents=True, exist_ok=True)
    ddp_barrier(ddp)
    gate_report = validate_training_gate(
        str(args.acceptance_report),
        str(args.readiness_report or ""),
        str(args.formal_preflight_report or ""),
        str(args.roadmap_constraints_report or ""),
        str(args.training_stage),
    )
    if ddp["enabled"] and args.cpu:
        raise ValueError("--cpu cannot be combined with torchrun/DDP.")
    device = torch.device(
        f"cuda:{int(ddp['local_rank'])}"
        if ddp["enabled"]
        else ("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    )
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    config["augment"] = False
    config["input_sha256"] = {
        "markush_label_csv": file_sha256(args.markush_label_csv),
        "eval_markush_label_csv": file_sha256(args.eval_markush_label_csv or args.markush_label_csv),
        "negative_csv": [file_sha256(path) for path in args.negative_csv],
        "eval_negative_csv": [file_sha256(path) for path in (args.eval_negative_csv or args.negative_csv)],
    }
    encoder, model_args = load_frozen_encoder(str(args.base_checkpoint), config, device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()

    train_df = make_frame(
        markush_csv=str(args.markush_label_csv),
        negative_csvs=list(args.negative_csv),
        max_markush_rows=int(args.max_markush_rows),
        max_negative_rows=int(args.max_negative_rows),
        layout_queries=int(args.layout_queries),
    )
    eval_df = make_frame(
        markush_csv=str(args.eval_markush_label_csv or args.markush_label_csv),
        negative_csvs=list(args.eval_negative_csv or args.negative_csv),
        max_markush_rows=int(args.max_eval_markush_rows),
        max_negative_rows=int(args.max_eval_negative_rows),
        layout_queries=int(args.layout_queries),
    )
    if train_df.empty or eval_df.empty:
        raise ValueError("No Markush layout expert rows were loaded.")
    image_audit = None
    if is_main_process(ddp):
        image_audit = {
            "train": audit_frame_images(train_df, name="train", max_rows=int(args.image_audit_max_rows)),
            "eval": audit_frame_images(eval_df, name="eval", max_rows=int(args.image_audit_max_rows)),
        }
    ddp_barrier(ddp)

    probe_loader = make_loader(model_args, train_df.head(1), batch_size=1, num_workers=0, weighted=False)
    with torch.no_grad():
        _row_ids, images, _refs = next(iter(probe_loader))
        features, _ = encoder(images.to(device))
    input_dim = int(features.shape[-1] if features.ndim >= 3 and features.shape[-1] <= 4096 else features.shape[1])
    markush_expert = MarkushLayoutExpert(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        depth=int(args.expert_depth),
        num_heads=int(args.expert_heads),
        patch_size=int(args.patch_size),
        layout_queries=int(args.layout_queries),
    ).to(device)
    train_model: nn.Module = markush_expert
    if ddp["enabled"]:
        train_model = DistributedDataParallel(
            markush_expert,
            device_ids=[int(ddp["local_rank"])],
            output_device=int(ddp["local_rank"]),
        )
    class_weight_tensors = {
        "markush_presence": torch.ones(2, dtype=torch.float32, device=device),
        "variable_count_bucket": effective_class_weights(
            [
                int(value)
                for value, presence in zip(
                    train_df["variable_count_bucket_label"].tolist(),
                    train_df["markush_presence_label"].tolist(),
                )
                if int(presence) == 1
            ],
            len(COUNT_BUCKETS),
            beta=float(args.class_balance_beta),
            max_weight=float(args.max_class_weight),
            device=device,
        ),
        "sidecar_risk": torch.ones(2, dtype=torch.float32, device=device),
    }
    optimizer = torch.optim.AdamW(
        markush_expert.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_loader = make_loader(
        model_args,
        train_df,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        weighted=not bool(args.no_weighted_sampler),
        ddp=ddp,
    )
    grad_accum_steps = max(1, int(args.gradient_accumulation_steps))
    optimizer_steps_per_epoch = max(1, (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps)
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * int(args.epochs))
    warmup_ratio = max(0.0, float(args.warmup_ratio))
    warmup_steps = int(total_optimizer_steps * warmup_ratio)
    if warmup_ratio > 0.0 and total_optimizer_steps > 1:
        warmup_steps = max(1, warmup_steps)
    scheduler = build_lr_scheduler(
        optimizer,
        scheduler_name=str(args.scheduler),
        warmup_steps=warmup_steps,
        total_steps=total_optimizer_steps,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(device=device.type, enabled=amp_enabled)
    best_checkpoint_path = output_dir / "markush_layout_expert_best.pth"
    last_checkpoint_path = output_dir / "markush_layout_expert.pth"
    best_calibration_score: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    stop_training = False
    metrics: dict[str, Any] = {
        "training_type": "frozen_molnextr_encoder_markush_layout_expert",
        "sidecar_only": True,
        "complete_path_mutated": False,
        "encoder_trainable": False,
        "decoder_loaded": False,
        "formal_full_checkpoint_training_allowed": False,
        "gate_report": gate_report,
        "consumes_markush_split": True,
        "image_audit": image_audit,
        "debug_only": bool(args.training_stage == "micro_smoke"),
        "model_scale": build_model_scale_report(
            encoder=encoder,
            expert=markush_expert,
            input_dim=int(input_dim),
            input_size=int(getattr(model_args, "input_size", 0) or 0),
            args=args,
            training_stage=str(args.training_stage),
            debug_only=False,
            ddp=ddp,
            amp_enabled=bool(amp_enabled),
        ),
        "distributed": {
            "enabled": bool(ddp["enabled"]),
            "world_size": int(ddp["world_size"]),
            "backend": "nccl" if ddp["enabled"] else None,
            "batch_size_is_per_process": bool(ddp["enabled"]),
            "weighted_sampler_disabled_under_ddp": bool(ddp["enabled"] and not bool(args.no_weighted_sampler)),
        },
        "training_controls": {
            "optimizer": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "scheduler": str(args.scheduler),
            "warmup_ratio": float(args.warmup_ratio),
            "warmup_steps": int(warmup_steps),
            "total_optimizer_steps": int(total_optimizer_steps),
            "gradient_accumulation_steps": int(grad_accum_steps),
            "max_grad_norm": float(args.max_grad_norm),
            "amp_enabled": bool(amp_enabled),
            "early_stop_patience": int(args.early_stop_patience),
            "best_checkpoint_metric": "selected_threshold_score",
            "presence_loss": "unweighted_cross_entropy_by_default",
            "count_loss": (
                "class_balanced_focal_bucket_plus_optional_corn_ordinal_on_positive_markush_rows_"
                "and_explicit_ordinary_negative_bucket_zero_focal_loss"
            ),
            "risk_loss": "unweighted_cross_entropy_by_default",
            "class_balance_beta": float(args.class_balance_beta),
            "max_class_weight": float(args.max_class_weight),
            "presence_focal_gamma": float(args.presence_focal_gamma),
            "count_focal_gamma": float(args.count_focal_gamma),
            "count_corn_loss_weight": float(args.count_corn_loss_weight),
            "negative_count_zero_loss_weight": float(args.negative_count_zero_loss_weight),
            "count_prediction_source": str(args.count_prediction_source),
            "risk_focal_gamma": float(args.risk_focal_gamma),
            "risk_target": str(args.risk_target),
            "risk_loss_weight": float(args.risk_loss_weight),
            "sidecar_emission_policy": "pred_markush_presence=1 and pred_count_bucket!=0 before calibrated confidence threshold",
        },
        "config": config,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "train_presence_counts": {str(k): int(v) for k, v in Counter(train_df["markush_presence_label"]).items()},
        "eval_presence_counts": {str(k): int(v) for k, v in Counter(eval_df["markush_presence_label"]).items()},
        "epochs": [],
    }
    start_time = time.time()
    for epoch in range(int(args.epochs)):
        encoder.eval()
        train_model.train()
        if ddp["enabled"] and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        total_loss = 0.0
        steps = 0
        loss_parts: dict[str, Any] = {}
        progress = tqdm(
            train_loader,
            desc=f"markush-layout-expert epoch {epoch + 1}",
            leave=False,
            disable=not is_main_process(ddp),
        )
        for row_ids, images, _refs in progress:
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                features, _ = encoder(images)
            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = train_model(features, images)
                boxes, objectness = layout_targets_for(train_df, row_ids, device)
                labels = {
                    "markush_presence": labels_for(train_df, row_ids, "markush_presence_label", device),
                    "variable_count_bucket": labels_for(train_df, row_ids, "variable_count_bucket_label", device),
                    "layout_boxes": boxes,
                    "layout_objectness": objectness,
                }
                loss, loss_parts = loss_for(
                    outputs,
                    labels,
                    class_weight_tensors,
                    presence_focal_gamma=float(args.presence_focal_gamma),
                    count_focal_gamma=float(args.count_focal_gamma),
                    count_corn_loss_weight=float(args.count_corn_loss_weight),
                    negative_count_zero_loss_weight=float(args.negative_count_zero_loss_weight),
                    count_prediction_source=str(args.count_prediction_source),
                    risk_focal_gamma=float(args.risk_focal_gamma),
                    risk_target=str(args.risk_target),
                    risk_loss_weight=float(args.risk_loss_weight),
                )
                scaled_loss = loss / float(grad_accum_steps)
            if steps % grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            scaler.scale(scaled_loss).backward()
            should_step = ((steps + 1) % grad_accum_steps == 0) or ((steps + 1) == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(markush_expert.parameters(), float(args.max_grad_norm))
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_was_skipped = amp_enabled and float(scaler.get_scale()) < previous_scale
                if scheduler is not None and not optimizer_step_was_skipped:
                    scheduler.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
            if is_main_process(ddp):
                progress.set_postfix(
                    {
                        "loss": total_loss / max(1, steps),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
        loss_sum = torch.tensor([total_loss, float(steps)], dtype=torch.float64, device=device)
        if ddp["enabled"]:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        train_loss = float(loss_sum[0].detach().cpu()) / max(1.0, float(loss_sum[1].detach().cpu()))
        ddp_barrier(ddp)
        if is_main_process(ddp):
            prediction_csv = output_dir / f"eval_predictions_epoch_{epoch + 1}.csv"
            eval_report = evaluate(
                encoder=encoder,
                markush_expert=markush_expert,
                model_args=model_args,
                df=eval_df,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                count_prediction_source=str(args.count_prediction_source),
                prediction_csv=prediction_csv,
            )
            confidence_report = build_markush_confidence_report(
                predictions_csv=prediction_csv,
                thresholds=parse_thresholds(str(args.confidence_thresholds)),
                ece_bins=int(args.confidence_ece_bins),
                max_negative_accepted=int(args.confidence_max_negative_accepted),
                min_positive_layout_precision=float(args.confidence_min_positive_layout_precision),
                min_positive_accepted=int(args.confidence_min_positive_accepted),
                min_positive_precision_wilson_lower=float(args.confidence_min_positive_precision_wilson_lower),
                error_limit=int(args.confidence_error_limit),
            )
            confidence_dir = output_dir / f"confidence_epoch_{epoch + 1}"
            confidence_dir.mkdir(parents=True, exist_ok=True)
            confidence_report_path = confidence_dir / "markush_layout_confidence_report.json"
            confidence_report_path.write_text(
                json.dumps(confidence_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            calibration_score, calibration_score_detail = selected_threshold_score(confidence_report)
            improved = best_calibration_score is None or calibration_score > best_calibration_score
            if improved:
                best_calibration_score = float(calibration_score)
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                save_markush_checkpoint(
                    best_checkpoint_path,
                    markush_expert=markush_expert,
                    config=config,
                    input_dim=input_dim,
                    epoch=epoch + 1,
                    calibration_score=float(calibration_score),
                )
            else:
                epochs_without_improvement += 1
            metrics["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "last_loss_parts": loss_parts,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "eval": eval_report,
                    "prediction_csv": str(prediction_csv),
                    "confidence_report": str(confidence_report_path),
                    "calibration_score": calibration_score,
                    "calibration_score_detail": calibration_score_detail,
                    "best_checkpoint_updated": bool(improved),
                    "confidence_summary": {
                        "deployment_allowed": bool(confidence_report.get("deployment_allowed")),
                        "selected_threshold": confidence_report.get("selected_threshold"),
                        "selected_high_confidence_errors": confidence_report.get("selected_high_confidence_errors"),
                        "sidecar_confidence_brier": (
                            confidence_report.get("confidence", {})
                            .get("sidecar_confidence", {})
                            .get("brier")
                        ),
                        "sidecar_confidence_ece": (
                            confidence_report.get("confidence", {})
                            .get("sidecar_confidence", {})
                            .get("ece")
                        ),
                    },
                }
            )
            metrics["best_checkpoint"] = {
                "path": str(best_checkpoint_path),
                "epoch": best_epoch,
                "calibration_score": best_calibration_score,
            }
            if int(args.early_stop_patience) > 0 and epochs_without_improvement >= int(args.early_stop_patience):
                stop_training = True
        ddp_barrier(ddp)
        if ddp["enabled"]:
            stop_tensor = torch.tensor([1 if stop_training else 0], dtype=torch.long, device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(int(stop_tensor.item()))
        if stop_training:
            break
    elapsed_seconds = time.time() - start_time
    local_peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if device.type == "cuda"
        else 0.0
    )
    per_gpu_peak_memory_gb = all_gather_float(local_peak_memory, ddp, device)
    if is_main_process(ddp):
        metrics["runtime"] = {
            "elapsed_seconds": float(elapsed_seconds),
            "train_rows": int(len(train_df)),
            "eval_rows": int(len(eval_df)),
            "throughput_samples_per_second": float(
                (int(len(train_df)) * int(len(metrics["epochs"])) + int(len(eval_df)) * int(len(metrics["epochs"])))
                / max(1e-6, float(elapsed_seconds))
            ),
            "per_gpu_peak_memory_gb": per_gpu_peak_memory_gb,
        }
        metrics["model_scale"] = build_model_scale_report(
            encoder=encoder,
            expert=markush_expert,
            input_dim=int(input_dim),
            input_size=int(getattr(model_args, "input_size", 0) or 0),
            args=args,
            training_stage=str(args.training_stage),
            debug_only=False,
            ddp=ddp,
            amp_enabled=bool(amp_enabled),
            runtime=metrics["runtime"],
        )
        save_markush_checkpoint(
            last_checkpoint_path,
            markush_expert=markush_expert,
            config=config,
            input_dim=input_dim,
            epoch=len(metrics["epochs"]),
            calibration_score=metrics["epochs"][-1].get("calibration_score") if metrics["epochs"] else None,
        )
        metrics["checkpoint"] = str(last_checkpoint_path)
        if best_calibration_score is None:
            save_markush_checkpoint(
                best_checkpoint_path,
                markush_expert=markush_expert,
                config=config,
                input_dim=input_dim,
                epoch=len(metrics["epochs"]),
                calibration_score=None,
            )
            metrics["best_checkpoint"] = {
                "path": str(best_checkpoint_path),
                "epoch": len(metrics["epochs"]),
                "calibration_score": None,
            }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))
    ddp_barrier(ddp)
    cleanup_distributed(ddp)


if __name__ == "__main__":
    main()
