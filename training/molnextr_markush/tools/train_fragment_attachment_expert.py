from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from tqdm import tqdm
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.config import namespace_from_config
from utils.MolNexTR.components import Encoder
from utils.MolNexTR.dataset import TrainDataset, bms_collate
from training.molnextr_markush.tools.calibrate_fragment_attachment_expert import (
    DEFAULT_THRESHOLDS,
    build_report as build_confidence_report,
    parse_thresholds,
)

csv.field_size_limit(sys.maxsize)


SIDE_NAMES = ["left", "right", "top", "bottom"]
SIDE_TO_ID = {name: index for index, name in enumerate(SIDE_NAMES)}
IGNORE_INDEX = -100
POINT_IGNORE = -1.0


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


def file_sha256_one_or_many(paths: str | Path | list[str] | tuple[str, ...]) -> str | list[str]:
    if isinstance(paths, (list, tuple)):
        hashes = [file_sha256(path) for path in paths]
        return hashes[0] if len(hashes) == 1 else hashes
    return file_sha256(paths)


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


def distributed_context() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        probe_cuda_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
    return {"enabled": enabled, "world_size": world_size, "rank": rank, "local_rank": local_rank}


def probe_cuda_device(device_index: int, *, attempts: int = 5) -> torch.device:
    init_error: Exception | None = None
    for _attempt in range(max(1, int(attempts))):
        try:
            torch.cuda.set_device(int(device_index))
            torch.cuda.init()
            probe = torch.ones(1, device=f"cuda:{int(device_index)}")
            _ = float(probe.item())
            return torch.device(f"cuda:{int(device_index)}")
        except Exception as exc:
            init_error = exc
            time.sleep(1.0)
    raise RuntimeError(
        f"CUDA initialization failed on device_index={int(device_index)}; "
        "training cannot use GPU without a working CUDA context."
    ) from init_error


def select_single_process_device(args: argparse.Namespace) -> torch.device:
    if bool(args.cpu):
        return torch.device("cpu")
    try:
        return probe_cuda_device(0)
    except Exception:
        if str(args.training_stage) in {"measured_smoke", "formal"}:
            raise
        return torch.device("cpu")


def is_main_process(ddp: dict[str, Any]) -> bool:
    return int(ddp.get("rank", 0)) == 0


def cleanup_distributed(ddp: dict[str, Any]) -> None:
    if ddp.get("enabled") and dist.is_initialized():
        dist.destroy_process_group()


def ddp_barrier(ddp: dict[str, Any]) -> None:
    if ddp.get("enabled") and dist.is_initialized():
        dist.barrier(device_ids=[int(ddp.get("local_rank", 0))])


def require_measured_runtime(args: argparse.Namespace, ddp: dict[str, Any]) -> None:
    if str(args.training_stage) not in {"measured_smoke", "formal"}:
        return
    if bool(args.cpu):
        raise ValueError("--cpu is allowed only for debug/research runs, not measured_smoke or formal training.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for measured_smoke and formal fragment expert training.")
    if int(ddp.get("world_size") or 1) < 2:
        raise ValueError(
            "measured_smoke and formal fragment expert training require torchrun/DDP with at least two GPUs."
        )
    if int(torch.cuda.device_count()) < 2:
        raise ValueError("measured_smoke and formal fragment expert training require two visible CUDA devices.")


def all_gather_float(value: float, ddp: dict[str, Any], device: torch.device) -> list[float]:
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if ddp.get("enabled") and dist.is_initialized():
        gathered = [torch.zeros_like(tensor) for _ in range(int(ddp["world_size"]))]
        dist.all_gather(gathered, tensor)
        return [float(item.detach().cpu().item()) for item in gathered]
    return [float(value)]


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
    if int(args.endpoint_queries) < 4:
        scale_blockers.append("endpoint_queries is below the current formal expert minimum of 4")
    if int(args.patch_size) > 16:
        scale_blockers.append("patch_size is larger than the current formal maximum of 16")
    limited_rows = any(
        int(value) > 0
        for value in [
            args.max_positive_rows,
            args.max_negative_rows,
            args.max_eval_positive_rows,
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
        scale_blockers.append("dual-GPU evidence is missing for measured/formal fragment run")
    grad_accum = max(1, int(args.gradient_accumulation_steps))
    runtime = runtime if isinstance(runtime, dict) else {}
    image_resolution = [int(input_size), int(input_size)] if int(input_size) > 0 else []
    patch_size = int(args.patch_size)
    image_token_count = int((int(input_size) // patch_size) ** 2) if int(input_size) > 0 and patch_size > 0 else 0
    return {
        "schema_version": "model_scale_report_v1",
        "architecture": "fragment_attachment_expert",
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
            "trainable_layer_ranges": ["fragment_attachment_expert:all"],
        },
        "routing": {
            "expert_branch": "fragment_attachment",
            "complete_molecules_enter_expert": False,
            "complete_path": "direct_original_molnextr",
            "requires_explicit_router_decision": True,
        },
        "scale_minimums": {
            "hidden_dim": 256,
            "fusion_transformer_depth": 4,
            "fusion_transformer_heads": 8,
            "endpoint_queries": 4,
            "patch_size_max": 16,
        },
        "scale_observed": {
            "hidden_dim": int(args.hidden_dim),
            "fusion_transformer_depth": int(args.expert_depth),
            "fusion_transformer_heads": int(args.expert_heads),
            "endpoint_queries": int(args.endpoint_queries),
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
            "endpoint_queries": int(args.endpoint_queries),
            "dropout": float(args.dropout),
            "patch_size": int(args.patch_size),
            "learned_endpoint_query_decoder": True,
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


def validate_training_gate(
    *,
    acceptance_report: str,
    readiness_report: str,
    formal_preflight_report: str,
    roadmap_constraints_report: str,
    training_stage: str,
    allow_ungated_debug_run: bool,
) -> dict[str, Any]:
    if allow_ungated_debug_run:
        return {
            "enforced": False,
            "training_stage": training_stage,
            "accepted": False,
            "reason": "Explicit --allow-ungated-debug-run was set. This run is not valid for accepted training.",
        }
    if not acceptance_report:
        raise ValueError(
            "--acceptance-report is required. Training must prove schema/visual/leak/coverage gates passed before sidecar fitting."
        )
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
    if (
        training_stage in {"measured_smoke", "formal"}
        and report.get("policy", {}).get("positive_only_markush_measured_training") is True
    ):
        raise ValueError(
            "Fragment expert training cannot use a Markush-only capacity measured acceptance report."
        )
    required_fragment_gates = [
        "accepted_manifest_accepted",
        "visual_review_passed",
        "source_leak_check_passed",
        "attachment_role_contract_passed",
    ]
    if training_stage in {"measured_smoke", "formal"}:
        missing_fragment_gates = [key for key in required_fragment_gates if gates.get(key) is not True]
        if (
            gates.get("fragment_taxonomy_coverage_passed") is not True
            and gates.get("real_fragment_taxonomy_alignment_passed") is not True
        ):
            missing_fragment_gates.append(
                "fragment_taxonomy_coverage_passed_or_real_fragment_taxonomy_alignment_passed"
            )
        if missing_fragment_gates:
            raise ValueError(f"Fragment acceptance gates are not green: {missing_fragment_gates}")
    readiness_required = training_stage in {"measured_smoke", "formal"}
    readiness_gate: dict[str, Any] = {"required": readiness_required, "provided": bool(readiness_report)}
    if readiness_report:
        readiness = load_json_object(readiness_report)
        if training_stage == "formal":
            runtime_allowed = readiness.get("formal_training_start_allowed") is True
            runtime_blockers = readiness.get("formal_blockers")
            runtime_key = "formal_training_start_allowed"
        else:
            if "fragment_measured_smoke_start_allowed" in readiness:
                runtime_allowed = readiness.get("fragment_measured_smoke_start_allowed") is True
                runtime_blockers = readiness.get("fragment_measured_blockers")
                runtime_key = "fragment_measured_smoke_start_allowed"
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
            "fragment_measured_smoke_start_allowed": readiness.get("fragment_measured_smoke_start_allowed")
            is True,
            "markush_measured_smoke_start_allowed": readiness.get("markush_measured_smoke_start_allowed") is True,
        }
    elif readiness_required:
        raise ValueError("--readiness-report is required when --training-stage measured_smoke or formal.")
    preflight_gate: dict[str, Any] = {"required": training_stage == "formal", "provided": bool(formal_preflight_report)}
    if formal_preflight_report:
        preflight = load_json_object(formal_preflight_report)
        gates = preflight.get("gates") if isinstance(preflight.get("gates"), dict) else {}
        red_gates = sorted(name for name, gate in gates.items() if not isinstance(gate, dict) or gate.get("passed") is not True)
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
        }
    elif training_stage == "formal":
        raise ValueError("--formal-preflight-report is required when --training-stage formal.")
    return {
        "enforced": True,
        "training_stage": training_stage,
        "accepted": True,
        "acceptance_report": str(acceptance_report),
        "roadmap_constraints_gate": roadmap_gate,
        "readiness_gate": readiness_gate,
        "formal_preflight_gate": preflight_gate,
        "required_key": required_key,
        "measured_sidecar_smoke_allowed": smoke_allowed,
        "formal_training_allowed": formal_allowed,
        "counts": report.get("counts") if isinstance(report.get("counts"), dict) else {},
        "manual_gates": report.get("manual_gates") if isinstance(report.get("manual_gates"), dict) else {},
    }


def load_frozen_encoder(base_checkpoint: str, config: dict[str, Any], device: torch.device) -> tuple[nn.Module, Any]:
    checkpoint = torch.load(base_checkpoint, map_location="cpu")
    args = namespace_from_config(
        {
            "output_dir": str(config["output_dir"]),
            "formats": [],
            "augment": bool(config.get("augment", True)),
            "real_match": bool(config.get("real_match", False)),
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


def side_from_point(point: torch.Tensor) -> torch.Tensor:
    distances = torch.stack(
        [
            point[:, 0],
            1.0 - point[:, 0],
            point[:, 1],
            1.0 - point[:, 1],
        ],
        dim=1,
    )
    return distances.argmin(dim=1)


class FragmentAttachmentExpert(nn.Module):
    """Frozen-base expert for fragment attachment localization and risk scoring."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        anchor_classes: int,
        *,
        depth: int = 4,
        num_heads: int = 8,
        patch_size: int = 16,
        endpoint_queries: int = 4,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.endpoint_queries = int(endpoint_queries)
        self.image_patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.image_token_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.molnextr_token_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.endpoint_queries_param = nn.Parameter(torch.randn(1, self.endpoint_queries, hidden_dim) * 0.02)
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
        self.endpoint_decoder = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.endpoint_norm = nn.LayerNorm(hidden_dim)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.presence_trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.anchor_trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.endpoint_objectness = nn.Linear(hidden_dim, 1)
        self.no_endpoint_objectness = nn.Linear(hidden_dim, 1)
        self.endpoint_point = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.endpoint_heatmap = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.endpoint_side = nn.Linear(hidden_dim, len(SIDE_NAMES))
        self.endpoint_presence = nn.Linear(hidden_dim, 2)
        self.anchor_token = nn.Linear(hidden_dim, anchor_classes)
        self.backbone_eligibility = nn.Linear(hidden_dim, 2)
        self.sidecar_applicability = nn.Linear(hidden_dim, 2)
        self.sidecar_risk = nn.Linear(hidden_dim, 2)

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
            if features.shape[-1] == self.input_dim:
                tokens = features
            elif features.shape[1] == self.input_dim:
                tokens = features.permute(0, 2, 1).contiguous()
            else:
                tokens = features
            coords = torch.zeros(tokens.shape[0], tokens.shape[1], 2, device=features.device, dtype=features.dtype)
        else:
            tokens = features.flatten(start_dim=1).unsqueeze(1)
            coords = torch.zeros(tokens.shape[0], tokens.shape[1], 2, device=features.device, dtype=features.dtype)
        return torch.cat([tokens, coords], dim=-1)

    def _image_tokens(self, images: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self.image_patch_embed(images.float()).to(dtype=dtype)
        batch, channels, height, width = patches.shape
        tokens = patches.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        y_coords = torch.linspace(0.0, 1.0, height, device=images.device, dtype=dtype)
        x_coords = torch.linspace(0.0, 1.0, width, device=images.device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        centers = torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2).expand(batch, -1, -1)
        pos_embed = torch.zeros(batch, height * width, self.hidden_dim, device=images.device, dtype=dtype)
        pos_embed[..., :2] = centers
        return tokens + pos_embed + self.image_token_type.to(dtype=dtype), centers

    def forward(self, features: torch.Tensor, images: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        molnextr_tokens = self.molnextr_projection(self._tokens_with_coords(features))
        batch_size = molnextr_tokens.shape[0]
        dtype = molnextr_tokens.dtype
        molnextr_tokens = molnextr_tokens + self.molnextr_token_type.to(dtype=dtype)
        if images is None:
            image_tokens = torch.zeros(batch_size, 0, self.hidden_dim, device=features.device, dtype=dtype)
            image_centers = torch.zeros(batch_size, 0, 2, device=features.device, dtype=dtype)
        else:
            image_tokens, image_centers = self._image_tokens(images, dtype)
        cls_tokens = self.cls_token.to(dtype=dtype).expand(batch_size, -1, -1)
        fused_tokens = self.fusion_encoder(torch.cat([cls_tokens, image_tokens, molnextr_tokens], dim=1))
        global_hidden = self.global_norm(fused_tokens[:, 0])
        field_tokens = fused_tokens[:, 1:]
        image_field_tokens = fused_tokens[:, 1 : 1 + image_tokens.shape[1]]
        endpoint_queries = self.endpoint_queries_param.to(dtype=dtype).expand(batch_size, -1, -1)
        endpoint_contexts, _ = self.endpoint_decoder(
            query=endpoint_queries,
            key=field_tokens,
            value=field_tokens,
            need_weights=False,
        )
        endpoint_contexts = self.endpoint_norm(endpoint_contexts + endpoint_queries)
        endpoint_objectness_logits = self.endpoint_objectness(endpoint_contexts).squeeze(-1)
        no_endpoint_logit = self.no_endpoint_objectness(global_hidden)
        objectness_logits = torch.cat([endpoint_objectness_logits, no_endpoint_logit], dim=1)
        endpoint_objectness_probs = torch.softmax(endpoint_objectness_logits, dim=1)
        endpoint_hidden = (endpoint_contexts * endpoint_objectness_probs.unsqueeze(-1)).sum(dim=1)
        endpoint_query_points = torch.sigmoid(self.endpoint_point(endpoint_contexts))
        endpoint_point = (endpoint_query_points * endpoint_objectness_probs.unsqueeze(-1)).sum(dim=1)
        presence_hidden = self.presence_trunk(torch.cat([global_hidden, endpoint_contexts.max(dim=1).values], dim=1))
        anchor_hidden = self.anchor_trunk(torch.cat([global_hidden, endpoint_hidden], dim=1))
        return {
            "endpoint_presence": self.endpoint_presence(presence_hidden),
            "endpoint_side": self.endpoint_side(endpoint_hidden),
            "anchor_token": self.anchor_token(anchor_hidden),
            "backbone_eligibility": self.backbone_eligibility(anchor_hidden),
            "sidecar_applicability": self.sidecar_applicability(anchor_hidden),
            "sidecar_risk": self.sidecar_risk(anchor_hidden),
            "endpoint_objectness": objectness_logits,
            "endpoint_query_points": endpoint_query_points,
            "endpoint_point": endpoint_point,
            "endpoint_heatmap": self.endpoint_heatmap(image_field_tokens).squeeze(-1),
            "endpoint_heatmap_centers": image_centers,
        }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def normalize_anchor(value: Any) -> str:
    token = str(value or "").strip()
    return token if token else ""


def as_float_or_none(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def normalized_point_from_row(row: dict[str, Any]) -> tuple[float, float, float, str]:
    width = as_float_or_none(row.get("image_width")) or 0.0
    height = as_float_or_none(row.get("image_height")) or 0.0
    x_names = [
        "endpoint_x",
        "endpoint_center_x",
        "attachment_x",
        "attachment_center_x",
        "target_x",
        "target_center_x",
    ]
    y_names = [
        "endpoint_y",
        "endpoint_center_y",
        "attachment_y",
        "attachment_center_y",
        "target_y",
        "target_center_y",
    ]
    for x_name in x_names:
        x_value = as_float_or_none(row.get(x_name))
        if x_value is None:
            continue
        for y_name in y_names:
            y_value = as_float_or_none(row.get(y_name))
            if y_value is None:
                continue
            x = x_value / width if x_value > 1.0 and width > 1.0 else x_value
            y = y_value / height if y_value > 1.0 and height > 1.0 else y_value
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                return float(x), float(y), 1.0, f"explicit:{x_name},{y_name}"

    direction = str(row.get("attachment_direction") or row.get("endpoint_side") or "").strip().lower()
    endpoint_side = str(row.get("endpoint_side") or "").strip().lower()
    text = f"{direction}_{endpoint_side}"
    if "left" in text:
        x = 0.08
    elif "right" in text:
        x = 0.92
    else:
        x = 0.5
    if "up" in text or "top" in text:
        y = 0.12
    elif "down" in text or "bottom" in text:
        y = 0.88
    else:
        y = 0.5
    if endpoint_side in SIDE_TO_ID or any(part in text for part in ["left", "right", "up", "down", "top", "bottom"]):
        return x, y, 0.25, "weak_direction"
    return POINT_IGNORE, POINT_IGNORE, 0.0, "missing"


def sample_positive_rows(rows: list[dict[str, Any]], *, max_rows: int, mode: str) -> list[dict[str, Any]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    if mode != "side_balanced":
        return random.sample(rows, max_rows)
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[int(row["endpoint_side_label"])].append(row)
    selected: list[dict[str, Any]] = []
    quota = max(1, max_rows // max(1, len(buckets)))
    remainder: list[dict[str, Any]] = []
    for side_id in sorted(buckets):
        bucket = buckets[side_id]
        random.shuffle(bucket)
        selected.extend(bucket[:quota])
        remainder.extend(bucket[quota:])
    if len(selected) < max_rows:
        selected.extend(random.sample(remainder, min(max_rows - len(selected), len(remainder))))
    random.shuffle(selected)
    return selected[:max_rows]


def positive_rows(path: str, *, max_rows: int = 0, sampling_mode: str = "side_balanced") -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(path):
        if not parse_bool(row.get("reliable_training_label")):
            continue
        side = str(row.get("endpoint_side") or "").strip().lower()
        anchor = normalize_anchor(row.get("attachment_anchor"))
        file_path = str(row.get("file_path") or "")
        if side not in SIDE_TO_ID or not anchor or not file_path:
            continue
        point = normalized_point_from_row(row)
        rows.append(
            {
                "file_path": file_path,
                "SMILES": str(row.get("smiles") or ""),
                "source_id": str(row.get("source_id") or row.get("review_id") or ""),
                "source_arrow": str(row.get("source_arrow") or row.get("record_kind") or "endpoint_positive"),
                "endpoint_presence_label": 1,
                "sidecar_applicability_label": 1,
                "endpoint_side_label": SIDE_TO_ID[side],
                "endpoint_x_label": point[0],
                "endpoint_y_label": point[1],
                "endpoint_point_weight": point[2],
                "endpoint_point_source": point[3],
                "anchor_token_label_text": anchor,
                "backbone_eligibility_label": (
                    1
                    if parse_bool(row.get("backbone_eligible_strict"))
                    or parse_bool(row.get("backbone_eligible_largest"))
                    else IGNORE_INDEX
                ),
            }
        )
    return sample_positive_rows(rows, max_rows=max_rows, mode=sampling_mode)


def negative_rows(paths: list[str], *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for row in read_csv(path):
            file_path = str(row.get("file_path") or "")
            if not file_path:
                continue
            label = str(row.get("structure_type_label") or "").strip()
            bucket = str(row.get("structure_type_bucket") or "").strip()
            if label and label != "complete_molecule":
                continue
            if bucket and bucket != "ordinary_structure":
                continue
            rows.append(
                {
                    "file_path": file_path,
                    "SMILES": str(row.get("SMILES") or row.get("smiles") or ""),
                    "source_id": str(row.get("source_id") or ""),
                    "source_arrow": str(row.get("source_arrow") or "endpoint_negative"),
                    "endpoint_presence_label": 0,
                    "sidecar_applicability_label": 0,
                    "endpoint_side_label": IGNORE_INDEX,
                    "endpoint_x_label": POINT_IGNORE,
                    "endpoint_y_label": POINT_IGNORE,
                    "endpoint_point_weight": 0.0,
                    "endpoint_point_source": "negative",
                    "anchor_token_label_text": "",
                    "backbone_eligibility_label": IGNORE_INDEX,
                }
            )
    if max_rows > 0 and len(rows) > max_rows:
        rows = random.sample(rows, max_rows)
    return rows


def make_frame(
    *,
    positive_csv: str,
    negative_csvs: list[str],
    max_positive_rows: int,
    max_negative_rows: int,
    positive_sampling_mode: str,
    anchor_vocab: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    all_pos = positive_rows(positive_csv, max_rows=0, sampling_mode="random")
    pos = sample_positive_rows(all_pos, max_rows=max_positive_rows, mode=positive_sampling_mode)
    neg = negative_rows(negative_csvs, max_rows=max_negative_rows)
    if anchor_vocab is None:
        anchor_vocab = sorted({row["anchor_token_label_text"] for row in all_pos if row["anchor_token_label_text"]})
    anchor_to_id = {token: index for index, token in enumerate(anchor_vocab)}
    rows = []
    for row in pos + neg:
        enriched = dict(row)
        token = str(enriched.get("anchor_token_label_text") or "")
        enriched["anchor_token_label"] = anchor_to_id.get(token, IGNORE_INDEX) if token else IGNORE_INDEX
        rows.append(enriched)
    random.shuffle(rows)
    frame = pd.DataFrame(rows).reset_index(drop=True)
    return frame, anchor_vocab


def audit_frame_images(df: pd.DataFrame, *, name: str, max_rows: int) -> dict[str, Any]:
    if "file_path" not in df.columns:
        raise ValueError(f"{name} frame has no file_path column")
    paths = [str(value) for value in df["file_path"].tolist()]
    missing = []
    unreadable = []
    checked = 0
    sample_paths = paths if max_rows <= 0 else paths[: max_rows]
    for path_text in sample_paths:
        checked += 1
        path = Path(path_text)
        if not path.is_absolute():
            raise ValueError(f"{name} image path is not absolute after normalization: {path_text}")
        if not path.exists():
            missing.append(path_text)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            unreadable.append({"path": path_text, "error": str(exc)})
    if missing or unreadable:
        raise ValueError(
            f"{name} image audit failed: missing={missing[:10]} unreadable={unreadable[:10]}"
        )
    return {
        "name": name,
        "rows": int(len(paths)),
        "checked_rows": int(checked),
        "all_paths_absolute": True,
        "missing": 0,
        "unreadable": 0,
    }


def class_weights(labels: list[int]) -> list[float]:
    counts = Counter(labels)
    total = float(sum(counts.values()))
    weights = {label: total / max(1.0, float(len(counts) * count)) for label, count in counts.items()}
    return [weights[int(label)] for label in labels]


def sampler_labels(df: pd.DataFrame) -> list[int]:
    labels = []
    for _, row in df.iterrows():
        if int(row["endpoint_presence_label"]) == 0:
            labels.append(0)
            continue
        side = int(row["endpoint_side_label"])
        labels.append(1 + max(0, side))
    return labels


def make_loader(
    model_args: Any,
    df: pd.DataFrame,
    *,
    batch_size: int,
    num_workers: int,
    weighted: bool,
    side_balanced_sampler: bool,
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
        labels = sampler_labels(df) if side_balanced_sampler else [
            int(value) for value in df["endpoint_presence_label"].tolist()
        ]
        weights = class_weights(labels)
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
    values = [int(df.iloc[int(row_id)][column]) for row_id in row_ids]
    return torch.tensor(values, dtype=torch.long, device=device)


def float_labels_for(df: pd.DataFrame, row_ids: list[int], column: str, device: torch.device) -> torch.Tensor:
    values = [float(df.iloc[int(row_id)][column]) for row_id in row_ids]
    return torch.tensor(values, dtype=torch.float32, device=device)


def endpoint_points_for(df: pd.DataFrame, row_ids: list[int], device: torch.device) -> torch.Tensor:
    values = [
        [float(df.iloc[int(row_id)]["endpoint_x_label"]), float(df.iloc[int(row_id)]["endpoint_y_label"])]
        for row_id in row_ids
    ]
    return torch.tensor(values, dtype=torch.float32, device=device)


def probability_for(logits: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1).gather(1, class_ids.unsqueeze(1)).squeeze(1)


def checkpoint_payload(
    *,
    fragment_expert: FragmentAttachmentExpert,
    anchor_vocab: list[str],
    encoder: nn.Module,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    epoch: int,
    eval_report: dict[str, Any] | None,
    confidence_report_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": "fragment_attachment_expert_checkpoint_v2",
        "fragment_expert": fragment_expert.state_dict(),
        "anchor_vocab": anchor_vocab,
        "side_names": SIDE_NAMES,
        "base_checkpoint": str(args.base_checkpoint),
        "input_dim": int(encoder.n_features),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "expert_depth": int(args.expert_depth),
        "expert_heads": int(args.expert_heads),
        "patch_size": int(args.patch_size),
        "endpoint_queries": int(args.endpoint_queries),
        "epoch": int(epoch),
        "eval_report": eval_report or {},
        "confidence_report": str(confidence_report_path) if confidence_report_path else "",
        "metrics": metrics,
    }


def training_state_payload(
    *,
    fragment_expert: FragmentAttachmentExpert,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    scaler: GradScaler,
    anchor_vocab: list[str],
    encoder: nn.Module,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    epoch: int,
    best_selection_score: float,
    best_checkpoint_path: str,
) -> dict[str, Any]:
    payload = checkpoint_payload(
        fragment_expert=fragment_expert,
        anchor_vocab=anchor_vocab,
        encoder=encoder,
        args=args,
        metrics=metrics,
        epoch=epoch,
        eval_report=metrics["epochs"][-1]["eval"] if metrics.get("epochs") else {},
        confidence_report_path=(
            Path(metrics["epochs"][-1]["confidence_report"])
            if metrics.get("epochs") and metrics["epochs"][-1].get("confidence_report")
            else None
        ),
    )
    payload.update(
        {
            "schema_version": "fragment_attachment_expert_training_state_v1",
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "next_epoch": int(epoch) + 1,
            "best_selection_score": float(best_selection_score),
            "best_checkpoint_path": str(best_checkpoint_path or ""),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }
    )
    return payload


def save_fragment_checkpoint(
    *,
    output_dir: Path,
    name: str,
    fragment_expert: FragmentAttachmentExpert,
    anchor_vocab: list[str],
    encoder: nn.Module,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    epoch: int,
    eval_report: dict[str, Any] | None,
    confidence_report_path: Path | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        fragment_expert=fragment_expert,
        anchor_vocab=anchor_vocab,
        encoder=encoder,
        args=args,
        metrics=metrics,
        epoch=epoch,
        eval_report=eval_report,
        confidence_report_path=confidence_report_path,
    )
    path = output_dir / name
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return path


def save_training_state_checkpoint(
    *,
    output_dir: Path,
    fragment_expert: FragmentAttachmentExpert,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    scaler: GradScaler,
    anchor_vocab: list[str],
    encoder: nn.Module,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    epoch: int,
    best_selection_score: float,
    best_checkpoint_path: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fragment_attachment_training_state_last.pth"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        training_state_payload(
            fragment_expert=fragment_expert,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            anchor_vocab=anchor_vocab,
            encoder=encoder,
            args=args,
            metrics=metrics,
            epoch=epoch,
            best_selection_score=best_selection_score,
            best_checkpoint_path=best_checkpoint_path,
        ),
        tmp_path,
    )
    tmp_path.replace(path)
    return path


def _config_value(config: dict[str, Any], key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else None


def validate_resume_training_state(
    state: dict[str, Any],
    *,
    args: argparse.Namespace,
    encoder: nn.Module,
    anchor_vocab: list[str],
    strict_config: bool,
) -> None:
    required = [
        "fragment_expert",
        "optimizer",
        "metrics",
        "input_dim",
        "hidden_dim",
        "dropout",
        "expert_depth",
        "expert_heads",
        "patch_size",
        "endpoint_queries",
        "anchor_vocab",
        "next_epoch",
    ]
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"resume training state is missing required keys: {missing}")
    mismatches = []
    comparisons = [
        ("input_dim", int(encoder.n_features)),
        ("hidden_dim", int(args.hidden_dim)),
        ("dropout", float(args.dropout)),
        ("expert_depth", int(args.expert_depth)),
        ("expert_heads", int(args.expert_heads)),
        ("patch_size", int(args.patch_size)),
        ("endpoint_queries", int(args.endpoint_queries)),
    ]
    for key, expected in comparisons:
        observed = state.get(key)
        if isinstance(expected, float):
            ok = abs(float(observed) - expected) <= 1e-12
        else:
            ok = int(observed) == int(expected)
        if not ok:
            mismatches.append(f"{key}: checkpoint={observed} current={expected}")
    if list(state.get("anchor_vocab") or []) != list(anchor_vocab):
        mismatches.append("anchor_vocab differs between resume checkpoint and current data")
    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    previous_config = metrics.get("config") if isinstance(metrics.get("config"), dict) else {}
    if strict_config:
        strict_keys = [
            "positive_label_csv",
            "eval_positive_label_csv",
            "negative_csv",
            "eval_negative_csv",
            "base_checkpoint",
            "training_stage",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "scheduler",
            "warmup_ratio",
            "class_balance_beta",
            "max_class_weight",
            "presence_focal_gamma",
            "side_loss_weight",
            "anchor_loss_weight",
            "backbone_eligibility_loss_weight",
            "sidecar_applicability_loss_weight",
            "point_loss_weight",
            "heatmap_loss_weight",
            "objectness_loss_weight",
            "risk_loss_weight",
        ]
        current_config = vars(args)
        for key in strict_keys:
            if str(_config_value(previous_config, key)) != str(_config_value(current_config, key)):
                mismatches.append(
                    f"config.{key}: checkpoint={_config_value(previous_config, key)} "
                    f"current={_config_value(current_config, key)}"
                )
    if mismatches:
        raise ValueError("resume training state is incompatible: " + "; ".join(mismatches))


def restore_rng_state(rng_state: dict[str, Any]) -> None:
    if not isinstance(rng_state, dict):
        return
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda"):
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def load_training_state(path: str | Path, *, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        raise FileNotFoundError(f"resume training state not found: {state_path}")
    state = torch.load(state_path, map_location=map_location)
    if not isinstance(state, dict):
        raise ValueError(f"resume training state must be a dict: {state_path}")
    if str(state.get("schema_version") or "") != "fragment_attachment_expert_training_state_v1":
        raise ValueError(f"unsupported resume training state schema: {state.get('schema_version')}")
    return state


def load_fragment_attachment_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[FragmentAttachmentExpert, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"fragment attachment expert checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    required = [
        "fragment_expert",
        "anchor_vocab",
        "input_dim",
        "hidden_dim",
        "dropout",
        "expert_depth",
        "expert_heads",
        "patch_size",
        "endpoint_queries",
    ]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"fragment attachment checkpoint is missing required keys: {missing}")
    expert = FragmentAttachmentExpert(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        anchor_classes=len(list(checkpoint["anchor_vocab"])),
        depth=int(checkpoint["expert_depth"]),
        num_heads=int(checkpoint["expert_heads"]),
        patch_size=int(checkpoint["patch_size"]),
        endpoint_queries=int(checkpoint["endpoint_queries"]),
    ).to(device)
    expert.load_state_dict(checkpoint["fragment_expert"], strict=True)
    expert.eval()
    return expert, checkpoint


@torch.no_grad()
def predict_fragment_attachment_batch(
    *,
    encoder: nn.Module,
    fragment_expert: FragmentAttachmentExpert,
    model_args: Any,
    image_paths: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> list[dict[str, Any]]:
    if not image_paths:
        return []
    rows = [
        {
            "file_path": str(Path(path).resolve()),
            "SMILES": "",
            "source_id": str(index),
        }
        for index, path in enumerate(image_paths)
    ]
    df = pd.DataFrame(rows)
    loader = make_loader(
        model_args,
        df,
        batch_size=batch_size,
        num_workers=num_workers,
        weighted=False,
        side_balanced_sampler=False,
    )
    encoder.eval()
    fragment_expert.eval()
    predictions: dict[int, dict[str, Any]] = {}
    for row_ids, images, _refs in loader:
        images = images.to(device, non_blocking=True)
        features, _ = encoder(images)
        outputs = fragment_expert(features, images)
        presence_probs = torch.softmax(outputs["endpoint_presence"], dim=1)
        side_probs = torch.softmax(outputs["endpoint_side"], dim=1)
        anchor_probs = torch.softmax(outputs["anchor_token"], dim=1)
        applicability_probs = torch.softmax(outputs["sidecar_applicability"], dim=1)
        risk_probs = torch.softmax(outputs["sidecar_risk"], dim=1)
        objectness_probs = torch.softmax(outputs["endpoint_objectness"], dim=1)
        endpoint_query_probs = objectness_probs[:, : outputs["endpoint_query_points"].shape[1]]
        selected_queries = endpoint_query_probs.argmax(dim=1)
        points = outputs["endpoint_query_points"][
            torch.arange(outputs["endpoint_query_points"].shape[0], device=device),
            selected_queries,
        ]
        pred_sides = side_probs.argmax(dim=1)
        pred_anchors = anchor_probs.argmax(dim=1)
        side_conf = probability_for(outputs["endpoint_side"], pred_sides)
        anchor_conf = probability_for(outputs["anchor_token"], pred_anchors)
        query_conf = endpoint_query_probs.gather(1, selected_queries.unsqueeze(1)).squeeze(1)
        sidecar_conf = (
            applicability_probs[:, 1]
            * presence_probs[:, 1]
            * side_conf
            * anchor_conf
            * query_conf
            * risk_probs[:, 1]
        )
        for local_index, row_id in enumerate(row_ids):
            predictions[int(row_id)] = {
                "endpoint_presence": int(presence_probs[local_index].argmax().detach().cpu().item()),
                "endpoint_presence_confidence": float(presence_probs[local_index, 1].detach().cpu().item()),
                "side": SIDE_NAMES[int(pred_sides[local_index].detach().cpu().item())],
                "side_confidence": float(side_conf[local_index].detach().cpu().item()),
                "anchor_index": int(pred_anchors[local_index].detach().cpu().item()),
                "anchor_confidence": float(anchor_conf[local_index].detach().cpu().item()),
                "applicability_confidence": float(applicability_probs[local_index, 1].detach().cpu().item()),
                "risk_confidence": float(risk_probs[local_index, 1].detach().cpu().item()),
                "query_objectness_confidence": float(query_conf[local_index].detach().cpu().item()),
                "no_endpoint_confidence": float(objectness_probs[local_index, -1].detach().cpu().item()),
                "sidecar_confidence": float(sidecar_conf[local_index].detach().cpu().item()),
                "endpoint_x": float(points[local_index, 0].detach().cpu().item()),
                "endpoint_y": float(points[local_index, 1].detach().cpu().item()),
            }
    return [predictions[index] for index in range(len(image_paths))]


def fragment_eval_selection_score(eval_report: dict[str, Any]) -> float:
    rows = max(1, int(eval_report.get("rows") or 0))
    return float(
        float(eval_report.get("presence_accuracy") or 0.0)
        + float(eval_report.get("side_anchor_accuracy") or 0.0)
        + 0.5 * float(eval_report.get("anchor_accuracy") or 0.0)
        - float(eval_report.get("negative_false_accepts") or 0) / float(rows)
    )


def endpoint_heatmap_targets(
    centers: torch.Tensor,
    points: torch.Tensor,
    weights: torch.Tensor,
    *,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if centers.numel() == 0:
        return centers.new_zeros(centers.shape[:2]), centers.new_zeros(centers.shape[:2])
    valid = weights > 0
    safe_points = points.clamp(0.0, 1.0)
    distance_sq = ((centers - safe_points[:, None, :]) ** 2).sum(dim=-1)
    target = torch.exp(-distance_sq / max(1e-6, 2.0 * float(sigma) * float(sigma)))
    mask = valid.float()[:, None].expand_as(target)
    return target * mask, mask


def matched_endpoint_query_targets(
    query_points: torch.Tensor,
    target_points: torch.Tensor,
    point_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, query_count, _ = query_points.shape
    objectness = torch.full((batch_size,), query_count, device=query_points.device, dtype=torch.long)
    matched_points = []
    matched_targets = []
    matched_weights = []
    matched_indices = torch.full((batch_size,), -1, device=query_points.device, dtype=torch.long)
    valid = point_weights > 0
    if torch.any(valid):
        distances = torch.cdist(query_points[valid], target_points[valid].clamp(0.0, 1.0).unsqueeze(1), p=1).squeeze(-1)
        selected = distances.argmin(dim=1)
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        for local_index, batch_index in enumerate(valid_indices):
            query_index = selected[local_index]
            objectness[batch_index] = query_index
            matched_indices[batch_index] = query_index
            matched_points.append(query_points[batch_index, query_index])
            matched_targets.append(target_points[batch_index].clamp(0.0, 1.0))
            matched_weights.append(point_weights[batch_index])
    if matched_points:
        return (
            objectness,
            torch.stack(matched_points, dim=0),
            torch.stack(matched_targets, dim=0),
            torch.stack(matched_weights, dim=0),
            matched_indices,
        )
    empty_points = query_points.new_zeros((0, 2))
    empty_weights = point_weights.new_zeros((0,))
    return objectness, empty_points, empty_points, empty_weights, matched_indices


def effective_class_weights(
    labels: list[int],
    num_classes: int,
    *,
    beta: float,
    max_weight: float,
    device: torch.device,
) -> torch.Tensor:
    valid = [int(label) for label in labels if int(label) >= 0]
    counts = Counter(valid)
    weights = torch.ones(num_classes, dtype=torch.float32)
    if not counts:
        return weights.to(device)
    beta = min(max(float(beta), 0.0), 0.9999)
    raw = {}
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


def loss_for(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    class_weight_tensors: dict[str, torch.Tensor],
    *,
    presence_focal_gamma: float,
    side_loss_weight: float,
    anchor_loss_weight: float,
    backbone_eligibility_loss_weight: float,
    sidecar_applicability_loss_weight: float,
    point_loss_weight: float,
    heatmap_loss_weight: float,
    objectness_loss_weight: float,
    risk_loss_weight: float,
    heatmap_sigma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    def focal_ce(name: str, gamma: float) -> torch.Tensor:
        logits = outputs[name]
        target = labels[name]
        ce_loss = nn.functional.cross_entropy(
            logits,
            target,
            weight=class_weight_tensors.get(name),
            reduction="none",
        )
        probability = torch.softmax(logits, dim=1)
        pt = probability.gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        return (((1.0 - pt) ** float(gamma)) * ce_loss).mean()

    def optional_ce(name: str) -> torch.Tensor:
        target = labels[name]
        if not torch.any(target != IGNORE_INDEX):
            return outputs[name].sum() * 0.0
        return nn.functional.cross_entropy(
            outputs[name],
            target,
            weight=class_weight_tensors.get(name),
            ignore_index=IGNORE_INDEX,
        )

    losses = {
        "endpoint_presence": focal_ce("endpoint_presence", presence_focal_gamma),
        "sidecar_applicability": focal_ce("sidecar_applicability", presence_focal_gamma),
        "endpoint_side": optional_ce("endpoint_side"),
        "anchor_token": optional_ce("anchor_token"),
        "backbone_eligibility": optional_ce("backbone_eligibility"),
    }
    if "endpoint_query_points" not in outputs:
        raise ValueError("FragmentAttachmentExpert must output endpoint_query_points; legacy fixed-index or pooled endpoint paths are forbidden.")
    if "endpoint_objectness" not in outputs:
        raise ValueError("FragmentAttachmentExpert must output endpoint_objectness for query matching.")
    if "endpoint_point" in outputs and "endpoint_point" in labels:
        point_weights = labels["endpoint_point_weight"].float()
        objectness_target, matched_points, matched_targets, matched_weights, matched_indices = matched_endpoint_query_targets(
            outputs["endpoint_query_points"],
            labels["endpoint_point"],
            point_weights,
        )
        if matched_points.numel() > 0:
            point_error = nn.functional.smooth_l1_loss(
                matched_points,
                matched_targets,
                reduction="none",
            ).sum(dim=1)
            losses["endpoint_point"] = (point_error * matched_weights).sum() / matched_weights.sum().clamp_min(1e-6)
            derived_side = side_from_point(matched_points)
            losses["endpoint_point_side_consistency"] = nn.functional.cross_entropy(
                outputs["endpoint_side"][matched_indices >= 0],
                derived_side,
            )
        else:
            losses["endpoint_point"] = outputs["endpoint_point"].sum() * 0.0
            losses["endpoint_point_side_consistency"] = outputs["endpoint_point"].sum() * 0.0
    if "endpoint_heatmap" in outputs and "endpoint_heatmap_centers" in outputs and "endpoint_point" in labels:
        heatmap_target, heatmap_mask = endpoint_heatmap_targets(
            outputs["endpoint_heatmap_centers"],
            labels["endpoint_point"],
            labels["endpoint_point_weight"].float(),
            sigma=heatmap_sigma,
        )
        if torch.any(heatmap_mask > 0):
            heatmap_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["endpoint_heatmap"],
                heatmap_target,
                reduction="none",
            )
            losses["endpoint_heatmap"] = (heatmap_loss * heatmap_mask).sum() / heatmap_mask.sum().clamp_min(1e-6)
        else:
            losses["endpoint_heatmap"] = outputs["endpoint_heatmap"].sum() * 0.0
    objectness_target, _matched_points, _matched_targets, _matched_weights, _matched_indices = matched_endpoint_query_targets(
        outputs["endpoint_query_points"],
        labels["endpoint_point"],
        labels["endpoint_point_weight"].float(),
    )
    if outputs["endpoint_objectness"].shape[1] != outputs["endpoint_query_points"].shape[1] + 1:
        raise ValueError("endpoint_objectness must contain endpoint queries plus one no-endpoint class.")
    objectness_loss = nn.functional.cross_entropy(
        outputs["endpoint_objectness"],
        objectness_target,
    )
    losses["endpoint_objectness"] = objectness_loss
    if "sidecar_risk" in outputs:
        risk_labels = (
            (labels["endpoint_presence"] == 1)
            & (labels["endpoint_side"] != IGNORE_INDEX)
            & (labels["anchor_token"] != IGNORE_INDEX)
        ).long()
        losses["sidecar_risk"] = nn.functional.cross_entropy(outputs["sidecar_risk"], risk_labels)
    total = (
        losses["endpoint_presence"]
        + float(side_loss_weight) * losses["endpoint_side"]
        + float(anchor_loss_weight) * losses["anchor_token"]
        + float(backbone_eligibility_loss_weight) * losses["backbone_eligibility"]
        + float(sidecar_applicability_loss_weight) * losses["sidecar_applicability"]
        + float(point_loss_weight) * losses.get("endpoint_point", 0.0)
        + 0.25 * float(point_loss_weight) * losses.get("endpoint_point_side_consistency", 0.0)
        + float(heatmap_loss_weight) * losses.get("endpoint_heatmap", 0.0)
        + float(objectness_loss_weight) * losses.get("endpoint_objectness", 0.0)
        + float(risk_loss_weight) * losses.get("sidecar_risk", 0.0)
    )
    return total, {name: float(value.detach().cpu()) for name, value in losses.items()}


@torch.no_grad()
def evaluate(
    *,
    encoder: nn.Module,
    fragment_expert: FragmentAttachmentExpert,
    model_args: Any,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    anchor_vocab: list[str],
    prediction_csv: Path | None = None,
) -> dict[str, Any]:
    encoder.eval()
    fragment_expert.eval()
    loader = make_loader(
        model_args,
        df,
        batch_size=batch_size,
        num_workers=num_workers,
        weighted=False,
        side_balanced_sampler=False,
    )
    counts = Counter()
    confusion_presence: dict[str, Counter[str]] = defaultdict(Counter)
    confusion_side: dict[str, Counter[str]] = defaultdict(Counter)
    confusion_anchor: dict[str, Counter[str]] = defaultdict(Counter)
    examples = []
    prediction_rows: list[dict[str, Any]] = []
    for row_ids, images, _refs in loader:
        images = images.to(device, non_blocking=True)
        features, _ = encoder(images)
        outputs = fragment_expert(features, images)
        presence_probs = torch.softmax(outputs["endpoint_presence"], dim=1)
        side_probs = torch.softmax(outputs["endpoint_side"], dim=1)
        anchor_probs = torch.softmax(outputs["anchor_token"], dim=1)
        applicability_probs = torch.softmax(outputs["sidecar_applicability"], dim=1)
        pred_presence_tensor = presence_probs.argmax(dim=1)
        pred_side_tensor = side_probs.argmax(dim=1)
        pred_anchor_tensor = anchor_probs.argmax(dim=1)
        pred_presence = pred_presence_tensor.cpu().tolist()
        pred_side = pred_side_tensor.cpu().tolist()
        pred_anchor = pred_anchor_tensor.cpu().tolist()
        applicability_confidences = applicability_probs[:, 1]
        presence_confidences = presence_probs[:, 1]
        side_confidences = probability_for(outputs["endpoint_side"], pred_side_tensor)
        anchor_confidences = probability_for(outputs["anchor_token"], pred_anchor_tensor)
        risk_probs = torch.softmax(outputs["sidecar_risk"], dim=1) if "sidecar_risk" in outputs else None
        risk_confidences_tensor = risk_probs[:, 1] if risk_probs is not None else torch.ones_like(applicability_confidences)
        if "endpoint_objectness" not in outputs or "endpoint_query_points" not in outputs:
            raise ValueError("Evaluation requires endpoint_objectness and endpoint_query_points; legacy pooled endpoint paths are forbidden.")
        query_objectness_probs = torch.softmax(outputs["endpoint_objectness"], dim=1)
        endpoint_query_probs = query_objectness_probs[:, : outputs["endpoint_query_points"].shape[1]]
        selected_query_tensor = endpoint_query_probs.argmax(dim=1)
        endpoint_points_tensor = outputs["endpoint_query_points"][
            torch.arange(outputs["endpoint_query_points"].shape[0], device=outputs["endpoint_query_points"].device),
            selected_query_tensor,
        ]
        query_objectness_confidences_tensor = endpoint_query_probs.gather(
            1, selected_query_tensor.unsqueeze(1)
        ).squeeze(1)
        no_endpoint_confidences_tensor = query_objectness_probs[:, -1]
        sidecar_confidences = (
            applicability_confidences
            * presence_confidences
            * side_confidences
            * anchor_confidences
            * query_objectness_confidences_tensor
            * risk_confidences_tensor
        ).cpu().tolist()
        risk_confidences = risk_confidences_tensor.cpu().tolist()
        query_objectness_confidences = query_objectness_confidences_tensor.cpu().tolist()
        no_endpoint_confidences = no_endpoint_confidences_tensor.cpu().tolist()
        endpoint_points = endpoint_points_tensor.detach().cpu().tolist() if "endpoint_point" in outputs else [[POINT_IGNORE, POINT_IGNORE] for _ in row_ids]
        applicability_confidences = applicability_confidences.cpu().tolist()
        presence_confidences = presence_confidences.cpu().tolist()
        side_confidences = side_confidences.cpu().tolist()
        anchor_confidences = anchor_confidences.cpu().tolist()
        gold_presence = labels_for(df, row_ids, "endpoint_presence_label", device).cpu().tolist()
        gold_applicability = labels_for(df, row_ids, "sidecar_applicability_label", device).cpu().tolist()
        gold_side = labels_for(df, row_ids, "endpoint_side_label", device).cpu().tolist()
        gold_anchor = labels_for(df, row_ids, "anchor_token_label", device).cpu().tolist()
        for row_id, ga_app, gp, pp, gs, ps, ga, pa, confidence, risk_conf, query_obj_conf, no_endpoint_conf, point, app_conf, pres_conf, side_conf, anchor_conf in zip(
            row_ids,
            gold_applicability,
            gold_presence,
            pred_presence,
            gold_side,
            pred_side,
            gold_anchor,
            pred_anchor,
            sidecar_confidences,
            risk_confidences,
            query_objectness_confidences,
            no_endpoint_confidences,
            endpoint_points,
            applicability_confidences,
            presence_confidences,
            side_confidences,
            anchor_confidences,
        ):
            counts["rows"] += 1
            counts["presence_correct"] += int(gp == pp)
            counts["applicability_correct"] += int((app_conf >= 0.5) == bool(ga_app))
            confusion_presence[str(gp)][str(pp)] += 1
            counts["confidence_sum"] += float(confidence)
            if gp == 1:
                counts["positive_confidence_sum"] += float(confidence)
                counts["positive_confidence_rows"] += 1
            else:
                counts["negative_confidence_sum"] += float(confidence)
                counts["negative_confidence_rows"] += 1
            source = df.iloc[int(row_id)]
            prediction_row = {
                "row_id": int(row_id),
                "source_id": str(source.get("source_id") or ""),
                "source_arrow": str(source.get("source_arrow") or ""),
                "smiles": str(source.get("SMILES") or ""),
                "gold_applicability": int(ga_app),
                "gold_presence": int(gp),
                "pred_presence": int(pp),
                "gold_side": SIDE_NAMES[gs] if gs != IGNORE_INDEX else "",
                "pred_side": SIDE_NAMES[ps],
                "gold_anchor": anchor_vocab[ga] if ga != IGNORE_INDEX else "",
                "pred_anchor": anchor_vocab[pa],
                "applicability_confidence": float(app_conf),
                "presence_confidence": float(pres_conf),
                "side_confidence": float(side_conf),
                "anchor_confidence": float(anchor_conf),
                "risk_confidence": float(risk_conf),
                "query_objectness_confidence": float(query_obj_conf),
                "no_endpoint_confidence": float(no_endpoint_conf),
                "sidecar_confidence": float(confidence),
                "pred_endpoint_x": float(point[0]),
                "pred_endpoint_y": float(point[1]),
                "gold_endpoint_x": float(source.get("endpoint_x_label", POINT_IGNORE)),
                "gold_endpoint_y": float(source.get("endpoint_y_label", POINT_IGNORE)),
                "endpoint_point_weight": float(source.get("endpoint_point_weight", 0.0)),
                "endpoint_point_source": str(source.get("endpoint_point_source") or ""),
            }
            prediction_rows.append(prediction_row)
            if gp == 0 and pp == 1:
                counts["negative_false_accepts"] += 1
            if gs != IGNORE_INDEX:
                counts["side_rows"] += 1
                counts["side_correct"] += int(gs == ps)
                confusion_side[SIDE_NAMES[gs]][SIDE_NAMES[ps]] += 1
            if ga != IGNORE_INDEX:
                counts["anchor_rows"] += 1
                counts["anchor_correct"] += int(ga == pa)
                confusion_anchor[anchor_vocab[ga]][anchor_vocab[pa]] += 1
            if gs != IGNORE_INDEX and ga != IGNORE_INDEX:
                counts["side_anchor_rows"] += 1
                counts["side_anchor_correct"] += int(gs == ps and ga == pa)
            if (gp != pp or (gs != IGNORE_INDEX and gs != ps) or (ga != IGNORE_INDEX and ga != pa)) and len(examples) < 24:
                examples.append(prediction_row)
    if prediction_csv is not None:
        prediction_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prediction_rows).to_csv(prediction_csv, index=False)
    return {
        "rows": int(counts["rows"]),
        "presence_accuracy": counts["presence_correct"] / max(1, counts["rows"]),
        "applicability_accuracy_at_0_5": counts["applicability_correct"] / max(1, counts["rows"]),
        "negative_false_accepts": int(counts["negative_false_accepts"]),
        "mean_sidecar_confidence": counts["confidence_sum"] / max(1, counts["rows"]),
        "positive_mean_sidecar_confidence": counts["positive_confidence_sum"] / max(1, counts["positive_confidence_rows"]),
        "negative_mean_sidecar_confidence": counts["negative_confidence_sum"] / max(1, counts["negative_confidence_rows"]),
        "side_rows": int(counts["side_rows"]),
        "side_accuracy": counts["side_correct"] / max(1, counts["side_rows"]),
        "side_per_class_recall": {
            side_name: (
                confusion_side.get(side_name, Counter()).get(side_name, 0)
                / max(1, sum(confusion_side.get(side_name, Counter()).values()))
            )
            for side_name in SIDE_NAMES
        },
        "side_balanced_accuracy": sum(
            confusion_side.get(side_name, Counter()).get(side_name, 0)
            / max(1, sum(confusion_side.get(side_name, Counter()).values()))
            for side_name in SIDE_NAMES
        )
        / len(SIDE_NAMES),
        "anchor_rows": int(counts["anchor_rows"]),
        "anchor_accuracy": counts["anchor_correct"] / max(1, counts["anchor_rows"]),
        "side_anchor_rows": int(counts["side_anchor_rows"]),
        "side_anchor_accuracy": counts["side_anchor_correct"] / max(1, counts["side_anchor_rows"]),
        "presence_confusion": {key: dict(value) for key, value in sorted(confusion_presence.items())},
        "side_confusion": {key: dict(value) for key, value in sorted(confusion_side.items())},
        "anchor_confusion": {key: dict(value) for key, value in sorted(confusion_anchor.items())},
        "mismatches": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen-base fragment attachment expert.")
    parser.add_argument("--positive-label-csv", required=True)
    parser.add_argument("--negative-csv", action="append", default=[])
    parser.add_argument("--eval-positive-label-csv", default="")
    parser.add_argument("--eval-negative-csv", action="append", default=[])
    parser.add_argument("--acceptance-report", default="")
    parser.add_argument("--readiness-report", default="")
    parser.add_argument("--formal-preflight-report", default="")
    parser.add_argument("--roadmap-constraints-report", default="")
    parser.add_argument(
        "--training-stage",
        choices=["micro_smoke", "research_only", "measured_smoke", "formal"],
        default="measured_smoke",
    )
    parser.add_argument(
        "--allow-ungated-debug-run",
        action="store_true",
        help="Allow a local debug run without an acceptance report. Metrics will mark the run as not accepted for training.",
    )
    parser.add_argument("--image-audit-max-rows", type=int, default=2048)
    parser.add_argument("--base-checkpoint", default="/workspace/models/molnextr_best.pth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--expert-depth", type=int, default=4)
    parser.add_argument("--expert-heads", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--endpoint-queries", type=int, default=4)
    parser.add_argument("--class-balance-beta", type=float, default=0.99)
    parser.add_argument("--max-class-weight", type=float, default=8.0)
    parser.add_argument("--presence-focal-gamma", type=float, default=2.0)
    parser.add_argument("--side-loss-weight", type=float, default=1.0)
    parser.add_argument("--anchor-loss-weight", type=float, default=1.0)
    parser.add_argument("--backbone-eligibility-loss-weight", type=float, default=0.25)
    parser.add_argument("--sidecar-applicability-loss-weight", type=float, default=0.5)
    parser.add_argument("--point-loss-weight", type=float, default=1.0)
    parser.add_argument("--heatmap-loss-weight", type=float, default=0.5)
    parser.add_argument("--objectness-loss-weight", type=float, default=0.25)
    parser.add_argument("--risk-loss-weight", type=float, default=0.25)
    parser.add_argument("--confidence-thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--confidence-ece-bins", type=int, default=15)
    parser.add_argument("--confidence-max-false-accept", type=int, default=0)
    parser.add_argument("--confidence-max-negative-accepted", type=int, default=0)
    parser.add_argument("--confidence-min-positive-triplet-precision", type=float, default=0.95)
    parser.add_argument("--confidence-error-limit", type=int, default=24)
    parser.add_argument("--heatmap-sigma", type=float, default=0.08)
    parser.add_argument("--positive-sampling-mode", choices=["random", "side_balanced"], default="side_balanced")
    parser.add_argument("--max-positive-rows", type=int, default=0)
    parser.add_argument("--max-negative-rows", type=int, default=0)
    parser.add_argument("--max-eval-positive-rows", type=int, default=0)
    parser.add_argument("--max-eval-negative-rows", type=int, default=0)
    parser.add_argument(
        "--resume-training-state",
        default="",
        help="Resume optimizer/scheduler/scaler/model state from fragment_attachment_training_state_last.pth.",
    )
    parser.add_argument(
        "--no-resume-strict-config",
        action="store_true",
        help="Allow resume when non-architecture config differs. Use only for explicit diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--no-side-balanced-sampler", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    ddp = distributed_context()
    require_measured_runtime(args, ddp)
    seed_all(int(args.seed) + int(ddp["rank"]))
    output_dir = Path(args.output_dir)
    if is_main_process(ddp):
        output_dir.mkdir(parents=True, exist_ok=True)
    ddp_barrier(ddp)
    gate_report = validate_training_gate(
        acceptance_report=str(args.acceptance_report or ""),
        readiness_report=str(args.readiness_report or ""),
        formal_preflight_report=str(args.formal_preflight_report or ""),
        roadmap_constraints_report=str(args.roadmap_constraints_report or ""),
        training_stage=str(args.training_stage),
        allow_ungated_debug_run=bool(args.allow_ungated_debug_run),
    )
    if ddp["enabled"] and args.cpu:
        raise ValueError("--cpu cannot be combined with torchrun/DDP.")
    device = torch.device(f"cuda:{int(ddp['local_rank'])}") if ddp["enabled"] else select_single_process_device(args)
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    # Train the endpoint expert WITH real-style degradation. The legacy
    # augment=False trained on clean RDKit images, so the expert's heads
    # miscalibrated on degraded real patent crops (sidecar_confidence 0.176 vs
    # 0.97 synthetic). At inference load_frozen_encoder is called with
    # augment=False / no real_match, so this only affects training.
    config["augment"] = True
    config["real_match"] = True
    config["input_sha256"] = {
        "positive_label_csv": file_sha256(args.positive_label_csv),
        "eval_positive_label_csv": file_sha256(args.eval_positive_label_csv or args.positive_label_csv),
        "negative_csv": file_sha256_one_or_many(args.negative_csv),
        "eval_negative_csv": file_sha256_one_or_many(args.eval_negative_csv or args.negative_csv),
    }
    encoder, model_args = load_frozen_encoder(args.base_checkpoint, config, device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()

    train_df, anchor_vocab = make_frame(
        positive_csv=args.positive_label_csv,
        negative_csvs=args.negative_csv,
        max_positive_rows=int(args.max_positive_rows),
        max_negative_rows=int(args.max_negative_rows),
        positive_sampling_mode=str(args.positive_sampling_mode),
    )
    if train_df.empty:
        raise ValueError("No training rows were loaded for the fragment attachment expert.")
    if not anchor_vocab:
        raise ValueError("No positive anchor-token labels were loaded for the fragment attachment expert.")
    eval_positive = args.eval_positive_label_csv or args.positive_label_csv
    eval_negative = args.eval_negative_csv or args.negative_csv
    eval_df, _ = make_frame(
        positive_csv=eval_positive,
        negative_csvs=eval_negative,
        max_positive_rows=int(args.max_eval_positive_rows),
        max_negative_rows=int(args.max_eval_negative_rows),
        positive_sampling_mode="random",
        anchor_vocab=anchor_vocab,
    )
    if eval_df.empty:
        raise ValueError("No evaluation rows were loaded for the fragment attachment expert.")
    image_audit = None
    if is_main_process(ddp):
        image_audit = {
            "train": audit_frame_images(train_df, name="train", max_rows=int(args.image_audit_max_rows)),
            "eval": audit_frame_images(eval_df, name="eval", max_rows=int(args.image_audit_max_rows)),
        }
    ddp_barrier(ddp)

    fragment_expert = FragmentAttachmentExpert(
        input_dim=int(encoder.n_features),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        anchor_classes=len(anchor_vocab),
        depth=int(args.expert_depth),
        num_heads=int(args.expert_heads),
        patch_size=int(args.patch_size),
        endpoint_queries=int(args.endpoint_queries),
    ).to(device)
    resume_state: dict[str, Any] | None = None
    if str(args.resume_training_state or ""):
        resume_state = load_training_state(args.resume_training_state, map_location=device)
        validate_resume_training_state(
            resume_state,
            args=args,
            encoder=encoder,
            anchor_vocab=anchor_vocab,
            strict_config=not bool(args.no_resume_strict_config),
        )
        fragment_expert.load_state_dict(resume_state["fragment_expert"], strict=True)
    train_model: nn.Module = fragment_expert
    if ddp["enabled"]:
        train_model = DistributedDataParallel(
            fragment_expert,
            device_ids=[int(ddp["local_rank"])],
            output_device=int(ddp["local_rank"]),
        )
    class_weight_tensors = {
        "endpoint_presence": effective_class_weights(
            [int(value) for value in train_df["endpoint_presence_label"].tolist()],
            2,
            beta=float(args.class_balance_beta),
            max_weight=float(args.max_class_weight),
            device=device,
        ),
        "sidecar_applicability": effective_class_weights(
            [int(value) for value in train_df["sidecar_applicability_label"].tolist()],
            2,
            beta=float(args.class_balance_beta),
            max_weight=float(args.max_class_weight),
            device=device,
        ),
        "endpoint_side": effective_class_weights(
            [int(value) for value in train_df["endpoint_side_label"].tolist()],
            len(SIDE_NAMES),
            beta=float(args.class_balance_beta),
            max_weight=float(args.max_class_weight),
            device=device,
        ),
        "anchor_token": effective_class_weights(
            [int(value) for value in train_df["anchor_token_label"].tolist()],
            len(anchor_vocab),
            beta=float(args.class_balance_beta),
            max_weight=float(args.max_class_weight),
            device=device,
        ),
        "backbone_eligibility": effective_class_weights(
            [int(value) for value in train_df["backbone_eligibility_label"].tolist()],
            2,
            beta=float(args.class_balance_beta),
            max_weight=float(args.max_class_weight),
            device=device,
        ),
    }
    optimizer = torch.optim.AdamW(
        fragment_expert.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    loader = make_loader(
        model_args,
        train_df,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        weighted=not bool(args.no_weighted_sampler),
        side_balanced_sampler=not bool(args.no_side_balanced_sampler),
        ddp=ddp,
    )
    grad_accum_steps = max(1, int(args.gradient_accumulation_steps))
    optimizer_steps_per_epoch = max(1, (len(loader) + grad_accum_steps - 1) // grad_accum_steps)
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
    resume_next_epoch = 1
    resume_best_selection_score = float("-inf")
    resume_best_checkpoint_path = ""
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        if scheduler is not None and resume_state.get("scheduler") is not None:
            scheduler.load_state_dict(resume_state["scheduler"])
        if resume_state.get("scaler"):
            scaler.load_state_dict(resume_state["scaler"])
        restore_rng_state(resume_state.get("rng_state") if isinstance(resume_state.get("rng_state"), dict) else {})
        resume_next_epoch = max(1, int(resume_state.get("next_epoch") or 1))
        resume_best_selection_score = float(resume_state.get("best_selection_score", float("-inf")))
        resume_best_checkpoint_path = str(resume_state.get("best_checkpoint_path") or "")
    metrics: dict[str, Any] = {
        "base_checkpoint": str(args.base_checkpoint),
        "config": config,
        "gate_report": gate_report,
        "image_audit": image_audit,
        "formal_full_checkpoint_training_allowed": False,
        "complete_path_mutated": False,
        "encoder_trainable": False,
        "decoder_loaded": False,
        "sidecar_only": True,
        "debug_only": bool(args.allow_ungated_debug_run or args.training_stage == "micro_smoke"),
        "model_scale": build_model_scale_report(
            encoder=encoder,
            expert=fragment_expert,
            input_dim=int(encoder.n_features),
            input_size=int(getattr(model_args, "input_size", 0) or 0),
            args=args,
            training_stage=str(args.training_stage),
            debug_only=bool(args.allow_ungated_debug_run),
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
        },
        "architecture": "fragment_attachment_expert",
        "expert_architecture": {
            "image_patch_embed": True,
            "molnextr_token_projection": True,
            "fusion_transformer_depth": int(args.expert_depth),
            "fusion_transformer_heads": int(args.expert_heads),
            "patch_size": int(args.patch_size),
            "endpoint_queries": int(args.endpoint_queries),
            "learned_endpoint_query_decoder": True,
            "endpoint_localization_heads": ["point", "heatmap", "objectness"],
            "multi_head_confidence": [
                "applicability",
                "presence",
                "side",
                "anchor",
                "backbone_eligibility",
                "risk",
            ],
            "complete_path": "direct_original_molnextr",
        },
        "loss_policy": {
            "endpoint_presence": "class_balanced_focal_cross_entropy",
            "endpoint_side": "class_balanced_cross_entropy_ignore_negative_rows",
            "anchor_token": "class_balanced_cross_entropy_ignore_negative_rows",
            "backbone_eligibility": "optional_ignore_only_safe_cross_entropy",
            "sidecar_applicability": "class_balanced_focal_cross_entropy",
            "endpoint_point": "smooth_l1_weighted_by_label_quality",
            "endpoint_heatmap": "gaussian_heatmap_bce_weighted_by_label_quality",
            "endpoint_objectness": "matched_query_cross_entropy_with_no_endpoint_class",
            "sidecar_risk": "cross_entropy_correctness_proxy",
            "presence_focal_gamma": float(args.presence_focal_gamma),
            "side_loss_weight": float(args.side_loss_weight),
            "anchor_loss_weight": float(args.anchor_loss_weight),
            "backbone_eligibility_loss_weight": float(args.backbone_eligibility_loss_weight),
            "sidecar_applicability_loss_weight": float(args.sidecar_applicability_loss_weight),
            "point_loss_weight": float(args.point_loss_weight),
            "heatmap_loss_weight": float(args.heatmap_loss_weight),
            "objectness_loss_weight": float(args.objectness_loss_weight),
            "risk_loss_weight": float(args.risk_loss_weight),
            "heatmap_sigma": float(args.heatmap_sigma),
        },
        "anchor_vocab": anchor_vocab,
        "side_names": SIDE_NAMES,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "train_label_counts": {
            "presence": dict(sorted(Counter(int(v) for v in train_df["endpoint_presence_label"]).items())),
            "side": dict(sorted(Counter(int(v) for v in train_df["endpoint_side_label"] if int(v) >= 0).items())),
            "anchor": dict(sorted(Counter(str(v) for v in train_df["anchor_token_label_text"] if str(v)).items())),
        },
        "class_weights": {
            name: [float(value) for value in tensor.detach().cpu().tolist()]
            for name, tensor in sorted(class_weight_tensors.items())
        },
        "eval_label_counts": {
            "presence": dict(sorted(Counter(int(v) for v in eval_df["endpoint_presence_label"]).items())),
            "side": dict(sorted(Counter(int(v) for v in eval_df["endpoint_side_label"] if int(v) >= 0).items())),
            "anchor": dict(sorted(Counter(str(v) for v in eval_df["anchor_token_label_text"] if str(v)).items())),
        },
        "epochs": [],
        "device": str(device),
    }
    if resume_state is not None:
        previous_metrics = resume_state.get("metrics") if isinstance(resume_state.get("metrics"), dict) else {}
        previous_epochs = previous_metrics.get("epochs") if isinstance(previous_metrics.get("epochs"), list) else []
        metrics["epochs"] = list(previous_epochs)
        metrics["resumed_from_training_state"] = str(args.resume_training_state)
        metrics["resume_next_epoch"] = int(resume_next_epoch)
        if previous_metrics.get("best_checkpoint"):
            metrics["best_checkpoint"] = previous_metrics.get("best_checkpoint")
        if previous_metrics.get("best_epoch"):
            metrics["best_epoch"] = previous_metrics.get("best_epoch")
        if previous_metrics.get("best_selection_score") is not None:
            metrics["best_selection_score"] = previous_metrics.get("best_selection_score")
    start = time.time()
    best_selection_score = float(resume_best_selection_score)
    best_checkpoint_path = str(resume_best_checkpoint_path)
    start_epoch = int(resume_next_epoch)
    target_epochs = int(args.epochs)
    if start_epoch > target_epochs:
        raise ValueError(
            f"resume state next_epoch={start_epoch} is beyond requested --epochs={target_epochs}; "
            "increase --epochs to continue training."
        )
    for epoch_number in range(start_epoch, target_epochs + 1):
        epoch_index = epoch_number - 1
        train_model.train()
        encoder.eval()
        if ddp["enabled"] and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch_index)
        total_loss = 0.0
        total_rows = 0
        loss_sums: Counter[str] = Counter()
        progress = tqdm(
            loader,
            desc=f"fragment attachment expert epoch {epoch_number}",
            disable=not is_main_process(ddp),
        )
        optimizer.zero_grad(set_to_none=True)
        for step, (row_ids, images, _refs) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            labels = {
                "endpoint_presence": labels_for(train_df, row_ids, "endpoint_presence_label", device),
                "sidecar_applicability": labels_for(train_df, row_ids, "sidecar_applicability_label", device),
                "endpoint_side": labels_for(train_df, row_ids, "endpoint_side_label", device),
                "anchor_token": labels_for(train_df, row_ids, "anchor_token_label", device),
                "backbone_eligibility": labels_for(train_df, row_ids, "backbone_eligibility_label", device),
                "endpoint_point": endpoint_points_for(train_df, row_ids, device),
                "endpoint_point_weight": float_labels_for(train_df, row_ids, "endpoint_point_weight", device),
            }
            with torch.no_grad():
                features, _ = encoder(images)
            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = train_model(features, images)
                loss, components = loss_for(
                    outputs,
                    labels,
                    class_weight_tensors,
                    presence_focal_gamma=float(args.presence_focal_gamma),
                    side_loss_weight=float(args.side_loss_weight),
                    anchor_loss_weight=float(args.anchor_loss_weight),
                    backbone_eligibility_loss_weight=float(args.backbone_eligibility_loss_weight),
                    sidecar_applicability_loss_weight=float(args.sidecar_applicability_loss_weight),
                    point_loss_weight=float(args.point_loss_weight),
                    heatmap_loss_weight=float(args.heatmap_loss_weight),
                    objectness_loss_weight=float(args.objectness_loss_weight),
                    risk_loss_weight=float(args.risk_loss_weight),
                    heatmap_sigma=float(args.heatmap_sigma),
                )
                scaled_loss = loss / float(grad_accum_steps)
            scaler.scale(scaled_loss).backward()
            should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fragment_expert.parameters(), float(args.max_grad_norm))
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_was_skipped = amp_enabled and float(scaler.get_scale()) < previous_scale
                if scheduler is not None and not optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            batch_rows = int(labels["endpoint_presence"].numel())
            total_loss += float(loss.detach().cpu()) * batch_rows
            total_rows += batch_rows
            for name, value in components.items():
                loss_sums[name] += value * batch_rows
            if is_main_process(ddp):
                progress.set_postfix({"loss": total_loss / max(1, total_rows), "lr": optimizer.param_groups[0]["lr"]})
        train_stats = torch.tensor([float(total_loss), float(total_rows)], dtype=torch.float64, device=device)
        if ddp["enabled"]:
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
        train_loss = float(train_stats[0].detach().cpu()) / max(1.0, float(train_stats[1].detach().cpu()))
        ddp_barrier(ddp)
        epoch_eval_report = None
        epoch_confidence_report_path = None
        if is_main_process(ddp):
            prediction_csv = output_dir / f"eval_predictions_epoch_{epoch_number}.csv"
            eval_report = evaluate(
                encoder=encoder,
                fragment_expert=fragment_expert,
                model_args=model_args,
                df=eval_df,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                anchor_vocab=anchor_vocab,
                prediction_csv=prediction_csv,
            )
            try:
                confidence_report = build_confidence_report(
                    predictions_csv=prediction_csv,
                    thresholds=parse_thresholds(str(args.confidence_thresholds)),
                    ece_bins=int(args.confidence_ece_bins),
                    max_false_accept=int(args.confidence_max_false_accept),
                    max_negative_accepted=int(args.confidence_max_negative_accepted),
                    min_positive_triplet_precision=float(args.confidence_min_positive_triplet_precision),
                    error_limit=int(args.confidence_error_limit),
                )
            except Exception as exc:
                if str(args.training_stage) != "micro_smoke":
                    raise
                confidence_report = {
                    "schema_version": "fragment_attachment_confidence_report_unavailable_v1",
                    "deployment_allowed": False,
                    "micro_smoke_only": True,
                    "error": str(exc),
                    "reason": "calibration_unavailable_for_tiny_micro_smoke",
                    "quality_constraints": {
                        "deployable": False,
                        "formal_or_measured_training_must_not_use_this_report": True,
                    },
                }
            confidence_dir = output_dir / f"confidence_epoch_{epoch_number}"
            confidence_dir.mkdir(parents=True, exist_ok=True)
            confidence_report_path = confidence_dir / "fragment_attachment_confidence_report.json"
            confidence_report_path.write_text(
                json.dumps(confidence_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            epoch_eval_report = eval_report
            epoch_confidence_report_path = confidence_report_path
            metrics["epochs"].append(
                {
                    "epoch": epoch_number,
                    "train_loss": train_loss,
                    "loss_components": {
                        name: float(value / max(1, total_rows)) for name, value in sorted(loss_sums.items())
                    },
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "eval": eval_report,
                    "confidence_report": str(confidence_report_path),
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
            epoch_checkpoint = save_fragment_checkpoint(
                output_dir=output_dir,
                name=f"fragment_attachment_expert_epoch_{epoch_number}.pth",
                fragment_expert=fragment_expert,
                anchor_vocab=anchor_vocab,
                encoder=encoder,
                args=args,
                metrics=metrics,
                epoch=epoch_number,
                eval_report=eval_report,
                confidence_report_path=confidence_report_path,
            )
            last_checkpoint = save_fragment_checkpoint(
                output_dir=output_dir,
                name="fragment_attachment_expert_last.pth",
                fragment_expert=fragment_expert,
                anchor_vocab=anchor_vocab,
                encoder=encoder,
                args=args,
                metrics=metrics,
                epoch=epoch_number,
                eval_report=eval_report,
                confidence_report_path=confidence_report_path,
            )
            selection_score = fragment_eval_selection_score(eval_report)
            metrics["latest_checkpoint"] = str(last_checkpoint)
            metrics["latest_epoch_checkpoint"] = str(epoch_checkpoint)
            metrics["latest_selection_score"] = float(selection_score)
            if selection_score > best_selection_score:
                best_selection_score = float(selection_score)
                best_checkpoint_path = str(output_dir / "fragment_attachment_expert_best.pth")
                shutil.copyfile(epoch_checkpoint, best_checkpoint_path)
                metrics["best_checkpoint"] = best_checkpoint_path
                metrics["best_epoch"] = int(epoch_number)
                metrics["best_selection_score"] = float(selection_score)
            training_state_checkpoint = save_training_state_checkpoint(
                output_dir=output_dir,
                fragment_expert=fragment_expert,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                anchor_vocab=anchor_vocab,
                encoder=encoder,
                args=args,
                metrics=metrics,
                epoch=epoch_number,
                best_selection_score=best_selection_score,
                best_checkpoint_path=best_checkpoint_path,
            )
            metrics["latest_training_state_checkpoint"] = str(training_state_checkpoint)
            (output_dir / "metrics.partial.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        ddp_barrier(ddp)
    elapsed_seconds = time.time() - start
    local_peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if device.type == "cuda"
        else 0.0
    )
    per_gpu_peak_memory_gb = all_gather_float(local_peak_memory, ddp, device)
    if is_main_process(ddp):
        metrics["elapsed_seconds"] = elapsed_seconds
        metrics["runtime"] = {
            "elapsed_seconds": float(elapsed_seconds),
            "train_rows": int(len(train_df)),
            "eval_rows": int(len(eval_df)),
            "throughput_samples_per_second": float(
                (int(len(train_df)) * len(metrics["epochs"]) + int(len(eval_df)) * len(metrics["epochs"]))
                / max(1e-6, float(elapsed_seconds))
            ),
            "per_gpu_peak_memory_gb": per_gpu_peak_memory_gb,
        }
        metrics["model_scale"] = build_model_scale_report(
            encoder=encoder,
            expert=fragment_expert,
            input_dim=int(encoder.n_features),
            input_size=int(getattr(model_args, "input_size", 0) or 0),
            args=args,
            training_stage=str(args.training_stage),
            debug_only=bool(args.allow_ungated_debug_run),
            ddp=ddp,
            amp_enabled=bool(amp_enabled),
            runtime=metrics["runtime"],
        )
        final_checkpoint = save_fragment_checkpoint(
            output_dir=output_dir,
            name="fragment_attachment_expert.pth",
            fragment_expert=fragment_expert,
            anchor_vocab=anchor_vocab,
            encoder=encoder,
            args=args,
            metrics=metrics,
            epoch=len(metrics["epochs"]),
            eval_report=metrics["epochs"][-1]["eval"] if metrics["epochs"] else {},
            confidence_report_path=(
                Path(metrics["epochs"][-1]["confidence_report"])
                if metrics["epochs"] and metrics["epochs"][-1].get("confidence_report")
                else None
            ),
        )
        metrics["final_checkpoint"] = str(final_checkpoint)
        if not metrics.get("best_checkpoint") and final_checkpoint.exists():
            best_checkpoint_path = str(output_dir / "fragment_attachment_expert_best.pth")
            shutil.copyfile(final_checkpoint, best_checkpoint_path)
            metrics["best_checkpoint"] = best_checkpoint_path
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))
    ddp_barrier(ddp)
    cleanup_distributed(ddp)


if __name__ == "__main__":
    main()
