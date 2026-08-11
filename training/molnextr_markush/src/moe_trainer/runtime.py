"""Distributed-training runtime: samplers, checkpointing, schedulers.

Extracted from the original ``tools/train_moe.py`` monolith.
"""
from __future__ import annotations

from ._common import *  # noqa: F401,F403

def is_rank0(rank: int) -> bool:
    return int(rank) == 0


def rank0_print(rank: int, *parts) -> None:
    if is_rank0(rank):
        print(*parts, flush=True)


def checkpoint_provenance(path: str | None) -> dict | None:
    if not path:
        return None
    resolved = os.path.abspath(path)
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": resolved,
        "size_bytes": int(os.path.getsize(resolved)),
        "sha256": digest.hexdigest(),
    }


class LabelBalancedSampler(Sampler[int]):
    """Deterministic label-balanced sampler with DDP rank slicing.

    Optional per-row ``weights`` enable size-aware oversampling within a label
    (e.g. upweight tiny fragments, which are ~2% of data and the main source of
    fragment over-generation, so the fragment expert actually learns them).
    """

    def __init__(
        self,
        labels,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        rows_per_label: int = 0,
        shuffle: bool = True,
        weights=None,
        focus_fraction: float = 0.0,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.label_to_indices = {
            int(label): np.flatnonzero(self.labels == int(label)).astype(np.int64)
            for label in sorted(np.unique(self.labels).tolist())
        }
        if not self.label_to_indices:
            raise ValueError("LabelBalancedSampler got no labels")
        if int(rows_per_label or 0) > 0:
            self.rows_per_label = int(rows_per_label)
        else:
            self.rows_per_label = min(len(indices) for indices in self.label_to_indices.values())
        self.global_size = len(self.label_to_indices) * self.rows_per_label
        self.total_size = (self.global_size // self.num_replicas) * self.num_replicas
        if self.total_size <= 0:
            raise ValueError("LabelBalancedSampler total size is zero")
        self.num_samples = self.total_size // self.num_replicas
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.focus_fraction = min(0.75, max(0.0, float(focus_fraction or 0.0)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        pieces = []
        for label, indices in sorted(self.label_to_indices.items()):
            weighted = False
            w = None
            if self.weights is not None:
                w = self.weights[indices].astype(np.float64)
                w = np.where(w > 0, w, 1e-6)
                weighted = not np.allclose(w, w[0])
            focus_rows = (
                min(self.rows_per_label, int(round(self.rows_per_label * self.focus_fraction)))
                if weighted
                else 0
            )
            coverage_rows = self.rows_per_label - focus_rows
            # Treat epochs as contiguous windows over independently shuffled
            # full-corpus passes. This guarantees every row is visited before a
            # new coverage pass starts. A separate weighted, no-replacement
            # focus slice keeps real/tiny/wavy hard domains present every epoch
            # without replacing the natural-coverage stream.
            cursor = self.epoch * coverage_rows
            cycle_id, offset = divmod(cursor, len(indices))
            remaining = coverage_rows
            cycles = []
            while remaining > 0:
                rng = np.random.default_rng(
                    self.seed + label * 9176 + cycle_id * 104729
                )
                permutation = rng.permutation(indices)
                take = min(remaining, len(indices) - offset)
                cycles.append(permutation[offset:offset + take])
                remaining -= take
                cycle_id += 1
                offset = 0
            coverage = np.concatenate(cycles) if cycles else np.empty(0, dtype=np.int64)
            if focus_rows > 0:
                focus_rng = np.random.default_rng(
                    self.seed + self.epoch * 1009 + label * 9176 + 48611
                )
                available_mask = ~np.isin(indices, coverage, assume_unique=False)
                focus_pool = indices[available_mask]
                focus_weights = w[available_mask]
                if len(focus_pool) < focus_rows:
                    focus_pool = indices
                    focus_weights = w
                focus = focus_rng.choice(
                    focus_pool,
                    size=focus_rows,
                    replace=focus_rows > len(focus_pool),
                    p=focus_weights / focus_weights.sum(),
                )
                picked = np.concatenate([coverage, focus])
            else:
                picked = coverage
            pieces.append(picked)
        merged = np.concatenate(pieces)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch * 3571 + 8191)
            merged = merged[rng.permutation(len(merged))]
        merged = merged[: self.total_size]
        rank_indices = merged[self.rank:self.total_size:self.num_replicas]
        return iter(int(x) for x in rank_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def report(self) -> dict:
        return {
            "strategy": "label_balanced",
            "rows_per_label_per_epoch": int(self.rows_per_label),
            "global_samples_per_epoch": int(self.total_size),
            "rank_samples_per_epoch": int(self.num_samples),
            "label_counts_available": {
                str(label): int(len(indices)) for label, indices in self.label_to_indices.items()
            },
            "num_replicas": int(self.num_replicas),
            "focus_fraction": float(self.focus_fraction),
            "coverage_policy": "natural_full_corpus_windows_plus_weighted_focus_without_replacement",
        }


def auto_rows_per_label(label_counts: dict[int, int], cap: int) -> int:
    """Coverage-aware epoch size with a hard upper bound.

    Small complete sets must not limit sidecar coverage, but a large complete
    set must not silently disable ``--auto-epoch-rows-cap`` either.  The
    sidecar scale is therefore capped directly; smaller labels are sampled in
    deterministic coverage cycles by :class:`LabelBalancedSampler`.
    """
    counts = {int(label): int(count) for label, count in label_counts.items()}
    if not counts or min(counts.values()) <= 0:
        raise ValueError(f"label counts must be positive, got {counts}")
    sidecar_counts = [count for label, count in counts.items() if label > 0]
    reference = max(sidecar_counts) if sidecar_counts else max(counts.values())
    return max(1, min(reference, max(1, int(cap))))


def make_summary_writer(args, rank: int):
    if not args.tensorboard or not is_rank0(rank):
        return None
    log_dir = args.tensorboard_dir or os.path.join(args.output_dir, "tensorboard")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        try:
            from tensorboardX import SummaryWriter
        except Exception as exc:
            raise RuntimeError(
                "TensorBoard logging requested but neither tensorboard nor tensorboardX is installed."
            ) from exc
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)


def linear_warmup_cosine(opt, warmup_steps, total_steps, min_lr_frac: float = 0.0):
    from torch.optim.lr_scheduler import LambdaLR
    floor = min(max(float(min_lr_frac or 0.0), 0.0), 0.95)
    def fn(step):
        if step < warmup_steps:
            return max(floor, max(1e-3, step / max(1, warmup_steps)))
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
        return floor + (1.0 - floor) * cosine
    return LambdaLR(opt, lr_lambda=fn)


def save_moe_state(model_for_loss, model_args, output_dir: str, *, tag: str | None = None,
                   confidence_head: bool = False, save_expert0: bool = False,
                   expert_kind: str = "lora",
                   encoder: torch.nn.Module | None = None,
                   encoder_trainable_names: list[str] | None = None,
                   encoder_finetune_stages: int = 0) -> dict[str, str | None]:
    """Save MoE checkpoints in the format consumed by ``model.py``.

    ``expert_kind="lora"`` (default): write only the LoRA adapter params
    (``moe_adapter.pth``) + router (+ optional confidence head). The frozen base
    weights live in the LoRAMoELinear buffers and are NOT duplicated — inference
    reloads them from ``molnextr_best.pth``. Tiny checkpoints (~MB, not 3×32 MB).
    ``expert_kind="full"``: legacy 3-full-decoder-copy path.
    """
    save_dir = output_dir if tag is None else os.path.join(output_dir, "checkpoints", tag)
    os.makedirs(save_dir, exist_ok=True)
    encoder_path = None
    selected_encoder_names = list(encoder_trainable_names or [])
    if encoder is not None and selected_encoder_names:
        encoder_state = encoder.state_dict()
        missing = [name for name in selected_encoder_names if name not in encoder_state]
        if missing:
            raise KeyError(
                f"trainable encoder parameters missing from state_dict: {missing[:5]}"
            )
        encoder_path = os.path.join(save_dir, "moe_encoder.pth")
        torch.save(
            {
                "encoder": {
                    name: encoder_state[name].detach().cpu()
                    for name in selected_encoder_names
                },
                "trainable_parameter_names": selected_encoder_names,
                "finetune_stages": int(encoder_finetune_stages),
            },
            encoder_path,
        )
    router_path = os.path.join(save_dir, "moe_router.pt")
    torch.save(
        {"router": model_for_loss.router.state_dict(),
         "feature_dim": model_args.encoder_dim,
         "num_experts": int(model_for_loss.num_experts),
         "router_kind": str(getattr(model_for_loss, "router_kind", "mean_mlp")),
         "token_fusion_mode": str(getattr(model_for_loss, "token_fusion_mode", "fixed")),
         "token_fusion_initial_sidecar_weight": float(
             getattr(model_for_loss, "token_fusion_initial_sidecar_weight", 0.2)
         ),
         "token_fusion_dispatch": str(
             getattr(model_for_loss, "token_fusion_dispatch", "soft")
         ),
         "token_fusion_hard_threshold": float(
             getattr(model_for_loss, "token_fusion_hard_threshold", 0.5)
         ),
         "specialist_ownership_scope": str(
             getattr(model_for_loss, "specialist_ownership_scope", "full")
         ),
         "attachment_set_enabled": bool(
             getattr(model_for_loss, "attachment_set_enabled", False)
         ),
         "attachment_set_feature_mode": str(
             getattr(model_for_loss, "attachment_set_feature_mode", "single_scale")
         ),
         "attachment_set_heatmap_logit_scale": float(
             getattr(model_for_loss, "attachment_set_heatmap_logit_scale", 4.0)
         ),
         "attachment_set_heatmap_prior_precision": float(
             getattr(model_for_loss, "attachment_set_heatmap_prior_precision", 8.0)
         ),
         "attachment_set_heads": (
             model_for_loss.attachment_set_heads.state_dict()
             if len(getattr(model_for_loss, "attachment_set_heads", ())) else None
         ),
         "fragment_terminal_action_head_enabled": bool(
             getattr(
                 model_for_loss,
                 "fragment_terminal_action_head_enabled",
                 False,
             )
         ),
         "fragment_terminal_action_head_hidden_dim": int(
             getattr(
                 model_for_loss,
                 "fragment_terminal_action_head_hidden_dim",
                 128,
             )
         ),
         "fragment_structured_terminal_edge_enabled": bool(
             getattr(
                 model_for_loss,
                 "fragment_structured_terminal_edge_enabled",
                 False,
             )
         ),
         "fragment_terminal_action_head": (
             model_for_loss.fragment_terminal_action_head.state_dict()
             if getattr(
                 model_for_loss,
                 "fragment_terminal_action_head",
                 None,
             ) is not None
             else None
         ),
         "decouple_fusion_policy_optimization": bool(
             getattr(model_for_loss, "decouple_fusion_policy_optimization", True)
         ),
         "one_sided_fusion_oracle": bool(
             getattr(model_for_loss, "one_sided_fusion_oracle", True)
         ),
         "token_fusion_gates": (
             model_for_loss.token_fusion_gates.state_dict()
             if len(getattr(model_for_loss, "token_fusion_gates", ())) else None
         )},
        router_path,
    )
    confidence_path = None
    if confidence_head and model_for_loss.confidence_head is not None:
        confidence_path = os.path.join(save_dir, "moe_confidence.pt")
        torch.save({"confidence_head": model_for_loss.confidence_head.state_dict()}, confidence_path)

    if expert_kind == "lora":
        from utils.MolNexTR.moe import lora_state_dict
        adapter_path = os.path.join(save_dir, "moe_adapter.pth")
        torch.save(
            {"decoder": lora_state_dict(model_for_loss._decoder),
             "lora_rank": int(model_for_loss.lora_rank),
             "lora_alpha": float(model_for_loss.lora_alpha),
             "num_experts": int(model_for_loss.num_experts),
             "num_adapter_experts": int(getattr(model_for_loss, "num_adapter_experts", 0))},
            adapter_path,
        )
        return {
            "adapter_path": adapter_path, "expert0_path": None,
            "expert1_path": None, "expert2_path": None,
            "router_path": router_path, "confidence_path": confidence_path,
            "regime_path": None, "encoder_path": encoder_path,
        }

    if expert_kind == "full_mixture":
        # expert0 is the frozen complete decoder (byte-identical base, reloaded
        # from molnextr_best.pth at inference) — skip unless debugging. In the
        # per-sidecar architecture, expert1=markush and expert2=fragment; legacy
        # collapsed runs may only have expert1.
        expert0_path = None
        if save_expert0:
            expert0_path = os.path.join(save_dir, "moe_expert0.pth")
            torch.save({"decoder": model_for_loss.experts[0].state_dict()}, expert0_path)
        expert1_path = os.path.join(save_dir, "moe_expert1.pth")
        torch.save({"decoder": model_for_loss.experts[1].state_dict()}, expert1_path)
        expert2_path = None
        if len(model_for_loss.experts) > 2:
            expert2_path = os.path.join(save_dir, "moe_expert2.pth")
            torch.save({"decoder": model_for_loss.experts[2].state_dict()}, expert2_path)
        return {
            "adapter_path": None, "expert0_path": expert0_path,
            "expert1_path": expert1_path, "expert2_path": expert2_path,
            "router_path": router_path, "confidence_path": confidence_path,
            "regime_path": None, "encoder_path": encoder_path,
        }

    # legacy full-decoder path
    expert0_path = None
    if save_expert0:
        expert0_path = os.path.join(save_dir, "moe_expert0.pth")
        torch.save({"decoder": model_for_loss.experts[0].state_dict()}, expert0_path)
    expert1_path = os.path.join(save_dir, "moe_expert1.pth")
    expert2_path = os.path.join(save_dir, "moe_expert2.pth")
    torch.save({"decoder": model_for_loss.experts[1].state_dict()}, expert1_path)
    torch.save({"decoder": model_for_loss.experts[2].state_dict()}, expert2_path)
    regime_path = None
    if getattr(model_for_loss, "regime_conditioning", False) and model_for_loss.regime_emb is not None:
        regime_path = os.path.join(save_dir, "moe_regime.pt")
        torch.save({"regime_emb": model_for_loss.regime_emb.state_dict()}, regime_path)
    return {
        "expert0_path": expert0_path, "expert1_path": expert1_path,
        "expert2_path": expert2_path, "router_path": router_path,
        "confidence_path": confidence_path, "regime_path": regime_path,
        "adapter_path": None, "encoder_path": encoder_path,
    }


