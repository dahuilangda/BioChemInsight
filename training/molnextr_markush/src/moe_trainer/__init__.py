"""MolNexTR MoE trainer package.

Re-exports the public API that was originally in ``tools/train_moe.py``.
The backward-compatible shim at ``tools/train_moe.py`` delegates here.
"""
from __future__ import annotations

# Library modules — import in dependency order
from .args import (  # noqa: F401
    parse_args,
    set_train_args,
    validate_direct_sidecar_training_contract,
)
from .model import (  # noqa: F401
    EncoderMoETrainingModel,
    configure_encoder_finetuning,
    configure_decoder_finetuning,
    decoder_scope_parameter_prefixes,
    decoder_trainability_manifest,
    init_distributed,
    validate_optimizer_parameter_coverage,
)
from .runtime import (  # noqa: F401
    LabelBalancedSampler,
    auto_rows_per_label,
    checkpoint_provenance,
    is_rank0,
    linear_warmup_cosine,
    make_summary_writer,
    rank0_print,
    save_moe_state,
)
from .losses import (  # noqa: F401
    decoded_prediction_smiles,
    finite_metric,
    fmt_metric,
    group_relative_rloo_loss,
    reward_weighted_counterfactual_dpo_loss,
    searched_terminal_action_loss,
    terminal_action_pair_margin_loss,
    terminal_dummy_edge_expected_reward,
)
from .data import (  # noqa: F401
    load_frozen_source_partition,
    load_real_original_training_frame,
    validate_attachment_set_capacity,
    validate_dataframe_contract,
)
from .calibration import (  # noqa: F401
    audit_sidecar_threshold,
    calibrate_sidecar_thresholds,
    calibrate_threshold,
)
from .entry import main  # noqa: F401

__all__ = [
    "EncoderMoETrainingModel",
    "LabelBalancedSampler",
    "audit_sidecar_threshold",
    "auto_rows_per_label",
    "calibrate_sidecar_thresholds",
    "calibrate_threshold",
    "checkpoint_provenance",
    "configure_decoder_finetuning",
    "configure_encoder_finetuning",
    "decoded_prediction_smiles",
    "decoder_scope_parameter_prefixes",
    "decoder_trainability_manifest",
    "finite_metric",
    "fmt_metric",
    "group_relative_rloo_loss",
    "init_distributed",
    "is_rank0",
    "linear_warmup_cosine",
    "load_frozen_source_partition",
    "load_real_original_training_frame",
    "main",
    "make_summary_writer",
    "parse_args",
    "rank0_print",
    "reward_weighted_counterfactual_dpo_loss",
    "save_moe_state",
    "searched_terminal_action_loss",
    "set_train_args",
    "terminal_action_pair_margin_loss",
    "terminal_dummy_edge_expected_reward",
    "validate_attachment_set_capacity",
    "validate_dataframe_contract",
    "validate_direct_sidecar_training_contract",
    "validate_optimizer_parameter_coverage",
]
