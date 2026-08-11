"""Backward-compatibility shim.

The real implementation now lives in
``training.molnextr_markush.src.moe_trainer``.  This file exists so that
existing shell scripts (``run_moe_production.sh``, ``run_bond_finetune.sh``)
and tests (``tests/test_moe_*.py``) that reference
``training.molnextr_markush.tools.train_moe`` continue to work.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Re-export the full public API.
from training.molnextr_markush.src.moe_trainer import (  # noqa: F401
    EncoderMoETrainingModel,
    LabelBalancedSampler,
    audit_sidecar_threshold,
    auto_rows_per_label,
    calibrate_sidecar_thresholds,
    calibrate_threshold,
    checkpoint_provenance,
    configure_decoder_finetuning,
    configure_encoder_finetuning,
    decoded_prediction_smiles,
    decoder_scope_parameter_prefixes,
    decoder_trainability_manifest,
    finite_metric,
    fmt_metric,
    group_relative_rloo_loss,
    init_distributed,
    is_rank0,
    linear_warmup_cosine,
    load_frozen_source_partition,
    load_real_original_training_frame,
    main,
    make_summary_writer,
    parse_args,
    rank0_print,
    reward_weighted_counterfactual_dpo_loss,
    save_moe_state,
    searched_terminal_action_loss,
    set_train_args,
    terminal_action_pair_margin_loss,
    terminal_dummy_edge_expected_reward,
    validate_attachment_set_capacity,
    validate_dataframe_contract,
    validate_direct_sidecar_training_contract,
    validate_optimizer_parameter_coverage,
)

# Private names and constants referenced by existing tests.
from training.molnextr_markush.src.moe_trainer._common import (  # noqa: F401
    ATOM_FORMAT,
    DATA_ROOT,
    EXPERT_NAMES,
    FORMAT_INFO,
    EOS_ID,
    MOE_DATA_CONTRACT_VERSION,
    PRODUCTION_RELATIVE_ROOT,
)
from training.molnextr_markush.src.moe_trainer.calibration import (  # noqa: F401
    _df_for_router_calibration,
    _progress,
    _router_probs_for_df,
    _sidecar_acceptance_from_probs,
    _sidecar_threshold_vector,
)

if __name__ == "__main__":
    main()
