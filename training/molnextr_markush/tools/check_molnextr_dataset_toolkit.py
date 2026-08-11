from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.molnextr_dataset_toolkit import (  # noqa: E402
    CandidatePlanWindow,
    ExplicitBucketPlanWindow,
    FormalSimDatasetConfig,
    MolNexTRDatasetPipeline,
    RetryPolicy,
    parse_bucket_window_overrides,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the class-based MolNexTR dataset toolkit contract.")
    parser.add_argument("--dataset-id", default="molnextr_moe_production_v1")
    parser.add_argument("--markush-shard", type=int, default=0)
    parser.add_argument("--markush-rows", type=int, default=500)
    parser.add_argument("--candidate-rows-per-bucket", type=int, default=650)
    parser.add_argument(
        "--bucket-window-overrides",
        default="",
        help="Explicit Markush per-bucket source windows for scheduled terminal tail shards.",
    )
    parser.add_argument("--renderer-seed-retries", type=int, default=3)
    parser.add_argument("--require-existing-gates", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    pipeline = MolNexTRDatasetPipeline(FormalSimDatasetConfig(dataset_id=str(args.dataset_id)))
    explicit_windows = parse_bucket_window_overrides(str(args.bucket_window_overrides))
    if explicit_windows:
        window = ExplicitBucketPlanWindow(
            int(args.markush_shard),
            accepted_rows_per_bucket=int(args.markush_rows),
            bucket_windows=explicit_windows,
        )
    else:
        window = CandidatePlanWindow.from_targets(
            int(args.markush_shard),
            accepted_rows_per_bucket=int(args.markush_rows),
            candidate_rows_per_bucket=int(args.candidate_rows_per_bucket),
        )
    retry = RetryPolicy(attempt_count=int(args.renderer_seed_retries))
    commands = pipeline.markush_commands(
        int(args.markush_shard),
        rows_per_bucket=int(args.markush_rows),
        seed=2026062208 + int(args.markush_shard),
        candidate_rows_per_bucket=int(args.candidate_rows_per_bucket),
        renderer_seed_retries=int(args.renderer_seed_retries),
        bucket_window_overrides=str(args.bucket_window_overrides),
    )
    command_stages = [command.stage for command in commands]
    expected_stages = [
        "generate_raw_markush_cdk_svg_formal_nonlinear",
        "validate_raw_markush_schema",
        "audit_raw_markush_formal_nonlinear_warp",
        "prepare_markush_accepted_candidate",
        "validate_accepted_markush_schema",
        "audit_accepted_markush_formal_nonlinear_warp",
        "audit_accepted_markush_pose_alignment",
        "audit_markush_substitution_anchor_contract",
    ]
    blockers: list[str] = []
    if command_stages != expected_stages:
        blockers.append("markush command stage order changed")
    if not explicit_windows and window.plan_end - window.plan_start != int(args.candidate_rows_per_bucket) * 5:
        blockers.append("candidate plan window size is inconsistent with five R-count buckets")
    if explicit_windows:
        for bucket, (start, end) in explicit_windows.items():
            if end - start < int(args.markush_rows):
                blockers.append(f"explicit bucket window {bucket} has fewer candidates than accepted target")
    contract = retry.contract()
    if int(args.renderer_seed_retries) > 1 and contract.get("thresholds_unchanged") is not True:
        blockers.append("retry contract does not prove thresholds are unchanged")
    if int(args.renderer_seed_retries) > 1 and contract.get("coordinate_recomputed_from_selected_renderer_output") is not True:
        blockers.append("retry contract does not require coordinate recomputation")

    gate_results = [result.to_dict(root=pipeline.config.root) for result in pipeline.markush_gate_results(int(args.markush_shard))]
    if args.require_existing_gates:
        for result in gate_results:
            if result["present"] is not True or result["passed"] is not True:
                blockers.append(f"required gate is not passing: {result['path']}")

    report = {
        "schema_version": "molnextr_dataset_toolkit_contract_check_v1",
        "dataset_id": str(args.dataset_id),
        "passed": not blockers,
        "blockers": blockers,
        "candidate_plan_window": window.to_dict(),
        "retry_contract": contract,
        "commands": [command.to_dict(root=pipeline.config.root) for command in commands],
        "gate_results": gate_results,
        "formal_training_start_allowed": False,
        "policy": {
            "toolkit_wraps_existing_formal_generators": True,
            "no_new_renderer_or_image_method": True,
            "retry_is_renderer_level_layout_diversity_not_threshold_relaxation": True,
            "formal_gates_remain_external_and_machine_readable": True,
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
