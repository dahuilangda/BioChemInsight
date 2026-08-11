from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COUNT_BUCKETS = ["1", "2", "3-4", "5-8", "9+"]


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def counts(value: Any) -> dict[str, int]:
    raw = value if isinstance(value, dict) else {}
    return {bucket: int(raw.get(bucket) or 0) for bucket in COUNT_BUCKETS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Report whether equal per-bucket Markush targets match current raw capacity.")
    parser.add_argument("--candidate-quality-audit", required=True)
    parser.add_argument("--capacity-risk", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-per-bucket", type=int, default=6000)
    parser.add_argument("--architecture-train-total", type=int, default=20000)
    parser.add_argument("--architecture-calibration-total", type=int, default=10000)
    args = parser.parse_args()

    quality = load_json(args.candidate_quality_audit)
    capacity = load_json(args.capacity_risk)
    available = counts(quality.get("available_by_bucket"))
    selected = counts(quality.get("selected_by_bucket"))
    extra = counts(quality.get("extra_capacity_by_bucket"))
    low_risk = counts((quality.get("risk_counts_by_bucket") or {}).get("low_risk_by_simple_features"))
    projected = counts(capacity.get("projected_accepted_by_bucket"))
    target = {bucket: int(args.target_per_bucket) for bucket in COUNT_BUCKETS}

    blockers: list[str] = []
    findings: list[str] = []
    for bucket in COUNT_BUCKETS:
        if available[bucket] < target[bucket]:
            blockers.append(f"bucket {bucket} raw source-document capacity {available[bucket]} < target {target[bucket]}")
        if projected[bucket] < target[bucket]:
            blockers.append(f"bucket {bucket} projected accepted {projected[bucket]} < target {target[bucket]}")
        if low_risk[bucket] < target[bucket]:
            findings.append(
                f"bucket {bucket} simple low-risk candidate count {low_risk[bucket]} < target {target[bucket]}; "
                "forcing equal accepted count may require accepting many structurally hard/fragmented candidates"
            )
        if extra[bucket] <= 100:
            findings.append(f"bucket {bucket} has only {extra[bucket]} unused source-document candidates")

    total_required = int(args.architecture_train_total) + int(args.architecture_calibration_total)
    proportional_targets = {}
    total_available = sum(available.values())
    for bucket in COUNT_BUCKETS:
        proportional_targets[bucket] = int(round(total_required * (available[bucket] / total_available))) if total_available else 0
    difference = total_required - sum(proportional_targets.values())
    for bucket in sorted(COUNT_BUCKETS, key=lambda item: available[item], reverse=True):
        if difference == 0:
            break
        proportional_targets[bucket] += 1 if difference > 0 else -1
        difference += -1 if difference > 0 else 1

    report = {
        "schema_version": "markush_standards_risk_report_v1",
        "candidate_quality_audit": str(args.candidate_quality_audit),
        "capacity_risk": str(args.capacity_risk),
        "current_equal_target_per_bucket": int(args.target_per_bucket),
        "architecture_total_required": total_required,
        "available_source_documents_by_bucket": available,
        "selected_candidates_by_bucket": selected,
        "unused_source_documents_by_bucket": extra,
        "simple_low_risk_candidates_by_bucket": low_risk,
        "projected_accepted_by_bucket": projected,
        "equal_bucket_target_supported": not blockers,
        "blockers": blockers,
        "findings": findings,
        "alternative_for_review": {
            "name": "source_capacity_proportional_30k",
            "description": (
                "Keep the 30k architecture-evaluation scale, but make bucket targets proportional to source-document "
                "capacity instead of forcing 6k per bucket. This preserves large validation scale while avoiding "
                "an impossible or low-quality equal-bucket target for bucket 1."
            ),
            "candidate_targets_by_bucket": proportional_targets,
            "requires_user_or_project_standards_acceptance": True,
            "does_not_change_current_gates_by_itself": True,
        },
        "policy": {
            "report_only": True,
            "does_not_relax_thresholds": True,
            "requires_explicit_standards_decision_before_replanning": True,
            "no_training_allowed_from_this_report": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
