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


def bucket_counts(mapping: Any) -> dict[str, int]:
    raw = mapping if isinstance(mapping, dict) else {}
    return {bucket: int(raw.get(bucket) or 0) for bucket in COUNT_BUCKETS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate whether a Markush candidate plan can meet accepted bucket targets.")
    parser.add_argument("--candidate-readiness", required=True)
    parser.add_argument("--probe-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--safety-margin", type=float, default=1.05)
    args = parser.parse_args()

    readiness = load_json(args.candidate_readiness)
    manifest = load_json(args.probe_manifest)
    targets = bucket_counts(readiness.get("bucket_targets"))
    selected = bucket_counts(readiness.get("selected_candidate_rows_by_bucket"))
    accepted = bucket_counts(manifest.get("counts_by_r_tag_bucket"))
    probe_candidate_plan = manifest.get("candidate_plan") if isinstance(manifest.get("candidate_plan"), dict) else {}
    probe_selected = bucket_counts(probe_candidate_plan.get("selected_candidate_rows_by_bucket"))
    projected: dict[str, int] = {}
    acceptance_rate: dict[str, float] = {}
    blockers: list[str] = []
    warnings: list[str] = []

    for bucket in COUNT_BUCKETS:
        denominator = int(probe_selected.get(bucket) or 0)
        rate = float(accepted[bucket]) / float(denominator) if denominator > 0 else 0.0
        acceptance_rate[bucket] = rate
        projected[bucket] = int(selected[bucket] * rate)
        required_with_margin = int(targets[bucket] * float(args.safety_margin))
        if projected[bucket] < targets[bucket]:
            blockers.append(
                f"bucket {bucket} projected accepted {projected[bucket]} < target {targets[bucket]} "
                f"from observed rate {rate:.3f}"
            )
        elif projected[bucket] < required_with_margin:
            warnings.append(
                f"bucket {bucket} projected accepted {projected[bucket]} is below safety-margin target "
                f"{required_with_margin} from observed rate {rate:.3f}"
            )

    report = {
        "schema_version": "markush_generation_capacity_risk_v1",
        "candidate_readiness": str(args.candidate_readiness),
        "probe_manifest": str(args.probe_manifest),
        "candidate_selected_by_bucket": selected,
        "accepted_targets_by_bucket": targets,
        "probe_selected_by_bucket": probe_selected,
        "probe_accepted_by_bucket": accepted,
        "observed_acceptance_rate_by_bucket": acceptance_rate,
        "projected_accepted_by_bucket": projected,
        "safety_margin": float(args.safety_margin),
        "passed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "policy": {
            "risk_audit_only": True,
            "does_not_relax_targets": True,
            "failed_projection_requires_more_candidates_or_replanning_before_architecture_claims": True,
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
