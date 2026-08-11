from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_DECISIONS = {
    "add_more_raw_data",
    "source_capacity_proportional_30k",
    "statistically_justified_smaller_eval",
    "keep_equal_bucket_target",
}


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate Markush data replanning on an explicit standards decision.")
    parser.add_argument("--standards-risk-report", required=True)
    parser.add_argument("--decision", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    risk = load_json(args.standards_risk_report)
    decision = load_json(args.decision) if args.decision else {}
    blockers: list[str] = []
    warnings: list[str] = []

    equal_supported = risk.get("equal_bucket_target_supported") is True
    if not equal_supported and not decision:
        blockers.append("equal-bucket target is not supported and no explicit standards decision was provided")

    selected_decision = str(decision.get("decision") or "").strip()
    if decision:
        if selected_decision not in ALLOWED_DECISIONS:
            blockers.append(f"decision {selected_decision!r} is not one of {sorted(ALLOWED_DECISIONS)}")
        if decision.get("accepted") is not True:
            blockers.append("standards decision is not explicitly accepted")
        if not str(decision.get("rationale") or "").strip():
            blockers.append("standards decision rationale is missing")
        if selected_decision == "keep_equal_bucket_target" and not equal_supported:
            warnings.append("equal-bucket target was kept despite current capacity-risk blockers")
        if selected_decision == "source_capacity_proportional_30k":
            alt = risk.get("alternative_for_review") if isinstance(risk.get("alternative_for_review"), dict) else {}
            if alt.get("name") != "source_capacity_proportional_30k":
                blockers.append("risk report does not contain the source_capacity_proportional_30k alternative")
            if decision.get("candidate_targets_by_bucket") != alt.get("candidate_targets_by_bucket"):
                blockers.append("decision candidate_targets_by_bucket does not match the risk-report alternative")

    report = {
        "schema_version": "markush_standards_decision_gate_v1",
        "standards_risk_report": str(args.standards_risk_report),
        "decision": str(args.decision),
        "equal_bucket_target_supported": equal_supported,
        "decision_provided": bool(decision),
        "selected_decision": selected_decision,
        "passed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "policy": {
            "requires_explicit_decision_when_equal_bucket_risk_fails": True,
            "does_not_generate_data": True,
            "does_not_accept_training_data": True,
            "no_training_allowed_from_report_only_decisions": True,
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
