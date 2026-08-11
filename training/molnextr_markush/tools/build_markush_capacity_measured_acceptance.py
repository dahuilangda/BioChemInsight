from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Markush-only measured-training acceptance report from strict machine data gates."
    )
    parser.add_argument("--validation", required=True)
    parser.add_argument("--pose-alignment", required=True)
    parser.add_argument("--source-leak", required=True)
    parser.add_argument("--substitution-anchor-contract", required=True)
    parser.add_argument("--assistant-visual-review", required=True)
    parser.add_argument("--visual-risk-report", required=True)
    parser.add_argument("--capacity-scale", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-preservation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    validation = load_json(args.validation)
    pose = load_json(args.pose_alignment)
    source_leak = load_json(args.source_leak)
    substitution = load_json(args.substitution_anchor_contract)
    assistant_visual = load_json(args.assistant_visual_review)
    visual_risk = load_json(args.visual_risk_report)
    capacity = load_json(args.capacity_scale)
    split_manifest = load_json(args.split_manifest)
    split_preservation = load_json(args.split_preservation)

    blockers: list[str] = []
    row_count = int(validation.get("row_count") or 0)
    if validation.get("trainable") is not True or int(validation.get("invalid_rows") or 0) != 0:
        blockers.append("schema validation is not trainable")
    if pose.get("passed") is not True:
        blockers.append("pose alignment did not pass")
    if source_leak.get("passed") is not True:
        blockers.append("source leak did not pass")
    if substitution.get("passed") is not True or int(substitution.get("failed_rows") or 0) != 0:
        blockers.append("substitution-anchor contract did not pass")
    if int(substitution.get("row_count") or 0) != row_count:
        blockers.append("substitution-anchor row_count does not match validation row_count")
    if assistant_visual.get("strict_machine_visual_acceptance_passed") is not True:
        blockers.append("strict machine visual acceptance did not pass")
    if int(visual_risk.get("background_context_rows") or 0) != row_count:
        blockers.append("document_context/background_realism does not cover every row")
    if capacity.get("formal_training_candidate_allowed") is not True:
        blockers.append("capacity-maximized scale does not allow measured/formal candidate training")
    split_stage = str(split_manifest.get("stage") or "").strip()
    split_is_candidate = split_stage == "candidate" and split_manifest.get("accepted_for_training") is not True
    split_is_formal = split_stage == "formal" and split_manifest.get("accepted_for_training") is True
    if not split_is_candidate and not split_is_formal:
        blockers.append(
            "Markush-only split manifest must be candidate/not-accepted or formal/accepted_for_training"
        )
    if split_preservation.get("passed") is not True:
        blockers.append("Markush-only split preservation did not pass")

    report = {
        "schema_version": "markush_capacity_measured_acceptance_v1",
        "accepted": not blockers,
        "measured_sidecar_smoke_allowed": not blockers,
        "formal_training_allowed": False,
        "formal_training_start_allowed": False,
        "formal_blockers": [
            "not a full formal preflight: confidence/router-negative/fragment/ordinary branch evidence still required"
            if split_is_formal
            else "split remains candidate-only; measured evidence only until formal preflight accepts the split"
        ],
        "smoke_blockers": blockers,
        "manual_gates": {
            "markush_validation_trainable": validation.get("trainable") is True,
            "markush_pose_mapping_review_passed": pose.get("passed") is True,
            "markush_source_leak_check_passed": source_leak.get("passed") is True,
            "markush_manifest_accepted": True,
            "strict_machine_visual_acceptance": assistant_visual.get("strict_machine_visual_acceptance_passed") is True,
            "markush_substitution_anchor_contract_passed": substitution.get("passed") is True
            and int(substitution.get("failed_rows") or 0) == 0,
        },
        "counts": {
            "markush_layout": row_count,
            "markush_train": split_manifest.get("markush", {}).get("train_rows")
            if isinstance(split_manifest.get("markush"), dict)
            else None,
            "markush_calibration": split_manifest.get("markush", {}).get("calibration_rows")
            if isinstance(split_manifest.get("markush"), dict)
            else None,
        },
        "evidence": {
            "validation": str(args.validation),
            "pose_alignment": str(args.pose_alignment),
            "source_leak": str(args.source_leak),
            "substitution_anchor_contract": str(args.substitution_anchor_contract),
            "assistant_visual_review": str(args.assistant_visual_review),
            "visual_risk_report": str(args.visual_risk_report),
            "capacity_scale": str(args.capacity_scale),
            "split_manifest": str(args.split_manifest),
            "split_preservation": str(args.split_preservation),
        },
        "policy": {
            "candidate_split_allowed_for_measured_capacity_gate_only": True,
            "candidate_split_cannot_start_formal_training": True,
            "manual_visual_csv_not_required": True,
            "strict_machine_visual_acceptance_required": True,
            "does_not_relax_pose_line_intersection_or_substitution_anchor_gates": True,
            "positive_only_markush_measured_training": True,
            "router_negative_evidence_not_claimed": True,
            "architecture_comparison_or_deployment_claim_not_allowed": True,
            "formal_training_requires_full_preflight_after_confidence": True,
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
