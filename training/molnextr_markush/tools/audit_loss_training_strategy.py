from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def current_timestamp() -> str:
    completed = subprocess.run(
        ["date", "+%Y-%m-%d %H:%M:%S %Z"],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def build_report(
    *,
    contract: dict[str, Any],
    acceptance_report: dict[str, Any],
    readiness_report: dict[str, Any],
    acceptance_path: str,
    attachment_role_contract_path: str,
    readiness_path: str,
) -> dict[str, Any]:
    base_model = contract.get("base_model") if isinstance(contract.get("base_model"), dict) else {}
    fragment = contract.get("fragment_sidecar") if isinstance(contract.get("fragment_sidecar"), dict) else {}
    markush = contract.get("markush_sidecar") if isinstance(contract.get("markush_sidecar"), dict) else {}
    validation_evidence = (
        acceptance_report.get("validation_evidence")
        if isinstance(acceptance_report.get("validation_evidence"), dict)
        else {}
    )
    counts = acceptance_report.get("counts") if isinstance(acceptance_report.get("counts"), dict) else {}

    formal_blockers = readiness_report.get("formal_blockers")
    if not isinstance(formal_blockers, list):
        formal_blockers = []
    measured_blockers = readiness_report.get("blockers")
    if not isinstance(measured_blockers, list):
        measured_blockers = []
    formal_acceptance_preflight = (
        readiness_report.get("formal_acceptance_preflight")
        if isinstance(readiness_report.get("formal_acceptance_preflight"), dict)
        else {}
    )
    formal_preflight_gates = (
        formal_acceptance_preflight.get("gates")
        if isinstance(formal_acceptance_preflight.get("gates"), dict)
        else {}
    )
    formal_preflight_red_gates = sorted(
        name
        for name, gate in formal_preflight_gates.items()
        if not isinstance(gate, dict) or gate.get("passed") is not True
    )

    return {
        "timestamp": current_timestamp(),
        "schema_version": "loss_training_strategy_audit_v2_readiness_synchronized",
        "decision": (
            "Use frozen-base routed experts only: FragmentAttachmentExpert for fragment endpoint/anchor evidence "
            "and MarkushLayoutExpert for Markush layout/OCR evidence. Do not use shared encoder/decoder fine-tuning "
            "for this project goal."
        ),
        "contract": {
            "base_model": {
                "path": base_model.get("path"),
                "immutable": base_model.get("immutable") is True,
                "complete_path": base_model.get("complete_path"),
                "complete_path_forbidden_mutations": base_model.get("complete_path_forbidden_mutations") or [],
            },
            "fragment_attachment_expert_status": fragment.get("status"),
            "markush_layout_expert_status": markush.get("status"),
        },
        "active_training_entries": {
            "fragment": {
                "path": "training/molnextr_markush/tools/train_fragment_attachment_expert.py",
                "training_type": "frozen_molnextr_encoder_fragment_attachment_expert",
                "updates_base_encoder": False,
                "updates_base_decoder": False,
                "saves_base_checkpoint": False,
                "requires_acceptance_report": True,
                "requires_runtime_readiness_report_for_formal": True,
                "requires_formal_preflight_report_for_formal": True,
                "rejects_blocked_readiness_when_provided": True,
                "requires_training_stage": True,
                "requires_formal_calibrated_contract_before_formal_training": True,
            },
            "markush": {
                "path": "training/molnextr_markush/tools/train_markush_layout_expert.py",
                "training_type": "frozen_molnextr_encoder_markush_layout_expert",
                "updates_base_encoder": False,
                "updates_base_decoder": False,
                "saves_base_checkpoint": False,
                "requires_acceptance_report": True,
                "requires_runtime_readiness_report_for_formal": True,
                "requires_formal_preflight_report_for_formal": True,
                "rejects_blocked_readiness_when_provided": True,
                "requires_training_stage": True,
                "requires_formal_calibrated_contract_before_formal_training": True,
            },
        },
        "active_training_entry": {
            "path": "training/molnextr_markush/tools/train_fragment_attachment_expert.py",
            "training_type": "frozen_molnextr_encoder_fragment_attachment_expert",
            "updates_base_encoder": False,
            "updates_base_decoder": False,
            "saves_base_checkpoint": False,
            "requires_acceptance_report": True,
            "requires_runtime_readiness_report_for_formal": True,
            "requires_formal_preflight_report_for_formal": True,
            "rejects_blocked_readiness_when_provided": True,
            "requires_training_stage": True,
        },
        "active_data_gates": {
            "acceptance_report": acceptance_path,
            "attachment_role_contract": attachment_role_contract_path,
            "training_readiness_report": readiness_path,
            "accepted_fragment_positive_rows": int(counts.get("attachment_fragment") or 0),
            "accepted_ordinary_negative_rows": int(counts.get("ordinary_complete_negative") or 0),
            "accepted_markush_layout_rows": int(counts.get("markush_layout") or 0),
            "ordinary_negative_attachment_policy": (
                "ordinary complete-molecule negatives must be attachment-free unless explicitly labeled as "
                "supported hard negatives"
            ),
            "validation_evidence": validation_evidence,
        },
        "active_losses": {
            "fragment": {
                "endpoint_presence": "class-balanced focal cross entropy",
                "sidecar_applicability": "class-balanced focal cross entropy",
                "endpoint_side": "class-balanced cross entropy with ordinary negatives ignored",
                "anchor_token": "class-balanced cross entropy with ordinary negatives ignored",
                "backbone_eligibility": "optional cross entropy where labels exist",
                "endpoint_point": "SmoothL1 weighted by endpoint label quality",
                "endpoint_heatmap": "Gaussian heatmap BCE weighted by endpoint label quality",
                "endpoint_objectness": "matched-query cross entropy with explicit no-endpoint class",
                "sidecar_risk": "accept/reject cross entropy for selective output",
            },
            "markush": {
                "markush_presence": "class-balanced cross entropy against accepted Markush/ordinary rows",
                "variable_count_bucket": "cross entropy over OCR/layout variable-count buckets",
                "layout_objectness": "set-prediction objectness over learned layout queries",
                "layout_box": "matched-query SmoothL1 box regression for OCR/layout evidence",
                "sidecar_risk": "accept/reject cross entropy for selective Markush output",
            },
        },
        "losses_that_must_not_drive_current_training": [
            "shared MolNexTR seq2seq decoder loss",
            "decoder-token attachment insertion loss",
            "attachment count decode constraint loss",
            "decoder logit-bias or star-budget losses",
            "post-decode SMILES mutation objectives",
            "encoder/decoder L2SP losses for shared fine-tuning",
        ],
        "confidence_strategy": {
            "keep_confidence": True,
            "router_confidence": {
                "current_policy": "frozen-base selective router with complete_to_sidecar=0 and calibrated sidecar acceptance thresholds",
                "required_metrics": [
                    "Brier",
                    "ECE",
                    "risk/coverage curve",
                    "ordinary false accepts",
                    "high-confidence errors",
                    "complete_to_sidecar=0",
                ],
                "status": "required_before_deployment",
            },
            "sidecar_confidence": {
                "scores": [
                    "endpoint_presence_probability",
                    "endpoint_objectness_probability",
                    "endpoint_point_or_heatmap_score",
                    "anchor_token_probability",
                    "expert_applicability_probability",
                    "expert_risk_probability",
                    "markush_presence_probability",
                    "markush_count_probability",
                    "markush_layout_box_quality",
                    "markush_risk_probability",
                ],
                "acceptance_policy": (
                    "Accept expert output only when the routed expert has a passing measured GPU contract, "
                    "a deployable formal calibrated confidence report, and all calibrated risk gates pass; "
                    "otherwise keep the original MolNexTR result or abstain."
                ),
                "status": "required_before_deployment",
            },
            "calibration_protocol": [
                "use calibration split only for threshold selection",
                "require zero accepted ordinary negatives at the selected threshold",
                "report Brier/ECE/risk coverage/high-confidence errors",
                "do not use raw decoder softmax as sidecar acceptance confidence",
                "formal readiness requires fragment and Markush formal calibrated run contracts",
            ],
        },
        "runtime_decision": {
            "data_gates_green": acceptance_report.get("measured_sidecar_smoke_allowed") is True
            and acceptance_report.get("formal_training_allowed") is True,
            "measured_smoke_allowed_by_data": acceptance_report.get("measured_sidecar_smoke_allowed") is True,
            "measured_smoke_allowed_by_runtime": readiness_report.get("measured_smoke_start_allowed") is True,
            "formal_training_allowed_by_runtime": readiness_report.get("formal_training_start_allowed") is True,
            "measured_smoke_blockers": measured_blockers,
            "formal_blockers": formal_blockers,
            "readiness_policy": readiness_report.get("policy") if isinstance(readiness_report.get("policy"), dict) else {},
            "formal_acceptance_preflight_ready": readiness_report.get("formal_acceptance_preflight_ready") is True,
            "formal_acceptance_preflight_report": readiness_report.get("formal_acceptance_preflight_report"),
            "formal_acceptance_preflight_red_gates": formal_preflight_red_gates,
            "formal_training_requires_formal_acceptance_preflight": (
                readiness_report.get("policy", {}).get("formal_training_requires_formal_acceptance_preflight") is True
                if isinstance(readiness_report.get("policy"), dict)
                else False
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit active loss/training strategy against current readiness gates.")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--acceptance-report", required=True)
    parser.add_argument("--readiness-report", required=True)
    parser.add_argument("--attachment-role-contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        contract=load_json(args.contract),
        acceptance_report=load_json(args.acceptance_report),
        readiness_report=load_json(args.readiness_report),
        acceptance_path=str(args.acceptance_report),
        attachment_role_contract_path=str(args.attachment_role_contract),
        readiness_path=str(args.readiness_report),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
