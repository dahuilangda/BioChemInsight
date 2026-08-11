from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def check_contract(contract: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    micro_training_blockers: list[str] = []
    deployment_blockers: list[str] = []
    warnings: list[str] = []

    base = contract.get("base_model") or {}
    if not base.get("immutable"):
        message = "base_model.immutable must be true"
        failures.append(message)
        micro_training_blockers.append(message)
    if base.get("complete_path") != "direct_original_molnextr":
        message = "complete_path must be direct_original_molnextr"
        failures.append(message)
        micro_training_blockers.append(message)

    router = contract.get("router") or {}
    confidence_contract = contract.get("confidence_contract") or {}
    if not confidence_contract.get("keep_confidence"):
        message = "confidence_contract.keep_confidence must be true"
        failures.append(message)
        micro_training_blockers.append(message)
    router_confidence = confidence_contract.get("router_confidence") or {}
    if not router_confidence.get("required"):
        message = "router confidence must remain required"
        failures.append(message)
        micro_training_blockers.append(message)
    sidecar_confidence = confidence_contract.get("sidecar_confidence") or {}
    if not sidecar_confidence.get("required"):
        message = "sidecar confidence must be required before expert training"
        failures.append(message)
        micro_training_blockers.append(message)

    strategy_reports = contract.get("strategy_audit_reports") or {}
    strategy_report_path = strategy_reports.get("loss_training_strategy_audit")
    strategy_report = load_json(strategy_report_path) if strategy_report_path else {}
    cleanliness_report_path = strategy_reports.get("workspace_cleanliness")
    cleanliness_report = load_json(cleanliness_report_path) if cleanliness_report_path else {}
    if strategy_report:
        training_audit = strategy_report.get("current_training_entry_audit") or {}
        if training_audit.get("shared_checkpoint_training_entry_conflicts_with_frozen_base_contract"):
            warnings.append(
                "legacy shared encoder/decoder training entry conflicts with frozen-base sidecar contract; "
                "use tools/train_fragment_attachment_expert.py for expert micro-training"
            )
        if not (strategy_report.get("confidence_strategy") or {}).get("keep_confidence"):
            message = "loss/training strategy audit does not preserve confidence"
            failures.append(message)
            micro_training_blockers.append(message)
    else:
        warnings.append("loss/training strategy audit report is missing")
    if not cleanliness_report:
        message = "workspace cleanliness report is missing"
        failures.append(message)
        micro_training_blockers.append(message)
    elif cleanliness_report.get("passed") is not True:
        message = "workspace cleanliness check did not pass"
        failures.append(message)
        micro_training_blockers.append(message)

    fragment = contract.get("fragment_sidecar") or {}
    fragment_gates = fragment.get("readiness_gates") or {}
    fragment_reports = fragment.get("audit_reports") or {}
    acceptance_path = fragment_reports.get("acceptance")
    acceptance_report = load_json(acceptance_path) if acceptance_path else {}
    if not acceptance_report:
        message = "current pose-factory acceptance report is missing"
        failures.append(message)
        micro_training_blockers.append(message)
    else:
        if acceptance_report.get("measured_sidecar_smoke_allowed") is not True:
            message = "current acceptance does not allow measured sidecar smoke"
            failures.append(message)
            micro_training_blockers.append(message)
        if acceptance_report.get("formal_training_allowed") is not True:
            message = "current acceptance does not allow formal expert training by data gates"
            failures.append(message)
            micro_training_blockers.append(message)
        manual_gates = acceptance_report.get("manual_gates") or {}
        required_manual_gates = [
            "accepted_manifest_accepted",
            "fragment_visual_contract_passed",
            "attachment_role_contract_passed",
            "source_leak_check_passed",
            "markush_validation_trainable",
            "markush_manifest_accepted",
            "markush_pose_mapping_review_passed",
            "markush_source_leak_check_passed",
        ]
        for key in required_manual_gates:
            if manual_gates.get(key) is not True:
                message = f"current acceptance manual gate is not true: {key}"
                failures.append(message)
                micro_training_blockers.append(message)
        if manual_gates.get("fragment_taxonomy_coverage_passed") is not True and manual_gates.get("real_fragment_taxonomy_alignment_passed") is not True:
            message = "current acceptance manual gate is not true: fragment_taxonomy_coverage_passed or real_fragment_taxonomy_alignment_passed"
            failures.append(message)
            micro_training_blockers.append(message)

    attachment_role_report_path = fragment_reports.get("attachment_role_contract")
    attachment_role_report = (
        load_json(attachment_role_report_path) if attachment_role_report_path else {}
    )
    if fragment_gates.get("attachment_role_contract_required") and not attachment_role_report:
        message = "attachment role contract report is missing"
        failures.append(message)
        micro_training_blockers.append(message)
    if attachment_role_report and attachment_role_report.get("passed") is not True:
        message = "attachment role contract did not pass"
        failures.append(message)
        micro_training_blockers.append(message)

    training_readiness = contract.get("training_readiness") or {}
    readiness_report_path = training_readiness.get("readiness_report")
    readiness_report = load_json(readiness_report_path) if readiness_report_path else {}
    if readiness_report:
        complete_max = as_int(nested(router, "policy", "complete_to_sidecar_max", default=0), default=0)
        if complete_max != 0:
            message = "router policy must keep complete_to_sidecar_max=0"
            failures.append(message)
            micro_training_blockers.append(message)
        if readiness_report.get("acceptance_ok") is not True:
            message = "current runtime readiness acceptance_ok is not true"
            failures.append(message)
            micro_training_blockers.append(message)
        if readiness_report.get("split_ok") is not True:
            message = "current runtime readiness split_ok is not true"
            failures.append(message)
            micro_training_blockers.append(message)
        readiness_policy = readiness_report.get("policy") if isinstance(readiness_report.get("policy"), dict) else {}
        if readiness_policy.get("debug_micro_contract_is_debug_only_not_measured_or_formal_gate") is not True:
            message = "current runtime readiness does not mark debug micro contracts as debug-only"
            failures.append(message)
            micro_training_blockers.append(message)
        if readiness_policy.get("formal_training_requires_formal_acceptance_preflight") is not True:
            message = "current runtime readiness does not require formal acceptance preflight"
            failures.append(message)
            micro_training_blockers.append(message)
        if "formal_acceptance_preflight_ready" not in readiness_report:
            message = "current runtime readiness is missing formal_acceptance_preflight_ready"
            failures.append(message)
            micro_training_blockers.append(message)
        formal_preflight = (
            readiness_report.get("formal_acceptance_preflight")
            if isinstance(readiness_report.get("formal_acceptance_preflight"), dict)
            else {}
        )
        if not formal_preflight:
            message = "current runtime readiness is missing embedded formal acceptance preflight"
            failures.append(message)
            micro_training_blockers.append(message)
        if readiness_report.get("measured_smoke_start_allowed") is not True:
            reasons = readiness_report.get("blockers") or ["measured smoke runtime readiness is not green"]
            micro_training_blockers.extend(str(reason) for reason in reasons)
            deployment_blockers.extend(str(reason) for reason in reasons)
        if readiness_report.get("formal_training_start_allowed") is not True:
            formal_reasons = readiness_report.get("formal_blockers")
            if not isinstance(formal_reasons, list) or not formal_reasons:
                formal_reasons = ["formal expert training readiness is not green"]
            micro_training_blockers.extend(str(reason) for reason in formal_reasons)
            deployment_blockers.extend(formal_reasons)
            warnings.extend(f"formal training blocked: {reason}" for reason in formal_reasons)
        if strategy_report:
            strategy_runtime = (
                strategy_report.get("runtime_decision")
                if isinstance(strategy_report.get("runtime_decision"), dict)
                else {}
            )
            strategy_formal_blockers = strategy_runtime.get("formal_blockers")
            if not isinstance(strategy_formal_blockers, list):
                strategy_formal_blockers = []
            readiness_formal_blockers = readiness_report.get("formal_blockers")
            if not isinstance(readiness_formal_blockers, list):
                readiness_formal_blockers = []
            if strategy_formal_blockers != readiness_formal_blockers:
                message = "loss/training strategy audit formal blockers are not synchronized with current readiness"
                failures.append(message)
                micro_training_blockers.append(message)
            if strategy_runtime.get("formal_training_allowed_by_runtime") != readiness_report.get(
                "formal_training_start_allowed"
            ):
                message = "loss/training strategy audit formal runtime decision is stale"
                failures.append(message)
                micro_training_blockers.append(message)
            if strategy_runtime.get("measured_smoke_allowed_by_runtime") != readiness_report.get(
                "measured_smoke_start_allowed"
            ):
                message = "loss/training strategy audit measured-smoke runtime decision is stale"
                failures.append(message)
                micro_training_blockers.append(message)
    else:
        message = "training readiness report is missing"
        failures.append(message)
        micro_training_blockers.append(message)

    markush = contract.get("markush_sidecar") or {}
    markush_gates = markush.get("readiness_gates") or {}
    if markush_gates.get("evidence_label_audit_required"):
        readiness_markush_ok = readiness_report.get("markush_measured_contract_ok") is True if readiness_report else False
        if readiness_markush_ok:
            warnings.append("Markush layout/OCR branch has passing measured GPU evidence; deployable Markush metrics still require formal scale, preflight, and calibration gates.")
        else:
            warnings.append("Markush evidence labels are accepted for data gates; Markush branch still requires passing measured GPU/formal eval.")

    contract_failures = list(failures)
    full_training_allowed = False
    sidecar_micro_training_allowed = not micro_training_blockers
    sidecar_deployment_allowed = sidecar_micro_training_allowed and not deployment_blockers
    status = (
        "sidecar_deployment_allowed"
        if sidecar_deployment_allowed
        else "sidecar_micro_training_allowed"
        if sidecar_micro_training_allowed
        else "pre_training_audit_only"
    )
    return {
        "status": status,
        "formal_full_checkpoint_training_allowed": full_training_allowed,
        "sidecar_micro_training_allowed": sidecar_micro_training_allowed,
        "sidecar_deployment_allowed": sidecar_deployment_allowed,
        "pre_training_audits_allowed": True,
        "contract_failures": contract_failures,
        "failures": contract_failures,
        "micro_training_blockers": micro_training_blockers,
        "deployment_blockers": deployment_blockers,
        "warnings": warnings,
        "acceptance": acceptance_report,
        "attachment_role_contract": attachment_role_report,
        "training_readiness": readiness_report,
        "workspace_cleanliness": cleanliness_report,
        "loss_training_strategy_audit": strategy_report,
        "decision": (
            "Sidecar may be deployed under frozen-base constraints."
            if sidecar_deployment_allowed
            else "Sidecar micro-training may start under frozen-base constraints; deployment remains blocked."
            if sidecar_micro_training_allowed
            else (
                "Do not train yet; formal preflight/readiness are blocked. Resolve manual visual acceptance, "
                "deployable confidence, candidate split promotion, and formal-scale routed expert contracts first."
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate frozen-base sidecar contract readiness."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract = load_json(args.contract)
    report = {
        "contract": args.contract,
        **check_contract(contract),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
