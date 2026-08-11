from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.tools import train_fragment_attachment_expert as fragment_train
from training.molnextr_markush.tools import train_markush_layout_expert as markush_train


def load_json(path_text: str | Path) -> dict[str, Any]:
    path = Path(path_text)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_positive_readiness_probe(source_readiness: dict[str, Any], source_path: str) -> dict[str, Any]:
    probe = dict(source_readiness)
    probe.update(
        {
            "schema_version": "positive_readiness_probe_v1",
            "probe_only": True,
            "source_readiness_report": str(source_path),
            "formal_training_start_allowed": True,
            "measured_smoke_start_allowed": True,
            "formal_blockers": [],
            "blockers": [],
        }
    )
    return probe


def make_positive_acceptance_probe(source_acceptance: dict[str, Any], source_path: str) -> dict[str, Any]:
    probe = dict(source_acceptance)
    policy = probe.get("policy") if isinstance(probe.get("policy"), dict) else {}
    policy = dict(policy)
    policy["positive_only_markush_measured_training"] = False
    policy["probe_all_branch_acceptance"] = True
    manual_gates = probe.get("manual_gates") if isinstance(probe.get("manual_gates"), dict) else {}
    manual_gates = dict(manual_gates)
    manual_gates.update(
        {
            "accepted_manifest_accepted": True,
            "markush_validation_trainable": True,
            "markush_manifest_accepted": True,
            "markush_pose_mapping_review_passed": True,
            "markush_source_leak_check_passed": True,
            "markush_visual_review_passed": True,
            "fragment_taxonomy_coverage_passed": True,
            "real_fragment_taxonomy_alignment_passed": True,
            "source_leak_check_passed": True,
            "attachment_role_contract_passed": True,
            "visual_review_passed": True,
        }
    )
    probe.update(
        {
            "schema_version": "positive_acceptance_probe_v1",
            "probe_only": True,
            "source_acceptance_report": str(source_path),
            "measured_sidecar_smoke_allowed": True,
            "formal_training_allowed": True,
            "smoke_blockers": [],
            "formal_blockers": [],
            "manual_gates": manual_gates,
            "policy": policy,
        }
    )
    return probe


def make_source_preflight_probe(source_path: str) -> dict[str, Any]:
    return {
        "schema_version": "markush_formal_acceptance_preflight_v1",
        "probe_only": True,
        "source_formal_preflight_report": str(source_path),
        "accepted_for_formal_split": False,
        "formal_training_start_allowed": False,
        "acceptance_blockers": ["probe source preflight is intentionally red"],
        "training_blockers": ["probe source preflight is intentionally red"],
        "gates": {
            "probe_source": {
                "passed": False,
                "blockers": ["probe source preflight is intentionally red"],
            }
        },
    }


def make_red_preflight_probe(source_preflight: dict[str, Any], source_path: str) -> dict[str, Any]:
    probe = dict(source_preflight)
    probe.update(
        {
            "schema_version": "markush_formal_acceptance_preflight_v1",
            "probe_only": True,
            "source_formal_preflight_report": str(source_path),
            "accepted_for_formal_split": False,
            "formal_training_start_allowed": False,
            "acceptance_blockers": ["probe red preflight must be rejected"],
            "training_blockers": ["probe red preflight must be rejected"],
        }
    )
    gates = probe.get("gates") if isinstance(probe.get("gates"), dict) else {}
    if gates:
        gates = dict(gates)
        confidence_gate = dict(gates.get("confidence") if isinstance(gates.get("confidence"), dict) else {})
        confidence_gate.update(
            {
                "passed": False,
                "blockers": ["probe red confidence gate must be rejected"],
            }
        )
        gates["confidence"] = confidence_gate
        probe["gates"] = gates
    return probe


def make_positive_preflight_probe(source_preflight: dict[str, Any], source_path: str) -> dict[str, Any]:
    probe = dict(source_preflight)
    probe.update(
        {
            "schema_version": "markush_formal_acceptance_preflight_v1",
            "probe_only": True,
            "source_formal_preflight_report": str(source_path),
            "accepted_for_formal_split": True,
            "formal_training_start_allowed": True,
            "acceptance_blockers": [],
            "training_blockers": [],
        }
    )
    gates = probe.get("gates") if isinstance(probe.get("gates"), dict) else {}
    if gates:
        clean_gates = {}
        for name, gate in gates.items():
            if isinstance(gate, dict):
                clean_gate = dict(gate)
                clean_gate["passed"] = True
                clean_gate["blockers"] = []
                clean_gates[name] = clean_gate
            else:
                clean_gates[str(name)] = {"passed": True, "blockers": [], "detail": {}}
        probe["gates"] = clean_gates
    return probe


def run_expected_rejection(
    *,
    name: str,
    call: Callable[[], dict[str, Any]],
    expected_substrings: list[str],
) -> dict[str, Any]:
    try:
        result = call()
    except ValueError as exc:
        message = str(exc)
        matched = [text for text in expected_substrings if text in message]
        return {
            "name": name,
            "passed": bool(matched),
            "expected_rejection": True,
            "matched_expected_substrings": matched,
            "expected_substrings": expected_substrings,
            "error": message,
        }
    return {
        "name": name,
        "passed": False,
        "expected_rejection": True,
        "error": "training gate accepted a run that should have been rejected",
        "unexpected_result": result,
    }


def run_expected_acceptance(
    *,
    name: str,
    call: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        result = call()
    except ValueError as exc:
        return {
            "name": name,
            "passed": False,
            "expected_rejection": False,
            "error": str(exc),
        }
    return {
        "name": name,
        "passed": True,
        "expected_rejection": False,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that routed expert training entrypoints cannot bypass the formal "
            "preflight gate. This runs gate functions only; it does not train."
        )
    )
    parser.add_argument("--acceptance-report", required=True)
    parser.add_argument("--readiness-report", required=True)
    parser.add_argument("--formal-preflight-report", default="")
    parser.add_argument("--roadmap-constraints-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    readiness = load_json(args.readiness_report)
    preflight = load_json(args.formal_preflight_report) if args.formal_preflight_report else make_source_preflight_probe("")
    roadmap = load_json(args.roadmap_constraints_report)
    red_gates = sorted(
        name
        for name, gate in (preflight.get("gates") if isinstance(preflight.get("gates"), dict) else {}).items()
        if not isinstance(gate, dict) or gate.get("passed") is not True
    )

    with tempfile.TemporaryDirectory(prefix="molnextr_entrypoint_gate_") as tmpdir:
        positive_acceptance_path = Path(tmpdir) / "positive_acceptance_probe.json"
        write_json(
            positive_acceptance_path,
            make_positive_acceptance_probe(load_json(args.acceptance_report), args.acceptance_report),
        )
        positive_readiness_path = Path(tmpdir) / "positive_readiness_probe.json"
        write_json(
            positive_readiness_path,
            make_positive_readiness_probe(readiness, args.readiness_report),
        )
        red_preflight_path = Path(tmpdir) / "red_preflight_probe.json"
        write_json(
            red_preflight_path,
            make_red_preflight_probe(preflight, args.formal_preflight_report),
        )
        positive_preflight_path = Path(tmpdir) / "positive_preflight_probe.json"
        write_json(
            positive_preflight_path,
            make_positive_preflight_probe(preflight, args.formal_preflight_report),
        )

        def fragment_formal_current_readiness() -> dict[str, Any]:
            return fragment_train.validate_training_gate(
                acceptance_report=str(positive_acceptance_path),
                readiness_report=args.readiness_report,
                formal_preflight_report=args.formal_preflight_report,
                roadmap_constraints_report=args.roadmap_constraints_report,
                training_stage="formal",
                allow_ungated_debug_run=False,
            )

        def markush_formal_current_readiness() -> dict[str, Any]:
            return markush_train.validate_training_gate(
                str(positive_acceptance_path),
                args.readiness_report,
                args.formal_preflight_report,
                args.roadmap_constraints_report,
                "formal",
            )

        def fragment_formal_missing_roadmap() -> dict[str, Any]:
            return fragment_train.validate_training_gate(
                acceptance_report=str(positive_acceptance_path),
                readiness_report=str(positive_readiness_path),
                formal_preflight_report=str(positive_preflight_path),
                roadmap_constraints_report="",
                training_stage="formal",
                allow_ungated_debug_run=False,
            )

        def markush_formal_missing_roadmap() -> dict[str, Any]:
            return markush_train.validate_training_gate(
                str(positive_acceptance_path),
                str(positive_readiness_path),
                str(positive_preflight_path),
                "",
                "formal",
            )

        def fragment_formal_missing_preflight() -> dict[str, Any]:
            return fragment_train.validate_training_gate(
                acceptance_report=str(positive_acceptance_path),
                readiness_report=str(positive_readiness_path),
                formal_preflight_report="",
                roadmap_constraints_report=args.roadmap_constraints_report,
                training_stage="formal",
                allow_ungated_debug_run=False,
            )

        def markush_formal_missing_preflight() -> dict[str, Any]:
            return markush_train.validate_training_gate(
                str(positive_acceptance_path),
                str(positive_readiness_path),
                "",
                args.roadmap_constraints_report,
                "formal",
            )

        def fragment_formal_red_preflight() -> dict[str, Any]:
            return fragment_train.validate_training_gate(
                acceptance_report=str(positive_acceptance_path),
                readiness_report=str(positive_readiness_path),
                formal_preflight_report=str(red_preflight_path),
                roadmap_constraints_report=args.roadmap_constraints_report,
                training_stage="formal",
                allow_ungated_debug_run=False,
            )

        def markush_formal_red_preflight() -> dict[str, Any]:
            return markush_train.validate_training_gate(
                str(positive_acceptance_path),
                str(positive_readiness_path),
                str(red_preflight_path),
                args.roadmap_constraints_report,
                "formal",
            )

        def fragment_measured_red_preflight() -> dict[str, Any]:
            return fragment_train.validate_training_gate(
                acceptance_report=args.acceptance_report,
                readiness_report=str(positive_readiness_path),
                formal_preflight_report=str(red_preflight_path),
                roadmap_constraints_report=args.roadmap_constraints_report,
                training_stage="measured_smoke",
                allow_ungated_debug_run=False,
            )

        def fragment_measured_positive_acceptance_red_preflight() -> dict[str, Any]:
            return fragment_train.validate_training_gate(
                acceptance_report=str(positive_acceptance_path),
                readiness_report=str(positive_readiness_path),
                formal_preflight_report=str(red_preflight_path),
                roadmap_constraints_report=args.roadmap_constraints_report,
                training_stage="measured_smoke",
                allow_ungated_debug_run=False,
            )

        def markush_measured_red_preflight() -> dict[str, Any]:
            return markush_train.validate_training_gate(
                args.acceptance_report,
                str(positive_readiness_path),
                str(red_preflight_path),
                args.roadmap_constraints_report,
                "measured_smoke",
            )

        def fragment_measured_missing_readiness() -> dict[str, Any]:
            return fragment_train.validate_training_gate(
                acceptance_report=str(positive_acceptance_path),
                readiness_report="",
                formal_preflight_report=str(red_preflight_path),
                roadmap_constraints_report=args.roadmap_constraints_report,
                training_stage="measured_smoke",
                allow_ungated_debug_run=False,
            )

        def markush_measured_missing_readiness() -> dict[str, Any]:
            return markush_train.validate_training_gate(
                args.acceptance_report,
                "",
                str(red_preflight_path),
                args.roadmap_constraints_report,
                "measured_smoke",
            )

        def fragment_research_without_roadmap_or_preflight() -> dict[str, Any]:
            return fragment_train.validate_training_gate(
                acceptance_report=args.acceptance_report,
                readiness_report="",
                formal_preflight_report="",
                roadmap_constraints_report="",
                training_stage="research_only",
                allow_ungated_debug_run=False,
            )

        def markush_research_without_roadmap_or_preflight() -> dict[str, Any]:
            return markush_train.validate_training_gate(
                args.acceptance_report,
                "",
                "",
                "",
                "research_only",
            )

        def fragment_formal_green_preflight() -> dict[str, Any]:
            return fragment_train.validate_training_gate(
                acceptance_report=str(positive_acceptance_path),
                readiness_report=str(positive_readiness_path),
                formal_preflight_report=str(positive_preflight_path),
                roadmap_constraints_report=args.roadmap_constraints_report,
                training_stage="formal",
                allow_ungated_debug_run=False,
            )

        def markush_formal_green_preflight() -> dict[str, Any]:
            return markush_train.validate_training_gate(
                str(positive_acceptance_path),
                str(positive_readiness_path),
                str(positive_preflight_path),
                args.roadmap_constraints_report,
                "formal",
            )

        def fragment_measured_cpu_runtime() -> dict[str, Any]:
            fragment_train.require_measured_runtime(
                type("Args", (), {"training_stage": "measured_smoke", "cpu": True})(),
                {"world_size": 1},
            )
            return {"accepted": True}

        def markush_measured_cpu_runtime() -> dict[str, Any]:
            markush_train.require_measured_runtime(
                type("Args", (), {"training_stage": "measured_smoke", "cpu": True})(),
                {"world_size": 1},
            )
            return {"accepted": True}

        def fragment_measured_single_process_runtime() -> dict[str, Any]:
            fragment_train.require_measured_runtime(
                type("Args", (), {"training_stage": "measured_smoke", "cpu": False})(),
                {"world_size": 1},
            )
            return {"accepted": True}

        def markush_measured_single_process_runtime() -> dict[str, Any]:
            markush_train.require_measured_runtime(
                type("Args", (), {"training_stage": "measured_smoke", "cpu": False})(),
                {"world_size": 1},
            )
            return {"accepted": True}

        def fragment_research_cpu_runtime() -> dict[str, Any]:
            fragment_train.require_measured_runtime(
                type("Args", (), {"training_stage": "research_only", "cpu": True})(),
                {"world_size": 1},
            )
            return {"accepted": True}

        def markush_research_cpu_runtime() -> dict[str, Any]:
            markush_train.require_measured_runtime(
                type("Args", (), {"training_stage": "research_only", "cpu": True})(),
                {"world_size": 1},
            )
            return {"accepted": True}

        probes = [
            run_expected_rejection(
                name="fragment_formal_current_readiness_rejected",
                call=fragment_formal_current_readiness,
                expected_substrings=["Readiness report does not allow formal"],
            ),
            run_expected_rejection(
                name="markush_formal_current_readiness_rejected",
                call=markush_formal_current_readiness,
                expected_substrings=["Readiness report does not allow formal"],
            ),
            run_expected_rejection(
                name="fragment_formal_missing_roadmap_rejected",
                call=fragment_formal_missing_roadmap,
                expected_substrings=["--roadmap-constraints-report is required"],
            ),
            run_expected_rejection(
                name="markush_formal_missing_roadmap_rejected",
                call=markush_formal_missing_roadmap,
                expected_substrings=["--roadmap-constraints-report is required"],
            ),
            run_expected_rejection(
                name="fragment_formal_missing_preflight_rejected",
                call=fragment_formal_missing_preflight,
                expected_substrings=["--formal-preflight-report is required"],
            ),
            run_expected_rejection(
                name="markush_formal_missing_preflight_rejected",
                call=markush_formal_missing_preflight,
                expected_substrings=[
                    "--formal-preflight-report is required",
                    "Markush acceptance gates are not green",
                ],
            ),
            run_expected_rejection(
                name="fragment_formal_red_preflight_rejected",
                call=fragment_formal_red_preflight,
                expected_substrings=[
                    "Formal preflight does not accept the split",
                    "Formal preflight does not allow expert training",
                ],
            ),
            run_expected_rejection(
                name="markush_formal_red_preflight_rejected",
                call=markush_formal_red_preflight,
                expected_substrings=[
                    "Formal preflight does not accept the split",
                    "Formal preflight does not allow expert training",
                ],
            ),
            run_expected_rejection(
                name="fragment_measured_markush_only_acceptance_rejected",
                call=fragment_measured_red_preflight,
                expected_substrings=["Fragment expert training cannot use a Markush-only capacity measured acceptance report"],
            ),
            run_expected_acceptance(
                name="fragment_measured_positive_acceptance_red_preflight_allowed",
                call=fragment_measured_positive_acceptance_red_preflight,
            ),
            run_expected_acceptance(
                name="markush_measured_red_preflight_allowed_with_markush_data_acceptance",
                call=markush_measured_red_preflight,
            ),
            run_expected_rejection(
                name="fragment_measured_missing_readiness_rejected",
                call=fragment_measured_missing_readiness,
                expected_substrings=["--readiness-report is required"],
            ),
            run_expected_rejection(
                name="markush_measured_missing_readiness_rejected",
                call=markush_measured_missing_readiness,
                expected_substrings=["--readiness-report is required"],
            ),
            run_expected_acceptance(
                name="fragment_research_without_roadmap_or_preflight_allowed",
                call=fragment_research_without_roadmap_or_preflight,
            ),
            run_expected_acceptance(
                name="markush_research_with_data_acceptance_allowed_without_roadmap_or_preflight",
                call=markush_research_without_roadmap_or_preflight,
            ),
            run_expected_acceptance(
                name="fragment_formal_green_preflight_accepted_with_positive_readiness",
                call=fragment_formal_green_preflight,
            ),
            run_expected_acceptance(
                name="markush_formal_green_preflight_accepted_with_positive_readiness",
                call=markush_formal_green_preflight,
            ),
            run_expected_rejection(
                name="fragment_measured_cpu_runtime_rejected",
                call=fragment_measured_cpu_runtime,
                expected_substrings=["--cpu is allowed only for debug/research runs"],
            ),
            run_expected_rejection(
                name="markush_measured_cpu_runtime_rejected",
                call=markush_measured_cpu_runtime,
                expected_substrings=["--cpu is allowed only for debug/research runs"],
            ),
            run_expected_rejection(
                name="fragment_measured_single_process_runtime_rejected",
                call=fragment_measured_single_process_runtime,
                expected_substrings=[
                    "CUDA is required for measured_smoke and formal fragment expert training.",
                    "require torchrun/DDP with at least two GPUs",
                    "require two visible CUDA devices",
                ],
            ),
            run_expected_rejection(
                name="markush_measured_single_process_runtime_rejected",
                call=markush_measured_single_process_runtime,
                expected_substrings=[
                    "CUDA is required for measured_smoke and formal Markush expert training.",
                    "require torchrun/DDP with at least two GPUs",
                    "require two visible CUDA devices",
                ],
            ),
            run_expected_acceptance(
                name="fragment_research_cpu_runtime_allowed",
                call=fragment_research_cpu_runtime,
            ),
            run_expected_acceptance(
                name="markush_research_cpu_runtime_allowed",
                call=markush_research_cpu_runtime,
            ),
        ]

    report = {
        "schema_version": "training_entrypoint_preflight_gate_check_v1",
        "passed": all(probe.get("passed") is True for probe in probes),
        "acceptance_report": str(args.acceptance_report),
        "readiness_report": str(args.readiness_report),
        "formal_preflight_report": str(args.formal_preflight_report),
        "roadmap_constraints_report": str(args.roadmap_constraints_report),
        "formal_preflight_summary": {
            "accepted_for_formal_split": preflight.get("accepted_for_formal_split") is True,
            "formal_training_start_allowed": preflight.get("formal_training_start_allowed") is True,
            "red_gates": red_gates,
            "acceptance_blockers": preflight.get("acceptance_blockers") if isinstance(preflight.get("acceptance_blockers"), list) else [],
            "training_blockers": preflight.get("training_blockers") if isinstance(preflight.get("training_blockers"), list) else [],
        },
        "roadmap_constraints_summary": {
            "passed": roadmap.get("passed") is True,
            "blockers": roadmap.get("blockers") if isinstance(roadmap.get("blockers"), list) else [],
        },
        "policy": {
            "does_not_train": True,
            "formal_stage_requires_formal_preflight_report": True,
            "formal_stage_requires_roadmap_constraints_report": True,
            "red_formal_preflight_cannot_be_used_as_training_evidence": True,
            "red_formal_preflight_cannot_be_used_as_formal_training_evidence": True,
            "fragment_nonformal_runs_record_red_preflight_without_blocking": True,
            "markush_measured_runs_require_data_acceptance_from_manifest_or_formal_preflight": True,
            "fragment_research_only_runs_do_not_require_formal_readiness_preflight_or_roadmap": True,
            "markush_research_runs_still_require_data_acceptance": True,
            "markush_measured_or_research_runs_can_use_green_data_acceptance_without_formal_preflight": True,
            "current_readiness_blockers_remain_enforced": True,
            "measured_and_formal_entrypoints_reject_cpu_runtime": True,
            "measured_and_formal_entrypoints_require_two_gpu_ddp": True,
        },
        "probes": probes,
    }
    write_json(Path(args.output), report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
