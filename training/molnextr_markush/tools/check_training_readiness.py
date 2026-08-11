from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def count_csv_rows(path: str | Path) -> int:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def load_optional_contract(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        return {
            "provided": True,
            "exists": False,
            "path": str(path),
            "passed": False,
            "blockers": [f"missing run contract: {path}"],
        }
    report = load_json(path)
    report.setdefault("provided", True)
    report.setdefault("exists", True)
    report.setdefault("path", str(path))
    return report


def load_optional_report(path_text: str, *, label: str) -> dict[str, Any]:
    if not path_text:
        return {
            "provided": False,
            "exists": False,
            "path": "",
            "blockers": [f"{label} report was not provided"],
        }
    path = Path(path_text)
    if not path.exists():
        return {
            "provided": True,
            "exists": False,
            "path": str(path),
            "blockers": [f"missing {label} report: {path}"],
        }
    report = load_json(path)
    report.setdefault("provided", True)
    report.setdefault("exists", True)
    report.setdefault("path", str(path))
    return report


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def contract_passed(
    report: dict[str, Any],
    *,
    required_policy_key: str,
    require_model_scale_policy: bool = False,
    expected_schema_version: str = "",
    split_manifest: dict[str, Any] | None = None,
    branch: str = "",
) -> bool:
    if not report or report.get("passed") is not True or report.get("blockers"):
        return False
    if expected_schema_version and report.get("schema_version") != expected_schema_version:
        return False
    policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    if policy.get(required_policy_key) is not True:
        return False
    if require_model_scale_policy:
        model_scale_policy_ok = (
            policy.get("model_scale_report_required") is True
            and policy.get("model_scale_gate_enforced") is True
            and policy.get("router_branch_gate_enforced") is True
            and policy.get("runtime_gpu_gate_enforced") is True
            and policy.get("debug_only_runs_cannot_satisfy_measured_or_formal") is True
        )
        if not model_scale_policy_ok:
            return False
    if split_manifest is not None and branch:
        if contract_split_hash_blockers(report, split_manifest=split_manifest, branch=branch):
            return False
    return True


def first_hash(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def contract_split_hash_blockers(
    report: dict[str, Any],
    *,
    split_manifest: dict[str, Any],
    branch: str,
) -> list[str]:
    if not report:
        return ["run contract was not provided"]
    policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    if policy.get("current_split_manifest_hash_must_match") is not True:
        return ["run contract does not require current split manifest hash matching"]
    expected = split_manifest.get("output_sha256") if isinstance(split_manifest.get("output_sha256"), dict) else {}
    if not expected:
        return ["split manifest has no output_sha256"]
    metrics_path = Path(str(report.get("metrics") or ""))
    if not metrics_path.exists():
        return [f"run contract metrics file is missing: {metrics_path}"]
    metrics = load_json(metrics_path)
    config = metrics.get("config") if isinstance(metrics.get("config"), dict) else {}
    observed = config.get("input_sha256") if isinstance(config.get("input_sha256"), dict) else {}
    if not observed:
        return ["metrics.config.input_sha256 is missing; run cannot be tied to the current split manifest"]

    if branch == "fragment":
        comparisons = [
            ("positive_label_csv", "fragment_train_csv", first_hash(observed.get("positive_label_csv"))),
            (
                "eval_positive_label_csv",
                "fragment_calibration_csv",
                first_hash(observed.get("eval_positive_label_csv")),
            ),
            ("negative_csv", "ordinary_train_csv", first_hash(observed.get("negative_csv"))),
            ("eval_negative_csv", "ordinary_calibration_csv", first_hash(observed.get("eval_negative_csv"))),
        ]
    elif branch == "markush":
        comparisons = [
            ("markush_label_csv", "markush_train_csv", first_hash(observed.get("markush_label_csv"))),
            (
                "eval_markush_label_csv",
                "markush_calibration_csv",
                first_hash(observed.get("eval_markush_label_csv")),
            ),
            ("negative_csv", "ordinary_train_csv", first_hash(observed.get("negative_csv"))),
            ("eval_negative_csv", "ordinary_calibration_csv", first_hash(observed.get("eval_negative_csv"))),
        ]
    else:
        return [f"unknown contract branch for split hash audit: {branch}"]

    blockers: list[str] = []
    for observed_name, manifest_name, observed_hash in comparisons:
        expected_hash = str(expected.get(manifest_name) or "")
        if not expected_hash:
            blockers.append(f"split manifest missing output_sha256.{manifest_name}")
        elif not observed_hash:
            blockers.append(f"metrics.config.input_sha256 missing {observed_name}")
        elif observed_hash != expected_hash:
            blockers.append(f"{observed_name} hash does not match current split manifest {manifest_name}")
    return blockers


def contract_blockers(
    report: dict[str, Any],
    *,
    label: str,
    expected_schema_version: str,
    split_manifest: dict[str, Any] | None = None,
    branch: str = "",
) -> list[str]:
    blockers: list[str] = []
    if not report:
        return [f"{label} contract was not provided"]
    if report.get("exists") is False:
        return [str(message) for message in report.get("blockers", [f"{label} contract is missing"])]
    if expected_schema_version and report.get("schema_version") != expected_schema_version:
        blockers.append(
            f"{label} contract schema_version {report.get('schema_version')!r} != {expected_schema_version!r}"
        )
    if report.get("passed") is not True:
        blockers.append(f"{label} contract did not pass")
    report_blockers = report.get("blockers")
    if isinstance(report_blockers, list):
        blockers.extend(f"{label}: {message}" for message in report_blockers)
    policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    for key in [
        "model_scale_report_required",
        "model_scale_gate_enforced",
        "router_branch_gate_enforced",
        "runtime_gpu_gate_enforced",
        "debug_only_runs_cannot_satisfy_measured_or_formal",
    ]:
        if policy.get(key) is not True:
            blockers.append(f"{label} policy.{key} is not true")
    if split_manifest is not None and branch:
        blockers.extend(f"{label}: {message}" for message in contract_split_hash_blockers(report, split_manifest=split_manifest, branch=branch))
    return blockers


def run_nvidia_smi_probe() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError:
        return {
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": "nvidia-smi not found",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "available": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": "nvidia-smi timed out",
        }
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def run_nvidia_smi(*, attempts: int = 3) -> dict[str, Any]:
    probes = [run_nvidia_smi_probe() for _ in range(max(1, int(attempts)))]
    success_count = sum(1 for probe in probes if probe.get("available") is True)
    return {
        "available": success_count == len(probes),
        "success_count": success_count,
        "attempt_count": len(probes),
        "which": shutil.which("nvidia-smi") or "",
        "path": os.environ.get("PATH", ""),
        "probes": probes,
        "returncode": probes[-1].get("returncode") if probes else None,
        "stdout": probes[-1].get("stdout", "") if probes else "",
        "stderr": probes[-1].get("stderr", "") if probes else "",
        "policy": {
            "python_subprocess_gpu_probe_must_pass_all_attempts": True,
            "direct_shell_gpu_visibility_is_not_sufficient_for_training_readiness": True,
        },
    }


def load_gpu_runtime_evidence(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {
            "available": False,
            "provided": True,
            "exists": False,
            "path": str(path),
            "blockers": [f"missing GPU runtime evidence: {path}"],
        }
    report = load_json(path)
    gpus = report.get("gpus") if isinstance(report.get("gpus"), list) else []
    blockers: list[str] = []
    if report.get("schema_version") != "gpu_runtime_evidence_v1":
        blockers.append("GPU runtime evidence schema_version is not gpu_runtime_evidence_v1")
    if report.get("passed") is not True:
        blockers.append("GPU runtime evidence did not pass")
    if int(report.get("gpu_count_observed") or len(gpus)) < 2:
        blockers.append("GPU runtime evidence observed fewer than two GPUs")
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        try:
            memory_total_gb = float(gpu.get("memory_total_gb") or 0.0)
        except (TypeError, ValueError):
            memory_total_gb = 0.0
        if memory_total_gb < 15.0:
            blockers.append(f"GPU {gpu.get('index')} memory {memory_total_gb:.2f} GB < required 15.00 GB")
    evidence_blockers = report.get("blockers")
    if isinstance(evidence_blockers, list):
        blockers.extend(str(item) for item in evidence_blockers)
    policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    return report | {
        "available": not blockers,
        "provided": True,
        "exists": True,
        "path": str(path),
        "blockers": blockers,
        "policy": policy
        | {
            "readiness_accepts_file_backed_gpu_runtime_evidence": True,
            "gpu_runtime_evidence_must_be_machine_checkable": True,
        },
    }


def entrypoint_policy_true(policy: dict[str, Any], *keys: str) -> bool:
    return any(policy.get(key) is True for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check sidecar training readiness without starting training.")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--acceptance-report", required=True)
    parser.add_argument("--debug-micro-contract", default="")
    parser.add_argument("--markush-debug-micro-contract", default="")
    parser.add_argument(
        "--cpu-smoke-contract",
        default="",
        help="Deprecated alias for --debug-micro-contract; debug-only and never measured/formal evidence.",
    )
    parser.add_argument(
        "--markush-cpu-smoke-contract",
        default="",
        help="Deprecated alias for --markush-debug-micro-contract; debug-only and never measured/formal evidence.",
    )
    parser.add_argument("--fragment-measured-contract", default="")
    parser.add_argument("--markush-measured-contract", default="")
    parser.add_argument("--fragment-calibrated-measured-contract", default="")
    parser.add_argument("--markush-calibrated-measured-contract", default="")
    parser.add_argument("--markush-architecture-eval-readiness-report", default="")
    parser.add_argument("--training-scale-adequacy-report", default="")
    parser.add_argument("--formal-acceptance-preflight-report", default="")
    parser.add_argument("--training-entrypoint-preflight-gate", default="")
    parser.add_argument("--markush-standards-decision-gate", default="")
    parser.add_argument("--split-pose-preservation-contract", default="")
    parser.add_argument("--fragment-formal-contract", default="")
    parser.add_argument("--markush-formal-contract", default="")
    parser.add_argument("--workspace-cleanliness-report", default="")
    parser.add_argument("--gpu-runtime-evidence", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument(
        "--scope",
        choices=["measured_smoke", "formal", "all"],
        default="all",
        help="Controls the process exit code. The JSON report always contains both measured and formal gates.",
    )
    args = parser.parse_args()

    debug_micro_contract_path = args.debug_micro_contract or args.cpu_smoke_contract
    markush_debug_micro_contract_path = args.markush_debug_micro_contract or args.markush_cpu_smoke_contract

    split_manifest = load_json(args.split_manifest)
    acceptance = load_json(args.acceptance_report)
    debug_micro_contract = (
        load_json(debug_micro_contract_path)
        if debug_micro_contract_path
        else {
            "provided": False,
            "passed": False,
            "blockers": ["debug micro contract was not provided; debug-only evidence is not required for readiness"],
        }
    )
    markush_debug_micro_contract = (
        load_json(markush_debug_micro_contract_path) if markush_debug_micro_contract_path else {}
    )
    fragment_measured_contract = load_optional_contract(args.fragment_measured_contract)
    markush_measured_contract = load_optional_contract(args.markush_measured_contract)
    fragment_calibrated_measured_contract = load_optional_contract(args.fragment_calibrated_measured_contract)
    markush_calibrated_measured_contract = load_optional_contract(args.markush_calibrated_measured_contract)
    markush_architecture_eval_readiness = load_optional_report(
        args.markush_architecture_eval_readiness_report,
        label="Markush architecture evaluation readiness",
    )
    training_scale_adequacy = load_optional_report(
        args.training_scale_adequacy_report,
        label="training scale adequacy",
    )
    formal_acceptance_preflight = load_optional_report(
        args.formal_acceptance_preflight_report,
        label="formal acceptance preflight",
    )
    training_entrypoint_preflight = load_optional_report(
        args.training_entrypoint_preflight_gate,
        label="training entrypoint preflight gate",
    )
    markush_standards_decision_gate = load_optional_report(
        args.markush_standards_decision_gate,
        label="Markush standards decision gate",
    )
    split_pose_preservation_contract = load_optional_report(
        args.split_pose_preservation_contract,
        label="split pose preservation contract",
    )
    fragment_formal_contract = load_optional_contract(args.fragment_formal_contract)
    markush_formal_contract = load_optional_contract(args.markush_formal_contract)
    cleanliness = load_json(args.workspace_cleanliness_report) if args.workspace_cleanliness_report else {}
    outputs = split_manifest.get("outputs") if isinstance(split_manifest.get("outputs"), dict) else {}
    purpose = split_manifest.get("purpose") if isinstance(split_manifest.get("purpose"), dict) else {}
    split_stage = str(split_manifest.get("stage") or "formal").strip()
    split_accepted_for_training = split_manifest.get("accepted_for_training")
    if split_accepted_for_training is None:
        split_accepted_for_training = split_stage == "formal"

    required_outputs = [
        "fragment_train_csv",
        "fragment_calibration_csv",
        "ordinary_train_csv",
        "ordinary_calibration_csv",
        "markush_train_csv",
        "markush_calibration_csv",
    ]
    missing_outputs = []
    row_counts: dict[str, int] = {}
    for key in required_outputs:
        path_text = str(outputs.get(key) or "")
        if not path_text or not Path(path_text).exists():
            missing_outputs.append(key)
            continue
        row_counts[key] = count_csv_rows(path_text)

    if args.skip_gpu_check:
        gpu = {"available": True, "skipped": True}
    elif args.gpu_runtime_evidence:
        gpu = load_gpu_runtime_evidence(args.gpu_runtime_evidence)
    else:
        gpu = run_nvidia_smi()
    acceptance_measured_ok = (
        acceptance.get("measured_sidecar_smoke_allowed") is True
        and not acceptance.get("smoke_blockers")
    )
    acceptance_ok = acceptance_measured_ok
    acceptance_formal_ok = (
        acceptance_measured_ok
        and acceptance.get("formal_training_allowed") is True
        and not acceptance.get("formal_blockers")
    )
    debug_micro_contract_ok = debug_micro_contract.get("passed") is True and not debug_micro_contract.get("blockers")
    markush_debug_micro_contract_ok = (
        bool(markush_debug_micro_contract)
        and markush_debug_micro_contract.get("passed") is True
        and not markush_debug_micro_contract.get("blockers")
    )
    fragment_measured_contract_ok = contract_passed(
        fragment_measured_contract,
        required_policy_key="measured_stage_required",
        require_model_scale_policy=True,
        expected_schema_version="fragment_attachment_expert_run_contract_v2",
        split_manifest=split_manifest,
        branch="fragment",
    )
    markush_measured_contract_ok = contract_passed(
        markush_measured_contract,
        required_policy_key="measured_stage_required",
        require_model_scale_policy=True,
        expected_schema_version="markush_layout_expert_run_contract_v2",
        split_manifest=split_manifest,
        branch="markush",
    )
    fragment_formal_contract_ok = contract_passed(
        fragment_calibrated_measured_contract or fragment_formal_contract,
        required_policy_key="calibrated_confidence_required",
        require_model_scale_policy=True,
        expected_schema_version="fragment_attachment_expert_run_contract_v2",
        split_manifest=split_manifest,
        branch="fragment",
    )
    markush_formal_contract_ok = contract_passed(
        markush_calibrated_measured_contract or markush_formal_contract,
        required_policy_key="calibrated_confidence_required",
        require_model_scale_policy=True,
        expected_schema_version="markush_layout_expert_run_contract_v2",
        split_manifest=split_manifest,
        branch="markush",
    )
    markush_architecture_eval_ready = (
        markush_architecture_eval_readiness.get("architecture_comparison_allowed") is True
        and markush_architecture_eval_readiness.get("formal_expert_training_candidate_allowed") is True
        and markush_architecture_eval_readiness.get("decision", {}).get("current_data_can_start_formal_expert_training")
        is True
    )
    training_scale_formal_ready = (
        training_scale_adequacy.get("schema_version") == "training_scale_adequacy_v1"
        and training_scale_adequacy.get("formal_training_start_allowed") is True
        and training_scale_adequacy.get("decision", {}).get("current_data_can_start_formal_branch_training") is True
    )
    training_scale_architecture_ready = (
        training_scale_adequacy.get("schema_version") == "training_scale_adequacy_v1"
        and training_scale_adequacy.get("architecture_comparison_allowed") is True
        and training_scale_adequacy.get("decision", {}).get("current_data_can_compare_architectures") is True
    )
    markush_capacity_measured_candidate_ready = (
        acceptance.get("schema_version") == "markush_capacity_measured_acceptance_v1"
        and acceptance.get("measured_sidecar_smoke_allowed") is True
        and acceptance.get("formal_training_allowed") is not True
        and acceptance.get("policy", {}).get("candidate_split_allowed_for_measured_capacity_gate_only") is True
        and acceptance.get("policy", {}).get("architecture_comparison_or_deployment_claim_not_allowed") is True
        and training_scale_adequacy.get("schema_version") == "markush_capacity_maximized_training_scale_v1"
        and training_scale_adequacy.get("formal_training_candidate_allowed") is True
        and training_scale_adequacy.get("architecture_comparison_allowed") is not True
        and training_scale_adequacy.get("deployment_certification_allowed") is not True
        and training_scale_adequacy.get("policy", {}).get("formal_training_can_use_best_effort_source_capacity") is True
        and training_scale_adequacy.get("policy", {}).get("does_not_allow_architecture_comparison_claim") is True
    )
    formal_acceptance_preflight_ready = (
        formal_acceptance_preflight.get("schema_version") == "markush_formal_acceptance_preflight_v1"
        and formal_acceptance_preflight.get("accepted_for_formal_split") is True
        and formal_acceptance_preflight.get("formal_training_start_allowed") is True
    )
    entrypoint_policy = (
        training_entrypoint_preflight.get("policy")
        if isinstance(training_entrypoint_preflight.get("policy"), dict)
        else {}
    )
    training_entrypoint_preflight_ready = (
        training_entrypoint_preflight.get("schema_version") == "training_entrypoint_preflight_gate_check_v1"
        and training_entrypoint_preflight.get("passed") is True
        and entrypoint_policy.get("formal_stage_requires_formal_preflight_report") is True
        and entrypoint_policy.get("formal_stage_requires_roadmap_constraints_report") is True
        and entrypoint_policy.get("red_formal_preflight_cannot_be_used_as_formal_training_evidence") is True
        and entrypoint_policy_true(
            entrypoint_policy,
            "nonformal_research_or_measured_runs_record_red_preflight_without_blocking",
            "fragment_nonformal_runs_record_red_preflight_without_blocking",
        )
        and entrypoint_policy_true(
            entrypoint_policy,
            "research_only_runs_do_not_require_formal_readiness_preflight_or_roadmap",
            "fragment_research_only_runs_do_not_require_formal_readiness_preflight_or_roadmap",
        )
        and entrypoint_policy.get("current_readiness_blockers_remain_enforced") is True
        and entrypoint_policy.get("measured_and_formal_entrypoints_reject_cpu_runtime") is True
        and entrypoint_policy.get("measured_and_formal_entrypoints_require_two_gpu_ddp") is True
    )
    markush_standards_decision_schema = str(markush_standards_decision_gate.get("schema_version") or "")
    markush_standards_decision_ready = (
        markush_standards_decision_schema == "markush_standards_decision_gate_v1"
        and markush_standards_decision_gate.get("passed") is True
    )
    markush_standards_decision_gate_is_report = (
        bool(args.markush_standards_decision_gate)
        and markush_standards_decision_schema == "markush_standards_decision_gate_v1"
    )
    split_pose_preservation_ready = (
        split_pose_preservation_contract.get("passed") is True
        and split_pose_preservation_contract.get("policy", {}).get("split_outputs_must_preserve_images") is True
        and split_pose_preservation_contract.get("policy", {}).get("split_outputs_must_preserve_graph_and_atom_coordinates") is True
        and split_pose_preservation_contract.get("policy", {}).get("fragment_rows_must_preserve_endpoint_coordinates_and_anchor_labels") is True
        and split_pose_preservation_contract.get("policy", {}).get("markush_rows_must_preserve_ocr_cells_r_tag_labels_annotations_and_render_provenance") is True
        and split_pose_preservation_contract.get("policy", {}).get("split_rows_must_preserve_image_to_graph_orientation_alignment") is True
    )
    split_ok = not missing_outputs and all(value > 0 for value in row_counts.values())
    cleanliness_ok = not args.workspace_cleanliness_report or cleanliness.get("passed") is True
    markush_purpose_text = str(purpose.get("markush_layout_expert") or purpose.get("markush_sidecar") or "")
    markush_purpose = markush_purpose_text.strip().lower()
    markush_consumed_by_current_training = (
        (bool(markush_purpose) and "not consumed" not in markush_purpose)
        or markush_measured_contract_ok
        or markush_formal_contract_ok
    )
    fragment_measured_blockers: list[str] = []
    markush_measured_blockers: list[str] = []
    formal_blockers: list[str] = []

    if not acceptance_measured_ok:
        fragment_measured_blockers.append("current acceptance report does not allow measured smoke")
        markush_measured_blockers.append("current acceptance report does not allow measured smoke")
    if not acceptance_formal_ok:
        formal_blockers.append("current acceptance report does not allow formal training")
    if not split_ok:
        fragment_measured_blockers.append(f"formal split outputs are missing or empty: {missing_outputs}")
        markush_measured_blockers.append(f"formal split outputs are missing or empty: {missing_outputs}")
        formal_blockers.append(f"formal split outputs are missing or empty: {missing_outputs}")
    split_allowed_for_measured = (split_stage == "formal" and split_accepted_for_training is True) or (
        split_stage == "candidate"
        and (training_scale_architecture_ready or markush_capacity_measured_candidate_ready)
    )
    if not split_allowed_for_measured:
        split_measured_message = (
            "split manifest is not accepted for measured expert training, architecture comparison, or "
            "Markush capacity-measured candidate evidence: "
            f"stage={split_stage or 'missing'} accepted_for_training={split_accepted_for_training} "
            f"architecture_comparison_allowed={training_scale_architecture_ready} "
            f"markush_capacity_measured_candidate_allowed={markush_capacity_measured_candidate_ready}"
        )
        fragment_measured_blockers.append(split_measured_message)
        markush_measured_blockers.append(split_measured_message)
    if markush_capacity_measured_candidate_ready:
        fragment_measured_blockers.append(
            "current measured acceptance is Markush-only capacity evidence; fragment measured training "
            "requires a fragment acceptance report"
        )
    if split_stage != "formal" or split_accepted_for_training is not True:
        formal_blockers.append(
            "split manifest is not accepted for training: "
            f"stage={split_stage or 'missing'} accepted_for_training={split_accepted_for_training}"
        )
    if not cleanliness_ok:
        fragment_measured_blockers.append("workspace cleanliness report did not pass")
        markush_measured_blockers.append("workspace cleanliness report did not pass")
        formal_blockers.append("workspace cleanliness report did not pass")
    if not gpu.get("available"):
        fragment_measured_blockers.append("GPU/NVML is not available; nvidia-smi failed")
        markush_measured_blockers.append("GPU/NVML is not available; nvidia-smi failed")
        formal_blockers.append("GPU/NVML is not available; nvidia-smi failed")
    if not markush_consumed_by_current_training:
        formal_blockers.append("current training entry does not consume Markush split; formal all-branch training is not ready")

    if not fragment_measured_contract_ok:
        formal_blockers.extend(
            contract_blockers(
                fragment_measured_contract,
                label="fragment measured GPU expert",
                expected_schema_version="fragment_attachment_expert_run_contract_v2",
                split_manifest=split_manifest,
                branch="fragment",
            )
            or ["fragment measured GPU expert contract has not passed"]
        )
    if not markush_measured_contract_ok:
        formal_blockers.extend(
            contract_blockers(
                markush_measured_contract,
                label="Markush measured GPU expert",
                expected_schema_version="markush_layout_expert_run_contract_v2",
                split_manifest=split_manifest,
                branch="markush",
            )
            or ["Markush measured GPU expert contract has not passed"]
        )
    if not fragment_formal_contract_ok:
        formal_blockers.extend(
            contract_blockers(
                fragment_calibrated_measured_contract or fragment_formal_contract,
                label="fragment calibrated measured expert",
                expected_schema_version="fragment_attachment_expert_run_contract_v2",
                split_manifest=split_manifest,
                branch="fragment",
            )
            or ["fragment calibrated measured expert contract has not passed"]
        )
    if not markush_formal_contract_ok:
        formal_blockers.extend(
            contract_blockers(
                markush_calibrated_measured_contract or markush_formal_contract,
                label="Markush calibrated measured expert",
                expected_schema_version="markush_layout_expert_run_contract_v2",
                split_manifest=split_manifest,
                branch="markush",
            )
            or ["Markush calibrated measured expert contract has not passed"]
        )
    if not markush_architecture_eval_ready:
        report_blockers = markush_architecture_eval_readiness.get("blockers")
        if not isinstance(report_blockers, list):
            report_blockers = []
        architecture_blockers = markush_architecture_eval_readiness.get("architecture_blockers")
        if isinstance(architecture_blockers, list):
            report_blockers.extend(architecture_blockers)
        formal_training_blockers = markush_architecture_eval_readiness.get("formal_training_blockers")
        if isinstance(formal_training_blockers, list):
            report_blockers.extend(formal_training_blockers)
        if not report_blockers:
            report_blockers = ["Markush architecture/data sufficiency readiness is not green"]
        formal_blockers.extend(report_blockers)
    if not training_scale_formal_ready:
        scale_blockers: list[str] = []
        if training_scale_adequacy.get("provided") is not True:
            scale_blockers.append("training scale adequacy report was not provided")
        elif training_scale_adequacy.get("exists") is False:
            scale_blockers.extend(str(item) for item in training_scale_adequacy.get("blockers") or [])
        elif training_scale_adequacy.get("schema_version") != "training_scale_adequacy_v1":
            scale_blockers.append("training scale adequacy schema_version is not training_scale_adequacy_v1")
        for key in ["formal_scale_blockers", "architecture_blockers"]:
            values = training_scale_adequacy.get(key)
            if isinstance(values, list):
                scale_blockers.extend(str(item) for item in values)
        if not scale_blockers:
            scale_blockers.append("training scale adequacy does not allow formal branch training")
        formal_blockers.extend(scale_blockers)
    if not formal_acceptance_preflight_ready:
        preflight_blockers: list[str] = []
        if formal_acceptance_preflight.get("provided") is not True:
            preflight_blockers.append("formal acceptance preflight report was not provided")
        elif formal_acceptance_preflight.get("exists") is False:
            preflight_blockers.extend(str(item) for item in formal_acceptance_preflight.get("blockers") or [])
        elif formal_acceptance_preflight.get("schema_version") != "markush_formal_acceptance_preflight_v1":
            preflight_blockers.append("formal acceptance preflight schema_version is not markush_formal_acceptance_preflight_v1")
        acceptance_blockers = formal_acceptance_preflight.get("acceptance_blockers")
        training_blockers = formal_acceptance_preflight.get("training_blockers")
        if isinstance(acceptance_blockers, list):
            preflight_blockers.extend(f"formal acceptance preflight: {item}" for item in acceptance_blockers)
        if isinstance(training_blockers, list):
            preflight_blockers.extend(f"formal acceptance preflight: {item}" for item in training_blockers)
        if not preflight_blockers:
            preflight_blockers.append("formal acceptance preflight does not allow formal training")
        formal_blockers.extend(preflight_blockers)
    if not training_entrypoint_preflight_ready:
        entrypoint_blockers: list[str] = []
        if training_entrypoint_preflight.get("provided") is not True:
            entrypoint_blockers.append("training entrypoint preflight gate report was not provided")
        elif training_entrypoint_preflight.get("exists") is False:
            entrypoint_blockers.extend(str(item) for item in training_entrypoint_preflight.get("blockers") or [])
        elif training_entrypoint_preflight.get("schema_version") != "training_entrypoint_preflight_gate_check_v1":
            entrypoint_blockers.append(
                "training entrypoint preflight gate schema_version is not training_entrypoint_preflight_gate_check_v1"
            )
        if training_entrypoint_preflight.get("passed") is not True:
            entrypoint_blockers.append("training entrypoint preflight gate did not pass")
        for key in [
            "formal_stage_requires_formal_preflight_report",
            "formal_stage_requires_roadmap_constraints_report",
            "red_formal_preflight_cannot_be_used_as_formal_training_evidence",
            "current_readiness_blockers_remain_enforced",
            "measured_and_formal_entrypoints_reject_cpu_runtime",
            "measured_and_formal_entrypoints_require_two_gpu_ddp",
        ]:
            if entrypoint_policy.get(key) is not True:
                entrypoint_blockers.append(f"training entrypoint preflight policy.{key} is not true")
        if not entrypoint_policy_true(
            entrypoint_policy,
            "nonformal_research_or_measured_runs_record_red_preflight_without_blocking",
            "fragment_nonformal_runs_record_red_preflight_without_blocking",
        ):
            entrypoint_blockers.append(
                "training entrypoint preflight policy.nonformal/fragment red preflight nonformal allowance is not true"
            )
        if not entrypoint_policy_true(
            entrypoint_policy,
            "research_only_runs_do_not_require_formal_readiness_preflight_or_roadmap",
            "fragment_research_only_runs_do_not_require_formal_readiness_preflight_or_roadmap",
        ):
            entrypoint_blockers.append(
                "training entrypoint preflight policy.research-only formal preflight/roadmap exemption is not true"
            )
        probes = training_entrypoint_preflight.get("probes")
        if not isinstance(probes, list) or not probes:
            entrypoint_blockers.append("training entrypoint preflight gate has no probes")
        elif any(not isinstance(probe, dict) or probe.get("passed") is not True for probe in probes):
            entrypoint_blockers.append("training entrypoint preflight gate has failed probes")
        formal_blockers.extend(entrypoint_blockers)
    if markush_standards_decision_gate_is_report and not markush_standards_decision_ready:
        standards_blockers = markush_standards_decision_gate.get("blockers")
        if not isinstance(standards_blockers, list):
            standards_blockers = ["Markush standards decision gate is not green"]
        fragment_measured_blockers.extend(str(item) for item in standards_blockers)
        markush_measured_blockers.extend(str(item) for item in standards_blockers)
        formal_blockers.extend(str(item) for item in standards_blockers)
    if not split_pose_preservation_ready:
        preservation_blockers = split_pose_preservation_contract.get("blockers")
        if not isinstance(preservation_blockers, list) or not preservation_blockers:
            preservation_blockers = ["split pose preservation contract is not green"]
        fragment_measured_blockers.extend(str(item) for item in preservation_blockers)
        markush_measured_blockers.extend(str(item) for item in preservation_blockers)
        formal_blockers.extend(str(item) for item in preservation_blockers)

    fragment_measured_blockers = unique_strings(fragment_measured_blockers)
    markush_measured_blockers = unique_strings(markush_measured_blockers)
    formal_blockers = unique_strings(formal_blockers)

    report = {
        "acceptance_report": str(args.acceptance_report),
        "split_manifest": str(args.split_manifest),
        "debug_micro_contract": str(debug_micro_contract_path),
        "markush_debug_micro_contract": str(markush_debug_micro_contract_path),
        "deprecated_cpu_smoke_contract_alias": str(args.cpu_smoke_contract),
        "deprecated_markush_cpu_smoke_contract_alias": str(args.markush_cpu_smoke_contract),
        "fragment_measured_contract": str(args.fragment_measured_contract),
        "markush_measured_contract": str(args.markush_measured_contract),
        "fragment_calibrated_measured_contract": str(args.fragment_calibrated_measured_contract),
        "markush_calibrated_measured_contract": str(args.markush_calibrated_measured_contract),
        "markush_architecture_eval_readiness_report": str(args.markush_architecture_eval_readiness_report),
        "training_scale_adequacy_report": str(args.training_scale_adequacy_report),
        "formal_acceptance_preflight_report": str(args.formal_acceptance_preflight_report),
        "training_entrypoint_preflight_gate_report": str(args.training_entrypoint_preflight_gate),
        "markush_standards_decision_gate_report": str(args.markush_standards_decision_gate),
        "split_pose_preservation_contract_report": str(args.split_pose_preservation_contract),
        "fragment_formal_contract": str(args.fragment_formal_contract),
        "markush_formal_contract": str(args.markush_formal_contract),
        "workspace_cleanliness_report": str(args.workspace_cleanliness_report),
        "gpu_runtime_evidence_report": str(args.gpu_runtime_evidence),
        "row_counts": row_counts,
        "split_stage": split_stage,
        "split_accepted_for_training": split_accepted_for_training is True,
        "acceptance_ok": acceptance_ok,
        "acceptance_measured_ok": acceptance_measured_ok,
        "acceptance_formal_ok": acceptance_formal_ok,
        "debug_micro_contract_ok": debug_micro_contract_ok,
        "markush_debug_micro_contract_ok": markush_debug_micro_contract_ok,
        "fragment_measured_contract_ok": fragment_measured_contract_ok,
        "markush_measured_contract_ok": markush_measured_contract_ok,
        "fragment_formal_contract_ok": fragment_formal_contract_ok,
        "markush_formal_contract_ok": markush_formal_contract_ok,
        "markush_architecture_eval_ready": markush_architecture_eval_ready,
        "training_scale_formal_ready": training_scale_formal_ready,
        "training_scale_architecture_ready": training_scale_architecture_ready,
        "markush_capacity_measured_candidate_ready": markush_capacity_measured_candidate_ready,
        "formal_acceptance_preflight_ready": formal_acceptance_preflight_ready,
        "training_entrypoint_preflight_ready": training_entrypoint_preflight_ready,
        "markush_standards_decision_ready": markush_standards_decision_ready,
        "split_pose_preservation_ready": split_pose_preservation_ready,
        "split_ok": split_ok,
        "workspace_cleanliness_ok": cleanliness_ok,
        "markush_consumed_by_current_training": markush_consumed_by_current_training,
        "training_scope": {
            "measured_smoke": "fragment_attachment_expert_current_entry",
            "formal": "all_current_routed_experts_require_measured_gpu_and_formal_calibrated_contracts",
            "markush_layout_expert_purpose": markush_purpose_text,
        },
        "measured_contracts": {
            "fragment": fragment_measured_contract,
            "markush": markush_measured_contract,
        },
        "calibrated_measured_contracts": {
            "fragment": fragment_calibrated_measured_contract,
            "markush": markush_calibrated_measured_contract,
        },
        "markush_architecture_eval_readiness": markush_architecture_eval_readiness,
        "training_scale_adequacy": training_scale_adequacy,
        "formal_acceptance_preflight": formal_acceptance_preflight,
        "training_entrypoint_preflight_gate": training_entrypoint_preflight,
        "markush_standards_decision_gate": markush_standards_decision_gate,
        "split_pose_preservation_contract": split_pose_preservation_contract,
        "formal_contracts": {
            "fragment": fragment_formal_contract,
            "markush": markush_formal_contract,
        },
        "gpu": gpu,
        "validation_scope": args.scope,
        "measured_smoke_start_allowed": not markush_measured_blockers,
        "markush_measured_smoke_start_allowed": not markush_measured_blockers,
        "fragment_measured_smoke_start_allowed": not fragment_measured_blockers,
        "formal_training_start_allowed": not formal_blockers,
        "blockers": markush_measured_blockers,
        "markush_measured_blockers": markush_measured_blockers,
        "fragment_measured_blockers": fragment_measured_blockers,
        "formal_blockers": formal_blockers,
        "policy": {
            "must_use_accepted_split": True,
            "split_manifest_must_be_formal_and_accepted_for_training": True,
            "formal_training_requires_formal_accepted_split": True,
            "markush_capacity_measured_candidate_is_not_formal_or_architecture_evidence": True,
            "candidate_split_can_satisfy_measured_architecture_comparison_only": True,
            "candidate_split_can_satisfy_markush_capacity_measured_evidence_only": True,
            "must_have_green_acceptance": True,
            "debug_micro_contract_is_debug_only_not_measured_or_formal_gate": True,
            "measured_smoke_requires_formal_accepted_split_or_architecture_ready_candidate_split_or_markush_capacity_measured_candidate": True,
            "measured_smoke_requires_split_pose_preservation_contract": True,
            "fragment_debug_micro_contract_is_debug_only_not_formal_gate": True,
            "markush_debug_micro_contract_is_debug_only_not_formal_gate": True,
            "formal_training_requires_fragment_measured_gpu_contract": True,
            "formal_training_requires_markush_measured_gpu_contract": True,
            "formal_training_requires_fragment_calibrated_measured_contract": True,
            "formal_training_requires_markush_calibrated_measured_contract": True,
            "formal_training_requires_markush_architecture_eval_readiness": True,
            "formal_training_requires_training_scale_adequacy": True,
            "formal_training_requires_formal_acceptance_preflight": True,
            "formal_training_requires_training_entrypoint_preflight_gate": True,
            "formal_training_requires_markush_standards_decision_when_provided": markush_standards_decision_gate_is_report,
            "markush_standards_decision_json_is_planning_evidence_not_a_gate": bool(args.markush_standards_decision_gate)
            and not markush_standards_decision_gate_is_report,
            "formal_training_requires_split_pose_preservation_contract": True,
            "formal_run_contracts_validate_finished_formal_outputs": True,
            "measured_and_formal_run_contracts_require_model_scale_policy": True,
            "debug_only_runs_cannot_satisfy_measured_or_formal": True,
            "must_have_clean_training_workspace": bool(args.workspace_cleanliness_report),
            "must_have_working_gpu_for_measured_or_formal_training": True,
            "formal_training_must_consume_markush_split": True,
            "complete_molnextr_path_remains_frozen": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if args.scope == "measured_smoke":
        should_fail = bool(fragment_measured_blockers)
    elif args.scope == "formal":
        should_fail = bool(formal_blockers)
    else:
        should_fail = bool(fragment_measured_blockers or formal_blockers)
    if should_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
