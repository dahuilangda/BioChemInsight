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


def resolve_report_path(raw_path: Any, *, run_dir: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = run_dir / path
        if candidate.exists():
            return candidate
        try:
            archived = run_dir / path.relative_to(run_dir.name)
            if archived.exists():
                return archived
        except ValueError:
            pass
        for marker in ["training/molnextr_markush/runs/sidecar_smoke"]:
            try:
                archived = Path("training/molnextr_markush/runs/legacy_runs/sidecar_smoke") / path.relative_to(marker)
                if archived.exists():
                    return archived
            except ValueError:
                continue
    return path


def audit_split_manifest(metrics: dict[str, Any], split_manifest_path: str) -> list[str]:
    if not split_manifest_path:
        return []
    blockers: list[str] = []
    manifest = load_json(split_manifest_path)
    expected = manifest.get("output_sha256") if isinstance(manifest.get("output_sha256"), dict) else {}
    if not expected:
        return [f"split manifest has no output_sha256: {split_manifest_path}"]
    config = metrics.get("config") if isinstance(metrics.get("config"), dict) else {}
    observed = config.get("input_sha256") if isinstance(config.get("input_sha256"), dict) else {}
    if not observed:
        return ["metrics.config.input_sha256 is missing; run cannot be tied to the current split manifest"]
    comparisons = [
        ("positive_label_csv", "fragment_train_csv"),
        ("eval_positive_label_csv", "fragment_calibration_csv"),
        ("negative_csv", "ordinary_train_csv"),
        ("eval_negative_csv", "ordinary_calibration_csv"),
    ]
    for observed_name, manifest_name in comparisons:
        observed_hash = observed.get(observed_name)
        expected_hash = expected.get(manifest_name)
        if not expected_hash:
            blockers.append(f"split manifest missing output_sha256.{manifest_name}")
        elif not observed_hash:
            blockers.append(f"metrics.config.input_sha256 missing {observed_name}")
        elif str(observed_hash) != str(expected_hash):
            blockers.append(f"{observed_name} hash does not match current split manifest {manifest_name}")
    return blockers


def audit_confidence_report(path: Path, *, require_deployable: bool) -> tuple[list[str], list[str], dict[str, Any]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return [f"missing confidence report: {path}"], warnings, {}
    report = load_json(path)
    confidence = report.get("confidence") if isinstance(report.get("confidence"), dict) else {}
    for head in [
        "applicability_confidence",
        "presence_confidence",
        "side_confidence",
        "anchor_confidence",
        "query_objectness_confidence",
        "no_endpoint_confidence",
        "risk_confidence",
        "sidecar_confidence",
    ]:
        head_report = confidence.get(head) if isinstance(confidence.get(head), dict) else {}
        if "brier" not in head_report:
            blockers.append(f"confidence report missing {head}.brier")
        if "ece" not in head_report:
            blockers.append(f"confidence report missing {head}.ece")
    if not isinstance(report.get("threshold_curve"), list) or not report.get("threshold_curve"):
        blockers.append("confidence report has no threshold_curve")
    if not isinstance(report.get("high_confidence_errors"), list):
        blockers.append("confidence report missing high_confidence_errors list")
    if require_deployable:
        selected = report.get("selected_threshold")
        constraints = report.get("quality_constraints") if isinstance(report.get("quality_constraints"), dict) else {}
        sidecar_confidence = confidence.get("sidecar_confidence") if isinstance(confidence.get("sidecar_confidence"), dict) else {}
        hard_min_coverage = max(0.05, float(constraints.get("min_selected_coverage") or 0.0))
        hard_min_positive_recall = max(0.10, float(constraints.get("min_positive_triplet_recall") or 0.0))
        hard_min_positive_accepted = max(100, int(constraints.get("min_positive_accepted") or 0))
        hard_min_wilson = max(0.70, float(constraints.get("min_positive_precision_wilson_lower") or 0.0))
        hard_max_sidecar_ece = min(0.20, float(constraints.get("max_sidecar_confidence_ece") or 0.20))
        hard_max_sidecar_brier = min(0.25, float(constraints.get("max_sidecar_confidence_brier") or 0.25))
        if not isinstance(selected, dict):
            blockers.append("formal run has no selected calibrated threshold")
        else:
            max_negative_attachment = int(constraints.get("max_negative_predicted_attachment") or 0)
            max_negative_accepted = int(constraints.get("max_negative_accepted") or 0)
            if int(selected.get("negative_predicted_attachment") or 0) > max_negative_attachment:
                blockers.append("selected threshold exceeds negative predicted-attachment limit")
            if int(selected.get("negative_accepted") or 0) > max_negative_accepted:
                blockers.append("selected threshold exceeds negative sidecar-accepted limit")
            if float(selected.get("positive_triplet_precision") or 0.0) < float(
                constraints.get("min_positive_triplet_precision") or 0.95
            ):
                blockers.append("selected threshold does not meet positive triplet precision")
            if int(selected.get("positive_accepted") or 0) < hard_min_positive_accepted:
                blockers.append(
                    f"selected threshold accepts too few positives for formal confidence "
                    f"({int(selected.get('positive_accepted') or 0)} < {hard_min_positive_accepted})"
                )
            if float(selected.get("coverage") or 0.0) < hard_min_coverage:
                blockers.append(
                    f"selected threshold coverage is too low for formal confidence "
                    f"({float(selected.get('coverage') or 0.0):.6f} < {hard_min_coverage:.6f})"
                )
            if float(selected.get("positive_triplet_recall") or 0.0) < hard_min_positive_recall:
                blockers.append(
                    f"selected threshold positive triplet recall is too low for formal confidence "
                    f"({float(selected.get('positive_triplet_recall') or 0.0):.6f} < {hard_min_positive_recall:.6f})"
                )
            if "positive_triplet_precision_wilson_lower" not in selected:
                blockers.append("selected threshold is missing Wilson precision lower bound")
            elif float(selected.get("positive_triplet_precision_wilson_lower") or 0.0) < hard_min_wilson:
                blockers.append("selected threshold does not meet Wilson precision lower bound")
            if float(sidecar_confidence.get("ece") or 1.0) > hard_max_sidecar_ece:
                blockers.append(
                    f"fragment sidecar confidence ECE is too high for formal confidence "
                    f"({float(sidecar_confidence.get('ece') or 1.0):.6f} > {hard_max_sidecar_ece:.6f})"
                )
            if float(sidecar_confidence.get("brier") or 1.0) > hard_max_sidecar_brier:
                blockers.append(
                    f"fragment sidecar confidence Brier is too high for formal confidence "
                    f"({float(sidecar_confidence.get('brier') or 1.0):.6f} > {hard_max_sidecar_brier:.6f})"
                )
        selected_errors = report.get("selected_error_summary")
        if isinstance(selected_errors, dict):
            if int(selected_errors.get("negative_accepted_errors") or 0) > max_negative_accepted:
                blockers.append("selected threshold has deployment-blocking ordinary-negative accepted errors")
            if int(selected_errors.get("negative_predicted_attachment_errors") or 0) > max_negative_attachment:
                blockers.append("selected threshold has deployment-blocking ordinary-negative attachment errors")
        elif report.get("selected_high_confidence_errors") not in (0, 0.0):
            blockers.append("legacy confidence report has selected-threshold high-confidence errors")
        if report.get("deployment_allowed") is not True:
            blockers.append("confidence report does not allow deployment")
    else:
        if report.get("deployment_allowed") is not True:
            warnings.append("confidence report is present but not deployable")
    return blockers, warnings, report


def audit_model_scale(metrics: dict[str, Any], *, require_evidence_scale: bool) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    model_scale = metrics.get("model_scale") if isinstance(metrics.get("model_scale"), dict) else {}
    if not model_scale:
        blockers.append("metrics.model_scale is missing")
        return blockers, warnings
    parameters = model_scale.get("parameters") if isinstance(model_scale.get("parameters"), dict) else {}
    for key in ["total_parameters", "trainable_parameters", "frozen_parameters", "trainable_ratio"]:
        if key not in parameters:
            blockers.append(f"metrics.model_scale.parameters.{key} is missing")
    expert_branch = model_scale.get("expert_branch") if isinstance(model_scale.get("expert_branch"), dict) else {}
    required_expert_keys = ["hidden_dim", "fusion_transformer_depth", "fusion_transformer_heads", "endpoint_queries"]
    for key in required_expert_keys:
        if key not in expert_branch:
            blockers.append(f"metrics.model_scale.expert_branch.{key} is missing")
    scale_observed = model_scale.get("scale_observed") if isinstance(model_scale.get("scale_observed"), dict) else {}
    scale_minimums = model_scale.get("scale_minimums") if isinstance(model_scale.get("scale_minimums"), dict) else {}
    if not scale_observed:
        blockers.append("metrics.model_scale.scale_observed is missing")
    if not scale_minimums:
        blockers.append("metrics.model_scale.scale_minimums is missing")
    observed_hidden = int(scale_observed.get("hidden_dim") or expert_branch.get("hidden_dim") or 0)
    observed_depth = int(
        scale_observed.get("fusion_transformer_depth") or expert_branch.get("fusion_transformer_depth") or 0
    )
    observed_heads = int(
        scale_observed.get("fusion_transformer_heads") or expert_branch.get("fusion_transformer_heads") or 0
    )
    observed_queries = int(scale_observed.get("endpoint_queries") or expert_branch.get("endpoint_queries") or 0)
    observed_patch = int(scale_observed.get("patch_size") or expert_branch.get("patch_size") or 0)
    if observed_hidden < 256:
        blockers.append(f"fragment expert hidden_dim {observed_hidden} < 256")
    if observed_depth < 4:
        blockers.append(f"fragment expert fusion depth {observed_depth} < 4")
    if observed_heads < 8:
        blockers.append(f"fragment expert attention heads {observed_heads} < 8")
    if observed_queries < 4:
        blockers.append(f"fragment endpoint queries {observed_queries} < 4")
    if observed_patch <= 0 or observed_patch > 16:
        blockers.append(f"fragment patch_size {observed_patch} is not in the formal range <=16")
    image_resolution = scale_observed.get("image_resolution")
    if not isinstance(image_resolution, list) or len(image_resolution) != 2 or not all(int(value or 0) > 0 for value in image_resolution):
        blockers.append("metrics.model_scale.scale_observed.image_resolution is missing or invalid")
    if int(scale_observed.get("image_token_count") or 0) <= 0:
        blockers.append("metrics.model_scale.scale_observed.image_token_count is missing or invalid")
    if int(scale_observed.get("graph_feature_dim") or 0) <= 0:
        blockers.append("metrics.model_scale.scale_observed.graph_feature_dim is missing or invalid")
    backbone = model_scale.get("backbone") if isinstance(model_scale.get("backbone"), dict) else {}
    if backbone.get("pretrained_backbone_used") is not True:
        blockers.append("metrics.model_scale.backbone.pretrained_backbone_used is not true")
    if str(backbone.get("complete_path") or "") != "direct_original_molnextr":
        blockers.append("metrics.model_scale.backbone.complete_path is not direct_original_molnextr")
    if "molnextr_best.pth" not in str(backbone.get("checkpoint") or ""):
        blockers.append("metrics.model_scale.backbone.checkpoint does not reference molnextr_best.pth")
    routing = model_scale.get("routing") if isinstance(model_scale.get("routing"), dict) else {}
    if routing.get("expert_branch") != "fragment_attachment":
        blockers.append("metrics.model_scale.routing.expert_branch is not fragment_attachment")
    if routing.get("complete_molecules_enter_expert") is not False:
        blockers.append("metrics.model_scale.routing.complete_molecules_enter_expert must be false")
    if routing.get("requires_explicit_router_decision") is not True:
        blockers.append("metrics.model_scale.routing.requires_explicit_router_decision must be true")
    training_scale = model_scale.get("training_scale") if isinstance(model_scale.get("training_scale"), dict) else {}
    for key in [
        "batch_size_per_process",
        "effective_batch_size",
        "epochs",
        "ddp",
        "world_size",
        "gpu_count_requested",
        "gpu_count_observed",
        "distributed_strategy",
        "per_gpu_peak_memory_gb",
        "throughput_samples_per_second",
    ]:
        if key not in training_scale:
            blockers.append(f"metrics.model_scale.training_scale.{key} is missing")
    if require_evidence_scale:
        if int(training_scale.get("gpu_count_observed") or 0) < 2:
            blockers.append("fragment measured/formal evidence requires two observed GPUs or a measured single-GPU rationale")
        if int(training_scale.get("world_size") or 0) < 2 and not str(training_scale.get("single_gpu_rationale") or "").strip():
            blockers.append("fragment measured/formal evidence lacks DDP world_size>=2 and has no measured single-GPU rationale")
        peak_memory = training_scale.get("per_gpu_peak_memory_gb")
        if not isinstance(peak_memory, list) or not peak_memory:
            blockers.append("metrics.model_scale.training_scale.per_gpu_peak_memory_gb is missing")
        if training_scale.get("throughput_samples_per_second") in (None, ""):
            blockers.append("metrics.model_scale.training_scale.throughput_samples_per_second is missing")
    if metrics.get("debug_only") is True or model_scale.get("debug_only") is True:
        message = "run is marked debug_only and cannot be measured/formal evidence"
        if require_evidence_scale:
            blockers.append(message)
        else:
            warnings.append(message)
    if require_evidence_scale and model_scale.get("allowed_for_measured_or_formal_evidence") is not True:
        blockers.append("model_scale does not allow measured/formal evidence")
    scale_blockers = model_scale.get("scale_blockers") if isinstance(model_scale.get("scale_blockers"), list) else []
    if require_evidence_scale:
        blockers.extend(f"model scale blocker: {message}" for message in scale_blockers)
    else:
        warnings.extend(f"model scale blocker: {message}" for message in scale_blockers)
    return blockers, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a fragment attachment expert run against the frozen-base contract.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--require-measured-stage", action="store_true")
    parser.add_argument("--require-formal-stage", action="store_true")
    parser.add_argument(
        "--require-calibrated-confidence",
        action="store_true",
        help="Require the run confidence report to select a deployable calibrated threshold.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "fragment_attachment_expert.pth"
    last_checkpoint_path = run_dir / "fragment_attachment_expert_last.pth"
    best_checkpoint_path = run_dir / "fragment_attachment_expert_best.pth"
    training_state_path = run_dir / "fragment_attachment_training_state_last.pth"
    epoch_checkpoints = sorted(run_dir.glob("fragment_attachment_expert_epoch_*.pth"))
    blockers: list[str] = []
    warnings: list[str] = []
    if not metrics_path.exists():
        blockers.append(f"missing metrics.json: {metrics_path}")
        metrics: dict[str, Any] = {}
    else:
        metrics = load_json(metrics_path)
    recoverable_checkpoint = last_checkpoint_path.exists() or best_checkpoint_path.exists() or bool(epoch_checkpoints)
    if not checkpoint_path.exists():
        blockers.append(f"missing fragment attachment expert checkpoint: {checkpoint_path}")
        if recoverable_checkpoint:
            warnings.append(
                "final checkpoint is missing, but resumable/evaluable partial checkpoint(s) exist; "
                "use last/best only for resume or explicit diagnostic eval, not formal deployment"
            )
    if not training_state_path.exists():
        message = f"missing resumable training state checkpoint: {training_state_path}"
        if args.require_measured_stage or args.require_formal_stage:
            blockers.append(message)
        else:
            warnings.append(message)
    if metrics:
        for key, path in [
            ("latest_checkpoint", last_checkpoint_path),
            ("best_checkpoint", best_checkpoint_path),
            ("final_checkpoint", checkpoint_path),
        ]:
            recorded = str(metrics.get(key) or "")
            if recorded and not Path(recorded).exists() and not (run_dir / recorded).exists():
                warnings.append(f"metrics.{key} points to a missing file: {recorded}")

    if metrics:
        if metrics.get("sidecar_only") is not True:
            blockers.append("metrics.sidecar_only is not true")
        if metrics.get("complete_path_mutated") is not False:
            blockers.append("complete MolNexTR path was marked mutated")
        if metrics.get("encoder_trainable") is not False:
            blockers.append("encoder_trainable must remain false")
        if metrics.get("decoder_loaded") is not False:
            blockers.append("decoder_loaded must remain false")
        if metrics.get("formal_full_checkpoint_training_allowed") is not False:
            blockers.append("run must not mark full checkpoint training allowed")
        scale_blockers, scale_warnings = audit_model_scale(
            metrics,
            require_evidence_scale=bool(args.require_measured_stage or args.require_formal_stage),
        )
        blockers.extend(scale_blockers)
        warnings.extend(scale_warnings)
        blockers.extend(audit_split_manifest(metrics, str(args.split_manifest or "")))
        config = metrics.get("config") if isinstance(metrics.get("config"), dict) else {}
        if args.require_measured_stage or args.require_formal_stage:
            if config.get("cpu") is True:
                blockers.append("measured/formal validation requires a non-CPU GPU run")

        gate = metrics.get("gate_report") if isinstance(metrics.get("gate_report"), dict) else {}
        if gate.get("enforced") is not True:
            blockers.append("gate_report.enforced is not true")
        if gate.get("accepted") is not True:
            blockers.append("gate_report.accepted is not true")
        if gate.get("measured_sidecar_smoke_allowed") is not True:
            blockers.append("acceptance report did not allow measured sidecar smoke")
        if args.require_formal_stage and gate.get("formal_training_allowed") is not True:
            blockers.append("acceptance report did not allow formal training")
        readiness_gate = gate.get("readiness_gate") if isinstance(gate.get("readiness_gate"), dict) else {}
        if args.require_measured_stage:
            if gate.get("training_stage") != "measured_smoke":
                blockers.append("measured validation requires gate_report.training_stage=measured_smoke")
            if readiness_gate.get("provided") is not True:
                blockers.append("measured validation requires a readiness gate report")
            if readiness_gate.get("measured_smoke_start_allowed") is not True:
                blockers.append("measured validation requires readiness measured_smoke_start_allowed=true")
            if readiness_gate.get("accepted") is not True:
                blockers.append("measured validation readiness gate is not accepted")
        if args.require_formal_stage and gate.get("training_stage") != "formal":
            blockers.append("training_stage is not formal")
        if args.require_formal_stage:
            if readiness_gate.get("provided") is not True:
                blockers.append("formal validation requires a readiness gate report")
            if readiness_gate.get("formal_training_start_allowed") is not True:
                blockers.append("formal validation requires readiness formal_training_start_allowed=true")
            if readiness_gate.get("accepted") is not True:
                blockers.append("formal validation readiness gate is not accepted")
            formal_preflight_gate = (
                gate.get("formal_preflight_gate") if isinstance(gate.get("formal_preflight_gate"), dict) else {}
            )
            if formal_preflight_gate.get("provided") is not True:
                blockers.append("formal validation requires formal_preflight_gate.provided=true")
            if formal_preflight_gate.get("accepted_for_formal_split") is not True:
                blockers.append("formal validation requires formal preflight accepted_for_formal_split=true")
            if formal_preflight_gate.get("formal_training_start_allowed") is not True:
                blockers.append("formal validation requires formal preflight formal_training_start_allowed=true")
            red_gates = formal_preflight_gate.get("red_gates")
            if isinstance(red_gates, list) and red_gates:
                blockers.append(f"formal validation requires no formal preflight red gates: {red_gates}")

        image_audit = metrics.get("image_audit") if isinstance(metrics.get("image_audit"), dict) else {}
        for split in ["train", "eval"]:
            split_report = image_audit.get(split) if isinstance(image_audit.get(split), dict) else {}
            if split_report.get("all_paths_absolute") is not True:
                blockers.append(f"{split} image paths were not all absolute")
            if int(split_report.get("missing") or 0) != 0:
                blockers.append(f"{split} image audit has missing files")
            if int(split_report.get("unreadable") or 0) != 0:
                blockers.append(f"{split} image audit has unreadable files")
            if int(split_report.get("checked_rows") or 0) <= 0:
                blockers.append(f"{split} image audit checked no rows")

        epochs = metrics.get("epochs") if isinstance(metrics.get("epochs"), list) else []
        if not epochs:
            blockers.append("metrics.epochs is empty")
        else:
            eval_report = epochs[-1].get("eval") if isinstance(epochs[-1], dict) else {}
            if isinstance(eval_report, dict):
                if float(eval_report.get("negative_mean_sidecar_confidence") or 0.0) >= float(
                    eval_report.get("positive_mean_sidecar_confidence") or 0.0
                ):
                    warnings.append("positive mean sidecar confidence is not greater than negative mean confidence")
            confidence_report_path = resolve_report_path(
                epochs[-1].get("confidence_report") if isinstance(epochs[-1], dict) else None,
                run_dir=run_dir,
            )
            require_deployable_confidence = args.require_formal_stage or args.require_calibrated_confidence
            if require_deployable_confidence:
                if confidence_report_path is None:
                    blockers.append("run missing epoch confidence_report path required for calibrated confidence")
                else:
                    confidence_blockers, confidence_warnings, _confidence_report = audit_confidence_report(
                        confidence_report_path,
                        require_deployable=True,
                    )
                    blockers.extend(confidence_blockers)
                    warnings.extend(confidence_warnings)
            elif confidence_report_path is not None:
                confidence_blockers, confidence_warnings, _confidence_report = audit_confidence_report(
                    confidence_report_path,
                    require_deployable=False,
                )
                warnings.extend(f"confidence report issue: {message}" for message in confidence_blockers)
                warnings.extend(confidence_warnings)

    report = {
        "schema_version": "fragment_attachment_expert_run_contract_v2",
        "run_dir": str(run_dir),
        "metrics": str(metrics_path),
        "checkpoint": str(checkpoint_path),
        "checkpoints": {
            "final": str(checkpoint_path),
            "final_exists": checkpoint_path.exists(),
            "last": str(last_checkpoint_path),
            "last_exists": last_checkpoint_path.exists(),
            "best": str(best_checkpoint_path),
            "best_exists": best_checkpoint_path.exists(),
            "training_state": str(training_state_path),
            "training_state_exists": training_state_path.exists(),
            "epoch_checkpoints": [str(path) for path in epoch_checkpoints],
            "recoverable_checkpoint_exists": bool(recoverable_checkpoint),
        },
        "passed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "policy": {
            "complete_path_must_not_mutate": True,
            "encoder_decoder_frozen": True,
            "gate_report_required": True,
            "current_split_manifest_hash_must_match": bool(args.split_manifest),
            "image_audit_required": True,
            "measured_stage_required": bool(args.require_measured_stage),
            "measured_stage_must_be_gpu": bool(args.require_measured_stage),
            "measured_stage_readiness_required": bool(args.require_measured_stage),
            "formal_confidence_report_required": bool(args.require_formal_stage),
            "calibrated_confidence_required": bool(args.require_calibrated_confidence),
            "calibrated_confidence_requires_min_coverage": bool(args.require_calibrated_confidence or args.require_formal_stage),
            "calibrated_confidence_requires_min_positive_recall": bool(args.require_calibrated_confidence or args.require_formal_stage),
            "calibrated_confidence_requires_wilson_lower_bound": bool(args.require_calibrated_confidence or args.require_formal_stage),
            "calibrated_confidence_requires_sidecar_ece_brier_caps": bool(args.require_calibrated_confidence or args.require_formal_stage),
            "formal_selected_threshold_must_be_deployable": bool(args.require_formal_stage),
            "formal_stage_readiness_required": bool(args.require_formal_stage),
            "formal_preflight_gate_required": bool(args.require_formal_stage),
            "model_scale_report_required": True,
            "model_scale_gate_enforced": True,
            "router_branch_gate_enforced": True,
            "runtime_gpu_gate_enforced": True,
            "dual_gpu_or_measured_single_gpu_required": bool(args.require_measured_stage or args.require_formal_stage),
            "debug_only_runs_cannot_satisfy_measured_or_formal": True,
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
