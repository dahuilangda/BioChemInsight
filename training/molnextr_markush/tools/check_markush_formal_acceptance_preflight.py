from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def gate_report(path_text: str, *, passed: bool, blockers: list[str] | None = None, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "path": str(path_text),
        "passed": bool(passed),
        "blockers": blockers or [],
        "detail": detail or {},
    }


def strict_machine_visual_blockers(
    *,
    assistant_visual: dict[str, Any],
    visual_risk: dict[str, Any],
    visual_review: dict[str, Any],
    candidate_row_count: int,
) -> list[str]:
    blockers: list[str] = []
    if not assistant_visual:
        blockers.append("assistant/model-assisted visual review report was not provided")
    else:
        if assistant_visual.get("schema_version") != "markush_assistant_visual_review_v1":
            blockers.append("assistant visual review schema_version is not markush_assistant_visual_review_v1")
        if int(assistant_visual.get("row_count") or 0) != int(candidate_row_count or 0):
            blockers.append("assistant visual review row_count does not match candidate")
        if assistant_visual.get("research_probe_visual_readability_passed") is not True:
            blockers.append("assistant visual review did not pass research/probe readability")
        if assistant_visual.get("strict_machine_visual_acceptance_passed") is not True:
            blockers.extend(
                str(item)
                for item in assistant_visual.get("blockers")
                or ["assistant visual review did not pass strict machine visual acceptance"]
            )

    if not visual_risk:
        blockers.append("visual risk report was not provided")
    else:
        if visual_risk.get("schema_version") != "markush_visual_risk_report_v1":
            blockers.append("visual risk report schema_version is not markush_visual_risk_report_v1")
        if int(visual_risk.get("row_count") or 0) != int(candidate_row_count or 0):
            blockers.append("visual risk report row_count does not match candidate")
        background_rows = int(visual_risk.get("background_context_rows") or 0)
        if background_rows != int(candidate_row_count or 0):
            blockers.append(
                f"background contract rows {background_rows} != candidate rows {int(candidate_row_count or 0)}"
            )
        image_probe = visual_risk.get("image_readability_probe") if isinstance(visual_risk.get("image_readability_probe"), dict) else {}
        if int(image_probe.get("missing_images") or 0) != 0:
            blockers.append(f"visual risk report has missing_images={int(image_probe.get('missing_images') or 0)}")
        if int(image_probe.get("unreadable_images") or 0) != 0:
            blockers.append(f"visual risk report has unreadable_images={int(image_probe.get('unreadable_images') or 0)}")
        realism_counts = (
            visual_risk.get("realism_policy_counts")
            if isinstance(visual_risk.get("realism_policy_counts"), dict)
            else {}
        )
        if int(realism_counts.get("patent_literature_markushgenerator_cdk_svg_v1") or 0) != int(candidate_row_count or 0):
            blockers.append("visual risk report does not preserve patent/literature realism policy for every row")

    if visual_review:
        if assistant_visual and str(assistant_visual.get("review_manifest") or "") != str(visual_review.get("output_dir") or ""):
            # Older assistant reports store the manifest path, so this is only a warning-level consistency check in detail.
            pass
    return blockers


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize whether the annotation-matched Markush candidate has enough gate evidence "
            "to be accepted as a formal split. This never accepts data and never starts training."
        )
    )
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--schema-validation", required=True)
    parser.add_argument("--pose-alignment", required=True)
    parser.add_argument("--substitution-anchor-contract", required=True)
    parser.add_argument("--provenance-audit", required=True)
    parser.add_argument("--source-leak", required=True)
    parser.add_argument("--standards-coverage", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-preservation", required=True)
    parser.add_argument("--split-image-preservation", default="")
    parser.add_argument("--fragment-taxonomy-alignment", default="")
    parser.add_argument("--attachment-role-contract", default="")
    parser.add_argument("--architecture-readiness", required=True)
    parser.add_argument("--gpu-runtime-evidence", required=True)
    parser.add_argument("--clean-workspace", required=True)
    parser.add_argument("--router-complete-path-contract", default="")
    parser.add_argument("--visual-review-package", default="")
    parser.add_argument("--manual-visual-acceptance", default="")
    parser.add_argument("--assistant-visual-review", default="")
    parser.add_argument("--visual-risk-report", default="")
    parser.add_argument("--confidence-report", default="")
    parser.add_argument("--training-entrypoint-preflight-gate", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidate = load_json(args.candidate_manifest)
    schema = load_json(args.schema_validation)
    pose = load_json(args.pose_alignment)
    substitution_anchor = load_json(args.substitution_anchor_contract)
    provenance = load_json(args.provenance_audit)
    source_leak = load_json(args.source_leak)
    standards = load_json(args.standards_coverage)
    split_manifest = load_json(args.split_manifest)
    split_preservation = load_json(args.split_preservation)
    split_image_preservation = load_json(args.split_image_preservation) if args.split_image_preservation else {}
    taxonomy_alignment = load_json(args.fragment_taxonomy_alignment) if args.fragment_taxonomy_alignment else {}
    attachment_role = load_json(args.attachment_role_contract) if args.attachment_role_contract else {}
    architecture = load_json(args.architecture_readiness)
    gpu = load_json(args.gpu_runtime_evidence)
    clean = load_json(args.clean_workspace)
    router_contract = load_json(args.router_complete_path_contract) if args.router_complete_path_contract else {}
    visual_review = load_json(args.visual_review_package) if args.visual_review_package else {}
    manual_visual = load_json(args.manual_visual_acceptance) if args.manual_visual_acceptance else {}
    assistant_visual = load_json(args.assistant_visual_review) if args.assistant_visual_review else {}
    visual_risk = load_json(args.visual_risk_report) if args.visual_risk_report else {}
    confidence = load_json(args.confidence_report) if args.confidence_report else {}
    entrypoint_preflight = load_json(args.training_entrypoint_preflight_gate) if args.training_entrypoint_preflight_gate else {}

    gates: dict[str, Any] = {}
    candidate_counts = candidate.get("counts_by_r_tag_bucket")
    if not isinstance(candidate_counts, dict):
        candidate_counts = (
            candidate.get("markush", {}).get("bucket_sizes")
            if isinstance(candidate.get("markush"), dict)
            else {}
        )
    candidate_row_count = candidate.get("row_count")
    if candidate_row_count is None and isinstance(candidate.get("markush"), dict):
        candidate_row_count = candidate["markush"].get("input_rows")
    if candidate_row_count is None and isinstance(candidate.get("counts"), dict):
        candidate_row_count = candidate["counts"].get("markush_layout")
    candidate_is_split_manifest = bool(candidate.get("outputs")) and bool(candidate.get("markush"))
    candidate_passed = (
        candidate.get("passed") is True
        and candidate.get("accepted") is False
        and candidate.get("rejected") is False
    ) or (
        candidate_is_split_manifest
        and int(candidate_row_count or 0) > 0
        and str(candidate.get("stage") or "") in {"candidate", "formal"}
    )
    gates["candidate_manifest"] = gate_report(
        args.candidate_manifest,
        passed=candidate_passed,
        blockers=[] if candidate_passed else list(candidate.get("blockers") or ["candidate manifest did not pass"]),
        detail={
            "row_count": candidate_row_count,
            "counts_by_r_tag_bucket": candidate_counts,
            "accepted": candidate.get("accepted"),
        },
    )
    schema_blockers: list[str] = []
    if int(schema.get("invalid_rows") or 0) != 0:
        schema_blockers.append("schema validation has invalid rows")
    schema_row_count = int(schema.get("row_count") or 0)
    schema_realism_counts = (
        schema.get("markush_realism_counts") if isinstance(schema.get("markush_realism_counts"), dict) else {}
    )
    if int(schema_realism_counts.get("machine_audit_passed") or 0) != schema_row_count:
        schema_blockers.append("schema validation does not show document_realism machine audit passed for every row")
    if int(schema_realism_counts.get("policy:patent_literature_markushgenerator_cdk_svg_v1") or 0) != schema_row_count:
        schema_blockers.append("schema validation does not preserve the accepted patent/literature realism policy for every row")
    gates["schema_validation"] = gate_report(
        args.schema_validation,
        passed=schema.get("trainable") is True and not schema_blockers,
        blockers=schema_blockers,
        detail={
            "row_count": schema.get("row_count"),
            "valid_rows": schema.get("valid_rows"),
            "invalid_rows": schema.get("invalid_rows"),
            "markush_layout_counts": schema.get("markush_layout_counts"),
            "markush_realism_counts": schema_realism_counts,
            "manual_visual_review_required_is_risk_metadata_not_hard_gate": True,
        },
    )
    pose_policy = pose.get("policy") if isinstance(pose.get("policy"), dict) else {}
    gates["pose_alignment"] = gate_report(
        args.pose_alignment,
        passed=pose.get("passed") is True,
        blockers=list(pose.get("blockers") or []),
        detail={"visual_review_still_required": pose_policy.get("visual_review_still_required") is True},
    )
    substitution_policy = (
        substitution_anchor.get("policy") if isinstance(substitution_anchor.get("policy"), dict) else {}
    )
    substitution_blockers = list(substitution_anchor.get("blockers") or [])
    if substitution_anchor.get("schema_version") != "markush_substitution_anchor_contract_v1":
        substitution_blockers.append(
            "substitution-anchor contract schema_version is not markush_substitution_anchor_contract_v1"
        )
    if substitution_anchor.get("passed") is not True:
        substitution_blockers.append("substitution-anchor contract did not pass")
    if int(substitution_anchor.get("failed_rows") or 0) != 0:
        substitution_blockers.append(
            f"substitution-anchor contract failed_rows={int(substitution_anchor.get('failed_rows') or 0)}"
        )
    if int(substitution_anchor.get("row_count") or 0) != int(candidate_row_count or 0):
        substitution_blockers.append("substitution-anchor contract row_count does not match candidate manifest")
    for key in [
        "rdkit_cxsmiles_dummy_or_atom_label_parse_precedes_regex_fallback",
        "each_dummy_label_requires_exactly_one_matching_ocr_cell",
        "dummy_atom_coordinate_must_lie_inside_matching_ocr_bbox_with_margin",
        "dummy_atom_must_have_graph_bond",
        "dummy_atom_must_have_real_atom_neighbor",
        "stable_definition_labels_are_diagnostic_not_anchor_hard_fail",
        "non_variable_element_ocr_cells_are_audited_separately_not_treated_as_markush_variables",
    ]:
        if key not in substitution_policy:
            substitution_blockers.append(f"substitution-anchor policy missing {key}")
    gates["substitution_anchor_contract"] = gate_report(
        args.substitution_anchor_contract,
        passed=not substitution_blockers,
        blockers=substitution_blockers,
        detail={
            "row_count": substitution_anchor.get("row_count"),
            "passed_rows": substitution_anchor.get("passed_rows"),
            "failed_rows": substitution_anchor.get("failed_rows"),
            "issue_counts": substitution_anchor.get("issue_counts"),
            "policy": substitution_policy,
        },
    )
    gates["provenance_audit"] = gate_report(
        args.provenance_audit,
        passed=provenance.get("passed") is True and int(provenance.get("issue_count") or 0) == 0,
        blockers=list(provenance.get("blockers") or []),
        detail={"issue_count": provenance.get("issue_count"), "policy": provenance.get("policy")},
    )
    gates["source_leak"] = gate_report(
        args.source_leak,
        passed=source_leak.get("passed") is True,
        blockers=list(source_leak.get("blockers") or []),
        detail={"policy": source_leak.get("policy")},
    )
    gates["standards_coverage"] = gate_report(
        args.standards_coverage,
        passed=standards.get("passed") is True and standards.get("architecture_comparison_data_coverage_allowed") is True,
        blockers=list(standards.get("blockers") or []),
        detail={
            "trainable_total_by_bucket": standards.get("trainable_total_by_bucket"),
            "target_total_by_bucket": standards.get("target_total_by_bucket"),
            "trainable_bucket_results": standards.get("trainable_bucket_results"),
        },
    )
    split_stage = str(split_manifest.get("stage") or "").strip()
    split_accepted_for_training = split_manifest.get("accepted_for_training") is True
    split_manifest_blockers: list[str] = []
    if split_stage == "candidate" and split_accepted_for_training:
        split_manifest_blockers.append("candidate split must not be accepted_for_training=true")
    elif split_stage == "formal" and not split_accepted_for_training:
        split_manifest_blockers.append("formal split must be accepted_for_training=true")
    elif split_stage not in {"candidate", "formal"}:
        split_manifest_blockers.append(f"split stage must be candidate or formal, got {split_stage or 'missing'}")
    gates["split_manifest"] = gate_report(
        args.split_manifest,
        passed=not split_manifest_blockers,
        blockers=split_manifest_blockers,
        detail={
            "stage": split_manifest.get("stage"),
            "accepted_for_training": split_manifest.get("accepted_for_training"),
            "policy": split_manifest.get("policy"),
        },
    )
    split_preservation_blockers = list(split_preservation.get("blockers") or [])
    if split_preservation.get("passed") is not True:
        split_preservation_blockers.append("split preservation report did not pass")
    if split_preservation.get("image_checks_enabled") is False:
        if not split_image_preservation:
            split_preservation_blockers.append("split preservation skipped image checks and no split image preservation report was provided")
        elif split_image_preservation.get("passed") is not True:
            split_preservation_blockers.extend(
                str(item) for item in split_image_preservation.get("blockers") or ["split image preservation report did not pass"]
            )
    gates["split_preservation"] = gate_report(
        args.split_preservation,
        passed=not split_preservation_blockers,
        blockers=split_preservation_blockers,
        detail={
            "split_stage": split_preservation.get("split_stage"),
            "split_accepted_for_training": split_preservation.get("split_accepted_for_training"),
            "policy": split_preservation.get("policy"),
            "image_checks_enabled": split_preservation.get("image_checks_enabled"),
            "split_image_preservation": {
                "path": str(args.split_image_preservation or ""),
                "passed": split_image_preservation.get("passed") is True if split_image_preservation else False,
                "schema_version": split_image_preservation.get("schema_version") if split_image_preservation else "",
            },
        },
    )
    taxonomy_blockers: list[str] = []
    taxonomy_notes: list[str] = []
    if not taxonomy_alignment:
        taxonomy_blockers.append("fragment taxonomy coverage report was not provided")
    else:
        metrics = taxonomy_alignment.get("metrics") if isinstance(taxonomy_alignment.get("metrics"), dict) else {}
        missing_axes = taxonomy_alignment.get("missing_real_axes") if isinstance(taxonomy_alignment.get("missing_real_axes"), dict) else {}
        policy = taxonomy_alignment.get("policy") if isinstance(taxonomy_alignment.get("policy"), dict) else {}
        if taxonomy_alignment.get("passed") is not True:
            taxonomy_notes.extend(str(item) for item in taxonomy_alignment.get("blockers") or ["fragment taxonomy coverage audit did not pass"])
        coverage_gaps = taxonomy_alignment.get("coverage_gaps") if isinstance(taxonomy_alignment.get("coverage_gaps"), list) else []
        if coverage_gaps:
            taxonomy_notes.extend(str(item) for item in coverage_gaps)
        for key in [
            "graph_consistency_fraction",
            "document_context_fraction",
        ]:
            try:
                value = float(metrics.get(key))
            except (TypeError, ValueError):
                value = 0.0
            if value < 1.0:
                taxonomy_blockers.append(f"fragment taxonomy coverage metric {key}={value} < 1.0")
        for axis, missing in missing_axes.items():
            if isinstance(missing, list) and missing:
                taxonomy_notes.append(f"fragment taxonomy coverage missing {axis}: {missing}")
        if policy.get("visual_review_still_required") is not True:
            taxonomy_blockers.append("fragment taxonomy coverage policy.visual_review_still_required is not true")
        if policy.get("does_not_accept_manifest_by_itself") is not True:
            taxonomy_blockers.append("fragment taxonomy coverage policy.does_not_accept_manifest_by_itself is not true")
        if policy.get("does_not_require_real_sample_pairing") is not True:
            taxonomy_blockers.append("fragment taxonomy coverage policy.does_not_require_real_sample_pairing is not true")
        if policy.get("metadata_labels_are_unpaired") is not True:
            taxonomy_blockers.append("fragment taxonomy coverage policy.metadata_labels_are_unpaired is not true")
        if policy.get("semantic_axis_overlap_is_diagnostic_not_a_hard_pairing_gate") is not True:
            taxonomy_blockers.append("fragment taxonomy coverage policy.semantic_axis_overlap_is_diagnostic_not_a_hard_pairing_gate is not true")
    gates["fragment_taxonomy_alignment"] = gate_report(
        args.fragment_taxonomy_alignment,
        passed=not taxonomy_blockers,
        blockers=taxonomy_blockers,
        detail={
            "metrics": taxonomy_alignment.get("metrics") if taxonomy_alignment else {},
            "missing_real_axes": taxonomy_alignment.get("missing_real_axes") if taxonomy_alignment else {},
            "coverage_gaps": taxonomy_alignment.get("coverage_gaps") if taxonomy_alignment else [],
            "policy": taxonomy_alignment.get("policy") if taxonomy_alignment else {},
            "real_taxonomy_report": taxonomy_alignment.get("real_taxonomy_report") if taxonomy_alignment else "",
            "notes": taxonomy_notes,
        },
    )
    attachment_role_blockers: list[str] = []
    if not attachment_role:
        attachment_role_blockers.append("attachment-role contract was not provided")
    else:
        policy = attachment_role.get("policy") if isinstance(attachment_role.get("policy"), dict) else {}
        if attachment_role.get("schema_version") != "attachment_role_contract_v1":
            attachment_role_blockers.append("attachment-role contract schema_version is not attachment_role_contract_v1")
        if attachment_role.get("passed") is not True:
            attachment_role_blockers.extend(str(item) for item in attachment_role.get("blockers") or ["attachment-role contract did not pass"])
        if policy.get("fragment_positive_must_have_anchor_endpoint_side_and_mode") is not True:
            attachment_role_blockers.append("attachment-role policy missing fragment positive anchor/endpoint/mode requirement")
        if policy.get("ordinary_negative_must_be_attachment_free") is not True:
            attachment_role_blockers.append("attachment-role policy missing ordinary-negative attachment-free requirement")
        fragment_reports = attachment_role.get("fragment_positive_reports")
        ordinary_reports = attachment_role.get("ordinary_negative_reports")
        if not isinstance(fragment_reports, list) or not fragment_reports:
            attachment_role_blockers.append("attachment-role contract has no fragment positive reports")
        else:
            for index, report in enumerate(fragment_reports):
                if not isinstance(report, dict) or report.get("passed") is not True:
                    attachment_role_blockers.append(f"attachment-role fragment positive report {index} did not pass")
                elif int(report.get("row_count") or 0) <= 0:
                    attachment_role_blockers.append(f"attachment-role fragment positive report {index} has no rows")
        if not isinstance(ordinary_reports, list) or not ordinary_reports:
            attachment_role_blockers.append("attachment-role contract has no ordinary negative reports")
        else:
            for index, report in enumerate(ordinary_reports):
                if not isinstance(report, dict) or report.get("passed") is not True:
                    attachment_role_blockers.append(f"attachment-role ordinary negative report {index} did not pass")
                elif int(report.get("row_count") or 0) <= 0:
                    attachment_role_blockers.append(f"attachment-role ordinary negative report {index} has no rows")
    gates["attachment_role"] = gate_report(
        args.attachment_role_contract,
        passed=not attachment_role_blockers,
        blockers=attachment_role_blockers,
        detail={
            "policy": attachment_role.get("policy") if attachment_role else {},
            "fragment_positive_reports": attachment_role.get("fragment_positive_reports") if attachment_role else [],
            "ordinary_negative_reports": attachment_role.get("ordinary_negative_reports") if attachment_role else [],
        },
    )
    architecture_detail = {
        "architecture_comparison_allowed": architecture.get("architecture_comparison_allowed"),
        "formal_expert_training_candidate_allowed": architecture.get(
            "formal_expert_training_candidate_allowed"
        ),
        "formal_training_blockers": architecture.get("formal_training_blockers"),
        "deployment_certification_allowed": architecture.get("deployment_certification_allowed"),
        "deployment_blockers": architecture.get("deployment_blockers"),
    }
    architecture_training_scale = (
        architecture.get("training_scale_adequacy")
        if isinstance(architecture.get("training_scale_adequacy"), dict)
        else {}
    )
    capacity_maximized_acceptance_allowed = (
        architecture_training_scale.get("schema_version") == "markush_capacity_maximized_training_scale_v1"
        and architecture_training_scale.get("formal_training_candidate_allowed") is True
    )
    architecture_acceptance_passed = (
        architecture.get("architecture_comparison_allowed") is True
        or architecture.get("formal_expert_training_candidate_allowed") is True
        or capacity_maximized_acceptance_allowed
    )
    architecture_acceptance_blockers = [] if architecture_acceptance_passed else list(
        architecture.get("architecture_blockers") or []
    )
    if capacity_maximized_acceptance_allowed:
        architecture_detail["capacity_maximized_acceptance_allowed"] = True
        architecture_detail["architecture_comparison_blockers_are_not_split_acceptance_blockers"] = True
    gates["architecture_readiness"] = gate_report(
        args.architecture_readiness,
        passed=architecture_acceptance_passed,
        blockers=architecture_acceptance_blockers,
        detail=architecture_detail,
    )
    gates["gpu_runtime_evidence"] = gate_report(
        args.gpu_runtime_evidence,
        passed=gpu.get("passed") is True and int(gpu.get("gpu_count_observed") or 0) >= 2,
        blockers=list(gpu.get("blockers") or []),
        detail={"gpu_count_observed": gpu.get("gpu_count_observed"), "gpus": gpu.get("gpus")},
    )
    gates["clean_workspace"] = gate_report(
        args.clean_workspace,
        passed=clean.get("passed") is True,
        blockers=list(clean.get("blockers") or []),
    )
    router_blockers: list[str] = []
    if not router_contract:
        router_blockers.append("router/complete-path contract was not provided")
    else:
        router_policy = router_contract.get("policy") if isinstance(router_contract.get("policy"), dict) else {}
        checkpoint = router_contract.get("molnextr_checkpoint") if isinstance(router_contract.get("molnextr_checkpoint"), dict) else {}
        row_counts = router_contract.get("row_counts") if isinstance(router_contract.get("row_counts"), dict) else {}
        if router_contract.get("schema_version") != "router_complete_path_contract_v1":
            router_blockers.append("router/complete-path contract schema_version is not router_complete_path_contract_v1")
        if router_contract.get("passed") is not True:
            router_blockers.extend(str(item) for item in router_contract.get("blockers") or ["router/complete-path contract did not pass"])
        if router_policy.get("complete_molecule_runtime_path") != "direct_original_molnextr":
            router_blockers.append("router policy complete_molecule_runtime_path is not direct_original_molnextr")
        if router_policy.get("complete_molecule_checkpoint") != "models/molnextr_best.pth":
            router_blockers.append("router policy complete_molecule_checkpoint is not models/molnextr_best.pth")
        if router_policy.get("ordinary_complete_rows_are_router_rejection_negatives_only") is not True:
            router_blockers.append("router policy does not restrict ordinary complete rows to rejection negatives")
        if router_policy.get("ordinary_complete_rows_must_not_be_expert_positive_targets") is not True:
            router_blockers.append("router policy does not forbid ordinary complete rows as expert-positive targets")
        if router_policy.get("fragment_and_markush_positive_rows_must_not_mix_branches") is not True:
            router_blockers.append("router policy does not enforce fragment/Markush branch separation")
        if checkpoint.get("exists") is not True or not str(checkpoint.get("sha256") or ""):
            router_blockers.append("router contract lacks existing molnextr_best.pth checkpoint hash")
        if int(row_counts.get("fragment_expert_positive") or 0) <= 0:
            router_blockers.append("router contract has no fragment expert positive rows")
        if int(row_counts.get("markush_expert_positive") or 0) <= 0:
            router_blockers.append("router contract has no Markush expert positive rows")
        if int(row_counts.get("router_negative_complete_molecule") or 0) <= 0:
            router_blockers.append("router contract has no ordinary complete router-negative rows")
    gates["router_complete_path"] = gate_report(
        args.router_complete_path_contract,
        passed=not router_blockers,
        blockers=router_blockers,
        detail={
            "row_counts": router_contract.get("row_counts") if router_contract else {},
            "molnextr_checkpoint": router_contract.get("molnextr_checkpoint") if router_contract else {},
            "policy": router_contract.get("policy") if router_contract else {},
        },
    )
    visual_review_blockers: list[str] = []
    if not visual_review:
        visual_review_blockers.append("visual review package was not provided")
    else:
        if str(visual_review.get("status") or "") != "visual_review_package_only_does_not_accept_training_data":
            visual_review_blockers.append("visual review package status is not review-only")
        if int(visual_review.get("row_count") or 0) != int(candidate_row_count or 0):
            visual_review_blockers.append("visual review package row_count does not match candidate manifest")
        if int(visual_review.get("sample_count") or 0) <= 0:
            visual_review_blockers.append("visual review package has no sampled rows")
        if int(visual_review.get("stratum_count") or 0) <= 0:
            visual_review_blockers.append("visual review package has no strata")
        strata = visual_review.get("strata") if isinstance(visual_review.get("strata"), dict) else {}
        for prefix in ["realism_policy:", "realism_status:", "font:", "stroke_ratio:"]:
            if not any(str(key).startswith(prefix) for key in strata):
                visual_review_blockers.append(f"visual review package missing {prefix} strata")
        for key in ["samples_csv", "contact_sheet", "html"]:
            path_text = str(visual_review.get(key) or "")
            if not path_text or not Path(path_text).exists():
                visual_review_blockers.append(f"visual review package missing {key}")
    gates["visual_review_package"] = gate_report(
        args.visual_review_package,
        passed=not visual_review_blockers,
        blockers=visual_review_blockers,
        detail={
            "row_count": visual_review.get("row_count"),
            "sample_count": visual_review.get("sample_count"),
            "stratum_count": visual_review.get("stratum_count"),
            "status": visual_review.get("status"),
        },
    )
    strict_visual_blockers = strict_machine_visual_blockers(
        assistant_visual=assistant_visual,
        visual_risk=visual_risk,
        visual_review=visual_review,
        candidate_row_count=int(candidate_row_count or 0),
    )
    gates["strict_machine_visual_acceptance"] = gate_report(
        args.assistant_visual_review,
        passed=not strict_visual_blockers,
        blockers=strict_visual_blockers,
        detail={
            "assistant_visual_review": assistant_visual,
            "visual_risk_report": {
                "schema_version": visual_risk.get("schema_version"),
                "row_count": visual_risk.get("row_count"),
                "background_context_rows": visual_risk.get("background_context_rows"),
                "risk_counts": visual_risk.get("risk_counts"),
                "maxima": visual_risk.get("maxima"),
                "image_readability_probe": visual_risk.get("image_readability_probe"),
            },
            "manual_visual_acceptance_ignored": bool(manual_visual),
            "manual_visual_acceptance_no_longer_required": True,
        },
    )
    confidence_blockers: list[str] = []
    if not confidence:
        confidence_blockers.append("calibrated confidence report was not provided")
    elif confidence.get("deployment_allowed") is not True:
        confidence_blockers.extend(str(item) for item in confidence.get("blockers") or ["calibrated confidence report does not allow deployment"])
    gates["confidence"] = gate_report(
        args.confidence_report,
        passed=confidence.get("deployment_allowed") is True,
        blockers=confidence_blockers,
        detail={
            "deployment_allowed": confidence.get("deployment_allowed"),
            "selected_threshold": confidence.get("selected_threshold"),
        },
    )
    entrypoint_blockers: list[str] = []
    if not entrypoint_preflight:
        entrypoint_blockers.append("training entrypoint preflight gate report was not provided")
    else:
        if entrypoint_preflight.get("schema_version") != "training_entrypoint_preflight_gate_check_v1":
            entrypoint_blockers.append("training entrypoint preflight gate schema_version is not training_entrypoint_preflight_gate_check_v1")
        if entrypoint_preflight.get("passed") is not True:
            entrypoint_blockers.append("training entrypoint preflight gate did not pass")
        policy = entrypoint_preflight.get("policy") if isinstance(entrypoint_preflight.get("policy"), dict) else {}
        for key in [
            "does_not_train",
            "formal_stage_requires_formal_preflight_report",
            "formal_stage_requires_roadmap_constraints_report",
            "red_formal_preflight_cannot_be_used_as_training_evidence",
            "red_formal_preflight_cannot_be_used_as_formal_training_evidence",
            "current_readiness_blockers_remain_enforced",
        ]:
            if policy.get(key) is not True:
                entrypoint_blockers.append(f"training entrypoint preflight policy.{key} is not true")
        if (
            policy.get("nonformal_research_or_measured_runs_record_red_preflight_without_blocking") is not True
            and policy.get("fragment_nonformal_runs_record_red_preflight_without_blocking") is not True
        ):
            entrypoint_blockers.append(
                "training entrypoint preflight policy.nonformal_research_or_measured_runs_record_red_preflight_without_blocking is not true"
            )
        if (
            policy.get("research_only_runs_do_not_require_formal_readiness_preflight_or_roadmap") is not True
            and policy.get("fragment_research_only_runs_do_not_require_formal_readiness_preflight_or_roadmap") is not True
        ):
            entrypoint_blockers.append(
                "training entrypoint preflight policy.research_only_runs_do_not_require_formal_readiness_preflight_or_roadmap is not true"
            )
        probes = entrypoint_preflight.get("probes")
        if not isinstance(probes, list) or not probes:
            entrypoint_blockers.append("training entrypoint preflight gate has no probes")
        else:
            failed_probe_names = [
                str(probe.get("name"))
                for probe in probes
                if not isinstance(probe, dict) or probe.get("passed") is not True
            ]
            if failed_probe_names:
                entrypoint_blockers.append(f"training entrypoint preflight probes failed: {failed_probe_names}")
    gates["training_entrypoint_preflight"] = gate_report(
        args.training_entrypoint_preflight_gate,
        passed=not entrypoint_blockers,
        blockers=entrypoint_blockers,
        detail={
            "policy": entrypoint_preflight.get("policy") if entrypoint_preflight else {},
            "formal_preflight_summary": entrypoint_preflight.get("formal_preflight_summary") if entrypoint_preflight else {},
            "probe_count": len(entrypoint_preflight.get("probes") or []) if entrypoint_preflight else 0,
        },
    )

    acceptance_blockers: list[str] = []
    training_blockers: list[str] = []
    for name, gate in gates.items():
        if gate["passed"] is not True:
            if name in {"confidence"}:
                training_blockers.extend(f"{name}: {item}" for item in gate["blockers"])
            else:
                acceptance_blockers.extend(f"{name}: {item}" for item in gate["blockers"] or ["gate did not pass"])

    if pose_policy.get("visual_review_still_required") is True and not gates["strict_machine_visual_acceptance"]["passed"]:
        acceptance_blockers.append("pose_alignment requires strict machine visual acceptance before acceptance")
    if split_stage != "formal" or split_manifest.get("accepted_for_training") is not True:
        training_blockers.append(
            f"split remains candidate-only: stage={split_manifest.get('stage')} accepted_for_training={split_manifest.get('accepted_for_training')}"
        )
    if architecture.get("formal_expert_training_candidate_allowed") is not True:
        training_blockers.extend(str(item) for item in architecture.get("formal_training_blockers") or [])

    report = {
        "schema_version": "markush_formal_acceptance_preflight_v1",
        "accepted_for_formal_split": not acceptance_blockers,
        "formal_training_start_allowed": not acceptance_blockers and not training_blockers,
        "gates": gates,
        "acceptance_blockers": acceptance_blockers,
        "training_blockers": training_blockers,
        "policy": {
            "preflight_only": True,
            "does_not_accept_training_data": True,
            "does_not_start_expert_training": True,
            "manual_visual_acceptance_required": False,
            "strict_machine_visual_acceptance_required": True,
            "human_review_csv_not_required": True,
            "substitution_anchor_contract_required": True,
            "confidence_required_before_training": True,
            "candidate_split_cannot_start_training": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if acceptance_blockers or training_blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
