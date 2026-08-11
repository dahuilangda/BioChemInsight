from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def count(report: dict[str, Any], axis: str, key: str) -> int:
    coverage = report.get("coverage") if isinstance(report.get("coverage"), dict) else {}
    values = coverage.get(axis) if isinstance(coverage.get(axis), dict) else {}
    return int(values.get(key) or 0)


def min_axis_count(report: dict[str, Any], axis: str, keys: list[str]) -> int:
    return min(count(report, axis, key) for key in keys) if keys else 0


def validation_count(report: dict[str, Any], section: str, key: str) -> int:
    values = report.get(section) if isinstance(report.get(section), dict) else {}
    return int(values.get(key) or 0)


def validation_nested_count(report: dict[str, Any], section: str, key: str, bucket: str) -> int:
    values = report.get(section) if isinstance(report.get(section), dict) else {}
    nested = values.get(key) if isinstance(values.get(key), dict) else {}
    return int(nested.get(bucket) or 0)


def fragment_validation_evidence(report: dict[str, Any]) -> dict[str, Any]:
    row_count = int(report.get("row_count") or 0)
    return {
        "row_count": row_count,
        "trainable": report.get("trainable") is True,
        "invalid_rows": int(report.get("invalid_rows") or 0),
        "all_rows_are_attachment_fragments": validation_count(report, "structure_type_counts", "attachment_fragment")
        == row_count,
        "all_rows_have_atom_coordinates": validation_count(report, "molnextr_pose_counts", "rows_with_atom_coordinates")
        == row_count,
        "all_rows_have_bonds": validation_count(report, "molnextr_pose_counts", "rows_with_bonds") == row_count,
        "all_atom_coordinates_in_range": validation_count(
            report, "molnextr_pose_counts", "rows_without_bad_atom_coordinates"
        )
        == row_count,
        "all_rows_have_endpoint": validation_count(report, "fragment_attachment_counts", "endpoint_in_unit_square")
        == row_count,
        "all_rows_have_anchor": validation_count(report, "fragment_attachment_counts", "anchor_present") == row_count,
        "all_rows_have_single_dummy_atom": validation_count(report, "fragment_attachment_counts", "single_dummy_atom")
        == row_count,
        "all_rows_have_anchor_dummy_bond": validation_count(
            report, "fragment_attachment_counts", "anchor_dummy_bond_present"
        )
        == row_count,
        "all_rows_graph_consistent": validation_nested_count(
            report, "graph_consistency_counts", "row_smiles_canonical_matches_mol", "true"
        )
        == row_count,
        "all_images_readable": validation_count(report, "image_quality_counts", "readable") == row_count,
        "endpoint_mark_counts": report.get("endpoint_mark_counts") if isinstance(report.get("endpoint_mark_counts"), dict) else {},
        "fragment_geometry_counts": report.get("fragment_geometry_counts")
        if isinstance(report.get("fragment_geometry_counts"), dict)
        else {},
        "numeric_summaries": report.get("numeric_summaries") if isinstance(report.get("numeric_summaries"), dict) else {},
    }


def ordinary_negative_validation_evidence(report: dict[str, Any]) -> dict[str, Any]:
    row_count = int(report.get("row_count") or 0)
    return {
        "row_count": row_count,
        "trainable": report.get("trainable") is True,
        "invalid_rows": int(report.get("invalid_rows") or 0),
        "all_rows_are_complete_compounds": validation_count(report, "structure_type_counts", "complete_compound")
        == row_count,
        "all_rows_have_atom_coordinates": validation_count(report, "molnextr_pose_counts", "rows_with_atom_coordinates")
        == row_count,
        "all_rows_have_bonds": validation_count(report, "molnextr_pose_counts", "rows_with_bonds") == row_count,
        "all_atom_coordinates_in_range": validation_count(
            report, "molnextr_pose_counts", "rows_without_bad_atom_coordinates"
        )
        == row_count,
        "all_rows_graph_consistent": validation_nested_count(
            report, "graph_consistency_counts", "row_smiles_canonical_matches_mol", "true"
        )
        == row_count,
        "all_rows_have_no_dummy_attachment_atom": validation_nested_count(
            report, "graph_consistency_counts", "single_dummy_atom", "false"
        )
        == row_count,
        "all_rows_have_no_anchor_dummy_bond": validation_nested_count(
            report, "graph_consistency_counts", "anchor_dummy_bond_present", "false"
        )
        == row_count,
        "all_images_readable": validation_count(report, "image_quality_counts", "readable") == row_count,
        "numeric_summaries": report.get("numeric_summaries") if isinstance(report.get("numeric_summaries"), dict) else {},
    }


def markush_validation_evidence(report: dict[str, Any]) -> dict[str, Any]:
    row_count = int(report.get("row_count") or 0)
    return {
        "row_count": row_count,
        "trainable": report.get("trainable") is True,
        "invalid_rows": int(report.get("invalid_rows") or 0),
        "all_rows_are_markush_layouts": validation_count(report, "structure_type_counts", "markush_layout") == row_count,
        "all_rows_have_atom_coordinates": validation_count(report, "molnextr_pose_counts", "rows_with_atom_coordinates")
        == row_count,
        "all_rows_have_bonds": validation_count(report, "molnextr_pose_counts", "rows_with_bonds") == row_count,
        "all_atom_coordinates_in_range": validation_count(
            report, "molnextr_pose_counts", "rows_without_bad_atom_coordinates"
        )
        == row_count,
        "all_rows_graph_consistent": validation_nested_count(
            report, "graph_consistency_counts", "row_smiles_canonical_matches_mol", "true"
        )
        == row_count,
        "all_rows_have_ocr_cells": validation_count(report, "markush_layout_counts", "rows_with_ocr_cells") == row_count,
        "all_rows_have_valid_ocr_boxes": validation_count(
            report, "markush_layout_counts", "rows_with_valid_ocr_boxes"
        )
        == row_count,
        "all_rows_pass_pose_mapping_rmse": validation_count(
            report, "markush_layout_counts", "rows_with_pose_mapping_rmse_pass"
        )
        == row_count,
        "all_rows_pass_document_realism_machine_audit": validation_count(
            report, "markush_realism_counts", "machine_audit_passed"
        )
        == row_count,
        "all_rows_require_manual_visual_review": validation_count(
            report, "markush_realism_counts", "manual_visual_review_required"
        )
        == row_count,
        "all_rows_use_patent_literature_realism_policy": validation_count(
            report,
            "markush_realism_counts",
            "policy:patent_literature_markushgenerator_cdk_svg_v1",
        )
        == row_count,
        "all_images_readable": validation_count(report, "image_quality_counts", "readable") == row_count,
        "numeric_summaries": report.get("numeric_summaries") if isinstance(report.get("numeric_summaries"), dict) else {},
    }


def failed_evidence_keys(evidence: dict[str, Any]) -> list[str]:
    ignored = {"row_count", "invalid_rows", "endpoint_mark_counts", "fragment_geometry_counts", "numeric_summaries"}
    return [key for key, value in evidence.items() if key not in ignored and value is not True]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether pose-factory shards are acceptable for sidecar training.")
    parser.add_argument("--positive-validation", required=True)
    parser.add_argument("--negative-validation", default="")
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-leak-report", default="")
    parser.add_argument("--real-fragment-taxonomy-report", default="")
    parser.add_argument("--fragment-taxonomy-alignment-report", default="")
    parser.add_argument("--fragment-visual-contract-report", default="")
    parser.add_argument("--attachment-role-contract-report", default="")
    parser.add_argument("--accepted-manifest", default="")
    parser.add_argument("--markush-validation", default="")
    parser.add_argument("--markush-manifest", default="")
    parser.add_argument("--markush-source-leak-report", default="")
    parser.add_argument("--markush-pose-alignment-report", default="")
    parser.add_argument("--markush-substitution-anchor-contract", default="")
    parser.add_argument("--markush-visual-review-passed", action="store_true")
    parser.add_argument("--markush-pose-mapping-review-passed", action="store_true")
    parser.add_argument("--visual-review-passed", action="store_true")
    parser.add_argument("--source-leak-check-passed", action="store_true")
    parser.add_argument("--min-fragment-rows-for-smoke", type=int, default=128)
    parser.add_argument("--min-fragment-rows-for-formal", type=int, default=5000)
    parser.add_argument("--min-negative-rows-for-smoke", type=int, default=512)
    parser.add_argument("--min-negative-rows-for-formal", type=int, default=20000)
    parser.add_argument("--min-per-side-for-smoke", type=int, default=24)
    parser.add_argument("--min-per-mode-for-smoke", type=int, default=24)
    parser.add_argument("--require-markush-for-formal", action="store_true", default=True)
    args = parser.parse_args()

    positive_validation = load_json(args.positive_validation)
    negative_validation = load_json(args.negative_validation) if args.negative_validation else {}
    coverage = load_json(args.coverage)
    source_leak_report = load_json(args.source_leak_report) if args.source_leak_report else {}
    real_taxonomy_report = load_json(args.real_fragment_taxonomy_report) if args.real_fragment_taxonomy_report else {}
    taxonomy_alignment_report = load_json(args.fragment_taxonomy_alignment_report) if args.fragment_taxonomy_alignment_report else {}
    fragment_visual_contract_report = load_json(args.fragment_visual_contract_report) if args.fragment_visual_contract_report else {}
    attachment_role_contract_report = load_json(args.attachment_role_contract_report) if args.attachment_role_contract_report else {}
    accepted_manifest = load_json(args.accepted_manifest) if args.accepted_manifest else {}
    markush_validation = load_json(args.markush_validation) if args.markush_validation else {}
    markush_manifest = load_json(args.markush_manifest) if args.markush_manifest else {}
    markush_source_leak_report = load_json(args.markush_source_leak_report) if args.markush_source_leak_report else {}
    markush_pose_alignment_report = load_json(args.markush_pose_alignment_report) if args.markush_pose_alignment_report else {}
    markush_substitution_anchor_contract = (
        load_json(args.markush_substitution_anchor_contract) if args.markush_substitution_anchor_contract else {}
    )

    manifest_acceptance = accepted_manifest.get("acceptance") if isinstance(accepted_manifest.get("acceptance"), dict) else {}
    manifest_counts = accepted_manifest.get("counts") if isinstance(accepted_manifest.get("counts"), dict) else {}
    manifest_status = str(accepted_manifest.get("status") or "").strip().lower()
    manifest_accepted = (
        accepted_manifest.get("accepted") is True
        or manifest_acceptance.get("accepted") is True
        or manifest_status == "accepted"
    )
    manifest_rejected = (
        accepted_manifest.get("rejected") is True
        or manifest_acceptance.get("rejected") is True
        or manifest_status == "rejected"
    )
    visual_review_passed = bool(args.visual_review_passed) or manifest_acceptance.get("visual_review_passed") is True
    taxonomy_coverage_passed = (
        manifest_acceptance.get("real_fragment_taxonomy_alignment_passed") is True
        or manifest_acceptance.get("fragment_taxonomy_coverage_passed") is True
        or taxonomy_alignment_report.get("passed") is True
    )
    source_leak_passed = (
        bool(args.source_leak_check_passed)
        or source_leak_report.get("passed") is True
        or manifest_acceptance.get("source_leak_check_passed") is True
    )
    markush_acceptance = markush_manifest.get("acceptance") if isinstance(markush_manifest.get("acceptance"), dict) else {}
    markush_manifest_status = str(markush_manifest.get("status") or "").strip().lower()
    markush_manifest_accepted = (
        markush_manifest.get("accepted") is True
        or markush_acceptance.get("accepted") is True
        or markush_manifest_status == "accepted"
    )
    markush_manifest_rejected = (
        markush_manifest.get("rejected") is True
        or markush_acceptance.get("rejected") is True
        or markush_manifest_status == "rejected"
    )
    markush_visual_review_passed = bool(args.markush_visual_review_passed) or markush_acceptance.get("visual_review_passed") is True
    markush_pose_mapping_review_passed = (
        bool(args.markush_pose_mapping_review_passed)
        or markush_acceptance.get("pose_mapping_review_passed") is True
        or markush_pose_alignment_report.get("passed") is True
    )
    markush_source_leak_passed = (
        markush_source_leak_report.get("passed") is True
        or markush_acceptance.get("source_leak_check_passed") is True
    )
    markush_substitution_anchor_passed = (
        markush_substitution_anchor_contract.get("schema_version") == "markush_substitution_anchor_contract_v1"
        and markush_substitution_anchor_contract.get("passed") is True
        and int(markush_substitution_anchor_contract.get("failed_rows") or 0) == 0
    )
    markush_manifest_counts = markush_manifest.get("counts") if isinstance(markush_manifest.get("counts"), dict) else {}
    markush_manifest_rows = int(markush_manifest_counts.get("markush_layout") or markush_manifest.get("row_count") or 0)
    positive_validation_evidence = fragment_validation_evidence(positive_validation)
    negative_validation_evidence = (
        ordinary_negative_validation_evidence(negative_validation) if negative_validation else {}
    )
    markush_validation_evidence_report = markush_validation_evidence(markush_validation) if markush_validation else {}

    fragment_rows = count(coverage, "structure_type", "attachment_fragment")
    ordinary_rows = count(coverage, "structure_type", "complete_compound")
    markush_rows = count(coverage, "structure_type", "markush_layout")
    endpoint_min = min_axis_count(coverage, "endpoint_side", ["left", "right", "top", "bottom"])
    mode_min = min_axis_count(coverage, "attachment_render_mode", ["wavy", "cut", "query_attachment", "dummy_atom"])
    fragment_geometries = coverage.get("coverage", {}).get("attachment_render_geometry", {})
    taxonomy_alignment_metrics = (
        taxonomy_alignment_report.get("metrics") if isinstance(taxonomy_alignment_report.get("metrics"), dict) else {}
    )
    taxonomy_alignment_policy = (
        taxonomy_alignment_report.get("policy") if isinstance(taxonomy_alignment_report.get("policy"), dict) else {}
    )

    blockers_smoke = []
    if not accepted_manifest:
        blockers_smoke.append("no accepted positive fragment manifest was provided")
    elif manifest_rejected:
        blockers_smoke.append("positive fragment manifest is explicitly rejected")
    elif not manifest_accepted:
        blockers_smoke.append("positive fragment manifest is not explicitly accepted")
    manifest_fragment_rows = int(manifest_counts.get("attachment_fragment") or manifest_counts.get("fragment_rows") or 0)
    if accepted_manifest and not manifest_rejected and manifest_fragment_rows <= 0:
        blockers_smoke.append("accepted manifest does not declare attachment fragment rows")
    if positive_validation.get("trainable") is not True:
        blockers_smoke.append("positive fragment validation is not trainable")
    positive_evidence_failures = failed_evidence_keys(positive_validation_evidence)
    if positive_evidence_failures:
        blockers_smoke.append(f"positive fragment validation evidence failed: {positive_evidence_failures}")
    if negative_validation and negative_validation.get("trainable") is not True:
        blockers_smoke.append("negative ordinary validation is not trainable")
    if negative_validation:
        negative_evidence_failures = failed_evidence_keys(negative_validation_evidence)
        if negative_evidence_failures:
            blockers_smoke.append(f"negative ordinary validation evidence failed: {negative_evidence_failures}")
    if fragment_rows < int(args.min_fragment_rows_for_smoke):
        blockers_smoke.append(f"fragment rows {fragment_rows} < smoke minimum {args.min_fragment_rows_for_smoke}")
    if ordinary_rows < int(args.min_negative_rows_for_smoke):
        blockers_smoke.append(f"ordinary negative rows {ordinary_rows} < smoke minimum {args.min_negative_rows_for_smoke}")
    if endpoint_min < int(args.min_per_side_for_smoke):
        blockers_smoke.append(f"minimum endpoint-side count {endpoint_min} < {args.min_per_side_for_smoke}")
    if mode_min < int(args.min_per_mode_for_smoke):
        blockers_smoke.append(f"minimum fragment render-mode count {mode_min} < {args.min_per_mode_for_smoke}")
    if not visual_review_passed:
        blockers_smoke.append("visual review has not been manually accepted")
    if not real_taxonomy_report:
        blockers_smoke.append("real fragment taxonomy report was not provided")
    elif int(real_taxonomy_report.get("summary", {}).get("row_count") or 0) <= 0:
        blockers_smoke.append("real fragment taxonomy report has no rows")
    if args.fragment_taxonomy_alignment_report and taxonomy_alignment_report.get("passed") is not True:
        blockers_smoke.append("fragment taxonomy coverage report did not pass")
    taxonomy_policy = taxonomy_alignment_policy
    if taxonomy_alignment_report and taxonomy_policy.get("does_not_require_real_sample_pairing") is not True:
        blockers_smoke.append("fragment taxonomy coverage report still requires real sample pairing")
    if taxonomy_alignment_report and taxonomy_policy.get("metadata_labels_are_unpaired") is not True:
        blockers_smoke.append("fragment taxonomy coverage report does not declare metadata labels unpaired")
    if not taxonomy_coverage_passed:
        blockers_smoke.append("fragment taxonomy/metadata coverage audit has not passed")
    if not source_leak_passed:
        blockers_smoke.append("source-leak check has not passed")
    if not fragment_visual_contract_report:
        blockers_smoke.append("fragment visual attachment contract report was not provided")
    elif fragment_visual_contract_report.get("passed") is not True:
        kept_rows = int(fragment_visual_contract_report.get("kept_rows") or 0)
        rejected_rows = int(fragment_visual_contract_report.get("rejected_rows") or 0)
        blockers_smoke.append(
            f"fragment visual attachment contract failed: kept_rows={kept_rows} rejected_rows={rejected_rows}"
        )
    if args.attachment_role_contract_report and attachment_role_contract_report.get("passed") is not True:
        blockers_smoke.append("attachment role contract did not pass")

    blockers_formal = list(blockers_smoke)
    if fragment_rows < int(args.min_fragment_rows_for_formal):
        blockers_formal.append(f"fragment rows {fragment_rows} < formal minimum {args.min_fragment_rows_for_formal}")
    if ordinary_rows < int(args.min_negative_rows_for_formal):
        blockers_formal.append(f"ordinary negative rows {ordinary_rows} < formal minimum {args.min_negative_rows_for_formal}")
    if args.require_markush_for_formal:
        if markush_rows <= 0:
            blockers_formal.append("no Markush layout/R-group/page rows in coverage")
        if not markush_validation:
            blockers_formal.append("no Markush validation report was provided")
        elif markush_validation.get("trainable") is not True:
            blockers_formal.append("Markush validation is not trainable")
        if markush_validation:
            markush_evidence_failures = failed_evidence_keys(markush_validation_evidence_report)
            if markush_evidence_failures:
                blockers_formal.append(f"Markush validation evidence failed: {markush_evidence_failures}")
        if not markush_manifest:
            blockers_formal.append("no accepted Markush manifest was provided")
        elif markush_manifest_rejected:
            blockers_formal.append("Markush manifest is explicitly rejected")
        elif not markush_manifest_accepted:
            blockers_formal.append("Markush manifest is not explicitly accepted")
        if markush_manifest and markush_manifest_rows <= 0:
            blockers_formal.append("Markush manifest does not declare markush_layout rows")
        if not markush_visual_review_passed:
            blockers_formal.append("Markush visual review has not been manually accepted")
        if not markush_pose_mapping_review_passed:
            blockers_formal.append("Markush pose mapping review has not been accepted")
        if args.markush_pose_alignment_report and markush_pose_alignment_report.get("passed") is not True:
            blockers_formal.append("Markush pose alignment report did not pass")
        if not markush_source_leak_passed:
            blockers_formal.append("Markush source-leak check has not passed")
        if not markush_substitution_anchor_contract:
            blockers_formal.append("Markush substitution-anchor contract was not provided")
        elif not markush_substitution_anchor_passed:
            blockers_formal.append("Markush substitution-anchor contract did not pass")
        elif int(markush_substitution_anchor_contract.get("row_count") or 0) != int(markush_manifest_rows or 0):
            blockers_formal.append("Markush substitution-anchor contract row_count does not match Markush manifest")

    report = {
        "positive_validation": str(args.positive_validation),
        "negative_validation": str(args.negative_validation),
        "coverage": str(args.coverage),
        "source_leak_report": str(args.source_leak_report),
        "real_fragment_taxonomy_report": str(args.real_fragment_taxonomy_report),
        "fragment_taxonomy_alignment_report": str(args.fragment_taxonomy_alignment_report),
        "fragment_visual_contract_report": str(args.fragment_visual_contract_report),
        "attachment_role_contract_report": str(args.attachment_role_contract_report),
        "accepted_manifest": str(args.accepted_manifest),
        "markush_substitution_anchor_contract": str(args.markush_substitution_anchor_contract),
        "counts": {
            "attachment_fragment": fragment_rows,
            "ordinary_complete_negative": ordinary_rows,
            "markush_layout": markush_rows,
            "endpoint_side_min": endpoint_min,
            "fragment_mode_min": mode_min,
            "fragment_geometries": fragment_geometries,
        },
        "taxonomy_coverage": {
            "passed": taxonomy_alignment_report.get("passed") is True if taxonomy_alignment_report else False,
            "metrics": taxonomy_alignment_metrics,
            "coverage_gaps": taxonomy_alignment_report.get("coverage_gaps")
            if isinstance(taxonomy_alignment_report.get("coverage_gaps"), list)
            else [],
            "missing_real_axes": taxonomy_alignment_report.get("missing_real_axes")
            if isinstance(taxonomy_alignment_report.get("missing_real_axes"), dict)
            else {},
            "visual_shape_aliases": taxonomy_alignment_policy.get("visual_shape_aliases")
            if isinstance(taxonomy_alignment_policy.get("visual_shape_aliases"), dict)
            else {},
            "review_only_visual_shapes_excluded_from_axis_coverage": taxonomy_alignment_policy.get(
                "review_only_visual_shapes_excluded_from_axis_coverage"
            )
            if isinstance(taxonomy_alignment_policy.get("review_only_visual_shapes_excluded_from_axis_coverage"), list)
            else [],
        },
        "validation_evidence": {
            "positive_fragment": positive_validation_evidence,
            "ordinary_negative": negative_validation_evidence,
            "markush_layout": markush_validation_evidence_report,
        },
        "manual_gates": {
            "accepted_manifest_present": bool(accepted_manifest),
            "accepted_manifest_accepted": manifest_accepted,
            "accepted_manifest_rejected": manifest_rejected,
            "visual_review_passed": visual_review_passed,
            "real_fragment_taxonomy_report_present": bool(real_taxonomy_report),
            "fragment_taxonomy_alignment_report_present": bool(taxonomy_alignment_report),
            "fragment_visual_contract_report_present": bool(fragment_visual_contract_report),
            "fragment_visual_contract_passed": fragment_visual_contract_report.get("passed") is True
            if fragment_visual_contract_report
            else False,
            "attachment_role_contract_report_present": bool(attachment_role_contract_report),
            "attachment_role_contract_passed": attachment_role_contract_report.get("passed") is True
            if attachment_role_contract_report
            else False,
            "fragment_taxonomy_coverage_passed": taxonomy_coverage_passed,
            "real_fragment_taxonomy_alignment_passed": taxonomy_coverage_passed,
            "source_leak_check_passed": source_leak_passed,
            "markush_validation_present": bool(markush_validation),
            "markush_validation_trainable": markush_validation.get("trainable") is True if markush_validation else False,
            "markush_manifest_present": bool(markush_manifest),
            "markush_manifest_accepted": markush_manifest_accepted,
            "markush_manifest_rejected": markush_manifest_rejected,
            "markush_visual_review_passed": markush_visual_review_passed,
            "markush_pose_mapping_review_passed": markush_pose_mapping_review_passed,
            "markush_pose_alignment_report_present": bool(markush_pose_alignment_report),
            "markush_source_leak_check_passed": markush_source_leak_passed,
            "markush_substitution_anchor_contract_present": bool(markush_substitution_anchor_contract),
            "markush_substitution_anchor_contract_passed": markush_substitution_anchor_passed,
        },
        "measured_sidecar_smoke_allowed": not blockers_smoke,
        "formal_training_allowed": not blockers_formal,
        "smoke_blockers": blockers_smoke,
        "formal_blockers": blockers_formal,
        "decision": (
            "start_measured_sidecar_smoke"
            if not blockers_smoke
            else "keep_building_and_reviewing_pose_factory_shards"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
