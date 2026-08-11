from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.pose_factory import (
    formal_nonlinear_pose_contract_passed,
    fragment_formal_nonlinear_contract_passed,
    validate_pose_factory_shard,
)


SPLIT_OUTPUT_KEYS = (
    "fragment_train_csv",
    "fragment_calibration_csv",
    "ordinary_train_csv",
    "ordinary_calibration_csv",
    "markush_train_csv",
    "markush_calibration_csv",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def has_point(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        x = float(value.get("x"))
        y = float(value.get("y"))
    except (TypeError, ValueError):
        return False
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def compact_pose_report(report: Any) -> dict[str, Any]:
    value = report.to_dict()
    return {
        "csv_path": value.get("csv_path"),
        "row_count": value.get("row_count"),
        "valid_rows": value.get("valid_rows"),
        "invalid_rows": value.get("invalid_rows"),
        "trainable": value.get("trainable") is True,
        "structure_type_counts": value.get("structure_type_counts", {}),
        "endpoint_side_counts": value.get("endpoint_side_counts", {}),
        "endpoint_mark_counts": value.get("endpoint_mark_counts", {}),
        "render_style_counts": value.get("render_style_counts", {}),
        "issue_severity_counts": value.get("issue_severity_counts", {}),
        "issue_code_counts": value.get("issue_code_counts", {}),
        "missing_required_field_counts": value.get("missing_required_field_counts", {}),
        "quality_gate_counts": value.get("quality_gate_counts", {}),
        "graph_consistency_counts": value.get("graph_consistency_counts", {}),
        "molnextr_pose_counts": value.get("molnextr_pose_counts", {}),
        "fragment_attachment_counts": value.get("fragment_attachment_counts", {}),
        "markush_layout_counts": value.get("markush_layout_counts", {}),
        "image_quality_counts": value.get("image_quality_counts", {}),
        "numeric_summaries": value.get("numeric_summaries", {}),
        "issues": value.get("issues", [])[:20],
    }


def inspect_csv(path: Path, *, role: str, max_examples: int) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = {
        "structure_type": Counter(),
        "backend": Counter(),
        "generator_version": Counter(),
        "render_style": Counter(),
        "source_dataset": Counter(),
        "fragment_anchor": Counter(),
        "fragment_endpoint": Counter(),
        "fragment_graph": Counter(),
        "markush_ocr": Counter(),
        "markush_r_tags": Counter(),
        "markush_annotation": Counter(),
        "markush_provenance": Counter(),
        "ordinary_attachment_metadata": Counter(),
    }
    row_count = 0
    issue_count = 0
    issues: list[dict[str, Any]] = []

    def add_issue(row_index: int, code: str, message: str, source_id: str = "") -> None:
        nonlocal issue_count
        issue_count += 1
        if len(issues) < max_examples:
            issues.append(
                {
                    "row_index": row_index,
                    "source_id": source_id,
                    "code": code,
                    "message": message,
                }
            )

    with path.open(newline="", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=2):
            row_count += 1
            source_id = str(row.get("source_id") or "")
            quality = parse_json_object(row.get("render_quality"))
            structure_type = str(quality.get("structure_type") or row.get("structure_type") or "").strip()
            counters["structure_type"].update([structure_type or "missing"])
            counters["backend"].update([str(quality.get("backend") or "missing")])
            counters["generator_version"].update([str(quality.get("generator_version") or "missing")])
            counters["render_style"].update([str(quality.get("render_style") or "missing")])
            counters["source_dataset"].update([str(quality.get("source_dataset") or "missing")])

            backend_references = quality.get("backend_references")
            if not isinstance(backend_references, list) or not backend_references:
                add_issue(row_index, "missing_backend_references", "render provenance must include backend references", source_id)
            if not quality.get("coord_policy"):
                add_issue(row_index, "missing_coord_policy", "render provenance must include coordinate policy", source_id)
            if "layout_seed" not in quality or quality.get("layout_seed") in {None, ""}:
                add_issue(row_index, "missing_layout_seed", "render provenance must include layout seed", source_id)

            pose_alignment = quality.get("pose_alignment") if isinstance(quality.get("pose_alignment"), dict) else {}
            if not pose_alignment:
                add_issue(row_index, "missing_pose_alignment", "render provenance must preserve image-to-graph orientation alignment", source_id)
            elif pose_alignment.get("image_to_graph_orientation_alignment") is not True:
                add_issue(row_index, "pose_alignment_not_synchronized", "split rows must preserve synchronized image-to-graph orientation", source_id)
            elif (
                pose_alignment.get("coordinate_mutation_after_render") is not False
                and not formal_nonlinear_pose_contract_passed(quality)
                and not fragment_formal_nonlinear_contract_passed(quality)
            ):
                add_issue(
                    row_index,
                    "pose_alignment_mutated_after_render",
                    "split rows may carry post-render nonlinear coordinates only with a passing formal Markush SVG-polyline or fragment endpoint/connector/mark contract",
                    source_id,
                )

            atom_coordinates = quality.get("atom_coordinates")
            bonds = quality.get("bonds")
            if not isinstance(atom_coordinates, list) or not atom_coordinates:
                add_issue(row_index, "missing_atom_coordinates", "generated split rows must preserve atom coordinates", source_id)
            if not isinstance(bonds, list):
                add_issue(row_index, "missing_bonds", "generated split rows must preserve graph bonds", source_id)

            graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
            if graph.get("row_smiles_canonical_matches_mol") is not True:
                add_issue(row_index, "smiles_graph_mismatch", "graph consistency must remain attached to split rows", source_id)

            if role == "fragment":
                endpoint = quality.get("attachment_endpoint")
                anchor = str(quality.get("attachment_anchor") or row.get("attachment_anchor") or "").strip()
                endpoint_mode = str(quality.get("attachment_render_mode") or row.get("attachment_render_mode") or "").strip()
                endpoint_side = str(quality.get("attachment_direction") or row.get("endpoint_side") or "").strip()
                counters["fragment_anchor"].update(["present" if anchor else "missing"])
                counters["fragment_endpoint"].update(["in_unit_square" if has_point(endpoint) else "missing_or_invalid"])
                counters["fragment_graph"].update(
                    ["anchor_dummy_bond_present" if graph.get("anchor_dummy_bond_present") is True else "anchor_dummy_bond_missing"]
                )
                if not anchor:
                    add_issue(row_index, "fragment_missing_anchor_label", "fragment split rows must preserve anchor labels", source_id)
                if not has_point(endpoint):
                    add_issue(row_index, "fragment_missing_endpoint_coordinates", "fragment split rows must preserve endpoint coordinates", source_id)
                if endpoint_mode not in {"wavy", "cut", "query_attachment", "dummy_atom"}:
                    add_issue(row_index, "fragment_invalid_attachment_mode", f"invalid attachment mode={endpoint_mode!r}", source_id)
                if endpoint_side not in {"left", "right", "top", "bottom"}:
                    add_issue(row_index, "fragment_invalid_endpoint_side", f"invalid endpoint side={endpoint_side!r}", source_id)
                if graph.get("anchor_dummy_bond_present") is not True:
                    add_issue(row_index, "fragment_missing_anchor_dummy_bond", "fragment graph must preserve anchor-dummy bond", source_id)

            if role == "markush":
                markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
                cells = markush.get("ocr_cells") if isinstance(markush.get("ocr_cells"), list) else []
                r_tag_count = markush.get("r_tag_count")
                annotation = str(markush.get("annotation") or "").strip()
                render_provenance = quality.get("render_provenance")
                counters["markush_ocr"].update(["present" if cells else "missing"])
                counters["markush_r_tags"].update([str(r_tag_count) if r_tag_count else "missing"])
                counters["markush_annotation"].update(["present" if annotation else "missing"])
                counters["markush_provenance"].update(["present" if isinstance(render_provenance, dict) and render_provenance else "missing"])
                if not cells:
                    add_issue(row_index, "markush_missing_ocr_cells", "Markush split rows must preserve OCR/layout cells", source_id)
                if not r_tag_count:
                    add_issue(row_index, "markush_missing_r_tag_count", "Markush split rows must preserve R-tag labels/counts", source_id)
                if not annotation:
                    add_issue(row_index, "markush_missing_annotation", "Markush split rows must preserve annotation labels", source_id)
                if not isinstance(render_provenance, dict) or not render_provenance:
                    add_issue(row_index, "markush_missing_render_provenance", "Markush split rows must preserve CDK render provenance", source_id)

            if role == "ordinary":
                attachment_fields = [
                    key
                    for key in set(row) | set(quality)
                    if ("attach" in key.lower() or "endpoint" in key.lower() or "anchor" in key.lower())
                    and str(quality.get(key) if key in quality else row.get(key) or "").strip()
                ]
                counters["ordinary_attachment_metadata"].update(["present" if attachment_fields else "absent"])
                if attachment_fields:
                    add_issue(
                        row_index,
                        "ordinary_contains_attachment_metadata",
                        f"ordinary split rows must remain attachment-free: {sorted(attachment_fields)}",
                        source_id,
                    )

    return {
        "csv_path": str(path),
        "role": role,
        "row_count": row_count,
        "issue_count": issue_count,
        "passed": row_count > 0 and issue_count == 0,
        "counters": {name: dict(sorted(counter.items())) for name, counter in sorted(counters.items()) if counter},
        "issues": issues,
    }


def role_for_key(key: str) -> str:
    if key.startswith("fragment_"):
        return "fragment"
    if key.startswith("markush_"):
        return "markush"
    if key.startswith("ordinary_"):
        return "ordinary"
    raise ValueError(f"unknown split output key: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that formal/candidate split CSVs preserve generated pose, graph, endpoint/anchor, Markush labels, and provenance."
    )
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-image-checks", action="store_true")
    parser.add_argument(
        "--full-pose-factory-validation",
        action="store_true",
        help="Rerun full pose-factory validation on split CSVs. Disabled by default for full-source splits because aggregate validation already covers the same rows before split.",
    )
    parser.add_argument("--max-issues", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()

    manifest_path = Path(args.split_manifest)
    manifest = load_json(manifest_path)
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}

    blockers: list[str] = []
    reports: dict[str, Any] = {}
    split_stage = str(manifest.get("stage") or "").strip()
    accepted_for_training = manifest.get("accepted_for_training") is True

    for index, key in enumerate(SPLIT_OUTPUT_KEYS, start=1):
        path_text = str(outputs.get(key) or "")
        if not path_text:
            blockers.append(f"split manifest missing output path for {key}")
            continue
        path = Path(path_text)
        if not path.exists():
            blockers.append(f"split output does not exist for {key}: {path}")
            continue
        role = role_for_key(key)
        if args.progress_every > 0 and (index == 1 or index % int(args.progress_every) == 0):
            print(f"checking {key}: {path}", file=sys.stderr, flush=True)
        preservation_report = inspect_csv(path, role=role, max_examples=int(args.max_issues))
        reports[key] = {"preservation": preservation_report}
        if args.full_pose_factory_validation:
            pose_report = validate_pose_factory_shard(
                path,
                check_images=not args.skip_image_checks,
                max_issues=int(args.max_issues),
            )
            reports[key]["pose_factory_validation"] = compact_pose_report(pose_report)
            if pose_report.trainable is not True:
                blockers.append(f"{key} failed pose-factory validation")
        if preservation_report.get("passed") is not True:
            blockers.append(f"{key} failed split preservation checks")

    report = {
        "schema_version": "split_pose_preservation_contract_v1",
        "split_manifest": str(manifest_path),
        "split_stage": split_stage,
        "split_accepted_for_training": accepted_for_training,
        "image_checks_enabled": not args.skip_image_checks,
        "full_pose_factory_validation_enabled": bool(args.full_pose_factory_validation),
        "passed": not blockers,
        "blockers": blockers,
        "reports": reports,
        "policy": {
            "split_outputs_must_preserve_images": True,
            "split_outputs_must_preserve_graph_and_atom_coordinates": True,
            "fragment_rows_must_preserve_endpoint_coordinates_and_anchor_labels": True,
            "fragment_rows_must_preserve_anchor_dummy_bond": True,
            "markush_rows_must_preserve_ocr_cells_r_tag_labels_annotations_and_render_provenance": True,
            "split_rows_must_preserve_image_to_graph_orientation_alignment": True,
            "post_render_coordinate_mutation_requires_formal_markush_or_fragment_nonlinear_contract": True,
            "ordinary_rows_must_remain_attachment_free": True,
            "complete_molecules_remain_on_original_molnextr_path": True,
            "fragment_and_markush_rows_are_routed_expert_branch_only": True,
            "aggregate_pose_factory_validation_must_pass_before_split_preservation": True,
            "full_source_split_preservation_does_not_repeat_aggregate_schema_validation_by_default": True,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
