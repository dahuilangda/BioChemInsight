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
    FORMAL_NONLINEAR_WARP_POLICY,
    formal_nonlinear_pose_contract_passed,
)


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def valid_point(point: Any) -> bool:
    if not isinstance(point, list) or len(point) != 2:
        return False
    x = as_float(point[0])
    y = as_float(point[1])
    return x is not None and y is not None and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def valid_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    values = [as_float(value) for value in bbox]
    if any(value is None for value in values):
        return False
    x1, y1, x2, y2 = [float(value) for value in values]
    return 0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0


def row_issues(row: dict[str, str]) -> list[str]:
    issues: list[str] = []
    quality = parse_quality(row)
    if not quality:
        return ["render_quality_json_invalid"]
    if not formal_nonlinear_pose_contract_passed(quality):
        issues.append("formal_nonlinear_pose_contract_not_passed")

    warp = quality.get("nonlinear_document_warp") if isinstance(quality.get("nonlinear_document_warp"), dict) else {}
    if warp.get("policy") != FORMAL_NONLINEAR_WARP_POLICY:
        issues.append("formal_nonlinear_policy_mismatch")
    if warp.get("research_only") is True or warp.get("debug_only") is True:
        issues.append("formal_nonlinear_mislabeled_research_or_debug")
    if warp.get("formal_training_allowed") is not True:
        issues.append("formal_nonlinear_not_allowed_by_contract")
    if int(warp.get("atom_outside_count") or 0) != 0:
        issues.append("atom_outside_count_nonzero")
    if int(warp.get("invalid_bbox_count") or 0) != 0:
        issues.append("invalid_bbox_count_nonzero")
    if warp.get("blank") is True or warp.get("dense") is True:
        issues.append("warped_image_blank_or_dense")

    pose = quality.get("nonlinear_pose_preservation") if isinstance(quality.get("nonlinear_pose_preservation"), dict) else {}
    if pose.get("passed") is not True:
        issues.append("nonlinear_pose_preservation_failed")
    checks = [
        ("line_constraint_rmse_svg_units", "line_constraint_rmse_threshold", "nonlinear_rmse_exceeds_threshold"),
        ("line_constraint_abs_p95_svg_units", "line_constraint_abs_p95_threshold", "nonlinear_line_p95_exceeds_threshold"),
        ("line_constraint_abs_max_svg_units", "line_constraint_abs_max_threshold", "nonlinear_line_max_exceeds_threshold"),
        ("intersection_anchor_rmse_svg_units", "intersection_anchor_rmse_threshold", "nonlinear_anchor_rmse_exceeds_threshold"),
        ("intersection_anchor_abs_max_svg_units", "intersection_anchor_abs_max_threshold", "nonlinear_anchor_max_exceeds_threshold"),
    ]
    for value_key, threshold_key, issue in checks:
        value = as_float(pose.get(value_key))
        threshold = as_float(pose.get(threshold_key))
        if value is None or threshold is None or value > threshold:
            issues.append(issue)
    if int(pose.get("intersection_anchor_count") or 0) < 2:
        issues.append("nonlinear_too_few_intersection_anchors")

    atoms = quality.get("atom_coordinates")
    if not isinstance(atoms, list) or not atoms:
        issues.append("missing_atom_coordinates")
    else:
        for atom in atoms:
            if not isinstance(atom, dict) or not valid_point([atom.get("x"), atom.get("y")]):
                issues.append("invalid_warped_atom_coordinate")
                break

    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    cells = markush.get("ocr_cells")
    if not isinstance(cells, list) or not cells:
        issues.append("missing_ocr_cells")
    else:
        for cell in cells:
            if not isinstance(cell, dict) or not valid_bbox(cell.get("bbox")):
                issues.append("invalid_warped_ocr_bbox")
                break
            if cell.get("bbox_transform_policy") != "nonlinear_warp_edge_sampling_envelope":
                issues.append("ocr_bbox_transform_policy_missing")
                break

    geometry = quality.get("svg_bond_geometry") if isinstance(quality.get("svg_bond_geometry"), dict) else {}
    if geometry.get("nonlinear_warp_synchronized") is not True:
        issues.append("svg_bond_geometry_not_nonlinear_synchronized")
    axis_count = 0
    warped_axis_count = 0
    for bond in geometry.get("bonds") if isinstance(geometry.get("bonds"), list) else []:
        axis = bond.get("selected_atom_center_axis") if isinstance(bond.get("selected_atom_center_axis"), dict) else {}
        source_points = axis.get("sampled_points_normalized")
        if not isinstance(source_points, list) or len(source_points) < 2 or not all(
            valid_point(point) for point in source_points
        ):
            issues.append("not_all_svg_bond_selected_axes_source_sampled")
        points = axis.get("sampled_points_normalized_after_warp")
        axis_count += 1
        if isinstance(points, list) and len(points) >= 2 and all(valid_point(point) for point in points):
            warped_axis_count += 1
        for line in bond.get("visible_lines") if isinstance(bond.get("visible_lines"), list) else []:
            line_points = line.get("sampled_points_normalized")
            if not isinstance(line_points, list) or len(line_points) < 2 or not all(
                valid_point(point) for point in line_points
            ):
                issues.append("not_all_svg_visible_lines_source_sampled")
            warped_line_points = line.get("sampled_points_normalized_after_warp")
            if not isinstance(warped_line_points, list) or len(warped_line_points) < 2 or not all(
                valid_point(point) for point in warped_line_points
            ):
                issues.append("not_all_svg_visible_lines_warped")
    if axis_count <= 0:
        issues.append("missing_svg_bond_selected_axes")
    if axis_count != warped_axis_count:
        issues.append("not_all_svg_bond_selected_axes_warped")
    return sorted(set(issues))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit formal-capable nonlinear Markush document warp contracts.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=40)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    issue_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=2):
        issues = row_issues(row)
        if issues:
            issue_counts.update(issues)
            if len(examples) < int(args.max_examples):
                examples.append({"row_index": index, "source_id": row.get("source_id"), "issues": issues})

    report = {
        "schema_version": "markush_formal_nonlinear_document_warp_contract_audit_v1",
        "csv": str(csv_path),
        "row_count": len(rows),
        "passed": not issue_counts,
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_examples": examples,
        "policy": {
            "formal_policy": FORMAL_NONLINEAR_WARP_POLICY,
            "research_only": False,
            "requires_synchronized_image_atom_ocr_svg_bond_geometry": True,
            "requires_warped_svg_polyline_pose_preservation": True,
            "does_not_replace_substitution_visual_source_leak_router_model_scale_or_readiness_gates": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if issue_counts:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
