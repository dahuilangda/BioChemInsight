from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.pose_factory import (
    MARKUSH_ATOM_INDEX_ALIGNMENT_SCHEMA_VERSION,
    formal_nonlinear_pose_contract_passed,
    validate_markush_atom_index_alignment_contract,
)


REAL_ATOM_TOKENS = {
    "B",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "Si",
    "Se",
    "Te",
    "As",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


SVG_CENTER_FIT_METHOD = "svg_bond_axis_atom_center_hybrid_intersections_with_single_axis_affine_seed_projection"


def is_svg_center_fit_method(method: str) -> bool:
    return SVG_CENTER_FIT_METHOD in str(method or "")


def is_supported_pose_fit_method(method: str) -> bool:
    method = str(method or "")
    return (
        is_svg_center_fit_method(method)
        or ("line" in method and "constraint" in method)
        or ("bond_axis" in method and "perpendicular_constraints" in method)
    )


def pose_affine_diagnostics(pose_mapping: dict[str, Any], fit_diagnostics: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return the affine sanity source for the pose policy.

    The SVG-center policy uses the final fit for line/intersection residuals and
    an affine seed only to place single-axis atoms along the rendered bond axis.
    Old line-constrained affine rows store the affine in final fit diagnostics.
    """
    if is_svg_center_fit_method(str(pose_mapping.get("fit_method") or "")):
        seed = pose_mapping.get("affine_seed_diagnostics")
        if isinstance(seed, dict) and isinstance(seed.get("affine"), dict):
            return seed["affine"], "affine_seed_diagnostics.affine"
        nested = fit_diagnostics.get("affine_seed_diagnostics")
        if isinstance(nested, dict) and isinstance(nested.get("affine"), dict):
            return nested["affine"], "fit_diagnostics.affine_seed_diagnostics.affine"
        return {}, "missing_affine_seed_diagnostics"
    affine = fit_diagnostics.get("affine") if isinstance(fit_diagnostics.get("affine"), dict) else {}
    return affine, "fit_diagnostics.affine" if affine else "missing_affine_diagnostics"


def validate_scheduled_tail_shard(
    *,
    csv_path: Path,
    row_count: int,
    min_rows: int,
    manifest_path: Path | None,
    schedule_path: Path | None,
    shard_index: int | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": False,
        "applies": False,
        "blockers": [],
        "reason": "schedule, manifest, and shard index were not all provided",
    }
    if manifest_path is None or schedule_path is None or shard_index is None:
        return result

    result["enabled"] = True
    manifest = read_json(manifest_path)
    schedule = read_json(schedule_path)
    shards = schedule.get("shards")
    if not isinstance(shards, list) or not shards:
        result["blockers"].append("schedule.shards is missing or empty")
        return result
    if shard_index < 0 or shard_index >= len(shards):
        result["blockers"].append(f"shard index {shard_index} outside schedule shard range 0..{len(shards) - 1}")
        return result

    scheduled = shards[shard_index]
    if not isinstance(scheduled, dict):
        result["blockers"].append(f"schedule shard {shard_index} is not an object")
        return result

    is_last_shard = shard_index == len(shards) - 1
    scheduled_candidate_rows = int(scheduled.get("candidate_rows") or -1)
    scheduled_output_dir = Path(str(scheduled.get("output_dir") or ""))
    expected_csv = scheduled_output_dir / "markush_layout_positive.csv"
    manifest_plan = manifest.get("candidate_plan") if isinstance(manifest.get("candidate_plan"), dict) else {}
    manifest_rows = int(manifest.get("row_count") or -1)
    failure_count = int(manifest.get("failure_count") or 0)
    selected_candidate_rows = int(manifest_plan.get("selected_candidate_rows") or -1)
    plan_start = int(manifest_plan.get("candidate_plan_slice_start") or -1)
    plan_end = int(manifest_plan.get("candidate_plan_slice_end") or -1)
    scheduled_plan_start = int(scheduled.get("plan_start") or -1)
    scheduled_plan_end = int(scheduled.get("plan_end") or -1)

    result.update(
        {
            "reason": "terminal schedule shard is smaller than the normal per-shard statistical minimum",
            "shard_index": shard_index,
            "schedule_shard_count": len(shards),
            "is_last_shard": is_last_shard,
            "scheduled_candidate_rows": scheduled_candidate_rows,
            "scheduled_plan_start": scheduled_plan_start,
            "scheduled_plan_end": scheduled_plan_end,
            "scheduled_output_dir": str(scheduled_output_dir),
            "expected_csv": str(expected_csv),
            "manifest_path": str(manifest_path),
            "manifest_row_count": manifest_rows,
            "manifest_failure_count": failure_count,
            "manifest_selected_candidate_rows": selected_candidate_rows,
            "manifest_plan_start": plan_start,
            "manifest_plan_end": plan_end,
            "csv_row_count": row_count,
            "min_rows": min_rows,
        }
    )

    if not is_last_shard:
        result["blockers"].append("only the final scheduled shard may use the tail-shard min_rows exemption")
    if scheduled_candidate_rows <= 0:
        result["blockers"].append("scheduled candidate_rows is missing or invalid")
    if scheduled_candidate_rows >= min_rows:
        result["blockers"].append(
            f"scheduled candidate_rows {scheduled_candidate_rows} is not below min_rows {min_rows}"
        )
    if csv_path != expected_csv:
        result["blockers"].append(f"csv path {csv_path} does not match scheduled output {expected_csv}")
    if manifest_rows != row_count:
        result["blockers"].append(f"manifest row_count {manifest_rows} != csv row_count {row_count}")
    if selected_candidate_rows != scheduled_candidate_rows:
        result["blockers"].append(
            f"manifest selected candidates {selected_candidate_rows} != scheduled candidate_rows {scheduled_candidate_rows}"
        )
    if manifest_rows + failure_count != selected_candidate_rows:
        result["blockers"].append(
            f"manifest row_count + failure_count {manifest_rows + failure_count} != selected candidates {selected_candidate_rows}"
        )
    if plan_start != scheduled_plan_start or plan_end != scheduled_plan_end:
        result["blockers"].append(
            f"manifest plan range {plan_start}..{plan_end} != scheduled range {scheduled_plan_start}..{scheduled_plan_end}"
        )

    result["applies"] = not result["blockers"]
    return result


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
    return number


def bbox_valid(cell: Any) -> bool:
    if not isinstance(cell, dict):
        return False
    bbox = cell.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    values = [as_float(item) for item in bbox]
    if any(value is None for value in values):
        return False
    x1, y1, x2, y2 = values
    return bool(0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0)


def bbox_area(cell: Any) -> float:
    if not bbox_valid(cell):
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in cell["bbox"]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def real_atom_count(atom_coordinates: Any) -> int:
    if not isinstance(atom_coordinates, list):
        return 0
    return sum(1 for item in atom_coordinates if isinstance(item, dict) and str(item.get("token") or "") in REAL_ATOM_TOKENS)


def atom_coordinate_audit(atom_coordinates: Any, graph: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "issues": [],
        "min_margin": None,
        "min_pair_distance": None,
        "count": 0,
    }
    if not isinstance(atom_coordinates, list) or not atom_coordinates:
        result["issues"].append("missing_atom_coordinates")
        return result
    graph_atom_count = int(graph.get("atom_count") or -1)
    if graph_atom_count >= 0 and graph_atom_count != len(atom_coordinates):
        result["issues"].append("atom_count_coordinate_mismatch")
    coords: list[tuple[float, float]] = []
    seen_indices: set[int] = set()
    for item in atom_coordinates:
        if not isinstance(item, dict):
            result["issues"].append("invalid_atom_coordinate_record")
            continue
        try:
            atom_index = int(item.get("atom_index"))
        except (TypeError, ValueError):
            result["issues"].append("invalid_atom_coordinate_index")
            continue
        if atom_index in seen_indices:
            result["issues"].append("duplicate_atom_coordinate_index")
        seen_indices.add(atom_index)
        if graph_atom_count >= 0 and not (0 <= atom_index < graph_atom_count):
            result["issues"].append("atom_coordinate_index_out_of_range")
        x = as_float(item.get("x"))
        y = as_float(item.get("y"))
        if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
            result["issues"].append("invalid_atom_coordinate_value")
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            result["issues"].append("atom_coordinate_out_of_range")
        coords.append((float(x), float(y)))
    if graph_atom_count >= 0 and seen_indices != set(range(graph_atom_count)):
        result["issues"].append("atom_coordinate_indices_not_contiguous")
    if coords:
        result["count"] = len(coords)
        result["min_margin"] = min(min(x, y, 1.0 - x, 1.0 - y) for x, y in coords)
        min_pair_distance = None
        for left in range(len(coords)):
            for right in range(left + 1, len(coords)):
                distance = math.dist(coords[left], coords[right])
                if min_pair_distance is None or distance < min_pair_distance:
                    min_pair_distance = distance
        result["min_pair_distance"] = min_pair_distance
    return result


def bond_index_issues(bonds: Any, atom_count: int) -> list[str]:
    if not isinstance(bonds, list):
        return ["missing_bonds"]
    issues: list[str] = []
    for bond in bonds:
        if isinstance(bond, dict):
            begin = bond.get("begin_atom_index")
            end = bond.get("end_atom_index")
        elif isinstance(bond, list) and len(bond) >= 2:
            begin, end = bond[0], bond[1]
        else:
            issues.append("invalid_bond_record")
            continue
        try:
            begin_index = int(begin)
            end_index = int(end)
        except (TypeError, ValueError):
            issues.append("invalid_bond_atom_index")
            continue
        if begin_index == end_index or not (0 <= begin_index < atom_count) or not (0 <= end_index < atom_count):
            issues.append("bond_atom_index_out_of_range")
    return issues


def bucket_count(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 5:
        return "1-5"
    if value <= 12:
        return "6-12"
    if value <= 25:
        return "13-25"
    return "26+"


def markush_r_tag_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    if value <= 8:
        return "5-8"
    return "9+"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Markush pose mapping and OCR/layout evidence before acceptance.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rmse", type=float, default=0.05)
    parser.add_argument("--max-line-abs-p95", type=float, default=0.05)
    parser.add_argument("--max-line-abs-max", type=float, default=0.10)
    parser.add_argument("--max-intersection-anchor-rmse", type=float, default=0.05)
    parser.add_argument("--max-intersection-anchor-abs-max", type=float, default=0.10)
    parser.add_argument("--min-fit-points", type=int, default=6)
    parser.add_argument("--max-affine-scale-ratio", type=float, default=1.25)
    parser.add_argument("--min-atom-coordinate-margin", type=float, default=0.0)
    parser.add_argument("--min-atom-pair-distance", type=float, default=0.005)
    parser.add_argument("--min-intersection-anchors", type=int, default=2)
    parser.add_argument("--min-rows", type=int, default=128)
    parser.add_argument("--min-variable-cell-fraction", type=float, default=0.60)
    parser.add_argument("--min-r-tag-rows", type=int, default=64)
    parser.add_argument("--manifest", default="", help="Shard manifest used to verify scheduled terminal tail shards.")
    parser.add_argument("--schedule", default="", help="Generation schedule used to verify scheduled terminal tail shards.")
    parser.add_argument("--shard-index", type=int, default=-1, help="Shard index in --schedule for terminal tail-shard validation.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows = read_rows(csv_path)
    blockers: list[str] = []
    rmse_values: list[float] = []
    line_abs_p95_values: list[float] = []
    line_abs_max_values: list[float] = []
    intersection_anchor_rmse_values: list[float] = []
    intersection_anchor_abs_max_values: list[float] = []
    legacy_rmse_values: list[float] = []
    affine_scale_ratios: list[float] = []
    affine_determinants: list[float] = []
    intersection_anchor_counts: list[int] = []
    fit_points: list[int] = []
    atom_coordinate_margins: list[float] = []
    atom_pair_distances: list[float] = []
    counters: dict[str, Counter[str]] = {
        "atom_index_alignment_policy": Counter(),
        "atom_index_alignment_schema": Counter(),
        "cell_count_bucket": Counter(),
        "r_tag_count_bucket": Counter(),
        "annotation_r_tag_bucket": Counter(),
        "dummy_count_bucket": Counter(),
        "backend": Counter(),
        "render_style": Counter(),
        "graph_atom_count_bucket": Counter(),
        "graph_bond_count_bucket": Counter(),
        "variable_cell_text": Counter(),
    }
    row_failures: list[dict[str, Any]] = []
    rows_with_cells = 0
    rows_with_variable_cells = 0
    rows_with_r_tags = 0
    graph_consistent_rows = 0
    rows_with_real_backbone = 0
    rows_with_acceptable_text_area = 0

    for index, row in enumerate(rows, start=2):
        quality = parse_quality(row)
        markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
        pose_alignment = quality.get("pose_alignment") if isinstance(quality.get("pose_alignment"), dict) else {}
        pose_mapping = quality.get("pose_mapping") if isinstance(quality.get("pose_mapping"), dict) else {}
        graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
        quality_gates = quality.get("quality_gates") if isinstance(quality.get("quality_gates"), dict) else {}
        atom_coordinates = quality.get("atom_coordinates") if isinstance(quality.get("atom_coordinates"), list) else []
        atom_index_alignment = (
            quality.get("atom_index_alignment") if isinstance(quality.get("atom_index_alignment"), dict) else {}
        )
        cells = markush.get("ocr_cells") if isinstance(markush.get("ocr_cells"), list) else []
        bad_cells = [cell for cell in cells if not bbox_valid(cell)]
        cell_area_sum = sum(bbox_area(cell) for cell in cells)
        cell_area_max = max([bbox_area(cell) for cell in cells] or [0.0])
        real_atoms = real_atom_count(atom_coordinates)
        atom_audit = atom_coordinate_audit(atom_coordinates, graph)
        rmse = as_float(pose_mapping.get("line_constraint_rmse_svg_units"))
        line_abs_p95 = as_float(pose_mapping.get("line_constraint_abs_p95_svg_units"))
        line_abs_max = as_float(pose_mapping.get("line_constraint_abs_max_svg_units"))
        intersection_anchor_rmse = as_float(pose_mapping.get("intersection_anchor_rmse_svg_units"))
        intersection_anchor_abs_max = as_float(pose_mapping.get("intersection_anchor_abs_max_svg_units"))
        threshold = as_float(pose_mapping.get("line_constraint_rmse_threshold")) or float(args.max_rmse)
        fit_count = int(pose_mapping.get("line_constraint_fit_equation_count") or 0)
        legacy_rmse = as_float(pose_mapping.get("legacy_midpoint_rmse_svg_units", pose_mapping.get("affine_rmse_svg_units")))
        pose_metric = str(pose_mapping.get("fit_method") or "")
        fit_diagnostics = pose_mapping.get("fit_diagnostics") if isinstance(pose_mapping.get("fit_diagnostics"), dict) else {}
        affine_diagnostics, affine_diagnostics_source = pose_affine_diagnostics(pose_mapping, fit_diagnostics)
        affine_scale_ratio = as_float(affine_diagnostics.get("scale_ratio"))
        affine_determinant = as_float(affine_diagnostics.get("determinant"))
        intersection_anchor_count = int(fit_diagnostics.get("intersection_anchor_count") or 0)
        variable_texts = [
            str(cell.get("text") or "")
            for cell in cells
            if isinstance(cell, dict) and str(cell.get("text") or "") not in {"", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
        ]
        r_tag_count = int(markush.get("r_tag_count") or 0)
        dummy_count = int(markush.get("dummy_count") or 0)

        counters["backend"].update([str(quality.get("backend") or "missing")])
        counters["atom_index_alignment_schema"].update([str(atom_index_alignment.get("schema_version") or "missing")])
        counters["atom_index_alignment_policy"].update([str(atom_index_alignment.get("policy") or "missing")])
        counters["render_style"].update([str(quality.get("render_style") or "missing")])
        counters["cell_count_bucket"].update([bucket_count(len(cells))])
        counters["r_tag_count_bucket"].update([bucket_count(r_tag_count)])
        counters["annotation_r_tag_bucket"].update([markush_r_tag_bucket(r_tag_count)])
        counters["dummy_count_bucket"].update([bucket_count(dummy_count)])
        counters["graph_atom_count_bucket"].update([bucket_count(int(graph.get("atom_count") or 0))])
        counters["graph_bond_count_bucket"].update([bucket_count(int(graph.get("bond_count") or 0))])
        counters["variable_cell_text"].update(variable_texts or ["none"])

        if cells:
            rows_with_cells += 1
        if variable_texts:
            rows_with_variable_cells += 1
        if r_tag_count > 0:
            rows_with_r_tags += 1
        if graph.get("row_smiles_canonical_matches_mol") is True:
            graph_consistent_rows += 1
        if real_atoms >= 3:
            rows_with_real_backbone += 1
        if cell_area_sum <= 0.12 and cell_area_max <= 0.08:
            rows_with_acceptable_text_area += 1
        if rmse is not None:
            rmse_values.append(rmse)
        if line_abs_p95 is not None:
            line_abs_p95_values.append(line_abs_p95)
        if line_abs_max is not None:
            line_abs_max_values.append(line_abs_max)
        if intersection_anchor_rmse is not None:
            intersection_anchor_rmse_values.append(intersection_anchor_rmse)
        if intersection_anchor_abs_max is not None:
            intersection_anchor_abs_max_values.append(intersection_anchor_abs_max)
        if legacy_rmse is not None:
            legacy_rmse_values.append(legacy_rmse)
        if affine_scale_ratio is not None:
            affine_scale_ratios.append(affine_scale_ratio)
        if affine_determinant is not None:
            affine_determinants.append(affine_determinant)
        intersection_anchor_counts.append(intersection_anchor_count)
        fit_points.append(fit_count)
        if atom_audit.get("min_margin") is not None:
            atom_coordinate_margins.append(float(atom_audit["min_margin"]))
        if atom_audit.get("min_pair_distance") is not None:
            atom_pair_distances.append(float(atom_audit["min_pair_distance"]))

        reasons: list[str] = []
        if str(quality.get("structure_type") or "") != "markush_layout":
            reasons.append("not_markush_layout")
        atom_index_alignment_issues = validate_markush_atom_index_alignment_contract(quality)
        reasons.extend(atom_index_alignment_issues)
        expected_coord_policy = "cdk_svg_bond_axis_atom_center_hybrid_mapping"
        if str(quality.get("coord_policy") or "") != expected_coord_policy:
            reasons.append("coord_policy_failed")
        if pose_alignment.get("image_to_graph_orientation_alignment") is not True:
            reasons.append("pose_orientation_alignment_failed")
        formal_nonlinear_pose = formal_nonlinear_pose_contract_passed(quality)
        if pose_alignment.get("coordinate_mutation_after_render") is not False and not formal_nonlinear_pose:
            reasons.append("pose_coordinate_mutation_after_render")
        if pose_alignment.get("coordinate_mutation_after_render") is True and formal_nonlinear_pose:
            nonlinear_pose = quality.get("nonlinear_pose_preservation") if isinstance(quality.get("nonlinear_pose_preservation"), dict) else {}
            if nonlinear_pose.get("passed") is not True:
                reasons.append("formal_nonlinear_pose_preservation_failed")
        if pose_alignment.get("synchronized_after_augmentation") is not True:
            reasons.append("pose_not_synchronized_after_augmentation")
        if str(pose_alignment.get("orientation_policy") or "") != expected_coord_policy:
            reasons.append("pose_orientation_policy_failed")
        if str(pose_alignment.get("pose_mapping_metric") or "") != "line_constraint_rmse_svg_units":
            reasons.append("pose_mapping_metric_failed")
        for gate_name in [
            "atom_coordinates_present",
            "external_backend_referenced",
            "image_readable",
            "image_to_graph_orientation_alignment",
            "line_constraint_pose_mapping_rmse_passed",
            "markush_ocr_cells_present",
            "markush_visual_quality_passed",
            "pose_mapping_rmse_passed",
        ]:
            if quality_gates.get(gate_name) is not True:
                reasons.append(f"quality_gate_{gate_name}_failed")
        if not cells:
            reasons.append("missing_ocr_cells")
        if bad_cells:
            reasons.append(f"{len(bad_cells)}_bad_ocr_bboxes")
        if rmse is None or rmse > min(threshold, float(args.max_rmse)):
            reasons.append("rmse_failed")
        if float(args.max_line_abs_p95) > 0.0 and (
            line_abs_p95 is None or line_abs_p95 > float(args.max_line_abs_p95)
        ):
            reasons.append("line_abs_p95_failed")
        if float(args.max_line_abs_max) > 0.0 and (
            line_abs_max is None or line_abs_max > float(args.max_line_abs_max)
        ):
            reasons.append("line_abs_max_failed")
        if float(args.max_intersection_anchor_rmse) > 0.0 and (
            intersection_anchor_rmse is None
            or intersection_anchor_rmse > float(args.max_intersection_anchor_rmse)
        ):
            reasons.append("intersection_anchor_rmse_failed")
        if float(args.max_intersection_anchor_abs_max) > 0.0 and (
            intersection_anchor_abs_max is None
            or intersection_anchor_abs_max > float(args.max_intersection_anchor_abs_max)
        ):
            reasons.append("intersection_anchor_abs_max_failed")
        if not is_supported_pose_fit_method(pose_metric):
            reasons.append("missing_line_constraint_fit_method")
        if not fit_diagnostics:
            reasons.append("missing_fit_diagnostics")
        if not affine_diagnostics:
            reasons.append(affine_diagnostics_source)
        else:
            if affine_determinant is None or affine_determinant >= 0.0:
                reasons.append("invalid_affine_orientation")
            if affine_scale_ratio is None or affine_scale_ratio > float(args.max_affine_scale_ratio):
                reasons.append("affine_scale_ratio_failed")
            if intersection_anchor_count < int(args.min_intersection_anchors):
                reasons.append("too_few_intersection_anchors")
        if fit_count < int(args.min_fit_points):
            reasons.append("too_few_fit_points")
        if graph.get("row_smiles_canonical_matches_mol") is not True:
            reasons.append("graph_smiles_mismatch")
        if atom_audit.get("issues"):
            reasons.extend(str(item) for item in atom_audit.get("issues", []))
        atom_count = int(graph.get("atom_count") or atom_audit.get("count") or 0)
        reasons.extend(bond_index_issues(quality.get("bonds"), atom_count))
        min_margin = atom_audit.get("min_margin")
        if (
            float(args.min_atom_coordinate_margin) > 0.0
            and (min_margin is None or float(min_margin) < float(args.min_atom_coordinate_margin))
        ):
            reasons.append("atom_coordinate_margin_failed")
        min_pair_distance = atom_audit.get("min_pair_distance")
        if (
            float(args.min_atom_pair_distance) > 0.0
            and min_pair_distance is not None
            and float(min_pair_distance) < float(args.min_atom_pair_distance)
        ):
            reasons.append("atom_pair_distance_failed")
        if real_atoms < 3:
            reasons.append("label_only_or_no_real_backbone")
        if cell_area_sum > 0.12:
            reasons.append("text_dominated_image")
        if cell_area_max > 0.08:
            reasons.append("oversized_ocr_cell")
        if reasons:
            row_failures.append(
                {
                    "row_index": index,
                    "source_id": str(row.get("source_id") or ""),
                    "reasons": reasons,
                    "rmse": rmse,
                    "line_abs_p95": line_abs_p95,
                    "line_abs_max": line_abs_max,
                    "intersection_anchor_rmse": intersection_anchor_rmse,
                    "intersection_anchor_abs_max": intersection_anchor_abs_max,
                    "legacy_midpoint_rmse": legacy_rmse,
                    "fit_point_count": fit_count,
                    "affine_scale_ratio": affine_scale_ratio,
                    "affine_determinant": affine_determinant,
                    "intersection_anchor_count": intersection_anchor_count,
                    "atom_coordinate_min_margin": min_margin,
                    "atom_pair_min_distance": min_pair_distance,
                    "cell_count": len(cells),
                    "real_atom_count": real_atoms,
                    "ocr_cell_area_sum": cell_area_sum,
                    "ocr_cell_area_max": cell_area_max,
                }
            )

    row_count = len(rows)
    variable_fraction = rows_with_variable_cells / row_count if row_count else 0.0
    graph_fraction = graph_consistent_rows / row_count if row_count else 0.0
    tail_shard_policy = validate_scheduled_tail_shard(
        csv_path=csv_path,
        row_count=row_count,
        min_rows=int(args.min_rows),
        manifest_path=Path(args.manifest) if args.manifest else None,
        schedule_path=Path(args.schedule) if args.schedule else None,
        shard_index=int(args.shard_index) if int(args.shard_index) >= 0 else None,
    )
    if row_count < int(args.min_rows) and not tail_shard_policy.get("applies"):
        blockers.append(f"Markush rows {row_count} < minimum {args.min_rows}")
        blockers.extend(str(item) for item in tail_shard_policy.get("blockers", []))
    if row_failures:
        blockers.append(f"{len(row_failures)} Markush rows failed pose/OCR checks")
    if variable_fraction < float(args.min_variable_cell_fraction):
        blockers.append(f"variable-cell row fraction {variable_fraction:.3f} < {args.min_variable_cell_fraction:.3f}")
    if rows_with_r_tags < int(args.min_r_tag_rows):
        blockers.append(f"rows with R-tags {rows_with_r_tags} < {args.min_r_tag_rows}")
    if graph_fraction < 1.0:
        blockers.append(f"graph consistency fraction {graph_fraction:.3f} < 1.000")
    if rows_with_real_backbone < row_count:
        blockers.append(f"rows with real chemical backbone {rows_with_real_backbone} < {row_count}")
    if rows_with_acceptable_text_area < row_count:
        blockers.append(f"rows with acceptable Markush text area {rows_with_acceptable_text_area} < {row_count}")

    rmse_sorted = sorted(rmse_values)
    line_abs_p95_sorted = sorted(line_abs_p95_values)
    line_abs_max_sorted = sorted(line_abs_max_values)
    intersection_anchor_rmse_sorted = sorted(intersection_anchor_rmse_values)
    intersection_anchor_abs_max_sorted = sorted(intersection_anchor_abs_max_values)
    legacy_rmse_sorted = sorted(legacy_rmse_values)
    affine_scale_sorted = sorted(affine_scale_ratios)
    affine_determinant_sorted = sorted(affine_determinants)
    intersection_anchor_sorted = sorted(intersection_anchor_counts)
    atom_coordinate_margin_sorted = sorted(atom_coordinate_margins)
    atom_pair_distance_sorted = sorted(atom_pair_distances)
    report = {
        "csv": str(csv_path),
        "row_count": row_count,
        "passed": not blockers,
        "blockers": blockers,
        "metrics": {
            "rmse_mean": mean(rmse_values) if rmse_values else None,
            "rmse_max": max(rmse_values) if rmse_values else None,
            "rmse_p95": rmse_sorted[int(0.95 * (len(rmse_sorted) - 1))] if rmse_sorted else None,
            "line_abs_p95_mean": mean(line_abs_p95_values) if line_abs_p95_values else None,
            "line_abs_p95_p95": (
                line_abs_p95_sorted[int(0.95 * (len(line_abs_p95_sorted) - 1))]
                if line_abs_p95_sorted
                else None
            ),
            "line_abs_p95_max": max(line_abs_p95_values) if line_abs_p95_values else None,
            "line_abs_max_mean": mean(line_abs_max_values) if line_abs_max_values else None,
            "line_abs_max_p95": (
                line_abs_max_sorted[int(0.95 * (len(line_abs_max_sorted) - 1))]
                if line_abs_max_sorted
                else None
            ),
            "line_abs_max_max": max(line_abs_max_values) if line_abs_max_values else None,
            "intersection_anchor_rmse_mean": mean(intersection_anchor_rmse_values) if intersection_anchor_rmse_values else None,
            "intersection_anchor_rmse_p95": (
                intersection_anchor_rmse_sorted[int(0.95 * (len(intersection_anchor_rmse_sorted) - 1))]
                if intersection_anchor_rmse_sorted
                else None
            ),
            "intersection_anchor_rmse_max": max(intersection_anchor_rmse_values) if intersection_anchor_rmse_values else None,
            "intersection_anchor_abs_max_p95": (
                intersection_anchor_abs_max_sorted[int(0.95 * (len(intersection_anchor_abs_max_sorted) - 1))]
                if intersection_anchor_abs_max_sorted
                else None
            ),
            "intersection_anchor_abs_max_max": (
                max(intersection_anchor_abs_max_values) if intersection_anchor_abs_max_values else None
            ),
            "legacy_midpoint_rmse_mean": mean(legacy_rmse_values) if legacy_rmse_values else None,
            "legacy_midpoint_rmse_max": max(legacy_rmse_values) if legacy_rmse_values else None,
            "legacy_midpoint_rmse_p95": legacy_rmse_sorted[int(0.95 * (len(legacy_rmse_sorted) - 1))] if legacy_rmse_sorted else None,
            "affine_scale_ratio_max": max(affine_scale_ratios) if affine_scale_ratios else None,
            "affine_scale_ratio_p95": affine_scale_sorted[int(0.95 * (len(affine_scale_sorted) - 1))] if affine_scale_sorted else None,
            "affine_determinant_min": min(affine_determinants) if affine_determinants else None,
            "affine_determinant_max": max(affine_determinants) if affine_determinants else None,
            "intersection_anchor_min": min(intersection_anchor_counts) if intersection_anchor_counts else None,
            "intersection_anchor_p05": intersection_anchor_sorted[int(0.05 * (len(intersection_anchor_sorted) - 1))] if intersection_anchor_sorted else None,
            "fit_point_min": min(fit_points) if fit_points else 0,
            "fit_point_mean": mean(fit_points) if fit_points else None,
            "atom_coordinate_min_margin_min": min(atom_coordinate_margins) if atom_coordinate_margins else None,
            "atom_coordinate_min_margin_p01": (
                atom_coordinate_margin_sorted[int(0.01 * (len(atom_coordinate_margin_sorted) - 1))]
                if atom_coordinate_margin_sorted
                else None
            ),
            "atom_pair_min_distance_min": min(atom_pair_distances) if atom_pair_distances else None,
            "atom_pair_min_distance_p01": (
                atom_pair_distance_sorted[int(0.01 * (len(atom_pair_distance_sorted) - 1))]
                if atom_pair_distance_sorted
                else None
            ),
            "rows_with_ocr_cells": rows_with_cells,
            "rows_with_variable_cells": rows_with_variable_cells,
            "rows_with_variable_cells_fraction": variable_fraction,
            "rows_with_r_tags": rows_with_r_tags,
            "graph_consistency_fraction": graph_fraction,
            "rows_with_real_chemical_backbone": rows_with_real_backbone,
            "rows_with_acceptable_text_area": rows_with_acceptable_text_area,
        },
        "counts": {name: dict(counter.most_common()) for name, counter in counters.items()},
        "row_failures": row_failures[:80],
        "tail_shard_policy": tail_shard_policy,
        "policy": {
            "visual_review_still_required": True,
            "does_not_accept_manifest_by_itself": True,
            "purpose": "automatic Markush pose/OCR evidence check before human visual acceptance",
            "markush_pose_rmse_required": True,
            "rmse_source": (
                "render_quality.pose_mapping.line_constraint_rmse_svg_units from rendered CDK SVG bond-axis "
                "atom-center reconstruction for cdk_svg_bond_axis_atom_center_hybrid_mapping; legacy affine "
                "midpoint RMSE is diagnostic only"
            ),
            "legacy_midpoint_rmse_is_diagnostic_only": True,
            "affine_diagnostics_required": True,
            "affine_diagnostics_role": (
                "final affine for legacy line-constrained rows; affine seed sanity check for SVG-center rows"
            ),
            "max_affine_scale_ratio": float(args.max_affine_scale_ratio),
            "min_intersection_anchors": int(args.min_intersection_anchors),
            "max_line_abs_p95": float(args.max_line_abs_p95),
            "max_line_abs_max": float(args.max_line_abs_max),
            "max_intersection_anchor_rmse": float(args.max_intersection_anchor_rmse),
            "max_intersection_anchor_abs_max": float(args.max_intersection_anchor_abs_max),
            "min_atom_coordinate_margin": float(args.min_atom_coordinate_margin),
            "min_atom_pair_distance": float(args.min_atom_pair_distance),
            "rmse_must_be_at_or_below_threshold": True,
            "fit_points_must_be_at_least_min_fit_points": True,
            "min_rows_is_statistical_shard_guard": True,
            "scheduled_terminal_tail_shard_requires_manifest_and_schedule_alignment": True,
            "atom_index_alignment_required": True,
            "atom_index_alignment_schema_version": MARKUSH_ATOM_INDEX_ALIGNMENT_SCHEMA_VERSION,
            "atom_index_alignment_policy": (
                "CDK/V3000/SVG atom indices must be explicitly aligned to RDKit CXSMILES atom indices before "
                "training atom_coordinates, bonds, dummy atoms, and OCR cells are accepted."
            ),
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
