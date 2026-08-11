from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.tools.audit_markush_substitution_anchor_contract import (
    validate_row as validate_substitution_anchor_contract,
)
from training.molnextr_markush.src.pose_factory import (
    formal_nonlinear_pose_contract_passed,
    validate_markush_atom_index_alignment_contract,
)


COUNT_BUCKETS = ["1", "2", "3-4", "5-8", "9+"]
SVG_CENTER_FIT_METHOD = "svg_bond_axis_atom_center_hybrid_intersections_with_single_axis_affine_seed_projection"
DOCUMENT_REALISM_SCHEMA_VERSION = "markush_document_realism_v1"
DOCUMENT_REALISM_POLICY = "patent_literature_markushgenerator_cdk_svg_v1"
REALISM_RENDER_PARAMETER_KEYS = {
    "seed",
    "stroke_ratio",
    "bond_separation",
    "symbol_margin_ratio",
    "font_name",
    "font_size",
    "render_atom_numbers",
    "render_carbon_symbols",
    "render_aromatic_display",
    "render_deuterium_symbol",
    "render_terminal_carbons",
}


def parse_bucket_targets(value: str) -> dict[str, int]:
    text = str(value or "").strip()
    if not text:
        return {}
    if text.isdigit():
        return {bucket: int(text) for bucket in COUNT_BUCKETS}
    targets: dict[str, int] = {}
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid bucket target {item!r}; expected bucket=count")
        bucket, count_text = [part.strip() for part in item.split("=", 1)]
        if bucket not in COUNT_BUCKETS:
            raise ValueError(f"invalid bucket {bucket!r}; expected one of {COUNT_BUCKETS}")
        count = int(count_text)
        if count < 0:
            raise ValueError(f"bucket target must be non-negative: {item!r}")
        targets[bucket] = count
    return {bucket: int(targets.get(bucket, 0)) for bucket in COUNT_BUCKETS}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def validate_document_realism_contract(quality: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    document_realism = quality.get("document_realism") if isinstance(quality.get("document_realism"), dict) else {}
    if not document_realism:
        return ["missing patent/literature document_realism contract"]
    if document_realism.get("schema_version") != DOCUMENT_REALISM_SCHEMA_VERSION:
        issues.append("document_realism schema mismatch")
    if document_realism.get("policy") != DOCUMENT_REALISM_POLICY:
        issues.append("document_realism policy mismatch")
    if document_realism.get("generation_method") != "existing_markushgenerator_cdk_svg_depictor":
        issues.append("document_realism generation method is not MarkushGenerator/CDK SVG")
    if document_realism.get("no_new_renderer_or_image_method") is not True:
        issues.append("document_realism does not forbid unreviewed renderer/image-method replacement")
    if document_realism.get("machine_audit_passed") is not True:
        issues.append("document_realism machine audit did not pass")
    if document_realism.get("manual_visual_review_required") is not True:
        issues.append("document_realism does not require manual visual review")
    if document_realism.get("manual_visual_review_passed") is True:
        issues.append("raw/accepted candidate must not self-certify manual visual review")
    if str(document_realism.get("source_visual_domain") or "") != "patent_literature_markush_document_crop":
        issues.append("document_realism source visual domain is not patent/literature Markush document crop")
    if not isinstance(document_realism.get("external_basis"), list) or not document_realism.get("external_basis"):
        issues.append("document_realism external basis is missing")
    missing_params = document_realism.get("render_parameter_keys_missing")
    if not isinstance(missing_params, list):
        issues.append("document_realism render parameter audit is missing")
    elif missing_params:
        issues.append(f"document_realism missing render parameters: {missing_params}")
    render_params = document_realism.get("render_parameters") if isinstance(document_realism.get("render_parameters"), dict) else {}
    if not render_params:
        issues.append("document_realism render parameters are missing")
    gate = quality.get("quality_gates") if isinstance(quality.get("quality_gates"), dict) else {}
    if gate.get("patent_literature_realism_machine_audit_passed") is not True:
        issues.append("quality gate patent_literature_realism_machine_audit_passed is not true")
    provenance = quality.get("render_provenance") if isinstance(quality.get("render_provenance"), dict) else {}
    provenance_params = provenance.get("parameters") if isinstance(provenance.get("parameters"), dict) else {}
    missing_from_provenance = sorted(REALISM_RENDER_PARAMETER_KEYS - set(provenance_params))
    if missing_from_provenance:
        issues.append(f"render_provenance missing realism parameters: {missing_from_provenance}")
    return issues


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def row_image_path(row: dict[str, str], csv_path: Path) -> Path:
    raw = str(row.get("file_path") or row.get("image_path") or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    return path if path.is_absolute() else csv_path.parent / path


def bbox_values(cell: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(cell, dict):
        return None
    bbox = cell.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    values = [as_float(item) for item in bbox]
    if any(value is None for value in values):
        return None
    x1, y1, x2, y2 = [float(value) for value in values]
    if not (0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0):
        return None
    return x1, y1, x2, y2


def bbox_area(cell: Any) -> float | None:
    values = bbox_values(cell)
    if values is None:
        return None
    x1, y1, x2, y2 = values
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def atom_coordinate_audit(atom_coordinates: Any, graph: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": False,
        "count": 0,
        "min_margin": None,
        "min_pair_distance": None,
        "issues": [],
    }
    if not isinstance(atom_coordinates, list) or not atom_coordinates:
        result["issues"].append("missing atom_coordinates")
        return result
    graph_atom_count = int(graph.get("atom_count") or -1)
    if graph_atom_count >= 0 and graph_atom_count != len(atom_coordinates):
        result["issues"].append("atom coordinate count does not match graph atom_count")
    coords: list[tuple[float, float]] = []
    seen_indices: set[int] = set()
    for item in atom_coordinates:
        if not isinstance(item, dict):
            result["issues"].append("invalid atom coordinate record")
            continue
        try:
            atom_index = int(item.get("atom_index"))
        except (TypeError, ValueError):
            result["issues"].append("invalid atom coordinate index")
            continue
        if atom_index in seen_indices:
            result["issues"].append("duplicate atom coordinate index")
        seen_indices.add(atom_index)
        if graph_atom_count >= 0 and not (0 <= atom_index < graph_atom_count):
            result["issues"].append("atom coordinate index outside graph atom_count")
        x = as_float(item.get("x"))
        y = as_float(item.get("y"))
        if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
            result["issues"].append("invalid atom coordinate value")
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            result["issues"].append("atom coordinate outside normalized image")
        coords.append((float(x), float(y)))
    if graph_atom_count >= 0 and seen_indices != set(range(graph_atom_count)):
        result["issues"].append("atom coordinate indices are not contiguous over graph atoms")
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
    result["valid"] = not result["issues"]
    return result


def literal_star_variable_issues(quality: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
    atom_coordinates = quality.get("atom_coordinates") if isinstance(quality.get("atom_coordinates"), list) else []
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    ocr_cells = markush.get("ocr_cells") if isinstance(markush.get("ocr_cells"), list) else []
    visible_star_indices = graph.get("visible_star_token_indices")
    if isinstance(visible_star_indices, list) and visible_star_indices:
        issues.append("literal star atom coordinate tokens are not accepted Markush variables")
    if any(isinstance(atom, dict) and str(atom.get("token") or "").strip() == "*" for atom in atom_coordinates):
        issues.append("atom_coordinates contains literal '*' token")
    if any(isinstance(cell, dict) and str(cell.get("text") or "").strip() == "*" for cell in ocr_cells):
        issues.append("Markush OCR cells contain literal '*' text")
    return issues


def bond_index_issues(bonds: Any, atom_count: int) -> list[str]:
    if not isinstance(bonds, list):
        return ["missing bonds"]
    issues: list[str] = []
    for bond in bonds:
        if isinstance(bond, dict):
            begin = bond.get("begin_atom_index")
            end = bond.get("end_atom_index")
        elif isinstance(bond, list) and len(bond) >= 2:
            begin, end = bond[0], bond[1]
        else:
            issues.append("invalid bond record")
            continue
        try:
            begin_index = int(begin)
            end_index = int(end)
        except (TypeError, ValueError):
            issues.append("invalid bond atom index")
            continue
        if begin_index == end_index or not (0 <= begin_index < atom_count) or not (0 <= end_index < atom_count):
            issues.append("bond atom index outside graph atom_count")
    return issues


def markush_bucket(row: dict[str, str]) -> str:
    quality = parse_quality(row)
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    bucket = str(markush.get("r_tag_count_bucket") or "").strip()
    if bucket in COUNT_BUCKETS:
        return bucket
    try:
        count = int(markush.get("r_tag_count") or 0)
    except (TypeError, ValueError):
        count = 0
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if 3 <= count <= 4:
        return "3-4"
    if 5 <= count <= 8:
        return "5-8"
    if count >= 9:
        return "9+"
    return ""


def candidate_plan_index(row: dict[str, str]) -> int:
    quality = parse_quality(row)
    try:
        return int(quality.get("candidate_plan_index"))
    except (TypeError, ValueError):
        return 10**12


def is_svg_center_fit_method(method: str) -> bool:
    return SVG_CENTER_FIT_METHOD in str(method or "")


def pose_affine_diagnostics(pose_mapping: dict[str, Any], fit_diagnostics: dict[str, Any]) -> dict[str, Any]:
    if is_svg_center_fit_method(str(pose_mapping.get("fit_method") or "")):
        seed = pose_mapping.get("affine_seed_diagnostics")
        if isinstance(seed, dict) and isinstance(seed.get("affine"), dict):
            return seed["affine"]
        nested = fit_diagnostics.get("affine_seed_diagnostics")
        if isinstance(nested, dict) and isinstance(nested.get("affine"), dict):
            return nested["affine"]
        return {}
    return fit_diagnostics.get("affine") if isinstance(fit_diagnostics.get("affine"), dict) else {}


def validate_markush_row(row: dict[str, str], csv_path: Path, args: argparse.Namespace) -> list[str]:
    issues: list[str] = []
    quality = parse_quality(row)
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    render_provenance = quality.get("render_provenance") if isinstance(quality.get("render_provenance"), dict) else {}
    graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
    pose_alignment = quality.get("pose_alignment") if isinstance(quality.get("pose_alignment"), dict) else {}
    pose_mapping = quality.get("pose_mapping") if isinstance(quality.get("pose_mapping"), dict) else {}
    quality_gates = quality.get("quality_gates") if isinstance(quality.get("quality_gates"), dict) else {}
    fit_diagnostics = pose_mapping.get("fit_diagnostics") if isinstance(pose_mapping.get("fit_diagnostics"), dict) else {}
    affine = pose_affine_diagnostics(pose_mapping, fit_diagnostics)
    source_id = str(row.get("source_id") or "").strip()
    if not source_id:
        issues.append("missing source_id")
    if str(row.get("structure_type_bucket") or "") != "markush_layout":
        issues.append("row structure_type_bucket is not markush_layout")
    if str(quality.get("structure_type") or "") != "markush_layout":
        issues.append("render_quality structure_type is not markush_layout")
    atom_audit = atom_coordinate_audit(quality.get("atom_coordinates"), graph)
    issues.extend(str(item) for item in atom_audit.get("issues", []))
    issues.extend(literal_star_variable_issues(quality))
    issues.extend(
        f"Markush atom index alignment contract failed: {issue}"
        for issue in validate_markush_atom_index_alignment_contract(quality)
    )
    atom_count = int(graph.get("atom_count") or atom_audit.get("count") or 0)
    issues.extend(bond_index_issues(quality.get("bonds"), atom_count))
    cells = markush.get("ocr_cells") if isinstance(markush.get("ocr_cells"), list) else []
    if not cells:
        issues.append("missing Markush OCR cells")
    cell_area_sum = 0.0
    cell_area_max = 0.0
    for cell in cells:
        area = bbox_area(cell)
        if area is None:
            issues.append("invalid Markush OCR cell bbox")
            continue
        cell_area_sum += area
        cell_area_max = max(cell_area_max, area)
        if "atom_index" in cell:
            try:
                cell_atom_index = int(cell.get("atom_index"))
            except (TypeError, ValueError):
                issues.append("invalid Markush OCR cell atom_index")
                continue
            if not (0 <= cell_atom_index < atom_count):
                issues.append("Markush OCR cell atom_index outside graph atom_count")
    if cell_area_sum > float(args.max_ocr_cell_area_sum):
        issues.append("Markush OCR cell area sum exceeds accepted-candidate threshold")
    if cell_area_max > float(args.max_ocr_cell_area_max):
        issues.append("Markush OCR cell area max exceeds accepted-candidate threshold")
    if not str(markush.get("annotation") or "").strip():
        issues.append("missing Markush annotation")
    if markush_bucket(row) not in COUNT_BUCKETS:
        issues.append("missing valid Markush r_tag_count_bucket")
    issues.extend(f"Markush document realism contract failed: {issue}" for issue in validate_document_realism_contract(quality))
    substitution_issues, _substitution_metrics = validate_substitution_anchor_contract(
        row,
        csv_path=csv_path,
        bbox_margin=float(args.max_variable_bbox_atom_margin),
        require_real_atom_neighbor=not bool(args.allow_dummy_only_variable_neighbor),
        check_image=not bool(args.skip_substitution_image_check),
    )
    issues.extend(f"Markush substitution anchor contract failed: {issue}" for issue in substitution_issues)
    if graph.get("row_smiles_canonical_matches_mol") is not True:
        issues.append("graph consistency is not true")
    rmse = as_float(pose_mapping.get("line_constraint_rmse_svg_units"))
    threshold = as_float(pose_mapping.get("line_constraint_rmse_threshold"))
    line_abs_p95 = as_float(pose_mapping.get("line_constraint_abs_p95_svg_units"))
    line_abs_max = as_float(pose_mapping.get("line_constraint_abs_max_svg_units"))
    anchor_rmse = as_float(pose_mapping.get("intersection_anchor_rmse_svg_units"))
    anchor_abs_max = as_float(pose_mapping.get("intersection_anchor_abs_max_svg_units"))
    scale_ratio = as_float(affine.get("scale_ratio"))
    determinant = as_float(affine.get("determinant"))
    fit_points = int(pose_mapping.get("line_constraint_fit_equation_count") or 0)
    intersection_anchors = int(fit_diagnostics.get("intersection_anchor_count") or 0)
    formal_nonlinear_pose = formal_nonlinear_pose_contract_passed(quality)
    if str(pose_alignment.get("pose_mapping_metric") or "") != "line_constraint_rmse_svg_units":
        issues.append("pose_alignment metric is not line_constraint_rmse_svg_units")
    if pose_alignment.get("image_to_graph_orientation_alignment") is not True:
        issues.append("pose_alignment does not preserve image-to-graph orientation")
    if pose_alignment.get("coordinate_mutation_after_render") is not False and not formal_nonlinear_pose:
        issues.append("pose coordinates were mutated after render without a passing formal nonlinear SVG-polyline pose contract")
    if pose_alignment.get("synchronized_after_augmentation") is not True:
        issues.append("pose alignment is not synchronized after augmentation")
    if str(pose_alignment.get("orientation_policy") or "") != str(args.expected_coord_policy):
        issues.append("pose_alignment orientation_policy is not the accepted MolNexTR policy")
    if str(quality.get("coord_policy") or "") != str(args.expected_coord_policy):
        issues.append("coord_policy is not the accepted MolNexTR policy")
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
            issues.append(f"quality gate {gate_name} is not true")
    if rmse is None or threshold is None or rmse > min(float(args.max_rmse), threshold):
        issues.append("line-constrained pose RMSE failed")
    if line_abs_p95 is None or line_abs_p95 > float(args.max_line_abs_p95):
        issues.append("line residual p95 exceeds MolNexTR accepted-candidate threshold")
    if line_abs_max is None or line_abs_max > float(args.max_line_abs_max):
        issues.append("line residual max exceeds MolNexTR accepted-candidate threshold")
    if anchor_rmse is None or anchor_rmse > float(args.max_intersection_anchor_rmse):
        issues.append("intersection anchor RMSE exceeds MolNexTR accepted-candidate threshold")
    if anchor_abs_max is None or anchor_abs_max > float(args.max_intersection_anchor_abs_max):
        issues.append("intersection anchor max residual exceeds MolNexTR accepted-candidate threshold")
    if formal_nonlinear_pose:
        nonlinear_pose = quality.get("nonlinear_pose_preservation") if isinstance(quality.get("nonlinear_pose_preservation"), dict) else {}
        nonlinear_rmse = as_float(nonlinear_pose.get("line_constraint_rmse_svg_units"))
        nonlinear_p95 = as_float(nonlinear_pose.get("line_constraint_abs_p95_svg_units"))
        nonlinear_max = as_float(nonlinear_pose.get("line_constraint_abs_max_svg_units"))
        nonlinear_anchor_rmse = as_float(nonlinear_pose.get("intersection_anchor_rmse_svg_units"))
        nonlinear_anchor_max = as_float(nonlinear_pose.get("intersection_anchor_abs_max_svg_units"))
        if nonlinear_rmse is None or nonlinear_rmse > float(args.max_rmse):
            issues.append("formal nonlinear warped SVG-polyline pose RMSE failed")
        if nonlinear_p95 is None or nonlinear_p95 > float(args.max_line_abs_p95):
            issues.append("formal nonlinear warped SVG-polyline line residual p95 failed")
        if nonlinear_max is None or nonlinear_max > float(args.max_line_abs_max):
            issues.append("formal nonlinear warped SVG-polyline line residual max failed")
        if nonlinear_anchor_rmse is None or nonlinear_anchor_rmse > float(args.max_intersection_anchor_rmse):
            issues.append("formal nonlinear warped SVG-polyline intersection anchor RMSE failed")
        if nonlinear_anchor_max is None or nonlinear_anchor_max > float(args.max_intersection_anchor_abs_max):
            issues.append("formal nonlinear warped SVG-polyline intersection anchor max residual failed")
    if determinant is None or determinant >= 0.0:
        issues.append("affine determinant does not preserve SVG/image orientation convention")
    if scale_ratio is None or scale_ratio > float(args.max_affine_scale_ratio):
        issues.append("affine scale ratio exceeds MolNexTR accepted-candidate threshold")
    if fit_points < int(args.min_fit_points):
        issues.append("too few line-constrained fit equations")
    if intersection_anchors < int(args.min_intersection_anchors):
        issues.append("too few multibond intersection anchors")
    min_margin = atom_audit.get("min_margin")
    if (
        float(args.min_atom_coordinate_margin) > 0.0
        and (min_margin is None or float(min_margin) < float(args.min_atom_coordinate_margin))
    ):
        issues.append("atom coordinate margin below MolNexTR accepted-candidate threshold")
    min_pair_distance = atom_audit.get("min_pair_distance")
    if (
        float(args.min_atom_pair_distance) > 0.0
        and min_pair_distance is not None
        and float(min_pair_distance) < float(args.min_atom_pair_distance)
    ):
        issues.append("atom coordinates are too close for MolNexTR coordinate supervision")
    if "candidate_plan_index" not in quality:
        issues.append("missing candidate_plan_index")
    if not str(quality.get("source_document_key") or "").strip():
        issues.append("missing source_document_key")
    source_dataset = str(quality.get("source_dataset") or "").strip()
    source_file = str(quality.get("source_file") or "").strip()
    if "markushgrapher-synthetic-training" in source_file and source_dataset != "markushgrapher-synthetic-training":
        issues.append(
            "source_dataset does not match MG1 source_file provenance: "
            f"source_dataset={source_dataset!r}, source_file={source_file!r}"
        )
    if not render_provenance.get("seeded_depictor"):
        issues.append("render provenance does not declare seeded_depictor")
    if "parameters" not in render_provenance:
        issues.append("missing render provenance parameters")
    image_path = row_image_path(row, csv_path)
    if not image_path.exists():
        issues.append(f"image file missing: {image_path}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge generated Markush CDK shards into a deterministic candidate shard for downstream gates."
    )
    parser.add_argument("--csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket-targets", default="")
    parser.add_argument("--output-name", default="markush_layout_positive.csv")
    parser.add_argument("--max-issues", type=int, default=80)
    parser.add_argument("--max-rmse", type=float, default=0.05)
    parser.add_argument("--max-line-abs-p95", type=float, default=0.05)
    parser.add_argument("--max-line-abs-max", type=float, default=0.10)
    parser.add_argument("--max-intersection-anchor-rmse", type=float, default=0.05)
    parser.add_argument("--max-intersection-anchor-abs-max", type=float, default=0.10)
    parser.add_argument("--max-affine-scale-ratio", type=float, default=1.25)
    parser.add_argument("--min-atom-coordinate-margin", type=float, default=0.0)
    parser.add_argument("--min-atom-pair-distance", type=float, default=0.005)
    parser.add_argument("--max-ocr-cell-area-sum", type=float, default=0.12)
    parser.add_argument("--max-ocr-cell-area-max", type=float, default=0.08)
    parser.add_argument("--max-variable-bbox-atom-margin", type=float, default=0.015)
    parser.add_argument("--allow-dummy-only-variable-neighbor", action="store_true")
    parser.add_argument("--skip-substitution-image-check", action="store_true")
    parser.add_argument("--min-fit-points", type=int, default=6)
    parser.add_argument("--min-intersection-anchors", type=int, default=2)
    parser.add_argument(
        "--expected-coord-policy",
        default="cdk_svg_bond_axis_atom_center_hybrid_mapping",
    )
    args = parser.parse_args()

    targets = parse_bucket_targets(args.bucket_targets)
    rows_with_source: list[tuple[dict[str, str], Path]] = []
    fieldnames: list[str] = []
    input_counts: dict[str, int] = {}
    for csv_text in args.csv:
        csv_path = Path(csv_text)
        rows = read_rows(csv_path)
        input_counts[str(csv_path)] = len(rows)
        if rows and not fieldnames:
            fieldnames = list(rows[0].keys())
        for row in rows:
            rows_with_source.append((row, csv_path))

    rows_with_source.sort(key=lambda item: (candidate_plan_index(item[0]), str(item[0].get("source_id") or "")))
    selected: list[tuple[dict[str, str], Path]] = []
    rejected: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    seen_plan_indexes: set[int] = set()
    selected_by_bucket: Counter[str] = Counter()
    source_document_keys: list[str] = []

    for row, csv_path in rows_with_source:
        source_id = str(row.get("source_id") or "").strip()
        plan_index = candidate_plan_index(row)
        bucket = markush_bucket(row)
        issues = validate_markush_row(row, csv_path, args)
        if source_id in seen_source_ids:
            issues.append("duplicate source_id")
        if plan_index in seen_plan_indexes:
            issues.append("duplicate candidate_plan_index")
        if targets and selected_by_bucket[bucket] >= int(targets.get(bucket, 0)):
            issues.append("bucket target already filled")
        if issues:
            rejected.append(
                {
                    "source_id": source_id,
                    "candidate_plan_index": plan_index,
                    "bucket": bucket,
                    "issues": issues,
                    "csv": str(csv_path),
                }
            )
            continue
        selected.append((row, csv_path))
        seen_source_ids.add(source_id)
        seen_plan_indexes.add(plan_index)
        selected_by_bucket[bucket] += 1
        quality = parse_quality(row)
        source_document_keys.append(str(quality.get("source_document_key") or ""))

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / args.output_name
    output_rows: list[dict[str, str]] = []
    copied_images = 0
    for row, csv_path in selected:
        source_image = row_image_path(row, csv_path)
        output_row = dict(row)
        if source_image.exists():
            target_image = image_dir / source_image.name
            if not target_image.exists() or source_image.resolve() != target_image.resolve():
                shutil.copy2(source_image, target_image)
                copied_images += 1
            output_row["file_path"] = str(Path("images") / target_image.name)
            if "image_path" in output_row:
                output_row["image_path"] = output_row["file_path"]
        output_rows.append(output_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    blockers: list[str] = []
    if targets:
        for bucket in COUNT_BUCKETS:
            if int(selected_by_bucket.get(bucket, 0)) < int(targets.get(bucket, 0)):
                blockers.append(
                    f"bucket {bucket} selected {int(selected_by_bucket.get(bucket, 0))} < target {int(targets.get(bucket, 0))}"
                )
    if len(source_document_keys) != len(set(source_document_keys)):
        blockers.append("selected rows contain duplicate source_document_key")

    manifest = {
        "csv": str(output_csv),
        "input_csv": args.csv,
        "input_counts": input_counts,
        "row_count": len(output_rows),
        "copied_images": copied_images,
        "rejected_rows": len(rejected),
        "rejected_examples": rejected[: int(args.max_issues)],
        "bucket_targets": targets,
        "counts": {"markush_layout": len(output_rows)},
        "counts_by_r_tag_bucket": {bucket: int(selected_by_bucket.get(bucket, 0)) for bucket in COUNT_BUCKETS},
        "source_document_overlap": len(source_document_keys) - len(set(source_document_keys)),
        "candidate_plan_index_overlap": len(seen_plan_indexes) - len(set(seen_plan_indexes)),
        "status": "merged_markush_candidate_requires_validation_visual_review_and_leak_check",
        "molnextr_pose_filter": {
            "purpose": "accepted-candidate filtering for MolNexTR image-to-graph orientation supervision",
            "max_rmse": float(args.max_rmse),
            "max_line_abs_p95": float(args.max_line_abs_p95),
            "max_line_abs_max": float(args.max_line_abs_max),
            "max_intersection_anchor_rmse": float(args.max_intersection_anchor_rmse),
            "max_intersection_anchor_abs_max": float(args.max_intersection_anchor_abs_max),
            "max_affine_scale_ratio": float(args.max_affine_scale_ratio),
            "min_atom_coordinate_margin": float(args.min_atom_coordinate_margin),
            "min_atom_pair_distance": float(args.min_atom_pair_distance),
            "max_ocr_cell_area_sum": float(args.max_ocr_cell_area_sum),
            "max_ocr_cell_area_max": float(args.max_ocr_cell_area_max),
            "max_variable_bbox_atom_margin": float(args.max_variable_bbox_atom_margin),
            "allow_dummy_only_variable_neighbor": bool(args.allow_dummy_only_variable_neighbor),
            "min_fit_points": int(args.min_fit_points),
            "min_intersection_anchors": int(args.min_intersection_anchors),
            "expected_coord_policy": str(args.expected_coord_policy),
            "rmse_unit": "CDK/SVG units on 289x289 generated images",
            "generation_method_unchanged": True,
            "markush_substitution_anchor_contract_required": True,
            "patent_literature_document_realism_contract_required": True,
            "patent_literature_document_realism_policy": DOCUMENT_REALISM_POLICY,
            "manual_visual_review_still_required": True,
        },
        "accepted": False,
        "rejected": False,
        "passed": not blockers,
        "blockers": blockers,
        "acceptance": {
            "accepted": False,
            "rejected": False,
            "visual_review_passed": False,
            "source_leak_check_passed": False,
            "pose_mapping_review_passed": False,
            "reason": "Merged Markush candidate; downstream schema, pose, visual, leak, coverage, confidence, and runtime gates are still required.",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
