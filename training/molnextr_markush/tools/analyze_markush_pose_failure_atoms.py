from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.markush_layout_labels import cxsmiles_dummy_labels  # noqa: E402
from training.molnextr_markush.tools import build_pose_factory_markush_shard as gen  # noqa: E402
from utils.markush_labels import normalize_label  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def classify_failure(error: str) -> str:
    lowered = error.lower()
    if "invalidsmilesexception" in lowered or "kekul" in lowered:
        return "invalid_smiles_or_kekule"
    if "line-constrained pose mapping rmse" in lowered:
        return "pose_mapping_rmse_over_threshold"
    if "line residual" in lowered or "scale ratio" in lowered:
        return "pose_gate_other"
    return "other"


def read_candidate_plan(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_id = str(row.get("source_id") or "")
            if source_id:
                rows[source_id] = row
    return rows


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def bbox_center_svg(bbox: Any, viewbox: tuple[float, float, float, float]) -> np.ndarray | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    values = [safe_float(item) for item in bbox]
    if any(value is None for value in values):
        return None
    x1, y1, x2, y2 = [float(value) for value in values]
    return np.asarray([(x1 + x2) * 0.5 * viewbox[2], (y1 + y2) * 0.5 * viewbox[3]], dtype=np.float64)


def bbox_contains_norm(bbox: Any, point: np.ndarray, viewbox: tuple[float, float, float, float], *, margin: float) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    values = [safe_float(item) for item in bbox]
    if any(value is None for value in values):
        return False
    x = float(point[0] / viewbox[2])
    y = float(point[1] / viewbox[3])
    x1, y1, x2, y2 = [float(value) for value in values]
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def line_residuals(point: np.ndarray, axes: list[tuple[np.ndarray, float]]) -> list[float]:
    return [abs(float(np.dot(axis[0], point) - axis[1])) for axis in axes]


def segment_report(lines: list[tuple[float, float, float, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for x1, y1, x2, y2 in lines:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(math.hypot(dx, dy))
        rows.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "length": length,
                "midpoint": [float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)],
                "unit_direction": [float(dx / length), float(dy / length)] if length > 1e-9 else None,
            }
        )
    rows.sort(key=lambda item: -float(item.get("length") or 0.0))
    return rows


def residual_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def analyze_row(
    *,
    row_id: str,
    cxsmiles: str,
    dataset_root: Path,
    max_atoms: int,
    bbox_margin: float,
) -> dict[str, Any]:
    svg_path = dataset_root / "images" / f"{row_id}.svg"
    mol_path = dataset_root / "molfiles" / f"{row_id}.mol"
    result: dict[str, Any] = {
        "row_id": row_id,
        "svg_path": str(svg_path),
        "mol_path": str(mol_path),
        "cdk_outputs_present": svg_path.exists() and mol_path.exists(),
    }
    if not svg_path.exists() or not mol_path.exists():
        return result

    atoms, bonds = gen.parse_v3000_mol(mol_path)
    viewbox, lines_by_bond, atom_path_cells = gen.parse_svg_geometry(svg_path)
    affine, affine_seed = gen.fit_line_constrained_affine(atoms, bonds, lines_by_bond)
    centers, diagnostics = gen.reconstruct_svg_atom_centers(atoms, bonds, lines_by_bond, affine)
    dummy_labels = cxsmiles_dummy_labels(cxsmiles)

    try:
        ocr_cells = gen.parse_cdk_ocr_cells(cxsmiles, mol_path, svg_path)
        ocr_error = ""
    except Exception as exc:
        ocr_cells = []
        ocr_error = f"{type(exc).__name__}: {exc}"

    atom_path_by_index = {
        int(cell["atom_index"]): cell
        for cell in atom_path_cells
        if isinstance(cell, dict) and str(cell.get("atom_index", "")).lstrip("-").isdigit()
    }
    ocr_by_atom_text: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    ocr_by_atom: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in ocr_cells:
        if not isinstance(cell, dict) or "atom_index" not in cell:
            continue
        try:
            atom_index = int(cell.get("atom_index"))
        except (TypeError, ValueError):
            continue
        text = normalize_label(str(cell.get("text") or ""))
        ocr_by_atom_text[(atom_index, text)].append(cell)
        ocr_by_atom[atom_index].append(cell)

    atom_by_index = {int(atom["atom_index"]): atom for atom in atoms}
    selected_axes = gen.select_svg_bond_axes_with_endpoint_policy(bonds, lines_by_bond)
    axes_by_atom: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
    incident_bonds_by_atom: dict[int, list[dict[str, Any]]] = defaultdict(list)
    neighbors: dict[int, list[int]] = defaultdict(list)
    axis_policy_counts: Counter[str] = Counter()
    for bond in bonds:
        bond_index = int(bond["bond_index"])
        begin = int(bond["begin_atom_index"])
        end = int(bond["end_atom_index"])
        neighbors[begin].append(end)
        neighbors[end].append(begin)
        selected = selected_axes.get(bond_index)
        if selected is None:
            continue
        axis, policy = selected
        axis_policy_counts[str(policy.get("selection") or "unknown")] += 1
        bond_report = {
            "bond_index": int(bond_index),
            "bond_order": str(bond.get("bond_order") or ""),
            "begin_atom_index": int(begin),
            "end_atom_index": int(end),
            "begin_token": str(atom_by_index.get(begin, {}).get("token") or ""),
            "end_token": str(atom_by_index.get(end, {}).get("token") or ""),
            "axis_policy": policy,
            "line_segments": segment_report(lines_by_bond.get(bond_index) or []),
        }
        incident_bonds_by_atom[begin].append(bond_report)
        incident_bonds_by_atom[end].append(bond_report)
        axes_by_atom[begin].append(axis)
        axes_by_atom[end].append(axis)

    seed_points: dict[int, np.ndarray] = {}
    for atom in atoms:
        atom_index = int(atom["atom_index"])
        seed_points[atom_index] = np.asarray([atom["mol_x"], atom["mol_y"], 1.0], dtype=np.float64) @ affine

    atom_reports: list[dict[str, Any]] = []
    all_current_residuals: list[float] = []
    all_ocr_center_residuals_for_dummy: list[float] = []
    dummy_ocr_center_improves = 0
    dummy_ocr_center_worsens = 0
    dummy_ocr_center_passes_axes = 0
    for atom in atoms:
        atom_index = int(atom["atom_index"])
        token = str(atom.get("token") or "")
        axes = axes_by_atom.get(atom_index, [])
        center = centers.get(atom_index)
        current_residuals = line_residuals(center, axes) if center is not None else []
        all_current_residuals.extend(current_residuals)

        axis_intersection = gen.least_squares_axis_point(axes)
        atom_path_center = bbox_center_svg(atom_path_by_index.get(atom_index, {}).get("bbox"), viewbox)
        dummy_label = dummy_labels.get(atom_index)
        matching_ocr = ocr_by_atom_text.get((atom_index, normalize_label(dummy_label or "")), []) if dummy_label else []
        ocr_center = bbox_center_svg(matching_ocr[0].get("bbox"), viewbox) if len(matching_ocr) == 1 else None

        current_max = max(current_residuals) if current_residuals else None
        atom_path_residuals = line_residuals(atom_path_center, axes) if atom_path_center is not None else []
        ocr_residuals = line_residuals(ocr_center, axes) if ocr_center is not None else []
        if dummy_label and ocr_residuals:
            all_ocr_center_residuals_for_dummy.extend(ocr_residuals)
            ocr_max = max(ocr_residuals)
            if current_max is not None and ocr_max < current_max:
                dummy_ocr_center_improves += 1
            if current_max is not None and ocr_max > current_max:
                dummy_ocr_center_worsens += 1
            if ocr_max <= 0.10:
                dummy_ocr_center_passes_axes += 1

        source = "affine_seed_no_svg_axis"
        if len(axes) >= 2 and axis_intersection is not None:
            source = "svg_axis_intersection_lstsq"
        elif len(axes) == 1:
            source = "single_axis_affine_seed_projection"

        atom_reports.append(
            {
                "atom_index": atom_index,
                "token": token,
                "dummy_label": dummy_label,
                "degree": int(len(neighbors.get(atom_index, []))),
                "neighbor_indices": [int(item) for item in neighbors.get(atom_index, [])],
                "neighbor_tokens": [str(atom_by_index.get(item, {}).get("token") or "") for item in neighbors.get(atom_index, [])],
                "incident_bonds": incident_bonds_by_atom.get(atom_index, []),
                "center_source": source,
                "axis_count": int(len(axes)),
                "current_line_abs_max_svg_units": current_max,
                "current_line_abs_mean_svg_units": float(np.mean(current_residuals)) if current_residuals else None,
                "seed_line_abs_max_svg_units": max(line_residuals(seed_points[atom_index], axes)) if axes else None,
                "atom_path_bbox_present": atom_index in atom_path_by_index,
                "atom_path_bbox_contains_current_center": (
                    bbox_contains_norm(atom_path_by_index.get(atom_index, {}).get("bbox"), center, viewbox, margin=bbox_margin)
                    if center is not None
                    else False
                ),
                "atom_path_center_line_abs_max_svg_units": max(atom_path_residuals) if atom_path_residuals else None,
                "matching_ocr_cell_count": int(len(matching_ocr)),
                "ocr_cells_for_atom_count": int(len(ocr_by_atom.get(atom_index, []))),
                "ocr_center_contains_current_center": (
                    bbox_contains_norm(matching_ocr[0].get("bbox"), center, viewbox, margin=bbox_margin)
                    if center is not None and len(matching_ocr) == 1
                    else False
                ),
                "ocr_center_line_abs_max_svg_units": max(ocr_residuals) if ocr_residuals else None,
                "ocr_center_distance_from_current_svg_units": (
                    float(np.linalg.norm(ocr_center - center)) if ocr_center is not None and center is not None else None
                ),
            }
        )

    atom_reports.sort(
        key=lambda item: (
            -1.0
            if item.get("current_line_abs_max_svg_units") is None
            else -float(item.get("current_line_abs_max_svg_units") or 0.0),
            int(item["atom_index"]),
        )
    )
    result.update(
        {
            "atom_count": int(len(atoms)),
            "bond_count": int(len(bonds)),
            "dummy_label_count": int(len(dummy_labels)),
            "ocr_cell_count": int(len(ocr_cells)),
            "ocr_error": ocr_error,
            "fit_diagnostics": diagnostics,
            "affine_seed_diagnostics": affine_seed,
            "axis_policy_counts": dict(sorted(axis_policy_counts.items())),
            "current_line_residual_stats": residual_stats(all_current_residuals),
            "dummy_ocr_center_line_residual_stats": residual_stats(all_ocr_center_residuals_for_dummy),
            "dummy_ocr_center_improves_atom_count": int(dummy_ocr_center_improves),
            "dummy_ocr_center_worsens_atom_count": int(dummy_ocr_center_worsens),
            "dummy_ocr_center_passes_axis_abs_max_0p10_atom_count": int(dummy_ocr_center_passes_axes),
            "worst_atoms": atom_reports[: int(max_atoms)],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only atom-level audit for Markush pose failures; does not create or accept training data."
    )
    parser.add_argument("--manifest", required=True, help="Generated shard manifest with failures.")
    parser.add_argument("--candidate-plan-csv", required=True)
    parser.add_argument("--dataset-name", required=True, help="Existing /tmp/MarkushGenerator dataset name.")
    parser.add_argument("--row-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--categories", default="pose_mapping_rmse_over_threshold,pose_gate_other")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--max-atoms", type=int, default=8)
    parser.add_argument("--bbox-margin", type=float, default=0.002)
    args = parser.parse_args()

    manifest = load_json(Path(args.manifest))
    candidate_plan = read_candidate_plan(Path(args.candidate_plan_csv))
    wanted_categories = {item.strip() for item in str(args.categories).split(",") if item.strip()}
    dataset_root = gen.MARKUSH_GENERATOR_ROOT / "data" / "dataset" / str(args.dataset_name)

    rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    missing_plan = 0
    for failure in manifest.get("failures") or []:
        if not isinstance(failure, dict):
            continue
        error = str(failure.get("error") or "")
        category = classify_failure(error)
        category_counts[category] += 1
        if category not in wanted_categories:
            continue
        source_id = str(failure.get("source_id") or "")
        plan_row = candidate_plan.get(source_id)
        if plan_row is None:
            missing_plan += 1
            continue
        try:
            plan_index = int(plan_row.get("plan_index"))
        except (TypeError, ValueError):
            missing_plan += 1
            continue
        row_id = str(failure.get("attempt_row_id") or "")
        if not row_id:
            row_id = f"{args.row_prefix}_{plan_index:06d}"
        row = analyze_row(
            row_id=row_id,
            cxsmiles=str(failure.get("cxsmiles") or plan_row.get("cxsmiles") or ""),
            dataset_root=dataset_root,
            max_atoms=int(args.max_atoms),
            bbox_margin=float(args.bbox_margin),
        )
        row.update(
            {
                "category": category,
                "source_id": source_id,
                "plan_index": int(plan_index),
                "bucket": str(plan_row.get("annotation_r_bucket") or ""),
                "original_error": error,
                "audit_only": True,
            }
        )
        rows.append(row)
        if len(rows) >= int(args.limit):
            break

    worst_atom_counter: Counter[str] = Counter()
    worst_dummy_counter: Counter[str] = Counter()
    dummy_ocr_improves_rows = 0
    dummy_ocr_worsens_rows = 0
    for row in rows:
        worst_atoms = row.get("worst_atoms") if isinstance(row.get("worst_atoms"), list) else []
        if worst_atoms:
            worst = worst_atoms[0]
            worst_atom_counter[str(worst.get("token") or "")] += 1
            if worst.get("dummy_label"):
                worst_dummy_counter[str(worst.get("dummy_label"))] += 1
        if int(row.get("dummy_ocr_center_improves_atom_count") or 0) > 0:
            dummy_ocr_improves_rows += 1
        if int(row.get("dummy_ocr_center_worsens_atom_count") or 0) > 0:
            dummy_ocr_worsens_rows += 1

    report = {
        "schema_version": "markush_pose_failure_atom_audit_v1",
        "audit_only": True,
        "does_not_accept_training_data": True,
        "does_not_relax_pose_visual_or_substitution_anchor_gates": True,
        "manifest": str(args.manifest),
        "candidate_plan_csv": str(args.candidate_plan_csv),
        "dataset_name": str(args.dataset_name),
        "row_prefix": str(args.row_prefix),
        "dataset_root": str(dataset_root),
        "requested_categories": sorted(wanted_categories),
        "manifest_failure_category_counts": dict(sorted(category_counts.items())),
        "missing_candidate_plan_rows": int(missing_plan),
        "sample_count": int(len(rows)),
        "summary": {
            "worst_atom_token_counts": dict(sorted(worst_atom_counter.items())),
            "worst_dummy_label_examples": dict(worst_dummy_counter.most_common(20)),
            "rows_where_dummy_ocr_center_would_improve_some_atom": int(dummy_ocr_improves_rows),
            "rows_where_dummy_ocr_center_would_worsen_some_atom": int(dummy_ocr_worsens_rows),
        },
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["schema_version", "sample_count", "summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
