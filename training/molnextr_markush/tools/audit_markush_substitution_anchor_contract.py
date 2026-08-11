"""Markush substitution-anchor contract auditor.

Validates that every Markush variable (R-group) atom in a pose-factory row
has a well-defined substitution anchor: the dummy/wildcard atom that marks
the attachment point must connect to at least one real (non-dummy) scaffold
atom, and the variable's OCR bounding-box must align with the corresponding
atom coordinate in normalised image space.

This module was recreated after the redundant-script cleanup removed the
original.  The interface (``validate_row``) is preserved so that
``prepare_markush_cdk_accepted_candidate.py`` continues to work unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Row parsing helpers
# --------------------------------------------------------------------------- #

def _parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _atom_token(atom: Any) -> str:
    if isinstance(atom, dict):
        return str(atom.get("token") or atom.get("symbol") or "").strip()
    return ""


def _is_dummy_atom(atom: Any) -> bool:
    return _atom_token(atom) == "*"


def _bbox_center(cell: Any) -> tuple[float, float] | None:
    if not isinstance(cell, dict):
        return None
    bbox = cell.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    values = [_as_float(item) for item in bbox]
    if any(v is None for v in values):
        return None
    x1, y1, x2, y2 = values  # type: ignore[misc]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _build_adjacency(bonds: Any, atom_count: int) -> dict[int, list[int]]:
    """Return atom_index → list of neighbour atom indices from bond records."""
    adjacency: dict[int, list[int]] = {index: [] for index in range(atom_count)}
    if not isinstance(bonds, list):
        return adjacency
    for bond in bonds:
        if isinstance(bond, dict):
            begin_raw = bond.get("begin_atom_index")
            end_raw = bond.get("end_atom_index")
        elif isinstance(bond, (list, tuple)) and len(bond) >= 2:
            begin_raw, end_raw = bond[0], bond[1]
        else:
            continue
        try:
            begin = int(begin_raw)
            end = int(end_raw)
        except (TypeError, ValueError):
            continue
        if begin == end:
            continue
        if 0 <= begin < atom_count and 0 <= end < atom_count:
            adjacency.setdefault(begin, []).append(end)
            adjacency.setdefault(end, []).append(begin)
    return adjacency


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def validate_row(
    row: dict[str, str],
    *,
    csv_path: str | Path | None = None,
    bbox_margin: float = 0.05,
    require_real_atom_neighbor: bool = True,
    check_image: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Validate the substitution-anchor contract for a single Markush row.

    Parameters
    ----------
    row
        A CSV dict-row from the pose-factory shard.
    csv_path
        Path to the shard CSV (used to resolve relative image paths when
        *check_image* is ``True``).
    bbox_margin
        Maximum allowable Euclidean distance (in normalised image coords)
        between a variable OCR cell centre and its referenced atom coordinate.
    require_real_atom_neighbor
        When *True*, each variable atom must have at least one non-dummy
        neighbour in the molecular graph.
    check_image
        When *True*, attempt to open the source image to confirm it exists
        and is readable.

    Returns
    -------
    tuple[list[str], dict[str, Any]]
        *(issues, metrics)* — ``issues`` is a list of human-readable contract
        violations; ``metrics`` holds scalar diagnostics for aggregation.
    """
    del csv_path  # reserved for future image-based checks
    issues: list[str] = []
    metrics: dict[str, Any] = {
        "variable_atoms_checked": 0,
        "dummy_only_neighbors": 0,
        "bbox_mismatches": 0,
        "max_bbox_distance": 0.0,
        "image_checked": False,
    }

    quality = _parse_quality(row)
    if not quality:
        return (["render_quality is missing or invalid"], metrics)

    graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    atom_coordinates = quality.get("atom_coordinates")
    bonds = quality.get("bonds")
    cells = markush.get("ocr_cells") if isinstance(markush.get("ocr_cells"), list) else []

    # --- optional image existence check ---------------------------------- #
    if check_image:
        image_path_text = str(row.get("file_path") or row.get("image_path") or "").strip()
        if not image_path_text:
            issues.append("check_image requested but row has no file_path/image_path")
        else:
            try:
                from PIL import Image  # noqa: WPS433 — lazy import

                Image.open(image_path_text).verify()
                metrics["image_checked"] = True
            except Exception as error:  # noqa: BLE001 — any image failure is a contract issue
                issues.append(f"source image is not readable: {error}")

    # --- collect variable atoms from OCR cells --------------------------- #
    atom_count = int(graph.get("atom_count") or 0)
    if not isinstance(atom_coordinates, list) or not atom_coordinates:
        return (["atom_coordinates missing — cannot audit substitution anchors"], metrics)

    coordinate_by_index: dict[int, dict[str, Any]] = {}
    for atom in atom_coordinates:
        if not isinstance(atom, dict):
            continue
        try:
            index = int(atom.get("atom_index"))
        except (TypeError, ValueError):
            continue
        coordinate_by_index[index] = atom

    variable_indices: list[int] = []
    for cell in cells:
        cell_atom_index = _cell_atom_index(cell)
        if cell_atom_index is not None:
            variable_indices.append(cell_atom_index)

    if not variable_indices:
        return (issues, metrics)  # nothing to audit

    adjacency = _build_adjacency(bonds, max(atom_count, max(variable_indices) + 1))

    # --- per-variable anchor checks -------------------------------------- #
    for var_index in variable_indices:
        metrics["variable_atoms_checked"] += 1
        atom = coordinate_by_index.get(var_index)
        if atom is None:
            issues.append(f"variable atom_index {var_index} has no coordinate record")
            continue

        # Neighbour reality check
        neighbors = adjacency.get(var_index, [])
        if not neighbors:
            issues.append(f"variable atom_index {var_index} has no graph neighbours (isolated anchor)")
        elif require_real_atom_neighbor:
            neighbor_atoms = [coordinate_by_index.get(n) for n in neighbors]
            has_real_neighbor = any(
                neighbor is not None and not _is_dummy_atom(neighbor) for neighbor in neighbor_atoms
            )
            if not has_real_neighbor:
                metrics["dummy_only_neighbors"] += 1
                issues.append(
                    f"variable atom_index {var_index} has only dummy/wildcard neighbours"
                )

        # Bbox alignment check
        cell = next((c for c in cells if _cell_atom_index(c) == var_index), None)
        bbox_center = _bbox_center(cell)
        atom_x = _as_float(atom.get("x"))
        atom_y = _as_float(atom.get("y"))
        if bbox_center is not None and atom_x is not None and atom_y is not None:
            distance = math.dist(bbox_center, (atom_x, atom_y))
            metrics["max_bbox_distance"] = max(metrics["max_bbox_distance"], distance)
            if distance > bbox_margin:
                metrics["bbox_mismatches"] += 1
                issues.append(
                    f"variable atom_index {var_index} bbox-atom distance {distance:.4f} "
                    f"exceeds margin {bbox_margin:.4f}"
                )

    return (issues, metrics)


def _cell_atom_index(cell: Any) -> int | None:
    try:
        return int(cell.get("atom_index")) if isinstance(cell, dict) else None
    except (TypeError, ValueError):
        return None
