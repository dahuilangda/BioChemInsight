from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


MAX_CUT_CONNECTOR_PX_AT_384 = 56.0
MAX_DUMMY_CONNECTOR_PX_AT_384 = 54.0
MAX_QUERY_CONNECTOR_PX_AT_384 = 58.0


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unit(vector: tuple[float, float]) -> tuple[float, float] | None:
    norm = math.hypot(vector[0], vector[1])
    if norm < 1e-8:
        return None
    return vector[0] / norm, vector[1] / norm


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def image_path_for_row(input_csv: Path, row: dict[str, str]) -> Path:
    path = Path(str(row.get("file_path") or row.get("image_path") or ""))
    if path.is_absolute():
        return path
    return input_csv.parent / path


def dark_near(arr: Any, x: float, y: float, radius: int, threshold: int = 215) -> bool:
    h, w = arr.shape[:2]
    left = max(0, int(math.floor(x - radius)))
    right = min(w, int(math.ceil(x + radius + 1)))
    top = max(0, int(math.floor(y - radius)))
    bottom = min(h, int(math.ceil(y + radius + 1)))
    if left >= right or top >= bottom:
        return False
    return bool((arr[top:bottom, left:right] < threshold).any())


def center_cross_pixels_visible(arr: Any, mark_geometry: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    center = mark_geometry.get("wavy_center") if isinstance(mark_geometry.get("wavy_center"), dict) else {}
    axis = mark_geometry.get("connector_vector") if isinstance(mark_geometry.get("connector_vector"), dict) else {}
    try:
        cx = float(center.get("x"))
        cy = float(center.get("y"))
        ux = float(axis.get("x"))
        uy = float(axis.get("y"))
    except (TypeError, ValueError):
        return ["custom_wavy_center_cross_pixel_geometry_missing"]
    if math.hypot(ux, uy) < 1e-6:
        return ["custom_wavy_center_cross_connector_axis_missing"]
    draw_width = as_float(mark_geometry.get("draw_line_width_px"), 0.0)
    line_width = max(0.75, draw_width if draw_width > 0.0 else as_float(mark_geometry.get("line_width_px"), 1.0))
    radius = max(1, int(math.ceil(line_width * 1.35)))
    required = max(
        4.2,
        min(
            as_float(mark_geometry.get("center_cross_required_cross_past_px"), 5.5) * 0.52,
            line_width * 4.2 + 1.5,
        ),
    )
    checks = [
        ("center", 0.0),
        ("before_inner", -max(1.6, line_width * 1.8)),
        ("after_inner", max(1.6, line_width * 1.8)),
        ("before_outer", -required),
        ("after_outer", required),
    ]
    missing = [
        name
        for name, offset in checks
        if not dark_near(arr, cx + ux * offset, cy + uy * offset, radius=radius, threshold=210)
    ]
    if "center" in missing:
        reasons.append("custom_wavy_center_cross_missing_center_pixel")
    before_missing = {"before_inner", "before_outer"}.issubset(set(missing))
    after_missing = {"after_inner", "after_outer"}.issubset(set(missing))
    if before_missing:
        reasons.append("custom_wavy_center_cross_missing_before_center_pixels")
    if after_missing:
        reasons.append("custom_wavy_center_cross_missing_after_center_pixels")
    return reasons


def terminal_pixel_connection_reasons(
    *,
    image_path: Path,
    quality: dict[str, Any],
    mode: str,
    mark_geometry: dict[str, Any],
) -> list[str]:
    if mode not in {"cut", "wavy"}:
        return []
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return ["pixel_connection_audit_unavailable"]

    if not image_path.exists():
        return ["image_file_missing_for_pixel_connection_audit"]
    try:
        arr = np.asarray(Image.open(image_path).convert("L"))
    except Exception:
        return ["image_file_unreadable_for_pixel_connection_audit"]

    endpoint = quality.get("attachment_endpoint") if isinstance(quality.get("attachment_endpoint"), dict) else {}
    image_width = as_float(quality.get("image_width"), 0.0)
    image_height = as_float(quality.get("image_height"), 0.0)
    if image_width <= 0 or image_height <= 0:
        return ["pixel_connection_missing_image_dimensions"]
    ex = as_float(endpoint.get("x"), -1.0) * image_width
    ey = as_float(endpoint.get("y"), -1.0) * image_height
    axis = mark_geometry.get("connector_vector") if isinstance(mark_geometry.get("connector_vector"), dict) else {}
    ux = as_float(axis.get("x"), 0.0)
    uy = as_float(axis.get("y"), 0.0)
    if math.hypot(ux, uy) < 1e-6 or ex < 0 or ey < 0:
        return ["pixel_connection_missing_endpoint_or_connector_vector"]

    line_width = max(1, int(round(as_float(mark_geometry.get("line_width_px"), 1.0))))
    radius = max(2, line_width + 1)
    overlap = max(5.0, as_float(mark_geometry.get("mark_connector_overlap_px"), 0.0))
    reasons: list[str] = []

    geometry = str(quality.get("attachment_render_geometry") or "").strip()
    if geometry == "rdkit_moldraw2d_terminal_perpendicular_cut_bar":
        if not str(mark_geometry.get("terminal_mark_render_source") or "").startswith("rdkit_moldraw2d_"):
            reasons.append("terminal_mark_not_renderer_native")
        if not dark_near(arr, ex, ey, radius + 1):
            reasons.append("terminal_endpoint_has_no_ink")
        mark_center = mark_geometry.get("mark_center") if isinstance(mark_geometry.get("mark_center"), dict) else {}
        mx = as_float(mark_center.get("x"), ex)
        my = as_float(mark_center.get("y"), ey)
        if math.hypot(mx - ex, my - ey) > 2.5:
            reasons.append("terminal_mark_center_not_at_connector_endpoint")
        if not dark_near(arr, mx, my, radius + 2):
            reasons.append("terminal_mark_center_has_no_ink")
        return reasons

    if mode == "wavy" and geometry == "custom_markush_attachment_perpendicular_wavy":
        if mark_geometry.get("not_rdkit_stereo_wavy") is not True:
            reasons.append("custom_wavy_allows_rdkit_stereo")
        if str(mark_geometry.get("terminal_mark_render_source") or "") != "custom_moldraw2d_image_primitive_terminal_perpendicular_wavy":
            reasons.append("custom_wavy_render_source_not_declared")
        if mark_geometry.get("line_width_matches_native_bonds") is not True:
            reasons.append("custom_wavy_line_width_not_native_bond_width")
        measured_width = as_float(mark_geometry.get("measured_native_line_width_px"), 0.0)
        draw_width = as_float(mark_geometry.get("draw_line_width_px"), 0.0)
        if measured_width <= 0.0 or draw_width <= 0.0:
            reasons.append("custom_wavy_line_width_measurement_missing")
        elif draw_width > max(1.15, measured_width * 1.02):
            reasons.append("custom_wavy_draw_line_width_too_thick")
        if str(mark_geometry.get("line_width_source") or "") not in {
            "rdkit_native_bond_line_width",
            "final_rendered_native_attachment_bond_width",
        }:
            reasons.append("custom_wavy_line_width_source_not_native")
        if not str(mark_geometry.get("depiction_profile") or ""):
            reasons.append("custom_wavy_missing_depiction_profile")
        font_range = mark_geometry.get("atom_font_size_range_px") if isinstance(mark_geometry.get("atom_font_size_range_px"), list) else []
        if len(font_range) != 2 or as_float(font_range[0], 0.0) < 10.0 or as_float(font_range[1], 0.0) < as_float(font_range[0], 0.0):
            reasons.append("custom_wavy_invalid_atom_font_size_profile")
        if str(mark_geometry.get("wavy_cut_style") or "") != "patent_terminal_perpendicular_wavy_mark":
            reasons.append("custom_wavy_style_not_terminal_perpendicular")
        straight = as_float(mark_geometry.get("straight_connector_length_px"), 0.0)
        wavy_length = as_float(mark_geometry.get("wavy_length_px") or mark_geometry.get("wavy_bond_length_px"), 0.0)
        draw_start = mark_geometry.get("connector_draw_start") if isinstance(mark_geometry.get("connector_draw_start"), dict) else {}
        erase_start = mark_geometry.get("native_connector_erase_start") if isinstance(mark_geometry.get("native_connector_erase_start"), dict) else {}
        if not draw_start or not erase_start:
            reasons.append("custom_wavy_missing_connector_draw_start")
        else:
            draw_erase_gap = math.hypot(
                as_float(draw_start.get("x"), -999.0) - as_float(erase_start.get("x"), 999.0),
                as_float(draw_start.get("y"), -999.0) - as_float(erase_start.get("y"), 999.0),
            )
            if draw_erase_gap > max(0.95, as_float(mark_geometry.get("draw_line_width_px"), 1.0) * 0.65):
                reasons.append("custom_wavy_connector_draw_start_not_erased_start")
        clearance = mark_geometry.get("visible_anchor_label_clearance") if isinstance(mark_geometry.get("visible_anchor_label_clearance"), dict) else {}
        start_policy = mark_geometry.get("visible_connector_start_policy") if isinstance(mark_geometry.get("visible_connector_start_policy"), dict) else {}
        anchor_depiction_mode = str(quality.get("attachment_anchor_depiction_mode") or mark_geometry.get("anchor_depiction_mode") or "")
        anchor_label_is_visible_text = (
            quality.get("attachment_anchor_label_is_visible_text") is True
            or mark_geometry.get("anchor_label_is_visible_text") is True
        )
        if anchor_depiction_mode == "visible_carbon_attachment_label" or anchor_label_is_visible_text:
            if start_policy.get("protects_visible_atom_label") is not True:
                reasons.append("custom_wavy_visible_anchor_label_not_protected")
            if mark_geometry.get("anchor_label_is_visible_text") is not True:
                reasons.append("custom_wavy_visible_anchor_label_detection_missing")
        if anchor_depiction_mode == "implicit_carbon_skeleton_endpoint" and anchor_label_is_visible_text:
            reasons.append("custom_wavy_implicit_carbon_misdeclared_visible")
        if start_policy.get("protects_visible_atom_label") is True:
            if clearance.get("passed") is not True:
                reasons.append("custom_wavy_visible_anchor_label_clearance_failed")
            minimum_start = as_float(clearance.get("minimum_visible_start_offset_px"), 0.0)
            measured_start = as_float(clearance.get("measured_visible_start_offset_px"), 0.0)
            if measured_start + 1e-6 < minimum_start:
                reasons.append("custom_wavy_connector_starts_inside_anchor_label")
            minimum_straight = as_float(clearance.get("minimum_straight_connector_length_px"), 0.0)
            straight_for_clearance = as_float(clearance.get("straight_connector_length_px"), straight)
            if straight_for_clearance + 1e-6 < minimum_straight:
                reasons.append("custom_wavy_connector_too_short_after_anchor_label")
        if wavy_length <= 0.0:
            reasons.append("custom_wavy_missing_length")
        length_ratio = as_float(
            mark_geometry.get("wavy_length_to_connector_ratio")
            or mark_geometry.get("mark_length_to_connector_ratio"),
            0.0,
        )
        if length_ratio < 0.35:
            reasons.append("custom_wavy_mark_too_short")
        if straight <= 0.0:
            reasons.append("custom_wavy_missing_connector")
        if as_float(mark_geometry.get("wavy_cycles"), 0.0) < 2.0 or as_float(mark_geometry.get("wavy_cycles"), 0.0) > 5.8:
            reasons.append("custom_wavy_cycle_count_not_patent_terminal")
        if as_float(mark_geometry.get("wavy_axis_dot_connector_abs"), 1.0) > 0.15:
            reasons.append("custom_wavy_axis_not_perpendicular_to_connector")
        if str(mark_geometry.get("wavy_connector_intersection_style") or "") not in {"center_cross", "side_touch"}:
            reasons.append("custom_wavy_missing_or_invalid_junction_style")
        if (
            str(mark_geometry.get("wavy_connector_intersection_style") or "") == "side_touch"
            and mark_geometry.get("side_touch_generation_allowed") is not True
        ):
            reasons.append("custom_wavy_side_touch_not_allowed_in_production")
        if str(mark_geometry.get("wavy_connector_intersection_style") or "") == "center_cross":
            if mark_geometry.get("connector_crosses_wavy_center") is not True:
                reasons.append("custom_wavy_center_cross_not_declared")
            if mark_geometry.get("center_cross_curve_passes_connector") is not True:
                reasons.append("custom_wavy_center_cross_curve_not_declared")
            if mark_geometry.get("center_cross_connector_is_solid_line") is not True:
                reasons.append("custom_wavy_center_cross_not_solid_connector")
            if str(mark_geometry.get("center_cross_draw_order") or "") != "wavy_then_solid_connector":
                reasons.append("custom_wavy_center_cross_draw_order_invalid")
            if as_float(mark_geometry.get("connector_wavy_center_distance_px"), 999.0) > max(1.5, line_width + 0.8):
                reasons.append("custom_wavy_connector_not_through_center")
            reasons.extend(center_cross_pixels_visible(arr, mark_geometry))
        if str(mark_geometry.get("wavy_connector_intersection_style") or "") == "side_touch":
            if mark_geometry.get("connector_crosses_wavy_center") is True:
                reasons.append("custom_wavy_side_touch_declares_center_cross")
            if mark_geometry.get("connector_contact_is_terminal_wavy_endpoint") is not True:
                reasons.append("custom_wavy_side_touch_endpoint_contact_not_declared")
            half_length = max(1.0, 0.5 * wavy_length)
            if abs(as_float(mark_geometry.get("connector_wavy_center_distance_px"), -999.0) - half_length) > max(2.5, as_float(mark_geometry.get("wavy_amplitude_px"), 0.0) * 1.2):
                reasons.append("custom_wavy_side_touch_center_distance_invalid")

        terminal_points = mark_geometry.get("terminal_mark_sampled_points_px")
        if not isinstance(terminal_points, list) or len(terminal_points) < 8:
            reasons.append("custom_wavy_missing_terminal_samples")
        else:
            visible = 0
            min_endpoint_distance = None
            for point in terminal_points:
                if not isinstance(point, dict):
                    continue
                px = as_float(point.get("x"), -1.0)
                py = as_float(point.get("y"), -1.0)
                distance = math.hypot(px - ex, py - ey)
                min_endpoint_distance = distance if min_endpoint_distance is None else min(min_endpoint_distance, distance)
                if dark_near(arr, px, py, radius + 2):
                    visible += 1
            if visible < max(5, int(round(len(terminal_points) * 0.45))):
                reasons.append("custom_wavy_terminal_samples_have_pixel_gap")
            if str(mark_geometry.get("wavy_connector_intersection_style") or "") == "center_cross" and (
                min_endpoint_distance is None or min_endpoint_distance > max(1.25, line_width * 0.75 + 0.75)
            ):
                reasons.append("custom_wavy_center_cross_curve_not_through_connector_endpoint")
            if str(mark_geometry.get("wavy_connector_intersection_style") or "") == "side_touch" and (
                min_endpoint_distance is None or min_endpoint_distance > max(1.75, line_width + 0.75)
            ):
                reasons.append("custom_wavy_side_touch_not_at_connector_endpoint")

        connector_points = mark_geometry.get("connector_sampled_points_px")
        if isinstance(connector_points, list) and connector_points:
            visible = 0
            leading_visible = 0
            for point in connector_points:
                if not isinstance(point, dict):
                    continue
                if dark_near(arr, as_float(point.get("x"), -1.0), as_float(point.get("y"), -1.0), radius + 1):
                    visible += 1
                    if visible == leading_visible + 1:
                        leading_visible += 1
            if visible < max(1, int(round(len(connector_points) * 0.40))):
                reasons.append("custom_wavy_connector_samples_have_pixel_gap")
            if leading_visible < min(3, len(connector_points)):
                reasons.append("custom_wavy_connector_start_has_pixel_gap")
        return reasons

    if not dark_near(arr, ex, ey, radius):
        reasons.append("terminal_endpoint_has_no_ink")

    samples = max(6, int(math.ceil(overlap)))
    missing = 0
    consecutive_missing = 0
    max_consecutive_missing = 0
    for index in range(samples + 1):
        distance = overlap * index / samples
        x = ex - ux * distance
        y = ey - uy * distance
        if dark_near(arr, x, y, radius):
            consecutive_missing = 0
        else:
            missing += 1
            consecutive_missing += 1
            max_consecutive_missing = max(max_consecutive_missing, consecutive_missing)
    if missing > max(1, samples // 5) or max_consecutive_missing > 1:
        reasons.append("terminal_connector_overlap_has_pixel_gap")

    mark_center = mark_geometry.get("mark_center") if isinstance(mark_geometry.get("mark_center"), dict) else {}
    mx = as_float(mark_center.get("x"), ex)
    my = as_float(mark_center.get("y"), ey)
    if math.hypot(mx - ex, my - ey) > max(2.5, overlap * 0.35):
        reasons.append("terminal_mark_center_not_at_connector_endpoint")
    if mode == "wavy" and not dark_near(arr, mx, my, radius + 1):
        reasons.append("wavy_center_has_no_ink")
    bridge_past = as_float(mark_geometry.get("mark_connector_bridge_past_endpoint_px"), 0.0)
    if mode == "wavy":
        cut_style = str(mark_geometry.get("wavy_cut_style") or "").strip()
        if cut_style != "patent_terminal_perpendicular_cut_mark":
            reasons.append("wavy_cut_style_not_patent_terminal_mark")
        if bridge_past > 0.75:
            reasons.append("wavy_connector_crosses_cut_center_like_stereo_bond")
    return reasons


def atom_by_index(quality: dict[str, Any]) -> dict[int, dict[str, Any]]:
    atoms = quality.get("atom_coordinates") if isinstance(quality.get("atom_coordinates"), list) else []
    output = {}
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        try:
            output[int(atom.get("atom_index"))] = atom
        except (TypeError, ValueError):
            continue
    return output


def outward_alignment(quality: dict[str, Any]) -> float | None:
    try:
        anchor_index = int(quality.get("attachment_anchor_index"))
        dummy_index = int(quality.get("attachment_dummy_index"))
    except (TypeError, ValueError):
        return None
    atoms = atom_by_index(quality)
    anchor = atoms.get(anchor_index)
    dummy = atoms.get(dummy_index)
    if not anchor or not dummy:
        return None
    neighbors = []
    bonds = quality.get("bonds") if isinstance(quality.get("bonds"), list) else []
    for bond in bonds:
        if not isinstance(bond, dict):
            continue
        try:
            begin = int(bond.get("begin_atom_index"))
            end = int(bond.get("end_atom_index"))
        except (TypeError, ValueError):
            continue
        other = None
        if begin == anchor_index and end != dummy_index:
            other = end
        elif end == anchor_index and begin != dummy_index:
            other = begin
        if other is not None and other in atoms:
            neighbors.append(atoms[other])
    if not neighbors:
        return None
    anchor_xy = (float(anchor["x"]), float(anchor["y"]))
    dummy_xy = (float(dummy["x"]), float(dummy["y"]))
    neighbor_x = sum(float(atom["x"]) for atom in neighbors) / len(neighbors)
    neighbor_y = sum(float(atom["y"]) for atom in neighbors) / len(neighbors)
    connector = unit((dummy_xy[0] - anchor_xy[0], dummy_xy[1] - anchor_xy[1]))
    outward = unit((anchor_xy[0] - neighbor_x, anchor_xy[1] - neighbor_y))
    if connector is None or outward is None:
        return None
    return float(connector[0] * outward[0] + connector[1] * outward[1])


def visual_reasons(
    row: dict[str, str],
    *,
    input_csv: Path,
    min_outward_dot: float,
    min_connector_px: float,
    min_mark_px: float,
    min_line_width_px: float,
    min_short_side_px: int,
) -> list[str]:
    reasons: list[str] = []
    quality = parse_quality(row)
    mode = str(row.get("attachment_render_mode") or quality.get("attachment_render_mode") or "").strip()
    geometry = str(quality.get("attachment_render_geometry") or "").strip()
    visual_shape = str(quality.get("visual_shape") or "").strip()
    if mode == "wavy" and isinstance(quality.get("wavy_geometry"), dict):
        mark_geometry = quality["wavy_geometry"]
    else:
        mark_geometry = quality.get("fragment_mark_geometry") if isinstance(quality.get("fragment_mark_geometry"), dict) else {}
    mark_count = int(mark_geometry.get("mark_count") or 0)
    mark_label = str(mark_geometry.get("mark_label") or "").strip()
    image_width = int(as_float(row.get("image_width") or quality.get("image_width"), 0.0))
    image_height = int(as_float(row.get("image_height") or quality.get("image_height"), 0.0))
    connector_length = as_float(mark_geometry.get("connector_length_px"), 0.0)
    mark_length = as_float(mark_geometry.get("mark_length_px") or mark_geometry.get("wavy_cut_length_px"), 0.0)
    line_width = as_float(mark_geometry.get("line_width_px"), 0.0)
    effective_min_short_side_px = int(min_short_side_px)
    if geometry == "custom_markush_attachment_perpendicular_wavy" and quality.get("real_tight_crop_style") is True:
        effective_min_short_side_px = min(effective_min_short_side_px, 50)

    if geometry == "straight_connector_with_terminal_stub_cut" or visual_shape == "left_terminal_short_stub":
        reasons.append("ambiguous_unmarked_terminal_stub")
    if min(image_width, image_height) < effective_min_short_side_px:
        reasons.append("image_short_side_too_small")
    if geometry == "rdkit_moldraw2d_terminal_perpendicular_cut_bar":
        connector_min_px = min(float(min_connector_px), 22.0)
    elif geometry == "custom_markush_attachment_perpendicular_wavy":
        connector_min_px = 8.0
    elif mode in {"query_attachment", "dummy_atom"}:
        connector_min_px = min(float(min_connector_px), 22.0)
    else:
        connector_min_px = float(min_connector_px)
    if connector_length < connector_min_px:
        reasons.append("connector_too_short")
    molnextr_quality = (
        quality.get("fragment_molnextr_input_quality")
        if isinstance(quality.get("fragment_molnextr_input_quality"), dict)
        else {}
    )
    if mode in {"cut", "dummy_atom", "query_attachment"} and molnextr_quality:
        connector_at_384 = as_float(molnextr_quality.get("connector_length_px_at_384"), 0.0)
        thresholds = (
            molnextr_quality.get("thresholds")
            if isinstance(molnextr_quality.get("thresholds"), dict)
            else {}
        )
        declared_max = as_float(thresholds.get("max_connector_length_px_at_384"), 0.0)
        expected_max = {
            "cut": MAX_CUT_CONNECTOR_PX_AT_384,
            "dummy_atom": MAX_DUMMY_CONNECTOR_PX_AT_384,
            "query_attachment": MAX_QUERY_CONNECTOR_PX_AT_384,
        }[mode]
        if declared_max <= 0.0 or declared_max > expected_max:
            reasons.append("connector_max_threshold_missing_or_loose")
        if connector_at_384 > expected_max:
            reasons.append("connector_too_long_after_molnextr_resize")
    if mode in {"cut", "wavy"} and geometry != "custom_markush_attachment_perpendicular_wavy" and mark_length < float(min_mark_px):
        reasons.append("terminal_mark_too_short")
    if line_width < float(min_line_width_px):
        reasons.append("attachment_line_too_thin")
    if mode == "cut" and mark_count <= 0 and not mark_label:
        reasons.append("cut_mode_without_terminal_mark")
    if mode in {"query_attachment", "dummy_atom"} and not mark_label:
        reasons.append("missing_terminal_attachment_label")
    if (
        mode in {"query_attachment", "dummy_atom"}
        and mark_geometry.get("terminal_label_render_source") not in {"rdkit_native_atom_label", "rdkit_moldraw2d_native_atom_label"}
        and isinstance(mark_geometry.get("mark_label_position"), dict)
    ):
        label_position = mark_geometry.get("mark_label_position") or {}
        endpoint = quality.get("attachment_endpoint") if isinstance(quality.get("attachment_endpoint"), dict) else {}
        width = max(1.0, float(image_width))
        height = max(1.0, float(image_height))
        ex = as_float(endpoint.get("x"), 0.0) * width
        ey = as_float(endpoint.get("y"), 0.0) * height
        lx = as_float(label_position.get("x"), ex)
        ly = as_float(label_position.get("y"), ey)
        axis = mark_geometry.get("connector_vector") if isinstance(mark_geometry.get("connector_vector"), dict) else {}
        ux = as_float(axis.get("x"), 0.0)
        uy = as_float(axis.get("y"), 0.0)
        parallel = abs((lx - ex) * ux + (ly - ey) * uy)
        perpendicular = abs((lx - ex) * (-uy) + (ly - ey) * ux)
        if perpendicular < 4.0 and parallel < 14.0:
            reasons.append("terminal_label_aligned_with_connector")
    if mode == "wavy":
        if geometry == "custom_markush_attachment_perpendicular_wavy":
            if mark_geometry.get("not_rdkit_stereo_wavy") is not True:
                reasons.append("custom_wavy_allows_rdkit_stereo")
            if as_float(mark_geometry.get("wavy_axis_dot_connector_abs"), 1.0) > 0.15:
                reasons.append("custom_wavy_axis_not_perpendicular_to_connector")
        elif as_float(mark_geometry.get("mark_connector_overlap_px"), 0.0) < 2.0:
            reasons.append("terminal_mark_not_connected_to_connector")
        if geometry == "custom_markush_attachment_perpendicular_wavy":
            if str(mark_geometry.get("wavy_cut_style") or "") != "patent_terminal_perpendicular_wavy_mark":
                reasons.append("custom_wavy_style_not_terminal_perpendicular")
            if as_float(mark_geometry.get("wavy_cycles"), 0.0) < 2.0 or as_float(mark_geometry.get("wavy_cycles"), 0.0) > 5.8:
                reasons.append("custom_wavy_cycle_count_not_patent_terminal")
            externality = quality.get("terminal_wavy_externality") if isinstance(quality.get("terminal_wavy_externality"), dict) else {}
            if externality.get("passed") is not True:
                reasons.append("terminal_wavy_externality_failed")
        else:
            if as_float(mark_geometry.get("mark_length_to_connector_ratio"), 0.0) < 0.6:
                reasons.append("wavy_mark_too_short")
            if as_float(mark_geometry.get("mark_axis_dot_connector_abs"), 1.0) > 0.15:
                reasons.append("wavy_axis_not_perpendicular_to_connector")
    if geometry in {"straight_connector_with_terminal_perpendicular_cut_bar", "rdkit_moldraw2d_terminal_perpendicular_cut_bar"}:
        if geometry != "rdkit_moldraw2d_terminal_perpendicular_cut_bar" and as_float(mark_geometry.get("mark_connector_overlap_px"), 0.0) < 2.0:
            reasons.append("terminal_mark_not_connected_to_connector")
        if as_float(mark_geometry.get("mark_length_to_connector_ratio"), 0.0) < 0.6:
            reasons.append("cut_bar_too_short")
        if as_float(mark_geometry.get("mark_axis_dot_connector_abs"), 1.0) > 0.15:
            reasons.append("cut_bar_not_perpendicular_to_connector")

    reasons.extend(
        terminal_pixel_connection_reasons(
            image_path=image_path_for_row(input_csv, row),
            quality=quality,
            mode=mode,
            mark_geometry=mark_geometry,
        )
    )

    alignment = outward_alignment(quality)
    if alignment is None:
        reasons.append("missing_connector_outward_alignment")
    elif alignment < min_outward_dot:
        reasons.append("connector_not_outward_from_anchor")

    if quality.get("graph_consistency", {}).get("anchor_dummy_bond_present") is not True:
        reasons.append("missing_anchor_dummy_bond")
    if quality.get("quality_gates", {}).get("endpoint_in_image") is not True:
        reasons.append("endpoint_not_in_image")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether attachment fragment positives have visible, geometrically valid terminal evidence.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-clean-csv", default="")
    parser.add_argument(
        "--filter-rejected-rows",
        action="store_true",
        help=(
            "Treat rejected rows as filtered candidates when --output-clean-csv is set. "
            "The report records the rejected raw rows, and the command passes only if at least one clean row remains."
        ),
    )
    parser.add_argument("--min-outward-dot", type=float, default=0.35)
    parser.add_argument("--min-connector-px", type=float, default=40.0)
    parser.add_argument("--min-mark-px", type=float, default=34.0)
    parser.add_argument("--min-line-width-px", type=float, default=1.0)
    parser.add_argument("--min-short-side-px", type=int, default=80)
    parser.add_argument("--max-examples", type=int, default=120)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    rows, fieldnames = read_rows(input_csv)
    kept: list[dict[str, str]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    for row in rows:
        if not parse_bool(row.get("reliable_training_label")):
            continue
        quality = parse_quality(row)
        mode_counts[str(row.get("attachment_render_mode") or quality.get("attachment_render_mode") or "missing")] += 1
        reasons = visual_reasons(
            row,
            input_csv=input_csv,
            min_outward_dot=float(args.min_outward_dot),
            min_connector_px=float(args.min_connector_px),
            min_mark_px=float(args.min_mark_px),
            min_line_width_px=float(args.min_line_width_px),
            min_short_side_px=int(args.min_short_side_px),
        )
        if reasons:
            reason_counts.update(reasons)
            rejected.append(
                {
                    "source_id": row.get("source_id") or "",
                    "file_path": row.get("file_path") or "",
                    "smiles": row.get("SMILES") or row.get("smiles") or "",
                    "attachment_render_mode": row.get("attachment_render_mode") or "",
                    "endpoint_side": row.get("endpoint_side") or "",
                    "reasons": reasons,
                }
            )
        else:
            kept.append(row)

    if args.output_clean_csv:
        write_rows(Path(args.output_clean_csv), kept, fieldnames)

    filter_mode = bool(args.filter_rejected_rows and args.output_clean_csv)
    passed = (len(rejected) == 0) if not filter_mode else bool(kept)
    report = {
        "input_csv": str(input_csv),
        "output_clean_csv": str(args.output_clean_csv or ""),
        "row_count": len(rows),
        "reliable_rows": int(sum(parse_bool(row.get("reliable_training_label")) for row in rows)),
        "kept_rows": len(kept),
        "rejected_rows": len(rejected),
        "passed": passed,
        "filter_rejected_rows": filter_mode,
        "clean_csv_contains_only_visual_contract_passing_rows": filter_mode,
        "min_outward_dot": float(args.min_outward_dot),
        "min_connector_px": float(args.min_connector_px),
        "min_mark_px": float(args.min_mark_px),
        "min_line_width_px": float(args.min_line_width_px),
        "min_short_side_px": int(args.min_short_side_px),
        "mode_counts": dict(sorted(mode_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rejected_examples": rejected[: int(args.max_examples)],
        "policy": {
            "plain_unmarked_stub_is_not_trainable_positive": True,
            "connector_must_extend_outward_from_anchor": True,
            "terminal_cut_or_wavy_must_be_visible": True,
            "terminal_mark_must_be_long_enough": True,
            "attachment_line_must_not_be_too_thin": True,
            "crop_must_not_shrink_fragment_below_minimum": True,
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
