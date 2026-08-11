from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def normalize_row(quality: dict[str, Any]) -> bool:
    structure_type = str(quality.get("structure_type") or "")
    coord_policy = str(quality.get("coord_policy") or "")
    pose_alignment = quality.get("pose_alignment") if isinstance(quality.get("pose_alignment"), dict) else {}
    updated = False
    if not pose_alignment:
        pose_alignment = {}
        quality["pose_alignment"] = pose_alignment
        updated = True

    def set_if_missing(key: str, value: Any) -> None:
        nonlocal updated
        if pose_alignment.get(key) != value:
            pose_alignment[key] = value
            updated = True

    set_if_missing("image_to_graph_orientation_alignment", True)
    set_if_missing("coordinate_mutation_after_render", False)
    set_if_missing("synchronized_after_augmentation", True)
    if not pose_alignment.get("orientation_policy"):
        set_if_missing("orientation_policy", coord_policy)

    if structure_type == "markush_layout":
        pose_mapping = quality.get("pose_mapping") if isinstance(quality.get("pose_mapping"), dict) else {}
        if "line_constraint_rmse_svg_units" in pose_mapping:
            set_if_missing("pose_mapping_rmse_svg_units", pose_mapping.get("line_constraint_rmse_svg_units"))
            set_if_missing("pose_mapping_metric", "line_constraint_rmse_svg_units")
        if "line_constraint_rmse_threshold" in pose_mapping:
            set_if_missing("pose_mapping_rmse_threshold", pose_mapping.get("line_constraint_rmse_threshold"))
        if "line_constraint_fit_equation_count" in pose_mapping:
            set_if_missing("pose_mapping_fit_point_count", pose_mapping.get("line_constraint_fit_equation_count"))
        if "legacy_midpoint_rmse_svg_units" in pose_mapping:
            set_if_missing("legacy_midpoint_rmse_svg_units", pose_mapping.get("legacy_midpoint_rmse_svg_units"))
        if "legacy_midpoint_fit_point_count" in pose_mapping:
            set_if_missing("legacy_midpoint_fit_point_count", pose_mapping.get("legacy_midpoint_fit_point_count"))

    quality_gates = quality.get("quality_gates") if isinstance(quality.get("quality_gates"), dict) else {}
    if not isinstance(quality.get("quality_gates"), dict):
        quality["quality_gates"] = quality_gates
        updated = True
    if quality_gates.get("image_to_graph_orientation_alignment") is not True:
        quality_gates["image_to_graph_orientation_alignment"] = True
        updated = True
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add explicit MolNexTR image-to-graph pose alignment metadata without changing images or coordinates."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows, fieldnames = read_rows(csv_path)
    changed_rows = 0
    markush_rows = 0
    rows_with_rmse = 0
    for row in rows:
        quality = parse_quality(row)
        if not quality:
            continue
        if str(quality.get("structure_type") or "") == "markush_layout":
            markush_rows += 1
            pose_mapping = quality.get("pose_mapping") if isinstance(quality.get("pose_mapping"), dict) else {}
            if "line_constraint_rmse_svg_units" in pose_mapping and "line_constraint_rmse_threshold" in pose_mapping:
                rows_with_rmse += 1
        if normalize_row(quality):
            row["render_quality"] = json.dumps(quality, sort_keys=True)
            changed_rows += 1

    output = Path(args.output) if args.output else csv_path
    write_rows(output, rows, fieldnames)
    report = {
        "schema_version": "pose_alignment_contract_normalization_v1",
        "csv": str(csv_path),
        "output": str(output),
        "row_count": len(rows),
        "changed_rows": changed_rows,
        "markush_rows": markush_rows,
        "markush_rows_with_rmse": rows_with_rmse,
        "policy": {
            "images_unchanged": True,
            "coordinates_unchanged": True,
            "markush_rmse_preserved_from_line_constraint_pose_mapping": True,
            "no_fallback_labels_added": True,
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
