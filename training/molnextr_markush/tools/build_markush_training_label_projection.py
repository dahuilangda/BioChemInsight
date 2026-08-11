from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)


MARKUSH_FIELDS = [
    "source_id",
    "source_arrow",
    "file_path",
    "SMILES",
    "smiles",
    "structure_type_bucket",
    "render_quality",
    "reliable_training_label",
]

ORDINARY_FIELDS = [
    "source_id",
    "source_arrow",
    "file_path",
    "SMILES",
    "smiles",
    "structure_type_bucket",
    "reliable_training_label",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slim_markush_quality(text: str) -> str:
    quality = json.loads(text or "{}")
    if not isinstance(quality, dict):
        quality = {}
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    slim = {
        "markush": {
            "annotation": markush.get("annotation", ""),
            "cxsmiles": markush.get("cxsmiles", ""),
            "ocr_cells": markush.get("ocr_cells", []),
        }
    }
    return json.dumps(slim, separators=(",", ":"), ensure_ascii=False)


def project_csv(input_path: Path, output_path: Path, *, role: str) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = MARKUSH_FIELDS if role == "markush" else ORDINARY_FIELDS
    rows = 0
    with input_path.open(newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            projected = {field: row.get(field, "") for field in fields}
            if role == "markush":
                projected["render_quality"] = slim_markush_quality(str(row.get("render_quality") or "{}"))
            writer.writerow(projected)
            rows += 1
    return {
        "input": str(input_path),
        "output": str(output_path),
        "parent_sha256": sha256_file(input_path),
        "projection_sha256": sha256_file(output_path),
        "row_count": rows,
        "passed": rows > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact training-label CSV projections for Markush measured runs.")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split_manifest_path = Path(args.split_manifest)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    outputs = split_manifest.get("outputs") if isinstance(split_manifest.get("outputs"), dict) else {}
    expected_hashes = split_manifest.get("output_sha256") if isinstance(split_manifest.get("output_sha256"), dict) else {}
    output_dir = Path(args.output_dir)

    specs = {
        "markush_train_csv": ("markush_label_csv", "markush_train.csv", "markush"),
        "markush_calibration_csv": ("eval_markush_label_csv", "markush_calibration.csv", "markush"),
        "ordinary_train_csv": ("negative_csv", "ordinary_train.csv", "ordinary"),
        "ordinary_calibration_csv": ("eval_negative_csv", "ordinary_calibration.csv", "ordinary"),
    }
    projections: dict[str, Any] = {}
    blockers: list[str] = []
    for manifest_name, (training_arg, filename, role) in specs.items():
        input_path = Path(str(outputs.get(manifest_name) or ""))
        if not input_path.exists():
            blockers.append(f"missing split output {manifest_name}: {input_path}")
            continue
        report = project_csv(input_path, output_dir / filename, role=role)
        report["training_arg"] = training_arg
        expected_parent = str(expected_hashes.get(manifest_name) or "")
        if expected_parent and report["parent_sha256"] != expected_parent:
            blockers.append(f"parent hash mismatch for {manifest_name}")
            report["passed"] = False
        projections[manifest_name] = report

    report = {
        "schema_version": "markush_training_label_projection_manifest_v1",
        "split_manifest": str(split_manifest_path),
        "output_dir": str(output_dir),
        "passed": not blockers and all(item.get("passed") is True for item in projections.values()),
        "blockers": blockers,
        "projections": projections,
        "policy": {
            "projection_preserves_row_count": True,
            "projection_preserves_image_paths": True,
            "projection_preserves_smiles_and_source_identity": True,
            "projection_preserves_markush_layout_cells": True,
            "projection_does_not_modify_images_coordinates_or_labels": True,
            "parent_split_hashes_must_match_current_manifest": True,
            "complete_molnextr_path_remains_frozen": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
