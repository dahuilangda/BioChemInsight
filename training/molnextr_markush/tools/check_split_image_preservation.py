from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

csv.field_size_limit(sys.maxsize)


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


def resolve_image_path(csv_path: Path, row: dict[str, str]) -> Path:
    raw_path = str(row.get("file_path") or row.get("image_path") or "").strip()
    if not raw_path:
        return Path("")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = csv_path.parent / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def verify_image(path: Path) -> tuple[bool, str, tuple[int, int] | None]:
    if not path.exists():
        return False, "missing", None
    if not path.is_file():
        return False, "not_file", None
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:  # noqa: BLE001 - this is a data audit boundary.
        return False, f"unreadable:{type(exc).__name__}", None
    if width <= 0 or height <= 0:
        return False, "invalid_dimensions", (int(width), int(height))
    return True, "", (int(width), int(height))


def audit_csv(csv_path: Path, *, max_examples: int, progress_every: int) -> dict[str, Any]:
    row_count = 0
    missing_path_count = 0
    missing_file_count = 0
    unreadable_count = 0
    invalid_dimension_count = 0
    unique_paths: set[str] = set()
    image_cache: dict[str, tuple[bool, str, tuple[int, int] | None]] = {}
    width_min: int | None = None
    width_max: int | None = None
    height_min: int | None = None
    height_max: int | None = None
    examples: list[dict[str, Any]] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=2):
            row_count += 1
            if progress_every > 0 and row_count % progress_every == 0:
                print(f"{csv_path}: checked {row_count} rows", file=sys.stderr, flush=True)
            image_path = resolve_image_path(csv_path, row)
            if not str(image_path):
                missing_path_count += 1
                if len(examples) < max_examples:
                    examples.append({"row_index": row_index, "source_id": row.get("source_id", ""), "issue": "missing_path"})
                continue
            path_key = str(image_path)
            unique_paths.add(path_key)
            if path_key not in image_cache:
                image_cache[path_key] = verify_image(image_path)
            ok, issue, size = image_cache[path_key]
            if ok and size is not None:
                width, height = size
                width_min = width if width_min is None else min(width_min, width)
                width_max = width if width_max is None else max(width_max, width)
                height_min = height if height_min is None else min(height_min, height)
                height_max = height if height_max is None else max(height_max, height)
                continue
            if issue == "missing":
                missing_file_count += 1
            elif issue == "invalid_dimensions":
                invalid_dimension_count += 1
            else:
                unreadable_count += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "row_index": row_index,
                        "source_id": row.get("source_id", ""),
                        "image_path": path_key,
                        "issue": issue,
                    }
                )

    return {
        "csv_path": str(csv_path),
        "row_count": row_count,
        "unique_image_paths": len(unique_paths),
        "missing_path_count": missing_path_count,
        "missing_file_count": missing_file_count,
        "unreadable_count": unreadable_count,
        "invalid_dimension_count": invalid_dimension_count,
        "dimension_summary": {
            "width_min": width_min,
            "width_max": width_max,
            "height_min": height_min,
            "height_max": height_max,
        },
        "passed": row_count > 0
        and missing_path_count == 0
        and missing_file_count == 0
        and unreadable_count == 0
        and invalid_dimension_count == 0,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that split CSV image paths are preserved and readable.")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    manifest_path = Path(args.split_manifest)
    manifest = load_json(manifest_path)
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    blockers: list[str] = []
    reports: dict[str, Any] = {}
    for key in SPLIT_OUTPUT_KEYS:
        path_text = str(outputs.get(key) or "")
        if not path_text:
            blockers.append(f"split manifest missing output path for {key}")
            continue
        csv_path = Path(path_text)
        if not csv_path.exists():
            blockers.append(f"split output does not exist for {key}: {csv_path}")
            continue
        report = audit_csv(csv_path, max_examples=int(args.max_examples), progress_every=int(args.progress_every))
        reports[key] = report
        if report.get("passed") is not True:
            blockers.append(f"{key} image preservation failed")

    report = {
        "schema_version": "split_image_preservation_v1",
        "split_manifest": str(manifest_path),
        "split_stage": manifest.get("stage"),
        "split_accepted_for_training": manifest.get("accepted_for_training") is True,
        "passed": not blockers,
        "blockers": blockers,
        "reports": reports,
        "policy": {
            "all_split_rows_must_have_image_path": True,
            "all_split_images_must_exist": True,
            "all_split_images_must_be_readable": True,
            "image_dimensions_must_be_positive": True,
            "does_not_start_training": True,
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
