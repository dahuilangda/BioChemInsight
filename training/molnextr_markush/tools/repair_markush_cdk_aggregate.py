from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


UNIQUE_KEYS = [
    "source_id",
    "source_record_id",
    "source_document_key",
    "candidate_plan_index",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def candidate_plan_index(row: dict[str, str]) -> str:
    quality = parse_quality(row)
    value = quality.get("candidate_plan_index")
    if value is None or str(value).strip() == "":
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip()


def row_keys(row: dict[str, str]) -> dict[str, str]:
    quality = parse_quality(row)
    return {
        "source_id": str(row.get("source_id") or "").strip(),
        "source_record_id": str(quality.get("source_record_id") or "").strip(),
        "source_document_key": str(quality.get("source_document_key") or "").strip(),
        "candidate_plan_index": candidate_plan_index(row),
    }


def row_image_path(row: dict[str, str], csv_path: Path) -> Path:
    raw = str(row.get("file_path") or row.get("image_path") or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    return path if path.is_absolute() else csv_path.parent / path


def markush_bucket(row: dict[str, str]) -> str:
    quality = parse_quality(row)
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    bucket = str(markush.get("r_tag_count_bucket") or "").strip()
    if bucket:
        return bucket
    try:
        count = int(markush.get("r_tag_count") or 0)
    except (TypeError, ValueError):
        return "missing"
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
    return "missing"


def duplicate_values(rows: list[dict[str, str]], key_name: str) -> list[str]:
    values = [row_keys(row)[key_name] for row in rows]
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def missing_key_rows(rows: list[dict[str, str]], *, csv_path: Path) -> list[dict[str, Any]]:
    missing = []
    for index, row in enumerate(rows, start=2):
        keys = row_keys(row)
        absent = [name for name in UNIQUE_KEYS if not keys[name]]
        if absent:
            missing.append(
                {
                    "csv": str(csv_path),
                    "row_index": index,
                    "source_id": keys.get("source_id", ""),
                    "missing_keys": absent,
                }
            )
    return missing


def copy_image(row: dict[str, str], *, csv_path: Path, image_dir: Path, used_names: set[str]) -> tuple[dict[str, str], bool, str]:
    source = row_image_path(row, csv_path)
    if not source.exists() or not source.is_file():
        return dict(row), False, f"image missing: {source}"
    target_name = source.name
    if target_name in used_names:
        source_id = row_keys(row)["source_id"] or "row"
        target_name = f"{source_id}_{source.name}"
    target = image_dir / target_name
    if target_name in used_names:
        return dict(row), False, f"image filename collision after deterministic rename: {target_name}"
    shutil.copy2(source, target)
    used_names.add(target_name)
    output_row = dict(row)
    output_row["file_path"] = str(Path("images") / target_name)
    if "image_path" in output_row:
        output_row["image_path"] = output_row["file_path"]
    return output_row, True, ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append globally indexed recovered Markush CDK rows to an aggregate candidate without overwriting original rows."
    )
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--recovered-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="markush_layout_positive.csv")
    parser.add_argument("--max-issues", type=int, default=80)
    args = parser.parse_args()

    base_csv = Path(args.base_csv)
    recovered_csv = Path(args.recovered_csv)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_csv = output_dir / args.output_name

    base_rows = read_rows(base_csv)
    recovered_rows = read_rows(recovered_csv)
    fieldnames = list(base_rows[0].keys()) if base_rows else list(recovered_rows[0].keys()) if recovered_rows else []
    if not fieldnames:
        raise ValueError("base or recovered CSV must contain a header")

    blockers: list[str] = []
    base_missing = missing_key_rows(base_rows, csv_path=base_csv)
    recovered_missing = missing_key_rows(recovered_rows, csv_path=recovered_csv)
    if base_missing:
        blockers.append(f"base rows are missing global identity keys: {len(base_missing)}")
    if recovered_missing:
        blockers.append(f"recovered rows are missing global identity keys: {len(recovered_missing)}")
    base_duplicate_keys = {name: duplicate_values(base_rows, name) for name in UNIQUE_KEYS}
    recovered_duplicate_keys = {name: duplicate_values(recovered_rows, name) for name in UNIQUE_KEYS}
    for name, duplicates in base_duplicate_keys.items():
        if duplicates:
            blockers.append(f"base CSV contains duplicate {name}: {len(duplicates)}")
    for name, duplicates in recovered_duplicate_keys.items():
        if duplicates:
            blockers.append(f"recovered CSV contains duplicate {name}: {len(duplicates)}")

    seen: dict[str, set[str]] = {name: set() for name in UNIQUE_KEYS}
    for row in base_rows:
        keys = row_keys(row)
        for name in UNIQUE_KEYS:
            if keys[name]:
                seen[name].add(keys[name])

    selected_recovered: list[dict[str, str]] = []
    skipped_recovered: list[dict[str, Any]] = []
    for row in sorted(recovered_rows, key=lambda item: (int(candidate_plan_index(item) or 10**12), row_keys(item)["source_id"])):
        keys = row_keys(row)
        duplicate_key_names = [name for name in UNIQUE_KEYS if keys[name] in seen[name]]
        if duplicate_key_names:
            skipped_recovered.append(
                {
                    "source_id": keys["source_id"],
                    "candidate_plan_index": keys["candidate_plan_index"],
                    "duplicate_keys": duplicate_key_names,
                }
            )
            continue
        selected_recovered.append(row)
        for name in UNIQUE_KEYS:
            seen[name].add(keys[name])

    output_rows: list[dict[str, str]] = []
    image_errors: list[str] = []
    copied_images = 0
    used_image_names: set[str] = set()
    image_dir.mkdir(parents=True, exist_ok=True)
    for row, csv_path in [(row, base_csv) for row in base_rows] + [(row, recovered_csv) for row in selected_recovered]:
        output_row, copied, error = copy_image(row, csv_path=csv_path, image_dir=image_dir, used_names=used_image_names)
        if error:
            image_errors.append(error)
        if copied:
            copied_images += 1
        output_rows.append(output_row)
    if image_errors:
        blockers.append(f"image copy failures: {len(image_errors)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    final_duplicate_keys = {name: duplicate_values(output_rows, name) for name in UNIQUE_KEYS}
    for name, duplicates in final_duplicate_keys.items():
        if duplicates:
            blockers.append(f"repaired output contains duplicate {name}: {len(duplicates)}")

    bucket_counts = Counter(markush_bucket(row) for row in output_rows)
    manifest = {
        "schema_version": "markush_cdk_repaired_aggregate_v1",
        "csv": str(output_csv),
        "base_csv": str(base_csv),
        "recovered_csv": str(recovered_csv),
        "row_count": len(output_rows),
        "base_rows_preserved": len(base_rows),
        "recovered_rows_input": len(recovered_rows),
        "recovered_rows_added": len(selected_recovered),
        "recovered_rows_skipped_as_existing": len(skipped_recovered),
        "copied_images": copied_images,
        "counts": {"markush_layout": len(output_rows)},
        "counts_by_r_tag_bucket": {key: int(bucket_counts.get(key, 0)) for key in ["1", "2", "3-4", "5-8", "9+", "missing"]},
        "base_missing_key_examples": base_missing[: int(args.max_issues)],
        "recovered_missing_key_examples": recovered_missing[: int(args.max_issues)],
        "base_duplicate_keys": {name: values[: int(args.max_issues)] for name, values in base_duplicate_keys.items()},
        "recovered_duplicate_keys": {name: values[: int(args.max_issues)] for name, values in recovered_duplicate_keys.items()},
        "skipped_recovered_examples": skipped_recovered[: int(args.max_issues)],
        "image_error_examples": image_errors[: int(args.max_issues)],
        "status": "repaired_candidate_requires_schema_pose_visual_source_leak_coverage_confidence_runtime_and_model_scale_gates",
        "accepted": False,
        "rejected": False,
        "passed": not blockers,
        "blockers": blockers,
        "acceptance": {
            "accepted": False,
            "rejected": False,
            "visual_review_passed": False,
            "source_leak_check_passed": False,
            "coverage_check_passed": False,
            "confidence_gate_passed": False,
            "runtime_gate_passed": False,
            "model_scale_gate_passed": False,
            "reason": "Repair-only aggregate candidate; expert training remains blocked until all gates pass.",
        },
        "policy": {
            "preserve_original_base_rows": True,
            "append_only_missing_global_recovered_rows": True,
            "no_overwrite_by_recovered_rows": True,
            "complete_molecules_remain_on_original_molnextr_path": True,
            "fragment_and_markush_rows_are_expert_routed_only": True,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
