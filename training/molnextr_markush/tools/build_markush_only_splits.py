from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.markush_layout_labels import markush_variable_count


BUCKETS = ["1", "2", "3-4", "5-8", "9+"]


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def row_image_path(row: dict[str, str], csv_path: Path) -> Path:
    raw = str(row.get("file_path") or row.get("image_path") or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    return path if path.is_absolute() else csv_path.parent / path


def bucket_name(row: dict[str, str]) -> str:
    quality = parse_json_object(str(row.get("render_quality") or "{}"))
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    bucket = str(markush.get("r_tag_count_bucket") or "").strip()
    return bucket if bucket in BUCKETS else "missing"


def source_document_key(row: dict[str, str]) -> str:
    quality = parse_json_object(str(row.get("render_quality") or "{}"))
    for key in ["source_document_key", "source_record_id", "source_file"]:
        value = str(quality.get(key) or "").strip()
        if value:
            return value
    return str(row.get("source_id") or "").strip()


def is_trainable_markush(row: dict[str, str]) -> bool:
    return (
        str(row.get("reliable_training_label") or "").strip().lower() == "true"
        and str(row.get("structure_type_bucket") or "").strip() == "markush_layout"
        and markush_variable_count(row) > 0
    )


def make_absolute_image_row(row: dict[str, str], csv_path: Path) -> dict[str, str]:
    out = dict(row)
    image = row_image_path(row, csv_path).resolve()
    out["file_path"] = str(image)
    if "image_path" in out:
        out["image_path"] = str(image)
    return out


def split_rows(
    rows: list[dict[str, str]],
    *,
    calibration_fraction: float,
    min_calibration_per_bucket: int,
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    groups_by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_sources: set[str] = set()
    duplicate_sources = 0
    for row in rows:
        source_key = source_document_key(row)
        if source_key in seen_sources:
            duplicate_sources += 1
            continue
        seen_sources.add(source_key)
        groups_by_bucket[bucket_name(row)].append(row)

    train: list[dict[str, str]] = []
    calibration: list[dict[str, str]] = []
    bucket_report: dict[str, Any] = {}
    for bucket in BUCKETS:
        bucket_rows = list(groups_by_bucket.get(bucket, []))
        rng.shuffle(bucket_rows)
        desired = int(round(len(bucket_rows) * float(calibration_fraction)))
        if len(bucket_rows) >= int(min_calibration_per_bucket):
            desired = max(desired, int(min_calibration_per_bucket))
        desired = min(max(1 if bucket_rows else 0, desired), max(0, len(bucket_rows) - 1))
        calibration.extend(bucket_rows[:desired])
        train.extend(bucket_rows[desired:])
        bucket_report[bucket] = {
            "input_rows": len(bucket_rows),
            "train_rows": max(0, len(bucket_rows) - desired),
            "calibration_rows": desired,
        }

    rng.shuffle(train)
    rng.shuffle(calibration)
    return train, calibration, {"bucket_report": bucket_report, "duplicate_source_document_rows_rejected": duplicate_sources}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build source-document-disjoint Markush-only train/calibration splits from a strict accepted Markush CSV."
    )
    parser.add_argument("--markush-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--min-calibration-per-bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026062103)
    args = parser.parse_args()

    csv_path = Path(args.markush_csv)
    raw_rows, fieldnames = read_rows(csv_path)
    rows = [make_absolute_image_row(row, csv_path) for row in raw_rows if is_trainable_markush(row)]
    train, calibration, split_detail = split_rows(
        rows,
        calibration_fraction=float(args.calibration_fraction),
        min_calibration_per_bucket=int(args.min_calibration_per_bucket),
        rng=random.Random(int(args.seed)),
    )

    output_dir = Path(args.output_dir)
    train_csv = output_dir / "markush_train.csv"
    calibration_csv = output_dir / "markush_calibration.csv"
    write_rows(train_csv, train, fieldnames)
    write_rows(calibration_csv, calibration, fieldnames)

    train_sources = {source_document_key(row) for row in train}
    calibration_sources = {source_document_key(row) for row in calibration}
    overlap = sorted(train_sources & calibration_sources)
    blockers: list[str] = []
    if not train:
        blockers.append("markush train split is empty")
    if not calibration:
        blockers.append("markush calibration split is empty")
    if overlap:
        blockers.append(f"train/calibration source_document_key overlap is nonzero: {len(overlap)}")
    for bucket in BUCKETS:
        report = split_detail["bucket_report"].get(bucket, {})
        if int(report.get("input_rows") or 0) > 0 and int(report.get("calibration_rows") or 0) <= 0:
            blockers.append(f"bucket {bucket} has no calibration rows")
        if int(report.get("input_rows") or 0) > 1 and int(report.get("train_rows") or 0) <= 0:
            blockers.append(f"bucket {bucket} has no train rows")

    manifest = {
        "schema_version": "markush_only_split_manifest_v1",
        "stage": "capacity_maximized_measured",
        "accepted_for_training": not blockers,
        "seed": int(args.seed),
        "inputs": {"markush_csv": str(csv_path)},
        "outputs": {
            "markush_train_csv": str(train_csv),
            "markush_calibration_csv": str(calibration_csv),
        },
        "output_sha256": {
            "markush_train_csv": file_sha256(train_csv),
            "markush_calibration_csv": file_sha256(calibration_csv),
        },
        "markush": {
            "input_rows": len(raw_rows),
            "trainable_input_rows": len(rows),
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "train_bucket_counts": dict(sorted(Counter(bucket_name(row) for row in train).items())),
            "calibration_bucket_counts": dict(sorted(Counter(bucket_name(row) for row in calibration).items())),
            **split_detail,
        },
        "policy": {
            "source_document_disjoint_train_calibration": True,
            "markush_rows_are_routed_expert_branch_only": True,
            "complete_molecules_remain_on_original_molnextr_path": True,
            "does_not_mix_legacy_fragment_or_ordinary_generated_data": True,
            "positive_only_markush_layout_training": True,
            "router_negative_evidence_not_claimed": True,
            "architecture_comparison_or_deployment_claim_not_allowed_by_this_split": True,
        },
        "passed": not blockers,
        "blockers": blockers,
    }
    manifest_path = output_dir / "markush_only_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
