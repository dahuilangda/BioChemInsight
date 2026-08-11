from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.markush_layout_labels import (
    count_bucket_name as markush_count_bucket_name,
)


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def read_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            raw_path = str(row.get("file_path") or row.get("image_path") or "").strip()
            if raw_path:
                image_path = Path(raw_path)
                if not image_path.is_absolute():
                    row["file_path"] = str((csv_path.parent / image_path).resolve())
            rows.append(row)
    return rows, fieldnames


def iter_rows(path: str | Path) -> Iterator[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def csv_fieldnames(path: str | Path) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def json_string_value(text: str, key: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', str(text or ""))
    if not match:
        return ""
    try:
        return str(json.loads(f'"{match.group(1)}"'))
    except json.JSONDecodeError:
        return match.group(1)


def json_int_value(text: str, key: str) -> int:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+)', str(text or ""))
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


def markush_fast_quality_value(row: dict[str, str], key: str) -> str:
    return json_string_value(str(row.get("render_quality") or ""), key)


def markush_r_tag_count_bucket(row: dict[str, str]) -> str:
    fast_bucket = markush_fast_quality_value(row, "r_tag_count_bucket").strip()
    if fast_bucket in {"1", "2", "3-4", "5-8", "9+"}:
        return fast_bucket
    quality = parse_quality(row)
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    bucket = str(markush.get("r_tag_count_bucket") or "").strip()
    if bucket in {"1", "2", "3-4", "5-8", "9+"}:
        return bucket
    try:
        count = int(markush.get("r_tag_count") or 0)
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    return "9+"


def markush_trainable_variable_present(row: dict[str, str]) -> bool:
    fast_bucket = markush_fast_quality_value(row, "r_tag_count_bucket").strip()
    if fast_bucket in {"1", "2", "3-4", "5-8", "9+"}:
        return True
    render_quality_text = str(row.get("render_quality") or "")
    for key in ["r_tag_count", "annotation_matched_ocr_cell_count", "variable_anchor_count", "layout_cell_count"]:
        if json_int_value(render_quality_text, key) > 0:
            return True
    quality = parse_quality(row)
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    bucket = str(markush.get("r_tag_count_bucket") or "").strip()
    if bucket in {"1", "2", "3-4", "5-8", "9+"}:
        return True
    for key in ["r_tag_count", "annotation_matched_ocr_cell_count", "variable_anchor_count", "layout_cell_count"]:
        try:
            if int(markush.get(key) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    cells = markush.get("ocr_cells")
    return isinstance(cells, list) and len(cells) > 0


def source_record_id(row: dict[str, str]) -> str:
    if str(row.get("structure_type_bucket") or "").strip() == "markush_layout":
        fast_record = markush_fast_quality_value(row, "source_record_id").strip()
        if fast_record:
            return fast_record
    quality = parse_quality(row)
    return str(quality.get("source_record_id") or row.get("source_id") or "").strip()


def source_document_key(row: dict[str, str], *, default_dataset: str) -> str:
    if default_dataset == "markush":
        dataset = str(
            markush_fast_quality_value(row, "source_dataset") or row.get("source_arrow") or default_dataset
        ).strip()
    else:
        quality = parse_quality(row)
        dataset = str(quality.get("source_dataset") or row.get("source_arrow") or default_dataset).strip()
    record = source_record_id(row)
    if not record:
        text = "|".join(str(row.get(name) or "") for name in ["SMILES", "smiles", "file_path"])
        record = hashlib.sha1(text.encode("utf-8")).hexdigest()
    if ":" in record:
        prefix, tail = record.rsplit(":", 1)
    else:
        prefix, tail = "", record
    if ".pdf_" in tail:
        document = tail.split(".pdf_", 1)[0] + ".pdf"
    else:
        document = re.sub(r"_[0-9]+(?:_[0-9]+)+$", "", tail)
    return f"{dataset}:{prefix}:{document}" if prefix else f"{dataset}:{document}"


def source_key(row: dict[str, str], *, default_dataset: str) -> str:
    if default_dataset == "markush":
        return source_document_key(row, default_dataset=default_dataset)
    quality = parse_quality(row)
    dataset = str(quality.get("source_dataset") or row.get("source_arrow") or default_dataset).strip()
    if default_dataset == "fragment":
        record = str(row.get("source_id") or quality.get("source_record_id") or "").strip()
    else:
        record = str(quality.get("source_record_id") or row.get("source_id") or "").strip()
    if not record:
        text = "|".join(str(row.get(name) or "") for name in ["SMILES", "smiles", "file_path"])
        record = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{dataset}:{record}"


def bucket_key(row: dict[str, str], *, kind: str) -> tuple[str, str, str, str]:
    quality = parse_quality(row)
    source_dataset = str(quality.get("source_dataset") or row.get("source_arrow") or "missing").strip()
    if kind == "fragment":
        return (
            kind,
            str(row.get("endpoint_side") or "missing").strip().lower(),
            str(row.get("attachment_anchor") or "missing").strip(),
            str(row.get("attachment_render_mode") or quality.get("attachment_render_mode") or "missing").strip(),
        )
    if kind == "ordinary":
        return (kind, source_dataset, "ordinary", "negative")
    return (kind, source_dataset, "markush_r_tag_count", markush_r_tag_count_bucket(row))


def reliable_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if parse_bool(row.get("reliable_training_label"))]


def trainable_markush_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    trainable = []
    for row in rows:
        if not parse_bool(row.get("reliable_training_label")):
            continue
        if str(row.get("structure_type_bucket") or "").strip() != "markush_layout":
            continue
        if markush_variable_count(row) <= 0:
            continue
        trainable.append(row)
    return trainable


def is_trainable_row(row: dict[str, str], *, kind: str) -> bool:
    if kind == "markush":
        if not parse_bool(row.get("reliable_training_label")):
            return False
        if str(row.get("structure_type_bucket") or "").strip() != "markush_layout":
            return False
        return markush_trainable_variable_present(row)
    return parse_bool(row.get("reliable_training_label"))


def collect_source_group_index(path: str | Path, *, kind: str) -> tuple[list[str], dict[str, dict[str, Any]], int, int]:
    fieldnames = csv_fieldnames(path)
    groups: dict[str, dict[str, Any]] = {}
    raw_rows = 0
    trainable_rows = 0
    for row in iter_rows(path):
        raw_rows += 1
        raw_path = str(row.get("file_path") or row.get("image_path") or "").strip()
        if raw_path:
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                row["file_path"] = str((Path(path).parent / image_path).resolve())
        if not is_trainable_row(row, kind=kind):
            continue
        trainable_rows += 1
        key = source_key(row, default_dataset=kind)
        bucket = bucket_key(row, kind=kind)
        entry = groups.setdefault(key, {"count": 0, "bucket": bucket})
        entry["count"] = int(entry["count"]) + 1
    return fieldnames, groups, raw_rows, trainable_rows


def split_source_group_index(
    groups: dict[str, dict[str, Any]],
    *,
    kind: str,
    trainable_rows: int,
    calibration_fraction: float,
    min_calibration_per_bucket: int,
    min_calibration_rows_by_bucket: dict[str, int] | None = None,
    rng: random.Random,
) -> tuple[set[str], set[str], dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for key, entry in groups.items():
        buckets[entry["bucket"]].append(key)

    train_keys: set[str] = set()
    calibration_keys: set[str] = set()
    fraction = min(max(float(calibration_fraction), 0.0), 0.8)
    min_calibration_rows_by_bucket = min_calibration_rows_by_bucket or {}
    for bucket, keys in buckets.items():
        rng.shuffle(keys)
        if len(keys) <= 1:
            train_keys.update(keys)
            continue
        bucket_name = bucket[-1]
        bucket_minimum = int(min_calibration_rows_by_bucket.get(bucket_name) or 0)
        calibration_count = max(int(min_calibration_per_bucket), int(round(len(keys) * fraction)), bucket_minimum)
        calibration_count = min(calibration_count, len(keys) - 1)
        calibration_keys.update(keys[:calibration_count])
        train_keys.update(keys[calibration_count:])

    train_rows = sum(int(groups[key]["count"]) for key in train_keys)
    calibration_rows = sum(int(groups[key]["count"]) for key in calibration_keys)
    overlap = sorted(train_keys & calibration_keys)
    report = {
        "kind": kind,
        "input_rows": int(trainable_rows),
        "source_groups": len(groups),
        "train_rows": int(train_rows),
        "calibration_rows": int(calibration_rows),
        "train_source_groups": len(train_keys),
        "calibration_source_groups": len(calibration_keys),
        "source_group_overlap": len(overlap),
        "bucket_count": len(buckets),
        "bucket_sizes": {
            "|".join(key): len(value) for key, value in sorted(buckets.items(), key=lambda item: str(item[0]))
        },
        "min_calibration_rows_by_bucket": {
            str(key): int(value) for key, value in sorted(min_calibration_rows_by_bucket.items())
        },
        "streaming_split": True,
    }
    return train_keys, calibration_keys, report


def write_streaming_split_rows(
    input_csv: str | Path,
    *,
    kind: str,
    fieldnames: list[str],
    train_keys: set[str],
    calibration_keys: set[str],
    train_csv: Path,
    calibration_csv: Path,
) -> dict[str, Any]:
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    train_counts: Counter[str] = Counter()
    calibration_counts: Counter[str] = Counter()
    train_aux_counts: dict[str, Counter[str]] = defaultdict(Counter)
    calibration_aux_counts: dict[str, Counter[str]] = defaultdict(Counter)
    with train_csv.open("w", newline="", encoding="utf-8") as train_handle, calibration_csv.open(
        "w", newline="", encoding="utf-8"
    ) as calibration_handle:
        train_writer = csv.DictWriter(train_handle, fieldnames=fieldnames, extrasaction="ignore")
        calibration_writer = csv.DictWriter(calibration_handle, fieldnames=fieldnames, extrasaction="ignore")
        train_writer.writeheader()
        calibration_writer.writeheader()
        for row in iter_rows(input_csv):
            raw_path = str(row.get("file_path") or row.get("image_path") or "").strip()
            if raw_path:
                image_path = Path(raw_path)
                if not image_path.is_absolute():
                    row["file_path"] = str((Path(input_csv).parent / image_path).resolve())
            if not is_trainable_row(row, kind=kind):
                continue
            key = source_key(row, default_dataset=kind)
            if key in calibration_keys:
                calibration_writer.writerow({field: row.get(field, "") for field in fieldnames})
                calibration_counts.update(["rows"])
                if kind == "fragment":
                    calibration_aux_counts["endpoint_side"].update([str(row.get("endpoint_side") or "missing")])
                    calibration_aux_counts["attachment_render_mode"].update(
                        [str(row.get("attachment_render_mode") or "missing")]
                    )
            else:
                train_writer.writerow({field: row.get(field, "") for field in fieldnames})
                train_counts.update(["rows"])
                if kind == "fragment":
                    train_aux_counts["endpoint_side"].update([str(row.get("endpoint_side") or "missing")])
                    train_aux_counts["attachment_render_mode"].update([str(row.get("attachment_render_mode") or "missing")])
    return {
        "train_rows_written": int(train_counts["rows"]),
        "calibration_rows_written": int(calibration_counts["rows"]),
        "train_side_counts": dict(sorted(train_aux_counts["endpoint_side"].items())),
        "calibration_side_counts": dict(sorted(calibration_aux_counts["endpoint_side"].items())),
        "train_mode_counts": dict(sorted(train_aux_counts["attachment_render_mode"].items())),
        "calibration_mode_counts": dict(sorted(calibration_aux_counts["attachment_render_mode"].items())),
    }


def split_by_source(
    rows: list[dict[str, str]],
    *,
    kind: str,
    calibration_fraction: float,
    min_calibration_per_bucket: int,
    min_calibration_rows_by_bucket: dict[str, int] | None = None,
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[source_key(row, default_dataset=kind)].append(row)

    buckets: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    representative: dict[str, dict[str, str]] = {}
    for key, group_rows in groups.items():
        representative[key] = group_rows[0]
        buckets[bucket_key(group_rows[0], kind=kind)].append(key)

    train_keys: set[str] = set()
    calibration_keys: set[str] = set()
    fraction = min(max(float(calibration_fraction), 0.0), 0.8)
    min_calibration_rows_by_bucket = min_calibration_rows_by_bucket or {}
    for bucket, keys in buckets.items():
        rng.shuffle(keys)
        if len(keys) <= 1:
            train_keys.update(keys)
            continue
        bucket_name = bucket[-1]
        bucket_minimum = int(min_calibration_rows_by_bucket.get(bucket_name) or 0)
        calibration_count = max(int(min_calibration_per_bucket), int(round(len(keys) * fraction)), bucket_minimum)
        calibration_count = min(calibration_count, len(keys) - 1)
        calibration_keys.update(keys[:calibration_count])
        train_keys.update(keys[calibration_count:])

    train_rows: list[dict[str, str]] = []
    calibration_rows: list[dict[str, str]] = []
    for key, group_rows in groups.items():
        if key in calibration_keys:
            calibration_rows.extend(group_rows)
        else:
            train_rows.extend(group_rows)
    rng.shuffle(train_rows)
    rng.shuffle(calibration_rows)
    overlap = sorted(train_keys & calibration_keys)
    report = {
        "kind": kind,
        "input_rows": len(rows),
        "source_groups": len(groups),
        "train_rows": len(train_rows),
        "calibration_rows": len(calibration_rows),
        "train_source_groups": len(train_keys),
        "calibration_source_groups": len(calibration_keys),
        "source_group_overlap": len(overlap),
        "bucket_count": len(buckets),
        "bucket_sizes": {
            "|".join(key): len(value) for key, value in sorted(buckets.items(), key=lambda item: str(item[0]))
        },
        "min_calibration_rows_by_bucket": {
            str(key): int(value) for key, value in sorted(min_calibration_rows_by_bucket.items())
        },
    }
    return train_rows, calibration_rows, report


def count_values(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(column) or "missing") for row in rows).items()))


def load_markush_calibration_targets(path: str, *, markush_row_count: int, calibration_fraction: float) -> dict[str, Any]:
    if not path:
        return {"path": "", "targets": {}, "blockers": []}
    decision = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decision, dict):
        raise ValueError("--markush-standards-decision must contain a JSON object")
    blockers: list[str] = []
    if decision.get("accepted") is not True:
        blockers.append("standards decision is not accepted")
    if str(decision.get("decision") or "") != "source_capacity_proportional_30k":
        blockers.append("standards decision is not source_capacity_proportional_30k")
    policy = decision.get("policy") if isinstance(decision.get("policy"), dict) else {}
    if policy.get("does_not_start_expert_training") is not True:
        blockers.append("standards decision must declare does_not_start_expert_training=true")
    raw_targets = decision.get("candidate_targets_by_bucket") if isinstance(decision.get("candidate_targets_by_bucket"), dict) else {}
    candidate_targets = {bucket: int(raw_targets.get(bucket) or 0) for bucket in ["1", "2", "3-4", "5-8", "9+"]}
    target_total = sum(candidate_targets.values())
    calibration_total = int(round(int(markush_row_count) * min(max(float(calibration_fraction), 0.0), 0.8)))
    targets = {
        bucket: int(math.floor(count * float(calibration_total) / float(target_total)))
        for bucket, count in candidate_targets.items()
        if target_total > 0
    }
    return {
        "path": str(path),
        "targets": targets,
        "candidate_targets_by_bucket": candidate_targets,
        "target_total": int(target_total),
        "calibration_total_estimate": int(calibration_total),
        "blockers": blockers,
        "policy": {
            "standards_decision_sets_markush_calibration_minimums_only": True,
            "does_not_accept_training_data": True,
            "does_not_start_expert_training": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed train/calibration CSV splits for formal sidecar training.")
    parser.add_argument("--fragment-csv", required=True)
    parser.add_argument("--ordinary-csv", required=True)
    parser.add_argument("--markush-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--ordinary-calibration-fraction", type=float, default=0.2)
    parser.add_argument("--markush-calibration-fraction", type=float, default=0.2)
    parser.add_argument("--min-calibration-per-bucket", type=int, default=1)
    parser.add_argument("--markush-standards-decision", default="")
    parser.add_argument("--seed", type=int, default=2026062102)
    parser.add_argument(
        "--stage",
        choices=["formal", "candidate"],
        default="formal",
        help="Use candidate for repaired or not-yet-accepted data splits that must not satisfy training readiness.",
    )
    parser.add_argument("--markush-acceptance-report", default="")
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    fragment_fields, fragment_groups, _fragment_raw_rows, fragment_row_count = collect_source_group_index(
        args.fragment_csv,
        kind="fragment",
    )
    ordinary_fields, ordinary_groups, _ordinary_raw_rows, ordinary_row_count = collect_source_group_index(
        args.ordinary_csv,
        kind="ordinary",
    )
    markush_fields, markush_groups, _markush_raw_rows, markush_row_count = collect_source_group_index(
        args.markush_csv,
        kind="markush",
    )
    markush_calibration_targets = load_markush_calibration_targets(
        args.markush_standards_decision,
        markush_row_count=markush_row_count,
        calibration_fraction=float(args.markush_calibration_fraction),
    )
    if markush_calibration_targets["blockers"]:
        raise ValueError("; ".join(str(item) for item in markush_calibration_targets["blockers"]))

    fragment_train_keys, fragment_calib_keys, fragment_report = split_source_group_index(
        fragment_groups,
        kind="fragment",
        trainable_rows=fragment_row_count,
        calibration_fraction=float(args.calibration_fraction),
        min_calibration_per_bucket=int(args.min_calibration_per_bucket),
        rng=rng,
    )
    ordinary_train_keys, ordinary_calib_keys, ordinary_report = split_source_group_index(
        ordinary_groups,
        kind="ordinary",
        trainable_rows=ordinary_row_count,
        calibration_fraction=float(args.ordinary_calibration_fraction),
        min_calibration_per_bucket=int(args.min_calibration_per_bucket),
        rng=rng,
    )
    markush_train_keys, markush_calib_keys, markush_report = split_source_group_index(
        markush_groups,
        kind="markush",
        trainable_rows=markush_row_count,
        calibration_fraction=float(args.markush_calibration_fraction),
        min_calibration_per_bucket=int(args.min_calibration_per_bucket),
        min_calibration_rows_by_bucket=markush_calibration_targets["targets"],
        rng=rng,
    )
    markush_acceptance_report = {}
    if args.markush_acceptance_report:
        parsed = json.loads(Path(args.markush_acceptance_report).read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("--markush-acceptance-report must contain a JSON object")
        markush_acceptance_report = parsed
    markush_formal_split_accepted = (
        markush_acceptance_report.get("accepted_for_formal_split") is True
        or markush_acceptance_report.get("training_allowed") is True
        or markush_acceptance_report.get("accepted") is True
    )

    output_dir = Path(args.output_dir)
    paths = {
        "fragment_train_csv": output_dir / "fragment_train.csv",
        "fragment_calibration_csv": output_dir / "fragment_calibration.csv",
        "ordinary_train_csv": output_dir / "ordinary_train.csv",
        "ordinary_calibration_csv": output_dir / "ordinary_calibration.csv",
        "markush_train_csv": output_dir / "markush_train.csv",
        "markush_calibration_csv": output_dir / "markush_calibration.csv",
    }
    fragment_write_report = write_streaming_split_rows(
        args.fragment_csv,
        kind="fragment",
        fieldnames=fragment_fields,
        train_keys=fragment_train_keys,
        calibration_keys=fragment_calib_keys,
        train_csv=paths["fragment_train_csv"],
        calibration_csv=paths["fragment_calibration_csv"],
    )
    ordinary_write_report = write_streaming_split_rows(
        args.ordinary_csv,
        kind="ordinary",
        fieldnames=ordinary_fields,
        train_keys=ordinary_train_keys,
        calibration_keys=ordinary_calib_keys,
        train_csv=paths["ordinary_train_csv"],
        calibration_csv=paths["ordinary_calibration_csv"],
    )
    markush_write_report = write_streaming_split_rows(
        args.markush_csv,
        kind="markush",
        fieldnames=markush_fields,
        train_keys=markush_train_keys,
        calibration_keys=markush_calib_keys,
        train_csv=paths["markush_train_csv"],
        calibration_csv=paths["markush_calibration_csv"],
    )

    report = {
        "seed": int(args.seed),
        "stage": str(args.stage),
        "accepted_for_training": str(args.stage) == "formal"
        and (not markush_acceptance_report or markush_formal_split_accepted),
        "split_policy": "source_group_disjoint_stratified_by_kind_side_anchor_mode_or_dataset_markush_r_tag_count_bucket",
        "policy": {
            "complete_molecules_remain_on_original_molnextr_path": True,
            "fragment_and_markush_rows_are_expert_routed_only": True,
            "candidate_stage_does_not_allow_training": str(args.stage) == "candidate",
            "markush_acceptance_report_required_for_repaired_data": bool(args.markush_acceptance_report),
        },
        "purpose": {
            "fragment_attachment_expert": "train on fragment_train + ordinary_train; select confidence thresholds on fragment_calibration + ordinary_calibration",
            "markush_layout_expert": "kept as separate train/calibration inputs for the Markush layout expert; consumed by train_markush_layout_expert.py, not by fragment attachment expert training",
            "complete_molecule_path": "ordinary rows are negatives only; complete molecules continue to bypass sidecar unless calibrated router explicitly accepts a branch",
        },
        "inputs": {
            "fragment_csv": str(args.fragment_csv),
            "ordinary_csv": str(args.ordinary_csv),
            "markush_csv": str(args.markush_csv),
        },
        "acceptance_inputs": {
            "markush_acceptance_report": str(args.markush_acceptance_report or ""),
            "markush_standards_decision": str(args.markush_standards_decision or ""),
            "markush_training_allowed": markush_acceptance_report.get("training_allowed")
            if markush_acceptance_report
            else None,
            "markush_accepted": markush_acceptance_report.get("accepted") if markush_acceptance_report else None,
            "markush_accepted_for_formal_split": markush_acceptance_report.get("accepted_for_formal_split")
            if markush_acceptance_report
            else None,
            "markush_formal_training_start_allowed": markush_acceptance_report.get("formal_training_start_allowed")
            if markush_acceptance_report
            else None,
            "markush_calibration_targets": markush_calibration_targets,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "output_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "fragment": fragment_report | fragment_write_report,
        "ordinary": ordinary_report | ordinary_write_report,
        "markush": markush_report | markush_write_report,
        "markush_filter_policy": {
            "reliable_training_label_required": True,
            "structure_type_bucket_required": "markush_layout",
            "annotation_matched_ocr_cell_count_must_be_positive": True,
            "label_source": "render_quality.markush.annotation <r> labels matched to render_quality.markush.ocr_cells text",
            "split_stratification_bucket": "render_quality.markush.r_tag_count_bucket",
            "matches_train_markush_layout_expert_positive_filter": True,
        },
        "formal_endpoint_sidecar_command_inputs": {
            "positive_label_csv": str(paths["fragment_train_csv"]),
            "negative_csv": str(paths["ordinary_train_csv"]),
            "eval_positive_label_csv": str(paths["fragment_calibration_csv"]),
            "eval_negative_csv": str(paths["ordinary_calibration_csv"]),
        },
    }
    report_path = output_dir / "formal_sidecar_split_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
