from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

csv.field_size_limit(sys.maxsize)

COUNT_BUCKETS = ["0", "1", "2", "3-4", "5-8", "9+"]
DEFAULT_MAX_SOURCE_DOCUMENT_BUCKET_EXAMPLES = 20


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_optional_json(path: str | Path | None, *, label: str) -> dict[str, Any]:
    if not path:
        return {
            "provided": False,
            "exists": False,
            "path": "",
            "blockers": [f"{label} report was not provided"],
        }
    report_path = Path(path)
    if not report_path.exists():
        return {
            "provided": True,
            "exists": False,
            "path": str(report_path),
            "blockers": [f"missing {label} report: {report_path}"],
        }
    report = load_json(report_path)
    report.setdefault("provided", True)
    report.setdefault("exists", True)
    report.setdefault("path", str(report_path))
    return report


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def count_bucket(value: int) -> int:
    if value <= 0:
        return 0
    if value == 1:
        return 1
    if value == 2:
        return 2
    if value <= 4:
        return 3
    if value <= 8:
        return 4
    return 5


def parse_render_quality(row: dict[str, Any]) -> dict[str, Any]:
    try:
        quality = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        quality = {}
    return quality if isinstance(quality, dict) else {}


def source_record_id(row: dict[str, Any], quality: dict[str, Any] | None = None) -> str:
    quality = quality if isinstance(quality, dict) else parse_render_quality(row)
    return str(quality.get("source_record_id") or row.get("source_id") or "").strip()


def source_document_key(row: dict[str, Any], quality: dict[str, Any] | None = None) -> str:
    quality = quality if isinstance(quality, dict) else parse_render_quality(row)
    dataset = str(quality.get("source_dataset") or row.get("source_arrow") or "missing").strip()
    record = source_record_id(row, quality)
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


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p_hat = float(successes) / float(total)
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / float(total)
    center = p_hat + z2 / (2.0 * float(total))
    margin = float(z) * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * float(total))) / float(total))
    return max(0.0, float((center - margin) / denominator))


def markush_positive_count(row: dict[str, Any], quality: dict[str, Any]) -> int | None:
    if not parse_bool(row.get("reliable_training_label")):
        return None
    if str(row.get("structure_type_bucket") or "").strip() != "markush_layout":
        return None
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    try:
        count = int(markush.get("variable_anchor_count") or 0)
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        return None
    return count


def compact_source_document_bucket_counts(
    counts: Counter[str],
    *,
    max_examples: int = DEFAULT_MAX_SOURCE_DOCUMENT_BUCKET_EXAMPLES,
) -> dict[str, Any]:
    documents = set()
    bucket_document_counts: dict[str, set[str]] = {bucket: set() for bucket in COUNT_BUCKETS[1:]}
    for key in counts:
        source_document, bucket = key.rsplit("|", 1)
        documents.add(source_document)
        if bucket in bucket_document_counts:
            bucket_document_counts[bucket].add(source_document)
    top_counts = sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))[:max_examples]
    return {
        "unique_positive_source_document_keys": int(len(documents)),
        "source_document_bucket_key_count": int(len(counts)),
        "bucket_unique_source_document_counts": {
            bucket: int(len(bucket_document_counts[bucket])) for bucket in COUNT_BUCKETS[1:]
        },
        "top_source_document_bucket_counts": [
            {"source_document_bucket": str(key), "rows": int(value)} for key, value in top_counts
        ],
        "full_source_document_bucket_counts_omitted": True,
        "max_top_source_document_bucket_counts": int(max_examples),
    }


def summarize_markush_csv(path: str | Path) -> dict[str, Any]:
    bucket_counts: Counter[str] = Counter()
    variable_count_counts: Counter[int] = Counter()
    source_ids: set[str] = set()
    source_records: set[str] = set()
    source_documents: set[str] = set()
    source_document_bucket_counts: Counter[str] = Counter()
    filtered_out = 0
    raw_row_count = 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_row_count += 1
            quality = parse_render_quality(row)
            source_id = str(row.get("source_id") or "").strip()
            if source_id:
                source_ids.add(source_id)
            source_document = source_document_key(row, quality)
            source_documents.add(source_document)
            variable_count = markush_positive_count(row, quality)
            if variable_count is None:
                filtered_out += 1
                continue
            bucket_name = COUNT_BUCKETS[int(count_bucket(variable_count))]
            variable_count_counts[variable_count] += 1
            bucket_counts[bucket_name] += 1
            source_document_bucket_counts[f"{source_document}|{bucket_name}"] += 1
            record_id = source_record_id(row, quality)
            if record_id:
                source_records.add(record_id)
    positive_rows = sum(bucket_counts.values())
    positive_bucket_counts = {name: int(bucket_counts.get(name, 0)) for name in COUNT_BUCKETS[1:]}
    return {
        "path": str(path),
        "raw_rows": int(raw_row_count),
        "positive_rows_after_training_filter": int(positive_rows),
        "filtered_out_rows": int(filtered_out),
        "count_bucket_counts": positive_bucket_counts,
        "min_positive_bucket_count": min(positive_bucket_counts.values()) if positive_bucket_counts else 0,
        "exact_variable_count_counts": {str(key): int(value) for key, value in sorted(variable_count_counts.items())},
        "unique_source_ids": int(len(source_ids)),
        "unique_source_record_ids": int(len(source_records)),
        "unique_source_document_keys": int(len(source_documents)),
        "source_document_bucket_summary": compact_source_document_bucket_counts(source_document_bucket_counts),
    }


def summarize_negative_csv(path: str | Path) -> dict[str, Any]:
    rows = read_csv(path)
    bucket_counts = Counter(str(row.get("structure_type_bucket") or "").strip() for row in rows)
    return {
        "path": str(path),
        "rows": int(len(rows)),
        "structure_type_bucket_counts": {str(key): int(value) for key, value in sorted(bucket_counts.items())},
    }


def overlap_report(train: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    train_rows = read_csv(train["path"])
    calibration_rows = read_csv(calibration["path"])
    train_source_ids = {str(row.get("source_id") or "").strip() for row in train_rows if str(row.get("source_id") or "").strip()}
    calibration_source_ids = {
        str(row.get("source_id") or "").strip()
        for row in calibration_rows
        if str(row.get("source_id") or "").strip()
    }
    train_source_records = {source_record_id(row) for row in train_rows if source_record_id(row)}
    calibration_source_records = {source_record_id(row) for row in calibration_rows if source_record_id(row)}
    train_source_documents = {source_document_key(row) for row in train_rows}
    calibration_source_documents = {source_document_key(row) for row in calibration_rows}
    source_id_overlap = sorted(train_source_ids & calibration_source_ids)
    source_record_overlap = sorted(train_source_records & calibration_source_records)
    source_document_overlap = sorted(train_source_documents & calibration_source_documents)
    return {
        "train_calibration_source_id_overlap_count": int(len(source_id_overlap)),
        "train_calibration_source_id_overlap_examples": source_id_overlap[:20],
        "train_calibration_source_record_overlap_count": int(len(source_record_overlap)),
        "train_calibration_source_record_overlap_examples": source_record_overlap[:20],
        "train_calibration_source_document_overlap_count": int(len(source_document_overlap)),
        "train_calibration_source_document_overlap_examples": source_document_overlap[:20],
    }


def confidence_evidence(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"provided": False}
    report = load_json(path)
    selected = report.get("selected_threshold")
    threshold_curve = report.get("threshold_curve") if isinstance(report.get("threshold_curve"), list) else []
    best_zero_negative_by_precision = None
    best_zero_negative_by_coverage = None
    zero_negative_rows = [row for row in threshold_curve if int(row.get("negative_accepted") or 0) == 0]
    if zero_negative_rows:
        best_zero_negative_by_precision = max(
            zero_negative_rows,
            key=lambda row: (
                float(row.get("positive_layout_precision") or 0.0),
                int(row.get("positive_accepted") or 0),
                float(row.get("positive_layout_recall") or 0.0),
            ),
        )
        best_zero_negative_by_coverage = max(
            zero_negative_rows,
            key=lambda row: (
                int(row.get("positive_accepted") or 0),
                float(row.get("positive_layout_precision_wilson_lower") or 0.0),
                float(row.get("positive_layout_precision") or 0.0),
            ),
        )
    return {
        "provided": True,
        "path": str(path),
        "deployment_allowed": report.get("deployment_allowed") is True,
        "selected_threshold": selected,
        "positive_rows": int(report.get("positive_rows") or 0),
        "negative_rows": int(report.get("negative_rows") or 0),
        "best_zero_negative_threshold": best_zero_negative_by_precision,
        "max_coverage_zero_negative_threshold": best_zero_negative_by_coverage,
    }


def deployable_zero_negative_threshold(confidence: dict[str, Any]) -> dict[str, Any]:
    selected = confidence.get("selected_threshold")
    if isinstance(selected, dict) and int(selected.get("negative_accepted") or 0) == 0:
        return selected
    max_coverage = confidence.get("max_coverage_zero_negative_threshold")
    if isinstance(max_coverage, dict):
        return max_coverage
    best_precision = confidence.get("best_zero_negative_threshold")
    return best_precision if isinstance(best_precision, dict) else {}


def markush_formal_scale_blockers(scale_report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if scale_report.get("provided") is not True:
        return ["training scale adequacy report was not provided for Markush formal training readiness"]
    if scale_report.get("exists") is False:
        return [str(item) for item in scale_report.get("blockers") or ["training scale adequacy report is missing"]]
    if scale_report.get("schema_version") == "markush_capacity_maximized_training_scale_v1":
        if scale_report.get("formal_training_candidate_allowed") is True:
            return []
        return [
            str(item)
            for item in scale_report.get("blockers")
            or ["capacity-maximized Markush scale report does not allow formal training candidate"]
        ]
    if scale_report.get("schema_version") != "training_scale_adequacy_v1":
        blockers.append("training scale adequacy schema_version is not training_scale_adequacy_v1")
    branches = scale_report.get("branches") if isinstance(scale_report.get("branches"), dict) else {}
    markush = branches.get("markush") if isinstance(branches.get("markush"), dict) else {}
    if markush.get("formal_training_candidate") is not True:
        branch_blockers = markush.get("formal_blockers")
        if isinstance(branch_blockers, list) and branch_blockers:
            blockers.extend(str(item) for item in branch_blockers)
        else:
            blockers.append("Markush training scale does not allow formal training candidate")
    return blockers


def capacity_maximized_formal_scale_allowed(scale_report: dict[str, Any]) -> bool:
    return (
        scale_report.get("provided") is True
        and scale_report.get("exists") is not False
        and scale_report.get("schema_version") == "markush_capacity_maximized_training_scale_v1"
        and scale_report.get("formal_training_candidate_allowed") is True
    )


def deployment_scale_blockers(scale_report: dict[str, Any]) -> list[str]:
    if scale_report.get("provided") is not True:
        return ["training scale adequacy report was not provided for deployment certification"]
    if scale_report.get("exists") is False:
        return [str(item) for item in scale_report.get("blockers") or ["training scale adequacy report is missing"]]
    blockers: list[str] = []
    if scale_report.get("schema_version") != "training_scale_adequacy_v1":
        blockers.append("training scale adequacy schema_version is not training_scale_adequacy_v1")
    if scale_report.get("deployment_certification_allowed") is not True:
        scale_blockers = scale_report.get("deployment_blockers")
        if isinstance(scale_blockers, list) and scale_blockers:
            blockers.extend(str(item) for item in scale_blockers)
        else:
            blockers.append("training scale adequacy does not allow deployment certification")
    return blockers


def independent_benchmark_blockers(report: dict[str, Any], *, min_rows: int) -> list[str]:
    if report.get("provided") is not True:
        return ["independent real benchmark report was not provided for deployment certification"]
    if report.get("exists") is False:
        return [str(item) for item in report.get("blockers") or ["independent real benchmark report is missing"]]
    blockers: list[str] = []
    if report.get("passed") is not True:
        blockers.append("independent real benchmark report did not pass")
    policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    if policy.get("source_disjoint_from_training") is not True:
        blockers.append("independent real benchmark is not explicitly source-disjoint from training")
    if policy.get("real_images_or_real_document_crops") is not True:
        blockers.append("independent real benchmark does not explicitly use real images or real document crops")
    row_count = int(report.get("row_count") or report.get("rows") or report.get("markush_rows") or 0)
    if row_count < int(min_rows):
        blockers.append(f"independent real benchmark rows are {row_count}; required >= {int(min_rows)}")
    report_blockers = report.get("blockers")
    if isinstance(report_blockers, list):
        blockers.extend(str(item) for item in report_blockers)
    return blockers


def enough_bucket_counts(summary: dict[str, Any], *, minimum: int) -> tuple[bool, list[str]]:
    blockers = []
    counts = summary["count_bucket_counts"]
    for bucket in COUNT_BUCKETS[1:]:
        count = int(counts.get(bucket) or 0)
        if count < minimum:
            blockers.append(f"{Path(summary['path']).name} bucket {bucket} has {count} positives; required >= {minimum}")
    return not blockers, blockers


def load_standards_bucket_targets(path: str, *, train_total: int, calibration_total: int) -> dict[str, Any]:
    if not path:
        return {
            "provided": False,
            "accepted": False,
            "decision": "",
            "train_targets_by_bucket": {},
            "calibration_targets_by_bucket": {},
            "blockers": [],
        }
    decision = load_json(path)
    blockers: list[str] = []
    if decision.get("accepted") is not True:
        blockers.append("standards decision is not accepted")
    if str(decision.get("decision") or "") != "source_capacity_proportional_30k":
        blockers.append("standards decision is not source_capacity_proportional_30k")
    policy = decision.get("policy") if isinstance(decision.get("policy"), dict) else {}
    if policy.get("does_not_start_expert_training") is not True:
        blockers.append("standards decision must not start expert training")
    raw_targets = decision.get("candidate_targets_by_bucket") if isinstance(decision.get("candidate_targets_by_bucket"), dict) else {}
    candidate_targets: dict[str, int] = {}
    for bucket in COUNT_BUCKETS[1:]:
        try:
            candidate_targets[bucket] = int(raw_targets.get(bucket) or 0)
        except (TypeError, ValueError):
            candidate_targets[bucket] = 0
        if candidate_targets[bucket] <= 0:
            blockers.append(f"standards decision target for bucket {bucket} is missing")
    target_total = sum(candidate_targets.values())
    if target_total < int(train_total) + int(calibration_total):
        blockers.append(
            f"standards decision target total {target_total} < required split total {int(train_total) + int(calibration_total)}"
        )
    train_targets: dict[str, int] = {}
    calibration_targets: dict[str, int] = {}
    if target_total > 0:
        for bucket, count in candidate_targets.items():
            train_targets[bucket] = int(math.floor(count * float(train_total) / float(target_total)))
            calibration_targets[bucket] = int(math.floor(count * float(calibration_total) / float(target_total)))
    return {
        "provided": True,
        "accepted": decision.get("accepted") is True,
        "decision": str(decision.get("decision") or ""),
        "path": str(path),
        "candidate_targets_by_bucket": candidate_targets,
        "target_total": int(target_total),
        "train_targets_by_bucket": train_targets,
        "calibration_targets_by_bucket": calibration_targets,
        "blockers": blockers,
        "policy": {
            "standards_decision_can_change_bucket_targets_only": True,
            "does_not_reduce_total_train_or_calibration_thresholds": True,
            "does_not_relax_confidence_or_deployment_gates": True,
        },
    }


def enough_bucket_targets(
    summary: dict[str, Any],
    *,
    targets_by_bucket: dict[str, int],
    split_name: str,
) -> tuple[bool, list[str]]:
    blockers = []
    counts = summary["count_bucket_counts"]
    for bucket in COUNT_BUCKETS[1:]:
        count = int(counts.get(bucket) or 0)
        target = int(targets_by_bucket.get(bucket) or 0)
        if target <= 0:
            blockers.append(f"{split_name} bucket {bucket} has no positive standards-derived target")
        elif count < target:
            blockers.append(f"{Path(summary['path']).name} bucket {bucket} has {count} positives; required >= {target}")
    return not blockers, blockers


def positive_source_document_count(summary: dict[str, Any]) -> int:
    compact = summary.get("source_document_bucket_summary")
    if isinstance(compact, dict):
        return int(compact.get("unique_positive_source_document_keys") or 0)
    return int(summary.get("unique_source_document_keys") or 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether Markush data is large and stratified enough for architecture-quality conclusions."
    )
    parser.add_argument("--markush-train-csv", required=True)
    parser.add_argument("--markush-calibration-csv", required=True)
    parser.add_argument("--ordinary-train-csv", default="")
    parser.add_argument("--ordinary-calibration-csv", default="")
    parser.add_argument("--confidence-report", default="")
    parser.add_argument("--standards-decision", default="")
    parser.add_argument("--training-scale-adequacy-report", default="")
    parser.add_argument("--independent-real-benchmark-report", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scope",
        choices=["architecture_comparison", "formal_training_candidate", "deployment_certification"],
        default="architecture_comparison",
        help="Controls the process exit code. The JSON report always contains all readiness levels.",
    )
    parser.add_argument("--min-train-positive-rows-for-architecture", type=int, default=20000)
    parser.add_argument("--min-calibration-positive-rows-for-architecture", type=int, default=10000)
    parser.add_argument("--min-train-per-count-bucket-for-architecture", type=int, default=4000)
    parser.add_argument("--min-calibration-per-count-bucket-for-architecture", type=int, default=2000)
    parser.add_argument("--min-calibration-positive-for-deployment", type=int, default=10000)
    parser.add_argument("--min-zero-negative-positive-accepted-for-deployment", type=int, default=500)
    parser.add_argument("--min-zero-negative-wilson-lower-for-deployment", type=float, default=0.70)
    parser.add_argument("--min-source-documents-for-architecture", type=int, default=200)
    parser.add_argument("--min-real-benchmark-rows-for-deployment", type=int, default=5000)
    args = parser.parse_args()

    train = summarize_markush_csv(args.markush_train_csv)
    calibration = summarize_markush_csv(args.markush_calibration_csv)
    ordinary_train = summarize_negative_csv(args.ordinary_train_csv) if args.ordinary_train_csv else {}
    ordinary_calibration = summarize_negative_csv(args.ordinary_calibration_csv) if args.ordinary_calibration_csv else {}
    confidence = confidence_evidence(args.confidence_report or None)
    training_scale = load_optional_json(args.training_scale_adequacy_report, label="training scale adequacy")
    independent_benchmark = load_optional_json(
        args.independent_real_benchmark_report,
        label="independent real benchmark",
    )
    split_overlap = overlap_report(train, calibration)
    standards_targets = load_standards_bucket_targets(
        args.standards_decision,
        train_total=int(args.min_train_positive_rows_for_architecture),
        calibration_total=int(args.min_calibration_positive_rows_for_architecture),
    )

    architecture_blockers: list[str] = []
    formal_training_blockers: list[str] = []
    deployment_blockers: list[str] = []
    smoke_warnings: list[str] = []

    if train["positive_rows_after_training_filter"] < args.min_train_positive_rows_for_architecture:
        architecture_blockers.append(
            "Markush train positives after training filter are "
            f"{train['positive_rows_after_training_filter']}; required >= {args.min_train_positive_rows_for_architecture}"
        )
    if calibration["positive_rows_after_training_filter"] < args.min_calibration_positive_rows_for_architecture:
        architecture_blockers.append(
            "Markush calibration positives after training filter are "
            f"{calibration['positive_rows_after_training_filter']}; required >= "
            f"{args.min_calibration_positive_rows_for_architecture}"
        )
    if standards_targets.get("provided") and not standards_targets.get("blockers"):
        _ok, blockers = enough_bucket_targets(
            train,
            targets_by_bucket=standards_targets["train_targets_by_bucket"],
            split_name="markush_train",
        )
        architecture_blockers.extend(blockers)
        _ok, blockers = enough_bucket_targets(
            calibration,
            targets_by_bucket=standards_targets["calibration_targets_by_bucket"],
            split_name="markush_calibration",
        )
        architecture_blockers.extend(blockers)
    else:
        if standards_targets.get("blockers"):
            architecture_blockers.extend(str(item) for item in standards_targets["blockers"])
        _ok, blockers = enough_bucket_counts(
            train,
            minimum=int(args.min_train_per_count_bucket_for_architecture),
        )
        architecture_blockers.extend(blockers)
        _ok, blockers = enough_bucket_counts(
            calibration,
            minimum=int(args.min_calibration_per_count_bucket_for_architecture),
        )
        architecture_blockers.extend(blockers)
    if split_overlap["train_calibration_source_id_overlap_count"] > 0:
        architecture_blockers.append(
            "train/calibration source_id overlap is nonzero: "
            f"{split_overlap['train_calibration_source_id_overlap_count']}"
        )
    if split_overlap["train_calibration_source_record_overlap_count"] > 0:
        architecture_blockers.append(
            "train/calibration source_record_id overlap is nonzero: "
            f"{split_overlap['train_calibration_source_record_overlap_count']}"
        )
    if split_overlap["train_calibration_source_document_overlap_count"] > 0:
        architecture_blockers.append(
            "train/calibration source-document overlap is nonzero: "
            f"{split_overlap['train_calibration_source_document_overlap_count']}"
        )
    total_source_documents = (
        positive_source_document_count(train)
        + positive_source_document_count(calibration)
        - int(split_overlap["train_calibration_source_document_overlap_count"])
    )
    if total_source_documents < args.min_source_documents_for_architecture:
        architecture_blockers.append(
            f"Markush source-document groups are {total_source_documents}; required >= "
            f"{args.min_source_documents_for_architecture} for architecture comparison"
        )

    if capacity_maximized_formal_scale_allowed(training_scale):
        formal_training_blockers.extend(
            item
            for item in architecture_blockers
            if "overlap is nonzero" in str(item) or "source-document groups are 0" in str(item)
        )
        smoke_warnings.append(
            "Formal training uses source-capacity-maximized scale; architecture comparison thresholds remain blocked."
        )
    else:
        formal_training_blockers.extend(architecture_blockers)
    formal_training_blockers.extend(markush_formal_scale_blockers(training_scale))
    if not confidence.get("provided"):
        formal_training_blockers.append("no calibrated confidence report was provided")
    elif confidence.get("deployment_allowed") is not True:
        formal_training_blockers.append("calibrated confidence report does not allow formal Markush expert training")

    deployment_blockers.extend(formal_training_blockers)
    deployment_blockers.extend(deployment_scale_blockers(training_scale))
    deployment_blockers.extend(
        independent_benchmark_blockers(
            independent_benchmark,
            min_rows=int(args.min_real_benchmark_rows_for_deployment),
        )
    )
    if calibration["positive_rows_after_training_filter"] < args.min_calibration_positive_for_deployment:
        deployment_blockers.append(
            "Markush calibration positives after training filter are "
            f"{calibration['positive_rows_after_training_filter']}; required >= "
            f"{args.min_calibration_positive_for_deployment} for deployment certification"
        )
    if not confidence.get("provided"):
        deployment_blockers.append("no calibrated confidence report was provided")
    elif confidence.get("deployment_allowed") is not True:
        deployment_blockers.append("calibrated confidence report does not allow deployment")
    if confidence.get("provided"):
        zero = deployable_zero_negative_threshold(confidence)
        positive_accepted = int(zero.get("positive_accepted") or 0)
        positive_correct = int(zero.get("positive_layout_correct") or 0)
        wilson = float(zero.get("positive_layout_precision_wilson_lower") or wilson_lower_bound(positive_correct, positive_accepted))
        if positive_accepted < args.min_zero_negative_positive_accepted_for_deployment:
            deployment_blockers.append(
                "deployable zero-negative confidence slice accepts "
                f"{positive_accepted} positives; required >= {args.min_zero_negative_positive_accepted_for_deployment}"
            )
        if wilson < args.min_zero_negative_wilson_lower_for_deployment:
            deployment_blockers.append(
                "deployable zero-negative confidence slice Wilson lower is "
                f"{wilson:.4f}; required >= {args.min_zero_negative_wilson_lower_for_deployment:.4f}"
            )

    if train["positive_rows_after_training_filter"] < 128 or calibration["positive_rows_after_training_filter"] < 64:
        smoke_warnings.append("Markush split is very small even for measured smoke; use only for debugging.")
    for split_name, summary in [("train", train), ("calibration", calibration)]:
        sparse = {
            bucket: int(count)
            for bucket, count in summary["count_bucket_counts"].items()
            if int(count) < 20
        }
        if sparse:
            smoke_warnings.append(f"{split_name} has sparse long-tail buckets: {sparse}")

    report = {
        "schema_version": "markush_architecture_eval_readiness_v1",
        "policy": {
            "small_splits_allowed_for_smoke_only": True,
            "architecture_comparison_requires_sufficient_stratified_validation": True,
            "do_not_treat_small_split_improvements_as_generalization_evidence": True,
            "do_not_relax_confidence_or_wilson_gates_to_pass_small_slices": True,
            "new_architecture_or_loss_must_reference_official_or_mature_public_implementations": True,
            "complete_molecules_remain_on_immutable_original_molnextr_path": True,
            "markush_rows_are_routed_expert_branch_only": True,
            "accepted_standards_decision_may_define_bucket_targets_without_relaxing_total_or_confidence_gates": True,
        },
        "thresholds": {
            "min_train_positive_rows_for_architecture": int(args.min_train_positive_rows_for_architecture),
            "min_calibration_positive_rows_for_architecture": int(args.min_calibration_positive_rows_for_architecture),
            "min_train_per_count_bucket_for_architecture": int(args.min_train_per_count_bucket_for_architecture),
            "min_calibration_per_count_bucket_for_architecture": int(args.min_calibration_per_count_bucket_for_architecture),
            "min_calibration_positive_for_deployment": int(args.min_calibration_positive_for_deployment),
            "min_zero_negative_positive_accepted_for_deployment": int(
                args.min_zero_negative_positive_accepted_for_deployment
            ),
            "min_zero_negative_wilson_lower_for_deployment": float(
                args.min_zero_negative_wilson_lower_for_deployment
            ),
            "min_source_documents_for_architecture": int(args.min_source_documents_for_architecture),
            "min_real_benchmark_rows_for_deployment": int(args.min_real_benchmark_rows_for_deployment),
        },
        "markush_train": train,
        "markush_calibration": calibration,
        "standards_targets": standards_targets,
        "ordinary_train": ordinary_train,
        "ordinary_calibration": ordinary_calibration,
        "split_overlap": split_overlap,
        "confidence": confidence,
        "training_scale_adequacy": training_scale,
        "independent_real_benchmark": independent_benchmark,
        "architecture_comparison_allowed": not architecture_blockers,
        "formal_expert_training_candidate_allowed": not formal_training_blockers,
        "deployment_certification_allowed": not deployment_blockers,
        "smoke_debug_allowed": True,
        "smoke_warnings": smoke_warnings,
        "architecture_blockers": architecture_blockers,
        "formal_training_blockers": formal_training_blockers,
        "deployment_blockers": deployment_blockers,
        "decision": {
            "current_data_can_debug_code_paths": True,
            "current_data_can_select_advanced_architecture": not architecture_blockers,
            "current_data_can_start_formal_expert_training": not formal_training_blockers,
            "current_data_can_support_deployment_certification": not deployment_blockers,
        },
        "recommended_next_steps": [
            "Expand or regenerate Markush positives with explicit stratification for 1, 2, 3-4, 5-8, and 9+ buckets.",
            "Reserve a larger independent calibration/test split before comparing advanced architectures.",
            "Use repeated split, bootstrap, or cross-validation once source-leak grouping is enforced.",
            "Rerun V6 nominal focal, V8 query-evidence nominal, and any new architecture on the same sufficient split.",
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.scope == "architecture_comparison":
        should_fail = bool(architecture_blockers)
    elif args.scope == "formal_training_candidate":
        should_fail = bool(formal_training_blockers)
    else:
        should_fail = bool(deployment_blockers)
    if should_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
