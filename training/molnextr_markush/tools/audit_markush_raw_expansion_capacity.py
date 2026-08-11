from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COUNT_BUCKETS = ["1", "2", "3-4", "5-8", "9+"]


def read_arrow_or_parquet_rows(path: Path) -> list[dict[str, Any]]:
    columns = ["id", "annotation", "cxsmiles_dataset", "cxsmiles", "cxsmiles_opt", "cells", "image_name", "page_image_path"]
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError(f"pyarrow is required for raw Markush parquet audit: {exc}") from exc
        schema = set(pq.read_schema(path).names)
        selected = [name for name in columns if name in schema]
        return pq.read_table(path, columns=selected).to_pylist()
    if path.suffix == ".arrow":
        try:
            import pyarrow.ipc as ipc
            from pyarrow.lib import ArrowInvalid
        except Exception as exc:
            raise RuntimeError(f"pyarrow is required for raw Markush arrow audit: {exc}") from exc
        with path.open("rb") as handle:
            try:
                table = ipc.open_file(handle).read_all()
            except ArrowInvalid:
                handle.seek(0)
                table = ipc.open_stream(handle).read_all()
        selected = [name for name in columns if name in set(table.column_names)]
        return table.select(selected).to_pylist()
    return []


def count_bucket(value: int) -> str | None:
    if value <= 0:
        return None
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    if value <= 8:
        return "5-8"
    return "9+"


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def primary_cxsmiles(row: dict[str, Any]) -> str:
    return clean_text(row.get("cxsmiles") or row.get("cxsmiles_dataset") or row.get("cxsmiles_opt") or "")


def annotation_r_labels(annotation: str) -> list[str]:
    return [match.strip() for match in re.findall(r"<r>(.*?)</r>", annotation or "", flags=re.IGNORECASE | re.DOTALL)]


def markush_r_labels(annotation: str, cxsmiles_opt: str = "") -> list[str]:
    labels = annotation_r_labels(annotation)
    if labels:
        return labels
    return annotation_r_labels(cxsmiles_opt)


def cxsmiles_star_count(cxsmiles: str) -> int:
    return len(re.findall(r"\*", cxsmiles or ""))


def is_variable_label(text: str) -> bool:
    token = text.strip()
    upper = token.upper()
    if not token:
        return False
    if token in {"*", "_AP"} or upper.startswith("_AP"):
        return True
    if upper.startswith(("R", "X", "Z")):
        return True
    if upper in {"A", "M", "G", "Q", "Y", "HET", "AR", "ALK", "HAL"}:
        return True
    if any(part in upper for part in ["R1", "R2", "R3", " OR", "NR", "NHR"]):
        return True
    return False


def raw_cell_variable_count(row: dict[str, Any]) -> int:
    cells = row.get("cells")
    if not isinstance(cells, list):
        return 0
    count = 0
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        bbox = cell.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if not (0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0):
            continue
        if is_variable_label(clean_text(cell.get("text"))):
            count += 1
    return count


def source_document_key(subset: str, row: dict[str, Any]) -> str:
    raw_id = clean_text(row.get("id"))
    if not raw_id:
        fallback = "|".join([primary_cxsmiles(row), clean_text(row.get("image_name")), clean_text(row.get("page_image_path"))])
        raw_id = hashlib.sha1(fallback.encode("utf-8")).hexdigest()
    if ".pdf_" in raw_id:
        document = raw_id.split(".pdf_", 1)[0] + ".pdf"
    else:
        document = re.sub(r"_[0-9]+(?:_[0-9]+)+$", "", raw_id)
    return f"{subset}:{document}"


def increment_bucket(counter: Counter[str], count: int) -> None:
    bucket = count_bucket(count)
    if bucket is not None:
        counter[bucket] += 1


def summarize_counter(counter: Counter[str]) -> dict[str, int]:
    return {bucket: int(counter.get(bucket, 0)) for bucket in COUNT_BUCKETS}


def exact_counter_to_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def audit_raw_markush(root: Path, subset_glob: str) -> dict[str, Any]:
    raw_paths = sorted(root.glob(subset_glob))
    total_rows = 0
    candidate_rows = 0
    annotation_positive_rows = 0
    cxsmiles_star_positive_rows = 0
    raw_cell_variable_positive_rows = 0
    subset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    subset_rows: Counter[str] = Counter()
    subset_annotation_positive: Counter[str] = Counter()
    subset_star_positive: Counter[str] = Counter()
    subset_raw_cell_positive: Counter[str] = Counter()
    annotation_bucket_counts: Counter[str] = Counter()
    star_bucket_counts: Counter[str] = Counter()
    raw_cell_bucket_counts: Counter[str] = Counter()
    annotation_exact_counts: Counter[int] = Counter()
    star_exact_counts: Counter[int] = Counter()
    raw_cell_exact_counts: Counter[int] = Counter()
    source_documents: set[str] = set()
    source_documents_by_annotation_bucket: dict[str, set[str]] = defaultdict(set)
    examples_by_annotation_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    mismatch_examples: list[dict[str, Any]] = []

    for path in raw_paths:
        try:
            subset = path.parent.relative_to(root).as_posix()
        except ValueError:
            subset = path.parent.name
        if not subset or subset == ".":
            subset = path.parent.name
        rows = read_arrow_or_parquet_rows(path)
        for row in rows:
            total_rows += 1
            subset_rows[subset] += 1
            annotation = clean_text(row.get("annotation"))
            cxsmiles_opt = clean_text(row.get("cxsmiles_opt"))
            cxsmiles = primary_cxsmiles(row)
            has_markush_markup = "<markush" in annotation.lower()
            if not cxsmiles or (not has_markush_markup and not cxsmiles_opt):
                continue
            candidate_rows += 1
            r_count = len(markush_r_labels(annotation, cxsmiles_opt))
            star_count = cxsmiles_star_count(cxsmiles)
            cell_count = raw_cell_variable_count(row)
            source_doc = source_document_key(subset, row)
            source_documents.add(source_doc)

            if r_count > 0:
                annotation_positive_rows += 1
                subset_annotation_positive[subset] += 1
                annotation_exact_counts[r_count] += 1
                increment_bucket(annotation_bucket_counts, r_count)
                bucket = count_bucket(r_count)
                if bucket:
                    subset_counts[subset][bucket] += 1
                    source_documents_by_annotation_bucket[bucket].add(source_doc)
                    if len(examples_by_annotation_bucket[bucket]) < 8:
                        examples_by_annotation_bucket[bucket].append(
                            {
                                "subset": subset,
                                "id": clean_text(row.get("id")),
                                "annotation_r_count": str(r_count),
                                "cxsmiles_star_count": str(star_count),
                                "raw_cell_variable_count": str(cell_count),
                                "cxsmiles": cxsmiles[:240],
                            }
                        )
            if star_count > 0:
                cxsmiles_star_positive_rows += 1
                subset_star_positive[subset] += 1
                star_exact_counts[star_count] += 1
                increment_bucket(star_bucket_counts, star_count)
            if cell_count > 0:
                raw_cell_variable_positive_rows += 1
                subset_raw_cell_positive[subset] += 1
                raw_cell_exact_counts[cell_count] += 1
                increment_bucket(raw_cell_bucket_counts, cell_count)
            if r_count > 0 and len(mismatch_examples) < 40 and abs(r_count - star_count) >= 3:
                mismatch_examples.append(
                    {
                        "subset": subset,
                        "id": clean_text(row.get("id")),
                        "annotation_r_count": r_count,
                        "cxsmiles_star_count": star_count,
                        "raw_cell_variable_count": cell_count,
                        "cxsmiles": cxsmiles[:240],
                    }
                )

    return {
        "raw_root": str(root),
        "subset_glob": subset_glob,
        "raw_files": [str(path) for path in raw_paths],
        "total_rows": int(total_rows),
        "markush_markup_candidate_rows": int(candidate_rows),
        "annotation_r_positive_rows": int(annotation_positive_rows),
        "cxsmiles_star_positive_rows": int(cxsmiles_star_positive_rows),
        "raw_cell_variable_positive_rows": int(raw_cell_variable_positive_rows),
        "annotation_r_count_bucket_counts": summarize_counter(annotation_bucket_counts),
        "cxsmiles_star_count_bucket_counts": summarize_counter(star_bucket_counts),
        "raw_cell_variable_count_bucket_counts": summarize_counter(raw_cell_bucket_counts),
        "annotation_r_exact_counts": exact_counter_to_dict(annotation_exact_counts),
        "cxsmiles_star_exact_counts": exact_counter_to_dict(star_exact_counts),
        "raw_cell_variable_exact_counts": exact_counter_to_dict(raw_cell_exact_counts),
        "subset_rows": dict(sorted((key, int(value)) for key, value in subset_rows.items())),
        "subset_annotation_r_positive_rows": dict(sorted((key, int(value)) for key, value in subset_annotation_positive.items())),
        "subset_cxsmiles_star_positive_rows": dict(sorted((key, int(value)) for key, value in subset_star_positive.items())),
        "subset_raw_cell_variable_positive_rows": dict(sorted((key, int(value)) for key, value in subset_raw_cell_positive.items())),
        "subset_annotation_r_bucket_counts": {
            subset: summarize_counter(counter)
            for subset, counter in sorted(subset_counts.items())
        },
        "unique_source_document_keys": int(len(source_documents)),
        "source_document_keys_by_annotation_bucket": {
            bucket: int(len(source_documents_by_annotation_bucket.get(bucket, set())))
            for bucket in COUNT_BUCKETS
        },
        "examples_by_annotation_bucket": {
            bucket: examples_by_annotation_bucket.get(bucket, [])
            for bucket in COUNT_BUCKETS
        },
        "mismatch_examples": mismatch_examples,
    }


def readiness_from_capacity(report: dict[str, Any], *, min_total: int, min_bucket: int, min_source_docs: int) -> dict[str, Any]:
    blockers = []
    annotation_positive = int(report.get("annotation_r_positive_rows") or 0)
    if annotation_positive < min_total:
        blockers.append(f"annotation-r positive rows are {annotation_positive}; required >= {min_total}")
    counts = report.get("annotation_r_count_bucket_counts") if isinstance(report.get("annotation_r_count_bucket_counts"), dict) else {}
    for bucket in COUNT_BUCKETS:
        count = int(counts.get(bucket) or 0)
        if count < min_bucket:
            blockers.append(f"annotation-r bucket {bucket} has {count} rows; required >= {min_bucket}")
    source_docs = int(report.get("unique_source_document_keys") or 0)
    if source_docs < min_source_docs:
        blockers.append(f"source-document groups are {source_docs}; required >= {min_source_docs}")
    return {
        "can_attempt_large_markush_pose_generation_from_local_mg2": not blockers,
        "blockers": blockers,
        "thresholds": {
            "min_total_annotation_r_positive_rows": int(min_total),
            "min_annotation_r_rows_per_bucket": int(min_bucket),
            "min_source_document_groups": int(min_source_docs),
        },
        "caveat": (
            "This is a raw-source capacity audit. It estimates count coverage from annotation <r> tags; "
            "CDK/MarkushGenerator rendering, OCR-cell extraction, pose mapping, validation, visual review, "
            "source-leak checks, and acceptance gates still have to pass before generated rows are trainable."
        ),
    }


def formal_capacity_from_local_mg2(report: dict[str, Any], *, min_formal_positive_rows: int) -> dict[str, Any]:
    annotation_positive = int(report.get("annotation_r_positive_rows") or 0)
    accepted_upper_bound = annotation_positive
    blockers: list[str] = []
    if accepted_upper_bound < int(min_formal_positive_rows):
        blockers.append(
            "local MarkushGrapher-2 annotation-positive upper bound is "
            f"{accepted_upper_bound}; formal Markush branch target is >= {int(min_formal_positive_rows)}"
        )
    return {
        "local_mg2_alone_can_supply_formal_markush_branch": not blockers,
        "annotation_positive_upper_bound": accepted_upper_bound,
        "min_formal_positive_rows": int(min_formal_positive_rows),
        "blockers": blockers,
        "caveat": (
            "This is an optimistic raw upper bound. Accepted trainable rows can only decrease after "
            "rendering, pose mapping, source-leak filtering, visual review, and acceptance gates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local MarkushGrapher-2 raw data capacity before large pose generation.")
    parser.add_argument("--raw-root", default="training/molnextr_markush/data/raw/markushgrapher2")
    parser.add_argument("--subset-glob", default="*/*.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-csv", default="")
    parser.add_argument("--min-total-annotation-r-positive-rows", type=int, default=3000)
    parser.add_argument("--min-annotation-r-rows-per-bucket", type=int, default=250)
    parser.add_argument("--min-source-document-groups", type=int, default=2000)
    parser.add_argument("--min-formal-positive-rows", type=int, default=100000)
    args = parser.parse_args()

    report = audit_raw_markush(Path(args.raw_root), str(args.subset_glob))
    report["readiness"] = readiness_from_capacity(
        report,
        min_total=int(args.min_total_annotation_r_positive_rows),
        min_bucket=int(args.min_annotation_r_rows_per_bucket),
        min_source_docs=int(args.min_source_document_groups),
    )
    report["formal_capacity"] = formal_capacity_from_local_mg2(
        report,
        min_formal_positive_rows=int(args.min_formal_positive_rows),
    )
    report["policy"] = {
        "raw_audit_only": True,
        "does_not_create_trainable_rows": True,
        "formal_capacity_is_upper_bound_not_training_acceptance": True,
        "pose_factory_validation_required_before_training": True,
        "visual_review_required_before_training": True,
        "source_leak_check_required_before_training": True,
        "acceptance_gate_required_before_training": True,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    if args.sample_csv:
        sample_path = Path(args.sample_csv)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["bucket", "subset", "id", "annotation_r_count", "cxsmiles_star_count", "raw_cell_variable_count", "cxsmiles"],
            )
            writer.writeheader()
            for bucket, examples in report["examples_by_annotation_bucket"].items():
                for example in examples:
                    writer.writerow({"bucket": bucket, **example})

    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if report["readiness"]["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
