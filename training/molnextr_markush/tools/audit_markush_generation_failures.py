from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COUNT_BUCKETS = ["1", "2", "3-4", "5-8", "9+"]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def bucket_from_source_id(source_id: str, candidate_plan: dict[str, dict[str, str]]) -> str:
    row = candidate_plan.get(source_id)
    if row:
        bucket = str(row.get("annotation_r_bucket") or "").strip()
        if bucket in COUNT_BUCKETS:
            return bucket
    return "missing"


def read_candidate_plan(path: Path) -> dict[str, dict[str, str]]:
    import csv

    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row.get("source_id") or ""): dict(row) for row in csv.DictReader(handle)}


def classify_error(error: str) -> str:
    text = str(error or "").strip()
    if re.fullmatch(r"\d+", text):
        return "opaque_numeric_exception"
    if text.startswith("pose mapping RMSE ") or text.startswith("line-constrained pose mapping RMSE "):
        return "pose_mapping_rmse_over_threshold"
    if text.startswith("InvalidSmilesException") or "Could not Kekulise" in text or "valid kekulé structure could not be assigned" in text:
        return "invalid_smiles_or_kekule"
    if "fewer than 3 real chemical atoms" in text:
        return "label_only_or_too_few_real_atoms"
    if "not enough bond midpoint correspondences" in text:
        return "insufficient_bond_midpoint_correspondences"
    if "could not convert string to float" in text:
        return "molfile_parse_text_coordinate_error"
    if "no valid Markush OCR/layout cells" in text:
        return "ocr_cell_extraction_empty"
    if "OCR text boxes occupy too much image area" in text or "single Markush OCR text box is too large" in text:
        return "text_dominated_diagram"
    if "CDK output svg/mol/render metadata missing" in text:
        return "cdk_output_missing"
    if "java depictor failed" in text:
        return "cdk_depictor_failed"
    return "other"


def rmse_value(error: str) -> float | None:
    match = re.search(r"(?:line-constrained )?pose mapping RMSE ([0-9.]+) exceeds", error)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Markush generation failure modes from shard manifests.")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples-per-category", type=int, default=12)
    args = parser.parse_args()

    category_counts: Counter[str] = Counter()
    category_bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)
    exact_error_counts: Counter[str] = Counter()
    shard_summaries: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rmse_values: list[float] = []
    total_reported_failures = 0
    total_classified_failure_details = 0
    truncated_failure_details = False

    for manifest_text in args.manifest:
        manifest_path = Path(manifest_text)
        manifest = load_json(manifest_path)
        plan_path = manifest_path.parent / "candidate_plan.csv"
        candidate_plan = read_candidate_plan(plan_path)
        failures = manifest.get("failures") if isinstance(manifest.get("failures"), list) else []
        generated_by_bucket = manifest.get("counts_by_r_tag_bucket") if isinstance(manifest.get("counts_by_r_tag_bucket"), dict) else {}
        selected_by_bucket = {}
        candidate_plan_report = manifest.get("candidate_plan") if isinstance(manifest.get("candidate_plan"), dict) else {}
        selected_raw = candidate_plan_report.get("selected_candidate_rows_by_bucket")
        if isinstance(selected_raw, dict):
            selected_by_bucket = {bucket: int(selected_raw.get(bucket) or 0) for bucket in COUNT_BUCKETS}
        reported_failure_count = int(manifest.get("failure_count") or len(failures))
        total_reported_failures += reported_failure_count
        total_classified_failure_details += len(failures)
        details_are_truncated = reported_failure_count > len(failures)
        truncated_failure_details = truncated_failure_details or details_are_truncated

        shard_failure_counts: Counter[str] = Counter()
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            error = str(failure.get("error") or "").strip()
            source_id = str(failure.get("source_id") or "").strip()
            category = classify_error(error)
            bucket = bucket_from_source_id(source_id, candidate_plan)
            category_counts[category] += 1
            category_bucket_counts[category][bucket] += 1
            exact_error_counts[error] += 1
            shard_failure_counts[category] += 1
            value = rmse_value(error)
            if value is not None:
                rmse_values.append(value)
            if len(examples[category]) < int(args.max_examples_per_category):
                examples[category].append(
                    {
                        "manifest": str(manifest_path),
                        "source_id": source_id,
                        "bucket": bucket,
                        "error": error,
                        "cxsmiles": str(failure.get("cxsmiles") or "")[:260],
                    }
                )

        shard_summaries.append(
            {
                "manifest": str(manifest_path),
                "row_count": int(manifest.get("row_count") or 0),
                "reported_failure_count": reported_failure_count,
                "classified_failure_detail_count": len(failures),
                "failure_details_truncated": details_are_truncated,
                "accepted": manifest.get("accepted") is True,
                "generated_by_bucket": {bucket: int(generated_by_bucket.get(bucket) or 0) for bucket in COUNT_BUCKETS},
                "selected_by_bucket": selected_by_bucket,
                "classified_failure_categories": {key: int(value) for key, value in sorted(shard_failure_counts.items())},
            }
        )

    blockers: list[str] = []
    if category_counts.get("opaque_numeric_exception", 0):
        blockers.append(
            "opaque numeric generation exceptions exist; generator must preserve typed error context before failures can be treated as source-quality evidence"
        )

    rmse_summary = {}
    if rmse_values:
        sorted_rmse = sorted(rmse_values)
        rmse_summary = {
            "count": len(sorted_rmse),
            "min": sorted_rmse[0],
            "mean": sum(sorted_rmse) / len(sorted_rmse),
            "p50": sorted_rmse[len(sorted_rmse) // 2],
            "p95": sorted_rmse[min(len(sorted_rmse) - 1, int(len(sorted_rmse) * 0.95))],
            "max": sorted_rmse[-1],
        }

    report = {
        "schema_version": "markush_generation_failure_audit_v1",
        "manifests": [str(Path(value)) for value in args.manifest],
        "shards": shard_summaries,
        "total_generated_rows": sum(int(item["row_count"]) for item in shard_summaries),
        "total_reported_failures": total_reported_failures,
        "total_classified_failure_details": total_classified_failure_details,
        "failure_details_truncated": truncated_failure_details,
        "classified_failure_category_counts": {key: int(value) for key, value in sorted(category_counts.items())},
        "classified_failure_category_bucket_counts": {
            category: {bucket: int(counter.get(bucket, 0)) for bucket in [*COUNT_BUCKETS, "missing"]}
            for category, counter in sorted(category_bucket_counts.items())
        },
        "top_exact_errors": [{"error": key, "count": int(value)} for key, value in exact_error_counts.most_common(40)],
        "rmse_failure_summary": rmse_summary,
        "examples": examples,
        "passed": not blockers,
        "blockers": blockers,
        "policy": {
            "audit_only": True,
            "does_not_accept_training_data": True,
            "opaque_numeric_exceptions_are_generator_bugs_not_quality_labels": True,
            "rmse_threshold_is_not_relaxed": True,
            "label_only_or_text_dominated_rows_must_not_be_fallback_accepted": True,
            "failure_categories_apply_only_to_manifest_failure_details": True,
            "manifest_failure_details_may_be_truncated": True,
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
