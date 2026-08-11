from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

csv.field_size_limit(sys.maxsize)

COUNT_BUCKETS = ["1", "2", "3-4", "5-8", "9+"]
EXPECTED_DECISION = "source_capacity_proportional_30k"


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def int_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for bucket in COUNT_BUCKETS:
        try:
            counts[bucket] = int(value.get(bucket) or 0)
        except (TypeError, ValueError):
            counts[bucket] = 0
    return counts


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def markush_bucket(row: dict[str, str], quality: dict[str, Any] | None = None) -> str:
    quality = quality if isinstance(quality, dict) else parse_quality(row)
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    bucket = str(markush.get("r_tag_count_bucket") or "").strip()
    if bucket in COUNT_BUCKETS:
        return bucket
    try:
        count = int(markush.get("r_tag_count") or 0)
    except (TypeError, ValueError):
        count = 0
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
    return ""


def resolve_manifest_csv(manifest: dict[str, Any], manifest_path: Path) -> Path:
    csv_text = str(manifest.get("csv") or "").strip()
    if not csv_text:
        return Path("")
    path = Path(csv_text)
    if path.exists() or path.is_absolute():
        return path
    candidate = manifest_path.parent / path
    return candidate if candidate.exists() else path


def trainable_counts_by_bucket(csv_path: Path) -> dict[str, Any]:
    counts = {bucket: 0 for bucket in COUNT_BUCKETS}
    skipped = {bucket: 0 for bucket in COUNT_BUCKETS}
    skipped_unknown_bucket = 0
    total_rows = 0
    trainable_rows = 0
    if not csv_path.exists():
        return {
            "csv": str(csv_path),
            "exists": False,
            "row_count": 0,
            "trainable_rows": 0,
            "counts_by_r_tag_bucket": counts,
            "skipped_by_r_tag_bucket": skipped,
            "skipped_unknown_bucket": 0,
        }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            quality = parse_quality(row)
            markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
            bucket = markush_bucket(row, quality)
            try:
                variable_anchor_count = int(markush.get("variable_anchor_count") or 0)
            except (TypeError, ValueError):
                variable_anchor_count = 0
            if not (
                parse_bool(row.get("reliable_training_label"))
                and str(row.get("structure_type_bucket") or "").strip() == "markush_layout"
                and variable_anchor_count > 0
            ):
                if bucket in skipped:
                    skipped[bucket] += 1
                else:
                    skipped_unknown_bucket += 1
                continue
            if bucket in counts:
                counts[bucket] += 1
                trainable_rows += 1
            else:
                skipped_unknown_bucket += 1
    return {
        "csv": str(csv_path),
        "exists": True,
        "row_count": int(total_rows),
        "trainable_rows": int(trainable_rows),
        "counts_by_r_tag_bucket": counts,
        "skipped_by_r_tag_bucket": skipped,
        "skipped_unknown_bucket": int(skipped_unknown_bucket),
            "filter_policy": {
                "reliable_training_label_required": True,
                "structure_type_bucket_required": "markush_layout",
                "annotation_matched_ocr_cell_count_must_be_positive": True,
                "matches_train_markush_layout_expert_positive_filter": True,
                "label_source": "render_quality.markush.annotation <r> labels matched to render_quality.markush.ocr_cells text",
            },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check a Markush candidate aggregate against the accepted standards decision. "
            "This gate only verifies candidate coverage; it never accepts data or starts training."
        )
    )
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--standards-decision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.candidate_manifest)
    manifest = load_json(manifest_path)
    decision = load_json(args.standards_decision)

    observed = int_counts(manifest.get("counts_by_r_tag_bucket"))
    targets = int_counts(decision.get("candidate_targets_by_bucket"))
    trainable_coverage = trainable_counts_by_bucket(resolve_manifest_csv(manifest, manifest_path))
    trainable_observed = int_counts(trainable_coverage.get("counts_by_r_tag_bucket"))
    bucket_results: dict[str, dict[str, Any]] = {}
    trainable_bucket_results: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    warnings: list[str] = []

    if decision.get("schema_version") != "markush_standards_decision_v1":
        blockers.append("standards decision schema_version is not markush_standards_decision_v1")
    if decision.get("accepted") is not True:
        blockers.append("standards decision is not accepted")
    if str(decision.get("decision") or "").strip() != EXPECTED_DECISION:
        blockers.append(f"standards decision is not {EXPECTED_DECISION}")
    policy = decision.get("policy") if isinstance(decision.get("policy"), dict) else {}
    if policy.get("does_not_start_expert_training") is not True:
        blockers.append("standards decision policy must declare does_not_start_expert_training=true")
    if policy.get("does_not_accept_training_data") is not True:
        blockers.append("standards decision policy must declare does_not_accept_training_data=true")

    if manifest.get("passed") is not True:
        blockers.append("candidate manifest did not pass its own aggregate gate")
    if manifest.get("accepted") is True:
        blockers.append("candidate manifest is already accepted; this gate expects candidate-only data")
    if manifest.get("rejected") is True:
        blockers.append("candidate manifest is explicitly rejected")
    if int(manifest.get("source_document_overlap") or 0) != 0:
        blockers.append("candidate manifest reports source_document_overlap > 0")
    if int(manifest.get("candidate_plan_index_overlap") or 0) != 0:
        blockers.append("candidate manifest reports candidate_plan_index_overlap > 0")
    if trainable_coverage.get("exists") is not True:
        blockers.append("candidate CSV for trainable coverage check does not exist")

    for bucket in COUNT_BUCKETS:
        count = int(observed.get(bucket) or 0)
        trainable_count = int(trainable_observed.get(bucket) or 0)
        target = int(targets.get(bucket) or 0)
        passed = count >= target and target > 0
        trainable_passed = trainable_count >= target and target > 0
        bucket_results[bucket] = {
            "observed": count,
            "target": target,
            "margin": count - target,
            "passed": passed,
        }
        trainable_bucket_results[bucket] = {
            "observed": trainable_count,
            "target": target,
            "margin": trainable_count - target,
            "passed": trainable_passed,
            "raw_minus_trainable": count - trainable_count,
        }
        if target <= 0:
            blockers.append(f"bucket {bucket} has no positive target in standards decision")
        elif count < target:
            blockers.append(f"bucket {bucket} observed {count} < target {target}")
        elif trainable_count < target:
            blockers.append(f"trainable bucket {bucket} observed {trainable_count} < target {target}")

    observed_total = sum(observed.values())
    trainable_total = sum(trainable_observed.values())
    target_total = sum(targets.values())
    manifest_rows = int(manifest.get("row_count") or 0)
    if manifest_rows != observed_total:
        blockers.append(f"manifest row_count {manifest_rows} != observed bucket total {observed_total}")
    if target_total < 30000:
        blockers.append(f"standards target total {target_total} < 30000")
    if observed_total < target_total:
        blockers.append(f"observed bucket total {observed_total} < target total {target_total}")
    if trainable_total < target_total:
        blockers.append(f"trainable bucket total {trainable_total} < target total {target_total}")

    if manifest.get("accepted") is not False:
        warnings.append("candidate manifest does not explicitly use accepted=false")

    report = {
        "schema_version": "markush_candidate_coverage_against_standards_v1",
        "candidate_manifest": str(args.candidate_manifest),
        "standards_decision": str(args.standards_decision),
        "selected_decision": str(decision.get("decision") or ""),
        "candidate_row_count": manifest_rows,
        "observed_total_by_bucket": observed_total,
        "trainable_total_by_bucket": trainable_total,
        "target_total_by_bucket": target_total,
        "bucket_results": bucket_results,
        "trainable_coverage": trainable_coverage,
        "trainable_bucket_results": trainable_bucket_results,
        "passed": not blockers,
        "accepted": False,
        "formal_training_allowed": False,
        "architecture_comparison_data_coverage_allowed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "remaining_required_gates_before_training": [
            "manual_visual_acceptance",
            "source_disjoint_formal_split",
            "schema_validation",
            "pose_alignment",
            "source_leak",
            "coverage",
            "taxonomy",
            "attachment_role",
            "confidence",
            "runtime",
            "model_scale",
            "router_contract",
        ],
        "policy": {
            "candidate_coverage_gate_only": True,
            "does_not_accept_training_data": True,
            "does_not_start_expert_training": True,
            "does_not_relax_confidence_or_runtime_gates": True,
            "does_not_override_visual_review": True,
            "trainable_coverage_must_match_markush_layout_expert_filter": True,
            "trainable_markush_labels_require_annotation_matched_ocr_cells": True,
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
