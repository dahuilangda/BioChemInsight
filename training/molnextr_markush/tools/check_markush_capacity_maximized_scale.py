from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUCKETS = ["1", "2", "3-4", "5-8", "9+"]


def load_json(path_text: str) -> dict[str, Any]:
    value = json.loads(Path(path_text).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path_text} must contain a JSON object")
    return value


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def source_key(row: dict[str, str], quality: dict[str, Any]) -> str:
    for key in ["source_document_key", "source_record_id", "source_file"]:
        value = str(quality.get(key) or "").strip()
        if value:
            return value
    return str(row.get("source_id") or "").strip()


def bucket_name(quality: dict[str, Any]) -> str:
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    bucket = str(markush.get("r_tag_count_bucket") or "").strip()
    return bucket if bucket in BUCKETS else "missing"


def variable_anchor_count(quality: dict[str, Any]) -> int:
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    try:
        return int(markush.get("variable_anchor_count") or 0)
    except (TypeError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a Markush candidate under a source-capacity-maximized policy. "
            "This can allow formal training start when all available high-quality source-disjoint data is used, "
            "but it never allows architecture comparison or deployment certification by itself."
        )
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--source-leak", required=True)
    parser.add_argument("--pose-alignment", required=True)
    parser.add_argument("--substitution-anchor-contract", required=True)
    parser.add_argument("--assistant-visual-review", required=True)
    parser.add_argument("--visual-risk-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-formal-capacity-rows", type=int, default=3000)
    parser.add_argument("--source-capacity-exhausted", action="store_true")
    parser.add_argument("--capacity-note", default="")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    source_leak = load_json(args.source_leak)
    pose = load_json(args.pose_alignment)
    substitution = load_json(args.substitution_anchor_contract)
    assistant_visual = load_json(args.assistant_visual_review)
    visual_risk = load_json(args.visual_risk_report)

    rows = 0
    trainable = 0
    buckets: Counter[str] = Counter()
    sources: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            quality = parse_quality(row)
            buckets.update([bucket_name(quality)])
            key = source_key(row, quality)
            if key:
                sources.add(key)
            if (
                str(row.get("reliable_training_label") or "").strip().lower() == "true"
                and str(row.get("structure_type_bucket") or "") == "markush_layout"
                and variable_anchor_count(quality) > 0
            ):
                trainable += 1

    blockers: list[str] = []
    warnings: list[str] = []
    if rows < int(args.min_formal_capacity_rows):
        blockers.append(f"row_count {rows} < min formal capacity rows {int(args.min_formal_capacity_rows)}")
    if trainable != rows:
        blockers.append(f"trainable rows {trainable} != row_count {rows}")
    if args.source_capacity_exhausted is not True:
        blockers.append("source_capacity_exhausted flag was not provided")
    if source_leak.get("passed") is not True:
        blockers.append("source leak report did not pass")
    if pose.get("passed") is not True:
        blockers.append("pose alignment report did not pass")
    if substitution.get("passed") is not True or int(substitution.get("failed_rows") or 0) != 0:
        blockers.append("substitution-anchor contract did not pass")
    if assistant_visual.get("strict_machine_visual_acceptance_passed") is not True:
        blockers.append("strict machine visual acceptance did not pass")
    if int(visual_risk.get("background_context_rows") or 0) != rows:
        blockers.append("background/document-context rows do not cover all rows")

    for bucket in BUCKETS:
        if int(buckets.get(bucket, 0)) == 0:
            warnings.append(f"bucket {bucket} has zero rows under current source capacity")

    report = {
        "schema_version": "markush_capacity_maximized_training_scale_v1",
        "csv": str(csv_path),
        "row_count": int(rows),
        "trainable_rows": int(trainable),
        "source_groups": int(len(sources)),
        "counts_by_r_tag_bucket": {bucket: int(buckets.get(bucket, 0)) for bucket in BUCKETS},
        "source_capacity_exhausted": bool(args.source_capacity_exhausted),
        "best_effort_maximized": bool(args.source_capacity_exhausted and rows >= int(args.min_formal_capacity_rows)),
        "capacity_note": str(args.capacity_note or ""),
        "formal_training_candidate_allowed": not blockers,
        "architecture_comparison_allowed": False,
        "deployment_certification_allowed": False,
        "blockers": blockers,
        "warnings": warnings,
        "policy": {
            "formal_training_can_use_best_effort_source_capacity": True,
            "does_not_allow_architecture_comparison_claim": True,
            "does_not_allow_deployment_or_final_generalization_claim": True,
            "quality_gates_not_relaxed": True,
            "manual_visual_csv_not_required": True,
            "strict_machine_visual_acceptance_required": True,
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
