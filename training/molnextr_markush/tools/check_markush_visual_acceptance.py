from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_SAMPLE_FIELDS = ["stratum", "row_index", "source_id", "review_image", "source_image", "summary"]
DECISION_VALUES = {"accept", "reject", "uncertain"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sample_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("stratum") or "").strip(),
        str(row.get("row_index") or "").strip(),
        str(row.get("source_id") or "").strip(),
    )


def write_template(samples_csv: Path, output_csv: Path) -> None:
    samples = read_csv(samples_csv)
    fieldnames = REQUIRED_SAMPLE_FIELDS + [
        "decision",
        "reviewer",
        "reviewed_at",
        "issue_code",
        "notes",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in samples:
            writer.writerow(
                {field: str(row.get(field) or "") for field in REQUIRED_SAMPLE_FIELDS}
                | {
                    "decision": "",
                    "reviewer": "",
                    "reviewed_at": "",
                    "issue_code": "",
                    "notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate manual visual acceptance decisions for a Markush review package. "
            "This tool never infers acceptance from the images; it only validates explicit human decisions."
        )
    )
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--decisions-csv", default="")
    parser.add_argument("--write-template", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    review_manifest = load_json(Path(args.review_manifest))
    samples_csv = Path(str(review_manifest.get("samples_csv") or ""))
    if not samples_csv.exists():
        raise FileNotFoundError(f"samples_csv from review manifest does not exist: {samples_csv}")
    if args.write_template:
        write_template(samples_csv, Path(args.write_template))

    samples = read_csv(samples_csv)
    sample_keys = [sample_key(row) for row in samples]
    sample_key_set = set(sample_keys)
    blockers: list[str] = []
    warnings: list[str] = []
    decision_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    reviewer_counts: Counter[str] = Counter()
    missing_decision_examples: list[dict[str, str]] = []
    missing_metadata_examples: list[dict[str, str]] = []
    missing_metadata_example_keys: set[tuple[str, str, str]] = set()
    reject_examples: list[dict[str, str]] = []
    reviewed_keys: set[tuple[str, str, str]] = set()
    missing_reviewer_count = 0
    missing_reviewed_at_count = 0

    if not args.decisions_csv:
        blockers.append("manual visual decisions CSV was not provided")
    else:
        decisions_path = Path(args.decisions_csv)
        if not decisions_path.exists():
            blockers.append(f"manual visual decisions CSV does not exist: {decisions_path}")
        else:
            decisions = read_csv(decisions_path)
            if len(decisions) != len(samples):
                blockers.append(f"decision row count {len(decisions)} != sample row count {len(samples)}")
            for row in decisions:
                key = sample_key(row)
                if key not in sample_key_set:
                    blockers.append(f"decision row does not match review sample: {key}")
                    continue
                if key in reviewed_keys:
                    blockers.append(f"duplicate decision row for sample: {key}")
                    continue
                reviewed_keys.add(key)
                decision = str(row.get("decision") or "").strip().lower()
                reviewer = str(row.get("reviewer") or "").strip()
                reviewed_at = str(row.get("reviewed_at") or "").strip()
                issue_code = str(row.get("issue_code") or "").strip()
                decision_counts.update([decision or "missing"])
                if issue_code:
                    issue_counts.update([issue_code])
                if reviewer:
                    reviewer_counts.update([reviewer])
                if decision not in DECISION_VALUES:
                    if len(missing_decision_examples) < 20:
                        missing_decision_examples.append(row)
                if not reviewer:
                    missing_reviewer_count += 1
                    if key not in missing_metadata_example_keys and len(missing_metadata_examples) < 20:
                        missing_metadata_examples.append(row)
                        missing_metadata_example_keys.add(key)
                if not reviewed_at:
                    missing_reviewed_at_count += 1
                    if key not in missing_metadata_example_keys and len(missing_metadata_examples) < 20:
                        missing_metadata_examples.append(row)
                        missing_metadata_example_keys.add(key)
                if decision in {"reject", "uncertain"}:
                    if len(reject_examples) < 20:
                        reject_examples.append(row)
            missing_keys = sorted(sample_key_set - reviewed_keys)
            for key in missing_keys[:20]:
                blockers.append(f"missing decision row for sample: {key}")
            if missing_keys[20:]:
                blockers.append(f"{len(missing_keys) - 20} additional samples missing decisions")

    if decision_counts.get("missing", 0) > 0 or missing_decision_examples:
        blockers.append("one or more review samples are missing an accept/reject/uncertain decision")
    if missing_reviewer_count:
        blockers.append(f"{missing_reviewer_count} review samples are missing reviewer")
    if missing_reviewed_at_count:
        blockers.append(f"{missing_reviewed_at_count} review samples are missing reviewed_at")
    if decision_counts.get("reject", 0) > 0:
        blockers.append("one or more review samples were rejected")
    if decision_counts.get("uncertain", 0) > 0:
        blockers.append("one or more review samples were marked uncertain")
    if not reviewer_counts and args.decisions_csv:
        blockers.append("no reviewer identity was recorded")

    passed = not blockers
    report = {
        "schema_version": "markush_manual_visual_acceptance_v1",
        "review_manifest": str(args.review_manifest),
        "samples_csv": str(samples_csv),
        "decisions_csv": str(args.decisions_csv or ""),
        "template_csv": str(args.write_template or ""),
        "sample_count": len(samples),
        "reviewed_sample_count": len(reviewed_keys),
        "decision_counts": dict(sorted(decision_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "reviewer_counts": dict(sorted(reviewer_counts.items())),
        "missing_reviewer_count": missing_reviewer_count,
        "missing_reviewed_at_count": missing_reviewed_at_count,
        "accepted": passed,
        "passed": passed,
        "blockers": blockers,
        "warnings": warnings,
        "missing_decision_examples": missing_decision_examples[:20],
        "missing_metadata_examples": missing_metadata_examples[:20],
        "reject_or_uncertain_examples": reject_examples[:20],
        "policy": {
            "manual_human_decisions_required": True,
            "every_review_sample_requires_decision": True,
            "reject_or_uncertain_blocks_acceptance": True,
            "reviewer_and_reviewed_at_required": True,
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
