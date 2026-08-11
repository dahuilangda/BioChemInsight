from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CLASS_COMPLETE = "complete_molecule"
CLASS_MARKUSH = "markush"
CLASS_FRAGMENT = "fragment_attachment"
DEFAULT_THRESHOLDS = [
    0.95,
    0.98,
    0.99,
    0.995,
    0.997,
    0.999,
    0.9995,
    0.9999,
]


def parse_thresholds(value: str) -> list[float]:
    if not value:
        return DEFAULT_THRESHOLDS
    return [float(part) for part in value.split(",") if part.strip()]


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def decide(row: dict[str, Any], *, fragment_threshold: float, markush_threshold: float) -> str:
    fragment_prob = as_float(row, "calibrated_prob_fragment_attachment")
    markush_prob = as_float(row, "calibrated_prob_markush")
    if fragment_prob >= fragment_threshold and fragment_prob >= markush_prob:
        return CLASS_FRAGMENT
    if markush_prob >= markush_threshold and markush_prob > fragment_prob:
        return CLASS_MARKUSH
    return "original"


def summarize_policy(
    rows: list[dict[str, Any]],
    *,
    fragment_threshold: float,
    markush_threshold: float,
) -> dict[str, Any]:
    counts = {
        "complete_support": 0,
        "fragment_support": 0,
        "markush_support": 0,
        "sidecar_emitted": 0,
        "complete_to_sidecar": 0,
        "complete_to_fragment": 0,
        "complete_to_markush": 0,
        "fragment_to_fragment": 0,
        "fragment_to_markush": 0,
        "fragment_to_original": 0,
        "markush_to_markush": 0,
        "markush_to_fragment": 0,
        "markush_to_original": 0,
    }
    for row in rows:
        expected = str(row.get("expected") or "")
        decision = decide(
            row,
            fragment_threshold=fragment_threshold,
            markush_threshold=markush_threshold,
        )
        if decision != "original":
            counts["sidecar_emitted"] += 1
        if expected == CLASS_COMPLETE:
            counts["complete_support"] += 1
            if decision != "original":
                counts["complete_to_sidecar"] += 1
            if decision == CLASS_FRAGMENT:
                counts["complete_to_fragment"] += 1
            elif decision == CLASS_MARKUSH:
                counts["complete_to_markush"] += 1
        elif expected == CLASS_FRAGMENT:
            counts["fragment_support"] += 1
            if decision == CLASS_FRAGMENT:
                counts["fragment_to_fragment"] += 1
            elif decision == CLASS_MARKUSH:
                counts["fragment_to_markush"] += 1
            else:
                counts["fragment_to_original"] += 1
        elif expected == CLASS_MARKUSH:
            counts["markush_support"] += 1
            if decision == CLASS_MARKUSH:
                counts["markush_to_markush"] += 1
            elif decision == CLASS_FRAGMENT:
                counts["markush_to_fragment"] += 1
            else:
                counts["markush_to_original"] += 1

    fragment_support = counts["fragment_support"]
    markush_support = counts["markush_support"]
    complete_support = counts["complete_support"]
    rows_count = len(rows)
    return {
        "fragment_threshold": float(fragment_threshold),
        "markush_threshold": float(markush_threshold),
        **counts,
        "rows": rows_count,
        "sidecar_coverage": counts["sidecar_emitted"] / rows_count if rows_count else 0.0,
        "complete_sidecar_rate": (
            counts["complete_to_sidecar"] / complete_support if complete_support else 0.0
        ),
        "fragment_recall": (
            counts["fragment_to_fragment"] / fragment_support if fragment_support else 0.0
        ),
        "markush_recall": (
            counts["markush_to_markush"] / markush_support if markush_support else 0.0
        ),
        "fragment_precision": (
            counts["fragment_to_fragment"]
            / (
                counts["fragment_to_fragment"]
                + counts["complete_to_fragment"]
                + counts["markush_to_fragment"]
            )
            if (
                counts["fragment_to_fragment"]
                + counts["complete_to_fragment"]
                + counts["markush_to_fragment"]
            )
            else 0.0
        ),
        "markush_precision": (
            counts["markush_to_markush"]
            / (
                counts["markush_to_markush"]
                + counts["complete_to_markush"]
                + counts["fragment_to_markush"]
            )
            if (
                counts["markush_to_markush"]
                + counts["complete_to_markush"]
                + counts["fragment_to_markush"]
            )
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a conservative sidecar router: complete rows default to the "
            "original MolNexTR path unless a non-complete class exceeds a high threshold."
        )
    )
    parser.add_argument("--row-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fragment-thresholds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS))
    parser.add_argument("--markush-thresholds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS))
    parser.add_argument("--max-complete-to-sidecar", type=int, default=0)
    args = parser.parse_args()

    rows = read_rows(Path(args.row_predictions))
    policies = [
        summarize_policy(rows, fragment_threshold=fragment_threshold, markush_threshold=markush_threshold)
        for fragment_threshold in parse_thresholds(args.fragment_thresholds)
        for markush_threshold in parse_thresholds(args.markush_thresholds)
    ]
    safe_policies = [
        policy
        for policy in policies
        if int(policy["complete_to_sidecar"]) <= int(args.max_complete_to_sidecar)
    ]
    safe_policies.sort(
        key=lambda row: (
            -row["fragment_recall"],
            -row["markush_recall"],
            -row["sidecar_coverage"],
            row["fragment_threshold"],
            row["markush_threshold"],
        )
    )
    policies.sort(
        key=lambda row: (
            row["complete_to_sidecar"],
            -row["fragment_recall"],
            -row["markush_recall"],
            -row["sidecar_coverage"],
        )
    )
    report = {
        "row_predictions": args.row_predictions,
        "rows": len(rows),
        "max_complete_to_sidecar": int(args.max_complete_to_sidecar),
        "best_safe_policy": safe_policies[0] if safe_policies else None,
        "safe_policy_count": len(safe_policies),
        "safe_policies": safe_policies[:24],
        "top_policies": policies[:24],
        "decision": (
            "complete_path_can_be_isolated"
            if safe_policies
            else "no_zero_complete_sidecar_policy"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
