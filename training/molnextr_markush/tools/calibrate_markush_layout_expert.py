from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_THRESHOLDS = "0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.5,0.7,0.9"
CONFIDENCE_COLUMNS = [
    "presence_confidence",
    "count_confidence",
    "risk_confidence",
    "sidecar_confidence",
]


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_thresholds(value: str) -> list[float]:
    return [float(part) for part in str(value).split(",") if part.strip()]


def score_adaptive_thresholds(frame: pd.DataFrame, thresholds: list[float]) -> list[float]:
    values = {float(threshold) for threshold in thresholds}
    scores = [float(value) for value in frame["sidecar_confidence"].astype(float).tolist()]
    values.update(scores)
    negative_scores = frame.loc[~frame["positive_row"], "sidecar_confidence"].astype(float)
    if len(negative_scores):
        max_negative = float(negative_scores.max())
        values.add(max_negative)
        values.add(math.nextafter(max_negative, math.inf))
    values.add(0.0)
    return sorted(value for value in values if math.isfinite(value))


def load_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, low_memory=False)
    missing_confidence_columns = [column for column in CONFIDENCE_COLUMNS if column not in frame.columns]
    if missing_confidence_columns:
        raise ValueError(f"Predictions CSV missing required confidence columns: {missing_confidence_columns}")
    for column in CONFIDENCE_COLUMNS:
        frame[column] = frame[column].map(as_float)
    frame["gold_markush_presence_int"] = frame["gold_markush_presence"].map(as_int)
    frame["pred_markush_presence_int"] = frame["pred_markush_presence"].map(as_int)
    frame["positive_row"] = frame["gold_markush_presence_int"] == 1
    frame["presence_correct"] = frame["gold_markush_presence_int"] == frame["pred_markush_presence_int"]
    frame["count_correct"] = frame["gold_count_bucket"].astype(str) == frame["pred_count_bucket"].astype(str)
    frame["pred_nonzero_count_bucket"] = frame["pred_count_bucket"].astype(str) != "0"
    frame["markush_sidecar_emitted"] = (frame["pred_markush_presence_int"] == 1) & frame["pred_nonzero_count_bucket"]
    frame["layout_box_ok"] = frame["layout_box_l1"].map(as_float) <= 0.25
    frame["markush_accept_correct"] = (
        (~frame["positive_row"] & (frame["pred_markush_presence_int"] == 0))
        | (frame["positive_row"] & frame["presence_correct"] & frame["count_correct"] & frame["layout_box_ok"])
    )
    return frame


def binary_brier(scores: pd.Series, labels: pd.Series) -> float:
    if len(scores) == 0:
        return 0.0
    return float(((scores.astype(float) - labels.astype(float)) ** 2).mean())


def ece(scores: pd.Series, labels: pd.Series, *, bins: int) -> tuple[float, list[dict[str, Any]]]:
    total = max(1, len(scores))
    value = 0.0
    rows = []
    for index in range(bins):
        lo = index / bins
        hi = (index + 1) / bins
        mask = (scores >= lo) & (scores <= hi) if index == bins - 1 else (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": index, "lo": lo, "hi": hi, "count": 0})
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(labels[mask].mean())
        gap = abs(accuracy - confidence)
        value += (count / total) * gap
        rows.append(
            {
                "bin": index,
                "lo": lo,
                "hi": hi,
                "count": count,
                "mean_confidence": confidence,
                "accuracy": accuracy,
                "abs_gap": gap,
            }
        )
    return float(value), rows


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p_hat = float(successes) / float(total)
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / float(total)
    center = p_hat + z2 / (2.0 * float(total))
    margin = float(z) * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * float(total))) / float(total))
    return max(0.0, float((center - margin) / denominator))


def confidence_report(frame: pd.DataFrame, *, bins: int) -> dict[str, Any]:
    markush_emit_correct = (
        frame["positive_row"]
        & frame["markush_sidecar_emitted"]
        & frame["presence_correct"]
        & frame["count_correct"]
        & frame["layout_box_ok"]
    )
    labels = {
        "presence_confidence": frame["positive_row"].astype(float),
        "count_confidence": (frame["positive_row"] & frame["count_correct"]).astype(float),
        "risk_confidence": frame["markush_accept_correct"].astype(float),
        "sidecar_confidence": markush_emit_correct.astype(float),
    }
    output = {}
    for column, label in labels.items():
        score = frame[column].astype(float)
        ece_value, rows = ece(score, label, bins=bins)
        output[column] = {
            "mean_score": float(score.mean()) if len(score) else 0.0,
            "brier": binary_brier(score, label),
            "ece": ece_value,
            "reliability_bins": rows,
        }
    return output


def evaluate_thresholds(frame: pd.DataFrame, thresholds: list[float]) -> list[dict[str, Any]]:
    rows = []
    positive_count = int(frame["positive_row"].sum())
    negative_count = int((~frame["positive_row"]).sum())
    for threshold in thresholds:
        accepted = frame["markush_sidecar_emitted"] & (frame["sidecar_confidence"] >= float(threshold))
        accepted_frame = frame[accepted]
        positive_accepted = accepted_frame[accepted_frame["positive_row"]]
        correct_positive = int(
            (
                positive_accepted["presence_correct"]
                & positive_accepted["count_correct"]
                & positive_accepted["layout_box_ok"]
            ).sum()
        )
        positive_accepted_count = int(len(positive_accepted))
        negative_accepted = int((~accepted_frame["positive_row"]).sum())
        correct = int(accepted_frame["markush_accept_correct"].sum())
        positive_layout_precision = correct_positive / positive_accepted_count if positive_accepted_count else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "accepted": int(len(accepted_frame)),
                "coverage": len(accepted_frame) / len(frame) if len(frame) else 0.0,
                "correct": correct,
                "precision": correct / len(accepted_frame) if len(accepted_frame) else 0.0,
                "negative_accepted": negative_accepted,
                "negative_accepted_rate": negative_accepted / negative_count if negative_count else 0.0,
                "positive_accepted": positive_accepted_count,
                "positive_layout_correct": correct_positive,
                "positive_layout_precision": positive_layout_precision,
                "positive_layout_precision_wilson_lower": wilson_lower_bound(
                    correct_positive,
                    positive_accepted_count,
                ),
                "positive_layout_recall": correct_positive / positive_count if positive_count else 0.0,
            }
        )
    return rows


def select_threshold(
    rows: list[dict[str, Any]],
    *,
    max_negative_accepted: int,
    min_positive_layout_precision: float,
    min_positive_accepted: int,
    min_positive_precision_wilson_lower: float,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if int(row["negative_accepted"]) <= int(max_negative_accepted)
        and float(row["positive_layout_precision"]) >= float(min_positive_layout_precision)
        and int(row["positive_accepted"]) >= int(min_positive_accepted)
        and float(row.get("positive_layout_precision_wilson_lower") or 0.0)
        >= float(min_positive_precision_wilson_lower)
    ]
    candidates.sort(key=lambda row: (-float(row["positive_layout_recall"]), -float(row["coverage"]), float(row["threshold"])))
    return candidates[0] if candidates else None


def confusion(frame: pd.DataFrame) -> dict[str, Any]:
    presence = defaultdict(Counter)
    count = defaultdict(Counter)
    for row in frame.to_dict(orient="records"):
        presence[str(row.get("gold_markush_presence") or "")][str(row.get("pred_markush_presence") or "")] += 1
        if str(row.get("gold_markush_presence") or "") == "1":
            count[str(row.get("gold_count_bucket") or "")][str(row.get("pred_count_bucket") or "")] += 1
    return {
        "presence": {key: dict(value) for key, value in sorted(presence.items())},
        "count_bucket": {key: dict(value) for key, value in sorted(count.items())},
    }


def high_confidence_errors(frame: pd.DataFrame, *, limit: int) -> list[dict[str, Any]]:
    errors = frame[~frame["markush_accept_correct"]].sort_values("sidecar_confidence", ascending=False)
    keep = [
        "row_id",
        "source_id",
        "source_arrow",
        "smiles",
        "gold_markush_presence",
        "pred_markush_presence",
        "gold_count_bucket",
        "pred_count_bucket",
        "presence_confidence",
        "count_confidence",
        "risk_confidence",
        "sidecar_confidence",
        "layout_box_l1",
    ]
    return errors[keep].head(limit).to_dict(orient="records")


def high_confidence_error_count(frame: pd.DataFrame, *, threshold: float) -> int:
    accepted = frame["markush_sidecar_emitted"] & (frame["sidecar_confidence"] >= float(threshold))
    return int((accepted & ~frame["markush_accept_correct"]).sum())


def selected_error_summary(frame: pd.DataFrame, *, threshold: float | None) -> dict[str, Any] | None:
    if threshold is None:
        return None
    accepted = frame["markush_sidecar_emitted"] & (frame["sidecar_confidence"] >= float(threshold))
    positive = frame["positive_row"]
    negative = ~positive
    positive_task_error = accepted & positive & ~frame["markush_accept_correct"]
    negative_accept_error = accepted & negative
    return {
        "total_accepted_errors": int((accepted & ~frame["markush_accept_correct"]).sum()),
        "positive_layout_errors": int(positive_task_error.sum()),
        "negative_accepted_errors": int(negative_accept_error.sum()),
        "deployment_blocking_error_classes": [
            "negative_accepted_errors",
        ],
        "audited_non_blocking_error_classes": [
            "positive_layout_errors",
        ],
    }


def build_report(
    *,
    predictions_csv: str | Path,
    thresholds: list[float],
    ece_bins: int,
    max_negative_accepted: int,
    min_positive_layout_precision: float,
    min_positive_accepted: int,
    min_positive_precision_wilson_lower: float,
    error_limit: int,
) -> dict[str, Any]:
    frame = load_frame(predictions_csv)
    threshold_candidates = score_adaptive_thresholds(frame, thresholds)
    threshold_rows = evaluate_thresholds(frame, threshold_candidates)
    selected = select_threshold(
        threshold_rows,
        max_negative_accepted=max_negative_accepted,
        min_positive_layout_precision=min_positive_layout_precision,
        min_positive_accepted=min_positive_accepted,
        min_positive_precision_wilson_lower=min_positive_precision_wilson_lower,
    )
    selected_threshold = float(selected["threshold"]) if selected else None
    selected_high_confidence_errors = (
        high_confidence_error_count(frame, threshold=selected_threshold) if selected_threshold is not None else None
    )
    selected_errors = selected_error_summary(frame, threshold=selected_threshold)
    confidence = confidence_report(frame, bins=int(ece_bins))
    deployment_allowed = bool(
        selected
        and int(selected["negative_accepted"]) <= int(max_negative_accepted)
        and float(selected["positive_layout_precision"]) >= float(min_positive_layout_precision)
        and int(selected["positive_accepted"]) >= int(min_positive_accepted)
        and float(selected.get("positive_layout_precision_wilson_lower") or 0.0)
        >= float(min_positive_precision_wilson_lower)
    )
    return {
        "predictions_csv": str(predictions_csv),
        "rows": int(len(frame)),
        "positive_rows": int(frame["positive_row"].sum()),
        "negative_rows": int((~frame["positive_row"]).sum()),
        "confidence_design": {
            "task": "Markush layout/OCR evidence sidecar selective output",
            "heads": [
                "presence_confidence",
                "count_confidence",
                "risk_confidence",
                "selective_sidecar_confidence",
            ],
            "selection_policy": "emit Markush sidecar evidence only when calibrated risk/coverage threshold satisfies zero ordinary-negative acceptance and layout precision constraints",
            "threshold_policy": "fixed thresholds plus score-adaptive calibration candidates; a row is accepted only when pred_markush_presence=1, pred_count_bucket!=0, and sidecar_confidence clears threshold",
        },
        "confusion": confusion(frame),
        "confidence": confidence,
        "threshold_curve": threshold_rows,
        "threshold_candidate_count": len(threshold_candidates),
        "fixed_thresholds": [float(value) for value in thresholds],
        "quality_constraints": {
            "max_negative_accepted": int(max_negative_accepted),
            "min_positive_layout_precision": float(min_positive_layout_precision),
            "min_positive_accepted": int(min_positive_accepted),
            "min_positive_precision_wilson_lower": float(min_positive_precision_wilson_lower),
            "wilson_confidence_z": 1.96,
            "selected_threshold_requires_zero_high_confidence_errors": False,
            "positive_task_errors_are_governed_by_precision": True,
        },
        "selected_threshold": selected,
        "selected_high_confidence_errors": selected_high_confidence_errors,
        "selected_error_summary": selected_errors,
        "deployment_allowed": deployment_allowed,
        "high_confidence_errors": high_confidence_errors(frame, limit=int(error_limit)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Markush layout expert confidence predictions.")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--max-negative-accepted", type=int, default=0)
    parser.add_argument("--min-positive-layout-precision", type=float, default=0.95)
    parser.add_argument("--min-positive-accepted", type=int, default=10)
    parser.add_argument("--min-positive-precision-wilson-lower", type=float, default=0.70)
    parser.add_argument("--error-limit", type=int, default=24)
    args = parser.parse_args()

    report = build_report(
        predictions_csv=args.predictions_csv,
        thresholds=parse_thresholds(str(args.thresholds)),
        ece_bins=int(args.ece_bins),
        max_negative_accepted=int(args.max_negative_accepted),
        min_positive_layout_precision=float(args.min_positive_layout_precision),
        min_positive_accepted=int(args.min_positive_accepted),
        min_positive_precision_wilson_lower=float(args.min_positive_precision_wilson_lower),
        error_limit=int(args.error_limit),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "markush_layout_confidence_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
