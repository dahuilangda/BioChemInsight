from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from training.molnextr_markush.tools.calibrate_markush_layout_expert import load_frame


CONFIDENCE_COLUMNS = [
    "presence_confidence",
    "count_confidence",
    "nominal_count_confidence",
    "corn_count_confidence",
    "nominal_positive_mass",
    "risk_confidence",
    "sidecar_confidence",
]
TASK_COLUMNS = [
    "layout_box_l1",
]
SUMMARY_PERCENTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
COUNT_BUCKET_ORDER = {"0": 0, "1": 1, "2": 2, "3-4": 3, "5-8": 4, "9+": 5}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def describe_by_presence(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for presence, group in frame.groupby("gold_markush_presence_int"):
        values = group[column].astype(float)
        if len(values) == 0:
            continue
        row: dict[str, Any] = {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std()) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
        for percentile in SUMMARY_PERCENTILES:
            row[f"p{int(percentile * 100):02d}"] = float(values.quantile(percentile))
        output[str(int(presence))] = row
    return output


def count_confusion(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return count_confusion_for(frame, "pred_count_bucket")


def count_confusion_for(frame: pd.DataFrame, prediction_column: str) -> dict[str, dict[str, int]]:
    positive = frame[frame["positive_row"]]
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in positive.to_dict(orient="records"):
        table[str(row.get("gold_count_bucket", ""))][str(row.get(prediction_column, ""))] += 1
    return {gold: dict(pred_counts) for gold, pred_counts in sorted(table.items())}


def count_ordinal_distance(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    positive = frame[frame["positive_row"]].copy()
    if len(positive) == 0 or prediction_column not in positive.columns:
        return {"available": False}
    gold = positive["gold_count_bucket"].astype(str).map(COUNT_BUCKET_ORDER)
    pred = positive[prediction_column].astype(str).map(COUNT_BUCKET_ORDER)
    valid = gold.notna() & pred.notna()
    if not bool(valid.any()):
        return {"available": False}
    distance = (pred[valid].astype(int) - gold[valid].astype(int)).abs()
    signed = pred[valid].astype(int) - gold[valid].astype(int)
    by_gold = {}
    for bucket, indices in positive[valid].groupby("gold_count_bucket").groups.items():
        bucket_distance = distance.loc[list(indices)]
        bucket_signed = signed.loc[list(indices)]
        by_gold[str(bucket)] = {
            "rows": int(len(bucket_distance)),
            "mean_abs_distance": float(bucket_distance.mean()),
            "max_abs_distance": int(bucket_distance.max()),
            "mean_signed_distance": float(bucket_signed.mean()),
        }
    return {
        "available": True,
        "rows": int(len(distance)),
        "mean_abs_distance": float(distance.mean()),
        "max_abs_distance": int(distance.max()),
        "exact_accuracy": float((distance == 0).mean()),
        "within_one_bucket_accuracy": float((distance <= 1).mean()),
        "mean_signed_distance": float(signed.mean()),
        "by_gold_bucket": by_gold,
    }


def presence_confusion(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in frame.to_dict(orient="records"):
        table[str(row.get("gold_markush_presence", ""))][str(row.get("pred_markush_presence", ""))] += 1
    return {gold: dict(pred_counts) for gold, pred_counts in sorted(table.items())}


def top_rows(frame: pd.DataFrame, *, positive: bool, limit: int) -> list[dict[str, Any]]:
    keep = [
        "row_id",
        "source_id",
        "source_arrow",
        "gold_markush_presence",
        "pred_markush_presence",
        "gold_count_bucket",
        "pred_count_bucket",
        "pred_variable_count",
        "presence_confidence",
        "count_confidence",
        "risk_confidence",
        "sidecar_confidence",
        "layout_box_l1",
        "accept_correct",
    ]
    subset = frame[frame["positive_row"] == positive].sort_values("sidecar_confidence", ascending=False)
    present = [column for column in keep if column in subset.columns]
    rows = []
    for row in subset[present].head(limit).to_dict(orient="records"):
        cleaned = {}
        for key, value in row.items():
            if pd.isna(value):
                cleaned[key] = None
            elif key.endswith("confidence") or key == "layout_box_l1":
                cleaned[key] = finite_float(value)
            else:
                cleaned[key] = value
        rows.append(cleaned)
    return rows


def sidecar_overlap(frame: pd.DataFrame) -> dict[str, Any]:
    positives = frame[frame["positive_row"]]
    emitted = (frame["pred_markush_presence_int"] == 1) & (frame["pred_count_bucket"].astype(str) != "0")
    emitted_negatives = frame[(~frame["positive_row"]) & emitted]
    emitted_positives = positives[emitted.loc[positives.index]]
    if len(emitted_negatives) == 0:
        max_negative = None
        positive_above_max_negative = int(len(emitted_positives))
        correct_positive_above_max_negative = int(emitted_positives["markush_accept_correct"].sum())
    else:
        max_negative = float(emitted_negatives["sidecar_confidence"].astype(float).max())
        above = emitted_positives["sidecar_confidence"].astype(float) > max_negative
        positive_above_max_negative = int(above.sum())
        correct_positive_above_max_negative = int((above & emitted_positives["markush_accept_correct"]).sum())
    correct_positive_total = int((positives["markush_accept_correct"]).sum())
    return {
        "max_negative_sidecar_confidence": max_negative,
        "positive_above_max_negative": positive_above_max_negative,
        "correct_positive_above_max_negative": correct_positive_above_max_negative,
        "correct_positive_total": correct_positive_total,
        "positive_rows": int(len(positives)),
        "negative_rows": int((~frame["positive_row"]).sum()),
        "emitted_positive_rows": int(len(emitted_positives)),
        "emitted_negative_rows": int(len(emitted_negatives)),
        "selection_semantics": "pred_markush_presence=1, pred_count_bucket!=0, and sidecar_confidence threshold",
    }


def zero_negative_threshold_summary(confidence_report: dict[str, Any]) -> dict[str, Any]:
    curve = confidence_report.get("threshold_curve")
    if not isinstance(curve, list):
        return {"available": False, "max_coverage": None, "max_precision": None}
    candidates = [row for row in curve if int(row.get("negative_accepted") or 0) == 0]
    if not candidates:
        return {"available": False, "max_coverage": None, "max_precision": None}
    max_coverage = sorted(
        candidates,
        key=lambda row: (
            -int(row.get("positive_accepted") or 0),
            -float(row.get("positive_layout_precision") or 0.0),
            float(row.get("threshold") or 0.0),
        ),
    )[0]
    max_precision = sorted(
        candidates,
        key=lambda row: (
            -float(row.get("positive_layout_precision") or 0.0),
            -int(row.get("positive_accepted") or 0),
            float(row.get("threshold") or 0.0),
        ),
    )[0]
    return {
        "available": True,
        "max_coverage": max_coverage,
        "max_precision": max_precision,
    }


def build_report(
    *,
    predictions_csv: str | Path,
    confidence_report: str | Path | None,
    top_limit: int,
) -> dict[str, Any]:
    frame = load_frame(predictions_csv)
    for column in TASK_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].map(finite_float)
    for column in ["gold_markush_presence", "pred_markush_presence", "pred_variable_count", "accept_correct"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: int(finite_float(value)))
    confidence_payload: dict[str, Any] = {}
    if confidence_report:
        confidence_payload = json.loads(Path(confidence_report).read_text(encoding="utf-8"))
        if not isinstance(confidence_payload, dict):
            raise ValueError(f"{confidence_report} must contain a JSON object")

    sidecar_emitted = (frame["pred_markush_presence_int"] == 1) & (frame["pred_count_bucket"].astype(str) != "0")
    negative_false_accepts = int(((frame["gold_markush_presence_int"] == 0) & sidecar_emitted).sum())
    positive = frame[frame["positive_row"]]
    report = {
        "predictions_csv": str(predictions_csv),
        "confidence_report": str(confidence_report or ""),
        "rows": int(len(frame)),
        "positive_rows": int(frame["positive_row"].sum()),
        "negative_rows": int((~frame["positive_row"]).sum()),
        "negative_false_accepts": negative_false_accepts,
        "positive_accept_correct": int(positive["markush_accept_correct"].sum()),
        "positive_accept_precision_ceiling_without_negatives": (
            int(positive["markush_accept_correct"].sum()) / max(1, int(len(positive)))
        ),
        "presence_confusion": presence_confusion(frame),
        "count_confusion_positive_rows": count_confusion(frame),
        "count_confusions_by_head": {
            column: count_confusion_for(frame, column)
            for column in [
                "pred_count_bucket",
                "pred_nominal_count_bucket",
                "pred_corn_count_bucket",
                "pred_query_count_bucket",
            ]
            if column in frame.columns
        },
        "count_ordinal_distance": {
            column: count_ordinal_distance(frame, column)
            for column in [
                "pred_count_bucket",
                "pred_nominal_count_bucket",
                "pred_corn_count_bucket",
                "pred_query_count_bucket",
            ]
            if column in frame.columns
        },
        "score_overlap": sidecar_overlap(frame),
        "score_descriptions": {
            column: describe_by_presence(frame, column)
            for column in [*CONFIDENCE_COLUMNS, *TASK_COLUMNS]
            if column in frame.columns
        },
        "top_negative_sidecar_scores": top_rows(frame, positive=False, limit=top_limit),
        "top_positive_sidecar_scores": top_rows(frame, positive=True, limit=top_limit),
        "calibrated_selected_threshold": confidence_payload.get("selected_threshold") if confidence_payload else None,
        "zero_negative_threshold_summary": (
            zero_negative_threshold_summary(confidence_payload)
            if confidence_payload
            else {"available": False, "max_coverage": None, "max_precision": None}
        ),
        "diagnosis": [],
    }
    overlap = report["score_overlap"]
    if int(overlap["positive_above_max_negative"]) == 0:
        report["diagnosis"].append(
            "No positive row scores above the highest ordinary-negative sidecar score; zero-negative selective calibration cannot accept any positive row."
        )
    if negative_false_accepts:
        report["diagnosis"].append(
            f"Sidecar emission policy false-accepted {negative_false_accepts} ordinary negatives before calibration."
        )
    if report["positive_accept_correct"] < report["positive_rows"]:
        report["diagnosis"].append(
            "Positive task correctness is limited by count/layout errors before confidence selection."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Markush confidence/count failures from an eval prediction CSV.")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--confidence-report", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-limit", type=int, default=24)
    args = parser.parse_args()

    report = build_report(
        predictions_csv=args.predictions_csv,
        confidence_report=args.confidence_report or None,
        top_limit=int(args.top_limit),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
