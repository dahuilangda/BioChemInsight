from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.isotonic import IsotonicRegression


CONFIDENCE_COLUMNS = [
    "applicability_confidence",
    "presence_confidence",
    "side_confidence",
    "anchor_confidence",
    "query_objectness_confidence",
    "no_endpoint_confidence",
    "risk_confidence",
    "sidecar_confidence",
]

DEFAULT_THRESHOLDS = "0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.5,0.7,0.9"


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


def load_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, low_memory=False)
    missing_columns = [column for column in CONFIDENCE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Prediction CSV is missing required confidence columns: {missing_columns}")
    for column in CONFIDENCE_COLUMNS:
        frame[column] = frame[column].map(as_float)
    frame["gold_presence_int"] = frame["gold_presence"].map(as_int)
    frame["presence_correct"] = frame["gold_presence"].astype(str) == frame["pred_presence"].astype(str)
    frame["side_correct"] = frame["gold_side"].astype(str) == frame["pred_side"].astype(str)
    frame["anchor_correct"] = frame["gold_anchor"].astype(str) == frame["pred_anchor"].astype(str)
    frame["positive_row"] = frame["gold_presence_int"] == 1
    frame["endpoint_triplet_correct"] = (
        frame["positive_row"] & frame["presence_correct"] & frame["side_correct"] & frame["anchor_correct"]
    )
    frame["sidecar_accept_correct"] = (
        (~frame["positive_row"] & (frame["pred_presence"].astype(str) == "0"))
        | frame["endpoint_triplet_correct"]
    )
    return frame


def binary_brier(scores: pd.Series, labels: pd.Series) -> float:
    if len(scores) == 0:
        return 0.0
    return float(((scores.astype(float) - labels.astype(float)) ** 2).mean())


def ece(scores: pd.Series, labels: pd.Series, *, bins: int) -> tuple[float, list[dict[str, Any]]]:
    total = max(1, len(scores))
    rows = []
    value = 0.0
    for index in range(bins):
        lo = index / bins
        hi = (index + 1) / bins
        if index == bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
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


def confidence_report(frame: pd.DataFrame, *, bins: int) -> dict[str, Any]:
    label_map = {
        "applicability_confidence": frame["positive_row"].astype(float),
        "presence_confidence": frame["positive_row"].astype(float),
        "side_confidence": (frame["positive_row"] & frame["side_correct"]).astype(float),
        "anchor_confidence": (frame["positive_row"] & frame["anchor_correct"]).astype(float),
        "query_objectness_confidence": frame["endpoint_triplet_correct"].astype(float),
        "no_endpoint_confidence": (~frame["positive_row"] & (frame["pred_presence"].astype(str) == "0")).astype(float),
        "risk_confidence": frame["sidecar_accept_correct"].astype(float),
        "sidecar_confidence": frame["endpoint_triplet_correct"].astype(float),
    }
    output = {}
    for column, labels in label_map.items():
        score = frame[column].astype(float)
        ece_value, bins_rows = ece(score, labels, bins=bins)
        output[column] = {
            "mean_score": float(score.mean()) if len(score) else 0.0,
            "brier": binary_brier(score, labels),
            "ece": ece_value,
            "reliability_bins": bins_rows,
        }
    return output


def apply_isotonic_sidecar_calibration(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = frame["endpoint_triplet_correct"].astype(float)
    raw_scores = frame["sidecar_confidence"].astype(float)
    positive_labels = int(labels.sum())
    negative_labels = int(len(labels) - positive_labels)
    if positive_labels <= 0 or negative_labels <= 0:
        raise ValueError(
            "Isotonic sidecar calibration requires both correct and incorrect triplet labels in the calibration split."
        )
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrated_scores = model.fit_transform(raw_scores.to_numpy(), labels.to_numpy())
    calibrated = frame.copy()
    calibrated["raw_sidecar_confidence"] = raw_scores
    calibrated["sidecar_confidence"] = calibrated_scores
    thresholds_x = [float(value) for value in model.X_thresholds_.tolist()]
    thresholds_y = [float(value) for value in model.y_thresholds_.tolist()]
    return calibrated, {
        "method": "isotonic_regression",
        "library": "sklearn.isotonic.IsotonicRegression",
        "fit_score_column": "raw_sidecar_confidence",
        "target_column": "endpoint_triplet_correct",
        "calibrated_score_column": "sidecar_confidence",
        "fit_rows": int(len(frame)),
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
        "out_of_bounds": "clip",
        "threshold_count": int(len(thresholds_x)),
        "x_thresholds": thresholds_x,
        "y_thresholds": thresholds_y,
        "monotonic_probability_mapping": True,
    }


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


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = phat + z2 / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * total)) / total)
    return max(0.0, (centre - margin) / denominator)


def evaluate_thresholds(frame: pd.DataFrame, thresholds: list[float]) -> list[dict[str, Any]]:
    rows = []
    positive_count = int(frame["positive_row"].sum())
    for threshold in thresholds:
        accepted = frame["sidecar_confidence"] >= float(threshold)
        accepted_frame = frame[accepted]
        accepted_count = int(len(accepted_frame))
        negative_accepted = int((~accepted_frame["positive_row"]).sum())
        negative_predicted_attachment = int(
            ((~accepted_frame["positive_row"]) & (accepted_frame["pred_presence"].astype(str) == "1")).sum()
        )
        positive_accepted = accepted_frame[accepted_frame["positive_row"]]
        triplet_correct = int(positive_accepted["endpoint_triplet_correct"].sum())
        positive_accepted_count = int(len(positive_accepted))
        correct = int(accepted_frame["sidecar_accept_correct"].sum())
        rows.append(
            {
                "threshold": float(threshold),
                "accepted": accepted_count,
                "coverage": accepted_count / len(frame) if len(frame) else 0.0,
                "correct": correct,
                "precision": correct / accepted_count if accepted_count else 0.0,
                "negative_accepted": negative_accepted,
                "negative_accepted_rate": negative_accepted / int((~frame["positive_row"]).sum())
                if int((~frame["positive_row"]).sum())
                else 0.0,
                "negative_predicted_attachment": negative_predicted_attachment,
                "negative_predicted_attachment_rate": negative_predicted_attachment
                / int((~frame["positive_row"]).sum())
                if int((~frame["positive_row"]).sum())
                else 0.0,
                "positive_accepted": positive_accepted_count,
                "positive_triplet_correct": triplet_correct,
                "positive_triplet_precision": triplet_correct / positive_accepted_count
                if positive_accepted_count
                else 0.0,
                "positive_triplet_precision_wilson_lower": wilson_lower_bound(
                    triplet_correct,
                    positive_accepted_count,
                ),
                "positive_triplet_recall": triplet_correct / positive_count if positive_count else 0.0,
            }
        )
    return rows


def select_threshold(
    threshold_rows: list[dict[str, Any]],
    *,
    max_false_accept: int,
    max_negative_accepted: int,
    min_positive_triplet_precision: float,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in threshold_rows
        if int(row["negative_predicted_attachment"]) <= max_false_accept
        and int(row["negative_accepted"]) <= max_negative_accepted
        and float(row["positive_triplet_precision"]) >= min_positive_triplet_precision
    ]
    candidates.sort(
        key=lambda row: (
            -float(row["positive_triplet_recall"]),
            -float(row["coverage"]),
            float(row["threshold"]),
        )
    )
    return candidates[0] if candidates else None


def confusion(frame: pd.DataFrame) -> dict[str, Any]:
    presence = defaultdict(Counter)
    side = defaultdict(Counter)
    anchor = defaultdict(Counter)
    for row in frame.to_dict(orient="records"):
        presence[str(row.get("gold_presence") or "")][str(row.get("pred_presence") or "")] += 1
        if str(row.get("gold_side") or ""):
            side[str(row.get("gold_side") or "")][str(row.get("pred_side") or "")] += 1
        if str(row.get("gold_anchor") or ""):
            anchor[str(row.get("gold_anchor") or "")][str(row.get("pred_anchor") or "")] += 1
    return {
        "presence": {key: dict(value) for key, value in sorted(presence.items())},
        "side": {key: dict(value) for key, value in sorted(side.items())},
        "anchor": {key: dict(value) for key, value in sorted(anchor.items())},
    }


def high_confidence_errors(frame: pd.DataFrame, *, limit: int) -> list[dict[str, Any]]:
    errors = frame[~frame["sidecar_accept_correct"]].sort_values("sidecar_confidence", ascending=False)
    keep = [
        "row_id",
        "source_id",
        "source_arrow",
        "smiles",
        "gold_presence",
        "pred_presence",
        "gold_side",
        "pred_side",
        "gold_anchor",
        "pred_anchor",
        "applicability_confidence",
        "presence_confidence",
        "side_confidence",
        "anchor_confidence",
        "query_objectness_confidence",
        "no_endpoint_confidence",
        "sidecar_confidence",
    ]
    return errors[keep].head(limit).to_dict(orient="records")


def high_confidence_error_count(frame: pd.DataFrame, *, threshold: float) -> int:
    accepted = frame["sidecar_confidence"] >= float(threshold)
    return int((accepted & ~frame["sidecar_accept_correct"]).sum())


def selected_error_summary(frame: pd.DataFrame, *, threshold: float | None) -> dict[str, Any] | None:
    if threshold is None:
        return None
    accepted = frame["sidecar_confidence"] >= float(threshold)
    positive = frame["positive_row"]
    negative = ~positive
    positive_task_error = accepted & positive & ~frame["endpoint_triplet_correct"]
    negative_accept_error = accepted & negative
    negative_attachment_error = accepted & negative & (frame["pred_presence"].astype(str) == "1")
    return {
        "total_accepted_errors": int((accepted & ~frame["sidecar_accept_correct"]).sum()),
        "positive_triplet_errors": int(positive_task_error.sum()),
        "negative_accepted_errors": int(negative_accept_error.sum()),
        "negative_predicted_attachment_errors": int(negative_attachment_error.sum()),
        "deployment_blocking_error_classes": [
            "negative_accepted_errors",
            "negative_predicted_attachment_errors",
        ],
        "audited_non_blocking_error_classes": [
            "positive_triplet_errors",
        ],
    }


def build_report(
    *,
    predictions_csv: str | Path,
    thresholds: list[float],
    ece_bins: int,
    max_false_accept: int,
    max_negative_accepted: int,
    min_positive_triplet_precision: float,
    error_limit: int,
) -> dict[str, Any]:
    raw_frame = load_frame(predictions_csv)
    frame, sidecar_calibration = apply_isotonic_sidecar_calibration(raw_frame)
    threshold_candidates = score_adaptive_thresholds(frame, thresholds)
    threshold_rows = evaluate_thresholds(frame, threshold_candidates)
    selected = select_threshold(
        threshold_rows,
        max_false_accept=max_false_accept,
        max_negative_accepted=max_negative_accepted,
        min_positive_triplet_precision=min_positive_triplet_precision,
    )
    selected_threshold = float(selected["threshold"]) if selected else None
    selected_high_confidence_errors = (
        high_confidence_error_count(frame, threshold=selected_threshold) if selected_threshold is not None else None
    )
    selected_errors = selected_error_summary(frame, threshold=selected_threshold)
    confidence = confidence_report(frame, bins=int(ece_bins))
    deployment_allowed = bool(
        selected
        and int(selected["negative_predicted_attachment"]) <= int(max_false_accept)
        and int(selected["negative_accepted"]) <= int(max_negative_accepted)
        and float(selected["positive_triplet_precision"]) >= float(min_positive_triplet_precision)
    )
    return {
        "predictions_csv": str(predictions_csv),
        "rows": int(len(frame)),
        "positive_rows": int(frame["positive_row"].sum()),
        "negative_rows": int((~frame["positive_row"]).sum()),
        "confidence_design": {
            "inspired_by": "Protenix-style separation of task heads, confidence heads, calibration, and selective output; not a copied protein architecture.",
            "post_hoc_calibration": (
                "isotonic regression fitted on the routed calibration split maps raw selective sidecar scores "
                "to calibrated triplet-correct probabilities before ECE/Brier and risk-coverage selection"
            ),
            "heads": [
                "applicability_confidence",
                "presence_confidence",
                "side_confidence",
                "anchor_confidence",
                "query_objectness_confidence",
                "no_endpoint_confidence",
                "risk_confidence",
                "selective_sidecar_confidence",
            ],
            "selection_policy": "selective classification: emit sidecar only when calibrated risk/coverage threshold satisfies ordinary-negative false-accept and positive triplet precision constraints",
            "deployment_rule": "fragment attachment output may emit only after selective router and all calibrated expert risk gates pass; otherwise keep the original MolNexTR result or abstain",
            "threshold_policy": "fixed thresholds plus score-adaptive calibration candidates, including the next representable value above the highest ordinary-negative score",
        },
        "sidecar_calibration": sidecar_calibration,
        "confusion": confusion(frame),
        "confidence": confidence,
        "threshold_curve": threshold_rows,
        "threshold_candidate_count": len(threshold_candidates),
        "fixed_thresholds": [float(value) for value in thresholds],
        "selected_threshold": selected,
        "selected_high_confidence_errors": selected_high_confidence_errors,
        "selected_error_summary": selected_errors,
        "deployment_allowed": deployment_allowed,
        "micro_training_allowed": True,
        "quality_constraints": {
            "max_negative_predicted_attachment": int(max_false_accept),
            "max_negative_accepted": int(max_negative_accepted),
            "min_positive_triplet_precision": float(min_positive_triplet_precision),
            "selected_threshold_requires_zero_high_confidence_errors": False,
            "positive_task_errors_are_governed_by_precision": True,
        },
        "high_confidence_errors": high_confidence_errors(frame, limit=int(error_limit)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate and risk-scan fragment attachment expert predictions."
    )
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--max-false-accept", type=int, default=0)
    parser.add_argument("--max-negative-accepted", type=int, default=0)
    parser.add_argument("--min-positive-triplet-precision", type=float, default=0.95)
    parser.add_argument("--error-limit", type=int, default=24)
    args = parser.parse_args()

    report = build_report(
        predictions_csv=args.predictions_csv,
        thresholds=parse_thresholds(args.thresholds),
        ece_bins=int(args.ece_bins),
        max_false_accept=int(args.max_false_accept),
        max_negative_accepted=int(args.max_negative_accepted),
        min_positive_triplet_precision=float(args.min_positive_triplet_precision),
        error_limit=int(args.error_limit),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fragment_attachment_confidence_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
