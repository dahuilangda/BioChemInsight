"""Confidence calibration for the Markush layout expert.

Reads the per-row prediction CSV emitted by ``evaluate()`` in
``train_markush_layout_expert.py`` and selects a deployable sidecar
threshold on ``sidecar_confidence`` that maximises positive-layout
recall subject to hard negative-acceptance and precision floors.

The layout expert differs from the fragment-attachment expert: there is
no side/anchor triplet.  A row is *positive* when ``gold_markush_presence``
is 1, and *layout-correct* when ``accept_correct`` is 1 (presence match,
count-bucket match, and box-error within tolerance).
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.isotonic import IsotonicRegression


CONFIDENCE_COLUMNS = [
    "presence_confidence",
    "count_confidence",
    "nominal_count_confidence",
    "corn_count_confidence",
    "query_count_confidence",
    "risk_confidence",
    "sidecar_confidence",
]

DEFAULT_THRESHOLDS = "0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.5,0.7,0.9"


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_thresholds(value: str) -> list[float]:
    """Parse a comma-separated threshold string into a sorted float list."""
    return [float(part) for part in str(value).split(",") if part.strip()]


def load_frame(path: str | Path) -> pd.DataFrame:
    """Load the prediction CSV and annotate derived boolean columns."""
    frame = pd.read_csv(path, dtype=str, low_memory=False)
    missing = [column for column in CONFIDENCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing required confidence columns: {missing}")
    for column in CONFIDENCE_COLUMNS:
        frame[column] = frame[column].map(_as_float)
    frame["gold_presence_int"] = frame["gold_markush_presence"].map(_as_int)
    frame["pred_presence_int"] = frame["pred_markush_presence"].map(_as_int)
    frame["accept_correct_int"] = frame["accept_correct"].map(_as_int)
    frame["positive_row"] = frame["gold_presence_int"] == 1
    frame["layout_correct"] = frame["accept_correct_int"] == 1
    return frame


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #

def binary_brier(scores: pd.Series, labels: pd.Series) -> float:
    if len(scores) == 0:
        return 0.0
    return float(((scores.astype(float) - labels.astype(float)) ** 2).mean())


def ece(scores: pd.Series, labels: pd.Series, *, bins: int) -> tuple[float, list[dict[str, Any]]]:
    """Expected Calibration Error with uniform-width bins."""
    total = max(1, len(scores))
    rows: list[dict[str, Any]] = []
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
            continue
        mean_score = float(scores[mask].mean())
        mean_label = float(labels[mask].mean())
        gap = abs(mean_score - mean_label)
        value += gap * count / total
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": count,
                "mean_score": mean_score,
                "mean_label": mean_label,
                "abs_gap": gap,
            }
        )
    return value, rows


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = phat + z2 / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * total)) / total)
    return max(0.0, (centre - margin) / denominator)


def apply_isotonic_sidecar_calibration(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit isotonic regression on sidecar_confidence against layout correctness."""
    positive = frame[frame["positive_row"]].copy()
    if len(positive) < 10:
        return frame, {"applied": False, "reason": "insufficient positive samples"}
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(positive["sidecar_confidence"], positive["layout_correct"].astype(float))
    calibrated = frame.copy()
    calibrated["sidecar_confidence"] = iso.predict(calibrated["sidecar_confidence"])
    return calibrated, {
        "applied": True,
        "x_thresholds": iso.X_thresholds_.tolist(),
        "y_thresholds": iso.y_thresholds_.tolist(),
        "monotonic_probability_mapping": True,
    }


# --------------------------------------------------------------------------- #
# Threshold evaluation
# --------------------------------------------------------------------------- #

def evaluate_thresholds(frame: pd.DataFrame, thresholds: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positive_count = int(frame["positive_row"].sum())
    for threshold in thresholds:
        accepted = frame["sidecar_confidence"] >= float(threshold)
        accepted_frame = frame[accepted]
        accepted_count = int(len(accepted_frame))
        negative_accepted = int((~accepted_frame["positive_row"]).sum())
        positive_accepted = accepted_frame[accepted_frame["positive_row"]]
        layout_correct = int(positive_accepted["layout_correct"].sum())
        positive_accepted_count = int(len(positive_accepted))
        rows.append(
            {
                "threshold": float(threshold),
                "accepted": accepted_count,
                "coverage": accepted_count / len(frame) if len(frame) else 0.0,
                "negative_accepted": negative_accepted,
                "negative_accepted_rate": negative_accepted / max(1, int((~frame["positive_row"]).sum())),
                "positive_accepted": positive_accepted_count,
                "positive_layout_correct": layout_correct,
                "positive_layout_precision": layout_correct / positive_accepted_count if positive_accepted_count else 0.0,
                "positive_layout_precision_wilson_lower": wilson_lower_bound(layout_correct, positive_accepted_count),
                "positive_layout_recall": layout_correct / positive_count if positive_count else 0.0,
            }
        )
    return rows


def select_threshold(
    threshold_rows: list[dict[str, Any]],
    *,
    max_negative_accepted: int,
    min_positive_layout_precision: float,
    min_positive_accepted: int,
    min_positive_precision_wilson_lower: float,
) -> dict[str, Any] | None:
    """Pick the highest-recall threshold meeting all deployment gates."""
    candidates = [
        row
        for row in threshold_rows
        if int(row["negative_accepted"]) <= max_negative_accepted
        and int(row["positive_accepted"]) >= min_positive_accepted
        and float(row["positive_layout_precision"]) >= min_positive_layout_precision
        and float(row["positive_layout_precision_wilson_lower"]) >= min_positive_precision_wilson_lower
    ]
    candidates.sort(
        key=lambda row: (
            -float(row["positive_layout_recall"]),
            -float(row["coverage"]),
            float(row["threshold"]),
        )
    )
    return candidates[0] if candidates else None


# --------------------------------------------------------------------------- #
# Confusion + error summaries
# --------------------------------------------------------------------------- #

def confusion(frame: pd.DataFrame) -> dict[str, Any]:
    presence: dict[str, Counter] = defaultdict(Counter)
    for record in frame.to_dict(orient="records"):
        presence[str(record.get("gold_markush_presence") or "")][
            str(record.get("pred_markush_presence") or "")
        ] += 1
    return {"markush_presence": {key: dict(value) for key, value in sorted(presence.items())}}


def high_confidence_errors(frame: pd.DataFrame, *, limit: int) -> list[dict[str, Any]]:
    errors = frame[~frame["layout_correct"] | (frame["pred_presence_int"] != frame["gold_presence_int"])]
    errors = errors.sort_values("sidecar_confidence", ascending=False)
    keep = [
        "row_id",
        "source_id",
        "source_arrow",
        "smiles",
        "gold_markush_presence",
        "pred_markush_presence",
        "gold_count_bucket",
        "pred_count_bucket",
        "layout_box_l1",
        "sidecar_confidence",
        "presence_confidence",
        "count_confidence",
        "risk_confidence",
    ]
    available = [column for column in keep if column in errors.columns]
    return errors[available].head(limit).to_dict(orient="records")


# --------------------------------------------------------------------------- #
# Top-level report builder
# --------------------------------------------------------------------------- #

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
    """Build the full calibration report for checkpoint selection."""
    raw_frame = load_frame(predictions_csv)
    frame, sidecar_calibration = apply_isotonic_sidecar_calibration(raw_frame)

    positive_frame = frame[frame["positive_row"]]
    negative_frame = frame[~frame["positive_row"]]

    presence_brier = binary_brier(frame["presence_confidence"], frame["gold_presence_int"].astype(float))
    sidecar_brier = binary_brier(
        frame["sidecar_confidence"],
        (frame["positive_row"] & frame["layout_correct"]).astype(float),
    )
    presence_ece, presence_ece_bins = ece(
        positive_frame["presence_confidence"], positive_frame["layout_correct"].astype(float), bins=ece_bins
    )
    sidecar_ece, sidecar_ece_bins = ece(
        frame["sidecar_confidence"],
        (frame["positive_row"] & frame["layout_correct"]).astype(float),
        bins=ece_bins,
    )

    threshold_curve = evaluate_thresholds(frame, thresholds)
    selected = select_threshold(
        threshold_curve,
        max_negative_accepted=max_negative_accepted,
        min_positive_layout_precision=min_positive_layout_precision,
        min_positive_accepted=min_positive_accepted,
        min_positive_precision_wilson_lower=min_positive_precision_wilson_lower,
    )

    return {
        "rows": int(len(frame)),
        "positive_rows": int(frame["positive_row"].sum()),
        "negative_rows": int((~frame["positive_row"]).sum()),
        "presence_accuracy": float(
            (frame["pred_presence_int"] == frame["gold_presence_int"]).mean()
        ) if len(frame) else 0.0,
        "layout_accuracy_on_positive": float(positive_frame["layout_correct"].mean()) if len(positive_frame) else 0.0,
        "negative_false_accepts": int(
            ((frame["gold_presence_int"] == 0) & (frame["pred_presence_int"] == 1)).sum()
        ),
        "sidecar_calibration": sidecar_calibration,
        "presence_brier": presence_brier,
        "sidecar_brier": sidecar_brier,
        "presence_ece": presence_ece,
        "presence_ece_bins": presence_ece_bins,
        "sidecar_ece": sidecar_ece,
        "sidecar_ece_bins": sidecar_ece_bins,
        "threshold_curve": threshold_curve,
        "selected_threshold": selected,
        "confusion": confusion(frame),
        "high_confidence_errors": high_confidence_errors(frame, limit=error_limit),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-negative-accepted", type=int, default=0)
    parser.add_argument("--min-positive-layout-precision", type=float, default=0.95)
    parser.add_argument("--min-positive-accepted", type=int, default=50)
    parser.add_argument("--min-positive-precision-wilson-lower", type=float, default=0.90)
    parser.add_argument("--error-limit", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(
        predictions_csv=args.predictions_csv,
        thresholds=parse_thresholds(args.thresholds),
        ece_bins=int(args.ece_bins),
        max_negative_accepted=int(args.max_negative_accepted),
        min_positive_layout_precision=float(args.min_positive_layout_precision),
        min_positive_accepted=int(args.min_positive_accepted),
        min_positive_precision_wilson_lower=float(args.min_positive_precision_wilson_lower),
        error_limit=int(args.error_limit),
    )
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
