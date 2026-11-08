"""Unified per-structure risk score (harness roadmap: replace fixed gate
ordering with one calibrated bucket that every gate can read).

The pipeline already tracks several independent signals per structure:
MolNexTR calibrated confidence, MoE routing confidence, detector (box) score,
quality-issue strings, and review verdicts. Gates historically hard-coded
their own threshold on one signal each; this module merges them into a single
0-1 risk with named buckets so gate policy becomes data, not scattered
constants.

risk = 1 - weighted_goodness, where each signal contributes evidence that the
decode is safe. Missing signals are neutral (0.5 goodness), never evidence of
safety — a structure with no detector score must not look safer than one with
a bad score.
"""
from __future__ import annotations

from typing import Any

# Signal weights sum to 1.0; confidence carries the most evidence because it
# is the calibrated E[Tanimoto] head, routing second (expert-fit mismatch is
# a strong error predictor), detector score third (localization only).
SIGNAL_WEIGHTS = {
    "confidence": 0.45,
    "routing": 0.25,
    "detector": 0.15,
    "issues": 0.15,
}

RISK_BUCKET_EDGES = {"low": 0.34, "medium": 0.67}  # risk <= edge -> bucket


def _clip01(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return max(0.0, min(1.0, value))


def _issue_goodness(issues: Any) -> float:
    """Any non-empty quality-issue string is a hard safety deduction; more
    distinct issues deduce more but never below 0."""
    text = str(issues or '').strip()
    if not text:
        return 1.0
    distinct = [part for part in text.split(';') if part.strip()]
    return max(0.0, 1.0 - 0.35 * len(distinct))


def compute_structure_risk(record: dict) -> dict:
    """Unified risk for one structure record (dict with the standard columns).

    Returns {'risk': float, 'bucket': 'low'|'medium'|'high',
             'signals': {name: (goodness|None, weight)}, 'missing': [names]}.
    """
    confidence = _clip01(record.get('MOLNEXTR_CONFIDENCE'))
    routing = _clip01(record.get('MOLNEXTR_ROUTING_CONFIDENCE'))
    detector = _clip01(record.get('DETECTOR_SCORE'))
    issue_goodness = _issue_goodness(record.get('MOLNEXTR_QUALITY_ISSUES'))

    signal_values = {
        "confidence": confidence,
        "routing": routing,
        "detector": detector,
        "issues": issue_goodness,
    }
    missing = [name for name, value in signal_values.items() if value is None]
    total_weight = 0.0
    weighted = 0.0
    for name, weight in SIGNAL_WEIGHTS.items():
        value = signal_values[name]
        if value is None:
            # Missing evidence is neutral 0.5 but its weight still counts, so
            # a record with many missing signals trends to mediocre goodness
            # rather than inheriting full confidence from the one present one.
            weighted += 0.5 * weight
        else:
            weighted += value * weight
        total_weight += weight
    goodness = weighted / total_weight
    risk = 1.0 - goodness
    # Escalation rules matching existing gate semantics: two or more distinct
    # quality issues, or confidence+routing both far below the cascade floors,
    # mean the decode is not trustworthy regardless of the blended score.
    distinct_issues = len([p for p in str(record.get('MOLNEXTR_QUALITY_ISSUES') or '').split(';') if p.strip()])
    if distinct_issues >= 2 or (
            confidence is not None and confidence < 0.60
            and routing is not None and routing < 0.80):
        bucket = "high"
    elif risk <= RISK_BUCKET_EDGES["low"]:
        bucket = "low"
    elif risk <= RISK_BUCKET_EDGES["medium"]:
        bucket = "medium"
    else:
        bucket = "high"
    return {
        "risk": round(risk, 4),
        "bucket": bucket,
        "signals": {name: (signal_values[name], weight)
                    for name, weight in SIGNAL_WEIGHTS.items()},
        "missing": missing,
    }


def annotate_records(records: list) -> dict:
    """Annotate a list of structure dicts in place with RISK_* columns and
    return bucket counts. Non-dict entries are skipped."""
    counts = {"low": 0, "medium": 0, "high": 0}
    for record in records or []:
        if not isinstance(record, dict):
            continue
        result = compute_structure_risk(record)
        record['STRUCTURE_RISK'] = result['risk']
        record['STRUCTURE_RISK_BUCKET'] = result['bucket']
        counts[result['bucket']] += 1
    return counts
