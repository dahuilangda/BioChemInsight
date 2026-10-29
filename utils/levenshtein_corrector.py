"""Weighted Levenshtein corrector for chemical OCR confusions .

Hand-tuned substitution costs for OCR-confusable character pairs (MolGrapher
abbreviation_detector.py pattern) plus a population-frequency prior from our
markush label corpus (R1: 35k, R2: 34k, R3: 24k, …). Used to correct OCR
misreads of R-group labels (R1→Rl, R5→RS, 0↔O, 8↔B) before isotope rewrite.
"""
from __future__ import annotations

import re

# Hand-tuned substitution costs (lower = more likely confusion).
# Source: MolGrapher abbreviation_detector.py L595-721 pattern, adapted
# for chemical R-label context.
_SUBSTITUTION_COSTS = {
    ("0", "O"): 0.1, ("O", "0"): 0.1,
    ("1", "l"): 0.1, ("l", "1"): 0.1,
    ("5", "S"): 0.1, ("S", "5"): 0.1,
    ("8", "B"): 0.1, ("B", "8"): 0.1,
    ("6", "G"): 0.3, ("G", "6"): 0.3,
    ("2", "Z"): 0.3, ("Z", "2"): 0.3,
    ("1", "I"): 0.2, ("I", "1"): 0.2,
    ("0", "D"): 0.4, ("D", "0"): 0.4,
    ("4", "A"): 0.5, ("A", "4"): 0.5,
}

# Population frequency prior over the extraction corpus.
# Labels are sorted by descending frequency; the corrector prefers common
# labels when Levenshtein distances tie.
_POPULATION_PRIOR = {
    1: 35047, 2: 34490, 3: 24094, 4: 16119, 5: 11631,
    6: 8476, 7: 6556, 8: 5348, 9: 4102, 10: 3202,
    11: 2600, 12: 2200, 13: 1800, 14: 1500, 15: 1200,
}


def _subst_cost(a: str, b: str) -> float:
    return _SUBSTITUTION_COSTS.get((a, b), 1.0 if a != b else 0.0)


def _levenshtein_weighted(a: str, b: str) -> float:
    """Weighted Levenshtein distance with chemical confusion costs."""
    m, n = len(a), len(b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,                      # delete
                    dp[i][j - 1] + 1,                      # insert
                    dp[i - 1][j - 1] + _subst_cost(a[i - 1], b[j - 1]),  # substitute
                )
    return dp[m][n]


def correct_r_label(raw: str, known_labels: list[int] | None = None,
                    max_distance: float = 1.5) -> int | None:
    """Correct an OCR-mangled R-label to the nearest known label.

    Args:
        raw: raw OCR text (e.g. 'Rl', 'RS', 'r1', 'R1a')
        known_labels: valid label numbers (from the document's scope);
            if None, uses the population prior.
        max_distance: maximum weighted Levenshtein distance.

    Returns:
        Corrected label number, or None if no known label is within range.
    """
    # Extract digits from raw text first — if already clean, pass through.
    digits = re.findall(r"\d+", str(raw or ""))
    if digits:
        number = int(digits[0])
        if known_labels is None or number in known_labels:
            return number
    # Normalize: strip non-alphanumeric, uppercase for comparison.
    clean = re.sub(r"[^A-Za-z0-9]", "", str(raw or "")).upper()
    if not clean:
        return None
    # Build candidate set from known labels or population prior.
    candidates = set(known_labels) if known_labels is not None else set(_POPULATION_PRIOR)
    best_label, best_score = None, float("inf")
    for number in candidates:
        candidate = f"R{number}"
        distance = _levenshtein_weighted(clean, candidate)
        # Population-frequency tiebreak: prefer common labels.
        frequency_bonus = _POPULATION_PRIOR.get(number, 0) / max(1, max(_POPULATION_PRIOR.values()))
        score = distance - frequency_bonus * 0.3
        if score < best_score:
            best_label, best_score = number, score
    if best_label is not None and best_score <= max_distance:
        return best_label
    return None


def batch_correct_r_labels(raw_labels: list[str | int],
                           known_labels: list[int] | None = None,
                           max_distance: float = 1.5) -> list[int | None]:
    """Correct a batch of raw label strings."""
    return [correct_r_label(raw, known_labels, max_distance) for raw in raw_labels]
