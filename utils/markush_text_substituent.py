from __future__ import annotations

import re

from utils.markush_labels import is_markush_label, normalize_label

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
VARIABLE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:R\s*\d+|R\s*[′']+|R[a-z]|Ar|Het|Hal|[XYZ]\d*|[A-Z]\d+|[ADEGJKLMQTUVWXYZ])(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)
ASSIGNMENT_RE = re.compile(
    rf"(?P<variables>{VARIABLE_TOKEN_RE.pattern}(?:\s*(?:,|/|and|or|及|和)\s*{VARIABLE_TOKEN_RE.pattern})*)"
    r"\s*(?:=|:|is|are|represents?|denotes?)\s*"
    r"(?P<value>[^;\n。]+)",
    flags=re.IGNORECASE,
)
VARIABLE_SPLIT_RE = re.compile(r"\s*(?:,|/|and|or|及|和)\s*", flags=re.IGNORECASE)

# NMR spectral data leaks into OCR markdown on patent pages that interleave a
# Markush table with characterization data. The coupling-constant notation
# "J = 3.7 Hz" is structurally indistinguishable from a Markush assignment
# "J = <substituent>", so ASSIGNMENT_RE captures it as a bogus variable "J".
# These signatures never appear in a real substituent definition, so rejecting
# the parsed value cleanly separates the two content types (not a fallback: a
# Markush substituent is objectively not an NMR peak list).
_NMR_SPECTRAL_VALUE_RE = re.compile(
    r"H\s*z"                       # Hz frequency unit (any spacing / LaTeX-split "H z")
    r"|\d+\s*H\s*\)"               # proton count 1H) 2H) 3H) — only inside NMR peak lists
    r"|J\s*=\s*\d"                 # coupling constant J=3.7
    r"|\b13C\b|\b19F\b|\b31P\b|\b15N\b"  # other NMR nuclei
    r"|δ\s*\d",                    # chemical-shift marker δ7.24
    flags=re.IGNORECASE,
)


def is_nmr_spectral_value(value: str) -> bool:
    """True when an assignment value is NMR characterization, not a substituent."""
    return bool(_NMR_SPECTRAL_VALUE_RE.search(str(value or "")))


def normalize_substituent_text(value: str) -> str:
    text = str(value or "").strip()
    text = text.translate(_SUBSCRIPT_MAP).translate(_SUPERSCRIPT_MAP)
    text = text.replace("′", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;()[]{}")
    text = re.sub(r"\s*:\s*\d+\s*$", "", text)
    text = re.sub(r"^(?:=|is|are)\s+", "", text, flags=re.IGNORECASE)
    if re.fullmatch(r"none|null|n/a|na|nd|-", text, flags=re.IGNORECASE):
        return ""
    return text


def normalize_variable_position(value: str) -> str:
    text = normalize_label(str(value or "").strip())
    text = text.translate(_SUBSCRIPT_MAP).translate(_SUPERSCRIPT_MAP)
    text = re.sub(r"\s+", "", text)
    if not text:
        return ""
    lowered = text.lower()
    if lowered == "ar":
        return "Ar"
    if lowered == "het":
        return "Het"
    if lowered == "hal":
        return "Hal"
    if re.fullmatch(r"r\d+", lowered):
        return "R" + lowered[1:]
    if re.fullmatch(r"r'+", lowered):
        return "R" + text[1:]
    if re.fullmatch(r"r[a-z]", lowered):
        return "R" + lowered[1:]
    if re.fullmatch(r"[a-z]\d*", lowered):
        candidate = lowered[0].upper() + lowered[1:]
        return candidate if is_markush_label(candidate) else ""
    normalized = text[0].upper() + text[1:] if text else ""
    return normalized if is_markush_label(normalized) else ""


def _split_variables(value: str) -> list[str]:
    variables = []
    for item in VARIABLE_SPLIT_RE.split(str(value or "")):
        normalized = normalize_variable_position(item)
        if normalized:
            variables.append(normalized)
    return list(dict.fromkeys(variables))


def _split_parallel_values(value: str, count: int) -> list[str] | None:
    if count <= 1:
        return None
    parts = [normalize_substituent_text(part) for part in re.split(r"\s*/\s*", str(value or ""))]
    parts = [part for part in parts if part]
    if len(parts) == count:
        return parts
    return None


def parse_assignment_line(line: str) -> list[dict]:
    assignments = []
    for match in ASSIGNMENT_RE.finditer(str(line or "")):
        variables = _split_variables(match.group("variables"))
        value = normalize_substituent_text(match.group("value"))
        if not value:
            continue
        # Reject NMR characterization (e.g. "J = 3.7 Hz, 1H), (s,1H)") that
        # ASSIGNMENT_RE mis-parses as a Markush variable assignment.
        if is_nmr_spectral_value(value):
            continue
        parallel_values = _split_parallel_values(value, len(variables))
        for index, variable in enumerate(variables):
            assignments.append(
                {
                    "variable_position": variable,
                    "substituent_text": parallel_values[index] if parallel_values else value,
                    "evidence_type": "text_assignment",
                }
            )
    return assignments
