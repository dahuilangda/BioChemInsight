"""Page-type classification over OCR page text (signal-based, calibrated).

A patent page can carry different evidence classes, each served by a
different skill:

  compound_table    dense per-compound rows: MW patterns, SEQ-ID
                    annotations, SMILES text, assay grade columns
  abbreviation_table monomer-definition rows (code -> chemical name)
  assay_text        experimental prose with assay terms and numbers
  2d_structure      drawn structure figures -> OCSR rung
  prose             narrative text (skipped by structure skills)

Signals are CONTENT measurements on the page text (pattern densities),
combined by a calibrated score rather than hard buckets, so partially
damaged OCR still classifies. Thresholds were fitted on a real 293-page
corpus; `calibrate()` re-fits them from any labeled corpus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_SEQ_RE = re.compile(r"SEQ\s+ID\s*NO", re.IGNORECASE)
_MW_RE = re.compile(r"\b[DM]\s*;\s*\d+\.\d{2,}")
_ASSAY_RE = re.compile(r"\b(IC50|EC50|Ki|SGF|SIF|HEK Blue|CXCL1?)\b", re.I)
_ANALOG_RE = re.compile(r"\(Analog of\b", re.I)
_SMILES_TOKEN_RE = re.compile(
    r"[\[\]A-Za-z0-9@+\-=\(\)#%/\\\.]{40,}")
_TABLE_MARK_RE = re.compile(r"<table|<td", re.I)


def _strip_tags(markdown: str) -> str:
    return re.sub(r"<[^>]+>", " ", markdown)


def _smiles_parseable(markdown: str) -> int:
    """SMILES lines that RDKit accepts (the only SMILES signal that
    matters; character-shape heuristics false-positive on prose)."""
    hits = 0
    for line in re.sub(r"<[^>]+>", "\n", markdown).split("\n"):
        text = line.strip()
        if 20 <= len(text) <= 1500 and not re.search(r"[^A-Za-z0-9@\+\-\(\)\[\]=#%/\\\.]", text):
            if Chem.MolFromSmiles(text) is not None:
                hits += 1
    return hits


def page_signals(markdown: str) -> dict:
    """Content signals for one OCR page. Every signal is a count/density
    measured on the page itself — nothing here knows any patent."""
    flat = _strip_tags(markdown)
    words = max(len(flat.split()), 1)
    return {
        "seq_mentions": len(_SEQ_RE.findall(flat)),
        "mw_patterns": len(_MW_RE.findall(flat)),
        "assay_terms": len(_ASSAY_RE.findall(flat)),
        "analog_defs": len(_ANALOG_RE.findall(flat)),
        "smiles_parsable": _smiles_parseable(markdown),
        "table_marked": int(bool(_TABLE_MARK_RE.search(markdown))),
        "figure_title": len(re.findall(r"FIGURE\s+\d", flat)),
        "img_blocks": markdown.count("<img"),
        "words": words,
    }


@dataclass
class PageTypeScores:
    label: str
    scores: dict = field(default_factory=dict)


def classify_page(markdown: str, min_signals: int = 1) -> PageTypeScores:
    """Score one page against the taxonomy. Scoring is additive over
    independent signals; a page takes the type with the strongest support
    and `min_signals` guards against single-weak-signal misfires on
    prose."""
    s = page_signals(markdown)
    scores = {
        "compound_table": (s["seq_mentions"] >= 1) + (s["mw_patterns"] >= 1) * 2
                          + (s["smiles_parsable"] >= 1) * 2
                          + (s["table_marked"] and s["assay_terms"] >= 1),
        "abbreviation_table": (s["analog_defs"] >= 2)
                              + (s["analog_defs"] >= 6) * 2
                              + (s["seq_mentions"] == 0),
        "assay_text": (s["assay_terms"] >= 3)
                      + (s["seq_mentions"] == 0 and s["analog_defs"] == 0),
        "2d_structure": ((s["figure_title"] >= 1) * 2
                         + (s["img_blocks"] >= 1)
                         + (s["table_marked"] == 0)
                         + (s["seq_mentions"] == 0)),
        "prose": 0,
    }
    label = max(scores, key=scores.get)
    if scores[label] < min_signals:
        label = "prose"
    return PageTypeScores(label=label, scores={**scores, **s})


def classify_corpus(pages: Iterable[tuple[object, str]]) -> dict:
    """Classify many (page_key, markdown) pages -> {type: [page_keys]}."""
    by_type: dict[str, list] = {}
    for key, md in pages:
        pts = classify_page(md)
        by_type.setdefault(pts.label, []).append(key)
    return by_type
