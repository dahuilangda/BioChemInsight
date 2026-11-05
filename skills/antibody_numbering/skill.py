"""Atomic skill: antibody chain numbering.

Anarcii (OPIG's ANARCI reimplementation) assigns IMGT numbers to a heavy
or light chain; CDR intervals are read off the numbering directly. A low
model score flags the chain for review rather than guessing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import re
from skills.registry import register

_MIN_SCORE = 20.0   # anarcii sequence log-likelihood scale

IMGT_CDR = {1: (27, 38), 2: (56, 65), 3: (105, 117)}

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from anarcii import Anarcii
        _MODEL = Anarcii(seq_type="antibody", mode="accuracy", cpu=True)
    return _MODEL


def _cdrs(numbering) -> dict:
    out = {}
    for name, (lo, hi) in IMGT_CDR.items():
        residues = [aa for (pos, _ins), aa in numbering if lo <= pos <= hi]
        out[f"CDR{name}"] = {
            "range": [lo, hi],
            "length": sum(1 for a in residues if a != "-"),
            "sequence": "".join(residues).replace("-", ""),
        }
    return out


def _clean(seq: str) -> str:
    return re.sub(r"[^A-Za-z]", "", str(seq or "")).upper()


@register("antibody_numbering", "model",
          "antibody chain -> IMGT numbering, chain type, CDR intervals")
async def number_chain(payload: dict) -> dict:
    """One or more chain sequences -> numbering summary per chain."""
    seqs = payload.get("sequences") or ([payload["sequence"]]
                                        if payload.get("sequence") else [])
    seqs = [_clean(s) for s in seqs if _clean(s)]
    if not seqs:
        return {"ok": False, "error": "no amino-acid sequence given"}
    scheme = str(payload.get("scheme", "imgt")).lower()
    if scheme != "imgt":
        return {"ok": False,
                "error": f"scheme '{scheme}' not supported: CDR intervals in "
                         "this skill are IMGT-specific and to_scheme "
                         "conversion is not wired; omit 'scheme' or pass "
                         "'imgt'"}
    model = _model()
    raw = model.number(seqs if len(seqs) > 1 else seqs[0])
    # anarcii keys results by sequence label ("Sequence" by default)
    if isinstance(raw, dict):
        results = list(raw.values())
    else:
        results = list(raw)
    out = []
    for r in results:
        chain = {
            "chain_type": r.get("chain_type"),
            "scheme": r.get("scheme", "imgt"),
            "score": round(float(r.get("score") or 0), 3),
            "length": len(r.get("numbering") or []),
            "numbered_sequence": "".join(aa for _, aa in r.get("numbering", [])),
            "cdrs": _cdrs(r.get("numbering") or []),
            "needs_review": float(r.get("score") or 0) < _MIN_SCORE,
            "error": r.get("error"),
        }
        out.append(chain)
    return {"ok": True, "chains": out}
