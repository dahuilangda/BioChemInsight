"""Atomic skill: antibody chain numbering.

Anarcii (OPIG's ANARCI reimplementation) assigns IMGT numbers to a heavy
or light chain; CDR intervals are read off the numbering directly. A low
model score flags the chain for review rather than guessing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

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


@register("antibody_numbering", "model",
          "Number antibody chains using IMGT and report CDR intervals.",
          input_schema={
              "type": "object",
              "properties": {
                  "sequence": {"type": "string", "minLength": 1},
                  "sequences": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                  "scheme": {"type": "string", "enum": ["imgt"]},
              },
              "oneOf": [{"required": ["sequence"]}, {"required": ["sequences"]}],
          })
async def number_chain(payload: dict) -> dict:
    """One or more chain sequences -> numbering summary per chain."""
    if "sequence" in payload and "sequences" in payload:
        return {"ok": False, "error": "Specify sequence or sequences, not both"}
    seqs = payload.get("sequences", [payload.get("sequence")])
    if not isinstance(seqs, list) or not seqs:
        return {"ok": False, "error": "sequences must be a non-empty list"}
    for index, sequence in enumerate(seqs):
        if (not isinstance(sequence, str) or not sequence
                or not sequence.isascii() or not sequence.isalpha() or not sequence.isupper()):
            return {"ok": False, "error": f"Sequence {index + 1} must contain uppercase residue letters without separators"}
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
    if not out:
        return {"ok": False, "chains": [], "error": "No chain numbering returned"}
    return {"ok": all(not chain["error"] for chain in out), "chains": out}
