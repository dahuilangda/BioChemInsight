"""Atomic skill: HGVS variant notation parsing.

Parses 'NM_000546.6:c.743G>A' style strings into structured sites
(accession, kind, position, ref, alt). Bare protein notation
('p.Thr284Ile') is accepted by supplying a placeholder accession for the
parse and reporting the accession as null.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from skills.registry import register

_PARSER = None


def _parser():
    global _PARSER
    if _PARSER is None:
        import hgvs.parser
        _PARSER = hgvs.parser.Parser()
    return _PARSER


def _parse_one(text: str) -> dict:
    import hgvs.exceptions
    bare_protein = False
    try:
        v = _parser().parse_hgvs_variant(text)
    except hgvs.exceptions.HGVSParseError:
        # 'p.Ala100Gly' without an accession: parse under a placeholder
        if not text.startswith(("NP_", "ENSP", "p.")):
            raise
        v = _parser().parse_hgvs_variant(
            text if ":" in text else f"NP_0:{text}")
        bare_protein = ":" not in text
    pos = v.posedit.pos
    def _pos(p):
        return getattr(p, "position", None) or getattr(p, "pos", None)
    site = {
        "hgvs": str(v),
        "kind": v.type,
        "accession": None if bare_protein else str(v.ac),
        "gene": getattr(v.gene, "name", None) if v.gene else None,
        "start": _pos(pos.start),
        "end": _pos(pos.end) if getattr(pos, "end", None) else _pos(pos.start),
        "ref": getattr(v.posedit.edit, "ref", None),
        "alt": getattr(v.posedit.edit, "alt", None),
    }
    return site


@register("variant_sites", "chemistry",
          "HGVS strings -> structured mutation sites")
async def parse_variants(payload: dict) -> dict:
    """List of HGVS strings (or one) -> parsed sites; failures are named
    per input, never silently dropped."""
    texts = payload.get("hgvs") or ([payload["hgvs_string"]]
                                    if payload.get("hgvs_string") else [])
    if isinstance(texts, str):
        texts = [texts]
    sites, failures = [], []
    for t in texts:
        t = str(t).strip()
        if not t:
            continue
        try:
            sites.append(_parse_one(t))
        except Exception as exc:
            failures.append({"input": t, "error": str(exc)[:120]})
    return {"ok": bool(sites) or not failures, "sites": sites,
            "failures": failures}
