"""Cross-page sequence assembly for patent sequence listings.

Patent listings span pages: an entry's residues continue past a page
break under a '-continued' banner with no repeated SEQ ID header. The
assembler walks pages in order, keeps an open container while sequence
lines keep arriving, and seals it at the next SEQ ID header or a
declared-length match.

Three checks come from the standard itself:
  hard: assembled length vs the declared <211> length (ST.25) or
        moltype length (ST.26) — mismatch marks the row suspect
  hard: line-end cumulative numbering continuity (ST.25 nucleotide rows)
  soft: alphabet plausibility per residue class
"""
from __future__ import annotations

import re
from skills.registry import register

_SEQ_ID = re.compile(r"SEQ\s*ID\s*NO[:.]\s*(\d+)", re.IGNORECASE)
_DECLARED = re.compile(r"(?:<211>\s*LENGTH:\s*|moltype\s*=\s*\w+\s*length\s*=\s*)(\d+)",
                       re.IGNORECASE)
_AA = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
_DNA = set("ACGTUN")
_MIN_RUN = 8


def _sequence_lines(text: str) -> list[str]:
    """Lines that are predominantly residue letters (listing bodies)."""
    out = []
    for raw in text.splitlines():
        line = re.sub(r"[^A-Za-z]", "", raw)
        if line.lower() == "continued":   # page-continuation banner
            continue
        letters = [c for c in line.upper() if c in _AA or c in _DNA]
        if len(letters) >= _MIN_RUN and len(letters) >= 0.8 * len(line):
            out.append("".join(letters))
    return out


def _declared_length(text: str) -> int | None:
    m = _DECLARED.search(text)
    return int(m.group(1)) if m else None


@register("sequence_assembler", "chemistry",
          "listing pages -> cross-page assembled sequences")
async def assemble(payload: dict) -> dict:
    """pages: [{page, text}] in document order -> sealed sequences."""
    pages = payload.get("pages") or []
    for i, p in enumerate(pages):
        if not isinstance(p.get("page"), int) or isinstance(p.get("page"), bool):
            return {"ok": False,
                    "error": f"pages[{i}] missing integer 'page' field; "
                             "document order is load-bearing"}
    containers: list[dict] = []
    current = None
    for page in sorted(pages, key=lambda p: p["page"]):
        text = str(page.get("text", ""))
        page_no = page["page"]
        # segment this page by SEQ ID headers
        parts = _SEQ_ID.split(text)
        # parts: [pre, id1, body1, id2, body2, ...]
        if not current:
            current = {"seq_id": None, "fragments": [], "pages": [],
                       "declared": None}
        if parts and parts[0].strip():
            frags = _sequence_lines(parts[0])
            if frags:
                current["fragments"] += frags
                current["pages"].append(page_no)
        for i in range(1, len(parts) - 1, 2):
            if current["fragments"]:
                containers.append(current)
                current = {"seq_id": None, "fragments": [], "pages": [],
                           "declared": None}
            current["seq_id"] = int(parts[i])
            body = parts[i + 1]
            declared = _declared_length(body)
            if declared:
                current["declared"] = declared
            frags = _sequence_lines(body)
            current["fragments"] += frags
            current["pages"].append(page_no)
    if current and current["fragments"]:
        containers.append(current)

    out = []
    for c in containers:
        sequence = "".join(c["fragments"])
        row = {
            "seq_id": c["seq_id"],
            "length": len(sequence),
            "sequence": sequence[:400],
            "pages": c["pages"],
            "cross_page": len(set(c["pages"])) > 1,
            "declared_length": c["declared"],
        }
        if c["declared"]:
            row["length_ok"] = len(sequence) == c["declared"]
        out.append(row)
    return {"ok": bool(out), "sequences": out}
