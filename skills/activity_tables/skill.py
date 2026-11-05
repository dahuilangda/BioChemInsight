"""Atomic skill: activity tables -> compound-id to assay-value rows.

Input is page markdown (PaddleOCR HTML tables or plain text). A row is
kept when an identifier cell (SEQ ID NO / Compound / Example) and at
least one assay cell (IC50 / EC50 / Ki / ED50 / Kd / IC-50 ...) both
parse. Values normalize to a number, a unit, and the comparison operator
they carried ('<50' stays '<50' with value 50.0).
"""
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from skills.registry import register

_ID_HINTS = ("seq id", "compound", "example", "peptide", "cpd")
_ASSAY = re.compile(r"\b(IC\s*-?\s*50|EC\s*-?\s*50|Ki|ED\s*-?\s*50|Kd|GI\s*50)\b", re.I)
_VALUE = re.compile(r"([<>≤≥~])?\s*([0-9]+(?:\.[0-9]+)?)\s*([nμmu]M|mM|M)?")
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)


def _clean(cell: str) -> str:
    import html
    c = html.unescape(re.sub(r"<[^>]+>", "", str(cell)))
    return re.sub(r"\s+", " ", c.replace("\\n", " ")).strip()


def _parse_value(text: str):
    m = _VALUE.search(text)
    if not m:
        return None
    op = m.group(1) or ""
    value = float(m.group(2))
    unit = (m.group(3) or "nM").replace("μ", "u").replace("uM", "nM") \
        if not m.group(3) else m.group(3).replace("μ", "u")
    return {"raw": text, "operator": op, "value": value, "unit": unit}


def _classify(header: list[str], cells: list[str]) -> dict | None:
    """Header cells name the columns; data cells carry the values."""
    id_cols = [i for i, h in enumerate(header)
               if any(hint in h.lower() for hint in _ID_HINTS)]
    assay_cols = {i: re.sub(r"\s+", "", _ASSAY.search(h).group(1)).upper()
                  .replace("-50", "50")
                  for i, h in enumerate(header) if _ASSAY.search(h)}
    if not id_cols or not assay_cols:
        return None
    def cell(i):
        return cells[i] if i < len(cells) else ""
    row_id = None
    id_kind = None
    for i in id_cols:
        num = re.search(r"(\d{1,4}[A-Za-z]?)", cell(i))
        if num:
            row_id = num.group(1)
            id_kind = next(h for h in _ID_HINTS
                           if h in header[i].lower())
            break
    if row_id is None:
        num = re.search(r"^(\d{1,4}[A-Za-z]?)$", cells[0].strip()) if cells else None
        if num:
            row_id = num.group(1)
            id_kind = "seq id"
    if row_id is None:
        return None
    assays = {}
    for i, name in assay_cols.items():
        value = _parse_value(cell(i))
        if value:
            assays[name] = value
    return {"compound_id": row_id, "id_kind": id_kind,
            "assays": assays} if assays else None


@register("activity_tables", "chemistry",
          "activity-table markdown -> compound id x assay values")
async def parse(payload: dict) -> dict:
    text = str(payload.get("markdown") or payload.get("text") or "")
    if not text:
        return {"ok": False, "error": "no markdown given"}
    rows = []
    header = []
    for tr in _ROW.finditer(text):
        cells = [_clean(c) for c in _CELL.findall(tr.group(1))]
        if not header:
            header = cells
            continue
        row = _classify(header, cells)
        if row:
            rows.append(row)
    if not rows:
        # plain-text fallback: id head and assay value in one line
        for line in text.splitlines():
            if _ASSAY.search(line):
                row = _classify([line], [line])
                if row:
                    rows.append(row)
    return {"ok": bool(rows), "rows": rows,
            "assay_types": sorted({a for r in rows for a in r["assays"]})}
