"""Definition-legend page mining from PP-OCR text-line boxes."""
from __future__ import annotations

import re

_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(_[A-Za-z0-9]+)?$")


def mine_definition_page(lines: list[dict], page: str,
                         page_width: float) -> list[dict]:
    if not lines:
        return []
    split_x = page_width * 0.38          # code region | definition region
    codes = [l for l in lines
             if l["box"][0] < split_x and _plausible_code(l["text"])]
    if not codes:
        return []
    codes.sort(key=lambda l: l["box"][1])
    entries: list[dict] = []
    for i, code_line in enumerate(codes):
        band_top = code_line["box"][1]
        band_bottom = (codes[i + 1]["box"][1]
                       if i + 1 < len(codes) else float("inf"))
        names, rdefs = [], []
        for line in lines:
            if line is code_line or line["box"][0] < split_x:
                continue
            yc = (line["box"][1] + line["box"][3]) / 2
            if not (band_top - 30 <= yc < band_bottom):
                continue
            text = line["text"].strip()
            if text.lower().startswith(("r1 ", "r2 ", "r1-", "r2-")) \
                    or "SMILES is" in text:
                rdefs.append(text)
            elif _name_like(text):
                names.append(text)
        entries.append({
            "page": page,
            "code": code_line["text"].strip(),
            "names": names,
            "rdefs": rdefs,
            "has_structure_drawing": not names,
        })
    return entries


def _plausible_code(text: str) -> bool:
    t = text.strip().rstrip(";")
    if len(t) > 40:
        return False
    parts = re.split(r"\s+or\s+", t)
    return all(re.match(r"^\(?[A-Za-z][A-Za-z0-9_ ]{0,24}\)?$", p.strip())
               for p in parts)


def _name_like(text: str) -> bool:
    t = text.strip()
    if not (3 <= len(t) <= 80):
        return False
    if t.startswith("[") or t.startswith("("):
        return False
    return bool(re.search(r"[a-z]{3}", t))
