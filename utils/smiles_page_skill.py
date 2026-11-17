"""SMILES extraction skill for compound-table pages.

A compound-table page prints each compound's SMILES as text. Two
extraction routes exist and are A/B measurable:

  paddle   SMILES candidate strings from the page's own OCR markdown
           cells (route works without any model server)
  vlm      crop the SMILES row region and transcribe through the vision
           model (needs a live vision server)
  llm      paddle candidates repaired through the text model using
           chemistry constraints (needs a live text server)

Every candidate from every route faces the same arbitration: RDKit parse,
then declared-mass match (mono or average convention). The route
comparison is therefore apples-to-apples — a route wins on surviving
verified rows per page, not on raw candidate count.
"""
from __future__ import annotations

import json
import re
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_WS_RE = re.compile(r"\s+")
_CAND_RE = re.compile(r"[\[\]A-Za-z0-9@+\-=\(\)#%/\\\.]{15,}")


def _clean_candidate(text: str) -> str:
    text = text.replace("\\n", " ")
    return _WS_RE.sub("", text)


def candidates_from_markdown(markdown: str) -> list[str]:
    """Route 'paddle': SMILES candidate strings from the page's OCR
    markdown cells. Cell text concatenated per row before candidate
    splitting (the printed SMILES wraps across lines). Candidates are
    inert strings — parse and mass arbitration decide their fate."""
    out, seen = [], set()
    for row_match in re.finditer(r"<tr>(.*?)</tr>", markdown, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_match.group(1), re.S)
        for cell in cells[1:]:
            text = _clean_candidate(re.sub(r"<[^>]+>", "", cell))
            for cand in _CAND_RE.findall(text):
                if cand not in seen and "(=O)" in cand:
                    seen.add(cand)
                    out.append(cand)
    return out


def candidates_via_llm(candidates: list[str], context: str) -> list[str]:
    """Route 'llm': repair damaged paddle candidates through the text
    model under chemistry constraints. Requires a live text server; every
    repaired string still faces parse + mass arbitration. Skips itself
    (returns empty) when the server is unavailable — call sites decide
    whether to retry later."""
    if not candidates:
        return []
    try:
        from constants import (LLM_OPENAI_COMPATIBLE_MODEL_KEY,
                               LLM_OPENAI_COMPATIBLE_MODEL_NAME,
                               LLM_OPENAI_COMPATIBLE_MODEL_URL)
        from openai import OpenAI
        client = OpenAI(api_key=LLM_OPENAI_COMPATIBLE_MODEL_KEY,
                        base_url=LLM_OPENAI_COMPATIBLE_MODEL_URL)
        reply = client.chat.completions.create(
            model=LLM_OPENAI_COMPATIBLE_MODEL_NAME,
            messages=[{"role": "system",
                       "content": "You repair OCR-damaged SMILES strings "
                                  "for peptide compounds. For each input, "
                                  'output {"in": ..., "out": "<repaired '
                                  'SMILES or empty>"}; never invent atoms '
                                  "that are not evidenced by the input. "
                                  "Output a JSON array only."},
                      {"role": "user",
                       "content": json.dumps(candidates[:8])
                       + (f"\nContext: {context}" if context else "")}],
            temperature=0, timeout=120)
        items = json.loads(_json_block(reply.choices[0].message.content))
        repaired = []
        for item in items if isinstance(items, list) else []:
            smiles = str(item.get("out", "")).strip()
            if smiles and Chem.MolFromSmiles(smiles) is not None:
                repaired.append(smiles)
        return repaired
    except Exception:
        return []


def _json_block(text: str):
    m = re.search(r"\[.*\]", text or "", re.S)
    return m.group(0) if m else "[]"


def parse_and_mass_gate(candidate: str, declared_mass: float | None,
                        tolerance: float = 0.8) -> dict:
    """The only acceptance decision: RDKit parse + declared-mass match in
    either convention (patent MW columns print mono or average)."""
    verdict = {"candidate": candidate[:80], "parsed": False, "ok": False,
               "delta": None, "convention": None}
    mol = Chem.MolFromSmiles(candidate)
    if mol is None:
        return verdict
    verdict["parsed"] = True
    if declared_mass is None:
        return verdict
    from rdkit.Chem import Descriptors
    exact = Descriptors.ExactMolWt(mol)
    avg = Descriptors.MolWt(mol)
    declared = float(declared_mass)
    for convention, mass in (("mono", exact), ("avg", avg)):
        if abs(mass - declared) <= tolerance:
            verdict.update({"ok": True, "delta": round(mass - declared, 4),
                            "convention": convention,
                            "smiles": Chem.MolToSmiles(mol)})
            return verdict
    verdict["delta"] = round(exact - declared, 4)
    return verdict


def extract_verified_smiles(markdown: str, declared_masses: list[float],
                            routes: tuple[str, ...] = ("paddle",),
                            llm_context: str = "") -> dict:
    """Run the enabled routes over one page and arbitrate. Returns the
    verified mapping {declared_mass: smiles} plus per-route stats — the
    A/B measurement the route comparison is judged on."""
    stats: dict[str, dict] = {}
    verified: dict[float, str] = {}
    pool: list[str] = []
    if "paddle" in routes:
        cands = candidates_from_markdown(markdown)
        stats["paddle"] = {"candidates": len(cands)}
        pool.extend(cands)
    if "llm" in routes:
        cands = candidates_from_markdown(markdown)
        repaired = candidates_via_llm(cands[:8], llm_context)
        stats["llm"] = {"candidates": len(repaired)}
        pool.extend(repaired)
    parsed = 0
    for cand in pool:
        for declared in declared_masses:
            v = parse_and_mass_gate(cand, declared)
            parsed += v["parsed"]
            if v["ok"] and declared not in verified:
                verified[declared] = Chem.MolToSmiles(
                    Chem.MolFromSmiles(cand))
                for route, st in stats.items():
                    st.setdefault("verified", 0)
                    st["verified"] += 1
    total = {"parsed": parsed}
    return {"verified": verified, "stats": stats, "total_parsed": parsed,
            "candidates_tried": len(pool)}


def candidates_from_json_rows(rows: list[dict]) -> list[tuple[str, float | None]]:
    """Rows mined from OCR (compound dicts with smiles_reading/mw) ->
    (candidate, declared mass) pairs for arbitration."""
    out = []
    for row in rows:
        cand = re.sub(r"\s+", "", str(row.get("smiles_reading", "")))
        if cand:
            out.append((cand, row.get("mw")))
    return out
