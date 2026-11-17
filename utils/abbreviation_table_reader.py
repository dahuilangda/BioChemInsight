"""Read a document's own monomer/abbreviation definition table into
(code, chemical name) evidence pairs.

The table pages are the authoritative dictionary a peptide patent ships for
its own codes. The reader TRANSCRIBES only: printed codes and the chemical
names printed next to them. It deliberately does not ask the model for
SMILES — structure inference belongs to name resolution (OPSIN/PubChem)
plus arbitration, not to transcription. Two independent reads are taken
and intersected per code; disagreements stay in the audit trail. Pairs
land in the resolution harness as name evidence (never as fragments).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from utils.llm_utils import call_visual_model

_READER_PROMPT = """You are transcribing a monomer definition table from a \
patent document. The table maps abbreviation codes to chemical definitions.

For EVERY table row visible, output one JSON object:
{"code": "<the abbreviation code(s) exactly as printed in the left column>",
 "names": ["<each full chemical name exactly as printed for this row>"],
 "has_structure_drawing": <true when the row shows a drawn structure>,
 "note": "<empty, or one line when the row cannot be read>"}

Rules:
- Transcribe names verbatim, including stereo descriptors like (S) or (2R,4S).
- A row may define a generic R-group scaffold instead of naming a compound: \
then "names" is empty and has_structure_drawing is true.
- Do NOT invent, expand, or correct names. Do NOT output SMILES.
- Skip rows that are page furniture (headers, continuation markers).
Output a JSON array of the row objects and nothing else."""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_rows(text: str) -> list[dict]:
    if not text:
        return []
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return []
    try:
        rows = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        code = str(row.get("code", "")).strip()
        if not code:
            continue
        names = [str(n).strip() for n in row.get("names") or []
                 if str(n).strip()]
        out.append({"code": code, "names": names,
                    "has_structure_drawing":
                        bool(row.get("has_structure_drawing")),
                    "note": str(row.get("note", "")).strip()})
    return out


_ANNOTATION_RE = re.compile(r"[\n(]")   # '(Analog of Trp)' style note lines


def _code_variants(code: str) -> set[str]:
    """Printed-code normalization for MATCHING table rows to sequence
    tokens: the code cell may carry an annotation line under the code
    ('1Nal' / '(Analog of Trp)'), so variants include the first line plus
    the parenthesis-stripped form; casefold and the OCR-unstable
    underscore/whitespace and '1/l' confusion are folded too. Matching
    only — the harness arbitrates on chemistry."""
    primary = code.split("\n")[0].strip()
    candidates = {code, primary, _ANNOTATION_RE.split(code)[0].strip()}
    out = set()
    for cand in candidates:
        fold = re.sub(r"[\s_]+", "", cand).casefold()
        if fold:
            out |= {fold, fold.replace("l", "1"), fold.replace("1", "l")}
    return out


def read_abbreviation_table(image_file: str, reads: int = 2,
                            audit_path: str | None = None) -> dict:
    """One table page -> transcribed (code, names) pairs with a per-code
    agreement count across independent reads. A code appears in the output
    only when at least one read produced a name for it; the audit keeps
    every read verbatim so disagreements are visible, never averaged away."""
    read_rows, errors = [], []
    for attempt in range(1, max(1, reads) + 1):
        try:
            text = call_visual_model(image_file, _READER_PROMPT)
        except Exception as exc:
            errors.append(f"read {attempt}: {type(exc).__name__}: "
                          f"{str(exc)[:120]}")
            continue
        rows = _parse_rows(text)
        read_rows.append(rows)
        if not rows:
            errors.append(f"read {attempt}: no parseable rows")
    by_code: dict[str, dict] = {}
    for rows in read_rows:
        for row in rows:
            variants = _code_variants(row["code"])
            entry = None
            for fold, existing in by_code.items():
                if variants & existing["variants"]:
                    entry = existing
                    break
            if entry is None:
                entry = {"code": row["code"], "variants": variants,
                         "names": [], "reads": 0,
                         "has_structure_drawing":
                             row["has_structure_drawing"]}
                by_code[row["code"]] = entry
            entry["reads"] += 1
            for name in row["names"]:
                if name not in entry["names"]:
                    entry["names"].append(name)
    pairs = [(entry["code"], name) for entry in by_code.values()
             for name in entry["names"]]
    audit = {"image": str(image_file), "reads": len(read_rows),
             "codes": len(by_code),
             "disagreements": [entry["code"] for entry in by_code.values()
                               if len(entry["names"]) > 1],
             "errors": errors}
    if audit_path:
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
        transcribed = [{k: (sorted(v) if k == "variants" else v)
                        for k, v in entry.items()}
                       for entry in by_code.values()]
        Path(audit_path).write_text(json.dumps(
            {"audit": audit, "transcribed": transcribed}, indent=1),
            encoding="utf-8")
    return {"pairs": pairs, "audit": audit, "entries": by_code}


def pairs_from_pages(page_images: list[str], audit_dir: str | None = None,
                     ) -> tuple[list[tuple[str, str]], list[dict]]:
    """Transcribe several table pages in order. Returns (pairs, page audits);
    callers feed the pairs into PeptideResolutionHarness.
    ingest_abbreviation_names."""
    all_pairs: list[tuple[str, str]] = []
    audits = []
    for page in page_images:
        audit_path = (f"{audit_dir}/table_{Path(page).stem}.json"
                      if audit_dir else None)
        result = read_abbreviation_table(page, audit_path=audit_path)
        all_pairs.extend(result["pairs"])
        audits.append(result["audit"])
    return all_pairs, audits


_STRUCTURE_PROMPT = """This patent table page defines monomer codes. For \
each row whose definition is a DRAWN STRUCTURE (with R1/R2/R3 placeholders \
marking the backbone termini), transcribe the structure to SMILES using \
the row's own legend: R1 marked H means a free N-terminus, R2 marked OH \
means a free C-terminus, side-chain placeholders (R3...) become the \
printed atom/group. Output one JSON object per requested code:
{"code": "<requested code>", "smiles": "<full free amino acid as \
N...C(=O)O, or empty when the structure cannot be read>",
 "note": "<one line, or empty>"}
Do not guess: an unreadable structure gets an empty smiles and a note. \
Output a JSON array and nothing else."""


def read_structure_definitions(image_file: str, codes: list[str],
                               audit_path: str | None = None) -> dict:
    """Structure-only table rows: a model READING of the drawn structure
    per requested code. Output is candidate evidence (source 'vision') —
    the resolution harness arbitrates it exactly like every other reading;
    an unreadable drawing returns empty with its note, never a guess."""
    if not codes:
        return {"pairs": [], "audit": {"image": str(image_file),
                                       "codes": 0, "errors": []}}
    listing = "\n".join(f"- {code}" for code in codes)
    prompt = f"{_STRUCTURE_PROMPT}\nRequested codes:\n{listing}"
    try:
        text = call_visual_model(image_file, prompt)
    except Exception as exc:
        return {"pairs": [], "audit": {"image": str(image_file),
                                       "codes": len(codes),
                                       "errors": [f"{type(exc).__name__}: "
                                                  f"{str(exc)[:120]}"]}}
    match = _JSON_ARRAY_RE.search(text or "")
    pairs: list[tuple[str, str]] = []
    unread = []
    if match:
        try:
            rows = json.loads(match.group(0))
        except json.JSONDecodeError:
            rows = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            code = str(row.get("code", "")).strip()
            smiles = str(row.get("smiles", "")).strip()
            note = str(row.get("note", "")).strip()
            if not code:
                continue
            if smiles:
                pairs.append((code, smiles))
            else:
                unread.append({"code": code, "note": note})
    audit = {"image": str(image_file), "codes": len(codes),
             "read": len(pairs), "unread": unread, "errors": []}
    if audit_path:
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
        Path(audit_path).write_text(json.dumps(
            {"audit": audit, "pairs": pairs}, indent=1), encoding="utf-8")
    return {"pairs": pairs, "audit": audit}
