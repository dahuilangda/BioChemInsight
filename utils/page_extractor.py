"""Page-level VLM entity extraction.

One VLM call per page returns structured JSON: compounds (label, name,
structure type, sequence), sequences, rgroup_definitions and a page
summary. Page content is never pattern-filtered. Failures degrade to an
empty page record.
"""
from __future__ import annotations

import base64
import json
import os
import re

_SEQUENCE_TABLE_PROMPT = """Read the sequence/analytical table actually visible on this page.
Return every printed row as STRICT JSON:
{"sequence_entries": [{"seq_id": str, "sequence_text": str, "procedure": str,
  "molecular_weight": float, "molecular_formula": str, "exact_mass": float,
  "purity_percent": float, "measured_mass": float, "observed_ion": str}]}
Preserve residue modifications, terminal groups, ring-closure annotations and stereochemical
notation verbatim in sequence_text. Do not convert modified sequences into unmodified
one-letter strings. Never reinterpret a non-sequence results table as peptide sequences.
Do not invent missing rows, molecular formulas, residue identities or attachment positions.
Use null for absent numerical fields. Return JSON only."""

_PAGE_SCHEMA_INSTRUCTION = """You are the document-level page reader of a chemical SAR harness.
Read the ENTIRE page image and return STRICT JSON:
{"compounds": [{"label_in_page": str, "name": str, "structure_type": "complete_compound|markush|fragment|sequence_only", "evidence_kind": "drawn_structure|text_name|result_row|sequence|template_member", "evidence": str, "sequence_one_letter": str}],
 "sequences": [{"seq_id": str, "sequence_one_letter": str, "sequence_text": str, "cyclic": bool}],
 "has_sequence_table": bool,
 "rgroup_definitions": [{"variable": str, "substituents": [str]}],
 "page_summary": str}
Transcribe local identifiers and names exactly as printed. Classify HOW each subject appears using evidence_kind. A result row is not a drawn complete molecule; a variable-table member is not a separately drawn structure. Only visible chemical bond drawings count as drawn_structure; interaction plots and docked complexes do not establish a complete molecular graph. Never infer full chemical names from a scaffold and a substituent table. A standard one-letter sequence
may only represent an explicitly unmodified canonical sequence; preserve every modification
and closure annotation in sequence_text. Do not infer a missing cyclic linkage or residue.
R-group definitions require actual variable-definition evidence. has_sequence_table is true
only when a visible table actually contains peptide/biopolymer sequences or their typed
residue notation. Assay/ADMET tables alone are not sequence tables.
Unknown string fields are empty, unknown lists are empty. Do not invent entities."""

_LEDGER_HEADER = """ENTITIES ALREADY KNOWN IN THIS DOCUMENT (resolve references against these):
{ledger}
"""


def _render_page(pdf_file: str, page_num: int, dpi: int = 150) -> str | None:
    import fitz
    doc = fitz.open(pdf_file)
    try:
        if not (1 <= page_num <= doc.page_count):
            return None
        pix = doc[page_num - 1].get_pixmap(dpi=dpi)
        path = f"/tmp/harness_page_{os.getpid()}_{page_num}.png"
        pix.save(path)
        return path
    finally:
        doc.close()


def _parse_strict_json(text: str) -> dict:
    """Strip code fences and return the first JSON object found."""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return {}
    return {}


class PageEntityExtractor:
    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import constants
            from openai import OpenAI
            self._client = OpenAI(
                base_url=constants.LLM_OPENAI_COMPATIBLE_MODEL_URL,
                api_key=constants.LLM_OPENAI_COMPATIBLE_MODEL_KEY)
            self._model = constants.VISUAL_MODEL_NAME
        return self._client

    def extract_page(self, pdf_file: str, page_num: int, ledger_digest: str = '') -> dict:
        """One page -> entity record. Fail-open to an empty record."""
        record = {'page_num': page_num, 'compounds': [], 'sequences': [],
                  'sequence_entries': [], 'rgroup_definitions': [],
                  'page_summary': '', 'ok': False, 'inventory_version': 2}
        png = None
        try:
            png = _render_page(pdf_file, page_num)
            if not png:
                return record
            img_b64 = base64.b64encode(open(png, 'rb').read()).decode()
            client = self._ensure_client()

            prompt = _PAGE_SCHEMA_INSTRUCTION
            if ledger_digest:
                prompt += "\n\n" + _LEDGER_HEADER.format(ledger=ledger_digest)
            resp = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}],
                temperature=0, max_tokens=3500, timeout=180)
            payload = _parse_strict_json(resp.choices[0].message.content)
            if payload:
                record['compounds'] = payload.get('compounds') or []
                record['sequences'] = payload.get('sequences') or []
                record['rgroup_definitions'] = payload.get('rgroup_definitions') or []
                record['page_summary'] = str(payload.get('page_summary') or '')
                record['ok'] = True
                if payload.get('has_sequence_table') is True:
                    seq_resp = client.chat.completions.create(model=self._model,
                        messages=[{"role": "user", "content": [{"type": "text", "text": _SEQUENCE_TABLE_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}],
                        temperature=0, max_tokens=4000, timeout=180)
                    seq_data = _parse_strict_json(seq_resp.choices[0].message.content)
                    record['sequence_entries'] = seq_data.get('sequence_entries') or []
        except Exception as exc:
            record['error'] = f"{type(exc).__name__}: {str(exc)[:120]}"
        finally:
            if png and os.path.exists(png):
                try:
                    os.remove(png)
                except OSError:
                    pass
        return record

    def extract_pages(self, pdf_file: str, pages: list[int]) -> list[dict]:
        """Extract pages in document order, feeding previously extracted
        entities back as context."""
        records = []
        entities: list[str] = []
        for page_num in sorted(set(pages)):
            digest = '\n'.join(entities[-40:]) if entities else ''
            record = self.extract_page(pdf_file, page_num, digest)
            records.append(record)
            for compound in record.get('compounds', []) or []:
                label = str(compound.get('label_in_page') or '').strip()
                name = str(compound.get('name') or '').strip()
                if label or name:
                    entities.append(f"page {page_num}: [{label}] {name}")
            for seq in record.get('sequences', []) or []:
                sid = str(seq.get('seq_id') or '').strip()
                if sid:
                    entities.append(f"page {page_num}: {sid} sequence recorded")
        return records
