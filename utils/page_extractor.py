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

_PAGE_SCHEMA_INSTRUCTION = """You are the page reader of a patent-mining harness.
Read the ENTIRE page image and return STRICT JSON with EXACTLY this schema:
{"compounds": [{"label_in_page": str, "name": str, "structure_type": "complete_compound|markush|fragment|sequence_only", "sequence_one_letter": str}],
 "sequences": [{"seq_id": str, "sequence_one_letter": str, "cyclic": bool}],
 "rgroup_definitions": [{"variable": str, "substituents": [str]}],
 "page_summary": str}
Rules:
- label_in_page: the label EXACTLY as printed near the structure (e.g. "22", "SEQ ID NO. 53", "R4").
- For peptides, transcribe sequences EXACTLY as printed; convert three-letter codes to one-letter (Cys=C, Gln=Q, ...). Use "" if the structure is a drawn macrocycle too modified to transcribe — do not invent residues.
- rgroup_definitions: only when the page defines markush variables ("R4 is selected from methyl, ethyl, ...").
- If a field is unknown use "" / [] / false. Return JSON only."""

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
                  'rgroup_definitions': [], 'page_summary': '', 'ok': False}
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
