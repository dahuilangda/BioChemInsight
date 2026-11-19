"""Page-level molecular and sequence evidence."""

from __future__ import annotations

import logging
import math
import os
import tempfile
from pathlib import Path

from utils.model_harness import (
    ModelContractError,
    ModelEvidenceError,
    require_json_object,
)

logger = logging.getLogger(__name__)

def _object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_STRING = {"type": "string"}
_BOOLEAN = {"type": "boolean"}
_NUMBER_OR_NULL = {"type": ["number", "null"]}
_STRUCTURE_TYPES = ("complete_compound", "markush", "fragment", "sequence_only")
_EVIDENCE_KINDS = ("drawn_structure", "text_name", "result_row", "sequence", "template_member")
_COMPOUND_SCHEMA = _object_schema({
    "label_in_page": _STRING,
    "name": _STRING,
    "structure_type": {"type": "string", "enum": list(_STRUCTURE_TYPES)},
    "evidence_kind": {"type": "string", "enum": list(_EVIDENCE_KINDS)},
    "evidence": _STRING,
    "sequence_one_letter": _STRING,
})
_SEQUENCE_SCHEMA = _object_schema({
    "seq_id": _STRING,
    "sequence_one_letter": _STRING,
    "sequence_text": _STRING,
    "cyclic": _BOOLEAN,
})
_VARIABLE_SCHEMA = _object_schema({
    "variable": _STRING,
    "substituents": {"type": "array", "items": _STRING},
})
_PAGE_SCHEMA = _object_schema({
    "compounds": {"type": "array", "items": _COMPOUND_SCHEMA},
    "sequences": {"type": "array", "items": _SEQUENCE_SCHEMA},
    "has_sequence_table": _BOOLEAN,
    "rgroup_definitions": {"type": "array", "items": _VARIABLE_SCHEMA},
    "page_summary": _STRING,
})
_TABLE_ROW_SCHEMA = _object_schema({
    "seq_id": _STRING,
    "sequence_text": _STRING,
    "procedure": _STRING,
    "molecular_formula": _STRING,
    "observed_ion": _STRING,
    "molecular_weight": _NUMBER_OR_NULL,
    "exact_mass": _NUMBER_OR_NULL,
    "purity_percent": _NUMBER_OR_NULL,
    "measured_mass": _NUMBER_OR_NULL,
})
_SEQUENCE_TABLE_SCHEMA = _object_schema({
    "sequence_entries": {"type": "array", "items": _TABLE_ROW_SCHEMA},
})


_SEQUENCE_TABLE_PROMPT = """Read the sequence/analytical table actually visible on this page.
Return every printed row as STRICT JSON:
{"sequence_entries": [{"seq_id": str, "sequence_text": str, "procedure": str,
  "molecular_weight": float, "molecular_formula": str, "exact_mass": float,
  "purity_percent": float, "measured_mass": float, "observed_ion": str}]}
Preserve residue modifications, terminal groups, ring-closure annotations and stereochemical
notation verbatim in sequence_text. Do not convert modified sequences into unmodified
one-letter strings. Never reinterpret a non-sequence results table as peptide sequences.
Do not invent missing rows, molecular formulas, residue identities or attachment positions.
Use null for absent numerical fields and empty strings for absent text fields. Return JSON only."""

_PAGE_SCHEMA_INSTRUCTION = """You are the document-level page reader of a chemical SAR harness.
Read the ENTIRE page image and return STRICT JSON:
{"compounds": [{"label_in_page": str, "name": str, "structure_type": "complete_compound|markush|fragment|sequence_only", "evidence_kind": "drawn_structure|text_name|result_row|sequence|template_member", "evidence": str, "sequence_one_letter": str}],
 "sequences": [{"seq_id": str, "sequence_one_letter": str, "sequence_text": str, "cyclic": bool}],
 "has_sequence_table": bool,
 "rgroup_definitions": [{"variable": str, "substituents": [str]}],
 "page_summary": str}
Transcribe local identifiers and names exactly as printed. Classify HOW each subject appears using evidence_kind. A result row is not a drawn complete molecule; a variable-table member is not a separately drawn structure. Only visible chemical bond drawings count as drawn_structure; interaction plots and docked complexes do not establish a complete molecular graph. Never infer full chemical names from a scaffold and a substituent table. A standard one-letter sequence
may only represent an explicitly unmodified canonical sequence. sequence_one_letter must
use uppercase ASCII residue codes without whitespace or modification notation. Use an empty
sequence_one_letter when the notation is modified or cannot be represented unambiguously;
preserve every modification and closure annotation in sequences[].sequence_text. Do not infer a missing cyclic linkage or residue.
R-group definitions require actual variable-definition evidence. has_sequence_table is true
only when a visible table actually contains peptide/biopolymer sequences or their typed
residue notation. Assay/ADMET tables alone are not sequence tables.
Unknown string fields are empty, unknown lists are empty. Do not invent entities."""

_LEDGER_HEADER = """ENTITIES ALREADY KNOWN IN THIS DOCUMENT (resolve references against these):
{ledger}
"""


def _render_page(pdf_file: str, page_num: int, dpi: int = 150) -> str:
    import fitz

    with fitz.open(pdf_file) as document:
        if type(page_num) is not int or not 1 <= page_num <= document.page_count:
            raise ModelEvidenceError(f"Invalid source page: {page_num!r}")
        pixmap = document[page_num - 1].get_pixmap(dpi=dpi)
    descriptor, path = tempfile.mkstemp(prefix="document_page_", suffix=".png")
    os.close(descriptor)
    try:
        pixmap.save(path)
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    return path


def _require_keys(item, keys, context):
    if not isinstance(item, dict) or set(item) != set(keys):
        raise ModelContractError(f"{context} requires exactly {', '.join(keys)}")


def _require_strings(item, keys, context):
    for key in keys:
        if not isinstance(item[key], str):
            raise ModelContractError(f"{context}.{key} must be a string")


def validate_one_letter_sequence(sequence: str, context: str) -> None:
    if not isinstance(sequence, str) or any(
        not "A" <= letter <= "Z" for letter in sequence
    ):
        raise ModelContractError(
            f"{context} must contain uppercase ASCII residue codes without whitespace or modification notation"
        )


def parse_page_entities(response_text: str) -> dict:
    payload = require_json_object(
        response_text, "extract_page_entities", allow_wrapped=False
    )
    _require_keys(payload, _PAGE_SCHEMA["required"], "page")
    for key in ("compounds", "sequences", "rgroup_definitions"):
        if not isinstance(payload[key], list):
            raise ModelContractError(f"page.{key} must be a list")
    if type(payload["has_sequence_table"]) is not bool:
        raise ModelContractError("page.has_sequence_table must be a boolean")
    _require_strings(payload, ("page_summary",), "page")
    compound_keys = _COMPOUND_SCHEMA["required"]
    for index, compound in enumerate(payload["compounds"]):
        context = f"page.compounds[{index}]"
        _require_keys(compound, compound_keys, context)
        _require_strings(compound, compound_keys, context)
        validate_one_letter_sequence(
            compound["sequence_one_letter"], f"{context}.sequence_one_letter"
        )
        if compound["structure_type"] not in _STRUCTURE_TYPES:
            raise ModelContractError(f"{context}.structure_type is invalid")
        if compound["evidence_kind"] not in _EVIDENCE_KINDS:
            raise ModelContractError(f"{context}.evidence_kind is invalid")
        if not compound["evidence"].strip():
            raise ModelContractError(f"{context}.evidence must not be empty")
    sequence_keys = _SEQUENCE_SCHEMA["required"]
    for index, sequence in enumerate(payload["sequences"]):
        context = f"page.sequences[{index}]"
        _require_keys(sequence, sequence_keys, context)
        _require_strings(sequence, sequence_keys[:-1], context)
        validate_one_letter_sequence(
            sequence["sequence_one_letter"], f"{context}.sequence_one_letter"
        )
        if type(sequence["cyclic"]) is not bool:
            raise ModelContractError(f"{context}.cyclic must be a boolean")
        if (
            not sequence["sequence_one_letter"].strip()
            and not sequence["sequence_text"].strip()
        ):
            raise ModelContractError(f"{context} requires sequence text")
    for index, definition in enumerate(payload["rgroup_definitions"]):
        context = f"page.rgroup_definitions[{index}]"
        _require_keys(definition, ("variable", "substituents"), context)
        if (
            not isinstance(definition["variable"], str)
            or not definition["variable"].strip()
        ):
            raise ModelContractError(f"{context}.variable must be a non-empty string")
        if not isinstance(definition["substituents"], list) or any(
            not isinstance(value, str) for value in definition["substituents"]
        ):
            raise ModelContractError(f"{context}.substituents must be a string list")
    return payload


def parse_sequence_entries(response_text: str) -> dict:
    payload = require_json_object(
        response_text, "extract_sequence_table", allow_wrapped=False
    )
    _require_keys(payload, ("sequence_entries",), "sequence_table")
    if not isinstance(payload["sequence_entries"], list):
        raise ModelContractError("sequence_table.sequence_entries must be a list")
    string_keys = (
        "seq_id",
        "sequence_text",
        "procedure",
        "molecular_formula",
        "observed_ion",
    )
    number_keys = ("molecular_weight", "exact_mass", "purity_percent", "measured_mass")
    for index, entry in enumerate(payload["sequence_entries"]):
        context = f"sequence_table.sequence_entries[{index}]"
        _require_keys(entry, (*string_keys, *number_keys), context)
        _require_strings(entry, string_keys, context)
        if not entry["sequence_text"].strip():
            raise ModelContractError(f"{context}.sequence_text must not be empty")
        for key in number_keys:
            value = entry[key]
            if value is not None and (
                type(value) not in (int, float)
                or (type(value) is float and not math.isfinite(value))
            ):
                raise ModelContractError(
                    f"{context}.{key} must be a finite number or null"
                )
    return payload


class PageEntityExtractor:
    def __init__(self, audit_path: str | Path | None = None, *, check_cancel=None):
        self.audit_path = audit_path
        self.check_cancel = check_cancel

    def extract_page(
        self, pdf_file: str, page_num: int, ledger_digest: str = ""
    ) -> dict:
        from utils.llm_utils import run_vision_json_task

        image = _render_page(pdf_file, page_num)
        try:
            prompt = _PAGE_SCHEMA_INSTRUCTION
            if ledger_digest:
                prompt += "\n\n" + _LEDGER_HEADER.format(ledger=ledger_digest)
            payload = run_vision_json_task(
                task_name="extract_page_entities",
                image_file=image,
                prompt=prompt,
                parser=parse_page_entities,
                response_schema=_PAGE_SCHEMA,
                preserve_response=True,
                audit_path=self.audit_path,
                metadata={
                    "page": page_num,
                    "pdf_file": os.fspath(pdf_file),
                    "scope": "document_page_entities",
                },
            )
            entries = []
            if payload["has_sequence_table"]:
                if self.check_cancel:
                    self.check_cancel()
                table = run_vision_json_task(
                    task_name="extract_sequence_table",
                    image_file=image,
                    prompt=_SEQUENCE_TABLE_PROMPT,
                    parser=parse_sequence_entries,
                    response_schema=_SEQUENCE_TABLE_SCHEMA,
                    preserve_response=True,
                    audit_path=self.audit_path,
                    metadata={
                        "page": page_num,
                        "pdf_file": os.fspath(pdf_file),
                        "scope": "document_sequence_table",
                    },
                )
                entries = table["sequence_entries"]
                if not entries:
                    raise ModelEvidenceError(
                        f"Sequence table on page {page_num} produced no rows"
                    )
            return {
                "page_num": page_num,
                "compounds": payload["compounds"],
                "sequences": payload["sequences"],
                "sequence_entries": entries,
                "rgroup_definitions": payload["rgroup_definitions"],
                "page_summary": payload["page_summary"],
                "ok": True,
                "inventory_version": 2,
            }
        finally:
            try:
                Path(image).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Cannot remove page image %s: %s", image, exc)

    def extract_pages(self, pdf_file: str, pages: list[int]) -> list[dict]:
        records = []
        entities: list[str] = []
        for page_num in sorted(set(pages)):
            if self.check_cancel:
                self.check_cancel()
            digest = "\n".join(entities[-40:])
            record = self.extract_page(pdf_file, page_num, digest)
            records.append(record)
            for compound in record["compounds"]:
                label = compound["label_in_page"].strip()
                name = compound["name"].strip()
                if label or name:
                    entities.append(f"page {page_num}: [{label}] {name}")
            for sequence in record["sequences"]:
                if sequence["seq_id"].strip():
                    entities.append(
                        f"page {page_num}: {sequence['seq_id']} sequence recorded"
                    )
        if self.check_cancel:
            self.check_cancel()
        return records
