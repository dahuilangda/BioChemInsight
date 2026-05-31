---
name: biocheminsight-text-models
description: Use when a BioChemInsight language model needs to normalize assay markdown into JSON or extract a final compound ID from free-form textual reasoning or OCR output.
model: inherit
effort: high
context: auto
---

# BioChemInsight Text Models

Use this skill for runtime text-model tasks.

This skill inherits shared runtime conventions from `biocheminsight-model-common` and only overrides text-task-specific behavior.

## Scope
- Assay markdown/table extraction into strict JSON dictionaries
- Compound-ID extraction from free-form text or model reasoning
- Stable parsing rules for downstream machine-readable outputs

## Runtime rules
1. Output machine-readable JSON only when the caller asks for JSON.
2. Never hallucinate compound IDs outside the provided allowlist.
3. Represent assay cells as rich objects: `value`, `unit`, `method`, `description`, `confidence`, `reason`.
4. Keep prompt contracts short; use examples and schemas for detailed shape.
5. Prefer returning `"None"` over speculative IDs.
6. For page-detection tasks, emit exactly one `decisions` item per provided page.
7. For assay extraction, preserve method/unit/symbol descriptions in the rich
   assay object so exports do not lose scientific context.
8. Treat continuation context as read-only table metadata. It may explain
   headers, units, methods, and assay ownership for a current page/table chunk,
   but output records must come only from the current chunk.
9. Use a planning step for multi-page assay extraction when page/table
   continuation is ambiguous. The planner decides context inheritance; extractor
   prompts execute that plan without inventing records.
10. If a bioactivity page overlaps a structure page, allow the planner to choose
    a visual-structure anchor strategy. Text extraction must not invent compound
    IDs; visual anchoring should be handled by a bounded harness step with source
    page and structure-record provenance.

## Runtime references
- `references/content_to_dict_prompt.md`
- `references/content_to_multi_assay_dict_prompt.md`
- `references/detect_assay_pages_prompt.md`
- `references/plan_assay_extraction_context_prompt.md`
- `references/get_compound_id_from_description_prompt.md`
- `references/resolve_compound_id_alias_prompt.md`
- `references/runtime.json`
- `references/output_schemas.json`
