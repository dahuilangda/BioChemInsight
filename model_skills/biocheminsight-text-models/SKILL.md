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
1. Return JSON only when the caller asks for JSON.
2. Never invent compound IDs, assay names, values, or page relationships.
3. Use allowlists when provided; otherwise require visible current-record IDs.
4. Represent assay cells as rich objects: `value`, `unit`, `method`, `description`, `confidence`, `reason`, `assay_match`.
5. Continuation context is read-only metadata for headers, units, methods, assay ownership, and scope.
6. Output records must come from the current page/chunk/record, not from inherited context.
7. Use planning prompts for page/context inheritance; extraction prompts execute the plan.
8. Prefer `"None"`, `{}`, or `extract=false` over speculative output.
9. If structure anchors are needed, text prompts may plan the anchor strategy but must not create structure-derived IDs.
10. Keep prompt contracts short; schemas and parsers enforce the detailed shape.

## Runtime references
- `references/content_to_dict_prompt.md`
- `references/content_to_multi_assay_dict_prompt.md`
- `references/detect_assay_pages_prompt.md`
- `references/plan_assay_extraction_context_prompt.md`
- `references/get_compound_id_from_description_prompt.md`
- `references/resolve_compound_id_alias_prompt.md`
- `references/runtime.json`
- `references/output_schemas.json`
