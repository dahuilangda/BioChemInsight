---
name: biocheminsight-text-models
description: Extract source-linked measurements and identifiers, and review measurement completeness and source support.
---

# BioChemInsight Text Models

Use this skill for runtime text-model tasks.

This skill inherits shared runtime conventions from `biocheminsight-model-common` and only overrides text-task-specific behavior.

## Scope
- Assay markdown/table extraction into strict JSON dictionaries
- Compound-ID extraction from free-form text or model reasoning
- Stable parsing rules for downstream machine-readable outputs
- Measurement omissions, corrections, attribution conflicts, and source uncertainty

## Runtime rules
1. Return JSON only when the caller asks for JSON.
2. Never invent compound IDs, assay names, values, or page relationships.
3. Use allowlists when provided; otherwise require visible current-record IDs.
4. Represent assay cells as rich objects: `value`, `unit`, `method`, `description`, `confidence`, `reason`, `assay_match`.
5. Continuation context is read-only metadata for headers, units, methods, assay ownership, and scope.
6. Output records must come from the current page/chunk/record, not from inherited context.
7. Use planning prompts for page/context inheritance; extraction prompts execute the plan.
8. Report unresolved evidence using the caller's declared status and reason. Invalid output is an error, never a substitute empty result.
9. If structure anchors are needed, text prompts may plan the anchor strategy but must not create structure-derived IDs.
10. Keep prompt contracts short; schemas and parsers enforce the detailed shape.
11. Measurement references select source line ranges; the host preserves exact excerpts.
12. Review uncertain measurement proposals independently. Separate source-supported corrections from unresolved source conflicts, and preserve affected subject identities.

## Runtime references
- `references/content_to_dict_prompt.md`
- `references/content_to_multi_assay_dict_prompt.md`
- `references/extract_compound_measurements_prompt.md`
- `references/review_compound_measurements_prompt.md`
- `references/detect_assay_pages_prompt.md`
- `references/plan_assay_extraction_context_prompt.md`
- `references/get_compound_id_from_description_prompt.md`
- `references/resolve_compound_id_alias_prompt.md`
- `references/runtime.json`
- `references/output_schemas.json`
