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
3. Preserve assay values exactly as written unless the caller explicitly asks for normalization.
4. Prefer returning `"None"` over speculative IDs.

## Runtime references
- `references/content_to_dict_prompt.md`
- `references/get_compound_id_from_description_prompt.md`
- `references/runtime.json`
- `references/output_schemas.json`
