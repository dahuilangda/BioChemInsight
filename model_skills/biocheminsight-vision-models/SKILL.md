---
name: biocheminsight-vision-models
description: Use when a BioChemInsight vision model needs to classify candidate structure images, reject Markush or fragment noise, or extract a compound ID from a highlighted structure image.
model: inherit
effort: high
context: auto
---

# BioChemInsight Vision Models

Use this skill for runtime vision-model tasks.

This skill inherits shared runtime conventions from `biocheminsight-model-common` and only overrides vision-task-specific behavior.

## Scope
- Structure candidate filtering before OCR/SMILES/ID extraction
- Conservative complete-compound gating
- Compound-ID extraction from highlighted structure images
- Visual audit of requested assay cells, including visible units and symbol/grade descriptions

## Runtime rules
1. Be conservative: cropped or ambiguous candidates must not pass as complete compounds.
2. Any variable-group placeholder means Markush, not a complete compound.
3. If a bond, atom, ring, label, or substituent is clipped by the image boundary, treat it as incomplete.
4. If the image is mainly text, arrows, legends, tables, or multiple unrelated objects, treat it as noise.
5. Respect repository strictness policy: `strict`, `balanced`, or `permissive`.
6. For assay-cell audit, do not rewrite tables; only verify requested cells and report visible `value`, `unit`, and symbol/grade `description`.
7. For structure-page detection, emit exactly one `decisions` item per provided page.
8. For structure ID extraction, use `COMPOUND_ID="None"` only with `ID_SOURCE="none"` and evidence explaining why no reliable ID is visible.

## Runtime references
- `references/classify_structure_candidate_prompt.md`
- `references/classify_structure_crop_check_prompt.md`
- `references/classify_structure_border_review_prompt.md`
- `references/detect_structure_pages_prompt.md`
- `references/review_structure_pages_prompt.md`
- `references/structure_to_id_prompt.md`
- `references/review_structure_primary_id_prompt.md`
- `references/runtime.json`
- `references/output_schemas.json`
