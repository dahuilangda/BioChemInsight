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
1. Use visible evidence only; do not complete structures, IDs, values, or roles from chemistry intuition.
2. Cropped or ambiguous candidates must not pass as complete compounds.
3. Explicit variable placeholders or variable bonds are Markush, not complete compounds.
4. Text, arrows, legends, tables, or multiple unrelated objects are not single complete structures.
5. Respect repository strictness policy: `strict`, `balanced`, or `permissive`.
6. For assay-cell audit, verify only requested cells and report visible `value`, `unit`, and symbol/grade `description`.
7. For page detection, emit exactly one `decisions` item per provided page.
8. For structure ID extraction, `COMPOUND_ID="None"` must use `ID_SOURCE="none"` and visible evidence.
9. Markush assembly evidence is layered: text assignment, visual attachment, and MolNexTR graph evidence are distinct.

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
