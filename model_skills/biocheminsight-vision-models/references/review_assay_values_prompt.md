Task
Review specified cells in the current assay extraction draft against the table image.
Review only requested items. Do not rewrite the whole table, and do not infer invisible values.

Current draft
{{ASSAY_DICTS_JSON}}

Requested review items
{{REVIEW_PAYLOAD_JSON}}

Rules
1) Review only the requested assay / compound_id / cell. Do not expand to other cells.
2) Use `replace` only when the visual value is clearly readable and differs from the draft.
3) Use `keep` when the draft is visually confirmed as correct.
4) Use `uncertain` when the cell is unclear or cannot be verified reliably.
5) Preserve glyphs actually visible in the image, such as `<`, `>`, `+`, `*`, `%`, `NA`.
6) Write `unit` and `description` only from visible content; otherwise use an empty string.
7) Do not guess values from chemical intuition, ordering, or neighboring rows.

Output contract
Return only JSON:
```json
{
  "corrections": [
    {
      "assay_name": "...",
      "compound_id": "...",
      "current_value": "...",
      "visual_value": "...",
      "unit": "...",
      "description": "...",
      "action": "keep|replace|uncertain",
      "confidence": "high|medium|low",
      "evidence": "short visual reason"
    }
  ]
}
```
