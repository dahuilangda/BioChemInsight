Task
Audit OCR/text-model assay extraction against the table image.

Inputs
Current extracted draft:
{{ASSAY_DICTS_JSON}}

Requested visual audit items:
{{REVIEW_PAYLOAD_JSON}}

Rules
1. Use the image only to audit the requested table cells against the current extracted draft.
2. Do not infer values from chemistry, compound order, or surrounding examples.
3. Do not rewrite the whole assay table.
4. For each requested item, find the matching row/cell in the visible table pages and report whether the draft value is visually correct.
5. Use `action="replace"` only when the visual cell is readable and differs from the draft.
6. Use `action="keep"` when the draft is visually correct.
7. Use `action="uncertain"` when the cell cannot be reliably verified.
8. Preserve the table's actual glyphs exactly when visible, including +, *, dagger/cross-like symbols, percentages, NA, or numbers.
9. Do not apply hard-coded mappings from OCR noise; read the glyphs visually.
10. Always report `confidence` as `high` / `medium` / `low`.
11. If you can only weakly see a differing value, prefer `action="uncertain"` over a low-confidence `replace`.

Output
Return JSON only:
{
  "corrections": [
    {
      "assay_name": "...",
      "compound_id": "...",
      "current_value": "...",
      "visual_value": "...",
      "action": "keep|replace|uncertain",
      "confidence": "high|medium|low",
      "evidence": "short visual reason"
    }
  ]
}
