Task
Merge the OCR/text assay draft with the cell-by-cell visual review report and output the final assay dictionary.

OCR / Markdown
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

Current draft
{{ASSAY_DICTS_JSON}}

Visual review report
{{VISUAL_REPORT_JSON}}

Rules
1) Preserve the top-level assay-name structure from the current draft.
2) Replace `value` only for items in the visual report with `action="replace"`, `confidence="high"`, and a non-empty `visual_value`.
3) If the visual report provides `unit` or `description`, update those fields too.
4) Keep the draft unchanged for `keep`, `uncertain`, and low/medium-confidence `replace` items.
5) Do not add compound_id entries that do not exist in the draft; do not delete existing compound_id entries.
6) Do not infer character mappings on your own. The visual report is the basis for changing values; OCR is only for confirming range.
7) Keep every final item as a rich assay object and preserve `assay_match`.
8) If the visual report proves the original assignment was wrong, you may set `assay_match.compatible=false` and remove that item from the final output; otherwise keep it.

Output contract
Return only JSON in the same format as the current draft:
```json
{
  "Assay Name": {
    "Compound ID": {
      "value": "final value",
      "unit": "unit or empty",
      "method": "method/context",
      "description": "description or empty",
      "confidence": "high|medium|low",
      "reason": "short reason",
      "assay_match": {
        "target": "...",
        "candidate": "...",
        "compatible": true,
        "best_requested_assay": "Assay Name",
        "reason": "..."
      }
    }
  }
}
```
