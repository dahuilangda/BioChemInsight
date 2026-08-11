Task
Decide which requested assays in the current OCR/Markdown should proceed to formal extraction.

requested assays
{{ASSAY_NAMES_JSON}}

OCR / Markdown
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

Rules
1) Decide independently for each requested assay whether there is an extractable bioactivity record.
2) The current content must contain a compound-level result column, result table, or result record that belongs to that assay.
3) If one column resembles multiple requested assays, choose the most specific, direct, and explanatory assay for method/platform/endpoint/unit/context.
4) A broader assay cannot take a column that belongs to a more specific sibling assay.
5) An exact header string does not get automatic priority; use task and continuation context.
6) If the content contains only structure, synthesis, references, figures, or no numeric/graded results, set all `extract=false`.

Output contract
Return only JSON:
```json
{
  "assays": {
    "__REQUESTED_ASSAY_NAME__": {
      "extract": true,
      "confidence": "high|medium|low",
      "reason": "short reason"
    }
  }
}
```
