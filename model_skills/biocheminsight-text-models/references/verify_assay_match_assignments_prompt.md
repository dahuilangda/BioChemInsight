Task
Review which requested assay each candidate value in the current assay extraction result should belong to.
Do not add compound_id entries, and do not rewrite `value`.

Current top-level assay
{{ASSAY_NAME}}

requested assays
{{ASSAY_NAMES_JSON}}

OCR / Markdown
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

Results to review
{{ASSAY_PAYLOAD_JSON}}

Rules
0) Return one entry for EVERY compound_id received, in the same order.
1) Judge each compound_id independently for the best requested assay for its candidate column/record.
2) `best_requested_assay` must be the most specific, direct, and explanatory requested assay for method/platform/endpoint/unit/context.
3) If a sibling assay is more specific or direct, do not assign the candidate to the current broader assay.
4) An exact header string does not get automatic priority; use continuation, task, and document context.
5) If the candidate conflicts with the current assay on method/endpoint/unit/biology context, or belongs to another assay, set `compatible_current=false`.
6) If the candidate values are computed, simulated, or otherwise predicted by a computational procedure instead of experimentally measured, set `compatible_current=false`; predicted quantities are not assay results.

Output contract
Return only JSON:
```json
{
  "__COMPOUND_ID__": {
    "compatible_current": true,
    "best_requested_assay": "{{ASSAY_NAME}}",
    "confidence": "high|medium|low",
    "reason": "short reason"
  }
}
```
