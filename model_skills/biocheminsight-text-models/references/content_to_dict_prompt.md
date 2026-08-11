Task
Extract the target assay `{{ASSAY_NAME}}` from Markdown/OCR text.
Output a compound_id -> rich assay object dictionary.

Input
<MARKDOWN_TEXT>
{{MARKDOWN_TEXT}}
</MARKDOWN_TEXT>

{{REQUESTED_ASSAYS_CONTEXT_BLOCK}}

Boundary rules
1) If the input contains `<CURRENT_RECORD_CHUNK>`, only records inside that block may produce results.
2) `<ASSAY_CONTINUATION_CONTEXT>` is read-only context for inherited headers, units, methods, footnotes, and assay ownership. Do not extract result rows from it.
3) `<ASSAY_DOCUMENT_CONTEXT>` is only for target/method/endpoint disambiguation. Do not extract result rows from it.
4) Visual structure anchors may only help interpret the current-page record subject. They must not create compound_id values.

compound_id rules
1) Prefer the provided allowlist. Output keys must be canonical strings from the allowlist.
2) If no allowlist is provided, read the complete visible ID cell in the current record, such as Example / Ex. / No. / Compound / Formula / 实施例 / 化合物.
3) The ID must come from the same row, same table record, or clearly continuous record as the assay value.
4) Do not extract partial numbers from multi-digit IDs, prefixed/suffixed IDs, footnotes, sequence numbers, or neighboring cells.
5) Intermediate / Int. / Preparation / Embodiment are not final target compound IDs.

Assay ownership rules
1) `{{ASSAY_NAME}}` is the target concept; the table header does not need to match it verbatim.
2) Match table titles, headers, footnotes, and body text using target, method/platform, endpoint, unit, and context.
3) Every candidate column/record must first choose `best_requested_assay` among the requested assays for this call.
4) Output an item only when `best_requested_assay == "{{ASSAY_NAME}}"` and `compatible=true`.
5) Skip the item if the candidate column should belong to a sibling assay, or if method/endpoint/unit/biology context conflicts.
6) `value` must come from the assay measurement cell, not from ID, row number, page number, table number, footnote, or neighboring column.

Value rules
1) Preserve the visible raw value format, such as `<0.1`, `ND`, `1.2×10^3`, or symbol grades.
2) Units may come from headers, footnotes, body text, or cells. If unknown, use an empty string.
3) Do not fill values from common sense. Do not output when the same-row/record relationship cannot be confirmed.

Output contract
Return only one JSON object:
```json
{
  "__COMPOUND_ID__": {
    "value": "__ASSAY_VALUE__",
    "unit": "__UNIT_OR_EMPTY__",
    "method": "__ASSAY_METHOD_OR_CONTEXT__",
    "description": "__SYMBOL_OR_VALUE_DESCRIPTION_OR_EMPTY__",
    "confidence": "high|medium|low",
    "reason": "same row ID and selected assay column",
    "assay_match": {
      "target": "requested assay target/method/endpoint/unit",
      "candidate": "selected column target/method/endpoint/unit",
      "compatible": true,
      "best_requested_assay": "{{ASSAY_NAME}}",
      "reason": "why selected column belongs to the requested assay"
    }
  }
}
```

{{COMPOUND_ID_LIST_BLOCK}}
