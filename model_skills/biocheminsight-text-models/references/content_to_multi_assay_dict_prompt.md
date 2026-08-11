Task
Extract multiple requested assays from Markdown/OCR text at the same time.
The top level must contain the original name of every requested assay.

Target assay list
{{ASSAY_NAMES_JSON}}

Input
<MARKDOWN_TEXT>
{{MARKDOWN_TEXT}}
</MARKDOWN_TEXT>

Boundary rules
1) If the input contains `<CURRENT_RECORD_CHUNK>`, only records inside that block may produce results.
2) `<ASSAY_CONTINUATION_CONTEXT>` is read-only context for inherited headers, units, methods, footnotes, and assay ownership. Do not extract result rows from it.
3) `<ASSAY_DOCUMENT_CONTEXT>` is only for target/method/endpoint disambiguation. Do not extract result rows from it.
4) Visual structure anchors may only help interpret the current-page record subject. They must not create compound_id values.

compound_id rules
1) When an allowlist is provided, each child-dictionary key must be the canonical string from the allowlist.
2) When no allowlist is provided, read the complete visible ID cell from the current record.
3) The ID must come from the same row, same table record, or clearly continuous record as the assay value.
4) Do not treat row indices, page numbers, footnotes, ID fragments, or assay values as compound_id.
5) Intermediate / Int. / Preparation / Embodiment are not final target compound_id values.

Assay ownership rules
1) For each candidate column/record, choose exactly one `best_requested_assay` using target, method/platform, endpoint, unit, and context.
2) Write into a top-level assay only when `best_requested_assay` equals that top-level assay name and `compatible=true`.
3) A broader assay cannot take a column that belongs to a more specific sibling assay, unless context clearly shows the column belongs only to the broader assay.
4) An exact header string does not get automatic priority; judge using the full page, continuation, and task context.
5) Do not omit, rename, or merge top-level assays. If an assay has no results, output an empty object `{}`.

Value rules
1) `value` must come from the assay measurement cell, not from an ID cell or neighboring column.
2) Preserve the visible raw value format. If the unit is unknown, use an empty string.
3) For symbol/grade/non-numeric columns, write `description` only when context is sufficient to explain the meaning; otherwise use an empty string.
4) Do not output an item when the same-row/record relationship cannot be confirmed.

Output contract
Return only one JSON object:
```json
{
  "__REQUESTED_ASSAY_NAME__": {
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
        "best_requested_assay": "__REQUESTED_ASSAY_NAME__",
        "reason": "why selected column belongs to this requested assay"
      }
    }
  }
}
```

{{COMPOUND_ID_LIST_BLOCK}}
