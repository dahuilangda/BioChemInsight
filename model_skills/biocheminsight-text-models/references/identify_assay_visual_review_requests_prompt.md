Task
Decide which extracted assay cells need a vision model to reread the image.
Only decide whether visual rereading is needed. Do not infer corrected values.

OCR / Markdown
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

Extracted assay results
{{ASSAY_DICTS_JSON}}

Parsed table grids
{{PARSED_TABLES_JSON}}

Rules
1) Use full OCR context: title, grouped headers, column names, units, other values in the same column, other assay values in the same row, footnotes, and continuation tables.
2) Do not use fixed single-character mappings, and do not request a reread just because a value looks unusual.
3) When an entire column is symbol grades, qualitative grades, response grades, mixed glyphs/short codes, and OCR is easy to confuse, list the relevant cells for that assay in the current chunk.
4) For normal numeric columns with clear units and stable same-column value ranges, do not request reread only because the value is short.
5) Output only cells that need visual rereading. Do not include others.

Output contract
Return only JSON:
```json
{
  "Assay Name": [
    {
      "compound_id": "Example 1",
      "ocr_value": "raw value",
      "unit": "",
      "method": "assay method",
      "description": "",
      "confidence": "high|medium|low",
      "reason": "why this cell needs visual reread",
      "context": "optional table/row/column context"
    }
  ]
}
```
