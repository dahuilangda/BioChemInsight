Task
Independently verify whether each `value` is the true target assay measurement.
If needed, correct `value` and `unit` using OCR evidence from the same row or record.

Target assay
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
1) For each compound_id, locate the same row, same table record, or clearly continuous record independently.
2) `value` must come from the target assay measurement cell, not from ID, row number, page number, table number, footnotes, condition values, NMR/MS, or neighboring assay columns.
3) If the current value was mistakenly taken from the ID or row number, but the correct assay value is confirmable in the same row, return `valid_assay_value=true` and provide the corrected value.
4) If the current value is correct, also return `valid_assay_value=true` and fill the current or confirmed value and unit.
5) If assay assignment conflicts or the same row/record cannot be confirmed, return `valid_assay_value=false` and `corrected_value="None"`.
6) Do not fill values from common sense.
7) If the located value was produced by a computational procedure — derived, estimated, simulated, or predicted — rather than an experimental measurement, return `valid_assay_value=false` and `corrected_value="None"`.

Output contract
Return only JSON:
```json
{
  "__COMPOUND_ID__": {
    "valid_assay_value": true,
    "corrected_value": "__MEASURED_VALUE_OR_None__",
    "corrected_unit": "__UNIT_OR_EMPTY__",
    "confidence": "high|medium|low",
    "reason": "short reason"
  }
}
```
