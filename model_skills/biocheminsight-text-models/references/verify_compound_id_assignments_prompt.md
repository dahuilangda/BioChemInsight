Task
Review whether each compound_id in the assay extraction result is the complete final compound ID from the same row or record, and map it to the allowlist.

Allowlist
{{COMPOUND_ID_LIST_JSON}}

OCR / Markdown
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

Results to review
{{ASSAY_PAYLOAD_JSON}}

Rules
1) Judge each current compound_id independently.
2) The ID must be in the same row, same table record, or clearly continuous record as the assay value.
3) Do not treat row numbers, page numbers, table numbers, footnotes, cell indices, value fragments, or multi-digit ID fragments as compound_id.
4) If the current key is only a fragment of a complete ID, set `valid_current_id=false`; if the allowlist contains a confirmable complete ID, fill in that original string.
5) Format differences or aliases may be mapped only when context confirms the same final compound.
6) Choose only raw strings from the allowlist; use `"None"` when you cannot confirm.

Output contract
Return only JSON:
```json
{
  "__CURRENT_COMPOUND_ID__": {
    "valid_current_id": true,
    "canonical_compound_id": "__ALLOWLIST_ID_OR_None__",
    "confidence": "high|medium|low",
    "reason": "short reason"
  }
}
```
