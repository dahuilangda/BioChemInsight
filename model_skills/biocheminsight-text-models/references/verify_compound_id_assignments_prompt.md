Task
Review whether each compound_id in the assay extraction result is the complete final compound ID from the same row or record, and map it to the allowlist when the source supports the mapping.

Allowlist
{{COMPOUND_ID_LIST_JSON}}

OCR / Markdown
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

Results to review
{{ASSAY_PAYLOAD_JSON}}

Rules
0) Return one entry for EVERY compound_id in the results, in the same order; never
   review only the first one.
1) Judge each current compound_id independently.
2) The ID must be in the same row, same table record, or clearly continuous record as the assay value.
3) Do not treat row numbers, page numbers, table numbers, footnotes, cell indices, value fragments, or multi-digit ID fragments as compound_id.
4) If the current key is only a fragment of a complete ID, set `valid_current_id=false`; if the allowlist contains a confirmable complete ID, fill in that original string.
5) Identify the printed subject of the value from its row or complete column header. An allowlist entry alone gives no ownership evidence. Use an alias only if the source explicitly links the printed subject to that ID. A generic formula or series cannot represent an individually named comparator.
6) Choose only raw strings from the allowlist for canonical_compound_id, or use `"None"` when no mapping is supported. A complete, source-supported identifier can be valid even when canonical_compound_id is `"None"`.

Output contract
Return a JSON object with exactly one entry per compound_id received above:
```json
{
  "__CURRENT_COMPOUND_ID_1__": {
    "valid_current_id": true,
    "canonical_compound_id": "__ALLOWLIST_ID_OR_None__",
    "confidence": "high|medium|low",
    "reason": "short reason"
  },
  "__CURRENT_COMPOUND_ID_2__": {
    "valid_current_id": true,
    "canonical_compound_id": "__ALLOWLIST_ID_OR_None__",
    "confidence": "high|medium|low",
    "reason": "short reason"
  }
}
```
