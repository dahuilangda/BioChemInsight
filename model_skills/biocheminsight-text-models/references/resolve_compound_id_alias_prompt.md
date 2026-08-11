Task
Map the raw compound ID alias to a unique canonical Compound ID in the allowlist.

Input
<RAW_ID>
{{RAW_ID}}
</RAW_ID>

<OPTIONAL_CONTEXT>
{{OPTIONAL_CONTEXT}}
</OPTIONAL_CONTEXT>

<ALLOWLIST>
{{ALLOWLIST}}
</ALLOWLIST>

Rules
1) Output only one raw string from the allowlist, or `"None"`.
2) Case, whitespace, punctuation, parentheses, common prefixes, and minor OCR differences are allowed, but the result must point to the same final/target compound.
3) If multiple allowlist items could match and the context is not unique, return `"None"`.
4) Intermediate / Int. / Preparation / Embodiment are not automatically equivalent to Example / Compound / No.
5) Do not treat row numbers, page numbers, table numbers, or spectrum/condition values as compound IDs.
6) If the evidence is not enough to map uniquely to the allowlist, return `"None"` and `CONFIDENCE="low"`.

Output contract
Return only JSON:
```json
{
  "COMPOUND_ID": "<ALLOWLIST item or None>",
  "CONFIDENCE": "high|medium|low",
  "REASON": "short reason"
}
```
