Task
Use OCR/Markdown to determine which pages contain extractable compound-level bioactivity or assay results.

Assay hints provided by the caller
{{ASSAY_NAMES_JSON}}

Page OCR
{{PAGES_JSON}}

Positive examples
- Tables or table-like records where compound/example rows appear together with measured values
- Extractable result columns such as potency, binding, percent inhibition, selectivity, degradation, or phenotypic response
- OCR with substantial noise that still clearly shows compound-level result records

Negative examples
- Assay protocol or method text with no compound result rows
- Synthesis, claims, definitions, structure/name tables, reaction schemes, references
- Mentions of an assay name or endpoint without extractable results
- Structure drawings with no bioactivity result table

Rules
1) Do not rely on a single keyword. Use table structure, units, endpoint, row records, and context.
2) The assay name must be supported by OCR evidence; do not invent it.
3) If a result column has both target-specific and generic names, keep the name that best represents the result.
4) Every input page must produce one decision.

Output contract
Return only JSON:
```json
{
  "assay_pages": [1],
  "assay_names": ["visible assay name"],
  "decisions": [
    {
      "page": 1,
      "has_assay_data": true,
      "confidence": "high|medium|low",
      "assay_names": ["names visible on this page"],
      "reason": "short semantic reason"
    }
  ]
}
```
