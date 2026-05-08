Task
You are reading OCR/markdown text from pages of a medicinal chemistry patent.
Classify which pages contain extractable bioactivity / assay result data.

Use semantic understanding of the OCR content. Do not rely on any single keyword.

Positive assay pages
- Pages with tables or table-like OCR listing compounds/examples and measured activity values.
- Pages with bioactivity result columns such as IC50, EC50, Ki, Kd, percent inhibition, potency, selectivity, binding, degradation, or similar measured endpoints.
- Pages where assay result data can plausibly be extracted for compound IDs.
- Pages may be noisy OCR from scanned patent images; infer tables from markdown, repeated rows, separators, numeric units, and assay/result context.

Negative pages
- Assay protocol/method descriptions without compound result rows.
- Synthesis examples, structure/name tables, reaction schemes, claims, definitions, background, references, search reports, or plain prose.
- Pages that merely mention an assay or endpoint but do not contain extractable compound-level result data.
- Pages with chemical structures but no bioactivity result table.

Assay names
- If an assay name or endpoint is visible, extract a concise user-facing name, e.g. "CDK2/cyclin E IC50", "NanoBRET CDK9 IC50", "Enzyme CDK1/cyclin B Ki".
- Prefer names that identify target + endpoint.
- Do not invent assay names not supported by the OCR.
- If the caller provides assay names, use them only as hints, not as mandatory matches.

Caller-provided assay-name hints:
{{ASSAY_NAMES_JSON}}

OCR pages in this batch:
{{PAGES_JSON}}

Output contract
Return JSON only:
{
  "assay_pages": [1, 2],
  "assay_names": ["name if present"],
  "decisions": [
    {
      "page": 1,
      "has_assay_data": true,
      "confidence": "high|medium|low",
      "assay_names": ["names visible on this page"],
      "reason": "short semantic reason"
    },
    {
      "page": 2,
      "has_assay_data": false,
      "confidence": "high|medium|low",
      "assay_names": [],
      "reason": "short semantic reason"
    }
  ]
}
