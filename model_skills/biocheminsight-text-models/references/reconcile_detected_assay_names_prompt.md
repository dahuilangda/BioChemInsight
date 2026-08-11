Task
Review candidate assay names and output the minimal name set to keep for formal extraction.

Candidate names
{{ASSAY_NAMES_JSON}}

Page-level evidence
{{PAGE_DECISIONS_JSON}}

OCR / Markdown
<OCR_CONTEXT>
{{OCR_CONTEXT}}
</OCR_CONTEXT>

Rules
1) Choose only from the original candidate strings; do not create new names.
2) Keep names that truly correspond to a compound-level result column, result table, result record, or result value domain.
3) When multiple names point to the same result, keep the most specific one that best explains target/method/endpoint/unit.
4) Protocol text, background description, target-only text, or method paragraphs are not independent assays unless they correspond to independent result records.
5) If it is unclear whether two candidates are the same result, keep both.
6) Use `keep=false` only when high-confidence evidence shows that the candidate is covered by a kept name.
7) `keep=true` must include real page numbers and result evidence type from page-level evidence.

Output contract
Return only JSON:
```json
{
  "assay_names": ["__KEPT_ORIGINAL_CANDIDATE_NAME__"],
  "decisions": {
    "__ORIGINAL_CANDIDATE_NAME__": {
      "keep": true,
      "canonical_assay_name": "__KEPT_ORIGINAL_CANDIDATE_NAME__",
      "confidence": "high|medium|low",
      "evidence_kind": "result_column|result_table|result_record|result_value|result_unit",
      "evidence_pages": [1],
      "evidence_summary": "short evidence summary",
      "reason": "short reason"
    }
  }
}
```
