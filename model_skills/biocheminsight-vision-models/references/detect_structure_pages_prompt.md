Task
Review the PDF page thumbnail contact sheet and identify which pages show visible chemical structure drawings.

Positive examples
- Visible 2D molecular line-bond drawings
- Visible structure drawings inside reaction schemes
- Visible Markush/core scaffold/R-group structure drawings
- Structure tables with cells that actually contain structure drawings

Negative examples
- Pure text, claims, definitions, references, indexes
- Assay tables, name tables, or ID tables with no structure drawing
- Only words such as Example / Formula / compound without a visible structure drawing
- Charts, screenshots, stamps, non-chemical graphics

Rules
1) Use visual evidence, not OCR keywords.
2) Set `has_structure=true` only when a structure drawing is actually visible.
3) Be conservative for text pages and uncertain pages.
4) Output exactly one decision for each provided page.

Page list
{{PAGES_JSON}}

Output contract
Return only JSON:
```json
{
  "decisions": [
    {
      "page": 1,
      "has_structure": true,
      "confidence": "high|medium|low",
      "reason": "short visual reason"
    }
  ]
}
```
