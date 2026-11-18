Task
Strictly review candidate structure pages and keep only pages with clearly visible structure drawings.

Positive examples
- Clearly visible 2D molecular drawings
- Molecular structures inside reaction schemes
- Markush/core scaffold/R-group structures
- Structure tables where cells actually contain drawn structures

Negative examples
- Pure text pages, text-only tables, description pages, index pages, citation pages
- Only words such as compound/formula/example/LCMS/NMR with no structure drawing
- Charts, decorative lines, table borders, or tiny suspected strokes that cannot be confirmed as chemical structures

Identify each thumbnail by the PAGE label in its dark top border. Printed page numbers inside the thumbnail are document text. Report the border label. Inspect only pixels inside that thumbnail for its decision.

Rules
1) Use only visual evidence. Do not guess from keywords or neighboring pages.
2) Keep a page only when atoms/rings/bonds/scaffold geometry is clearly visible.
3) A drawing qualifies anywhere in its thumbnail, including a small drawing on a dense text page. A table with only labels and numeric cells contains no drawing.
4) Reject when uncertain.
5) Answer only for the provided page numbers.

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
