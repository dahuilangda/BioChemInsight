Task
You are performing a strict second-pass visual review of candidate PDF page thumbnails from a medicinal chemistry patent.
Each thumbnail is labeled with its exact page number.

Goal
Keep only pages that clearly and visibly contain 2D chemical structure drawings suitable for downstream structure extraction.
This is a precision review: reject uncertain pages rather than guessing.

Positive page: set has_structure=true only if the thumbnail visibly shows at least one of:
- 2D molecular line-bond structure drawings.
- Reaction schemes where molecule structures are drawn.
- Markush/core scaffold drawings or R-group structure drawings.
- Structure/example tables where chemical structures themselves are drawn in cells.

Negative page: set has_structure=false for:
- Plain patent prose, claims, abstract, definitions, descriptions, examples, preparation text, references, search reports, indexes, or bibliographic pages without visible molecular drawings.
- Tables that contain only text, compound IDs, names, formulas, masses, NMR/LCMS values, assay values, page numbers, or metadata but no visible chemical drawings.
- Pages that mention words such as compound, formula, scheme, example, structure, intermediate, preparation, LCMS, NMR, or m/z but do not visibly show a drawn 2D structure.
- Charts, plots, screenshots, seals, signatures, forms, or non-chemical graphics.
- Ambiguous tiny marks, punctuation, table borders, text fragments, or decorative lines that could be mistaken for chemical bonds.

Strict decision rules
1. Use only visual evidence from the image, not OCR keywords or assumptions about surrounding pages.
2. A positive page must have clearly visible chemical drawing geometry: atoms/rings/bonds/scaffold/reaction molecules.
3. If the page looks mostly like dense text or a text-only table, reject it even if the topic is chemistry.
4. If you are unsure whether the visible marks are real chemical drawings, reject the page.
5. Do not add pages that are not in the provided page list.

Pages shown in this contact sheet:
{{PAGES_JSON}}

Output contract
Return JSON only. The response must contain exactly these top-level keys:
- `decisions`: one object for every provided page, no missing pages and no extra pages.

Each decision object must contain exactly:
- `page`: integer page number from the provided list.
- `has_structure`: boolean.
- `confidence`: `high`, `medium`, or `low`.
- `reason`: short visual reason.

Canonical shape:
{
  "decisions": [
    {"page": 1, "has_structure": true, "confidence": "high|medium|low", "reason": "short visual reason"},
    {"page": 2, "has_structure": false, "confidence": "high|medium|low", "reason": "short visual reason"}
  ]
}
