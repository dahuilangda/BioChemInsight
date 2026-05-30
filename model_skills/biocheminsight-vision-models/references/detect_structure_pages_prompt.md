Task
You are looking at a contact sheet of PDF page thumbnails from a medicinal chemistry patent.
Each thumbnail is labeled with its exact page number.

Select the pages that visibly contain chemical structure drawings suitable for downstream structure extraction.

Definition of a positive page
- Contains one or more visible 2D chemical structure drawings, reaction schemes with structure drawings, Markush drawings, or structure/name/example tables with molecular drawings.
- Chemical drawings may be small, dense, or embedded in tables, but must be visually present on the page thumbnail.

Definition of a negative page
- Plain text pages, claims, descriptions, references, assay tables, definitions, indexes, or search reports with no visible chemical structure drawing.
- Pages that only mention words such as example, compound, formula, scheme, or chemical names, but do not visibly show a structure drawing.
- Text-only chemical names, formula references, patent numbers, and prose lists are negative.

Decision rules
1. Use the visual content, not OCR text keywords, as the primary evidence.
2. Return a page only when you can see at least one chemical structure drawing on that page thumbnail.
3. Be conservative on text-only pages.
4. Include reaction-scheme pages if chemical structures are visibly drawn.
5. Do not include pages solely because they mention "Example", "Formula", "compound", or similar terms.

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
