Task
Review the PDF page thumbnail contact sheet and identify which pages show visible chemical structure drawings.

Inclusion criteria
- Visible 2D molecular line-bond drawings
- Visible structure drawings inside reaction schemes
- Visible Markush/core scaffold/R-group structure drawings
- Structure tables with cells that actually contain structure drawings

Exclusion criteria
- Pure text, claims, definitions, references, indexes
- Assay tables, name tables, or ID tables with no structure drawing
- Text labels without a visible structure drawing
- Charts, screenshots, stamps, non-chemical graphics

Identify each thumbnail by the PAGE label in its dark top border. Printed page numbers inside the thumbnail are document text. Report the border label. Inspect only pixels inside that thumbnail for its decision.

Inspect every thumbnail from top to bottom. A small recognizable drawing qualifies anywhere on the page.

Rules
1) Use visual evidence, not OCR keywords.
2) Set `has_structure=true` only when a structure drawing is actually visible.
3) Be conservative for text pages and uncertain pages.
4) Output exactly one decision for each provided page.

Page list
{{PAGES_JSON}}

Output contract
Return only a JSON object with a `decisions` array. Each entry must have an integer `page`
matching the thumbnail label, a boolean `has_structure`, a `confidence` string selected
from `high`, `medium`, or `low`, and a brief `reason` describing visible evidence.
