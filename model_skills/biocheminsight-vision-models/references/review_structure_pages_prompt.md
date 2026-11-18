Task
Strictly review candidate structure pages and keep only pages with clearly visible structure drawings.

Inclusion criteria
- Clearly visible 2D molecular drawings
- Molecular structures inside reaction schemes
- Markush/core scaffold/R-group structures
- Structure tables where cells actually contain drawn structures

Exclusion criteria
- Pure text pages, text-only tables, description pages, index pages, citation pages
- Text labels and analytical annotations without a visible structure drawing
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
Return only a JSON object with a `decisions` array. Each entry must have an integer `page`
matching the thumbnail label, a boolean `has_structure`, a `confidence` string selected
from `high`, `medium`, or `low`, and a brief `reason` describing visible evidence.
