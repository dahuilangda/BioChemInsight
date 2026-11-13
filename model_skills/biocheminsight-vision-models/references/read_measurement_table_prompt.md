Recover compound-level result records from this page image. Include biological activity,
ADME, toxicity and drug-property results, experimental or predicted. Use the entire
caption, grouped headers and footnotes to assign endpoint, target, unit and provenance.
Exclude synthesis/analytical identity characterization and unassigned general discussion.

Return compact JSON:
{"tables": [{"caption": "visible source caption or panel heading", "notes": ["applicable printed footnote/category definition, preserving source page when provided"], "columns": [
 {"endpoint": "printed scientific endpoint symbol/name", "target": "visible target or empty",
  "unit": "printed unit or empty", "source_kind": "experimental|predicted|calculated|unknown",
  "header_evidence": "verbatim header and relevant group heading"}],
 "rows": [{"compound_id": "complete identifier from the same record", "values": ["verbatim cell value without unit"],
 "row_evidence": "visible identifier/record anchor"}]}], "gaps": ["illegible or ambiguous record"]}
Each values array must have exactly one entry per column in the same order. Use null
for an illegible/missing cell; a blank is not zero. Preserve uncertainty, inequality and
categorical text. Keep distinct tables/panels separate. Do not fill a row from a neighbor,
from a known drug's typical properties, or from a familiar series. Return an empty tables
array when no compound-level results are extractable. Structure captions with associated
results may be represented as records. Do not invent identifiers for unlabeled structures.

Repeated side-by-side blocks with identical endpoint headers are continuations, not different assays. Recover every subject once across all blocks. Preserve categorical symbols and untested labels exactly; do not invent numeric point estimates from categories.
