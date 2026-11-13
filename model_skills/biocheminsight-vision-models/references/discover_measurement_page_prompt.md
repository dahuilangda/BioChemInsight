Inspect the whole page for compound-level results. Include biological/biochemical activity,
ADME, pharmacokinetics, toxicity, physicochemical properties and computational predictions.
Exclude analytical identity-confirmation records and spectral/elemental characterization: they establish what a substance is, rather than its biological, disposition, safety or drug-property outcome.
A result may be in a table, a structure caption, a panel or a prose record. Protocols alone
and unassigned general statements are not result records. Do not require a particular
endpoint vocabulary or a familiar table layout. Use row/subject ownership, headers, units,
captions and footnotes. Distinguish experimental from predicted/calculated provenance;
computational provenance is a classification, never a reason to discard a result.

Return only JSON:
{"has_result_records": true, "confidence": "high|medium|low",
 "endpoints": [{"endpoint": "source-supported endpoint name", "target": "visible target or empty",
 "unit": "visible unit or empty", "source_kind": "experimental|predicted|calculated|unknown"}],
 "evidence": "brief visible result evidence", "gap": "missing headers, illegibility or empty"}
Use false and an empty endpoints array when there are no compound-level results. For endpoint, preserve a visible scientific endpoint symbol/abbreviation when the result header supplies one; do not replace it with a broad activity description. Names
must be understandable scientific endpoint names, not arbitrary assay labels or table numbers.
Do not invent units or targets. Keep distinct result columns distinct.
