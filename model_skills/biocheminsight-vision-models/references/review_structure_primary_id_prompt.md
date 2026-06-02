{{BASE_STRUCTURE_TO_ID_PROMPT}}

Primary Compound ID review
The current visual result is:
{{INITIAL_RESULT_JSON}}

Re-read the red-boxed structure and decide the primary target Compound ID that should be used for SAR/data merging.

Review rules
1) First identify the full visual group containing the red box: table row, structure cell, reaction product block, peak/enantiomer pair, caption block, or heading block.
2) If the boxed structure is in a table row or product/result block with a row-leading or governing `Example`, `Ex.`, `No.`, `Compound`, `Formula`, or equivalent target ID, that primary row/heading ID should be returned for the boxed target compound.
3) Treat local labels such as peak/enantiomer/stereoisomer/form/salt/component labels, short one-letter labels, and short numeric labels as secondary attributes when a primary row/heading target ID visibly governs the same boxed structure or product block.
4) Return the secondary local label only when it is the only visible identifier that actually identifies the boxed molecule, or when the table/header proves that local-label column is the primary compound-ID column.
5) Do not strip or shorten any returned identifier. If the correct identifier is local, keep the full local label. If the correct identifier is row/heading-based, keep the full row/heading label.
6) Do not use a nearby ID from another row, another product, another peak/enantiomer, a paragraph number, page number, scheme/table/figure number, reaction step, condition, yield, analytical value, or reagent.
7) If no primary target ID can be visually confirmed for the boxed structure, return `COMPOUND_ID` as `"None"` with `ID_SOURCE` set to `none`.
8) In a continued table, inherited header context can identify the primary ID column only while the same table scope continues.
   It cannot supply a missing row value, and it must stop at a new table/core/scaffold, changed layout, or unrelated record block.

Return JSON only with the same five required keys and allowed enum values from the base contract.
