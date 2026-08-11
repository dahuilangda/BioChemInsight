{{BASE_STRUCTURE_TO_ID_PROMPT}}

Secondary ID review
Current result:
{{INITIAL_RESULT_JSON}}

Review rules
1) Re-start from the red-boxed molecule itself. Do not inherit the first-pass conclusion.
2) Confirm that the ID truly belongs to the red-boxed molecule: the same row, same cell/group, a directly attached local label, or a heading that clearly governs that molecule.
3) The ID must be complete. Do not truncate it to a single number, suffix, or piece of a longer label.
4) Reject row numbers, page numbers, table/figure numbers, step numbers, footnotes, analytical values, and condition text.
5) If the first-pass ID is wrong, incomplete, or borrowed from another molecule, correct it if possible. If it cannot be corrected reliably, output `"None"`.
6) Preserve the full printed prefix/suffix/hyphen/parentheses.
7) Continuation-table context may only help identify which column is the ID column. It cannot invent a missing current-row ID value.

Return only JSON with the same contract as the base prompt.
