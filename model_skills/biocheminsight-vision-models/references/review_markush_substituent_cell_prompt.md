Task
Review whether the red box is a substituent cell in a Markush/R-group table or inline definition.
Extract only visible assignment evidence. Do not generate SMILES, and do not infer attachment sites.

Page/table context summary:
{{PAGE_CONTEXT_JSON}}

Candidate evidence JSON:
{{CANDIDATE_JSON}}

Review rules
1) Use only the red-box content, same-row labels, and immediately adjacent headers.
2) Use `visual_role="substituent_cell"` for a variable-column cell or inline definition cell.
3) `compound_id` may come only from a visible same-row Ex./No./Compound/Example/record ID. If not visible, write "None".
4) `variable_position` may come only from the column header or a visible variable name inside the cell, such as R4, X, or Y. If not visible, write an empty string.
5) `substituent_text` must contain only visible text or structure labels inside the red box. Do not expand or normalize long names.
6) If the red box contains a structure/fragment drawing, set `has_visual_structure=true`; pure-text substituents are false.
7) `has_attachment_evidence=true` requires a visible star, open bond, wavy bond, R-group connector, or explicit attachment prefix.
8) Text assignments such as `R1 = Cl`, `X = OH`, or `A = CH` are not attachment/pose evidence. If no visible connection mark exists, set `has_attachment_evidence=false`.
9) If unclear, not a cell, uncertain across headers, or discontinuous in scope, output low/medium confidence. Do not guess.

Output contract
Return only one JSON object:
```json
{
  "visual_role": "substituent_cell|table_header|scaffold|noise|unknown",
  "compound_id": "row/example id or None",
  "compound_id_source": "row_label|local_label|none|unknown",
  "variable_position": "R4",
  "substituent_text": "visible text only",
  "has_visual_structure": false,
  "has_attachment_evidence": true,
  "confidence": "high|medium|low",
  "evidence": "short visual evidence summary"
}
```
