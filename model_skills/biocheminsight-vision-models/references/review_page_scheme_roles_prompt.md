Task
Read one page containing several numbered, color-coded boxes. Decide every boxed
structure's role in the same reading of the page, based on the graphics that are
actually drawn.

Box legend
Each structure on the page is framed by a colored rectangle; the number printed at the
box corner identifies it in the JSON you return. Use the printed numbers exactly.

Procedure
0) Context gate: if the page shows NO reaction arrow and NO plus sign, do not assign
   reaction roles at all — every box is a standalone depiction (`standalone_compound`,
   or a table/grid entry when applicable).
1) Reconstruct each reaction flow on the page: which boxes are joined by plus signs,
   which arrows point from which boxes to which box, and which flows continue from an
   earlier flow's product.
2) A box joined by "+" to another box is a reactant. A box an arrow points to is that
   step's product; when that product then feeds a later arrow, it is an intermediate.
   The box after the final arrow of the whole page's scheme, with nothing consuming it,
   is the final product.
3) Identify the governing heading or record label on the page (embodiment / example /
   compound number). It names the record the scheme produces.
4) Decide the product box: normally the final-product box. Only when the page clearly
   isolates one boxed target structure (single-structure page, table entry, or boxed
   title product) is that box the record box.

Output contract
Return only JSON:
```json
{
  "boxes": {
    "__BOX_NUMBER__": {
      "role": "final_product|product|intermediate|reactant|reagent_or_condition|table_entry|standalone_compound|unknown",
      "compound_id": "verbatim transcription of the identifier printed with the boxed structure, or \"None\"",
      "annotation_text": "verbatim transcription of any other text printed with the boxed structure, or \"\"",
      "evidence": "short positional evidence"
    }
  },
  "confidence": "high|medium|low"
}
```

Rules
- compound_id per box: identify and transcribe verbatim the identifier that belongs to
  the boxed structure itself — whatever its printed form. An identifier belongs to the
  box when it is visually bound to that drawing (same panel, same row or cell, or
  directly attached to it); it does not belong to the box when it only names the page,
  figure, table, scheme, or a different structure. Use "None" only when no such
  identifier is visible for that box. Report it even when the page has no governing
  heading.
- annotation_text per box: transcribe verbatim any other text printed with the boxed
  structure that is not its identifier (measurements, conditions, notes), preserving the
  original notation. Empty string when none. Never interpret, normalize, or complete
  transcribed text.
- Judge all boxes in one consistent reading: never call two different boxes the final
  product of the same flow, and never leave a flow's after-arrow box unassigned.
  page element: patent/application numbers, figure or table numbers, paragraph brackets,
  page headers or footers are not compound identifiers; if only those are visible,
  compound_id="None".
- If no box can be adjudicated as a record target, omit any record claim and rely on the per-box fields.
