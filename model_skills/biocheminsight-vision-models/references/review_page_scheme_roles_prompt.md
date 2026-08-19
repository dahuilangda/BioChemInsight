Task
Read one patent page that contains a reaction scheme (or several) with several numbered,
color-coded boxes. Decide every boxed structure's role in the SAME reading of the page,
and name the single box that carries the page's record.

Box legend
Each structure on the page is framed by a colored rectangle; the number printed at the
box corner identifies it in the JSON you return. Use the printed numbers exactly.

Procedure
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
      "role": "final_product|product|intermediate|reactant|reagent_or_condition|table_entry|unknown",
      "evidence": "short positional evidence"
    }
  },
  "record_box": "__BOX_NUMBER__OR_NONE__",
  "record_id": "visible id string or None",
  "record_id_source": "heading|local_label|row_label|none",
  "confidence": "high|medium|low"
}
```

Rules
- Judge all boxes in one consistent reading: never call two different boxes the final
  product of the same flow, and never leave a flow's after-arrow box unassigned.
- record_id must be a record identifier (embodiment, example, or compound label), never a
  page element: patent/application numbers, figure or table numbers, paragraph brackets,
  page headers or footers are not record IDs; if only those are visible, record_id="None".
- record_id comes only from visible page text; do not invent or complete IDs.
- If the page has no scheme and no single boxed target, set record_box to "None".
