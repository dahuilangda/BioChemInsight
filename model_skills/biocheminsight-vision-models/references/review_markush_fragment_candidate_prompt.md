Task
Review whether the red-boxed Markush fragment/substituent candidate can serve
as evidence for a specific Markush structure relationship.
Verify evidence only. Do not generate SMILES, do not complete structures, and
do not infer invisible attachment sites.

The image is a patent page with a red box highlighting the fragment.  It may
also include the previous page merged on the left for context.  Search the
area around the red box; including the left margin, row labels, table cells,
and the merged previous page; for a compound ID or row number.

MolNexTR candidate JSON:
{{FRAGMENT_CANDIDATE_JSON}}

Candidate relationship JSON:
{{RELATIONSHIP_JSON}}

Page/table context JSON:
{{PAGE_CONTEXT_JSON}}

Evidence layers
- `text_assignment`: visible textual variable assignment. This is not
  a structural fragment and does not provide attachment or pose evidence.
- `visual_attachment`: visible star, open bond, wavy bond, R/X/Y/Ar/Het label,
  or connection direction. This only proves that the image has an attachment cue.
- `molnextr_graph`: dummy atom, star atom, R-group atom, and bonds inside the
  MolNexTR MOLBLOCK/graph. Only this layer proves that MolNexTR recognized an
  attachment atom.

Review rules
1. Output `fragment` or `substituent` only when the red box contains a visible
   chemical structure, fragment drawing, or substituent drawing with an
   attachment cue.
2. If the red box contains only text, names, table lines, blank area, or noise,
   output `text_cell`, `noise`, or `unknown`.
3. If candidate relationship JSON provides `compound_id`, `fragment_refs`, and
   `variable_positions`, first verify that the red box corresponds to that
   relationship. Output that `compound_id` and variable only when supported by
   a visible row ID, same-scope continuation, or page context.
4. `compound_id` may come only from a visible same-row ID, the current-row ID
   already bound in candidate relationship JSON, or same-scope continuation
   context. Continuation context may explain column meaning but must not invent
   a current-row ID.
5. `variable_position` may come only from a same-table column header, nearby
   red-box label, candidate relationship `variable_positions`, or same-scope
   inherited variable header.
6. `variable_position` must always be a JSON string. If unknown, output the
   empty string `""`. Never output `null`, an array, or an object.
7. `has_attachment_evidence=true` requires a visible star, open bond, wavy bond,
   R-group connector, or explicit attachment mark.
8. `molnextr_has_attachment_atom=true` may come only from an explicit dummy,
   star, or R-group atom in the MolNexTR candidate JSON/MOLBLOCK/graph.
9. Visual attachment evidence cannot substitute for a MolNexTR attachment atom.
   If the MolNexTR graph lacks it, output false.
10. `attachment_site_consistent=true` means the MolNexTR attachment atom aligns
    with the visible attachment location. If either side is missing, output false.
11. The MolNexTR MOLBLOCK/graph must match the red-boxed visible structure.
    When a two-panel image is available, compare the rendered SMILES (right
    panel) against the source drawing (left panel) atom-by-atom.  A wrong bond
    order, a missing or extra atom, a missing ring, or a wrong stereo
    configuration is a rejection; even if the overall structure looks similar.
    SMILES text alone is not pose or attachment-site evidence.
12. High confidence requires all of: relationship candidate match, non-empty
    string variable position, visible attachment, MolNexTR graph attachment atom,
    and consistent attachment site.
13. If row ID, variable position, structure consistency, or attachment evidence
    is unclear, use low/medium confidence. Do not guess.

Output contract
Return only one JSON object:
```json
{
  "visual_role": "fragment|substituent|scaffold|text_cell|noise|unknown",
  "compound_id": "row/example id or None",
  "compound_id_source": "row_label|local_label|none|unknown",
  "variable_position": "R4",
  "molnextr_consistent": true,
  "has_attachment_evidence": true,
  "molnextr_has_attachment_atom": true,
  "attachment_site_consistent": true,
  "confidence": "high|medium|low",
  "evidence": "short visual evidence summary"
}
```
