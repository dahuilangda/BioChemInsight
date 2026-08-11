Task
Plan Markush scaffold, variable-position, fragment/substituent, and cross-page
continuation relationships for candidate structure pages.
Plan evidence relationships only. Do not generate final SMILES, do not invent
invisible structures, and do not write assay values.

Input
Page context JSON:
{{PAGE_CONTEXTS_JSON}}

`text_assignments` are deterministic variable-assignment candidates extracted
from OCR/Markdown.
They can help identify variable headers, scope, and compound-variable
relationships, but they cannot substitute for a visible fragment drawing or a
MolNexTR graph.

Structure candidate JSON:
{{STRUCTURE_CANDIDATES_JSON}}

Evidence layers
- `text_assignment`: inline or table textual variable assignment. It may
  support compound ID and variable assignment, but it does
  not provide attachment, pose, or MolNexTR structural-fragment evidence.
- `visual_attachment`: visible star, open bond, wavy bond, R/X/Y/Ar/Het label,
  or connection direction. It may support pose reasoning, but it cannot replace
  an attachment atom in the MolNexTR graph.
- `molnextr_graph`: explicit dummy atom, star atom, R-group atom, and ordinary
  bonds in the MolNexTR MOLBLOCK/graph. Only this layer authorizes later
  structure assembly.

Planning rules
1. Output one page decision for every input page.
2. Page role must be one of:
   `markush_scaffold`, `fragment_table`, `continuation`, `complete_compound`, `non_structure`, `unknown`.
3. A continuation table may inherit context from non-adjacent prior pages, but
   only when table/scope, scaffold, variable headers, and record range remain
   continuous. Stop inheriting at a new table, new scaffold, new header, or
   scope reset.
4. `relationships` may link only evidence visible in the input or already
   present in candidate summaries:
   - `scaffold_ref`: scaffold candidate, page scaffold description, or null.
   - `fragment_refs`: fragment candidate, table row, text-assignment candidate,
     or an empty array.
   - `variable_positions`: visible or same-scope inherited variable names only.
   - `compound_id`: current-record visible ID, or current-row record ID in a
     same-scope continuation table. If uncertain, write "None".
5. Assay tables and assay values belong only to assay extraction context and
   must not enter structure relationships.
6. If a relationship already has `compound_id`, `scaffold_ref`, and exactly one
   `variable_positions` item, and the `source_pages` have fragment/substituent
   structure candidates, you must cite one or more candidate refs in
   `fragment_refs` as candidates for later visual verification. Do not leave
   `fragment_refs` empty merely because the connection is not yet explicitly
   confirmed. The later visual relationship review will reject wrong refs.
   If a prior page has a Markush scaffold with a visible variable label, and
   following contiguous table pages contain row-labeled fragment drawings with
   visible attachment cues, create one candidate relationship per row-labeled
   fragment using the inherited scaffold ref, the inherited variable label, the
   visible row ID, and that fragment candidate ref.
7. `fragment_refs` may be empty only when the input candidates truly contain no
   usable fragment/substituent candidate, or the relationship is only a pure
   text assignment. In that case, the reason must state that no usable candidate
   exists.
   A missing/null `active_scope_id` alone is not a reset when page layout,
   table headers, row numbering, and fragment column structure visibly continue
   from the prior Markush table. In that case, inherit the nearest compatible
   scaffold and variable header and state the inheritance in the reason.
8. `assembly_status` meanings:
   - `ready`: scaffold, one fragment, one variable position, record ID, and
     consistent pose are available, and any existing fragment review proves a
     `molnextr_graph` attachment atom.
   - `needs_context`: additional inherited or missing context is required.
   - `uncertain`: a relationship may exist but evidence is insufficient.
   - `not_applicable`: not a Markush/fragment assembly relationship.
9. `ready` only means the candidate may enter later visual relationship review
   and deterministic assembly harness. Final structures enter normal output only
   after fragment visual review, MolBlock/dummy, connectivity, and pose checks.
10. If there is only `text_assignment`, do not mark `ready`; `pose_consistency`
    must be `not_applicable` or `unknown`.
11. If fragment review already exists and does not simultaneously satisfy:
   `molnextr_consistent=true`, `has_attachment_evidence=true`, `molnextr_has_attachment_atom=true`, `attachment_site_consistent=true`,
    do not mark `ready`.
12. If fragment review has not run yet, but candidates include scaffold,
    fragment_ref, one variable position, current-row ID, and same-scope table
    relationship, you may output `ready` as a candidate awaiting visual
    relationship verification. The downstream harness will downgrade or block
    relationships that fail rule 11.
13. Do not hard-code decisions from page number, patent identity, compound name,
    or common chemistry fragments.

Output contract
Return only one JSON object. Top level must contain `pages` and `relationships`.
The `pages` array must cover every input page.

Page decision format:
```json
{
  "page": 1,
  "role": "markush_scaffold|fragment_table|continuation|complete_compound|non_structure|unknown",
  "use_prior_markush_context": false,
  "context_source_page": null,
  "confidence": "high|medium|low",
  "reason": "short generic reason"
}
```

Relationship format:
```json
{
  "record_id": "stable local relationship id",
  "compound_id": "Example/Compound/row ID or None",
  "compound_id_source": "row_label|local_label|heading|caption|inherited_context|none|unknown",
  "source_pages": [1],
  "scaffold_ref": "candidate/page reference or null",
  "fragment_refs": ["candidate/page/table-row reference"],
  "variable_positions": ["visible variable label"],
  "assembly_status": "ready|needs_context|not_applicable|uncertain",
  "pose_consistency": "consistent|inconsistent|not_applicable|unknown",
  "confidence": "high|medium|low",
  "reason": "short evidence summary"
}
```
