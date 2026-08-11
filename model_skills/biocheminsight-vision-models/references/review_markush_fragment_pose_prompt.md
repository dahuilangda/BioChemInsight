Task
Review whether the candidate Markush relationship has enough visual evidence to enter the deterministic assembly harness.
Verify evidence consistency only. Do not generate final SMILES, and do not complete the scaffold or fragment.

The image you see is a two-panel side-by-side comparison:
- Left panel: the source fragment drawing cropped from the patent page.
- Right panel: a computer rendering of the decoded structure.

The decoded structure's SMILES is NOT provided in the JSON below.  You MUST
judge correctness purely by comparing the two panels atom-by-atom.  If the
two panels do not show the same atoms, connectivity, ring sizes, and bond
orders, the decoded structure is wrong regardless of anything else in the
JSON.

Input relationship JSON:
{{RELATIONSHIP_JSON}}

MolNexTR fragment evidence JSON:
{{FRAGMENT_CANDIDATE_JSON}}

MolNexTR scaffold evidence JSON:
{{SCAFFOLD_CANDIDATE_JSON}}

Page/table context summary:
{{PAGE_CONTEXT_JSON}}

Evidence layers
- `text_assignment`: compound_id and variable-assignment evidence; does not provide pose.
- `visual_attachment`: visible connection direction, star, open bond, wavy bond, or variable label in the red box.
- `molnextr_graph`: explicit attachment atom and bond in the MolNexTR MOLBLOCK/graph; this is required for assembly.

Review rules
1) The relationship must have a scaffold candidate, fragment candidate, single variable position, and record ID. If any is missing, it cannot be ready.
2) The variable position must be consistent. If the relationship says R4, the scaffold/page context and fragment/table column must both support R4.
3) `has_attachment_evidence=true` requires a visible attachment cue in the red box; pure text assignment does not count.
4) `molnextr_consistent=true` requires the MolNexTR MOLBLOCK/graph to match the visible red-box structure. When a two-panel image is available, you MUST compare the rendered SMILES (right panel) against the source drawing (left panel) atom-by-atom. Specifically check:
   a. **Extra atoms**: count the atoms along each chain from the attachment point to each ring or heteroatom. If the right panel has more atoms in any chain than the left panel, reject.
   b. **Missing atoms**: if the left panel shows a linker atom that is absent in the right panel, reject.
   c. **Wrong attachment atom**: if the wavy bond in the left panel connects to a different heteroatom than the right panel shows, reject.
   d. **Ring size**: ring size mismatch between the two panels is a rejection.
   e. **Wrong bond order**: single vs double bond mismatch is a rejection.
   f. Only set `molnextr_consistent=true` if ALL of the above checks pass.
5) If the fragment candidate review shows that MolNexTR has no attachment atom, it cannot be ready, even when a wavy/open bond is visible.
6) Use `pose_consistency="consistent"` only when scaffold variable site, fragment variable column/connection direction, and MolNexTR graph evidence do not conflict.
7) If unclear, severely cropped, not the fragment for this relationship, scope-mismatched, or evidence-conflicted, output `uncertain` or `needs_context`.
8) `assembly_status="ready"` only means the later deterministic harness may attempt assembly. The final structure must still pass MolBlock dummy, connectivity, and pose checks.
9) `evidence` must describe the atom-by-atom comparison result, including the atom count from the attachment point to each ring in both panels and whether they match.

Output contract
Return only one JSON object:
```json
{
  "visual_role": "scaffold|fragment|substituent|table_cell|noise|unknown",
  "molnextr_consistent": true,
  "has_attachment_evidence": true,
  "variable_position_visible": true,
  "pose_consistency": "consistent|inconsistent|not_applicable|unknown",
  "assembly_status": "ready|needs_context|not_applicable|uncertain",
  "confidence": "high|medium|low",
  "evidence": "short visual evidence summary"
}
```
