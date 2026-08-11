Task
Review whether an automatically assembled Markush compound matches the visible
red-boxed drawings it was assembled from. The image contains three panels in a
fixed order:

- Left panel: the red-boxed Markush scaffold drawing (as cropped from the
  patent page, may contain R-group labels like R1/R2 and attachment marks).
- Middle panel: the red-boxed substituent/fragment drawing (the fragment that
  was attached, may contain a wavy bond, star, or R-group attachment mark).
- Right panel: the assembled molecule rendered from the assembled MOLBLOCK
  (the final product of connecting the substituent onto the scaffold at the
  labeled variable position).

Verify evidence only. Do not generate SMILES, do not propose alternative
structures, and do not guess what is not visible.

Assembly JSON:
{{ASSEMBLY_JSON}}

Scaffold candidate JSON:
{{SCAFFOLD_JSON}}

Fragment candidate JSON:
{{FRAGMENTS_JSON}}

Review layers
- `scaffold_match`: the scaffold region of the assembled molecule (right panel)
  must show the same atoms, rings, and bond orders as the visible scaffold
  drawing (left panel). Aromatic rings must stay aromatic; ring sizes and
  heteroatoms must match; no atom or bond may appear in the assembly that is
  not present in the scaffold drawing.
- `fragment_match`: the substituent region of the assembled molecule must show
  the same atoms and connectivity as the visible fragment drawing (middle
  panel), ignoring the attachment mark itself.
- `attachment_site_match`: the substituent must be attached at the labeled
  variable position (R1/R2/etc.) or at the visible attachment mark in the
  scaffold drawing; not at a different position.

Review rules
1. `consistent=true` requires all three layers to match. Any visible mismatch
   in scaffold region, fragment region, or attachment position is a rejection.
2. A wrong bond order, a missing atom, an extra atom, or a substituent
   attached at the wrong position are all rejections, even if the rest of
   the molecule looks similar.
3. **Critical: fragment atom count check**. Compare the fragment drawing
   (middle panel) atom-by-atom with the substituent region of the assembled
   molecule (right panel). Count the atoms from the attachment point to each
   ring or heteroatom in the fragment drawing, then verify the same count
   exists in the assembly. Specific checks:
   a. Extra linker atom: if the assembly has more atoms in any chain from
      the attachment point to a ring than the fragment drawing, reject.
   b. Missing linker atom: if the assembly has fewer atoms in any chain from
      the attachment point to a ring than the fragment drawing, reject.
   c. Wrong attachment heteroatom: if the wavy bond in the fragment drawing
      connects to a different heteroatom than the assembly, reject.
   d. Wrong ring size: ring size mismatch between fragment drawing and
      assembly is a rejection.
4. The right panel is a computer rendering; ignore rendering style differences
   (bond lengths, angles, font) and only compare chemistry: atom identity,
   ring membership, bond order, and attachment position.
5. If the scaffold or fragment drawing is too blurry, cut off, or ambiguous to
   compare, output `confidence=low` and `consistent=false`; do not guess.
6. If the assembled molecule contains an R-group label (R1/R2/...) still
   present in the right panel, that is a rejection: the assembly must be a
   fully defined molecule with no remaining variable positions.
7. `issues` must list every specific mismatch found. Empty only when
   consistent.
8. `confidence` reflects how clearly the visible drawings support the
   comparison: high = all three panels clearly readable; medium = some
   ambiguity but enough to decide; low = cannot reliably compare.

Output contract
Return only one JSON object:
```json
{
  "consistent": true,
  "confidence": "high",
  "issues": [],
  "evidence": "short visual evidence summary"
}
```
- `consistent`: boolean; the assembled molecule matches the visible scaffold
  and fragment drawings and is attached at the labeled position.
- `confidence`: "high" | "medium" | "low".
- `issues`: list of specific mismatches (empty when consistent).
- `evidence`: short summary of what was compared and what was found.
