Task
Decide whether two depictions of the same chemical structure agree atom by
atom, and justify the decision with a counted comparison.

Panel A (left, "A: SOURCE") is a structure cropped from a patent image.
Panel B (right, "B: PARSED (same layout)") is a rendering of the structure
parsed from panel A. By construction panel B is drawn with the same 2D layout
and orientation as panel A: every atom occupies the same relative position in
both panels. Layout correspondence is guaranteed; your task is to find any
atom-level disagreement between the two depictions.

Structure JSON:
{{STRUCTURE_JSON}}

Procedure - execute every step in order and carry the results into evidence

1. Element census. For every element in N, O, S, P, F, Cl, Br, I, count the
   atom labels visible in panel A and the atom labels visible in panel B.
   Record both counts for each element, plus the ring count for each panel.
2. Arbitration. If any recorded count differs between the panels, re-examine
   both panels for that element before concluding: locate each disputed label
   by its position among the neighboring vertices and confirm which count the
   drawing supports. A difference that survives re-examination is a chemical
   difference.
3. Placement. When the census agrees, confirm that each heteroatom label in
   panel A is answered by the same element label at the same relative
   position in panel B, and that each terminal substituent is attached at
   the same vertex. An unlabeled vertex denotes carbon.
4. Connectivity. Confirm that fused-ring edges and chain branching agree.

Verdict rules
- consistent=true when census, placement and connectivity all agree.
- consistent=false when, after arbitration, an element or ring count still
  differs; or a position carries one element in panel A and a different
  element or an unlabeled vertex in panel B; or a terminal substituent
  attaches at a different vertex; or a ring or bond is missing or extra.
- Every rejection must name the element, its position, and which panel is
  missing or misplacing it. A difference that cannot be located precisely is
  not a difference.
- confidence is high when the census was completed for every element,
  medium or low otherwise.

Rendering equivalence - never a difference: stroke weight, fonts, label
colors, Kekule versus aromatic-circle notation, rotation, scale, shift,
resolution, antialiasing, hydrogens shown or omitted on N, O and S, isotope
NOTATION only (D and 2H are the same isotope, as are T and 3H - but an
isotope label present in one panel and absent in the other, or facing a
different element, is a real difference), group abbreviations expanded in
panel B (Boc, Cbz, Ts, TBDMS and similar protective or common groups drawn
out atom by atom are the same group), stereo wedges drawn in panel A but
absent in panel B (the parsed depiction carries no wedge geometry - report
stereo differences only when both panels show wedges and they disagree),
label text order that only reflects the attachment side of a bond, and text
or marks near the structure that are not part of the molecule.

The Structure JSON is metadata about the parse; never treat it as evidence
of what the panels depict.

Output contract
```json
{
  "consistent": true,
  "confidence": "high",
  "issues": [],
  "evidence": "element-by-element census counts for A and B, ring counts, and the placement and connectivity conclusion"
}
```
