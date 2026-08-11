Task
Review whether the red box fully contains exactly one structure image that can
be sent to MolNexTR.
Judge box/crop quality only. Do not decide whether it is a final complete
compound, do not repair the structure, and do not suggest a new box.

Candidate JSON:
{{CANDIDATE_JSON}}

Rules
1. `complete_single_structure`: the red box fully contains one structure or one
   recognizable fragment, with no boundary clipping. A fragment with a wavy
   bond, open bond, star, or R-group attachment can be a complete box, but it is
   still not a final complete compound.
2. `partial_or_truncated`: the box contains only part of a structure, or any
   bond, atom, ring, label, or attachment cue is visibly clipped.
3. `multiple_structures`: the box contains multiple independent structure
   drawings.
4. `non_structure`: mostly text, table lines, headers/footers, blank area, or
   noise.
5. `uncertain`: use when box completeness cannot be judged reliably.
6. Judge only from the red box and immediately adjacent boundary content. Do
   not use context to guess a complete structure.
7. Do not interpret `complete_single_structure` as `complete_compound`; this
   result only allows later MolNexTR recognition to continue.

Output contract
Return only JSON:
```json
{
  "box_status": "complete_single_structure|partial_or_truncated|multiple_structures|non_structure|uncertain",
  "is_single_structure": true,
  "is_complete_box": true,
  "confidence": "high|medium|low",
  "evidence": "short visual evidence summary"
}
```
