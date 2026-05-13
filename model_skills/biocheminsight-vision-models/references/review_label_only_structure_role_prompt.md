{{BASE_STRUCTURE_TO_ID_PROMPT}}

Second-pass role review
- The first pass found a local label for the red-boxed molecule, but did not cite product/result text.
- Re-check the full current-page image carefully. The local label identifies the molecule only; it does not prove final/product role.
- Inspect nearby prose/captions, reaction arrows, wrapped multi-line scheme continuation, and whether the molecule is used in a later step.
- If the boxed molecule is described or positioned as an intermediate, precursor, reactant, or material for a later step, set `VISUAL_ROLE` accordingly.
- Keep `COMPOUND_ID` as the visible local label when it belongs to the boxed molecule.
- Candidate local label from first pass: {{COMPOUND_ID}}

Return JSON only with the same five keys.
