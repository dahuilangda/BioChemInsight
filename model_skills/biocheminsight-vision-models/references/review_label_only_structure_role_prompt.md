{{BASE_STRUCTURE_TO_ID_PROMPT}}

Secondary role review
- The first pass found a local label, but that label only identifies which molecule it is; it does not automatically prove that it is the final product.
- Re-check the whole-page context: reaction arrows, nearby text, captions, cross-row continuation, and whether the molecule is used again later.
- If the molecule is described as an intermediate, precursor, reactant, or starting material for a later step, adjust `VISUAL_ROLE` accordingly.
- `COMPOUND_ID` should still remain the visible label truly attached to the red-boxed molecule. If no reliable label is found, return `"None"`.
- First-pass candidate label: {{COMPOUND_ID}}

Return only JSON with the same contract as the base prompt.
