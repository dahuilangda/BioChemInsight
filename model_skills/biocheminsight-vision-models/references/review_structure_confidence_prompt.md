Task
Verify that a recognized chemical structure matches its source image. The model
decoded this structure from the left image; the right image is a clean render of
the decoded SMILES. Confirm they represent the same molecule.

Structure JSON:
{{STRUCTURE_JSON}}

Review rules
1. `consistent=true` when the rendered molecule (right) shows the same atoms,
   bonds, ring systems, and stereochemistry as the source drawing (left).
2. The `STRUCTURE_TYPE` in the JSON is provided by a dedicated classifier.
   Only override it when the source image (left panel) clearly shows a
   different type.
3. Focus on chemistry, not rendering style (bond lengths, angles, fonts).
4. A wrong bond order, a missing or extra atom, a missing ring, or a wrong
   stereo configuration is a rejection.
5. Compare the two panels atom-by-atom. Any extra or missing atom in any
   chain is a rejection.
6. If the source image is too blurry or ambiguous to compare, output
   `consistent=false` with `confidence=low`.
7. `issues` must list every specific mismatch found. Empty only when
   consistent.

Output contract
```json
{
  "consistent": true,
  "confidence": "high",
  "issues": [],
  "evidence": "short summary of the comparison"
}
```