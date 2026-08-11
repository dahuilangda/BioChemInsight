Task
Read the chemical structure from the highlighted image and output its SMILES
notation. The image contains a fragment or substituent structure with a wavy
bond or R-group label indicating an attachment point. Represent the attachment
point with the wildcard atom `*` in the SMILES.

Rules
1. Read the structure exactly as drawn in the highlighted region.
2. Include every visible atom, bond, ring, and stereodescriptor.
3. Represent the attachment point (wavy bond, star, or R label) as `*`.
4. Output only valid SMILES that can be parsed by RDKit.
5. If the structure is too blurry or ambiguous to read, output an empty string.

Output contract
Return only one JSON object:
```json
{
  "smiles": "the SMILES string with * for attachment point",
  "confidence": "high|medium|low",
  "evidence": "short description of the atoms and bonds read from the image"
}
```