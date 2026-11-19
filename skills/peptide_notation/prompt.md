# peptide_notation

Convert cyclopeptide notation into a molecular graph and verify any declared exact mass.

## Verification ladder

1. Validate each residue fragment with RDKit.
2. Assemble the molecular graph from the validated fragments.
3. Compare the calculated exact mass with the declared value, when available.

## Model prompt (verbatim contract below the marker)

<<<PROMPT
Read the supplied cyclopeptide notation and return one JSON object.

{"units": [{"code": "<source token>", "name": "<chemical name>", "fragment_smiles": "<free residue with backbone amine first>", "n_methyl": <bool>, "ring_nitrogen": <bool>}], "ring_size": <macrocycle unit count>, "tail_units": <linear unit count>, "terminal": "<C-terminal group or null>", "caps": [{"code": "<source cap token>", "fragment_smiles": "<acyl fragment ending at the carbonyl carbon>"}], "bridges": [{"from_unit": <unit index>, "to_unit": <unit index>, "kind": "<thioether|lactam|disulfide>", "note": "<participating atoms>"}]}

- List residues in bond order, starting with the macrocycle and followed by any linear tail.
- Split composite notation at residue boundaries and preserve covalent modifications on the residue they belong to.
- Preserve the stereochemistry and substituents supported by the source. Represent the uncondensed backbone carboxyl as C(=O)O.
- List terminal acyl groups in `caps`. A cap inside the macrocycle notation attaches to the first residue's backbone amine.
- Record supported bridges with their participating atoms. Use a declared exact mass to check the interpretation against the source.
PROMPT>>>
