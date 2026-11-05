# peptide_notation

Atomic skill: cyclopeptide line notation -> built, mass-checked SMILES.

## Verification ladder

1. RDKit per-unit gate (parse + amine-first skeleton), failures fed back.
2. Deterministic molecule-graph assembly (RWMol bonding).
3. Declared exact-mass gate when provided; delta fed back with likely causes.
Every attempt is recorded; unresolved notations end as named gaps.

## Model prompt (verbatim contract below the marker)

<<<PROMPT
Read this cyclopeptide line notation from a patent and list every amino-acid unit in order, with its chemical identity. Respond with one JSON object:

{"units": [{"code": "<token as written>", "name": "<full chemical name>", "fragment_smiles": "<free amino acid: backbone amine N first, alpha carbon with side chain, carboxyl as C(=O)O>", "n_methyl": <bool>, "ring_nitrogen": <bool>}], "ring_size": <units in the head-to-tail macrocycle>, "tail_units": <linear units after the ring>, "terminal": "<C-terminal group like NH2 / OH, or null>", "caps": [{"code": "<cap token>", "fragment_smiles": "<acyl group ending in C(=O), e.g. chloroacetyl ClCC(=O)>"}], "bridges": [{"from_unit": <unit index>, "to_unit": <unit index>, "kind": "<thioether|lactam|disulfide>", "note": "<side-chain atoms>"}]}

Rules: units[0..ring_size-1] form the macrocycle in bond order; a cap inside cyclo(...) acylates the first unit's amine (that amine is N,N-diacyl); beta-methyl amino acids are N[C@@H]([C@H](C)side)C(=O)O; Sarcosine is N(C)CC(=O)O. JSON only.
PROMPT>>>