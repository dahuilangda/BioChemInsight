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

Composite tokens decompose into units; work these canonical examples:

- CLAc (chloroacetyl cap) is one acyl unit, NOT a residue: {"code": "CLAc", "fragment_smiles": "NCCl"} — as a cap inside cyclo(...) it acylates the next unit's amine AND its CH2Cl bridges the Cys side chain later (record that in bridges).
- Aea ([2-(2-aminoethoxy)acetic acid] linker) is N(CCO)CC(=O)O — one unit.
- K(N-Biotin) is lysine with biotinylated side chain: one unit {"code": "K(N-Biotin)", "fragment_smiles": "N[C@@H](CCCCN)C(=O)O", "n_methyl": false}; note the biotin bulk in "note" (mass of biotin excluded unless the declared mass demands it).
- SbMeW7OMe ... C is TWO units forming a side-chain thioether bridge: SbMeW7OMe = beta-methyl 7-methoxy tryptophan N[C@@H]([C@H](C)c1c[nH]2c(OC)cccc12)C(=O)O, C = cysteine N[C@@H](CS)C(=O)O, with bridges [{"from_unit": <W index>, "to_unit": <C index>, "kind": "thioether", "note": "side-chain CH2-S bond"}].
- The thioether bridge loses 2 H total from the assembled molecule — when a declared exact mass sits ~2 Da below the naive build, check bridges first.
PROMPT>>>