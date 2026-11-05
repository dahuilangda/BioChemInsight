# peptide_notation

Read a cyclopeptide line notation from a patent and emit every amino-acid
unit in order with a validated chemical definition.

## Contract

Input: free text (any patent notation style, OCR noise tolerated), optional
context (SITE notes, table row), optional declared exact mass.

Output JSON:

    {"units": [{"code", "name", "fragment_smiles", "n_methyl",
                "ring_nitrogen"}],
     "ring_size": int,
     "tail_units": int,
     "terminal": "NH2" | "OH" | null,
     "caps": [{"code", "fragment_smiles"}],
     "bridges": [{"from_unit", "to_unit", "kind", "note"}]}

fragment_smiles: free amino acid — backbone amine N first, alpha carbon
with side chain, carboxyl as C(=O)O. N-methyl amine written N(C).
Ring-N acids start at the ring nitrogen. beta-methyl acids are
N[C@@H]([C@H](C)side)C(=O)O. Sarcosine is N(C)CC(=O)O.

## Verification ladder

1. RDKit per-unit gate (parse + amine-first skeleton), failures fed back.
2. Deterministic assembly to full SMILES.
3. Declared exact-mass gate when provided; delta fed back with likely causes.
Every attempt is recorded; unresolved notations end as named gaps.
