"""Cyclic-peptide SMILES generator for the large-macrocycle training corpus.

Target chemistry: the heavily modified ~25-30 residue macrocycles drawn in
cyclic-peptide patent figure pages (WO2024163691 SEQ22/53 class) — SMILES of
300-900 characters. The grammar is calibrated against a residue-by-residue
reading of those real molecules:

  - backbone letters weighted toward the real chemistry: arylalanines
    (4-carbamoyl / 4-carboxy / 4-pyridyl / phenoxyacetic), Trp family
    (7-Cl, 7-OMe, 5-CN, N-carboxymethyl, beta-methyl), imino acids
    (Pro, 4-hydroxyproline incl. O-carboxymethyl, 5-hydroxypipecolic acid,
    cis-3-aminocyclopentane carboxamide), sarcosine, norleucine,
    alpha-methylserine, Asn; the common aliphatic/charged residues appear
    at low weight
  - N-methylation 45% on glycine positions (sarcosine, as in the real
    molecules); ring-nitrogen residues are never N-methylated (valence)
  - head-to-tail amide rings, size drawn from a triangular distribution
    over 18-30 residues, plus a linear fraction

Every generated SMILES is RDKit-validated before use.
"""
import random
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# side chain as written inside N[C@H](...)C(=O); '' = glycine; None = proline
STANDARD = {
    'A': 'C', 'R': 'CCCNC(=N)N', 'N': 'CC(=O)N', 'D': 'CC(=O)O', 'C': 'CS',
    'E': 'CCC(=O)O', 'Q': 'CCC(=O)N', 'G': '', 'H': 'Cc1cnc[nH]1',
    'I': 'C(C)CC', 'L': 'CC(C)C', 'K': 'CCCCN', 'M': 'CCSC', 'F': 'Cc1ccccc1',
    'P': None, 'S': 'CO', 'T': 'C(O)C', 'W': 'Cc1c[nH]c2ccccc12',
    'Y': 'Cc1ccc(O)cc1', 'V': 'C(C)C',
}
ALPHA_ME = {'S': ('CO', 'C')}   # alpha-methylserine
ARYL_PHE = {          # para-decorated phenylalanines (every visible Phe)
    '4conPhe': 'Cc1ccc(C(N)=O)cc1',
    '4carPhe': 'Cc1ccc(C(=O)O)cc1',
    '4pyA': 'Cc1ccncc1',
    '4phOAc': 'Cc1ccc(OCC(=O)O)cc1',
}
TRP_VARIANTS = {      # all values are SIDE CHAINS attached to the backbone Ca
    'W': 'Cc1c[nH]c2ccccc12',
    '7ClW': '[C@H](C)c1c[nH]c2c(Cl)cccc12',        # beta-methyl-7-Cl
    '7OMeW': '[C@H](C)c1c[nH]c2c(OC)cccc12',       # beta-methyl-7-OMe (real)
    '5CNW': 'Cc1c[nH]c2ccc(C#N)cc12',              # 5-cyano
    'NCMW': 'Cc1cn(CC(=O)O)c2ccccc12',             # indole-N carboxymethyl
    'Oxa': 'CC1C(=O)Nc2ccccc12',                   # oxindolylalanine
}
TRP_WEIGHTS = {'W': 0.15, '7ClW': 0.20, '7OMeW': 0.15,
               '5CNW': 0.15, 'NCMW': 0.15, 'Oxa': 0.20}

N_METHYL_GLY_PROB = 0.45   # sarcosine share among glycine positions
D_STEREO_PROB = 0.25
ALPHA_ME_PROB = 0.10
ARYL_PHE_PROB = 0.85       # every visible Phe is para-substituted
IMINO_PROB = 0.20          # combined Pro/PipOH/ACPC/HypOCM share

# backbone-embedded motifs seen in the real patent family (WO2024163691
# SEQ52/60/77 class) that the plain residue grammar never produced:
#   lanthionine  — two alpha carbons joined through CH2-S, no amide between
#   pseudoproline — N,O-acetal six-ring: N-CH2-C(CH3)2-O-CH2-CH
MOTIF_LANTHIONINE_PROB = 0.18
MOTIF_PSEUDOPRO_PROB = 0.14

_LANTHIONINE_UNITS = [
    "N[C@@H](CC(N)=O)CS[C@@H](CC)C(=O)",
    "N[C@H](CC(N)=O)CS[C@@H](C)C(=O)",
    "N[C@@H](CC)CS[C@H](CCC)C(=O)",
    "N[C@H](Cc1ccc(O)cc1)CS[C@@H](C)C(=O)",
]

_PSEUDOPRO_UNITS = [
    "N(C)CC(C)(C)OC[C@@H](Cc1ccccc1)C(=O)",
    "N(C)CC(C)(C)OC[C@H](C)C(=O)",
    "N(C)CC(C)(C)OC[C@@H](CO)C(=O)",
    # This pendant carbon has two heavy neighbours: it is CH2, not [C@H].
    "N[C@@H](COCC(C)(C)OCCCc1ccc(C(O)=O)cc1)C(=O)",
]

# weighted backbone letters: aryl/trp/imino families dominate, common
# aliphatic and charged residues stay as a low-weight tail
_BACKBONE = {
    'F': 0.10, 'W': 0.08, '4conPhe': 0.05, '4pyA': 0.045, '4carPhe': 0.02,
    'Y': 0.02, 'Sar': 0.05, 'Nle': 0.05, 'N': 0.05, 'G': 0.03,
    'P': 0.05, 'HypOCM': 0.04, 'PipOH': 0.05, 'ACPC': 0.05, 'S': 0.04,
    'V': 0.03, 'T': 0.02, 'A': 0.02,
    'R': 0.008, 'K': 0.008, 'D': 0.008, 'E': 0.008, 'C': 0.005,
    'H': 0.005, 'M': 0.005, 'I': 0.005, 'L': 0.008, 'Q': 0.01,
}
_BACKBONE_LETTERS = list(_BACKBONE)
_BACKBONE_WEIGHTS = [_BACKBONE[k] for k in _BACKBONE_LETTERS]


def _weighted_letter(rng: random.Random) -> str:
    return rng.choices(_BACKBONE_LETTERS, weights=_BACKBONE_WEIGHTS, k=1)[0]


def _residue_fragment(rng: random.Random, spec: str) -> str:
    # imino-acid family: ring nitrogen is never N-methylated (valence)
    if spec == 'P':
        return ("N1CCC[C@@H](O)C1C(=O)" if rng.random() < 0.3
                else "N1CCC[C@@H]1C(=O)")
    if spec == 'HypOCM':
        return "N1CCC[C@@H](OCC(=O)O)C1C(=O)"
    if spec == 'PipOH':
        return "N1CC[C@H](O)C[C@@H]1C(=O)"
    if spec == 'ACPC':
        return "N[C@H]1CC[C@H](C1)C(=O)"
    # sarcosine
    if spec == 'Sar':
        return "N(C)CC(=O)" if rng.random() < N_METHYL_GLY_PROB else "NCC(=O)"
    # tryptophan family
    if spec == 'W':
        variant = rng.choices(list(TRP_WEIGHTS), weights=list(TRP_WEIGHTS.values()), k=1)[0]
        side = TRP_VARIANTS[variant]
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        return f"N{centre}({side})C(=O)"
    # arylalanine family
    if spec in ARYL_PHE:
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        return f"N{centre}({ARYL_PHE[spec]})C(=O)"
    if spec == 'F':
        aryl = rng.choices(list(ARYL_PHE),
                           weights=[0.5, 0.3, 0.1, 0.1], k=1)[0]
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        return f"N{centre}({ARYL_PHE[aryl]})C(=O)"
    # alpha-methylserine
    if spec == 'S' and rng.random() < ALPHA_ME_PROB * 3:
        w1 = "[C@@]" if rng.random() < 0.5 else "[C@]"
        return f"N{w1}(CO)(C)C(=O)"
    if spec in ALPHA_ME and rng.random() < ALPHA_ME_PROB:
        s1, s2 = ALPHA_ME[spec]
        w1 = "[C@@]" if rng.random() < 0.5 else "[C@]"
        return f"N{w1}({s1})({s2})C(=O)"
    # norvaline/leucine tail residues
    if spec == 'L':
        return "N[C@@H](CC(C)C)C(=O)"
    side = STANDARD.get(spec, '')
    centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
    if spec in ('D', 'E') and rng.random() < 0.3:
        side = side.replace('=O)O', '=O)N(C)C')
    if not side:
        if rng.random() < N_METHYL_GLY_PROB:
            return "N(C)CC(=O)"
        return "NCC(=O)"
    return f"N{centre}({side})C(=O)"


def _ring_size(rng: random.Random) -> int:
    """Triangular over [16, 40] centered at 28, covering SEQ22/53 class."""
    size = int(rng.triangular(16, 40, 28))
    return max(14, min(40, size))


def generate_cyclopeptide(rng: random.Random, size: int | None = None) -> str | None:
    size = size or _ring_size(rng)
    seq = [_weighted_letter(rng) for _ in range(size)]
    if seq[0] in ('P', 'HypOCM', 'PipOH'):
        seq[0] = 'S'
    head_spec = seq[0]
    n0 = "N9"
    if head_spec == 'Sar':
        head = f"{n0}(C)CC(=O)"
    elif head_spec == 'Nle':
        head = f"{n0}[C@@H](CCCC)C(=O)"
    elif head_spec == 'ACPC':
        head = f"{n0}[C@H]1CC[C@H](C1)C(=O)"
    elif head_spec == 'W':
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        head = f"{n0}{centre}({TRP_VARIANTS['W']})C(=O)"
    elif head_spec == 'G':
        head = f"{n0}CC(=O)"
    elif head_spec in ARYL_PHE:
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        head = f"{n0}{centre}({ARYL_PHE[head_spec]})C(=O)"
    elif head_spec == 'N':
        head = f"{n0}[C@@H](CC(N)=O)C(=O)"
    else:
        side = STANDARD.get(head_spec, '')
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        if not side:
            head = f"{n0}CC(=O)"
        else:
            head = f"{n0}{centre}({side})C(=O)"
    slot, unit = _draw_molecule_motif(rng, len(seq) - 1)
    body = ''.join(_motif_or_residue(rng, s, slot, pos, unit)
                   for pos, s in enumerate(seq[1:]))
    smiles = head + body + '9'
    mol = Chem.MolFromSmiles(smiles)
    return _closed_shell_smiles(mol)


def _closed_shell_smiles(mol):
    if mol is None or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    return Chem.MolToSmiles(mol)


def _inject_backbone_motifs(rng: random.Random, smiles: str) -> str:
    """Kept for interface compatibility; motifs are injected at the
    fragment level inside generate_cyclopeptide (string splicing on SMILES
    is not safe)."""
    return smiles


def _draw_molecule_motif(rng: random.Random, n_body: int):
    """Zero or one backbone motif per molecule (the real figures carry zero
    or one): returns (slot index in the body, unit SMILES fragment)."""
    roll = rng.random()
    if roll < MOTIF_LANTHIONINE_PROB and n_body > 0:
        return rng.randrange(n_body), rng.choice(_LANTHIONINE_UNITS)
    if (roll < MOTIF_LANTHIONINE_PROB + MOTIF_PSEUDOPRO_PROB
            and n_body > 1):
        return rng.randrange(n_body), rng.choice(_PSEUDOPRO_UNITS)
    return -1, None


def _motif_or_residue(rng: random.Random, spec: str, slot: int, pos: int,
                      unit):
    if slot >= 0 and pos == slot:
        return unit
    return _residue_fragment(rng, spec)


def generate_linear(rng: random.Random) -> str | None:
    size = rng.randint(10, 30)
    seq = [_weighted_letter(rng) for _ in range(size)]
    parts = []
    for spec in seq:
        parts.append(_residue_fragment(rng, spec))
    mol = Chem.MolFromSmiles(''.join(parts) + 'O')
    return _closed_shell_smiles(mol)


def generate_batch(n: int, seed: int = 0, linear_frac: float = 0.15,
                   max_len: int = 2048) -> list[str]:
    rng = random.Random(seed)
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 10:
        tries += 1
        smi = generate_linear(rng) if rng.random() < linear_frac else generate_cyclopeptide(rng)
        if not smi or len(smi) > max_len or smi in seen:
            continue
        seen.add(smi)
        out.append(smi)
    return out
