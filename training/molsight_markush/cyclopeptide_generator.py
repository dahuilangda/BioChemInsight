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
  - N-methylation 8% concentrated on glycine (sarcosine, as in the real
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
SPECIAL_FRAGMENTS = {
    'Sar': 'N(C)CC(=O)',                            # sarcosine
    'Nle': 'NCCCC(=O)',                             # norleucine
    'PipOH': 'N1CC[C@H](O)C[C@@H]1C(=O)',           # 5-hydroxypipecolic acid
    'ACPC': 'N[C@H]1CC[C@H](C(=O))C1',              # cis-3-aminocyclopentane carboxamide
    'HypOCM': 'N1CC[C@@H](OCC(=O)O)C1C(=O)',        # Hyp(4-O-carboxymethyl)
}
SPECIAL_WEIGHTS = {
    'Sar': 0.05, 'Nle': 0.05, 'PipOH': 0.06, 'ACPC': 0.05, 'HypOCM': 0.04,
}
ALPHA_ME = {          # alpha-methyl variants: (sub1, sub2) on the quaternary Ca
    'S': ('CO', 'C'),          # alpha-methylserine (real, 2 occurrences)
    'F': ('Cc1ccccc1', 'C'),
    'W': ('Cc1c[nH]c2ccccc12', 'C'),
}
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
        return ("N1CC[C@@H](O)C1C(=O)" if rng.random() < 0.3
                else "N1CC[C@@H]1C(=O)")
    if spec == 'HypOCM':
        return "N1CC[C@@H](OCC(=O)O)C1C(=O)"
    if spec == 'PipOH':
        return "N1CC[C@H](O)C[C@@H]1C(=O)"
    if spec == 'ACPC':
        return "N[C@H]1CC[C@H](C(=O))C1"
    # sarcosine
    if spec == 'Sar':
        return "N(C)CC(=O)" if rng.random() < N_METHYL_GLY_PROB else "NCC(=O)"
    # tryptophan family
    if spec == 'W':
        variant = rng.choices(list(TRP_WEIGHTS), weights=list(TRP_WEIGHTS.values()), k=1)[0]
        side = TRP_VARIANTS[variant]
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        if variant in ('7ClW', '7OMeW'):
            # beta-methyl: the side chain carries its own chiral C-beta
            return f"N{centre}({side})C(=O)"
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
    """Triangular over [18, 30] centered at 26, matching the real molecules."""
    size = int(rng.triangular(18, 30, 26))
    return max(14, min(30, size))


def generate_cyclopeptide(rng: random.Random, size: int | None = None) -> str | None:
    size = size or _ring_size(rng)
    seq = [_weighted_letter(rng) for _ in range(size)]
    if seq[0] in ('P', 'HypOCM', 'PipOH'):
        seq[0] = 'S'
    head_spec = seq[0]
    n0 = "N9"
    if head_spec == 'S' or head_spec == 'G':
        s0 = 'CO' if head_spec == 'S' else ''
        head = f"{n0}[C@@H]({s0})C(=O)" if s0 else f"{n0}CC(=O)"
    elif head_spec in ARYL_PHE:
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        head = f"{n0}{centre}({ARYL_PHE[head_spec]})C(=O)"
    elif head_spec == 'W':
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        head = f"{n0}{centre}({TRP_VARIANTS['W']})C(=O)"
    elif head_spec == 'N':
        head = f"{n0}[C@@H](CC(N)=O)C(=O)"
    else:
        side = STANDARD.get(head_spec, '')
        centre = "[C@@H]" if rng.random() < D_STEREO_PROB else "[C@H]"
        if not side:
            head = f"{n0}CC(=O)"
        else:
            head = f"{n0}{centre}({side})C(=O)"
    body = ''.join(_residue_fragment(rng, s) for s in seq[1:])
    smiles = head + body + '9'
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def generate_linear(rng: random.Random) -> str | None:
    size = rng.randint(10, 30)
    seq = [_weighted_letter(rng) for _ in range(size)]
    parts = []
    for spec in seq:
        if spec == 'W':
            variant = rng.choices(list(TRP_WEIGHTS),
                                  weights=list(TRP_WEIGHTS.values()), k=1)[0]
            parts.append(TRP_VARIANTS[variant])
            continue
        parts.append(_residue_fragment(rng, spec))
    mol = Chem.MolFromSmiles(''.join(parts) + 'O')
    return Chem.MolToSmiles(mol) if mol is not None else None


def generate_batch(n: int, seed: int = 0, linear_frac: float = 0.15,
                   max_len: int = 1400) -> list[str]:
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
