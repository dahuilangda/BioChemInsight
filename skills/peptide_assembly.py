"""Patent sequence candidate alignment; identity requires separate review.

US20240287138A1 defaults to the explicitly named, typed-port compiler in
skills/markush_assemble/rebuild_row_candidates.py. Chemical interpretation
is fixed before mass validation. No mass-selected topology/isomer variants
are used on that path. `ok` means chemical consistency only, never reference
identity or image/SMILES pair approval. Legacy notation readers for other
patents remain available as unreviewed candidate generators.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__" and __package__ is None:   # executable entry
    sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger                      # noqa: E402
from rdkit.Chem import rdMolDescriptors               # noqa: E402

from skills.peptide_notation.skill import (   # noqa: E402
    _assemble, _find_amine_and_carboxyl)

RDLogger.DisableLog("rdApp.*")

TABLE1_PATH = "/tmp/table1_rows.json"
LISTING_PATH = "/tmp/listing_entries.json"
NAME_CORPUS_PATH = "/tmp/name_corpus.json"
LEXICON_PATH = os.path.join(ROOT, "data", "residue_lexicon.json")
RESULT_PATH = "/tmp/peptide_assembly_result.json"

MASS_TOLERANCE = 0.8

# ---------------------------------------------------------------------------
# Notation styles (generalization over patent families)
# ---------------------------------------------------------------------------
# US baseline : cyclo(...) sequences, head-to-tail assembly, SITE-note
#               listings; unchanged defaults.
# kras        : same cyclo(...) wrapper, but several series close the ring
#               through a chloroacetyl cap thioethered to a Cys/Cysteamine
#               sulfur instead of a head-to-tail peptide bond; no SITE notes
#               (code-first lexicon reading + mass arbitration).
# gpc3        : linear hyphen notation WITHOUT cyclo(); position 1 carries
#               the chloroacetyl (ClAc) cap and the LAST residue (Cys) closes
#               the ring by S-alkylation; 'Me' prefix marks N-methylated
#               residues and a lowercase 'd' prefix marks D-forms; the
#               single letter 'l' prints for isoleucine (OCR-stable I/l swap
#               resolved by the patent's building-block tables).
STYLE_OPTIONS = {
    "us": {"linear_notation": False, "thioether": False,
           "strict_named_chemistry": True,
           "n_methyl_prefix": False, "d_prefix": False,
           "lowercase_l_is_isoleucine": False,
           "token_aliases": {}},
    "kras": {"linear_notation": False, "thioether": True,
             "n_methyl_prefix": True, "d_prefix": True,
             "lowercase_l_is_isoleucine": False,
             # compound-table OCR prints Cys as C1/Cl (the dC forms hit the
             # d-prefix first) and 1-vs-l slips into naphthylalanine /
             # cyclobutane keys
             "token_aliases": {
                 "c": "C", "c1": "C", "c2": "C", "cl": "C",
                 "1nai": "1Nal", "c3amcb1c": "c3amCblc",
                 "flac": "FLac", "amecl": "aMeCl", "amec1": "aMeCl",
                 "amec": "aMeCl", "nmeomme": "NMeOrnMe",
                 "nmesbmef": "NMeRbMeF", "mec1": "MeC1", "nvafl": "NvaF",
             },
             "token_alias_d_form": set(),
             # the h* codes collide across families: the US lexicon stores
             # a different skeleton for them than this patent's monomer
             # table defines; overridden in run() before alignment
             "lexicon_overrides": {
                 "hL": "N[C@H](C=O)CCC(C)C",   # homo-leucine
                 "hK": "N[C@H](C=O)CCCCCN",    # homo-lysine
                 "hS": "N[C@H](C=O)CCO",       # homo-serine
                 "hL3F": "N[C@H](C=O)CCC(C)CF",
             }},
    "gpc3": {"linear_notation": True, "thioether": True,
             "n_methyl_prefix": True, "d_prefix": True,
             "lowercase_l_is_isoleucine": True,
             "token_aliases": {"nai2": "Nal2", "nal2": "Nal2",
                               "nai27n": "Nal27N", "nal27n": "Nal27N",
                               "f4c00": "F4COO", "f4aa0": "F4aao",
                               "f4aa": "F4aao", "f4aao": "F4aao",
                               "f4amcopipzaa": "F4aaopipzaa"},
             "token_alias_d_form": set(),
             "lexicon_overrides": {}},
}


def _style_opts(style: str | None, overrides: dict | None) -> dict:
    opts = dict(STYLE_OPTIONS.get(style or "us", STYLE_OPTIONS["us"]))
    for key, value in (overrides or {}).items():
        if value is not None:
            opts[key] = value
    return opts

# anchor vocabulary: standard letter -> standard residue name fragment
STD_ANCHOR = {
    "W": "tryptophan", "Y": "tyrosine", "F": "phenylalanine",
    "S": "serine", "N": "asparagine", "G": "glycine", "C": "cysteine",
    "K": "lysine", "R": "arginine", "H": "histidine", "P": "proline",
    "T": "threonine", "M": "methionine", "I": "isoleucine",
    "L": "leucine", "V": "valine", "A": "alanine", "D": "aspartic",
    "E": "glutamic", "Q": "glutamine",
}
STD_SET = set(STD_ANCHOR)
_ANCHOR_RE = {ch: re.compile(r"(?<![a-z])" + name) for ch, name in STD_ANCHOR.items()}

# DP scoring weights (deterministic, unit-less)
W_ANCHOR = 6.0          # standard letter <-> its standard name
W_AGREE = 6.0           # name and code resolve to the same fragment
W_NAME = 2.0            # the SITE name resolves in the lexicon
W_CODE = 0.5            # only the code resolves
W_BAD = -2.0            # neither side resolves
W_STD_MISMATCH = -8.0   # standard letter forced onto a foreign SITE name
W_SKIP_NOTE = -4.0      # an unexplained chemical name
W_SKIP_CODE = -0.4      # a residue without SITE name (the normal case)


def _load_lexicon() -> dict:
    try:
        raw = json.load(open(LEXICON_PATH, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    lex = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict) or not entry.get("fragment_smiles"):
            continue
        lex[key.casefold()] = entry
    return lex


LEXICON = _load_lexicon()
# vocabulary for merged-token splitting: multi-char lexicon keys plus the
# uppercase standard letters (case-sensitive; lowercase letters such as
# the 'e' in 'NMe' are not codes)
MULTI_KEYS = {k for k in LEXICON if len(k) >= 2}


# fixed-chemistry fallback for a fresh checkout; lexicon entries override
STD_BASELINE = {
    'A': 'N[C@@H](C)C=O',
    'C': 'N[C@H](C=O)CS',
    'D': 'N[C@H](C=O)CC(=O)O',
    'E': 'N[C@H](C=O)CCC(=O)O',
    'F': 'N[C@H](C=O)Cc1ccccc1',
    'G': 'NCC=O',
    'H': 'N[C@H](C=O)Cc1cnc[nH]1',
    'I': 'N[C@H](C=O)[C@H](C)CC',
    'K': 'N[C@H](C=O)CCCCN',
    'L': 'N[C@H](C=O)CC(C)C',
    'M': 'N[C@H](C=O)CCSC',
    'N': 'N[C@H](C=O)CC(N)=O',
    'P': 'N1CCC[C@H]1C=O',
    'Q': 'N[C@H](C=O)CCC(N)=O',
    'R': 'N[C@H](C=O)CCCNC(=N)N',
    'S': 'N[C@H](C=O)CO',
    'T': 'N[C@H](C=O)[C@@H](C)O',
    'V': 'N[C@H](C=O)C(C)C',
    'W': 'N[C@H](C=O)Cc1c[nH]c2ccccc12',
    'Y': 'N[C@H](C=O)Cc1ccc(O)cc1',
}


def _std_fragments() -> dict:
    """Backbone-form fragments for the 20 standard letters: lexicon
    entries (figure-verified) override the fixed baseline."""
    frags = dict(STD_BASELINE)
    for ch in sorted(STD_SET):
        entry = LEXICON.get(ch.casefold())
        if entry:
            frags[ch] = entry["fragment_smiles"]
    # asparagine override (lexicon 'N' entry is N-methylglycine)
    frags["N"] = "N[C@H](C=O)CC(N)=O"
    return frags


STD_FRAGMENTS = _std_fragments()

# OCR aliases established by the corpus itself (every aligned occurrence):
# * 'SasMeW7OMe' fuses two codes with the 'rb' of 'Sar-SbMeW7OMe' lost;
#   its SITE notes always carry BOTH the Sarcosine and the
#   7-methoxy-beta-Me-Trp names.
# * 'TyrOcm' has no lexicon key, but every SITE note that aligns to it is
#   the carboxymethoxy-phenylalanine entry below.
# * 'dCClAc' fuses the D-Cys linker with the chloroacetyl cap (KRAS).
SPLIT_ALIASES = {
    "sasmebw7ome": ["Sar", "SbMeW7OMe"],
    # second OCR spelling of the same fusion ('r' of 'Sar' lost only);
    # every listing note that pairs with this token carries BOTH the
    # Sarcosine and the 7-methoxy-beta-Me-Trp names
    "sasmew7ome": ["Sar", "SbMeW7OMe"],
    # fusion of NhE + G: once 'hE' entered the lexicon the greedy cover
    # would split this token; keep it whole so the SITE note
    # ('4-(Carboxymethylamino)butanoic acid') can resolve it
    "nheg": ["NhEG"],
    "dcclac": ["dC", "ClAc"],
}
CODE_ALIASES = {
    "tyrocm": "(S)-2-Amino-3-(4-(carboxymethoxy)phenyl)propanoic acid",
    # codes whose lexicon keys are malformed skeletons; the correct
    # fragments are already in the lexicon under the SITE names that the
    # listings pair with these codes (verified by the mass arbitration)
    "trpch2cooh": "(S)-2-Amino-3-[1-(carboxymethyl)indol-3-yl]propanoic acid",
    "trp7azch2cooh": "(S)-2-Amino-3-[1-(carboxymethyl)pyrrolo[2,3-b]pyridin-3-yl]propanoic acid",
    "dprot4nme2": "(2R,4S)-4-(Dimethylamino)pyrrolidine-2-carboxylic acid",
    "morpip6dime": "(S)-6,6-dimethylmorpholine-3-carboxylic acid",
    # OCR digit/letter swap, resolved to the figure-verified 7-chloro key
    "trp7c1": "Trp7Cl",
    # 'SbMe' family = beta-methyl amino acids (SbMeW7OMe / SbMeW2Me notes
    # confirm the reading); 'dProt4Me3' = the dimethylamino-proline entry
    "sbmew4pal": "(2S,3S)-2-Amino-3-pyridin-4-ylbutanoic acid",
    "dprot4me3": "(2R,4S)-4-(Dimethylamino)pyrrolidine-2-carboxylic acid",
}
# Aliases whose target is absent from the lexicon (fresh checkout) simply
# resolve to nothing downstream; the token surfaces as an unknown gap.
CODE_ALIASES = {k: v for k, v in CODE_ALIASES.items()
                if v.casefold() in LEXICON}

# ---------------------------------------------------------------------------
# Mass-refuted lexicon families
# ---------------------------------------------------------------------------
# Several recurring lexicon code families store skeletons that the patent's
# own declared exact masses refute (each gap below was verified against
# dozens of rows before the family was admitted):
#   * bhcLeu / dbhcLeu  -> stored as a beta-hydroxy-leucine chain; rows whose
#     SITE name resolves to ACPC pass the check with the aminocyclopentane
#     skeleton, so that is the family variant.
#   * bhcLeut4OH / bhcLeut45diOH -> stored as the same chain skeleton; the
#     failing rows sit exactly +2.016 Da (one degree of unsaturation) off,
#     i.e. a (di)hydroxy-ACPC ring. Isomer-level caveat: the OH placement
#     follows the ring skeleton, not figure evidence.
#   * Pip5OH / PipT5OH / Pipt5OH -> stored as 5-hydroxyPIPERIDINE (6-ring);
#     some rows sit exactly one CH2 (14.016 Da) heavy, i.e. a 5-ring
#     hydroxyproline skeleton, others confirm the 6-ring entry, so both
#     ring sizes are offered and the arbitration picks.
#   * ClAc -> behaves as a glycine-mass unit where rows pass, exactly one
#     CH2 heavier (N-methylglycine, the ClAc~NMeG confusion) or as a plain
#     acetyl cap (43.018 Da) elsewhere.
#   * 'N' -> asparagine per the standard-letter convention; the N-methyl-
#     asparagine skeleton (+14.016 Da, same composition as glutamine) is
#     offered as the fallback for rows that reject asparagine.
#   * Phe4COOH / Phe4CONH2 -> acid/amide side-chain suffix confusion
#     (0.984 Da); the sibling fragment is the fallback.
# All variants are tried only AFTER the lexicon-first reading of a
# candidate fails, and a row is accepted solely when the assembled exact
# mass matches the declared value within MASS_TOLERANCE.
_VARIANT_ACPC = LEXICON.get("acpc", {}).get("fragment_smiles") \
    or "N[C@H]1CC[C@@H](C=O)C1"
_VARIANT_OH_PRO = "N1C[C@H](O)C[C@@H]1C=O"          # 115.0633 (5-ring)
_VARIANT_OH_ACPC = "N[C@H]1CC[C@@H](O)C1C=O"        # 129.0790 (hydroxy-ACPC)
_VARIANT_DIOH_ACPC = "N[C@H]1CC(O)C(O)C1C=O"        # 145.0739 (diOH-ACPC)
_VARIANT_NME_G = "N(C)CC=O"                         # 73.0528
_VARIANT_NME_ASN = "N[C@H](C=O)CC(N(C)C)=O"         # 144.0899

# per-family fallback options (lexicon reading = None comes first); the
# Pip5OH family offers both ring sizes because the table's 6-ring entry
# is mass-refuted in some rows and mass-confirmed in others
_FAMILY_OPTIONS = {
    "clac": [None, _VARIANT_NME_G, "cap:CC=O"],
    "n": [None, _VARIANT_NME_ASN],
    "pip5oh": [None, _VARIANT_OH_PRO, "pip:VARIANT_PIP6"],
    "pipt5oh": [None, _VARIANT_OH_PRO, "pip:VARIANT_PIP6"],
    "pipc4oh": [None, "pip:VARIANT_PIP6"],
    "pipc5oh": [None, "pip:VARIANT_PIP6"],
    "bhcleu": [None, _VARIANT_ACPC],
    "dbhcleu": [None, _VARIANT_ACPC],
    "bhcleut4oh": [None, _VARIANT_OH_ACPC],
    "bhcleut45dioh": [None, _VARIANT_DIOH_ACPC],
    # acid vs amide side chain: 'COOH'/'CONH2' differ by one suffix in the
    # table abbreviations and by 0.984 Da in composition; the sibling
    # fragment is offered as a fallback for rows sitting one NH/O apart
    "phe4cooh": [None, LEXICON.get("phe4conh2", {}).get("fragment_smiles")],
    "phe4conh2": [None, LEXICON.get("phe4cooh", {}).get("fragment_smiles")],
    "phe3cooh": [None, LEXICON.get("phe3conh2", {}).get("fragment_smiles")],
    "phe3conh2": [None, LEXICON.get("phe3cooh", {}).get("fragment_smiles")],
}


def split_notation(sequence: str,
                   linear_notation: bool = False) -> tuple[list[str], list[str]]:
    """Sequence line -> (ring codes, tail codes). Parentheses are
    respected so side-chain tags like K(N-Biotin) stay one token.

    US/KRAS families wrap the ring in cyclo(...); the GPC3 family prints a
    bare hyphen-joined linear sequence (whole string is the ring, optional
    -NH2 tail is stripped)."""
    seq = sequence.strip()
    if linear_notation:
        # drop the printed C-terminal amide marker; it describes the last
        # residue's cap, not an extra ring member
        seq = re.sub(r"-\s*N\s*H\s*2\s*$", "", seq, flags=re.I)
        inner = seq              # whole linear string is the ring body
        tail_text = ""
    else:
        m = re.match(r"^cyclo\(", seq)
        if not m:
            # KRAS sequence-table rows print the ring body without the
            # cyclo() wrapper; treat a bare hyphenated token string as the
            # ring (documented notation variant, not a silent reinterpret
            # of a wrapped row)
            inner = seq
            tail_text = ""
            depth = 0
            close = len(seq) - 1
        else:
            depth = 0
            close = -1
            for i, ch in enumerate(seq):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        close = i
                        break
            if close < 0:
                return [], []
            inner = seq[m.end():close]
            tail_text = seq[close + 1:]

    tokens: list[str] = []
    buf = ""
    depth = 0
    for ch in inner:
        if ch == "(":
            depth += 1
            buf += ch
        elif ch == ")":
            depth -= 1
            buf += ch
        elif ch == "-" and depth == 0:
            if buf:
                tokens.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        tokens.append(buf)

    tail = []
    buf = ""
    depth = 0
    for ch in tail_text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "-" and depth == 0:
            if buf.strip():
                tail.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        tail.append(buf.strip())
    tail = [t for t in tail if t.upper() != "NH2"]
    return tokens, tail


def split_merged(token: str) -> list[str]:
    """Split an OCR-fused token ('aMeSF4SO2NH2' = 'aMeS-F4SO2NH2') by
    greedy longest lexicon-key cover. A cover into >=2 parts wins over
    the whole-key reading; single parts must be multi-char keys or
    uppercase standard letters so ordinary names never fall apart.
    Documented corpus aliases (SPLIT_ALIASES) run first."""
    alias = SPLIT_ALIASES.get(token.casefold())
    if alias:
        return list(alias)

    def cover(tok: str) -> list[str] | None:
        parts: list[str] = []
        i, n = 0, len(tok)
        while i < n:
            for j in range(min(n, i + 25), i, -1):
                seg = tok[i:j]
                if j - i == 1:
                    if seg in STD_SET:          # uppercase standard letter
                        parts.append(seg)
                        i = j
                        break
                elif seg.casefold() in MULTI_KEYS:
                    parts.append(seg)
                    i = j
                    break
            else:
                return None
        return parts

    whole = cover(token)
    if whole is None:
        return [token]                          # unresolved: keep as-is
    if len(whole) >= 2:
        return whole
    best: list[str] | None = None
    for i in range(1, len(token)):
        left, right = cover(token[:i]), cover(token[i:])
        if left and right:
            cand = left + right
            if best is None or len(cand) > len(best):
                best = cand
    return best if best else whole


# ---------------------------------------------------------------------------
# new-family token normalization (Me / d prefixes, single-letter canon)
# ---------------------------------------------------------------------------

_NME_RE = re.compile(r"^Me(?=[A-Z])")
_DFORM_RE = re.compile(r"^d(?=[A-Z0-9])")
_DAME_RE = re.compile(r"^da(?=[A-Z])")   # 'daMeCl' = D-alpha-methyl-Cys


def normalize_token(token: str, opts: dict) -> tuple[str, dict]:
    """One sequence token -> (base code, flags). Flags carry the N-methyl
    and D-form markers that the notation prints as prefixes ('MeK',
    'dF'), plus the OCR-stable lowercase-'l'-for-isoleucine convention of
    the GPC3 family and the per-style token alias table (compound-table
    OCR of the Cys series). Everything is recorded, never silently
    folded."""
    flags: dict = {}
    code = token
    if opts.get("d_prefix"):
        m = _DAME_RE.match(code)
        if m:
            code = code[1:]          # strip the D marker only: daMeCl->aMeCl
            flags["d_form"] = True
            flags["daMe_rewrite"] = True
    if opts.get("d_prefix"):
        m = _DFORM_RE.match(code)
        if m:
            code = code[m.end():]
            flags["d_form"] = True
    if opts.get("n_methyl_prefix"):
        m = _NME_RE.match(code)
        if m:
            code = code[m.end():]
            flags["n_methyl"] = True
    alias = opts.get("token_aliases", {}).get(code.casefold())
    if alias is not None:
        if code.casefold() in opts.get("token_alias_d_form", set()):
            flags["d_form"] = True
        flags["alias"] = f"{code}->{alias}"
        code = alias
    if opts.get("lowercase_l_is_isoleucine") and code == "l":
        code = "I"
        flags["l_canonicalized_to_I"] = True
    return code, flags


def _n_methyl_fragment(fragment: str) -> str:
    """Backbone-form fragment -> its N-methyl variant (leading backbone
    amine gains a methyl branch). Ring-nitrogen fragments (Pro-line)
    transform the same way since their N is the leading atom."""
    if fragment.startswith("N("):
        return fragment
    if fragment.startswith("N"):
        return "N(C)" + fragment[1:]
    return fragment


def _std_n_methyl(code: str) -> str | None:
    """N-methyl variant of a standard single-letter fragment, else None."""
    frag = STD_FRAGMENTS.get(code.upper())
    return _n_methyl_fragment(frag) if frag else None


def resolve_token(code: str, flags: dict) -> tuple[str | None, str | None]:
    """Flag-aware token resolution -> (fragment, key). Resolution order:
    flag-qualified standard letter, lexicon (flag-aware n_methyl guard),
    CODE_ALIASES. Returns (None, None) when unknown."""
    if len(code) == 1 and code.upper() in STD_FRAGMENTS:
        frag = STD_FRAGMENTS[code.upper()]
        key = "std:" + code.upper()
        if flags.get("n_methyl"):
            frag, key = _std_n_methyl(code), key + "+NMe"
        return frag, key
    fold = code.casefold()
    entry = LEXICON.get(fold)
    if entry is None and flags.get("n_methyl") and len(code) > 1:
        # composite code whose lexicon key stores the N-methyl form whole
        # (e.g. GPC3 'MeKCOpipzaa'): the Me prefix was already stripped,
        # retry the lookup under the printed form
        entry = LEXICON.get("me" + fold)
        if entry:
            return entry["fragment_smiles"], "me" + fold
    if entry:
        # the lexicon entry already carries the methyl (e.g. Sar): a Me
        # prefix in the sequence is OCR noise, not a second methylation
        if flags.get("n_methyl") and not entry.get("n_methyl"):
            return _n_methyl_fragment(entry["fragment_smiles"]), fold + "+NMe"
        return entry["fragment_smiles"], fold
    alias = CODE_ALIASES.get(fold)
    if alias:
        entry = LEXICON.get(alias.casefold())
        if entry:
            frag = entry["fragment_smiles"]
            if flags.get("n_methyl"):
                frag = _n_methyl_fragment(frag)
            return frag, alias.casefold() + ("+NMe" if flags.get("n_methyl") else "")
    return None, None


# ---------------------------------------------------------------------------
# chloroacetyl-cap thioether assembly (KRAS Cysteamine/ClAc rows and the
# whole GPC3 family: ring closes S-alkylation of the cap onto a terminal
# thiol instead of a head-to-tail peptide bond)
# ---------------------------------------------------------------------------

def _assemble_thioether(units: list[dict], cap_fragment: str,
                        ) -> tuple[str | None, str, list[str]]:
    """Linear chain of units + N-terminal chloroacetyl cap + intramolecular
    S-alkylation of the cap's CH2-Cl onto the LAST unit's free thiol.
    Deterministic RWMol construction; the exact-mass arbitration still decides
    the reading. Returns (smiles, error, notes)."""
    notes: list[str] = []
    try:
        em = Chem.RWMol()
        amine_idx, carboxyl_idx, thiol_idx = [], [], None
        for u in units:
            mol = Chem.MolFromSmiles(u["entry"]["fragment_smiles"])
            if mol is None:
                return None, f"unit '{u['code']}' does not parse", notes
            amine, carboxyl = _find_amine_and_carboxyl(mol)
            offset = em.GetNumAtoms()
            for a in mol.GetAtoms():
                em.AddAtom(Chem.Atom(a))
            for b in mol.GetBonds():
                em.AddBond(b.GetBeginAtomIdx() + offset,
                           b.GetEndAtomIdx() + offset, b.GetBondType())
            amine_idx.append(offset + amine if amine is not None else None)
            carboxyl_idx.append(offset + carboxyl if carboxyl is not None
                                else None)
            # free side-chain thiol: S with exactly one heavy neighbor
            for a in mol.GetAtoms():
                if a.GetSymbol() == "S" and a.GetDegree() == 1:
                    thiol_idx = offset + a.GetIdx()

        cap = Chem.MolFromSmiles(cap_fragment)
        if cap is None:
            return None, "cap fragment does not parse", notes
        cap_amine, cap_carboxyl = _find_amine_and_carboxyl(cap)
        if cap_carboxyl is None:
            return None, "cap has no acyl carbon", notes
        cap_cl = None
        cap_ch2 = None
        for a in cap.GetAtoms():
            if a.GetSymbol() == "Cl":
                cap_cl = a.GetIdx()
                for b in a.GetBonds():
                    other = cap.GetAtomWithIdx(b.GetOtherAtomIdx(a.GetIdx()))
                    if other.GetSymbol() == "C":
                        cap_ch2 = other.GetIdx()
        if cap_cl is None or cap_ch2 is None:
            return None, "cap has no chloroacetyl CH2-Cl motif", notes
        offset = em.GetNumAtoms()
        for a in cap.GetAtoms():
            em.AddAtom(Chem.Atom(a))
        for b in cap.GetBonds():
            em.AddBond(b.GetBeginAtomIdx() + offset,
                       b.GetEndAtomIdx() + offset, b.GetBondType())
        cap_acyl = offset + cap_carboxyl
        cap_ch2_em = offset + cap_ch2
        cap_cl_em = offset + cap_cl

        head = amine_idx[0]
        if head is None or cap_acyl is None:
            return None, "head amine or cap acyl missing", notes
        em.AddBond(cap_acyl, head, Chem.BondType.SINGLE)
        for i in range(len(units) - 1):
            if carboxyl_idx[i] is None or amine_idx[i + 1] is None:
                return None, (f"unit {i + 1} lacks backbone "
                              "amine/carboxyl for the chain bond"), notes
            em.AddBond(carboxyl_idx[i], amine_idx[i + 1], Chem.BondType.SINGLE)
        if thiol_idx is None:
            return None, "no free side-chain thiol for the thioether closure", notes
        # the last chain unit keeps its C-terminus as a free acid in this
        # family (ring closure goes through the cap thioether, not a
        # peptide bond): promote an aldehyde terminus to COOH
        last_c = carboxyl_idx[-1] if carboxyl_idx else None
        if last_c is not None:
            carb = em.GetAtomWithIdx(last_c)
            has_o = any(b.GetBondTypeAsDouble() == 2 and
                        em.GetAtomWithIdx(b.GetOtherAtomIdx(last_c)).GetSymbol() == "O"
                        for b in carb.GetBonds())
            no_other = carb.GetDegree() == 2 and \
                sum(1 for b in carb.GetBonds()
                    if em.GetAtomWithIdx(b.GetOtherAtomIdx(last_c)).GetSymbol() == "O") == 1
            if has_o and no_other:
                o = em.AddAtom(Chem.Atom("O"))
                em.AddBond(last_c, o, Chem.BondType.SINGLE)
        # S-alkylation onto the cap's CH2 carbon (the Cl atom itself only
        # marks the leaving group and is removed below)
        em.AddBond(thiol_idx, cap_ch2_em, Chem.BondType.SINGLE)
        # the chloride leaves: remove Cl, clear the sulfur's implicit H
        em.RemoveAtom(cap_cl_em)
        mol = em.GetMol()
        sulfur = [a for a in mol.GetAtoms()
                  if a.GetSymbol() == "S" and a.GetDegree() == 2]
        for a in sulfur:
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol), "", notes
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:160]}", notes


CLAC_CAP_FRAGMENT = "ClCC=O"   # chloroacetyl; aldehyde-H vanishes on
                               # acylation (RDKit implicit-H adjustment)

def _thioether_reading(codes: list[str],
                       opts: dict) -> tuple[list[str], str] | None:
    """Detect the cap/thiol reading for a thioether-style row.
    Returns (chain_codes, cap_fragment) or None when the row has no
    chloroacetyl-cap pattern."""
    fold = [c.casefold() for c in codes]
    cap = None
    chain = list(codes)
    if opts.get("linear_notation"):
        # GPC3: the cap is implicit (position 1 carries ClAc, last residue
        # is the thiol) — every row of this family takes the reading
        cap = CLAC_CAP_FRAGMENT
    elif fold[-1:] == ["clac"]:
        cap = CLAC_CAP_FRAGMENT
        chain = chain[:-1]
    if cap is None:
        return None
    return chain, cap


_MISC_NOTE_RE = re.compile(
    r"connected to position|cross.?link|^(acetylated|amidated) residue$",
    re.I)


def filter_notes(notes: list[str]) -> list[str]:
    """Drop cross-link descriptors and Acetylated/Amidated-residue
    markers; they describe caps/topology, not residues."""
    return [n for n in notes if not _MISC_NOTE_RE.search(n)]


def _name_variants(name: str) -> list[str]:
    """Deterministic OCR normalization: casefold, drop the listing's
    dedup suffixes ('Sarcosine 13'), and repair the classic 1/l swap
    ('1-Norleucine' -> 'l-norleucine')."""
    s = " ".join(str(name).split())
    out = []
    folded = s.casefold()
    out.append(folded)
    trimmed = re.sub(r"\s+\d+$", "", folded)
    out.append(trimmed)
    out.append(re.sub(r"(^|-|\s)1(?=[-\s])", r"\1l", trimmed))
    return out


def resolve_name(name: str) -> tuple[str | None, str | None]:
    """SITE name -> (fragment_smiles, lexicon key). None when unknown."""
    for variant in _name_variants(name):
        entry = LEXICON.get(variant)
        if entry:
            return entry["fragment_smiles"], variant
    return None, None


def resolve_code(code: str) -> tuple[str | None, str | None]:
    """Code -> (fragment_smiles, lexicon key), e.g. 'W', 'Sar', 'Nle'.
    Single-letter standard codes resolve through the standard fragment
    table (with the asparagine override), not through arbitrary lexicon
    keys that may collide with another notation."""
    if len(code) == 1 and code.upper() in STD_FRAGMENTS:
        return STD_FRAGMENTS[code.upper()], "std:" + code.upper()
    alias = CODE_ALIASES.get(code.casefold())
    if alias:
        entry = LEXICON.get(alias.casefold())
        if entry:
            return entry["fragment_smiles"], alias.casefold()
    entry = LEXICON.get(code.casefold())
    if entry:
        return entry["fragment_smiles"], code.casefold()
    return None, None


def _is_anchor(code: str, note: str) -> bool:
    return (len(code) == 1 and code.upper() in STD_SET
            and bool(_ANCHOR_RE[code.upper()].search(note.casefold())))


def _pair_score(code: str, note: str, frags: dict[str, str]) -> float:
    """Score for pairing one code with one note (both resolved forms are
    pre-computed in frags: code -> fragment or '', note -> fragment or '')."""
    if len(code) == 1 and code.upper() in STD_SET:
        return W_ANCHOR if _is_anchor(code, note) else W_STD_MISMATCH
    nfrag = frags.get("n:" + note) or ""
    cfrag = frags.get("c:" + code) or ""
    if nfrag:
        return W_AGREE if (cfrag and nfrag == cfrag) else W_NAME
    if cfrag:
        return W_CODE
    return W_BAD


def dp_align(codes: list[str], notes: list[str],
             frags: dict[str, str]) -> tuple[float, list[tuple[int, int]]]:
    """Monotone alignment of notes onto codes maximizing the anchor /
    fragment-agreement score. Returns (score, [(code_idx, note_idx)])."""
    n, m = len(codes), len(notes)
    if m == 0:
        return 0.0, []
    NEG = float("-inf")
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    take = [[0] * (m + 1) for _ in range(n + 1)]   # 2=pair 1=skip note 0=skip code
    for i in range(n + 1):
        dp[i][m] = W_SKIP_CODE * 0                  # no notes left: free
    for j in range(m):
        dp[n][j] = W_SKIP_NOTE * (m - j)            # leftover notes penalized
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best = NEG
            move = 0
            c1 = dp[i + 1][j] + W_SKIP_CODE
            if c1 > best:
                best, move = c1, 0
            c2 = dp[i][j + 1] + W_SKIP_NOTE
            if c2 > best:
                best, move = c2, 1
            c3 = dp[i + 1][j + 1] + _pair_score(codes[i], notes[j], frags)
            if c3 > best:
                best, move = c3, 2
            dp[i][j] = best
            take[i][j] = move
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        move = take[i][j]
        if move == 2:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif move == 1:
            j += 1
        else:
            i += 1
    return dp[0][0], pairs


def candidate_alignments(codes: list[str],
                         notes: list[str]) -> list[dict]:
    """Deterministic hypothesis list: the direct order first, then ring
    rotations whose DP score at least ties the direct one, best first.
    Each candidate maps note index -> original code index."""
    if not notes:
        return [{"score": 0.0, "rotation": 0, "pairs": []}]
    def prep(cs):
        f: dict[str, str] = {}
        for c in cs:
            f["c:" + c] = resolve_code(c)[0] or ""
        for nt in notes:
            f["n:" + nt] = resolve_name(nt)[0] or ""
        return f
    base_frags = prep(codes)
    score, pairs = dp_align(codes, notes, base_frags)
    out = [{"score": score, "rotation": 0, "pairs": pairs}]
    n = len(codes)
    for r in range(1, n):
        rot = codes[r:] + codes[:r]
        score_r, pairs_r = dp_align(rot, notes, prep(rot))
        if score_r >= score:
            fixed = [((ci + r) % n, nj) for ci, nj in pairs_r]
            out.append({"score": score_r, "rotation": r, "pairs": fixed})
    out.sort(key=lambda c: (-c["score"], c["rotation"]))
    return out


def _variant_sets(codes: list[str]) -> list[tuple[str, dict]]:
    """Deterministic fallback fragment sets for the recurring mass-refuted
    code families, as a bounded Cartesian product of per-family options.
    A 'cap:' prefix marks an N-acyl cap that is lifted out of the ring.
    Options are ordered lexicon-first; the exact-mass arbitration decides which
    reading (if any) holds for the row. Bounded: rows carry at most a few
    families, so the product stays small."""
    families = _FAMILY_OPTIONS
    present = []
    for code in codes:
        cf = code.casefold()
        if cf in families and cf not in [p[0] for p in present]:
            present.append((cf, families[cf]))
    if not present:
        return [("lexicon", {})]
    names = [cf for cf, _ in present]
    sets = []
    for combo in itertools.product(*[opts for _, opts in present]):
        vmap = {}
        label = []
        for (cf, _), choice in zip(present, combo):
            if choice is not None:
                vmap[cf] = choice
                label.append(f"{cf}={choice[:14]}")
        sets.append(("lexicon" if not label else "+".join(label), vmap))
    return sets


def build_units(codes: list[str], notes: list[str],
                pairs: list[tuple[int, int]],
                variant_map: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Resolve every code: the paired SITE name outranks the code itself
    (the listing name is the authoritative description). Returns
    (resolvable units in ring order, unknown positions). Codes present
    in variant_map get that fragment as a fallback override; the mass
    arbitration decides whether the override holds for the row."""
    name_of = {ci: notes[nj] for ci, nj in pairs}
    units: list[dict] = []
    caps: list[str] = []
    unknown: list[dict] = []
    for ci, code in enumerate(codes):
        name = name_of.get(ci)
        frag = key = None
        if name:
            frag, key = resolve_name(name)
        if frag is None:
            frag, key = resolve_code(code)
        source = "lexicon"
        if variant_map and code.casefold() in variant_map:
            spec = variant_map[code.casefold()]
            if spec.startswith("cap:"):
                caps.append(spec[4:])
                source = "variant-cap"
                continue
            if spec.startswith("pip:"):
                frag = LEXICON.get("pipt5oh", {}).get("fragment_smiles") \
                    or "N1CCC[C@@H](O)[C@@H]1C=O"   # 6-ring 5-OH-piperidine
                key = "variant:pipt5oh"
                source = "variant"
                units.append({"code": code, "name": name,
                              "entry": {"fragment_smiles": frag},
                              "lexicon_key": key, "source": source})
                continue
            frag = spec
            key = "variant:" + code.casefold()
            source = "variant"
        if frag is None:
            unknown.append({"index": ci, "code": code, "name": name})
            continue
        units.append({"code": code, "name": name,
                      "entry": LEXICON.get(key or "", {"fragment_smiles": frag})
                      if not key.startswith("variant:")
                      else {"fragment_smiles": frag},
                      "lexicon_key": key, "source": source})
    return units, caps, unknown


def exact_mass(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcExactMolWt(mol) if mol else None


def _build_units_tokens(tokens: list[tuple[str, dict]],
                        ) -> tuple[list[dict], list[dict]]:
    """Flag-aware token resolution -> (units, unknown). Units carry the
    same shape as build_units output; every fragment comes from
    resolve_token (standard-letter table, lexicon, aliases)."""
    units: list[dict] = []
    unknown: list[dict] = []
    for ci, (code, flags) in enumerate(tokens):
        frag, key = resolve_token(code, flags)
        if frag is None:
            unknown.append({"index": ci, "code": code, "name": None,
                            "flags": flags})
            continue
        units.append({"code": code, "name": None, "flags": flags,
                      "entry": {"fragment_smiles": frag},
                      "lexicon_key": key, "source": "lexicon+flags"})
    return units, unknown


def _mass_trial(trial: dict, smiles: str | None, error: str,
                declared: float) -> dict:
    """Shared mass arbitration: fills trial in place; returns it."""
    if not smiles:
        trial["failure"] = f"assembly failed: {error}"
        return trial
    mass = exact_mass(smiles)
    if mass is None:
        trial["failure"] = "assembled SMILES does not parse"
        return trial
    delta = round(mass - float(declared), 4)
    trial["delta"] = delta
    if abs(delta) <= MASS_TOLERANCE:
        trial["ok"] = True
        trial["smiles"] = smiles
    else:
        trial["failure"] = (f"mass mismatch: computed {mass:.4f} vs declared "
                            f"{float(declared):.4f}")
    return trial


def _align_tokens_thioether(tokens: list[tuple[str, dict]],
                            declared: float, opts: dict) -> dict | None:
    """Chloroacetyl-cap thioether reading (KRAS Cys/ClAc series, GPC3
    family). Returns a trial dict, or None when the row has no cap
    pattern. No DP/rotation: this family's row order is the chain order."""
    codes = [c for c, _ in tokens]
    reading = _thioether_reading(codes, opts)
    if reading is None:
        return None
    chain, cap = reading
    chain_tokens = tokens[:len(chain)]
    units, unknown = _build_units_tokens(chain_tokens)
    trial = {
        "ok": False, "smiles": None,
        "codes": [c for c, _ in chain_tokens] + ["ClAc(cap)"],
        "names": [], "delta": None, "declared_mass": declared,
        "tail": [], "rotation": None, "variant": "clac-thioether",
        "unknown": unknown, "failure": None,
        "closure": "thioether",
    }
    if unknown:
        trial["failure"] = ("unresolved positions: "
                            + ", ".join(f"{u['code']!r}" for u in unknown[:6]))
        return trial
    smiles, error, _notes = _assemble_thioether(units, cap)
    return _mass_trial(trial, smiles, error, declared)


def _align_tokens_direct(tokens: list[tuple[str, dict]], notes: list[str],
                         declared: float, tail: list[str],
                         variant: str = "flags-direct") -> dict:
    """Head-to-tail assembly over flag-aware tokens (Me/d-prefixed rows of
    the new families). No SITE notes exist for these patents; the mass
    arbitration arbitrates."""
    units, unknown = _build_units_tokens(tokens)
    trial = {
        "ok": False, "smiles": None, "codes": [c for c, _ in tokens],
        "names": [], "delta": None, "declared_mass": declared,
        "tail": tail, "rotation": None, "variant": variant,
        "unknown": unknown, "failure": None,
    }
    if unknown:
        trial["failure"] = ("unresolved positions: "
                            + ", ".join(f"{u['code']!r}" for u in unknown[:6]))
        return trial
    smiles, error, _notes = _assemble(
        units, {"ring_size": len(units), "caps": [], "terminal": ""})
    return _mass_trial(trial, smiles, error, declared)


def align_row(row: dict, entry: dict | None, opts: dict | None = None) -> dict:
    """One TABLE 1 row -> reference record (ok, smiles, codes, names, delta).
    opts (from _style_opts) switches on the new-family notation paths;
    the US default path below is unchanged."""
    opts = opts or _style_opts("us", None)
    if opts.get("strict_named_chemistry"):
        from skills.markush_assemble.rebuild_row_candidates import rebuild_row
        candidate = rebuild_row(row)
        error = candidate.get('error')
        unknown = []
        if error and error.startswith('unsupported_named_residue:'):
            unknown = [{'code':error.split(':',1)[1], 'name':None}]
        consistent = candidate['status'] == 'chemical_consistency_pending_source_review'
        return {**candidate, 'ok':consistent,
                'smiles':candidate.get('smiles'), 'codes':candidate.get('codes',[]),
                'names':[], 'delta':candidate.get('mass_delta'),
                'declared_mass':row.get('exact_mass'), 'tail':[], 'rotation':None,
                'variant':'named-graph-compiler', 'unknown':unknown,
                'identity_verified':False, 'review_status':'pending_source_review',
                'failure':None if consistent else error or candidate['status']}
    linear = bool(opts.get("linear_notation"))
    if not row.get("sequence"):
        return {"ok": False, "smiles": None, "codes": [], "names": [],
                "delta": None, "declared_mass": row.get("exact_mass"),
                "tail": [], "rotation": None, "unknown": [],
                "failure": "row has no sequence string"}
    codes, tail = split_notation(row["sequence"], linear_notation=linear)
    codes = [c for tok in codes for c in split_merged(tok)]
    tokens = [normalize_token(c, opts) for c in codes]
    notes = filter_notes((entry or {}).get("notes", []))
    declared = row.get("exact_mass")

    record: dict = {
        "ok": False,
        "smiles": None,
        "codes": codes,
        "names": [],
        "delta": None,
        "declared_mass": declared,
        "tail": tail,
        "rotation": None,
        "unknown": [],
        "failure": None,
    }
    if not codes:
        record["failure"] = "no ring codes parsed from the notation"
        return record
    if declared is None:
        record["failure"] = "row has no declared exact_mass"
        return record

    # new-family readings first: they carry the flags the US machinery
    # cannot represent; the mass arbitration reviews every reading
    best_fail: dict | None = None
    if opts.get("thioether"):
        trial = _align_tokens_thioether(tokens, float(declared), opts)
        if trial is not None:
            if trial["ok"]:
                record.update(trial)
                return record
            best_fail = trial
    if any(flags for _, flags in tokens):
        trial = _align_tokens_direct(tokens, notes, float(declared), tail)
        if trial["ok"]:
            record.update(trial)
            return record
        if best_fail is None:
            best_fail = trial

    # keep any named failure from the new-family readings; the US DP path
    # below only fills best_fail when still unset
    vsets = _variant_sets(codes)
    for cand in candidate_alignments(codes, notes):
        pairs = cand["pairs"]
        for label, vmap in vsets:
            units, caps, unknown = build_units(codes, notes, pairs, vmap)
            names = [notes[nj] for _, nj in pairs]
            trial = {
                "ok": False, "smiles": None, "codes": codes, "names": names,
                "delta": None, "declared_mass": declared,
                "tail": tail, "rotation": cand["rotation"],
                "variant": label,
                "unknown": unknown, "failure": None,
            }
            if unknown:
                trial["failure"] = (
                    "unresolved positions: "
                    + ", ".join(f"{u['code']!r}({(u['name'] or '-')[:40]})"
                                for u in unknown[:6]))
            else:
                # cheap sum-mass prefilter before any RDKit work: an
                # assembled ring weighs sum(frag) - 2H per unit; a cap
                # adds its acyl (frag - H) and additionally acylates the
                # head amine one H below its ring reading (- H)
                expected = sum(exact_mass(u["entry"]["fragment_smiles"])
                               or 0.0 for u in units) \
                    - 2.01565 * len(units) \
                    + sum(exact_mass(c) or 0.0 for c in caps) \
                    - 2.01565 * len(caps)
                if abs(expected - float(declared)) <= MASS_TOLERANCE + 0.05:
                    smiles, error, _notes = _assemble(
                        units, {"ring_size": len(units),
                                "caps": [{"fragment_smiles": c} for c in caps],
                                "terminal": ""})
                    if not smiles:
                        trial["failure"] = f"assembly failed: {error}"
                    else:
                        mass = exact_mass(smiles)
                        if mass is None:
                            trial["failure"] = "assembled SMILES does not parse"
                        else:
                            delta = round(mass - float(declared), 4)
                            trial["delta"] = delta
                            if abs(delta) <= MASS_TOLERANCE:
                                trial["ok"] = True
                                trial["smiles"] = smiles
                                return trial
                            trial["failure"] = (
                                f"mass mismatch: computed {mass:.4f} vs declared "
                                f"{float(declared):.4f}")
                else:
                    trial["failure"] = (
                        f"mass prefilter: expected {expected:.4f} vs declared "
                        f"{float(declared):.4f}")
            # keep the highest-scoring failed candidate for the report
            if best_fail is None:
                best_fail = trial
    # Fallback: a corrupted listing (e.g. dozens of duplicated OCR notes)
    # can pair garbage SITE names over correct code entries, because the
    # DP prefers pairing a resolvable name to skipping it.  When every
    # name-first reading failed, retry the pure code-first reading (SITE
    # names ignored, lexicon code entries + family variants only).  The
    # mass arbitration still decides: only a +/-0.8 Da match is accepted.
    for label, vmap in vsets:
        units, caps, unknown = build_units(codes, [], [], vmap)
        trial = {
            "ok": False, "smiles": None, "codes": codes, "names": [],
            "delta": None, "declared_mass": declared,
            "tail": tail, "rotation": None,
            "variant": label + "+codefirst",
            "unknown": unknown, "failure": None,
        }
        if unknown:
            trial["failure"] = (
                "unresolved positions: "
                + ", ".join(f"{u['code']!r}({(u['name'] or '-')[:40]})"
                            for u in unknown[:6]))
        else:
            expected = sum(exact_mass(u["entry"]["fragment_smiles"])
                           or 0.0 for u in units) \
                - 2.01565 * len(units) \
                + sum(exact_mass(c) or 0.0 for c in caps) \
                - 2.01565 * len(caps)
            if abs(expected - float(declared)) <= MASS_TOLERANCE + 0.05:
                smiles, error, _notes = _assemble(
                    units, {"ring_size": len(units),
                            "caps": [{"fragment_smiles": c} for c in caps],
                            "terminal": ""})
                if not smiles:
                    trial["failure"] = f"assembly failed: {error}"
                else:
                    mass = exact_mass(smiles)
                    if mass is None:
                        trial["failure"] = "assembled SMILES does not parse"
                    else:
                        delta = round(mass - float(declared), 4)
                        trial["delta"] = delta
                        if abs(delta) <= MASS_TOLERANCE:
                            trial["ok"] = True
                            trial["smiles"] = smiles
                            return trial
                        trial["failure"] = (
                            f"mass mismatch: computed {mass:.4f} vs declared "
                            f"{float(declared):.4f}")
            else:
                trial["failure"] = (
                    f"mass prefilter: expected {expected:.4f} vs declared "
                    f"{float(declared):.4f}")
        if best_fail is None:
            best_fail = trial
    assert best_fail is not None
    return best_fail


def run(limit: int = 0, table_path: str | None = None,
        listing_path: str | None = None, result_path: str | None = None,
        style: str | None = None, opts: dict | None = None) -> dict:
    """All table rows -> result file + summary statistics.

    table_path/listing_path/result_path default to the US TABLE 1
    contract; `style` picks a new-family notation profile (kras/gpc3)
    and `opts` overrides individual style flags. The listing is optional
    for the new families (they carry no SITE-note sequence listings)."""
    rows = json.load(open(table_path or TABLE1_PATH, encoding="utf-8"))
    listing = {}
    lp = listing_path or LISTING_PATH
    if lp and os.path.exists(lp):
        listing = json.load(open(lp, encoding="utf-8"))
    order = sorted(rows, key=lambda k: int(k))
    if limit:
        order = order[:limit]

    align_opts = _style_opts(style, opts)
    # style-scoped lexicon overrides (h* family collisions across patents):
    # applied to the module lexicon for this run only; the data file on
    # disk is not modified
    for key, frag in (align_opts.get("lexicon_overrides") or {}).items():
        LEXICON[key.casefold()] = {"fragment_smiles": frag,
                                   "n_methyl": False,
                                   "note": "style override",
                                   "ring_nitrogen": False}
    result: dict[str, dict] = {}
    for sid in order:
        result[sid] = align_row(rows[sid], listing.get(sid), align_opts)

    ok_rows = [sid for sid in order if result[sid]["ok"]]
    with_notes = [sid for sid in order if result[sid]["names"]]
    fail_unknown = [sid for sid in order
                    if not result[sid]["ok"] and result[sid]["unknown"]]
    fail_assembly = [sid for sid in order if not result[sid]["ok"]
                     and not result[sid]["unknown"]
                     and str(result[sid].get("failure", "")).startswith("assembly")]
    fail_parse = [sid for sid in order if not result[sid]["ok"]
                  and result[sid]["delta"] is None and not result[sid]["unknown"]
                  and not str(result[sid].get("failure", "")).startswith("assembly")]
    fail_mass = [sid for sid in order
                 if not result[sid]["ok"] and result[sid]["delta"] is not None]
    # alignment rate over rows that carry SITE notes: fraction of notes
    # that the chosen reading actually paired to a code
    total_notes = sum(len(result[sid]["names"]) for sid in order)
    notes_seen = 0
    for sid in order:
        entry = listing.get(sid) or {}
        notes_seen += len(filter_notes(entry.get("notes", [])))
    variant_rows = [sid for sid in ok_rows
                    if result[sid].get("variant", "lexicon") not in ("lexicon", "named-graph-compiler")]

    unknown_names = Counter()
    unknown_codes = Counter()
    for sid in order:
        for u in result[sid]["unknown"]:
            if u["name"]:
                unknown_names[u["name"]] += 1
            else:
                unknown_codes[u["code"]] += 1

    summary = {
        "rows": len(order),
        "ok": len(ok_rows),
        "ok_seq_ids": [int(s) for s in ok_rows],
        "ok_via_variant": [int(s) for s in variant_rows],
        "rows_with_paired_names": len(with_notes),
        "alignment_rate": (round(total_notes / notes_seen, 4)
                           if notes_seen else None),
        "notes_total": notes_seen,
        "notes_paired": total_notes,
        "fail_mass": len(fail_mass),
        "fail_unknown_position": len(fail_unknown),
        "fail_assembly": len(fail_assembly),
        "fail_no_candidate_in_tolerance": len(fail_parse),
        "unknown_names_top20": unknown_names.most_common(20),
        "unknown_codes_top20": unknown_codes.most_common(20),
        "mass_tolerance": .025 if align_opts.get("strict_named_chemistry") else MASS_TOLERANCE,
        "candidate_only": True, "identity_verified_rows": 0,
    }
    json.dump({"summary": summary, "rows": result},
              open(result_path or RESULT_PATH, "w", encoding="utf-8"),
              indent=1)
    summary["style"] = align_opts
    return {"summary": summary, "rows": result}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("limit_pos", nargs="?", type=int, default=0,
                        help="row limit (US default run, positional compat)")
    parser.add_argument("--table", default=TABLE1_PATH)
    parser.add_argument("--listing", default=LISTING_PATH,
                        help="SITE-note listing json (absent file = none)")
    parser.add_argument("--result", default=RESULT_PATH)
    parser.add_argument("--style", default="us",
                        choices=sorted(STYLE_OPTIONS),
                        help="notation family profile")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    out = run(limit=args.limit or args.limit_pos, table_path=args.table,
              listing_path=args.listing, result_path=args.result,
              style=args.style)
    s = out["summary"]
    print(f"rows processed : {s['rows']}")
    print(f"ok (reference) : {s['ok']}")
    if s.get("alignment_rate") is not None:
        print(f"alignment rate : {s['alignment_rate']} "
              f"({s['notes_paired']}/{s['notes_total']} SITE names)")
    print(f"fail mass      : {s['fail_mass']}")
    print(f"fail unknown   : {s['fail_unknown_position']}")
    print(f"fail assembly  : {s['fail_assembly']}")
    print(f"fail no-match  : {s['fail_no_candidate_in_tolerance']}")
    print(f"result written : {args.result}")
    return 0 if s["ok"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
