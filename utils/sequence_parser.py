"""Sequence extraction: amino-acid/nucleotide sequences from OCR page
text, with peptide→SMILES support.

Patent context: cyclic-peptide patents (scanned PDFs with no text layer)
carry sequences as OCR'd tables and ST.25-style listings.

  extract_sequences_from_text(text, page_num) -> list[SequenceRecord]
    - one-letter amino-acid runs (>= MIN_AA_LEN, line-broken like ST.25)
    - three-letter code tables (Ala/Arg/... -> one letter)
    - SEQ ID NO: N association (nearest preceding label)
    - nucleotide runs classified separately (DNA/RNA)
    - cyclic annotation detection (cyclo/cyclic/head-to-tail)
  sequence_to_smiles(seq, cyclic=True) -> canonical SMILES
    - standard 20 residues, L-configuration, head-to-tail macrocycle
    - enables the cross-check: sequence-derived structure vs image-decoded
      structure
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_AA_LEN = 5

THREE_TO_ONE = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q',
    'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K',
    'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W',
    'Tyr': 'Y', 'Val': 'V', 'Sec': 'U', 'Pyl': 'O', 'Asx': 'B', 'Glx': 'Z',
    'Xle': 'J', 'Xaa': 'X',
}

# L-amino-acid side chains (as SMILES substituent starting at Cα attachment).
_SIDECHAIN = {
    'A': 'C', 'R': 'CCCNC(=[NH2+])N', 'N': 'CC(=O)N', 'D': 'CC(=O)O',
    'C': 'CS', 'E': 'CCC(=O)O', 'Q': 'CCC(=O)N', 'G': '', 'H': 'Cc1cnc[nH]1',
    'I': 'C(C)CC', 'L': 'CC(C)C', 'K': 'CCCCN', 'M': 'CCSC', 'F': 'Cc1ccccc1',
    'P': None, 'S': 'CO', 'T': 'C(O)C', 'W': 'Cc1c[nH]c2ccccc12',
    'Y': 'Cc1ccc(O)cc1', 'V': 'C(C)C', 'U': '[SeH]', 'O': 'CCCN',
}

_SEQ_ID_RE = re.compile(r"SEQ\s+ID\s*NO[:.]?\s*([0-9]{1,4})", re.IGNORECASE)
_CYCLIC_RE = re.compile(
    r"\bcycl(?:o|ic)\b|\bhead[- ]to[- ]tail\b|\bmacrocycl\w*|\bcyclo\b", re.IGNORECASE)
# Compressed English headers that OCR flattens into letter-runs of valid AA
# codes ("SEQ ID NO" -> SEQIDNO). Rejected before any length/label logic.
_AA_STOPWORDS = {
    'SEQIDNO', 'SEQIDNOS', 'SEQUENCE', 'SEQUENCES', 'LISTING', 'AMINOACID',
    'AMINOACIDS', 'POLYPEPTIDE', 'POLYPEPTIDES', 'POLYNUCLEOTIDE', 'PEPTIDE',
    'PEPTIDES', 'IDENTIF', 'IDENTIFY', 'MODIFIED', 'MODIFICATION', 'SUBSTITUT',
    'DETAILED', 'DESCRIPTION', 'EMBOSS', 'PATENTIN', 'NASDAQ', 'WORD',
    'DOCUMENT', 'VERSION', 'CREATION', 'DATABASE', 'QUERY', 'SBJCT', 'SCORE',
}
_AA_RUN_RE = re.compile(
    r"(?<![A-Za-z])([ACDEFGHIKLMNPQRSTVWY][ACDEFGHIKLMNPQRSTVWYBZXJUO\s\-]{%d,})" % (MIN_AA_LEN - 1))
_NUC_RUN_RE = re.compile(r"(?<![A-Za-z])([ATGCU][ATGCU\s\-]{15,})")
@dataclass
class SequenceRecord:
    sequence: str
    kind: str = 'aa'                 # 'aa' | 'nuc'
    seq_id: str = ''
    page_num: int = 0
    cyclic: bool = False
    confidence: float = 1.0
    source_span: str = field(default='', repr=False)

    def asdict(self) -> dict:
        return {
            'sequence': self.sequence, 'kind': self.kind, 'seq_id': self.seq_id,
            'page_num': self.page_num, 'cyclic': self.cyclic,
            'confidence': self.confidence,
        }


def _clean_candidate(raw: str) -> str:
    """Strip whitespace/dashes/positions digits from an OCR'd sequence run."""
    text = re.sub(r"[^A-Za-z]", "", raw)
    return text.upper()


def extract_sequences_from_text(text: str, page_num: int = 0) -> list[SequenceRecord]:
    """Extract sequence records from one page of OCR/markdown text.

    Combines one-letter runs (label-anchored first), three-letter residue
    names, and nucleotide runs. Compressed English headers that OCR flattens
    into AA-code letter-runs (SEQ ID NO → SEQIDNO) are stripped as prefixes
    before the orphan-length gate.
    """
    if not text:
        return []
    records: list[SequenceRecord] = []
    seen: set[str] = set()
    cyclic_hint = bool(_CYCLIC_RE.search(text or ''))

    def strip_stopword_prefix(seq: str) -> str:
        changed = True
        while changed and len(seq) >= MIN_AA_LEN:
            changed = False
            for word in _AA_STOPWORDS:
                if seq.startswith(word) and len(seq) > len(word):
                    seq = seq[len(word):]
                    changed = True
                    break
        return seq

    def seq_id_before(start: int) -> str:
        label_span = text[max(0, start - 200):start]
        found = None
        for found in _SEQ_ID_RE.finditer(label_span):
            pass
        return found.group(1) if found else ''

    for match in _AA_RUN_RE.finditer(text):
        seq = strip_stopword_prefix(_clean_candidate(match.group(1)))
        if len(seq) < MIN_AA_LEN or seq in _AA_STOPWORDS:
            continue
        seq_id = seq_id_before(match.start())
        # Orphan runs (no SEQ ID label nearby) carry the prose-false-positive
        # risk; hold them to a higher bar than label-anchored runs.
        if not seq_id and len(seq) < 6:
            continue
        key = (seq, seq_id)
        if key in seen:
            continue
        seen.add(key)
        records.append(SequenceRecord(
            sequence=seq, kind='aa', seq_id=seq_id, page_num=page_num,
            cyclic=cyclic_hint, confidence=0.9 if seq_id else 0.6,
            source_span=match.group(1)[:80]))

    for record in extract_from_three_letter(text, page_num=page_num):
        key = (record.sequence, '')
        if key in seen:
            continue
        seen.add(key)
        records.append(record)

    for match in _NUC_RUN_RE.finditer(text):
        seq = _clean_candidate(match.group(1))
        if len(seq) < 16:
            continue
        key = (seq, 'nuc')
        if key in seen:
            continue
        seen.add(key)
        records.append(SequenceRecord(
            sequence=seq, kind='nuc', page_num=page_num, confidence=0.7,
            source_span=match.group(1)[:80]))

    return records


def extract_from_three_letter(text: str, page_num: int = 0) -> list[SequenceRecord]:
    """Extract sequences written in three-letter code (Ala-Arg-...).

    Residues must be separated ONLY by whitespace/dash/comma — any other
    letter material between names (prose, labels, table headers) terminates
    the chain, so 'Cys-Ala-Lys-Gly ... text ... Gly Gly Ser' yields two
    records, not one merged 12-mer.
    """
    records: list[SequenceRecord] = []
    if not text:
        return records
    names = "|".join(THREE_TO_ONE)
    unit = re.compile(rf"\b({names})\b((?:\s*[-–,]\s*|\s+)({names})\b)*",
                      re.IGNORECASE)
    for match in unit.finditer(text):
        raw_names = re.findall(rf"\b({names})\b", match.group(0), re.IGNORECASE)
        seq = ''.join(THREE_TO_ONE[name.title()] for name in raw_names)
        if len(seq) < MIN_AA_LEN:
            continue
        label_span = text[max(0, match.start() - 200):match.start()]
        found = None
        for found in _SEQ_ID_RE.finditer(label_span):
            pass
        seq_id = found.group(1) if found else ''
        records.append(SequenceRecord(
            sequence=seq, kind='aa', seq_id=seq_id, page_num=page_num,
            confidence=0.9 if seq_id else 0.5))
    return records


def sequence_to_smiles(seq: str, cyclic: bool = True) -> str | None:
    """One-letter peptide -> canonical SMILES (L-residues, optional
    head-to-tail macrocycle). Non-standard letters (B/Z/J/X/O) return None.

    Proline fragments use ring digit 1 (internal pyrrolidine); the macrocycle
    closes on ring digit 9, so digit clashes are avoided by construction.
    """
    seq = re.sub(r"[^A-Za-z]", '', str(seq or '')).upper()
    if not seq or any(ch not in _SIDECHAIN for ch in seq):
        return None
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")

        def residue(ch: str) -> str:
            if ch == 'P':
                return "N1CCC[C@@H]1C(=O)"   # Cα is a pyrrolidine ring carbon
            side = _SIDECHAIN[ch]
            return f"N[C@@H]({side})C(=O)" if side else "NCC(=O)"

        if cyclic:
            if seq[0] == 'P':
                return None                  # head-proline macrocycle: skip
            side0 = _SIDECHAIN[seq[0]]
            head = f"N9[C@@H]({side0})C(=O)" if side0 else "N9CC(=O)"
            smiles = head + ''.join(residue(ch) for ch in seq[1:]) + '9'
        else:
            smiles = ''.join(residue(ch) for ch in seq) + 'O'
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


# Modified-residue grammar for patent cyclopeptides (the residues that
# one-letter sequence_to_smiles cannot express). Fragment values attach at
# the backbone Cα, same convention as _SIDECHAIN.
# multi-alpha backbone units observed in real patent figures (each spans
# one list position and chains like a residue: starts with the amine N,
# ends with C(=O)); used verbatim before the single-side-chain path
EXOTIC_RESIDUE_UNITS = {
    # lanthionine-type thioether bridge, alpha-carboxamide + ethyl (SEQ60)
    'Sbridge_AsnEt': 'N[C@@H](C(N)=O)CS[C@@H](CC)C(=O)',
    # 6-membered N,O-acetal pseudoproline, gem-dimethyl, N-methyl (SEQ60)
    'Pseudoproline': 'N(C)CC(C)(C)OC[C@@H]C(=O)',
    # N-alpha-methyl-glutamic acid (SEQ60)
    'NMeGlu': 'N(C)[C@@H](CCC(O)=O)C(=O)',
    # tyrosine
    'Tyr': 'Cc1ccc(O)cc1',
    # SEQ77-specific residues (WO2024163691 SEQ77 gold enumeration)
    '7azaTrp': 'N[C@@H](Cc2cnc3[nH]ccc3c2)C(=O)',   # 7-azaindolylalanine
    'HypOCM': 'N1CC[C@@H](OCC(O)=O)C1C(=O)',       # O-carboxymethyl-Hyp
    'MeAPeC': 'N[C@@H]1CC[C@H](C)C1C(=O)',         # methyl-ACPC (5-ring, C-Me)
    # SEQ52-specific units (WO2024163691 SEQ52 gold enumeration)
    'Slinked_Asn': 'N[C@@H](C(N)=O)CS',      # backbone carboxyl replaced by CH2-S
    'Thioglycolyl': 'CC(=O)',                # mercaptoacetyl: CH2-C(=O) after S
    'Pipz2COOH': 'N1C(C(=O))CNCC1',          # piperazine-2-carboxylic acid
    'TrpN1CH2COOH': 'N[C@@H](Cc2cn(CC(O)=O)c3ccccc23)C(=O)',
    'aMeTrp7OMe': 'N[C@](C)(c2c[nH]c3cc(OC)ccc23)C(=O)',
}

# three-letter aliases for the standard residues (agent enumerations use them)
EXOTIC_RESIDUE_ALIASES = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q',
    'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K',
    'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W',
    'Tyr': 'Y', 'Val': 'V',
}

EXOTIC_RESIDUE_FRAGMENTS = {
    # standard
    'A': 'C', 'R': 'CCCNC(=N)N', 'N': 'CC(N)=O', 'D': 'CC(O)=O',
    'E': 'CCC(O)=O', 'Q': 'CCC(N)=O', 'G': '', 'H': 'Cc1cnc[nH]1',
    'I': '[C@H](C)CC', 'L': 'CC(C)C', 'K': 'CCCCN', 'M': 'CCSC',
    'F': 'Cc1ccccc1', 'S': 'CO', 'C': 'CS', 'T': '[C@H](O)C',
    'W': 'Cc1c[nH]c2ccccc12', 'Y': 'Cc1ccc(O)cc1', 'V': 'C(C)C',
    # modified / non-standard (patent cyclopeptide canon)
    'NMeG': 'SARCOSINE',         # backbone N-Me, no side chain
    'Nle': 'CCCC',               # norleucine
    '4Pal': 'Cc1ccncc1',         # 4-pyridylalanine
    'Phe4CONH2': 'Cc1ccc(C(N)=O)cc1',
    'Phe4COOH': 'Cc1ccc(C(O)=O)cc1',
    'Trp7Cl': 'Cc1c[nH]c2c(Cl)cccc12',
    'Trp7OMe': 'Cc1c[nH]c2c(OC)cccc12',
    'bMeTrp7Cl': '[C@H](C)c1c[nH]c2c(Cl)cccc12',
    'bMeTrp7OMe': '[C@H](C)c1c[nH]c2c(OC)cccc12',
    'aMeS': ('C', 'CO'),         # alpha-methylserine: two Calpha substituents
    'Hyp': None,                 # ring-N residue: handled as its own ring
    'PipOH': None,
    'ACPC': None,
    '5CNW': 'Cc1c[nH]c2ccc(C#N)cc12',
    'NCMW': 'Cc1cn(CC(O)=O)c2ccccc12',
    'Oxa': 'CC1C(=O)Nc2ccccc12',
}

# Ring-N residues carry their own secondary amine ring; the macrocycle
# closure digit must not be attached to their nitrogen.
_EXOTIC_RING_RESIDUES = {
    'P': 'N1CCC[C@@H]1C(=O)',   # proline: backbone amine is the ring N
    'Hyp': 'N1CC[C@@H](O)C1C(=O)',
    'PipOH': 'N1CC[C@H](O)C[C@@H]1C(=O)',
    'ACPC': 'N[C@H]1CC[C@H](C1)C(=O)',
}


def _unit_fragment(code: str, position: int, n_methyl, head: bool = False):
    """One macrocycle unit: returns the fragment starting with its amine
    (N9-tagged for the head position). Units start with the amine and end
    with C(=O), so they chain head-to-tail unchanged."""
    import re as _re
    if code in ('NMeG', 'Sar'):
        frag = 'N(C)CC(=O)'          # sarcosine: backbone N-Me, no side chain
    else:
        unit = EXOTIC_RESIDUE_UNITS.get(code)
        if unit:
            frag = unit
        elif code in _EXOTIC_RING_RESIDUES:
            frag = _EXOTIC_RING_RESIDUES[code]
        else:
            spec = EXOTIC_RESIDUE_FRAGMENTS.get(code)
            if spec is None:
                return None
            amine_methyl = '(C)' if position in n_methyl else ''
            if isinstance(spec, tuple):
                s1, s2 = spec
                frag = f"N{amine_methyl}[C@]({s1})({s2})C(=O)"
            else:
                frag = f"N{amine_methyl}[C@@H]({spec})C(=O)" if spec else f"N{amine_methyl}CC(=O)"
    if head:
        frag = _re.sub(r'^N', 'N9', frag, count=1)
    return frag


def build_cyclopeptide_smiles(residues, n_methyl_positions=None):
    """Deterministic head-to-tail cyclic peptide from residue codes.

    Builds via the grammar N9...tail 9 pattern with globally unique ring
    digits (each fragment's aromatic ring digits are remapped to a fresh
    namespace). Accepts the modified-residue vocabulary (4Pal, Trp7Cl, Nle,
    aMeS, NMeG, ACPC, PipOH, Hyp, Tyr, NMeGlu, Sbridge_AsnEt,
    Pseudoproline, 7azaTrp, HypOCM, MeAPeC). N-methylation applies at the
    given 0-based positions on plain residues.
    Returns canonical SMILES or None on assembly failure."""
    import re as _re
    codes = [str(code).strip() for code in residues or [] if str(code).strip()]
    codes = [EXOTIC_RESIDUE_ALIASES.get(c, c) for c in codes]
    if len(codes) < 3:
        return None
    n_methyl = set(n_methyl_positions or [])
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return None

    # build body fragments with original digits, then remap each fragment's
    # ring digits to a globally unique namespace
    body_frags = []
    for pos, code in enumerate(codes[1:], start=1):
        frag = _unit_fragment(code, pos, n_methyl)
        if frag is None:
            return None
        body_frags.append(frag)

    # head fragment (amine N becomes N9 for head-to-tail closure)
    head_frag = _unit_fragment(codes[0], 0, n_methyl, head=True)
    if head_frag is None:
        return None

    # global digit remap: assign fresh digits to each fragment's ring
    # closures (the head fragment keeps digit 9 for macrocycle + gets new
    # digits for its aromatic rings)
    all_text = head_frag + ''.join(body_frags)
    used = {9}
    remapped_parts = [head_frag]
    for bf in body_frags:
        digits_in = sorted(set(int(d) for d in _re.findall(r'\d', bf)))
        mapping = {}
        for d in digits_in:
            while d in used or d in mapping.values():
                d += 1
            mapping[d] = d
            used.add(d)
        remapped = bf
        for old, new in sorted(mapping.items()):
            remapped = remapped.replace(str(old), str(new))
        remapped_parts.append(remapped)

    smiles = head_frag + ''.join(remapped_parts[1:]) + '9'
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def parse_cyclopeptide_notation(text: str) -> list[str]:
    """'cyclo(Nle-4Pal-PipOH-W)' -> ['Nle', '4Pal', 'PipOH', 'W'].

    Splits the head-to-tail notation on dashes, keeping parenthesised
    groups (side chains) as single tokens. Also accepts a bare dash- or
    comma-separated list without the cyclo() wrapper."""
    text = str(text or '').strip()
    wrapper = re.match(r'^cyclo\s*\((.*)\)$', text, re.IGNORECASE | re.DOTALL)
    if wrapper:
        text = wrapper.group(1)
    tokens = []
    depth = 0
    current = ''
    for ch in text:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
        elif ch in '-,' and depth == 0:
            if current.strip():
                tokens.append(current.strip())
            current = ''
        else:
            current += ch
    if current.strip():
        tokens.append(current.strip())
    return tokens
