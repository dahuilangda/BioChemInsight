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
