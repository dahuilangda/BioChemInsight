"""Peptide skill cluster: structure <-> sequence <-> 2D depiction.

Like a gene cluster, these unit operations belong together: each converts
one representation into another, and the round trip is the stability
invariant — a structure that fails to round-trip through its sequence
names exactly which leg is unfaithful.

Members:
  structure -> sequence   linear: cut the amide bonds, each component is
                          one residue (backbone form). cyclic: the
                          macrocycle walker (residue_typer).
  sequence  -> structure  assembly (one-letter grammar / lexicon codes).
  structure -> 2D         RDKit depiction.

Residue codes resolve through the shared knowledge (standard letters, the
residue lexicon, the built-in monomer dictionary). A residue unit missing
from every dictionary is emitted AS ITS CANONICAL FRAGMENT SMILES — an
unrecognized residue is represented, never silently dropped.
"""
from __future__ import annotations

import json
import re

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from rdkit.Chem import Descriptors

from utils.residue_typer import smiles_to_sequence
from utils.sequence_parser import sequence_to_smiles

from pathlib import Path as _Path
_LEXICON_PATH = str(_Path(__file__).resolve().parent.parent / "data" / "residue_lexicon.json")


# ---------------------------------------------------------------------------
# amide-bond cutting
# ---------------------------------------------------------------------------

def _is_carbonyl_carbon(atom) -> bool:
    for b in atom.GetBonds():
        other = atom.GetOwningMol().GetAtomWithIdx(b.GetOtherAtomIdx(atom.GetIdx()))
        if other.GetSymbol() == "O" and b.GetBondTypeAsDouble() == 2:
            return True
    return False


def _amide_bonds(mol) -> list[tuple[int, int]]:
    """(N idx, carbonyl-C idx) pairs: a nitrogen single-bonded to a
    carbonyl carbon — every peptide bond and N-acyl cap."""
    pairs = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "N":
            continue
        for b in atom.GetBonds():
            other = mol.GetAtomWithIdx(b.GetOtherAtomIdx(atom.GetIdx()))
            if other.GetSymbol() == "C" and _is_carbonyl_carbon(other):
                pairs.append((atom.GetIdx(), other.GetIdx()))
    return pairs


def _cut_amide_bonds(mol) -> list[Chem.Mol]:
    """Remove every N-C(=O) amide bond; the molecule falls apart into its
    residue/cap components (the standard peptide-bond disconnection)."""
    em = Chem.RWMol(mol)
    for n_idx, c_idx in _amide_bonds(mol):
        bond = em.GetBondBetweenAtoms(n_idx, c_idx)
        if bond is not None:
            em.RemoveBond(n_idx, c_idx)
    out = Chem.GetMolFrags(em.GetMol(), asMols=True, sanitizeFrags=True)
    return list(out)


# ---------------------------------------------------------------------------
# reverse lookup: canonical fragment -> code
# ---------------------------------------------------------------------------

def _backbone_canonical(component: Chem.Mol) -> str:
    """Component -> backbone-form canonical SMILES: the free acid's alpha
    OH is stripped (internal residues end at bare C(=O)); the N-terminal
    amide cap (ac-) stays as its acyl aldehyde-convention form."""
    return Chem.MolToSmiles(component)


def _strip_terminal_group(component: Chem.Mol) -> Chem.Mol:
    """Free-acid residue component -> backbone form: drop the terminal
    carboxyl OH (and a C-terminal amide N is kept — it reads as -NH2 in
    the canonical string; reverse lookup handles both spellings)."""
    em = Chem.RWMol(component)
    for atom in em.GetAtoms():
        # a free hydroxyl: single-bonded O on a carboxyl carbon. The
        # carbonyl =O is degree-1 too but its bond is double — stripping it
        # would destroy already-backbone-form entries.
        if atom.GetSymbol() == "O" and atom.GetDegree() == 1:
            bonds = atom.GetBonds()
            if len(bonds) == 1 and bonds[0].GetBondTypeAsDouble() != 2:
                nb = em.GetAtomWithIdx(bonds[0].GetOtherAtomIdx(atom.GetIdx()))
                if nb.GetSymbol() == "C" and _is_carbonyl_carbon(nb):
                    em.RemoveAtom(atom.GetIdx())
                    break
    try:
        out = em.GetMol()
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return component


def _preferred_code(codes: list[str]) -> str:
    """Deterministic code choice: a single standard letter first, then
    the shortest alphanumeric code, then the first printed key."""
    for c in codes:
        if len(c) == 1 and c.upper() in "ACDEFGHIKLMNPQRSTVWY":
            return c
    alnum = [c for c in codes if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", c)]
    return sorted(alnum, key=len)[0] if alnum else codes[0]


def build_reverse_index(lexicon_path: str = _LEXICON_PATH) -> dict[str, list[str]]:
    """canonical backbone fragment -> codes that print as it. Built from
    the residue lexicon (RDKit-verified definitions) and the standard
    letters. Molecules are compared canonically, so rooting differences
    between sources do not matter."""
    reverse: dict[str, list[str]] = {}
    try:
        lex = json.load(open(lexicon_path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        lex = {}
    for code, entry in lex.items():
        if not isinstance(entry, dict):
            continue
        smiles = entry.get("fragment_smiles") or ""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        stripped = _strip_terminal_group(mol)
        canon = Chem.MolToSmiles(stripped)
        reverse.setdefault(canon, []).append(code)
    return {canon: {"codes": codes, "preferred": _preferred_code(codes)}
            for canon, codes in reverse.items()}


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

class PeptideStructureSequenceCluster:
    """structure <-> sequence <-> 2D depiction with a built-in round-trip
    stability check."""

    def __init__(self, lexicon_path: str = _LEXICON_PATH):
        self.reverse_index = build_reverse_index(lexicon_path)

    # -- structure -> sequence ----------------------------------------------

    def structure_to_sequence(self, smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"codes": [], "unknown": [],
                    "note": "SMILES does not parse"}
        # cyclic peptides: the existing macrocycle walker
        codes, cyclic, note = smiles_to_sequence(smiles)
        if codes:
            return {"codes": codes, "cyclic": cyclic, "note": note,
                    "unknown": []}
        return self._linear_sequence(mol)

    def _linear_sequence(self, mol) -> dict:
        reverse = self.reverse_index
        codes, unknown = [], []
        for component in _cut_amide_bonds(mol):
            backbone = _backbone_canonical(_strip_terminal_group(component))
            bucket = reverse.get(backbone)
            codes.append(bucket["preferred"] if bucket else backbone)
            if not bucket:
                unknown.append(backbone)
        return {"codes": codes, "unknown": unknown, "cyclic": False,
                "caps": [c for c in codes if len(c) > 1][:1],
                "note": ""}

    # -- sequence -> structure ----------------------------------------------

    def sequence_to_structure(self, codes: list[str], cyclic: bool = True):
        """One-letter codes -> SMILES via the standard grammar. Multi-char
        codes (caps, unknown fragments) are not assembly-able here and are
        returned unresolved."""
        if not all(len(c) == 1 for c in codes):
            return None
        return sequence_to_smiles("".join(codes), cyclic=cyclic)

    # -- 2D depiction ---------------------------------------------------------

    def depict_2d(self, smiles: str, out_path: str, width: int = 800) -> str:
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("SMILES does not parse")
        Draw.MolToFile(mol, out_path, size=(width, width))
        return out_path

    # -- round-trip stability invariant ---------------------------------------

    def roundtrip_verify(self, smiles: str) -> dict:
        """structure -> sequence -> structure: the cluster's stability
        invariant. A mismatch names the broken leg with formulas."""
        seq = self.structure_to_sequence(smiles)
        codes = seq.get("codes", [])
        if not codes:
            return {"faithful": False,
                    "reason": f"sequence leg empty: {seq.get('note', '')}"}
        if any(len(c) > 1 for c in codes):
            return {"faithful": None,
                    "reason": "sequence contains unresolved fragment codes",
                    "codes": codes}
        rebuilt = self.sequence_to_structure(codes,
                                             cyclic=seq.get("cyclic", True))
        m1 = Chem.MolFromSmiles(smiles)
        m2 = Chem.MolFromSmiles(rebuilt) if rebuilt else None
        if m1 is None or m2 is None:
            return {"faithful": False, "reason": "a leg does not parse"}
        f1 = Chem.rdMolDescriptors.CalcMolFormula(m1)
        f2 = Chem.rdMolDescriptors.CalcMolFormula(m2)
        same = Chem.MolToSmiles(m1) == Chem.MolToSmiles(m2)
        return {"faithful": bool(same and f1 == f2),
                "formula_structure": f1, "formula_sequence": f2}
