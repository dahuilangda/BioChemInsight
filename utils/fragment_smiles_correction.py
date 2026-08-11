"""VLM-assisted fragment SMILES correction.

When the vision model reads a fragment SMILES directly from the patent image,
this module decides whether to accept the VLM's reading over MolNexTR's
decoded SMILES, based on RDKit-computed similarity and atom-count difference.

Decision matrix (no fallback, no hard-coded examples):

- VLM SMILES unparseable / empty          -> keep MolNexTR
- Tanimoto >= 0.95 and |dummy-stripped atom count diff| <= 1
                                           -> accept VLM (simple correction)
- Tanimoto < 0.5                           -> reject (too divergent)
- otherwise                                -> keep MolNexTR (uncertain)
"""
from __future__ import annotations

from typing import Any

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    Chem = None
    AllChem = None
    DataStructs = None
    _HAS_RDKIT = False


def _strip_dummy(mol) -> Chem.Mol | None:
    """Remove dummy atoms (atomic number 0) from *mol*, returning None on
    failure or if the stripped molecule is invalid / disconnected."""
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    dummy_ids = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(dummy_ids, reverse=True):
        rw.RemoveAtom(idx)
    stripped = rw.GetMol()
    try:
        Chem.SanitizeMol(stripped)
    except Exception:
        return None
    if len(Chem.GetMolFrags(stripped)) != 1:
        return None
    return stripped


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not smiles or Chem is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None
    return mol


def _tanimoto(mol_a, mol_b) -> float:
    if mol_a is None or mol_b is None or AllChem is None or DataStructs is None:
        return 0.0
    try:
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=1024)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=1024)
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return 0.0


def _atom_count(mol) -> int:
    if mol is None:
        return 0
    return mol.GetNumAtoms()


def _validate_fragment_smiles(smiles: str) -> bool:
    """A usable fragment SMILES must sanitize, have exactly one dummy atom of
    degree 1, and be a single connected fragment."""
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return False
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return False
    if dummies[0].GetDegree() != 1:
        return False
    return len(Chem.GetMolFrags(mol)) == 1


def _canonical_smiles(mol) -> str | None:
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except Exception:
        return None


def _ring_signature(mol) -> tuple | None:
    """Sorted tuple of ring sizes — a compact signature of the ring system.
    None if the molecule has no rings or the ring info is unavailable."""
    if mol is None:
        return None
    try:
        ri = mol.GetRingInfo()
        if not ri or ri.NumRings() == 0:
            return None
        return tuple(sorted(len(r) for r in ri.AtomRings()))
    except Exception:
        return None


def _is_isomorphic(mol_a, mol_b) -> bool:
    if mol_a is None or mol_b is None:
        return False
    canon_a = _canonical_smiles(mol_a)
    canon_b = _canonical_smiles(mol_b)
    return canon_a is not None and canon_a == canon_b


def _mcs_coverage(mol_a, mol_b) -> float:
    """Fraction of atoms covered by the maximum common substructure,
    a more robust similarity than Tanimoto for direction-reversed SMILES."""
    if mol_a is None or mol_b is None or Chem is None:
        return 0.0
    try:
        from rdkit.Chem import rdFMCS
        mcs = rdFMCS.FindMCS([mol_a, mol_b])
        num = mcs.numAtoms
        denom = max(mol_a.GetNumAtoms(), mol_b.GetNumAtoms())
        return num / denom if denom else 0.0
    except Exception:
        return 0.0


def _looks_truncated(smiles: str) -> bool:
    """MolNexTR collapsed to a trivial fragment (e.g. *CC) — a sign the
    decode failed and the VLM reading should be trusted."""
    stripped = smiles.replace('*', '').strip()
    if not stripped:
        return False
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return False
    non_dummy = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 0]
    return len(non_dummy) <= 2


def correct_fragment_smiles(
    molnextr_smiles: str,
    vlm_smiles: str,
    molnextr_quality_issues: str = '',
) -> dict[str, Any]:
    """Compare MolNexTR's decoded SMILES with the VLM's reading.

    Decision logic (no fallback):
    - VLM unparseable                    -> keep MolNexTR
    - Same molecule (isomorphic)         -> keep MolNexTR (no correction)
    - MolNexTR truncated (*CC or <=2 atoms) -> accept VLM if it is a valid
      fragment and not absurdly larger (atom_diff <= 4)
    - MolNexTR has quality issues        -> accept VLM only when very close
      (mcs >= 0.9 and |atom_diff| <= 1)
    - Very divergent (mcs < 0.5)         -> reject
    - Otherwise                          -> keep MolNexTR

    Returns a dict with:
      - status: 'no_change' | 'accepted' | 'rejected' | 'keep_molnextr'
      - corrected_smiles: the SMILES to use (MolNexTR's unless accepted)
      - similarity: Morgan Tanimoto between the two (0.0 on parse failure)
      - mcs_coverage: maximum-common-substructure atom coverage
      - atom_diff: dummy-stripped atom count difference (vlm - molnextr)
      - reason: short human-readable decision reason
    """
    if not _HAS_RDKIT:
        return {
            'status': 'keep_molnextr',
            'corrected_smiles': molnextr_smiles,
            'similarity': 0.0,
            'mcs_coverage': 0.0,
            'atom_diff': 0,
            'reason': 'rdkit_unavailable',
        }

    mx = _mol_from_smiles(molnextr_smiles)
    vlm = _mol_from_smiles(vlm_smiles)

    if vlm is None or not vlm_smiles.strip():
        return {
            'status': 'keep_molnextr',
            'corrected_smiles': molnextr_smiles,
            'similarity': 0.0,
            'mcs_coverage': 0.0,
            'atom_diff': 0,
            'reason': 'vlm_smiles_unparseable',
        }

    # Compare dummy-stripped structures
    mx_stripped = _strip_dummy(mx) if mx is not None else None
    vlm_stripped = _strip_dummy(vlm)

    if mx_stripped is None or vlm_stripped is None:
        return {
            'status': 'keep_molnextr',
            'corrected_smiles': molnextr_smiles,
            'similarity': 0.0,
            'mcs_coverage': 0.0,
            'atom_diff': 0,
            'reason': 'stripped_parse_failed',
        }

    similarity = _tanimoto(mx_stripped, vlm_stripped)
    mcs_coverage = _mcs_coverage(mx_stripped, vlm_stripped)
    atom_diff = _atom_count(vlm_stripped) - _atom_count(mx_stripped)
    isomorphic = _is_isomorphic(mx_stripped, vlm_stripped)
    vlm_valid = _validate_fragment_smiles(vlm_smiles)
    has_quality_issues = bool(molnextr_quality_issues and molnextr_quality_issues.strip())
    mx_truncated = _looks_truncated(molnextr_smiles)
    # Same ring system (same ring sizes) in both readings is a strong signal
    # that the only difference is on a side chain / linker — a simple error.
    ring_match = _ring_signature(mx_stripped) == _ring_signature(vlm_stripped)

    # Same molecule (even if written differently) -> no correction needed
    if isomorphic:
        return {
            'status': 'no_change',
            'corrected_smiles': molnextr_smiles,
            'similarity': similarity,
            'mcs_coverage': mcs_coverage,
            'atom_diff': 0,
            'reason': 'isomorphic',
        }

    # MolNexTR collapsed to a trivial fragment -> the decode failed; the VLM
    # reading is the only evidence.  Accept it whenever it is a valid fragment
    # and is not absurdly large (<= 12 non-dummy atoms is a sane fragment cap).
    if mx_truncated:
        if vlm_valid and atom_diff <= 12:
            return {
                'status': 'accepted',
                'corrected_smiles': vlm_smiles,
                'similarity': similarity,
                'mcs_coverage': mcs_coverage,
                'atom_diff': atom_diff,
                'reason': 'molnextr_truncated_vlm_recovered',
            }
        return {
            'status': 'keep_molnextr',
            'corrected_smiles': molnextr_smiles,
            'similarity': similarity,
            'mcs_coverage': mcs_coverage,
            'atom_diff': atom_diff,
            'reason': 'molnextr_truncated_vlm_invalid',
        }

    # MolNexTR reported quality issues -> accept VLM when structurally close
    # and the atom count difference is small.
    if has_quality_issues:
        if mcs_coverage >= 0.7 and abs(atom_diff) <= 2 and vlm_valid:
            return {
                'status': 'accepted',
                'corrected_smiles': vlm_smiles,
                'similarity': similarity,
                'mcs_coverage': mcs_coverage,
                'atom_diff': atom_diff,
                'reason': 'quality_issue_simple_correction',
            }
        return {
            'status': 'keep_molnextr',
            'corrected_smiles': molnextr_smiles,
            'similarity': similarity,
            'mcs_coverage': mcs_coverage,
            'atom_diff': atom_diff,
            'reason': 'quality_issue_uncertain',
        }

    # Highly divergent -> reject (do not assemble with either)
    if mcs_coverage < 0.5:
        return {
            'status': 'rejected',
            'corrected_smiles': molnextr_smiles,
            'similarity': similarity,
            'mcs_coverage': mcs_coverage,
            'atom_diff': atom_diff,
            'reason': 'vlm_and_molnextr_diverge',
        }

    # Healthy MolNexTR decode: accept a correction only when the structures
    # are nearly identical (mcs >= 0.95) and differ by at most one atom, OR
    # when they share the same ring system and differ by at most one atom
    # (a simple side-chain / linker length error).
    if mcs_coverage >= 0.95 and abs(atom_diff) <= 1:
        if vlm_valid:
            return {
                'status': 'accepted',
                'corrected_smiles': vlm_smiles,
                'similarity': similarity,
                'mcs_coverage': mcs_coverage,
                'atom_diff': atom_diff,
                'reason': 'simple_atom_correction',
            }
        return {
            'status': 'keep_molnextr',
            'corrected_smiles': molnextr_smiles,
            'similarity': similarity,
            'mcs_coverage': mcs_coverage,
            'atom_diff': atom_diff,
            'reason': 'vlm_smiles_invalid_fragment',
        }

    # Same ring system and a single-atom difference on the linker: this is
    # the classic "one extra / one missing atom" case, safe to correct.
    # Only meaningful when both molecules actually have rings.
    both_ringed = (
        _ring_signature(mx_stripped) is not None
        and _ring_signature(vlm_stripped) is not None
    )
    if both_ringed and ring_match and abs(atom_diff) == 1 and vlm_valid:
        return {
            'status': 'accepted',
            'corrected_smiles': vlm_smiles,
            'similarity': similarity,
            'mcs_coverage': mcs_coverage,
            'atom_diff': atom_diff,
            'reason': 'ring_match_single_atom_correction',
        }

    # Medium divergence: keep MolNexTR
    return {
        'status': 'keep_molnextr',
        'corrected_smiles': molnextr_smiles,
        'similarity': similarity,
        'mcs_coverage': mcs_coverage,
        'atom_diff': atom_diff,
        'reason': 'medium_divergence',
    }
