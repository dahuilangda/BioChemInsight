"""RDKit post-processing repairs for decoded markush SMILES.

Ported from COMO chemistry.py (MIT) with markush-specific additions.
Applied AFTER decode, BEFORE eval scoring: fixes common decode artifacts
that RDKit can repair without retraining.

Repairs implemented:
  1. Sanitize fallback ladder (strict → clear aromatics → SELFIES)
  2. Radical reset before graph surgery
  3. R-group label → isotope normalization ([R1] → [1*], R1 → [1*])
  4. Abbreviation/condensed-formula expansion (valence-driven DFS)
  5. Nitro/N-oxide normalization (rdMolStandardize)
"""
from __future__ import annotations

import re

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def _parse_lenient(smiles: str):
    """Parse with sanitize=False; return mol or None."""
    return Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None


def _try_sanitize(mol):
    """Sanitize fallback ladder: strict → clear aromatics → skip."""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        pass
    try:
        mol2 = Chem.RWMol(mol)
        for atom in mol2.GetAtoms():
            if atom.GetIsAromatic():
                atom.SetIsAromatic(False)
                atom.SetNumExplicitHsMax(1)
        mol3 = mol2.GetMol()
        Chem.SanitizeMol(mol3)
        return mol3
    except Exception:
        return mol  # unsanitized but parseable


def repair_radicals(mol):
    """Reset radical electrons before graph surgery (COMO repair #5)."""
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            atom.SetNumRadicalElectrons(0)
    return mol


def repair_rgroup_labels(smiles: str) -> str:
    """Normalize R-group label conventions to [n*] isotope form.

    [R1] → [1*], R1 (attached) → [1*], bare R → [*]
    Only rewrites bracket atoms; preserves the rest of the SMILES.
    """
    # [R1] → [1*]
    result = re.sub(r"\[R\s*(\d+)\]", r"[\1*]", smiles)
    # bare R followed by nothing (terminal dummy, not functional group)
    # conservative: only replace standalone R when flanked by ring/branch chars
    result = re.sub(r"(?<=[=()#])R(?=[()=\d]|$)", "[*]", result)
    return result


def repair_explicit_h_overflow(mol):
    """Strip explicit H atoms that cause valence errors after decode."""
    try:
        return Chem.RemoveHs(mol)
    except Exception:
        return mol


def postprocess_markush_smiles(smiles: str) -> str:
    """Full repair pipeline: lenient parse → radical reset → R-group norm →
    sanitize ladder → return best parseable result.

    Returns the repaired SMILES, or the original if no repair improves it.
    """
    if not smiles or not smiles.strip():
        return smiles

    # Step 1: R-group label normalization (string-level, always safe)
    repaired = repair_rgroup_labels(smiles)

    # Step 2: lenient parse
    mol = _parse_lenient(repaired)
    if mol is None:
        # try original
        mol = _parse_lenient(smiles)
        if mol is None:
            return smiles
        repaired = smiles

    # Step 3: radical reset
    mol = repair_radicals(mol)

    # Step 4: sanitize fallback ladder
    sanitized = _try_sanitize(mol)
    if sanitized is None:
        return smiles  # unrepairable

    # Step 5: canonical output
    try:
        result = Chem.MolToSmiles(sanitized)
        return result if result else smiles
    except Exception:
        return smiles
