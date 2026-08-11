from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any


BACKEND_REFERENCES = [
    {
        "name": "RDKit",
        "url": "https://www.rdkit.org/docs/Install.html",
        "role": "2D layout, SVG metadata, atom draw coordinates, and renderer-native attachment primitives.",
    },
    {
        "name": "MolDepictor",
        "url": "https://github.com/DS4SD/MolDepictor/blob/main/mol_depict/generation/generation.py",
        "role": "Reference for randomized RDKit depiction options and metadata-derived keypoints.",
    },
    {
        "name": "MolDepictor image transformations",
        "url": "https://github.com/DS4SD/MolDepictor/blob/main/mol_depict/utils/image_transformation.py",
        "role": "Reference for realistic noise/caption/line perturbation domains.",
    },
]


def import_rdkit():
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    return Chem, rdDepictor, rdMolDraw2D


def stable_id(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def can_accept_attachment_dummy(mol: Any, atom_index: int) -> bool:
    Chem, _, _ = import_rdkit()

    try:
        atom = mol.GetAtomWithIdx(int(atom_index))
        if atom.GetNumRadicalElectrons() != 0:
            return False
        editable = Chem.RWMol(mol)
        dummy = Chem.Atom("*")
        dummy.SetNoImplicit(True)
        dummy_index = int(editable.AddAtom(dummy))
        editable.AddBond(int(atom_index), dummy_index, Chem.BondType.SINGLE)
        fragment = editable.GetMol()
        Chem.SanitizeMol(fragment)
        smiles = Chem.MolToSmiles(fragment, canonical=True)
        parsed = Chem.MolFromSmiles(smiles)
        return parsed is not None and Chem.MolToSmiles(parsed, canonical=True) == smiles
    except Exception:
        return False


def choose_anchor(mol: Any, rng: random.Random) -> int:
    preferred = []
    secondary = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if atom.GetDegree() >= 4:
            continue
        if symbol not in {"C", "N", "O", "S", "P"}:
            continue
        if atom.GetNumRadicalElectrons() != 0:
            continue
        if not can_accept_attachment_dummy(mol, int(atom.GetIdx())):
            continue
        if atom.GetTotalNumHs() > 0:
            preferred.append(atom.GetIdx())
        secondary.append(atom.GetIdx())
    candidates = preferred or secondary
    if not candidates:
        candidates = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetDegree() < 4
            and atom.GetTotalNumHs() > 0
            and atom.GetNumRadicalElectrons() == 0
            and can_accept_attachment_dummy(mol, int(atom.GetIdx()))
        ]
    if not candidates:
        raise ValueError("no usable anchor atoms")
    return int(rng.choice(candidates))


def atom_label(atom: Any) -> str:
    if atom.GetSymbol() == "*":
        return "*"
    if atom.HasProp("atomLabel"):
        return atom.GetProp("atomLabel")
    return atom.GetSymbol()


def normalize_coord(x: float, y: float, width: int, height: int) -> dict[str, float]:
    return {"x": max(0.0, min(1.0, x / float(width))), "y": max(0.0, min(1.0, y / float(height)))}


def side_from_endpoint(x: float, y: float) -> str:
    distances = {
        "left": x,
        "right": 1.0 - x,
        "top": y,
        "bottom": 1.0 - y,
    }
    return min(distances, key=distances.get)


def configure_draw_options(options: Any, *, style: str, seed: int, allow_rotation: bool = True) -> None:
    if style == "rdkit_patent_thin":
        options.bondLineWidth = 1.4
        options.minFontSize = 18
        options.maxFontSize = 28
    elif style == "rdkit_patent_bold":
        options.bondLineWidth = 2.8
        options.minFontSize = 22
        options.maxFontSize = 34
    elif style == "rdkit_margin_crop":
        options.bondLineWidth = 2.0
        options.minFontSize = 20
        options.maxFontSize = 32
        options.padding = 0.18
    elif style == "rdkit_scan_like":
        options.bondLineWidth = 2.2
        options.minFontSize = 20
        options.maxFontSize = 32
        options.padding = 0.10
    else:
        options.bondLineWidth = 2.0
        options.minFontSize = 20
        options.maxFontSize = 32
        options.padding = 0.08
    options.rotate = int(seed % 360) if allow_rotation else 0
    options.comicMode = style == "rdkit_scan_like"
    options.setAtomPalette({17: (0, 0, 0)})


def graph_consistency(fragment: Any, smiles: str, anchor_index: int, dummy_index: int) -> dict[str, Any]:
    Chem, _, _ = import_rdkit()
    parsed = Chem.MolFromSmiles(smiles)
    canonical_from_row = Chem.MolToSmiles(parsed) if parsed is not None else ""
    canonical_from_mol = Chem.MolToSmiles(fragment)
    dummy_atoms = [atom.GetIdx() for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
    anchor_dummy_bond = fragment.GetBondBetweenAtoms(int(anchor_index), int(dummy_index)) is not None
    return {
        "canonical_smiles": canonical_from_mol,
        "row_smiles_canonical_matches_mol": bool(canonical_from_row == canonical_from_mol),
        "atom_count": int(fragment.GetNumAtoms()),
        "bond_count": int(fragment.GetNumBonds()),
        "dummy_atom_indices": [int(index) for index in dummy_atoms],
        "single_dummy_atom": bool(dummy_atoms == [int(dummy_index)]),
        "anchor_dummy_bond_present": bool(anchor_dummy_bond),
        "anchor_index": int(anchor_index),
        "dummy_index": int(dummy_index),
    }


def perturb_image(path: Path, style: str, rng: random.Random) -> dict[str, Any]:
    from PIL import Image, ImageFilter
    import numpy as np

    image = Image.open(path).convert("L")
    operations = []
    if style == "rdkit_scan_like":
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.65)))
        operations.append("gaussian_blur")
        arr = np.asarray(image).astype("int16")
        noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, rng.uniform(1.0, 4.0), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype("uint8")
        image = Image.fromarray(arr, mode="L")
        operations.append("mild_noise")
    image.convert("RGB").save(path)

    arr = np.asarray(Image.open(path).convert("L"))
    dark_ratio = float((arr < 245).mean())
    return {
        "operations": operations,
        "dark_pixel_ratio": dark_ratio,
        "blank": dark_ratio < 0.0005,
        "dense": dark_ratio > 0.85,
    }


def bond_records(mol: Any) -> list[dict[str, Any]]:
    records = []
    for bond in mol.GetBonds():
        records.append(
            {
                "begin_atom_index": int(bond.GetBeginAtomIdx()),
                "end_atom_index": int(bond.GetEndAtomIdx()),
                "bond_order": str(bond.GetBondType()),
                "bond_dir": str(bond.GetBondDir()),
                "is_aromatic": bool(bond.GetIsAromatic()),
            }
        )
    return records
