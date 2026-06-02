from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from utils.structure_recognition import StructureRecognizer


def _draw_mol(smiles: str, path: Path, wavy_bond: tuple[int, int] | None = None) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    rdDepictor.Compute2DCoords(mol)
    if wavy_bond:
        bond = mol.GetBondBetweenAtoms(*wavy_bond)
        if bond is None:
            raise ValueError(f"missing bond {wavy_bond} in {smiles}")
        bond.SetBondDir(Chem.BondDir.UNKNOWN)
    drawer = rdMolDraw2D.MolDraw2DCairo(320, 160)
    drawer.drawOptions().useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    path.write_bytes(drawer.GetDrawingText())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/codex_layout_check/molnextr_attachment_capability")
    parser.add_argument("--real-crop", action="append", default=[])
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        {"name": "star_direct", "smiles": "[*]N(CC)CCO", "expect_dummy": False},
        {"name": "numbered_dummy_direct", "smiles": "[4*]N(CC)CCO", "expect_dummy": False},
        {"name": "star_wavy", "smiles": "[*]N(CC)CCO", "wavy_bond": (0, 1), "expect_dummy": True},
        {"name": "scaffold_r4", "smiles": "c1cc([4*])ncc1", "expect_dummy": True},
    ]
    image_paths = []
    for case in cases:
        path = output / f"{case['name']}.png"
        _draw_mol(case["smiles"], path, case.get("wavy_bond"))
        case["image"] = str(path)
        image_paths.append(path)
    for path in args.real_crop:
        image_paths.append(Path(path))

    recognizer = StructureRecognizer()
    results = []
    for path in image_paths:
        prediction = recognizer.predict_segment_file(str(path))
        raw = prediction.raw if isinstance(prediction.raw, dict) else {}
        results.append({
            "image": str(path),
            "predicted_smiles": prediction.smiles,
            "quality_issues": list(prediction.quality_issues),
            "atom_symbols": [atom.get("atom_symbol") for atom in raw.get("atom_sets", []) if isinstance(atom, dict)],
            "bond_types": [bond.get("bond_type") for bond in raw.get("bond_sets", []) if isinstance(bond, dict)],
            "molblock_atom_lines": (prediction.molblock or "").splitlines()[4:14],
        })
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
