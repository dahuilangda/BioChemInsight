import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

from training.peptide_data.quality import geometry_report, stereo_roundtrip


def _coordinates(smiles, points):
    molecule = Chem.MolFromSmiles(smiles)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, (x, y) in enumerate(points):
        conformer.SetAtomPosition(index, (x, y, 0.0))
    molecule.AddConformer(conformer)
    return molecule


def test_bond_crossing_is_rejected():
    molecule = _coordinates("CCCC", [(0, 0), (1, 1), (0, 1), (1, 0)])
    report = geometry_report(molecule)
    assert not report["accepted"]
    assert report["crossings"] == [(0, 2)]
    assert "bond_crossing" in report["reasons"]


def test_atom_on_nonincident_bond_is_rejected():
    molecule = _coordinates("CCCC", [(0, 0), (2, 0), (2, 2), (1, 0)])
    report = geometry_report(molecule)
    assert not report["accepted"]
    assert "atom_bond_contact" in report["reasons"]


def test_collapsed_bond_is_rejected():
    molecule = _coordinates("CC", [(0, 0), (0, 0)])
    assert geometry_report(molecule)["reasons"] == ["collapsed_bond"]


def test_valid_chain_is_accepted():
    molecule = Chem.MolFromSmiles("N[C@@H](C)C(=O)NCC(=O)O")
    rdDepictor.Compute2DCoords(molecule, forceRDKit=True)
    assert geometry_report(molecule)["accepted"]
    assert stereo_roundtrip(molecule)


def test_nonfinite_coordinates_are_rejected():
    molecule = _coordinates("CC", [(0, 0), (np.nan, 0)])
    assert geometry_report(molecule)["reasons"] == ["nonfinite_coordinates"]
