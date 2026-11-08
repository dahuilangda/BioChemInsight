import numpy as np
import pytest
from rdkit import Chem

from cyclopeptide2d import depict
from cyclopeptide2d.frames import peptide_frames


@pytest.mark.parametrize("residues", [3, 4, 5])
def test_cycle_template_angles_follow_ring_geometry(residues):
    smiles = "O=C1" + "NCC(=O)" * (residues - 1) + "NC1"
    molecule = Chem.MolFromSmiles(smiles)
    frames, order, _ = peptide_frames(molecule, backbone_template="auto")
    expected_angle = np.pi - 2 * np.pi / len(order)
    assert len(frames) == residues
    for frame in frames:
        assert frame.backbone_template == "cycle"
        local = dict(zip(frame.atoms, frame.local, strict=True))
        a, b, c = frame.backbone
        first, second = local[a] - local[b], local[c] - local[b]
        cosine = first @ second / np.linalg.norm(first) / np.linalg.norm(second)
        assert np.isclose(cosine, np.cos(expected_angle))
        for bond in molecule.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in local and b in local:
                assert np.isclose(np.linalg.norm(local[a] - local[b]), 1)
        neighbours = [
            atom.GetIdx() for atom in molecule.GetAtomWithIdx(frame.c_atom).GetNeighbors()
        ]
        alpha = frame.backbone[-2]
        oxygen = next(
            atom for atom in neighbours if molecule.GetAtomWithIdx(atom).GetAtomicNum() == 8
        )
        assert np.isclose(frame.port_cosines[frame.c_atom, alpha], np.cos(expected_angle))
        assert np.isclose(
            frame.port_cosines[frame.c_atom, oxygen], np.cos((2 * np.pi - expected_angle) / 2)
        )


def test_native_template_and_larger_cycle_policy_remain_available():
    small = Chem.MolFromSmiles("O=C1NCC(=O)NCC(=O)NCC1")
    frames, _, _ = peptide_frames(small, backbone_template="residue")
    assert all(frame.backbone_template == "residue" for frame in frames)
    large = Chem.MolFromSmiles("O=C1" + "NCC(=O)" * 5 + "NC1")
    frames, _, _ = peptide_frames(large, backbone_template="auto")
    assert all(frame.backbone_template == "residue" for frame in frames)


@pytest.mark.parametrize(
    "smiles",
    [
        "O=C1N[C@@H](Cc2ccccc2)C(=O)N[C@@H](CC)C(=O)N[C@@H](CO)1",
        "O=C1NCC(=O)N(C)[C@@H](C)C(=O)N[C@H](CC(C)C)1",
        "O=C1N2CCC[C@H]2C(=O)N[C@@H](C)C(=O)NCC1",
    ],
)
def test_small_cycles_pass_full_validation_without_changing_chemical_identity(smiles):
    result = depict(smiles)
    assert result.accepted
    assert Chem.MolToSmiles(result.molecule) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert result.report["stereo_roundtrip"]
    assert set(result.report["backbone_templates"]) == {"cycle"}
    constraints = result.report["constraints"]
    assert constraints["max_link_error"] < 1e-7
    assert constraints["max_bond_relative_error"] < 1e-6
    assert constraints["min_angle_slack"] >= -1e-6
    assert constraints["min_global_separation_slack"] >= -1e-6
    assert result.report["metrics"]["bond_crossings"] == 0
    assert result.report["metrics"]["n_ring_intrusions"] == 0


def test_template_branch_normalization_keeps_anchors_and_input():
    from cyclopeptide2d.frames import _unit_tree_bonds

    molecule = Chem.MolFromSmiles("CCCC")
    original = np.array([[0.0, 0.0], [1.0, 0.0], [1.9, 0.0], [1.9, 0.9]])
    result = _unit_tree_bonds(molecule, original, {0, 1})
    np.testing.assert_array_equal(result[:2], original[:2])
    np.testing.assert_array_equal(original[2:], [[1.9, 0.0], [1.9, 0.9]])
    for bond in molecule.GetBonds():
        assert np.isclose(
            np.linalg.norm(result[bond.GetBeginAtomIdx()] - result[bond.GetEndAtomIdx()]), 1
        )


def test_template_branch_normalization_rejects_conflicting_anchors():
    from cyclopeptide2d.frames import _unit_tree_bonds

    with pytest.raises(ValueError, match="inconsistent_template_anchors"):
        _unit_tree_bonds(Chem.MolFromSmiles("CC"), np.array([[0.0, 0.0], [0.9, 0.0]]), {0, 1})
