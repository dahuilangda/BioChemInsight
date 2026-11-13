import pytest
from rdkit import Chem

pytest.importorskip("peptidecore")

from peptidecore import decompose_smiles, from_helm, from_sequence

from cyclopeptide2d import depict_record, depict_sequence

CYCLIC = "PEPTIDE1{G.G.G.G.G.G.G.G}$PEPTIDE1,PEPTIDE1,8:R2-1:R1$$$V2.0"


def test_helm_to_layout_preserves_atoms_and_monomer_ownership():
    record = from_helm(CYCLIC)
    result = depict_record(record, maxiter=150)
    assert result.accepted
    assert Chem.MolToSmiles(result.molecule) == record.smiles
    assert result.report["chemical_record"] == record.as_dict()
    for index, atom in enumerate(result.molecule.GetAtoms()):
        assert atom.GetProp("_peptide_instance") == record.atom_instances[index]
        assert atom.GetIntProp("_peptide_origin") == record.atom_origins[index]
    for connection in record.connections:
        assert result.molecule.GetBondBetweenAtoms(connection.source_atom, connection.target_atom)


def test_sequence_entry_point_uses_the_existing_layout_validation():
    assert depict_sequence(CYCLIC, maxiter=150).accepted
    linear = depict_sequence("AGK", input_format="sequence", maxiter=150)
    assert linear.accepted
    assert linear.report["topology"] == "linear"
    assert Chem.MolToSmiles(linear.molecule) == from_sequence("AGK").smiles


def test_biln_and_recovered_smiles_reach_the_same_layout_pipeline():
    biln = "G(1,1)-G-G-G-G-G-G-G(1,2)"
    assert depict_sequence(biln, input_format="BILN", maxiter=150).accepted
    source = from_helm(CYCLIC).smiles
    result = decompose_smiles(source)
    assert result.unique
    layout = depict_record(result.records[0], maxiter=150)
    assert layout.accepted
    assert layout.report["chemical_record"]["source"] == {"text": source, "format": "SMILES"}


@pytest.mark.parametrize(
    "symbols",
    [
        "T.[meA].[meF].F.P",
        "D.D.K.[meF].T",
        "A.P.[meF].R.E",
        "[meF].W.E.[meL].W",
    ],
)
def test_modified_small_cycles_preserve_stereo_and_unit_bonds(symbols):
    helm = f"PEPTIDE1{{{symbols}}}$PEPTIDE1,PEPTIDE1,5:R2-1:R1$$$V2.0"
    record = from_helm(helm)
    result = depict_record(record, maxiter=200)
    assert result.accepted
    assert Chem.MolToSmiles(result.molecule) == record.smiles
    assert result.report["constraints"]["max_bond_relative_error"] < 1e-6
    assert result.report["constraints"]["min_global_separation_slack"] >= -1e-6
    assert result.report["stereo_roundtrip"]


def test_attachment_moves_keep_native_angles_branch_order_and_ports():
    import math

    import numpy as np

    from cyclopeptide2d.frames import peptide_frames
    from cyclopeptide2d.solver import PoseProblem

    record = from_helm("PEPTIDE1{D.D.K.[meF].T}$PEPTIDE1,PEPTIDE1,5:R2-1:R1$$$V2.0")
    molecule = record.molecule
    frames, order, _ = peptide_frames(molecule, backbone_template="auto")
    problem = PoseProblem(molecule, frames, order)
    initial = problem.local.copy()
    moves = list(problem.attachment_moves(range(len(frames))))
    assert moves
    assert any(
        pivot not in set(order)
        and molecule.GetAtomWithIdx(pivot).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for _, pivot, _, _ in moves
    )
    for _, pivot, moving, angle in moves:
        problem.local[:] = initial
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        origin = initial[pivot]
        problem.local[moving] = (initial[moving] - origin) @ rotation.T + origin
        np.testing.assert_array_equal(problem.local[order], initial[order])
        for bond in molecule.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            assert np.isclose(
                np.linalg.norm(problem.local[a] - problem.local[b]),
                np.linalg.norm(initial[a] - initial[b]),
            )
        neighbours = [atom.GetIdx() for atom in molecule.GetAtomWithIdx(pivot).GetNeighbors()]
        for a in neighbours:
            for b in neighbours:
                if a == b:
                    continue
                u, v = problem.local[a] - origin, problem.local[b] - origin
                theta = math.acos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))
                assert abs(theta - problem.junction_angles[a, pivot, b]) <= math.radians(12) + 1e-9
                old_u, old_v = initial[a] - origin, initial[b] - origin
                old_cross = old_u[0] * old_v[1] - old_u[1] * old_v[0]
                new_cross = u[0] * v[1] - u[1] * v[0]
                assert old_cross * new_cross >= -1e-12
