import json
from pathlib import Path

import pytest
from rdkit import Chem

from cyclopeptide2d.chemistry import linear_backbone_order, peptide_macroring
from cyclopeptide2d.frames import peptide_frames


@pytest.mark.parametrize(
    "smiles",
    [
        "N[C@@H](C)C(=O)NCC(=O)N[C@@H](CCCCN)C(=O)O",
        "CC(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](CCCCN)C(N)=O",
        "CN[C@@H](C)C(=O)NCC(=O)N(C)[C@@H](CC(=O)N)C(=O)O",
        "[NH3+][C@@H](C)C(=O)NCC(=O)N[C@@H](CCC(N)=O)C(=O)[O-]",
    ],
)
def test_linear_backbones_are_complete_after_atom_reordering(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    for _ in range(5):
        shuffled = Chem.MolFromSmiles(Chem.MolToSmiles(molecule, doRandom=True))
        order = linear_backbone_order(shuffled)
        assert order is not None
        assert len(order) == 9
        assert len(set(order)) == len(order)
        assert all(
            shuffled.GetBondBetweenAtoms(a, b) is not None
            for a, b in zip(order, order[1:], strict=False)
        )


def test_linear_backbone_search_does_not_recurse_over_residues():
    molecule = Chem.MolFromSmiles("NCC(=O)" * 1100 + "O")
    order = linear_backbone_order(molecule)
    assert len(order) == 3300


def test_disulfide_does_not_become_a_backbone_closure():
    molecule = Chem.MolFromSmiles("N[C@@H]1CSSC[C@@H](C(=O)O)NC(=O)[C@H](C)NC(=O)[C@H](C)NC1=O")
    frames, order, crosslinks = peptide_frames(molecule)
    assert len(frames) == 4
    assert all(frame.backbone_template == "linear" for frame in frames)
    assert len(crosslinks) == 1
    assert [molecule.GetAtomWithIdx(atom).GetSymbol() for atom in crosslinks[0]] == ["S", "S"]
    assert len(order) == 12


def test_amide_cycle_survives_urea_and_an_extra_ring():
    molecule = Chem.MolFromSmiles(
        "O=C1NC(CS2)C(=O)NCC(=O)NCC(=O)NCC(=O)NC(CS2)C(=O)NCC(=O)NCC(=O)NCC(=O)N1"
    )
    cycle = peptide_macroring(molecule)
    assert cycle is not None
    assert sum(molecule.GetAtomWithIdx(atom).GetAtomicNum() == 7 for atom in cycle) == 9
    assert not any(molecule.GetAtomWithIdx(atom).GetAtomicNum() == 16 for atom in cycle)
    _, _, crosslinks = peptide_frames(molecule)
    assert len(crosslinks) == 1


def test_beta_residue_backbone_cycle_is_retained():
    molecule = Chem.MolFromSmiles("O=C1NCC(=O)NCC(=O)NCC1")
    assert linear_backbone_order(molecule) is None
    cycle = peptide_macroring(molecule)
    assert cycle is not None and len(cycle) == 10
    frames, order, crosslinks = peptide_frames(molecule)
    assert len(frames) == 3 and len(order) == 10
    assert not crosslinks


def test_head_to_side_closure_uses_the_acylated_terminal_nitrogen():
    from cyclopeptide2d import depict

    references = json.loads(
        (Path(__file__).parents[1] / "examples/reviewed_references.json").read_text()
    )
    molecule = Chem.MolFromSmiles(references["68"]["smiles"])
    for _ in range(3):
        smiles = Chem.MolToSmiles(molecule, doRandom=True)
        parsed = Chem.MolFromSmiles(smiles)
        frames, _, crosslinks = peptide_frames(parsed, backbone_template="auto")
        assert all(frame.backbone_template != "linear" for frame in frames)
        assert crosslinks == []
        result = depict(smiles)
        assert result.accepted
        assert result.report["stereo_roundtrip"]
        assert Chem.MolToSmiles(result.molecule) == Chem.MolToSmiles(molecule)
