import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from cyclopeptide2d import LayoutResult, depict, render_svg, validate_input


@pytest.mark.parametrize(
    "smiles,reason",
    [
        ("CC[CH]CC", "input_contains_radical_electrons"),
        ("NCC(=O)*", "input_contains_unresolved_atoms"),
        ("NCC(=O)O.CC", "input_contains_disconnected_components"),
    ],
)
def test_input_rejects_parseable_unsupported_labels(smiles, reason):
    with pytest.raises(ValueError, match=reason):
        validate_input(smiles)


def test_failed_candidate_cannot_be_rendered():
    result = LayoutResult(Chem.MolFromSmiles("NCC(=O)O"), {"qa_pass": False})
    with pytest.raises(ValueError, match="layout_failed_geometry_qa"):
        render_svg(result)


def test_acyclic_input_returns_explicit_topology_failure():
    result = depict("NCC(=O)O")
    assert not result.accepted
    assert result.report["error"] == "molecule has no ring"


def test_documented_cli_molecule_solves_and_rendering_preserves_coordinates():
    smiles = "N1CC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC1=O"
    result = depict(smiles, maxiter=150)
    assert result.accepted
    original = result.molecule.GetConformer().GetPositions().copy()
    assert "<svg" in render_svg(result)
    assert np.array_equal(original, result.molecule.GetConformer().GetPositions())
    assert Chem.MolToSmiles(result.molecule) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_contacts_can_be_resolved_after_local_rotations_stall():
    case = json.loads((Path(__file__).parent / "fixtures/contact_stall.json").read_text())
    result = depict(case["smiles"], seed=case["seed"])
    assert result.accepted
    assert result.report["constraints"]["min_global_separation_slack"] >= -1e-6
    assert result.report["stereo_roundtrip"]
