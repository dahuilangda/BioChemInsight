import pytest
from peptidecore import core_library, from_helm
from cyclopeptide2d.frames import registered_frames

from training.peptide_data.coverage_design import classification, make_record


def test_unusable_registry_attachment_is_quarantined():
    assert classification(core_library().get("D-Ala-ol")) is None
    assert classification(core_library().get("M")) == ("internal", "alpha")


@pytest.mark.parametrize(
    "ordinal,expected_cycles", [(0, 0), (1, 1), (2, 2), (3, 0), (7, 1)]
)
def test_coverage_topologies_preserve_the_requested_residue(ordinal, expected_cycles):
    row = make_record("M", ordinal, 20261112)
    record = from_helm(row["helm"])
    assert any("M" in chain.monomers for chain in record.graph.chains)
    _, graph = registered_frames(record)
    assert graph.cycle_rank == expected_cycles
    assert record.smiles == row["chemical_record"]["smiles"]
    assert not row["training_release_approved"]
