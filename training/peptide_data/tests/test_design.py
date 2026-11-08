from collections import Counter

from peptidecore import from_helm
from rdkit import Chem

from training.peptide_data.design import STRATA, build_design


def test_review_matrix_is_reproducible_and_unique():
    design = build_design()
    assert design == build_design()
    assert len(design["records"]) == 96
    assert design["counts"] == {stratum: 8 for stratum in STRATA}
    assert len({row["chemical_record"]["smiles"] for row in design["records"]}) == 96
    assert not design["training_release_approved"]
    for row in design["records"]:
        record = from_helm(row["helm"])
        molecule = record.molecule
        assert record.as_dict() == row["chemical_record"]
        assert len(Chem.GetMolFrags(molecule)) == 1
        assert not row["training_release_approved"]
        assert row["split"] == "review"
        if row["stratum"] == "multiple_closures":
            assert len(record.graph.connections) == 2
        if row["stratum"] == "branched":
            assert len(record.graph.chains) == 2
            assert len(record.graph.connections) == 1


def test_seed_changes_chemistry_without_changing_allocations():
    first, second = build_design(per_stratum=1), build_design(per_stratum=1, seed=29)
    assert Counter(row["stratum"] for row in first["records"]) == Counter(
        row["stratum"] for row in second["records"]
    )
    assert [row["id"] for row in first["records"]] != [
        row["id"] for row in second["records"]
    ]
