from collections import Counter

from peptidecore import Chain, PeptideGraph, core_library, from_helm
from rdkit import Chem

from training.peptide_data.design import (
    STRATA,
    build_design,
    build_design_parallel,
    composition_group,
)


def test_review_matrix_is_reproducible_and_unique():
    design = build_design()
    assert design == build_design()
    assert len(design["records"]) == 8 * len(STRATA)
    assert design["counts"] == {stratum: 8 for stratum in STRATA}
    assert len(
        {row["chemical_record"]["smiles"] for row in design["records"]}
    ) == 8 * len(STRATA)
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


def test_parallel_design_is_independent_of_worker_count():
    serial = build_design_parallel(seed=20261109, per_stratum=1, workers=1)
    parallel = build_design_parallel(seed=20261109, per_stratum=1, workers=2)
    assert serial == parallel
    assert len(parallel["records"]) == len(STRATA)


def test_seed_changes_chemistry_without_changing_allocations():
    first, second = build_design(per_stratum=1), build_design(per_stratum=1, seed=29)
    assert Counter(row["stratum"] for row in first["records"]) == Counter(
        row["stratum"] for row in second["records"]
    )
    assert [row["id"] for row in first["records"]] != [
        row["id"] for row in second["records"]
    ]


def test_composition_guard_ignores_sequence_order_and_chain_partition():
    library = core_library()
    first = PeptideGraph((Chain("PEPTIDE1", ("A", "G", "G")),))
    reordered = PeptideGraph((Chain("PEPTIDE1", ("G", "A", "G")),))
    partitioned = PeptideGraph(
        (Chain("PEPTIDE1", ("A",)), Chain("PEPTIDE2", ("G", "G")))
    )
    modified = PeptideGraph((Chain("PEPTIDE1", ("dA", "G", "G")),))
    assert composition_group(first, library) == composition_group(reordered, library)
    assert composition_group(first, library) == composition_group(partitioned, library)
    assert composition_group(first, library) != composition_group(modified, library)


def test_aromatic_cycles_cover_methylation_and_halogen_variants():
    design = build_design(per_stratum=20, strata=("head_to_tail_aromatic",))
    lengths = set()
    for row in design["records"]:
        record = from_helm(row["helm"])
        symbols = record.graph.chains[0].monomers
        lengths.add(len(symbols))
        assert 8 <= len(symbols) <= 18
        assert (
            sum(symbol.startswith("me") for symbol in symbols) == 2 + row["ordinal"] % 3
        )
        halogens = sum(
            atom.GetAtomicNum() in (9, 17, 35, 53)
            for atom in record.molecule.GetAtoms()
        )
        assert halogens == int(row["ordinal"] % 2 == 0)
        assert sum(atom.GetIsAromatic() for atom in record.molecule.GetAtoms()) >= 24
        assert len(record.graph.connections) == 1
    assert len(lengths) >= 3


def test_standard_sampling_alphabet_contains_all_twenty_residues():
    from training.peptide_data.design import STANDARD

    assert len(STANDARD) == 20
    assert set(STANDARD) == set("ACDEFGHIKLMNPQRSTVWY")
    for symbol in STANDARD:
        assert core_library().get(symbol).monomer_type == "Backbone"
