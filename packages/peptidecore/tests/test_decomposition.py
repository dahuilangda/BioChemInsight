import json
import random
from pathlib import Path

import pytest
from rdkit import Chem

from peptidecore import (
    Attachment,
    Chain,
    Connection,
    Endpoint,
    Monomer,
    PeptideError,
    PeptideGraph,
    assemble,
    core_library,
    decompose_smiles,
    from_helm,
    from_sequence,
    to_helm,
    to_sequence,
)
from peptidecore.library import identity, parse_smiles

MONOMERS = tuple(core_library().monomers.values())


@pytest.mark.parametrize("monomer", MONOMERS, ids=lambda m: m.symbol)
def test_all_source_monomers_have_a_verified_structural_decomposition(monomer):
    source = identity(parse_smiles(monomer.smiles))
    result = decompose_smiles(source, max_solutions=4, max_states=2000)
    assert result.records, (monomer.symbol, result.as_dict())
    assert all(record.smiles == source for record in result.records)
    assert result.coverage == 1
    assert not result.unidentified


@pytest.mark.parametrize(
    "helm",
    [
        "PEPTIDE1{A.G.K}$$$$V2.0",
        "PEPTIDE1{I.[dA].T.P.[dT]}$$$$V2.0",
        "PEPTIDE1{A.G.K}$PEPTIDE1,PEPTIDE1,3:R2-1:R1$$$V2.0",
        "PEPTIDE1{C.A.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-5:R3$$$V2.0",
        "PEPTIDE1{C.A.C}|PEPTIDE2{C.G.C}$PEPTIDE1,PEPTIDE2,1:R3-3:R3|PEPTIDE1,PEPTIDE2,3:R3-1:R3$$$V2.0",
        "PEPTIDE1{[ac].A.[Aib].[meA].[am]}$$$$V2.0",
        "PEPTIDE1{C.A.K.D.C}$PEPTIDE1,PEPTIDE1,1:R3-5:R3|PEPTIDE1,PEPTIDE1,3:R3-4:R3|PEPTIDE1,PEPTIDE1,1:R1-5:R2$$$V2.0",
    ],
)
def test_connections_and_stereochemistry_survive_smiles_without_metadata(helm):
    source = from_helm(helm)
    plain = Chem.MolFromSmiles(source.smiles)
    assert not any(a.HasProp("_peptide_instance") for a in plain.GetAtoms())
    result = decompose_smiles(Chem.MolToSmiles(plain), max_solutions=16)
    assert result.records
    for record in result.records:
        assert record.smiles == source.smiles
        assert from_helm(to_helm(record.graph)).smiles == source.smiles
        assert len(record.atom_instances) == plain.GetNumAtoms()
        for connection in record.connections:
            assert plain.GetBondBetweenAtoms(connection.source_atom, connection.target_atom)
    assert result.verification_failures == 0


def test_unique_sequence_is_recovered_from_an_independent_atomic_structure():
    molecule = Chem.MolFromSequence("AGK")
    result = decompose_smiles(Chem.MolToSmiles(molecule))
    assert result.unique and result.search_exhausted
    assert to_sequence(result.records[0]) == "AGK"


def test_atom_order_does_not_change_the_resolved_residue_graph():
    source = from_helm("PEPTIDE1{A.G.K}$PEPTIDE1,PEPTIDE1,3:R2-1:R1$$$V2.0")
    expected = to_helm(decompose_smiles(source.smiles).records[0].graph)
    for random_smiles in Chem.MolToRandomSmilesVect(source.molecule, 12, randomSeed=20261108):
        result = decompose_smiles(random_smiles)
        assert result.unique
        assert to_helm(result.records[0].graph) == expected
        assert Chem.MolToSmiles(result.records[0].molecule) == source.smiles


def test_chemical_equivalence_does_not_imply_unique_residue_assignment():
    result = decompose_smiles(from_sequence("E").smiles)
    assert result.status == "ambiguous"
    assert not result.unique
    assert {record.graph.chains[0].monomers for record in result.records} == {("E",), ("gGlu",)}
    assert len({record.smiles for record in result.records}) == 1


def test_unseen_isotope_residue_is_retained_without_a_false_sequence():
    monomer = Monomer(
        "isotope",
        "PEPTIDE",
        "Isotope residue",
        "[H:1]N[13CH2]C(=O)[OH:2]",
        (Attachment("R1", "[*:1][H]"), Attachment("R2", "O[*:2]")),
    )
    library = core_library().extend((monomer,), name="isotopes")
    source = from_helm("PEPTIDE1{A.[isotope].G}$$$$V2.0", library=library)
    unknown = decompose_smiles(source.smiles)
    assert unknown.status == "partial" and not unknown.records
    assert unknown.unidentified and any("13C" in f.smiles for f in unknown.unidentified)
    assert {m.symbol for m in unknown.matches} >= {"A", "G"}
    recognized = {a for m in unknown.matches for a in m.atoms}
    unresolved = {a for f in unknown.unidentified for a in f.atoms}
    assert recognized.isdisjoint(unresolved)
    assert recognized | unresolved == set(range(source.molecule.GetNumAtoms()))
    assert all(Chem.MolFromSmiles(f.smiles) is not None for f in unknown.unidentified)
    restored = decompose_smiles(source.smiles, library=library)
    assert restored.records
    assert all(record.smiles == source.smiles for record in restored.records)


def test_custom_chemical_linkers_are_recovered_as_chemical_components():
    monomer = Monomer(
        "ethylene",
        "CHEM",
        "Ethylene linker",
        "[H:1]CC[H:2]",
        (Attachment("R1", "[*:1][H]"), Attachment("R2", "[*:2][H]")),
    )
    library = core_library().extend((monomer,), name="linkers")
    record = from_helm(
        "PEPTIDE1{C.A.C}|CHEM1{[ethylene]}$PEPTIDE1,CHEM1,1:R3-1:R1|PEPTIDE1,CHEM1,3:R3-1:R2$$$V2.0",
        library=library,
    )
    result = decompose_smiles(record.smiles, library=library)
    assert result.records
    assert any(
        any(chain.polymer_type == "CHEM" for chain in r.graph.chains) for r in result.records
    )
    assert all(r.smiles == record.smiles for r in result.records)


@pytest.mark.parametrize(
    "chain",
    [
        "[H:1]CCCCCC[H:2]",
        "[H:1]C[C@@H](C)CC[C@H](F)C[H:2]",
    ],
)
def test_two_ports_of_a_monomer_can_meet_the_same_external_atom(chain):
    attachments = (Attachment("R1", "[*:1][H]"), Attachment("R2", "[*:2][H]"))
    library = core_library().extend(
        (
            Monomer("chain", "CHEM", "Chain", chain, attachments),
            Monomer("nitrogen", "CHEM", "Ammonia", "[H:1]N[H:2]", attachments),
        ),
        name="macrocycles",
    )
    record = from_helm(
        "CHEM1{[chain]}|CHEM2{[nitrogen]}$CHEM1,CHEM2,1:R1-1:R1|CHEM1,CHEM2,1:R2-1:R2$$$V2.0",
        library=library,
    )
    result = decompose_smiles(record.smiles, library=library)
    assert result.records
    assert result.verification_failures == 0
    assert all(r.smiles == record.smiles for r in result.records)


def test_input_atom_maps_do_not_change_the_canonical_residue_graph():
    source = from_sequence("AGK")
    molecule = source.molecule
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(molecule.GetNumAtoms() - atom.GetIdx())
    result = decompose_smiles(Chem.MolToSmiles(molecule))
    assert result.unique
    assert to_helm(result.records[0].graph) == to_helm(
        decompose_smiles(source.smiles).records[0].graph
    )


def test_equivalent_library_symbols_are_reported_as_aliases():
    source = from_helm("PEPTIDE1{[Bmt_E]}$$$$V2.0")
    result = decompose_smiles(source.smiles)
    assert result.records
    assert any("Bmt_E" in m.aliases for m in result.matches if m.symbol == "Bmt")


def test_unseen_combinations_of_modified_monomers_reassemble_exactly():
    library = core_library()
    codes = [
        m.symbol
        for m in library.monomers.values()
        if m.monomer_type == "Backbone" and {a.label for a in m.attachments} >= {"R1", "R2"}
    ]
    rng = random.Random(20261109)
    for index in range(30):
        size = (4, 8, 14)[index % 3]
        symbols = tuple(rng.choices(codes, k=size))
        edges = (
            (Connection(Endpoint("PEPTIDE1", size, "R2"), Endpoint("PEPTIDE1", 1, "R1")),)
            if index % 2
            else ()
        )
        source = assemble(PeptideGraph((Chain("PEPTIDE1", symbols),), edges))
        result = decompose_smiles(source.smiles, timeout=5, max_solutions=4)
        assert result.records, (symbols, result.as_dict())
        assert all(record.smiles == source.smiles for record in result.records)


def test_patent_structures_require_registered_monomers_and_reassemble_with_them():
    path = Path(__file__).parents[1] / "examples/patent_extension.json"
    document = json.loads(path.read_text())
    library = core_library().extend(
        (Monomer.from_helm(m) for m in document["monomers"]), name=document["namespace"]
    )
    for reference in document["references"].values():
        strict = decompose_smiles(reference["smiles"])
        assert strict.status == "partial"
        assert not strict.records and strict.unidentified
        registered = decompose_smiles(reference["smiles"], library=library)
        assert registered.records
        assert registered.coverage == 1
        assert not registered.unidentified
        assert all(record.smiles == reference["smiles"] for record in registered.records)
        assert registered.verification_failures == 0


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"max_states": 1}, "states"),
        ({"max_matches": 1}, "candidates"),
        ({"timeout": 1e-9}, "timeout"),
    ],
)
def test_interrupted_search_does_not_claim_unique_identification(options, reason):
    result = decompose_smiles(from_sequence("AGK").smiles, **options)
    assert result.status == "budget_exceeded"
    assert not result.unique and not result.search_exhausted
    assert reason in result.limits_reached
    assert json.loads(json.dumps(result.as_dict()))["status"] == result.status


def test_a_solution_limit_is_reported_even_when_verified_structures_exist():
    result = decompose_smiles(from_sequence("MENQ").smiles, max_solutions=2)
    assert result.status == "ambiguous"
    assert len(result.records) == 2
    assert not result.search_exhausted
    assert "solutions" in result.limits_reached


def test_unspecified_stereochemistry_is_not_assigned_from_the_library():
    result = decompose_smiles("NC(C)C(=O)O")
    assert not result.records
    assert not result.unique
    assert result.status == "partial"


@pytest.mark.parametrize("source", ["", "*CC", "CC[CH]CC", "invalid", "C[C@H](N)C(=O)O |&1:1|"])
def test_unresolved_or_unsupported_atomic_structures_are_rejected(source):
    with pytest.raises(PeptideError):
        decompose_smiles(source)
