import random

import pytest
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from peptidecore import Attachment, Monomer, core_library, from_helm, from_sequence


def test_standard_sequences_match_rdkit_including_stereochemistry():
    rng = random.Random(20261108)
    sequences = ["ACDEFGIKLMNPQRSTVWY", "C", "I", "T", "P"]
    sequences += [
        "".join(rng.choices("ACDEFGIKLMNPQRSTVWY", k=rng.randrange(2, 25))) for _ in range(50)
    ]
    for sequence in sequences:
        record = from_sequence(sequence)
        expected = Chem.MolFromSequence(sequence)
        assert record.smiles == Chem.MolToSmiles(expected, isomericSmiles=True)
        assert len(record.connections) == len(sequence) - 1


@pytest.mark.parametrize(
    "text",
    [
        "PEPTIDE1{A.G.K}$$$$V2.0",
        "PEPTIDE1{A.[dA].I.[dT].C.P}$$$$V2.0",
        "PEPTIDE1{A.G.K}$PEPTIDE1,PEPTIDE1,3:R2-1:R1$$$V2.0",
        "PEPTIDE1{C.A.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-5:R3$$$V2.0",
        "PEPTIDE1{C.A.C}|PEPTIDE2{C.G.C}$PEPTIDE1,PEPTIDE2,1:R3-3:R3|PEPTIDE1,PEPTIDE2,3:R3-1:R3$$$V2.0",
        "PEPTIDE1{[ac].A.I.P.[am]}$$$$V2.0",
    ],
)
def test_helm_assembly_matches_independent_rdkit_implementation(text):
    record = from_helm(text)
    expected = Chem.MolFromHELM(text)
    assert expected is not None
    assert record.smiles == Chem.MolToSmiles(expected, isomericSmiles=True)
    for connection in record.connections:
        assert record.molecule.GetBondBetweenAtoms(connection.source_atom, connection.target_atom)
        assert (
            record.atom_instances[connection.source_atom]
            == f"{connection.connection.source.chain}:{connection.connection.source.position}"
        )
        assert (
            record.atom_instances[connection.target_atom]
            == f"{connection.connection.target.chain}:{connection.connection.target.position}"
        )


def test_modified_residues_and_terminal_groups_match_explicit_structure():
    result = from_helm("PEPTIDE1{[ac].A.[Aib].[meA].[am]}$$$$V2.0")
    expected = Chem.MolFromSmiles("CC(=O)N[C@@H](C)C(=O)NC(C)(C)C(=O)N(C)[C@@H](C)C(N)=O")
    assert result.smiles == Chem.MolToSmiles(expected)


def test_custom_chemical_linker_joins_two_peptide_side_chains():
    monomer = Monomer(
        "ethylene",
        "CHEM",
        "Ethylene linker",
        "[H:1]CC[H:2]",
        (Attachment("R1", "[*:1][H]"), Attachment("R2", "[*:2][H]")),
    )
    library = core_library().extend((monomer,), name="linkers")
    result = from_helm(
        "PEPTIDE1{C.A.C}|CHEM1{[ethylene]}$PEPTIDE1,CHEM1,1:R3-1:R1|PEPTIDE1,CHEM1,3:R3-1:R2$$$V2.0",
        library=library,
    )
    assert len(Chem.GetMolFrags(result.molecule)) == 1
    assert set(result.atom_instances) == {"PEPTIDE1:1", "PEPTIDE1:2", "PEPTIDE1:3", "CHEM1:1"}
    assert len(result.connections) == 4
    assert sum(a.GetAtomicNum() == 16 for a in result.molecule.GetAtoms()) == 2


def test_isotopes_and_formal_charges_survive_assembly_and_capping():
    monomer = Monomer(
        "isotope",
        "PEPTIDE",
        "Isotope",
        "[H:1]N[13CH2]C(=O)[OH:2]",
        (Attachment("R1", "[*:1][H]"), Attachment("R2", "O[*:2]")),
    )
    charged = Monomer(
        "charged", "CHEM", "Ammonium", "[H:1][N+](C)(C)C", (Attachment("R1", "[*:1][H]"),)
    )
    library = core_library().extend((monomer, charged), name="isotopes")
    result = from_helm("PEPTIDE1{A.[isotope]}|CHEM1{[charged]}$$$$V2.0", library=library)
    assert sum(a.GetIsotope() == 13 for a in result.molecule.GetAtoms()) == 1
    assert Chem.GetFormalCharge(result.molecule) == 1
    assert len(Chem.GetMolFrags(result.molecule)) == 2


def test_record_returns_a_copy_of_its_structure():
    record = from_sequence("AGK")
    molecule = record.molecule
    molecule.GetAtomWithIdx(0).SetIsotope(13)
    assert Chem.MolToSmiles(molecule) != record.smiles
    assert Chem.MolToSmiles(record.molecule) == record.smiles


def test_histidine_uses_the_source_library_tautomer():
    record = from_sequence("H")
    expected = Chem.MolFromSequence("H")
    assert record.smiles == "N[C@@H](Cc1cnc[nH]1)C(=O)O"
    normalizer = rdMolStandardize.TautomerEnumerator()
    normalizer.SetRemoveSp3Stereo(False)
    normalizer.SetRemoveBondStereo(False)
    assert Chem.MolToSmiles(normalizer.Canonicalize(record.molecule)) == Chem.MolToSmiles(
        normalizer.Canonicalize(expected)
    )
