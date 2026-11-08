import json
from dataclasses import FrozenInstanceError

import pytest
from rdkit import Chem

from peptidecore import (
    Attachment,
    Chain,
    Monomer,
    MonomerLibrary,
    PeptideError,
    PeptideGraph,
    assemble,
    core_library,
)
from peptidecore.library import identity, parse_smiles

MONOMERS = tuple(core_library().monomers.values())


@pytest.mark.parametrize("monomer", MONOMERS, ids=lambda m: m.symbol)
def test_every_core_monomer_reassembles_to_its_source_structure(monomer):
    graph = PeptideGraph((Chain("PEPTIDE1", (monomer.symbol,)),))
    result = assemble(graph)
    assert result.smiles == identity(parse_smiles(monomer.smiles))
    assert len(result.atom_instances) == result.molecule.GetNumAtoms()
    assert set(result.atom_instances) == {"PEPTIDE1:1"}
    assert not any(a.GetAtomicNum() == 0 for a in result.molecule.GetAtoms())


def test_core_definitions_and_molecule_templates_cannot_be_changed_by_callers():
    library = core_library()
    assert len(library.monomers) == 322
    assert library.source_revision == "91361a1099df74e08c9cd53e95c708025b321244"
    with pytest.raises(FrozenInstanceError):
        library.digest = "changed"
    with pytest.raises(TypeError):
        library.monomers["PEPTIDE", "A"] = library.get("G")
    monomer = library.get("A")
    first = monomer.template()
    first.GetAtomWithIdx(0).SetIsotope(13)
    assert Chem.MolToSmiles(first) != Chem.MolToSmiles(monomer.template())


def test_library_identity_does_not_depend_on_entry_order():
    forward = MonomerLibrary(MONOMERS, name="first")
    reverse = MonomerLibrary(reversed(MONOMERS), name="second")
    assert forward.digest == reverse.digest


def test_library_snapshot_preserves_definitions_and_checks_its_digest():
    original = core_library()
    restored = MonomerLibrary.from_json(original.to_json())
    assert restored.digest == original.digest
    assert restored.monomers == original.monomers
    assert restored.source_revision == original.source_revision
    document = json.loads(original.to_json())
    document["monomers"][0]["smiles"] = "NCC(=O)O"
    with pytest.raises(PeptideError):
        MonomerLibrary.from_json(json.dumps(document))


def test_custom_chemical_linker_and_duplicate_identifier_validation():
    linker = Monomer(
        "ethylene",
        "CHEM",
        "Ethylene linker",
        "[H:1]CC[H:2]",
        (Attachment("R1", "[*:1][H]"), Attachment("R2", "[*:2][H]")),
    )
    library = core_library().extend((linker,), name="custom")
    assert library.get("ethylene", "CHEM") == linker
    assert library.digest != core_library().digest
    with pytest.raises(PeptideError, match="duplicate_monomer"):
        library.extend((linker,), name="duplicate")


@pytest.mark.parametrize(
    "smiles,attachments,code",
    [
        ("NCC(=O)O", (Attachment("R1", "[*:1][H]"),), "attachment_map_mismatch"),
        (
            "[H:1]NCC(=O)[OH:2]",
            (Attachment("R1", "[*:1][H]"), Attachment("R1", "O[*:1]")),
            "duplicate_attachment",
        ),
        ("[H:1]NC", (Attachment("R1", "O[*:1]"),), "cap_roundtrip_mismatch"),
        ("[H:1]NC", (Attachment("R1", "[H]"),), "invalid_cap"),
    ],
)
def test_invalid_library_definitions_are_rejected(smiles, attachments, code):
    monomer = Monomer("test", "PEPTIDE", "Test", smiles, attachments)
    with pytest.raises(PeptideError, match=code):
        MonomerLibrary((monomer,), name="invalid")
