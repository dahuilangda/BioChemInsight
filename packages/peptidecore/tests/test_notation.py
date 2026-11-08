import pytest

from peptidecore import (
    Monomer,
    MonomerLibrary,
    PeptideError,
    core_library,
    from_fasta,
    from_helm,
    from_sequence,
    parse_helm,
    to_helm,
    to_sequence,
)


@pytest.mark.parametrize(
    "text",
    [
        "PEPTIDE1{A.G.K}$$$$V2.0",
        "PEPTIDE1{[ac].A.[dA].[Aib].[meA].[am]}$$$$V2.0",
        "PEPTIDE1{C.A.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$V2.0",
        "PEPTIDE1{K.G}|PEPTIDE2{G.E}$PEPTIDE1,PEPTIDE2,1:R3-2:R3$$$V2.0",
        "PEPTIDE1{[L-hArg(Et,Et)].A}$$$$V2.0",
    ],
)
def test_helm_graph_and_structure_roundtrips(text):
    graph = parse_helm(text)
    assert parse_helm(to_helm(graph)) == graph
    first = from_helm(text)
    second = from_helm(to_helm(first.graph))
    assert first.smiles == second.smiles
    assert first.atom_instances == second.atom_instances
    assert first.source == text


@pytest.mark.parametrize(
    "text,code",
    [
        ("PEPTIDE1{A.G}$$$", "invalid_helm_sections"),
        ("PEPTIDE1{A.G}$$$$V9.0", "unsupported_helm_version"),
        ("PEPTIDE1{A.G}$$G1(PEPTIDE1)$$V2.0", "unsupported_helm_sections"),
        ('PEPTIDE1{A.G}$$${"label":"value"}$V2.0', "unsupported_helm_sections"),
        ("PEPTIDE1{A.G'5'}$$$$V2.0", "unsupported_helm_monomer"),
        ("PEPTIDE1{(A,G).G}$$$$V2.0", "unsupported_helm_monomer"),
        ("PEPTIDE1{[[*:1]NC([*:2])=O]}$$$$V2.0", "unsupported_helm_monomer"),
        ("PEPTIDE1{X.G}$$$$V2.0", "unsupported_helm_monomer"),
        ("PEPTIDE1{A.[unknown]}$$$$V2.0", "unknown_monomer"),
        ("PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE1,1:R2-2:R1$$$V2.0", "attachment_already_used"),
        ("PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE1,3:R2-1:R1$$$V2.0", "invalid_monomer_position"),
        ("PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE1,1:R3-2:R3$$$V2.0", "unknown_attachment"),
        ("PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE2,1:R1-1:R1$$$V2.0", "unknown_chain"),
        ("PEPTIDE1{A}|PEPTIDE1{G}$$$$V2.0", "duplicate_or_missing_chain"),
        ("PEPTIDE1{A.[G}$$$$V2.0", "invalid_helm_syntax"),
        ("PEPTIDE1{A.}$$$$V2.0", "unsupported_helm_monomer"),
        ("CHEM1{[one].[two]}$$$$V2.0", "invalid_chemical_chain"),
        ("PEPTIDE1{[am].A}$$$$V2.0", "unknown_attachment"),
        ("PEPTIDE1{A.[ac]}$$$$V2.0", "unknown_attachment"),
    ],
)
def test_unsupported_or_invalid_helm_is_not_silently_changed(text, code):
    with pytest.raises(PeptideError) as error:
        from_helm(text)
    assert error.value.code == code


def test_fasta_and_plain_sequence_import_preserve_source():
    text = ">example\nACDE\nFGHIK\n"
    record = from_fasta(text)
    assert record.source == text
    assert record.source_format == "FASTA"
    assert to_sequence(record) == "ACDEFGHIK"
    assert record.smiles == from_sequence("acde fghik").smiles


@pytest.mark.parametrize("text", ["", "ACDE", ">one\nAC\n>two\nDE", ">one\n"])
def test_fasta_import_rejects_missing_or_multiple_records(text):
    with pytest.raises(PeptideError):
        from_fasta(text)


@pytest.mark.parametrize(
    "text",
    [
        "PEPTIDE1{A.[dA]}$$$$V2.0",
        "PEPTIDE1{[ac].A}$$$$V2.0",
        "PEPTIDE1{A.G.K}$PEPTIDE1,PEPTIDE1,3:R2-1:R1$$$V2.0",
        "PEPTIDE1{C.A.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$V2.0",
        "PEPTIDE1{A}|PEPTIDE2{G}$$$$V2.0",
    ],
)
def test_sequence_export_rejects_loss_of_chemical_information(text):
    with pytest.raises(PeptideError, match="sequence_information_loss"):
        to_sequence(from_helm(text))


def test_sequence_export_checks_the_structure_of_custom_identifiers():
    glycine = core_library().get("G")
    renamed = Monomer("A", "PEPTIDE", "Custom A", glycine.smiles, glycine.attachments)
    library = MonomerLibrary((renamed,), name="custom")
    record = from_sequence("A", library=library)
    with pytest.raises(PeptideError, match="sequence_information_loss"):
        to_sequence(record)
