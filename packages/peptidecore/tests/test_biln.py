import pytest

from peptidecore import PeptideError, from_biln, from_helm, parse_biln, to_biln, to_helm


@pytest.mark.parametrize(
    "biln,helm",
    [
        ("P-E-P-T-I-D-E", "PEPTIDE1{P.E.P.T.I.D.E}$$$$V2.0"),
        ("ac-D-T-H-F-E-I-A-am", "PEPTIDE1{[ac].D.T.H.F.E.I.A.[am]}$$$$V2.0"),
        ("C(1,3)-A-A-A-C(1,3)", "PEPTIDE1{C.A.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-5:R3$$$V2.0"),
        ("A(1,1)-G-K(1,2)", "PEPTIDE1{A.G.K}$PEPTIDE1,PEPTIDE1,1:R1-3:R2$$$V2.0"),
        (
            "A-G-Q-A-A-K(1,3)-E-F-I-A-A.G-L-E-E(1,3)",
            "PEPTIDE1{A.G.Q.A.A.K.E.F.I.A.A}|PEPTIDE2{G.L.E.E}$PEPTIDE1,PEPTIDE2,6:R3-4:R3$$$V2.0",
        ),
        (
            "C(1,1)(2,3)-A-C(1,2)(2,3)",
            "PEPTIDE1{C.A.C}$PEPTIDE1,PEPTIDE1,1:R1-3:R2|PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$V2.0",
        ),
        ("[D-Ala-ol]-[L-hArg(Et,Et)]", "PEPTIDE1{[D-Ala-ol].[L-hArg(Et,Et)]}$$$$V2.0"),
        ("[C(4,3)]-A-[C](4,3)", "PEPTIDE1{C.A.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$V2.0"),
    ],
)
def test_biln_and_helm_reconstruct_the_same_structure(biln, helm):
    result = from_biln(biln)
    assert result.smiles == from_helm(helm).smiles
    assert result.source == biln and result.source_format == "BILN"
    assert from_biln(to_biln(result.graph)).smiles == result.smiles
    assert from_helm(to_helm(result.graph)).smiles == result.smiles


@pytest.mark.parametrize(
    "text",
    [
        "",
        "A-",
        "-A",
        "A..G",
        "A(1,3)-C",
        "C(1,3)-C(1,3)-C(1,3)",
        "C(1,0)-C(1,3)",
        "C(1,-1)-C(1,3)",
        "C(1,3)-A(1,3)",
        "C(1,3)-C(1,3)(2,3).C(2,3)",
        "[C](1,3)junk-C(1,3)",
        "A(1,2)-G(1,1)",
        "[A-G",
        "C(0,3)-C(0,3)",
    ],
)
def test_invalid_biln_is_rejected(text):
    with pytest.raises(PeptideError):
        from_biln(text)


def test_biln_export_preserves_connections_with_renamed_chains():
    graph = from_helm("PEPTIDE8{C.A.C}$PEPTIDE8,PEPTIDE8,3:R3-1:R3$$$V2.0").graph
    assert from_biln(to_biln(graph)).smiles == from_helm(to_helm(graph)).smiles
    assert parse_biln(to_biln(graph)).chains[0].identifier == "PEPTIDE1"
