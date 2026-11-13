import pytest
from molsight.source_labels import cxsmiles_to_esmiles, parse_source_label
from molsight.structure import StructureError, structure_equal


def test_source_variables_keep_visible_names():
    source = "C*OC(OCC)C(=N)OC |atomProp:1.dummyLabel.E|"
    assert cxsmiles_to_esmiles(source) == "C*OC(OCC)C(=N)OC<sep><a>1:E</a>"
    assert not structure_equal(cxsmiles_to_esmiles(source), "C[1*]OC(OCC)C(=N)OC")


def test_atom_order_and_labels_are_preserved():
    source = "*CC* |atomProp:0.dummyLabel.Rr:3.dummyLabel.L30|"
    assert cxsmiles_to_esmiles(source) == "*CC*<sep><a>0:Rr</a><a>3:L30</a>"


def test_coordinate_fields_do_not_change_chemical_labels():
    assert (
        cxsmiles_to_esmiles("C* |(0,0,;1,0,),atomProp:1.dummyLabel.E|")
        == "C*<sep><a>1:E</a>"
    )


@pytest.mark.parametrize(
    "source",
    [
        "*CC* |atomProp:0.dummyLabel.R1,Sg:n:1,2:n:ht:::|",
        "*CC* |atomProp:0.custom.value|",
        "C* |atomProp:1.dummyLabel.E",
    ],
)
def test_unrepresented_semantics_are_rejected(source):
    with pytest.raises(StructureError):
        cxsmiles_to_esmiles(source)


def test_markushgrapher_position_variation_is_preserved():
    source = "c1ncc(O)cc1.<r>R</r>C|m:8:0.1.2"
    document = parse_source_label(source, notation="markushgrapher")
    assert document.variation_records
    assert "m:" in document.model_text
    assert structure_equal(document.model_text, document.model_text)


def test_named_variables_and_repeat_units_survive_position_variation():
    source = "<r>Rr</r>=c1ccc2c(c(O)ccc2CCNS)<r>L30</r>1.<r>R</r>C.<r>R0</r>C|Sg:n:11,12,13,14:Q:ht,m:17:1.2.3.13.14.15,m:19:3.4.6.7.8.13"
    document = parse_source_label(source, notation="markushgrapher")
    from molsight.structure import parse_structure

    recovered = parse_structure(document.model_text)
    assert document.equivalent(recovered)
    assert len(recovered.variation_records) == 2
    assert not any(atom.HasQuery() for atom in document.molecule.GetAtoms())


@pytest.mark.parametrize(
    "source",
    [
        "O=<r>Ri</r>1ccc2c(cc<r>Y0</r>2)n1",
        "<r>Rt</r>=<r>Rz</r>1<r>T5</r>N=<r>B28</r>2cc<r>R9</r>:<r>W9</r>c2=C1C(=O)O",
        "<r>E1</r>C.O=<r>Z0</r>1ncc2c<r>Z</r>(c3<r>Rl</r>cc<r>V4</r>3)ccc21|m:1:6.7.8.14.15.16",
    ],
)
def test_aromatic_variables_use_a_lossless_bond_notation(source):
    from molsight.structure import parse_structure

    document = parse_source_label(source, notation="markushgrapher")
    recovered = parse_structure(document.model_text)
    assert document.equivalent(recovered)
    assert document.model_text == recovered.model_text
