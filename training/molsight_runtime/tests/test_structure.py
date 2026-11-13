import json
from pathlib import Path

import pytest
from molsight.structure import StructureError, parse_structure, structure_equal


@pytest.mark.parametrize(
    "first,second",
    [
        ("CC", "CC<sep>"),
        ("C", "[H]C([H])([H])[H]"),
        ("*CO<sep><a>0:R[1]</a>", "OC*<sep><a>2:R1</a>"),
        ("C[1*]", "*C<sep><a>0:R[1]</a>"),
        ("*C<sep><a>0:</a>", "*C<sep><d>0:<dum></d>"),
        ("[*:1]C", "C[*:1]<sep><d>1:<dum></d>"),
        (
            "C[C@H](*)*<sep><a>2:R[1]</a><a>3:R[2]</a>",
            "C[C@@H](*)*<sep><a>2:R[2]</a><a>3:R[1]</a>",
        ),
        ("c1ccccc1-c1ccccn1<sep><r>0:R[1]</r>", "n1ccccc1-c1ccccc1<sep><r>1:R[1]</r>"),
        (
            "*Cc1ccccc1<sep><c>0:Ar</c><r><c>0:R[1]</r>",
            "c1ccccc1C*<sep><r><c>7:R[1]</r><c>7:Ar</c>",
        ),
    ],
)
def test_structure_equivalence_retains_annotations_and_remaps_indices(first, second):
    assert structure_equal(first, second)
    assert structure_equal(second, first)


@pytest.mark.parametrize(
    "first,second",
    [
        ("C[*:1]", "C[1*]"),
        ("[13CH4]", "C"),
        ("CC.Cl", "CC"),
        ("*CO<sep><a>0:R[1]</a>", "*CO<sep><a>0:R[2]</a>"),
        ("*CO<sep><a>0:R[1]</a>", "*CO<sep><a>2:R[1]</a>"),
        ("*C<sep><c>0:Ar</c>", "*C<sep><a>0:Ar</a>"),
        ("c1ccccc1<sep><r>0:R[1]?n</r>", "c1ccccc1<sep><r>0:R[1]?m</r>"),
        ("c1ccccc1<sep><r>0:<dum></r>", "c1ccccc1<sep>"),
        (
            "*CC*<sep><a>0:<dum></a><a>3:<dum></a>|Sg:n|",
            "*CC*<sep><a>0:<dum></a><a>3:<dum></a>|Sg:2|",
        ),
        (
            "C[C@H](*)*<sep><a>2:R[1]</a><a>3:R[2]</a>",
            "C[C@@H](*)*<sep><a>2:R[1]</a><a>3:R[2]</a>",
        ),
        ("F/C=C/F", "F/C=C\\F"),
    ],
)
def test_structure_equivalence_rejects_chemical_and_semantic_changes(first, second):
    assert not structure_equal(first, second)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "C C",
        "C<sep><a>3:R</a>",
        "c1ccccc1<sep><r>5:R</r>",
        "C<sep><d>0:<dum></d>",
        "*C<sep><a>0:R</r>",
        "*C<sep><r><c>0:R[1]</r>",
        "*C<sep><a>0:R</a>unparsed",
    ],
)
def test_invalid_records_fail_without_dropping_information(text):
    with pytest.raises(StructureError):
        parse_structure(text)


def test_explicit_hydrogens_do_not_shift_dataset_annotation_indices():
    text = "[C@@]12([H])CN(*)C[C@]1([H])CNC2<sep><a>3:<dum></a>"
    document = parse_structure(text)
    assert document.molecule.GetAtomWithIdx(3).GetAtomicNum() == 0
    assert structure_equal(text, document.to_esmiles())


def test_atom_and_double_bond_stereo_are_independent_evaluation_views():
    first = parse_structure("N[C@H](C)/C=C/F")
    second = parse_structure("N[C@H](C)/C=C\\F")
    enantiomer = parse_structure("N[C@@H](C)/C=C/F")
    assert first.without_stereo().equivalent(second.without_stereo())
    assert not first.without_stereo().equivalent(enantiomer.without_stereo())
    assert first.without_stereo(atoms=True).equivalent(
        enantiomer.without_stereo(atoms=True)
    )


def test_native_cxsmiles_named_variables_match_esmiles():
    assert structure_equal("C* |atomProp:1.dummyLabel.E|", "C*<sep><a>1:E</a>")
    assert structure_equal("[1*]CC |$X;;$|", "*CC<sep><a>0:X</a>")
    assert not structure_equal("[1*]CC |$X;;$|", "[1*]CC")


def test_native_repeat_membership_and_label_are_chemical_information():
    first = parse_structure("*CCCO* |Sg:n:1,2:n:ht:::|")
    assert structure_equal(first.model_text, "*CCCO* |Sg:n:1,2:n:ht:::|")
    assert not structure_equal(first.model_text, "*CCCO* |Sg:n:2,3:n:ht:::|")
    assert not structure_equal(first.model_text, "*CCCO* |Sg:n:1,2:m:ht:::|")
    assert not structure_equal(first.model_text, "*CCCO*")


def test_native_repeat_indices_follow_atom_permutations():
    first = parse_structure("*CCCO* |Sg:n:1,2:n:ht:::|")
    assert structure_equal(first.model_text, "*OCCC* |Sg:n:3,4:n:ht:::|")


def test_native_coordinates_are_not_part_of_chemical_identity():
    assert structure_equal(
        "C* |(0,0,;1,0,),atomProp:1.dummyLabel.E|", "C* |atomProp:1.dummyLabel.E|"
    )


def test_variable_attachment_sites_survive_serialization():
    source = "C*.c1ncc(O)cc1 |m:1:2.3|"
    document = parse_structure(source)
    assert document.variation_records
    assert document.equivalent(parse_structure(document.model_text))
    assert not structure_equal(source, "C*.c1ncc(O)cc1")
    assert not structure_equal(source, "C*.c1ncc(O)cc1 |m:1:4.5|")


def test_atom_query_constraints_are_not_dropped_from_identity():
    assert not structure_equal("CC |s:0:2|", "CC")
    document = parse_structure("CC |s:0:2|")
    with pytest.raises(StructureError, match="serialization_changed_structure"):
        document.model_text


def test_position_variation_compares_query_atoms_with_query_atoms():
    source = "C*.c1ccccc1 |$;X_p;;;;;;$,m:1:2.3|"
    document = parse_structure(source)
    assert document.equivalent(parse_structure(document.model_text))


def test_abbreviated_polymer_record_is_delimited_before_position_variation():
    document = parse_structure("CC(=O)N.C*.c1ccccc1 |Sg:n:1,2:n:ht,m:5:6.7|")
    assert document.equivalent(parse_structure(document.model_text))


@pytest.mark.parametrize(
    "record",
    json.loads((Path(__file__).parent / "fixtures/markush_queries.json").read_text()),
)
def test_native_aromatic_query_serialization_preserves_the_source_graph(record):
    document = parse_structure(record["label"])
    assert document.equivalent(parse_structure(document.model_text))
    assert " |(" not in document.model_text
