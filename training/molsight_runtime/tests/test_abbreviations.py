import pytest
from rdkit import Chem

from molsight.abbreviations import REGISTRY, resolve_abbreviations
from molsight.structure import parse_structure


@pytest.mark.parametrize("label", REGISTRY)
def test_registry_expansions_match_explicit_port_templates(label):
    expected = REGISTRY[label].replace("*", "Cl", 1)
    result = resolve_abbreviations(f"Cl*<sep><a>1:{label}</a>")
    assert result.replacements == ((1, label),)
    assert result.document.equivalent(parse_structure(expected))
    assert not result.unresolved


@pytest.mark.parametrize(
    "source,expected",
    [
        ("*c1ccccc1<sep><a>0:CF3</a>", "FC(F)(F)c1ccccc1"),
        ("N*<sep><a>1:Boc</a>", "NC(=O)OC(C)(C)C"),
        ("*C[C@@H](F)Cl<sep><a>0:Ph</a>", "c1ccccc1C[C@@H](F)Cl"),
        ("C[C@H](*)Br<sep><a>2:Ph</a>", "C[C@H](c1ccccc1)Br"),
        ("*/C=C/F<sep><a>0:Ph</a>", "c1ccccc1/C=C/F"),
        (
            "*C(*)*<sep><a>0:Ph</a><a>2:Boc</a><d>3:<dum></d>",
            "c1ccccc1C(C(=O)OC(C)(C)C)*<sep><d>14:<dum></d>",
        ),
        ("*c1ccccc1<sep><a>0:Ph</a><r>0:R1</r>", "c1ccccc1-c1ccccc1<sep><r>1:R1</r>"),
        ("N* |$;Boc$|", "NC(=O)OC(C)(C)C"),
    ],
)
def test_defined_expansions_preserve_stereo_and_other_annotations(source, expected):
    result = resolve_abbreviations(source)
    assert result.document.equivalent(parse_structure(expected))
    assert not parse_structure(source).equivalent(result.document)
    assert not resolve_abbreviations(result.document).replacements


@pytest.mark.parametrize(
    "text",
    [
        "*C<sep><a>0:Ar</a>",
        "*C<sep><a>0:Py</a>",
        "*C<sep><a>0:R1</a>",
        "C* |$;NBoc$|",
        "*C<sep><a>0:Tol</a>",
        "*C<sep><c>0:Ph</c>",
        "C*<sep><a>1:NC</a>",
    ],
)
def test_undefined_and_ambiguous_labels_are_preserved(text):
    document = parse_structure(text)
    result = resolve_abbreviations(document)
    assert result.document is document
    assert not result.replacements


@pytest.mark.parametrize(
    "text",
    [
        "C*C<sep><a>1:Ph</a>",
        "C=*<sep><a>1:Ph</a>",
        "C[*:1]<sep><a>1:Ph</a>",
        "C[1*]<sep><a>1:R1</a>",
        "C*<sep><a>1:Ph</a><d>1:<dum></d>",
    ],
)
def test_conflicting_attachment_semantics_are_not_overwritten(text):
    result = resolve_abbreviations(text)
    assert not result.replacements
    assert result.document.equivalent(parse_structure(text))


def test_document_definitions_can_reserve_standard_abbreviation_names():
    result = resolve_abbreviations("C*<sep><a>1:Ph</a>", reserved_labels={"Ph"})
    assert not result.replacements


def test_native_polymer_membership_expands_with_its_abbreviation():
    result = resolve_abbreviations("*CC* |$Ph;;;$,Sg:n:0,1:n:ht:::|")
    groups = Chem.GetMolSubstanceGroups(result.document.molecule)
    assert len(groups) == 1
    assert len(groups[0].GetAtoms()) == 7
    assert result.document.equivalent(parse_structure(result.document.model_text))
