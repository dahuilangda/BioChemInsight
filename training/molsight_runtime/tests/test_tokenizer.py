import json
from pathlib import Path

import pytest
from molsight.tokenizer import CharTokenizer


@pytest.fixture
def tokenizer(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(
        json.dumps({"<pad>": 0, "<sos>": 1, "<eos>": 2, "<mask>": 3, "C": 4, "*": 5})
    )
    return CharTokenizer(path)


def test_unknown_character_rejected(tokenizer):
    with pytest.raises(ValueError, match="vocabulary"):
        tokenizer.smiles_to_sequence("CN")


def test_dropped_input_rejected(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.smiles_to_sequence("C C")


def test_tokenizer_preserves_known_input(tokenizer):
    sequence, indices = tokenizer.smiles_to_sequence("CC")
    assert sequence == [1, 4, 4, 2]
    assert indices == [1, 2]


@pytest.mark.parametrize(
    "text",
    [
        "CC*<sep><a>2:R[1]</a>",
        "c1ccccc1<sep><r>0:R[2]</r>",
        "*CC<sep><a>0:<dum></a>",
        "CC*<sep><a>2:NHAc</a>|Sg:n|",
    ],
)
def test_extended_labels_round_trip_without_extra_atoms(text):
    tokenizer = CharTokenizer(Path(__file__).parents[1] / "vocab/vocab_chars.json")
    sequence, indices = tokenizer.smiles_to_sequence(text)
    decoded = tokenizer.sequence_to_smiles(sequence[1:])
    assert decoded["smiles"] == text
    assert decoded["base_smiles"] == text.split("<sep>")[0]
    assert decoded["indices"] == [index - 1 for index in indices]
    assert len(indices) == len(decoded["atoms"])
    assert len(indices) == (6 if text.startswith("c1") else 3)


def test_mask_uses_vocabulary_id():
    tokenizer = CharTokenizer(Path(__file__).parents[1] / "vocab/vocab_chars.json")
    assert tokenizer.stoi["<mask>"] == 4
    sequence, _ = tokenizer.smiles_to_sequence("CC", mask_ratio=1)
    assert sequence == [1, 4, 4, 2]


def test_generated_padding_does_not_turn_truncation_into_a_valid_molecule(tokenizer):
    assert tokenizer.sequence_to_smiles([4, 0, 2])["smiles"] == "C<pad>"


def test_native_cxsmiles_tokenization_keeps_only_base_atom_indices(tmp_path):
    vocabulary = json.loads(
        (Path(__file__).parents[1] / "vocab/vocab_chars.json").read_text()
    )
    vocabulary[" "] = len(vocabulary)
    path = tmp_path / "extended.json"
    path.write_text(json.dumps(vocabulary))
    tokenizer = CharTokenizer(path)
    text = "*CCCO* |Sg:n:1,2:n:ht:::|"
    sequence, indices = tokenizer.smiles_to_sequence(text)
    decoded = tokenizer.sequence_to_smiles(sequence[1:])
    assert decoded["smiles"] == text
    assert decoded["base_smiles"] == "*CCCO*"
    assert len(indices) == len(decoded["atoms"]) == 6
