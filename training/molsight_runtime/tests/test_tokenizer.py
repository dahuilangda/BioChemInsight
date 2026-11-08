import json

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
