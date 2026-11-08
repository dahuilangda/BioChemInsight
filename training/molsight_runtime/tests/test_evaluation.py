import pytest
from evaluate import SmilesEvaluator, canonicalize_smiles


def test_invalid_predictions_remain_in_denominator():
    evaluator = SmilesEvaluator(["CC", "CO"], num_workers=1)
    scores = evaluator.evaluate(["CC", "invalid"], include_details=True)
    assert scores["exact"] == 0.5
    assert scores["parse_success"] == 0.5
    assert scores["exact_match"] == [True, False]


@pytest.mark.parametrize("gold", [[""], ["invalid"], []])
def test_invalid_gold_rejected(gold):
    with pytest.raises(ValueError):
        SmilesEvaluator(gold, num_workers=1)


def test_unknown_labels_are_not_replaced_with_wildcards():
    assert canonicalize_smiles("C[Unknown]") == ("", False)


def test_structure_attributes_preserved():
    gold = ["C[*:1]", "[13CH4]", "CC.Cl", "F/C=C/F"]
    pred = ["C[*:2]", "C", "CC", "F/C=C\\F"]
    scores = SmilesEvaluator(gold, num_workers=1).evaluate(pred)
    assert scores["exact"] == 0
    assert scores["ignore_cistrans"] == 0.25


def test_prediction_count_checked():
    with pytest.raises(ValueError):
        SmilesEvaluator(["CC"], num_workers=1).evaluate([])
