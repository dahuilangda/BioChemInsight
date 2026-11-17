"""Extraction scoring and evaluation status tests."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from utils.extraction_metrics import (
    canonicalize_smiles,
    combine_evaluation_statuses,
    score_assay_values,
    score_structure_core,
    score_structure_matches,
)

VALUES = {"A": {"value": "2.0 ± 0.1"}, "B": {"value": ">750"}}
STRUCTURES = {"A": "CCO", "B": "C[C@H](O)C(=O)O"}


def _structure_rows(references=STRUCTURES):
    return [
        {"COMPOUND_ID": identifier, "SMILES": smiles}
        for identifier, smiles in references.items()
    ]


def test_empty_assay_output_fails_complete_reference_set():
    result = score_assay_values({}, VALUES)
    assert result["status"] == "failed"
    assert result["missing"] == ["A", "B"]
    assert result["correct"] == 0 and result["coverage"] == 0


def test_complete_assay_values_pass():
    result = score_assay_values({"A": {"value": "2.0±0.1"}, "B": "> 750"}, VALUES)
    assert result["status"] == "passed" and result["coverage"] == 1


def test_missing_control_value_fails():
    result = score_assay_values({"A": "2.0 ± 0.1"}, VALUES)
    assert result["missing"] == ["B"] and result["status"] == "failed"
    assert result["coverage"] == 0.5


def test_wrong_or_unexpected_assay_value_fails():
    result = score_assay_values({"A": "3", "B": ">750", "unknown": "2"}, VALUES)
    assert result["status"] == "failed"
    assert result["wrong"][0]["id"] == "A"
    assert result["unexpected_keys"] == ["unknown"]


@pytest.mark.parametrize(
    "value", [None, "", " ", True, [], {}, float("nan"), float("inf")]
)
def test_invalid_assay_value_is_not_silently_skipped(value):
    result = score_assay_values({"A": value}, {"A": {"value": "2"}})
    assert result["status"] == "failed" and result["invalid"]


def test_duplicate_assay_ids_fail_without_last_row_overwrite():
    rows = [{"COMPOUND_ID": "A", "value": "wrong"}, {"COMPOUND_ID": "A", "value": "2"}]
    result = score_assay_values(rows, {"A": {"value": "2"}})
    assert result["status"] == "failed" and result["duplicates"] == ["A"]
    assert result["emitted"] == 2 and result["correct"] == 0


def test_conflicting_assay_ids_are_invalid():
    result = score_assay_values(
        {"A": {"COMPOUND_ID": "B", "value": "2"}}, {"A": {"value": "2"}}
    )
    assert result["status"] == "failed" and result["invalid"]
    assert result["correct"] == 0


def test_missing_reference_value_is_inconclusive():
    result = score_assay_values({"A": "2"}, {"A": {}})
    assert result["status"] == "inconclusive" and result["reference_errors"]


def test_empty_reference_set_is_not_a_pass():
    result = score_assay_values({}, {})
    assert result["status"] == "inconclusive" and result["coverage"] is None


def test_complete_structures_pass():
    rows = _structure_rows()
    rows[0]["SMILES"] = "OCC"
    result = score_structure_matches(rows, STRUCTURES)
    assert result["status"] == "passed"
    assert result["matched"] == 2 and result["coverage"] == 1
    assert result["full_structure_accuracy"] == 1


def test_empty_structure_output_fails():
    result = score_structure_matches([], STRUCTURES)
    assert result["status"] == "failed"
    assert result["missing"] == ["A", "B"] and result["coverage"] == 0


def test_partial_structure_output_fails():
    result = score_structure_matches(_structure_rows({"A": "CCO"}), STRUCTURES)
    assert result["status"] == "failed" and result["missing"] == ["B"]
    assert result["coverage"] == 0.5


def test_opposite_enantiomer_fails():
    rows = _structure_rows()
    rows[1]["SMILES"] = "C[C@@H](O)C(=O)O"
    result = score_structure_matches(rows, STRUCTURES)
    assert result["status"] == "failed" and result["wrong"][0]["id"] == "B"
    assert canonicalize_smiles(STRUCTURES["B"]) != canonicalize_smiles(
        rows[1]["SMILES"]
    )


def test_missing_stereochemistry_is_not_equal_to_specified_reference():
    result = score_structure_matches(
        _structure_rows({"B": "CC(O)C(=O)O"}), {"B": STRUCTURES["B"]}
    )
    assert result["status"] == "failed" and result["wrong"]


@pytest.mark.parametrize("smiles", [None, "", "not a molecule", "CCO unwanted-label"])
def test_invalid_structure_is_not_silently_skipped(smiles):
    result = score_structure_matches(_structure_rows({"A": smiles}), {"A": "CCO"})
    assert result["status"] == "failed" and result["invalid"]
    assert result["emitted"] == 1


def test_duplicate_structure_ids_fail():
    rows = _structure_rows({"A": "CN"}) + _structure_rows({"A": "CCO"})
    result = score_structure_matches(rows, {"A": "CCO"})
    assert result["status"] == "failed" and result["duplicates"] == ["A"]
    assert result["matched"] == 0


def test_unexpected_structure_id_fails():
    result = score_structure_matches(
        _structure_rows({"A": "CCO", "X": "CCO"}), {"A": "CCO"}
    )
    assert result["status"] == "failed" and result["unexpected_keys"] == ["X"]


def test_missing_structure_id_is_reported_as_invalid():
    result = score_structure_matches([{"SMILES": "CCO"}], {"A": "CCO"})
    assert result["status"] == "failed" and result["invalid"]
    assert result["missing"] == ["A"]


def test_unavailable_reference_does_not_shrink_denominator():
    result = score_structure_matches(_structure_rows(), {"A": "CCO", "B": None})
    assert result["status"] == "inconclusive"
    assert result["reference_members"] == 2 and result["available_references"] == 1
    assert result["coverage"] == 0.5 and result["output_coverage"] == 1
    assert result["full_structure_accuracy"] is None
    assert [row["id"] for row in result["reference_errors"]] == ["B"]


def test_expected_ids_remain_in_denominator_when_reference_mapping_is_incomplete():
    result = score_structure_matches(
        _structure_rows(), {"A": "CCO"}, expected_ids=["A", "B"]
    )
    assert result["status"] == "inconclusive" and result["reference_members"] == 2
    assert result["coverage"] == 0.5


def test_missing_output_is_reported_even_when_reference_is_unavailable():
    result = score_structure_matches([], {"A": None})
    assert result["status"] == "failed" and result["missing"] == ["A"]
    assert result["reference_errors"] and result["coverage"] == 0


def test_core_match_is_not_full_structure_success():
    result = score_structure_core(_structure_rows({"A": "c1ccccc1"}), ["A"], "c1ccccc1")
    assert result["core_status"] == "passed"
    assert (
        result["status"] == "inconclusive" and result["full_structure_accuracy"] is None
    )
    assert result["coverage"] == 1


def test_empty_core_output_still_reports_missing_members():
    result = score_structure_core([], ["A", "B"], "c1ccccc1")
    assert result["status"] == "failed" and result["core_status"] == "failed"
    assert result["missing"] == ["A", "B"] and result["coverage"] == 0


def test_core_mismatch_fails():
    result = score_structure_core(_structure_rows({"A": "CCO"}), ["A"], "c1ccccc1")
    assert result["status"] == "failed" and result["violations"][0]["id"] == "A"


def test_invalid_core_reference_is_inconclusive():
    result = score_structure_core(_structure_rows({"A": "CCO"}), ["A"], "[")
    assert result["status"] == result["core_status"] == "inconclusive"
    assert result["reference_errors"]


@pytest.mark.parametrize("expected", [["A", "A"], [""], [None]])
def test_invalid_expected_id_set_is_rejected(expected):
    with pytest.raises(ValueError):
        score_structure_matches([], {}, expected_ids=expected)


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([], "inconclusive"),
        (["passed"], "passed"),
        (["passed", "inconclusive"], "inconclusive"),
        (["inconclusive", "failed"], "failed"),
    ],
)
def test_combined_status_requires_all_checks_to_pass(statuses, expected):
    assert combine_evaluation_statuses(statuses) == expected


def test_unknown_evaluation_status_is_rejected():
    with pytest.raises(ValueError):
        combine_evaluation_statuses(["completed"])


@pytest.fixture
def literature_evaluation():
    path = Path(__file__).resolve().parents[1] / "evals" / "literature_eval.py"
    spec = importlib.util.spec_from_file_location("literature_evaluation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_resolution_failure_is_inconclusive(
    literature_evaluation, monkeypatch
):
    def resolve(name):
        return (
            (Chem.MolFromSmiles("CCO"), "")
            if name == "ethanol"
            else (None, "unresolved")
        )

    monkeypatch.setattr(literature_evaluation, "name_to_molecule", resolve)
    truth = {
        "series_id": "1-2",
        "members": {"1": {"name": "ethanol"}, "2": {"name": "unknown"}},
    }
    result = literature_evaluation.judge_structure_benzimidazole(
        _structure_rows({"1": "CCO", "2": "C[C@H](O)C(=O)O"}), truth
    )
    assert result["status"] == "inconclusive" and result["reference_members"] == 2
    assert result["reference_resolution_errors"][0]["id"] == "2"


def test_literature_core_check_does_not_claim_full_accuracy(literature_evaluation):
    truth = {
        "series_id": "9a-b",
        "members": {"9a": {}, "9b": {}},
        "core_smarts": "c1ccccc1",
    }
    result = literature_evaluation.judge_structure_quinoline(
        _structure_rows({"9a": "c1ccccc1", "9b": "Cc1ccccc1"}), truth
    )
    assert (
        result["status"] == "inconclusive" and result["full_structure_accuracy"] is None
    )
    assert result["expected_members"] == 2


@pytest.mark.parametrize(
    "assay_status,structure_status,expected",
    [
        ("passed", "passed", "passed"),
        ("failed", "passed", "failed"),
        ("passed", "inconclusive", "inconclusive"),
    ],
)
def test_literature_entry_point_uses_grader_statuses(
    literature_evaluation,
    monkeypatch,
    tmp_path,
    assay_status,
    structure_status,
    expected,
):
    monkeypatch.setattr(sys, "argv", ["literature_eval.py", "1", "benzimidazole"])
    monkeypatch.setattr(literature_evaluation, "ARTIFACTS", str(tmp_path))
    monkeypatch.setattr(
        literature_evaluation, "run_assay_eval", lambda *args: {"status": assay_status}
    )
    monkeypatch.setattr(
        literature_evaluation,
        "run_structure_eval",
        lambda *args: {
            "status": structure_status,
            "coverage": 1 if structure_status == "passed" else 0,
        },
    )
    result = literature_evaluation.main()
    report = json.loads((tmp_path / "runs.jsonl").read_text())
    assert report["status"] == expected
    assert report["pass"] is (expected == "passed")
    assert report["papers"]["benzimidazole"]["pass"] is (expected == "passed")
    assert result == (0 if expected == "passed" else 1)


def test_unannotated_member_name_does_not_shrink_expected_set(
    literature_evaluation, monkeypatch
):
    resolved = []

    def resolve(name):
        resolved.append(name)
        return Chem.MolFromSmiles("CCO"), ""

    monkeypatch.setattr(literature_evaluation, "name_to_molecule", resolve)
    truth = {"series_id": "1-2", "members": {"1": {"name": "ethanol"}, "2": {}}}
    rows = _structure_rows({"1": "CCO", "2": "CN"})
    result = literature_evaluation.judge_structure_benzimidazole(rows, truth)
    assert result["status"] == "inconclusive"
    assert result["reference_members"] == 2 and result["available_references"] == 1
    assert result["coverage"] == 0.5 and result["full_structure_accuracy"] is None
    assert resolved == ["ethanol"]


def test_reference_resolver_exception_is_inconclusive(
    literature_evaluation, monkeypatch
):
    def unavailable(name):
        raise RuntimeError("reference service unavailable")

    monkeypatch.setattr(literature_evaluation, "name_to_molecule", unavailable)
    truth = {
        "series_id": "1-2",
        "members": {"1": {"name": "ethanol"}, "2": {"name": "amine"}},
    }
    result = literature_evaluation.judge_structure_benzimidazole(
        _structure_rows({"1": "CCO", "2": "CN"}), truth
    )
    assert result["status"] == "inconclusive" and result["reference_members"] == 2
    assert (
        result["available_references"] == 0
        and result["full_structure_accuracy"] is None
    )


def test_unannotated_structure_scope_is_inconclusive(literature_evaluation):
    result = literature_evaluation.judge_structure_benzimidazole(
        _structure_rows(), {"members": {}}
    )
    assert result["status"] == "inconclusive"
    assert (
        result["expected_members"] is None and result["full_structure_accuracy"] is None
    )


def test_core_check_respects_annotated_stereochemistry():
    result = score_structure_core(
        _structure_rows({"A": "C[C@@H](O)C(=O)O"}), ["A"], "C[C@H](O)C(=O)O"
    )
    assert result["core_status"] == result["status"] == "failed"
    assert result["full_structure_accuracy"] is None


def test_zero_structure_trials_are_rejected(literature_evaluation, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["literature_eval.py", "0", "benzimidazole"])
    with pytest.raises(ValueError):
        literature_evaluation.main()
