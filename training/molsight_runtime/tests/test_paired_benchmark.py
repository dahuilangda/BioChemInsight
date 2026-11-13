import json
import hashlib

import pytest

from paired_benchmark import compare, summarize


def test_paired_changes_count_gains_and_losses():
    result = summarize([(True, True), (True, False), (False, True), (False, True)])
    assert result["before_correct"] == 2
    assert result["after_correct"] == 3
    assert result["gained"] == 2 and result["lost"] == 1
    assert result["accuracy_change"] == 0.25
    assert result["mcnemar_exact_two_sided_p"] == 1.0
    assert summarize([(False, True)] * 10)["mcnemar_exact_two_sided_p"] == 2 / 1024


def test_paired_comparison_rejects_misaligned_predictions(tmp_path):
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        json.dumps(
            [
                {
                    "record_key": "key",
                    "component": "component",
                    "SMILES": "CCO",
                    "source": "real",
                    "atom_bin": "1_20",
                }
            ]
        )
    )
    digest = hashlib.sha256(pairs.read_bytes()).hexdigest()
    before, after = [tmp_path / name for name in ("before", "after")]
    for directory in (before, after):
        directory.mkdir()
        (directory / "report.json").write_text(
            json.dumps({"pairs_sha256": digest, "examples": 1, "precision": "fp32"})
        )
        (directory / "predictions.jsonl").write_text(
            json.dumps(
                {"index": 0, "record_key": "key", "gold": "CCO", "resolved_exact": True}
            )
            + "\n"
        )
    assert compare(before, after, pairs)["results"]["all"]["lost"] == 0
    (after / "predictions.jsonl").write_text(
        json.dumps(
            {
                "index": 0,
                "record_key": "different",
                "gold": "CCO",
                "resolved_exact": True,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="identity"):
        compare(before, after, pairs)
