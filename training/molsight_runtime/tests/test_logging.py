import json

import pytest
from molsight.logging import ScalarWriter


def test_training_metrics_are_persisted_without_optional_services(tmp_path):
    writer = ScalarWriter(tmp_path)
    writer.add_scalar("valid/exact", 0.75, 3)
    writer.close()
    records = [
        json.loads(line)
        for line in (tmp_path / "scalars.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["tag"] == "valid/exact"
    assert records[0]["value"] == 0.75
    assert records[0]["step"] == 3


def test_nonfinite_training_metrics_fail(tmp_path):
    writer = ScalarWriter(tmp_path)
    with pytest.raises(ValueError):
        writer.add_scalar("loss", float("nan"), 0)
    writer.close()
