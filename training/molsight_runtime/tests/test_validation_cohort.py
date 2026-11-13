from collections import Counter

import pyarrow as pa

from fit import validation_indices


class ValidationRecords:
    def __init__(self):
        self.table = pa.Table.from_pylist(
            [
                {
                    "record_key": key,
                    "component": component,
                    "category": category,
                    "input_size": 512,
                    "peptide_stratum": None,
                }
                for key, component, category in [
                    ("a", "one", "complete"),
                    ("b", "two", "peptide"),
                    ("c", "three", "complete"),
                    ("d", "three", "markush"),
                    ("e", "four", "complete"),
                    ("f", "one", "complete"),
                ]
            ]
        )

    def _open(self):
        pass

    def metadata(self, tags):
        return [
            {"peptide_candidate": index != 0, "cyclic_peptide_candidate": index != 4}
            for index in range(len(self.table))
        ]


def test_discovered_peptides_extend_validation_without_changing_retention_cohort():
    dataset = ValidationRecords()
    limits = {"complete": 1, "peptide": 1}
    original, _ = validation_indices(dataset, limits)
    selected, rows = validation_indices(dataset, limits, "verified_tags.json")
    assert selected[: len(original)] == original
    assert selected == [0, 1, 2, 4]
    assert len({rows[index]["component"] for index in selected}) == len(selected)
    counts = Counter(
        rows[index].get("evaluation_category", rows[index]["category"])
        for index in selected
    )
    assert counts == {"complete": 1, "peptide": 1, "discovered_peptide": 2}
    assert all(row["category"] == "complete" for row in (rows[2], rows[4]))
