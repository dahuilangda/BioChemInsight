"""Reproducible large-record audit checks; temporary data only, no production jobs."""

import copy
import json
import random
import tempfile
import time
from pathlib import Path
from utils.review_audit import changes, restore_records, read_events

rng = random.Random(713)
rows = [
    {
        "RECORD_ID": f"row-{i}",
        "COMPOUND_ID": str(i),
        **{f"Endpoint {j}": str(i + j) for j in range(12)},
    }
    for i in range(10000)
]
after = copy.deepcopy(rows)
for _ in range(5000):
    index = rng.randrange(len(after))
    field = f"Endpoint {rng.randrange(12)}"
    after[index][field] = str(rng.randrange(1000000))
start = time.perf_counter()
diff = changes(rows, after)
diff_ms = (time.perf_counter() - start) * 1000
expected = {
    row["RECORD_ID"]: {field for field in row if row[field] != after[i][field]}
    for i, row in enumerate(rows)
}
expected = {key: fields for key, fields in expected.items() if fields}
assert len(diff) == len(expected)
for change in diff:
    assert {f["field"] for f in change["fields"]} == expected[change["record_id"]]
event = {"records": rows, "after_records": after}
start = time.perf_counter()
for change in rng.sample(diff, 32):
    field = change["fields"][0]["field"]
    result = restore_records(
        after, event, scope="field", record_id=change["record_id"], field=field
    )
    index = int(change["record_id"].split("-")[1])
    assert result[index][field] == rows[index][field]
    for other in (0, 9999):
        if index != other:
            assert result[other] == after[other]
restore_ms = (time.perf_counter() - start) * 1000
with tempfile.TemporaryDirectory() as folder:
    for i in range(20):
        Path(folder, f"{i:03d}.json").write_text(
            json.dumps({"revision": str(i), "records": rows if i % 2 == 0 else after})
        )
    start = time.perf_counter()
    events = read_events(folder, after)
    history_ms = (time.perf_counter() - start) * 1000
    assert len(events) == 20 and events[0]["after_records"] == after
report = {
    "records": 10000,
    "measurement_columns": 12,
    "seeded_mutations": 5000,
    "changed_records": len(diff),
    "selective_restores": 32,
    "legacy_snapshots": 20,
    "diff_ms": round(diff_ms, 2),
    "selective_restores_total_ms": round(restore_ms, 2),
    "history_read_ms": round(history_ms, 2),
    "passed": True,
}
print(json.dumps(report))
