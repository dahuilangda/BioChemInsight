"""Result diffs and revision restore helpers."""

import copy
import hashlib
import json
import math
from pathlib import Path


_IGNORED_FIELDS = {"MEASUREMENT_REVIEWED_AT"}
_SET_FIELDS = {"USER_LOCKED_FIELDS", "USER_REMOVED_MEASUREMENTS"}


def comparable_value(field, value):
    if value is None or value == "" or isinstance(value, float) and math.isnan(value):
        return None
    if field == "MOLBLOCK" and isinstance(value, str):
        lines = value.replace("\r\n", "\n").splitlines()
        return (
            "\n".join(line.rstrip() for line in lines[3:]).rstrip()
            if len(lines) > 3
            else value
        )
    parsed = value
    if isinstance(value, str) and field not in {"COMPOUND_ID", "RECORD_ID", "SMILES"}:
        try:
            parsed = (
                json.loads(value) if value.lstrip().startswith(("{", "[")) else value
            )
        except ValueError:
            pass
    if isinstance(parsed, (list, dict)) and not parsed:
        return None
    if field in _SET_FIELDS and isinstance(parsed, list):
        return sorted({json.dumps(item, sort_keys=True) for item in parsed})
    return str(parsed) if isinstance(parsed, (int, float, bool)) else parsed


def record_values(record):
    if record is None or not (
        record.get("BIOACTIVITY_OBSERVATIONS") or record.get("USER_MEASUREMENT_EDITS")
    ):
        return record
    from utils.measurement_evidence import observation_values, user_measurement_edits

    return {**record, **observation_values(record), **user_measurement_edits(record)}


def changes(before, after):
    def indexed(rows):
        return {
            str(row.get("RECORD_ID") or f"legacy:{index}"): row
            for index, row in enumerate(rows)
        }

    old, new = indexed(before), indexed(after)
    result = []
    for key in dict.fromkeys([*old, *new]):
        a, b = record_values(old.get(key)), record_values(new.get(key))
        fields = []
        for field in sorted(
            (set(a or {}) | set(b or {})) - _IGNORED_FIELDS - {"RECORD_ID"}
        ):
            av, bv = (a or {}).get(field), (b or {}).get(field)
            if comparable_value(field, av) != comparable_value(field, bv):
                fields.append({"field": field, "before": av, "after": bv})
        if a is None or b is None or fields:
            result.append(
                {
                    "record_id": key,
                    "compound_id": (b or a or {}).get("COMPOUND_ID", ""),
                    "operation": "add"
                    if a is None
                    else "delete"
                    if b is None
                    else "update",
                    "fields": fields,
                }
            )
    return result


def digest(rows):
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def read_events(directory, current):
    events = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            item = json.loads(path.read_text())
            if isinstance(item.get("records"), list):
                events.append(item)
        except (OSError, ValueError):
            continue
    for i, event in enumerate(events):
        event.setdefault(
            "after_records",
            events[i + 1]["records"] if i + 1 < len(events) else current,
        )
        if "changes" not in event:
            event["changes"] = changes(event["records"], event["after_records"])
        event.setdefault(
            "actor", {"kind": "unknown", "label": "Legacy edit · author not recorded"}
        )
    return events


def restore_records(
    current, event, *, scope="version", record_id=None, field=None, side="before"
):
    if side not in {"before", "after"}:
        raise ValueError("Choose before or after")
    target = copy.deepcopy(
        event["records"] if side == "before" else event["after_records"]
    )
    if scope == "version":
        return target
    if scope not in {"record", "field"} or not record_id:
        raise ValueError("Choose a record")

    def find(rows):
        return next(
            (
                i
                for i, row in enumerate(rows)
                if str(row.get("RECORD_ID") or f"legacy:{i}") == record_id
            ),
            None,
        )

    rows = copy.deepcopy(current)
    index, saved_index = find(rows), find(target)
    if scope == "record":
        if saved_index is None:
            if index is not None:
                rows.pop(index)
        elif index is None:
            rows.insert(min(saved_index, len(rows)), target[saved_index])
        else:
            rows.pop(index)
            rows.insert(min(saved_index, len(rows)), target[saved_index])
        return rows
    if not field or field == "RECORD_ID" or index is None or saved_index is None:
        raise ValueError(
            "This field cannot be restored independently; restore the record instead"
        )
    if field in {"SMILES", "MOLBLOCK"} or field.startswith("USER_"):
        raise ValueError(
            "Restore the record to keep structure and correction metadata consistent"
        )
    if field in target[saved_index]:
        rows[index][field] = target[saved_index][field]
    else:
        rows[index].pop(field, None)
    # Restore the correction overlay as well as its displayed measurement.
    edits = rows[index].get("USER_MEASUREMENT_EDITS")
    observation_columns = set()
    for row in (rows[index], target[saved_index]):
        try:
            observations = json.loads(row.get("BIOACTIVITY_OBSERVATIONS") or "[]")
            observation_columns.update(
                item.get("column") for item in observations if isinstance(item, dict)
            )
        except (ValueError, TypeError):
            pass
    import re

    is_measurement = field in observation_columns or not re.fullmatch(
        r"[A-Z][A-Z0-9_]*", field
    )
    if edits or is_measurement:
        try:
            values = json.loads(edits) if isinstance(edits, str) else dict(edits or {})
            if field in values or is_measurement:
                values[field] = rows[index].get(field, "")
                rows[index]["USER_MEASUREMENT_EDITS"] = json.dumps(
                    values, ensure_ascii=False
                )
            if is_measurement and field in target[saved_index]:
                removed = json.loads(
                    rows[index].get("USER_REMOVED_MEASUREMENTS") or "[]"
                )
                if field in removed:
                    rows[index]["USER_REMOVED_MEASUREMENTS"] = json.dumps(
                        [v for v in removed if v != field]
                    )
        except (ValueError, TypeError):
            pass
    return rows
