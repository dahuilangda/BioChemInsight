from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)


FRAGMENT_TYPES = {"attachment_fragment", "wavy_fragment", "cut_fragment"}
ORDINARY_TYPES = {"complete_compound", "complete_molecule"}
ORDINARY_LABELS = {"complete_molecule"}
ORDINARY_BUCKETS = {"ordinary_structure"}
SUPPORTED_HARD_NEGATIVE_TYPES = {
    "fragment_hard_negative",
    "ordinary_attachment_like_hard_negative",
}


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalized_structure_type(row: dict[str, Any], quality: dict[str, Any]) -> str:
    return str(
        quality.get("structure_type")
        or row.get("structure_type")
        or row.get("structure_type_label")
        or ""
    ).strip()


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return bool(str(value).strip())


def attachment_fields(row: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in sorted(set(row) | set(quality)):
        if "attach" not in key.lower() and "endpoint" not in key.lower() and "anchor" not in key.lower():
            continue
        value = quality.get(key) if key in quality else row.get(key)
        if is_nonempty(value):
            fields[key] = value
    return fields


def inspect_fragment_csv(path: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    counts = Counter()
    mode_counts = Counter()
    side_counts = Counter()
    rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=1):
            rows += 1
            quality = parse_json_object(row.get("render_quality"))
            structure_type = normalized_structure_type(row, quality)
            mode = str(quality.get("attachment_render_mode") or row.get("attachment_render_mode") or "").strip()
            side = str(quality.get("attachment_direction") or row.get("endpoint_side") or "").strip()
            anchor = quality.get("attachment_anchor") or row.get("attachment_anchor")
            endpoint = quality.get("attachment_endpoint") or {
                "x": row.get("endpoint_x") or row.get("attachment_x"),
                "y": row.get("endpoint_y") or row.get("attachment_y"),
            }
            counts[structure_type or "missing"] += 1
            mode_counts[mode or "missing"] += 1
            side_counts[side or "missing"] += 1
            if structure_type not in FRAGMENT_TYPES:
                issues.append(
                    {
                        "row_index": row_index,
                        "severity": "error",
                        "code": "fragment_csv_contains_non_fragment_structure",
                        "message": f"fragment CSV row has structure_type={structure_type!r}",
                    }
                )
            if not is_nonempty(anchor):
                issues.append(
                    {
                        "row_index": row_index,
                        "severity": "error",
                        "code": "fragment_missing_attachment_anchor",
                        "message": "fragment-positive rows require attachment_anchor",
                    }
                )
            if not is_nonempty(endpoint):
                issues.append(
                    {
                        "row_index": row_index,
                        "severity": "error",
                        "code": "fragment_missing_attachment_endpoint",
                        "message": "fragment-positive rows require attachment_endpoint",
                    }
                )
            if side not in {"left", "right", "top", "bottom"}:
                issues.append(
                    {
                        "row_index": row_index,
                        "severity": "error",
                        "code": "fragment_invalid_attachment_side",
                        "message": f"fragment-positive row has endpoint side={side!r}",
                    }
                )
            if mode not in {"wavy", "cut", "query_attachment", "dummy_atom"}:
                issues.append(
                    {
                        "row_index": row_index,
                        "severity": "error",
                        "code": "fragment_invalid_attachment_mode",
                        "message": f"fragment-positive row has attachment mode={mode!r}",
                    }
                )
    return {
        "csv": str(path),
        "role": "fragment_positive",
        "row_count": rows,
        "structure_type_counts": dict(counts),
        "attachment_render_mode_counts": dict(mode_counts),
        "endpoint_side_counts": dict(side_counts),
        "issues": issues[:200],
        "issue_count": len(issues),
        "passed": rows > 0 and not issues,
    }


def inspect_ordinary_csv(path: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    counts = Counter()
    label_counts = Counter()
    bucket_counts = Counter()
    attachment_like_rows = 0
    hard_negative_rows = 0
    rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=1):
            rows += 1
            quality = parse_json_object(row.get("render_quality"))
            structure_type = normalized_structure_type(row, quality)
            label = str(row.get("structure_type_label") or "").strip()
            bucket = str(row.get("structure_type_bucket") or "").strip()
            row_role = str(
                row.get("role")
                or row.get("negative_role")
                or quality.get("negative_role")
                or quality.get("role")
                or ""
            ).strip()
            counts[structure_type or "missing"] += 1
            label_counts[label or "missing"] += 1
            bucket_counts[bucket or "missing"] += 1
            attachment = attachment_fields(row, quality)
            allowed_hard_negative = row_role in SUPPORTED_HARD_NEGATIVE_TYPES
            if allowed_hard_negative:
                hard_negative_rows += 1
            if attachment:
                attachment_like_rows += 1
            if structure_type not in ORDINARY_TYPES:
                issues.append(
                    {
                        "row_index": row_index,
                        "severity": "error",
                        "code": "ordinary_csv_contains_non_ordinary_structure",
                        "message": f"ordinary-negative row has structure_type={structure_type!r}",
                    }
                )
            if label and label not in ORDINARY_LABELS:
                issues.append(
                    {
                        "row_index": row_index,
                        "severity": "error",
                        "code": "ordinary_csv_unexpected_structure_label",
                        "message": f"ordinary-negative row has structure_type_label={label!r}",
                    }
                )
            if bucket and bucket not in ORDINARY_BUCKETS:
                issues.append(
                    {
                        "row_index": row_index,
                        "severity": "error",
                        "code": "ordinary_csv_unexpected_structure_bucket",
                        "message": f"ordinary-negative row has structure_type_bucket={bucket!r}",
                    }
                )
            if attachment and not allowed_hard_negative:
                issues.append(
                    {
                        "row_index": row_index,
                        "severity": "error",
                        "code": "ordinary_negative_contains_attachment_metadata",
                        "message": "ordinary negatives must be attachment-free unless explicitly marked as a supported hard-negative role",
                        "attachment_fields": attachment,
                    }
                )
    return {
        "csv": str(path),
        "role": "ordinary_negative",
        "row_count": rows,
        "structure_type_counts": dict(counts),
        "structure_type_label_counts": dict(label_counts),
        "structure_type_bucket_counts": dict(bucket_counts),
        "attachment_like_rows": attachment_like_rows,
        "explicit_hard_negative_rows": hard_negative_rows,
        "issues": issues[:200],
        "issue_count": len(issues),
        "passed": rows > 0 and not issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate attachment semantics by dataset role.")
    parser.add_argument("--fragment-csv", action="append", default=[])
    parser.add_argument("--ordinary-csv", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fragment_reports = [inspect_fragment_csv(Path(path)) for path in args.fragment_csv]
    ordinary_reports = [inspect_ordinary_csv(Path(path)) for path in args.ordinary_csv]
    reports = fragment_reports + ordinary_reports
    blockers = []
    if not fragment_reports:
        blockers.append("no fragment-positive CSV was provided")
    if not ordinary_reports:
        blockers.append("no ordinary-negative CSV was provided")
    for report in reports:
        if not report["passed"]:
            blockers.append(f"{report['role']} contract failed for {report['csv']}")

    output = {
        "schema_version": "attachment_role_contract_v1",
        "fragment_positive_reports": fragment_reports,
        "ordinary_negative_reports": ordinary_reports,
        "passed": not blockers,
        "blockers": blockers,
        "policy": {
            "ordinary_negative_must_be_attachment_free": True,
            "fragment_positive_must_have_anchor_endpoint_side_and_mode": True,
            "attachment_like_negative_rows_must_be_explicit_hard_negatives": True,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
