from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_candidate_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row.get("source_id") or ""): dict(row) for row in csv.DictReader(handle)}


def manifest_plan_slice_start(manifest: dict[str, Any]) -> int:
    for key in ["candidate_plan_source", "candidate_plan"]:
        value = manifest.get(key)
        if not isinstance(value, dict):
            continue
        for start_key in ["slice_start", "candidate_plan_slice_start"]:
            try:
                return int(value.get(start_key))
            except (TypeError, ValueError):
                continue
    return 0


def globalize_plan_index(row: dict[str, str], *, slice_start: int) -> dict[str, str]:
    output = dict(row)
    local_text = str(row.get("plan_index") or "").strip()
    try:
        local_index = int(local_text)
    except ValueError as exc:
        raise ValueError(f"candidate row {row.get('source_id')} has invalid plan_index={local_text!r}") from exc
    output["local_plan_index"] = str(local_index)
    output["plan_index"] = str(int(slice_start) + local_index)
    return output


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = {name: str(row.get(name) or "") for name in fieldnames}
            output["plan_index"] = str(row.get("plan_index") or index)
            writer.writerow(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract candidate_plan rows for selected Markush generation failure categories.")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--category", choices=["opaque_numeric_exception"], required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    selected_rows: list[dict[str, str]] = []
    selected_keys: set[str] = set()
    missing: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    fieldnames = [
        "plan_index",
        "local_plan_index",
        "source_id",
        "source_file",
        "source_record_id",
        "source_document_key",
        "subset",
        "annotation_r_count",
        "annotation_r_bucket",
        "cxsmiles_star_count",
        "selection_hash",
        "annotation",
        "cxsmiles",
    ]

    for manifest_text in args.manifest:
        manifest_path = Path(manifest_text)
        manifest = load_json(manifest_path)
        slice_start = manifest_plan_slice_start(manifest)
        candidate_rows = read_candidate_rows(manifest_path.parent / "candidate_plan.csv")
        failures = manifest.get("failures") if isinstance(manifest.get("failures"), list) else []
        manifest_selected = 0
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            error = str(failure.get("error") or "").strip()
            if args.category == "opaque_numeric_exception" and not re.fullmatch(r"\d+", error):
                continue
            source_id = str(failure.get("source_id") or "").strip()
            row = candidate_rows.get(source_id)
            if row is None:
                missing.append({"manifest": str(manifest_path), "source_id": source_id, "error": error})
                continue
            row = globalize_plan_index(row, slice_start=slice_start)
            dedupe_key = "|".join(
                [
                    str(row.get("source_id") or source_id),
                    str(row.get("source_record_id") or ""),
                    str(row.get("source_document_key") or ""),
                    str(row.get("plan_index") or ""),
                ]
            )
            if dedupe_key in selected_keys:
                continue
            selected_keys.add(dedupe_key)
            selected_rows.append(row)
            manifest_selected += 1
        source_counts[str(manifest_path)] = manifest_selected

    write_rows(Path(args.output_csv), selected_rows, fieldnames)
    report = {
        "schema_version": "markush_failure_candidate_extract_v1",
        "category": args.category,
        "manifests": [str(Path(value)) for value in args.manifest],
        "output_csv": str(args.output_csv),
        "selected_rows": len(selected_rows),
        "source_manifest_selected_rows": source_counts,
        "missing_count": len(missing),
        "missing": missing[:80],
        "policy": {
            "candidate_rows_only": True,
            "does_not_accept_training_data": True,
            "used_for_controlled_regeneration_after_generator_fix": True,
        },
    }
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_report).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
