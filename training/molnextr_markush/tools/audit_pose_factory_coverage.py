from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("render_quality") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def bucket_count(value: int) -> str:
    if value <= 5:
        return "1-5"
    if value <= 12:
        return "6-12"
    if value <= 25:
        return "13-25"
    if value <= 50:
        return "26-50"
    return "51+"


def markush_r_count_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    if value <= 8:
        return "5-8"
    return "9+"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit coverage of generated pose-factory shards.")
    parser.add_argument("--csv", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    counters: dict[str, Counter[str]] = {
        "source_arrow": Counter(),
        "structure_type": Counter(),
        "backend": Counter(),
        "render_style": Counter(),
        "coord_policy": Counter(),
        "atom_count_bucket": Counter(),
        "bond_count_bucket": Counter(),
        "endpoint_side": Counter(),
        "attachment_render_mode": Counter(),
        "attachment_render_geometry": Counter(),
        "markush_cell_count_bucket": Counter(),
        "markush_r_tag_count_bucket": Counter(),
        "augmentations": Counter(),
    }
    rows_total = 0
    for csv_text in args.csv:
        csv_path = Path(csv_text)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows_total += 1
                quality = parse_quality(row)
                graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
                counters["source_arrow"].update([str(row.get("source_arrow") or "missing")])
                counters["structure_type"].update([str(quality.get("structure_type") or "missing")])
                counters["backend"].update([str(quality.get("backend") or "missing")])
                counters["render_style"].update([str(quality.get("render_style") or "missing")])
                counters["coord_policy"].update([str(quality.get("coord_policy") or "missing")])
                counters["atom_count_bucket"].update([bucket_count(int(graph.get("atom_count") or 0))])
                counters["bond_count_bucket"].update([bucket_count(int(graph.get("bond_count") or 0))])
                counters["endpoint_side"].update([str(quality.get("attachment_direction") or "none")])
                counters["attachment_render_mode"].update([str(quality.get("attachment_render_mode") or "none")])
                counters["attachment_render_geometry"].update([str(quality.get("attachment_render_geometry") or "none")])
                cells = quality.get("ocr_cells") if isinstance(quality.get("ocr_cells"), list) else []
                if not cells and isinstance(quality.get("markush"), dict):
                    nested_cells = quality["markush"].get("ocr_cells")
                    cells = nested_cells if isinstance(nested_cells, list) else []
                markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
                counters["markush_r_tag_count_bucket"].update([markush_r_count_bucket(int(markush.get("r_tag_count") or 0))])
                counters["markush_cell_count_bucket"].update([bucket_count(len(cells)) if cells else "none"])
                augmentations = quality.get("style_augmentations") if isinstance(quality.get("style_augmentations"), list) else []
                if augmentations:
                    counters["augmentations"].update(str(item) for item in augmentations)
                else:
                    counters["augmentations"].update(["none"])

    report = {
        "row_count": rows_total,
        "csv": args.csv,
        "coverage": {name: dict(counter.most_common()) for name, counter in counters.items()},
        "status": "coverage_audit_only_not_acceptance",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
