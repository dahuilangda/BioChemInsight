from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


csv.field_size_limit(sys.maxsize)


def parse_quality(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("render_quality") or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def literal_star_reasons(quality: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    graph = quality.get("graph_consistency") if isinstance(quality.get("graph_consistency"), dict) else {}
    atom_coordinates = quality.get("atom_coordinates") if isinstance(quality.get("atom_coordinates"), list) else []
    markush = quality.get("markush") if isinstance(quality.get("markush"), dict) else {}
    ocr_cells = markush.get("ocr_cells") if isinstance(markush.get("ocr_cells"), list) else []
    if isinstance(graph.get("visible_star_token_indices"), list) and graph.get("visible_star_token_indices"):
        reasons.append("visible_star_token_indices")
    if any(isinstance(atom, dict) and str(atom.get("token") or "").strip() == "*" for atom in atom_coordinates):
        reasons.append("literal_star_atom_coordinate_token")
    if any(isinstance(cell, dict) and str(cell.get("text") or "").strip() == "*" for cell in ocr_cells):
        reasons.append("literal_star_ocr_cell")
    return reasons


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove accepted Markush rows that still expose literal '*' tokens.")
    parser.add_argument(
        "--root",
        default="training/molnextr_markush/data/generated/pose_factory/molnextr_moe_production_v1/markush",
    )
    parser.add_argument(
        "--report",
        default="training/molnextr_markush/runs/molnextr_moe_production_v1_contracts/markush_literal_star_filter.json",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--backup-root",
        default="training/molnextr_markush/runs/molnextr_moe_production_v1_contracts/literal_star_backups",
        help="Directory for rejected-source backups. Backups are kept outside production generated data.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    csv_paths = sorted(root.glob("s*/accepted_candidate/markush_layout_positive.csv"))
    reason_counts: Counter[str] = Counter()
    shard_reports: list[dict[str, Any]] = []
    rejected_examples: list[dict[str, Any]] = []
    total_rows = 0
    total_rejected = 0
    total_kept = 0

    for csv_path in csv_paths:
        rows, fieldnames = read_rows(csv_path)
        kept: list[dict[str, str]] = []
        rejected: list[dict[str, Any]] = []
        for row in rows:
            total_rows += 1
            quality = parse_quality(row)
            reasons = literal_star_reasons(quality)
            if reasons:
                reason_counts.update(reasons)
                rejected.append(
                    {
                        "source_id": row.get("source_id"),
                        "reasons": reasons,
                        "candidate_plan_index": quality.get("candidate_plan_index"),
                    }
                )
                continue
            kept.append(row)
        total_rejected += len(rejected)
        total_kept += len(kept)
        if rejected:
            shard_reports.append(
                {
                    "csv": str(csv_path),
                    "shard": csv_path.parents[1].name,
                    "input_rows": len(rows),
                    "kept_rows": len(kept),
                    "rejected_rows": len(rejected),
                    "rejected_examples": rejected[:10],
                }
            )
            rejected_examples.extend(
                {"csv": str(csv_path), **example} for example in rejected[: max(0, 20 - len(rejected_examples))]
            )
            if args.execute:
                backup_root = Path(args.backup_root)
                backup_path = backup_root / csv_path.parents[1].name / csv_path.name
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                if not backup_path.exists():
                    shutil.copy2(csv_path, backup_path)
                write_rows(csv_path, fieldnames, kept)

    report = {
        "schema_version": "markush_literal_star_accepted_filter_v1",
        "root": str(root),
        "execute": bool(args.execute),
        "csv_count": len(csv_paths),
        "input_rows": total_rows,
        "kept_rows": total_kept,
        "rejected_rows": total_rejected,
        "reason_counts": dict(sorted(reason_counts.items())),
        "affected_shards": len(shard_reports),
        "backup_root": str(args.backup_root),
        "shards": shard_reports,
        "rejected_examples": rejected_examples[:20],
        "raw_source_data_untouched": True,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report -> {report_path}")
    if total_rejected and not args.execute:
        raise SystemExit("literal star rows found; rerun with --execute to filter accepted derived CSVs")


if __name__ == "__main__":
    main()
