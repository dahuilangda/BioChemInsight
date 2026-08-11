from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BUILDER = ROOT / "training/molnextr_markush/tools/build_fragment_backbone_seeds.py"


def resolve_glob(pattern: str) -> list[Path]:
    text = str(pattern or "").strip()
    if not text:
        return []
    if Path(text).is_absolute():
        return sorted(Path(path) for path in glob.glob(text, recursive=True))
    return sorted(Path(path) for path in glob.glob(str(ROOT / text), recursive=True))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["source_id", "SMILES", "smiles", "fragment_seed_method", "parent_smiles", "parent_source_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def run_shard(
    *,
    source_path: Path,
    shard_dir: Path,
    min_atoms: int,
    max_atoms: int,
    anchor_symbols: str,
    source_row_timeout_seconds: int,
    force: bool,
) -> dict[str, Any]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    output_csv = shard_dir / "seeds.csv"
    report_path = shard_dir / "report.json"
    stdout_path = shard_dir / "stdout.log"
    stderr_path = shard_dir / "stderr.log"
    if not force and output_csv.exists() and report_path.exists():
        report = load_json(report_path)
        return {
            "input": str(source_path),
            "returncode": 0,
            "output_csv": str(output_csv),
            "report": report,
            "report_path": str(report_path),
            "reused_existing": True,
        }
    command = [
        sys.executable,
        str(BUILDER),
        "--input",
        str(source_path),
        "--output-csv",
        str(output_csv),
        "--report",
        str(report_path),
        "--max-source-rows-per-input",
        "0",
        "--max-seeds",
        "0",
        "--min-atoms",
        str(int(min_atoms)),
        "--max-atoms",
        str(int(max_atoms)),
        "--source-row-timeout-seconds",
        str(int(source_row_timeout_seconds)),
    ]
    if anchor_symbols:
        command.extend(["--anchor-symbols", str(anchor_symbols)])
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=str(ROOT), text=True, stdout=stdout, stderr=stderr, check=False)
    report = load_json(report_path) if report_path.exists() else {}
    return {
        "input": str(source_path),
        "returncode": int(completed.returncode),
        "command": command,
        "output_csv": str(output_csv),
        "report": report,
        "report_path": str(report_path),
        "stdout_tail": tail(stdout_path),
        "stderr_tail": tail(stderr_path),
        "reused_existing": False,
    }


def merge_shards(results_by_index: dict[int, dict[str, Any]], output_csv: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    merged_rows: list[dict[str, str]] = []
    seen_backbone: set[str] = set()
    duplicate_rows = 0
    method_counts: Counter[str] = Counter()
    source_column_counts: Counter[str] = Counter()
    source_file_counts: Counter[str] = Counter()
    source_file_row_counts: Counter[str] = Counter()
    source_rows_scanned = 0
    failure_count = 0
    failures: list[dict[str, str]] = []
    shard_summaries: list[dict[str, Any]] = []

    for index in sorted(results_by_index):
        result = results_by_index[index]
        report = result.get("report") if isinstance(result.get("report"), dict) else {}
        source_rows_scanned += int(report.get("source_rows_scanned") or 0)
        failure_count += int(report.get("failure_count") or 0)
        failures.extend(list(report.get("failures") or [])[:50])
        source_column_counts.update({str(k): int(v) for k, v in dict(report.get("source_column_counts") or {}).items()})
        source_file_counts.update({str(k): int(v) for k, v in dict(report.get("source_file_counts") or {}).items()})
        source_file_row_counts.update(
            {str(k): int(v) for k, v in dict(report.get("source_file_row_counts") or {}).items()}
        )
        shard_csv = Path(str(result.get("output_csv") or ""))
        shard_rows = 0
        shard_accepted_after_global_dedupe = 0
        if shard_csv.exists():
            for row in read_csv_rows(shard_csv):
                shard_rows += 1
                backbone = str(row.get("SMILES") or row.get("smiles") or "").strip()
                if not backbone:
                    continue
                if backbone in seen_backbone:
                    duplicate_rows += 1
                    continue
                seen_backbone.add(backbone)
                method_counts.update([str(row.get("fragment_seed_method") or "missing")])
                merged_rows.append(row)
                shard_accepted_after_global_dedupe += 1
        shard_summaries.append(
            {
                "index": int(index),
                "input": result.get("input"),
                "returncode": int(result.get("returncode") or 0),
                "source_rows_scanned": int(report.get("source_rows_scanned") or 0),
                "seed_rows_before_global_dedupe": int(report.get("seed_rows") or shard_rows),
                "seed_rows_after_global_dedupe": int(shard_accepted_after_global_dedupe),
                "failure_count": int(report.get("failure_count") or 0),
                "reused_existing": bool(result.get("reused_existing")),
                "report_path": result.get("report_path"),
                "output_csv": result.get("output_csv"),
            }
        )

    write_csv(output_csv, merged_rows)
    summary = {
        "source_rows_scanned": int(source_rows_scanned),
        "seed_rows": int(len(merged_rows)),
        "duplicate_seed_rows_removed_by_global_backbone_dedupe": int(duplicate_rows),
        "method_counts": dict(sorted(method_counts.items())),
        "source_column_counts": dict(sorted(source_column_counts.items())),
        "source_file_counts": dict(sorted(source_file_counts.items())),
        "source_file_row_counts": dict(sorted(source_file_row_counts.items())),
        "failure_count": int(failure_count),
        "failures": failures[:50],
        "shards": shard_summaries,
    }
    return merged_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full-source fragment backbone seed CSV by running per-source all-row shards in parallel."
    )
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--min-atoms", type=int, default=3)
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--source-row-timeout-seconds", type=int, default=5)
    parser.add_argument("--anchor-symbols", default="")
    parser.add_argument("--force", action="store_true", help="Rebuild shard outputs even when existing reports are present.")
    args = parser.parse_args()

    input_paths = [Path(path) if Path(path).is_absolute() else ROOT / path for path in args.input]
    for pattern in args.input_glob:
        input_paths.extend(resolve_glob(str(pattern)))
    input_paths = list(dict.fromkeys(input_paths))
    if not input_paths:
        raise SystemExit("at least one --input or --input-glob source is required")
    if int(args.jobs) <= 0:
        raise ValueError("--jobs must be positive")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv)
    report_path = Path(args.report)

    results_by_index: dict[int, dict[str, Any]] = {}
    blockers: list[str] = []
    with ThreadPoolExecutor(max_workers=int(args.jobs)) as executor:
        futures = {}
        for index, source_path in enumerate(input_paths):
            shard_dir = work_dir / f"source_{index:05d}"
            future = executor.submit(
                run_shard,
                source_path=source_path,
                shard_dir=shard_dir,
                min_atoms=int(args.min_atoms),
                max_atoms=int(args.max_atoms),
                anchor_symbols=str(args.anchor_symbols or ""),
                source_row_timeout_seconds=int(args.source_row_timeout_seconds),
                force=bool(args.force),
            )
            futures[future] = {"index": index, "source_path": source_path}
            print(
                json.dumps({"event": "scheduled_fragment_seed_source", "index": index, "input": str(source_path)}),
                file=sys.stderr,
                flush=True,
            )
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "input": str(task["source_path"]),
                    "returncode": -1,
                    "error": str(exc),
                    "report": {},
                }
            results_by_index[int(task["index"])] = result
            returncode = int(result.get("returncode") or 0)
            report = result.get("report") if isinstance(result.get("report"), dict) else {}
            if returncode != 0:
                blockers.append(f"source {task['index']} failed with returncode {returncode}: {task['source_path']}")
            print(
                json.dumps(
                    {
                        "event": "finished_fragment_seed_source",
                        "index": int(task["index"]),
                        "input": str(task["source_path"]),
                        "returncode": returncode,
                        "source_rows_scanned": int(report.get("source_rows_scanned") or 0),
                        "seed_rows": int(report.get("seed_rows") or 0),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    _rows, merged = merge_shards(results_by_index, output_csv)
    report = {
        "schema_version": "fragment_backbone_seed_parallel_report_v1",
        "inputs": [str(path) for path in input_paths],
        "input_globs": [str(item) for item in args.input_glob],
        "output_csv": str(output_csv),
        "work_dir": str(work_dir),
        "jobs": int(args.jobs),
        "all_source_rows_requested": True,
        "max_source_rows_per_input": 0,
        "max_seeds": 0,
        "min_atoms": int(args.min_atoms),
        "max_atoms": int(args.max_atoms),
        "source_row_timeout_seconds": int(args.source_row_timeout_seconds),
        "anchor_symbols": [item.strip() for item in str(args.anchor_symbols or "").split(",") if item.strip()],
        **merged,
        "blockers": blockers,
        "policy": {
            "seeds_are_not_trainable_rows": True,
            "no_sampling_or_row_limit": True,
            "invalid_or_unrenderable_seed_candidates_are_filtered": True,
            "global_backbone_dedupe_is_quality_preserving_not_sampling": True,
            "fragment_renderer_must_preserve_molnextr_input_contract": True,
            "source_leak_and_visual_gates_required_before_training": True,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
