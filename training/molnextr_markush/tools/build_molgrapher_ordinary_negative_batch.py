from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CONVERTER = ROOT / "training/molnextr_markush/tools/convert_molgrapher_pose_shard.py"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty ordinary-negative batch")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run_converter(
    *,
    parquet_path: Path,
    shard_dir: Path,
    start_row: int,
    source_rows: int,
    max_rows: int,
    heldout_csv: list[str],
    heldout_image_csv: list[str],
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(CONVERTER),
        "--parquet",
        str(parquet_path),
        "--output-dir",
        str(shard_dir),
        "--start-row",
        str(start_row),
        "--max-source-rows",
        str(source_rows),
        "--max-rows",
        str(max_rows),
    ]
    for csv_path in heldout_csv:
        command.extend(["--heldout-csv", csv_path])
    for csv_path in heldout_image_csv:
        command.extend(["--heldout-image-csv", csv_path])
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    manifest_path = shard_dir / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "manifest": manifest,
        "manifest_path": str(manifest_path),
    }


def copy_rows_with_images(shard_csv: Path, output_dir: Path, merged_count: int, limit: int) -> tuple[list[dict[str, str]], int]:
    rows = []
    copied = 0
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for row in read_rows(shard_csv):
        if 0 <= limit <= len(rows):
            break
        source_image = Path(str(row.get("file_path") or ""))
        if not source_image.is_absolute():
            source_image = shard_csv.parent / source_image
        if not source_image.exists():
            raise FileNotFoundError(f"missing converted image: {source_image}")
        target_image = image_dir / f"{merged_count + len(rows):08d}_{source_image.name}"
        target_image.write_bytes(source_image.read_bytes())
        copied += 1
        merged_row = dict(row)
        merged_row["file_path"] = str(Path("images") / target_image.name)
        rows.append(merged_row)
    return rows, copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MolGrapher ordinary-negative rows with strict parallel shard conversion.")
    parser.add_argument("--parquet", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-rows", type=int, required=True)
    parser.add_argument("--source-window", type=int, default=5000)
    parser.add_argument("--max-accepted-per-window", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--heldout-csv", action="append", default=[])
    parser.add_argument("--heldout-image-csv", action="append", default=[])
    args = parser.parse_args()

    if int(args.target_rows) <= 0:
        raise ValueError("--target-rows must be positive")
    if int(args.source_window) <= 0:
        raise ValueError("--source-window must be positive")
    if int(args.jobs) <= 0:
        raise ValueError("--jobs must be positive")

    output_dir = Path(args.output_dir)
    shard_root = output_dir / "shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for parquet_text in args.parquet:
        parquet_path = Path(parquet_text)
        total_rows = parquet_rows(parquet_path)
        for start in range(0, total_rows, int(args.source_window)):
            tasks.append(
                {
                    "parquet_path": parquet_path,
                    "start_row": start,
                    "source_rows": min(int(args.source_window), total_rows - start),
                }
            )

    merged_rows: list[dict[str, str]] = []
    copied_images = 0
    shard_reports: list[dict[str, Any]] = []
    stop_scheduling = False
    task_index = 0
    with ThreadPoolExecutor(max_workers=int(args.jobs)) as executor:
        in_flight = {}
        while (task_index < len(tasks) and not stop_scheduling) or in_flight:
            while task_index < len(tasks) and not stop_scheduling and len(in_flight) < int(args.jobs):
                task = tasks[task_index]
                shard_dir = shard_root / f"molgrapher_ordinary_{task_index:05d}"
                future = executor.submit(
                    run_converter,
                    parquet_path=task["parquet_path"],
                    shard_dir=shard_dir,
                    start_row=int(task["start_row"]),
                    source_rows=int(task["source_rows"]),
                    max_rows=int(args.max_accepted_per_window),
                    heldout_csv=list(args.heldout_csv),
                    heldout_image_csv=list(args.heldout_image_csv),
                )
                in_flight[future] = {"task_index": task_index, "shard_dir": shard_dir, **task}
                log_progress(
                    f"scheduled shard={task_index} parquet={task['parquet_path'].name} "
                    f"start={task['start_row']} source_rows={task['source_rows']}"
                )
                task_index += 1
            if not in_flight:
                break
            for future in as_completed(list(in_flight)):
                task = in_flight.pop(future)
                result = future.result()
                report = {"task": {key: str(value) for key, value in task.items()}, **result}
                shard_reports.append(report)
                if int(result.get("returncode") or 0) != 0:
                    stop_scheduling = True
                    continue
                shard_csv = Path(str(result["manifest"].get("csv") or ""))
                if shard_csv.exists() and len(merged_rows) < int(args.target_rows):
                    remaining = int(args.target_rows) - len(merged_rows)
                    new_rows, new_images = copy_rows_with_images(shard_csv, output_dir, len(merged_rows), remaining)
                    merged_rows.extend(new_rows)
                    copied_images += new_images
                log_progress(
                    f"finished shard={task['task_index']} returncode={result.get('returncode')} "
                    f"merged_rows={len(merged_rows)} target={int(args.target_rows)}"
                )
                if len(merged_rows) >= int(args.target_rows):
                    stop_scheduling = True
                    for pending in in_flight:
                        pending.cancel()
                break

    output_csv = output_dir / "ordinary_negative.csv"
    if merged_rows:
        write_rows(output_csv, merged_rows)
    blockers = []
    failed = [report for report in shard_reports if int(report.get("returncode") or 0) != 0]
    if failed:
        blockers.append(f"{len(failed)} converter shard(s) failed")
    if len(merged_rows) < int(args.target_rows):
        blockers.append(f"accepted rows {len(merged_rows)} < target {int(args.target_rows)}")

    manifest = {
        "schema_version": "molgrapher_ordinary_negative_batch_v1",
        "csv": str(output_csv),
        "row_count": len(merged_rows),
        "target_rows": int(args.target_rows),
        "copied_images": copied_images,
        "parquet_inputs": args.parquet,
        "source_window": int(args.source_window),
        "max_accepted_per_window": int(args.max_accepted_per_window),
        "jobs": int(args.jobs),
        "heldout_csv": args.heldout_csv,
        "heldout_image_csv": args.heldout_image_csv,
        "shard_reports": shard_reports,
        "passed": not blockers,
        "blockers": blockers,
        "status": "candidate_large_ordinary_negative_requires_validation_source_leak_attachment_role_and_visual_review",
        "policy": {
            "uses_strict_single_shard_converter": True,
            "no_converter_fallback": True,
            "heldout_blacklist_forwarded_to_each_shard": True,
            "heldout_image_blacklist_forwarded_to_each_shard": True,
            "accepted_rows_must_be_validated_after_merge": True,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
