from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize)


BRANCH_SPECS = {
    "fragment": {
        "glob": "fragment/s*/attachment_fragment_positive.csv",
        "output_name": "attachment_fragment_positive.csv",
        "structure_type": "attachment_fragment",
    },
    "ordinary": {
        "glob": "ordinary/s*/ordinary_negative.csv",
        "output_name": "ordinary_negative.csv",
        "structure_type": "complete_compound",
    },
    "markush": {
        "glob": "markush/s*/accepted_candidate/markush_layout_positive.csv",
        "output_name": "markush_layout_positive.csv",
        "structure_type": "markush_layout",
    },
}


def resolve_image_path(row: dict[str, str], csv_path: Path) -> str:
    raw = str(row.get("file_path") or row.get("image_path") or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        path = csv_path.parent / path
    return str(path.resolve())


def parse_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge production pose-factory shard CSVs into aggregate CSVs for "
            "source-disjoint sidecar training splits. Image paths are written "
            "as absolute paths so the aggregate directory does not need to copy "
            "hundreds of thousands of PNGs."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--branch", choices=sorted(BRANCH_SPECS), action="append", default=[])
    parser.add_argument("--require-reliable", action="store_true")
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Stat every resolved image path. Disabled by default because shard-level QC already checked images and full aggregation can include hundreds of thousands of PNGs.",
    )
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50000)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    branches = args.branch or sorted(BRANCH_SPECS)
    manifest: dict[str, object] = {
        "schema_version": "pose_factory_aggregate_from_shards_v1",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "branches": {},
        "policy": {
            "absolute_image_paths": True,
            "does_not_copy_images": True,
            "image_existence_check": bool(args.check_images),
            "raw_source_data_unchanged": True,
            "aggregate_is_training_input_not_acceptance_by_itself": True,
        },
    }
    blockers: list[str] = []

    for branch in branches:
        started = time.time()
        spec = BRANCH_SPECS[branch]
        csv_paths = sorted(dataset_root.glob(str(spec["glob"])))
        print(f"[aggregate] branch={branch} input_csvs={len(csv_paths)}", flush=True)
        output_csv = output_dir / str(spec["output_name"])
        fieldnames: list[str] = []
        rows_written = 0
        skipped_unreliable = 0
        missing_images = 0
        source_counts: Counter[str] = Counter()
        branch_counts: Counter[str] = Counter()
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer: csv.DictWriter[str] | None = None
            for csv_path in csv_paths:
                with csv_path.open(newline="", encoding="utf-8") as input_handle:
                    reader = csv.DictReader(input_handle)
                    if not fieldnames:
                        fieldnames = list(reader.fieldnames or [])
                        if "file_path" not in fieldnames:
                            fieldnames.append("file_path")
                        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                        writer.writeheader()
                    for row in reader:
                        if args.require_reliable and not parse_bool(row.get("reliable_training_label", "")):
                            skipped_unreliable += 1
                            continue
                        resolved = resolve_image_path(row, csv_path)
                        if not resolved:
                            missing_images += 1
                            continue
                        if args.check_images and not Path(resolved).exists():
                            missing_images += 1
                            continue
                        row = dict(row)
                        row["file_path"] = resolved
                        if "image_path" in row:
                            row["image_path"] = resolved
                        if writer is None:
                            raise RuntimeError("CSV writer was not initialized")
                        writer.writerow(row)
                        rows_written += 1
                        if int(args.progress_every) > 0 and rows_written % int(args.progress_every) == 0:
                            print(
                                f"[aggregate] branch={branch} rows={rows_written} csv={csv_path}",
                                flush=True,
                            )
                        source_counts[str(csv_path)] += 1
                        branch_counts[str(row.get("structure_type_bucket") or spec["structure_type"])] += 1

        if rows_written < int(args.min_rows):
            blockers.append(f"{branch} aggregate rows {rows_written} < {int(args.min_rows)}")
        manifest["branches"][branch] = {
            "input_csv_count": len(csv_paths),
            "output_csv": str(output_csv),
            "row_count": rows_written,
            "elapsed_seconds": float(time.time() - started),
            "skipped_unreliable": skipped_unreliable,
            "missing_images": missing_images,
            "structure_type": str(spec["structure_type"]),
            "source_counts": dict(source_counts),
            "structure_type_counts": dict(branch_counts),
        }
        print(
            f"[aggregate] branch={branch} done rows={rows_written} elapsed={time.time() - started:.1f}s output={output_csv}",
            flush=True,
        )

    manifest["passed"] = not blockers
    manifest["blockers"] = blockers
    manifest_path = output_dir / "aggregate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
