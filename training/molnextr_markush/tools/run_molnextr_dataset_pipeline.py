from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.molnextr_markush.src.molnextr_dataset_toolkit import (
    FormalSimDatasetConfig,
    MolNexTRDatasetPipeline,
)


def parse_shard_list(value: str) -> list[int]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("shard list must be non-empty")
    shards: list[int] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = [part.strip() for part in item.split("-", 1)]
            start = int(start_text)
            end = int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid shard range {item!r}")
            shards.extend(range(start, end + 1))
        else:
            shard = int(item)
            if shard < 0:
                raise ValueError(f"invalid shard index {item!r}")
            shards.append(shard)
    if not shards:
        raise ValueError("shard list must be non-empty")
    return sorted(set(shards))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production MolNexTR dataset pipeline wrapper for complete/Markush/fragment generation."
    )
    parser.add_argument("--dataset-id", default="molnextr_moe_production_v1")
    parser.add_argument("--branch", choices=["markush", "fragment", "ordinary", "attachment-role"], default="markush")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--markush-rows", type=int, default=500, help="Accepted rows per Markush R-count bucket.")
    parser.add_argument("--markush-candidate-multiplier", type=int, default=2)
    parser.add_argument("--markush-candidate-rows-per-bucket", type=int, default=0)
    parser.add_argument(
        "--markush-bucket-window-overrides",
        default="",
        help="Explicit Markush per-bucket source windows for scheduled tail shards.",
    )
    parser.add_argument(
        "--markush-bucket-targets",
        default="",
        help="Explicit accepted targets per Markush bucket, e.g. 1=256,2=256,3-4=128,5-8=0,9+=0.",
    )
    parser.add_argument("--markush-renderer-seed-retries", type=int, default=3)
    parser.add_argument("--fragment-rows", type=int, default=64)
    parser.add_argument("--ordinary-rows", type=int, default=64)
    parser.add_argument("--ordinary-parquet", default="", help="Explicit MolGrapher parquet file for this ordinary shard.")
    parser.add_argument("--ordinary-source-start-row", type=int, default=-1)
    parser.add_argument("--fragment-shards", default="", help="Comma/range list for attachment-role aggregate gate, e.g. 0-10.")
    parser.add_argument("--ordinary-shards", default="", help="Comma/range list for attachment-role aggregate gate, e.g. 0-3.")
    parser.add_argument("--seed", type=int, default=2026062208)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--inspect-retry", action="store_true", help="Print retry diagnostics for existing Markush shards.")
    parser.add_argument("--output", default="", help="Optional JSON dry-run/diagnostic report path.")
    args = parser.parse_args()

    pipeline = MolNexTRDatasetPipeline(FormalSimDatasetConfig(dataset_id=str(args.dataset_id)))
    commands = []
    diagnostics = []
    if args.branch == "attachment-role":
        commands.append(
            pipeline.attachment_role_contract_command(
                fragment_shards=parse_shard_list(str(args.fragment_shards)),
                ordinary_shards=parse_shard_list(str(args.ordinary_shards)),
            )
        )
    else:
        for index in range(int(args.start_shard), int(args.start_shard) + int(args.shards)):
            if args.branch == "markush":
                commands.extend(
                    pipeline.markush_commands(
                        index,
                        rows_per_bucket=int(args.markush_rows),
                        seed=int(args.seed) + index,
                        candidate_multiplier=int(args.markush_candidate_multiplier),
                        candidate_rows_per_bucket=(
                            int(args.markush_candidate_rows_per_bucket)
                            if int(args.markush_candidate_rows_per_bucket) > 0
                            else None
                        ),
                        renderer_seed_retries=int(args.markush_renderer_seed_retries),
                        bucket_window_overrides=str(args.markush_bucket_window_overrides),
                        bucket_targets_override=str(args.markush_bucket_targets),
                    )
                )
                if args.inspect_retry:
                    diagnostics.append(pipeline.markush_retry_diagnostics(index).to_dict())
            elif args.branch == "fragment":
                commands.extend(pipeline.fragment_commands(index, rows=int(args.fragment_rows), seed=int(args.seed) + 1000 + index))
            elif args.branch == "ordinary":
                commands.extend(
                    pipeline.ordinary_commands(
                        index,
                        rows=int(args.ordinary_rows),
                        source_start_row=(
                            int(args.ordinary_source_start_row)
                            if int(args.ordinary_source_start_row) >= 0
                            else None
                        ),
                        molgrapher_parquet=str(args.ordinary_parquet) if str(args.ordinary_parquet).strip() else None,
                    )
                )
            else:
                raise ValueError(f"unsupported branch {args.branch!r}")

    records = [
        command.run(cwd=pipeline.config.root, execute=bool(args.execute), skip_existing=bool(args.skip_existing))
        for command in commands
    ]
    report = pipeline.dry_run_manifest(commands)
    report["executed"] = bool(args.execute)
    report["command_records"] = records
    if diagnostics:
        report["retry_diagnostics"] = diagnostics
    if args.output:
        pipeline.write_json(report, Path(args.output))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
