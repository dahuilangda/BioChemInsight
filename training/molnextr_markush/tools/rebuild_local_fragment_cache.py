#!/usr/bin/env python
"""Rebuild the MoE fragment training split from local pose-factory shard CSVs.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.molnextr_markush.src.moe_dataset import build_moe_df  # noqa: E402
from training.molnextr_markush.tools.train_moe import (  # noqa: E402
    MOE_DATA_CONTRACT_VERSION,
    validate_dataframe_contract,
)

DEFAULT_SHARD_GLOB = (
    "training/molnextr_markush/data/generated/pose_factory/"
    "molnextr_moe_production_v1/fragment/s*/attachment_fragment_positive.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-glob", default=DEFAULT_SHARD_GLOB)
    parser.add_argument("--max-rows", type=int, default=60000)
    parser.add_argument("--out", default="experiments/moe/fragment_cache.parquet")
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(args.shard_glob))
    if not shard_paths:
        raise SystemExit(f"no shard CSVs matched: {args.shard_glob}")
    # Pre-filter to shards whose images exist on disk (cheap existence check).
    usable = []
    for path in shard_paths:
        shard_dir = Path(path).resolve().parent
        images_dir = shard_dir / "images"
        if images_dir.is_dir() and any(images_dir.iterdir()):
            usable.append(path)
    print(f"shards matched: {len(shard_paths)}  usable (images present): {len(usable)}")
    if not usable:
        raise SystemExit("no usable shards (images missing)")

    df = build_moe_df(
        sources_by_label={2: usable},
        per_label_limit=int(args.max_rows),
        out_path=None,
        tokenizer=None,
    )
    if df.empty:
        raise SystemExit("build produced zero rows — shard schema or contract mismatch")

    # Re-validate the fragment contract (atom-index alignment + linearization)
    # exactly as train_moe.validate_dataframe_contract enforces it.
    validate_dataframe_contract(df, context="rebuilt_local_fragments",
                                native_attachment_extension=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    # Verify images on the written frame too.
    import pandas as pd  # noqa: PLC0415
    written = pd.read_parquet(out_path, columns=["file_path"])
    img_ok = sum(1 for p in written["file_path"] if os.path.isfile(p))
    print(f"wrote {len(df)} fragment rows -> {out_path}")
    print(f"  structure_type_label: {df['structure_type_label'].value_counts().to_dict()}")
    print(f"  attachment_render_mode: {df['attachment_render_mode'].value_counts().to_dict()}")
    print(f"  images on disk: {img_ok}/{len(written)}")
    print(f"  contract version: {df['data_contract_version'].unique().tolist()}")


if __name__ == "__main__":
    main()
