#!/usr/bin/env python
"""Build the bond fine-tune training cache from the base cache and real fragment sources.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.molnextr_markush.tools.train_moe import (  # noqa: E402
    validate_dataframe_contract,
)

DEFAULT_BASE_CACHE = (
    REPO / "experiments/moe/bond_finetune_base_cache.parquet"
)
DEFAULT_REAL_FRAGMENTS = (
    REPO / "training/molnextr_markush/data/generated/real_fragments/data.parquet"
)
DEFAULT_LOCAL_FRAGMENT_CACHE = (
    REPO / "experiments/moe/fragment_cache.parquet"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", default=str(DEFAULT_BASE_CACHE))
    parser.add_argument("--real-fragments", default=str(DEFAULT_REAL_FRAGMENTS))
    parser.add_argument(
        "--local-fragment-cache",
        default=str(DEFAULT_LOCAL_FRAGMENT_CACHE),
        help="Rebuilt local fragment cache (rebuild_local_fragment_cache.py output)",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(args.base_cache)
    validate_dataframe_contract(base, context="base_cache")
    print(f"base cache: {len(base)} rows "
          f"({base['structure_type_label'].value_counts().to_dict()})")
    seen = set(base["file_path"].tolist())
    merged_parts = [base]

    # Local rebuilt fragments (rebuild_local_fragment_cache.py output).
    local_frag_path = Path(args.local_fragment_cache)
    if local_frag_path.exists():
        local_fragments = pd.read_parquet(local_frag_path)
        validate_dataframe_contract(
            local_fragments, context="local_fragments", native_attachment_extension=True
        )
        added_local = local_fragments[~local_fragments["file_path"].isin(seen)]
        print(f"local fragment cache: {len(local_fragments)} rows "
              f"({len(added_local)} new vs base)")
        merged_parts.append(added_local)
        seen.update(added_local["file_path"].tolist())
    else:
        print(f"warning: local fragment cache not found: {local_frag_path}")

    # Real patent fragments (small but high-value; always include).
    fragments = pd.read_parquet(args.real_fragments)
    validate_dataframe_contract(
        fragments, context="real_fragments", native_attachment_extension=True
    )
    added_real = fragments[~fragments["file_path"].isin(seen)]
    print(f"real fragments: {len(fragments)} rows ({len(added_real)} new)")
    merged_parts.append(added_real)

    merged = pd.concat(merged_parts, ignore_index=True)
    # Canonical column order from the base cache.
    merged = merged[base.columns.tolist()]
    validate_dataframe_contract(merged, context="merged_cache")
    out_path = out_dir / "expanded_bond_finetune_cache.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"merged cache: {len(merged)} rows "
          f"({merged['structure_type_label'].value_counts().to_dict()})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
