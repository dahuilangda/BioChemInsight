#!/usr/bin/env python
"""Assemble the pure-real-data SFT train_df (labels 0/1/2) for MoE warm-start.

Mixes:
  label=0  real_uspto_complete       (~300k real USPTO patent crops, symbol-only)
  label=1  real_markushgrapher_v2    (25,506 pose-verified markush rows)
  label=2  real_fragments + wavy     (894 + 31 pose-verified fragment rows)

All sources already carry contract-v3 columns; this script only unions the
column sets, fills benign NaNs (string columns -> ""), enforces file_path
uniqueness with source priority, and writes the single --df-cache parquet the
trainer consumes with --reuse-df-cache --no-use-real-original-data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output train_df parquet")
    parser.add_argument("--complete", type=Path, default=ROOT / "training/molnextr_markush/data/generated/real_uspto_complete/data.parquet")
    parser.add_argument("--markush", type=Path, default=ROOT / "training/molnextr_markush/data/generated/real_markushgrapher_ocsr_v2/train/data.parquet")
    parser.add_argument("--fragments", type=Path, nargs="*", default=[
        ROOT / "training/molnextr_markush/data/generated/real_fragments/data.parquet",
        ROOT / "training/molnextr_markush/data/generated/real_wavy_fragment_finetune/data.parquet",
    ])
    parser.add_argument("--extra", type=Path, nargs="*", default=[],
                        help="extra contract-v3 parquets appended as-is "
                             "(rows carry their own labels; e.g. rgroup / "
                             "recovered symbol-only markush shards)")
    args = parser.parse_args()

    import pandas as pd

    frames = []
    for label, name, path in [
        (1, "markush", args.markush),
        (2, "fragments", None),
        (0, "complete", args.complete),
    ]:
        if path is None:
            for frag_path in args.fragments:
                df = pd.read_parquet(frag_path)
                df["__source"] = f"fragments:{frag_path.parent.name}"
                frames.append(df)
            continue
        df = pd.read_parquet(path)
        df["__source"] = name
        frames.append(df)
    for extra_path in args.extra:
        df = pd.read_parquet(extra_path)
        df["__source"] = f"extra:{extra_path.parent.name}"
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Fill NaN introduced by the column union: boolean columns get False,
    # string columns get "", numeric columns keep NaN (contract only audits
    # label=2 rows for the 19 alignment columns, and those rows come fully
    # populated from their builders).
    for column in combined.columns:
        if not combined[column].isna().any():
            continue
        non_null = combined[column].dropna()
        if len(non_null) and non_null.map(type).eq(bool).all():
            combined[column] = combined[column].fillna(False)
        elif combined[column].dtype == object and non_null.map(lambda v: isinstance(v, str)).all():
            combined[column] = combined[column].fillna("")

    before = len(combined)
    combined = combined.drop_duplicates(subset=["file_path"], keep="first")
    dropped = before - len(combined)

    label_counts = combined["structure_type_label"].value_counts().sort_index().to_dict()
    combined = combined.drop(columns=["__source"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out, index=False)

    report = {
        "rows_total": len(combined),
        "duplicate_file_paths_dropped": dropped,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "sources": {
            "complete": str(args.complete),
            "markush": str(args.markush),
            "fragments": [str(p) for p in args.fragments],
        },
    }
    print(json.dumps(report, indent=2))
    with open(args.out.with_suffix(".report.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
