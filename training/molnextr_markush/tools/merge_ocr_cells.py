"""Merge precomputed ChemicalOCR cells into a training df (§44.53).

Left-joins {file_path -> ocr_cells JSON} parquet(s) onto the training frame,
emitting the ``ocr_cells`` column the OCR-conditioning dataset path consumes.
Rows without cells get an empty string → zero/null-token memory (safe).

Usage:
  python tools/merge_ocr_cells.py \
      --df experiments/moe/uspto_label_v1_train_df.parquet \
      --cells /tmp/ocr_cells_precompute.shard*.parquet \
      --out experiments/moe/uspto_label_ocr_cond_df.parquet
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--df", required=True)
    p.add_argument("--cells", nargs="+", required=True,
                   help="cells parquet(s), globs allowed")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.df)
    frames = []
    for pattern in args.cells:
        for path in sorted(glob.glob(pattern)):
            frames.append(pd.read_parquet(path))
    if not frames:
        raise SystemExit("no cells parquet matched")
    cells = pd.concat(frames, ignore_index=True)
    cells = cells.drop_duplicates(subset=["file_path"], keep="last")
    non_empty = (cells["ocr_cells"].fillna("").str.len() > 0).sum()
    print(f"cells: {len(cells)} images ({non_empty} non-empty)")

    merged = df.merge(
        cells[["file_path", "ocr_cells"]],
        on="file_path", how="left",
    )
    merged["ocr_cells"] = merged["ocr_cells"].fillna("")
    # The trainer's contract rejects duplicate file_path rows; a left merge on
    # a deduped key preserves uniqueness, but assert it loudly anyway.
    if merged["file_path"].duplicated().any():
        raise SystemExit("merge introduced duplicate file_path rows")
    merged.to_parquet(args.out, index=False)
    with_cells = (merged["ocr_cells"].str.len() > 0).sum()
    print(f"merged: {len(merged)} rows -> {args.out} "
          f"({with_cells} rows with cells, {with_cells / max(1, len(merged)):.1%})")


if __name__ == "__main__":
    main()
