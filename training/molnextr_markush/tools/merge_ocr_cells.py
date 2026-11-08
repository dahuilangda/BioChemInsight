"""Merge precomputed ChemicalOCR cells into a training df (§44.53).

Left-joins {file_path -> ocr_cells JSON} parquet(s) onto the training frame,
emitting the ``ocr_cells`` column the OCR-conditioning dataset path consumes.

--gold-hybrid: per row, sample the stream from EITHER the real ChemicalOCR
read OR noise-augmented gold cells (MG2's standard OCR-noise augmentation:
label drop-out matching observed recall, R-prefix/suffix text variants,
confusable-digit substitutions, bbox jitter). The 50/50 regime mix prevents
shortcut learning — the decoder cannot know which regime a row is in, so it
must learn to read whichever cells are present. Rows with no real read use
noisy gold alone.

Usage:
  python tools/merge_ocr_cells.py \
      --df experiments/moe/uspto_label_v1_train_df.parquet \
      --cells /tmp/ocr_cells_precompute.shard*.parquet \
      --gold-hybrid --seed 20261005 \
      --out experiments/moe/uspto_label_ocr_cond_df.parquet
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

_GOLD_KEEP_WITH_REAL = 0.5   # noisy-gold regime share when a real read exists
_GOLD_KEEP_ALONE = 0.5       # drop-out within noisy-gold-only rows
_DIGIT_CONFUSABLE = {"0": "O", "O": "0", "1": "l", "l": "1",
                     "5": "S", "S": "5", "8": "B", "B": "8"}


def _confusable(text: str, rng) -> str:
    return "".join(
        _DIGIT_CONFUSABLE.get(ch, ch) if rng.random() < 0.5 else ch
        for ch in text)


def noisy_gold_cells(smiles: str, coords, rng) -> list[dict]:
    """Gold isotope labels + node coords → noise-augmented cell list."""
    try:
        pts = ast.literal_eval(coords) if isinstance(coords, str) and coords.strip() else []
    except (ValueError, SyntaxError):
        pts = []
    cells = []
    for k, num in enumerate(re.findall(r"\[(\d+)\*\]", str(smiles))):
        if rng.random() > 0.65:
            continue
        if k < len(pts):
            cx, cy = float(pts[k][0]), float(pts[k][1])
        else:
            cx, cy = 0.08 + 0.18 * (k % 5), 0.06 + 0.05 * (k // 5)
        roll = rng.random()
        if roll < 0.4:
            text = f"R{num}"
        elif roll < 0.8:
            text = num
        elif roll < 0.9:
            text = f"R{num}{rng.choice('ab')}"
        else:
            text = _confusable(num, rng)

        def jitter():
            return rng.uniform(-0.015, 0.015)
        x0 = min(0.98, max(0.0, cx - 0.045 + jitter()))
        y0 = min(0.98, max(0.0, cy - 0.028 + jitter()))
        x1 = min(1.0, max(x0 + 0.01, cx + 0.045 + jitter()))
        y1 = min(1.0, max(y0 + 0.01, cy + 0.028 + jitter()))
        cells.append({"bbox": [x0, y0, x1, y1], "text": text})
    return cells


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--df", required=True)
    p.add_argument("--cells", nargs="+", required=True,
                   help="cells parquet(s), globs allowed")
    p.add_argument("--out", required=True)
    p.add_argument("--gold-hybrid", action="store_true",
                   help="mix real reads with noise-augmented gold cells")
    p.add_argument("--seed", type=int, default=20261005)
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
    if args.gold_hybrid:
        import random
        real = merged.set_index("file_path")["ocr_cells"]
        composed = []
        for row in merged.itertuples(index=False):
            rng = random.Random(f"{args.seed}:{row.file_path}")
            gold = noisy_gold_cells(
                getattr(row, "SMILES", "") or "",
                getattr(row, "node_coords", None), rng)
            real_raw = real.get(row.file_path, "")
            try:
                real_cells = json.loads(real_raw) if real_raw else []
            except (ValueError, TypeError):
                real_cells = []
            if real_cells and gold:
                # 50/50 regime mix; real-only rows keep their true read.
                chosen = gold if rng.random() < 0.5 else real_cells
            elif gold:
                chosen = [c for c in gold if rng.random() < _GOLD_KEEP_ALONE + 0.2]
            else:
                chosen = real_cells
            composed.append(json.dumps(chosen))
        merged["ocr_cells"] = composed
    else:
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
