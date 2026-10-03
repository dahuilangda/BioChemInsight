"""Build (label crop, label class) training pairs from USPTO-MOL-M 54k.

For every row: ink-bbox placement (validated §44.23), mol-space dummy
coords mapped affinely into the crop, a window cropped around each dummy
atom, and the $...$ variable label as the class. Numeric labels only
(R1..R99); placeholder labels (X) and non-numeric variants are skipped.

Output: images dir + labels.json {crop_file: label} + class histogram.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys_path = str(REPO)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from rdkit import Chem, RDLogger  # noqa: E402

from training.molnextr_markush.src.moe_dataset import (  # noqa: E402
    parse_cxsmiles_dummy_labels,
)
from training.molnextr_markush.tools.prototype_uspto_placement import (  # noqa: E402
    ink_crop_from_page_bytes,
)

RDLogger.DisableLog("rdApp.*")

WINDOW = 72  # crop half-window in crop pixels around the dummy point
# Labels are typically drawn above-right of the attachment atom; bias the
# window so the glyph lands near the center.
LABEL_OFFSET_X = 0.35
LABEL_OFFSET_Y = -0.30


def numeric_label(label: str) -> int | None:
    digits = re.findall(r"\d+", str(label or ""))
    return int(digits[0]) if digits else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit-rows", type=int, default=0)
    args = parser.parse_args()

    import glob as globlib

    import pyarrow.parquet as pq

    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    scanned = written = skipped = 0
    labels_map = {}
    class_hist: dict[int, int] = {}
    label_list_path = out_dir / "labels.json"

    for parquet_path in sorted(globlib.glob(args.parquet_glob)):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                scanned += 1
                if args.limit_rows and written >= args.limit_rows:
                    print(f"limit reached: written={written}", flush=True)
                    json.dump(labels_map, open(label_list_path, "w"))
                    return
                try:
                    cxsmiles = str(row.get("cxsmiles_dataset") or "")
                    if not cxsmiles:
                        continue
                    labels = parse_cxsmiles_dummy_labels(cxsmiles)
                    if not labels:
                        continue
                    params = Chem.SmilesParserParams()
                    params.allowCXSMILES = True
                    params.strictCXSMILES = False
                    params.removeHs = False
                    mol = Chem.MolFromSmiles(cxsmiles, params)
                    if mol is None or mol.GetNumConformers() == 0:
                        skipped += 1
                        continue
                    conf = mol.GetConformer()
                    crop, bbox = ink_crop_from_page_bytes(
                        row["page_image"]["bytes"], margin=12)
                    if crop is None:
                        skipped += 1
                        continue
                    crop_h, crop_w = crop.shape[:2]
                    bx0, by0, bx1, by1 = bbox
                    mol_xs = [conf.GetAtomPosition(a.GetIdx()).x for a in mol.GetAtoms()]
                    mol_ys = [conf.GetAtomPosition(a.GetIdx()).y for a in mol.GetAtoms()]
                    min_x, max_x = min(mol_xs), max(mol_xs)
                    min_y, max_y = min(mol_ys), max(mol_ys)
                    span_x = (max_x - min_x) or 1.0
                    span_y = (max_y - min_y) or 1.0
                    span_px_x = max(1, bx1 - bx0)
                    span_px_y = max(1, by1 - by0)
                    image_name = Path(str(row.get("image_name") or "page")).stem

                    for atom in mol.GetAtoms():
                        if atom.GetAtomicNum() != 0:
                            continue
                        label = labels.get(atom.GetIdx())
                        number = numeric_label(label)
                        if number is None:
                            continue
                        pos = conf.GetAtomPosition(atom.GetIdx())
                        nx_ = (pos.x - min_x) / span_x
                        ny_ = (pos.y - min_y) / span_y
                        px = (bx0 + nx_ * span_px_x - (bx0 - 12)) / crop_w
                        py = (by0 + ny_ * span_px_y - (by0 - 12)) / crop_h
                        px = int(min(1.0, max(0.0, px)) * crop_w)
                        py = int(min(1.0, max(0.0, py)) * crop_h)
                        px += int(LABEL_OFFSET_X * WINDOW)
                        py += int(LABEL_OFFSET_Y * WINDOW)
                        x1_, x2_ = max(0, px - WINDOW), min(crop_w, px + WINDOW)
                        y1_, y2_ = max(0, py - WINDOW), min(crop_h, py + WINDOW)
                        window = crop[y1_:y2_, x1_:x2_]
                        if window.size == 0 or window.shape[0] < 12 or window.shape[1] < 12:
                            continue
                        crop_file = f"{image_name}_{atom.GetIdx()}_{written}_{scanned}.png"
                        cv2.imwrite(str(crops_dir / crop_file), window)
                        labels_map[crop_file] = number
                        class_hist[number] = class_hist.get(number, 0) + 1
                        written += 1
                except Exception:
                    skipped += 1
        print(f"{Path(parquet_path).name}: scanned={scanned} written={written} "
              f"skipped={skipped}", flush=True)

    json.dump(labels_map, open(label_list_path, "w"))
    print(f"DONE: scanned={scanned} written={written} skipped={skipped} "
          f"classes={len(class_hist)} top={sorted(class_hist.items(), key=lambda kv: -kv[1])[:8]}",
          flush=True)


if __name__ == "__main__":
    main()
