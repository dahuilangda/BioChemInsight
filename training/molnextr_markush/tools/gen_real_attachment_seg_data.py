#!/usr/bin/env python
"""Generate a COCO-format attachment-point detector training set from REAL patent
annotations.

The production detector (experiments/attachment_seg/attachment_maskrcnn_best.pth)
was trained on synthetic PIL glyphs, so its bbox centers are imprecise on real
patents — the root cause of graft regressions on exact_graph. This script builds
a real-patent training set so the detector learns the actual rendering of
attachment marks (wavy bonds / R-groups / asterisks) in patent literature.

Source: training/.../real_markushgrapher_ocsr_v2/train/data.parquet
  - file_path: real patent structure crop (1024x1024)
  - attachment_points: [[x,y],...] normalized coordinates of attachment marks
  - dummy_count: number of attachment points

Output: COCO annotations.json + symlinked images/, consumed verbatim by
train_attachment_maskrcnn.py (category_id 1 = wavy/attachment mark).

Each attachment point becomes one bbox annotation: a square box centered on the
point, sized to the typical attachment-glyph extent (~3.5% of image width, i.e.
~36px on a 1024px crop). The mask is the filled bbox (Mask R-CNN learns the
region; the graft consumer only uses the bbox center).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def build_coco(
    parquet_path: str,
    out_dir: str,
    *,
    box_fraction: float = 0.035,
    negative_fraction: float = 0.0,
    seed: int = 42,
) -> None:
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    t = pq.read_table(parquet_path)
    fp_col = t.column("file_path").to_pylist()
    ap_col = t.column("attachment_points").to_pylist()

    images, annotations = [], []
    ann_id = 1
    used = 0
    skipped_missing = 0
    for img_id, (fp, ap) in enumerate(zip(fp_col, ap_col), start=1):
        if not os.path.isfile(fp):
            skipped_missing += 1
            continue
        pts = json.loads(ap) if ap else []
        # Symlink (or copy) the image into the output dir with a stable name.
        dst = img_dir / f"real_{img_id:06d}.png"
        if not dst.exists():
            try:
                os.symlink(os.path.abspath(fp), dst)
            except OSError:
                shutil.copy(fp, dst)
        with Image.open(dst) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": dst.name, "width": w, "height": h})
        box_w = max(8, int(round(w * box_fraction)))
        box_h = max(8, int(round(h * box_fraction)))
        for pt in pts:
            cx, cy = float(pt[0]) * w, float(pt[1]) * h
            x1 = max(0, int(cx - box_w / 2))
            y1 = max(0, int(cy - box_h / 2))
            x2 = min(w, int(cx + box_w / 2))
            y2 = min(h, int(cy + box_h / 2))
            bw, bh = x2 - x1, y2 - y1
            if bw < 3 or bh < 3:
                continue
            # Segmentation = filled bbox polygon (Mask R-CNN mask target).
            seg = [[x1, y1, x2, y1, x2, y2, x1, y2]]
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [x1, y1, bw, bh], "area": float(bw * bh),
                "segmentation": seg, "iscrowd": 0,
            })
            ann_id += 1
        used += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "wavy", "supercategory": "attachment"},
            {"id": 2, "name": "rgroup", "supercategory": "attachment"},
            {"id": 3, "name": "asterisk", "supercategory": "attachment"},
            {"id": 4, "name": "dashed", "supercategory": "attachment"},
        ],
    }
    (out / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    print(f"built {out}: {used} images ({skipped_missing} missing skipped), "
          f"{len(annotations)} attachment annotations")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="training/molnextr_markush/data/generated/real_markushgrapher_ocsr_v2/train/data.parquet")
    p.add_argument("--out", default="training/molnextr_markush/data/real_attachment_seg/train")
    p.add_argument("--box-fraction", type=float, default=0.035)
    a = p.parse_args()
    build_coco(a.parquet, a.out, box_fraction=a.box_fraction)
