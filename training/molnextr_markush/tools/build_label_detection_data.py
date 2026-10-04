"""Extract label-detection training data from USPTO-MOL-M 54k (§44.28).

Per row: save the ink-bbox structure crop (the adapter's placement) and the
label centers (cells bbox centers mapped into crop coordinates). The
detector (train_label_detector.py) learns to predict these centers as
heatmap peaks; at inference the peaks give 'which crop position holds which
label', closing the position mapping.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit-rows", type=int, default=0)
    args = parser.parse_args()

    import glob as globlib

    import pyarrow.parquet as pq

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    centers = {}
    scanned = written = skipped = 0

    for parquet_path in sorted(globlib.glob(args.parquet_glob)):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                scanned += 1
                if args.limit_rows and written >= args.limit_rows:
                    print(f"limit reached: written={written}", flush=True)
                    json.dump(centers, open(out_dir / "centers.json", "w"))
                    return
                try:
                    image_bytes = (row.get("page_image") or {}).get("bytes")
                    if not image_bytes:
                        continue
                    arr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    ink = (gray < 128).astype(np.uint8)
                    ys, xs = np.where(ink > 0)
                    if ys.size == 0:
                        skipped += 1
                        continue
                    height, width = image.shape[:2]
                    x0, x1 = int(xs.min()), int(xs.max())
                    y0, y1 = int(ys.min()), int(ys.max())
                    margin = 12
                    cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
                    cx1, cy1 = min(width, x1 + margin), min(height, y1 + margin)
                    crop = image[cy0:cy1, cx0:cx1]
                    crop_h, crop_w = crop.shape[:2]
                    if crop_h < 40 or crop_w < 40:
                        skipped += 1
                        continue
                    pts = []
                    for cell in row.get("cells") or []:
                        text = str(cell.get("text") or "").strip()
                        # audit §44.34: R-prefix narrowed the label domain —
                        # USPTO-MOL-M also uses X/Y/A/B/E variable labels.
                        if not text or not text.upper().startswith(("R", "X", "Y", "A", "B", "E")):
                            continue
                        bbox = cell.get("bbox")
                        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                            continue
                        # cells bbox is normalized [0,1]; convert the center
                        # to page pixels, then into crop coordinates.
                        center_px = (bbox[0] + (bbox[2] - bbox[0]) / 2) * width
                        center_py = (bbox[1] + (bbox[3] - bbox[1]) / 2) * height
                        px = (center_px - cx0) / crop_w
                        py = (center_py - cy0) / crop_h
                        if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0:
                            pts.append([round(px, 4), round(py, 4),
                                        round((bbox[2] - bbox[0]) * width, 1)])
                    if not pts:
                        skipped += 1
                        continue
                    image_name = Path(str(row.get("image_name") or "page")).stem
                    file_name = f"{image_name}_{written}.png"
                    cv2.imwrite(str(images_dir / file_name), crop)
                    centers[file_name] = pts
                    written += 1
                except Exception:
                    skipped += 1
        print(f"{Path(parquet_path).name}: scanned={scanned} written={written} "
              f"skipped={skipped}", flush=True)

    json.dump(centers, open(out_dir / "centers.json", "w"))
    print(f"DONE: scanned={scanned} written={written} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
