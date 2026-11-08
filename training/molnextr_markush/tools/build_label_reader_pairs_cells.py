"""Build label training pairs from the dataset's own cells bboxes (§44.26).

The cells field carries exact label positions and text — cropping there
gives perfectly aligned (crop, class) pairs, fixing the ~50-70% glyph hit
rate of the affine-dummy-point placement (§44.26). Numeric labels only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def numeric_label(text: str) -> int | None:
    match = re.fullmatch(r"R?\s*(\d+)[a-z]?", str(text or "").strip())
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--margin-frac", type=float, default=0.35,
                        help="crop padding around the cell bbox, as a fraction of the bbox size")
    args = parser.parse_args()

    import glob as globlib

    import pyarrow.parquet as pq

    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.json"

    labels = {}
    class_hist: dict[int, int] = {}
    scanned = written = skipped = 0

    for parquet_path in sorted(globlib.glob(args.parquet_glob)):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                scanned += 1
                if args.limit_rows and written >= args.limit_rows:
                    print(f"limit reached: written={written}", flush=True)
                    json.dump(labels, open(labels_path, "w"))
                    return
                try:
                    import io

                    from PIL import Image
                    image_bytes = (row.get("page_image") or {}).get("bytes")
                    if not image_bytes:
                        continue
                    image = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    height, width = image.shape[:2]
                    image_name = Path(str(row.get("image_name") or "page")).stem
                    for cell_index, cell in enumerate(row.get("cells") or []):
                        number = numeric_label(cell.get("text"))
                        if number is None:
                            continue
                        bbox = cell.get("bbox")
                        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                            continue
                        x0, y0, x1, y1 = [float(v) for v in bbox]
                        pad_x = (x1 - x0) * args.margin_frac
                        pad_y = (y1 - y0) * args.margin_frac
                        px0 = max(0, int((x0 - pad_x) * width))
                        py0 = max(0, int((y0 - pad_y) * height))
                        px1 = min(width, int((x1 + pad_x) * width))
                        py1 = min(height, int((y1 + pad_y) * height))
                        if px1 - px0 < 8 or py1 - py0 < 8:
                            continue
                        crop = image[py0:py1, px0:px1]
                        if crop.size == 0:
                            continue
                        crop_file = f"{image_name}_{cell_index}_{written}_{scanned}.png"
                        cv2.imwrite(str(crops_dir / crop_file), crop)
                        labels[crop_file] = number
                        class_hist[number] = class_hist.get(number, 0) + 1
                        written += 1
                except Exception:
                    skipped += 1
        print(f"{Path(parquet_path).name}: scanned={scanned} written={written} "
              f"skipped={skipped}", flush=True)

    json.dump(labels, open(labels_path, "w"))
    print(f"DONE: scanned={scanned} written={written} skipped={skipped} "
          f"classes={len(class_hist)}", flush=True)


if __name__ == "__main__":
    main()
