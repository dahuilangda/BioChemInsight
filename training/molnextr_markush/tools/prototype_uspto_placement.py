"""Prototype placement repair for the USPTO-MOL-M 54k shard (§44.23).

The real_sft_v3 build emitted all 33,792 uspto-mol-m rows as symbol-only
shells (empty smiles, reliable=False): structure placement inside the full
patent page failed for every row, and --train-symbol-only-fallback kept the
shells. This prototype demonstrates the placement fix: each page render is
a single-markush layout, so the ink bounding box (with a small margin) is
the structure crop.

Validated visually on shard train-00000 rows 0-1: complete structures with
variable labels (X) and repeat brackets captured. Production adapter next:
ink-crop placement + implant_markush_attachment_isotopes + graph edges ->
MoE graph contract rows.
"""
from __future__ import annotations

import cv2
import numpy as np


def ink_crop_from_page_bytes(page_image_bytes: bytes, margin: int = 12):
    """Crop the structure region from a USPTO-MOL-M page render.

    Returns (crop_bgr, ink_bbox) or (None, None) for blank pages.
    """
    arr = np.frombuffer(page_image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = (gray < 128).astype(np.uint8)
    ys, xs = np.where(ink > 0)
    if ys.size == 0 or xs.size == 0:
        return None, None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    height, width = image.shape[:2]
    crop = image[
        max(0, y0 - margin):min(height, y1 + margin),
        max(0, x0 - margin):min(width, x1 + margin),
    ]
    bbox = (x0, y0, x1, y1)
    return crop, bbox
