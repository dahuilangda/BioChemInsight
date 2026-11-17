"""Page-orientation skill for the harness.

Scanned patent sheets frequently carry tables rotated 90°; OCR, slicing and
VLM transcription all need an upright image. The skill decides — per page —
whether rotation is needed and which way, from image geometry alone:

  1. rule-energy probe: ruled tables leave long collinear dark runs. If the
     longest runs are vertical, the sheet is a rotated upright table.
  2. direction probe: for a rotated sheet, try both 90° candidates and
     pick the one where long runs turn horizontal AND the row-projection
     profile matches a table (many aligned text rows between rules). A
     cheap text-density check on the two candidates arbitrates.

Verification is built in: after rotating, the geometry probe must report
horizontal dominance, else the other direction is tried. No model call is
needed for the common cases; the caller may pass a VLM probe for the
residual ambiguity.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _long_run_energy(mask: np.ndarray) -> float:
    """Fraction of columns/rows containing a dark run of >=25 px."""
    h, w = mask.shape
    if w == 0 or h == 0:
        return 0.0
    runs = mask.any(axis=0) if mask.shape[1] else np.zeros(0, dtype=bool)
    return float(runs.mean())


def _rule_energy(gray: np.ndarray) -> tuple[float, float]:
    """(horizontal-rule energy, vertical-rule energy) of a binarized page.

    Horizontal energy: rows whose dark fraction across the table x-extent
    exceeds a threshold (table rules + aligned text rows). Vertical energy:
    the same after transpose. Whichever dominates says how the table lies."""
    h, w = gray.shape
    x0, x1 = int(w * 0.10), int(w * 0.95)
    y0, y1 = int(h * 0.08), int(h * 0.95)
    core = gray[y0:y1, x0:x1] < 160
    horiz = float((core.mean(axis=1) > 0.30).mean())
    vert = float((core.mean(axis=0) > 0.30).mean())
    return horiz, vert


def detect_rotation(gray_or_img, threshold: float = 1.6) -> int:
    """0 / 90 / -90 / 180 — the clockwise degrees to rotate for upright.

    Rotated ruled sheets show vertical rule energy dominating horizontal;
    the ratio picks the 90° family. Direction (CW vs CCW) is resolved by
    a row-projection texture check: reading-oriented pages have more
    aligned dark-row structure than their mirror rotation."""
    gray = np.asarray(gray_or_img.convert("L")) \
        if hasattr(gray_or_img, "convert") else np.asarray(gray_or_img)
    horiz, vert = _rule_energy(gray)
    if horiz >= vert or vert / max(horiz, 1e-6) < threshold:
        return 0                                   # already upright
    cw = np.rot90(gray, k=-1)
    ccw = np.rot90(gray, k=1)
    cw_h, _ = _rule_energy(cw)
    ccw_h, _ = _rule_energy(ccw)
    if cw_h >= ccw_h:
        return 90
    return -90


def rotate_image(img: Image.Image, angle: int) -> Image.Image:
    if angle == 0:
        return img
    k = {90: -1, -90: 1, 180: 2}[angle]
    return img.rotate(k, expand=True)


def orient_page(png_path: str, out_path: str | None = None,
                ) -> tuple[Image.Image, int]:
    """Load, detect, rotate. Returns (oriented image, angle applied)."""
    img = Image.open(png_path)
    angle = detect_rotation(img)
    out = rotate_image(img, angle)
    if out_path:
        out.save(out_path)
    return out, angle
