"""Image-level wavy-bond inpainting for fragment OCSR (RGReco-style).

The MolNexTR backbone decoder follows the bond-line convention "every vertex is
a carbon atom". Our synthetic (and real patent) wavy attachment bonds are drawn
as triangle-wave zigzags with 5-6 turning points, so the decoder faithfully
decodes those turning points as spurious "ghost" carbons: ``*CNCCCO`` becomes
``*C(C)NCCCO``. MolScribe/MolNexTR have no "wavy" bond class, so this is a
field-wide ambiguity, not a model bug.

The fix (validated against the deployed model on real_wavy_hard, 31 fragments):
mask the wavy region's image pixels before the encoder sees them and draw a
single unambiguous attachment marker, so the decoder never encounters the
vertex-rich zigzag. Detection is deterministic for already-routed fragment
crops: the wavy sits on the structure's outer edge, and the decoded ghost atom
lands at the extreme x (or y) of ``atom_sets``. We inpaint the edge band
containing that extreme atom.

This is a pure inference-time, image-level transform. Complete rows are never
masked; a single config flag disables it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Fraction of the structure's in-image span that an edge band covers. Real
# patent wavy bonds span ~12-22px at 384px input (~3-6% of width), but the ghost
# atom can sit slightly inside, so the band is sized to cover the whole wavy
# extent plus a margin. 0.30 matches the detect_wavy_endpoints prototype band.
EDGE_BAND_FRACTION = 0.30
# An atom is considered "on the outer edge" (a candidate wavy/ghost location)
# when its normalized coordinate is within this margin of the image edge.
EDGE_MARGIN = 0.22
# Minimum number of decoded atoms required before we trust the geometry.
MIN_ATOMS = 3

MARKER_X = "x"
MARKER_DOT = "dot"


@dataclass(frozen=True)
class WavyMaskResult:
    """Outcome of a single-image wavy-mask pass.

    ``masked`` is True when the image was modified; the caller re-decodes only
    in that case. ``side`` records the inpainted edge for diagnostics.
    """

    masked: bool
    image: np.ndarray
    side: str = ""
    ghost_atom_index: int = -1
    reason: str = ""


def _foreground_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fixed = (gray < 235).astype(np.uint8) * 255
    fg = cv2.bitwise_or(otsu, fixed)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.where(fg > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _edge_side_from_atoms(
    atom_sets: list[dict[str, Any]],
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[str, int] | None:
    """Pick the outer edge that carries the wavy attachment.

    The most reliable locator is the decoder's own ``*`` dummy atom: it is the
    learned attachment endpoint token and sits at the wavy bond's outer tip. When
    a dummy is present we map its normalized coordinate to the nearest image
    edge. When no dummy decoded, we fall back to the most extreme non-dummy atom
    along the axis of greatest backbone spread (the chain runs away from the
    attachment, so the tail is the attachment).
    """
    if len(atom_sets) < MIN_ATOMS:
        return None

    dummies = [
        (index, atom)
        for index, atom in enumerate(atom_sets)
        if isinstance(atom, dict) and "*" in str(atom.get("atom_symbol") or "")
    ]
    if dummies:
        # Average dummy position in case of multiple, then map to nearest edge.
        dx = sum(float(a["coords"][0]) for _, a in dummies if a.get("coords")) / len(dummies)
        dy = sum(float(a["coords"][1]) for _, a in dummies if a.get("coords")) / len(dummies)
        candidates = (("left", dx), ("right", 1.0 - dx), ("top", dy), ("bottom", 1.0 - dy))
        side, _ = min(candidates, key=lambda item: item[1])
        ghost_index = dummies[0][0]
        return side, ghost_index

    backbone = [
        (index, atom)
        for index, atom in enumerate(atom_sets)
        if isinstance(atom, dict)
        and "*" not in str(atom.get("atom_symbol") or "")
    ]
    if len(backbone) < MIN_ATOMS:
        return None
    xs = [float(a["coords"][0]) for _, a in backbone if a.get("coords")]
    ys = [float(a["coords"][1]) for _, a in backbone if a.get("coords")]
    if len(xs) < MIN_ATOMS:
        return None
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    if span_x >= span_y:
        if min(xs) <= 1.0 - max(xs):
            side, target = "left", min(xs)
        else:
            side, target = "right", 1.0 - max(xs)
    else:
        if min(ys) <= 1.0 - max(ys):
            side, target = "top", min(ys)
        else:
            side, target = "bottom", 1.0 - max(ys)
    if target > EDGE_MARGIN:
        return None
    if side == "left":
        ghost_index, _ = min(backbone, key=lambda item: float(item[1]["coords"][0]))
    elif side == "right":
        ghost_index, _ = max(backbone, key=lambda item: float(item[1]["coords"][0]))
    elif side == "top":
        ghost_index, _ = min(backbone, key=lambda item: float(item[1]["coords"][1]))
    else:
        ghost_index, _ = max(backbone, key=lambda item: float(item[1]["coords"][1]))
    return side, ghost_index


def _band_slice(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[slice, slice]:
    """Row/col slices of the edge band on the full image array."""
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    bw = max(6, int(round(width * EDGE_BAND_FRACTION)))
    bh = max(6, int(round(height * EDGE_BAND_FRACTION)))
    h, w = image.shape[:2]
    if side == "left":
        return slice(y1, min(y2 + bh // 2, h)), slice(max(0, x1 - 2), min(x2, x1 + bw))
    if side == "right":
        return slice(y1, min(y2 + bh // 2, h)), slice(max(x1, x2 - bw), min(w, x2 + 2))
    if side == "top":
        return slice(max(0, y1 - 2), min(y2, y1 + bh)), slice(x1, min(x2 + bw // 2, w))
    return slice(max(y1, y2 - bh), min(h, y2 + 2)), slice(x1, min(x2 + bw // 2, w))


def _draw_marker(
    image_rgb: np.ndarray,
    side: str,
    row_slice: slice,
    col_slice: slice,
    marker: str,
) -> None:
    """Draw the attachment marker onto the (in-place) masked edge band.

    ``marker`` selects the glyph: ``MARKER_X`` draws an "x" character (RGReco's
    choice, robust across OCSR tools); ``MARKER_DOT`` draws a filled disk so the
    decoder reads a single explicit atom vertex instead of a vertex-rich zigzag.
    The glyph is sized to the bond line width (derived from foreground density).
    """
    band = image_rgb[row_slice, col_slice]
    if band.size == 0:
        return
    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    ink = (gray < 160).sum()
    ink_frac = ink / float(gray.size)
    # Estimate bond line width from foreground: thicker strokes -> larger glyph.
    radius = max(3, int(round(2.5 + ink_frac * 14.0)))
    h, w = band.shape[:2]
    # Marker center: the side wall of the band (where the wavy sat).
    if side == "left":
        cx, cy = max(radius, w // 5), h // 2
    elif side == "right":
        cx, cy = w - max(radius, w // 5), h // 2
    elif side == "top":
        cx, cy = w // 2, max(radius, h // 5)
    else:
        cx, cy = w // 2, h - max(radius, h // 5)
    if marker == MARKER_DOT:
        cv2.circle(band, (cx, cy), radius, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        return
    # MARKER_X: draw via PIL for a clean glyph.
    pil = Image.fromarray(band)
    draw = ImageDraw.Draw(pil)
    font_size = max(10, int(round(radius * 2.4)))
    try:
        from PIL import ImageFont

        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
    text = "x"
    try:
        tbbox = draw.textbbox((0, 0), text, font=font)
        tw, th = tbbox[2] - tbbox[0], tbbox[3] - tbbox[1]
    except Exception:
        tw, th = font_size, font_size
    tx = max(0, min(w - tw, cx - tw // 2 - tbbox[0]))
    ty = max(0, min(h - th, cy - th // 2 - tbbox[1]))
    draw.text((tx, ty), text, fill=(0, 0, 0), font=font)
    image_rgb[row_slice, col_slice] = np.asarray(pil)


def mask_wavy_attachment(
    image_rgb: np.ndarray,
    atom_sets: list[dict[str, Any]] | None,
    *,
    expected_type: str = "",
    marker: str = MARKER_X,
) -> WavyMaskResult:
    """Inpaint the wavy attachment region of a fragment crop and draw a marker.

    Returns the (possibly modified) image and whether masking happened. When
    ``expected_type`` is not fragment/markush, or no decoded atom sits on an
    outer edge, the image is returned unchanged (``masked=False``).
    """
    if str(expected_type or "").strip().lower() not in {"fragment", "markush"}:
        return WavyMaskResult(False, image_rgb, reason="not_fragment_or_markush")
    if image_rgb is None or image_rgb.size == 0:
        return WavyMaskResult(False, image_rgb, reason="empty_image")
    if not atom_sets:
        return WavyMaskResult(False, image_rgb, reason="no_atoms")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
    bbox = _foreground_bbox(gray)
    if bbox is None:
        return WavyMaskResult(False, image_rgb, reason="no_foreground")

    picked = _edge_side_from_atoms(atom_sets, bbox, image_rgb.shape)
    if picked is None:
        return WavyMaskResult(False, image_rgb, reason="no_edge_atom")
    side, ghost_index = picked

    row_slice, col_slice = _band_slice(image_rgb, bbox, side)
    band = image_rgb[row_slice, col_slice]
    if band.size == 0:
        return WavyMaskResult(False, image_rgb, reason="empty_band", side=side)

    out = image_rgb.copy()
    band_region = out[row_slice, col_slice]
    # Inpaint the foreground ink inside the band so the zigzag pixels vanish.
    band_gray = cv2.cvtColor(band_region, cv2.COLOR_RGB2GRAY)
    ink_mask = ((band_gray < 160).astype(np.uint8)) * 255
    # Dilate slightly so residual anti-aliased edges are covered.
    ink_mask = cv2.dilate(ink_mask, np.ones((3, 3), np.uint8), iterations=1)
    if int(ink_mask.sum()) > 0:
        inpainted = cv2.inpaint(band_region, ink_mask, 3, cv2.INPAINT_TELEA)
        out[row_slice, col_slice] = inpainted
    _draw_marker(out, side, row_slice, col_slice, marker)
    return WavyMaskResult(True, out, side=side, ghost_atom_index=ghost_index, reason="masked")
