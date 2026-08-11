#!/usr/bin/env python3
"""Multi-class attachment-point segmentation dataset generator (Stage A fusion).

Extends gen_wavy_seg_data.py to render FOUR attachment-point mask classes, so a
single Mask R-CNN can localize every non-standard attachment notation found in
patent Markush/fragment structures. Drives the mask-guided graph repair that is
the DECIMER+MolNexTR organic fusion.

Classes (category_id):
  1 = wavy bond (triangle-wave zigzag attachment bond)
  2 = R-group label (R1/R'/superscript substituent identifier glyph region)
  3 = asterisk (explicit * attachment-point glyph)
  4 = dashed/open-valence (dashed bond indicating an open attachment)

Each sample = (image.png, COCO annotations.json). Reuses the patent-calibrated
wavy geometry; R-group/asterisk/dashed are drawn with PIL at the attachment
vertex. Negatives are complete molecules (no attachment marks).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "training" / "molnextr_markush" / "tools"
sys.path.insert(0, str(TOOLS))

from build_pose_factory_fragment_shard import (  # noqa: E402
    connector_basis,
    markush_wavy_points,
    patent_style_perpendicular_wavy_geometry,
    center_cross_perpendicular_wavy_style,
)

NATIVE_SIZE_RANGE = (80, 120)
OUTPUT_SIZE = 384
PEN_WIDTH_RANGE = (1.6, 2.6)
BG_MEAN = 253.0
BG_STD = 10.0

CAT_WAVY = 1
CAT_RGROUP = 2
CAT_ASTERISK = 3
CAT_DASHED = 4
CATEGORY_NAMES = {1: "wavy", 2: "rgroup", 3: "asterisk", 4: "dashed"}


def draw_antialiased_line_mask(size, points, *, width_px, scale=4):
    w, h = size
    big = Image.new("L", (w * scale, h * scale), 0)
    draw = ImageDraw.Draw(big)
    scaled = [(x * scale, y * scale) for x, y in points]
    ww = max(1, int(round(width_px * scale)))
    if len(scaled) >= 2:
        draw.line(scaled, fill=255, width=ww, joint="curve")
    return big.resize((w, h), Image.LANCZOS)


def composite_mask(image, mask, fill):
    arr = np.asarray(image).astype(np.float32)
    m = np.asarray(mask).astype(np.float32) / 255.0
    for c in range(3):
        arr[..., c] = arr[..., c] * (1 - m) + fill[c] * m
    return Image.fromarray(arr.astype(np.uint8))


def add_background_noise(image, rng):
    arr = np.asarray(image).astype(np.float32)
    noise = np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(0.0, BG_STD, arr.shape[:2])
    bg_mask = arr.mean(axis=-1) > 180
    for c in range(3):
        chan = arr[..., c]
        chan[bg_mask] = np.clip(chan[bg_mask] + noise[bg_mask], 0, 255)
    arr += rng.uniform(-3, 3)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _atoms_for_side(side, size, n_atoms, rng):
    margin = rng.uniform(0.12, 0.20) * size
    if side in ("left", "right"):
        start = margin if side == "left" else size - margin
        end = size - margin if side == "left" else margin
        cy = size * rng.uniform(0.40, 0.60)
        xs = np.linspace(start, end, n_atoms)
        amp = rng.uniform(size * 0.05, size * 0.08)
        ys = cy + amp * np.array([(-1) ** i for i in range(n_atoms)])
        anchor_idx = 0 if side == "left" else n_atoms - 1
    else:
        start = margin if side == "top" else size - margin
        end = size - margin if side == "top" else margin
        cx = size * rng.uniform(0.40, 0.60)
        ys = np.linspace(start, end, n_atoms)
        amp = rng.uniform(size * 0.05, size * 0.08)
        xs = cx + amp * np.array([(-1) ** i for i in range(n_atoms)])
        anchor_idx = 0 if side == "top" else n_atoms - 1
    return list(zip(xs.tolist(), ys.tolist())), anchor_idx


def _endpoint_for_side(side, ax, ay, native, rng):
    off = rng.uniform(native * 0.04, native * 0.07)
    return {"left": (ax - off, ay), "right": (ax + off, ay),
            "top": (ax, ay - off), "bottom": (ax, ay + off)}[side]


def _draw_chain(draw, atoms, bond_width):
    for i in range(len(atoms) - 1):
        x1, y1 = atoms[i]; x2, y2 = atoms[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=max(1, int(round(bond_width))))


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_glyph_mask(image_size, text, cx, cy, font_size):
    """Return (composited_into_image, glyph_mask_L) for a text glyph centered at cx,cy."""
    w, h = image_size
    layer = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    font = _font(font_size)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = font_size, font_size
    tx = max(0, int(cx - tw / 2 - bbox[0]))
    ty = max(0, int(cy - th / 2 - bbox[1]))
    draw.text((tx, ty), text, fill=255, font=font)
    return layer


def _dashed_line_mask(image_size, p1, p2, width_px, dash_len, gap_len):
    w, h = image_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return mask
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    ww = max(1, int(round(width_px)))
    t = 0.0
    while t < length:
        a = (x1 + ux * t, y1 + uy * t)
        b_t = min(t + dash_len, length)
        b = (x1 + ux * b_t, y1 + uy * b_t)
        draw.line([a, b], fill=255, width=ww)
        t += dash_len + gap_len
    return mask


def _contour_from_mask(mask_arr):
    contours, _ = cv2.findContours(mask_arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segs = []
    for c in contours:
        if len(c) >= 3:
            segs.append(c.reshape(-1).astype(float).tolist())
    return segs


def _annotation(mask_np, category_id):
    ys, xs = np.where(mask_np)
    if len(xs) < 3:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    segs = _contour_from_mask(mask_np)
    if not segs:
        return None
    return {"bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1], "area": int(mask_np.sum()),
            "segmentation": segs, "category_id": category_id}


def render_sample(rng):
    """Render one fragment crop with ONE attachment mark (random class). Returns
    (image, [annotations]) at OUTPUT_SIZE, or None."""
    native = rng.randint(*NATIVE_SIZE_RANGE)
    bond_width = rng.uniform(*PEN_WIDTH_RANGE)
    n_atoms = rng.randint(3, 6)
    side = rng.choice(("left", "right", "top", "bottom"))
    atoms, anchor_idx = _atoms_for_side(side, native, n_atoms, rng)
    ax, ay = atoms[anchor_idx]
    endpoint = _endpoint_for_side(side, ax, ay, native, rng)

    image = Image.new("RGB", (native, native), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    _draw_chain(draw, atoms, bond_width)
    if rng.random() < 0.4 and n_atoms >= 4:
        far = atoms[-1]
        r = native * rng.uniform(0.06, 0.10)
        ring = [(far[0] + r * math.cos(a), far[1] + r * math.sin(a)) for a in [0, 2.1, 4.2]]
        for i in range(3):
            draw.line([ring[i], ring[(i + 1) % 3]], fill=(0, 0, 0), width=max(1, int(round(bond_width))))

    cat = rng.choice([CAT_WAVY, CAT_WAVY, CAT_WAVY, CAT_RGROUP, CAT_ASTERISK, CAT_DASHED])  # wavy weighted (most common)
    native_mask = np.zeros((native, native), np.uint8)

    if cat == CAT_WAVY:
        ux, uy, px, py, norm = connector_basis((ax, ay), endpoint)
        if norm < 1e-3:
            return None
        style = patent_style_perpendicular_wavy_geometry(norm)
        if rng.random() < 0.94:
            style = center_cross_perpendicular_wavy_style(style)
        wc = endpoint
        ws = (wc[0] - px * style["length"] * 0.5, wc[1] - py * style["length"] * 0.5)
        we = (wc[0] + px * style["length"] * 0.5, wc[1] + py * style["length"] * 0.5)
        wpts = markush_wavy_points(ws, we, amplitude=float(style["amplitude"]),
                                   cycles=float(style["cycles"]), segments=int(style["segments"]))
        cmask = draw_antialiased_line_mask(image.size, [(ax, ay), endpoint], width_px=bond_width, scale=4)
        image = composite_mask(image, cmask, (0, 0, 0))
        wmask = draw_antialiased_line_mask(image.size, wpts, width_px=bond_width, scale=4)
        image = composite_mask(image, wmask, (0, 0, 0))
        native_mask = np.asarray(wmask) > 127

    elif cat == CAT_ASTERISK:
        cmask = draw_antialiased_line_mask(image.size, [(ax, ay), endpoint], width_px=bond_width, scale=4)
        image = composite_mask(image, cmask, (0, 0, 0))
        ex, ey = endpoint
        gmask = _text_glyph_mask(image.size, "*", ex, ey, font_size=max(10, int(native * 0.10)))
        image = composite_mask(image, gmask, (0, 0, 0))
        native_mask = np.asarray(gmask) > 127

    elif cat == CAT_RGROUP:
        cmask = draw_antialiased_line_mask(image.size, [(ax, ay), endpoint], width_px=bond_width, scale=4)
        image = composite_mask(image, cmask, (0, 0, 0))
        label = rng.choice(["R1", "R2", "R'", "Ra", "X", "Y", "Z"])
        ex, ey = endpoint
        gmask = _text_glyph_mask(image.size, label, ex, ey, font_size=max(9, int(native * 0.085)))
        image = composite_mask(image, gmask, (0, 0, 0))
        native_mask = np.asarray(gmask) > 127

    elif cat == CAT_DASHED:
        dmask = _dashed_line_mask(image.size, (ax, ay), endpoint, bond_width,
                                  dash_len=max(2, native * 0.025), gap_len=max(2, native * 0.025))
        image = composite_mask(image, dmask, (0, 0, 0))
        native_mask = np.asarray(dmask) > 127

    image = image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
    mask_full = cv2.resize(native_mask.astype(np.uint8), (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
    ann = _annotation(mask_full, cat)
    if ann is None:
        return None
    return image, [ann]


def render_negative(rng):
    native = rng.randint(*NATIVE_SIZE_RANGE)
    bond_width = rng.uniform(*PEN_WIDTH_RANGE)
    image = Image.new("RGB", (native, native), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    cx = cy = native / 2
    r = native * rng.uniform(0.12, 0.20)
    n_ring = rng.randint(5, 6)
    pts = [(cx + r * math.cos(2 * math.pi * i / n_ring), cy + r * math.sin(2 * math.pi * i / n_ring)) for i in range(n_ring)]
    for i in range(n_ring):
        draw.line([pts[i], pts[(i + 1) % n_ring]], fill=(0, 0, 0), width=max(1, int(round(bond_width))))
    for _ in range(rng.randint(1, 3)):
        a = pts[rng.randrange(n_ring)]
        ang = rng.uniform(0, 2 * math.pi)
        b = (a[0] + r * math.cos(ang), a[1] + r * math.sin(ang))
        draw.line([a, b], fill=(0, 0, 0), width=max(1, int(round(bond_width))))
    image = image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
    image = add_background_noise(image, rng)
    return image, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-positive", type=int, default=8000)
    parser.add_argument("--n-negative", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--coco", action="store_true", default=True)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    images_meta = []; anns = []; ann_id = 1; idx = 0
    for _ in range(args.n_positive):
        res = render_sample(rng)
        if res is None:
            continue
        image, sample_anns = res
        image = add_background_noise(image, rng)
        name = f"pos_{idx:06d}.png"
        image.save(out / "images" / name)
        images_meta.append({"id": idx, "file_name": name, "width": OUTPUT_SIZE, "height": OUTPUT_SIZE})
        for a in sample_anns:
            a["id"] = ann_id; a["image_id"] = idx; ann_id += 1; anns.append(a)
        idx += 1
    for _ in range(args.n_negative):
        image, _ = render_negative(rng)
        name = f"neg_{idx:06d}.png"
        image.save(out / "images" / name)
        images_meta.append({"id": idx, "file_name": name, "width": OUTPUT_SIZE, "height": OUTPUT_SIZE})
        idx += 1
    coco = {"images": images_meta, "annotations": anns,
            "categories": [{"id": k, "name": v} for k, v in CATEGORY_NAMES.items()]}
    (out / "annotations.json").write_text(json.dumps(coco))
    from collections import Counter
    cc = Counter(a["category_id"] for a in anns)
    print(f"Generated {len(images_meta)} samples, annotations by class: {dict(cc)}")


if __name__ == "__main__":
    main()
