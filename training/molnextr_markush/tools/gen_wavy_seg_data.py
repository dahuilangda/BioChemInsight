#!/usr/bin/env python3
"""Synthetic wavy-bond segmentation dataset generator (Phase 0).

Renders fragment depictions with a terminal wavy attachment + a matching
pixel-accurate wavy mask, calibrated against real patent fragments:

- Real fragment crops are SMALL (~80-110px wide), then upscaled to 384 by
  MolNexTR. We render at the native small resolution then upscale to mimic the
  real pipeline, so the wavy's anti-aliasing/noise matches.
- Pen width ~5.7px (measured on real patents via distance transform).
- Background is near-white (mean ~253) with scan noise (std ~10), not pure
  white — we add Gaussian noise + a faint gray jitter.
- Wavy geometry reuses the patent-calibrated production functions
  (markush_wavy_points, patent_style_perpendicular_wavy_geometry) so the
  zigzag amplitude/length/cycle ranges match real patents.

Each sample = (image.png, wavy_mask.png); mask is single-channel uint8
(255 = wavy pixels). The generator emits positive (with wavy) and negative
(complete molecule, all-zero mask) samples.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "training" / "molnextr_markush" / "tools"
sys.path.insert(0, str(TOOLS))

from build_pose_factory_fragment_shard import (  # noqa: E402
    connector_basis,
    markush_wavy_points,
    patent_style_perpendicular_wavy_geometry,
    center_cross_perpendicular_wavy_style,
)

# Calibrated to real patent fragments (see docstring).
NATIVE_SIZE_RANGE = (80, 120)      # native crop width/height before upscale
OUTPUT_SIZE = 384                  # MolNexTR input size
PEN_WIDTH_RANGE = (1.6, 2.6)       # pen width at NATIVE resolution (upscale ~3-4x -> ~5-9px at 384)
BG_MEAN = 253.0
BG_STD = 10.0


def draw_antialiased_line_mask(
    size: tuple[int, int],
    points: list[tuple[float, float]],
    *,
    width_px: float,
    scale: int = 4,
) -> Image.Image:
    w, h = size
    big = Image.new("L", (w * scale, h * scale), 0)
    draw = ImageDraw.Draw(big)
    scaled = [(x * scale, y * scale) for x, y in points]
    ww = max(1, int(round(width_px * scale)))
    if len(scaled) >= 2:
        draw.line(scaled, fill=255, width=ww, joint="curve")
    elif scaled:
        draw.ellipse([scaled[0][0] - ww, scaled[0][1] - ww, scaled[0][0] + ww, scaled[0][1] + ww], fill=255)
    return big.resize((w, h), Image.LANCZOS)


def composite_mask(image: Image.Image, mask: Image.Image, fill: tuple[int, int, int]) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    m = np.asarray(mask).astype(np.float32) / 255.0
    for c in range(3):
        arr[..., c] = arr[..., c] * (1 - m) + fill[c] * m
    return Image.fromarray(arr.astype(np.uint8))


def add_background_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    """Mimic patent scan noise: near-white bg with Gaussian jitter."""
    arr = np.asarray(image).astype(np.float32)
    noise = np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(0.0, BG_STD, arr.shape[:2])
    # Only perturb background pixels (light ones) to avoid washing out ink.
    bg_mask = arr.mean(axis=-1) > 180
    for c in range(3):
        chan = arr[..., c]
        chan[bg_mask] = np.clip(chan[bg_mask] + noise[bg_mask], 0, 255)
    # Slight global brightness jitter
    arr += rng.uniform(-3, 3)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _draw_chain(draw, atoms, bond_width):
    for i in range(len(atoms) - 1):
        x1, y1 = atoms[i]
        x2, y2 = atoms[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=max(1, int(round(bond_width))))


def _atoms_for_side(side, size, n_atoms, rng):
    """Place backbone atoms so the wavy attaches on `side`."""
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


def render_positive(rng: random.Random) -> tuple[Image.Image, Image.Image] | None:
    native = rng.randint(*NATIVE_SIZE_RANGE)
    bond_width = rng.uniform(*PEN_WIDTH_RANGE)
    n_atoms = rng.randint(3, 6)
    side = rng.choice(("left", "right", "top", "bottom"))
    atoms, anchor_idx = _atoms_for_side(side, native, n_atoms, rng)
    ax, ay = atoms[anchor_idx]
    # endpoint just outside anchor, toward the side
    off = rng.uniform(native * 0.04, native * 0.07)
    endpoint = {
        "left": (ax - off, ay), "right": (ax + off, ay),
        "top": (ax, ay - off), "bottom": (ax, ay + off),
    }[side]

    image = Image.new("RGB", (native, native), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    _draw_chain(draw, atoms, bond_width)
    # occasional ring at far end
    if rng.random() < 0.4 and n_atoms >= 4:
        far = atoms[-1]
        r = native * rng.uniform(0.06, 0.10)
        ring = [(far[0] + r * math.cos(a), far[1] + r * math.sin(a)) for a in [0, 2.1, 4.2]]
        for i in range(3):
            draw.line([ring[i], ring[(i + 1) % 3]], fill=(0, 0, 0), width=max(1, int(round(bond_width))))

    # wavy via production geometry
    ux, uy, px, py, norm = connector_basis((ax, ay), endpoint)
    if norm < 1e-3:
        return None
    style = patent_style_perpendicular_wavy_geometry(norm)
    if rng.random() < 0.94:
        style = center_cross_perpendicular_wavy_style(style)
    wc = endpoint
    ws = (wc[0] - px * style["length"] * 0.5, wc[1] - py * style["length"] * 0.5)
    we = (wc[0] + px * style["length"] * 0.5, wc[1] + py * style["length"] * 0.5)
    wpts = markush_wavy_points(ws, we, amplitude=float(style["amplitude"]), cycles=float(style["cycles"]), segments=int(style["segments"]))
    # connector anchor->wavy center (NOT part of the wavy label)
    cmask = draw_antialiased_line_mask(image.size, [(ax, ay), endpoint], width_px=bond_width, scale=4)
    image = composite_mask(image, cmask, (0, 0, 0))
    # wavy polyline — the training label. Render at the SAME native resolution
    # so the mask aligns 1:1 with the wavy pixels, then upscale image+mask
    # together (NEAREST on mask to keep crisp boundaries).
    wmask_full = draw_antialiased_line_mask(image.size, wpts, width_px=bond_width, scale=4)
    image = composite_mask(image, wmask_full, (0, 0, 0))

    image = image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
    # Slight dilation ONLY on the wavy mask (post-upscale) to give the model a
    # sub-pixel tolerance margin; do NOT include the connector.
    label = wmask_full.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST)
    label = label.filter(ImageFilter.MaxFilter(3))
    image = add_background_noise(image, rng)
    return image, label


def render_negative(rng: random.Random) -> tuple[Image.Image, Image.Image]:
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
    label = Image.new("L", (OUTPUT_SIZE, OUTPUT_SIZE), 0)
    image = add_background_noise(image, rng)
    return image, label


def render_positive_coco(rng: random.Random) -> tuple[Image.Image, list[dict]] | None:
    """Like render_positive but returns COCO-style instance annotations.

    annotations: list of dicts with keys:
      bbox [x,y,w,h], area, segmentation (RLE-free polygon list), category_id 1 (wavy)
    """
    res = render_positive(rng)
    if res is None:
        return None
    image, mask_img = res
    mask = np.asarray(mask_img) > 127
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    # Build a coarse polygon from the mask contour.
    import cv2 as _cv2
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = _cv2.findContours(mask_u8, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for c in contours:
        if len(c) >= 3:
            segmentation.append(c.reshape(-1).astype(float).tolist())
    if not segmentation:
        return None
    ann = {
        "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
        "area": int(mask.sum()),
        "segmentation": segmentation,
        "category_id": 1,
    }
    return image, [ann]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic wavy segmentation data.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-positive", type=int, default=8000)
    parser.add_argument("--n-negative", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--coco", action="store_true", help="Also emit COCO-format annotations.json")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    idx = 0
    coco_images = []
    coco_anns = []
    ann_id = 1
    for _ in range(args.n_positive):
        if args.coco:
            res = render_positive_coco(rng)
            if res is None:
                continue
            image, anns = res
            name = f"pos_{idx:06d}.png"
            image.save(out / "images" / name)
            mask_img = Image.new("L", image.size, 0)
            # rebuild mask for saving (mask not returned by coco variant)
            Image.fromarray(np.zeros((image.size[1], image.size[0]), np.uint8)).save(out / "masks" / name)
            coco_images.append({"id": idx, "file_name": name, "width": image.size[0], "height": image.size[1]})
            for a in anns:
                a["id"] = ann_id; a["image_id"] = idx; ann_id += 1
                coco_anns.append(a)
        else:
            res = render_positive(rng)
            if res is None:
                continue
            image, mask_img = res
            name = f"pos_{idx:06d}.png"
            image.save(out / "images" / name)
            mask_img.save(out / "masks" / name)
        rows.append((name, 1))
        idx += 1
    for _ in range(args.n_negative):
        image, mask_img = render_negative(rng)
        name = f"neg_{idx:06d}.png"
        image.save(out / "images" / name)
        mask_img.save(out / "masks" / name)
        if args.coco:
            coco_images.append({"id": idx, "file_name": name, "width": image.size[0], "height": image.size[1]})
        rows.append((name, 0))
        idx += 1
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "has_wavy"])
        w.writerows(rows)
    if args.coco:
        import json
        coco = {
            "images": coco_images,
            "annotations": coco_anns,
            "categories": [{"id": 1, "name": "wavy"}],
        }
        (out / "annotations.json").write_text(json.dumps(coco))
    print(f"Generated {len(rows)} samples in {out} (pos+neg)" + (" + COCO" if args.coco else ""))


if __name__ == "__main__":
    main()
