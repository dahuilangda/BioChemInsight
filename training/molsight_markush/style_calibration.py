"""Render-style calibration profiles for the cyclopeptide pipeline.

Encapsulates the two MolDraw2D parameter sets used for real-domain style
alignment studies:

  baseline  — the stage-11 production parameters (canvas 1400 px, bond
              line 3.6-4.6 px, font 32-46 px, loose tight-crop margin)
  aligned   — parameters calibrated against the measured style of the 255
              verified real patent crops (see style_metrics.py /
              compare_render_style.py): serif label font, fixed bond
              length and font size so stroke/label-to-bond ratios match
              the real distribution, thinner relative strokes, tighter
              label padding.

`draw_molecule()` is the single render path shared by
render_style_pairs.py (paired style comparison) and the stage-11 batch
renderer, so the training corpus and the style study cannot drift.

Calibration notes (measured, round-by-round against the verified crops):
  - stroke_ratio real 0.176 pooled / ~0.10-0.11 at US-crop scale: the
    pooled value is inflated by low-res KRAS crops; 2.6 px at L=24
    reproduces the US/GPC3 scale.
  - serif (Liberation Serif, Times-metric) labels match the patent font;
    16 px keeps glyph strokes intact after rasterization.
  - additionalAtomLabelPadding is a fraction of font size: beyond ~0.2
    the MolDraw2D bond-truncation logic corrupts the drawing (bonds
    break into stubs), so the real label clearance (~0.28 x bond length)
    is only partially reproducible — 0.14 gives ~0.19 measured.
  - light print-gray (ink_level 0.86 + noise sigma 1.5) matches the mean
    ink darkness of scanned real crops without blurring geometry.
"""
from __future__ import annotations

import io
import math
import random

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

SERIF_FONT = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"

BASELINE = {
    "canvas": 1400,
    "bond_line": (3.6, 4.6),
    "font_min": 32,
    "font_max": 46,
    "label_padding": 0.0,
    "tight_crop_pad": (10, 45),
    "font_file": "",
    "fixed_bond_length": -1.0,
    "fixed_font_size": -1.0,
    "multiple_bond_offset": 0.15,
}

ALIGNED = {
    # fixed bond length in px pins the absolute scale so the measured
    # ratios (stroke/label vs bond length) land on the real distribution;
    # the canvas is derived from the scaled extent
    "canvas": 0,
    "bond_line": (2.6, 2.6),
    "font_min": 8,
    "font_max": 16,
    # see calibration note above — do not raise past ~0.2
    "label_padding": 0.14,
    "tight_crop_pad": (6, 14),
    "font_file": SERIF_FONT,
    "fixed_bond_length": 24.0,
    "fixed_font_size": 16,
    "multiple_bond_offset": 0.22,
    # light print-gray + faint noise so ink darkness matches the scanned
    # real crops (mean ink gray ~0.17); no blur, geometry stays crisp
    "ink_level": 0.86,
    "noise_sigma": 1.5,
}


def apply_profile(opts: rdMolDraw2D.MolDrawOptions, profile: dict,
                  rng: random.Random) -> None:
    """Write a calibration profile into MolDrawOptions."""
    lo, hi = profile["bond_line"]
    opts.bondLineWidth = rng.uniform(lo, hi) if hi > lo else float(lo)
    ff = profile.get("fixed_font_size", 0) or 0
    if ff > 0:
        opts.fixedFontSize = int(ff)
    else:
        opts.minFontSize = profile["font_min"]
        opts.maxFontSize = profile["font_max"]
    opts.additionalAtomLabelPadding = profile["label_padding"]
    opts.multipleBondOffset = profile["multiple_bond_offset"]
    opts.fixedBondLength = profile["fixed_bond_length"]
    if profile["font_file"]:
        opts.fontFile = profile["font_file"]
    opts.useBWAtomPalette = True
    opts.setAtomPalette({k: (0.0, 0.0, 0.0) for k in range(-1, 121)})
    opts.singleColourWedgeBonds = True


def coordgen_coords(mol: Chem.Mol):
    """Atom-shuffle (depiction randomization, as in render_stage11.py) and
    compute CoordGen coordinates. Returns the renumbered molecule carrying
    the coordinates, or None when CoordGen failed (caller skips it)."""
    order = list(range(mol.GetNumAtoms()))
    rng = random.Random(sum(a.GetAtomicNum() for a in mol.GetAtoms())
                        + mol.GetNumBonds() * 17)
    rng.shuffle(order)
    shuffled = Chem.RenumberAtoms(mol, order)
    try:
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(shuffled, clearConfs=True)
        rdDepictor.SetPreferCoordGen(False)
    except Exception:
        rdDepictor.SetPreferCoordGen(False)
        return None
    return shuffled


def draw_molecule(mol: Chem.Mol, profile: dict, rng: random.Random) -> Image:
    """Render a molecule (coords must already exist) to a tight-cropped
    grayscale PIL image in the profile's style. A uniform rotation is
    applied through the coordinates so labels stay upright, mirroring the
    arbitrary orientation of real patent crops."""
    bond_len = profile["fixed_bond_length"]
    if bond_len <= 0:
        bond_len = 40.0
    font_size = int(profile.get("fixed_font_size", 0) or 0)

    angle = rng.uniform(0, 360)
    rad = math.radians(angle)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        x = p.x * math.cos(rad) - p.y * math.sin(rad)
        y = p.x * math.sin(rad) + p.y * math.cos(rad)
        conf.SetAtomPosition(i, Point3D(x, y, 0.0))

    # coord units are not px: fixedBondLength rescales the layout, so the
    # canvas is derived from the SCALED extent (baseline keeps its fixed
    # large canvas; the tight crop removes the surplus either way)
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    if profile["canvas"] == 0:
        blens = []
        for b in mol.GetBonds():
            pa = conf.GetAtomPosition(b.GetBeginAtomIdx())
            pb = conf.GetAtomPosition(b.GetEndAtomIdx())
            blens.append(math.hypot(pb.x - pa.x, pb.y - pa.y))
        unit = float(np.median(blens)) or 1.0
        size = int(extent * (bond_len / unit) + 6 * bond_len)
        size = max(220, min(4000, size))
    else:
        size = profile["canvas"]

    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    apply_profile(opts, profile, rng)
    if font_size > 0:
        opts.fixedFontSize = font_size
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, kekulize=True)
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("L")

    # print-gray: scanned patent ink is not pure black
    ink_level = profile.get("ink_level")
    if ink_level:
        arr = np.array(img).astype(np.float32)
        arr = 255.0 - (255.0 - arr) * ink_level
        sigma = profile.get("noise_sigma", 0.0)
        if sigma:
            arr = arr + np.random.normal(0.0, sigma, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # tight-crop to ink bbox (patent crops carry only a few px of margin)
    arr = np.array(img)
    rows = np.nonzero((arr < 200).any(axis=1))[0]
    cols = np.nonzero((arr < 200).any(axis=0))[0]
    if len(rows) == 0:
        return img
    lo, hi = profile["tight_crop_pad"]
    margin = int(rng.uniform(lo, hi))
    y0 = max(0, rows.min() - margin)
    y1 = min(arr.shape[0], rows.max() + 1 + margin)
    x0 = max(0, cols.min() - margin)
    x1 = min(arr.shape[1], cols.max() + 1 + margin)
    return Image.fromarray(arr[y0:y1, x0:x1])
