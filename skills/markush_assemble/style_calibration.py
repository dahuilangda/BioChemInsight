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


def draw_molecule(mol: Chem.Mol, profile: dict, rng: random.Random,
                  opts_hook=None) -> Image:
    """Render a molecule (coords must already exist) to a tight-cropped
    grayscale PIL image in the profile's style. A uniform rotation is
    applied through the coordinates so labels stay upright, mirroring the
    arbitrary orientation of real patent crops.  `opts_hook`, when given,
    is called with the MolDrawOptions after the profile is applied —
    e.g. render_via_engine applies its patent black-white palette
    through it."""
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
    if opts_hook is not None:
        opts_hook(opts)
    if font_size > 0:
        opts.fixedFontSize = font_size
    # patent C_alpha convention: carbons never print H or C symbols.
    # RDKit derives a floating "H" label from the parsed explicit-H count
    # on chiral carbons and prints a bare "C" on some wedge geometries;
    # stripping the counts keeps SMILES/stereo identical (verified on the
    # seq53 gold), and the label override silences the remaining fallback.
    # degree-2 chiral carbons keep their explicit H (it defines the tag).
    for a in mol.GetAtoms():
        if a.GetSymbol() == "C" and a.GetDegree() >= 3:
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(False)
            opts.atomLabels[int(a.GetIdx())] = ""
    # patent wedge spec (wedge_spec.py): the planner picks the wedge bond
    # per stereocentre under collision + convention scoring, and the
    # glyphs are drawn as slim overlays.  RDKit's own wedging draws fat
    # triangles on arbitrary bonds (a piperidine ring bond landed a
    # glyph inside an NH crowd on the r7 seq68 render), so the molecule
    # is drawn wedge-less and the overlay carries the stereo.
    marks = []
    used_overlay = False
    try:
        from wedge_spec import plan_wedge_marks
        # page scale: the planner's collision constants live in OUTPUT
        # pixels, while the conformer is in layout units — scale once here
        import numpy as _np
        _conf = mol.GetConformer()
        _blens = [math.hypot(_conf.GetAtomPosition(b.GetBeginAtomIdx()).x
                             - _conf.GetAtomPosition(b.GetEndAtomIdx()).x,
                             _conf.GetAtomPosition(b.GetBeginAtomIdx()).y
                             - _conf.GetAtomPosition(b.GetEndAtomIdx()).y)
                  for b in mol.GetBonds()]
        _unit = float(_np.median(_blens)) or 1.0
        _page_scale = float(bond_len / _unit)
        marks, _report = plan_wedge_marks(mol, font_px=float(
            font_size or 16), page_scale=_page_scale)
        if not marks and _report.get("n_centers", 0) > 0:
            # a chiral molecule with zero planned marks means the planner
            # could not PROVE its mark set: drawing wedge-less would drop
            # the stereo from the image entirely -- bail to the stock
            # wedging path instead (fat glyphs, correct stereo)
            raise RuntimeError(
                f"empty wedge plan for chiral molecule: {_report}")
        if _report.get("fallbacks"):
            import sys as _sys
            print(f"[wedge_spec] fallback centres {_report['fallbacks']}"
                  " (writer marks drawn slim)", file=_sys.stderr)
        prep = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True,
                                                addChiralHs=False,
                                                wedgeBonds=False)
        for b in prep.GetBonds():
            if b.GetBondDir() != Chem.BondDir.NONE:
                b.SetBondDir(Chem.BondDir.NONE)
        # draw the prepped molecule AS IS: DrawMolecule would otherwise
        # re-run its own preparation (prepareMolsBeforeDrawing defaults
        # to True) and its fat stock wedging could sneak back in
        opts.prepareMolsBeforeDrawing = False
        drawer.DrawMolecule(prep)
        used_overlay = True
    except Exception:
        # planner failure must never cost the render: fall back to the
        # stock wedging path
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, kekulize=True)
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("L")
    if used_overlay and marks:
        from wedge_spec import draw_wedge_marks
        draw_wedge_marks(img, marks,
                         lambda i: (drawer.GetDrawCoords(i).x,
                                    drawer.GetDrawCoords(i).y),
                         line_px=float(profile["bond_line"][0]))

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
