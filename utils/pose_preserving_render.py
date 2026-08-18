"""Pose-preserving rendering for the structure confidence visual review.

The recognized molblock keeps the image-derived 2D pose, so rendering it with
its own conformer coordinates yields a panel with the same layout as the
patent crop; the vision review then looks for local disagreements instead of
comparing across drawing styles.  Entry points return None on any parsing or
rendering problem.
"""
from __future__ import annotations

import os
import tempfile

# RDKit drawing canvas: width fixed, height from the conformer aspect ratio.
# Kept >= 640 so atom labels stay legible after the panel compositing pass.
REVIEW_RENDER_WIDTH = 720
REVIEW_RENDER_MIN_HEIGHT = 240
REVIEW_RENDER_MAX_ASPECT = 3.2  # height/width cap for tall molecules

# Two-panel review composite geometry, chosen by calibration against the
# production vision endpoint.
REVIEW_PANEL_HEIGHT = 720
REVIEW_PANEL_MIN_WIDTH = 640    # each panel is at least this wide
REVIEW_PANEL_MAX_WIDTH = 1560   # content wider than this is scaled down to fit
REVIEW_PANEL_MARGIN = 44        # white border on every side
REVIEW_PANEL_TOP_BAND = 68      # top white band carrying the label text
REVIEW_PANEL_FONT_SCALE = 0.62

PANEL_LABEL_SOURCE = 'A: SOURCE'
PANEL_LABEL_PARSED = 'B: PARSED (same layout)'


def _suppress_rdkit_logs():
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        pass


def mol_from_molblock_tolerant(molblock):
    """Parse a molblock leniently: no sanitization, then best-effort sanitize.

    Returns the mol — possibly unsanitized, still drawable with raw bond
    orders — or ``None`` when the block cannot be parsed at all.
    """
    if not molblock:
        return None
    try:
        from rdkit import Chem
        from utils.molecule_2d_layout import normalize_molblock_header
    except ImportError:
        return None
    try:
        mol = Chem.MolFromMolBlock(
            normalize_molblock_header(molblock), sanitize=False, removeHs=False
        )
    except Exception:
        return None
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            pass  # draw with raw bond orders from the block
    return mol


def _conformer_aspect(mol):
    """Height/width bbox aspect of the first conformer, or ``None``."""
    try:
        import numpy as np
        conf = mol.GetConformer()
        pts = np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
             for i in range(mol.GetNumAtoms())],
            dtype=float,
        )
        if pts.shape[0] == 0 or not np.isfinite(pts).all():
            return None
        span = pts.max(axis=0) - pts.min(axis=0)
        if span.max() <= 0:
            return None
        return min(REVIEW_RENDER_MAX_ASPECT, max(0.30, span[1] / max(span[0], 1e-6)))
    except Exception:
        return None


def _kekulized_copy(mol):
    """Kekulized copy for patent-style drawing, or ``None`` when the molecule
    cannot be kekulized (the caller then draws aromatic circles instead)."""
    try:
        from rdkit import Chem
        out = Chem.Mol(mol)
        Chem.Kekulize(out, clearAromaticFlags=True)
        return out
    except Exception:
        return None


def render_mol_own_coords(mol, width=None, height=None):
    """Render *mol* with its existing conformer coordinates (no re-layout).

    Exactly one of *width* / *height* should be given; the other is derived
    from the conformer aspect so the depiction is drawn at native size.
    Returns a BGR ndarray or ``None``.
    """
    try:
        import cv2
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None
    if mol is None or mol.GetNumConformers() == 0:
        return None
    aspect = _conformer_aspect(mol)
    if aspect is None:
        return None
    try:
        if height and not width:
            width = max(1, int(round(height / aspect)))
        elif not width:
            width = REVIEW_RENDER_WIDTH
        canvas_height = max(REVIEW_RENDER_MIN_HEIGHT, int(round(width * aspect)))
        drawer = rdMolDraw2D.MolDraw2DCairo(int(width), int(canvas_height))
        kek_mol = _kekulized_copy(mol)
        try:
            drawer.drawOptions().kekulize = kek_mol is not None
        except AttributeError:
            pass  # older RDKit: aromatic rings drawn with inner circles
        drawer.DrawMolecule(kek_mol if kek_mol is not None else mol)
        drawer.FinishDrawing()
        path = tempfile.mktemp(suffix='_pose_render.png')
        with open(path, 'wb') as handle:
            handle.write(drawer.GetDrawingText())
        img = cv2.imread(path)
        try:
            os.remove(path)
        except OSError:
            pass
        return img
    except Exception:
        return None


def render_molblock_pose_preserving(molblock, width=REVIEW_RENDER_WIDTH, height=None):
    """Render a molblock with its own coordinates (pose-preserving)."""
    _suppress_rdkit_logs()
    mol = mol_from_molblock_tolerant(molblock)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    return render_mol_own_coords(mol, width=width, height=height)


def molblock_has_usable_pose(molblock):
    """True when the molblock parses, carries a conformer, and its
    coordinates span a non-degenerate finite bbox (so rendering with its own
    coordinates truly preserves the image pose)."""
    mol = mol_from_molblock_tolerant(molblock) if molblock else None
    return (mol is not None and mol.GetNumConformers() > 0
            and _conformer_aspect(mol) is not None)


def render_structure_for_review_panel(molblock,
                                      content_height=None,
                                      panel_height=REVIEW_PANEL_HEIGHT,
                                      margin=REVIEW_PANEL_MARGIN,
                                      top_band=REVIEW_PANEL_TOP_BAND,
                                      min_width=REVIEW_PANEL_MIN_WIDTH,
                                      max_width=REVIEW_PANEL_MAX_WIDTH):
    """Render the molblock at the native size of a review panel's content box.

    Drawing directly at the final content height keeps atom labels at full
    size for the vision model.  Returns None when the molblock carries no
    usable pose.
    """
    _suppress_rdkit_logs()
    if not molblock_has_usable_pose(molblock):
        return None
    if content_height is None:
        content_height = int(panel_height - top_band - margin)
    mol = mol_from_molblock_tolerant(molblock)
    width = int(round(content_height / _conformer_aspect(mol)))
    width = min(max(width, min_width), max_width)
    return render_mol_own_coords(mol, width=width)


def build_review_panel(content_img, label,
                       panel_height=REVIEW_PANEL_HEIGHT,
                       panel_min_width=REVIEW_PANEL_MIN_WIDTH,
                       panel_max_width=REVIEW_PANEL_MAX_WIDTH,
                       margin=REVIEW_PANEL_MARGIN,
                       top_band=REVIEW_PANEL_TOP_BAND):
    """Place a content image on a white panel with margins and a top label.

    The label is drawn inside the top white band so it never covers the
    structure's ink.  Content is scaled (LANCZOS4) to fill the content area
    while preserving its aspect ratio; oversized content is scaled down to
    *panel_max_width* and letterboxed vertically.  Returns a BGR ndarray or
    ``None``.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    if content_img is None or content_img.size == 0:
        return None
    try:
        content_h = int(panel_height - top_band - margin)
        if content_h < 40:
            return None
        scale = content_h / float(content_img.shape[0])
        scaled_w = max(1, int(round(content_img.shape[1] * scale)))
        content = None
        if scaled_w <= panel_max_width:
            content = cv2.resize(content_img, (scaled_w, content_h),
                                 interpolation=cv2.INTER_LANCZOS4)
            y0 = int(top_band)
        else:
            width_scale = panel_max_width / float(scaled_w)
            fit_h = max(1, int(round(content_h * width_scale)))
            content = cv2.resize(content_img, (panel_max_width, fit_h),
                                 interpolation=cv2.INTER_LANCZOS4)
            y0 = int(top_band) + (content_h - fit_h) // 2
        panel_w = max(int(panel_min_width), content.shape[1] + 2 * int(margin))
        panel = np.full((int(panel_height), panel_w, 3), 255, dtype=np.uint8)
        x0 = (panel_w - content.shape[1]) // 2
        panel[y0:y0 + content.shape[0], x0:x0 + content.shape[1]] = content
        cv2.putText(
            panel, str(label), (int(margin), int(top_band) - 18),
            cv2.FONT_HERSHEY_SIMPLEX, REVIEW_PANEL_FONT_SCALE,
            (0, 0, 0), 2, cv2.LINE_AA,
        )
        return panel
    except Exception:
        return None


def compose_review_composite(source_img, rendered_img,
                             source_label=PANEL_LABEL_SOURCE,
                             parsed_label=PANEL_LABEL_PARSED,
                             panel_height=REVIEW_PANEL_HEIGHT,
                             panel_min_width=REVIEW_PANEL_MIN_WIDTH,
                             margin=REVIEW_PANEL_MARGIN,
                             top_band=REVIEW_PANEL_TOP_BAND):
    """Build the [source | parsed] comparison image for the vision review.

    Both panels are equal-height, carry their labels in a white top band,
    and are separated by a white gap.  Returns a BGR ndarray or ``None``.
    """
    source_panel = build_review_panel(
        source_img, source_label, panel_height=panel_height,
        panel_min_width=panel_min_width, margin=margin, top_band=top_band)
    parsed_panel = build_review_panel(
        rendered_img, parsed_label, panel_height=panel_height,
        panel_min_width=panel_min_width, margin=margin, top_band=top_band)
    if source_panel is None or parsed_panel is None:
        return None
    try:
        import cv2
        import numpy as np
        gap = np.full((int(panel_height), int(margin), 3), 255, dtype=np.uint8)
        return cv2.hconcat([source_panel, gap, parsed_panel])
    except Exception:
        return None
