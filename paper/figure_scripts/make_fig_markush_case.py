"""Manuscript figure: literature series (Markush R-group) expansion case.

Panel A: source material from Sci Rep benzimidazole paper — the series design
scaffold (variable X) and the activity table (compounds 6a-r, substituents,
experimental IC50). Panel B: six of the 18 members resolved by the pipeline
with their linked experimental IC50 values.

Reproducibility: paper/figure_data/markush_case/ holds the source crops and
the synthesized member rows (regenerate the latter with
evals/literature_eval.py). Missing crops are re-extracted from the source PDF
below automatically.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimage
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "figure_data", "markush_case")
PDF = os.path.join(ROOT, "..", "data", "literature_test", "scirep_benzimidazole.pdf")
CROP_SPECS = [  # page, (x0, y0, x1, y1) in PDF points, output name
    (2, (85, 428, 568, 727), "source_figure1.png"),
    (4, (85, 146, 572, 404), "source_table1.png"),
]

os.makedirs(SRC, exist_ok=True)
for page_no, box, name in CROP_SPECS:
    path = os.path.join(SRC, name)
    if not os.path.exists(path):
        import fitz

        doc = fitz.open(os.path.abspath(PDF))
        pix = doc[page_no - 1].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), clip=fitz.Rect(*box))
        pix.save(path)
        print("re-extracted", path)

SHOW = [
    ("6a", "Benzyl", "153.7 \u00b1 0.9"),
    ("6j", "4-Bromobenzyl", "28.0 \u00b1 0.6"),
    ("6k", "3,4-Dichlorobenzyl", "99.4 \u00b1 0.7"),
    ("6o", "2,3-Dimethylbenzyl", "126.9 \u00b1 0.5"),
    ("6p", "4-Nitrobenzyl", "89.2 \u00b1 1.3"),
    ("6r", "Ethyl", "300.7 \u00b1 2.0"),
]

rows = {r["COMPOUND_ID"]: r for r in json.load(open(os.path.join(SRC, "members.json")))["rows"]}

fig = plt.figure(figsize=(6.85, 6.3), dpi=300)
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.16,
                      left=0.02, right=0.98, top=0.93, bottom=0.015)

# --- Panel A: source excerpts ---
axA = fig.add_subplot(gs[0])
axA.axis("off")
fig1 = mpimage.imread(os.path.join(SRC, "source_figure1.png"))
tab1 = mpimage.imread(os.path.join(SRC, "source_table1.png"))


def trim_white(arr, pad=14):
    """Crop uniform white margins so displayed content fills its box."""
    ink = (arr[..., :3].min(axis=2) < 246)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return arr
    r0, r1 = max(0, rows[0] - pad), min(arr.shape[0] - 1, rows[-1] + pad)
    c0, c1 = max(0, cols[0] - pad), min(arr.shape[1] - 1, cols[-1] + pad)
    return arr[r0:r1 + 1, c0:c1 + 1]


fig1, tab1 = trim_white(fig1), trim_white(tab1)
h_fig, w_fig = fig1.shape[:2]
h_tab, w_tab = tab1.shape[:2]
# equal display heights, widths proportional
H = 1.0
wA = H * w_fig / h_fig
wB = H * w_tab / h_tab
total = wA + wB + 0.03
scale = 1.0 / total
axA.imshow(fig1, extent=[0, wA * scale, 0, H], aspect="auto", interpolation="lanczos")
axA.imshow(tab1, extent=[wA * scale + 0.03, (wA + 0.03 + wB) * scale, 0, H], aspect="auto", interpolation="lanczos")
axA.set_xlim(0, 1)
axA.set_ylim(0, 1)
axA.text(0.0, 1.06, "A", transform=axA.transAxes, fontsize=11, fontweight="bold", va="bottom")
axA.text(0.005, 1.015, "series definition (scaffold with variable X)", transform=axA.transAxes, fontsize=7.5, va="bottom", style="italic")
axA.text((wA + 0.03 + wB / 2) * scale, 1.015, "activity table (6a\u2013r, experimental IC\u2085\u2080)", transform=axA.transAxes, fontsize=7.5, va="bottom", style="italic", ha="center")

# arrow between panels
axA.annotate("", xy=(0.5, -0.10), xytext=(0.5, -0.02), xycoords="axes fraction",
             arrowprops=dict(arrowstyle="-|>", linewidth=1.0, color="black"))
axA.text(0.515, -0.06, "series expansion \u2192 member structures \u2192 activity linking", transform=axA.transAxes,
         fontsize=7.5, va="center", ha="left")

# --- Panel B: resolved members ---
axB = fig.add_subplot(gs[1])
axB.axis("off")
axB.set_xlim(0, 3)
axB.set_ylim(-0.05, 2.30)

# structure cells must have the same pixel-per-unit aspect as their bitmap,
# otherwise rings stretch. Compute the axes' px density before adding images.
fig.canvas.draw()
_bbox = axB.get_window_extent()
_px_per_x = _bbox.width / 3.0
_px_per_y = _bbox.height / 2.35
_density_ratio = _px_per_x / _px_per_y

IMG_W, IMG_H = 420, 330
Y_SPAN = 0.74
X_SPAN = Y_SPAN * (IMG_W / IMG_H) / _density_ratio

for idx, (mid, subst, ic50) in enumerate(SHOW):
    col, row = idx % 3, idx // 3
    mol = Chem.MolFromMolBlock(rows[mid]["MOLBLOCK"])
    img = Draw.MolToImage(mol, size=(IMG_W, IMG_H), kekulize=True)
    cx = col + 0.5
    y1 = 2.10 - row * 1.16
    y0 = y1 - Y_SPAN
    axB.imshow(img, extent=[cx - X_SPAN / 2, cx + X_SPAN / 2, y0, y1], aspect="auto", interpolation="lanczos")
    axB.text(cx, y0 - 0.08, f"{mid} \u00b7 {subst}", ha="center", fontsize=8)
    axB.text(cx, y0 - 0.22, f"IC\u2085\u2080 = {ic50} \u00b5M", ha="center", fontsize=8, color="#333333")
axB.text(0.0, 1.06, "B", transform=axB.transAxes, fontsize=11, fontweight="bold", va="bottom")
axB.text(0.025, 1.02, "six of 18 resolved members (all 18 matched reference structures exactly)", transform=axB.transAxes, fontsize=7.5, va="bottom", style="italic")

fig.savefig(os.path.join(ROOT, "figures", "fig_series_case.png"), dpi=300)
print("saved fig_series_case.png")
