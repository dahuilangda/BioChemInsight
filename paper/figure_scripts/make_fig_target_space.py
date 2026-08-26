"""Manuscript figure: target-space comparison (patent-derived vs ChEMBL).

Per therapeutic target, compounds from each source are pooled into one Morgan
fingerprint occupancy vector (radius 2, 2048 bits); the 30 vectors (15 target
groups x 2 sources) are embedded in 2D with UMAP. Same-target pairs are
connected to visualize source divergence per target.

Inputs: paper/patent_statistics.xlsx (patent -> target),
        data/seatable_validation/outputs/<patent>/structures.csv,
        paper/figure_data/chembl_target_compounds.json
Output: paper/figures/fig_target_space.png,
        paper/figure_data/target_space_points.csv
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import umap
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
CACHE = os.path.join(ROOT, "figure_data", "chembl_target_compounds.json")
OUT_PNG = os.path.join(ROOT, "figures", "fig_target_space.png")
OUT_CSV = os.path.join(ROOT, "figure_data", "target_space_points.csv")

GROUP_SYMBOLS = {
    "Axl-Mer-Tyro3": ["AXL", "MERTK", "TYRO3"],
    "CDK8": ["CDK8"],
    "DHODH": ["DHODH"],
    "EGFR": ["EGFR"],
    "GSPT1": ["GSPT1"],
    "HPK1": ["MAP4K1"],
    "JAK1": ["JAK1"],
    "NLRP3": ["NLRP3"],
    "NLRP3+TNF": ["NLRP3", "TNF"],
    "TNKS2": ["TNKS2"],
    "WRN": ["WRN"],
    "cMet": ["MET"],
    "cMet+Axl-Mer-Tyro3": ["MET", "AXL", "MERTK", "TYRO3"],
    "cMet+Axl-Mer-Tyro3+VEGFR-2": ["MET", "AXL", "MERTK", "TYRO3", "KDR"],
    "cMet+VEGFR-2+RET": ["MET", "KDR", "RET"],
}

chembl = json.load(open(CACHE))


def patent_targets():
    wb = openpyxl.load_workbook(os.path.join(ROOT, "patent_statistics.xlsx"), read_only=True)
    ws = wb["Per-Patent Statistics"]
    out = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        pat, tgt = row[0], row[1]
        if pat and tgt and tgt in GROUP_SYMBOLS:
            out.append((str(pat), tgt))
    return out


def pool_vector(smiles_list):
    occ = np.zeros(2048, dtype=np.float64)
    n = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float64)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        occ += arr
        n += 1
    return (occ / n) if n else None, n


def main():
    targets = patent_targets()
    patent_smiles = {g: set() for g in GROUP_SYMBOLS}
    for pat, tgt in targets:
        f = os.path.join(REPO, "data", "seatable_validation", "outputs", pat, "structures.csv")
        if not os.path.exists(f):
            continue
        with open(f, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                smi = (row.get("SMILES") or "").strip()
                if smi and Chem.MolFromSmiles(smi) is not None:
                    patent_smiles[tgt].add(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))

    vectors, labels, sources, counts = [], [], [], []
    skip = []
    for group, symbols in GROUP_SYMBOLS.items():
        pv, pn = pool_vector(sorted(patent_smiles[group]))
        cs, used = [], []
        for s in symbols:
            smis = chembl.get(s, {}).get("smiles") or []
            if len(smis) >= 10:
                cs.extend(smis)
                used.append(s)
        cv, cn = pool_vector(sorted(set(cs)))
        if pv is None or pn < 10 or cv is None or cn < 15:
            skip.append((group, pn, cn, [s for s in symbols if s not in used]))
            continue
        vectors.extend([pv, cv])
        labels.extend([group, group])
        sources.extend(["patent", "chembl"])
        counts.extend([pn, cn])
    print("groups plotted:", len(labels) // 2)
    for g, pn, cn, missing_syms in skip:
        print(f"  skipped {g}: patent={pn} chembl={cn} missing={missing_syms}")

    X = np.array(vectors)
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.25, n_components=2,
                        metric="euclidean", random_state=42)
    emb = reducer.fit_transform(X)

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "source", "umap1", "umap2", "compounds"])
        for t, s, e, n in zip(labels, sources, emb, counts):
            w.writerow([t, s, round(float(e[0]), 4), round(float(e[1]), 4), n])

    fig, ax = plt.subplots(figsize=(6.85, 4.6), dpi=300)
    pts = {(t, s): e for t, s, e in zip(labels, sources, emb)}
    for group in GROUP_SYMBOLS:
        if (group, "patent") not in pts:
            continue
        p, c = pts[(group, "patent")], pts[(group, "chembl")]
        ax.plot([p[0], c[0]], [p[1], c[1]], color="#bbbbbb", linewidth=0.7, zorder=1)
    for (t, s), e in pts.items():
        if s == "patent":
            ax.scatter(e[0], e[1], s=42, color="#B2453A", edgecolor="black",
                       linewidth=0.4, zorder=3, label="Patent (BioChemInsight)")
        else:
            ax.scatter(e[0], e[1], s=46, facecolor="none", edgecolor="#2F5C88",
                       linewidth=1.1, zorder=3, label="ChEMBL")

    # --- labels with collision avoidance: measure real text extents, then
    # greedily shift overlapping labels away from labels and data points ---
    label_groups = [g for g in GROUP_SYMBOLS if (g, "patent") in pts]
    texts = []
    for g in label_groups:
        p, c = pts[(g, "patent")], pts[(g, "chembl")]
        t = ax.annotate(g.replace("+", "\n+"), ((p[0] + c[0]) / 2, (p[1] + c[1]) / 2),
                        fontsize=6.4, ha="center", va="center", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))
        texts.append(t)
    fig.canvas.draw()
    inv = ax.transData.inverted()

    def to_data(bbox):
        (x0, y0), (x1, y1) = inv.transform((bbox.x0, bbox.y0)), inv.transform((bbox.x1, bbox.y1))
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def label_box(t):
        patch = t.get_bbox_patch()
        if patch is not None:
            return to_data(patch.get_window_extent(renderer=fig.canvas.get_renderer()))
        return to_data(t.get_window_extent(renderer=fig.canvas.get_renderer()))

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xspan, yspan = xlim[1] - xlim[0], ylim[1] - ylim[0]
    axes_px = ax.get_window_extent()
    xpad = 20 * xspan / axes_px.width   # marker radius ~17px + fringe
    ypad = 20 * yspan / axes_px.height
    point_boxes = [(e[0] - xpad, e[1] - ypad, e[0] + xpad, e[1] + ypad) for e in emb]

    def overlap(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    # candidate offsets as expanding rings around each label's home position
    cands = []
    for ring in range(0, 9):
        for iy in range(0, ring + 1):
            ix = ring - iy
            for sy in ({0} if iy == 0 else (1, -1)):
                for sx in ({0} if ix == 0 else (1, -1)):
                    off = (iy * sy * yspan * 0.05, ix * sx * xspan * 0.035)
                    if off not in cands:
                        cands.append(off)

    placed = []
    for t in texts:
        home = t.get_position()
        best = home
        for dy, dx in cands:
            t.set_position((home[0] + dx, home[1] + dy))
            fig.canvas.draw()
            box = label_box(t)
            inside = xlim[0] < box[0] and box[2] < xlim[1] and ylim[0] < box[1] and box[3] < ylim[1]
            if inside and not any(overlap(box, o) for o in placed + point_boxes):
                best = (home[0] + dx, home[1] + dy)
                break
        t.set_position(best)
        fig.canvas.draw()
        placed.append(label_box(t))
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#B2453A", markeredgecolor="black", label="Patent (BioChemInsight)"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="#2F5C88", label="ChEMBL"),
    ]
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=8)
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT_PNG)
    print("saved", OUT_PNG)


if __name__ == "__main__":
    main()
