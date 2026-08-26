"""Manuscript figure: per-patent valid-SMILES rate distribution (293 patents)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

wb = openpyxl.load_workbook("/data/BioChemInsight/paper/patent_statistics.xlsx", read_only=True)
ws = wb["Per-Patent Statistics"]
rows = list(ws.iter_rows(values_only=True))[1:]
rates = [float(r[5]) for r in rows if r[3] and r[3] >= 1]
assert len(rates) == 293, len(rates)

fig, ax = plt.subplots(figsize=(3.35, 2.5), dpi=300)
ax.hist(rates, bins=np.arange(0, 105, 5), color="#31597E", edgecolor="white", linewidth=0.4)
ax.axvline(91.5, color="black", linestyle="--", linewidth=0.8)
ax.annotate("median 91.5%", xy=(91.5, 160), xytext=(60, 158), fontsize=8,
            arrowprops=dict(arrowstyle="-", linewidth=0.6))
ax.set_xlabel("Valid-SMILES rate per patent (%)", fontsize=8)
ax.set_ylabel("Number of patents", fontsize=8)
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 101, 20))
ax.tick_params(labelsize=8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig("/data/BioChemInsight/paper/figures/fig_validity_distribution.png")
print("written, n =", len(rates), "median", float(np.median(rates)))
