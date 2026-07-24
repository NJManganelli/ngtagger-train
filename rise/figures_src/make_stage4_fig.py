"""Stage-4 jet-tagger variant-matrix figure: macro AUC heatmap, views x features.

Data source: eval_refitq/stage4/stage4_summary.json (11-cell matrix, unified
coherent nanos 1111/1100/0000, 3 seeds/cell, best-of-seed macro AUC).

Run:  pixi run python rise/figures_src/make_stage4_fig.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
SRC = os.path.join(ROOT, "eval_refitq", "stage4", "stage4_summary.json")
OUT = os.path.join(HERE, os.pardir, "figures", "stage4_auc_matrix.png")

VIEWS = ["1111", "1100", "0000"]
SETS = [("baseline", "baseline"), ("refitbdt", "+ refit-BDT score"),
        ("vertexdxy", "+ vertex dxy"), ("both", "+ both")]


def main():
    cells = json.load(open(SRC))["cells"]
    M = np.full((len(VIEWS), len(SETS)), np.nan)
    S = np.full((len(VIEWS), len(SETS)), np.nan)
    for i, v in enumerate(VIEWS):
        for j, (key, _) in enumerate(SETS):
            c = cells.get(f"{v}__{key}")
            if c:
                M[i, j] = c["best_macro_auc"]
                S[i, j] = c["macro_auc_std"]

    plt.rcParams.update({"font.size": 16, "axes.titlesize": 18})
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, cmap="YlGnBu", vmin=0.67, vmax=0.74, aspect="auto")
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([lab for _, lab in SETS], fontsize=15)
    ax.set_yticks(range(len(VIEWS)))
    ax.set_yticklabels([
        "1111\nrefit, all 4 IT layers", "1100\nrefit, layers 1+2", "0000\nOT-only baseline",
    ], fontsize=14)
    for i in range(len(VIEWS)):
        for j in range(len(SETS)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=15, color="#999999")
            else:
                ax.text(j, i, f"{M[i, j]:.3f}\n$\\pm${S[i, j]:.3f}", ha="center",
                        va="center", fontsize=15, fontweight="bold",
                        color="white" if M[i, j] > 0.715 else "#1a1a1a")
    base_1111 = M[0, 0]
    base_1100 = M[1, 0]
    base_0000 = M[2, 0]
    ax.set_title(
        "Jet-tagger macro AUC (8-flavor, best of 3 seeds $\\pm$ seed spread)\n"
        f"coherent refit downstream vs OT-only: "
        f"+{base_1111 - base_0000:.3f} (1111), +{base_1100 - base_0000:.3f} (1100)",
        fontsize=16,
    )
    fig.colorbar(im, ax=ax, label="macro AUC")
    ax.text(0.0, -0.16,
            "Low stats: ~5.3k train / ~1.3k test jets per view — only the "
            "0.03–0.04 refit-vs-OT-only gap exceeds the seed spread.",
            transform=ax.transAxes, fontsize=13, color="#7a3b00")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
