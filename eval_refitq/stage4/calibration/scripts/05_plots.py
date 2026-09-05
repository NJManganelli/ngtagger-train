"""Calibration plots (big readable fonts):
  plots/learning_curve.png   macro AUC vs N (both views), fits + extrapolation,
                             embedded@0000 absolute bar band
  plots/gain_vs_N.png        paired gain(N) = AUC_1111 - AUC_0000
  plots/per_class_embedded_vs_scratch.png  per-class AUC, embedded vs scratch
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
PLOTS = os.path.join(CAL, "plots")

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 18,
    "legend.fontsize": 14, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "figure.facecolor": "white", "savefig.dpi": 150,
})
COL = {"1111": "#c62828", "0000": "#1565c0"}


def main():
    summ = json.load(open(os.path.join(CAL, "calibration_summary.json")))
    gaps = json.load(open(os.path.join(CAL, "paired_gaps.json")))
    bar = summ["inputs"]["bar_embedded_0000_test"]
    bar_std = summ["inputs"]["bar_boot_std"]

    # ---- learning curve ----
    fig, ax = plt.subplots(figsize=(11, 8))
    Ngrid = np.geomspace(200, 3e5, 400)
    for view in ["1111", "0000"]:
        v = summ["views"][view]
        Ns = [p["N"] for p in v["points"]]
        mu = [p["auc_mean"] for p in v["points"]]
        sd = [p["auc_std"] for p in v["points"]]
        ax.errorbar(Ns, mu, yerr=sd, fmt="o", ms=9, lw=2, capsize=4,
                    color=COL[view], label=f"scratch {view} (3 seeds)")
        e0, a, b = v["fit_macro_with_floor"]["params"]
        ax.plot(Ngrid, 1 - (e0 + a * Ngrid ** (-b)), "-", color=COL[view], lw=2.2,
                alpha=0.85,
                label=f"{view} fit: b={b:.2f}, AUC$_\\infty$={1 - e0:.3f}")
        a2, b2 = v["fit_macro_no_floor"]["params"]
        ax.plot(Ngrid, 1 - a2 * Ngrid ** (-b2), "--", color=COL[view], lw=1.6,
                alpha=0.55, label=f"{view} no-floor: b={b2:.2f}")
    ax.axhspan(bar - bar_std, bar + bar_std, color="0.75", alpha=0.5)
    ax.axhline(bar, color="0.2", lw=2.5, ls=":",
               label=f"embedded@0000 bar = {bar:.4f} $\\pm$ {bar_std:.4f}")
    nc = summ["n_cross"]["with_floor"]
    if nc.get("n_cross_jets_median"):
        ax.axvline(nc["n_cross_jets_median"], color="green", lw=2, ls="-.",
                   label=f"N_cross(median) $\\approx$ {nc['n_cross_jets_median']:.0f} jets")
    ax.set_xscale("log")
    ax.set_xlabel("training jets N")
    ax.set_ylabel("macro OvR AUC (fixed stage-4 test split)")
    ax.set_title("Learning curve vs embedded (central) SC4 NG bar")
    ax.set_ylim(0.45, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "learning_curve.png"))
    plt.close(fig)

    # ---- gain vs N ----
    fig, ax = plt.subplots(figsize=(10, 6.5))
    g = summ["gain_vs_N"]
    Ns = [x["N_1111"] for x in g]
    mu = [x["gain_mean"] for x in g]
    se = [x["gain_sem"] for x in g]
    ax.errorbar(Ns, mu, yerr=se, fmt="s-", ms=10, lw=2.2, capsize=5,
                color="#6a1b9a", label="gain(N), paired by seed")
    ax.axhline(0, color="0.3", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("training jets N")
    ax.set_ylabel("AUC$_{1111}$(N) $-$ AUC$_{0000}$(N)")
    ax.set_title("Refit-view gain vs training statistics")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "gain_vs_N.png"))
    plt.close(fig)

    # ---- per-class embedded vs scratch ----
    classes = list(gaps["views"]["0000"]["per_class"].keys())
    x = np.arange(len(classes))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, view in zip(axes, ["0000", "1111"]):
        pc = gaps["views"][view]["per_class"]
        emb = [pc[c]["embedded"] for c in classes]
        emb_e = [pc[c]["embedded_se_hm"] for c in classes]
        scr = [pc[c]["scratch_best"] for c in classes]
        scr_e = [pc[c]["scratch_se_hm"] for c in classes]
        npos = [pc[c]["n_pos"] for c in classes]
        ax.bar(x - 0.2, emb, 0.38, yerr=emb_e, capsize=4, color="0.35",
               label="embedded (central)")
        ax.bar(x + 0.2, scr, 0.38, yerr=scr_e, capsize=4, color=COL[view],
               alpha=0.85, label="scratch (best of 3)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(classes, npos)],
                           fontsize=13)
        ax.axhline(0.5, color="0.4", lw=1, ls=":")
        ax.set_title(f"view {view} — test split")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("one-vs-rest AUC")
    axes[0].set_ylim(0.4, 1.02)
    fig.suptitle("Per-class AUC: embedded central NG tagger vs scratch stage-4 "
                 "(same test jets; HM errors)", fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "per_class_embedded_vs_scratch.png"))
    plt.close(fig)
    print("plots written to", PLOTS)


if __name__ == "__main__":
    main()
