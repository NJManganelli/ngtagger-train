"""Stage-3 refit-quality BDT figure: val AUC across all 15 layer configs.

Data source: eval_refitq/stage3/models/stage3_summary.json (committed-adjacent
training summary; 10-file multi-mode nano, best of 5 seeds per config).

Run:  pixi run python rise/figures_src/make_stage3_fig.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
SRC = os.path.join(ROOT, "eval_refitq", "stage3", "models", "stage3_summary.json")
OUT = os.path.join(HERE, os.pardir, "figures", "stage3_auc_configs.png")


def main():
    res = json.load(open(SRC))["results"]
    # order by number of active layers, then by mask (outer configs last)
    items = sorted(res.items(), key=lambda kv: (kv[1]["activeSP"].count("1"), kv[1]["activeSP"]))
    labels = [f"{v['activeSP']}" for _, v in items]
    aucs = [v["best_val_auc"] for _, v in items]
    ntrk = [v["n_refit_tracks"] for _, v in items]
    nneg = [v["n_neg"] for _, v in items]

    plt.rcParams.update({"font.size": 16, "axes.titlesize": 19, "axes.labelsize": 17})
    fig, ax = plt.subplots(figsize=(14, 6.2))
    nlayers = [l.count("1") for l in labels]
    cmap = {1: "#a6bddb", 2: "#74a9cf", 3: "#2b8cbe", 4: "#045a8d"}
    bars = ax.bar(range(len(labels)), aucs, color=[cmap[n] for n in nlayers])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.set_xlabel("active SmartPixels layer mask (L1 L2 L3 L4)")
    ax.set_ylabel("genuine-vs-fake val AUC (best of 5 seeds)")
    ax.set_ylim(0.94, 0.99)
    ax.grid(axis="y", alpha=0.3)
    lo, hi = min(aucs), max(aucs)
    ax.axhspan(lo, hi, color="#fdd49e", alpha=0.35, zorder=0)
    ax.text(0.02, 0.95, f"all 15 configs: AUC {lo:.3f}–{hi:.3f}",
            transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")
    for i, (b, a) in enumerate(zip(bars, aucs)):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.0008, f"{a:.3f}",
                ha="center", va="bottom", fontsize=11, rotation=90)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[n]) for n in (1, 2, 3, 4)]
    ax.legend(handles, [f"{n} active layer{'s' if n > 1 else ''}" for n in (1, 2, 3, 4)],
              loc="lower right", fontsize=13)
    ax.set_title(
        "Refit-quality BDT (24-feature v1): genuine-vs-fake AUC across ALL 15 layer configs\n"
        f"({min(ntrk):,}–{max(ntrk):,} refit tracks/config, "
        f"{min(nneg):,}–{max(nneg):,} fakes; 1000 PU events)",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
