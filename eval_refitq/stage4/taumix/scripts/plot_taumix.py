"""Taumix deliverable plots:
  plots/per_class_auc.png   per-class AUC on the mixed test set, per view:
                            ttbar-only ctrl vs tau-mixed vs embedded (+refitbdt
                            for 1111), with stage-4 ttbar-only anchors marked.
  plots/tau_roc_overlay.png tau ROC curves (taup/taum, one-vs-rest) on the
                            mixed test set: mixed-trained vs ttbar-only-trained
                            vs embedded NG tagger.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAUMIX = os.path.dirname(HERE)
STAGE4 = os.path.dirname(TAUMIX)
PLOTS = os.path.join(TAUMIX, "plots")
DUMPS = os.path.join(TAUMIX, "dumps")

with open(os.path.join(TAUMIX, "taumix_summary.json")) as f:
    S = json.load(f)
with open(os.path.join(STAGE4, "stage4_summary.json")) as f:
    S4 = json.load(f)
CLASSES = S["meta"]["class_labels"]
VIEWS = ["1111", "0000"]


def cell_agg(key, setname="mixed_test"):
    return S["cells"][key]["agg"][setname]


def emb(view, setname="mixed_test"):
    return S["embedded"][view][setname]


def per_class_plot():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    x = np.arange(len(CLASSES))
    w = 0.22
    for ax, view in zip(axes, VIEWS):
        ctrl = cell_agg(f"{view}__baseline_ttonly_ctrl")
        mix = cell_agg(f"{view}__baseline_taumix")
        e = emb(view)
        bars = [("ttbar-only (ctrl)", ctrl, "#7f8fa6"),
                ("tau-mixed", mix, "#e1701a")]
        if view == "1111" and "1111__refitbdt_taumix" in S["cells"]:
            bars.append(("tau-mixed +refitbdt",
                         cell_agg("1111__refitbdt_taumix"), "#c23616"))
        n = len(bars) + 1
        for bi, (lab, a, col) in enumerate(bars):
            vals = [a[c]["mean"] for c in CLASSES]
            errs = [a[c]["std"] for c in CLASSES]
            ax.bar(x + (bi - n / 2 + 0.5) * w, vals, w, yerr=errs, capsize=2,
                   label=lab, color=col)
        ev = [e["per_class"][c]["auc"] for c in CLASSES]
        ee = [e["per_class"][c]["se_hm"] for c in CLASSES]
        ax.bar(x + (len(bars) - n / 2 + 0.5) * w, ev, w, yerr=ee, capsize=2,
               label="embedded NG", color="#273c75")
        anch = S4["cells"][f"{view}__baseline"]["best_per_flavor_auc"]
        ax.scatter(x, [anch[c] for c in CLASSES], marker="_", s=220, c="k",
                   zorder=5, label="stage-4 anchor (ttbar test)")
        ax.set_xticks(x, CLASSES, rotation=30)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
        ax.set_title(f"view {view} — mixed test set")
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("one-vs-rest AUC")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Per-class AUC: tau-mixed vs ttbar-only vs embedded (bars: mean±seed std; embedded: ±HM SE)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "per_class_auc.png"), dpi=150)
    plt.close(fig)


def best_seed(key, setname="mixed_test"):
    recs = S["cells"][key]["per_seed"]
    return max(recs, key=lambda r: r[setname]["macro"] or 0)["seed"]


def tau_roc_plot():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), sharex=True, sharey=True)
    for vi, view in enumerate(VIEWS):
        # embedded probs re-derived from a dump? embedded probs are not in cell
        # dumps; recompute alignment from the baseline_taumix dump ordering via
        # a dedicated embedded dump saved by run_taumix (probs_emb aligned to
        # test order) -- stored in dumps/embedded_<view>.npz
        with np.load(os.path.join(DUMPS, f"embedded_{view}.npz"),
                     allow_pickle=True) as f:
            y = f["y"]
            pe = f["probs"]
        curves = {
            "tau-mixed": os.path.join(
                DUMPS, f"{view}__baseline_taumix__s{best_seed(f'{view}__baseline_taumix')}.npz"),
            "ttbar-only": os.path.join(
                DUMPS, f"{view}__baseline_ttonly_ctrl__s{best_seed(f'{view}__baseline_ttonly_ctrl')}.npz"),
        }
        for ci, (tau, tidx) in enumerate((("taup", 4), ("taum", 5))):
            ax = axes[vi][ci]
            for lab, path in curves.items():
                with np.load(path, allow_pickle=True) as f:
                    yd, pd = f["y"], f["probs"]
                fpr, tpr, _ = roc_curve(yd == tidx, pd[:, tidx])
                a = S["cells"][f"{view}__baseline_taumix" if lab == "tau-mixed"
                              else f"{view}__baseline_ttonly_ctrl"]["agg"]["mixed_test"][tau]
                ax.plot(fpr, tpr, lw=1.6,
                        label=f"{lab} (AUC {a['mean']:.3f}±{a['std']:.3f})")
            fpr, tpr, _ = roc_curve(y == tidx, pe[:, tidx])
            ea = S["embedded"][view]["mixed_test"]["per_class"][tau]
            ax.plot(fpr, tpr, lw=1.6, color="#273c75",
                    label=f"embedded NG (AUC {ea['auc']:.3f}±{ea['se_hm']:.3f})")
            ax.plot([0, 1], [0, 1], "k:", lw=0.6)
            ax.set_title(f"view {view} — {tau} (n_pos={ea['n_pos']})", fontsize=10)
            ax.legend(fontsize=7.5, loc="lower right")
    for ax in axes[1]:
        ax.set_xlabel("false positive rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("true positive rate")
    fig.suptitle("Tau one-vs-rest ROC on the mixed stratified test set")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "tau_roc_overlay.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    per_class_plot()
    tau_roc_plot()
    print("plots written:", PLOTS)
