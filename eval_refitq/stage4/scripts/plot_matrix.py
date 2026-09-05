"""Bar plots of the Stage-4 matrix: macro AUC and b-AUC per view x feature-set,
with seed-spread error bars. Saves PNGs under eval_refitq/stage4/plots/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/stage4"
with open(os.path.join(OUT, "stage4_summary.json")) as f:
    cells = json.load(f)["cells"]

VIEWS = ["1111", "1100", "0000"]
FEATS = [("baseline", "baseline"), ("refitbdt", "+refitBDT"),
         ("vertexdxy", "+vtxDxy"), ("both", "+both")]
colors = {"baseline": "#888", "refitbdt": "#1f77b4",
          "vertexdxy": "#2ca02c", "both": "#d62728"}


def panel(ax, metric_fn, title, ylabel):
    x = np.arange(len(VIEWS))
    w = 0.2
    for j, (fk, lab) in enumerate(FEATS):
        vals, errs = [], []
        for v in VIEWS:
            key = f"{v}__{fk}"
            val, err = metric_fn(cells.get(key, {}))
            vals.append(val if val is not None else np.nan)
            errs.append(err if err is not None else 0)
        ax.bar(x + (j - 1.5) * w, vals, w, yerr=errs, label=lab,
               color=colors[fk], capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}\n(activeSP)" for v in VIEWS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, ls=":", c="k", lw=0.7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.5)


def macro(c):
    return c.get("best_macro_auc"), c.get("macro_auc_std")


def bauc(c):
    pf = c.get("best_per_flavor_auc", {})
    seeds = [s["per_flavor_auc"].get("b") for s in c.get("per_seed", [])
             if s.get("per_flavor_auc", {}).get("b") is not None]
    err = float(np.std(seeds)) if len(seeds) > 1 else 0
    return pf.get("b"), err


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
panel(axes[0], macro, "Macro AUC (8-class, one-vs-rest)", "macro AUC")
panel(axes[1], bauc, "b-jet AUC (b vs rest)", "b AUC")
fig.suptitle("Stage-4 tagger matrix — low-stats single-coherent-view (treat spread skeptically)",
             fontsize=11)
fig.tight_layout()
p = os.path.join(OUT, "plots", "stage4_auc_matrix.png")
fig.savefig(p, dpi=120)
print("wrote", p)
