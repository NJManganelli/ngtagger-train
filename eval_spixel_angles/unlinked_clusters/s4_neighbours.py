"""Step 4 (what ARE unlinked clusters?): same-module nearest-neighbour topology.

A delta-ray or other soft secondary knocked out by a nearby TP-linked track would sit CLOSE to a
linked cluster on the same module; noise, or an independent soft particle, would not. A looper or
low-energy shower would leave several unlinked clusters near each other. We compare, per class,
the local-plane distance to the nearest OTHER cluster on the same module in the same event, split
by whether that neighbour is linked or unlinked, against a position-shuffled null (each cluster's
localX/localY redrawn from the same module's clusters in OTHER events), which preserves module
occupancy and acceptance but destroys any causal proximity.

Writes results/neighbours.json, results/neighbours.md, figs/f4_neighbours.png.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import _common as c

RES = c.HERE / "results"
OFF = 1000.0   # cm offset per (event, module) group so a single KD-tree never mixes groups


def nn_dist(g, x, y, ref_mask, query_mask, exclude_self):
    pts = np.stack([x, y, g * OFF], 1)
    tree = cKDTree(pts[ref_mask])
    k = 2 if exclude_self else 1
    dd, _ = tree.query(pts[query_mask], k=k, distance_upper_bound=OFF / 2)
    dd = dd[:, -1] if exclude_self else dd
    return np.where(np.isfinite(dd), dd, np.nan)


def run(d, x, y):
    g = d.groupby(["evid", "detId"]).ngroup().to_numpy().astype(np.float64)
    u = d["unlinked"].to_numpy() == 1
    lk = ~u
    out = {}
    # nearest linked neighbour (excluding self if the query is linked)
    dl = np.full(len(d), np.nan)
    dl[lk] = nn_dist(g, x, y, lk, lk, True)
    dl[u] = nn_dist(g, x, y, lk, u, False)
    du = np.full(len(d), np.nan)
    du[u] = nn_dist(g, x, y, u, u, True)
    du[lk] = nn_dist(g, x, y, u, lk, False)
    out["d_linked"], out["d_unlinked"] = dl, du
    return out


def main():
    d = c.load()
    x = d["localX"].to_numpy().astype(np.float64)
    y = d["localY"].to_numpy().astype(np.float64)
    real = run(d, x, y)
    # null: shuffle positions within module across events (keeps per-module acceptance/occupancy)
    rng = np.random.default_rng(7)
    perm = d.groupby("detId").indices
    xs, ys = x.copy(), y.copy()
    for _, ii in perm.items():
        p = rng.permutation(ii)
        xs[ii], ys[ii] = x[p], y[p]
    null = run(d, xs, ys)

    u = d["unlinked"].to_numpy() == 1
    pt = d["tpPt"].to_numpy()
    groups = {"unlinked": u, "linked pT>=2": (~u) & (pt >= 2), "linked pT<0.3": (~u) & (pt < 0.3),
              "linked secondary vr>1cm": (~u) & (d["tpVr"].to_numpy() > 1), "linked all": ~u}
    rows = []
    for gname, m in groups.items():
        for which in ("d_linked", "d_unlinked"):
            for cut in (0.02, 0.05, 0.1):
                rows.append({"class": gname, "nearest": which.replace("d_", ""), "within [cm]": cut,
                             "real": np.nanmean(real[which][m] < cut),
                             "shuffled null": np.nanmean(null[which][m] < cut)})
    t = pd.DataFrame(rows)
    t["excess (real - null)"] = t["real"] - t["shuffled null"]
    (RES / "neighbours.md").write_text(
        "Fraction of clusters whose nearest same-module neighbour of the given kind lies within the cut "
        "(local-plane distance). Null = positions shuffled within the module across events.\n\n"
        + c.md(t.set_index(["class", "nearest", "within [cm]"]), "{:.4f}"))
    (RES / "neighbours.json").write_text(json.dumps(t.to_dict(orient="list"), indent=1, default=float))
    print(c.md(t.set_index(["class", "nearest", "within [cm]"]), "{:.4f}"))

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.logspace(-3, 0.5, 71)
    for ax, which, ttl in ((axs[0], "d_linked", "nearest TP-LINKED cluster"),
                           (axs[1], "d_unlinked", "nearest UNLINKED cluster")):
        for gname, col in (("unlinked", "#d62728"), ("linked pT>=2", "#1f77b4"), ("linked pT<0.3", "#9467bd")):
            m = groups[gname]
            ax.hist(real[which][m], bins, histtype="step", density=True, color=col, label=f"{gname}")
            ax.hist(null[which][m], bins, histtype="step", density=True, color=col, ls=":", label=f"{gname} (shuffled null)")
        ax.set_xscale("log")
        ax.set_xlabel(f"distance to {ttl}, same module & event [cm]")
        ax.set_ylabel("density [1/cm]")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(c.FIGS / "f4_neighbours.png", dpi=110)


if __name__ == "__main__":
    main()
