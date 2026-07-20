"""FastHisto peak-finder kernel study: two-close-vertices toy scan.

Weakness under study (user-identified): the flat boxcar window can prefer the
midpoint BETWEEN two similarly-hard vertices over either true peak. This scan
throws two vertices at a controlled z-separation and relative hardness, and
measures the wrong-vertex / midpoint-pick rate for the flat kernel vs the
tapered kernels (triangular, gaussian, epanechnikov). It also measures the
single-vertex resolution cost of each kernel.

Outputs (eval_refitq/vtxdxy/):
  kernel_scan.json   full grid + single-vertex resolution
  kernel_scan.png    midpoint-pick heatmaps (flat vs best taper) + res bars

NNVtx comparison hook: the real study replaces the flat/kernel arg-max with
nnvtx.compare_vertex_scores against a trained convolution (which effectively
learns a better kernel); left as a documented hook -- needs the future PU
production and a trained e2e_nnvtx model.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ngtagger.train.nnvtx import (HISTO_MAX, HISTO_MIN, HISTO_WIDTH,
                                  fast_histo_z0)

OUT = os.path.dirname(os.path.abspath(__file__))
KERNELS = {
    "flat": {},
    "triangular": {"kernel": "triangular"},
    "epanechnikov": {"kernel": "epanechnikov"},
    "gaussian": {"kernel": "gaussian", "sigma_bins": 1.0},
}
WINDOW_BINS = 3
Z_RES = 0.10  # per-track z0 resolution [cm] (PV-track scale)


def _throw_two_vertex(rng, z1, z2, hardness_ratio, n_base=20, pu=15):
    """One event: two vertices at z1, z2 with track multiplicities in ratio
    hardness_ratio (n2 = round(n_base * ratio)), plus flat PU. pt uniform so
    'hardness' = track count. Returns (z0, pt, mask) row vectors."""
    n2 = max(1, int(round(n_base * hardness_ratio)))
    z = np.concatenate([
        rng.normal(z1, Z_RES, n_base),
        rng.normal(z2, Z_RES, n2),
        rng.uniform(HISTO_MIN + 1, HISTO_MAX - 1, pu),
    ])
    pt = np.concatenate([
        rng.uniform(3, 20, n_base), rng.uniform(3, 20, n2), rng.uniform(2, 5, pu)])
    return z, pt


def two_vertex_scan(seps_bins, ratios, n_events=400, seed=0):
    """Grid over z-separation (in histogram bins) x relative hardness.
    A pick is 'correct' if it lands within 1.5 bins of the HARDER vertex;
    'midpoint' if it lands strictly between the two vertices (> 0.5 bin from
    both). z1 is randomised per event to avoid bin-phase artefacts."""
    rng = np.random.default_rng(seed)
    results = {k: {"midpoint_rate": [], "wrong_rate": []} for k in KERNELS}
    grid = {"sep_bins": list(seps_bins), "ratios": list(ratios)}
    for kname, kw in KERNELS.items():
        for sep in seps_bins:
            mid_row, wrong_row = [], []
            for ratio in ratios:
                rows_z, rows_pt = [], []
                z1s, z2s = [], []
                for _ in range(n_events):
                    z1 = rng.uniform(HISTO_MIN + 3, HISTO_MAX - 3)
                    z2 = z1 + sep * HISTO_WIDTH
                    zz, pp = _throw_two_vertex(rng, z1, z2, ratio)
                    rows_z.append(zz)
                    rows_pt.append(pp)
                    z1s.append(z1)
                    z2s.append(z2)
                m = max(len(r) for r in rows_z)
                Z = np.zeros((n_events, m))
                P = np.zeros((n_events, m))
                M = np.zeros((n_events, m))
                for i, (rz, rp) in enumerate(zip(rows_z, rows_pt)):
                    Z[i, :len(rz)] = rz
                    P[i, :len(rp)] = rp
                    M[i, :len(rz)] = 1.0
                found = fast_histo_z0(Z, P, M, window_bins=WINDOW_BINS, **kw)
                z1s, z2s = np.array(z1s), np.array(z2s)
                # vertex 1 has n_base tracks, vertex 2 has round(n_base*ratio);
                # the harder vertex is z1 for ratio<=1, z2 otherwise
                hard = z2s if ratio > 1.0 else z1s
                d_hard = np.abs(found - hard)
                lo = np.minimum(z1s, z2s)
                hi = np.maximum(z1s, z2s)
                between = (found > lo + 0.5 * HISTO_WIDTH) & (found < hi - 0.5 * HISTO_WIDTH)
                correct = d_hard < 1.5 * HISTO_WIDTH
                mid_row.append(float(between.mean()))
                wrong_row.append(float((~correct).mean()))
            results[kname]["midpoint_rate"].append(mid_row)
            results[kname]["wrong_rate"].append(wrong_row)
    return grid, results


def single_vertex_resolution(n_events=3000, seed=1):
    """Resolution cost of each kernel on an isolated single vertex + PU."""
    rng = np.random.default_rng(seed)
    ztrue = rng.uniform(HISTO_MIN + 3, HISTO_MAX - 3, n_events)
    n_pv, n_pu = 20, 20
    m = n_pv + n_pu
    Z = np.zeros((n_events, m))
    P = np.zeros((n_events, m))
    for i in range(n_events):
        Z[i, :n_pv] = rng.normal(ztrue[i], Z_RES, n_pv)
        Z[i, n_pv:] = rng.uniform(HISTO_MIN + 1, HISTO_MAX - 1, n_pu)
        P[i, :n_pv] = rng.uniform(3, 20, n_pv)
        P[i, n_pv:] = rng.uniform(2, 5, n_pu)
    M = np.ones_like(Z)
    out = {}
    for kname, kw in KERNELS.items():
        found = fast_histo_z0(Z, P, M, window_bins=WINDOW_BINS, **kw)
        res = found - ztrue
        out[kname] = {"res_mean": float(res.mean()),
                      "res_q68": float(np.quantile(np.abs(res), 0.68)),
                      "res_std": float(res.std())}
    return out


def _plot(grid, results, single, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seps, ratios = grid["sep_bins"], grid["ratios"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, kname in zip(axes[:2], ("flat", "triangular")):
        im = ax.imshow(np.array(results[kname]["midpoint_rate"]), origin="lower",
                       aspect="auto", vmin=0, vmax=max(0.05, np.max(results["flat"]["midpoint_rate"])),
                       extent=[min(ratios), max(ratios), min(seps), max(seps)], cmap="magma")
        ax.set_title(f"midpoint-pick rate: {kname}")
        ax.set_xlabel("hardness ratio n2/n1")
        ax.set_ylabel("separation [bins]")
        fig.colorbar(im, ax=ax)
    ax = axes[2]
    names = list(single)
    q68 = [single[k]["res_q68"] for k in names]
    ax.bar(names, q68, color=["#444", "#2a7", "#27a", "#a52"])
    ax.set_ylabel("single-vertex |res| q68 [cm]")
    ax.set_title("resolution cost")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    seps = [1, 2, 3, 4, 5]
    ratios = [0.6, 0.8, 1.0]
    grid, results = two_vertex_scan(seps, ratios)
    single = single_vertex_resolution()

    # headline: mean midpoint-pick rate over the grid, flat vs best taper
    flat_mid = float(np.mean(results["flat"]["midpoint_rate"]))
    tapers = {k: float(np.mean(v["midpoint_rate"])) for k, v in results.items() if k != "flat"}
    best = min(tapers, key=tapers.get)
    payload = {
        "config": {"window_bins": WINDOW_BINS, "z_res_cm": Z_RES,
                   "histo": [HISTO_MIN, HISTO_MAX, HISTO_WIDTH]},
        "grid": grid, "results": results, "single_vertex": single,
        "headline": {
            "flat_mean_midpoint_rate": flat_mid,
            "taper_mean_midpoint_rate": tapers,
            "best_taper": best,
            "midpoint_reduction_vs_flat": flat_mid - tapers[best],
            "single_vertex_res_penalty_cm": single[best]["res_q68"] - single["flat"]["res_q68"],
        },
    }
    with open(os.path.join(OUT, "kernel_scan.json"), "w") as f:
        json.dump(payload, f, indent=2)
    try:
        _plot(grid, results, single, os.path.join(OUT, "kernel_scan.png"))
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    h = payload["headline"]
    print(f"flat midpoint-pick rate    {h['flat_mean_midpoint_rate']:.3f}")
    print(f"best taper ({best})       {tapers[best]:.3f}")
    print(f"reduction                  {h['midpoint_reduction_vs_flat']:+.3f}")
    print(f"single-vtx res penalty     {h['single_vertex_res_penalty_cm']:+.4f} cm")


if __name__ == "__main__":
    main()
