"""Vertex (dx, dy) / fastHisto-kernel study drivers, importable from the
package so the CLI and the eval_refitq/vtxdxy/*.py scripts share one code path.

Two studies:

  run_vertex_dxy_smoke  real-data first-look on an extended-track nano
                        (L1TExtTrack): fast_histo_vtx per event ungated +
                        d0-gated (LSQ) + isotropic-gated, writes
                        realdata_smoke.json (+ .png).
  run_kernel_scan       two-close-vertices toy scan (flat vs tapered kernels)
                        + single-vertex resolution cost, writes
                        kernel_scan.json (+ .png).

Behaviour is identical to the original scripts (the committed JSONs are the
reference); the scripts are now thin wrappers over these functions.
"""
from __future__ import annotations

import json
import os

import awkward as ak
import numpy as np

from ngtagger.train.nnvtx import (HISTO_MAX, HISTO_MIN, HISTO_WIDTH,
                                  fast_histo_vtx, fast_histo_z0)

# --------------------------------------------------------------------------
# real-data (dx, dy) smoke
# --------------------------------------------------------------------------
SMOKE_TRACK_BRANCHES = ("pt", "phi", "z0", "d0")


def _load_ext_track_nano(files: list[str], track_table: str):
    import uproot

    br = [f"{track_table}_{b}" for b in SMOKE_TRACK_BRANCHES]
    br += ["GenVtx_x", "GenVtx_y", "GenVtx_z"]
    return uproot.concatenate([f"{f}:Events" for f in files], filter_name=br, how="zip")


def _pad(tracks, field, max_trk, fill=0.0):
    return ak.to_numpy(ak.fill_none(
        ak.pad_none(tracks[field], max_trk, axis=1, clip=True), fill)).astype(np.float64)


def _summ(a):
    a = np.asarray(a)[np.isfinite(a)]
    return {"n": int(a.size), "median": float(np.median(a)),
            "mean": float(np.mean(a)), "std": float(np.std(a)),
            "q16": float(np.quantile(a, 0.16)), "q84": float(np.quantile(a, 0.84))}


def run_vertex_dxy_smoke(files, out_dir, track_table: str = "L1TExtTrack",
                         d0_gate: float = 0.15, make_plot: bool = True):
    """Real-data fastHisto (dx, dy) smoke on an extended-track nano.

    Ungated (raw accumulator picture) + prompt-gated (|d0| <= d0_gate cm) LSQ +
    isotropic-gated: the extended-track collection carries displaced/loose
    tracks that violate the prompt approximation, so the gate is what yields a
    beam-spot-scale look. Writes {out_dir}/realdata_smoke.json (+ .png).
    Returns the JSON payload dict."""
    os.makedirs(out_dir, exist_ok=True)
    ev = _load_ext_track_nano(files, track_table)
    tracks = ev[track_table]
    counts = ak.num(tracks["z0"], axis=1)
    max_trk = int(ak.max(counts))
    mask = ak.to_numpy(ak.fill_none(
        ak.pad_none(ak.ones_like(tracks["z0"]), max_trk, axis=1, clip=True), 0.0)).astype(np.float64)

    z0p, ptp = _pad(tracks, "z0", max_trk), _pad(tracks, "pt", max_trk)
    d0p, phip = _pad(tracks, "d0", max_trk), _pad(tracks, "phi", max_trk)
    res = fast_histo_vtx(z0p, ptp, mask, d0p, phip, estimator="lsq")
    res_gated = fast_histo_vtx(z0p, ptp, mask, d0p, phip, estimator="lsq", d0_gate=d0_gate)
    res_iso = fast_histo_vtx(z0p, ptp, mask, d0p, phip, estimator="isotropic", d0_gate=d0_gate)

    genx = ak.to_numpy(ev["GenVtx"].x) if "GenVtx" in ev.fields else ak.to_numpy(ev["GenVtx_x"])
    geny = ak.to_numpy(ev["GenVtx"].y) if "GenVtx" in ev.fields else ak.to_numpy(ev["GenVtx_y"])

    ok = np.isfinite(res["dx"]) & np.isfinite(res["dy"])

    payload = {
        "files": list(files), "n_events": int(len(res["dx"])),
        "solved_fraction": float(ok.mean()),
        "n_window": _summ(res["n_window"]), "phi_condition": _summ(res["phi_condition"]),
        "d0_scatter_cm": _summ(res["d0_scatter"]),
        "lsq": {"dx": _summ(res["dx"]), "dy": _summ(res["dy"]),
                "dxsig": _summ(res["dxsig"]), "dysig": _summ(res["dysig"])},
        "lsq_gated_0p15": {"dx": _summ(res_gated["dx"]), "dy": _summ(res_gated["dy"]),
                           "n_window": _summ(res_gated["n_window"]),
                           "d0_scatter_cm": _summ(res_gated["d0_scatter"])},
        "isotropic_gated": {"dx": _summ(res_iso["dx"]), "dy": _summ(res_iso["dy"])},
        "gen_vtx": {"x_median": float(np.median(genx)), "y_median": float(np.median(geny)),
                    "x_std": float(np.std(genx)), "y_std": float(np.std(geny))},
        "dx_minus_genx": _summ(res["dx"][ok] - genx[ok]),
        "dy_minus_geny": _summ(res["dy"][ok] - geny[ok]),
    }
    with open(os.path.join(out_dir, "realdata_smoke.json"), "w") as f:
        json.dump(payload, f, indent=2)

    if make_plot:
        _plot_smoke(res, ok, os.path.join(out_dir, "realdata_smoke.png"))

    print(f"events {payload['n_events']}, solved {payload['solved_fraction']:.2f}, "
          f"window mult median {payload['n_window']['median']:.0f}")
    print(f"dx: median {payload['lsq']['dx']['median']*1e4:+.1f} um, "
          f"std {payload['lsq']['dx']['std']*1e4:.1f} um")
    print(f"dy: median {payload['lsq']['dy']['median']*1e4:+.1f} um, "
          f"std {payload['lsq']['dy']['std']*1e4:.1f} um")
    print(f"d0 scatter median {payload['d0_scatter_cm']['median']*1e4:.0f} um; "
          f"phi_condition median {payload['phi_condition']['median']:.2f}")
    g = payload["lsq_gated_0p15"]
    print(f"[gated |d0|<{d0_gate}] window mult {g['n_window']['median']:.0f}, "
          f"dx std {g['dx']['std']*1e4:.1f} um, dy std {g['dy']['std']*1e4:.1f} um, "
          f"scatter {g['d0_scatter_cm']['median']*1e4:.0f} um")
    return payload


def _plot_smoke(res, ok, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].hist(res["dx"][ok] * 1e4, bins=40, alpha=0.7, label="dx")
        ax[0].hist(res["dy"][ok] * 1e4, bins=40, alpha=0.7, label="dy")
        ax[0].set_xlabel("vertex transverse position [um]")
        ax[0].legend()
        ax[0].set_title("fastHisto (dx, dy) [LSQ]")
        ax[1].hist(res["dxsig"][np.isfinite(res["dxsig"])], bins=40, alpha=0.7, label="dx/sigma")
        ax[1].hist(res["dysig"][np.isfinite(res["dysig"])], bins=40, alpha=0.7, label="dy/sigma")
        ax[1].set_xlabel("significance")
        ax[1].legend()
        ax[1].set_title("transverse significance")
        ax[2].hist(res["n_window"], bins=30)
        ax[2].set_xlabel("PV-window track multiplicity")
        ax[2].set_title("window occupancy")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)


# --------------------------------------------------------------------------
# two-close-vertices kernel scan
# --------------------------------------------------------------------------
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
    'hardness' = track count. Returns (z0, pt) row vectors."""
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


def _plot_scan(grid, results, single, path):
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


def run_kernel_scan(out_dir, seps=(1, 2, 3, 4, 5), ratios=(0.6, 0.8, 1.0),
                    n_events=400, seed=0, make_plot: bool = True):
    """Two-close-vertices kernel scan + single-vertex resolution cost.
    Writes {out_dir}/kernel_scan.json (+ .png). Returns the JSON payload dict."""
    os.makedirs(out_dir, exist_ok=True)
    seps, ratios = list(seps), list(ratios)
    grid, results = two_vertex_scan(seps, ratios, n_events=n_events, seed=seed)
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
    with open(os.path.join(out_dir, "kernel_scan.json"), "w") as f:
        json.dump(payload, f, indent=2)
    if make_plot:
        try:
            _plot_scan(grid, results, single, os.path.join(out_dir, "kernel_scan.png"))
        except Exception as e:  # plotting is optional
            print("plot skipped:", e)
    h = payload["headline"]
    print(f"flat midpoint-pick rate    {h['flat_mean_midpoint_rate']:.3f}")
    print(f"best taper ({best})       {tapers[best]:.3f}")
    print(f"reduction                  {h['midpoint_reduction_vs_flat']:+.3f}")
    print(f"single-vtx res penalty     {h['single_vertex_res_penalty_cm']:+.4f} cm")
    return payload
