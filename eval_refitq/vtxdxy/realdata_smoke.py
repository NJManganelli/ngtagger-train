"""Real-data smoke for the fastHisto (dx, dy) estimator on an existing nano
file carrying the extended-track table (L1TExtTrack, 5-par d0).

This is a sanity first-look, NOT a measurement: with RelVal-scale statistics
the per-event PV-window (dx, dy) should cluster near the beam-spot origin with
a spread set by the d0 resolution / sqrt(N_window). Prints the dx/dy
distribution summary and the significance spread, and saves JSON + a plot.

Usage:
  pixi run python eval_refitq/vtxdxy/realdata_smoke.py [nano_file ...]
Default file: the PU100 TrkSmartPix withGen smoke nano (read-only).
"""
from __future__ import annotations

import json
import os
import sys

import awkward as ak
import numpy as np
import uproot

from ngtagger.train.nnvtx import fast_histo_vtx

OUT = os.path.dirname(os.path.abspath(__file__))
DEFAULT = ["/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/"
           "nano_pu100_TrkSmartPix_withGen.root"]
TRACK = "L1TExtTrack"


def load(files):
    br = [f"{TRACK}_{b}" for b in ("pt", "phi", "z0", "d0")]
    br += ["GenVtx_x", "GenVtx_y", "GenVtx_z"]
    ev = uproot.concatenate([f"{f}:Events" for f in files], filter_name=br, how="zip")
    return ev


def _pad(field, tracks, max_trk, fill=0.0):
    return ak.to_numpy(ak.fill_none(
        ak.pad_none(tracks[field], max_trk, axis=1, clip=True), fill)).astype(np.float64)


def main():
    files = sys.argv[1:] or DEFAULT
    ev = load(files)
    tracks = ev[TRACK]
    counts = ak.num(tracks["z0"], axis=1)
    max_trk = int(ak.max(counts))
    mask = ak.to_numpy(ak.fill_none(
        ak.pad_none(ak.ones_like(tracks["z0"]), max_trk, axis=1, clip=True), 0.0)).astype(np.float64)

    z0p, ptp = _pad("z0", tracks, max_trk), _pad("pt", tracks, max_trk)
    d0p, phip = _pad("d0", tracks, max_trk), _pad("phi", tracks, max_trk)
    # ungated (raw accumulator picture) + prompt-gated (|d0|<=0.15 cm): the
    # extended-track collection carries displaced/loose tracks that violate the
    # prompt approximation, so the gate is what makes a beam-spot-scale look
    res = fast_histo_vtx(z0p, ptp, mask, d0p, phip, estimator="lsq")
    res_gated = fast_histo_vtx(z0p, ptp, mask, d0p, phip, estimator="lsq", d0_gate=0.15)
    res_iso = fast_histo_vtx(z0p, ptp, mask, d0p, phip, estimator="isotropic", d0_gate=0.15)

    genx = ak.to_numpy(ev["GenVtx"].x) if "GenVtx" in ev.fields else ak.to_numpy(ev["GenVtx_x"])
    geny = ak.to_numpy(ev["GenVtx"].y) if "GenVtx" in ev.fields else ak.to_numpy(ev["GenVtx_y"])

    ok = np.isfinite(res["dx"]) & np.isfinite(res["dy"])

    def summ(a):
        a = np.asarray(a)[np.isfinite(a)]
        return {"n": int(a.size), "median": float(np.median(a)),
                "mean": float(np.mean(a)), "std": float(np.std(a)),
                "q16": float(np.quantile(a, 0.16)), "q84": float(np.quantile(a, 0.84))}

    payload = {
        "files": files, "n_events": int(len(res["dx"])), "solved_fraction": float(ok.mean()),
        "n_window": summ(res["n_window"]), "phi_condition": summ(res["phi_condition"]),
        "d0_scatter_cm": summ(res["d0_scatter"]),
        "lsq": {"dx": summ(res["dx"]), "dy": summ(res["dy"]),
                "dxsig": summ(res["dxsig"]), "dysig": summ(res["dysig"])},
        "lsq_gated_0p15": {"dx": summ(res_gated["dx"]), "dy": summ(res_gated["dy"]),
                           "n_window": summ(res_gated["n_window"]),
                           "d0_scatter_cm": summ(res_gated["d0_scatter"])},
        "isotropic_gated": {"dx": summ(res_iso["dx"]), "dy": summ(res_iso["dy"])},
        "gen_vtx": {"x_median": float(np.median(genx)), "y_median": float(np.median(geny)),
                    "x_std": float(np.std(genx)), "y_std": float(np.std(geny))},
        "dx_minus_genx": summ(res["dx"][ok] - genx[ok]),
        "dy_minus_geny": summ(res["dy"][ok] - geny[ok]),
    }
    with open(os.path.join(OUT, "realdata_smoke.json"), "w") as f:
        json.dump(payload, f, indent=2)

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
        fig.savefig(os.path.join(OUT, "realdata_smoke.png"), dpi=110)
        plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)

    print(f"events {payload['n_events']}, solved {payload['solved_fraction']:.2f}, "
          f"window mult median {payload['n_window']['median']:.0f}")
    print(f"dx: median {payload['lsq']['dx']['median']*1e4:+.1f} um, "
          f"std {payload['lsq']['dx']['std']*1e4:.1f} um")
    print(f"dy: median {payload['lsq']['dy']['median']*1e4:+.1f} um, "
          f"std {payload['lsq']['dy']['std']*1e4:.1f} um")
    print(f"d0 scatter median {payload['d0_scatter_cm']['median']*1e4:.0f} um; "
          f"phi_condition median {payload['phi_condition']['median']:.2f}")
    g = payload["lsq_gated_0p15"]
    print(f"[gated |d0|<0.15] window mult {g['n_window']['median']:.0f}, "
          f"dx std {g['dx']['std']*1e4:.1f} um, dy std {g['dy']['std']*1e4:.1f} um, "
          f"scatter {g['d0_scatter_cm']['median']*1e4:.0f} um")


if __name__ == "__main__":
    main()
