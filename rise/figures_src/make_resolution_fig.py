"""Resolution figure for the RISE deck: d0/z0 vs matched-TP truth, seed vs refit.

Data source: the post-sign-fix 10-file prime production
  nano_pF_PFTrkSmartPix_withGen_file{1..10}.root  (1000 PU events)
Selection: L1TTrack rows with a matched TrackingParticle (tp_pt > 0); the
digiRefit variant tables are row-aligned with L1TTrack (passthrough rows copy
the seed).  Metric: median |param - tp_param| in micrometers.

The computed numbers are cached in resolution_pF.json next to this script so
the figure regenerates without the (large, untracked) nano files.  Rerun with
--recompute when the nanos are present to refresh the cache.

Run:  pixi run python rise/figures_src/make_resolution_fig.py [--recompute]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "resolution_pF.json")
OUT = os.path.join(HERE, os.pardir, "figures", "resolution_d0z0.png")
NANO_GLOB = (
    "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/"
    "nano_pF_PFTrkSmartPix_withGen_file*.root"
)
CONFIGS = ["AIII", "AAII", "AAAI", "AAAA"]  # activeSP 1000/1100/1110/1111


def recompute():
    import awkward as ak
    import numpy as np
    import uproot

    files = [f for f in sorted(glob.glob(NANO_GLOB)) if "verify" not in f]
    if not files:
        raise SystemExit(f"no nano files match {NANO_GLOB}")
    acc = {c: {"d0": [], "z0": []} for c in CONFIGS}
    acc["seed"] = {"d0": [], "z0": []}
    nev = 0
    ntrk = 0
    for fn in files:
        with uproot.open(fn + ":Events") as t:
            nev += t.num_entries
            tp_d0 = ak.flatten(t["L1TTrack_tp_d0"].array()).to_numpy()
            tp_z0 = ak.flatten(t["L1TTrack_tp_z0"].array()).to_numpy()
            tp_pt = ak.flatten(t["L1TTrack_tp_pt"].array()).to_numpy()
            m = tp_pt > 0
            ntrk += int(m.sum())
            d0 = ak.flatten(t["L1TTrack_d0"].array()).to_numpy()
            z0 = ak.flatten(t["L1TTrack_z0"].array()).to_numpy()
            acc["seed"]["d0"].append(np.abs(d0 - tp_d0)[m])
            acc["seed"]["z0"].append(np.abs(z0 - tp_z0)[m])
            for c in CONFIGS:
                vt = f"L1TSmartPixelsTrackDigiRefit{c}"
                vd0 = ak.flatten(t[f"{vt}_d0"].array()).to_numpy()
                vz0 = ak.flatten(t[f"{vt}_z0"].array()).to_numpy()
                acc[c]["d0"].append(np.abs(vd0 - tp_d0)[m])
                acc[c]["z0"].append(np.abs(vz0 - tp_z0)[m])
    out = {
        "source": "nano_pF_PFTrkSmartPix_withGen_file{1..10}.root (post-sign-fix)",
        "selection": "matched TP (tp_pt>0); refit tables row-aligned, passthrough included",
        "n_events": nev,
        "n_matched_tracks": ntrk,
        "median_um": {},
    }
    for key in ["seed"] + CONFIGS:
        out["median_um"][key] = {
            p: float(np.median(np.concatenate(acc[key][p])) * 1e4) for p in ("d0", "z0")
        }
    json.dump(out, open(CACHE, "w"), indent=2)
    print("wrote", CACHE)
    return out


def main():
    if "--recompute" in sys.argv or not os.path.exists(CACHE):
        data = recompute()
    else:
        data = json.load(open(CACHE))
    med = data["median_um"]
    labels = ["OT-only\nseed"] + [f"OT+IT\n{c}" for c in CONFIGS]
    plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.labelsize": 18})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, par, title in [
        (axes[0], "d0", r"transverse impact parameter $d_0$"),
        (axes[1], "z0", r"longitudinal impact parameter $z_0$"),
    ]:
        vals = [med["seed"][par]] + [med[c][par] for c in CONFIGS]
        colors = ["#8a8a8a"] + ["#d95f02", "#c74a02", "#b03802", "#1b6ca8"]
        bars = ax.bar(labels, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=16, fontweight="bold")
        ax.set_ylabel(r"median $|$reco $-$ TP truth$|$  [$\mu$m]")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.22)
    imp_d0 = 100 * (1 - med["AAAA"]["d0"] / med["seed"]["d0"])
    imp_z0 = 100 * (1 - med["AAAA"]["z0"] / med["seed"]["z0"])
    fig.suptitle(
        f"5-par prompt track resolution vs matched-TP truth  —  "
        f"{data['n_matched_tracks']:,} tracks / {data['n_events']} PU events   "
        f"(AAAA: $d_0$ $-${imp_d0:.0f}%, $z_0$ $-${imp_z0:.0f}%)",
        fontsize=17,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT, dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
