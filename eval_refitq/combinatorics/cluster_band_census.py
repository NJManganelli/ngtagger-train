#!/usr/bin/env python
"""How many IT clusters must a refit actually test, per layer, per crossing?

THE QUESTION. Architecture A1 (match OT tracks against SmartPixels clusters in a
search window) lives or dies on combinatorics, and we have never measured them.
`windowMult` in the sidecar cannot answer it: it is counted AFTER the static
window cut and AFTER `maxHitsPerWindow` truncation, so it reports the size of a
list we already decided to build, not the size of the problem.

WHAT THIS MEASURES, in three steps of increasing realism:

  1. Cluster occupancy per TBPX layer, per event -- the raw population, and how
     it shrinks under a truth-pT requirement on each cluster's dominant charge
     contributor. (A truth-pT cut is not deployable; it bounds what an ideal
     on-sensor pT-discriminating readout could buy.)

  2. Clusters on the crossed module inside the q68 / q95 band of a NAIVE
     projection: the OT-only seed helix extrapolated to every layer with NO
     Kalman updates. This is the honest "cold start" number -- what a system
     that matches all layers in parallel from the seed would face.

  3. The same count for the RUNNING projection in outsideIn order, where each
     layer is projected from a state already updated by the layers outside it.
     The difference between (2) and (3) is what sequential refitting buys you in
     combinatoric reduction, as opposed to in fit quality.

BANDS ARE EMPIRICAL, NOT MODELLED. The q68/q95 half-widths are quantiles of the
observed |reco - proj| distribution for TRUTH-CORRECT hits (selHitClass == 0),
computed per layer and per coordinate from the input itself. This deliberately
avoids sqrt(S): the covariance collapses after the first update and never grows
(no process-noise term yet), so a covariance-derived band would be optimistic by
a layer-dependent and order-dependent factor -- exactly the effect under study.

SCOPE. Only clusters on the CROSSED MODULE are counted, which is what the
producer's candidate loop considers, so this is the right denominator for "how
many candidates per crossing". Where a band is wider than a module, the count is
a lower bound; that case is reported rather than hidden.

    pixi run python eval_refitq/combinatorics/cluster_band_census.py \
        -i <nano.root> -o eval_refitq/combinatorics/
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

SENTINEL = -900.0
LAYERS = (1, 2, 3, 4)
PT_CUTS = (0.0, 0.5, 1.0, 1.5, 2.0)

CLUSTER_TABLE = "L1TSpixCluster"
# Columns this study REQUIRES. The script is the specification: if they are
# absent it says exactly what to add rather than silently measuring something
# weaker.
NEED_CLUSTER = ["layer", "detId", "localX", "localY"]
NEED_HIT = ["trackIdx", "layer", "detId", "hitAccepted", "selHitClass",
            "recoLocalX", "recoLocalY", "projResX", "projResY",
            "projLocalX", "projLocalY", "projSeedLocalX", "projSeedLocalY"]


def _discover(path):
    with uproot.open(f"{path}:Events") as t:
        keys = set(t.keys())
    cfgs = sorted({m.group(1) for m in
                   (re.match(r"^L1TSmartPixelsRefitHitDigiRefit([A-Za-z0-9]+)_", k) for k in keys) if m})
    if not cfgs:
        raise SystemExit("no L1TSmartPixelsRefitHitDigiRefit* table in the input")
    hit = f"L1TSmartPixelsRefitHitDigiRefit{cfgs[-1]}"

    missing_hit = [c for c in NEED_HIT if f"{hit}_{c}" not in keys]
    missing_cl = ([] if any(k.startswith(CLUSTER_TABLE + "_") for k in keys)
                  else ["<the whole table>"])
    if not missing_cl:
        missing_cl = [c for c in NEED_CLUSTER if f"{CLUSTER_TABLE}_{c}" not in keys]
    if missing_hit or missing_cl:
        raise SystemExit(
            "input cannot support this study.\n"
            f"  missing from {hit}: {missing_hit or 'none'}\n"
            f"  missing from {CLUSTER_TABLE}: {missing_cl or 'none'}\n\n"
            "Required additions:\n"
            "  * an UNTRUNCATED per-cluster table (all IT clusters, not just selected\n"
            "    ones) carrying layer, detId and module-local localX/localY;\n"
            "  * per-crossing projLocalX/Y (running Kalman state) and projSeedLocalX/Y\n"
            "    (unmodified OT seed), stored for EVERY valid crossing rather than only\n"
            "    where a hit was accepted -- crossings with no accepted hit are exactly\n"
            "    the ones where combinatorics matter most.")
    return hit, cfgs[-1]


def load(paths):
    files = [f for p in paths for f in (sorted(_glob.glob(p)) or [p])]
    hit, cfg = _discover(files[0])

    hcols = NEED_HIT
    ccols = NEED_CLUSTER + [c for c in ("truthPt", "selHitClass") if True]
    h = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{hit}_{c}" for c in hcols])
    with uproot.open(f"{files[0]}:Events") as t:
        have_pt = f"{CLUSTER_TABLE}_truthPt" in set(t.keys())
    cnames = NEED_CLUSTER + (["truthPt"] if have_pt else [])
    c = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{CLUSTER_TABLE}_{x}" for x in cnames])
    n_ev = sum(uproot.open(f"{f}:Events").num_entries for f in files)
    print(f"files={len(files)} config={cfg} events={n_ev} truthPt={'yes' if have_pt else 'no'}")
    return hit, cfg, h, c, cnames, n_ev, have_pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputs", nargs="+", required=True)
    ap.add_argument("-o", "--outdir", default="eval_refitq/combinatorics")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    hit, cfg, H, C, cnames, n_ev, have_pt = load(args.inputs)
    out = {"config": cfg, "n_events": n_ev, "layers": {}}

    # ---------- (1) occupancy ------------------------------------------------
    cl_layer = C[f"{CLUSTER_TABLE}_layer"]
    cl_pt = C[f"{CLUSTER_TABLE}_truthPt"] if have_pt else None
    print("\n(1) clusters per event, per layer")
    hdr = "  L      all" + "".join(f"   >{p}GeV" for p in PT_CUTS[1:] if have_pt)
    print(hdr)
    occ = {}
    for L in LAYERS:
        per_ev = ak.to_numpy(ak.sum(cl_layer == L, axis=1)).astype(float)
        row = {"mean_all": float(per_ev.mean()), "std_all": float(per_ev.std())}
        line = f"  {L}  {per_ev.mean():>7.1f}"
        if have_pt:
            for p in PT_CUTS[1:]:
                sel = ak.to_numpy(ak.sum((cl_layer == L) & (cl_pt > p), axis=1)).astype(float)
                row[f"mean_pt{p}"] = float(sel.mean())
                line += f"  {sel.mean():>8.1f}"
        print(line)
        occ[L] = row
    out["occupancy"] = occ

    # ---------- flatten hits, keep event index for cluster lookup -----------
    counts = ak.to_numpy(ak.num(H[f"{hit}_layer"]))
    ev_of_hit = np.repeat(np.arange(len(counts)), counts)
    h = {c: ak.to_numpy(ak.flatten(H[f"{hit}_{c}"])) for c in NEED_HIT}

    # ---------- (2) empirical bands from truth-correct hits ------------------
    print("\n(2) empirical band half-widths from truth-correct hits [um]")
    print("  L        q68_x    q95_x    q68_y    q95_y      n")
    bands = {}
    for L in LAYERS:
        m = (h["hitAccepted"] > 0) & (h["selHitClass"] == 0) & (h["layer"] == L) \
            & (h["projResX"] > SENTINEL) & (h["projResY"] > SENTINEL)
        if m.sum() < 50:
            continue
        rx, ry = np.abs(h["projResX"][m]), np.abs(h["projResY"][m])
        b = {"q68_x": float(np.quantile(rx, 0.68)), "q95_x": float(np.quantile(rx, 0.95)),
             "q68_y": float(np.quantile(ry, 0.68)), "q95_y": float(np.quantile(ry, 0.95)),
             "n": int(m.sum())}
        bands[L] = b
        print(f"  {L}  {b['q68_x']*1e4:>11.1f} {b['q95_x']*1e4:>8.1f} "
              f"{b['q68_y']*1e4:>8.1f} {b['q95_y']*1e4:>8.1f} {b['n']:>7}")
    out["bands_cm"] = bands

    # ---------- (3) in-band cluster counts, naive vs running -----------------
    # Per crossing: how many clusters on the SAME module fall inside the band
    # around each projection.
    cl_det = C[f"{CLUSTER_TABLE}_detId"]
    cl_x = C[f"{CLUSTER_TABLE}_localX"]
    cl_y = C[f"{CLUSTER_TABLE}_localY"]
    # index clusters by (event, detId) once
    per_ev_det = {}
    for iev in range(len(cl_det)):
        d = ak.to_numpy(cl_det[iev]); x = ak.to_numpy(cl_x[iev]); y = ak.to_numpy(cl_y[iev])
        order = np.argsort(d, kind="stable")
        d, x, y = d[order], x[order], y[order]
        edges = np.searchsorted(d, np.unique(d), side="left")
        per_ev_det[iev] = (d, x, y, np.unique(d), edges)

    print("\n(3) clusters on the crossed module inside the band, per crossing")
    print("     (naive = OT seed projected with NO updates; running = outsideIn state)")
    print("  L    band     naive_mean  running_mean   naive_p95  running_p95")
    res = {}
    for L in LAYERS:
        if L not in bands:
            continue
        b = bands[L]
        sel = (h["layer"] == L) & (h["projSeedLocalX"] > SENTINEL) & (h["projLocalX"] > SENTINEL)
        idx = np.nonzero(sel)[0]
        rec = {}
        for qname in ("q68", "q95"):
            hx, hy = b[f"{qname}_x"], b[f"{qname}_y"]
            n_naive, n_run = [], []
            for i in idx:
                iev = ev_of_hit[i]
                d, x, y, ud, ed = per_ev_det[iev]
                k = np.searchsorted(ud, h["detId"][i])
                if k >= len(ud) or ud[k] != h["detId"][i]:
                    n_naive.append(0); n_run.append(0); continue
                lo = ed[k]
                hi = ed[k + 1] if k + 1 < len(ed) else len(d)
                mx, my = x[lo:hi], y[lo:hi]
                n_naive.append(int(np.sum((np.abs(mx - h["projSeedLocalX"][i]) < hx) &
                                          (np.abs(my - h["projSeedLocalY"][i]) < hy))))
                n_run.append(int(np.sum((np.abs(mx - h["projLocalX"][i]) < hx) &
                                        (np.abs(my - h["projLocalY"][i]) < hy))))
            n_naive = np.array(n_naive, dtype=float); n_run = np.array(n_run, dtype=float)
            rec[qname] = {"naive_mean": float(n_naive.mean()), "running_mean": float(n_run.mean()),
                          "naive_p95": float(np.quantile(n_naive, 0.95)),
                          "running_p95": float(np.quantile(n_run, 0.95)),
                          "n_crossings": int(len(n_naive))}
            print(f"  {L}   {qname}    {n_naive.mean():>10.2f}  {n_run.mean():>12.2f}"
                  f"  {np.quantile(n_naive,0.95):>10.0f}  {np.quantile(n_run,0.95):>11.0f}")
            rec[qname]["_naive_hist"] = np.bincount(n_naive.astype(int), minlength=12)[:12].tolist()
            rec[qname]["_run_hist"] = np.bincount(n_run.astype(int), minlength=12)[:12].tolist()
        res[L] = rec
    out["in_band"] = res

    # ---------- plots --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    xs = np.arange(len(LAYERS))
    ax.bar(xs - 0.2, [occ[L]["mean_all"] for L in LAYERS], 0.4, label="all clusters")
    if have_pt:
        ax.bar(xs + 0.2, [occ[L].get("mean_pt2.0", 0) for L in LAYERS], 0.4,
               label="dominant TP $p_T>2$ GeV")
    ax.set_xticks(xs); ax.set_xticklabels([f"L{L}" for L in LAYERS])
    ax.set_ylabel("clusters / event"); ax.set_title("(1) IT cluster occupancy")
    ax.legend(); ax.grid(alpha=.3)

    ax = axes[1]
    for qname, ls in (("q68", "-"), ("q95", "--")):
        ax.plot([L for L in LAYERS if L in res], [res[L][qname]["naive_mean"] for L in LAYERS if L in res],
                ls, marker="o", label=f"naive seed, {qname}")
        ax.plot([L for L in LAYERS if L in res], [res[L][qname]["running_mean"] for L in LAYERS if L in res],
                ls, marker="s", label=f"outsideIn running, {qname}")
    ax.set_xlabel("TBPX layer"); ax.set_ylabel("clusters in band / crossing")
    ax.set_title("(2) candidates to test per crossing"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[2]
    if res:
        L0 = sorted(res)[0]
        hn = np.array(res[L0]["q95"]["_naive_hist"], dtype=float)
        hr = np.array(res[L0]["q95"]["_run_hist"], dtype=float)
        w = np.arange(len(hn))
        ax.bar(w - 0.2, hn / max(hn.sum(), 1), 0.4, label="naive seed")
        ax.bar(w + 0.2, hr / max(hr.sum(), 1), 0.4, label="outsideIn running")
        ax.set_xlabel(f"clusters in q95 band (L{L0})"); ax.set_ylabel("fraction of crossings")
        ax.set_title("(3) per-crossing multiplicity"); ax.legend(); ax.grid(alpha=.3)
    fig.suptitle(f"SmartPixels cluster combinatorics — {cfg}, {n_ev} events", y=1.02)
    fig.tight_layout()
    png = os.path.join(args.outdir, "cluster_band_census.png")
    fig.savefig(png, dpi=140, bbox_inches="tight")
    print(f"\nwrote {png}")

    js = os.path.join(args.outdir, "cluster_band_census.json")
    with open(js, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {js}")


if __name__ == "__main__":
    main()
