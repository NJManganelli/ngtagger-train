#!/usr/bin/env python
"""OMNIBUS combinatorics study for the SmartPixels refit, on the Clusters tier.

*** THIS SCRIPT IS DELIBERATELY AN OMNIBUS. *** It is expected to grow many more
figures over time. Each study is a `study_*` function registered in STUDIES,
takes the same prepared arrays, and returns a dict of numbers for the JSON
sidecar. Add a function, add it to STUDIES, done -- do not fork this script.

THE QUESTION. A refit crossing must test the clusters on the module it crosses.
At PU200 that module carries 20-40 clusters (measured 40.6/32.7/31.6/19.7 on
L1-L4), the current static window admits ~2 and truncates at 8. Nothing so far
measures what the window discards, whether a covariance-derived cone would do
better, or what other handles (angle, charge) could cut the pool down.

WHAT IT NEEDS. The Clusters tier (L1PFTrkNanoSmartPixClusters[withGen]):
untruncated per-cluster table joined to the per-crossing refit records on
(event, detId). Both are produced by the same job, so the clusters are exactly
the ones the refit was offered.

THE CONE. Built from the SEED covariance: projSeedSigX/Y = sqrt(diag(H C H^T))
projected to the module, with NO Kalman updates -- the single-shot cold start.
q68 -> 1.00 sigma, q95 -> 1.96 sigma per coordinate. This is the track's own
uncertainty and excludes the measurement term, which is the right choice here:
we are asking how big the search region must be, not how well a hit fits.

    pixi run python eval_refitq/combinatorics/spix_combinatorics_omnibus.py \
        -i <clusters-tier nano.root> -o eval_refitq/combinatorics/
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
CONES = {"q68": 1.0, "q95": 1.959964}
PT_EDGES = np.array([2, 3, 4, 6, 10, 20, 1e9])
ETA_EDGES = np.array([-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4])
CLUSTER_TABLE = "L1TSmartPixelsCluster"
REF = "L1TTrack"


# --------------------------------------------------------------------------
# loading + the (event, detId) join
# --------------------------------------------------------------------------
def _discover(path):
    with uproot.open(f"{path}:Events") as t:
        keys = set(t.keys())
    cfgs = sorted({m.group(1) for m in
                   (re.match(r"^L1TSmartPixelsRefitHitDigiRefit([A-Za-z0-9]+)_", k) for k in keys) if m})
    if not cfgs:
        raise SystemExit("no L1TSmartPixelsRefitHitDigiRefit* table in the input")
    hit = f"L1TSmartPixelsRefitHitDigiRefit{cfgs[-1]}"
    need_hit = ["trackIdx", "layer", "detId", "hitAccepted", "selHitClass",
                "projSeedLocalX", "projSeedLocalY", "projSeedSigX", "projSeedSigY",
                "projSeedCotAlpha", "projSeedCotBeta", "projLocalX", "projLocalY",
                "recoLocalX", "recoLocalY"]
    miss_h = [c for c in need_hit if f"{hit}_{c}" not in keys]
    miss_c = ([CLUSTER_TABLE] if not any(k.startswith(CLUSTER_TABLE + "_") for k in keys)
              else [c for c in ("layer", "detId", "localX", "localY", "charge",
                                "truthPt", "truthCotAlpha", "truthCotBeta")
                    if f"{CLUSTER_TABLE}_{c}" not in keys])
    if miss_h or miss_c:
        raise SystemExit(
            "input cannot support this study.\n"
            f"  missing from {hit}: {miss_h or 'none'}\n"
            f"  missing from {CLUSTER_TABLE}: {miss_c or 'none'}\n\n"
            "Produce a Clusters-tier file:\n"
            "  test/makeSpixConfig.py --pu 200 --tier clusters-truth "
            "--variant digiRefit:1111 --needs-truth -o <out>.py")
    return hit, cfgs[-1], need_hit


def load(paths):
    files = [f for p in paths for f in (sorted(_glob.glob(p)) or [p])]
    hit, cfg, need_hit = _discover(files[0])
    ccols = ["layer", "detId", "localX", "localY", "charge", "truthPt",
             "truthCotAlpha", "truthCotBeta", "sizeX", "sizeY"]
    rcols = ["pt", "eta"]

    H = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{hit}_{c}" for c in need_hit])
    C = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{CLUSTER_TABLE}_{c}" for c in ccols])
    R = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{REF}_{c}" for c in rcols])
    n_ev = sum(uproot.open(f"{f}:Events").num_entries for f in files)

    # flatten crossings, tagging event index and the owning track's pt/eta
    ncross = ak.to_numpy(ak.num(H[f"{hit}_layer"]))
    ev_x = np.repeat(np.arange(len(ncross)), ncross)
    X = {c: ak.to_numpy(ak.flatten(H[f"{hit}_{c}"])) for c in need_hit}
    X["event"] = ev_x
    ntrk = ak.to_numpy(ak.num(R[f"{REF}_pt"]))
    off = np.concatenate([[0], np.cumsum(ntrk)])
    for c in rcols:
        flat = ak.to_numpy(ak.flatten(R[f"{REF}_{c}"]))
        X[f"trk_{c}"] = flat[off[ev_x] + X["trackIdx"].astype(np.int64)]

    ncl = ak.to_numpy(ak.num(C[f"{CLUSTER_TABLE}_layer"]))
    ev_c = np.repeat(np.arange(len(ncl)), ncl)
    K = {c: ak.to_numpy(ak.flatten(C[f"{CLUSTER_TABLE}_{c}"])) for c in ccols}
    K["event"] = ev_c

    print(f"files={len(files)} config={cfg} events={n_ev} "
          f"crossings={len(X['layer'])} clusters={len(K['layer'])}")
    return hit, cfg, X, K, n_ev


def join_on_module(X, K):
    """Expand to (crossing, cluster) PAIRS sharing (event, detId).

    Fully vectorized: sort clusters by the packed key, searchsorted the crossing
    keys, then expand with repeat + a ramp. A python loop over ~70k crossings x
    ~40 clusters is avoidable and would dominate the runtime.
    """
    ckey = K["event"].astype(np.int64) * (1 << 32) + K["detId"].astype(np.int64)
    order = np.argsort(ckey, kind="stable")
    cs = ckey[order]
    xkey = X["event"].astype(np.int64) * (1 << 32) + X["detId"].astype(np.int64)
    lo = np.searchsorted(cs, xkey, "left")
    hi = np.searchsorted(cs, xkey, "right")
    n = hi - lo
    xi = np.repeat(np.arange(len(xkey)), n)                       # crossing index
    ramp = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)     # 0..n-1 per crossing
    ci = order[np.repeat(lo, n) + ramp]                            # cluster index
    return xi, ci, n


def prepare(X, K):
    xi, ci, n_on_module = join_on_module(X, K)
    P = {}
    P["xi"], P["ci"] = xi, ci
    P["n_on_module"] = n_on_module
    # displacement of each cluster from the SEED projection, in sigma
    dx = K["localX"][ci] - X["projSeedLocalX"][xi]
    dy = K["localY"][ci] - X["projSeedLocalY"][xi]
    sx = X["projSeedSigX"][xi]
    sy = X["projSeedSigY"][xi]
    good = (sx > 0) & (sy > 0) & (X["projSeedLocalX"][xi] > SENTINEL)
    with np.errstate(divide="ignore", invalid="ignore"):
        P["nsigx"] = np.where(good, dx / sx, np.inf)
        P["nsigy"] = np.where(good, dy / sy, np.inf)
    P["good"] = good
    # angle mismatch vs the track's expectation at this module
    P["dCotA"] = K["truthCotAlpha"][ci] - X["projSeedCotAlpha"][xi]
    P["dCotB"] = K["truthCotBeta"][ci] - X["projSeedCotBeta"][xi]
    P["ang_ok"] = (K["truthCotAlpha"][ci] > SENTINEL) & (X["projSeedCotAlpha"][xi] > SENTINEL)
    P["is_selected"] = match_selected(X, K, P)
    return P


def in_cone(P, k):
    return P["good"] & (np.abs(P["nsigx"]) < k) & (np.abs(P["nsigy"]) < k)


def match_selected(X, K, P):
    """Boolean over PAIRS: is this cluster the one the crossing selected?

    Matched as the NEAREST cluster on the module to the stored reco position,
    not by equality. Both tables write positions with 10-bit nano mantissa
    precision, which on a ~0.5 cm coordinate is ~5 um -- comparable to the 25 um
    x pitch -- so an equality test finds essentially nothing (measured: 1 match in
    3368 crossings). Nearest-match is immune to that.

    A selClusterIdx column on the refit hit table would make this exact and is the
    right long-term fix; until then this is a reconstruction of a link the file
    does not carry.
    """
    xi, ci = P["xi"], P["ci"]
    acc = (X["hitAccepted"] > 0) & (X["recoLocalX"] > SENTINEL)
    d2 = ((K["localX"][ci] - X["recoLocalX"][xi]) ** 2
          + (K["localY"][ci] - X["recoLocalY"][xi]) ** 2)
    d2 = np.where(acc[xi], d2, np.inf)
    best = np.full(len(X["layer"]), np.inf)
    np.minimum.at(best, xi, d2)
    sel = np.isfinite(d2) & (d2 <= best[xi]) & acc[xi]
    # guard against ties selecting two clusters for one crossing
    first = np.zeros(len(X["layer"]), dtype=bool)
    keep = np.zeros(len(xi), dtype=bool)
    idx = np.flatnonzero(sel)
    for j in idx:                      # ties are rare; cheap loop over matches only
        if not first[xi[j]]:
            first[xi[j]] = True
            keep[j] = True
    return keep


# --------------------------------------------------------------------------
# studies
# --------------------------------------------------------------------------
def study_cone_occupancy(X, K, P, ax_row, out):
    """(1) clusters inside the seed cone, per layer and vs pT / eta."""
    xlay = X["layer"]
    res = {}
    ax = ax_row[0]
    for cname, k in CONES.items():
        m = in_cone(P, k)
        cnt = np.bincount(P["xi"][m], minlength=len(xlay)).astype(float)
        per_layer = [cnt[xlay == L].mean() if (xlay == L).any() else np.nan for L in LAYERS]
        res[cname] = {"per_layer_mean": [float(v) for v in per_layer]}
        ax.plot(LAYERS, per_layer, marker="o", label=f"seed cone {cname}")
        P[f"cnt_{cname}"] = cnt
    tot = np.bincount(P["xi"], minlength=len(xlay)).astype(float)
    ax.plot(LAYERS, [tot[xlay == L].mean() for L in LAYERS], marker="s", ls="--",
            color="k", label="all on module")
    ax.set_yscale("log"); ax.set_xlabel("TBPX layer"); ax.set_ylabel("clusters / crossing")
    ax.set_title("(1) candidates per crossing"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    res["all_on_module_per_layer"] = [float(tot[xlay == L].mean()) for L in LAYERS]

    for ax, var, edges, lab in ((ax_row[1], "trk_pt", PT_EDGES, r"track $p_T$ [GeV]"),
                                (ax_row[2], "trk_eta", ETA_EDGES, r"track $\eta$")):
        v = X[var]
        idx = np.digitize(v, edges) - 1
        for cname in CONES:
            cnt = P[f"cnt_{cname}"]
            ys = [cnt[idx == i].mean() if (idx == i).any() else np.nan
                  for i in range(len(edges) - 1)]
            ctr = [0.5 * (edges[i] + min(edges[i + 1], 40)) for i in range(len(edges) - 1)]
            ax.plot(ctr, ys, marker="o", label=cname)
            res.setdefault(f"vs_{var}", {})[cname] = {
                "bin_centres": [float(c) for c in ctr], "mean": [float(y) for y in ys]}
        ax.set_xlabel(lab); ax.set_ylabel("clusters in cone / crossing")
        ax.set_title(f"(1) cone occupancy vs {lab}"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    out["cone_occupancy"] = res


def study_cone_containment(X, K, P, ax_row, out):
    """(2) does the cone contain the cluster the track actually needs?

    CONDITIONING, stated because it biases the number upward: the only clusters
    known to belong to this track are the ones the refit SELECTED and truth
    labelled selHitClass==0. A truth cluster that the current static window never
    offered cannot appear here. An unbiased efficiency needs a TP identifier on
    every cluster row, which the tier does not yet carry -- listed in the writeup.
    """
    ok = (X["hitAccepted"] > 0) & (X["selHitClass"] == 0) & (X["recoLocalX"] > SENTINEL) \
         & (X["projSeedSigX"] > 0) & (X["projSeedLocalX"] > SENTINEL)
    res = {"n_reference_hits": int(ok.sum()), "conditioning":
           "selected-and-truth-correct hits only; biased upward, see docstring"}
    ax = ax_row[0]
    for cname, k in CONES.items():
        nx = np.abs(X["recoLocalX"] - X["projSeedLocalX"]) / np.where(X["projSeedSigX"] > 0, X["projSeedSigX"], np.nan)
        ny = np.abs(X["recoLocalY"] - X["projSeedLocalY"]) / np.where(X["projSeedSigY"] > 0, X["projSeedSigY"], np.nan)
        inside = ok & (nx < k) & (ny < k)
        eff = [inside[ok & (X["layer"] == L)].mean() if (ok & (X["layer"] == L)).any() else np.nan
               for L in LAYERS]
        ax.plot(LAYERS, eff, marker="o", label=cname)
        res[cname] = {"per_layer_eff": [float(e) for e in eff],
                      "overall_eff": float(inside[ok].mean()) if ok.any() else float("nan")}
    # The cone is a per-coordinate BOX cut at k sigma, so for an ideal Gaussian
    # projection the JOINT containment is (2*Phi(k)-1)^2, not 2*Phi(k)-1. Drawing
    # 0.68/0.95 here would make a correctly-sized cone look broken.
    from math import erf, sqrt
    for k, c in ((CONES["q68"], "grey"), (CONES["q95"], "grey")):
        ideal = erf(k / sqrt(2.0)) ** 2
        ax.axhline(ideal, color=c, ls=":", lw=1)
        ax.text(4.05, ideal, f" ideal {ideal:.3f}", fontsize=6, va="center", color=c)
        res.setdefault("ideal_box_containment", {})[f"k={k:.2f}"] = float(ideal)
    ax.set_xlabel("TBPX layer"); ax.set_ylabel("containment efficiency")
    ax.set_title("(2) correct hit inside the seed cone"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_ylim(0, 1.05)

    # cone size itself, the thing that drives (1) and (2) together
    ax = ax_row[1]
    for nm, key in (("$\\sigma_x$", "projSeedSigX"), ("$\\sigma_y$", "projSeedSigY")):
        v = [np.median(X[key][(X["layer"] == L) & (X[key] > 0)]) * 1e4 for L in LAYERS]
        ax.plot(LAYERS, v, marker="o", label=nm)
        res.setdefault("cone_sigma_um", {})[key] = [float(x) for x in v]
    ax.set_yscale("log"); ax.set_xlabel("TBPX layer"); ax.set_ylabel(r"median cone $\sigma$ [$\mu$m]")
    ax.set_title("(2) seed cone size"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    out["cone_containment"] = res


def study_angle_discrimination(X, K, P, ax_row, out):
    """(3) how much of the in-cone pool could an alpha/beta cut remove?

    UPPER BOUND, not a capability: the cluster angles here are UNSMEARED truth
    (truthCotAlpha/Beta). A real smart-pixel angle carries the PixelAV response
    smear, so achievable rejection is strictly worse than this.
    """
    m = in_cone(P, CONES["q95"]) & P["ang_ok"]
    correct = (X["hitAccepted"] > 0) & (X["selHitClass"] == 0)
    same = P["is_selected"] & correct[P["xi"]]
    res = {"note": "unsmeared truth angles: an UPPER BOUND on angle rejection"}
    for ax, d, nm in ((ax_row[0], P["dCotA"], r"$\Delta\cot\alpha$"),
                      (ax_row[1], P["dCotB"], r"$\Delta\cot\beta$")):
        a_sig, a_bkg = d[m & same], d[m & ~same]
        rng = np.nanpercentile(np.abs(d[m]), 99) if m.any() else 1.0
        bins = np.linspace(-rng, rng, 80)
        for arr, lab in ((a_bkg, "other clusters in cone"), (a_sig, "the track's own cluster")):
            if len(arr) > 10:
                ax.hist(arr, bins=bins, histtype="step", density=True, label=f"{lab} (n={len(arr)})")
        ax.set_xlabel(nm); ax.set_ylabel("density"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        ax.set_title(f"(3) {nm} vs track expectation")
        if len(a_sig) > 10 and len(a_bkg) > 10:
            for q in (0.68, 0.95):
                cut = np.quantile(np.abs(a_sig), q)
                res.setdefault(nm, {})[f"keep{int(q*100)}_cut"] = float(cut)
                res[nm][f"keep{int(q*100)}_bkg_rejected"] = float(np.mean(np.abs(a_bkg) > cut))
    out["angle_discrimination"] = res


def study_charge_gate(X, K, P, ax_row, out):
    """(4) the READOUT gate: only the N highest-charge clusters per module can be
    formed into L1 outputs; the rest wait for a Level-1 Accept and are useless to
    the trigger. So: what charge rank does the track's own cluster sit at?"""
    ci, xi = P["ci"], P["xi"]
    # rank each cluster within its module by descending charge
    key = K["event"].astype(np.int64) * (1 << 32) + K["detId"].astype(np.int64)
    order = np.lexsort((-K["charge"].astype(np.float64), key))
    sk = key[order]
    starts = np.r_[True, sk[1:] != sk[:-1]]
    grp = np.cumsum(starts) - 1
    pos = np.arange(len(sk)) - np.flatnonzero(starts)[grp]
    rank = np.empty(len(key), dtype=np.int64)
    rank[order] = pos

    correct = (X["hitAccepted"] > 0) & (X["selHitClass"] == 0)
    match = P["is_selected"] & correct[xi]
    r = rank[ci][match]
    res = {"n_matched": int(match.sum())}
    ax = ax_row[0]
    if len(r) > 10:
        maxr = int(np.quantile(r, 0.99)) + 1
        ax.hist(r, bins=np.arange(0, max(maxr, 8) + 1) - 0.5, histtype="stepfilled", alpha=.7)
        ax.set_xlabel("charge rank within module (0 = highest)")
        ax.set_ylabel("truth-correct hits")
        ax.set_title("(4) where the needed cluster sits in charge order")
        ax.grid(alpha=.3)
        keep = {N: float(np.mean(r < N)) for N in (1, 2, 4, 8, 16, 32)}
        res["survival_vs_topN"] = keep
        ax2 = ax_row[1]
        Ns = sorted(keep)
        ax2.plot(Ns, [keep[N] for N in Ns], marker="o")
        ax2.set_xscale("log", base=2); ax2.set_xlabel("read out top-N by charge per module")
        ax2.set_ylabel("fraction of needed clusters kept")
        ax2.set_title("(4) readout gate efficiency"); ax2.grid(alpha=.3); ax2.set_ylim(0, 1.05)
    out["charge_gate"] = res


STUDIES = [
    ("cone occupancy", study_cone_occupancy, 3),
    ("cone containment + size", study_cone_containment, 2),
    ("angle discrimination", study_angle_discrimination, 2),
    ("charge readout gate", study_charge_gate, 2),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputs", nargs="+", required=True)
    ap.add_argument("-o", "--outdir", default="eval_refitq/combinatorics")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    hit, cfg, X, K, n_ev = load(args.inputs)
    P = prepare(X, K)
    out = {"config": cfg, "n_events": n_ev, "n_crossings": int(len(X["layer"])),
           "n_clusters": int(len(K["layer"]))}

    ncols = max(n for _, _, n in STUDIES)
    fig, axes = plt.subplots(len(STUDIES), ncols, figsize=(5.2 * ncols, 4.2 * len(STUDIES)))
    axes = np.atleast_2d(axes)
    for row, (name, fn, n) in enumerate(STUDIES):
        print(f"  study: {name}")
        fn(X, K, P, axes[row], out)
        for c in range(n, ncols):
            axes[row][c].axis("off")
    fig.suptitle(f"SmartPixels combinatorics omnibus — {cfg}, {n_ev} events", y=1.005)
    fig.tight_layout()
    png = os.path.join(args.outdir, "spix_combinatorics_omnibus.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    js = os.path.join(args.outdir, "spix_combinatorics_omnibus.json")
    with open(js, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {png}\nwrote {js}")


if __name__ == "__main__":
    main()
