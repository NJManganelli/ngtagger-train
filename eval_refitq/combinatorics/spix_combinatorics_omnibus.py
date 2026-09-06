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
from matplotlib.colors import LogNorm
import numpy as np
from statistics import NormalDist
import uproot

SENTINEL = -900.0
LAYERS = (1, 2, 3, 4)
CONES = {"q68": 1.0, "q95": 1.959964}
PT_EDGES = np.array([2, 3, 4, 6, 10, 20, 1e9])
ETA_EDGES = np.array([-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4])
CLUSTER_TABLE = "L1TSmartPixelsCluster"
REF = "L1TTrack"
# eta bin EDGES for the cot(theta) resolution study; cot(theta) = sinh(eta)
ETA_RES_EDGES = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4])
Z_HALF_RANGE_CM = 15.0   # +-z0 span a seeding stage would have to slice

def robust_sigma(a):
    """MAD-scaled width and the half 16-84 interval, as (mad, q68).

    Both are quoted because they disagree exactly when it matters: the
    cot(theta) residual has tails, and a plain std would chase them.
    """
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return float("nan"), float("nan")
    mad = 1.4826 * np.median(np.abs(a - np.median(a)))
    lo, hi = np.percentile(a, [16, 84])
    return float(mad), float(0.5 * (hi - lo))


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
                "projCotAlpha", "projCotBeta", "projSigX", "projSigY",
                "recoLocalX", "recoLocalY", "selClusterIdx"]
    miss_h = [c for c in need_hit if f"{hit}_{c}" not in keys]
    miss_c = ([CLUSTER_TABLE] if not any(k.startswith(CLUSTER_TABLE + "_") for k in keys)
              else [c for c in ("layer", "detId", "localX", "localY", "charge",
                                "tpPt", "tpLocalCotAlpha", "tpLocalCotBeta", "tpIdx")
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
    ccols = ["layer", "detId", "localX", "localY", "charge", "tpPt",
             "tpLocalCotAlpha", "tpLocalCotBeta", "sizeX", "sizeY", "tpIdx"]
    # sensor angle + CPE sigma, present once the cluster table reads SmartPixelsRecHit
    optional = {"localCotAlpha": "clLocalCotAlpha", "localCotBeta": "clLocalCotBeta",
                "sigAlpha": "clSigAlpha", "sigBeta": "clSigBeta",
                "sigX": "sigX", "sigY": "sigY", "hasAlpha": "clHasAlpha",
                # global frame: cluster POSITION (globalR/Z/Phi) and the sensor's
                # estimate of the track DIRECTION there (gClPhi/gClCotTheta), with
                # rotated uncertainties. Study (7); absent in older files.
                "globalR": "globalR", "globalZ": "globalZ", "globalPhi": "globalPhi",
                "globalClusterPhi": "gClPhi",
                "globalClusterCotTheta": "gClCotTheta",
                "sigGlobalClusterPhi": "gSigPhi",
                "sigGlobalClusterCotTheta": "gSigCotTheta",
                "tpGlobalClusterPhi": "tpGClPhi",
                "tpGlobalClusterCotTheta": "tpGClCotTheta",
                "hasBeta": "clHasBeta"}
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

    vtrk = uproot.concatenate([f"{f}:Events" for f in files],
                              filter_name=[f"{hit.replace('RefitHit', 'Track')}_spixMatchedTpIdx"])
    vname = f"{hit.replace('RefitHit', 'Track')}_spixMatchedTpIdx"
    mtp = ak.to_numpy(ak.flatten(vtrk[vname]))
    nmt = ak.to_numpy(ak.num(vtrk[vname]))
    offm = np.concatenate([[0], np.cumsum(nmt)])
    X["trk_tpIdx"] = mtp[offm[ev_x] + X["trackIdx"].astype(np.int64)]

    ncl = ak.to_numpy(ak.num(C[f"{CLUSTER_TABLE}_layer"]))
    ev_c = np.repeat(np.arange(len(ncl)), ncl)
    K = {c: ak.to_numpy(ak.flatten(C[f"{CLUSTER_TABLE}_{c}"])) for c in ccols}
    with uproot.open(f"{files[0]}:Events") as t:
        avail = set(t.keys())
    got = [c for c in optional if f"{CLUSTER_TABLE}_{c}" in avail]
    if got:
        O = uproot.concatenate([f"{f}:Events" for f in files],
                               filter_name=[f"{CLUSTER_TABLE}_{c}" for c in got])
        for c in got:
            K[optional[c]] = ak.to_numpy(ak.flatten(O[f"{CLUSTER_TABLE}_{c}"]))
    K["event"] = ev_c
    K["_evt_base"] = np.concatenate([[0], np.cumsum(ncl)])

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
    # Same displacement against the REFIT-ORDER projection: the running covariance
    # after multiple-scattering Q, before this layer's update. projSig* is present on
    # every crossing, including the ones whose window came up empty, so this arm is
    # not silently restricted to crossings that already found a hit.
    dxr = K["localX"][ci] - X["projLocalX"][xi]
    dyr = K["localY"][ci] - X["projLocalY"][xi]
    sxr, syr = X["projSigX"][xi], X["projSigY"][xi]
    good_rf = (sxr > 0) & (syr > 0) & (X["projLocalX"][xi] > SENTINEL)
    with np.errstate(divide="ignore", invalid="ignore"):
        P["nsigx_rf"] = np.where(good_rf, dxr / sxr, np.inf)
        P["nsigy_rf"] = np.where(good_rf, dyr / syr, np.inf)
    P["good_rf"] = good_rf
    # angle mismatch vs the track's expectation at this module
    P["dCotA"] = K["tpLocalCotAlpha"][ci] - X["projSeedCotAlpha"][xi]
    P["dCotB"] = K["tpLocalCotBeta"][ci] - X["projSeedCotBeta"][xi]
    P["ang_ok"] = (K["tpLocalCotAlpha"][ci] > SENTINEL) & (X["projSeedCotAlpha"][xi] > SENTINEL)
    P["sel_gidx"] = selected_global_index(X, K)
    # over PAIRS: is this pair the crossing's selected cluster?
    P["is_selected"] = (P["sel_gidx"][P["xi"]] >= 0) & (P["ci"] == P["sel_gidx"][P["xi"]])
    return P


def in_cone(P, k):
    return P["good"] & (np.abs(P["nsigx"]) < k) & (np.abs(P["nsigy"]) < k)


def in_cone_refit(P, k):
    return P["good_rf"] & (np.abs(P["nsigx_rf"]) < k) & (np.abs(P["nsigy_rf"]) < k)


def selected_global_index(X, K):
    """Global cluster-table row of each crossing's SELECTED cluster, or -1.

    Uses the EXACT selClusterIdx link. Position matching was tried first and is
    unsafe: both tables store coordinates at 10-bit nano mantissa precision, which
    disagrees by up to 9.7 um against a 25 um pitch, so it can silently pick a
    neighbouring cluster.
    """
    g = np.where(X["selClusterIdx"] >= 0,
                 K["_evt_base"][X["event"]] + X["selClusterIdx"].astype(np.int64), -1)
    ok = g >= 0
    if ok.any():
        bad = int((K["detId"][g[ok]] != X["detId"][ok]).sum())
        if bad:
            raise SystemExit(
                f"selClusterIdx integrity check FAILED on {bad} crossings: the cluster it "
                "points at is on a different module. The refit and cluster tables have "
                "diverged in filter or iteration order; do not trust any result from this file.")
    return g


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
    (tpLocalCotAlpha/Beta). A real smart-pixel angle carries the PixelAV response
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


def study_true_containment(X, K, P, ax_row, out):
    """(5) UNBIASED containment: of the clusters that genuinely belong to this
    track's TrackingParticle and sit on the module it crosses, how many does the
    cone hold?

    This is the study that selClusterIdx and tpIdx were added for. Study (2)
    can only ever see clusters the current static window already offered, so it
    cannot detect a true cluster the window never showed the fit -- exactly the
    failure a cone redesign is meant to fix. Here the truth clusters are found by
    joining tpIdx to the track's spixMatchedTpIdx, independently of what the
    window did, so a cluster the window missed still counts against the cone.
    """
    xi, ci = P["xi"], P["ci"]
    have = (X["trk_tpIdx"][xi] >= 0) & (K["tpIdx"][ci] >= 0)
    is_true = have & (K["tpIdx"][ci] == X["trk_tpIdx"][xi])
    res = {"n_true_pairs": int(is_true.sum())}
    ax = ax_row[0]
    for cname, k in CONES.items():
        m = is_true & P["good"]
        if m.sum() < 20:
            continue
        inside = m & in_cone(P, k)
        eff = []
        for L in LAYERS:
            sel = m & (X["layer"][xi] == L)
            eff.append(float(inside[sel].mean()) if sel.any() else np.nan)
        ax.plot(LAYERS, eff, marker="o", label=f"{cname} (unbiased)")
        res[cname] = {"per_layer_eff": eff, "overall_eff": float(inside[m].mean())}
    # the biased version, for direct contrast on the same axes
    bc = out.get("cone_containment", {})
    for cname in CONES:
        if cname in bc:
            ax.plot(LAYERS, bc[cname]["per_layer_eff"], marker="s", ls="--", alpha=.6,
                    label=f"{cname} (window-conditioned)")
    ax.set_xlabel("TBPX layer"); ax.set_ylabel("containment efficiency")
    ax.set_title("(5) TRUE containment vs window-conditioned")
    ax.legend(fontsize=6); ax.grid(alpha=.3); ax.set_ylim(0, 1.05)

    # how many true clusters does a track even have on the module it crosses?
    ax = ax_row[1]
    n_true = np.bincount(xi[is_true], minlength=len(X["layer"]))
    for L in LAYERS:
        sel = X["layer"] == L
        if sel.any():
            ax.plot(L, n_true[sel].mean(), marker="o", color="C0")
    res["mean_true_clusters_on_module"] = [
        float(n_true[X["layer"] == L].mean()) if (X["layer"] == L).any() else float("nan")
        for L in LAYERS]
    ax.plot(LAYERS, res["mean_true_clusters_on_module"], color="C0")
    ax.set_xlabel("TBPX layer"); ax.set_ylabel("true clusters on crossed module")
    ax.set_title("(5) how many are there to find"); ax.grid(alpha=.3)
    out["true_containment"] = res


def study_chi2_weight_scan(X, K, P, ax_row, out):
    """(6) WHICH selection chi2 picks the cleanest hits?

    The refit currently selects the candidate minimising

        sel = (dx/sigx)^2 + (dy/sigy)^2 + [alpha] (dcotA/sigA)^2 + [beta] (dcotB/sigB)^2

    i.e. all four terms at unit weight. That is a choice, not a derivation: the
    angle terms come from a sensor estimator whose resolution is not commensurate
    with the CPE position resolution, and the r-phi and r-z terms are not equally
    informative either. This scans the weights and asks which combination picks
    the TRUTH-CORRECT cluster most often.

    THE FIGURE OF MERIT is per-crossing hit-selection purity: of the crossings
    where the correct cluster is present in the window at all, how often does the
    weighted metric rank it first. That is the quantity the refit's parameter
    resolution is downstream of -- measured earlier, one wrong hit annihilates the
    refit gain, and the whole outsideIn win came from selection rather than from
    fitting.

    Scans, as requested:
      * coarse over (w_rphi, w_rz) applied to the POSITION terms, from the
        physics-informed default (1, 1);
      * 1D over w_alpha alone (bending angle only, the beta term off);
      * 2D over (w_alpha, w_beta).

    The angle terms use the SENSOR estimate and its sigma, not truth, so the rule
    itself is deployable -- unlike anything scanned on truth angles.

    HISTORY WORTH KEEPING. This scan was meaningless until 2026-09-05, because
    SmartPixelsRecHitProducer gave NO angle to clusters with no simlink: hasAlpha
    was 99.5% for TP-linked clusters and 0.0% for unlinked ones, so "reports an
    angle" was a perfect proxy for "is real" and any angle weight bought that
    proxy rather than angle information. The scan now runs a LEAK SELF-CHECK on
    every invocation and says so if the two rates diverge again -- a measured
    guard, not a comment that can go stale.
    """
    xi, ci = P["xi"], P["ci"]
    need = ("clLocalCotAlpha", "clLocalCotBeta", "clSigAlpha", "clSigBeta")
    if not all(k in K for k in need):
        print("   (6) SKIPPED: cluster table lacks the sensor angle columns "
              "(localCotAlpha/localCotBeta/sigAlpha/sigBeta)")
        out["chi2_weight_scan"] = {"skipped": "cluster table has no sensor angle columns"}
        for a in ax_row:
            a.axis("off")
        return

    # per-pair residuals against the RUNNING projection (what the refit compares to)
    dx = (K["localX"][ci] - X["projLocalX"][xi]) / np.maximum(K["sigX"][ci], 1e-6)
    dy = (K["localY"][ci] - X["projLocalY"][xi]) / np.maximum(K["sigY"][ci], 1e-6)
    okA = (K["clLocalCotAlpha"][ci] > SENTINEL) & (K["clSigAlpha"][ci] > 0) \
          & (X["projCotAlpha"][xi] > SENTINEL)
    okB = (K["clLocalCotBeta"][ci] > SENTINEL) & (K["clSigBeta"][ci] > 0) \
          & (X["projCotBeta"][xi] > SENTINEL)
    # A cluster whose sensor reports NO angle must be neither rewarded nor punished
    # for it. Filling its normalized residual with 0 (the naive choice) makes it
    # cost-free, so any large angle weight simply selects angle-less clusters -- an
    # artefact that made angle-only selection look catastrophic (0.064) and made
    # huge weights look beneficial. The unbiased fill is the EXPECTATION of a
    # normalized residual squared, i.e. 1, so a missing term contributes its mean
    # and the comparison stays fair across candidates with different term counts.
    NEUTRAL = 1.0
    da2 = np.where(okA, ((K["clLocalCotAlpha"][ci] - X["projCotAlpha"][xi])
                         / np.maximum(K["clSigAlpha"][ci], 1e-9)) ** 2, NEUTRAL)
    db2 = np.where(okB, ((K["clLocalCotBeta"][ci] - X["projCotBeta"][xi])
                         / np.maximum(K["clSigBeta"][ci], 1e-9)) ** 2, NEUTRAL)

    # the correct cluster for each crossing, from the TP join (window-independent)
    have = (X["trk_tpIdx"][xi] >= 0) & (K["tpIdx"][ci] >= 0)
    is_true = have & (K["tpIdx"][ci] == X["trk_tpIdx"][xi])
    ncross = len(X["layer"])
    has_true = np.zeros(ncross, dtype=bool)
    has_true[xi[is_true]] = True
    inwin = P["good"]

    def purity(wrphi, wrz, wa, wb):
        """Fraction of crossings whose lowest-cost candidate is the correct one."""
        cost = wrphi * dx * dx + wrz * dy * dy + wa * da2 + wb * db2
        cost = np.where(inwin, cost, np.inf)
        best = np.full(ncross, np.inf)
        np.minimum.at(best, xi, cost)
        picked_true = np.zeros(ncross, dtype=bool)
        sel = np.isfinite(cost) & (cost <= best[xi]) & is_true
        picked_true[xi[sel]] = True
        d = has_true & np.isfinite(best)
        return float(picked_true[d].mean()) if d.any() else float("nan"), int(d.sum())

    # SCALE INVARIANCE: the argmin is unchanged by a global rescaling, so only
    # RATIOS matter and w_rphi is pinned to 1. Four weights therefore have THREE
    # free parameters, and they are scanned JOINTLY.
    #
    # The earlier version of this scan was wrong and is worth recording. It
    # optimised GREEDILY: first w_rz at unit angle weights, then the angle weights
    # at that frozen w_rz. That silently assumes the best position balance does not
    # depend on how much the angles are trusted, which is false -- once the angles
    # dominate, the position terms are a tiebreak and their optimal ratio changes.
    # It also mislabelled "position + alpha" as an "alpha-only scan" while the
    # LIMITS block used "alpha only" for alpha with NO position, so two different
    # things carried the same name.
    res = {"note": "w_rphi pinned to 1 (scale invariance); the remaining three weights "
                   "are scanned JOINTLY, not greedily"}
    base, n_den = purity(1, 1, 1, 1)
    res["baseline_all_unit_weights"] = {"purity": base, "n_crossings": n_den}
    print(f"   (6) baseline, all four terms at unit weight: purity {base:.4f} "
          f"over {n_den} crossings")

    rzgrid = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    # The angle-weight grid runs to 1e6 rather than stopping at 4096 because it
    # RAILED there: the earlier scan reported its optimum sitting exactly on the
    # ceiling, which means the optimum was outside the range and the reported
    # weights were an artefact of where the grid stopped. A scan that ends on its
    # own boundary has not found a maximum, it has found an edge. Extending far
    # enough to see the purity PLATEAU is what distinguishes "the angles should be
    # weighted very heavily" from "we never looked far enough".
    wgrid = [0.0, 0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0,
             16384.0, 65536.0, 262144.0, 1048576.0]

    # 3D scan: (w_rphi=1, w_rz, w_alpha), beta OFF. The bending angle only, but
    # the POSITION terms are scanned with it rather than frozen.
    s3 = {}
    for wz in rzgrid:
        for wa in wgrid:
            s3[f"{wz}_{wa}"] = purity(1.0, wz, wa, 0.0)[0]
    b3 = max(s3, key=s3.get)
    res["scan3D_rphi_rz_alpha"] = s3
    res["best3D"] = {"w_rz": float(b3.split("_")[0]), "w_alpha": float(b3.split("_")[1]),
                     "purity": s3[b3]}
    print(f"       3D (w_rz, w_alpha; beta off) best {b3} -> {s3[b3]:.4f}")

    # 4D scan: (w_rphi=1, w_rz, w_alpha, w_beta). Fully joint.
    s4 = {}
    for wz in rzgrid:
        for wa in wgrid:
            for wb in wgrid:
                s4[f"{wz}_{wa}_{wb}"] = purity(1.0, wz, wa, wb)[0]
    b4 = max(s4, key=s4.get)
    wz4, wa4, wb4 = (float(v) for v in b4.split("_"))
    res["scan4D_rphi_rz_alpha_beta"] = s4
    res["best4D"] = {"w_rphi": 1.0, "w_rz": wz4, "w_alpha": wa4, "w_beta": wb4,
                     "purity": s4[b4]}
    print(f"       4D (w_rz, w_alpha, w_beta) best {b4} -> {s4[b4]:.4f}   "
          f"(baseline {base:.4f}, gain {s4[b4]-base:+.4f})")
    # Does the 4D optimum contain the 3D one? If the best w_rz differs between the
    # two, the greedy version could not have found this point.
    if res["best3D"]["w_rz"] != wz4:
        print(f"       NOTE best w_rz differs between the 3D ({res['best3D']['w_rz']}) and "
              f"4D ({wz4}) scans, which is exactly what the greedy version could not see")
    wr0, wz0 = 1.0, wz4
    a1 = {str(w): s3[f"{wz4}_{w}"] for w in wgrid}   # alpha slice at the 4D-best w_rz
    a2 = {f"{wa}_{wb}": s4[f"{wz4}_{wa}_{wb}"] for wa in wgrid for wb in wgrid}

    # ABLATIONS: each row uses a STRICT SUBSET of the four terms, so a lower purity
    # means that subset carries less information -- NOT that adding information hurt.
    # Ordered smallest subset first so the monotone build-up is visible. These are a
    # different thing from the SCANS above, which always keep the position terms and
    # vary only the weights.
    lim = [("alpha alone              (0,0,1,0)", purity(0.0, 0.0, 1.0, 0.0)[0]),
           ("beta alone               (0,0,0,1)", purity(0.0, 0.0, 0.0, 1.0)[0]),
           ("both angles, no position (0,0,1,1)", purity(0.0, 0.0, 1.0, 1.0)[0]),
           ("position alone           (1,wz,0,0)", purity(1.0, wz0, 0.0, 0.0)[0]),
           ("ALL four, unit weights   (1,1,1,1)", base),
           ("ALL four, weights tuned  (4D best)", s4[b4])]
    res["ablations"] = {k.strip(): v for k, v in lim}
    print("       ABLATIONS -- each row is a STRICT SUBSET of the four terms, so purity")
    print("       RISES as information is added. Subsets, not regressions:")
    for k, v in lim:
        print(f"         {k:<38} {v:.4f}")

    # LEAK SELF-CHECK, measured not asserted. Before the noise payload existed,
    # unlinked clusters carried NO angle (hasAlpha 99.5% TP-linked vs 0.0%
    # unlinked), so "reports an angle" was a perfect proxy for "is real" and any
    # angle weight bought that proxy. Recomputed every run so it cannot go stale.
    if "clHasAlpha" in K:
        lk = K["tpIdx"] >= 0
        ra, rb = float(K["clHasAlpha"][lk].mean()), float(K["clHasAlpha"][~lk].mean())
        res["angle_presence_TPlinked"], res["angle_presence_unlinked"] = ra, rb
        if abs(ra - rb) > 0.05:
            print(f"       *** CONFOUNDED: hasAlpha {100*ra:.1f}% TP-linked vs {100*rb:.1f}% "
                  "unlinked -- angle weight partly buys that proxy. Do not tune on this.")
            res["CONFOUNDED"] = f"angle presence differs by {abs(ra-rb):.3f} between classes"
        else:
            print(f"       leak self-check OK: hasAlpha {100*ra:.1f}% TP-linked vs "
                  f"{100*rb:.1f}% unlinked, so angle presence carries no class information")
    out["chi2_weight_scan"] = res

    ax = ax_row[0]
    for wz in rzgrid:
        ax.plot(wgrid, [s3[f"{wz}_{w}"] for w in wgrid], marker=".",
                label=rf"$w_{{rz}}$={wz}")
    ax.axhline(base, color="k", ls=":", label="baseline (1,1,1,1)")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel(r"$w_\alpha$   ($w_{r\phi}\equiv1$, $w_\beta=0$)")
    ax.set_ylabel("hit-selection purity")
    ax.set_title("(6) 3D scan: position terms NOT frozen")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)

    ax = ax_row[1]
    M2 = np.array([[a2[f"{wa}_{wb}"] for wb in wgrid] for wa in wgrid])
    im2 = ax.imshow(M2, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(wgrid))); ax.set_xticklabels(wgrid, rotation=45, fontsize=7)
    ax.set_yticks(range(len(wgrid))); ax.set_yticklabels(wgrid, fontsize=7)
    ax.set_xlabel(r"$w_\beta$"); ax.set_ylabel(r"$w_\alpha$")
    ax.set_title("(6) angle-weight scan (1D = bottom row)"); plt.colorbar(im2, ax=ax)



def study_refit_cone_occupancy(X, K, P, ax_row, out):
    """(9) Candidates per crossing: naive SEED cone vs outsideIn REFIT-ORDER cone.

    THIS IS THE HALF OF THE COMBINATORICS QUESTION THAT WAS NEVER BUILT. Study (1)
    projects with the OT seed covariance -- one shot, no updates -- which answers
    "how many clusters would a naive projection have to test". The actual refit
    walks outsideIn and tightens its covariance at every layer, so the number it
    must test is smaller, and by how much is the thing that decides whether the
    per-track combinatorics are affordable. Nothing measured it until projSig*
    existed.

    IT ONLY BECAME MEANINGFUL AFTER THE Q TERM. Before process noise the running
    covariance was up to 2.7x too tight (correct-hit pull width 4.13 at L1), so a
    refit-order cone would have looked spectacularly better than the seed cone for
    the worst possible reason -- it was lying about its own precision and would
    have thrown away real hits. Any number from this study taken before Q is not a
    physics result, it is the bug.

    The reduction reported here is therefore an HONEST one: the covariance now
    passes a two-sided pull test (see eval_refitq/windows/q_acceptance.py).
    """
    xlay = X["layer"]
    res = {}
    ax = ax_row[0]
    for cname, k in CONES.items():
        ms, mr = in_cone(P, k), in_cone_refit(P, k)
        cs = np.bincount(P["xi"][ms], minlength=len(xlay)).astype(float)
        cr = np.bincount(P["xi"][mr], minlength=len(xlay)).astype(float)
        P[f"cnt_rf_{cname}"] = cr
        ys = [cs[xlay == L].mean() if (xlay == L).any() else np.nan for L in LAYERS]
        yr = [cr[xlay == L].mean() if (xlay == L).any() else np.nan for L in LAYERS]
        ax.plot(LAYERS, ys, marker="o", ls="--", label=f"seed {cname}")
        ax.plot(LAYERS, yr, marker="s", label=f"refit-order {cname}")
        res[cname] = {"seed_per_layer": [float(v) for v in ys],
                      "refit_per_layer": [float(v) for v in yr],
                      "reduction_per_layer": [float(a / b) if b > 0 else float("nan")
                                              for a, b in zip(ys, yr)]}
    tot = np.bincount(P["xi"], minlength=len(xlay)).astype(float)
    ax.plot(LAYERS, [tot[xlay == L].mean() for L in LAYERS], marker="^", ls=":",
            color="k", label="all on module")
    ax.set_yscale("log"); ax.set_xlabel("TBPX layer")
    ax.set_ylabel("candidates / crossing")
    ax.set_title("(9) seed cone vs refit-order cone")
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    # cone half-width itself, in microns: the mechanism behind the count above
    ax = ax_row[1]
    hs, hr = [], []
    for L in LAYERS:
        m = xlay == L
        a = X["projSeedSigX"][m]; b = X["projSigX"][m]
        hs.append(float(np.median(a[a > 0]) * 1e4) if (a > 0).any() else np.nan)
        hr.append(float(np.median(b[b > 0]) * 1e4) if (b > 0).any() else np.nan)
    ax.plot(LAYERS, hs, marker="o", ls="--", label="seed sigma_x")
    ax.plot(LAYERS, hr, marker="s", label="refit-order sigma_x")
    ax.set_xlabel("TBPX layer"); ax.set_ylabel(r"median projected $\sigma_x$ [$\mu$m]")
    ax.set_title("(9) projection cone half-width"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    res["median_sigma_x_um"] = {"seed": hs, "refit": hr}

    # does the gain survive at low pT, where scattering is worst?
    ax = ax_row[2]
    idx = np.digitize(X["trk_pt"], PT_EDGES) - 1
    ctr = [0.5 * (PT_EDGES[i] + min(PT_EDGES[i + 1], 40)) for i in range(len(PT_EDGES) - 1)]
    for cname in CONES:
        cs = P[f"cnt_{cname}"] if f"cnt_{cname}" in P else None
        cr = P[f"cnt_rf_{cname}"]
        if cs is None:
            continue
        ratio = []
        for i in range(len(PT_EDGES) - 1):
            m = idx == i
            if not m.any():
                ratio.append(np.nan); continue
            a, b = cs[m].mean(), cr[m].mean()
            ratio.append(float(a / b) if b > 0 else np.nan)
        ax.plot(ctr, ratio, marker="o", label=cname)
        res.setdefault("reduction_vs_pt", {})[cname] = {
            "bin_centres": [float(c) for c in ctr], "ratio": ratio}
    ax.axhline(1.0, color="k", lw=.8, ls=":")
    ax.set_xlabel(r"track $p_T$ [GeV]"); ax.set_ylabel("seed / refit-order candidates")
    ax.set_title("(9) combinatorics reduction vs $p_T$"); ax.legend(fontsize=7); ax.grid(alpha=.3)

    print("\n=== (9) refit-order cone vs naive seed cone ===")
    print(f"  median projected sigma_x [um] by layer")
    print(f"    seed        : " + "  ".join(f"{v:7.1f}" for v in hs))
    print(f"    refit-order : " + "  ".join(f"{v:7.1f}" for v in hr))
    for cname in CONES:
        r = res[cname]
        print(f"  {cname}: candidates/crossing seed "
              + "/".join(f"{v:.2f}" for v in r["seed_per_layer"])
              + "  refit " + "/".join(f"{v:.2f}" for v in r["refit_per_layer"])
              + "  reduction " + "/".join(f"{v:.2f}x" for v in r["reduction_per_layer"]))
    out["refit_cone_occupancy"] = res


def study_z0_resolution(X, K, P, ax_row, out):
    """(7) sigma(cot theta), and the z0 resolution it implies.

    THE QUESTION. A cluster measures both a POSITION (globalR, globalZ) and a
    DIRECTION (globalClusterCotTheta), so on its own it determines a longitudinal
    impact parameter:

        z0 = z - r * cot(theta)         =>   sigma(z0) = r * sigma(cot theta)

    If that z0 is sharp, a seeding Hough transform gains a third nearly free
    dimension: clusters can be sliced by z0 BEFORE the (phi0, q/pT) transform
    runs, and since random k-layer coincidences scale as the per-cell density to
    the k-th power, the fake rate falls as the CUBE of the slicing factor for a
    3-layer seed. `doc/SmartPixelsSeedingAndFitting.md` (sections 6, 11, 12)
    makes this the single largest factor in the design and flags it as ASSUMED.
    This measures it.

    THE SUPPRESSION FACTOR is Z_range / (4 sigma), not Z_range / (slice width).
    A hit must vote into every slice its z0 could belong to, so it occupies
    ~4 sigma / w of the w-wide slices and the w cancels. Choosing a fine slicing
    buys nothing on its own -- only a small sigma does.

    *** THIS RESIDUAL IS CIRCULAR. IT IS NOT A PHYSICAL RESOLUTION. ***
    SmartPixelsRecHitProducer builds the truth angle as the dominant TP's helix
    PROPAGATED TO THE HIT, with no multiple scattering (:292-319, and the file's
    own header at :32-40), then forms the reco angle as
    `cotB = trueCotB + corrBetaShift_->evaluate(...)` (:330-331). The nano truth
    column is that same trueCotBeta() rotated to global
    (L1SmartPixelsClusterTableProducer.cc:199-202). So (reco - truth) is
    IDENTICALLY the PixelAV payload draw, and what this study measures is the
    payload's own width, not how well a sensor knows a real track's direction.
    The tell is in the output below: sigma comes out equal for pT > 2 and for
    0.5-2 GeV to 0.3%, where a physical resolution would degrade toward low pT.

    What is missing is the scattering between the vertex and the module. It is
    not fatal for z0 -- a kink at radius r_s moves the EXTRAPOLATED z0 by
    r_s * delta(cot theta), not r * delta, so material outside the measurement
    radius does not bias it, and L1 has almost nothing inside it. Estimated
    inflation is +0% (L1, eta 0) to +27% (L4, eta 2, 0.5 GeV). See
    doc/SmartPixelsSeedingAndFitting.md section 4a.

    THE NEXT STEP is a closure test: with the dominant TP's production vz on the
    cluster truth block, compare z0_pred = globalZ - globalR * globalClusterCotTheta
    against the TP's actual z0. That is non-circular, and it is UNIFORM across PU
    and signal (TrackingParticles are post-mixing; 86.5% of clusters carry one).
    It gives a PESSIMISTIC BOUND rather than the true resolution, because the
    simulated cluster pairs a scattered position with an unscattered angle: the
    residual carries +(r - r_s)*delta where the physical term is -r_s*delta. Taken
    together with the number below it brackets the answer, which is enough to
    design against.

    NOT PSimHit. It is the only scattering-aware angle available, but it is
    signal-only at ~1.7% of PU200 clusters, so any resolution derived from it
    describes a population with different pT and eta spectra from pileup and would
    carry that asymmetry into everything downstream. SmartPixelsRecHitProducer
    already refuses PSimHit as a production input for this reason (:36-39); the
    same rule applies to using it as a validation reference.

    Resolving this EXACTLY needs the angle to come from the simulated cluster
    SHAPE for every cluster, rather than from helix-truth plus a payload draw.
    Until then, read every number below as the payload smear.

    METHOD. Robust widths only (MAD*1.4826 and the half 16-84 interval) -- the
    residual has tails a plain std would chase.

    BINNING IN ETA IS NOT OPTIONAL. cot(theta) is read out of cluster LENGTH,
    which is shortest at eta ~ 0, so the resolution is expected to be worst
    exactly where most tracks are. A single global number would be a fiction.
    The binning variable is the TRUE angle, eta = asinh(tpGlobalClusterCotTheta),
    so it is not the quantity being measured.

    PULLS. sigGlobalClusterCotTheta is the estimator's own claimed uncertainty.
    Every weighted vote in the seeding design inherits it, so its pull width is
    checked here rather than trusted; a width far from 1.0 means the segment
    lengths would be set from a miscalibrated sigma.
    """
    need = ("gClCotTheta", "tpGClCotTheta", "globalR", "globalZ")
    if not all(k in K for k in need):
        print("   (7) SKIPPED: cluster table lacks the global-frame angle columns "
              "(globalClusterCotTheta / tpGlobalClusterCotTheta / globalR / globalZ)")
        out["z0_resolution"] = {"skipped": "no global-frame angle columns"}
        for a in ax_row:
            a.axis("off")
        return

    ok = (K["tpIdx"] >= 0) & (K["gClCotTheta"] > SENTINEL) & (K["tpGClCotTheta"] > SENTINEL)
    if "clHasBeta" in K:
        ok &= K["clHasBeta"] > 0
    d = K["gClCotTheta"] - K["tpGClCotTheta"]          # cot(theta) residual
    dz0 = K["globalR"] * d                              # cm; z0 = z - r cot(theta)
    eta = np.arcsinh(K["tpGClCotTheta"])                # cot(theta) = sinh(eta)
    lay = K["layer"]
    Zspan = 2.0 * Z_HALF_RANGE_CM

    res = {"n_clusters_used": int(ok.sum()),
           "definition": "sigma = robust width of (reco - truth) on the same cluster",
           "suppression_formula": "Z_range / (4 sigma_z0), w cancels",
           "cotTheta_validated": (
               "globalClusterCotTheta IS the global polar slope dz/dr, verified after the "
               "module-flip fix: per-TP RMS of (z - r*cot) is 0.0087 cm against a 3.18 cm "
               "do-nothing baseline, a 365x collapse. An earlier revision wrongly also "
               "required |cot| to be layer-independent across the POPULATION; that is not a "
               "valid test, because TBPX is a barrel and high-|eta| tracks only reach the "
               "inner layers. Restricted to TPs that reach L4 the medians are flat "
               "(0.577/0.532/0.494/0.477 on L1-L4)."),
           "CIRCULAR": (
               "NOT A PHYSICAL RESOLUTION. SmartPixelsRecHitProducer builds the truth "
               "angle as the TP's helix propagated to the hit with NO multiple scattering "
               "(:292-319), then sets reco = truth + PixelAV draw (:330-331); the nano "
               "truth column is that same value rotated (L1SmartPixelsClusterTableProducer"
               ".cc:199-202). So (reco - truth) IS the payload draw and every sigma below "
               "is the payload's own width. Tell: sigma is equal for pT>2 and 0.5-2 GeV to "
               "0.3%, where a physical resolution would degrade toward low pT. Treat these "
               "as an OPTIMISTIC BOUND. See doc/SmartPixelsSeedingAndFitting.md section 4a.")}

    # ---- per layer, and per layer x |eta| -----------------------------------
    per_layer = {}
    for L in LAYERS:
        m = ok & (lay == L)
        s_cot = robust_sigma(d[m])
        s_z0 = robust_sigma(dz0[m])
        per_layer[f"L{L}"] = {
            "n": int(m.sum()),
            "median_r_cm": float(np.median(K["globalR"][m])) if m.any() else float("nan"),
            "sigma_cotTheta_mad": s_cot[0], "sigma_cotTheta_q68": s_cot[1],
            "sigma_z0_cm_mad": s_z0[0], "sigma_z0_cm_q68": s_z0[1],
            "z0_slices": float(Zspan / (4.0 * s_z0[0])) if s_z0[0] == s_z0[0] else float("nan"),
        }
    res["per_layer"] = per_layer

    eta_tab = {}
    for i in range(len(ETA_RES_EDGES) - 1):
        lo, hi = ETA_RES_EDGES[i], ETA_RES_EDGES[i + 1]
        me = ok & (np.abs(eta) >= lo) & (np.abs(eta) < hi)
        row = {}
        for L in LAYERS:
            s = robust_sigma(dz0[me & (lay == L)])
            row[f"L{L}"] = s[0]
        row["n"] = int(me.sum())
        row["sigma_cotTheta_mad"] = robust_sigma(d[me])[0]
        eta_tab[f"{lo:.1f}-{hi:.1f}"] = row
    res["vs_abs_eta"] = eta_tab

    # ---- the two design targets separately ---------------------------------
    if "tpPt" in K:
        for nm, sel in (("A_pt_gt_2", ok & (K["tpPt"] > 2.0)),
                        ("B_pt_0p5_to_2", ok & (K["tpPt"] > 0.5) & (K["tpPt"] <= 2.0))):
            s = robust_sigma(dz0[sel])
            res.setdefault("by_target", {})[nm] = {
                "n": int(sel.sum()), "sigma_z0_cm_mad": s[0],
                "z0_slices": float(Zspan / (4.0 * s[0])) if s[0] == s[0] else float("nan")}

    # ---- cluster-weighted headline + the pull check -------------------------
    s_all = robust_sigma(dz0[ok])
    res["overall"] = {"sigma_z0_cm_mad": s_all[0], "sigma_z0_cm_q68": s_all[1],
                      "z0_slices": float(Zspan / (4.0 * s_all[0])) if s_all[0] == s_all[0]
                      else float("nan"),
                      "sigma_for_30_slices_cm": float(Zspan / 120.0)}
    if "gSigCotTheta" in K:
        pm = ok & (K["gSigCotTheta"] > 0)
        pull = d[pm] / K["gSigCotTheta"][pm]
        pw = robust_sigma(pull)
        res["pull"] = {"n": int(pm.sum()), "width_mad": pw[0], "width_q68": pw[1],
                       "median": float(np.median(pull)) if pm.any() else float("nan"),
                       "note": "width far from 1.0 => the stored sigma is miscalibrated"}

    print(f"   (7) sigma(z0) overall {s_all[0]*1e4:.0f} um "
          f"-> {res['overall']['z0_slices']:.1f} usable z0 slices "
          f"(design assumed 30, which needs {Zspan/120.0*1e4:.0f} um)")
    for L in LAYERS:
        p = per_layer[f"L{L}"]
        print(f"       L{L}: r={p['median_r_cm']:.1f} cm  sigma(cot)={p['sigma_cotTheta_mad']:.4f}"
              f"  sigma(z0)={p['sigma_z0_cm_mad']*1e4:.0f} um  slices={p['z0_slices']:.1f}")
    if "pull" in res:
        print(f"       pull width {res['pull']['width_mad']:.3f} (want ~1.0)")

    # ---- figures -------------------------------------------------------------
    ax = ax_row[0]
    for k_eta, row in eta_tab.items():
        ys = [row[f"L{L}"] * 1e4 for L in LAYERS]
        ax.plot(LAYERS, ys, marker="o", label=rf"$|\eta|$ {k_eta}")
    ax.axhline(Zspan / 120.0 * 1e4, color="k", ls=":", lw=1)
    ax.text(4.05, Zspan / 120.0 * 1e4, " 30 slices", fontsize=6, va="center")
    ax.set_yscale("log"); ax.set_xlabel("TBPX layer")
    ax.set_ylabel(r"robust $\sigma(z_0)$ [$\mu$m]")
    ax.set_title(r"(7) per-cluster $z_0$ resolution"); ax.legend(fontsize=6); ax.grid(alpha=.3)

    ax = ax_row[1]
    if "pull" in res:
        pm = ok & (K["gSigCotTheta"] > 0)
        ax.hist(np.clip(d[pm] / K["gSigCotTheta"][pm], -5, 5), bins=80,
                histtype="step", density=True, label=f"pull (w={res['pull']['width_mad']:.2f})")
    rng = np.nanpercentile(np.abs(d[ok]), 99) if ok.any() else 1.0
    ax.hist(np.clip(d[ok] / max(rng, 1e-9), -5, 5), bins=80, histtype="step", density=True,
            label=r"$\Delta\cot\theta$ / q99")
    ax.set_xlabel("normalized residual"); ax.set_ylabel("density")
    ax.set_title(r"(7) $\cot\theta$ residual and pull"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    out["z0_resolution"] = res


# --------------------------------------------------------------------------
# (8) Hough example panels
# --------------------------------------------------------------------------
B_FIELD_T = 3.8
# phi_pos = phi0 - C_BEND * r[cm] * kappa[1/GeV]. The DIRECTION turns twice as
# fast as the position azimuth, which is what makes a cluster a SEGMENT.
C_BEND = 0.29979246 * B_FIELD_T / 2.0 / 100.0
HOUGH_PT_POINTS = (1.0, 2.0, 5.0, 10.0, 20.0)
HOUGH_ETA_POINTS = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4)
LAYER_COLOR = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4", 4: "#2ca02c"}  # red/orange/blue/green
BAND_ALPHA = (0.95, 0.40, 0.10)      # |d| < 1 sigma, 1-3 sigma, > 3 sigma
CONE_DPHI, CONE_DETA = 0.20, 0.20
SECTOR_DPHI = 2.0 * np.pi / 9.0
Z0_SLICE_NSIG = 2.0
SECTOR_DRAW_CAP = 3000


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _half_turn(phi_pos, phi_dir):
    """asin(C_BEND * r * kappa): the half turn from the beamline out to the hit.

    EXACT, not the small-angle form. For a helix from the origin the position
    azimuth lags phi0 by asin(c*r*kappa) and the DIRECTION by twice that, so
    phi_p - phi_d = asin(c*r*kappa) and kappa = sin(phi_p - phi_d) / (c*r).

    HISTORY, because the failure was silent and cost a full round of wrong
    results: before the module-flip fix, `globalClusterPhi` was propagated
    without accounting for modules being physically flipped within a ladder (a
    real feature of the detector, not a code invention). The stored direction
    then pointed inward on flipped modules, so phi_p - phi_d came out as
    pi - asin(...) on some clusters and as +/-asin(...) on others -- correct
    magnitude, scrambled sign. sin() absorbs the pi but NOT the sign, so kappa
    came out with a random sign per cluster and any segment-shortening built on
    it was meaningless.

    Post-fix on spix_postq_500.root: median |phi_p - phi_d| = 0.098 with ZERO
    clusters beyond 2.0 rad (the pi branch is gone), |kappa|*pT = 1.0035, and
    the per-cluster sign agrees with truth 98.1% on clean >=3-layer topologies
    (98.3/98.3/98.1/97.5 on L1-L4).
    """
    return _wrap(phi_pos - phi_dir)


def _hough_bands(centre, sigma, slope, intercept, xlo, xhi, has_angle):
    """Split y = intercept + slope*x into |x-centre| < 1s, 1-3s, >3s.

    Returns three (N,2,2) arrays for LineCollection. A cluster whose sensor
    reports NO angle has centre=sigma=0, which makes bands 1 and 2 empty and
    band 3 span the full range in two pieces -- i.e. it constrains nothing and
    is drawn faintest across the whole line. That is the correct behaviour and
    needs no special case.
    """
    ok = has_angle & np.isfinite(sigma) & (sigma > 0) & np.isfinite(centre)
    c = np.where(ok, centre, 0.0)
    s = np.where(ok, sigma, 0.0)

    def seg(a, b):
        a, b = np.clip(a, xlo, xhi), np.clip(b, xlo, xhi)
        m = b > a
        if not m.any():
            return None
        x0, x1 = a[m], b[m]
        return np.stack([np.stack([x0, intercept[m] + slope[m] * x0], -1),
                         np.stack([x1, intercept[m] + slope[m] * x1], -1)], 1)

    full = np.full_like(c, xlo), np.full_like(c, xhi)
    spans = [[(c - s, c + s)],
             [(c - 3 * s, c - s), (c + s, c + 3 * s)],
             [(full[0], c - 3 * s), (c + 3 * s, full[1])]]
    out = []
    for band in spans:
        parts = [p for p in (seg(a, b) for a, b in band) if p is not None]
        out.append(np.concatenate(parts) if parts else np.empty((0, 2, 2)))
    return out


def _draw_hough(ax, sel, K, xlo, xhi, mode, truth, rasterize, shade=True):
    """One Hough panel. mode='rphi' -> (kappa, phi0); mode='rz' -> (cotTheta, z0).

    shade=False draws the classic POSITION-ONLY Hough line: full range, one shade
    per layer, no angle information used. That is the honest fallback while the
    direction columns carry a per-module orientation convention (see the module
    docstring of study 8).
    """
    from matplotlib.collections import LineCollection
    r = K["globalR"][sel]
    if not shade and mode == "rphi":
        phi_p = K["globalPhi"][sel]
        slope, intercept = C_BEND * r, _wrap(phi_p - truth["phi0"])
        for L in LAYERS:
            m = K["layer"][sel] == L
            if not m.any():
                continue
            x0 = np.full(int(m.sum()), xlo); x1 = np.full(int(m.sum()), xhi)
            segs = np.stack([np.stack([x0, intercept[m] + slope[m] * x0], -1),
                             np.stack([x1, intercept[m] + slope[m] * x1], -1)], 1)
            ax.add_collection(LineCollection(segs, colors=LAYER_COLOR[L], linewidths=0.6,
                                             alpha=0.45, rasterized=rasterize))
        ax.set_xlim(xlo, xhi)
        return
    if mode == "rphi":
        phi_p = K["globalPhi"][sel]
        slope = C_BEND * r
        intercept = _wrap(phi_p - truth["phi0"])          # plot relative to truth
        half = _half_turn(phi_p, K["gClPhi"][sel])
        centre = np.sin(half) / np.maximum(slope, 1e-12)
        sigma = K["gSigPhi"][sel] * np.abs(np.cos(half)) / np.maximum(slope, 1e-12)
        has = K["gClPhi"][sel] > SENTINEL
    else:
        slope = -r
        intercept = K["globalZ"][sel]
        centre = K["gClCotTheta"][sel]
        sigma = K["gSigCotTheta"][sel]
        has = K["gClCotTheta"][sel] > SENTINEL
    for L in LAYERS:
        m = K["layer"][sel] == L
        if not m.any():
            continue
        bands = _hough_bands(centre[m], sigma[m], slope[m], intercept[m], xlo, xhi, has[m])
        for b, segs in enumerate(bands):
            if len(segs):
                ax.add_collection(LineCollection(
                    segs, colors=LAYER_COLOR[L], linewidths=0.6, alpha=BAND_ALPHA[b],
                    rasterized=rasterize))
    ax.set_xlim(xlo, xhi)



# --------------------------------------------------------------------------
# (10) combination sweep across activeSP configurations
# --------------------------------------------------------------------------
# Two-sided per-axis Gaussian quantiles, the SAME convention as CONES above
# (q68 -> 1.0, q95 -> 1.96): a "qX cone" means each axis is within k sigma where
# k = Phi^-1((1+X)/2). Joint containment of a 2D box is the square of that, so
# these are per-axis levels and not the probability of keeping the true hit.
SWEEP_Q = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
SWEEP_COMBO_EDGES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 1 << 30],
                             dtype=float)

# Report order: grouped by how many layers are INSTRUMENTED, and within a group by
# which ones. Grouping this way is the point -- the interesting comparisons are
# between builds of equal cost in ASICs, where the only difference is WHICH layers
# were chosen, and those differ by more than an order of magnitude in search cost.
SWEEP_ORDER = ["AIII", "IAII", "IIAI", "IIIA",
               "AAII", "AIAI", "AIIA", "IAAI", "IAIA", "IIAA",
               "AAAI", "AAIA", "AIAA", "IAAA",
               "AAAA"]
# CONE WIDTHS the design table is evaluated at. These are the realistic knob: the
# cone quantile IS how wide the search window is, so it sets both how many true
# hits enter the refit and how much combinatorics comes with them. One total-work
# number cannot express that trade, so every cone gets its own column.
#
# NOMINAL, NOT MEASURED. k = Phi^-1((1+q)/2) assumes unit-width pulls, and ours are
# 0.81 to 1.52 depending on layer and visit depth (see q_acceptance.py), so a
# "q99.99 cone" does NOT deliver 99.99% true-hit containment. That is exactly why
# the table also carries MEASURED containment beside the cost.
SWEEP_TABLE_CONES = [0.99, 0.999, 0.9999]


def _variant_crossings(files, suffix):
    """Crossings for ONE activeSP variant. Only the columns the sweep needs."""
    hit = f"L1TSmartPixelsRefitHitDigiRefit{suffix}"
    cols = ["trackIdx", "layer", "detId", "projLocalX", "projLocalY", "projSigX", "projSigY",
            "hitAccepted"]
    H = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{hit}_{c}" for c in cols])
    ncross = ak.to_numpy(ak.num(H[f"{hit}_layer"]))
    X = {c: ak.to_numpy(ak.flatten(H[f"{hit}_{c}"])) for c in cols}
    ev = np.repeat(np.arange(len(ncross)), ncross)
    X["event"] = ev
    # The matched TP is per TRACK and per VARIANT (each variant refits separately),
    # so it has to come from THIS variant's track table, not the one load() picked.
    tname = f"{hit.replace('RefitHit', 'Track')}_spixMatchedTpIdx"
    V = uproot.concatenate([f"{f}:Events" for f in files], filter_name=[tname])
    mtp = ak.to_numpy(ak.flatten(V[tname]))
    offm = np.concatenate([[0], np.cumsum(ak.to_numpy(ak.num(V[tname])))])
    X["trk_tpIdx"] = mtp[offm[ev] + X["trackIdx"].astype(np.int64)]
    return X


def _combinations_per_track(X, K, kvals):
    """For each k, the number of L1xL2xL3xL4 hit combinations each track must search.

    Counts are SUMMED over crossings within a layer before the product is taken:
    a track that clips two overlapping modules at one layer has a single candidate
    POOL there, so those crossings must add. Multiplying them would invent
    combinations that the refit never considers, since it takes one hit per layer.

    A layer with no candidate contributes a factor of 1, not 0. The refit skips it
    and carries on; treating it as 0 would erase the track from the distribution
    entirely and would bias the result toward the busy tracks.
    """
    xi, ci, _ = join_on_module(X, K)
    dx = K["localX"][ci] - X["projLocalX"][xi]
    dy = K["localY"][ci] - X["projLocalY"][xi]
    sx, sy = X["projSigX"][xi], X["projSigY"][xi]
    ok = (sx > 0) & (sy > 0) & (X["projLocalX"][xi] > SENTINEL)
    with np.errstate(divide="ignore", invalid="ignore"):
        nx = np.where(ok, np.abs(dx / sx), np.inf)
        ny = np.where(ok, np.abs(dy / sy), np.inf)

    tkey = X["event"].astype(np.int64) * (1 << 20) + X["trackIdx"].astype(np.int64)
    key2 = tkey * 8 + X["layer"].astype(np.int64)
    u2, inv2 = np.unique(key2, return_inverse=True)
    tk2 = u2 // 8
    ut, invt = np.unique(tk2, return_inverse=True)

    # TRUE-hit containment, the benefit side of widening the cone. A pair is the
    # true one when the cluster's dominant TP is the TP the track was matched to.
    # Denominator counts only crossings where such a cluster EXISTS on the module,
    # so a layer that simply had no true cluster does not count as an inefficiency.
    is_true = (K["tpIdx"][ci] >= 0) & (X["trk_tpIdx"][xi] >= 0) & \
              (K["tpIdx"][ci] == X["trk_tpIdx"][xi])
    have_true = np.zeros(len(X["layer"]), bool)
    have_true[xi[is_true]] = True
    n_have = int(have_true.sum())

    out, cont = {}, {}
    for k in kvals:
        m = (nx < k) & (ny < k)
        cnt = np.bincount(xi[m], minlength=len(X["layer"])).astype(np.float64)
        n_tl = np.bincount(inv2, weights=cnt, minlength=len(u2))
        prod = np.ones(len(ut), dtype=np.float64)
        np.multiply.at(prod, invt, np.maximum(n_tl, 1.0))
        out[k] = prod
        found = np.zeros(len(X["layer"]), bool)
        found[xi[is_true & m]] = True
        cont[k] = float((found & have_true).sum() / n_have) if n_have else float("nan")
    return out, len(ut), cont, n_have



def _write_sweep_table(res, sfx, out, n_ev):
    """Persistent activeSP design table: cost AND containment at each cone width.

    THE CONE WIDTH IS THE DESIGN KNOB, so it gets a column rather than a single
    chosen value. Widening the window admits more true hits into the refit and more
    combinatorics with them; a lone total-work number cannot express that trade and
    would let a build look cheap purely because it was quoted at a tight cone.
    Every cone therefore carries both its cost (TOTAL work) and what that cost
    bought (measured true-hit containment).

    CONTAINMENT IS MEASURED, NOT ASSUMED. The quantile is nominal: k =
    Phi^-1((1+q)/2) presumes unit-width pulls and ours run 0.81 to 1.52 depending
    on layer and visit depth, so a "q99.99 cone" does not deliver 99.99%. The
    denominator counts only crossings where a truth-matched cluster actually exists
    on the module, so a layer that had no true cluster is not scored as an
    inefficiency.

    Emitted on every run rather than pasted into a message: every number here moved
    at least once during development.
    """
    order = [s for s in SWEEP_ORDER if s in sfx] + [s for s in sfx if s not in SWEEP_ORDER]
    cones = res["table_cones"]
    ck = [f"{q:.4f}" for q in cones]
    ntrk_ref = max((res[s]["n_tracks"] for s in order), default=0)

    def qlab(q):
        return ("q%g" % (q * 100)).rstrip("0").rstrip(".") if q * 100 % 1 else "q%d" % (q * 100)

    labs = [qlab(q) for q in cones]
    trk_per_ev = ntrk_ref / max(n_ev, 1)
    W = 8
    blk = W * len(labs)
    grp = (f"{'':<10}  " + f"{'combinations / track':^{blk}}" + " | "
           + f"{'p99 (busy track)':^{blk}}" + " | " + f"{'containment':^{blk}}"
           + " | " + f"{'viability':^16}")
    sub = (f"{'cfg':<6}{'nSP':>4}  " + "".join(f"{l:>{W}}" for l in labs) + " | "
           + "".join(f"{l:>{W}}" for l in labs) + " | "
           + "".join(f"{l:>{W}}" for l in labs) + " | "
           + f"{'>=2hit':>8}{'>=3hit':>8}")
    hdr = grp + "\n" + sub
    lines = [
        "activeSP refit search cost vs CONE WIDTH",
        f"  sample        : {n_ev} events, PU200, {ntrk_ref:,} tracks per config",
        "  cone          : nominal per-axis two-sided Gaussian, k = Phi^-1((1+q)/2)",
        "                  " + " | ".join(
            f"{l} k={res['table_k_sigma'][i]:.3f}" for i, l in enumerate(labs)),
        f"                  {trk_per_ev:.0f} refit-able tracks per event",
        "  combinations  : PER TRACK, the number of (L1,L2,L3,L4) hit tuples a refit must",
        "                  test = product of the candidate counts over INSTRUMENTED layers.",
        "                  A layer with NO candidate contributes 1, not 0 -- the refit skips",
        "                  it and carries on. Worked: 2,1,0,2 -> 2*1*1*2 = 4;  3,2,1,1 -> 6.",
        "                  (pinned by tests/test_combination_counting.py)",
        "  cmb/trk <cone>: MEAN combinations per track refit at that cone width.",
        f"                  Multiply by {trk_per_ev:.0f} for per-event throughput.",
        "  p99 <cone>    : 99th-percentile track, i.e. the busy-track cost.",
        "  cont <cone>   : MEASURED fraction of crossings whose true cluster falls in the",
        "                  cone. The nominal quantile is NOT the achieved containment --",
        "                  pull widths are 0.81-1.52, so this is measured, not assumed.",
        "  >=2hit/>=3hit : fraction of tracks the refit gave that many hits. 0% for every",
        "                  two-layer build is structural, not performance.",
        "",
        hdr, "-" * len(sub)]
    group = {1: "1 instrumented layer", 2: "2 instrumented layers",
             3: "3 instrumented layers", 4: "4 instrumented layers"}
    last_n = None
    for st in order:
        n = st.count("A")
        if n != last_n:
            lines.append(f"--- {group.get(n, str(n))} ---")
            last_n = n
        r, f = res[st], res[st]["frac_tracks_with_hits"]
        bc = r["by_cone"]
        lines.append(
            f"{st:<6}{n:>4}  "
            + "".join(f"{bc[c]['mean']:>{W}.2f}" for c in ck) + " | "
            + "".join(f"{bc[c]['p99']:>{W}.0f}" for c in ck) + " | "
            + "".join(f"{100 * bc[c]['true_hit_containment']:>{W - 1}.1f}%" for c in ck)
            + " | " + f"{100 * f['ge2']:>7.0f}%{100 * f['ge3']:>7.0f}%")
    txt = "\n".join(lines)
    tp = os.path.join(out["_outdir"], "spix_activesp_table.txt")
    with open(tp, "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)
    print(f"\n   (10) wrote {tp}")

    cp = os.path.join(out["_outdir"], "spix_activesp_table.csv")
    with open(cp, "w") as fh:
        fh.write("activeSP,n_sp_layers,cone_q,k_sigma,combos_per_track_mean,"
                 "combos_per_track_median,combos_per_track_p99,combos_per_track_max,"
                 "combos_per_event,total_work_raw,true_hit_containment,"
                 "mean_accepted_hits,frac_ge2hit,frac_ge3hit\n")
        for st in order:
            r, f = res[st], res[st]["frac_tracks_with_hits"]
            for c, d in sorted(r["by_cone"].items(), key=lambda kv: float(kv[0])):
                fh.write(f"{st},{st.count('A')},{float(c):.4f},{d['k_sigma']:.4f},"
                         f"{d['mean']:.4f},{d['median']:.0f},{d['p99']:.0f},{d['max']:.0f},"
                         f"{d['combos_per_event']:.2f},{d['total_work']:.0f},"
                         f"{d['true_hit_containment']:.5f},"
                         f"{r['mean_accepted_hits']:.4f},{f['ge2']:.4f},{f['ge3']:.4f}\n")
    print(f"   (10) wrote {cp}")
    res["_table"] = os.path.basename(tp)
    res["_csv"] = os.path.basename(cp)


def study_combination_sweep(X, K, P, ax_row, out):
    """(10) Search-space size vs cone quantile, for every non-trivial activeSP mask.

    THE QUESTION THIS ANSWERS. Widening the cone buys hit-finding efficiency and
    costs combinatorics, and adding smart-pixel layers tightens the cone for free.
    Neither trade is readable from a covariance: the number that matters is how
    many L1xL2xL3xL4 hit combinations a refit must actually search, and how its
    DISTRIBUTION over tracks moves. A mean is not enough here, because the cost of
    the tail is what sets the hardware budget -- hence a 2D histogram per
    configuration rather than a curve.

    EACH MASK IS A DETECTOR BUILD, NOT AN ALGORITHM SETTING. activeSP is which IT
    layers are INSTRUMENTED with smart pixels. Smart pixels emit cluster data at L1
    latency; a conventional pixel layer is not read out until after an L1 accept. So
    an uninstrumented layer contributes NOTHING to L1 track building of any form --
    not "position but no angle", nothing at all. Measured crossing counts by layer:
    AAAA gives 436/374/296/227, AAII gives 435/373/0/0, IIIA gives 0/0/0/219.

    That means a mask with fewer A's searching fewer layers is not an unfairness in
    the comparison, it IS the trade: instrument less, get less information AND less
    combinatorics. These 15 panels are a cost curve across candidate builds.

    WHICH layers matters as much as HOW MANY, and asymmetrically. L1 is both the
    busiest layer (seed-cone occupancy 1.98 vs 0.82 at L4) and the smallest radius,
    so a build instrumenting only inner layers enters the densest region carrying
    the full untightened OT-seed cone (~308 um). An outer-first build gets
    progressive tightening before it arrives there. Two builds with the same number
    of A's are not interchangeable.

    COMBINATIONS ARE THE COST SIDE ONLY. A build can look cheap because it cannot do
    the job. The third panel therefore carries the viability counterpart -- what
    fraction of tracks even collect enough hits to refit -- and resolution/purity
    per build is still missing and must not be inferred from these histograms.

    The cone widths corroborate the outsideIn ordering: AAAA tightens monotonically
    133.9 / 84.8 / 61.9 / 35.2 um from L4 in to L1, while AAII visits L2 FIRST
    (235.8 um, no update yet) and then L1 (32.6 um). IIIA's only layer, L4, sits at
    133.9 um -- identical to AAAA's L4, which is the check that a first-visited
    layer receives no update in either configuration.
    """
    files = out.get("_inputs")
    if not files:
        return
    with uproot.open(f"{files[0]}:Events") as t:
        keys = set(t.keys())
    sfx = sorted({m.group(1) for m in
                  (re.match(r"^L1TSmartPixelsRefitHitDigiRefit([AI]{4})_", k) for k in keys) if m})
    if len(sfx) < 2:
        print(f"   (10) SKIPPED: input has {len(sfx)} activeSP variant(s); the sweep needs the "
              "15 non-trivial masks. Regenerate with a repeated --variant digiRefit:XXXX.")
        for a in ax_row:
            a.axis("off")
        return

    kvals = [NormalDist().inv_cdf(0.5 * (1.0 + q)) for q in SWEEP_Q]
    tabq = [q for q in SWEEP_TABLE_CONES]
    ktab = [NormalDist().inv_cdf(0.5 * (1.0 + q)) for q in tabq]
    # one pass over the join covers plot cones and the wider design cones
    allq = list(SWEEP_Q) + [q for q in tabq if q not in SWEEP_Q]
    allk = kvals + [k for q, k in zip(tabq, ktab) if q not in SWEEP_Q]
    res, per_cfg = {"quantiles": SWEEP_Q, "k_sigma": [float(k) for k in kvals],
                    "table_cones": tabq,
                    "table_k_sigma": [float(k) for k in ktab]}, {}
    n_ev_local = max(int(out.get("n_events", 0)), 1)
    for s in sfx:
        Xv = _variant_crossings(files, s)
        combos, ntrk, cont, n_have = _combinations_per_track(Xv, K, allk)
        per_cfg[s] = combos
        # Viability: an uninstrumented layer yields no L1 hit at all, so a build can
        # simply run out of hits. Counted from the refit's OWN acceptance rather than
        # from cone occupancy, since that is what it actually kept.
        tk = Xv["event"].astype(np.int64) * (1 << 20) + Xv["trackIdx"].astype(np.int64)
        ut2, inv3 = np.unique(tk, return_inverse=True)
        nacc = np.bincount(inv3, weights=(Xv["hitAccepted"] > 0).astype(float),
                           minlength=len(ut2))
        frac = {f"ge{j}": float((nacc >= j).mean()) for j in (1, 2, 3, 4)}
        res[s] = {"n_tracks": int(ntrk), "frac_tracks_with_hits": frac,
                  "mean_accepted_hits": float(nacc.mean()),
                  "n_crossings_with_true_cluster": int(n_have),
                  # keyed by CONE quantile: each is a different design point
                  "by_cone": {f"{q:.4f}": {
                      "k_sigma": float(k),
                      # RAW SUM over whatever sample ran -- not interpretable on its
                      # own, kept only so the normalised numbers can be rederived.
                      "total_work": float(combos[k].sum()),
                      "combos_per_event": float(combos[k].sum() / max(n_ev_local, 1)),
                      "mean": float(combos[k].mean()),
                      "median": float(np.median(combos[k])),
                      "p99": float(np.percentile(combos[k], 99)),
                      "max": float(combos[k].max()),
                      "true_hit_containment": float(cont[k]),
                  } for q, k in zip(allq, allk)},
                  "median": [float(np.median(combos[k])) for k in kvals],
                  "mean": [float(combos[k].mean()) for k in kvals],
                  "p99": [float(np.percentile(combos[k], 99)) for k in kvals],
                  # p99 ALONE INVERTS THE RANKING and must not be quoted by itself.
                  # At q99 AAAA beats AIII on p99 (12 vs 15) and loses badly on the
                  # deep tail (672 vs 48), because multiplying four layers lets rare
                  # busy tracks produce enormous products. Throughput budget and
                  # worst-case budget disagree here, so both are reported.
                  "p999": [float(np.percentile(combos[k], 99.9)) for k in kvals],
                  "p9999": [float(np.percentile(combos[k], 99.99)) for k in kvals],
                  "max": [float(combos[k].max()) for k in kvals],
                  "total_work": [float(combos[k].sum()) for k in kvals]}
        print(f"   (10) {s}: median combos "
              + "/".join(f"{v:.0f}" for v in res[s]["median"])
              + f"   q99: p99={res[s]['p99'][-1]:.0f} p99.9={res[s]['p999'][-1]:.0f} "
                f"p99.99={res[s]['p9999'][-1]:.0f} max={res[s]['max'][-1]:.0f} "
                f"tot={res[s]['total_work'][-1]:,.0f}"
              + f"   <hits>={res[s]['mean_accepted_hits']:.2f}"
              + f"  >=2hit {100 * frac['ge2']:.0f}%  >=3hit {100 * frac['ge3']:.0f}%")

    # ---- per-configuration 2D histograms -----------------------------------
    ncol = 4
    nrow = int(np.ceil(len(sfx) / ncol))
    f2, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow), squeeze=False)
    vmax = 1
    grids = {}
    for s in sfx:
        g = np.zeros((len(SWEEP_COMBO_EDGES) - 1, len(kvals)))
        for j, k in enumerate(kvals):
            g[:, j] = np.histogram(per_cfg[s][k], bins=SWEEP_COMBO_EDGES)[0]
        grids[s] = g
        vmax = max(vmax, g.max())
    # Clip the shared y-range to the highest OCCUPIED bin. The top edge exists to
    # catch an unbounded tail, but drawn literally it hands most of a log axis to
    # empty space and makes every panel look identical.
    occupied = max((np.flatnonzero(g.sum(axis=1)).max() for g in grids.values()
                    if g.sum() > 0), default=len(SWEEP_COMBO_EDGES) - 2)
    ytop = SWEEP_COMBO_EDGES[min(occupied + 1, len(SWEEP_COMBO_EDGES) - 1)]
    for i, s in enumerate(sfx):
        a = axs[i // ncol][i % ncol]
        mesh = a.pcolormesh(np.arange(len(kvals) + 1), SWEEP_COMBO_EDGES, grids[s],
                            norm=LogNorm(vmin=1, vmax=vmax), cmap="viridis")
        a.set_yscale("log")
        a.set_ylim(1, ytop)
        a.set_xticks(np.arange(len(kvals)) + 0.5)
        a.set_xticklabels([f"{int(q * 100)}" for q in SWEEP_Q], fontsize=7)
        a.set_title(f"activeSP {s}  ({s.count('A')} SP layer"
                    f"{'s' if s.count('A') != 1 else ''})", fontsize=9)
        a.set_xlabel("cone quantile qX", fontsize=8)
        a.set_ylabel("combinations / track", fontsize=8)
        f2.colorbar(mesh, ax=a, label="tracks")
    for i in range(len(sfx), nrow * ncol):
        axs[i // ncol][i % ncol].axis("off")
    f2.suptitle("(10) refit search space: L1xL2xL3xL4 combinations vs cone quantile, "
                "per activeSP configuration", y=1.002)
    f2.tight_layout()
    p2 = os.path.join(out["_outdir"], "spix_combination_sweep.png")
    f2.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(f2)
    print(f"   (10) wrote {p2}")
    res["_figure"] = os.path.basename(p2)

    # ---- summary panels in the omnibus figure -------------------------------
    order = [s for s in SWEEP_ORDER if s in sfx] + [s for s in sfx if s not in SWEEP_ORDER]
    cmap = plt.get_cmap("turbo")
    for a, stat, lab in ((ax_row[0], "total_work", "TOTAL combinations"),
                         (ax_row[1], "max", "worst-case (max)")):
        for i, s in enumerate(order):
            a.plot([q * 100 for q in SWEEP_Q], res[s][stat], marker="o", ms=3,
                   color=cmap(i / max(len(order) - 1, 1)), label=s)
        a.set_yscale("log"); a.set_xlabel("cone quantile qX")
        a.set_ylabel(f"{lab} combinations / track")
        a.set_title(f"(10) {lab} search space vs cone")
        a.grid(alpha=.3)
        a.legend(fontsize=5, ncol=3)
    a = ax_row[2]
    xs = np.arange(len(order))
    for j, mk in ((2, "s"), (3, "^"), (4, "v")):
        a.plot(xs, [res[s]["frac_tracks_with_hits"][f"ge{j}"] for s in order],
               marker=mk, ms=4, label=f"$\\geq${j} hits")
    a.set_xticks(xs); a.set_xticklabels(order, rotation=90, fontsize=6)
    a.set_ylabel("fraction of tracks"); a.set_ylim(0, 1.02)
    a.set_title("(10) can this build even refit?")
    a.grid(alpha=.3); a.legend(fontsize=7)
    _write_sweep_table(res, sfx, out, out.get("n_events", -1))
    out["combination_sweep"] = res


def study_hough_examples(X, K, P, ax_row, out):
    """(8) worked Hough transforms for example TrackingParticles.

    WHAT IS DRAWN. For each (pT, eta) cell one real TP is picked from the file
    and two populations are shown around it: a CONE (legible, shows the signal
    structure) and the full phi SECTOR one processing node would see (honest,
    shows the PU200 background it sits in). Each gets three panels: the r-z
    plane, the r-phi plane, and the r-phi plane again after z0 slicing.

    WHY A CLUSTER IS A SEGMENT, NOT A LINE. Position and direction turn at
    different rates -- phi_p = phi0 - c*r*kappa but phi_d = phi0 - 2*c*r*kappa --
    so a cluster's own angle fixes kappa_hat = (phi_p - phi_d)/(c*r) with
    sigma = sigma(phi_d)/(c*r). The Hough line is drawn dark within 1 sigma of
    that, mid to 3 sigma, faint beyond. Layers: L1 red, L2 orange, L3 blue,
    L4 green.

    THE r-z PLANE IS THE SAME CONSTRUCTION. z0 = z - r*cot(theta), so a cluster
    is a segment of slope -r there too, and the slope encodes the layer: L1's
    short lever arm makes a nearly flat, tightly-determined z0, L4's a steep and
    loose one. That is the whole reason z0 slicing works, drawn rather than
    asserted.

    ETA COVERAGE IS A RESULT, NOT A PLOTTING PROBLEM. TBPX is a barrel: measured
    on this file, the fraction of TPs lighting >=3 layers is 0.87 at |eta| 0.2-1.0
    but 0.21 at 1.4-1.8 and 0.03 above 1.8. That is the acceptance limit of
    barrel-only seeding (|eta| <~ 1.4). NOTE this is a BLOCKED DEPENDENCY, not a
    scope decision: forward coverage is wanted, but there is no PixelAV angle
    parametrisation for the disc sensors in their B-field configuration (a disc
    sits perpendicular to B where a barrel sits parallel), so a disc cluster has
    no usable angle yet. See doc/SmartPixelsSeedingAndFitting.md section 1.

    THE TP PER CELL IS THE TYPICAL ONE, NOT THE BEST ONE -- the median-ranked
    candidate by (n_layers, n_clusters). An earlier version took the argmax, which
    cherry-picked the rare 4-layer survivor at high |eta| and made the panels look
    as though seeding worked there. It also admitted 1-layer TPs to the pool, since
    requiring >=2 layers pre-selects away the very loss these panels exist to show.
    Each page states its pool's acceptance (fraction reaching >=3 and 4 layers)
    independently of which TP was drawn, so the acceptance claim does not rest on
    the single example.

    THE ANGLE SIGMAS ARE THE STORED ONES and sigGlobalClusterCotTheta is known to
    be ~21% optimistic (study 7), so the bands here are correspondingly tight.
    They are also drawn from a truth angle that neglects multiple scattering --
    see the CIRCULAR note in study (7).
    """
    from matplotlib.backends.backend_pdf import PdfPages
    need = ("globalR", "globalZ", "globalPhi", "gClPhi", "gClCotTheta",
            "gSigPhi", "gSigCotTheta", "tpGClPhi", "tpGClCotTheta")
    if not all(k in K for k in need):
        print("   (8) SKIPPED: cluster table lacks the global-frame columns")
        out["hough_examples"] = {"skipped": "no global-frame columns"}
        for a in ax_row:
            a.axis("off")
        return

    # ---- per-TP aggregates, fully vectorised -------------------------------
    ok = (K["tpIdx"] >= 0) & (K["tpGClPhi"] > SENTINEL) & (K["tpGClCotTheta"] > SENTINEL)
    key = K["event"].astype(np.int64) * (1 << 20) + K["tpIdx"].astype(np.int64)
    ukey, inv = np.unique(key[ok], return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    laymask = np.zeros(len(ukey), dtype=np.int64)
    np.bitwise_or.at(laymask, inv, (1 << K["layer"][ok].astype(np.int64)))
    nlay = sum(((laymask >> L) & 1) for L in LAYERS)

    r_ok = K["globalR"][ok]
    half = _half_turn(K["globalPhi"][ok], K["tpGClPhi"][ok])
    kap = np.sin(half) / (C_BEND * r_ok)
    phi0 = _wrap(K["globalPhi"][ok] + half)
    # NOTE z0 here is NOT trustworthy: globalClusterCotTheta is -localCotBeta,
    # not dz/dr (see doc section 4a retraction). Kept only so the r-z panels can
    # be drawn as BLOCKED rather than silently wrong.
    z0 = K["globalZ"][ok] - r_ok * K["tpGClCotTheta"][ok]

    def mean_by(v):
        return np.bincount(inv, weights=v) / cnt

    def median_by(v):
        """Per-TP median. A MEAN is unusable for kappa: a cluster at small r has
        kappa bounded only by 1/(C*r) ~ 60, so a single bad one (secondary,
        merged cluster, broken pi convention) drags the mean by an order of
        magnitude -- which is exactly what put the truth marker at kappa = -2.7
        for a 2.5 GeV track before this was fixed."""
        order = np.lexsort((v, inv))
        vs, gs = v[order], inv[order]
        g = np.arange(len(cnt))
        lo = np.searchsorted(gs, g, "left")
        return vs[lo + (np.searchsorted(gs, g, "right") - lo) // 2]
    tp_pt = median_by(K["tpPt"][ok])
    tp_cot = median_by(K["tpGClCotTheta"][ok])
    tp_eta = np.arcsinh(tp_cot)
    tp_z0 = median_by(z0)
    # kappa's SIGN cannot be taken from the direction columns: their orientation
    # convention varies per module, so (phi_p - phi_d) is sometimes the small bend
    # and sometimes pi minus it, with either sign. |kappa| = 1/tpPt is exact from
    # truth; take the sign from how the POSITION azimuth turns with radius, since
    # phi_p = phi0 - asin(C*r*kappa) means dphi_p/dr < 0 for kappa > 0.
    rc, phi_c = K["globalR"][ok], K["globalPhi"][ok]
    ref = np.arctan2(np.bincount(inv, weights=np.sin(phi_c)),
                     np.bincount(inv, weights=np.cos(phi_c)))     # circular mean
    phi_rel = _wrap(phi_c - ref[inv])                             # wrap-safe residual
    cov = (np.bincount(inv, weights=rc * phi_rel)
           - np.bincount(inv, weights=rc) * np.bincount(inv, weights=phi_rel) / cnt)
    tp_kap = np.where(cov > 0, -1.0, 1.0) / np.maximum(tp_pt, 1e-6)
    # phi0 is circular: average via unit vectors
    tp_phi0 = np.arctan2(mean_by(np.sin(phi0)), mean_by(np.cos(phi0)))
    tp_ev = (ukey >> 20).astype(np.int64)

    clu_eta = np.arcsinh(K["globalZ"] / np.maximum(K["globalR"], 1e-6))
    res, pages = {"cells": {}}, []

    for pt_t in HOUGH_PT_POINTS:
        for eta_t in HOUGH_ETA_POINTS:
            cell = f"pt{pt_t:g}_eta{eta_t:g}"
            # nlay >= 1, NOT >= 2: requiring two layers already pre-selects away
            # the acceptance loss these panels exist to show.
            cand = (np.abs(np.abs(tp_eta) - eta_t) < 0.15) & \
                   (np.abs(np.log(np.maximum(tp_pt, 1e-6) / pt_t)) < np.log(1.3)) & (nlay >= 1)
            if not cand.any():
                res["cells"][cell] = {"found": False,
                                      "reason": "no TP within (dEta<0.15, pT within 30%)"}
                pages.append((cell, None))
                continue
            # TYPICAL, not best. Taking the argmax over (nlay, cnt) cherry-picks the
            # rare 4-layer survivor at high |eta| and hides the barrel acceptance
            # collapse -- the panel then shows an algorithm working on a track that
            # almost no track in that cell resembles. The median-ranked candidate is
            # what a track at this (pT, eta) actually looks like.
            order = np.lexsort((cnt[cand], nlay[cand]))
            idx = np.flatnonzero(cand)[order[len(order) // 2]]
            pool_ge3 = float((nlay[cand] >= 3).mean())
            pool_eq4 = float((nlay[cand] == 4).mean())
            truth = {"phi0": tp_phi0[idx], "kap": tp_kap[idx],
                     "z0": tp_z0[idx], "cot": tp_cot[idx]}
            ev = tp_ev[idx]
            same_ev = K["event"] == ev
            dphi = _wrap(K["globalPhi"] - tp_phi0[idx])
            cone = same_ev & (np.abs(dphi) < CONE_DPHI) & \
                   (np.abs(clu_eta - tp_eta[idx]) < CONE_DETA)
            sect = same_ev & (np.abs(dphi) < 0.5 * SECTOR_DPHI)
            if sect.sum() > SECTOR_DRAW_CAP:      # keep the PDF finite
                keep = np.zeros(sect.sum(), bool)
                keep[np.random.default_rng(0).choice(sect.sum(), SECTOR_DRAW_CAP, False)] = True
                si = np.flatnonzero(sect); sect = np.zeros_like(sect); sect[si[keep]] = True
            res["cells"][cell] = {
                "found": True, "event": int(ev), "tpIdx": int(ukey[idx] & ((1 << 20) - 1)),
                "tp_pt": float(tp_pt[idx]), "tp_eta": float(tp_eta[idx]),
                "n_layers": int(nlay[idx]), "n_clusters_tp": int(cnt[idx]),
                "n_cone": int(cone.sum()), "n_sector": int(sect.sum()),
                "n_candidate_tps": int(cand.sum()),
                "selection": "median-ranked by (n_layers, n_clusters) -- TYPICAL, not best",
                # the acceptance statement for this cell, independent of which TP
                # happened to be drawn
                "pool_frac_ge3_layers": pool_ge3, "pool_frac_eq4_layers": pool_eq4}
            pages.append((cell, (idx, truth, cone, sect)))

    # ---- render -------------------------------------------------------------
    pdf_path = os.path.join(out["_outdir"], "spix_hough_examples.pdf")
    with PdfPages(pdf_path) as pdf:
        for cell, payload in pages:
            fig, axs = plt.subplots(2, 3, figsize=(16.5, 9))
            if payload is None:
                for a in axs.ravel():
                    a.axis("off")
                axs[0][1].text(0.5, 0.5, f"{cell}\n\nno TrackingParticle found\n"
                               "barrel-only acceptance ends near |eta| 1.4",
                               ha="center", va="center", fontsize=13)
            else:
                idx, truth, cone, sect = payload
                kmax = max(0.6, 1.6 * abs(truth["kap"]))
                for row, (sel, nm, rast) in enumerate(
                        ((cone, "cone", False), (sect, "phi sector", True))):
                    zpred = K["globalZ"][sel] - K["globalR"][sel] * K["gClCotTheta"][sel]
                    zsig = K["globalR"][sel] * np.maximum(K["gSigCotTheta"][sel], 1e-9)
                    sel_sliced = np.zeros_like(sel)
                    sel_sliced[np.flatnonzero(sel)[
                        np.abs(zpred - truth["z0"]) < Z0_SLICE_NSIG * zsig]] = True

                    a = axs[row][0]
                    _draw_hough(a, sel, K, truth["cot"] - 0.5, truth["cot"] + 0.5, "rz", truth, rast)
                    a.axhline(truth["z0"], color="k", lw=0.8, ls="--")
                    a.plot(truth["cot"], truth["z0"], "k*", ms=11, zorder=5)
                    a.set_ylim(truth["z0"] - 15, truth["z0"] + 15)
                    a.set_xlabel(r"$\cot\theta$"); a.set_ylabel(r"$z_0$ [cm]")
                    a.set_title(f"{nm}: r-z plane ({int(sel.sum())} clusters)", fontsize=9)

                    for col, (s, lab) in enumerate(((sel, "all"),
                                                    (sel_sliced, "in $z_0$ slice")), 1):
                        a = axs[row][col]
                        _draw_hough(a, s, K, -kmax, kmax, "rphi", truth, rast)
                        if nlay[idx] >= 2:
                            a.plot(truth["kap"], 0.0, "k*", ms=11, zorder=5)
                            a.axvline(truth["kap"], color="k", lw=0.6, ls=":")
                        else:
                            # one layer: the curvature SIGN comes from how the position
                            # azimuth turns with radius, which a single cluster cannot
                            # give. Show |kappa| = 1/pT on both sides instead of
                            # planting a star at an arbitrary sign.
                            for sgn in (-1.0, 1.0):
                                a.axvline(sgn * abs(truth["kap"]), color="k", lw=0.6, ls=":")
                        a.set_ylim(-0.30, 0.30)
                        a.set_xlabel(r"$q/p_T$ [GeV$^{-1}$]")
                        a.set_ylabel(r"$\phi_0 - \phi_0^{\rm true}$ [rad]")
                        a.set_title(f"{nm}: r-$\\phi$, {lab} ({int(s.sum())})", fontsize=9)
                    for a in axs[row]:
                        a.grid(alpha=.25)
                c = res["cells"][cell]
                fig.suptitle(f"{cell}   |   TYPICAL TP: $p_T$={c['tp_pt']:.2f} GeV, "
                             f"$\\eta$={c['tp_eta']:+.2f}, {c['n_layers']} TBPX layers   |   "
                             f"pool of {c['n_candidate_tps']}: "
                             f"{c['pool_frac_ge3_layers']*100:.0f}% reach $\\geq$3 layers, "
                             f"{c['pool_frac_eq4_layers']*100:.0f}% reach 4   |   "
                             f"L1 red, L2 orange, L3 blue, L4 green; "
                             f"shade = 1/3$\\sigma$ of the cluster's own angle", fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig, dpi=110)
            plt.close(fig)
    res["pdf"] = pdf_path
    res["n_pages"] = len(pages)
    res["n_cells_found"] = sum(1 for v in res["cells"].values() if v.get("found"))
    print(f"   (8) wrote {pdf_path} ({len(pages)} cells, "
          f"{res['n_cells_found']} with a TP)")

    # inline: one representative cell, before and after z0 slicing
    shown = next((c for c in ("pt2_eta0.4", "pt2_eta0", "pt5_eta0.4")
                  if res["cells"].get(c, {}).get("found")), None)
    if shown:
        idx, truth, cone, sect = dict(pages)[shown]
        kmax = max(0.6, 1.6 * abs(truth["kap"]))
        zp = K["globalZ"][cone] - K["globalR"][cone] * K["gClCotTheta"][cone]
        zs = K["globalR"][cone] * np.maximum(K["gSigCotTheta"][cone], 1e-9)
        sl = np.zeros_like(cone)
        sl[np.flatnonzero(cone)[np.abs(zp - truth["z0"]) < Z0_SLICE_NSIG * zs]] = True
        for a, (s, lab, rast) in zip(ax_row, ((cone, "cone, all", False),
                                              (sl, "cone, in $z_0$ slice", False))):
            _draw_hough(a, s, K, -kmax, kmax, "rphi", truth, rast)
            a.plot(truth["kap"], 0.0, "k*", ms=10, zorder=5)
            a.set_ylim(-0.30, 0.30); a.grid(alpha=.25)
            a.set_xlabel(r"$q/p_T$ [GeV$^{-1}$]"); a.set_ylabel(r"$\phi_0-\phi_0^{\rm true}$")
            a.set_title(f"(8) {shown} {lab} ({int(s.sum())})", fontsize=9)
    else:
        for a in ax_row:
            a.axis("off")
    out["hough_examples"] = res


STUDIES = [
    ("cone occupancy", study_cone_occupancy, 3),
    ("refit-order cone", study_refit_cone_occupancy, 3),
    ("cone containment + size", study_cone_containment, 2),
    ("angle discrimination", study_angle_discrimination, 2),
    ("charge readout gate", study_charge_gate, 2),
    ("unbiased containment", study_true_containment, 2),
    ("chi2 weight scan", study_chi2_weight_scan, 2),
    ("z0 resolution (seeding)", study_z0_resolution, 2),
    ("combination sweep (activeSP)", study_combination_sweep, 3),
    ("hough examples", study_hough_examples, 2),
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
           "n_clusters": int(len(K["layer"])),
           # side-artefact directory for studies that write their own files;
           # stripped before the JSON is dumped
           "_outdir": args.outdir,
           # studies that need to re-read the file for OTHER activeSP variants
           "_inputs": [f for p in args.inputs for f in (sorted(_glob.glob(p)) or [p])]}

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
    out.pop("_outdir", None)
    out.pop("_inputs", None)
    with open(js, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {png}\nwrote {js}")


if __name__ == "__main__":
    main()
