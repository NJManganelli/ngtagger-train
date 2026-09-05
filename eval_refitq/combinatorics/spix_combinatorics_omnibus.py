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
                "projCotAlpha", "projCotBeta",
                "recoLocalX", "recoLocalY", "selClusterIdx"]
    miss_h = [c for c in need_hit if f"{hit}_{c}" not in keys]
    miss_c = ([CLUSTER_TABLE] if not any(k.startswith(CLUSTER_TABLE + "_") for k in keys)
              else [c for c in ("layer", "detId", "localX", "localY", "charge",
                                "truthPt", "truthCotAlpha", "truthCotBeta", "truthTpIdx")
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
             "truthCotAlpha", "truthCotBeta", "sizeX", "sizeY", "truthTpIdx"]
    # sensor angle + CPE sigma, present once the cluster table reads SmartPixelsRecHit
    optional = {"recoCotAlpha": "clRecoCotAlpha", "recoCotBeta": "clRecoCotBeta",
                "sigAlpha": "clSigAlpha", "sigBeta": "clSigBeta",
                "sigX": "sigX", "sigY": "sigY"}
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
    # angle mismatch vs the track's expectation at this module
    P["dCotA"] = K["truthCotAlpha"][ci] - X["projSeedCotAlpha"][xi]
    P["dCotB"] = K["truthCotBeta"][ci] - X["projSeedCotBeta"][xi]
    P["ang_ok"] = (K["truthCotAlpha"][ci] > SENTINEL) & (X["projSeedCotAlpha"][xi] > SENTINEL)
    P["sel_gidx"] = selected_global_index(X, K)
    # over PAIRS: is this pair the crossing's selected cluster?
    P["is_selected"] = (P["sel_gidx"][P["xi"]] >= 0) & (P["ci"] == P["sel_gidx"][P["xi"]])
    return P


def in_cone(P, k):
    return P["good"] & (np.abs(P["nsigx"]) < k) & (np.abs(P["nsigy"]) < k)


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


def study_true_containment(X, K, P, ax_row, out):
    """(5) UNBIASED containment: of the clusters that genuinely belong to this
    track's TrackingParticle and sit on the module it crosses, how many does the
    cone hold?

    This is the study that selClusterIdx and truthTpIdx were added for. Study (2)
    can only ever see clusters the current static window already offered, so it
    cannot detect a true cluster the window never showed the fit -- exactly the
    failure a cone redesign is meant to fix. Here the truth clusters are found by
    joining truthTpIdx to the track's spixMatchedTpIdx, independently of what the
    window did, so a cluster the window missed still counts against the cone.
    """
    xi, ci = P["xi"], P["ci"]
    have = (X["trk_tpIdx"][xi] >= 0) & (K["truthTpIdx"][ci] >= 0)
    is_true = have & (K["truthTpIdx"][ci] == X["trk_tpIdx"][xi])
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

    *** RESULTS ARE CURRENTLY CONFOUNDED. DO NOT TUNE ON THEM. ***
    SmartPixelsRecHitProducer deliberately gives NO angle to clusters with no
    simlink, pending re-derivation of the smarthit_noise_* payload (which is an
    inverse CDF of the old, broken production-momentum angle). Measured
    consequence: clusters WITH a TrackingParticle have hasAlpha 99.5%, clusters
    WITHOUT one have hasAlpha 0.0%. So "reports an angle" is a perfect proxy for
    "is a real cluster", and a large angle weight is partly buying that proxy
    rather than any angle information -- which is why the optimum runs to the top
    of any grid. The scan becomes meaningful once noise clusters carry a
    (re-derived) angle; until then treat the optimum as an upper bound polluted by
    truth leakage, and the LIMITS block below as the honest comparison.
    """
    xi, ci = P["xi"], P["ci"]
    need = ("clRecoCotAlpha", "clRecoCotBeta", "clSigAlpha", "clSigBeta")
    if not all(k in K for k in need):
        print("   (6) SKIPPED: cluster table lacks the sensor angle columns "
              "(recoCotAlpha/recoCotBeta/sigAlpha/sigBeta)")
        out["chi2_weight_scan"] = {"skipped": "cluster table has no sensor angle columns"}
        for a in ax_row:
            a.axis("off")
        return

    # per-pair residuals against the RUNNING projection (what the refit compares to)
    dx = (K["localX"][ci] - X["projLocalX"][xi]) / np.maximum(K["sigX"][ci], 1e-6)
    dy = (K["localY"][ci] - X["projLocalY"][xi]) / np.maximum(K["sigY"][ci], 1e-6)
    okA = (K["clRecoCotAlpha"][ci] > SENTINEL) & (K["clSigAlpha"][ci] > 0) \
          & (X["projCotAlpha"][xi] > SENTINEL)
    okB = (K["clRecoCotBeta"][ci] > SENTINEL) & (K["clSigBeta"][ci] > 0) \
          & (X["projCotBeta"][xi] > SENTINEL)
    # A cluster whose sensor reports NO angle must be neither rewarded nor punished
    # for it. Filling its normalized residual with 0 (the naive choice) makes it
    # cost-free, so any large angle weight simply selects angle-less clusters -- an
    # artefact that made angle-only selection look catastrophic (0.064) and made
    # huge weights look beneficial. The unbiased fill is the EXPECTATION of a
    # normalized residual squared, i.e. 1, so a missing term contributes its mean
    # and the comparison stays fair across candidates with different term counts.
    NEUTRAL = 1.0
    da2 = np.where(okA, ((K["clRecoCotAlpha"][ci] - X["projCotAlpha"][xi])
                         / np.maximum(K["clSigAlpha"][ci], 1e-9)) ** 2, NEUTRAL)
    db2 = np.where(okB, ((K["clRecoCotBeta"][ci] - X["projCotBeta"][xi])
                         / np.maximum(K["clSigBeta"][ci], 1e-9)) ** 2, NEUTRAL)

    # the correct cluster for each crossing, from the TP join (window-independent)
    have = (X["trk_tpIdx"][xi] >= 0) & (K["truthTpIdx"][ci] >= 0)
    is_true = have & (K["truthTpIdx"][ci] == X["trk_tpIdx"][xi])
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

    # SCALE INVARIANCE: argmin of the cost is unchanged by a global rescaling, so
    # only RATIOS matter. w_rphi is therefore pinned to 1 and everything else is
    # measured relative to it -- otherwise "down-weight position" and "up-weight
    # angles" are the same move explored twice, and both scans run into a boundary
    # that means nothing.
    res = {"note": "w_rphi pinned to 1; the cost is scale-invariant so only ratios matter"}
    base, n_den = purity(1, 1, 1, 1)
    res["baseline_all_unit_weights"] = {"purity": base, "n_crossings": n_den}
    print(f"   (6) baseline (1,1,1,1): purity {base:.4f} over {n_den} crossings")

    # coarse scan of r-z relative to r-phi, at unit angle weights
    grid = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    pos = {str(wz): purity(1.0, wz, 1, 1)[0] for wz in grid}
    bestpos = max(pos, key=pos.get)
    res["rz_over_rphi_scan"] = pos
    res["rz_best"] = {"w_rz": float(bestpos), "purity": pos[bestpos]}
    print(f"       r-z / r-phi best w_rz={bestpos} -> {pos[bestpos]:.4f}")
    wr0, wz0 = 1.0, float(bestpos)

    # 1D: bending angle only
    wgrid = [0.0, 0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]
    a1 = {str(w): purity(wr0, wz0, w, 0.0)[0] for w in wgrid}
    res["alpha_only_scan"] = a1
    besta = max(a1, key=a1.get)
    print(f"       1D alpha-only best w_alpha={besta} -> {a1[besta]:.4f}")

    # 2D: both angles
    a2 = {}
    for wa in wgrid:
        for wb in wgrid:
            a2[f"{wa}_{wb}"] = purity(wr0, wz0, wa, wb)[0]
    res["alpha_beta_scan"] = a2
    best2 = max(a2, key=a2.get)
    res["best_overall"] = {"w_rphi": wr0, "w_rz": wz0,
                           "w_alpha": float(best2.split("_")[0]),
                           "w_beta": float(best2.split("_")[1]),
                           "purity": a2[best2]}
    print(f"       2D best (w_alpha,w_beta)={best2} -> {a2[best2]:.4f}   "
          f"(baseline {base:.4f}, gain {a2[best2]-base:+.4f})")
    # The optimum runs to the top of any angle grid, which means the scan is really
    # saying "position contributes little". Test the LIMITS explicitly rather than
    # reporting a boundary value as if it were an optimum.
    lim = {"position_only_(1,wz,0,0)": purity(1.0, wz0, 0.0, 0.0)[0],
           "angle_only_(0,0,1,1)": purity(0.0, 0.0, 1.0, 1.0)[0],
           "alpha_only_(0,0,1,0)": purity(0.0, 0.0, 1.0, 0.0)[0],
           "beta_only_(0,0,0,1)": purity(0.0, 0.0, 0.0, 1.0)[0],
           "baseline_(1,1,1,1)": base}
    res["limits"] = lim
    print("       *** CONFOUNDED: noise clusters currently carry NO angle, and")
    print("           hasAlpha is 99.5% for TP-linked vs 0.0% for unlinked clusters,")
    print("           so angle weight partly buys that proxy. Do not tune on this yet.")
    print("       LIMITS:")
    for k, v in sorted(lim.items(), key=lambda kv: -kv[1]):
        print(f"         {k:<28} {v:.4f}")
    res["CONFOUNDED"] = ("noise clusters carry no angle pending re-derivation of "
                         "smarthit_noise_*; hasAlpha is 99.5%/0.0% for TP-linked/unlinked, "
                         "so angle weight partly buys a truth proxy")
    out["chi2_weight_scan"] = res

    ax = ax_row[0]
    ax.plot(grid, [pos[str(w)] for w in grid], marker="o", label=r"scan $w_{rz}$ ($w_{r\phi}\equiv1$)")
    ax.plot(wgrid, [a1[str(w)] for w in wgrid], marker="s",
            label=r"scan $w_\alpha$ ($w_\beta=0$)")
    ax.axhline(base, color="grey", ls=":", label="baseline (1,1,1,1)")
    ax.set_xscale("symlog", linthresh=0.1); ax.set_xlabel("weight")
    ax.set_ylabel("hit-selection purity")
    ax.set_title("(6) 1D weight scans"); ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = ax_row[1]
    M2 = np.array([[a2[f"{wa}_{wb}"] for wb in wgrid] for wa in wgrid])
    im2 = ax.imshow(M2, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(wgrid))); ax.set_xticklabels(wgrid, rotation=45, fontsize=7)
    ax.set_yticks(range(len(wgrid))); ax.set_yticklabels(wgrid, fontsize=7)
    ax.set_xlabel(r"$w_\beta$"); ax.set_ylabel(r"$w_\alpha$")
    ax.set_title("(6) angle-weight scan (1D = bottom row)"); plt.colorbar(im2, ax=ax)


STUDIES = [
    ("cone occupancy", study_cone_occupancy, 3),
    ("cone containment + size", study_cone_containment, 2),
    ("angle discrimination", study_angle_discrimination, 2),
    ("charge readout gate", study_charge_gate, 2),
    ("unbiased containment", study_true_containment, 2),
    ("chi2 weight scan", study_chi2_weight_scan, 2),
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
