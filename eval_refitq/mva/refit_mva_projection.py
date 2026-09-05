#!/usr/bin/env python
"""Projected performance of a refit MVA, under BOTH readings of what it is for.

There are two different classifiers people mean by "the refit MVA", they have
different inputs, different targets and different deployment points, and mixing
them up is how the current `genuine`-trained model ended up nearly blind to the
failure mode the refit actually controls. This scores them separately.

--------------------------------------------------------------------------
A) TRACK-LEVEL TRUST GATE.  "Should I believe this refit?"
--------------------------------------------------------------------------
One row per refit track. Target = the refit used NO wrong hits, established by
the TP join (window-independent), not by the deployed `genuine` label.

Motivation, measured earlier: a clean refit improves d0 ~8x and z0 ~11.5x, ONE
wrong hit annihilates that, and only a trust gate makes the refit a strict
improvement. Under outsideIn ordering the clean fraction rose from 53.9% to
75.3%, which changes the operating point: the gate now has to reject a quarter of
tracks rather than half, so its cost in efficiency is much lower and its
achievable purity higher. That is the thing to quantify.

Features are all hardware-available at the refit output: hit counts and the layer
mask, window occupancy/truncation, the four chi2 totals, and the seed-vs-refit
diagnostics (chi2ITAtSeed/AtRefit, shiftChi2, logDetRatio) that need no stubs.

--------------------------------------------------------------------------
B) PER-CLUSTER COMPATIBILITY.  "Does this cluster belong to this track?"
--------------------------------------------------------------------------
One row per (crossing, cluster) pair inside the window. Target = the cluster's
dominant TP is the track's TP.

This is the classifier that would REPLACE the hand-weighted selection chi2. It is
evaluated against the SEED track for the first layer visited and against the
UPDATED track for later layers, exactly as the refit sees them -- which is why
the sidecar stores both projections per crossing. The script splits its scores by
first-visited vs later so the two regimes are not averaged together: the first
crossing has the full seed uncertainty and no updates behind it, and is a
genuinely different problem from the later ones.

Both use out-of-fold scoring (no track appears in both train and score), and both
report AUC against the practical quantity as well: what a working point buys.

    pixi run python eval_refitq/mva/refit_mva_projection.py -i <clusters nano> -o eval_refitq/mva/
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
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve

SENTINEL = -900.0
CLUSTER_TABLE = "L1TSmartPixelsCluster"
REF = "L1TTrack"
NFOLD = 3


def _discover(path):
    with uproot.open(f"{path}:Events") as t:
        keys = set(t.keys())
    cfgs = sorted({m.group(1) for m in
                   (re.match(r"^L1TSmartPixelsRefitHitDigiRefit([A-Za-z0-9]+)_", k) for k in keys) if m})
    if not cfgs:
        raise SystemExit("no refit hit table in the input")
    cfg = cfgs[-1]
    return f"L1TSmartPixelsRefitHitDigiRefit{cfg}", f"L1TSmartPixelsTrackDigiRefit{cfg}", cfg, keys


def load(paths):
    files = [f for p in paths for f in (sorted(_glob.glob(p)) or [p])]
    HIT, VAR, cfg, keys = _discover(files[0])

    hcols = ["trackIdx", "layer", "detId", "hitAccepted", "selHitClass",
             "projLocalX", "projLocalY", "projCotAlpha", "projCotBeta",
             "projSeedLocalX", "projSeedLocalY", "projSeedSigX", "projSeedSigY",
             "projSeedCotAlpha", "projSeedCotBeta", "windowMult", "selClusterIdx"]
    vcols = ["spixNCrossings", "spixNAcceptedHits", "spixLayerHitMask", "spixMaxWindowMult",
             "spixAnyWindowTruncated", "spixNKFUpdates",
             "spixChi2IncXTot", "spixChi2IncYTot", "spixChi2IncAlphaTot", "spixChi2IncBetaTot",
             "spixChi2ITAtSeed", "spixChi2ITAtRefit", "spixShiftChi2", "spixLogDetRatio",
             "spixRefitPerformed", "spixMatchedTpIdx"]
    ccols = ["layer", "detId", "localX", "localY", "sigX", "sigY", "charge",
             "sizeX", "sizeY", "truthTpIdx"]
    copt = ["size", "recoCotAlpha", "recoCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta"]

    miss = [c for c in vcols if f"{VAR}_{c}" not in keys]
    if miss:
        raise SystemExit(f"track table lacks {miss}; produce a current Clusters-tier nano")

    H = uproot.concatenate([f"{f}:Events" for f in files], filter_name=[f"{HIT}_{c}" for c in hcols])
    V = uproot.concatenate([f"{f}:Events" for f in files], filter_name=[f"{VAR}_{c}" for c in vcols])
    C = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{CLUSTER_TABLE}_{c}" for c in ccols
                                        + [x for x in copt if f"{CLUSTER_TABLE}_{x}" in keys]])
    R = uproot.concatenate([f"{f}:Events" for f in files], filter_name=[f"{REF}_pt", f"{REF}_eta"])
    n_ev = sum(uproot.open(f"{f}:Events").num_entries for f in files)

    nh = ak.to_numpy(ak.num(H[f"{HIT}_layer"]))
    nv = ak.to_numpy(ak.num(V[f"{VAR}_spixNCrossings"]))
    nc = ak.to_numpy(ak.num(C[f"{CLUSTER_TABLE}_layer"]))
    X = {c: ak.to_numpy(ak.flatten(H[f"{HIT}_{c}"])) for c in hcols}
    T = {c: ak.to_numpy(ak.flatten(V[f"{VAR}_{c}"])) for c in vcols}
    K = {c: ak.to_numpy(ak.flatten(C[f"{CLUSTER_TABLE}_{c}"]))
         for c in ccols + [x for x in copt if f"{CLUSTER_TABLE}_{x}" in keys]}
    X["event"] = np.repeat(np.arange(len(nh)), nh)
    T["event"] = np.repeat(np.arange(len(nv)), nv)
    K["event"] = np.repeat(np.arange(len(nc)), nc)
    T["_off"] = np.concatenate([[0], np.cumsum(nv)])
    K["_off"] = np.concatenate([[0], np.cumsum(nc)])
    X["gtrk"] = T["_off"][X["event"]] + X["trackIdx"].astype(np.int64)
    for c in ("pt", "eta"):
        T[f"trk_{c}"] = ak.to_numpy(ak.flatten(R[f"{REF}_{c}"]))
    print(f"files={len(files)} cfg={cfg} events={n_ev} tracks={len(T['spixNCrossings'])} "
          f"crossings={len(X['layer'])} clusters={len(K['layer'])}")
    return X, T, K, cfg, n_ev


def oof_score(Xf, y, groups, n_estimators=220, max_depth=5):
    """Out-of-fold XGB score. Folds split on TRACK so a track never trains on itself."""
    pred = np.zeros(len(y))
    fold = groups % NFOLD
    for f in range(NFOLD):
        tr, te = fold != f, fold == f
        if y[tr].sum() < 20 or (~y[tr]).sum() < 20 or te.sum() == 0:
            continue
        m = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.08,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                              tree_method="hist", n_jobs=8)
        m.fit(Xf[tr], y[tr])
        pred[te] = m.predict_proba(Xf[te])[:, 1]
    return pred, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputs", nargs="+", required=True)
    ap.add_argument("-o", "--outdir", default="eval_refitq/mva")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    X, T, K, cfg, n_ev = load(args.inputs)
    out = {"config": cfg, "n_events": n_ev}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # ---------- shared truth: which crossings picked a wrong hit ----------
    # via the TP join, independent of the deployed genuine label
    acc = X["hitAccepted"] > 0
    wrong = acc & (X["selHitClass"] == 1)
    ntrk = len(T["spixNCrossings"])
    n_wrong = np.zeros(ntrk)
    np.add.at(n_wrong, X["gtrk"][wrong], 1.0)

    # ================= A) TRACK-LEVEL TRUST GATE =========================
    feats_A = ["spixNCrossings", "spixNAcceptedHits", "spixLayerHitMask", "spixMaxWindowMult",
               "spixAnyWindowTruncated", "spixNKFUpdates", "spixChi2IncXTot", "spixChi2IncYTot",
               "spixChi2IncAlphaTot", "spixChi2IncBetaTot", "spixChi2ITAtSeed",
               "spixChi2ITAtRefit", "spixShiftChi2", "spixLogDetRatio"]
    good = (T["spixRefitPerformed"] > 0)
    FA = np.column_stack([T[f][good].astype(np.float64) for f in feats_A]
                         + [(T["spixChi2ITAtSeed"] - T["spixChi2ITAtRefit"])[good]])
    FA = np.nan_to_num(FA, nan=-999., posinf=1e12, neginf=-1e12)
    yA = (n_wrong[good] == 0)
    gA = np.arange(len(yA))
    print(f"\n(A) trust gate: {len(yA)} refit tracks, clean fraction {yA.mean():.4f}")
    pA, _ = oof_score(FA, yA, gA)
    aucA = roc_auc_score(yA, pA)
    print(f"    out-of-fold AUC = {aucA:.4f}")
    recA = {"n_tracks": int(len(yA)), "clean_fraction": float(yA.mean()), "auc": float(aucA),
            "features": feats_A + ["chi2ITSeedMinusRefit"], "working_points": []}
    for keep in (0.95, 0.90, 0.80, 0.60):
        cut = np.quantile(pA[yA], 1 - keep)
        sel = pA >= cut
        purity = float(yA[sel].mean())
        recA["working_points"].append(
            {"keep_clean": keep, "cut": float(cut), "kept_frac_of_all": float(sel.mean()),
             "purity": purity})
        print(f"      keep {keep:.0%} of clean refits -> keeps {sel.mean():.1%} of all tracks, "
              f"purity {purity:.3f} (baseline {yA.mean():.3f})")
    out["A_trust_gate"] = recA
    fpr, tpr, _ = roc_curve(yA, pA)
    axes[0].plot(fpr, tpr, label=f"AUC {aucA:.3f}")
    axes[0].plot([0, 1], [0, 1], "k:", lw=.8)
    axes[0].set_xlabel("fraction of DIRTY refits kept"); axes[0].set_ylabel("fraction of CLEAN kept")
    axes[0].set_title(f"(A) trust gate — clean = {yA.mean():.1%}")
    axes[0].legend(); axes[0].grid(alpha=.3)

    # ================= B) PER-CLUSTER COMPATIBILITY ======================
    # join clusters to crossings on (event, detId)
    ck = K["event"].astype(np.int64) * (1 << 32) + K["detId"].astype(np.int64)
    order = np.argsort(ck, kind="stable"); cs = ck[order]
    xk = X["event"].astype(np.int64) * (1 << 32) + X["detId"].astype(np.int64)
    lo, hi = np.searchsorted(cs, xk, "left"), np.searchsorted(cs, xk, "right")
    n = hi - lo
    xi = np.repeat(np.arange(len(xk)), n)
    ramp = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    ci = order[np.repeat(lo, n) + ramp]

    # FIRST-VISITED crossing per track: hitInfo records are pushed in VISIT order,
    # so the first record of a track is the crossing evaluated against the SEED.
    first_rec = np.zeros(len(X["layer"]), dtype=bool)
    seen = {}
    ordr = np.lexsort((np.arange(len(X["gtrk"])), X["gtrk"]))
    gsorted = X["gtrk"][ordr]
    firsts = ordr[np.r_[True, gsorted[1:] != gsorted[:-1]]]
    first_rec[firsts] = True
    is_first = first_rec[xi]

    # residuals against the projection the refit ACTUALLY used at that crossing
    px = np.where(is_first, X["projSeedLocalX"][xi], X["projLocalX"][xi])
    py = np.where(is_first, X["projSeedLocalY"][xi], X["projLocalY"][xi])
    pa = np.where(is_first, X["projSeedCotAlpha"][xi], X["projCotAlpha"][xi])
    pb = np.where(is_first, X["projSeedCotBeta"][xi], X["projCotBeta"][xi])
    sx = np.maximum(K["sigX"][ci], 1e-6); sy = np.maximum(K["sigY"][ci], 1e-6)
    cols = {"dx_sig": (K["localX"][ci] - px) / sx,
            "dy_sig": (K["localY"][ci] - py) / sy,
            "sigX": K["sigX"][ci], "sigY": K["sigY"][ci],
            "charge": K["charge"][ci], "sizeX": K["sizeX"][ci], "sizeY": K["sizeY"][ci],
            "layer": K["layer"][ci], "is_first": is_first.astype(float),
            "windowMult": X["windowMult"][xi],
            "projSeedSigX": X["projSeedSigX"][xi], "projSeedSigY": X["projSeedSigY"][xi]}
    if "size" in K:
        cols["size"] = K["size"][ci]
        cols["chargeDensity"] = K["charge"][ci] / np.maximum(K["size"][ci], 1)
    if "recoCotAlpha" in K:
        sa = np.maximum(K["sigAlpha"][ci], 1e-9); sb = np.maximum(K["sigBeta"][ci], 1e-9)
        cols["dcotA_sig"] = np.where(K["hasAlpha"][ci] > 0, (K["recoCotAlpha"][ci] - pa) / sa, 0.)
        cols["dcotB_sig"] = np.where(K["hasBeta"][ci] > 0, (K["recoCotBeta"][ci] - pb) / sb, 0.)
        cols["hasAlpha"] = K["hasAlpha"][ci].astype(float)
        cols["hasBeta"] = K["hasBeta"][ci].astype(float)

    valid = (px > SENTINEL) & (K["truthTpIdx"][ci] >= -1)
    tp_of_trk = T["spixMatchedTpIdx"]
    yB = (K["truthTpIdx"][ci] >= 0) & (tp_of_trk[X["gtrk"][xi]] >= 0) \
         & (K["truthTpIdx"][ci] == tp_of_trk[X["gtrk"][xi]])
    names = list(cols)
    FB = np.nan_to_num(np.column_stack([cols[c].astype(np.float64) for c in names]),
                       nan=0., posinf=1e12, neginf=-1e12)[valid]
    yBv = yB[valid]; gB = X["gtrk"][xi][valid]
    print(f"\n(B) per-cluster compatibility: {len(yBv)} (crossing,cluster) pairs, "
          f"{yBv.mean():.4f} are the track's own")
    pB, mB = oof_score(FB, yBv, gB)
    aucB = roc_auc_score(yBv, pB)
    print(f"    out-of-fold AUC = {aucB:.4f}   features: {names}")
    recB = {"n_pairs": int(len(yBv)), "positive_fraction": float(yBv.mean()),
            "auc": float(aucB), "features": names}
    firstv = cols["is_first"][valid] > 0
    for nm, m in (("first-visited (vs SEED track)", firstv), ("later (vs UPDATED track)", ~firstv)):
        if m.sum() > 100 and 0 < yBv[m].mean() < 1:
            a = roc_auc_score(yBv[m], pB[m])
            recB[nm] = {"auc": float(a), "n": int(m.sum()), "pos_frac": float(yBv[m].mean())}
            print(f"      {nm:<32} AUC {a:.4f}  (n={m.sum()}, pos {yBv[m].mean():.3f})")
    for L in (1, 2, 3, 4):
        m = valid.copy(); m[:] = False
        mm = (cols["layer"][valid] == L)
        if mm.sum() > 100 and 0 < yBv[mm].mean() < 1:
            recB[f"layer{L}_auc"] = float(roc_auc_score(yBv[mm], pB[mm]))
    # what a top-1 pick would achieve, i.e. the MVA used AS the selector
    ordr2 = np.lexsort((-pB, gB * 8 + cols["layer"][valid].astype(int)))
    keyb = (gB * 8 + cols["layer"][valid].astype(int))[ordr2]
    firstpick = np.r_[True, keyb[1:] != keyb[:-1]]
    picked = np.zeros(len(yBv), dtype=bool); picked[ordr2[firstpick]] = True
    hastrue = np.zeros(len(yBv), dtype=bool)
    key_all = gB * 8 + cols["layer"][valid].astype(int)
    ktrue = set(key_all[yBv].tolist())
    hastrue = np.array([k in ktrue for k in key_all])
    sel_purity = float(yBv[picked & hastrue].mean()) if (picked & hastrue).any() else float("nan")
    recB["mva_as_selector_purity"] = sel_purity
    print(f"      MVA used AS the selector (top-1 per crossing): purity {sel_purity:.4f}")
    # HONEST RE-SCORE. hasAlpha/hasBeta are currently a truth proxy: the producer
    # gives noise clusters no angle pending re-derivation of smarthit_noise_*, so
    # hasAlpha is 99.5% for TP-linked clusters and 0.0% for unlinked ones. Since
    # every unlinked cluster is automatically a negative, those two features let the
    # model separate a large slice of the negatives for free and the headline AUC is
    # not achievable in a world where noise clusters report an angle. Re-score
    # without them.
    # ALL angle features leak, not just the presence flags: dcot*_sig is zero-filled
    # when the sensor reports no angle, which is itself the tell. Drop the whole
    # angle block for the honest floor.
    leak = [n for n in names if n in ("hasAlpha", "hasBeta", "dcotA_sig", "dcotB_sig")]
    if leak:
        keep = [i for i, n in enumerate(names) if n not in leak]
        pB2, _ = oof_score(FB[:, keep], yBv, gB)
        auc2 = roc_auc_score(yBv, pB2)
        recB["auc_no_angle_features"] = float(auc2)
        recB["leakage_note"] = (
            "every angle feature is a truth proxy TODAY: noise clusters carry no angle "
            "(hasAlpha 99.5% TP-linked vs 0.0% unlinked) and dcot*_sig is zero-filled when "
            "absent. auc with angles is an upper bound; auc_no_angle_features is the floor "
            "with no angle information at all. The achievable value lies between, and is "
            "only measurable once smarthit_noise_* is re-derived and noise clusters get an "
            "angle.")
        print(f"      NO angle features at all (honest floor): AUC {auc2:.4f}")
        print(f"      => achievable value lies between {auc2:.4f} and {aucB:.4f}; the gap is")
        print( "         unmeasurable until noise clusters carry a re-derived angle.")
    out["B_cluster_compatibility"] = recB
    fpr, tpr, _ = roc_curve(yBv, pB)
    axes[1].plot(fpr, tpr, label=f"all, AUC {aucB:.3f}")
    for nm, m in (("first (seed)", firstv), ("later (updated)", ~firstv)):
        if m.sum() > 100 and 0 < yBv[m].mean() < 1:
            f2, t2, _ = roc_curve(yBv[m], pB[m]); axes[1].plot(f2, t2, ls="--", label=nm)
    axes[1].plot([0, 1], [0, 1], "k:", lw=.8)
    axes[1].set_xlabel("wrong clusters kept"); axes[1].set_ylabel("correct clusters kept")
    axes[1].set_title("(B) per-cluster compatibility"); axes[1].legend(fontsize=7); axes[1].grid(alpha=.3)

    imp = getattr(mB, "feature_importances_", None)
    if imp is not None:
        o = np.argsort(imp)[::-1][:12]
        axes[2].barh([names[i] for i in o][::-1], imp[o][::-1])
        axes[2].set_title("(B) feature importance"); axes[2].grid(alpha=.3)
        out["B_cluster_compatibility"]["importance"] = {names[i]: float(imp[i]) for i in o}

    fig.suptitle(f"Refit MVA projection — {cfg}, {n_ev} events", y=1.03)
    fig.tight_layout()
    png = os.path.join(args.outdir, "refit_mva_projection.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    with open(os.path.join(args.outdir, "refit_mva_projection.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
