#!/usr/bin/env python
"""Can a SmartPixels sensor tell, from the cluster ALONE, which clusters matter?

THE GOAL THIS SERVES. A smart-pixel sensor cannot run its ML angle estimate on
every cluster inside the L1 latency budget, and it cannot know about any track.
So it must decide, from locally available cluster observables only, which
clusters to spend its output bandwidth on. The clusters worth spending on are
those from tracks that will actually traverse outward through the rest of the IT
and into the OT, because only those can ever be matched to an OT track and
refitted. Everything else is bandwidth spent on a hit no refit will ever use.

WHY pT IS THE WRONG TARGET, AND WHAT WE USE INSTEAD. "high momentum" is a proxy
for "traverses outward"; the thing we actually care about is the traversal. With
truthTpIdx on every cluster we can measure the traversal directly: count the
distinct TBPX layers in which a given TrackingParticle deposits a cluster. So
this script scores TWO targets and compares them:

    TARGET_PT     truthPt > threshold                      (the usual proxy)
    TARGET_TRAV   the TP deposits clusters in >= 3 IT layers  (what we mean)

If they disagree substantially then pT-threshold thinking is mis-specifying the
problem, and that is itself a result.

FEATURES ARE SENSOR-OBSERVABLE ONLY. sizeX, sizeY, charge and combinations. NOT
localX/Y (needs no track, but carries no separation), NOT sigX/sigY (those are
offline CPE template errors, not something an ASIC computes), and obviously NO
truth. A feature an ASIC cannot compute is not a candidate readout gate.

KNOWN FEATURE GAP, stated because it weakens the charge-density variables: the
cluster table stores sizeX/sizeY, which are the BOUNDING-BOX extents, not
SiPixelCluster::size(), the actual fired-pixel count. So "charge per pixel" here
is really charge per bounding-box cell, which is wrong for L-shaped or sparse
clusters. Adding size() is a one-line producer change and would sharpen every
density feature below.

    pixi run python eval_refitq/sensor/cluster_pt_separability.py \
        -i <clusters-tier nano.root> -o eval_refitq/sensor/
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.stats import rankdata

SENTINEL = -900.0
CLUSTER_TABLE = "L1TSmartPixelsCluster"
LAYERS = (1, 2, 3, 4)
PT_THRESHOLDS = (1.0, 2.0)
MIN_TRAVERSED_LAYERS = 3


def auc(score, label):
    """Rank-based AUC. Ties handled by average rank (verified against
    sklearn.roc_auc_score previously; ~800x faster than a pairwise loop)."""
    label = label.astype(bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def load(paths):
    files = [f for p in paths for f in (sorted(_glob.glob(p)) or [p])]
    cols = ["layer", "detId", "charge", "sizeX", "sizeY",
            "truthPt", "truthCotAlpha", "truthCotBeta", "truthTpIdx", "truthChargeFrac"]
    with uproot.open(f"{files[0]}:Events") as t:
        keys = set(t.keys())
    miss = [c for c in cols if f"{CLUSTER_TABLE}_{c}" not in keys]
    if miss:
        raise SystemExit(
            f"input lacks {CLUSTER_TABLE} columns {miss}. Produce a Clusters-tier file:\n"
            "  test/makeSpixConfig.py --pu 200 --tier clusters-truth "
            "--variant digiRefit:1111 --needs-truth -o <out>.py")
    A = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{CLUSTER_TABLE}_{c}" for c in cols])
    n = ak.to_numpy(ak.num(A[f"{CLUSTER_TABLE}_layer"]))
    K = {c: ak.to_numpy(ak.flatten(A[f"{CLUSTER_TABLE}_{c}"])) for c in cols}
    K["event"] = np.repeat(np.arange(len(n)), n)
    n_ev = sum(uproot.open(f"{f}:Events").num_entries for f in files)
    print(f"files={len(files)} events={n_ev} clusters={len(K['layer'])}")
    return K, n_ev


def traversal_target(K):
    """Per cluster: how many distinct IT layers does its TP light up in this event?

    Grouped on (event, truthTpIdx) so a TP is never mixed across events. Clusters
    with no TP get 0 and are excluded from the traversal target entirely -- they
    cannot be scored against a property of a particle they have no link to.
    """
    ok = K["truthTpIdx"] >= 0
    key = np.where(ok, K["event"].astype(np.int64) * (1 << 20) + K["truthTpIdx"].astype(np.int64), -1)
    nlay = np.zeros(len(key), dtype=np.int8)
    idx = np.flatnonzero(ok)
    order = np.lexsort((K["layer"][idx], key[idx]))
    si = idx[order]
    sk, sl = key[si], K["layer"][si]
    # distinct-layer count per key
    starts = np.r_[True, sk[1:] != sk[:-1]]
    grp = np.cumsum(starts) - 1
    newlayer = np.r_[True, (sk[1:] != sk[:-1]) | (sl[1:] != sl[:-1])]
    counts = np.bincount(grp, weights=newlayer.astype(float)).astype(np.int8)
    nlay[si] = counts[grp]
    return nlay, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputs", nargs="+", required=True)
    ap.add_argument("-o", "--outdir", default="eval_refitq/sensor")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    K, n_ev = load(args.inputs)
    out = {"n_events": n_ev, "n_clusters": int(len(K["layer"]))}

    # ---- features an ASIC could compute -----------------------------------
    sx = K["sizeX"].astype(np.float64)
    sy = K["sizeY"].astype(np.float64)
    q = K["charge"].astype(np.float64)
    box = np.maximum(sx * sy, 1.0)
    FEATS = {
        "charge": q,
        "sizeX": sx,
        "sizeY": sy,
        "nBoxPix": box,
        "chargeDensity": q / box,             # charge per bounding-box cell
        "chargePerSizeY": q / np.maximum(sy, 1.0),
        "aspect_sizeY_over_sizeX": sy / np.maximum(sx, 1.0),
        "minus_sizeY": -sy,                   # sign-flipped: short clusters = signal?
    }

    nlay, has_tp = traversal_target(K)
    out["traversal"] = {
        "frac_clusters_with_tp": float(has_tp.mean()),
        "layer_count_hist": np.bincount(nlay, minlength=5)[:5].tolist(),
    }

    # ---- do the two targets even agree? -----------------------------------
    print("\n(1) do pT and traversal pick the same clusters?")
    trav = has_tp & (nlay >= MIN_TRAVERSED_LAYERS)
    agree = {}
    for thr in PT_THRESHOLDS:
        hp = has_tp & (K["truthPt"] > thr)
        both = hp & trav
        agree[str(thr)] = {
            "frac_highpt": float(hp[has_tp].mean()),
            "frac_traversing": float(trav[has_tp].mean()),
            "frac_both": float(both[has_tp].mean()),
            "P_traverse_given_highpt": float(trav[hp].mean()) if hp.any() else float("nan"),
            "P_highpt_given_traverse": float(hp[trav].mean()) if trav.any() else float("nan"),
        }
        a = agree[str(thr)]
        print(f"   pT>{thr}: highpT {a['frac_highpt']:.4f}  traversing {a['frac_traversing']:.4f}  "
              f"both {a['frac_both']:.4f}")
        print(f"            P(traverse | pT>{thr}) = {a['P_traverse_given_highpt']:.3f}   "
              f"P(pT>{thr} | traverse) = {a['P_highpt_given_traverse']:.3f}")
    out["target_agreement"] = agree

    # ---- single-feature separability, per layer and per target ------------
    print("\n(2) single-feature AUC (sensor-observable features only)")
    targets = {f"pt>{t}": (has_tp & (K['truthPt'] > t), has_tp) for t in PT_THRESHOLDS}
    targets[f"traverse>={MIN_TRAVERSED_LAYERS}L"] = (trav, has_tp)
    res = {}
    for tname, (lab, dom) in targets.items():
        print(f"   target {tname}   (positives {int(lab[dom].sum())} of {int(dom.sum())})")
        res[tname] = {}
        for fname, fv in FEATS.items():
            a_all = auc(fv[dom], lab[dom])
            per_layer = []
            for L in LAYERS:
                m = dom & (K["layer"] == L)
                per_layer.append(auc(fv[m], lab[m]) if m.sum() > 100 else float("nan"))
            res[tname][fname] = {"auc_all": a_all, "auc_per_layer": per_layer}
            print(f"      {fname:<26} AUC {a_all:.4f}   per-layer "
                  + " ".join(f"{v:.3f}" for v in per_layer))
    out["single_feature_auc"] = res

    # ---- what does the truth angle look like for each class? --------------
    print("\n(3) truth angles by class -- is 'grazing' really the confusable population?")
    ang = {}
    for tname, (lab, dom) in targets.items():
        m = dom & (K["truthCotAlpha"] > SENTINEL)
        s, b = m & lab, m & ~lab
        rec = {}
        for nm, arr in (("cotAlpha", np.abs(K["truthCotAlpha"])), ("cotBeta", np.abs(K["truthCotBeta"]))):
            rec[nm] = {"signal_median": float(np.median(arr[s])) if s.any() else float("nan"),
                       "signal_p90": float(np.quantile(arr[s], 0.9)) if s.any() else float("nan"),
                       "bkg_median": float(np.median(arr[b])) if b.any() else float("nan"),
                       "bkg_p90": float(np.quantile(arr[b], 0.9)) if b.any() else float("nan")}
            print(f"   {tname:<18} |{nm}|  signal med {rec[nm]['signal_median']:.4f} "
                  f"p90 {rec[nm]['signal_p90']:.4f}   bkg med {rec[nm]['bkg_median']:.4f} "
                  f"p90 {rec[nm]['bkg_p90']:.4f}")
        ang[tname] = rec
    out["truth_angles_by_class"] = ang

    # ---- working points: a readout budget --------------------------------
    print("\n(4) readout working points: keep X% of the target, at what total bandwidth?")
    wp = {}
    for tname, (lab, dom) in targets.items():
        best = None
        for fname, fv in FEATS.items():
            v = fv[dom]
            l = lab[dom]
            if l.sum() < 100:
                continue
            for keep in (0.90, 0.95, 0.99):
                cut = np.quantile(v[l], 1.0 - keep)   # keep the HIGH side
                frac_all = float((v >= cut).mean())
                eff = float((v[l] >= cut).mean())
                rec = {"feature": fname, "keep_target": keep, "cut": float(cut),
                       "eff": eff, "frac_all_clusters_kept": frac_all,
                       "enrichment": float(eff / frac_all) if frac_all > 0 else float("nan")}
                wp.setdefault(tname, []).append(rec)
                if keep == 0.95 and (best is None or frac_all < best["frac_all_clusters_kept"]):
                    best = rec
        if best:
            print(f"   {tname:<18} best@95%: {best['feature']:<24} keeps "
                  f"{best['frac_all_clusters_kept']*100:5.1f}% of ALL clusters "
                  f"(enrichment {best['enrichment']:.2f}x)")
    out["working_points"] = wp

    # ---- (4b) THE CEILING: what could a PERFECT angle measurement buy? ----
    # This is the number that decides whether a smart-pixel ML angle estimate is
    # worth its bandwidth. Raw sizeX is the shape proxy an ordinary sensor already
    # has; |cotAlpha| from truth is what a PERFECT angle estimate would give. The
    # gap between them is the headroom an on-sensor ML regressor is competing for.
    print("\n(4b) ceiling: perfect angle vs raw shape, for pT selection")
    print("     alpha carries the BENDING, so its power must grow with radius")
    ceil = {}
    dom = has_tp & (K["truthCotAlpha"] > SENTINEL)
    for thr in PT_THRESHOLDS:
        lab = dom & (K["truthPt"] > thr)
        rec = {}
        for nm, sc in (("perfect_alpha", -np.abs(K["truthCotAlpha"])),
                       ("perfect_beta", -np.abs(K["truthCotBeta"])),
                       ("raw_sizeX", -sx), ("raw_sizeY", -sy)):
            rec[nm] = {"auc_all": auc(sc[dom], lab[dom]),
                       "auc_per_layer": [auc(sc[dom & (K["layer"] == L)], lab[dom & (K["layer"] == L)])
                                         if (dom & (K["layer"] == L)).sum() > 100 else float("nan")
                                         for L in LAYERS]}
        ceil[f"pt>{thr}"] = rec
        print(f"   pT>{thr}:")
        for nm in ("perfect_alpha", "raw_sizeX"):
            pl = " ".join(f"{v:.3f}" for v in rec[nm]["auc_per_layer"])
            print(f"      {nm:<16} AUC {rec[nm]['auc_all']:.4f}   per-layer {pl}")
    out["perfect_angle_ceiling"] = ceil

    # ---- (5) how much could rule B possibly matter? ----------------------
    print("\n(5) charge-share near-ties: could the winner-takes-pixel rule flip the dominant TP?")
    cf = K["truthChargeFrac"]
    m = cf > SENTINEL
    rec = {"n": int(m.sum())}
    for lo, hi, nm in ((0.0, 0.55, "<0.55 (near-tie, rule could flip)"),
                       (0.55, 0.9, "0.55-0.9 (shared, unlikely to flip)"),
                       (0.9, 1.01, ">0.9 (clean)")):
        f = float(((cf[m] >= lo) & (cf[m] < hi)).mean())
        rec[nm] = f
        print(f"   chargeFrac {nm:<38} {f*100:5.2f}%")
    out["charge_share"] = rec
    print("   Only the first row can change the dominant TP under a fractional-ADC rule,")
    print("   so that row bounds how much question (B) can matter.")

    # ---- plots ------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))
    ax = axes[0][0]
    ax.bar(range(5), out["traversal"]["layer_count_hist"])
    ax.set_yscale("log"); ax.set_xlabel("distinct IT layers lit by the cluster's TP")
    ax.set_ylabel("clusters"); ax.set_title("(1) does the particle traverse?"); ax.grid(alpha=.3)

    ax = axes[0][1]
    names = list(FEATS)
    for tname in res:
        ax.plot([res[tname][f]["auc_all"] for f in names], marker="o", label=tname)
    ax.axhline(0.5, color="grey", ls=":")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("AUC"); ax.set_title("(2) single-feature separability"); ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = axes[0][2]
    dom = has_tp
    for lab, nm in ((trav[dom], f"traverses >={MIN_TRAVERSED_LAYERS}L"), (~trav[dom], "does not")):
        ax.hist(sy[dom][lab], bins=np.arange(0.5, 20.5), histtype="step", density=True, label=nm)
    ax.set_xlabel("sizeY (pixels)"); ax.set_ylabel("density")
    ax.set_title("(3) cluster length vs traversal"); ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = axes[1][0]
    for lab, nm in ((trav[dom], "traverses"), (~trav[dom], "does not")):
        ax.hist(np.log10(np.maximum(q[dom][lab], 1)), bins=60, histtype="step", density=True, label=nm)
    ax.set_xlabel(r"$\log_{10}$(cluster charge)"); ax.set_ylabel("density")
    ax.set_title("(3) charge vs traversal"); ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = axes[1][1]
    mm = dom & (K["truthCotBeta"] > SENTINEL)
    ax.scatter(np.abs(K["truthCotBeta"][mm])[:20000], sy[mm][:20000], s=1, alpha=.05)
    ax.set_xlabel(r"$|\cot\beta|$ (truth)"); ax.set_ylabel("sizeY (pixels)")
    ax.set_xlim(0, 4); ax.set_ylim(0, 25)
    ax.set_title("(3) is sizeY an angle proxy?"); ax.grid(alpha=.3)

    ax = axes[1][2]
    ax.hist(cf[m], bins=np.linspace(0, 1, 60), histtype="stepfilled", alpha=.75)
    ax.axvline(0.55, color="r", ls="--", lw=1)
    ax.set_xlabel("dominant contributor charge share"); ax.set_ylabel("clusters")
    ax.set_yscale("log"); ax.set_title("(5) shared clusters; red = rule-B sensitive"); ax.grid(alpha=.3)

    fig.suptitle(f"SmartPixels sensor-level separability — {n_ev} events, "
                 f"{len(K['layer'])} clusters", y=1.01)
    fig.tight_layout()
    png = os.path.join(args.outdir, "cluster_pt_separability.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    with open(os.path.join(args.outdir, "cluster_pt_separability.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
