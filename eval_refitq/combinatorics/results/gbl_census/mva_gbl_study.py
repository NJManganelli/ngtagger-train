"""Do smoothed-refit (GBL) outputs improve the site taggers?

Census PINNED: cache-tp-incl/tpcensus_038cb9d6de3d851e (PRE-menu-update seed
menu); GBL features from results/gbl_census (run_census_gbl.py). Same protocol as
export_site.build_scores: every 3rd EVENT held out, export_site._train_one
(HistGradientBoosting, 200 iterations, <= 1.5M training tracks), the same
targets (quality_taggers.make_target) and |fitted d0| windows, d0-derived
features withheld from the resolution taggers.

Variants per tagger:
  baseline      the site's inputs
  gbl_replace   the tagger's chi2/residual-type inputs REMOVED and the GBL set added
  aggregate     the tagger's inputs + the GBL set
  forward_only  the tagger's inputs + the forward-filter innovation-pull set
AUC uncertainty: paired bootstrap over held-out tracks (delta vs baseline).
"""
import sys, os, json, glob, time
import numpy as np
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import kf_emulation as KF          # noqa: E402
import export_site as ES           # noqa: E402
import quality_taggers as QT       # noqa: E402
import census_gbl as G             # noqa: E402
from sklearn.metrics import roc_auc_score

CENSUS = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/cache-tp-incl/tpcensus_038cb9d6de3d851e"
FEATD = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/gbl_census"
assert json.load(open(f"{FEATD}/manifest.json"))["census"] == CENSUS

# the census inputs that are chi2 or residual summaries of the census fit
CHI2_LIKE = ("chi2_rphi_per_layer", "chi2_rz_per_layer", "chi2_scaled", "chi2_angle_per_cl",
             "max_angle_pull", "second_angle_pull", "max_pos_pull", "chi2_rphi_it",
             "chi2_rphi_ot", "rank_score")
GBL_SET = tuple(n for n in G.FEATURE_NAMES if n.startswith("gbl_"))
FWD_SET = tuple(n for n in G.FEATURE_NAMES if n.startswith("fwd_"))


def load():
    Ts, Fs = [], []
    for sh in sorted(glob.glob(f"{CENSUS}/chunk_*.npz")):
        z = np.load(sh)
        f = np.load(f"{FEATD}/{os.path.basename(sh)}")
        for k in sorted(int(x[4:]) for x in z.files if x.startswith("trk_")):
            t = z[f"trk_{k}"]
            if not len(t):
                continue
            Ts.append(t); Fs.append(f[f"feat_{k}"])
            assert len(Fs[-1]) == len(t), f"{sh} seed {k}: feature/track row mismatch"
    return np.concatenate(Ts), np.concatenate(Fs)


def main():
    t0 = time.time()
    T0, Fg = load()
    col = {c: j for j, c in enumerate(KF.TRACK_COLS)}
    for j, n in enumerate(G.FEATURE_NAMES):
        col[n] = T0.shape[1] + j
    T = np.concatenate([T0, Fg], 1)
    del T0, Fg
    n = len(T)
    print(f"{n:,} census track rows, {time.time()-t0:.0f} s to load", flush=True)
    ev = T[:, col["event"]]
    is_te = np.isin(ev, np.unique(ev)[::3])
    d0f = np.abs(T[:, col["d0"]])
    WIN = {"prompt": d0f <= ES.D0_SPLIT_CM, "displaced": d0f > ES.D0_SPLIT_CM}
    fmat = lambda feats: np.nan_to_num(T[:, [col[f] for f in feats]], posinf=0, neginf=0)

    taggers = []
    y_leg = T[:, col["n_wrong"]] <= 1
    for name, spec in ES.LEGACY_DEFS.items():
        base = [c for c in (spec["feats"] or KF.MVA_FEATURES) if c in col]
        taggers.append((name, base, y_leg, np.ones(n, bool)))
    vfeats = [c for c in KF.MVA_FEATURES if c in col and c not in ES.D0_LAUNDER_FEATS]
    for tname in QT.TARGETS:
        if tname not in ES.VARIANT_KEEP:
            continue
        for wname, win in WIN.items():
            y, ok, _ = QT.make_target(T, col, tname, win)
            if y is not None:
                taggers.append((f"mva_{ES.VARIANT_KEEP[tname]}_{wname}", vfeats, y, ok))

    rng = np.random.default_rng(1)
    brng = np.random.default_rng(7)
    res = {"census": CENSUS, "seed_menu": "pre-menu-update", "n_tracks": int(n),
           "swapped_out_in_gbl_replace": list(CHI2_LIKE), "gbl_set": list(GBL_SET),
           "forward_set": list(FWD_SET), "taggers": {}}
    for name, base, y, ok in taggers:
        variants = {"baseline": base,
                    "gbl_replace": [c for c in base if c not in CHI2_LIKE] + list(GBL_SET),
                    "aggregate": base + list(GBL_SET),
                    "forward_only": base + list(FWD_SET)}
        preds, row = {}, {"swapped_out": [c for c in base if c in CHI2_LIKE]}
        for vn, feats in variants.items():
            X = fmat(feats)
            g, auc, te = ES._train_one(X, y, ok & np.isfinite(X).all(1), is_te, rng)
            if g is None:
                row[vn] = None
                continue
            preds[vn] = (te, g.predict_proba(X[te])[:, 1])
            row[vn] = {"auc": round(auc, 4), "n_features": len(feats)}
            del X
        # paired bootstrap of AUC(variant) - AUC(baseline) on the common held-out tracks
        if "baseline" in preds:
            te_b, s_b = preds["baseline"]
            for vn in ("gbl_replace", "aggregate", "forward_only"):
                if vn not in preds:
                    continue
                te_v, s_v = preds[vn]
                com, ib, iv = np.intersect1d(te_b, te_v, return_indices=True)
                sub = brng.choice(len(com), min(len(com), 400_000), replace=False)
                yy, sb, sv = y[com[sub]], s_b[ib[sub]], s_v[iv[sub]]
                ds = []
                for _ in range(20):
                    b = brng.integers(0, len(sub), len(sub))
                    ds.append(roc_auc_score(yy[b], sv[b]) - roc_auc_score(yy[b], sb[b]))
                row[vn]["delta_vs_baseline"] = round(float(np.mean(ds)), 4)
                row[vn]["delta_sd"] = round(float(np.std(ds)), 4)
        res["taggers"][name] = row
        print(f"{name:20s} " + "  ".join(
            f"{vn} {row[vn]['auc']:.4f}" + (f" ({row[vn]['delta_vs_baseline']:+.4f}+-{row[vn]['delta_sd']:.4f})"
                                             if row.get(vn) and "delta_vs_baseline" in row[vn] else "")
            for vn in ("baseline", "gbl_replace", "aggregate", "forward_only") if row.get(vn)), flush=True)
        json.dump(res, open(f"{FEATD}/mva_gbl_study.json", "w"), indent=1)
    print(f"done in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
