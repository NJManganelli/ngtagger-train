"""Track-quality MVA on the combined IT+OT track collection, two targets.

WHY A SECOND TARGET. The OT ships a track-quality MVA whose job is real versus
fake: a GBDT on seven binned features (tanl, z0_scaled, bendchi2_bin, nstub,
nlaymiss_interior, chi2rphi_bin, chi2rz_bin). That question is nearly solved
here by the fit alone. The question that is NOT solved, and that a downstream
impact-parameter tagger actually needs answered, is HOW MANY HITS ARE WRONG,
because that is what sets the d0 resolution it has to trust:

    MEASURED, true wrong-hit count against sigma(d0)
        0 wrong   53 um        2 wrong  417 um
        1 wrong   63 um       3+ wrong  667 um

One wrong hit is absorbed by the fit; two is catastrophic. So the useful
operating split is "at most one" against "two or more", not clean against
contaminated, and a per-track prediction of it lets a tagger weight each track's
d0 instead of applying one resolution to all of them.

THE LABELS ARE FREE. The census already knows which TrackingParticle every
attached hit belongs to, so the wrong-hit count needs no new machinery.

WHAT THE COMBINED SCENARIO ADDS over the OT's feature set: every SmartPixels
cluster carries its own direction, so the angle pulls localise a suspect hit
(max_angle_pull, second_angle_pull) instead of averaging into one summed
bendchi2, and chi2_rphi is split by system so IT-consistent-but-OT-inconsistent
is distinguishable from the reverse.

Run tp_findability.py with --features-per-chunk to produce the training rows.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_findability as TF   # noqa: E402
import kf_emulation as KF     # noqa: E402


def robust_sigma(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return float("nan")
    q = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q[1] - q[0]))


def load_rows(cache_dir):
    ds = sorted(glob.glob(f"{cache_dir}/tpcensus_*"))
    if not ds:
        raise SystemExit(f"no census under {cache_dir}")
    C = TF.load(ds[-1], verbose=False, with_tracks=True)
    if not C["tracks"]:
        raise SystemExit("census has no track rows; rerun with --export-tracks")
    T = np.concatenate([C["tracks"][i] for i in sorted(C["tracks"])])
    col = {c: j for j, c in enumerate(C["track_cols"])}
    # The MVA may see only the quality columns. Identity, fitted kinematics and
    # every truth column are excluded by name, so a column added to TRACK_COLS
    # for the interactive page cannot leak into training by accident.
    fidx = [col[c] for c in KF.MVA_FEATURES]
    X = T[:, fidx]
    Y = np.stack([T[:, col["n_wrong"]], T[:, col["is_clean"]],
                  T[:, col["tp_pt"]], T[:, col["d_d0"]]], axis=1)
    seed = T[:, col["seed_idx"]]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y[:, 0])
    return X[ok], Y[ok], seed[ok], C


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default=str(Path(__file__).parent / "cache"))
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier as GB
    from sklearn.metrics import roc_auc_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance

    X, Y, seed, C = load_rows(a.cache_dir)
    nm = list(KF.MVA_FEATURES)
    nw, d0res = Y[:, 0], Y[:, 3]
    print(f"{len(X):,} tracks, {len(nm)} features, "
          f"{C['n_events']} events, {len(C['seed_tags'])} seeds")
    print("\nsigma(d0) by TRUE wrong-hit count")
    for k in (0, 1, 2, 3):
        m = (nw == k) if k < 3 else (nw >= 3)
        if m.sum() < 30:
            continue
        print(f"  n_wrong {'>=3' if k == 3 else k}: {int(m.sum()):>7,} "
              f"sigma(d0) = {1e4 * robust_sigma(d0res[m]):>6.0f} um")

    out = {"n_tracks": int(len(X)), "n_events": C["n_events"],
           "features": list(nm), "sigma_d0_by_true_nwrong": {}}
    for k in (0, 1, 2, 3):
        m = (nw == k) if k < 3 else (nw >= 3)
        if m.sum() >= 30:
            out["sigma_d0_by_true_nwrong"][str(k)] = \
                [int(m.sum()), 1e4 * robust_sigma(d0res[m])]

    # ---- target (a): clean vs contaminated -----------------------------
    yb = (nw == 0).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, yb, test_size=a.test_frac,
                                          random_state=1, stratify=yb)
    print("\n(a) CLEAN vs contaminated")
    singles = {}
    for f in ("chi2_rphi_per_layer", "chi2_scaled", "max_angle_pull",
              "chi2_angle_per_cl"):
        if f in nm:
            s = roc_auc_score(yte, -Xte[:, nm.index(f)])
            singles[f] = s
            print(f"    {f:<24} alone   AUC {s:.4f}")
    g = GB(max_iter=250, learning_rate=0.1, random_state=1).fit(Xtr, ytr)
    auc_all = roc_auc_score(yte, g.predict_proba(Xte)[:, 1])
    # rank_score MUST be dropped from the no-angle ablation: rank_score is
    # nhit - 0.02*chi2/nhit - w_angle*chi2_angle/n_angle, so it EMBEDS the angle
    # chi2 by construction and is not caught by a name filter. Leaving it in
    # made the "without angles" model still see the angles, which is why the
    # ablation read as costing 0.0001 AUC while angle features occupied the top
    # three importances -- two results that cannot both be true.
    ANGLE_EMBEDDING = ("angle", "rank_score")
    noang = [i for i, n in enumerate(nm)
             if not any(k in n for k in ANGLE_EMBEDDING)]
    g2 = GB(max_iter=250, learning_rate=0.1, random_state=1).fit(Xtr[:, noang], ytr)
    auc_noang = roc_auc_score(yte, g2.predict_proba(Xte[:, noang])[:, 1])
    print(f"    GBDT, all features       AUC {auc_all:.4f}")
    print(f"    GBDT, no angle features  AUC {auc_noang:.4f}   "
          f"(angles worth {auc_all - auc_noang:+.4f})")
    print(f"      [no-angle set also drops rank_score, which embeds "
          f"chi2_angle by construction]")
    # the operationally useful split: 0 and 1 wrong hits give nearly the same
    # d0, while >= 2 is a different regime, so a binary target is both easier
    # and closer to what a tagger would act on than the 3-class one
    yb2 = (nw <= 1).astype(int)
    X2tr, X2te, y2tr, y2te, _, D2te = train_test_split(
        X, yb2, d0res, test_size=a.test_frac, random_state=1, stratify=yb2)
    g2b = GB(max_iter=250, learning_rate=0.1, random_state=1).fit(X2tr, y2tr)
    p2 = g2b.predict_proba(X2te)[:, 1]
    print(f"\n(a2) <=1 vs >=2 wrong hits   AUC {roc_auc_score(y2te, p2):.4f}")
    for thr in (0.5, 0.8, 0.95):
        m = p2 >= thr
        if m.sum() < 50:
            continue
        print(f"     cut {thr:.2f}: keeps {100 * m.mean():5.1f}% of tracks, "
              f"sigma(d0) = {1e4 * robust_sigma(D2te[m]):>5.0f} um, "
              f"purity {100 * (y2te[m] == 1).mean():.0f}%")
    out["binary_le1"] = {"auc": float(roc_auc_score(y2te, p2))}
    out["binary"] = {"singles": singles, "auc_all": auc_all,
                     "auc_no_angles": auc_noang}

    # ---- target (b): 0 / 1 / 2+ ----------------------------------------
    y3 = np.clip(nw, 0, 2).astype(int)
    X3tr, X3te, y3tr, y3te, _, D3te = train_test_split(
        X, y3, d0res, test_size=a.test_frac, random_state=1, stratify=y3)
    g3 = GB(max_iter=250, learning_rate=0.1, random_state=1).fit(X3tr, y3tr)
    p3 = g3.predict(X3te)
    print(f"\n(b) 3-class n_wrong 0 / 1 / 2+    accuracy {(p3 == y3te).mean():.3f}")
    print("    confusion (rows true, cols predicted)")
    print("   ", str(confusion_matrix(y3te, p3)).replace("\n", "\n    "))
    print("\n    sigma(d0) by PREDICTED class -- the calibration a tagger uses")
    cal = {}
    for k in (0, 1, 2):
        m = p3 == k
        if m.sum() < 30:
            continue
        sd = 1e4 * robust_sigma(D3te[m])
        pur = 100 * float((y3te[m] == k).mean())
        cal[str(k)] = [int(m.sum()), sd, pur]
        print(f"      pred {k}{'+' if k == 2 else ' '}: {int(m.sum()):>7,} "
              f"sigma(d0) = {sd:>6.0f} um   purity {pur:.0f}%")
    out["threeclass"] = {"accuracy": float((p3 == y3te).mean()),
                         "sigma_d0_by_pred": cal}

    r = permutation_importance(g, Xte[:4000], yte[:4000], n_repeats=3,
                               random_state=1, scoring="roc_auc")
    print("\ntop features (permutation importance, binary task)")
    imp = {}
    for i in np.argsort(-r.importances_mean)[:10]:
        imp[nm[i]] = float(r.importances_mean[i])
        print(f"   {nm[i]:<24}{r.importances_mean[i]:+.4f}")
    out["importance"] = imp
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
