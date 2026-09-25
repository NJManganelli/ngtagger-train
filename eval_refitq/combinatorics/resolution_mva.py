"""Per-track resolution estimate: predict the WIDTH of the truth residual.

The quality question downstream is not "is this track real" but "how much
should the tagger believe this d0". That is a conditional width -- the spread of
p(d0_fit - d0_true | features) -- and it is estimated here by a quantile pair,
sigma_hat = (q84.1 - q15.9)/2, which needs no Gaussian assumption and is not
dragged by the contaminated tail.

WHY NOT THE FIT'S OWN COVARIANCE. sqrt(var_d0) is the fit's CLAIMED error: a
function of the hit pattern and the assumed per-hit errors, never of the
measured residuals, so it cannot know a hit came from another particle. It is
measured to be optimistic by ~3.5x even for perfectly clean tracks, so it is
demoted to an input feature here and never used to normalise the target.

WHY THE RESIDUAL IS OWNER-REFERENCED. The exported d_d0 is taken against the
TP of the SEED cluster, which is not the majority owner for a large fraction of
tracks; training on it would teach the model the wrong particle's d0.

THE d0 FEATURE IS A TRAP, and it is tested rather than assumed. A model given
|d0| can learn "displaced implies contaminated implies distrust", because this
sample is prompt-dominated -- which is exactly backwards for a b-tagger, whose
signal IS displacement. The ablation below trains with and without the d0-like
features and reports what they buy; if they buy much, they are buying it by
penalising the tracks the tagger most wants.

Consumes the table written by hit_composition.py --npz.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kf_emulation as KF   # noqa: E402

# the base table's residuals are owner-referenced as of census format 16
TARGETS = {"d_d0": ("sigma(d0)", 1e4, "um"),
           "d_z0": ("sigma(z0)", 1e4, "um"),
           "d_kappa": ("sigma(1/pT)", 1.0, "1/GeV")}
# d0-like features, held out in the ablation
D0_FEATS = ("d0", "d0_over_sigma")


def sig(x, nmin=100):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < nmin:
        return float("nan")
    q = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q[1] - q[0]))


def nll(r, s):
    """Mean Gaussian negative log-likelihood -- the score a tagger inherits."""
    s = np.maximum(s, 1e-12)
    return float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * (r / s) ** 2))


def fit_quantiles(Xtr, ytr, Xs, seed=1):
    """Conditional 15.9 / 50 / 84.1 percentiles of the residual, per track.

    THE MEDIAN IS NOT OPTIONAL. The residual has a conditional BIAS -- a wrong
    hit pulls d0 in a direction set by where it sits, and the kappa-d0
    correlation adds more -- so a band can be correctly narrow about its own
    centre while the spread of centres across tracks is an order of magnitude
    wider. Judging a width by an UNCENTRED pull charges the model for a bias it
    was never asked to predict, which is exactly how the first version of this
    script produced predicted 7 um against observed 68 um and a Gaussian NLL of
    1e20.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor as GB
    out = {}
    for q in (0.15865, 0.5, 0.84135):
        g = GB(loss="quantile", quantile=q, max_iter=200, learning_rate=0.1,
               random_state=seed).fit(Xtr, ytr)
        out[q] = [g.predict(X) for X in Xs]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", default="results/hit_composition.npz")
    ap.add_argument("--dr-only", action="store_true")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=False)
    R, cols = z["rows"], [str(x) for x in z["cols"]]
    c = {n: i for i, n in enumerate(cols)}
    if a.dr_only:
        R = R[R[:, c["dr"]] > 0]
    feats = list(dict.fromkeys(f for f in KF.MVA_FEATURES if f in c))
    print(f"{len(R):,} tracks, {len(feats)} features")

    # SPLIT BY EVENT, not by track: tracks in one event share clusters and are
    # duplicates of each other by construction, so a random track split leaks
    # the answer across the boundary.
    ev = R[:, c["event"]]
    uev = np.unique(ev)
    te_ev = set(uev[::3].tolist())
    is_te = np.isin(ev, list(te_ev))
    print(f"  {len(uev)} events -> {is_te.sum():,} test / "
          f"{(~is_te).sum():,} train tracks")

    J = {"n_tracks": int(len(R)), "features": feats}
    pred = {}
    for tgt, (name, scale, unit) in TARGETS.items():
        r = R[:, c[tgt]] * scale
        # a track with no majority owner has no residual and no label; the
        # feature columns are handed to the tree with NaNs intact
        ok = np.isfinite(r)
        tr, te = ok & ~is_te, ok & is_te
        if tr.sum() < 20000 or te.sum() < 5000:
            print(f"{name}: too few labelled tracks, skipped")
            continue
        X = np.nan_to_num(R[:, [c[f] for f in feats]], posinf=0, neginf=0)
        Qs = fit_quantiles(X[tr], r[tr], [X[te], X])
        q16, q50, q84 = (Qs[k][0] for k in (0.15865, 0.5, 0.84135))
        s_te = np.maximum(0.5 * (q84 - q16), 1e-9)
        s_all = np.maximum(0.5 * (Qs[0.84135][1] - Qs[0.15865][1]), 1e-9)
        pred[tgt] = s_all
        rt = r[te]
        cen = rt - q50
        print(f"\n=== {name} ===  {tr.sum():,} train / {te.sum():,} test")
        print(f"  observed spread of the residual   {sig(rt):>10.4g} {unit}")
        print(f"  predicted sigma, 10-90 percentile "
              f"{np.percentile(s_te, 10):.4g} - {np.percentile(s_te, 90):.4g} {unit}")

        # ---- CALIBRATION: pull width per predicted-sigma decile ----------
        cov = float(np.mean((rt >= q16) & (rt <= q84)))
        print(f"  coverage of the predicted band: {100 * cov:.1f}% "
              f"(68.3% = calibrated)")
        print(f"  conditional bias |median|, 10-90 pct: "
              f"{np.percentile(np.abs(q50), 10):.4g} - "
              f"{np.percentile(np.abs(q50), 90):.4g} {unit}")
        print("  decile of predicted sigma: predicted  observed  observed")
        print("                                        (raw)     (centred)"
              "   coverage")
        d = np.percentile(s_te, np.arange(0, 101, 10))
        rows = []
        for k in range(10):
            m = (s_te >= d[k]) & (s_te <= d[k + 1] if k == 9 else s_te < d[k + 1])
            if m.sum() < 100:
                continue
            ck = float(np.mean((rt[m] >= q16[m]) & (rt[m] <= q84[m])))
            print(f"    {k + 1:>2}  {np.median(s_te[m]):>16.4g} {sig(rt[m]):>9.4g} "
                  f"{sig(cen[m]):>11.4g} {100 * ck:9.1f}%")
            rows.append({"decile": k + 1, "pred": float(np.median(s_te[m])),
                         "obs_raw": sig(rt[m]), "obs_centred": sig(cen[m]),
                         "coverage": ck, "n": int(m.sum())})

        # ---- PROPER SCORE against the alternatives -----------------------
        base = {
            "one global sigma": np.full(int(te.sum()), sig(r[tr] - np.median(r[tr]))),
            "sigma by TRUE wrong-hit class (oracle)": None,
            "fit's claimed sigma": None,
            "predicted per track": s_te,
        }
        nw = R[:, c["n_it_wrong"]] + R[:, c["n_ot_wrong"]]
        cls = np.clip(nw, 0, 3)
        lut = {k: sig(r[tr & (cls == k)]) for k in range(4)}
        base["sigma by TRUE wrong-hit class (oracle)"] = np.array(
            [lut.get(int(k), np.nan) for k in cls[te]])
        kf_col = {"d_d0": "sig_kf_d0"}.get(tgt)
        if kf_col and kf_col in c:
            base["fit's claimed sigma"] = R[te, c[kf_col]] * scale
        else:
            base.pop("fit's claimed sigma", None)
        print("  mean Gaussian NLL of the CENTRED residual (lower is better; "
              "the gap is the log-likelihood a tagger gains)")
        J.setdefault("targets", {})[tgt] = {"calibration": rows, "nll": {}}
        for k, s in base.items():
            good = np.isfinite(s) & (s > 0)
            v = nll(cen[good], s[good])
            print(f"    {k:<42} {v:9.3f}   (centred pull width "
                  f"{sig(cen[good] / s[good]):5.2f}, raw "
                  f"{sig(rt[good] / s[good]):5.2f})")
            J["targets"][tgt]["nll"][k] = round(v, 4)

        # ---- ABLATION: what the d0-like features buy ---------------------
        # d0 only: that is the target where "displaced implies distrust" would
        # do real damage to a tagger, and it halves the fit count.
        if tgt != "d_d0":
            continue
        keep = [f for f in feats if f not in D0_FEATS]
        Xk = np.nan_to_num(R[:, [c[f] for f in keep]], posinf=0, neginf=0)
        Q2 = fit_quantiles(Xk[tr], r[tr], [Xk[te]])
        a16, a50, a84 = (Q2[k][0] for k in (0.15865, 0.5, 0.84135))
        s2 = np.maximum(0.5 * (a84 - a16), 1e-9)
        cov2 = float(np.mean((rt >= a16) & (rt <= a84)))
        print(f"  without {', '.join(D0_FEATS)}: NLL {nll(rt - a50, s2):.3f} "
              f"and coverage {100 * cov2:.1f}%, against {nll(cen, s_te):.3f} "
              f"and {100 * cov:.1f}% with them")
        J["targets"][tgt]["ablation_no_d0_feats"] = {
            "nll": round(nll(rt - a50, s2), 4), "coverage": cov2}

        # ---- OPERATING POINTS: selecting the good-resolution tracks ------
        print("  keep the tracks with the smallest predicted sigma:")
        print("     kept    actual sigma   median predicted   coverage")
        for f in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
            thr = np.percentile(s_te, 100 * f)
            m = s_te <= thr
            print(f"    {100 * f:4.0f}%  {sig(rt[m]):>12.4g}   "
                  f"{np.median(s_te[m]):>16.4g}   "
                  f"{100 * np.mean((rt[m] >= q16[m]) & (rt[m] <= q84[m])):7.1f}%")
            J["targets"][tgt].setdefault("operating", []).append(
                {"kept": f, "actual_sigma": sig(rt[m]),
                 "median_pred": float(np.median(s_te[m]))})
        J["targets"][tgt]["coverage"] = cov

    # ---- does ONE scalar serve all three parameters? ---------------------
    if len(pred) > 1:
        print("\nrank correlation between the three predicted widths "
              "(1.00 would mean one scalar suffices)")
        ks = list(pred)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                m = np.isfinite(pred[ks[i]]) & np.isfinite(pred[ks[j]])
                a_ = np.argsort(np.argsort(pred[ks[i]][m]))
                b_ = np.argsort(np.argsort(pred[ks[j]][m]))
                rho = float(np.corrcoef(a_, b_)[0, 1])
                print(f"  {TARGETS[ks[i]][0]:<12} vs {TARGETS[ks[j]][0]:<12}"
                      f" rho = {rho:6.3f}")
                J.setdefault("rho", {})[f"{ks[i]}|{ks[j]}"] = round(rho, 4)
    if a.out:
        json.dump(J, open(a.out, "w"), indent=1, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
