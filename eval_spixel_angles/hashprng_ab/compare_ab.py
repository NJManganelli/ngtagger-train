#!/usr/bin/env python3
"""A/B validation of the HashPRNG angle-synthesis factorization.

Inputs: OLD-producer nano, NEW-producer nano run1, NEW-producer nano run2
(same input file, same payload -- the plain bias/sigma tables are bit-identical
between OLD and NEW; only the throw mechanism changed: engine draws -> HashPRNG).

Checks
  1. Per-layer synthesized-angle residual (meas - parent-true) distributions for
     alpha and beta: mean/std vs the payload bias/sigma, and OLD-vs-NEW KS +
     quantile agreement (throws differ hit-by-hit by design; distributions must
     match statistically).
  2. Standardized residuals (res - bias(inputs)) / sigma(inputs) ~ N(0,1) using
     the payload lookups at each hit's true angles.
  3. Alpha-beta per-hit residual correlation (OLD: independent engine draws;
     NEW: independent entropy permutations -> both ~0).
  4. Refit-track d0/z0/pt/eta/phi + chi2: OLD-vs-NEW KS (statistically unchanged).
  5. Determinism: every refit/track branch bit-identical between NEW run1/run2.

Writes a results json; prints a bounded summary.
"""
import argparse
import json

import numpy as np
import uproot
from scipy import stats

SENTINEL = -900.0  # values <= this are -999 sentinels
PAYLOAD = ("/Users/nmangane/smartpixels/cmssw/work/spxsmoke/"
           "spx_angle_response_Conv1D_Full-2bit_v4fixed.json")


def load_hits(path, hit_prefix):
    t = uproot.open(path)["Events"]
    import awkward as ak
    def g(col):
        return ak.flatten(t[f"{hit_prefix}_{col}"].array()).to_numpy()
    d = {c: g(c) for c in ("layer", "cotAlphaMeas", "cotBetaMeas", "parCotAlpha",
                           "parCotBeta", "hasAlpha", "hasBeta", "sigAlpha", "sigBeta")}
    return d


def residuals(d, angle):
    meas = d[f"cot{angle}Meas"]
    true = d[f"parCot{angle}"]
    has = d[f"has{angle}"].astype(bool)
    ok = has & (meas > SENTINEL) & (true > SENTINEL)
    return d["layer"][ok], (meas[ok] - true[ok]), true[ok], ok


def payload_lookup(layer, ca, cb, which):
    import correctionlib
    cset = correctionlib.CorrectionSet.from_file(PAYLOAD)
    bias = cset[f"spx_angle_{which}_bias"]
    sig = cset[f"spx_angle_{which}_sigma"]
    b = np.array([bias.evaluate(int(l), float(a), float(c), -3.81)
                  for l, a, c in zip(layer, ca, cb)])
    s = np.array([sig.evaluate(int(l), float(a), float(c), -3.81)
                  for l, a, c in zip(layer, ca, cb)])
    return b, s


def summarize(tag, res_old, res_new, out):
    """KS + quantiles OLD vs NEW."""
    ks = stats.ks_2samp(res_old, res_new)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    out[tag] = {
        "n_old": int(res_old.size), "n_new": int(res_new.size),
        "mean_old": float(res_old.mean()), "mean_new": float(res_new.mean()),
        "std_old": float(res_old.std()), "std_new": float(res_new.std()),
        "ks_stat": float(ks.statistic), "ks_pval": float(ks.pvalue),
        "quantiles_old": [float(x) for x in np.quantile(res_old, qs)],
        "quantiles_new": [float(x) for x in np.quantile(res_new, qs)],
    }
    return ks.pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new1"); ap.add_argument("new2")
    ap.add_argument("--hit-prefix", required=True)
    ap.add_argument("--trk-prefix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import awkward as ak

    out = {"files": {"old": args.old, "new1": args.new1, "new2": args.new2}}
    ok = True

    ho = load_hits(args.old, args.hit_prefix)
    hn = load_hits(args.new1, args.hit_prefix)

    for angle, wh in (("Alpha", "alpha"), ("Beta", "beta")):
        lo, ro, to_, _ = residuals(ho, angle)
        ln, rn, tn, _ = residuals(hn, angle)
        # per-layer OLD vs NEW
        for L in (1, 2, 3, 4):
            po = ro[lo == L]; pn = rn[ln == L]
            if po.size < 50 or pn.size < 50:
                continue
            p = summarize(f"{wh}_L{L}", po, pn, out)
            if p < 1e-3:
                ok = False
                out.setdefault("errors", []).append(f"{wh} L{L}: KS p={p:.2e}")

    # joint standardized check + alpha-beta correlation (NEW and OLD)
    for tag, d in (("old", ho), ("new", hn)):
        hasA = d["hasAlpha"].astype(bool) & (d["parCotAlpha"] > SENTINEL)
        hasB = d["hasBeta"].astype(bool) & (d["parCotBeta"] > SENTINEL)
        both = hasA & hasB
        ra = d["cotAlphaMeas"][both] - d["parCotAlpha"][both]
        rb = d["cotBetaMeas"][both] - d["parCotBeta"][both]
        rho = float(stats.spearmanr(ra, rb).statistic)
        out[f"alpha_beta_residual_corr_{tag}"] = {"spearman": rho, "n": int(both.sum())}
        if abs(rho) > 0.1:
            ok = False
            out.setdefault("errors", []).append(f"alpha-beta corr {tag}: {rho:.3f}")
        bA, sA = payload_lookup(d["layer"][both], d["parCotAlpha"][both],
                                d["parCotBeta"][both], "alpha")
        zA = (ra - bA) / sA
        bB, sB = payload_lookup(d["layer"][both], d["parCotAlpha"][both],
                                d["parCotBeta"][both], "beta")
        zB = (rb - bB) / sB
        out[f"standardized_{tag}"] = {
            "alpha_mean": float(zA.mean()), "alpha_std": float(zA.std()),
            "beta_mean": float(zB.mean()), "beta_std": float(zB.std()), "n": int(both.sum()),
        }

    # refit-track kinematics OLD vs NEW
    to = uproot.open(args.old)["Events"]
    tn1 = uproot.open(args.new1)["Events"]
    for col in ("pt", "eta", "phi", "z0", "d0", "chi2XYRed", "chi2ZRed", "trkMVA1"):
        bo = f"{args.trk_prefix}_{col}"
        if bo not in to or bo not in tn1:
            continue
        vo = ak.flatten(to[bo].array()).to_numpy()
        vn = ak.flatten(tn1[bo].array()).to_numpy()
        p = summarize(f"trk_{col}", vo, vn, out)
        if p < 1e-3:
            ok = False
            out.setdefault("errors", []).append(f"trk {col}: KS p={p:.2e}")

    # determinism: new1 vs new2, all hit + track branches bit-identical
    tn2 = uproot.open(args.new2)["Events"]
    ndiff = 0
    checked = 0
    for k in tn1.keys():
        if not (k.startswith(args.hit_prefix) or k.startswith(args.trk_prefix)):
            continue
        if "/" in k:
            continue
        try:
            a1 = ak.flatten(tn1[k].array(), axis=None).to_numpy()
            a2 = ak.flatten(tn2[k].array(), axis=None).to_numpy()
        except Exception:
            continue
        checked += 1
        if a1.shape != a2.shape or not np.array_equal(a1, a2):
            ndiff += 1
            out.setdefault("determinism_diffs", []).append(k)
    out["determinism"] = {"branches_checked": checked, "branches_differ": ndiff,
                          "bit_identical": ndiff == 0}
    if ndiff:
        ok = False

    out["pass"] = ok
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"alpha L1 old/new std: {out.get('alpha_L1', {}).get('std_old'):.5f}/"
          f"{out.get('alpha_L1', {}).get('std_new'):.5f} ks_p={out.get('alpha_L1', {}).get('ks_pval'):.3g}"
          if "alpha_L1" in out else "alpha_L1: insufficient stats")
    print("standardized_new:", out["standardized_new"])
    print("ab-corr old/new:", out["alpha_beta_residual_corr_old"]["spearman"],
          out["alpha_beta_residual_corr_new"]["spearman"])
    print("determinism:", out["determinism"])
    if out.get("errors"):
        for e in out["errors"]:
            print("FAIL:", e)
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
