#!/usr/bin/env python
"""Per-field codes-per-octave constants for the 4-bit refit chi2 delta encoding.

chi2_bitwidth_study.py established that 4 bits suffices and that the deployed
k=2 is mis-sized. This picks a SEPARATE k for each of X, Y, Alpha, Beta, since
their dynamic ranges differ (28.2 / 22.3 / 23.4 / 22.7 octaves) and there is no
structural cost to four constants instead of one.

Two measurements per (field, k):

  isolated : base features + ONLY this field's code. Most sensitive to that
             field's own resolution - no other chi2 field can mask the loss.
             This is the k-selection metric.
  in situ  : base + all four fields, this one quantized, the other three float.
             The deployment-relevant number, reported as a cross-check (a field
             whose information is partly redundant will look cheaper here).

A final run puts all four at their chosen k simultaneously and compares to all
four float, which also tests that the per-field choices do not interact.

Label is 'clean' (every accepted IT hit came from the track's own TP): it has
~17k negatives versus ~1k for 'genuine', and it is the label these features
actually respond to. See docs/refit-chi2-bitwidth-study.md.

    pixi run python eval_refitq/quantstudy/chi2_perfield_k.py
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from chi2_bitwidth_study import (DEFAULT_FILES, DIMS, auc, build_base, load,
                                 occ_bits, oof_scores, paired_bootstrap, q_log)
from chi2_code_efficiency import entropy_bits, spearman

NBITS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument("--k-grid", nargs="+", type=float,
                    default=[0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--bits", type=int, default=NBITS)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default="eval_refitq/quantstudy")
    args = ap.parse_args()

    flat, sums, _ = load(args.files)
    m = flat["var_spxRefitPerformed"] > 0
    d = {k: sums[k][m] for k in DIMS}
    fl = {k: v[m] for k, v in flat.items()}
    y = ((fl["n_wrong_hits"] + fl["n_noise_hits"] == 0)
         & (fl["n_acc_hits"] > 0)).astype(np.int32)
    base, base_names = build_base(fl)
    nb = args.bits

    print(f"tracks={len(y)}  pos={y.sum()}  neg={(y == 0).sum()}  bits={nb}")

    def fit(extra_cols):
        X = base if not extra_cols else np.column_stack(
            [base] + [np.asarray(v, float) for v in extra_cols])
        s = oof_scores(X, y, n_folds=args.n_folds, seed=args.seed, quick=args.quick)
        return s, auc(y, s)

    res = {"files": args.files, "bits": nb, "label": "clean",
           "base_features": base_names, "n_pos": int(y.sum()),
           "n_neg": int((y == 0).sum()), "fields": {}}

    # float references
    s_float_iso = {}
    for f in DIMS:
        s_float_iso[f], a = fit([d[f]])
        res["fields"][f] = {"float_isolated_auc": a, "scan": []}
        print(f"  float isolated {f:<5} AUC={a:.5f}")
    s_float_all, a_float_all = fit([d[f] for f in DIMS])
    res["float_all_auc"] = a_float_all
    print(f"  float all four   AUC={a_float_all:.5f}\n")

    # a pure range-matching prediction, for comparison with the measured optimum
    print("range-matching prediction k* = (2^n - 1)/octaves(p1..p99.9):")
    for f in DIMS:
        nz = d[f][d[f] > 0]
        oct_span = float(np.log2(np.quantile(nz, 0.999) / max(np.quantile(nz, 0.01), 1e-9)))
        k_pred = ((1 << nb) - 1) / oct_span
        res["fields"][f]["octaves_p1_p999"] = oct_span
        res["fields"][f]["k_range_matched"] = k_pred
        print(f"  {f:<5} octaves={oct_span:5.1f}  k*={k_pred:.2f}")

    print(f"\nper-field k scan at {nb} bits (label 'clean')")
    print("  field   k     H(code) sat%   rho     isolated dAUC vs float [95% CI]")
    for f in DIMS:
        for k in args.k_grid:
            c = q_log(d[f], nb, k)
            hi = (1 << nb) - 1
            s_q, a_q = fit([c])
            mu, lo, up = paired_bootstrap(y, s_q, s_float_iso[f], args.n_boot, args.seed)
            row = {"k": k, "auc": a_q, "dauc_vs_float": mu, "ci_lo": lo, "ci_hi": up,
                   "entropy_bits": entropy_bits(c),
                   "saturation_frac": float((c == hi).mean()),
                   "code0_frac": float((c == 0).mean()),
                   "spearman_vs_float": spearman(c, d[f])}
            res["fields"][f]["scan"].append(row)
            print(f"  {f:<5} {k:<5} {row['entropy_bits']:6.2f} "
                  f"{row['saturation_frac']*100:5.2f} "
                  f"{row['spearman_vs_float']:6.3f}  {mu:+.5f} [{lo:+.5f}, {up:+.5f}]")

    # choose per field: the smallest-|dAUC| k, breaking ties toward less
    # saturation (a populated top code is an information sink even where this
    # sample cannot resolve the loss)
    chosen = {}
    for f in DIMS:
        scan = res["fields"][f]["scan"]
        best = min(scan, key=lambda r: (round(abs(r["dauc_vs_float"]), 4),
                                        r["saturation_frac"]))
        chosen[f] = best["k"]
        res["fields"][f]["k_chosen"] = best["k"]
        print(f"\nchosen k[{f}] = {best['k']}  (isolated dAUC {best['dauc_vs_float']:+.5f}, "
              f"sat {best['saturation_frac']*100:.2f}%, H={best['entropy_bits']:.2f}/{nb})")

    # in-situ cross-check at the chosen k, and the joint run
    print("\nin-situ check (this field quantized at its chosen k, others float):")
    for f in DIMS:
        cols = [q_log(d[g], nb, chosen[g]) if g == f else d[g] for g in DIMS]
        s_q, a_q = fit(cols)
        mu, lo, up = paired_bootstrap(y, s_q, s_float_all, args.n_boot, args.seed)
        res["fields"][f]["in_situ_dauc"] = {"mean": mu, "lo": lo, "hi": up, "auc": a_q}
        print(f"  {f:<5} AUC={a_q:.5f}  dAUC vs all-float {mu:+.5f} [{lo:+.5f}, {up:+.5f}]")

    print("\njoint: all four at chosen k vs all four float")
    s_all, a_all = fit([q_log(d[f], nb, chosen[f]) for f in DIMS])
    mu, lo, up = paired_bootstrap(y, s_all, s_float_all, args.n_boot, args.seed)
    res["joint_chosen"] = {"k": chosen, "auc": a_all, "dauc_vs_float": mu,
                           "ci_lo": lo, "ci_hi": up}
    print(f"  AUC={a_all:.5f}  dAUC={mu:+.5f} [{lo:+.5f}, {up:+.5f}]")

    # for contrast: the single deployed constant everywhere
    s_k2, a_k2 = fit([q_log(d[f], nb, 2.0) for f in DIMS])
    mu2, lo2, up2 = paired_bootstrap(y, s_k2, s_float_all, args.n_boot, args.seed)
    res["joint_k2"] = {"auc": a_k2, "dauc_vs_float": mu2, "ci_lo": lo2, "ci_hi": up2}
    print(f"  (single deployed k=2 everywhere: AUC={a_k2:.5f}  "
          f"dAUC={mu2:+.5f} [{lo2:+.5f}, {up2:+.5f}])")

    out = pathlib.Path(args.outdir) / "chi2_perfield_k.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nchosen constants: {chosen}\nwrote {out}")


if __name__ == "__main__":
    main()
