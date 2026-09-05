#!/usr/bin/env python
"""Label-free companion to chi2_bitwidth_study.py: is the CODE itself efficient?

With ~1e3 negatives the BDT AUC differences between quantizer settings are at
or below the noise floor, so the bit-width decision cannot rest on AUC alone.
This script answers the part that does not depend on labels at all:

  * How much ENTROPY does an n-bit code actually carry on this distribution?
    A 4-bit field is only worth 4 bits if the 16 codes are populated; if 40% of
    tracks land in one code, the field is carrying ~2 bits and the register
    width is wasted. Efficiency = H(code) / n.
  * How much of the FLOAT feature's own ordering survives quantization?
    Reported as the Mann-Whitney agreement between the quantized code and the
    float value (a self-AUC that needs no truth labels), plus the saturation
    and bottom-code occupancies.
  * What dynamic range must a code cover, per dimension, post-guard?

Run after (or instead of) the BDT sweep:
    pixi run python eval_refitq/quantstudy/chi2_code_efficiency.py
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from chi2_bitwidth_study import (DEFAULT_FILES, DIMS, SENTINEL, load, q_log,
                                 q_quantile, validate_identity)


def entropy_bits(code: np.ndarray) -> float:
    _, cnt = np.unique(code, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument("--bits", nargs="+", type=int, default=[3, 4, 5, 6])
    ap.add_argument("--per-octave", nargs="+", type=float,
                    default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--outdir", default="eval_refitq/quantstudy")
    args = ap.parse_args()

    flat, sums, _ = load(args.files)
    ident = validate_identity(flat, sums)
    m = flat["var_spixRefitPerformed"] > 0
    d = {k: sums[k][m] for k in DIMS}
    nacc = np.maximum(flat["var_spixNAcceptedHits"][m].astype(float), 1.0)

    res = {"files": args.files, "identity": ident, "n_tracks": int(m.sum()),
           "dynamic_range": {}, "codes": [], "reduced": {}}

    print("== dynamic range of the per-dimension chi2 deltas (post-guard) ==")
    print("  dim     zero%     p50       p90       p99      p99.9      max     "
          "octaves(p1..p99.9)")
    for k in DIMS:
        v = d[k]
        nz = v[v > 0]
        qs = np.quantile(v, [0.5, 0.9, 0.99, 0.999, 1.0])
        oct_span = float(np.log2(np.quantile(nz, 0.999) / max(np.quantile(nz, 0.01), 1e-9)))
        res["dynamic_range"][k] = {
            "frac_zero": float((v == 0).mean()), "p50": float(qs[0]),
            "p90": float(qs[1]), "p99": float(qs[2]), "p999": float(qs[3]),
            "max": float(qs[4]), "octaves_p1_to_p999": oct_span}
        print(f"  {k:>5} {(v==0).mean()*100:7.2f} {qs[0]:9.3g} {qs[1]:9.3g} "
              f"{qs[2]:9.3g} {qs[3]:9.3g} {qs[4]:9.3g}   {oct_span:6.1f}")

    print("\n== code efficiency for the position deltas (X shown; Y in JSON) ==")
    print("  bits  k/oct   H(code)  eff=H/n   sat%   code0%   spearman(code,float)")
    for nb in args.bits:
        for k in args.per_octave:
            row = {"bits": nb, "per_octave": k}
            for dim in ("X", "Y"):
                c = q_log(d[dim], nb, k)
                hi = (1 << nb) - 1
                row[dim] = {
                    "entropy_bits": entropy_bits(c),
                    "efficiency": entropy_bits(c) / nb,
                    "saturation_frac": float((c == hi).mean()),
                    "code0_frac": float((c == 0).mean()),
                    "spearman_vs_float": spearman(c, d[dim]),
                }
            res["codes"].append(row)
            rx = row["X"]
            print(f"  {nb:>4}  {k:>5.2f}  {rx['entropy_bits']:7.3f}  "
                  f"{rx['efficiency']:7.3f}  {rx['saturation_frac']*100:5.2f}  "
                  f"{rx['code0_frac']*100:6.2f}   {rx['spearman_vs_float']:6.3f}")

    print("\n== equal-frequency reference (the best an n-bit code can carry) ==")
    for nb in args.bits:
        c = q_quantile(d["X"], nb)
        print(f"  {nb} bits: H={entropy_bits(c):.3f} (max {nb})  "
              f"spearman={spearman(c, d['X']):.3f}")
        res.setdefault("quantile_reference", []).append(
            {"bits": nb, "entropy_bits": entropy_bits(c),
             "spearman_vs_float": spearman(c, d["X"])})

    # A reduced (per-accepted-hit) delta has a narrower range, so the same
    # number of bits covers it with less saturation. Worth knowing before
    # spending bits on range instead of resolution.
    print("\n== reduced delta (per accepted hit): range and 4-bit behaviour ==")
    for dim in ("X", "Y"):
        r = d[dim] / nacc
        c = q_log(r, 4, 2.0)
        nz = r[r > 0]
        info = {
            "p50": float(np.quantile(r, 0.5)), "p99": float(np.quantile(r, 0.99)),
            "max": float(r.max()),
            "octaves_p1_to_p999": float(np.log2(np.quantile(nz, 0.999) /
                                                max(np.quantile(nz, 0.01), 1e-9))),
            "q4k2_entropy_bits": entropy_bits(c),
            "q4k2_saturation_frac": float((c == 15).mean()),
        }
        res["reduced"][dim] = info
        print(f"  {dim}: p50={info['p50']:.3g} p99={info['p99']:.3g} "
              f"max={info['max']:.3g} octaves={info['octaves_p1_to_p999']:.1f} "
              f"| 4b k=2: H={info['q4k2_entropy_bits']:.2f} "
              f"sat={info['q4k2_saturation_frac']*100:.2f}%")

    out = pathlib.Path(args.outdir) / "chi2_code_efficiency.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
