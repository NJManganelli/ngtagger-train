#!/usr/bin/env python3
"""Payload-level validation of the HashPRNG-factorized angle-synthesis compounds.

For each smear compound (spx_angle_{alpha,beta}_smear = sigma * HashPRNG-stdnormal):
the entropy IS the input tuple, so the throw distribution is sampled by evaluating at
many float-distinct cotAlpha values WITHIN one (layer, cotAlpha-bin, cotBeta-bin) cell
(sigma is constant per cell). Checks per sampled cell:
  * mean(smear) ~ 0            (tolerance: few % of sigma, statistics-limited)
  * std(smear)  ~ sigma(cell)  (tolerance: few %)
  * bias + smear reproduces N(bias, sigma): KS test of (bias+smear-bias)/sigma vs N(0,1)
Also: determinism (same inputs -> same value across repeated evals AND across a fresh
CorrectionSet load) and the valid_flat gate variate is U(0,1)-distributed and
decorrelated from the smear throw.

Usage: validate_hashprng_payload.py PAYLOAD.json [--out RESULTS.json]
"""
import argparse
import json
import sys

import numpy as np
from scipy import stats
import correctionlib

# sampled cells: (layer, alpha-bin center, cotBeta probe) -- alpha bins from the payload
N_SAMPLES = 4000
MEAN_TOL_FRAC = 0.05   # |mean| < 5% of sigma * (1 + few/sqrt(N) headroom)
STD_TOL_FRAC = 0.05    # |std/sigma - 1| < 5%
KS_PVAL_MIN = 1e-3
BLOCALY = -3.81


def sample_cell(comp, sigma_corr, bias_corr, layer, a_lo, a_hi, cb, n=N_SAMPLES):
    """Evaluate the compound at n float-distinct cotAlpha values inside one alpha bin."""
    # keep strictly inside the bin; avoid the exact edges
    pad = 0.02 * (a_hi - a_lo)
    alphas = np.linspace(a_lo + pad, a_hi - pad, n)
    smear = np.array([comp.evaluate(layer, float(a), cb, BLOCALY) for a in alphas])
    sig = sigma_corr.evaluate(layer, float(0.5 * (a_lo + a_hi)), cb, BLOCALY)
    bias = bias_corr.evaluate(layer, float(0.5 * (a_lo + a_hi)), cb, BLOCALY)
    return alphas, smear, sig, bias


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("payload")
    ap.add_argument("--out", default=None, help="results json (default: stdout summary only)")
    args = ap.parse_args()

    cset = correctionlib.CorrectionSet.from_file(args.payload)
    raw = json.load(open(args.payload))
    alpha_edges = None
    for c in raw["corrections"]:
        if c["name"] == "spx_angle_alpha_sigma":
            alpha_edges = c["data"]["content"][0]["value"]["edges"][0]
    assert alpha_edges, "cannot find alpha edges"

    results = {"payload": args.payload, "cells": [], "determinism": {}, "gate": {}, "errors": []}
    ok = True

    for x in ("alpha", "beta"):
        comp = cset.compound[f"spx_angle_{x}_smear"]
        sig_c = cset[f"spx_angle_{x}_sigma"]
        bias_c = cset[f"spx_angle_{x}_bias"]
        # several cells: 2 layers x 3 alpha bins x 2 cotBeta probes
        for layer in (1, 3):
            for ibin in (1, len(alpha_edges) // 2, len(alpha_edges) - 3):
                a_lo, a_hi = alpha_edges[ibin], alpha_edges[ibin + 1]
                for cb in (-0.15, 0.6):
                    alphas, smear, sig, bias = sample_cell(comp, sig_c, bias_c, layer, a_lo, a_hi, cb)
                    mean, std = float(smear.mean()), float(smear.std())
                    # standardized bias+smear vs N(0,1): smear/sigma vs N(0,1) (bias is additive const)
                    ks = stats.kstest(smear / sig, "norm")
                    cell = dict(angle=x, layer=layer, alpha_bin=[a_lo, a_hi], cotBeta=cb,
                                sigma_payload=sig, bias_payload=bias,
                                mean=mean, std=std, std_ratio=std / sig,
                                ks_pval=float(ks.pvalue), n=len(smear))
                    # tolerance: statistics-limited mean error is sigma/sqrt(N); allow 3x + frac
                    mean_tol = sig * (MEAN_TOL_FRAC + 3.0 / np.sqrt(len(smear)))
                    std_tol = STD_TOL_FRAC + 3.0 / np.sqrt(2 * len(smear))
                    cell["pass"] = bool(abs(mean) < mean_tol and abs(std / sig - 1) < std_tol
                                        and ks.pvalue > KS_PVAL_MIN)
                    if not cell["pass"]:
                        ok = False
                        results["errors"].append(f"{x} L{layer} bin{ibin} cb={cb}: "
                                                 f"mean={mean:.5f} std/sig={std/sig:.4f} ks_p={ks.pvalue:.2e}")
                    results["cells"].append(cell)

    # determinism: repeated evals + fresh load
    probe = (2, 0.123456789, -0.4, BLOCALY)
    v1 = cset.compound["spx_angle_alpha_smear"].evaluate(*probe)
    v2 = cset.compound["spx_angle_alpha_smear"].evaluate(*probe)
    cset2 = correctionlib.CorrectionSet.from_file(args.payload)
    v3 = cset2.compound["spx_angle_alpha_smear"].evaluate(*probe)
    p1 = cset["spx_angle_prng"].evaluate(*probe)
    p2 = cset2["spx_angle_prng"].evaluate(*probe)
    results["determinism"] = dict(repeat_identical=(v1 == v2), fresh_load_identical=(v1 == v3),
                                  prng_identical=(p1 == p2), value=v1)
    if not (v1 == v2 == v3 and p1 == p2):
        ok = False
        results["errors"].append("determinism violated")

    # valid_flat gate: U(0,1) + decorrelated from the smear deviate
    alphas = np.linspace(-0.05, 0.05, 2000)
    flat = np.array([cset["spx_angle_valid_flat"].evaluate(1, float(a), 0.6, BLOCALY) for a in alphas])
    z = np.array([cset["spx_angle_prng"].evaluate(1, float(a), 0.6, BLOCALY) for a in alphas])
    ks_u = stats.kstest(flat, "uniform")
    rho = float(stats.spearmanr(flat, z).statistic)
    results["gate"] = dict(ks_uniform_pval=float(ks_u.pvalue), in_unit=bool(((flat >= 0) & (flat <= 1)).all()),
                           spearman_vs_prng=rho)
    if ks_u.pvalue < KS_PVAL_MIN or not results["gate"]["in_unit"] or abs(rho) > 0.1:
        ok = False
        results["errors"].append(f"gate: ks_p={ks_u.pvalue:.2e} rho={rho:.3f}")

    results["pass"] = ok
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
    npass = sum(c["pass"] for c in results["cells"])
    print(f"cells: {npass}/{len(results['cells'])} pass | determinism: {results['determinism']} | "
          f"gate: ks_p={results['gate']['ks_uniform_pval']:.3g} rho={results['gate']['spearman_vs_prng']:.3f}")
    for e in results["errors"]:
        print("  FAIL:", e)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
