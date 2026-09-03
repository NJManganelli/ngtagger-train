#!/usr/bin/env python
"""Is the correct-hit pull excess actually multiple scattering? Test before coding.

The refit has no multiple-scattering term (the spec's Q1 table flags this against
TMTT, which inflates stub errors by sigmaScat = KalmanMultiScattTerm/pT with
0.00075). Cluster-era PU200 pulls for TRUTHFULLY CORRECT hits are 1.54 (x) and
2.34 (y) where they should be 1.0, and backing out the terms says residuals are
prediction-dominated. Scattering is the prime suspect.

But "add a term until the pulls come out at 1.0" is not evidence. Scattering has
a SHAPE: the deflection goes as 1/(beta p), so the missing variance should be
~1/pT^2 and the pull excess should vanish at high pT. This script tests that
shape before any code is written:

  * pull width vs pT, per layer, for correct hits only. If the excess is
    scattering, the width falls toward 1.0 as pT rises. If it is FLAT in pT, the
    excess is NOT scattering (think misalignment-like or a projection bias) and a
    1/pT term would be curve-fitting, not physics.
  * the residual spread and the predicted sigma separately vs pT, so it is
    visible WHICH of the two carries the pT dependence.
  * a fitted per-layer constant k [cm*GeV] for
        sigma_eff^2 = sigma_pred^2 + (k / pT)^2
    obtained from the measured excess, plus the pull width that constant would
    produce per pT bin -- i.e. does ONE constant flatten the whole range?

    pixi run python eval_refitq/windows/ms_term_shape.py -i <cluster-era nano.root>
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import re

import awkward as ak
import numpy as np
import uproot

SENTINEL = -900.0
PT_BINS = [(2, 3), (3, 4), (4, 6), (6, 10), (10, 20), (20, 1e9)]


def robust_sigma(v):
    if len(v) < 30:
        return float("nan")
    q1, q3 = np.quantile(v, [0.25, 0.75])
    return float((q3 - q1) / 1.349)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputs", nargs="+", required=True)
    ap.add_argument("--ref", default="L1TTrack")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    files = [f for p in args.inputs for f in (sorted(_glob.glob(p)) or [p])]

    with uproot.open(f"{files[0]}:Events") as t:
        keys = list(t.keys())
    cfg = sorted({m.group(1) for m in
                  (re.match(r"^L1TSmartPixelsRefitHitDigiRefit([A-Za-z0-9]+)_", k) for k in keys)
                  if m})[-1]
    HIT = f"L1TSmartPixelsRefitHitDigiRefit{cfg}"
    print(f"config={cfg}")

    hcols = ["trackIdx", "layer", "hitAccepted", "selHitClass",
             "projResX", "projResY", "pullX", "pullY"]
    ref = uproot.concatenate([f"{f}:Events" for f in files],
                             filter_name=[f"{args.ref}_pt"])
    hits = uproot.concatenate([f"{f}:Events" for f in files],
                              filter_name=[f"{HIT}_{c}" for c in hcols])
    counts = ak.to_numpy(ak.num(ref[f"{args.ref}_pt"]))
    off = np.concatenate([[0], np.cumsum(counts)])
    pt_flat = ak.to_numpy(ak.flatten(ref[f"{args.ref}_pt"]))

    d = {c: ak.to_numpy(ak.flatten(hits[f"{HIT}_{c}"])) for c in hcols}
    ev = np.repeat(np.arange(len(counts)), ak.to_numpy(ak.num(hits[f"{HIT}_trackIdx"])))
    g = off[ev] + d["trackIdx"].astype(np.int64)
    pt = pt_flat[g]

    out = {"files": files, "config": cfg, "pt_bins": PT_BINS, "dims": {}}

    for dim, res_name, pull_name in (("x", "projResX", "pullX"), ("y", "projResY", "pullY")):
        print(f"\n=== local {dim}: correct hits only (selHitClass == 0) ===")
        print("  L   pT bin      n    pull sigma   |res| sigma [um]   pred sigma [um]")
        rec = out["dims"].setdefault(dim, {})
        for L in (1, 2, 3, 4):
            base = ((d["hitAccepted"] > 0) & (d["selHitClass"] == 0) & (d["layer"] == L)
                    & (d[pull_name] > SENTINEL) & (np.abs(d[pull_name]) > 1e-9))
            rows = []
            for lo, hi in PT_BINS:
                m = base & (pt >= lo) & (pt < hi)
                if m.sum() < 30:
                    continue
                pl = d[pull_name][m]
                rs = d[res_name][m]
                # sqrt(S) is exact for x (relinearization vanishes on the first
                # scalar update) and approximate for y; stated in the writeup.
                s_pred = np.abs(rs) / np.abs(pl)
                sp = float(robust_sigma(pl)); sr = float(robust_sigma(rs) * 1e4)
                spred = float(np.median(s_pred) * 1e4)
                lbl = f"{lo}-{'inf' if hi > 1e8 else hi}"
                print(f"  {L}  {lbl:>8} {int(m.sum()):>7}   {sp:>9.3f}   {sr:>14.1f}   {spred:>13.1f}")
                rows.append({"pt_lo": lo, "pt_hi": hi, "n": int(m.sum()),
                             "pull_sigma": sp, "res_sigma_um": sr, "pred_sigma_um": spred})
            rec[str(L)] = rows

            # Is the excess consistent with ONE 1/pT constant?
            #   res^2 = pred^2 + (k/pT)^2  ->  k^2 = (res^2 - pred^2) * pT^2
            ks = []
            for r in rows:
                if not (np.isfinite(r["res_sigma_um"]) and np.isfinite(r["pred_sigma_um"])):
                    continue
                excess = r["res_sigma_um"] ** 2 - r["pred_sigma_um"] ** 2
                ptc = 0.5 * (r["pt_lo"] + min(r["pt_hi"], 40.0))
                ks.append(float(np.sqrt(excess) * ptc) if excess > 0 else float("nan"))
            ks = [k for k in ks if np.isfinite(k)]
            # Two-component decomposition. If the excess were pure scattering, k
            # would be constant across pT and c would be 0. A nonzero c is a
            # pT-INDEPENDENT floor, which scattering cannot produce.
            hi_bins = [r for r in rows if r["pt_lo"] >= 6 and np.isfinite(r["res_sigma_um"])]
            if hi_bins:
                c2 = np.median([r["res_sigma_um"] ** 2 - r["pred_sigma_um"] ** 2 for r in hi_bins])
                c = float(np.sqrt(max(c2, 0.0)))
                lo = [r for r in rows if r["pt_lo"] < 4 and np.isfinite(r["res_sigma_um"])]
                kk = float("nan")
                if lo:
                    r0 = lo[0]
                    rem = r0["res_sigma_um"] ** 2 - r0["pred_sigma_um"] ** 2 - c2
                    ptc = 0.5 * (r0["pt_lo"] + r0["pt_hi"])
                    kk = float(np.sqrt(max(rem, 0.0)) * ptc)
                print(f"      two-component: pT-INDEPENDENT floor c = {c:.1f} um"
                      f"  (from pT>=6 bins), scattering-like k = {kk:.0f} um*GeV")
                rec[f"{L}_two_component"] = {"floor_c_um": c, "k_um_GeV": kk}
            if ks:
                kmed = float(np.median(ks))
                spread = (max(ks) / min(ks)) if min(ks) > 0 else float("inf")
                print(f"      k per bin [um*GeV]: {['%.0f' % k for k in ks]}")
                print(f"      -> median k = {kmed:.0f} um*GeV, max/min spread = {spread:.1f}"
                      f"   ({'consistent with 1/pT' if spread < 2.5 else 'NOT a 1/pT law'})")
                rec[f"{L}_k_um_GeV"] = {"per_bin": ks, "median": kmed, "spread": spread}

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
