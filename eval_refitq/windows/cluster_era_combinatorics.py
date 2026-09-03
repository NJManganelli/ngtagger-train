#!/usr/bin/env python
"""Cluster-era re-measurement of the three numbers the digi model got wrong.

Context. Until v2.6 the refit took its hit candidates from raw PixelDigis, so a
"hit" was one fired pixel: an inclined particle contributed SEVERAL candidates,
the position was a pixel centre and its uncertainty a per-module
pitch/sqrt(12) constant. Candidates are now SiPixelRecHits (CPE position and
template errors). Every combinatorics and pull number measured on the digi
model therefore has to be re-measured, and the comparison is only meaningful at
the SAME pileup -- so this must run on a 20_1-compatible PU200 sample, not on
the noPU smoke.

Three deliverables, each against its digi-era value:

  1. in-window multiplicity per layer   (digi era, PU200: 6.09 / 6.07 / 5.64 / 5.14)
     How much of that was one particle's own cluster versus genuine ambiguity?
  2. correct-hit pull widths per dimension (digi era: 2.04 / 1.61 / 2.20 / 1.30)
     Does sigma collapse toward 1.0 now the error model is the CPE rather than
     pitch/sqrt(12)? This is the PREREQUISITE for covariance-derived windows: a
     window built from an optimistic covariance loses real hits.
  3. wrong-hit rate per layer            (digi era: 47.7 / 25.4 / 19.6 / 17.8%)
     plus the new cluster-merging rate, which the digi model could not express.

Also reports the innovation sigma per layer (sqrt(S) is now stored directly as
innovSig*, when present; else reconstructed as projRes/pull) against the static
window half-widths, i.e. the window-mis-sizing table recomputed in the cluster
era.

    pixi run python eval_refitq/windows/cluster_era_combinatorics.py -i <nano.root>
"""
from __future__ import annotations

import argparse
import glob as _glob
import json

import awkward as ak
import numpy as np
import uproot

SENTINEL = -900.0
WIN_RPHI = {1: 0.05, 2: 0.17, 3: 0.5, 4: 0.9}   # cm, per-layer static half-widths
WIN_Z = {1: 0.45, 2: 0.35, 3: 0.25, 4: 0.2}
PITCH_X_UM, PITCH_Y_UM = 25.0, 100.0            # from the phase2_IT_v7.1.1_25x100 payloads

DIGI_ERA = {                                     # PU200, digi hit model, for contrast
    "windowMult": {1: 6.09, 2: 6.07, 3: 5.64, 4: 5.14},
    "wrongFrac": {1: 0.477, 2: 0.254, 3: 0.196, 4: 0.178},
    "pullWidth": {"pullX": 2.04, "pullY": 1.61, "pullAlpha": 2.20, "pullBeta": 1.30},
}


def robust_sigma(v):
    if len(v) < 20:
        return float("nan")
    q1, q3 = np.quantile(v, [0.25, 0.75])
    return float((q3 - q1) / 1.349)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputs", nargs="+", required=True)
    ap.add_argument("--config", default=None, help="digiRefit config suffix; default: autodetect")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    files = [f for p in args.inputs for f in (sorted(_glob.glob(p)) or [p])]

    with uproot.open(f"{files[0]}:Events") as t:
        keys = list(t.keys())
    import re
    cfgs = sorted({m.group(1) for m in
                   (re.match(r"^L1TSmartPixelsRefitHitDigiRefit([A-Za-z0-9]+)_", k) for k in keys)
                   if m})
    if not cfgs:
        raise SystemExit("no L1TSmartPixelsRefitHitDigiRefit* tables in the input")
    cfg = args.config or cfgs[-1]
    HIT = f"L1TSmartPixelsRefitHitDigiRefit{cfg}"
    print(f"files={len(files)}  config={cfg}")

    want = ["layer", "windowMult", "hitAccepted", "selHitClass",
            "pullX", "pullY", "pullAlpha", "pullBeta", "projResX", "projResY"]
    optional = ["clusterMerged", "truthChargeFrac", "recoSizeX", "recoSizeY",
                "sigX", "sigY", "innovSigX", "innovSigY"]
    have = [c for c in want + optional if f"{HIT}_{c}" in keys]
    missing = [c for c in want if c not in have]
    if missing:
        raise SystemExit(f"input lacks required columns {missing} - is this cluster-era nano?")
    arr = uproot.concatenate([f"{f}:Events" for f in files],
                             filter_name=[f"{HIT}_{c}" for c in have])
    d = {c: ak.to_numpy(ak.flatten(arr[f"{HIT}_{c}"])) for c in have}
    n_ev = sum(uproot.open(f"{f}:Events").num_entries for f in files)
    lay, acc = d["layer"], d["hitAccepted"] > 0
    print(f"events={n_ev}  crossing records={len(lay)}  accepted={int(acc.sum())}")

    out = {"files": files, "config": cfg, "n_events": n_ev,
           "n_crossings": int(len(lay)), "hit_model": "clusters (SiPixelRecHit)",
           "digi_era_reference": DIGI_ERA, "layers": {}}

    print("\n(1) in-window multiplicity, cluster era vs digi era")
    print("  L    <windowMult>   digi era   ratio     median")
    for L in (1, 2, 3, 4):
        m = acc & (lay == L)
        if m.sum() < 20:
            print(f"  {L}   (too few)"); continue
        wm = d["windowMult"][m].astype(float)
        ref = DIGI_ERA["windowMult"][L]
        print(f"  {L}    {wm.mean():>9.2f}   {ref:>8.2f}   {wm.mean()/ref:>5.2f}x   {np.median(wm):>6.0f}")
        out["layers"].setdefault(str(L), {})["windowMult_mean"] = float(wm.mean())
        out["layers"][str(L)]["windowMult_digi_era"] = ref

    print("\n(3) hit-quality composition per layer")
    hdr = "  L    wrong%   digi era"
    if "clusterMerged" in have:
        hdr += "    merged%   <chargeFrac>"
    print(hdr)
    for L in (1, 2, 3, 4):
        m = acc & (lay == L)
        if m.sum() < 20:
            continue
        wrong = float((d["selHitClass"][m] == 1).mean())
        line = f"  {L}    {wrong*100:>5.1f}%   {DIGI_ERA['wrongFrac'][L]*100:>6.1f}%"
        rec = out["layers"].setdefault(str(L), {})
        rec["wrong_frac"] = wrong
        rec["wrong_frac_digi_era"] = DIGI_ERA["wrongFrac"][L]
        if "clusterMerged" in have:
            mg = float(np.asarray(d["clusterMerged"][m], dtype=bool).mean())
            cf = d["truthChargeFrac"][m]
            cf = cf[cf > SENTINEL]
            line += f"    {mg*100:>5.2f}%   {np.median(cf) if len(cf) else float('nan'):>10.3f}"
            rec["merged_frac"] = mg
        print(line)

    print("\n(2) pull widths by truth class (robust sigma; 1.0 = correct error model)")
    print("  dim         correct  digi era   wrong    ratio w/c")
    for dim in ("pullX", "pullY", "pullAlpha", "pullBeta"):
        ok = acc & (d[dim] > SENTINEL)
        c = d[dim][ok & (d["selHitClass"] == 0)]
        w = d[dim][ok & (d["selHitClass"] == 1)]
        sc, sw = robust_sigma(c), robust_sigma(w)
        ref = DIGI_ERA["pullWidth"][dim]
        ratio = sw / sc if sc and np.isfinite(sc) and sc > 0 else float("nan")
        print(f"  {dim:<10} {sc:>8.3f}  {ref:>8.2f}  {sw:>8.3f}   {ratio:>8.1f}")
        out.setdefault("pull_width", {})[dim] = {
            "correct": sc, "wrong": sw, "digi_era_correct": ref}

    if "sigX" in have:
        print("\nCPE position uncertainty vs the digital limit it replaced")
        for nm, pitch in (("sigX", PITCH_X_UM), ("sigY", PITCH_Y_UM)):
            v = d[nm][acc & (d[nm] > SENTINEL)] * 1e4
            dig = pitch / np.sqrt(12.0)
            print(f"  {nm}: median {np.median(v):>7.2f} um   pitch/sqrt(12) = {dig:>5.2f} um"
                  f"   -> {dig/np.median(v):>4.2f}x better")
            out.setdefault("cpe_sigma_um", {})[nm] = {
                "median": float(np.median(v)), "digital_limit": float(dig)}

    # innovation sigma: prefer the stored sqrt(S), else reconstruct (exact for x only)
    print("\ninnovation sigma vs the STATIC window half-width (window mis-sizing, cluster era)")
    print("  L    sqrt(S_x) um   window um   window/sigma      source")
    for L in (1, 2, 3, 4):
        m = acc & (lay == L)
        if "innovSigX" in have:
            s = d["innovSigX"][m]
            s = s[s > SENTINEL] * 1e4
            src = "stored"
        else:
            ok = m & (d["pullX"] > SENTINEL) & (np.abs(d["pullX"]) > 1e-9)
            s = np.abs(d["projResX"][ok]) / np.abs(d["pullX"][ok]) * 1e4
            src = "projRes/pull"
        if len(s) < 20:
            continue
        w = WIN_RPHI[L] * 1e4
        print(f"  {L}    {np.median(s):>10.1f}   {w:>9.0f}   {w/np.median(s):>10.1f}x      {src}")
        out["layers"].setdefault(str(L), {})["innovSigX_um"] = float(np.median(s))

    if "recoSizeX" in have:
        print("\ncluster size (the shape an angle estimator would use)")
        for nm in ("recoSizeX", "recoSizeY"):
            v = d[nm][acc].astype(float)
            print(f"  {nm}: mean {v.mean():.2f} pixels, median {np.median(v):.0f}, max {v.max():.0f}")

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
