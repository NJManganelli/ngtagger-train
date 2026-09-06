#!/usr/bin/env python
"""Acceptance test for the multiple-scattering process-noise term Q.

TWO-SIDED, deliberately. "Tune a term until the pulls come out at 1.0" is not
evidence: any monotone inflation can hit 1.0 on average. Scattering has a SHAPE --
the deflection goes as 1/(beta p), so a correct Q must ALSO flatten the pull in
pT. A term that centres the pull but leaves it sloping in pT is curve-fitting.

So both conditions must hold, per layer:
  (1) correct-hit pull width -> 1.0
  (2) that width is FLAT in pT

and the residual after both is the single-helix approximation showing itself: a
5-parameter helix cannot represent a scattered trajectory, so no single Q can
flatten all four layers. That floor is expected and is reported, not tuned away.

Compares Q-on against Q-off on the same events.

    pixi run python eval_refitq/windows/q_acceptance.py -a <Q-on.root> -b <Q-off.root>

CALIBRATION RESULT (PU200, 25 events, correct-hit pull width in x, target 1.0).
The scale scanned is digiRefitMultScattTerm; L4 is first-visited under outsideIn
so it gets no Q by construction and is unchanged throughout.

    term        L1      L2      L3      L4
    0 (off)   4.127   2.787   2.114   1.126
    0.00075   1.524   1.261   0.811   1.126     <- TMTT default, ADOPTED
    0.00150   1.056   0.785   0.461   1.126

NO SINGLE CONSTANT CENTRES ALL THREE, and the miss is monotone in VISIT DEPTH,
not in material: under outsideIn the visit order is L4 -> L3 -> L2 -> L1, and the
term each layer "wants" grows the later it is visited (L1 ~0.0015, L2 ~0.0011,
L3 ~0.0006). Material does not vary that way across TBPX. A trend that tracks the
number of accumulated updates rather than the radiation length is the single-helix
approximation leaking, so fitting a per-layer scale to it would be absorbing a
model error into a physics constant. Left alone until the seed-gap term exists.

WHY 0.00075 RATHER THAN THE BETTER-CENTRED 0.00150: the two errors are not
symmetric for our purpose. 0.00150 drives L3 to 0.461, i.e. a covariance roughly
2x too LARGE, which over-opens the search cone and admits exactly the
combinatorics this refit exists to suppress. 0.00075 under-corrects (L1 at 1.52)
but never over-opens any layer, and it is the value TMTT derives from the tracker
material rather than one fitted to these pulls. An honest under-correction beats
a tuned over-correction.
"""
from __future__ import annotations

import argparse
import json
import re

import awkward as ak
import numpy as np
import uproot

SENTINEL = -900.0
LAYERS = (1, 2, 3, 4)
PT_BINS = [(2, 3), (3, 4), (4, 6), (6, 10), (10, 1e9)]


def robust_sigma(v):
    if len(v) < 30:
        return float("nan")
    q1, q3 = np.quantile(v, [0.25, 0.75])
    return float((q3 - q1) / 1.349)


def load(path):
    with uproot.open(f"{path}:Events") as t:
        keys = set(t.keys())
    cfg = sorted({m.group(1) for m in
                  (re.match(r"^L1TSmartPixelsRefitHitDigiRefit([A-Za-z0-9]+)_", k) for k in keys) if m})[-1]
    H, REF = f"L1TSmartPixelsRefitHitDigiRefit{cfg}", "L1TTrack"
    hc = ["trackIdx", "layer", "hitAccepted", "selHitClass", "pullX", "pullY"]
    A = uproot.concatenate([f"{path}:Events"], filter_name=[f"{H}_{c}" for c in hc])
    R = uproot.concatenate([f"{path}:Events"], filter_name=[f"{REF}_pt"])
    n = ak.to_numpy(ak.num(A[f"{H}_layer"]))
    X = {c: ak.to_numpy(ak.flatten(A[f"{H}_{c}"])) for c in hc}
    ev = np.repeat(np.arange(len(n)), n)
    nt = ak.to_numpy(ak.num(R[f"{REF}_pt"]))
    off = np.concatenate([[0], np.cumsum(nt)])
    X["pt"] = ak.to_numpy(ak.flatten(R[f"{REF}_pt"]))[off[ev] + X["trackIdx"].astype(np.int64)]
    return X


def report(X, label, out):
    print(f"\n=== {label} ===")
    print("  L   pull_x   pull_y     pT-slope(x)   verdict")
    rec = {}
    for L in LAYERS:
        base = (X["hitAccepted"] > 0) & (X["selHitClass"] == 0) & (X["layer"] == L) \
               & (X["pullX"] > SENTINEL) & (np.abs(X["pullX"]) > 1e-9)
        if base.sum() < 50:
            continue
        sx, sy = robust_sigma(X["pullX"][base]), robust_sigma(X["pullY"][base])
        # condition (2): is the width flat in pT? Fit width vs 1/pT and report the
        # span; a real MS deficit shows up as width RISING at low pT.
        widths = []
        for lo, hi in PT_BINS:
            m = base & (X["pt"] >= lo) & (X["pt"] < hi)
            widths.append(robust_sigma(X["pullX"][m]) if m.sum() > 30 else np.nan)
        w = np.array(widths, dtype=float)
        ok = np.isfinite(w)
        span = float(np.nanmax(w) - np.nanmin(w)) if ok.sum() > 1 else float("nan")
        centred = abs(sx - 1.0) < 0.25
        flat = span < 0.35
        verdict = ("PASS" if (centred and flat) else
                   "centred, not flat" if centred else
                   "flat, not centred" if flat else "FAIL")
        print(f"  {L}  {sx:7.3f} {sy:8.3f}   span {span:8.3f}   {verdict}")
        rec[L] = {"pull_x": sx, "pull_y": sy, "pt_span_x": span,
                  "widths_by_pt": [None if not np.isfinite(v) else float(v) for v in w],
                  "verdict": verdict}
    out[label] = rec
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--q-on", required=True)
    ap.add_argument("-b", "--q-off", required=True)
    ap.add_argument("-o", "--output", default="eval_refitq/windows/q_acceptance.json")
    args = ap.parse_args()
    out = {}
    on = report(load(args.q_on), "Q ON", out)
    off = report(load(args.q_off), "Q OFF", out)
    print("\n=== effect of Q, correct-hit pull width in x (target 1.0) ===")
    print("  L      Q off     Q on    moved toward 1?")
    for L in LAYERS:
        if L in on and L in off:
            a, b = off[L]["pull_x"], on[L]["pull_x"]
            better = abs(b - 1.0) < abs(a - 1.0)
            print(f"  {L}   {a:8.3f} {b:8.3f}    {'yes' if better else 'NO'}")
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
