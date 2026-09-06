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

    term        L1      L2      L3      L4     |  L1 pT-span
    0 (off)   4.127   2.787   2.114   1.126    |    2.657
    0.00075   1.524   1.261   0.811   1.126    |    0.286   <- ADOPTED
    0.00150   1.056   0.785   0.461   1.126    |    0.391
    0.00300   0.617   0.445   0.242   1.126    |    0.412

THE SHAPE CRITERION PICKS 0.00075 ON ITS OWN, and this is the load-bearing
result. The L1 pT-span is MINIMISED at 0.00075 and rises again at 0.00150 and
0.00300: over-inflating C reintroduces pT structure with the opposite sign, the
same way under-inflating does. So the criterion that CANNOT be gamed by a scale
factor -- flatness, not centring -- has an interior optimum, and it sits on the
value TMTT derives from tracker material rather than anywhere near the values
that centre the pulls better. Centring and flatness disagree, and flatness is
the one carrying the physics.

NO SINGLE CONSTANT CENTRES ALL THREE, and the miss is monotone in VISIT DEPTH,
not in material: under outsideIn the visit order is L4 -> L3 -> L2 -> L1, and the
term each layer "wants" grows the later it is visited (L1 ~0.0015, L2 ~0.0011,
L3 ~0.0006). Material does not vary that way across TBPX. A trend that tracks the
number of accumulated updates rather than the radiation length is the single-helix
approximation leaking, so fitting a per-layer scale to it would be absorbing a
model error into a physics constant.

WHY RUNNING THE FILTER AGAINST THE PARTICLE'S DIRECTION OF TRAVEL IS FINE. The
particle crossed L1..L4 on its way OUT and was scattered most by the time it
reached L4, which is the hit this fit incorporates FIRST. That sounds like the Q
bookkeeping must be backwards. It is not, because the outbound scattering is not
an uncertainty for us: we never extrapolate from the production vertex. The seed
is an OT-only fit, so every kink the particle took inboard of the OT is already
baked into WHERE THE OT TRACK IS. It is not error, it is just the trajectory. The
only variance we owe is for material between CONSECUTIVE CONSTRAINT POINTS in the
fit's own visit order, which is what nCross counts. Covariance transport for an
unmodelled kink is also symmetric under swapping the two endpoints (the linearised
transport is invertible and the kink is small), which is why CMSSW propagates
oppositeToMomentum with the same MaterialEffectsUpdator it uses alongMomentum.
Energy loss is the one effect that genuinely is NOT reversible -- backwards you
must add dE/dx back rather than subtract it -- and it is not modelled here at all.

THE FIT-ORDER PICTURE MAKES A FALSIFIABLE PREDICTION AND IT HOLDS. If Q is
tracking material between constraint points, then Q-off widths should degrade
monotonically with VISIT DEPTH and each degraded layer should carry the 1/pT
fingerprint (width RISING toward low pT), while L4 -- charged nothing, being
first-visited -- should show no scattering deficit at all. Q-off widths by pT bin:

    L4   1.07  0.98  1.33  1.38  1.69     rises with pT: NOT scattering
    L3   2.14  2.08  2.43  2.40  1.67
    L1   4.68  3.42  3.43  2.20  2.02     falls with pT: scattering

L1 loses a factor 2.3 from high pT to low, the textbook 1/(beta p) signature. L4
slopes the OTHER WAY and is already ~1.0 where scattering would hurt most.

THAT KILLS THE SEED-GAP-SCATTERING EXPLANATION OF L4, which earlier versions of
this file asserted. Missing multiple scattering makes a pull width blow up at LOW
pT; L4's blows up at HIGH pT. The sign is wrong, so no MS term of any scale fixes
it -- adding one would spoil the already-correct low-pT bins to chase the high-pT
ones. L4's residual is the OT seed covariance being too OPTIMISTIC at high pT
(a seed-quality problem) or a pT-independent residual floor, and it should be
chased there rather than by inventing a seed-gap Q.

The centring criterion, taken alone, would have preferred 0.00150 or 0.00300 --
and that is the trap this test exists to avoid. Both over-open the cone badly
(L3 at 0.461 and 0.242 is a covariance 2x to 4x too LARGE) while admitting exactly
the combinatorics the refit exists to suppress, and both are WORSE by the shape
criterion. Had this test only checked pull widths it would have recommended a
term four times too large.
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
