#!/usr/bin/env python
"""Does the refit help or hurt, broken out by how many WRONG hits it used?

54.7% of refits on the reference sample include at least one hit from another
TP (or noise). The deployed refit-quality objective ('genuine') is a property of
the OT seed's truth match and is nearly blind to this, so the question of what
wrong hits actually cost has never been quantified. This does that, with no new
production: existing nano carries refit parameters (variant table), seed
parameters (reference table), TP truth (tp_*) and the per-hit truth class.

For each category n_wrong = 0, 1, 2, 3, 4+ it reports, per parameter:

  seed error   = median |seed  - truth|
  refit error  = median |refit - truth|
  improvement  = 1 - refit/seed  (positive = the refit helped)
  frac improved= fraction of tracks where |refit-truth| < |seed-truth|
  kick         = median |refit - seed|, i.e. how far the refit moved at all

Caveats stated in the output: sign/definition conventions between TTTrack and
TrackingParticle are NOT assumed to agree, so the median SIGNED offset is
printed alongside; a large offset means the comparison for that parameter is
convention-limited rather than resolution-limited.

    pixi run python eval_refitq/wronghits/param_vs_wronghits.py [-i GLOB ...]
"""
from __future__ import annotations

import argparse
import glob as _glob
import json

import awkward as ak
import numpy as np
import uproot

REF, VAR = "L1TTrack", "L1TSmartPixelsTrackDigiRefitAAAA"
HIT = "L1TSmartPixelsRefitHitDigiRefitAAAA"

# (name, reference/variant column, truth column, scale, unit)
PARAMS = [
    ("d0", "d0", "tp_d0", 1e4, "um"),
    ("z0", "z0", "tp_z0", 1e4, "um"),
    ("phi", "phi", "tp_phi", 1e3, "mrad"),
    ("tanL", "tanL", "tp_tanL", 1e3, "1e-3"),
    ("pt", "pt", "tp_pt", 1.0, "GeV"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputs", nargs="+", default=[
        "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/clamp_on_f*.root"])
    ap.add_argument("--label", default="genuine",
                    choices=["genuine", "looselyGenuine"],
                    help="truth-match quality required of the OT seed")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    files = [f for p in args.inputs for f in (sorted(_glob.glob(p)) or [p])]
    print(f"files: {len(files)}")

    ref_cols = [c for _, c, _, _, _ in PARAMS] + [t for _, _, t, _, _ in PARAMS] + \
               [args.label, "tpFromHardInteraction"]
    var_cols = [c for _, c, _, _, _ in PARAMS] + ["spixRefitPerformed", "spixNAcceptedHits"]
    hit_cols = ["trackIdx", "selHitClass", "hitAccepted"]

    def cat(prefix, cols):
        a = uproot.concatenate([f"{f}:Events" for f in files],
                               filter_name=[f"{prefix}_{c}" for c in cols])
        return {c: a[f"{prefix}_{c}"] for c in cols}

    ref, var, hits = cat(REF, ref_cols), cat(VAR, var_cols), cat(HIT, hit_cols)
    counts = ak.to_numpy(ak.num(ref[args.label]))
    off = np.concatenate([[0], np.cumsum(counts)])
    n_tracks = int(off[-1])

    R = {c: ak.to_numpy(ak.flatten(v)) for c, v in ref.items()}
    V = {c: ak.to_numpy(ak.flatten(v)) for c, v in var.items()}

    ti = ak.to_numpy(ak.flatten(hits["trackIdx"])).astype(np.int64)
    ev = np.repeat(np.arange(len(counts)), ak.to_numpy(ak.num(hits["trackIdx"])))
    g = off[ev] + ti
    cls = ak.to_numpy(ak.flatten(hits["selHitClass"]))
    acc = ak.to_numpy(ak.flatten(hits["hitAccepted"])) > 0
    n_wrong = np.zeros(n_tracks)
    np.add.at(n_wrong, g[acc & ((cls == 1) | (cls == 2))], 1.0)

    # refit tracks with a usable TP match
    m = (V["spixRefitPerformed"] > 0) & (R[args.label] > 0) & (R["tp_pt"] > 0)
    print(f"refit tracks with {args.label} TP match: {int(m.sum())} of {n_tracks}")
    nw = n_wrong[m]
    pv = R["tpFromHardInteraction"][m] > 0

    cats = [("0", nw == 0), ("1", nw == 1), ("2", nw == 2), ("3", nw == 3),
            ("4+", nw >= 4)]
    print("\npopulation by number of wrong hits used")
    print("  n_wrong   tracks    share    of which PV")
    for name, sel in cats:
        n = int(sel.sum())
        print(f"  {name:>7}   {n:>6}   {100*n/len(nw):>5.1f}%   "
              f"{100*pv[sel].mean() if n else 0:>7.1f}%")

    out = {"files": files, "label": args.label, "n_tracks": int(m.sum()),
           "categories": {}}

    for pname, col, tcol, scale, unit in PARAMS:
        seed = R[col][m].astype(np.float64)
        refit = V[col][m].astype(np.float64)
        truth = R[tcol][m].astype(np.float64)
        e_seed = np.abs(seed - truth) * scale
        e_refit = np.abs(refit - truth) * scale
        kick = np.abs(refit - seed) * scale
        off_seed = np.median((seed - truth) * scale)

        print(f"\n{pname}  [{unit}]   (median SIGNED seed-truth offset "
              f"{off_seed:+.4g} -- large means convention-limited)")
        print("  n_wrong   seed err   refit err   improvement   frac improved   kick")
        for name, sel in cats:
            if sel.sum() < 20:
                print(f"  {name:>7}   (too few)")
                continue
            s, r = np.median(e_seed[sel]), np.median(e_refit[sel])
            imp = 1.0 - r / s if s > 0 else float("nan")
            fi = float((e_refit[sel] < e_seed[sel]).mean())
            print(f"  {name:>7}   {s:>8.4g}   {r:>9.4g}   {imp*100:>+9.1f}%   "
                  f"{fi*100:>11.1f}%   {np.median(kick[sel]):>7.4g}")
            out["categories"].setdefault(name, {})[pname] = {
                "n": int(sel.sum()), "seed_err": s, "refit_err": r,
                "improvement_frac": imp, "frac_improved": fi,
                "median_kick": float(np.median(kick[sel])),
                "unit": unit}

    # What a TRUST GATE would be worth. The refit is only usable where it is
    # clean, so compare three policies on the same tracks: always take the seed,
    # always take the refit, or take the refit only when it used no wrong hit
    # (an oracle gate - the ceiling for any refit-quality classifier used this
    # way). The gap between 'always refit' and the oracle gate is what a
    # good gate can recover; the gap between the gate and 'always seed' is what
    # the refit is worth at all.
    print("\n" + "=" * 78)
    print("value of a TRUST GATE: use the refit only when it is clean")
    print("=" * 78)
    clean = nw == 0
    out["gate"] = {}
    for pname, col, tcol, scale, unit in PARAMS:
        seed = R[col][m].astype(np.float64)
        refit = V[col][m].astype(np.float64)
        truth = R[tcol][m].astype(np.float64)
        e_seed = np.abs(seed - truth) * scale
        e_refit = np.abs(refit - truth) * scale
        e_gate = np.where(clean, e_refit, e_seed)
        rows = [("always seed", e_seed), ("always refit", e_refit),
                ("oracle gate", e_gate)]
        print(f"\n{pname} [{unit}]        median      q68        q95")
        for name, e in rows:
            print(f"  {name:<14} {np.median(e):>9.4g} {np.quantile(e, 0.68):>9.4g} "
                  f"{np.quantile(e, 0.95):>9.4g}")
        out["gate"][pname] = {
            "unit": unit,
            **{k: {"median": float(np.median(v)), "q68": float(np.quantile(v, 0.68)),
                   "q95": float(np.quantile(v, 0.95))} for k, v in rows}}

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
