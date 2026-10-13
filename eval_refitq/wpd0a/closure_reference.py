"""WP-D0a check (plan D.9, C.4 row "closure reference from another build"): are the
census-vs-CMSSW closure references the production's own tracks?

    python eval_refitq/wpd0a/closure_reference.py --production F.root --dr DR.root \
        --nodr NODR.root -o out.json

The references (modecmp HYBRID_np5{,_noDR}_f01_100ev, made 2026-09-24 by another build,
cfg_HYBRID_np5.py) are usable only if they hold the production's tracks:
  --dr    the reference WITH duplicate removal must equal the production's L1TTrack
          collection event by event (every column both carry, bit for bit, as a multiset);
  --nodr  the reference WITHOUT duplicate removal must CONTAIN every production track of
          the event (duplicate removal only removes tracks) and hold at least as many.
Events are matched by (run, luminosityBlock, event). Known-bad: the 4-parameter reference
(HYBRID_np4) in place of --nodr must fail (other fitted parameters on every track). Also
reported: the share of production tracks found bit-identical in the no-DR reference.
Exit status 0 only when both hold in every event.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
import numpy as np
import awkward as ak
import uproot

COLS = ("pt", "eta", "phi", "z0", "d0", "chi2XYRed", "chi2ZRed", "nStubs", "hitPattern")
EVK = ("run", "luminosityBlock", "event")


def read(path):
    t = uproot.open(f"{path}:Events")
    have = [c for c in COLS if f"L1TTrack_{c}" in t.keys()]
    A = t.arrays([f"L1TTrack_{c}" for c in have] + list(EVK))
    ids = list(zip(*(ak.to_numpy(A[k]).tolist() for k in EVK)))
    ev = {}
    for i, k in enumerate(ids):
        cols = [ak.to_numpy(A[f"L1TTrack_{c}"][i]) for c in have]
        ev[k] = Counter(zip(*(c.tolist() for c in cols))) if cols else Counter()
    return have, ev


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--production", required=True)
    ap.add_argument("--dr", required=True)
    ap.add_argument("--nodr", required=True)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    cp, P = read(a.production)
    cd, D = read(a.dr)
    cn, N = read(a.nodr)
    out = {"production": a.production, "dr": a.dr, "nodr": a.nodr, "n_events": len(P),
           "n_tracks_production": int(sum(sum(c.values()) for c in P.values()))}
    common_d = [c for c in cp if c in cd]
    common_n = [c for c in cp if c in cn]
    jd, jn = [cp.index(c) for c in common_d], [cp.index(c) for c in common_n]
    proj = lambda C, idx, cols_src, cols: Counter({tuple(k[cols_src.index(c)] for c in cols): v
                                                   for k, v in C.items()})
    eq = cont = n_in = 0
    miss_d = []
    for k, c in P.items():
        if k not in D or k not in N:
            miss_d.append(str(k))
            continue
        if proj(c, jd, cp, common_d) == proj(D[k], None, cd, common_d):
            eq += 1
        pc, nc = proj(c, jn, cp, common_n), proj(N[k], None, cn, common_n)
        if not (pc - nc) and sum(nc.values()) >= sum(pc.values()):
            cont += 1
        n_in += sum((pc & nc).values())
    out.update(columns_dr=common_d, columns_nodr=common_n, events_missing=miss_d[:10],
               dr_identical_events=eq, nodr_containing_events=cont,
               n_tracks_dr=int(sum(sum(c.values()) for c in D.values())),
               n_tracks_nodr=int(sum(sum(c.values()) for c in N.values())),
               # context: production tracks found bit-identical in the no-DR reference (merge-style
               # duplicate removal refits the tracks it merges, so only unmerged ones can match)
               nodr_track_overlap=n_in / max(out["n_tracks_production"], 1))
    ok = eq == len(P) and cont == len(P) and not miss_d
    out["closure_reference_ok"] = bool(ok)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(P)} events: DR reference identical in {eq}, no-DR reference contains the production "
          f"tracks in {cont}; tracks production/DR/noDR {out['n_tracks_production']}/{out['n_tracks_dr']}/"
          f"{out['n_tracks_nodr']}; production tracks found in the no-DR reference "
          f"{100 * out['nodr_track_overlap']:.1f}% -> {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
