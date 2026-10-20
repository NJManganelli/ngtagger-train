"""Self-consistency of a v3 closure job: its double-precision dump against its own nano (v3cmssw/PLAN.md rev 1 §3).

    python eval_refitq/v3cmssw/dev_selfcheck.py --nano v3.root --dump v3_dump.jsonl [--table L1TSmartPixelsTrack]

Per OT track (index along axis 1 of the canonical SmartPixels track table, events paired by (run, lumi, event)):
- the refit flag and the accepted-hit count agree with spixRefitPerformed / spixNAcceptedHits, and spixRefitVersion is 3;
- a refit track's helix in the nano equals the dump's final helix rounded to float32, and a passthrough track's equals
  the OT track (L1TTrack);
- the per-crossing refit-hit entries (L1TSmartPixelsRefitHit, one per visited surface of a refit track; none for a
  passthrough track) carry the dump's surface, selected cluster index, candidate count and two-layer choice in visit
  order.
Exit 0 when every check holds.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import uproot

HELIX = (("rInv", "rInv"), ("phi0", "phi"), ("tanL", "tanL"), ("z0", "z0"), ("d0", "d0"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nano", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--table", default="L1TSmartPixelsTrack")
    ap.add_argument("--hits", default="L1TSmartPixelsRefitHit")
    a = ap.parse_args()
    T = uproot.open(a.nano)["Events"]
    tb, hb = a.table, a.hits
    cols = ["run", "luminosityBlock", "event"] + [f"{tb}_{c}" for _, c in HELIX] + \
           [f"L1TTrack_{c}" for _, c in HELIX] + \
           [f"{tb}_spixRefitPerformed", f"{tb}_spixNAcceptedHits", f"{tb}_spixRefitVersion",
            f"{hb}_trackIdx", f"{hb}_layer", f"{hb}_selClusterIdx", f"{hb}_windowMult", f"{hb}_kchoice",
            f"{hb}_hitAccepted"]
    A = T.arrays(cols)
    ev_index = {(int(r), int(l), int(e)): i for i, (r, l, e) in
                enumerate(zip(A["run"], A["luminosityBlock"], A["event"]))}
    bad = {}
    n = {"tracks": 0, "refit": 0, "visited": 0}

    def fail(what, msg):
        bad.setdefault(what, []).append(msg)

    for line in open(a.dump):
        d = json.loads(line)
        i = ev_index.get((d["run"], d["lumi"], d["event"]))
        if i is None:
            fail("event missing in nano", str((d["run"], d["lumi"], d["event"])))
            continue
        t = d["track"]
        n["tracks"] += 1
        tag = f"ev {d['event']} trk {t}"
        if bool(A[f"{tb}_spixRefitPerformed"][i][t]) != d["refit"]:
            fail("refit flag", tag)
        if int(A[f"{tb}_spixNAcceptedHits"][i][t]) != d["nacc"]:
            fail("nacc", tag)
        if int(A[f"{tb}_spixRefitVersion"][i][t]) != 3:
            fail("refitVersion", tag)
        for k, c in HELIX:
            got = np.float32(A[f"{tb}_{c}"][i][t])
            want = np.float32(d["final"][k]) if d["refit"] else np.float32(A[f"L1TTrack_{c}"][i][t])
            if got != want:
                fail(f"helix {c}", f"{tag} {got} vs {want}")
        n["refit"] += d["refit"]
        sel = np.asarray(A[f"{hb}_trackIdx"][i]) == t
        hits = [{k: np.asarray(A[f"{hb}_{k}"][i])[sel] for k in ("layer", "selClusterIdx", "windowMult", "kchoice",
                                                                 "hitAccepted")}][0]
        # a passthrough track has no refit-hit entries (RefitSidecarSpec §2), whatever the engine visited
        steps = [s for s in d["steps"] if s["visited"]] if d["refit"] else []
        n["visited"] += len(steps)
        if len(steps) != len(hits["layer"]):
            fail("hit count", f"{tag} {len(hits['layer'])} vs {len(steps)}")
            continue
        for j, s in enumerate(steps):
            if (int(hits["layer"][j]), bool(hits["hitAccepted"][j]), int(hits["windowMult"][j]),
                    int(hits["kchoice"][j])) != (s["surface"], s["accepted"], s["ncand"], s["kchoice"]):
                fail("hit entry", f"{tag} step {j}")
            if s["accepted"] and int(hits["selClusterIdx"][j]) != s["cluster"]:
                fail("selected cluster", f"{tag} step {j}")
    res = {"counts": n, "failures": {k: len(v) for k, v in bad.items()},
           "examples": {k: v[:5] for k, v in bad.items()}, "verdict": "pass" if not bad else "FAIL"}
    print(json.dumps(res, indent=1))
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
