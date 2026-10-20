"""Per-path counts of the v3 refit from the CMSSW closure dump (v3cmssw/PLAN.md rev 1 §3, item 5): how often each
rarely-exercised branch of the engine fired.

    python eval_refitq/v3cmssw/path_counts.py DUMP [DUMP ...] [--out counts.json]

A multi-threaded job writes one dump per stream (<name>, <name>.1, ...); pass them all. Counted:
- tracks: refit, passthrough, disc-going, seed model outside its fitted p / |eta| domain (clipped), end-of-track
  one-layer commits;
- visited crossings: window fallback (every module of the surface), skip price fallback level (0 its own bin, 1 the
  surface without the first-visit split, 2 / 3 the same for the barrel code of a disc-going track, 4 "*");
- applied updates: the prediction |cot| > 12 guard, updates with a Jacobian column zeroed by jacobianMaxAbs, the
  angle-table mode of the applied cluster per angle (0 no entry, 1, 2).
Each count is given in total and per surface code (1-4 barrel, 11-14 disc).
"""
from __future__ import annotations

import argparse
import collections
import json


def count(paths):
    tot = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                tot["tracks"] += 1
                tot["refit" if d["refit"] else "passthrough"] += 1
                tot["disc-going"] += bool(d.get("disc_going"))
                tot["seed outside the fitted domain"] += bool(d.get("seedClipped"))
                tot["end-of-track one-layer commit"] += bool(d.get("endCommit"))
                for s in d["steps"]:
                    if not s.get("visited"):
                        continue
                    S = s["surface"]
                    for k, v in (("visited crossings", 1),
                                 ("window fallback (every module)", bool(s.get("fallback"))),
                                 ("accepted", bool(s.get("accepted")))):
                        tot[k] += v
                        per[S][k] += v
                    if s.get("skipLevel", -1) >= 0:
                        k = f"skip price level {s['skipLevel']}"
                        tot[k] += 1
                        per[S][k] += 1
                    if s.get("accepted"):
                        for k, v in (("prediction |cot| > 12 guard", bool(s.get("predGuard"))),
                                     ("Jacobian column zeroed", s.get("jacZeroed", 0) > 0)):
                            tot[k] += v
                            per[S][k] += v
                        for ia, name in enumerate(("alpha", "beta")):
                            k = f"{name} table mode {s.get('angleMode', [-1, -1])[ia]}"
                            tot[k] += 1
                            per[S][k] += 1
    return {"total": dict(sorted(tot.items())), "per_surface": {S: dict(sorted(c.items())) for S, c in sorted(per.items())}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"dumps": a.dumps, **count(a.dumps)}
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(json.dumps(res, indent=1) + "\n")
    for k, v in res["total"].items():
        print(f"{k:40s} {v:9d}")


if __name__ == "__main__":
    main()
