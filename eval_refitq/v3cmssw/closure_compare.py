"""The v3 port's closure: CMSSW's double-precision dump against the replay's closure-arm output on the same
events (v3cmssw/PLAN.md rev 1 §3, items 4 and 5).

    python eval_refitq/v3cmssw/closure_compare.py select  --cmssw dump.jsonl --replay replay.jsonl [--eps E] [--out o.json]
    python eval_refitq/v3cmssw/closure_compare.py forced  --cmssw dump.jsonl --replay forced.jsonl [--out o.json]

Both inputs are JSON lines, one per OT track, in the spec's reference schema (run, lumi, event, track, refit, nacc,
disc_going, steps [surface, visited, accepted, detId, cluster, ncand, chi2, kchoice, ...], final, final_C); the
CMSSW dump adds per step scores (the committing two-layer scores of the window opened there), selMargin (runner-up
minus best ranking value) and, per accepted step, aPre / CPred / aPost / CPost.

select (item 5, free-running): each track is judged at its FIRST disagreement in (visited, accepted, cluster,
kchoice) only. Its margin is the smaller of the committing score margin (second-lowest minus lowest of the window's
three scores, the skip price included) and the ranking margin at that step, from the CMSSW dump. A first
disagreement with margin > eps fails; near-ties (margin <= eps) are capped at 0.1 % of the visited crossings per
file. eps is 10 x the largest |score difference| between the two sides over agreeing commits (needs the replay's
scores), with the floor 1e-6 (1 + |score|); --eps freezes it. Reported: every first disagreement with its margin,
surface, visit step and kchoice; agreement fractions; the final-state agreement on agreeing tracks.

forced (item 4): per accepted step, |a_cmssw - a_replay|_i / sigma_i <= 1e-6 (sigma from the CMSSW covariance at
the same point) for the state before (aPre) and after (aPost) the update, and |dC_ij| / sqrt(C_ii C_jj) <= 1e-6 for
CPred and CPost; the same on the final state and covariance.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

TOL = 1e-6
NEAR_TIE_CAP = 1e-3


def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out[(d["run"], d["lumi"], d["event"], d["track"])] = d
    return out


def visited_steps(d):
    return [s for s in d["steps"] if s.get("visited")]


def step_key(s):
    return (s.get("surface"), bool(s.get("accepted")), s.get("cluster") if s.get("accepted") else -1,
            s.get("kchoice", -1))


def score_margin(s):
    sc = [x for x in (s.get("scores") or []) if x is not None and math.isfinite(x)]
    if len(sc) < 2:
        return math.inf
    sc.sort()
    return sc[1] - sc[0]


def rank_margin(s):
    m = s.get("selMargin")
    return m if m is not None and math.isfinite(m) else math.inf


def select(a):
    C, R = load(a.cmssw), load(a.replay)
    keys = sorted(set(C) | set(R))
    missing = [k for k in keys if k not in C or k not in R]
    nvis = sum(len(visited_steps(C[k])) for k in keys if k in C)
    dscore = 0.0
    firsts, agree_tracks = [], 0
    final_dev = 0.0
    for k in keys:
        if k not in C or k not in R:
            continue
        c, r = visited_steps(C[k]), visited_steps(R[k])
        bad = None
        for j in range(max(len(c), len(r))):
            sc, sr = (c[j] if j < len(c) else None), (r[j] if j < len(r) else None)
            if sc is None or sr is None or step_key(sc) != step_key(sr):
                bad = (j, sc, sr)
                break
            if sc.get("scores") and sr.get("scores"):
                for x, y in zip(sc["scores"], sr["scores"]):
                    if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                        dscore = max(dscore, abs(x - y))
        if bad is None:
            agree_tracks += 1
            fc, fr, Cc = C[k]["final"], R[k]["final"], C[k]["final_C"]
            for i, p in enumerate(("rInv", "phi0", "tanL", "z0", "d0")):
                sg = math.sqrt(abs(Cc[i][i])) or 1.0
                final_dev = max(final_dev, abs(fc[p] - fr[p]) / sg)
            continue
        j, sc, sr = bad
        ref = sc if sc is not None else sr
        firsts.append({"track": list(k), "step": j, "surface": ref.get("surface"),
                       "cmssw": None if sc is None else step_key(sc), "replay": None if sr is None else step_key(sr),
                       "kchoice": None if sc is None else sc.get("kchoice"),
                       "margin": min(score_margin(sc), rank_margin(sc)) if sc is not None else math.inf})
    eps = a.eps if a.eps is not None else 10.0 * dscore
    near, fail = [], []
    for f in firsts:
        floor = 1e-6
        (near if f["margin"] <= max(eps, floor) else fail).append(f)
    res = {"tracks": len(keys), "missing": len(missing), "visited_crossings": nvis, "agreeing_tracks": agree_tracks,
           "first_disagreements": len(firsts), "near_ties": len(near), "failing": len(fail),
           "eps": eps, "max_dscore_agreeing": dscore, "near_tie_cap": NEAR_TIE_CAP * nvis,
           "max_final_dev_sigma_on_agreeing": final_dev,
           "verdict": "pass" if not fail and not missing and len(near) <= NEAR_TIE_CAP * nvis else "FAIL",
           "list": firsts[:200]}
    return res


def forced(a):
    C, R = load(a.cmssw), load(a.replay)
    worst = {"aPre": 0.0, "aPost": 0.0, "CPred": 0.0, "CPost": 0.0, "final": 0.0, "final_C": 0.0}
    nsteps = 0
    for k, c in C.items():
        r = R.get(k)
        if r is None:
            continue
        rs = {s["step"]: s for s in r["steps"]}
        for s in c["steps"]:
            if not s.get("accepted") or s["step"] not in rs:
                continue
            t = rs[s["step"]]
            nsteps += 1
            for av, cv in (("aPre", "CPred"), ("aPost", "CPost")):
                if av not in t:
                    continue
                Cm = s[cv]
                for i in range(5):
                    sg = math.sqrt(abs(Cm[i][i])) or 1.0
                    worst[av] = max(worst[av], abs(s[av][i] - t[av][i]) / sg)
                    for m in range(5):
                        nrm = math.sqrt(abs(Cm[i][i] * Cm[m][m])) or 1.0
                        worst[cv] = max(worst[cv], abs(Cm[i][m] - t[cv][i][m]) / nrm)
        Cc = c["final_C"]
        for i, p in enumerate(("rInv", "phi0", "tanL", "z0", "d0")):
            sg = math.sqrt(abs(Cc[i][i])) or 1.0
            worst["final"] = max(worst["final"], abs(c["final"][p] - r["final"][p]) / sg)
            for m in range(5):
                nrm = math.sqrt(abs(Cc[i][i] * Cc[m][m])) or 1.0
                worst["final_C"] = max(worst["final_C"], abs(Cc[i][m] - r["final_C"][i][m]) / nrm)
    return {"accepted_steps": nsteps, "worst": worst, "tolerance": TOL,
            "verdict": "pass" if all(v <= TOL for v in worst.values()) else "FAIL"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("select", "forced"))
    ap.add_argument("--cmssw", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = select(a) if a.mode == "select" else forced(a)
    text = json.dumps(res, indent=1, default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else x)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "list"}, indent=1))
    sys.exit(0 if res["verdict"] == "pass" else 1)


if __name__ == "__main__":
    main()
