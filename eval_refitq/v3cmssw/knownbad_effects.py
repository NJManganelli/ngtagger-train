"""The known-bads' effect on the dev events, in the units of the closure test each must fail (v3cmssw/PLAN.md rev 1
§3, item 7):

- arithmetic (item 4): on the tracks whose choices agree at every visited crossing, the largest deviation of the
  per-update states (aPre / aPost, CPred / CPost) and of the final state, in multiples of the 1e-6 tolerance;
- selection (item 5): the count of first disagreements whose margin exceeds the frozen eps (non-near-tie); and,
  re-aimed for the known-bads that rarely change a choice (the skip price's key), the count of tracks whose choices
  all agree but whose committing scores differ by more than eps at some window;
- the per-track disc-going flag, compared directly.

The required effect is >= 10x the tolerance (item 4) or >= 10 tracks per scenario (item 5); below that the known-bad
is below resolution.

    python eval_refitq/v3cmssw/knownbad_effects.py --kb-dir DIR --reference-dir SPEC/reference [--eps 3.9e-5] \
        [--file f04] [--out effects.json]

DIR holds refitV3NanoClosure dumps <scen>_<knownbad>.jsonl. Every known-bad is compared with the replay reference
(<scen>_<file>.jsonl, and <scen>_<file>_forced.jsonl for the per-update states), except the xyab skip price with the
angles off, which is compared with the engine's own angles-off run (<scen>_anglesoff.jsonl).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

from closure_compare import TOL, load, rank_margin, score_margin, step_key, visited_steps

REQUIRED_TOL_MULTIPLE = 10.0
REQUIRED_TRACKS = 10
PARS = ("rInv", "phi0", "tanL", "z0", "d0")


def state_dev(a, b, Cm):
    da = max(abs(a[i] - b[i]) / (math.sqrt(abs(Cm[i][i])) or 1.0) for i in range(5))
    return da


def cov_dev(Cm, Cr):
    return max(abs(Cm[i][m] - Cr[i][m]) / (math.sqrt(abs(Cm[i][i] * Cm[m][m])) or 1.0)
               for i in range(5) for m in range(5))


def effects(cm, ref, forced, eps):
    out = {"tracks": 0, "visited_crossings": 0, "agreeing_tracks": 0, "first_disagreements": 0, "near_ties": 0,
           "non_near_tie": 0, "non_near_tie_choice_only": 0, "agreeing_tracks_score_over_eps": 0,
           "max_dscore_agreeing": 0.0, "disc_going_flips": 0, "agreeing_tracks_over_10x_tol": 0,
           "max_state_dev": 0.0, "max_cov_dev": 0.0}
    for k, c in cm.items():
        r = ref.get(k)
        if r is None:
            continue
        out["tracks"] += 1
        vc, vr = visited_steps(c), visited_steps(r)
        out["visited_crossings"] += len(vc)
        out["disc_going_flips"] += bool(c.get("disc_going")) != bool(r.get("disc_going"))
        first, sr0, dscore = None, None, 0.0
        for j in range(max(len(vc), len(vr))):
            sc, sr = (vc[j] if j < len(vc) else None), (vr[j] if j < len(vr) else None)
            if sc is None or sr is None or step_key(sc) != step_key(sr):
                first, sr0 = sc, sr
                break
            for x, y in zip(sc.get("scores") or [], sr.get("scores") or []):
                if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                    dscore = max(dscore, abs(x - y))
        if first is not None or len(vc) != len(vr):
            out["first_disagreements"] += 1
            margin = min(score_margin(first), rank_margin(first)) if first is not None else math.inf
            if margin <= max(eps, 1e-6):
                out["near_ties"] += 1
            else:
                out["non_near_tie"] += 1
                if first is None or sr0 is None or step_key(first)[:3] != step_key(sr0)[:3]:
                    out["non_near_tie_choice_only"] += 1
            continue
        out["agreeing_tracks"] += 1
        out["max_dscore_agreeing"] = max(out["max_dscore_agreeing"], dscore)
        out["agreeing_tracks_score_over_eps"] += dscore > eps
        f = forced.get(k, r)
        fs = {s["step"]: s for s in f["steps"]}
        dev_a = state_dev([c["final"][p] for p in PARS], [r["final"][p] for p in PARS], c["final_C"])
        dev_c = cov_dev(c["final_C"], r["final_C"])
        for s in c["steps"]:
            t = fs.get(s["step"])
            if not s.get("accepted") or t is None:
                continue
            for av, cv in (("aPre", "CPred"), ("aPost", "CPost")):
                if av in s and av in t:
                    dev_a = max(dev_a, state_dev(s[av], t[av], s[cv]))
                    dev_c = max(dev_c, cov_dev(s[cv], t[cv]))
        out["max_state_dev"] = max(out["max_state_dev"], dev_a)
        out["max_cov_dev"] = max(out["max_cov_dev"], dev_c)
        if max(dev_a, dev_c) > REQUIRED_TOL_MULTIPLE * TOL:
            out["agreeing_tracks_over_10x_tol"] += 1
    out["arith_tol_multiple"] = max(out["max_state_dev"], out["max_cov_dev"]) / TOL
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb-dir", required=True)
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--eps", type=float, default=3.9e-5)
    ap.add_argument("--file", default="f04")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"eps": a.eps, "tolerance": TOL, "required": {"tol_multiple": REQUIRED_TOL_MULTIPLE,
                                                        "tracks_per_scenario": REQUIRED_TRACKS}, "effects": {}}
    refs = {}
    for path in sorted(glob.glob(os.path.join(a.kb_dir, "*_*.jsonl"))):
        scen, kb = os.path.basename(path)[:-6].split("_", 1)
        if kb in ("none", "anglesoff", "rankonly"):
            continue
        if kb == "anglesoff_xyab":
            ref = load(os.path.join(a.kb_dir, f"{scen}_anglesoff.jsonl"))
            forced = ref
        else:
            if scen not in refs:
                refs[scen] = (load(os.path.join(a.reference_dir, f"{scen}_{a.file}.jsonl")),
                              load(os.path.join(a.reference_dir, f"{scen}_{a.file}_forced.jsonl")))
            ref, forced = refs[scen]
        res["effects"].setdefault(kb, {})[scen] = effects(load(path), ref, forced, a.eps)
    for kb, per in res["effects"].items():
        per["min_over_scenarios"] = {q: min(v[q] for v in per.values())
                                     for q in ("non_near_tie", "agreeing_tracks_score_over_eps", "disc_going_flips",
                                               "arith_tol_multiple")}
    text = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(f"{'known-bad':28s} {'scen':5s} {'1st-dis':>7s} {'non-tie':>7s} {'choice':>6s} {'agree':>6s} {'dsc>eps':>7s} "
          f"{'disc':>4s} {'>10xtol':>7s} {'state/tol':>10s} {'cov/tol':>10s}")
    for kb, per in res["effects"].items():
        for scen, v in per.items():
            if scen == "min_over_scenarios":
                continue
            print(f"{kb:28s} {scen:5s} {v['first_disagreements']:7d} {v['non_near_tie']:7d} "
                  f"{v['non_near_tie_choice_only']:6d} {v['agreeing_tracks']:6d} {v['agreeing_tracks_score_over_eps']:7d} "
                  f"{v['disc_going_flips']:4d} {v['agreeing_tracks_over_10x_tol']:7d} {v['max_state_dev'] / TOL:10.3g} "
                  f"{v['max_cov_dev'] / TOL:10.3g}")


if __name__ == "__main__":
    main()
