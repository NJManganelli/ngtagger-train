"""Drivers of the CMSSW_17 statistical comparison (plan 3.4).

    python eval_refitq/cmssw17/run.py freeze                 # frozen cell map + tolerances from all D121 files
    python eval_refitq/cmssw17/run.py null [--split oddeven|halves|randomA|randomB]
    python eval_refitq/cmssw17/run.py defects                # the planted defects on the primary null split
    python eval_refitq/cmssw17/run.py defects-report         # each defect against its pre-declared expectation
    python eval_refitq/cmssw17/run.py compare                # CMSSW_17 (old TTbar) against D121 (needs the production)

Outputs under CM.OUT (WDMac): frozen/frozen.json, null/<split>.json, defects/<name>.json, compare/result.json.
Freezing rule (plan 3.4.4): the frozen tables are written before the CMSSW_17 production is read; `compare` refuses
to run without them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as CM  # noqa: E402
import engine as EN  # noqa: E402
import stats as ST  # noqa: E402

FROZEN = CM.OUT / "frozen" / "frozen.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _json(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(type(o))


def stage_freeze(a):
    ref = EN.Side("D121", "d121", CM.D121_FILES)
    edges = EN.compute_ml_edges(ref)          # the sigma_hat deciles (empty without the ML records)
    EN.set_ml_edges(edges)
    out = {"what": "frozen cell map and tolerances (plan 3.4.4-3.4.6), from all D121 files", "files": list(CM.D121_FILES),
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ml_edges": edges, "quantities": []}
    for q in EN.catalogue():
        for mask in q.masks:
            t0 = time.time()
            fz = EN.freeze_quantity(q, ref, mask, n_boot=a.n_boot)
            out["quantities"].append(fz)
            log(f"freeze {q.name} {mask}: {len(fz['cells'])} judged cells, {len(fz['report_only'])} fine cells "
                f"report-only ({time.time() - t0:.0f} s)")
    os.makedirs(FROZEN.parent, exist_ok=True)
    json.dump(out, open(FROZEN, "w"), indent=1, default=_json)
    log("wrote", FROZEN)


def frozen_map():
    if not FROZEN.exists():
        raise SystemExit(f"no frozen tables ({FROZEN}): run `freeze` first")
    fz = json.load(open(FROZEN))
    EN.set_ml_edges(fz.get("ml_edges", {}))
    infl = fz.get("se_inflation", {})
    out = {}
    for q in fz["quantities"]:
        q["se_inflation"] = infl
        out[(q["quantity"], q["mask"])] = q
    return out


def stage_fallback(a):
    """The null check's pre-declared fallback (plan 3.4.7), step 1: a class (effect kind and summary) whose null
    sd(z) exceeds 1.25 has its errors inflated by that sd. Classes below 0.8 are left as they are: their halves share
    the pileup library, so the null under-states their differences (documented deviation: deflating errors from a
    correlated null would understate the errors of the real comparison)."""
    js = json.load(open(CM.OUT / "null" / f"{a.split}.json"))
    infl = {k: v["sd_z"] for k, v in js["criteria"]["classes"].items() if v["sd_z"] and v["sd_z"] > 1.25}
    fz = json.load(open(FROZEN))
    fz["se_inflation"] = infl
    fz["se_inflation_from"] = f"null/{a.split}.json"
    json.dump(fz, open(FROZEN, "w"), indent=1, default=_json)
    log("se inflation:", infl)


def run_compare(sideA, sideB, a, defect=None, label=""):
    fz = frozen_map()
    res = []
    for q in EN.catalogue():
        for mask in q.masks:
            if (q.name, mask) not in fz:
                continue
            t0 = time.time()
            r = EN.compare_quantity(q, sideA, sideB, mask, fz[(q.name, mask)], n_boot=a.n_boot, n_perm=a.n_perm,
                                    defect=defect)
            res += r
            if a.verbose:
                log(f"{label} {q.name} {mask}: {len(r)} results ({time.time() - t0:.0f} s)")
    EN.verdicts(res, mixed_rule=a.mixed_rule)
    return res


def summarize(res):
    s = {"n": len(res), "judged": sum(r.judged for r in res)}
    for k in ("agrees", "disagrees", "inconclusive", "report-only"):
        s[k] = sum(r.verdict == k for r in res)
    for k in ("MATERIAL", "MINOR", "ok", "report"):
        s["class_" + k] = sum(r.cls == k for r in res)
    s["borderline_gross"] = sum(bool(r.extra.get("borderline_gross")) for r in res)
    return s


def null_criteria(res):
    """Pass criteria of the null (plan 3.4.7): (a) the sd of z = effect / error per class within 0.8-1.25;
    (b) agreeing cells >= 80 % of the share expected from each cell's null error (the plan's flat 80 % cannot be
    reached at half statistics: deviation recorded); (c) no cell disagrees."""
    by = {}
    for r in res:
        if not r.judged or r.summary == "shape KS" or not np.isfinite(r.se) or r.se <= 0:
            continue
        key = r.kind + ":" + r.summary.split("[")[0]
        by.setdefault(key, []).append(r)
    out = {"classes": {}}
    for k, rs in by.items():
        z = np.array([r.effect / r.se for r in rs if np.isfinite(r.effect)])
        out["classes"][k] = {"n": len(z), "sd_z": float(np.std(z, ddof=1)) if len(z) > 2 else None}
    exp, obs = 0.0, 0
    for r in res:
        if not r.judged or r.summary == "shape KS" or not np.isfinite(r.se) or r.se <= 0:
            continue
        m = r.tol - 1.645 * r.se
        exp += max(0.0, 2 * ST.phi(m / r.se) - 1) if m > 0 else 0.0
        obs += r.verdict == "agrees"
    out["agree_expected"], out["agree_observed"] = exp, obs
    out["disagrees"] = sum(r.verdict == "disagrees" for r in res)
    sd_ok = all(v["sd_z"] is None or 0.8 <= v["sd_z"] <= 1.25 for v in out["classes"].values())
    out["pass_a"], out["pass_b"], out["pass_c"] = sd_ok, obs >= 0.8 * exp, out["disagrees"] == 0
    out["pass"] = out["pass_a"] and out["pass_b"] and out["pass_c"]
    return out


def write(res, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    js = {"summary": summarize(res), "results": [asdict(r) for r in res]}
    if extra:
        js.update(extra)
    json.dump(js, open(path, "w"), indent=1, default=_json)
    log("wrote", path, js["summary"])


def null_sides(split):
    ref = EN.Side("D121", "d121", CM.D121_FILES)
    if split == "oddeven":
        return ref.subset_events([1, 3, 5, 7, 9]), ref.subset_events([2, 4, 6, 8, 10])
    if split == "halves":
        return ref.subset_events([1, 2, 3, 4, 5]), ref.subset_events([6, 7, 8, 9, 10])
    if split in ("randomA", "randomB"):
        salt = np.uint64(101 if split == "randomA" else 202)

        def pick(d, want):
            u = (ST.splitmix64(d["ev_uid"] ^ salt) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
            return (u < 0.5) == want
        return (ref.subset_events(keep_fn=lambda d: pick(d, True), label=f"D121[{split} A]"),
                ref.subset_events(keep_fn=lambda d: pick(d, False), label=f"D121[{split} B]"))
    raise SystemExit(f"unknown split {split}")


def stage_null(a):
    A, B = null_sides(a.split)
    res = run_compare(A, B, a, label=f"null:{a.split}")
    crit = null_criteria(res)
    write(res, str(CM.OUT / "null" / f"{a.split}.json"), {"criteria": crit, "split": a.split})
    log("null criteria:", json.dumps(crit, default=_json))


def stage_compare(a):
    A = EN.Side("CMSSW_17 old TTbar", "c17", CM.C17_FILES)
    B = EN.Side("D121", "d121", CM.D121_FILES)
    res = run_compare(A, B, a, label="compare")
    write(res, str(CM.OUT / "compare" / f"result_{a.mixed_rule}.json"), {"mixed_rule": a.mixed_rule})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("freeze", "null", "fallback", "defects", "defects-report", "compare"))
    ap.add_argument("--split", default="oddeven")
    ap.add_argument("--n-boot", type=int, default=CM.N_BOOT)
    ap.add_argument("--n-perm", type=int, default=CM.N_PERM)
    ap.add_argument("--mixed-rule", default="sole", choices=("sole", "any"))
    ap.add_argument("--defect", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.stage == "freeze":
        a.n_boot = min(a.n_boot, 100)
        stage_freeze(a)
    elif a.stage == "null":
        stage_null(a)
    elif a.stage == "fallback":
        stage_fallback(a)
    elif a.stage == "defects":
        import defects as DF
        DF.stage(a, run_compare, null_sides, write)
    elif a.stage == "defects-report":
        import defects as DF
        sys.exit(0 if DF.report() else 1)
    else:
        stage_compare(a)


if __name__ == "__main__":
    main()
