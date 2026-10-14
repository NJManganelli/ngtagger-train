"""Driver of the WP-C entry jobs (PLAN §6). Every job runs through ops/spheavy --frozen, in the background.

  cutfit   canon_v3 files 1-2, AAAA / AAAI / IAAA: PM (producer covariance) and INN (calibrated) with the
           own-cluster diagnostics and the leave-one-out chi2; fits the 99 % cuts (cuts.fit, PD1 + PD8);
           reports the unconditioned cuts, PD1's chi2-vs-helix comparison and the local disc pulls (C1b)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import arms as AR
import common as CM
import cuts as CU
import diagnostics as DG
from common import OP
from ngtagger.spdefs import angles as SA

OWN_KEEP = ("o", "step", "first", "surface", "ci", "mi", "clean", "e1", "t1", "z", "has", "g_xy", "g_xya",
            "g_xyb", "g_xyab", "rank_chi2", "classT", "table_pop", "keybin", "orient", "helix_d", "tp_pt",
            "primary", "helix_primary", "secondary")


def loo_flags(out, R, ctx):
    """own_primary: the accepted hit is an own primary (joined on (track, step, cluster)); clean_track:
    every accepted hit of the track is its own TP's."""
    L = out["loo"]
    key = lambda o, s, c: (np.asarray(o, np.int64) * 64 + np.asarray(s, np.int64)) * (1 << 31) + np.asarray(c, np.int64)
    pr = R["primary"]
    prim = np.isin(key(L["o"], L["step"], L["ci"]), key(R["o"][pr], R["step"][pr], R["ci"][pr]))
    rec = out["rec"]
    t, j = np.nonzero(rec["acc"])
    wrong = ctx["K"]["tp_global"][rec["ci"][t, j]] != ctx["trk_tp"][t]
    bad = np.zeros(ctx["ntrk"], bool)
    bad[t[wrong]] = True
    return prim, ~bad[L["o"]]


def cutfit(a):
    t0 = time.time()
    geo = OP.load_geometry(CM.GEOMETRY)
    own = {"producer": [], "calibrated": []}
    loo = {"producer": [], "calibrated": []}
    meta = []
    for scen in a.scen:
        for f in CM.FIT_FILES:
            path = CM.input_path(scen, f)
            cfg = CM.producer_config(path)
            assert CM.mask_of(cfg) == scen
            tools = AR.Tools(cfg)
            for arm in (AR.pm(cfg), AR.inn(cfg)):
                n = 0
                for start, A in CM.iterate(path, AR.branches(path), chunk=a.chunk, nev=a.nev):
                    ctx = AR.prepare(A, geo)
                    out = AR.run_arm(ctx, geo, cfg, scen, arm, tools, own=True, loo=True)
                    R = DG.mark_primary(out["own"])
                    prim, clean_track = loo_flags(out, R, ctx)
                    own[arm.cov_model].append({k: R[k] for k in OWN_KEEP})
                    L = out["loo"]
                    loo[arm.cov_model].append({"surface": L["surface"], "first": L["first"], "own_primary": prim,
                                               "clean_track": clean_track,
                                               **{f"d_{k}": v for k, v in L["dims"].items()}})
                    n += len(A)
                meta.append({"scen": scen, "file": f, "arm": arm.name, "events": n})
                print(f"[{time.time() - t0:7.1f}s] {scen} f{f:02d} {arm.name}: {n} events", flush=True)
    own = {k: CM.concat(v) for k, v in own.items()}
    loo = {k: CM.concat(v) for k, v in loo.items()}
    for k in loo:
        loo[k]["dims"] = {d: loo[k].pop(f"d_{d}") for d in AR.DIMS}
    prov = {"files": [str(CM.input_path(s, f)) for s in a.scen for f in CM.FIT_FILES], "runs": meta,
            "angle_table": AR.AI.TABLE, "calibration": AR.CALIBRATION, "plan": "PLAN §1 item 5, §10 PD1, PD2, PD8"}
    out_dir = os.path.join(a.out, "cuts")
    os.makedirs(out_dir, exist_ok=True)
    CU.fit(own, loo, prov, clean=True).save(os.path.join(out_dir, "wpc_cuts_v1.json"))
    CU.fit(own, loo, prov, clean=False).save(os.path.join(out_dir, "wpc_cuts_v1_unconditioned.json"))
    summary = {"pd1": pd1_report(own, loo, prov), "c1b_local_disc_pulls": disc_pulls(own["producer"]),
               "populations": population_counts(own, loo), "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "cutfit_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(out_dir, "cutfit_own_rows.npz"),
                        **{f"{cov}__{k}": v for cov, R in own.items() for k, v in R.items()})
    print(f"[{time.time() - t0:7.1f}s] cutfit done", flush=True)
    return 0


def population_counts(own, loo):
    out = {}
    for cov in own:
        g, w, r = CU.populations(own[cov], loo[cov], clean=True)
        g0, w0, r0 = CU.populations(own[cov], loo[cov], clean=False)
        out[cov] = {"gate": int(g.sum()), "gate_unconditioned": int(g0.sum()), "window": int(w.sum()),
                    "window_unconditioned": int(w0.sum()), "rr": int(r.sum()), "rr_unconditioned": int(np.sum(r0))}
    return out


def pd1_report(own, loo, prov):
    """PD1: how often the chi2 choice of the own primary differs from the helix-nearest choice, the
    secondary and off-path shares per surface, and the gate cuts fitted on the helix choice relative to
    the chi2 choice."""
    out = {}
    for cov, R in own.items():
        T = R["classT"]
        e = {"differ_frac": float(np.mean(R["primary"][T] != R["helix_primary"][T])), "per_surface": {}}
        for S in np.unique(R["surface"]):
            mS = R["surface"] == S
            e["per_surface"][int(S)] = {"secondary_share_of_classT": float(np.mean(R["secondary"][T & mS])),
                                        "off_path_share_of_own": float(np.mean(~R["classT"][mS]))}
        Rh = dict(R)
        Rh["primary"] = R["helix_primary"]
        ch = CU.fit({cov: R}, {cov: loo[cov]}, prov)
        hh = CU.fit({cov: Rh}, {cov: loo[cov]}, prov)
        ratio = {}
        for dims in ("xy", "xyab"):
            a_, b_ = ch.doc["cuts"][f"gate|{dims}|{cov}"], hh.doc["cuts"][f"gate|{dims}|{cov}"]
            ratio[dims] = {k: (b_[k]["value"] / a_[k]["value"]) if (a_[k]["value"] and k in b_ and b_[k]["value"])
                           else None for k in a_ if "|" in k}
        e["gate_cut_ratio_helix_over_chi2"] = ratio
        out[cov] = e
    return out


def disc_pulls(R):
    """C1b input (review L1): local disc angle pulls (published local sigma, PM's prediction, clean history)
    per disc x module facing x predicted-key bin: centre (25 %-trimmed mean) and weighting width."""
    out = {}
    m0 = R["table_pop"] & R["clean"] & (R["surface"] > 10)
    for S in np.unique(R["surface"][m0]):
        for fac in (-1, 1):
            for i, ang in enumerate(("alpha", "beta")):
                m = m0 & (R["surface"] == S) & (R["orient"] == fac) & np.isfinite(R["t1"][:, i])
                for b in np.unique(R["keybin"][m, i]):
                    x = R["t1"][m & (R["keybin"][:, i] == b), i]
                    if len(x) < 50:
                        continue
                    c = SA.centre_of(x)
                    out[f"{int(S)}|{fac}|{ang}|{int(b)}"] = {"n": int(len(x)), "centre": c,
                                                            "width": SA.weighting_width(np.abs(x - c))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("cutfit",))
    ap.add_argument("--scen", nargs="+", default=list(CM.SCENARIOS))
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--out", default=str(CM.OUT))
    a = ap.parse_args(argv)
    return {"cutfit": cutfit}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
