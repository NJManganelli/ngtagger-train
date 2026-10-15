"""Driver of the WP-C entry jobs (PLAN §6). Every job runs through ops/spheavy --frozen, in the background.

  cutfit   canon_v3 files 1-2, AAAA / AAAI / IAAA: PM (producer covariance) and INN (calibrated) with the
           own-cluster diagnostics and the leave-one-out chi2; fits the 99 % cuts (cuts.fit, PD1 + PD8; the
           rr cuts per scenario, PD15); reports the unconditioned cuts, PD1's chi2-vs-helix comparison and
           the local disc pulls (C1b); stores the own rows and the leave-one-out rows
  loocheck PD17, file 4 only: the smoothed leave-one-out's coverage per number of remaining inner hits,
           before / after, and the end-to-end smoke of the hit gate + reject-and-refit
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
            arms = (AR.pm(cfg), AR.inn(cfg))
            rr_statistic, = {arm.loo_pred for arm in arms}
            for arm in arms:
                n = 0
                for start, A in CM.iterate(path, AR.branches(path), chunk=a.chunk, nev=a.nev):
                    ctx = AR.prepare(A, geo)
                    out = AR.run_arm(ctx, geo, cfg, scen, arm, tools, own=True, loo=True)
                    R = DG.mark_primary(out["own"])
                    prim, clean_track = loo_flags(out, R, ctx)
                    own[arm.cov_model].append({k: R[k] for k in OWN_KEEP})
                    L = out["loo"]
                    loo[arm.cov_model].append({"surface": L["surface"], "first": L["first"], "own_primary": prim,
                                               "clean_track": clean_track, "scen": np.full(len(prim), scen),
                                               "n_inner": L["n_inner"],
                                               **{f"d_{k}": v for k, v in L["dims"].items()}})
                    n += len(A)
                meta.append({"scen": scen, "file": f, "arm": arm.name, "events": n})
                print(f"[{time.time() - t0:7.1f}s] {scen} f{f:02d} {arm.name}: {n} events", flush=True)
    own = {k: CM.concat(v) for k, v in own.items()}
    loo = {k: CM.concat(v) for k, v in loo.items()}
    out_dir = os.path.join(a.out, "cuts")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "cutfit_loo_rows.npz"),
                        **{f"{cov}__{k}": v for cov, L in loo.items() for k, v in L.items()})
    for k in loo:
        loo[k]["dims"] = {d: loo[k].pop(f"d_{d}") for d in AR.DIMS}
    prov = {"files": [str(CM.input_path(s, f)) for s in a.scen for f in CM.FIT_FILES], "runs": meta,
            "angle_table": AR.AI.TABLE, "calibration": AR.CALIBRATION,
            "plan": "PLAN §1 item 5, §10 PD1, PD2, PD8, PD15, PD17"}
    CU.fit(own, loo, prov, rr_statistic, clean=True).save(os.path.join(out_dir, CM.CUTS.name))
    CU.fit(own, loo, prov, rr_statistic, clean=False).save(os.path.join(out_dir, f"{CM.CUTS.stem}_unconditioned.json"))
    summary = {"pd1": pd1_report(own, loo, prov, rr_statistic), "c1b_local_disc_pulls": disc_pulls(own["producer"]),
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


def pd1_report(own, loo, prov, rr_statistic):
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
        ch = CU.fit({cov: R}, {cov: loo[cov]}, prov, rr_statistic)
        hh = CU.fit({cov: Rh}, {cov: loo[cov]}, prov, rr_statistic)
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


BEFORE_CUTS = CM.OUT / "cuts" / "wpc_cuts_v2.json"      # PD17's "before": the final-state rr cuts
SMOKE_EVENTS = 10                                      # PD17's end-to-end smoke: the first events per scenario


def smoke_counts(ctx, geo, cfg, scen, tools, off):
    """PD17's end-to-end smoke on one chunk: INN with the hit gate and reject-and-refit on, against the
    same chunk's INN with both off (off). Counts: gated and removed hits (own / wrong), removals per
    track, accepted hits and wrong ones (population tracks), prompt either-harm, changed selections."""
    import fom as FM
    on = AR.run_arm(ctx, geo, cfg, scen, AR.inn(cfg, name="INN-gate-rr", gate=True, rr=True), tools)
    ktp, ttp = ctx["K"]["tp_global"], ctx["trk_tp"]
    own = lambda t, ci: (ci >= 0) & (ttp[t] >= 0) & (ktp[np.maximum(ci, 0)] == ttp[t])
    truth = FM.track_truth(ctx)
    tg, jg = np.nonzero(on["rec"]["gated"])
    tr, jr = np.nonzero(on["removed"])
    c = {"tracks": int(on["refit"].sum()), "gated": len(tg), "gated_own": int(own(tg, on["rec"]["ci"][tg, jg]).sum()),
         "removed": len(tr), "removed_own": int(own(tr, on["rec"]["ci"][tr, jr]).sum())}
    per = on["removed"].sum(1)
    c.update({f"tracks_removals_{k}": int((per == k).sum()) for k in range(3)})
    pr = truth["pop"] & (truth["origin"] == 0)
    c["prompt"] = int(pr.sum())
    for lab, out in (("off", off), ("on", on)):
        t, _, _, w = FM.hits_of_record(out["rec"], ctx)
        pop = truth["pop"][t]
        c[f"acc_{lab}"], c[f"wrong_{lab}"] = int(pop.sum()), int(w[pop].sum())
        hm = FM.harm(out["a"][:, 4], out["a"][:, 3], ctx["T"]["d0"], ctx["T"]["z0"], truth)["either"]
        c[f"harm_{lab}"] = int(hm[pr].sum())
    sel = lambda out: np.where(out["rec"]["acc"], out["rec"]["ci"], -1)
    c["changed"] = int(np.any(sel(on) != sel(off), axis=1).sum())
    return c


def scenario_split(scen_list, cov="calibrated"):
    """PD17: is the per-scenario rr split still needed? Per shared (surface, first) bin and dims, each
    scenario's smoothed 99 % quantile with its 68 % order-statistic interval, from the cut fit's rows."""
    z = np.load(os.path.join(os.path.dirname(CM.CUTS), "cutfit_loo_rows.npz"))
    L = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(cov + "__")}
    m0 = L["own_primary"] & L["clean_track"]
    out = {}
    for dims in ("xy", "xyab"):
        for S, f in sorted({(int(s), bool(f)) for s, f in zip(L["surface"][m0], L["first"][m0])}):
            row = {}
            for scen in scen_list:
                x = np.sort(L[f"d_{dims}"][m0 & (L["scen"] == scen) & (L["surface"] == S) & (L["first"] == f)])
                if len(x) < CU.MIN_N:
                    continue
                k = int(np.ceil(CU.LEVEL * (len(x) + 1))) - 1
                d = int(np.ceil(np.sqrt(len(x) * CU.LEVEL * (1 - CU.LEVEL))))
                row[scen] = {"n": len(x), "q99": float(x[min(k, len(x) - 1)]),
                             "lo": float(x[max(k - d, 0)]), "hi": float(x[min(k + d, len(x) - 1)])}
            if len(row) > 1:
                out[f"{dims}|{S}|{int(f)}"] = row
    return out


def loocheck(a):
    """PD17, file 4 only (no gate): the reject-and-refit cut's coverage per scenario x surface x first /
    later x number of remaining inner hits, before (the final-state leave-one-out with the v2 cuts) and
    after (the smoothed one with the v3 cuts), on the same rows; the per-class medians; whether the
    per-scenario split is still needed; and the end-to-end smoke of the hit gate + reject-and-refit (the
    first SMOKE_EVENTS events). Writes entry/loocheck_summary.json."""
    import gates as G
    t0 = time.time()
    geo = OP.load_geometry(CM.GEOMETRY)
    cuts = {"after": CU.Cuts.load(str(CM.CUTS)), "before": CU.Cuts.load(str(BEFORE_CUTS))}
    pred = {"after": "smoothed", "before": "final"}
    assert all(cuts[k].rr_statistic == pred[k] for k in pred)
    entry = os.path.join(a.out, "entry")
    res = {"cuts": {k: str(CM.CUTS if k == "after" else BEFORE_CUTS) for k in pred}, "plan": "PLAN §10 PD17",
           "material": [], "minor": [], "scenarios": {}, "smoke": {}}
    for scen in a.scen:
        path = CM.input_path(scen, CM.PRED_FILE)
        cfg = CM.producer_config(path)
        assert CM.mask_of(cfg) == scen
        tools = AR.Tools(cfg, cuts=cuts["after"])
        rows, smoke, n = {k: [] for k in pred}, [], 0
        for start, A in CM.iterate(path, AR.branches(path), chunk=a.chunk, nev=a.nev):
            ctx = AR.prepare(A, geo)
            outs = {}
            for k, mode in pred.items():
                out = AR.run_arm(ctx, geo, cfg, scen, AR.inn(cfg, loo_pred=mode), tools, own=True, loo=True)
                prim, clean_track = loo_flags(out, DG.mark_primary(out["own"]), ctx)
                L = out["loo"]
                rows[k].append({"ev": ctx["tev"][L["o"]] + start, "surface": L["surface"], "first": L["first"],
                                "n_inner": L["n_inner"], "own_primary": prim, "clean_track": clean_track,
                                **{f"d_{d}": v for d, v in L["dims"].items()}})
                outs[k] = out
            if start < SMOKE_EVENTS:
                smoke.append(smoke_counts(ctx, geo, cfg, scen, tools, outs["after"]))
            n += len(A)
        sc = {"events": n, "classes": {}, "medians": {}}
        for k in pred:
            L = CM.concat(rows[k])
            m = L["own_primary"] & L["clean_track"]
            sc["classes"][k] = {}
            for dims in AR.DIMS:
                r = G.e1b_consumer(cuts[k], "rr", CU.table_key("rr", dims, "calibrated", scen), L[f"d_{dims}"][m],
                                   L["surface"][m], L["first"][m], L["ev"][m], cls=L["n_inner"][m])
                mat, minor, t = G.e1b_class_verdict(scen, dims, r)
                sc["classes"][k][dims] = {**r, "t_class": t, "material": mat, "minor": minor}
                if k == "after":
                    res["material"] += mat
                    res["minor"] += minor
            sc["medians"][k] = {f"{int(S)}|{int(f)}|n{int(c)}": {"n": int(mm.sum()), "xy": float(np.median(L["d_xy"][mm])),
                                                                "xyab": float(np.median(L["d_xyab"][mm]))}
                                for S, f, c in sorted({(int(s), bool(f), int(c)) for s, f, c in
                                                       zip(L["surface"][m], L["first"][m], L["n_inner"][m])})
                                for mm in [m & (L["surface"] == S) & (L["first"] == f) & (L["n_inner"] == c)]}
        before = sc["classes"]["before"]
        sc["known_bad_fails"] = any(abs(b["z"]) > d["t_class"] for d in before.values() for b in d["bins"]
                                    if b["n_test"] >= CU.MIN_N and np.isfinite(b["z"]))
        if not sc["known_bad_fails"]:
            res["material"].append(f"rr per class {scen}: the final-state known-bad does not fail the per-class test")
        res["scenarios"][scen] = sc
        res["smoke"][scen] = {k: int(sum(c[k] for c in smoke)) for k in smoke[0]}
        print(f"[{time.time() - t0:7.1f}s] {scen}: file 4, {n} events", flush=True)
    res["scenario_split"] = scenario_split(a.scen)
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(entry, "loocheck_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({"material": res["material"], "minor": res["minor"], "smoke": res["smoke"]}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("cutfit", "loocheck"))
    ap.add_argument("--scen", nargs="+", default=list(CM.SCENARIOS))
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--out", default=str(CM.OUT))
    a = ap.parse_args(argv)
    return {"cutfit": cutfit, "loocheck": loocheck}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
