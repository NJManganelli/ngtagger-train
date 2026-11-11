"""Post-scan items of the SmartPixels output-bits refit study, checkpointed like the scan (run through ops/spheavy).

  python postscan.py supervise --workers 2      start the workers and wait (the command the lane runs)
  python postscan.py worker                     one worker
  python postscan.py status

Units (a few minutes each; each writes its outputs atomically and then a marker <name>.done.json under
checks/postscan/; a finished unit is skipped; a unit runs only when the units it depends on are done; workers claim a
unit with an exclusive claim file, a claim of a dead process is taken over):

  defects_A, defects_B, defects_clusters, defects_eval   the planted defects (defects.py) -> checks/planted_defects.json
  d0sign               file 4 AAAA, the 8 priority arms -> checks/d0_sign_AAAA_f04.json (Pearson, Spearman, same-sign
                       share, median refit / true d0, on b/c daughters with |d0 true| > 200 um)
  rerun                the finished scan unit priority/AAAA_f05 recomputed and compared bit for bit
  sanity               the no-physics sanity pass (sanity.py) over every group of scan units
  cutrows_<scen>_f0N   (files 1, 2) the 16-bit point's correct-hit chi2 rows for the skip-price fit
  cuts                 the skip price P re-derived for 4/4/4/4: the gate|xyab table refitted by the existing cut fit
                       (cuts._fit_table, own primaries with a clean history, 99 % point) on those rows
  skip_<scen>_fNN      (scan files) the 4/4/4/4 arm with the re-derived P -> checks/postscan/skipprice/
  skip_eval            d0 / z0 68 % half-widths against the scan's alloc_4444 -> checks/skip_price_rederived.json
  direct_<scen>_fNN    (all ten files) the direct network-position point: barrel-only tracks with particle pT >= 50 GeV,
                       crossing clusters at the particle's helix + the drawn network position residual
  direct_eval          -> checks/direct_network_point.json (shown only if at least 200 such tracks)

Stops cleanly below 12 GB free; waits while the external disk is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

import run_scan as RS
from run_scan import log

HERE = Path(__file__).resolve().parent
PT_DIRECT = 50.0
FIT_FILES = (1, 2)
SKIP_ARM, SKIP_NAME = "alloc_4444", "alloc_4444_skipprice"
PRIORITY = ["implemented_now", "implemented_now_sizeblind", "implemented_now_newtable", "network_float", "alloc_4444",
            "alloc_4444_wholesensor", "alloc_6604", "bending_float"]
FORECAST = {"pos_err_scale_x": 7.6 / 12.3, "pos_err_scale_y": 2.0 / 3.1}
NEV = None                # development only (job "one" with --nev): the first events of each file
DEV_DIR = None            # development only: every output under this directory


def checks_dir():
    import study as ST
    return Path(DEV_DIR) if DEV_DIR else ST.LOCAL / "checks"


def out_dir():
    d = checks_dir() / "postscan"
    d.mkdir(parents=True, exist_ok=True)
    return d


def units():
    import study as ST
    u = [("defects_A", []), ("defects_B", []), ("defects_clusters", []),
         ("defects_eval", ["defects_A", "defects_B", "defects_clusters"]), ("d0sign", []), ("rerun", []), ("sanity", [])]
    rows = [f"cutrows_{s}_f{f:02d}" for f in FIT_FILES for s in ST.SCENARIOS]
    u += [(r, []) for r in rows] + [("cuts", rows)]
    skips = [f"skip_{s}_f{f:02d}" for f in ST.SCAN_FILES for s in ST.SCENARIOS]
    u += [(k, ["cuts"]) for k in skips] + [("skip_eval", skips)]
    direct = [f"direct_{s}_f{f:02d}" for f in ST.ALL_FILES for s in ST.SCENARIOS]
    u += [(k, []) for k in direct] + [("direct_eval", direct)]
    return u


def done(name):
    return (out_dir() / f"{name}.done.json").exists()


def mark(name, info=None):
    tmp = out_dir() / f"{name}.done.json.tmp{os.getpid()}"
    json.dump({"unit": name, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})}, open(tmp, "w"), indent=1,
              default=float)
    os.replace(tmp, out_dir() / f"{name}.done.json")


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    json.dump(doc, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, path)


def claim(name):
    c = out_dir() / f"{name}.claim"
    try:
        fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            os.kill(int(c.read_text() or "0"), 0)
            return False
        except (OSError, ValueError):
            c.unlink(missing_ok=True)                       # the claimant is gone: take it at the next pass
            return False


def w68(x):
    x = np.abs(x[np.isfinite(x)])
    return float(np.quantile(x, 0.6827)) if len(x) else float("nan")


# ---------------------------------------------------------------- units ----
def u_d0sign(S):
    from scipy import stats
    reg = json.load(open(HERE / "arms.json"))
    truth, D = RS.run_unit(S, reg, 4, "AAAA", PRIORITY, nev=NEV)
    RS.save_npz(out_dir() / "d0sign_AAAA_f04_truth.npz", truth)
    RS.save_npz(out_dir() / "d0sign_AAAA_f04_arms.npz", {f"{n}__{k}": v for n, d in D.items() for k, v in d.items()})
    m = (truth["origin"] == 1) & (np.abs(truth["tp_d0"]) > 0.02)
    res = {"what": "refit d0 against true d0, b/c daughters with |d0 true| > 200 um, file 4 AAAA; pre-declared: Pearson "
                   ">= 0.9; proposed (ruling with the physicist): same-sign share >= 0.9, Spearman >= 0.8, median "
                   "refit / true in [0.9, 1.1]", "n_tracks": int(m.sum()), "arms": {}}
    for n, d in D.items():
        ok = m & d["refit"]
        x, y = truth["tp_d0"][ok].astype(float), d["a"][ok, 4].astype(float)
        e = {"n_refit": int(ok.sum()), "pearson": float(np.corrcoef(x, y)[0, 1]), "spearman": float(stats.spearmanr(x, y)[0]),
             "same_sign_share": float((np.sign(x) == np.sign(y)).mean()), "median_refit_over_true": float(np.median(y / x))}
        e["passes_pre_declared_pearson_0.9"] = bool(e["pearson"] >= 0.9)
        e["passes_proposed_form"] = bool(e["same_sign_share"] >= 0.9 and e["spearman"] >= 0.8
                                         and 0.9 <= e["median_refit_over_true"] <= 1.1)
        res["arms"][n] = e
    import study as ST
    write_json(checks_dir() / "d0_sign_AAAA_f04.json", res)
    return {"arms": len(D)}


def u_rerun(S):
    import study as ST
    reg = RS.load_registry(ST.LOCAL)
    u = next(x for x in RS.units_of(reg, ST.SCAN_FILES, ST.SCENARIOS)
             if x["group"] == "priority" and x["scen"] == "AAAA" and x["file"] == 5)
    truth, arms = RS.run_unit(S, reg, 5, "AAAA", u["arms"], nev=NEV)
    Z = np.load(ST.LOCAL / "units" / u["name"])
    T = np.load(ST.LOCAL / "units" / "truth" / "AAAA_f05_truth.npz")
    bad = [f"{n}__{k}" for n, d in arms.items() for k, v in d.items() if not np.array_equal(v, Z[f"{n}__{k}"], equal_nan=True)]
    bad += [f"truth {k}" for k, v in truth.items() if not np.array_equal(v, T[k], equal_nan=True)]
    res = {"unit": u["name"], "arms": u["arms"], "arrays_compared": sum(len(d) for d in arms.values()) + len(truth),
           "tracks": int(len(truth["run"])), "different": bad, "bit_equal": not bad,
           "note": "recomputed from the working tree (branch spixbits) against the npz the frozen scan job stored"}
    write_json(checks_dir() / "rerun_bitequal_AAAA_f05_priority.json", res)
    return res


def u_sanity(S):
    import study as ST
    reg = RS.load_registry(ST.LOCAL)
    summary = {}
    for g in reg["group_order"]:
        r = subprocess.run([sys.executable, str(HERE / "sanity.py"), "--group", g], capture_output=True, text=True)
        d = json.load(open(checks_dir() / f"sanity_{g}.json"))
        summary[g] = {k: d[k] for k in ("units", "finished", "loadable", "arms_checked", "pass")}
        summary[g]["problems"] = d["problems"][:5]
    write_json(checks_dir() / "sanity_all_groups.json", summary)
    return summary


def _iter(S, scen, f, extra=()):
    ST, EM, AR, CM = S.ST, S.EM, S.AR, S.CM
    path = ST.input_path(scen, f)
    RS.wait_for_input(str(path))
    cfg = CM.producer_config(path)
    S.cfg = cfg
    keys = AR.OP._keys(str(path))
    br = list(dict.fromkeys(AR.branches(path) + ["run", "luminosityBlock", "event"]
                            + AR.SPN.cluster_branches(keys, list(EM.EXTRA_CLUSTER_COLS), disc_scope="all")))
    tools = ST.tools(cfg)
    for start, A in CM.iterate(path, br, chunk=10, nev=NEV):
        ctx = AR.prepare(A, S.geo)
        yield cfg, tools, start, A, ctx, EM.base(ctx, S.geo, A, S.net, S.links, S.LT)


def u_cutrows(S, scen, f):
    """The skip-price fit rows of the 16-bit point: the existing own-cluster diagnostics (K = 1, as the cut fit ran)
    with the arm's emulated clusters, measurements and errors; the statistic is the gate chi2 over positions and both
    angles plus the one-sided cost of a flag code (what the two-layer choice compares with P); sent clusters only."""
    import diagnostics as DG
    AR, EM = S.AR, S.EM
    spec = next(a["spec"] for a in json.load(open(HERE / "arms.json"))["arms"] if a["name"] == SKIP_ARM)
    parts = []
    for cfg, tools, start, A, ctx, B in _iter(S, scen, f):
        arm = replace(RS.build_arm(AR, cfg, SKIP_ARM, spec), K=1)
        t_a = S.tools_for(tools, spec)
        c = EM.emulate(ctx, B, spec, S.LT)
        R = DG.mark_primary(AR.run_arm(c, S.geo, cfg, scen, arm, t_a, own=True)["own"])
        if not len(R.get("v", [])):
            continue
        cost, _ = AR.flag_cost(arm, "rank", c, S.geo, R["ci"], R["mi"], R["cot_pred"][:, 0], R["cot_pred"][:, 1], t_a)
        parts.append({"g": R["g_xyab"] + cost, "g_no_cost": R["g_xyab"], "csurface": R["csurface"], "first": R["first"],
                      "gate_pop": R["primary"] & R["table_pop"] & R["clean"] & B["sent"][R["ci"]]})
    RS.save_npz(out_dir() / "skipprice" / f"cutrows_{scen}_f{f:02d}.npz", S.CM.concat(parts))
    return {"rows": int(sum(len(p["g"]) for p in parts))}


def cuts_path():
    return out_dir() / "skipprice" / "cuts_alloc_4444.json"


def u_cuts(S):
    import cuts as CU
    ST = S.ST
    rows = [np.load(out_dir() / "skipprice" / f"cutrows_{s}_f{f:02d}.npz") for f in FIT_FILES for s in ST.SCENARIOS]
    R = {k: np.concatenate([r[k] for r in rows]) for k in ("g", "csurface", "first", "gate_pop")}
    m = R["gate_pop"]
    doc = json.load(open(ST.PAYLOAD / "wpc_cuts_c1b_v1.json"))
    old = doc["cuts"]["gate|xyab|calibrated"]
    new = CU._fit_table(R["g"][m], R["csurface"][m], R["first"][m])
    doc["cuts"]["gate|xyab|calibrated"] = new
    doc["provenance"] = {"base": "wpc_cuts_c1b_v1.json (every other table unchanged)",
                         "refitted": "gate|xyab|calibrated: the skip price of the 4/4/4/4 sub-area arm, canon_v5 files 1-2 x "
                                     "AAAA / AAAI / IAAA pooled, own primaries with a clean history, sent clusters"}
    write_json(cuts_path(), doc)
    cmp_ = {k: {"today": old[k]["value"], "today_n": old[k]["n"], "rederived": new.get(k, {}).get("value"),
                "rederived_n": new.get(k, {}).get("n")} for k in old}
    write_json(out_dir() / "skipprice" / "skip_price_table_comparison.json",
               {"min_n_for_a_bin": CU.MIN_N, "n_fit_values": int(m.sum()), "bins (surface | first step)": cmp_})
    return {"n_fit_values": int(m.sum()), "pooled_P_today": old["*"]["value"], "pooled_P_rederived": new["*"]["value"]}


def u_skip(S, scen, f):
    reg = json.load(open(HERE / "arms.json"))
    r = json.loads(json.dumps(next(a for a in reg["arms"] if a["name"] == SKIP_ARM)))
    r["name"] = SKIP_NAME
    r["spec"]["cuts_file"] = str(cuts_path())
    reg["arms"].append(r)
    truth, arms = RS.run_unit(S, reg, f, scen, [SKIP_NAME], nev=NEV)
    RS.save_npz(out_dir() / "skipprice" / f"{scen}_f{f:02d}.npz", {f"{n}__{k}": v for n, d in arms.items() for k, v in d.items()})
    return {"tracks": int(len(truth["run"]))}


def _boot_ratio(ev, ra, rb, m, n=200, seed=20261111):
    """68 % half-width ratio W(a) / W(b) on tracks m and its paired event-bootstrap sd."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(ev[m], return_inverse=True)
    xa, xb = np.abs(ra[m]), np.abs(rb[m])
    val = w68(xa) / w68(xb)
    reps = []
    for _ in range(n):
        w = rng.multinomial(len(u), np.full(len(u), 1 / len(u)))[inv]
        o = np.repeat(np.arange(len(xa)), w)
        reps.append(np.quantile(xa[o], 0.6827) / np.quantile(xb[o], 0.6827))
    return val, float(np.std(reps))


def u_skip_eval(S):
    ST = S.ST
    reg = RS.load_registry(ST.LOCAL)
    res = {"what": "the 4/4/4/4 sub-area arm with the skip price re-derived for it (files 1-2) against the scan's value "
                   "(today's skip price): 68 % half-widths of (delivered - true) on prompt population tracks, scan files, "
                   "paired by track; ratio = re-derived / scan with its paired event-bootstrap sd. A bound on what not "
                   "recalibrating costs, not a tuning.", "scenarios": {}}
    for scen in ST.SCENARIOS:
        P = {k: [] for k in ("ev", "d0_s", "d0_r", "z0_s", "z0_r", "refit_s", "refit_r", "acc_s", "acc_r", "cross_s", "cross_r",
                             "skipchosen_s", "skipchosen_r")}
        for fi, f in enumerate(ST.SCAN_FILES):
            u = next(x for x in RS.units_of(reg, ST.SCAN_FILES, ST.SCENARIOS)
                     if SKIP_ARM in x["arms"] and x["scen"] == scen and x["file"] == f)
            Z = np.load(ST.LOCAL / "units" / u["name"])
            N = np.load(out_dir() / "skipprice" / f"{scen}_f{f:02d}.npz")
            T = np.load(ST.LOCAL / "units" / "truth" / f"{scen}_f{f:02d}_truth.npz")
            m = T["origin"] == 0
            P["ev"].append(T["entry"][m].astype(np.int64) + 1000 * fi)
            for tag, src, name in (("s", Z, SKIP_ARM), ("r", N, SKIP_NAME)):
                P["d0_" + tag].append((src[f"{name}__a"][m, 4] - T["tp_d0"][m]) * 1e4)
                P["z0_" + tag].append((src[f"{name}__a"][m, 3] - T["tp_z0"][m]) * 1e4)
                P["refit_" + tag].append(src[f"{name}__refit"][m])
                P["acc_" + tag].append(src[f"{name}__n_accepted"][m])
                P["cross_" + tag].append((src[f"{name}__hit_code"][m] == 1).sum(1))
                P["skipchosen_" + tag].append((src[f"{name}__hit_choice"][m] == 2).sum(1))
        P = {k: np.concatenate(v) for k, v in P.items()}
        e = {"prompt_tracks": int(len(P["ev"]))}
        allm = np.ones(len(P["ev"]), bool)
        for q in ("d0", "z0"):
            r, sd = _boot_ratio(P["ev"], P[q + "_r"], P[q + "_s"], allm)
            e[q] = {"w68_scan_um": w68(P[q + "_s"]), "w68_rederived_um": w68(P[q + "_r"]), "ratio": r, "ratio_sd": sd}
        for k, lab in (("refit", "refit_fraction"), ("acc", "accepted_hits_per_track"),
                       ("cross", "accepted_crossing_clusters_per_track"), ("skipchosen", "windows_committed_to_skip_per_track")):
            e[lab] = {"scan": float(P[k + "_s"].mean()), "rederived": float(P[k + "_r"].mean())}
        res["scenarios"][scen] = e
    res["skip_price_tables"] = json.load(open(out_dir() / "skipprice" / "skip_price_table_comparison.json"))
    write_json(checks_dir() / "skip_price_rederived.json", res)
    return {s: {q: [round(e[q]["ratio"], 4), round(e[q]["ratio_sd"], 4)] for q in ("d0", "z0")} for s, e in res["scenarios"].items()}


DIRECT_ARMS = {"network_float": {},                                   # today's positions (the stand-in), as in the scan
               "direct_network_errors": {"direct_positions": True, **FORECAST},     # helix + network residual, network errors
               "direct_today_errors": {"direct_positions": True}}     # the same positions, today's size-blind errors


def u_direct(S, scen, f):
    """Barrel-only population tracks with particle pT >= 50 GeV: the three DIRECT_ARMS run on those tracks only. The
    crossing clusters of these tracks sit at the particle's helix crossing of the module + the network's position
    residual of the cluster's own draw (CMS localX residual = - SmartPixels y residual, localY = - SmartPixels x)."""
    import fom as FM
    AR, EM, CM = S.AR, S.EM, S.CM
    base = next(a["spec"] for a in json.load(open(HERE / "arms.json"))["arms"] if a["name"] == "network_float")
    rows = []
    for cfg, tools, start, A, ctx, B in _iter(S, scen, f):
        tr = FM.track_truth(ctx)
        arm0 = RS.build_arm(AR, cfg, "network_float", base)
        path_, seq = AR.visit_plan(ctx, S.geo, cfg, scen, arm0)
        sel = tr["pop"] & (tr["pt"] >= PT_DIRECT) & ~AR.disc_going(path_) & (np.trace(ctx["C"], axis1=1, axis2=2) > 0)
        pop = np.flatnonzero(sel)
        if not len(pop):
            continue
        cx = RS.crossing_clusters(S, ctx, pop, seq, CM.pt_constant(ctx))
        ev_ = cx[6]
        n = len(B["x"])
        B["direct_x"], B["direct_y"] = np.full(n, np.nan), np.full(n, np.nan)
        j = B["draw_row"][ev_["ci"]]
        ok = (j >= 0) & np.isfinite(ev_["u"]) & np.isfinite(ev_["v"])
        B["direct_x"][ev_["ci"][ok]] = ev_["u"][ok] - S.net.ry_um[j[ok]] * 1e-4
        B["direct_y"][ev_["ci"][ok]] = ev_["v"][ok] - S.net.rx_um[j[ok]] * 1e-4
        toff = np.concatenate([[0], np.cumsum(ctx["T"]["_n"])])
        e = ctx["tev"][pop]
        d = {"run": np.asarray(A["run"])[e].astype(np.int64), "lumi": np.asarray(A["luminosityBlock"])[e].astype(np.int64),
             "event": np.asarray(A["event"])[e].astype(np.int64), "track": (pop - toff[e]).astype(np.int32),
             "entry": (start + e).astype(np.int32), "tp_pt": tr["pt"][pop], "tp_eta": tr["eta"][pop],
             "tp_d0": tr["d0"][pop], "tp_z0": tr["z0"][pop], "origin": tr["origin"][pop],
             "n_crossing_replaced": np.bincount(ev_["track"][ok], minlength=len(pop)).astype(np.int8)}
        for name, extra in DIRECT_ARMS.items():
            spec = {**base, **extra}
            out = AR.run_arm(EM.emulate(ctx, B, spec, S.LT), S.geo, cfg, scen, RS.build_arm(AR, cfg, name, spec),
                             S.tools_for(tools, spec), tracks=sel)
            d[f"{name}__a"] = out["a"][pop]
            d[f"{name}__cov_diag"] = np.einsum("nii->ni", out["C"][pop])
            d[f"{name}__refit"] = out["refit"][pop]
            d[f"{name}__n_accepted"] = out["nacc"][pop]
        rows.append(d)
    data = CM.concat(rows) if rows else {"run": np.zeros(0, np.int64)}
    RS.save_npz(out_dir() / "direct" / f"{scen}_f{f:02d}.npz", data)
    return {"tracks": int(len(data["run"]))}


def u_direct_eval(S):
    ST = S.ST
    reg = RS.load_registry(ST.LOCAL)
    res = {"what": "the direct network-position point (PLAN 4a item 3): barrel-only population tracks with particle pT >= "
                   f"{PT_DIRECT:g} GeV; network_float = today's positions (stand-in, upper bound); direct_network_errors = "
                   "crossing clusters at helix + drawn network position residual with the network's position errors; "
                   "direct_today_errors = the same positions with today's size-blind errors. 68 % half-widths of "
                   "(delivered - true) in um. Shown only with at least 200 tracks.", "min_tracks": 200, "samples": {}}
    for label, files in (("scan files 5 7 8 9 10", ST.SCAN_FILES), ("all ten files", ST.ALL_FILES)):
        res["samples"][label] = {}
        for scen in ST.SCENARIOS:
            Z = [np.load(out_dir() / "direct" / f"{scen}_f{f:02d}.npz") for f in files]
            Z = [z for z in Z if len(z["run"])]
            n = int(sum(len(z["run"]) for z in Z))
            e = {"tracks": n, "shown": n >= 200}
            if n:
                cat = lambda k: np.concatenate([z[k] for z in Z])
                prompt = cat("origin") == 0
                e["prompt_tracks"] = int(prompt.sum())
                e["crossing_clusters_replaced_per_track"] = float(cat("n_crossing_replaced").mean())
                for name in DIRECT_ARMS:
                    a = cat(f"{name}__a")
                    e[name] = {"refit_fraction": float(cat(f"{name}__refit").mean()),
                               "accepted_hits_per_track": float(cat(f"{name}__n_accepted").mean()),
                               "d0_w68_um": w68((a[:, 4] - cat("tp_d0")) * 1e4), "z0_w68_um": w68((a[:, 3] - cat("tp_z0")) * 1e4),
                               "d0_w68_um_prompt": w68(((a[:, 4] - cat("tp_d0")) * 1e4)[prompt]),
                               "z0_w68_um_prompt": w68(((a[:, 3] - cat("tp_z0")) * 1e4)[prompt]),
                               "d0_sigma_median_um": float(np.median(np.sqrt(cat(f"{name}__cov_diag")[:, 4])) * 1e4),
                               "z0_sigma_median_um": float(np.median(np.sqrt(cat(f"{name}__cov_diag")[:, 3])) * 1e4)}
            res["samples"][label][scen] = e
    # consistency: network_float run on the selected tracks alone against the scan's stored value (scan files)
    worst, ncmp = 0.0, 0
    for scen in ST.SCENARIOS:
        for f in ST.SCAN_FILES:
            z = np.load(out_dir() / "direct" / f"{scen}_f{f:02d}.npz")
            if not len(z["run"]):
                continue
            T = np.load(ST.LOCAL / "units" / "truth" / f"{scen}_f{f:02d}_truth.npz")
            u = next(x for x in RS.units_of(reg, ST.SCAN_FILES, ST.SCENARIOS)
                     if "network_float" in x["arms"] and x["scen"] == scen and x["file"] == f)
            a = np.load(ST.LOCAL / "units" / u["name"])["network_float__a"]
            key = {(int(r), int(l), int(e), int(t)): i for i, (r, l, e, t) in enumerate(zip(T["run"], T["lumi"], T["event"], T["track"]))}
            idx = np.array([key[(int(r), int(l), int(e), int(t))] for r, l, e, t in zip(z["run"], z["lumi"], z["event"], z["track"])])
            worst = max(worst, float(np.abs(a[idx] - z["network_float__a"].astype(np.float32)).max()))
            ncmp += len(idx)
    res["network_float_on_selected_tracks_vs_scan"] = {"tracks_compared": ncmp, "max_abs_parameter_difference": worst,
                                                       "identical": worst == 0.0}
    write_json(checks_dir() / "direct_network_point.json", res)
    return {k: {s: v[s]["tracks"] for s in v} for k, v in res["samples"].items()}


def run_one(S, name):
    import defects as DF
    if name in ("defects_A", "defects_B"):
        DF.arms_part(S, name[-1], out_dir(), NEV)
        return {}
    if name == "defects_clusters":
        DF.clusters_part(S, out_dir(), 50 if NEV is None else min(NEV, 50))
        return {}
    if name == "defects_eval":
        return {"detected": DF.evaluate(out_dir(), checks_dir())["detected"]}
    simple = {"d0sign": u_d0sign, "rerun": u_rerun, "sanity": u_sanity, "cuts": u_cuts, "skip_eval": u_skip_eval,
              "direct_eval": u_direct_eval}
    if name in simple:
        return simple[name](S)
    kind, scen, f = name.split("_")
    return {"cutrows": u_cutrows, "skip": u_skip, "direct": u_direct}[kind](S, scen, int(f[1:]))


def worker(a):
    import study as ST
    S = RS.Study()
    log(f"postscan worker pid {os.getpid()}")
    while True:
        todo = [(n, deps) for n, deps in units() if not done(n)]
        if not todo:
            log("worker: everything is done")
            return 0
        if RS.free_gb(ST.LOCAL) < RS.MIN_FREE_GB:
            log(f"STOP: free space {RS.free_gb(ST.LOCAL):.1f} GB < {RS.MIN_FREE_GB} GB")
            return 3
        ready = [n for n, deps in todo if all(done(d) for d in deps) and not (out_dir() / f"{n}.failed.txt").exists()]
        got = next((n for n in ready if claim(n)), None)
        if got is None:
            if not ready and not any((out_dir() / f"{n}.claim").exists() for n, _ in todo):
                log(f"worker: nothing can run ({len(todo)} units blocked by failures)")
                return 1
            time.sleep(20)
            continue
        t0 = time.time()
        try:
            RS.wait_for_input(f"{RS.MOUNT}/smartpixels_scratch/canon_v5/spstudy_AAAA_f04_100ev.root")
            info = run_one(S, got)
            mark(got, {"seconds": time.time() - t0, "result": info})
            log(f"done {got}: {time.time() - t0:.0f} s, free {RS.free_gb(ST.LOCAL):.1f} GB")
        except Exception:                                       # noqa: BLE001
            tb = traceback.format_exc()
            log(f"unit {got} FAILED:\n{tb}")
            if os.path.ismount(RS.MOUNT):
                (out_dir() / f"{got}.failed.txt").write_text(tb)
            else:
                time.sleep(60)
        finally:
            (out_dir() / f"{got}.claim").unlink(missing_ok=True)


def supervise(a):
    import study as ST
    logs = ST.LOCAL / "logs"
    procs = {}

    def start(i):
        lf = open(logs / f"postscan_worker{i}.log", "a")
        return subprocess.Popen([sys.executable, str(HERE / "postscan.py"), "worker"], stdout=lf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
    for i in range(a.workers):
        procs[i] = start(i)
        time.sleep(15)
    (logs / "postscan.pids").write_text(" ".join(str(p.pid) for p in procs.values()) + f"\nsupervisor {os.getpid()}\n")
    restarts, final = 0, {}
    while procs:
        time.sleep(20)
        for i, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            if rc in (0, 1, 3) or restarts >= 50:
                final[i] = rc
                del procs[i]
            else:
                restarts += 1
                log(f"supervisor: worker {i} exited {rc}; restart {restarts}")
                time.sleep(30)
                procs[i] = start(i)
    log(f"supervisor: workers exited {final}")
    return max(final.values(), default=0)


def status(a):
    u = units()
    d = [n for n, _ in u if done(n)]
    failed = sorted(p.name for p in out_dir().glob("*.failed.txt"))
    print(json.dumps({"done": len(d), "total": len(u), "not_done": [n for n, _ in u if not done(n)][:40], "failed": failed}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("supervise", "worker", "status", "one"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--unit", default="")
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--dev-dir", default=None)
    a = ap.parse_args(argv)
    global NEV, DEV_DIR
    NEV, DEV_DIR = a.nev, a.dev_dir
    if a.job == "one":                                          # development: one unit in this process, no marker
        print(json.dumps(run_one(RS.Study(), a.unit), indent=1, default=float))
        return 0
    return {"supervise": supervise, "worker": worker, "status": status}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
