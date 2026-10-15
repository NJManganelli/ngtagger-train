"""Driver of the C1b jobs (the replay's disc calibration; /Volumes/WDMac/smartpixels_scratch/wpc/c1b/PLAN.md rev 4).
Every job runs through ops/spheavy --frozen --on-busy wait, in the background.

  overlap  §1 item 2, files 1-2, AAAA only (one copy of each event): the same-surface overlap pairs on IL1
           and ID1-ID4; T0 (the toy closure in the real disc pair geometry, 1000 replicas, the planted R within
           3 % where the replica mean is known to 1 %, with the no-rotation known-bad), D0 (the IL1 closure against the same clusters' truth-residual
           R: the pooled ratio's one-sided 95 % bounds within [0.90, 1.10], per axis), D2 (per disc, the
           overlap R against v3's R at the pairs' own cells, within 1 +- 0.1 widened by t1(N) x the bootstrap
           sd; the same- vs opposite-facing agreement; B' < 0.02 cm GeV) and the facing-sign known-bad's
           size. Writes c1b/overlap/overlap_summary.json and the pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import arms as AR
import common as CM
import disc_r as DR
import gates as G
from common import OP

C1B_OUT = CM.OUT / "c1b"
IL1, DISCS = 1, (11, 12, 13, 14)
N_TOY, N_TOY_KB = 1000, 40                    # rev 4 amendment A3: enough replicas that the mean's error is <= 1 %
TOY_TOL, TOY_SE_MAX = 0.03, 0.01
D0_BAND = (0.90, 1.10)
D2_TOL = 0.10
NEV_FILE = 100


def _concat(parts):
    parts = [p for p in parts if p]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def judged(n_clusters):
    return n_clusters >= 400                      # >= 200 pairs: two clusters per pair


def overlap(a):
    t0 = time.time()
    geo = OP.load_geometry(CM.GEOMETRY)
    tools = AR.Tools(None, calibration=AR.CALIBRATION, angle_table=None)
    tables = tools.cal["R_tables"]
    parts = []
    for fi, f in enumerate(CM.FIT_FILES):
        path = CM.input_path("AAAA", f)
        for start, A in CM.iterate(path, AR.branches(path), chunk=a.chunk, nev=a.nev):
            ctx = AR.prepare(A, geo)
            parts.append(DR.pairs_of_chunk(ctx, geo, (IL1,) + DISCS, start=fi * NEV_FILE + start))
        print(f"[{time.time() - t0:7.1f}s] AAAA f{f:02d}: pairs collected", flush=True)
    P = _concat(parts)
    nev = NEV_FILE * len(CM.FIT_FILES)
    reps = G.replicas(nev=nev)
    res = {"plan": "c1b/PLAN.md rev 4 §1 item 2, §3 T0 D0 D2", "files": [str(CM.input_path("AAAA", f)) for f in CM.FIT_FILES],
           "n_pairs": {str(s): int((P["surface"] == s).sum()) for s in (IL1,) + DISCS}, "material": [], "minor": []}
    rng = np.random.default_rng(20261015)
    planted = DR.planted_widths(tables)

    # T0: the toy closure on the real disc pairs (and its no-rotation known-bad). A bin is judged when it holds
    # >= 200 pairs and the replica mean's standard error is <= 1 % (amendment A3); others are reported.
    res["T0"] = {}
    for s in DISCS:
        sel = P["surface"] == s
        out = {}
        jb = None
        for lab, rot, nrep in (("model", True, N_TOY), ("known_bad_no_rotation", False, N_TOY_KB)):
            fits = [DR.toy(P, tables, sel, planted, rng, rotation=rot) for _ in range(nrep)]
            rel = {ax: np.array([f[f"R_{ax}"] for f in fits]) / planted[ax] - 1 for ax in ("u", "v")}
            dev = {ax: np.nanmean(rel[ax], 0) for ax in ("u", "v")}
            se = {ax: np.nanstd(rel[ax], 0) / np.sqrt(np.sum(np.isfinite(rel[ax]), 0).clip(1)) for ax in ("u", "v")}
            nn = {ax: fits[0][f"n_{ax}"] for ax in ("u", "v")}
            if jb is None:                            # the known-bad is judged on the model's judged bins
                jb = {ax: [b for b in range(len(dev[ax])) if judged(nn[ax][b]) and se[ax][b] <= TOY_SE_MAX]
                      for ax in ("u", "v")}
            bad = [f"{ax}{b}" for ax in ("u", "v") for b in jb[ax] if not abs(dev[ax][b]) <= TOY_TOL]
            out[lab] = {"deviation": dev, "se": se, "n_clusters": nn, "judged": jb, "failing_bins": bad,
                        "replicas": nrep, "B_prime": np.nanmean([f["B_prime"] for f in fits], 0)}
        res["T0"][str(s)] = out
        if out["model"]["failing_bins"]:
            res["material"].append(f"T0 surface {s}: planted R not recovered within 3 % in {out['model']['failing_bins']}")
        if not out["known_bad_no_rotation"]["failing_bins"]:
            res["material"].append(f"T0 surface {s}: the no-rotation known-bad passes (the toy cannot see the mixing)"
                                   + ("" if any(jb.values()) else "; no bin judged"))
        print(f"[{time.time() - t0:7.1f}s] T0 surface {s}", flush=True)

    # fits with the event bootstrap: IL1 and the discs; facing splits on the discs
    fits = {s: DR.bootstrap(P, tables, s, reps) for s in (IL1,) + DISCS}

    # D0: the IL1 closure, like-for-like (the same clusters' truth-residual R), pooled ratio per axis
    sel = P["surface"] == IL1
    ev = P["ev"].astype(np.int64)
    res["D0"] = {}
    for ax in ("u", "v"):
        ref, n = DR.truth_width(P, tables, sel, ax)
        ratios = []
        for r, w in zip(fits[IL1], [None] + list(reps)):
            if w is None:
                rr = ref
            else:
                k = np.repeat(np.flatnonzero(sel), w[ev[sel]])
                Q = {kk: v[k] for kk, v in P.items()}
                rr, _ = DR.truth_width(Q, tables, np.ones(len(k), bool), ax)
            x = r[f"R_{ax}"] / rr
            m = np.isfinite(x) & (n > 0)
            ratios.append(float(np.sum(n[m] * x[m]) / np.sum(n[m])) if m.any() else np.nan)
        lo, hi = np.nanpercentile(ratios[1:], [5, 95])
        v3c, _ = DR.v3_at_cells(P, tables, sel, ax)
        res["D0"][ax] = {"pooled_ratio": ratios[0], "bounds95": [lo, hi], "pass": D0_BAND[0] <= lo and hi <= D0_BAND[1],
                         "per_bin_ratio": fits[IL1][0][f"R_{ax}"] / ref, "n_clusters": n,
                         "edge_vs_module": ref / v3c}
        if not res["D0"][ax]["pass"]:
            res["material"].append(f"D0 {ax}: IL1 pooled ratio {ratios[0]:.3f} [{lo:.3f}, {hi:.3f}] outside [0.90, 1.10]")
    print(f"[{time.time() - t0:7.1f}s] D0", flush=True)

    # D2 per disc; the facing agreement; B'
    res["D2"] = {}
    disc_fail = False
    for s in DISCS:
        sel = P["surface"] == s
        e = {}
        for ax in ("u", "v"):
            v3c, n = DR.v3_at_cells(P, tables, sel, ax)
            R = np.array([f[f"R_{ax}"] for f in fits[s]])
            ratio = R / v3c
            sd = np.nanstd(ratio[1:], 0)
            nj = sum(judged(x) for x in n)
            W = G.t_one(max(nj, 1))
            fail = [b for b in range(len(n)) if judged(n[b]) and np.isfinite(ratio[0][b])
                    and abs(ratio[0][b] - 1) > D2_TOL + W * sd[b]]
            e[ax] = {"R": R[0], "v3_at_cells": v3c, "ratio": ratio[0], "sd": sd, "n_clusters": n, "W": W, "failing": fail}
            disc_fail |= bool(fail)
        bp = np.array([f["B_prime"] for f in fits[s]])
        e["B_prime"], e["B_prime_sd"] = bp[0], np.nanstd(bp[1:], 0)
        e["B_prime_ok"] = bool(np.all(np.nan_to_num(bp[0]) < DR.B_PRIME_MAX))
        if not e["B_prime_ok"]:
            res["material"].append(f"D2 surface {s}: B' {np.round(bp[0], 4).tolist()} >= {DR.B_PRIME_MAX} cm GeV")
        # same- vs opposite-facing, paired bootstrap
        same = P["facing1"] == P["facing2"]
        fs = {lab: DR.bootstrap({k: v[m_] for k, v in P.items()}, tables, s, reps)
              for lab, m_ in (("same", same), ("opposite", ~same))}
        fac = {}
        for ax in ("u", "v"):
            d = np.array([a_[f"R_{ax}"] - b_[f"R_{ax}"] for a_, b_ in zip(fs["same"], fs["opposite"])])
            sd = np.nanstd(d[1:], 0)
            z = d[0] / np.where(sd > 0, sd, np.nan)
            nmin = np.minimum(fs["same"][0][f"n_{ax}"], fs["opposite"][0][f"n_{ax}"])
            bad = [b for b in range(len(z)) if judged(nmin[b]) and np.isfinite(z[b]) and abs(z[b]) > 2]
            fac[ax] = {"same": fs["same"][0][f"R_{ax}"], "opposite": fs["opposite"][0][f"R_{ax}"], "z": z, "disagree": bad}
            if bad:
                res["material"].append(f"D2 surface {s} {ax}: same- vs opposite-facing R disagree in bins {bad}")
        e["facing"] = fac
        kb = DR.fit_surface(P, tables, sel, drop_facing=-1)
        e["known_bad_facing_sign_dropped"] = {"B_prime": kb["B_prime"], "R_u": kb["R_u"], "R_v": kb["R_v"]}
        res["D2"][str(s)] = e
        print(f"[{time.time() - t0:7.1f}s] D2 surface {s}", flush=True)
    res["D2_decision"] = "build disc R tables from the overlap R" if disc_fail else "keep v3's R"
    res["seconds"] = round(time.time() - t0, 1)
    out_dir = os.path.join(a.out, "overlap")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "overlap_pairs.npz"), **P)
    with open(os.path.join(out_dir, "overlap_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({k: res[k] for k in ("n_pairs", "material", "D2_decision")}, indent=1, default=str))
    return 0


PT_BINS = (2.0, 3.0, 5.0, 10.0, 20.0, np.inf)          # v3's seed pT bins
D3_TOL = 0.10


def first_met_disc(a, planes, path):
    """Per track: the first disc the OT helix meets along the outside-in path, i.e. the outermost (largest
    |z|) TFPX disc (1-6) it crosses within its radial coverage; else (crossed only within the drPath margin)
    the outermost disc in the arm's path; 0 for none."""
    import disc_material as DM
    dp = [p for p in planes if p["kind"] == "disc"]
    _, _, on = DM.crossings(a, dp)
    disk = np.array([p["disk"] for p in dp])
    z = np.array([p["z"] for p in dp])
    out = np.where(on.any(1), disk[np.argmax(np.where(on, z[None, :], -1.0), 1)], 0)
    cols = [(j, S) for j, S in enumerate(path["surfaces"]) if OP.SPN.is_disc_surface(S)]
    for j, S in sorted(cols, key=lambda x: -x[1]):             # outermost first
        out = np.where((out == 0) & path["in_path"][:, j], S - 10, out)
    return out


def core_width(x):
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return float(np.median(np.abs(x - np.median(x))) / 0.6745)


def width_table(rows, keys, reps, min_n=200):
    """Per bin (the tuple of keys): n, the core widths of pu and pv and their event-bootstrap sd."""
    out = {}
    combo = np.stack([rows[k] for k in keys], 1)
    uniq = np.unique(combo, axis=0)
    ev = rows["ev"].astype(np.int64)
    for u in uniq:
        m = np.all(combo == u, axis=1)
        n = int(m.sum())
        e = {"n": n}
        for ax in ("pu", "pv"):
            x = rows[ax][m]
            e[ax] = core_width(x)
            if n >= min_n:
                idx = np.flatnonzero(m)
                bs = [core_width(rows[ax][np.repeat(idx, w[ev[idx]])]) for w in reps]
                e[ax + "_sd"] = float(np.nanstd(bs))
        out["|".join(str(int(x)) if np.isfinite(x) else str(x) for x in u)] = e
    return out


SCALE_GRID = (0.5, 0.71, 1.0, 1.41, 2.0)              # A5
CLIP = 25.0


def collect(a, disc_scale, cuts, disc_R):
    """One pass of the disc arm over files 1-2 x the scenarios at a disc material scale: own primaries with a
    clean history (pulls and innovation variances), with the population, group, step, surface and pT bin."""
    import diagnostics as DG
    geo = OP.load_geometry(CM.GEOMETRY)
    disc = AR.DiscTools(disc_scale=disc_scale, R_tables=disc_R)
    parts = []
    for scen in a.scen:
        for fi, f in enumerate(CM.FIT_FILES):
            path_f = CM.input_path(scen, f)
            cfg = CM.producer_config(path_f)
            assert CM.mask_of(cfg) == scen
            tools = AR.Tools(cfg, cuts=cuts, disc=disc)
            arm = AR.inn_disc(cfg)
            for start, A in CM.iterate(path_f, AR.branches(path_f), chunk=a.chunk, nev=a.nev):
                ctx = AR.prepare(A, geo)
                path, _ = AR.visit_plan(ctx, geo, cfg, scen, arm)
                dg = AR.disc_going(path)
                fmd = first_met_disc(ctx["a"], disc.planes, path)
                out = AR.run_arm(ctx, geo, cfg, scen, arm, tools, own=True)
                R = DG.mark_primary(out["own"])
                m = R["primary"] & R["clean"]
                o = R["o"][m]
                pt = ctx["T"]["pt"][o]
                parts.append({"scen": np.full(m.sum(), CM.SCENARIOS.index(scen)), "ev": R["ev"][m] + fi * NEV_FILE + start,
                              "surface": R["surface"][m], "step": R["step"][m], "first": R["first"][m],
                              "pu": R["pu"][m], "pv": R["pv"][m], "vu": R["vu"][m], "vv": R["vv"][m],
                              "dg": dg[o], "group": np.minimum(fmd[o], 5), "stubs": ctx["T"]["nStubs"][o],
                              "ptbin": np.clip(np.searchsorted(PT_BINS, pt, side="right") - 1, 0, len(PT_BINS) - 2)})
    return _concat(parts)


def judge_relative(table, ref, refkey, label, min_n=200):
    """A4: |width - width(reference)| <= 0.1 + t1(N) sqrt(sd^2 + sd_ref^2), per bin and axis; refkey maps a bin
    key to its reference key. Returns (failing lines, W, N)."""
    jb = []
    for k, e in table.items():
        r = ref.get(refkey(k))
        for ax in ("pu", "pv"):
            if e["n"] >= min_n and r and r["n"] >= min_n and np.isfinite(e[ax]) and np.isfinite(r[ax]):
                jb.append((k, ax, r))
    W = G.t_one(max(len(jb), 1))
    fails = [f"{label} {k} {ax}: width {table[k][ax]:.3f} vs reference {r[ax]:.3f} (n {table[k]['n']} / {r['n']})"
             for k, ax, r in jb
             if abs(table[k][ax] - r[ax]) > D3_TOL + W * np.hypot(table[k][ax + "_sd"], r[ax + "_sd"])]
    return fails, W, len(jb)


def judge_all(rows, reps):
    """D3a, D3b, D3c relative to the barrel-only references (A4), the stub-count test, the absolute widths."""
    sub = lambda m: {k: v[m] for k, v in rows.items()}
    bo = ~rows["dg"] & (rows["surface"] <= 4)
    out = {"ref_same": width_table(sub(bo), ("surface", "first", "ptbin"), reps),
           "ref_first": width_table(sub(bo & (rows["step"] == 0)), ("ptbin",), reps),
           "ref_later": width_table(sub(bo & (rows["step"] > 0)), ("ptbin",), reps)}
    first = rows["dg"] & (rows["step"] == 0)
    out["D3a"] = width_table(sub(first), ("group", "ptbin"), reps)
    pt_of = lambda k: k.split("|")[-1]
    out["D3a_fail"], out["D3a_W"], out["D3a_N"] = judge_relative(out["D3a"], out["ref_first"], pt_of, "D3a")
    out["D3a_masked_only"] = width_table(sub(first & (rows["surface"] <= 4)), ("group",), reps)
    out["D3a_stubs"] = width_table(sub(first), ("group", "ptbin", "stubs"), reps)
    sf, _, _ = judge_relative(out["D3a_stubs"], out["ref_first"], lambda k: k.split("|")[1], "D3a stub sub-bin")
    bad_groups = {f.split()[1] for f in out["D3a_fail"]}
    out["stub_split"] = sorted({f.split()[3].rsplit("|", 1)[0] for f in sf} - bad_groups)
    later_disc = rows["dg"] & (rows["surface"] > 10) & (rows["step"] > 0)
    out["D3b"] = width_table(sub(later_disc), ("surface", "ptbin"), reps)
    out["D3b_fail"], out["D3b_W"], out["D3b_N"] = judge_relative(out["D3b"], out["ref_later"], pt_of, "D3b")
    bar = rows["dg"] & (rows["surface"] <= 4)
    out["D3c"] = width_table(sub(bar), ("surface", "first", "ptbin"), reps)
    out["D3c_fail"], out["D3c_W"], out["D3c_N"] = judge_relative(out["D3c"], out["ref_same"], lambda k: k, "D3c")
    return out


def seedtest(a):
    """§1 item 3 with A4: the D3a test-or-fit decision on files 1-2 x 3 scenarios, the disc arm with the seed as
    is (each disc-going track's own v3 deficit) and the material scale fixed at 1; D3b / D3c are recorded, not
    judged (they are judged after the scale fit). Writes c1b/seed/seedtest_summary.json and the rows."""
    import cuts as CU
    t0 = time.time()
    cuts = CU.Cuts.load(a.cuts)
    rows = collect(a, 1.0, cuts, load_disc_R(a.disc_r) if a.disc_r else None)
    reps = G.replicas(nev=NEV_FILE * len(CM.FIT_FILES))
    J = judge_all(rows, reps)
    res = {"plan": "c1b/PLAN.md rev 4 §1 item 3, §8 A4", "cuts": a.cuts, "disc_r": a.disc_r, "disc_scale": 1.0,
           "material": [], "minor": [], **J,
           "decision": "refit the disc seed groups" if J["D3a_fail"] else "keep the seed (each track's v3 deficit)",
           "note": "D3b and D3c recorded at scale 1; judged after the material-scale fit (A4)",
           "seconds": round(time.time() - t0, 1)}
    out_dir = os.path.join(a.out, "seed")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "seedtest_rows.npz"), **rows)
    with open(os.path.join(out_dir, "seedtest_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({k: res[k] for k in ("D3a_fail", "decision")}, indent=1, default=str))
    return 0


def nll_per_event(rows, sel, nev):
    """Per event: sum over u and v of log V + min(r^2 / V, CLIP), the rows sel (A5)."""
    out = np.zeros(nev)
    for p_, v_ in (("pu", "vu"), ("pv", "vv")):
        x = np.log(rows[v_][sel]) + np.minimum(rows[p_][sel] ** 2, CLIP)
        ok = np.isfinite(x)
        out += np.bincount(rows["ev"][sel][ok].astype(np.int64), weights=x[ok], minlength=nev)
    return out


def quad_min(logs, y):
    """The minimum of a parabola through the grid's lowest point and its neighbours (log scale); the edge
    when the lowest point is at the grid's edge."""
    i = int(np.argmin(y))
    if i in (0, len(y) - 1):
        return float(np.exp(logs[i])), True
    x3, y3 = logs[i - 1:i + 2], y[i - 1:i + 2]
    c = np.polyfit(x3, y3, 2)
    return float(np.exp(-c[1] / (2 * c[0]))) if c[0] > 0 else float(np.exp(logs[i])), False


def scalefit(a):
    """§1 items 4-5 with A4 / A5: the disc material scale from the scan, its second-on vs third-on check, then
    one pass at the fitted scale: D3b, D3c (the stop rule) and the D3a re-check, relative to the barrel-only
    references; the absolute widths reported. Writes c1b/scale/scalefit_summary.json."""
    import cuts as CU
    t0 = time.time()
    cuts = CU.Cuts.load(a.cuts)
    disc_R = load_disc_R(a.disc_r) if a.disc_r else None
    nev = NEV_FILE * len(CM.FIT_FILES)
    reps = G.replicas(nev=nev)
    logs = np.log(np.array(SCALE_GRID))
    nll = {"second_on": [], "third_on": []}
    core_ratio = []                     # diagnostic (reported only): the later disc visits' core width over the
    for s in SCALE_GRID:                # barrel-only later visits', u and v averaged, per grid scale
        rows = collect(a, s, cuts, disc_R)
        later = rows["dg"] & (rows["surface"] > 10)
        nll["second_on"].append(nll_per_event(rows, later & (rows["step"] >= 1), nev))
        nll["third_on"].append(nll_per_event(rows, later & (rows["step"] >= 2), nev))
        ref = ~rows["dg"] & (rows["surface"] <= 4) & (rows["step"] > 0)
        dl = later & (rows["step"] >= 1)
        core_ratio.append(np.mean([core_width(rows[x][dl]) / core_width(rows[x][ref]) for x in ("pu", "pv")]))
        print(f"[{time.time() - t0:7.1f}s] scale {s}", flush=True)
    res = {"plan": "c1b/PLAN.md rev 4 §1 items 4-5, §8 A4 A5", "grid": SCALE_GRID, "material": [], "minor": []}
    fit = {}
    for k in nll:
        M = np.array(nll[k])                                     # (grid, nev)
        s_hat, edge = quad_min(logs, M.sum(1))
        bs = [quad_min(logs, M @ w)[0] for w in reps]
        fit[k] = {"scale": s_hat, "at_bound": edge, "nll": M.sum(1), "boot": np.array(bs)}
    d = fit["second_on"]["boot"] - fit["third_on"]["boot"]
    sd_d = float(np.std(d))
    res["scale"] = fit["second_on"]["scale"]
    res["scale_sd"] = float(np.std(fit["second_on"]["boot"]))
    res["third_on"] = fit["third_on"]["scale"]
    res["paired_sd"] = sd_d
    res["nll"] = {k: v["nll"] for k, v in fit.items()}
    cr = np.array(core_ratio)
    res["diagnostic_core_ratio"] = cr
    ok = np.flatnonzero(np.diff(np.sign(cr - 1.0)) != 0)
    res["diagnostic_core_matching_scale"] = (float(np.exp(np.interp(1.0, cr[ok[0]:ok[0] + 2][::-1] if cr[ok[0]] > cr[ok[0] + 1] else cr[ok[0]:ok[0] + 2],
                                                               logs[ok[0]:ok[0] + 2][::-1] if cr[ok[0]] > cr[ok[0] + 1] else logs[ok[0]:ok[0] + 2])))
                                             if len(ok) else None)
    if fit["second_on"]["at_bound"]:
        res["material"].append(f"the disc material scale is at a bound of the grid ({res['scale']:.2f})")
    if abs(res["scale"] - res["third_on"]) > 2 * sd_d:
        res["material"].append(f"second-on scale {res['scale']:.3f} vs third-on {res['third_on']:.3f}: beyond 2 paired sd ({sd_d:.3f})")
    if 1.25 <= res["scale"] <= 1.6:
        res["note_1p4"] = "a scale near 1.4 is most likely the barrel model's unexplained 1.37x excess, not disc material"
    rows = collect(a, res["scale"], cuts, disc_R)
    J = judge_all(rows, reps)
    res.update(J)
    res["material"] += [f"{x} (after the scale fit: a D3c failure stops the package)" for x in J["D3c_fail"]]
    res["material"] += J["D3b_fail"]
    res["D3a_recheck"] = "fails: the seed groups are refitted" if J["D3a_fail"] else "passes"
    res["seconds"] = round(time.time() - t0, 1)
    out_dir = os.path.join(a.out, "scale")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "scalefit_rows.npz"), **rows)
    with open(os.path.join(out_dir, "scalefit_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({k: res[k] for k in ("scale", "scale_sd", "third_on", "material", "D3a_recheck")}, indent=1, default=str))
    return 0


C1B_CUTS = "wpc_cuts_c1b_v1.json"


def cut_pass(a, scale, cuts, disc_R):
    """One pass of the disc arm (own and leave-one-out rows) over files 1-2 x the scenarios, with the barrel
    arm beside it for D5 (barrel-only tracks bit-identical, discs on vs off, under the same cuts). The rows are
    returned with surface = the cut-bin code (disc-going barrel visits + 100)."""
    import diagnostics as DG
    geo = OP.load_geometry(CM.GEOMETRY)
    disc = AR.DiscTools(disc_scale=scale, R_tables=disc_R)
    own, loo, d5 = [], [], {"tracks": 0, "different": 0}
    for scen in a.scen:
        for fi, f in enumerate(CM.FIT_FILES):
            path_f = CM.input_path(scen, f)
            cfg = CM.producer_config(path_f)
            assert CM.mask_of(cfg) == scen
            tools = AR.Tools(cfg, cuts=cuts, disc=disc)
            arm = AR.inn_disc(cfg)
            for start, A in CM.iterate(path_f, AR.branches(path_f), chunk=a.chunk, nev=a.nev):
                ctx = AR.prepare(A, geo)
                path, _ = AR.visit_plan(ctx, geo, cfg, scen, arm)
                dg = AR.disc_going(path)
                out = AR.run_arm(ctx, geo, cfg, scen, arm, tools, own=True, loo=True)
                off = AR.run_arm(ctx, geo, cfg, scen, AR.inn(cfg), tools)
                bo = np.flatnonzero(~dg & out["refit"])
                k = off["rec"]["acc"].shape[1]                  # barrel-only tracks visit the same barrel steps
                sel = lambda r: np.where(r["rec"]["acc"][bo, :k], r["rec"]["ci"][bo, :k], -1)
                diff = ~(np.all(out["a"][bo] == off["a"][bo], 1) & np.all(out["C"][bo] == off["C"][bo], (1, 2))
                         & np.all(sel(out) == sel(off), 1) & ~out["rec"]["acc"][bo, k:].any(1))
                d5["tracks"] += len(bo)
                d5["different"] += int(diff.sum())
                R = DG.mark_primary(out["own"])
                R = {k: (R["csurface"] if k == "surface" else v) for k, v in R.items()}
                R["ev"] = R["ev"] + fi * NEV_FILE + start
                own.append({k: R[k] for k in RUN_OWN_KEEP})
                L = out["loo"]
                import run_entry as RE
                prim, clean_track = RE.loo_flags(out, DG.mark_primary(out["own"]), ctx)
                loo.append({"surface": L["csurface"], "first": L["first"], "own_primary": prim, "clean_track": clean_track,
                            "scen": np.full(len(prim), scen), "n_inner": L["n_inner"], "n_acc": L["n_acc"],
                            "ev": ctx["tev"][L["o"]] + fi * NEV_FILE + start, **{f"d_{k}": v for k, v in L["dims"].items()}})
    return _concat(own), _concat(loo), d5


RUN_OWN_KEEP = ("o", "ev", "step", "first", "surface", "ci", "mi", "clean", "e1", "t1", "z", "has", "g_xy", "g_xya",
                "g_xyb", "g_xyab", "classT", "table_pop", "keybin", "dkeybin", "facing", "primary", "pu", "pv")


def c1b_cutfit(a):
    """§1 item 6: the cut fit with the disc bins, the barrel-visit bins and the barrel-only bins, files 1-2 x the
    scenarios, the disc arm with the fitted scale (J3) and the decided seed; the disc window starts from the
    provisional cut and the fit is repeated once if the fitted cuts move the populations (PD18's rule); D5 on the
    way. Writes c1b/cuts/wpc_cuts_c1b_v1.json (+ unconditioned), the rows and cutfit_summary.json."""
    import cuts as CU
    t0 = time.time()
    with open(os.path.join(a.out, "scale", "scalefit_summary.json")) as fh:
        scale = float(json.load(fh)["scale"])
    disc_R = load_disc_R(a.disc_r) if a.disc_r else None
    out_dir = os.path.join(a.out, "cuts")
    os.makedirs(out_dir, exist_ok=True)
    window = CU.Cuts.load(a.cuts)
    res = {"plan": "c1b/PLAN.md rev 4 §1 item 6, §3 D5", "disc_scale": scale, "window_cuts": a.cuts, "material": [],
           "minor": [], "passes": []}
    fits = []
    for it in range(2):
        own, loo, d5 = cut_pass(a, scale, window, disc_R)
        res["passes"].append({"window": window.doc.get("created", a.cuts), "D5": d5})
        if d5["different"]:
            res["material"].append(f"D5 (files 1-2, pass {it + 1}): {d5['different']} of {d5['tracks']} barrel-only tracks "
                                   "differ with the discs on")
        loo_f = dict(loo)
        loo_f["dims"] = {dm: loo_f.pop(f"d_{dm}") for dm in AR.DIMS}
        prov = {"files": [str(CM.input_path(s, f)) for s in a.scen for f in CM.FIT_FILES], "disc_scale": scale,
                "disc_R": a.disc_r, "window": a.cuts if it == 0 else "the pass-1 fit", "plan": res["plan"]}
        fit = CU.fit({"calibrated": own}, {"calibrated": loo_f}, prov, "smoothed", clean=True)
        fits.append((fit, own, loo_f))
        print(f"[{time.time() - t0:7.1f}s] cut pass {it + 1}", flush=True)
        if it == 0:
            iv = CU.intervals({"calibrated": own}, {"calibrated": loo_f})
            window = fit
            continue
        # PD18's rule: the pass-2 values against pass 1's 95 % intervals
        from scipy import stats
        n_b, out_ = 0, []
        for tk, tab in iv.items():
            for b, e in tab.items():
                ne = fit.doc["cuts"].get(tk, {}).get(b)
                if e["value"] is None or e["n"] < CU.MIN_N or ne is None or ne["value"] is None or ne["n"] < CU.MIN_N:
                    continue
                n_b += 1
                if not e["value"][0] <= ne["value"] <= e["value"][1]:
                    out_.append(f"{tk} {b}")
        bound = float(stats.binom.ppf(0.99, n_b, 0.05))
        res["window_refit"] = {"bins": n_b, "outside": len(out_), "bound": bound, "changed": len(out_) > bound}
    final = fits[1][0] if res["window_refit"]["changed"] else fits[0][0]
    res["adopted"] = "pass 2" if res["window_refit"]["changed"] else "pass 1"
    final.save(os.path.join(out_dir, C1B_CUTS))
    own, loo_f = (fits[1] if res["window_refit"]["changed"] else fits[0])[1:]
    CU.fit({"calibrated": own}, {"calibrated": loo_f}, final.doc["provenance"], "smoothed", clean=False).save(
        os.path.join(out_dir, C1B_CUTS.replace(".json", "_unconditioned.json")))
    np.savez_compressed(os.path.join(out_dir, "c1b_cut_rows.npz"), **{f"own__{k}": v for k, v in own.items()},
                        **{f"loo__{k}": v for k, v in loo_f.items() if k != "dims"},
                        **{f"loo__d_{k}": v for k, v in loo_f["dims"].items()})
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "cutfit_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({k: res[k] for k in ("material", "adopted")}, indent=1, default=str))
    return 0


def disc_r_tables(a):
    """Light, after J1: when D2 decided to build disc tables, write them from J1's overlap R in v3's layout
    (size x |cot| widths; the overlap R per size bin over every |cot| cell; a bin J1 left unconstrained keeps
    v3's width), per disc surface: c1b/overlap/disc_R_tables.json. Nothing to do when D2 kept v3's R."""
    src = os.path.join(a.out, "overlap", "overlap_summary.json")
    with open(src) as fh:
        S = json.load(fh)
    if S["D2_decision"] != "build disc R tables from the overlap R":
        print("D2 kept v3's R: no disc tables")
        return 0
    tables = AR.Tools(None, calibration=AR.CALIBRATION, angle_table=None).cal["R_tables"]
    out = {}
    for s, e in S["D2"].items():
        t = json.loads(json.dumps(tables))
        for ax in ("u", "v"):
            W = np.asarray(t[ax]["width_cm"], np.float64)
            R = np.asarray(e[ax]["R"], np.float64)
            for b in range(len(R)):
                if np.isfinite(R[b]) and e[ax]["n_clusters"][b] >= 400:
                    W[b, :] = R[b]
            t[ax]["width_cm"] = W.tolist()
        t["built_from"] = f"C1b overlap R, surface {s} (J1: {src})"
        out[s] = t
    with open(os.path.join(a.out, "overlap", "disc_R_tables.json"), "w") as fh:
        json.dump({"source": src, "tables": out}, fh, indent=1)
    return 0


def load_disc_R(path):
    """C1b's disc position-error tables (J1's D2 decision), per disc surface, in v3's table layout."""
    with open(path) as fh:
        return {int(k): v for k, v in json.load(fh)["tables"].items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("overlap", "discR", "seedtest", "scalefit", "cutfit"))
    ap.add_argument("--scen", nargs="+", default=list(CM.SCENARIOS))
    ap.add_argument("--cuts", default=str(CM.CUTS), help="the cut file in use (the window)")
    ap.add_argument("--disc-r", default=None, help="J1's disc position-error tables, if D2 built them")
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--out", default=str(C1B_OUT))
    a = ap.parse_args(argv)
    return {"overlap": overlap, "discR": disc_r_tables, "seedtest": seedtest, "scalefit": scalefit,
            "cutfit": c1b_cutfit}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
