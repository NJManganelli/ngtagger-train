"""Driver of the C1b jobs (the replay's disc calibration; /Volumes/WDMac/smartpixels_scratch/wpc/c1b/PLAN.md rev 4).
Every job runs through ops/spheavy --frozen --on-busy wait, in the background.

  pairs     §1 item 2 with A8 / A10 (c1b_overlap): the overlap pairs on IL1 and ID1-ID4 and IL1's own clusters,
            AAAA only (one copy of each event), the fit files or --files
  il1truth  A8 / A11 (c1b_overlap): IL1's truth width against p with the beam pipe's Highland expectation, IL1's
            reference by predicted offset, the findings on v3's barrel tables and their width definition
  fit       §1 item 2 with A8 / A11 (c1b_overlap; report-only, A13): the split-core shape fits with the event
            bootstrap, T0, the shape systematic, D0 (IL1), D2 with the facing agreement and B'
  seedtest  §1 item 3: the D3a test-or-fit decision
  scalefit  §1 items 4-5: the disc material scale, D3b / D3c, the D3a re-check
  cutfit    §1 item 6: the cut fit with the disc, barrel-visit and barrel-only bins
  checks    J5 (file 4, --known-bad) / J6 (file 6): the final configuration judged
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import arms as AR
import c1b_overlap as OV
import common as CM
import gates as G
from common import OP

C1B_OUT = CM.OUT / "c1b"
NEV_FILE = OV.NEV_FILE


def _concat(parts):
    parts = [p for p in parts if p]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


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
    """Per bin (the tuple of keys): n; per pull axis the A17 width (the variance within a +-5 sigma clip, a unit
    Gaussian giving 1) with its event-bootstrap sd, and the MAD core width (reported)."""
    import seedmodel as SMOD
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
            e[ax], e[ax + "_core"] = SMOD.var_width(x), core_width(x)
            if n >= min_n:
                idx = np.flatnonzero(m)
                e[ax + "_sd"] = float(np.nanstd([SMOD.var_width(rows[ax][np.repeat(idx, w[ev[idx]])]) for w in reps]))
        out["|".join(str(int(x)) if np.isfinite(x) else str(x) for x in u)] = e
    return out


SCALE_GRID = (0.5, 0.71, 1.0, 1.41, 2.0)              # A5
SCALE_GRID_EXT = (2.83, 4.0)                          # A17's chain: only if the minimum sits at 2.0
CLIP = 25.0


def collect(a, disc_scale):
    """One pass of the disc arm over the fit files (A10) x the scenarios at a disc material scale: own primaries
    with a clean history (pulls and innovation variances), with the population, group, step, surface and pT bin."""
    import diagnostics as DG
    geo = OP.load_geometry(CM.GEOMETRY)
    disc = AR.DiscTools(disc_scale=disc_scale)
    parts = []
    for scen in a.scen:
        for fi, f in enumerate(CM.FIT_FILES):
            path_f = CM.input_path(scen, f)
            cfg = CM.producer_config(path_f)
            assert CM.mask_of(cfg) == scen
            evr = fi * NEV_FILE + CM.event_rank(path_f)            # an event's identity across the scenarios (A14)
            tools = AR.Tools(cfg, disc=disc)                     # no consumer reads a cut: the 4-sigma window (A9)
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
                parts.append({"scen": np.full(m.sum(), CM.SCENARIOS.index(scen)), "ev": evr[R["ev"][m] + start],
                              "surface": R["surface"][m], "step": R["step"][m], "first": R["first"][m],
                              "pu": R["pu"][m], "pv": R["pv"][m], "vu": R["vu"][m], "vv": R["vv"][m],
                              "dg": dg[o], "group": np.minimum(fmd[o], 5), "stubs": ctx["T"]["nStubs"][o],
                              "ptbin": np.clip(np.searchsorted(PT_BINS, pt, side="right") - 1, 0, len(PT_BINS) - 2)})
    return _concat(parts)


def judge_absolute(table, label, min_n=200):
    """A17: |w - 1| <= 0.1 + t1(N) sd, per bin and axis (bins with >= min_n rows). Returns (failing lines, W, N)."""
    jb = [(k, ax) for k, e in table.items() for ax in ("pu", "pv") if e["n"] >= min_n and np.isfinite(e[ax])]
    W = G.t_one(max(len(jb), 1))
    fails = [f"{label} {k} {ax}: width {table[k][ax]:.3f} (n {table[k]['n']})" for k, ax in jb
             if abs(table[k][ax] - 1.0) > D3_TOL + W * table[k][ax + "_sd"]]
    return fails, W, len(jb)


def judge_all(rows, reps):
    """A17: D3a (first visits of disc-going tracks, per first-met disc x pT bin, with the stub-count sub-bins), D3b
    (later disc visits, per disc x pT bin), D3c (barrel visits of disc-going tracks, per layer x first / later x pT bin)
    and D3d (barrel-only tracks, the same bins; reported only), each judged absolutely."""
    sub = lambda m: {k: v[m] for k, v in rows.items()}
    out = {}
    bo = ~rows["dg"] & (rows["surface"] <= 4)
    out["D3d"] = width_table(sub(bo), ("surface", "first", "ptbin"), reps)
    out["D3d_fail"], out["D3d_W"], out["D3d_N"] = judge_absolute(out["D3d"], "D3d")
    first = rows["dg"] & (rows["step"] == 0)
    out["D3a"] = width_table(sub(first), ("group", "ptbin"), reps)
    out["D3a_fail"], out["D3a_W"], out["D3a_N"] = judge_absolute(out["D3a"], "D3a")
    out["D3a_masked_only"] = width_table(sub(first & (rows["surface"] <= 4)), ("group",), reps)
    out["D3a_stubs"] = width_table(sub(first), ("group", "ptbin", "stubs"), reps)
    sf, _, _ = judge_absolute(out["D3a_stubs"], "D3a stub sub-bin")
    bad_groups = {f.split()[1] for f in out["D3a_fail"]}
    out["stub_split"] = sorted({f.split()[3].rsplit("|", 1)[0] for f in sf} - bad_groups)
    later_disc = rows["dg"] & (rows["surface"] > 10) & (rows["step"] > 0)
    out["D3b"] = width_table(sub(later_disc), ("surface", "ptbin"), reps)
    out["D3b_fail"], out["D3b_W"], out["D3b_N"] = judge_absolute(out["D3b"], "D3b")
    bar = rows["dg"] & (rows["surface"] <= 4)
    out["D3c"] = width_table(sub(bar), ("surface", "first", "ptbin"), reps)
    out["D3c_fail"], out["D3c_W"], out["D3c_N"] = judge_absolute(out["D3c"], "D3c")
    return out


def seedtest(a):
    """§1 item 3 with A16 / A17: the seed check of the smooth seed on the fit files (A10) x 3 scenarios, the disc arm
    at material scale 1; D3a judged absolutely (a failure stops the package with the discs off: one attempt); D3b / D3c
    recorded, judged after the scale fit; D3d reported. Writes c1b/seed/seedtest_summary.json and the rows."""
    t0 = time.time()
    rows = collect(a, 1.0)
    reps = G.replicas(nev=NEV_FILE * len(CM.FIT_FILES))
    J = judge_all(rows, reps)
    res = {"plan": "c1b/PLAN.md rev 4 §1 item 3, §8 A9, A10, A16, A17", "files": CM.FIT_FILES, "seed": "smooth (A16)",
           "disc_scale": 1.0, "material": [f"{x} (the smooth seed's D3a: the discs go off, A16-A17)" for x in J["D3a_fail"]],
           "minor": [], **J,
           "note": "D3b and D3c recorded at scale 1, judged after the material-scale fit; D3d reported only (A17)",
           "seconds": round(time.time() - t0, 1)}
    out_dir = os.path.join(a.out, "seed")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "seedtest_rows.npz"), **rows)
    with open(os.path.join(out_dir, "seedtest_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({k: res[k] for k in ("material", "D3d_fail")}, indent=1, default=str))
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
    """§1 items 4-5 with A5 / A16 / A17: the disc material scale from the scan (extended to SCALE_GRID_EXT only if the
    minimum sits at 2.0), its second-on vs third-on check, then one pass at the fitted scale: D3a (re-check), D3b and D3c
    judged absolutely, D3d reported. Writes c1b/scale/scalefit_summary.json."""
    t0 = time.time()
    nev = NEV_FILE * len(CM.FIT_FILES)
    reps = G.replicas(nev=nev)
    grid = list(SCALE_GRID)
    nll = {"second_on": [], "third_on": []}

    def scan(points):
        for s in points:
            rows = collect(a, s)
            later = rows["dg"] & (rows["surface"] > 10)
            nll["second_on"].append(nll_per_event(rows, later & (rows["step"] >= 1), nev))
            nll["third_on"].append(nll_per_event(rows, later & (rows["step"] >= 2), nev))
            print(f"[{time.time() - t0:7.1f}s] scale {s}", flush=True)

    def fit_all():
        logs = np.log(np.array(grid))
        out = {}
        for k in nll:
            M = np.array(nll[k])                                 # (grid, nev)
            s_hat, edge = quad_min(logs, M.sum(1))
            out[k] = {"scale": s_hat, "at_bound": edge, "nll": M.sum(1),
                      "boot": np.array([quad_min(logs, M @ w)[0] for w in reps])}
        return out

    scan(grid)
    fit = fit_all()
    if fit["second_on"]["at_bound"] and fit["second_on"]["scale"] >= grid[-1]:
        grid += list(SCALE_GRID_EXT)                             # A17's chain: extend only if the minimum sits at 2.0
        scan(SCALE_GRID_EXT)
        fit = fit_all()
    res = {"plan": "c1b/PLAN.md rev 4 §1 items 4-5, §8 A5, A9, A10, A16, A17", "files": CM.FIT_FILES, "grid": grid,
           "seed": "smooth (A16)", "material": [], "minor": []}
    d = fit["second_on"]["boot"] - fit["third_on"]["boot"]
    sd_d = float(np.std(d))
    res.update(scale=fit["second_on"]["scale"], scale_sd=float(np.std(fit["second_on"]["boot"])),
               third_on=fit["third_on"]["scale"], paired_sd=sd_d, nll={k: v["nll"] for k, v in fit.items()})
    if fit["second_on"]["at_bound"]:
        res["material"].append(f"the disc material scale is at a bound of the grid ({res['scale']:.2f})")
    if abs(res["scale"] - res["third_on"]) > 2 * sd_d:
        res["material"].append(f"second-on scale {res['scale']:.3f} vs third-on {res['third_on']:.3f}: beyond 2 paired sd ({sd_d:.3f})")
    if res["scale"] > 2.0:
        res["minor"].append(f"the disc material scale {res['scale']:.2f} is above 2: physically suspect (A17's chain)")
    rows = collect(a, res["scale"])
    J = judge_all(rows, reps)
    res.update(J)
    res["material"] += [f"{x} (after the scale fit: the discs go off, A16-A17)" for x in J["D3a_fail"] + J["D3b_fail"] + J["D3c_fail"]]
    res["seconds"] = round(time.time() - t0, 1)
    out_dir = os.path.join(a.out, "scale")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "scalefit_rows.npz"), **rows)
    with open(os.path.join(out_dir, "scalefit_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({k: res[k] for k in ("scale", "scale_sd", "third_on", "material", "minor")}, indent=1, default=str))
    return 0


C1B_CUTS = "wpc_cuts_c1b_v1.json"


def cut_pass(a, scale):
    """One pass of the disc arm (own and leave-one-out rows) over the fit files (A10) x the scenarios, with the
    barrel arm beside it for D5 (barrel-only tracks bit-identical, discs on vs off). The rows are returned with
    surface = the cut-bin code (disc-going barrel visits + 100)."""
    import diagnostics as DG
    geo = OP.load_geometry(CM.GEOMETRY)
    disc = AR.DiscTools(disc_scale=scale)
    own, loo, d5 = [], [], {"tracks": 0, "different": 0}
    for scen in a.scen:
        for fi, f in enumerate(CM.FIT_FILES):
            path_f = CM.input_path(scen, f)
            cfg = CM.producer_config(path_f)
            assert CM.mask_of(cfg) == scen
            evr = fi * NEV_FILE + CM.event_rank(path_f)
            tools = AR.Tools(cfg, disc=disc)                     # no consumer reads a cut: the 4-sigma window (A9)
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
                R["ev"] = evr[R["ev"] + start]
                own.append({k: R[k] for k in RUN_OWN_KEEP})
                L = out["loo"]
                import run_entry as RE
                prim, clean_track = RE.loo_flags(out, DG.mark_primary(out["own"]), ctx)
                loo.append({"surface": L["csurface"], "first": L["first"], "own_primary": prim, "clean_track": clean_track,
                            "scen": np.full(len(prim), scen), "n_inner": L["n_inner"], "n_acc": L["n_acc"],
                            "ev": evr[ctx["tev"][L["o"]] + start], **{f"d_{k}": v for k, v in L["dims"].items()}})
    return _concat(own), _concat(loo), d5


RUN_OWN_KEEP = ("o", "ev", "step", "first", "surface", "ci", "mi", "clean", "e1", "t1", "z", "has", "g_xy", "g_xya",
                "g_xyb", "g_xyab", "classT", "table_pop", "keybin", "dkeybin", "facing", "primary", "pu", "pv")


def c1b_cutfit(a):
    """§1 item 6 with A9 / A10: the cut fit with the disc bins, the barrel-visit bins and the barrel-only bins, the fit
    files x the scenarios, the disc arm with the fitted scale (J3) and the decided seed, one pass (the 4-sigma window
    does not read the cuts); D5 on the way. Writes c1b/cuts/wpc_cuts_c1b_v1.json (+ unconditioned), the rows and
    cutfit_summary.json."""
    import cuts as CU
    t0 = time.time()
    with open(os.path.join(a.out, "scale", "scalefit_summary.json")) as fh:
        scale = float(json.load(fh)["scale"])
    out_dir = os.path.join(a.out, "cuts")
    os.makedirs(out_dir, exist_ok=True)
    res = {"plan": "c1b/PLAN.md rev 4 §1 item 6, §3 D5, §8 A9 A10", "files": CM.FIT_FILES, "disc_scale": scale,
           "material": [], "minor": []}
    own, loo, d5 = cut_pass(a, scale)
    res["D5"] = d5
    if d5["different"]:
        res["material"].append(f"D5 (the fit files): {d5['different']} of {d5['tracks']} barrel-only tracks differ "
                               "with the discs on")
    loo_f = dict(loo)
    loo_f["dims"] = {dm: loo_f.pop(f"d_{dm}") for dm in AR.DIMS}
    prov = {"files": [str(CM.input_path(s, f)) for s in a.scen for f in CM.FIT_FILES], "disc_scale": scale,
            "plan": res["plan"]}
    CU.fit({"calibrated": own}, {"calibrated": loo_f}, prov, "smoothed", clean=True).save(os.path.join(out_dir, C1B_CUTS))
    CU.fit({"calibrated": own}, {"calibrated": loo_f}, prov, "smoothed", clean=False).save(
        os.path.join(out_dir, C1B_CUTS.replace(".json", "_unconditioned.json")))
    np.savez_compressed(os.path.join(out_dir, "c1b_cut_rows.npz"), **{f"own__{k}": v for k, v in own.items()},
                        **{f"loo__{k}": v for k, v in loo_f.items() if k != "dims"},
                        **{f"loo__d_{k}": v for k, v in loo_f["dims"].items()})
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "cutfit_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({k: res[k] for k in ("material", "D5")}, indent=1, default=str))
    return 0


def checks(a):
    """J5 (file 4, --known-bad) / J6 (file 6): the final configuration (C1b's cut file, the fitted scale, the
    decided seed, v3's position errors, A13) judged by D1, D3 (relative, A4), D4 and D5; on file 4 the D1 false veto,
    the known-bads (D1 / D4 post hoc; D3 / D5 by arm variants) and the power statements. Writes
    c1b/checks/checks_f<NN>.json."""
    import cuts as CU
    import diagnostics as DG
    import disc_checks as DC
    import run_entry as RE
    t0 = time.time()
    files = tuple(a.files)
    nev = NEV_FILE * len(files)
    geo = OP.load_geometry(CM.GEOMETRY)
    cuts = CU.Cuts.load(os.path.join(a.out, "cuts", C1B_CUTS))
    with open(os.path.join(a.out, "scale", "scalefit_summary.json")) as fh:
        scale = float(json.load(fh)["scale"])
    variants = {"model": ({}, scale)}
    if a.known_bad:
        variants.update({"kb_radius_only": ({"disc_material": "radius"}, scale), "kb_no_disc_material": ({}, 0.0),
                         "kb_2x_scale": ({}, 2 * scale), "kb_1.3x_scale (power)": ({}, 1.3 * scale),
                         "kb_barrel_path_factor": ({"disc_material": "barrel_factor"}, scale),
                         "kb_disc_material_on_barrel_only": ({"disc_material": "all_tracks"}, scale)})
    rows = {k: [] for k in variants}
    own_all, loo_all, d5 = {}, {}, {k: [0, 0] for k in variants}
    for scen in a.scen:
        own_s, loo_s = [], []
        for fi, f in enumerate(files):
            path_f = CM.input_path(scen, f)
            cfg = CM.producer_config(path_f)
            assert CM.mask_of(cfg) == scen
            evr = fi * NEV_FILE + CM.event_rank(path_f)
            for start, A in CM.iterate(path_f, AR.branches(path_f), chunk=a.chunk, nev=a.nev):
                ctx = AR.prepare(A, geo)
                off = None
                for lab, (kw, sc) in variants.items():
                    disc = AR.DiscTools(disc_scale=sc)
                    tools = AR.Tools(cfg, cuts=cuts, disc=disc)
                    arm = AR.inn_disc(cfg, **kw)
                    path, _ = AR.visit_plan(ctx, geo, cfg, scen, arm)
                    dg = AR.disc_going(path)
                    fmd = first_met_disc(ctx["a"], disc.planes, path)
                    out = AR.run_arm(ctx, geo, cfg, scen, arm, tools, own=True, loo=(lab == "model"))
                    if off is None:
                        off = AR.run_arm(ctx, geo, cfg, scen, AR.inn(cfg), tools)
                    bo = np.flatnonzero(~dg & out["refit"])
                    k = off["rec"]["acc"].shape[1]
                    sel = lambda r: np.where(r["rec"]["acc"][bo, :k], r["rec"]["ci"][bo, :k], -1)
                    diff = ~(np.all(out["a"][bo] == off["a"][bo], 1) & np.all(sel(out) == sel(off), 1))
                    d5[lab][0] += len(bo)
                    d5[lab][1] += int(diff.sum())
                    R = DG.mark_primary(out["own"])
                    m = R["primary"] & R["clean"]
                    o = R["o"][m]
                    rows[lab].append({"scen": np.full(m.sum(), CM.SCENARIOS.index(scen)), "ev": evr[R["ev"][m] + start],
                                      "surface": R["surface"][m], "step": R["step"][m], "first": R["first"][m],
                                      "pu": R["pu"][m], "pv": R["pv"][m], "dg": dg[o], "group": np.minimum(fmd[o], 5),
                                      "stubs": ctx["T"]["nStubs"][o],
                                      "ptbin": np.clip(np.searchsorted(PT_BINS, ctx["T"]["pt"][o], side="right") - 1, 0,
                                                       len(PT_BINS) - 2)})
                    if lab == "model":
                        Rc = {kk: (R["csurface"] if kk == "surface" else v) for kk, v in R.items()}
                        Rc["ev"] = evr[R["ev"] + start]
                        own_s.append(Rc)
                        L = out["loo"]
                        prim, clean_track = RE.loo_flags(out, R, ctx)
                        loo_s.append({"surface": L["csurface"], "first": L["first"], "own_primary": prim,
                                      "clean_track": clean_track, "scen": np.full(len(prim), scen), "n_inner": L["n_inner"],
                                      "n_acc": L["n_acc"], "ev": evr[ctx["tev"][L["o"]] + start],
                                      **{f"d_{dd}": v for dd, v in L["dims"].items()}})
            print(f"[{time.time() - t0:7.1f}s] {scen} f{f:02d}", flush=True)
        own_all[scen], loo_all[scen] = _concat(own_s), _concat(loo_s)
    reps = G.replicas(nev=nev)
    res = {"plan": "c1b/PLAN.md rev 4 §3-§4", "files": list(files), "role": "known-good" if a.known_bad else "gate",
           "cuts": C1B_CUTS, "disc_scale": scale, "material": [], "minor": []}
    # D1 (pooled over scenarios: the disc rows are scenario-independent; one copy of each event is AAAA's)
    imp = AR.DiscTools().imp
    d1v = ("no_rows", "flip_centre", "global_rows", "key_from_local", "to_local", "width_div_1.15", "emitted_key") \
        if a.known_bad else ()
    res["D1"] = DC.d1(own_all["AAAA"], imp, geo, nev, false_veto=a.known_bad, variants=d1v)
    if res["D1"]["pooled"]["n_fail"]:
        res["material"].append(f"D1: {res['D1']['pooled']['n_fail']} failing bins (pooled facings)")
    if a.known_bad and res["D1"]["false_veto"]["material"]:
        res["material"].append(f"D1 false veto: lower bound {res['D1']['false_veto']['lower95']:.3f} > 5.7 %")
    if any(v["facing_only"] for v in res["D1"]["variants"].values()) and res["D1"]["per_facing"]["n_fail"]:
        res["material"].append(f"D1 per facing: {res['D1']['per_facing']['n_fail']} failing bins (a facing known-bad "
                               "is caught only per facing)")
    # D3 (A4) for the model and the D3 / D5 known-bads
    res["D3"] = {}
    for lab in variants:
        J = judge_all(_concat(rows[lab]), reps)
        res["D3"][lab] = J
        res["D3"][lab]["any_fail"] = bool(J["D3a_fail"] or J["D3b_fail"] or J["D3c_fail"])
    M = res["D3"]["model"]
    res["material"] += M["D3a_fail"] + M["D3b_fail"] + M["D3c_fail"]
    # D4 per scenario; its known-bads on file 4
    res["D4"] = {scen: DC.d4(own_all[scen], loo_all[scen], cuts, scen, nev) for scen in a.scen}
    for scen, e in res["D4"].items():
        res["material"] += e["material"]
        res["minor"] += e["minor"]
    if a.known_bad:
        z = np.load(os.path.join(a.out, "cuts", "c1b_cut_rows.npz"))
        R = {k[5:]: z[k] for k in z.files if k.startswith("own__")}
        m = R["primary"] & R["table_pop"] & R["clean"]
        old = CU.LEVEL
        CU.LEVEL = 0.975
        try:
            tab = CU._fit_table(R["g_xy"][m], R["surface"][m], R["first"][m])
        finally:
            CU.LEVEL = old
        c975 = CU.Cuts({"schema": CU.SCHEMA, "level": 0.975, "cuts": {"gate|xy|calibrated": tab}})
        res["D4_known_bads"] = DC.d4_known_bads(_concat(list(own_all.values())), c975)
    # D5
    res["D5"] = {lab: {"barrel_only_tracks": v[0], "different": v[1]} for lab, v in d5.items()}
    if d5["model"][1]:
        res["material"].append(f"D5: {d5['model'][1]} of {d5['model'][0]} barrel-only tracks differ with the discs on")
    # the known-bads' verdicts
    if a.known_bad:
        kb = {f"D1/{k}": v["fails"] for k, v in res["D1"]["variants"].items() if k != "emitted_key"}
        kb.update({f"D3/{k}": res["D3"][k]["any_fail"] for k in variants if k.startswith("kb_") and "power" not in k
                   and k != "kb_disc_material_on_barrel_only"})
        kb["D5/disc_material_on_barrel_only"] = d5["kb_disc_material_on_barrel_only"][1] > 0
        kb.update({f"D4/{k}": v["fails"] for k, v in res["D4_known_bads"].items()})
        res["known_bads"] = kb
        res["power"] = {"D1 width / 1.15": res["D1"]["variants"]["width_div_1.15"],
                        "D3 1.3 x scale": res["D3"]["kb_1.3x_scale (power)"]["any_fail"]}
        res["material"] += [f"known-bad {k} does not fail its check" for k, v in kb.items() if not v]
    res["seconds"] = round(time.time() - t0, 1)
    out_dir = os.path.join(a.out, "checks")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"checks_f{'_'.join(f'{x:02d}' for x in files)}.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else
                  (float(o) if np.isscalar(o) else str(o)))
    print(json.dumps({k: res.get(k) for k in ("material", "known_bads")}, indent=1, default=str))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("pairs", "il1truth", "fit", "seedtest", "scalefit", "cutfit", "checks"))
    ap.add_argument("--scen", nargs="+", default=list(CM.SCENARIOS))
    ap.add_argument("--files", type=int, nargs="+", default=None, help="pairs / il1truth / fit: files other than the fit files")
    ap.add_argument("--known-bad", action="store_true", help="file 4: the known-bads")
    ap.add_argument("--n-boot", type=int, default=OV.N_BOOT,
                    help="fit: the shape fits' bootstrap replicas (a disc: at most N_BOOT_DISC)")
    ap.add_argument("--n-toy", type=int, default=OV.N_TOY, help="fit: T0's replicas per surface and planted shape")
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--out", default=str(C1B_OUT))
    a = ap.parse_args(argv)
    return {"pairs": OV.pairs, "il1truth": OV.il1truth, "fit": OV.fit, "seedtest": seedtest,
            "scalefit": scalefit, "cutfit": c1b_cutfit, "checks": checks}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
