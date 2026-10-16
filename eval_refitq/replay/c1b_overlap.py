"""C1b J1 (c1b/PLAN.md rev 4 §1 item 2 with §8 A8, A10): the parts that do not depend on the single-cluster shape.

  pairs     AAAA only (one copy of each event), the fit files (A10) or --files: the overlap pairs on IL1 and ID1-ID4
            and IL1's own clusters with their truth residuals -> c1b/overlap[_fNN]/pairs.npz, il1_clusters.npz
  il1truth  the IL1 truth width against p per axis x size bin with the beam pipe's Highland expectation; the plateau
            p* (A8's rule); the finding on v3's barrel tables (did they pool scattering-inflated clusters?); the
            u-v tail independence; the count-only power of the IL1 method check -> il1truth_summary.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy import stats

import arms as AR
import common as CM
import disc_r as DR
import gates as G
from common import OP

IL1, DISCS = 1, (11, 12, 13, 14)
NEV_FILE = 100
P_BINS = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, np.inf)    # GeV: the truth width against p
PSTAR_EDGES = (10.0, 15.0, 20.0, 30.0)               # the plateau's candidate starts (A8)
PLATEAU_MIN_N, PBIN_MIN_N = 200, 50                  # clusters above p* for a bin to be tested; per p bin in its chi2
PLATEAU_CHI2_P, PLATEAU_INFL = 0.01, 0.01            # (i) flat: chi2 tail probability >= 0.01; (ii) Highland <= 1 %
FINDING_PT = (5.0, 10.0, 20.0, 50.0, np.inf)         # GeV: v3's tables were built at pT >= 5 GeV
V3_MIN_N = 40                                        # v3's cell minimum (refit_calibration_fit.R_MIN_N)
POWER_PSTAR = 30.0                                   # the count-only power at the worst-case p*
TAIL_Z = 3.0                                         # the u-v independence check's tail


def out_dir(a, files):
    return os.path.join(a.out, "overlap" if files == CM.FIT_FILES else f"overlap_f{'_'.join(f'{x:02d}' for x in files)}")


def _concat(parts):
    parts = [p for p in parts if p]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def r_tables():
    return AR.Tools(None, calibration=AR.CALIBRATION, angle_table=None).cal["R_tables"]


def pairs(a):
    t0 = time.time()
    files = tuple(a.files) if a.files else CM.FIT_FILES
    geo = OP.load_geometry(CM.GEOMETRY)
    P, C = [], []
    for fi, f in enumerate(files):
        path = CM.input_path("AAAA", f)
        for start, A in CM.iterate(path, AR.branches(path), chunk=a.chunk, nev=a.nev):
            ctx = AR.prepare(A, geo)
            ev0 = fi * NEV_FILE + start
            P.append(DR.pairs_of_chunk(ctx, geo, (IL1,) + DISCS, start=ev0))
            C.append(DR.clusters_of_chunk(ctx, geo, IL1, start=ev0))
        print(f"[{time.time() - t0:7.1f}s] AAAA f{f:02d}: pairs and IL1 clusters", flush=True)
    d = out_dir(a, files)
    os.makedirs(d, exist_ok=True)
    P, C = _concat(P), _concat(C)
    np.savez_compressed(os.path.join(d, "pairs.npz"), files=np.array(files), **P)
    np.savez_compressed(os.path.join(d, "il1_clusters.npz"), files=np.array(files), **C)
    n = {str(s): int((P["surface"] == s).sum()) for s in (IL1,) + DISCS}
    print(json.dumps({"files": files, "pairs": n, "il1_clusters": len(C["p"])}), flush=True)
    return 0


def load(a, files):
    d = out_dir(a, files)
    zp, zc = np.load(os.path.join(d, "pairs.npz")), np.load(os.path.join(d, "il1_clusters.npz"))
    assert tuple(zp["files"]) == tuple(files) == tuple(zc["files"])
    return ({k: zp[k] for k in zp.files if k != "files"}, {k: zc[k] for k in zc.files if k != "files"})


def _boot_width(x, ev, reps):
    """MAD / 0.6745 of x and its event-bootstrap sd."""
    return DR.wmad_width(x), float(np.nanstd([DR.wmad_width(x, r[ev]) for r in reps]))


def width_vs_p(C, tables, reps):
    """Per axis: the truth width per size bin x p bin (n >= PBIN_MIN_N), its bootstrap sd and the counts."""
    pb = np.clip(np.searchsorted(P_BINS, C["p"], side="right") - 1, 0, len(P_BINS) - 2)
    out = {}
    for ax in ("u", "v"):
        sk, _, rk, _ = DR.AX_KEYS[ax]
        sb = DR.size_bin(C[sk], tables[ax]["size_edges"])
        nb, npb = len(tables[ax]["size_edges"]) - 1, len(P_BINS) - 1
        W, SD, N = np.full((nb, npb), np.nan), np.full((nb, npb), np.nan), np.zeros((nb, npb), int)
        for i in range(nb):
            for k in range(npb):
                m = (sb == i) & (pb == k)
                N[i, k] = int(m.sum())
                if N[i, k] >= PBIN_MIN_N:
                    W[i, k], SD[i, k] = _boot_width(C[rk][m], C["ev"][m], reps)
        out[ax] = {"width": W, "sd": SD, "n": N, "size_bin": sb}
    return out, pb


def plateau(C, tables, tv, pb, rng):
    """A8's plateau rule: p* is the lowest edge in PSTAR_EDGES for which every axis x size bin with >= PLATEAU_MIN_N
    clusters above it has (i) widths of its p bins above p* consistent with a constant and (ii) a Highland inflation
    above p* of at most 1 %; else 30 GeV, reported."""
    tests = {}
    chosen = None
    for ps in PSTAR_EDGES:
        fails, rows = [], []
        for ax in ("u", "v"):
            _, _, rk, bk = DR.AX_KEYS[ax]
            W, SD, N, sb = (tv[ax][k] for k in ("width", "sd", "n", "size_bin"))
            for i in range(W.shape[0]):
                above = (sb == i) & (C["p"] >= ps)
                if above.sum() < PLATEAU_MIN_N:
                    continue
                ks = [k for k in range(W.shape[1]) if P_BINS[k] >= ps and N[i, k] >= PBIN_MIN_N and SD[i, k] > 0]
                pval = 1.0
                if len(ks) >= 2:
                    w, s = W[i, ks], SD[i, ks]
                    mean = np.sum(w / s ** 2) / np.sum(1 / s ** 2)
                    pval = float(stats.chi2.sf(np.sum((w - mean) ** 2 / s ** 2), len(ks) - 1))
                infl = DR.highland_inflation(C[rk][above], C[bk][above], rng) - 1.0
                rows.append({"axis": ax, "size_bin": i, "n": int(above.sum()), "p_bins": len(ks), "chi2_p": pval,
                             "highland_inflation": infl})
                if pval < PLATEAU_CHI2_P or infl > PLATEAU_INFL:
                    fails.append(f"{ax}{i}")
        tests[str(ps)] = {"bins": rows, "fails": fails}
        if chosen is None and not fails:
            chosen = ps
    return (chosen if chosen is not None else PSTAR_EDGES[-1]), chosen is None, tests


def highland_table(C, tables, tv, pb, pstar, rng):
    """Per axis x size bin x p bin: the measured width and the Highland expectation (the plateau residuals with each
    of the bin's clusters' beam-pipe offset), both over the plateau width."""
    out = {}
    for ax in ("u", "v"):
        _, _, rk, bk = DR.AX_KEYS[ax]
        W, N, sb = tv[ax]["width"], tv[ax]["n"], tv[ax]["size_bin"]
        E, Wp = np.full(W.shape, np.nan), np.full(W.shape[0], np.nan)
        for i in range(W.shape[0]):
            plat = (sb == i) & (C["p"] >= pstar)
            if plat.sum() < PBIN_MIN_N:
                continue
            Wp[i] = DR.wmad_width(C[rk][plat])
            for k in range(W.shape[1]):
                m = (sb == i) & (pb == k)
                if N[i, k] >= PBIN_MIN_N:
                    E[i, k] = DR.highland_expected(C[rk][plat], C[bk][m], rng)
        out[ax] = {"plateau_width": Wp, "measured_over_plateau": W / Wp[:, None], "expected_over_plateau": E / Wp[:, None],
                   "median_bp_offset_um": [[float(np.median(C[bk][(sb == i) & (pb == k)]) * DR.UM) if N[i, k] else np.nan
                                            for k in range(W.shape[1])] for i in range(W.shape[0])]}
    return out


def barrel_finding(C, tables, pstar):
    """Did v3's barrel R tables (IL1 truth residuals, pT >= 5 GeV) pool scattering-inflated clusters? Per v3 cell
    (size x |cot|) and pT bin: the truth width over the plateau width at the same cell; v3's own value and the pT >= 5
    pooled width (v3's selection, on this sample) over the plateau; summaries count-weighted over cells."""
    ptb = np.clip(np.searchsorted(FINDING_PT, C["pt"], side="right") - 1, -1, len(FINDING_PT) - 2)
    ptb = np.where(C["pt"] < FINDING_PT[0], -1, ptb)
    out = {}
    for ax in ("u", "v"):
        sk, ck, rk, _ = DR.AX_KEYS[ax]
        t = tables[ax]
        sb, cb = DR.size_bin(C[sk], t["size_edges"]), DR.size_bin(np.abs(C[ck]), t["cot_edges"])
        nb, nc, npt = len(t["size_edges"]) - 1, len(t["cot_edges"]) - 1, len(FINDING_PT) - 1
        plat, pooled = np.full((nb, nc), np.nan), np.full((nb, nc), np.nan)
        per_pt, n_pt = np.full((npt, nb, nc), np.nan), np.zeros((npt, nb, nc), int)
        for i in range(nb):
            for j in range(nc):
                cell = (sb == i) & (cb == j)
                m = cell & (C["p"] >= pstar)
                if m.sum() >= V3_MIN_N:
                    plat[i, j] = DR.wmad_width(C[rk][m])
                m = cell & (ptb >= 0)
                if m.sum() >= V3_MIN_N:
                    pooled[i, j] = DR.wmad_width(C[rk][m])
                for k in range(npt):
                    m = cell & (ptb == k)
                    n_pt[k, i, j] = int(m.sum())
                    if n_pt[k, i, j] >= V3_MIN_N:
                        per_pt[k, i, j] = DR.wmad_width(C[rk][m])
        v3 = np.asarray(t["width_cm"], np.float64)
        npool = n_pt.sum(0)

        def wmean(r, w):
            ok = np.isfinite(r) & (w > 0)
            return float(np.sum(r[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan

        out[ax] = {"plateau_cm": plat, "per_pt_over_plateau": per_pt / plat[None], "n_per_pt": n_pt,
                   "pooled_pt5_over_plateau": pooled / plat, "v3_over_plateau": v3 / plat,
                   "summary": {"per_pt_bin": {f"{FINDING_PT[k]:g}-{FINDING_PT[k + 1]:g}": wmean(per_pt[k] / plat, n_pt[k])
                                              for k in range(npt)},
                               "pooled_pt5": wmean(pooled / plat, npool), "v3_table": wmean(v3 / plat, npool)}}
    return out


def uv_independence(C, tables, pstar):
    """On the plateau clusters: the joint tail rate of the standardized u and v residuals (|z| > TAIL_Z, per size
    bin's median and width) against the product of the two marginal rates (A8's independence assumption)."""
    m = C["p"] >= pstar
    z = {}
    for ax in ("u", "v"):
        sk, _, rk, _ = DR.AX_KEYS[ax]
        sb = DR.size_bin(C[sk][m], tables[ax]["size_edges"])
        x = C[rk][m]
        zz = np.full(len(x), np.nan)
        for i in np.unique(sb):
            k = sb == i
            if k.sum() >= PBIN_MIN_N:
                med = np.median(x[k])
                zz[k] = (x[k] - med) / DR.wmad_width(x[k])
        z[ax] = zz
    ok = np.isfinite(z["u"]) & np.isfinite(z["v"])
    tu, tv_ = np.abs(z["u"][ok]) > TAIL_Z, np.abs(z["v"][ok]) > TAIL_Z
    n, joint = int(ok.sum()), int(np.sum(tu & tv_))
    expect = float(tu.mean() * tv_.mean() * n)
    return {"n": n, "tail_u": float(tu.mean()), "tail_v": float(tv_.mean()), "joint": joint, "expected_if_independent": expect,
            "ratio": joint / expect if expect > 0 else np.nan,
            "poisson_95": [float(stats.chi2.ppf(0.025, 2 * joint) / 2) / expect if joint and expect else 0.0,
                           float(stats.chi2.ppf(0.975, 2 * joint + 2) / 2) / expect if expect else np.nan]}


def count_power(P, C, reps, pstars):
    """A8's count-only power of the IL1 check: 1.645 sqrt(s_pair^2 + s_ref^2), s_pair the event-bootstrap relative
    sd of the model-free pooled robust width of the IL1 pair differences (p >= 2 GeV), s_ref = 1.166 / sqrt(n_ref)
    at each p*."""
    m = (P["surface"] == IL1) & (P["p"] >= DR.P_MIN["barrel"])
    Q = {k: v[m] for k, v in P.items()}
    D, _ = DR.differences(Q)
    out = {"pairs": int(m.sum())}
    for d, ax in (("phi", "u"), ("2", "v")):
        w0, sd = _boot_width(D[d], Q["ev"], reps)
        s_pair = sd / w0
        out[ax] = {"s_pair": s_pair}
        for ps in pstars:
            n_ref = int(np.sum(C["p"] >= ps))
            s_ref = 1.166 / np.sqrt(max(n_ref, 1))
            out[ax][f"pstar_{ps:g}"] = {"n_ref": n_ref, "s_ref": s_ref, "half_width95": 1.645 * np.hypot(s_pair, s_ref)}
    return out


def il1truth(a):
    t0 = time.time()
    files = tuple(a.files) if a.files else CM.FIT_FILES
    P, C = load(a, files)
    tables = r_tables()
    reps = G.replicas(nev=NEV_FILE * len(files))
    rng = np.random.default_rng(20261016)
    res = {"plan": "c1b/PLAN.md rev 4 §8 A8 (IL1's reference widths; the finding), A10", "files": files,
           "clusters": len(C["p"]), "p_bins": P_BINS, "material": [], "minor": []}
    tv, pb = width_vs_p(C, tables, reps)
    res["width_vs_p"] = {ax: {k: v for k, v in tv[ax].items() if k != "size_bin"} for ax in tv}
    print(f"[{time.time() - t0:7.1f}s] width against p", flush=True)
    pstar, fallback, tests = plateau(C, tables, tv, pb, rng)
    res.update({"pstar": pstar, "pstar_fallback": fallback, "plateau_tests": tests})
    if fallback:
        res["minor"].append(f"no plateau edge passes A8's rule: p* = {pstar:g} GeV (reported)")
    print(f"[{time.time() - t0:7.1f}s] p* = {pstar:g} GeV{' (fallback)' if fallback else ''}", flush=True)
    res["highland"] = highland_table(C, tables, tv, pb, pstar, rng)
    res["barrel_finding"] = barrel_finding(C, tables, pstar)
    res["uv_independence"] = uv_independence(C, tables, pstar)
    res["count_power"] = count_power(P, C, reps, sorted({pstar, POWER_PSTAR}))
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir(a, files), "il1truth_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({"pstar": pstar, "fallback": fallback,
                      "finding": {ax: res["barrel_finding"][ax]["summary"] for ax in ("u", "v")},
                      "uv": res["uv_independence"]["ratio"], "power": res["count_power"]}, indent=1, default=float))
    return 0
