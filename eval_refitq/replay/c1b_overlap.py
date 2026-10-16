"""C1b J1 (c1b/PLAN.md rev 4 §1 item 2 with §8 A8, A10): the parts that do not depend on the single-cluster shape.

  pairs     AAAA only (one copy of each event), the fit files (A10) or --files: the overlap pairs on IL1 and ID1-ID4
            and IL1's own clusters with their truth residuals -> c1b/overlap[_fNN]/pairs.npz, il1_clusters.npz
  il1truth  the IL1 truth width against p per axis x size bin with the beam pipe's Highland expectation; IL1's
            reference clusters (A11: each cluster's predicted beam-pipe offset below 15 % of its bin's width); the
            finding on v3's barrel tables (did they pool scattering-inflated clusters?); the tables' width (MAD)
            against the RMS; the u-v tail independence; the count-only power of the IL1 method check
            -> il1truth_summary.json
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
PBIN_MIN_N = 50                                      # clusters per p bin for a width
REF_OFFSET_FRAC, REF_ITER, REF_START_P = 0.15, 5, 30.0   # A11: offset < 15 % of the bin's width (<= 1 % inflation),
REF_MIN_N = 200                                      # iterated 5 times from the p >= 30 GeV width; judged with >= 200
FINDING_PT = (5.0, 10.0, 20.0, 50.0, np.inf)         # GeV: v3's tables were built at pT >= 5 GeV
V3_MIN_N = 40                                        # v3's cell minimum (refit_calibration_fit.R_MIN_N)
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


def reference(C, tables, rng):
    """A11: IL1's reference clusters per axis x size bin, those whose predicted beam-pipe offset is below
    REF_OFFSET_FRAC of the bin's width, the width iterated from the p >= 30 GeV one; per axis the cluster mask
    (judged bins only) and per bin n, width, the residual Highland inflation and whether it is judged."""
    out = {}
    for ax in ("u", "v"):
        sk, _, rk, bk = DR.AX_KEYS[ax]
        sb = DR.size_bin(C[sk], tables[ax]["size_edges"])
        mask = np.zeros(len(sb), bool)
        rows = []
        for i in range(len(tables[ax]["size_edges"]) - 1):
            b = sb == i
            hi = b & (C["p"] >= REF_START_P)
            W = DR.wmad_width(C[rk][hi]) if hi.sum() >= PBIN_MIN_N else DR.wmad_width(C[rk][b]) if b.any() else np.nan
            sel = b & (C[bk] < REF_OFFSET_FRAC * W)
            for _ in range(REF_ITER):
                if sel.sum() < PBIN_MIN_N:
                    break
                W = DR.wmad_width(C[rk][sel])
                sel = b & (C[bk] < REF_OFFSET_FRAC * W)
            n = int(sel.sum())
            judged = n >= REF_MIN_N
            infl = DR.highland_inflation(C[rk][sel], C[bk][sel], rng) - 1.0 if n >= PBIN_MIN_N else np.nan
            rows.append({"size_bin": i, "n": n, "width_cm": W if n >= PBIN_MIN_N else np.nan, "highland_inflation": infl,
                         "judged": judged})
            if judged:
                mask |= sel
        out[ax] = {"bins": rows, "mask": mask}
    return out


def highland_table(C, tables, tv, pb, ref, rng):
    """Per axis x size bin x p bin: the measured width and the Highland expectation (the reference residuals with
    each of the bin's clusters' beam-pipe offset), both over the reference width."""
    out = {}
    for ax in ("u", "v"):
        _, _, rk, bk = DR.AX_KEYS[ax]
        W, N, sb = tv[ax]["width"], tv[ax]["n"], tv[ax]["size_bin"]
        E = np.full(W.shape, np.nan)
        Wr = np.array([r["width_cm"] for r in ref[ax]["bins"]], np.float64)
        for i in range(W.shape[0]):
            x_ref = C[rk][(sb == i) & ref[ax]["mask"]]
            if len(x_ref) < PBIN_MIN_N:
                continue
            for k in range(W.shape[1]):
                m = (sb == i) & (pb == k)
                if N[i, k] >= PBIN_MIN_N:
                    E[i, k] = DR.highland_expected(x_ref, C[bk][m], rng)
        out[ax] = {"reference_width": Wr, "measured_over_reference": W / Wr[:, None],
                   "expected_over_reference": E / Wr[:, None],
                   "median_bp_offset_um": [[float(np.median(C[bk][(sb == i) & (pb == k)]) * DR.UM) if N[i, k] else np.nan
                                            for k in range(W.shape[1])] for i in range(W.shape[0])]}
    return out


def barrel_finding(C, tables, ref):
    """Did v3's barrel R tables (IL1 truth residuals, pT >= 5 GeV) pool scattering-inflated clusters? Per v3 cell
    (size x |cot|) and pT bin: the truth width over the reference width at the same cell; v3's own value and the
    pT >= 5 pooled width (v3's selection, on this sample) over the reference; summaries count-weighted over cells."""
    ptb = np.clip(np.searchsorted(FINDING_PT, C["pt"], side="right") - 1, -1, len(FINDING_PT) - 2)
    ptb = np.where(C["pt"] < FINDING_PT[0], -1, ptb)
    out = {}
    for ax in ("u", "v"):
        sk, ck, rk, _ = DR.AX_KEYS[ax]
        t = tables[ax]
        sb, cb = DR.size_bin(C[sk], t["size_edges"]), DR.size_bin(np.abs(C[ck]), t["cot_edges"])
        nb, nc, npt = len(t["size_edges"]) - 1, len(t["cot_edges"]) - 1, len(FINDING_PT) - 1
        refw, pooled = np.full((nb, nc), np.nan), np.full((nb, nc), np.nan)
        per_pt, n_pt = np.full((npt, nb, nc), np.nan), np.zeros((npt, nb, nc), int)
        for i in range(nb):
            for j in range(nc):
                cell = (sb == i) & (cb == j)
                m = cell & ref[ax]["mask"]
                if m.sum() >= V3_MIN_N:
                    refw[i, j] = DR.wmad_width(C[rk][m])
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

        out[ax] = {"reference_cm": refw, "per_pt_over_reference": per_pt / refw[None], "n_per_pt": n_pt,
                   "pooled_pt5_over_reference": pooled / refw, "v3_over_reference": v3 / refw,
                   "summary": {"per_pt_bin": {f"{FINDING_PT[k]:g}-{FINDING_PT[k + 1]:g}": wmean(per_pt[k] / refw, n_pt[k])
                                              for k in range(npt)},
                               "pooled_pt5": wmean(pooled / refw, npool), "v3_table": wmean(v3 / refw, npool)}}
    return out


def mad_vs_rms(C, tables, ref):
    """A11's cheap finding: on the reference clusters, per axis x size bin, the tables' width (MAD / 0.6745) against
    the RMS about the median (within the 1 mm the clusters are selected in); the occupancy-weighted variance ratio
    RMS^2 / width^2 over the judged bins."""
    out = {}
    for ax in ("u", "v"):
        sk, _, rk, _ = DR.AX_KEYS[ax]
        sb = DR.size_bin(C[sk], tables[ax]["size_edges"])
        rows, num, den = [], 0.0, 0
        for i in range(len(tables[ax]["size_edges"]) - 1):
            m = (sb == i) & ref[ax]["mask"]
            if m.sum() < REF_MIN_N:
                continue
            x = C[rk][m]
            w = DR.wmad_width(x)
            rms = float(np.sqrt(np.mean((x - np.median(x)) ** 2)))
            rows.append({"size_bin": i, "n": int(m.sum()), "width_cm": w, "rms_cm": rms, "rms_over_width": rms / w})
            num += m.sum() * (rms / w) ** 2
            den += m.sum()
        out[ax] = {"bins": rows, "variance_ratio": num / den if den else np.nan}
    return out


def uv_independence(C, tables, ref):
    """On the clusters in both axes' references: the joint tail rate of the standardized u and v residuals (|z| >
    TAIL_Z, per size bin's median and width) against the product of the two marginal rates (A8 / A11)."""
    m = ref["u"]["mask"] & ref["v"]["mask"]
    z = {}
    for ax in ("u", "v"):
        sk, _, rk, _ = DR.AX_KEYS[ax]
        sb = DR.size_bin(C[sk][m], tables[ax]["size_edges"])
        x = C[rk][m]
        zz = np.full(len(x), np.nan)
        for i in np.unique(sb):
            k = sb == i
            if k.sum() >= PBIN_MIN_N:
                zz[k] = (x[k] - np.median(x[k])) / DR.wmad_width(x[k])
        z[ax] = zz
    return tail_independence(z["u"], z["v"])


def tail_independence(zu, zv):
    """The joint |z| > TAIL_Z rate of two standardized variables against the product of the marginal rates."""
    ok = np.isfinite(zu) & np.isfinite(zv)
    tu, tv_ = np.abs(zu[ok]) > TAIL_Z, np.abs(zv[ok]) > TAIL_Z
    n, joint = int(ok.sum()), int(np.sum(tu & tv_))
    expect = float(tu.mean() * tv_.mean() * n) if n else 0.0
    return {"n": n, "tail_1": float(tu.mean()) if n else np.nan, "tail_2": float(tv_.mean()) if n else np.nan,
            "joint": joint, "expected_if_independent": expect, "ratio": joint / expect if expect > 0 else np.nan,
            "poisson_95": [float(stats.chi2.ppf(0.025, 2 * joint) / 2) / expect if joint and expect else 0.0,
                           float(stats.chi2.ppf(0.975, 2 * joint + 2) / 2) / expect if expect else np.nan]}


def count_power(P, ref, reps):
    """A8's count-only power of the IL1 check: 1.645 sqrt(s_pair^2 + s_ref^2), s_pair the event-bootstrap relative
    sd of the model-free pooled robust width of the IL1 pair differences (p >= 2 GeV), s_ref = 1.166 / sqrt(n_ref)
    with n_ref the axis's judged reference clusters."""
    m = (P["surface"] == IL1) & (P["p"] >= DR.P_MIN["barrel"])
    Q = {k: v[m] for k, v in P.items()}
    D, _ = DR.differences(Q)
    out = {"pairs": int(m.sum())}
    for d, ax in (("phi", "u"), ("2", "v")):
        w0, sd = _boot_width(D[d], Q["ev"], reps)
        n_ref = int(ref[ax]["mask"].sum())
        s_pair, s_ref = sd / w0, 1.166 / np.sqrt(max(n_ref, 1))
        out[ax] = {"s_pair": s_pair, "n_ref": n_ref, "s_ref": s_ref, "half_width95": 1.645 * np.hypot(s_pair, s_ref)}
    return out


def il1truth(a):
    t0 = time.time()
    files = tuple(a.files) if a.files else CM.FIT_FILES
    P, C = load(a, files)
    tables = r_tables()
    reps = G.replicas(nev=NEV_FILE * len(files))
    rng = np.random.default_rng(20261016)
    res = {"plan": "c1b/PLAN.md rev 4 §8 A8, A11 (IL1's reference; the findings), A10", "files": files,
           "clusters": len(C["p"]), "p_bins": P_BINS, "material": [], "minor": []}
    tv, pb = width_vs_p(C, tables, reps)
    res["width_vs_p"] = {ax: {k: v for k, v in tv[ax].items() if k != "size_bin"} for ax in tv}
    print(f"[{time.time() - t0:7.1f}s] width against p", flush=True)
    ref = reference(C, tables, rng)
    res["reference"] = {ax: ref[ax]["bins"] for ax in ref}
    res["minor"] += [f"IL1 reference {ax}{r['size_bin']}: {r['n']} clusters (< {REF_MIN_N}), reported, not judged"
                     for ax in ref for r in ref[ax]["bins"] if not r["judged"]]
    np.savez_compressed(os.path.join(out_dir(a, files), "il1_reference.npz"), u=ref["u"]["mask"], v=ref["v"]["mask"])
    res["highland"] = highland_table(C, tables, tv, pb, ref, rng)
    res["barrel_finding"] = barrel_finding(C, tables, ref)
    res["mad_vs_rms"] = mad_vs_rms(C, tables, ref)
    res["uv_independence"] = uv_independence(C, tables, ref)
    res["count_power"] = count_power(P, ref, reps)
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir(a, files), "il1truth_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({"reference": res["reference"], "finding": {ax: res["barrel_finding"][ax]["summary"] for ax in ("u", "v")},
                      "mad_vs_rms": {ax: res["mad_vs_rms"][ax]["variance_ratio"] for ax in ("u", "v")},
                      "uv": res["uv_independence"]["ratio"], "power": res["count_power"]}, indent=1, default=float))
    return 0


# ------------------------------------------------------------------ the shape fits (A8 with A11)
MODEL, SYSTEMATIC = "split2", "split1"               # A11: a split core plus 2 tails; the systematic, plus 1 tail
TOY_SHAPES = ("box", "t3", "mix2")                   # A11: flat-topped, heavy-tailed, two-Gaussian
# A12 (the lane budget): event-bootstrap replicas of the shape fits on IL1 and on a disc; of the facing and p splits
N_BOOT, N_BOOT_DISC, N_BOOT_SUB = 100, 50, 30
N_TOY, N_TOY_KB = 20, 10                             # T0 per surface and shape; the no-rotation known-bad
WORKERS = 3                                          # surfaces fitted in parallel (spawned processes)
JUDGE_CLUSTERS = 400                                 # a size bin is judged with >= 200 pairs (two clusters per pair)
TOY_TOL, TOY_SE_MAX = 0.03, 0.01                     # T0: within 3 % where the replica mean is known to 1 % (A3)
SYS_TOL = 0.03                                       # A8 / A11: the shape systematic's MATERIAL level
D0_BAND, D0_HALF_MAX = (0.90, 1.10), 0.10
D2_TOL, FACING_Z = 0.10, 2.0
IL1_P_SPLIT = (2.0, 5.0, 10.0, np.inf)
FIT_KEYS = ("R_u", "R_v", "B_prime", "b", "converged", "n_u", "n_v", "shape_u", "shape_v", "sparse_u", "sparse_v")


def _judged(F, ax):
    """The raw size bins a fit judges: their own shape (not sparse) and >= JUDGE_CLUSTERS clusters."""
    return F.own[ax] & (F.cnt[ax] >= JUDGE_CLUSTERS)


def _light(r):
    return {k: r[k] for k in FIT_KEYS}


def _rel(r, planted):
    return {ax: r[f"R_{ax}"] / planted[ax] - 1.0 for ax in ("u", "v")}


def t0_surface(F, Fkb, planted, rng, n_toy, n_kb):
    """T0 on one surface's pair geometry: per planted shape, n_toy replicas (each fit warm-started from the previous
    one); the deviation of the recovered width per bin, its standard error and A3's judged bins; the no-rotation
    known-bad (in-family plant) judged on the model's judged bins for that plant."""
    pw = {a: planted[a] * DR.UM for a in planted}
    out = {}
    for shape in TOY_SHAPES:
        th, rel = None, []
        for _ in range(n_toy):
            r = F.fit(D=F.toy_D(planted, shape, rng), theta0=th, start_widths=pw)
            th = r["theta"]
            rel.append(_rel(r, planted))
        e = {"replicas": n_toy}
        for ax in ("u", "v"):
            x = np.array([q[ax] for q in rel])
            dev = np.nanmean(x, 0)
            se = np.nanstd(x, 0) / np.sqrt(np.maximum(np.sum(np.isfinite(x), 0), 1))
            jb = [int(b) for b in range(len(dev)) if _judged(F, ax)[b] and se[b] <= TOY_SE_MAX]
            e[ax] = {"deviation": dev, "se": se, "judged": jb, "failing": [b for b in jb if not abs(dev[b]) <= TOY_TOL]}
        out[shape] = e
    if Fkb is not None:
        th, rel = None, []
        for _ in range(n_kb):
            r = Fkb.fit(D=F.toy_D(planted, "mix2", rng), theta0=th, start_widths=pw)
            th = r["theta"]
            rel.append(_rel(r, planted))
        e = {"replicas": n_kb}
        for ax in ("u", "v"):
            dev = np.nanmean(np.array([q[ax] for q in rel]), 0)
            e[ax] = {"deviation": dev, "failing": [b for b in out["mix2"][ax]["judged"] if not abs(dev[b]) <= TOY_TOL]}
        e["fails"] = bool(e["u"]["failing"] or e["v"]["failing"])
        out["known_bad_no_rotation"] = e
    return out


def _boot(F, reps, ev, c0, **kw):
    return [c0] + [F.fit(weights=r[ev[F.idx]], theta0=c0["theta"], **kw) for r in reps]


def surface_work(job):
    """One surface's shape fits (run in a spawned worker): the central fit, T0 (fit files), the bootstrap, the
    systematic family; on a disc the facing split and the facing-sign known-bad; on IL1 the fitted width against p."""
    s, d, files, n_boot, n_toy, fit_files = (job[k] for k in ("surface", "dir", "files", "n_boot", "n_toy", "fit_files"))
    t0 = time.time()
    z = np.load(os.path.join(d, "pairs.npz"))
    P = {k: z[k] for k in z.files if k != "files"}
    tables = r_tables()
    planted = DR.planted_widths(tables)
    pw = {ax: planted[ax] * DR.UM for ax in planted}
    rng = np.random.default_rng([20261016, s])
    reps = G.replicas(nev=NEV_FILE * len(files), n=n_boot)
    ev = P["ev"]
    on = P["surface"] == s
    F = DR.ShapeFit(P, tables, on, family=MODEL)
    c0 = F.fit(start_widths=pw)
    out = {"surface": s, "cnt": F.cnt, "own": F.own, "idx": F.idx, "central": _light(c0)}
    log = lambda what: print(f"[{time.time() - t0:7.1f}s] surface {s}: {what}", flush=True)
    log("central")
    if fit_files:
        Fkb = DR.ShapeFit(P, tables, on, family=MODEL, rotation=False) if s in DISCS else None
        out["T0"] = t0_surface(F, Fkb, planted, rng, n_toy, min(N_TOY_KB, n_toy))
        log("T0")
    out["fits"] = [_light(f) for f in _boot(F, reps, ev, c0)]
    log("bootstrap")
    rs = DR.ShapeFit(P, tables, on, family=SYSTEMATIC).fit(start_widths=pw)
    out["systematic"] = {"rel": {ax: rs[f"R_{ax}"] / c0[f"R_{ax}"] - 1.0 for ax in ("u", "v")}, "converged": rs["converged"]}
    if s in DISCS:
        same = P["facing1"] == P["facing2"]
        fs = {}
        for lab, m_ in (("same", same), ("opposite", ~same)):
            Fs = DR.ShapeFit(P, tables, on & m_, family=MODEL)
            fs[lab] = (Fs, _boot(Fs, reps[:N_BOOT_SUB], ev, Fs.fit(start_widths=pw)))
        fac = {}
        for ax in ("u", "v"):
            dd = np.array([x[f"R_{ax}"] - y[f"R_{ax}"] for x, y in zip(fs["same"][1], fs["opposite"][1])])
            sd = np.nanstd(dd[1:], 0)
            zz = dd[0] / np.where(sd > 0, sd, np.nan)
            jb = _judged(fs["same"][0], ax) & _judged(fs["opposite"][0], ax)
            fac[ax] = {"same": fs["same"][1][0][f"R_{ax}"], "opposite": fs["opposite"][1][0][f"R_{ax}"], "z": zz,
                       "disagree": [int(b) for b in np.flatnonzero(jb) if np.isfinite(zz[b]) and abs(zz[b]) > FACING_Z]}
        out["facing"] = fac
        kb = DR.ShapeFit(P, tables, on, family=MODEL, drop_facing=-1).fit(start_widths=pw)
        out["known_bad_facing_sign_dropped"] = {"B_prime": kb["B_prime"], "R_u": kb["R_u"], "R_v": kb["R_v"]}
        log("facing")
    else:
        wp = {}
        for plo, phi in zip(IL1_P_SPLIT[:-1], IL1_P_SPLIT[1:]):
            Fs = DR.ShapeFit(P, tables, on & (P["p"] >= plo) & (P["p"] < phi), family=MODEL)
            fb = _boot(Fs, reps[:N_BOOT_SUB], ev, Fs.fit(fixed_b=c0["b"], start_widths=pw), fixed_b=c0["b"])
            e = {}
            for ax in ("u", "v"):
                jb = _judged(F, ax) & Fs.own[ax]
                ov = [_pooled(f[f"R_{ax}"], c0[f"R_{ax}"], F.cnt[ax], jb) for f in fb]
                e[ax] = {"pooled_over_all_p": ov[0], "sd": float(np.nanstd(ov[1:]))}
            wp[f"{plo:g}-{phi:g}"] = e
        out["width_vs_p"] = wp
        log("width against p")
    return out


def d2_ratio(P, sel, fits_s, width_fn, edges, sys_rel=None):
    """D2 per axis: the overlap R (and its bootstrap replicas) over the R in use at the pairs' own (size, |cot|)
    cells, pair-weighted per size bin; the error is the bootstrap sd with the shape systematic in quadrature (A8);
    failing bins at 1 +- 0.1 widened by t1(N) x that error."""
    e = {}
    for ax in ("u", "v"):
        sx, cx = ("sx", "cotA") if ax == "u" else ("sy", "cotB")
        size = np.concatenate([P[f"{sx}{j}"][sel] for j in ("1", "2")])
        cot = np.concatenate([P[f"{cx}{j}"][sel] for j in ("1", "2")])
        b = DR.size_bin(size, edges[ax])
        n = np.bincount(b, minlength=len(edges[ax]) - 1)
        ref = np.bincount(b, weights=width_fn(ax, size, cot), minlength=len(edges[ax]) - 1) / np.maximum(n, 1)
        R = np.array([f[f"R_{ax}"] for f in fits_s])
        ratio = R / ref
        sd = np.nanstd(ratio[1:], 0)
        if sys_rel is not None:
            sd = np.hypot(sd, np.nan_to_num(np.abs(sys_rel[ax])) * ratio[0])
        jb = [k for k in range(len(n)) if n[k] >= JUDGE_CLUSTERS and np.isfinite(ratio[0][k])]
        W = G.t_one(max(len(jb), 1))
        fail = [k for k in jb if abs(ratio[0][k] - 1) > D2_TOL + W * sd[k]]
        e[ax] = {"R": R[0], "in_use_at_cells": ref, "ratio": ratio[0], "sd": sd, "n_clusters": n, "W": W, "failing": fail}
    return e


def d2_known_bads(P, sel, fits_s, use, tables, edges, sys_rel):
    """C1b §4, D2's known-bads on file 4 (each must fail D2; the decision-scale one with its power): the
    scattering-contaminated R (the production-helix residual's MAD at pT >= 5 GeV, the round-0 estimator), X / Y
    swapped, half width, R x 1.15, and v3's IL1 tables kept when C1b built disc tables."""
    out = {}
    hi = sel & (P["p"] / np.cosh(P["eta"]) >= 5.0)
    cont = {ax: DR.truth_width(P, tables, hi, ax)[0] for ax in ("u", "v")}
    variants = {
        "contaminated": lambda ax, sz, cot: cont[ax][DR.size_bin(sz, tables[ax]["size_edges"])],
        "xy_swapped": lambda ax, sz, cot: DR.v3_width(use, "v" if ax == "u" else "u", sz, cot),
        "half_width": lambda ax, sz, cot: 0.5 * DR.v3_width(use, ax, sz, cot),
        "x1.15 (power)": lambda ax, sz, cot: 1.15 * DR.v3_width(use, ax, sz, cot)}
    if use is not tables:
        variants["IL1_tables_kept"] = lambda ax, sz, cot: DR.v3_width(tables, ax, sz, cot)
    for k, fn in variants.items():
        e = d2_ratio(P, sel, fits_s, fn, edges, sys_rel)
        nf = sum(len(e[ax]["failing"]) for ax in ("u", "v"))
        out[k] = {"fails": nf > 0, "failing_bins": nf}
    return out


def _pooled(R, ref, n, judged):
    x = R / ref
    m = judged & np.isfinite(x)
    return float(np.sum(n[m] * x[m]) / np.sum(n[m])) if m.any() else np.nan


def fit(a):
    import multiprocessing as mp
    t0 = time.time()
    files = tuple(a.files) if a.files else CM.FIT_FILES
    fit_files = files == CM.FIT_FILES
    d = out_dir(a, files)
    P, C = load(a, files)
    zr = np.load(os.path.join(d, "il1_reference.npz"))
    ref_mask = {"u": zr["u"], "v": zr["v"]}
    with open(os.path.join(d, "il1truth_summary.json")) as fh:
        ref_bins = json.load(fh)["reference"]
    ref_judged = {ax: np.array([r["judged"] for r in ref_bins[ax]]) for ax in ("u", "v")}
    tables = r_tables()
    edges = {ax: tables[ax]["size_edges"] for ax in ("u", "v")}
    surfaces = (IL1,) + DISCS
    n_boot = {s: a.n_boot if s == IL1 else min(a.n_boot, N_BOOT_DISC) for s in surfaces}
    jobs = [{"surface": s, "dir": d, "files": files, "n_boot": n_boot[s], "n_toy": a.n_toy, "fit_files": fit_files}
            for s in (14, 13, 12, 11, IL1)]                       # the largest first
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        W = {o["surface"]: o for o in pool.imap_unordered(surface_work, jobs)}
    print(f"[{time.time() - t0:7.1f}s] surface fits done", flush=True)
    res = {"plan": "c1b/PLAN.md rev 4 §1 item 2, §3 T0 D0 D2, §8 A1-A3, A6, A8, A10-A12", "files": files,
           "role": "fit and decision" if fit_files else ("known-good" if a.known_bad else "gate"), 
           "model": MODEL, "systematic": SYSTEMATIC, "n_boot": n_boot, "n_toy": a.n_toy,
           "n_pairs": {str(s): int((P["surface"] == s).sum()) for s in surfaces}, "material": [], "minor": [],
           "decisions": []}
    for s in surfaces:
        w = W[s]
        if not w["central"]["converged"]:
            res["material"].append(f"surface {s}: the shape fit did not converge")
        if fit_files:
            for shape in TOY_SHAPES:
                bad = [f"{ax}{b}" for ax in ("u", "v") for b in w["T0"][shape][ax]["failing"]]
                if bad:
                    res["material"].append(f"T0 surface {s}, planted {shape}: width not recovered within 3 % in {bad}")
            if s in DISCS and not w["T0"]["known_bad_no_rotation"]["fails"]:
                res["material"].append(f"T0 surface {s}: the no-rotation known-bad passes (the toy cannot see the mixing)")
    res["T0"] = {str(s): W[s]["T0"] for s in surfaces} if fit_files else None
    res["fits"] = {str(s): {"central": W[s]["central"],
                            "R_sd": {ax: np.nanstd([f[f"R_{ax}"] for f in W[s]["fits"][1:]], 0) for ax in ("u", "v")},
                            "replicas_not_converged": int(sum(not f["converged"] for f in W[s]["fits"][1:]))}
                   for s in surfaces}

    # the shape systematic (A8 / A11)
    res["systematic"] = {}
    for s in surfaces:
        rel, e = W[s]["systematic"]["rel"], {"converged": W[s]["systematic"]["converged"]}
        for ax in ("u", "v"):
            jb = W[s]["own"][ax] & (W[s]["cnt"][ax] >= JUDGE_CLUSTERS) & np.isfinite(rel[ax])
            n = W[s]["cnt"][ax]
            mean_abs = float(np.sum(n[jb] * np.abs(rel[ax][jb])) / np.sum(n[jb])) if jb.any() else np.nan
            e[ax] = {"change": rel[ax], "judged": np.flatnonzero(jb), "mean_abs_change": mean_abs}
            if s in DISCS and mean_abs > SYS_TOL:
                res["material"].append(f"shape systematic surface {s} {ax}: mean |change| {mean_abs:.3f} > {SYS_TOL}")
        res["systematic"][str(s)] = e

    # D0: IL1, the method check (A8, A11)
    Wi = W[IL1]
    sel_il1 = np.zeros(len(P["p"]), bool)
    sel_il1[Wi["idx"]] = True
    reps = G.replicas(nev=NEV_FILE * len(files), n=n_boot[IL1])
    res["D0"] = {"width_vs_p": Wi["width_vs_p"]}
    for ax in ("u", "v"):
        jb = Wi["own"][ax] & (Wi["cnt"][ax] >= JUDGE_CLUSTERS) & ref_judged[ax]
        ratios = []
        for k, f in enumerate(Wi["fits"]):
            wp = None if k == 0 else reps[k - 1][P["ev"][sel_il1]]
            pc = DR.pair_cells(P, sel_il1, tables, ax, wp)
            ref, unm = DR.cell_reweighted_width(C, ref_mask[ax], pc, tables, ax, None if k == 0 else reps[k - 1])
            ratios.append(_pooled(f[f"R_{ax}"], ref, Wi["cnt"][ax], jb))
            if k == 0:
                per_bin, ref0, unm0 = f[f"R_{ax}"] / ref, ref, unm
        boot = np.array(ratios[1:])
        lo, hi = np.nanpercentile(boot, [5, 95])
        sd = float(np.nanstd(boot))
        e = {"pooled_ratio": ratios[0], "bounds95": [lo, hi], "sd": sd, "half_width95": 1.645 * sd,
             "powered": 1.645 * sd <= D0_HALF_MAX, "per_bin_ratio": per_bin, "reference_cm": ref0,
             "unmatched_share": unm0, "judged": np.flatnonzero(jb), "n_clusters": Wi["cnt"][ax],
             "pass": bool(D0_BAND[0] <= lo and hi <= D0_BAND[1]),
             "edge_clusters_p30_over_reference": DR.truth_width(P, tables, sel_il1 & (P["p"] >= REF_START_P), ax)[0] / ref0}
        res["D0"][ax] = e
        if not e["powered"]:
            res["decisions"].append(f"D0 {ax}: the IL1 check's 95 % half-width {1.645 * sd:.3f} exceeds {D0_HALF_MAX}: it "
                                    "cannot pass even at a true ratio of 1 (IL1 statistics)")
        elif not e["pass"]:
            (res["material"] if fit_files else res["minor"]).append(
                f"D0 {ax}: IL1 pooled ratio {ratios[0]:.3f} [{lo:.3f}, {hi:.3f}] outside [0.90, 1.10]")

    # D2 per disc; B'; the facing agreement; the facing-sign known-bad; u-v tails on the pairs
    res["D2"] = {}
    disc_fail = False
    for s in DISCS:
        w = W[s]
        sel = np.zeros(len(P["p"]), bool)
        sel[w["idx"]] = True
        e = d2_ratio(P, sel, w["fits"], lambda ax, sz, cot: DR.v3_width(tables, ax, sz, cot), edges, w["systematic"]["rel"])
        disc_fail |= any(e[ax]["failing"] for ax in ("u", "v"))
        if a.known_bad:
            e["known_bads"] = d2_known_bads(P, sel, w["fits"], tables, tables, edges, w["systematic"]["rel"])
        bp = np.array([f["B_prime"] for f in w["fits"]])
        e["B_prime"], e["B_prime_sd"] = bp[0], np.nanstd(bp[1:], 0)
        e["B_prime_ok"] = bool(np.all(np.nan_to_num(bp[0]) < DR.B_PRIME_MAX))
        if not e["B_prime_ok"]:
            res["material"].append(f"D2 surface {s}: B' {np.round(bp[0], 4).tolist()} >= {DR.B_PRIME_MAX} cm GeV")
        e["facing"] = w["facing"]
        for ax in ("u", "v"):
            if w["facing"][ax]["disagree"]:
                res["material"].append(f"D2 surface {s} {ax}: same- vs opposite-facing R disagree in bins "
                                       f"{w['facing'][ax]['disagree']}")
        e["known_bad_facing_sign_dropped"] = w["known_bad_facing_sign_dropped"]
        Q = {k: v[sel] for k, v in P.items()}
        D, _ = DR.differences(Q)
        m0 = ((DR.size_bin(Q["sx1"], edges["u"]) == 0) & (DR.size_bin(Q["sx2"], edges["u"]) == 0)
              & (DR.size_bin(Q["sy1"], edges["v"]) == 0) & (DR.size_bin(Q["sy2"], edges["v"]) == 0))
        if m0.sum() >= PBIN_MIN_N:
            zd = {k: (D[k][m0] - np.median(D[k][m0])) / DR.wmad_width(D[k][m0]) for k in D}
            e["uv_tails"] = tail_independence(zd["phi"], zd["2"])
        res["D2"][str(s)] = e
    res["D2_agrees_with_v3"] = not disc_fail           # report-only (A13): v3's R stays on the discs
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(d, "overlap_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({k: res.get(k) for k in ("material", "minor", "decisions", "D2_agrees_with_v3")}, indent=1,
                     default=str))
    return 0
