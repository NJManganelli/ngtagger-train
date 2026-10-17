"""Exploratory: a smooth seed-covariance correction for the calibrated arm, barrel and endcaps (WP-C, after C1b's J3;
the physicist's design, 2026-10-17). No gates, no pre-registration; the official chain and refit calibration v3 are
untouched. Fit files only (canon_v4 AAAA); files 4 and 6 are never read.

The model: the OT track's reported covariance C plus a correction

    dC(eta, p) = S [ A(eta) + B(eta) / p^2 ] S,          S = diag of typical reported sigmas (dimensionless A, B),

block-diagonal in the transverse helix parameters (rInv, phi0, d0) and the polar ones (tanL, z0). A (flat in p, the
measurement-like part) and B (the scattering inside the OT) are positive semi-definite, written as L L^T with every
Cholesky entry linear in |eta| within a segment; segments are either |eta| ranges whose edges follow the OT stub pattern
("eta") or the number of PS stubs ("ps"), the alternative keying. p = |p| = pT cosh(eta) of the OT track.

The estimator, at each track's FIRST inner-tracker surface with an own cluster (truth only picks the cluster; no update
precedes it, so the history is clean): the seed propagated with the calibrated material between the OT and that surface
(disc-going tracks: C1b's path-ordered material at scale 1), and

    r ~ N(0, J (C + Q + dC) J^T + R),

R the cluster's position error (v3's tables), Q the material: both known and subtracted by construction. One surface
constrains one projection of dC per axis; the next own-cluster surface (still without an update; its extra material
Q12 between the two surfaces known) is added jointly (a 4-dim innovation with the cross-covariance J1 (C+Q+dC) J2^T), so
that z0, tanL and their correlation (and phi0 / d0) separate within a track. The parameters
maximise the clipped Gaussian likelihood sum [log det V + min(r^T V^-1 r, 25)] (the calibration's own form).

Jobs (python seedstudy.py <job>): collect (rows of the fit files), fit (both keyings), slices (free per-slice fits with
an event bootstrap), widths (first-surface core widths: reported / v3 / the models, held-out file), discs (later disc
visits per disc with v3's and the model's seed), plots.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

import arms as AR
import common as CM
import diagnostics as DG
from common import OP, ORP

jax.config.update("jax_enable_x64", True)

OUT = CM.OUT / "seedstudy"
FIT_FILES, CHECK_FILE = (1, 2), 3                   # fit on two fit files, check on a third (all fit files, A10)
STUB_COLS = ("L1TTrackStub_trackIdx", "L1TTrackStub_otStubIdx", "L1TOTStub_isPS", "L1TOTStub_isBarrel")
TRANS, POLAR = (0, 1, 4), (2, 3)                    # helix parameters: rInv, phi0, tanL, z0, d0
CLIP = 25.0
P_EDGES = (2.0, 4.0, 8.0, 16.0, 32.0, np.inf)       # GeV, |p| bins of the width tables
ETA_STEP = 0.15                                     # |eta| slice width of the width tables and the slice fits
FIG_P_BINS = ((2.0, 3.0), (5.0, 6.0), (10.0, 15.0))  # the physicist's figure: |p| rows
N_BOOT_SLICE = 20


# ---------------------------------------------------------------- collection
def stub_features(A, ntrk):
    """Per flattened OT track: the number of PS and 2S stubs and of barrel and disc stubs."""
    nt = ak.to_numpy(ak.num(A["L1TTrack_pt"]))
    off = np.concatenate([[0], np.cumsum(nt)[:-1]])
    ti, si = A["L1TTrackStub_trackIdx"], A["L1TTrackStub_otStubIdx"]
    gi = ak.to_numpy(ak.flatten(ti + ak.Array(off))).astype(np.int64)
    ps = ak.to_numpy(ak.flatten(A["L1TOTStub_isPS"][si])).astype(np.float64)
    br = ak.to_numpy(ak.flatten(A["L1TOTStub_isBarrel"][si])).astype(np.float64)
    out = {"n_ps": np.bincount(gi, ps, ntrk), "n_2s": np.bincount(gi, 1.0 - ps, ntrk),
           "n_barrel_stubs": np.bincount(gi, br, ntrk), "n_disc_stubs": np.bincount(gi, 1.0 - br, ntrk)}
    return {k: v.astype(np.int16) for k, v in out.items()}


def _own_primary(arm, st, j, t, S, ctx, geo, tools, kpt):
    """The own primary cluster per track of t on surface S (the best-ranked module), its innovation r, projection J
    (positions) and cluster error R, at the current (un-updated) state."""
    R = DG.own_rows(arm, st, j, t, S, ctx, geo, tools, kpt)
    if not R:
        return None
    R = DG.mark_primary(R)
    m = np.flatnonzero(R["primary"])
    if not len(m):
        return None
    m = m[np.lexsort((R["rank_chi2"][m], R["v"][m]))]
    m = m[np.r_[True, R["v"][m][1:] != R["v"][m][:-1]]]
    v, ci, mi = R["v"][m], R["ci"][m], R["mi"][m]
    pj = AR.project(arm, st.a[v], st.C[v], S, geo, mi, kpt)
    meas, sig, _ = AR.measurements(ctx, geo, ci, mi, pj["cotA"], pj["cotB"], arm, tools)
    org = geo["origin"][mi]
    return {"v": v, "J": np.asarray(pj["J"])[:, :2, :], "r": np.stack([meas[:, 0] - pj["u"], meas[:, 1] - pj["v"]], 1),
            "R": sig[:, :2], "C": st.C[v].copy(), "zmod": org[:, 2], "rmod": np.hypot(org[:, 0], org[:, 1])}


def first_surface_rows(ctx, geo, cfg, arm, tools, feats, ev0):
    """One chunk: per OT track with a TP, its first two visited surfaces with an own primary cluster, with no update
    anywhere (the seed propagated with the calibrated material): per surface the innovation r, its projection J, the
    cluster error R and the propagated covariance (C + Q); the reported C, v3's deficit and the OT features. The
    second surface is absent (has2 False) when the track has only one."""
    kpt = ORP.pt_constant(ctx["T"])
    tracks = (np.trace(ctx["C"], axis1=1, axis2=2) > 0) & (ctx["trk_tp"] >= 0)
    o = np.flatnonzero(tracks)
    if not len(o):
        return None
    path, seq = AR.visit_plan(ctx, geo, cfg, "AAAA", arm)
    a, Crep = AR.seed(ctx, cfg, arm, tools, o)
    pt = kpt / np.maximum(np.abs(a[:, 0]), 1e-12)
    Dv3 = ORP.seed_cov(tools.cal, Crep, a[:, 2], ctx["n_disk_stubs"][o], ctx["T"]["nStubs"][o], pt) - Crep
    dg = AR.disc_going(path)[o]
    st = AR.State(a.copy(), Crep.copy(), o.copy(), seq[o], dg=dg)
    n = len(o)
    got = np.zeros(n, np.int8)                           # own surfaces recorded so far (0, 1, 2)
    rec = {k: {} for k in (1, 2)}
    for j in range(seq.shape[1]):
        for S in np.unique(st.seq[:, j][(st.seq[:, j] >= 0) & (got < 2)]):
            S = int(S)
            t = np.flatnonzero((st.seq[:, j] == S) & (got < 2))
            AR._q_step(arm, st, t, S, path, geo, tools, kpt, cfg)
            x = _own_primary(arm, st, j, t, S, ctx, geo, tools, kpt)
            if x is None:
                continue
            for k, v in enumerate(x["v"]):
                g = got[v] + 1
                rec[g][int(v)] = {"S": S, "step": j, **{kk: x[kk][k] for kk in ("J", "r", "R", "C", "zmod", "rmod")}}
                got[v] = g
    v1 = np.array(sorted(rec[1]), np.int64)
    if not len(v1):
        return None
    f1 = [rec[1][v] for v in v1]
    out = {"trk": o[v1], "ev": ctx["tev"][o[v1]] + ev0, "dg": dg[v1], "a": a[v1], "Crep": Crep[v1], "Dv3": Dv3[v1],
           "surface": np.array([x["S"] for x in f1], np.int16), "step": np.array([x["step"] for x in f1], np.int16),
           "J": np.stack([x["J"] for x in f1]), "r": np.stack([x["r"] for x in f1]), "R": np.stack([x["R"] for x in f1]),
           "Cq": np.stack([x["C"] for x in f1]), "zmod": np.array([x["zmod"] for x in f1]),
           "rmod": np.array([x["rmod"] for x in f1])}
    has2 = np.array([v in rec[2] for v in v1])
    f2 = [rec[2].get(int(v)) for v in v1]
    out["has2"] = has2
    out["surface2"] = np.array([x["S"] if x else -1 for x in f2], np.int16)
    out["J2"] = np.stack([x["J"] if x else np.zeros((2, 5)) for x in f2])
    out["r2"] = np.stack([x["r"] if x else np.zeros(2) for x in f2])
    out["R2"] = np.stack([x["R"] if x else np.ones(2) for x in f2])
    out["Q12"] = np.stack([x["C"] - y["C"] if x else np.zeros((5, 5)) for x, y in zip(f2, f1)])
    trk = out["trk"]
    tanl = out["a"][:, 2]
    out["eta"] = np.arcsinh(tanl)
    out["pt"] = kpt / np.maximum(np.abs(out["a"][:, 0]), 1e-12)
    out["p"] = out["pt"] * np.sqrt(1.0 + tanl ** 2)
    out["n_stubs"] = ctx["T"]["nStubs"][trk].astype(np.int16)
    out["n_disk"] = ctx["n_disk_stubs"][trk].astype(np.int16)
    for k, x in feats.items():
        out[k] = x[trk]
    return out


def collect(a):
    """Rows of the fit files (FIT_FILES + CHECK_FILE), canon_v4 AAAA; writes seedstudy/rows_fNN.npz."""
    os.makedirs(OUT, exist_ok=True)
    geo = OP.load_geometry(CM.GEOMETRY)
    for f in FIT_FILES + (CHECK_FILE,):
        assert f in CM.FIT_FILES
        t0 = time.time()
        path = CM.input_path("AAAA", f)
        cfg = CM.producer_config(path)
        tools = AR.Tools(cfg, disc=AR.DiscTools())
        arm = AR.inn_disc(cfg, seed_cov="reported")              # the seed as reported; the material as calibrated
        parts = []
        for start, A in CM.iterate(path, AR.branches(path) + list(STUB_COLS), chunk=10, nev=a.nev):
            ctx = AR.prepare(A, geo)
            r = first_surface_rows(ctx, geo, cfg, arm, tools, stub_features(A, ctx["ntrk"]), start)
            if r is not None:
                parts.append(r)
        rows = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
        np.savez_compressed(OUT / f"rows_f{f:02d}.npz", **rows)
        print(f"[{time.time() - t0:6.1f}s] AAAA f{f:02d}: {len(rows['eta'])} tracks", flush=True)
    return 0


def load(files):
    zs = [np.load(OUT / f"rows_f{f:02d}.npz") for f in files]
    out = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    off, ev = 0, []
    for z in zs:                                         # events unique across files
        ev.append(z["ev"] + off)
        off += int(z["ev"].max()) + 1
    out["ev"] = np.concatenate(ev)
    out["aeta"] = np.abs(out["eta"])
    return out


# ---------------------------------------------------------------- the model
def segments_eta(edges):
    return {"kind": "eta", "edges": tuple(edges)}


def segment_index(D, seg):
    if seg["kind"] == "eta":
        return np.clip(np.searchsorted(seg["edges"], D["aeta"], side="right") - 1, 0, len(seg["edges"]) - 2)
    return np.clip(np.searchsorted(seg["edges"], D["n_ps"], side="right") - 1, 0, len(seg["edges"]) - 2)


def segment_centre(D, seg, k):
    if seg["kind"] == "eta":
        e = seg["edges"]
        return 0.5 * (e[k] + min(e[k + 1], 3.0))
    return 1.2                                           # PS-keyed: one smooth curve per class over all |eta|


SCALE = np.array([3e-5, 2e-3, 4e-3, 0.15, 0.03])      # typical reported sigmas: rInv [1/cm], phi0, tanL, z0 [cm], d0 [cm]
N_CHOL = {"T": 6, "L": 3}                           # Cholesky entries: transverse 3x3, polar 2x2
NPAR_SEG = 2 * 2 * (N_CHOL["T"] + N_CHOL["L"])      # (A, B) x (constant, slope in |eta|) x entries = 36


def _chol(e, n):
    L = jnp.zeros(e.shape[:-1] + (n, n))
    idx = np.tril_indices(n)
    return L.at[..., idx[0], idx[1]].set(e)


def _block(e, n):
    L = _chol(e, n)
    return L @ jnp.swapaxes(L, -1, -2)


def dC_of(theta, seg_idx, x, p):
    """The correction per track (n, 5, 5) in helix units: theta (nseg, NPAR_SEG); x = |eta| - segment centre."""
    th = theta[seg_idx]                                  # (n, NPAR_SEG)
    half = NPAR_SEG // 2
    ent = th[:, :half] + th[:, half:] * x[:, None]       # linear in |eta| within the segment
    nT, nL = N_CHOL["T"], N_CHOL["L"]
    aT, aL = ent[:, :nT], ent[:, nT:nT + nL]
    bT, bL = ent[:, nT + nL:2 * nT + nL], ent[:, 2 * nT + nL:]
    inv_p2 = (1.0 / p ** 2)[:, None, None]
    MT = _block(aT, 3) + _block(bT, 3) * inv_p2
    ML = _block(aL, 2) + _block(bL, 2) * inv_p2
    M = jnp.zeros((len(p), 5, 5))
    ti, li = np.array(TRANS), np.array(POLAR)
    M = M.at[:, ti[:, None], ti[None, :]].set(MT)
    M = M.at[:, li[:, None], li[None, :]].set(ML)
    s = jnp.asarray(SCALE)
    return M * s[None, :, None] * s[None, None, :]


def _joint_nll(C1, J1, R1, r1, J2, R2, r2, Q12, w):
    """Clipped Gaussian -2 log L of the two surfaces' innovations (4-dim; tracks without a second surface carry J2 = 0,
    r2 = 0, so their second block is a constant). The clip scales with the dimension: 12.5 per measured coordinate."""
    V11 = jnp.einsum("nij,njk,nlk->nil", J1, C1, J1) + jax.vmap(jnp.diag)(R1 ** 2)
    V22 = jnp.einsum("nij,njk,nlk->nil", J2, C1 + Q12, J2) + jax.vmap(jnp.diag)(R2 ** 2)
    V12 = jnp.einsum("nij,njk,nlk->nil", J1, C1, J2)
    V = jnp.concatenate([jnp.concatenate([V11, V12], 2), jnp.concatenate([jnp.swapaxes(V12, 1, 2), V22], 2)], 1)
    rr = jnp.concatenate([r1, r2], 1)
    sign, logdet = jnp.linalg.slogdet(V)
    q = jnp.einsum("ni,ni->n", rr, jnp.linalg.solve(V, rr[..., None])[..., 0])
    dof = 2.0 + 2.0 * jnp.any(J2 != 0, axis=(1, 2))
    return jnp.sum(w * (logdet + jnp.minimum(q, CLIP / 2.0 * dof)))


def _nll(theta_flat, nseg, seg_idx, x, p, J, Cq, R, r, J2, R2, r2, Q12, w):
    theta = theta_flat.reshape(nseg, NPAR_SEG)
    return _joint_nll(Cq + dC_of(theta, seg_idx, x, p), J, R, r, J2, R2, r2, Q12, w)


_VG = jax.jit(jax.value_and_grad(_nll), static_argnums=(1,))


def _fit_arrays(D, m, seg):
    sidx = segment_index(D, seg)[m]
    cen = np.array([segment_centre(D, seg, k) for k in range(len(seg["edges"]) - 1)])
    x = D["aeta"][m] - cen[sidx]
    return (jnp.asarray(sidx), jnp.asarray(x), jnp.asarray(D["p"][m]),
            *(jnp.asarray(D[k][m]) for k in ("J", "Cq", "R", "r", "J2", "R2", "r2", "Q12")))


def fit_model(D, seg, m=None, w=None, theta0=None):
    """Maximum clipped likelihood of the model on rows m (event weights w). Returns theta (nseg, NPAR_SEG) and -2 log L."""
    m = np.ones(len(D["eta"]), bool) if m is None else m
    nseg = len(seg["edges"]) - 1
    arrs = _fit_arrays(D, m, seg)
    ww = jnp.ones(int(m.sum())) if w is None else jnp.asarray(w[m], jnp.float64)
    if theta0 is None:
        theta0 = np.zeros((nseg, NPAR_SEG))
        half = NPAR_SEG // 2
        diag = {"T": [0, 2, 5], "L": [0, 2]}
        for blk_off, n, dg in ((0, 3, diag["T"]), (N_CHOL["T"], 2, diag["L"]),
                               (N_CHOL["T"] + N_CHOL["L"], 3, diag["T"]), (2 * N_CHOL["T"] + N_CHOL["L"], 2, diag["L"])):
            theta0[:, [blk_off + i for i in dg]] = 0.3
        theta0[:, half:] = 0.0

    def fun(th):
        v, g = _VG(jnp.asarray(th), nseg, *arrs, ww)
        return float(v), np.asarray(g, np.float64)

    r = optimize.minimize(fun, np.asarray(theta0, np.float64).ravel(), jac=True, method="L-BFGS-B",
                          options={"maxiter": 10000, "maxcor": 30})
    return r.x.reshape(nseg, NPAR_SEG), float(r.fun), bool(r.success)


def model_dC(D, seg, theta, m=None):
    m = np.ones(len(D["eta"]), bool) if m is None else m
    sidx, x, p = _fit_arrays(D, m, seg)[:3]
    return np.asarray(dC_of(jnp.asarray(theta), sidx, x, p))


def model_curve(seg, theta, eta, p):
    """dC at given |eta| (array) and |p| (scalar); PS-keyed models need n_ps: not drawn as a curve."""
    D = {"aeta": np.abs(eta), "p": np.full(len(eta), p)}
    sidx = segment_index(D, seg)
    cen = np.array([segment_centre(D, seg, k) for k in range(len(seg["edges"]) - 1)])
    return np.asarray(dC_of(jnp.asarray(theta), jnp.asarray(sidx), jnp.asarray(D["aeta"] - cen[sidx]), jnp.asarray(D["p"])))


# ---------------------------------------------------------------- evaluation
def pulls(D, extra, m=None):
    """Innovation position pulls (n, 2) with the seed covariance Crep + extra (n, 5, 5) and the material."""
    m = np.ones(len(D["eta"]), bool) if m is None else m
    C = D["Cq"][m] + extra
    V = np.einsum("nij,njk,nlk->nil", D["J"][m], C, D["J"][m])
    V[:, 0, 0] += D["R"][m, 0] ** 2
    V[:, 1, 1] += D["R"][m, 1] ** 2
    return D["r"][m] / np.sqrt(np.stack([V[:, 0, 0], V[:, 1, 1]], 1))


def core(x):
    x = x[np.isfinite(x)]
    return (float(np.median(np.abs(x - np.median(x))) / 0.6745), len(x)) if len(x) >= 30 else (np.nan, len(x))


def v3_deficit_at(D, cal, p_mid, m):
    """v3's deficit for rows m with pT set to p_mid / cosh(eta): the bin a track of that |p| would use."""
    pt = p_mid / np.cosh(D["eta"][m])
    zero = np.zeros((int(m.sum()), 5, 5))
    return ORP.seed_cov(cal, zero, D["a"][m, 2], D["n_disk"][m], D["n_stubs"][m], pt)



# ---------------------------------------------------------------- jobs
SEG_ETA = segments_eta((0.0, 0.9, 1.4, 1.75, 2.25, 2.6))       # from the stub pattern (see the stub plot)
SEG_PS = {"kind": "ps", "edges": (0, 3, 4, 5, 99)}             # PS-stub classes: <= 2, 3, 4, >= 5


def nll_per_track(D, extra, m=None):
    m = np.ones(len(D["eta"]), bool) if m is None else m
    args = [jnp.asarray(D[k][m]) for k in ("J", "R", "r", "J2", "R2", "r2", "Q12")]
    C1 = jnp.asarray(D["Cq"][m] + extra)
    return float(_joint_nll(C1, *args, jnp.ones(int(m.sum())))) / int(m.sum())


def fit(a):
    D, H = load(FIT_FILES), load((CHECK_FILE,))
    res = {"fit_files": FIT_FILES, "check_file": CHECK_FILE, "n_fit": len(D["eta"]), "n_check": len(H["eta"]), "models": {}}
    for name, seg in (("eta", SEG_ETA), ("ps", SEG_PS)):
        t0 = time.time()
        th, f2, ok = fit_model(D, seg)
        nh = nll_per_track(H, model_dC(H, seg, th))
        res["models"][name] = {"seg": seg, "theta": th.tolist(), "m2lnL_fit": f2, "converged": ok,
                               "n_par": int(th.size), "check_nll_per_track": nh,
                               "seconds": round(time.time() - t0, 1)}
        print(f"{name}: converged {ok}, {th.size} parameters, -2lnL/track fit {f2 / len(D['eta']):.4f}, "
              f"check {nh:.4f} ({time.time() - t0:.0f}s)", flush=True)
    res["check_nll_per_track"] = {"reported": nll_per_track(H, np.zeros_like(H["Cq"])), "v3": nll_per_track(H, H["Dv3"])}
    res["v3_n_par"] = 75 * 15
    print(json.dumps(res["check_nll_per_track"]))
    with open(OUT / "fit.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return 0


def load_fit():
    with open(OUT / "fit.json") as fh:
        F = json.load(fh)
    return {k: (v["seg"] | {"edges": tuple(v["seg"]["edges"])}, np.array(v["theta"])) for k, v in F["models"].items()}, F


# the slice fits: a constant correction (9 Cholesky entries) per |eta| slice x figure |p| bin
def _dC_const(e):
    MT, ML = _block(e[:6], 3), _block(e[6:], 2)
    M = jnp.zeros((5, 5))
    ti, li = np.array(TRANS), np.array(POLAR)
    M = M.at[ti[:, None], ti[None, :]].set(MT).at[li[:, None], li[None, :]].set(ML)
    s = jnp.asarray(SCALE)
    return M * s[:, None] * s[None, :]


def _nll_const(e, J, Cq, R, r, J2, R2, r2, Q12, w):
    return _joint_nll(Cq + _dC_const(e)[None], J, R, r, J2, R2, r2, Q12, w)


_VG_CONST = jax.jit(jax.value_and_grad(_nll_const))


def fit_const(D, m, w=None, e0=None):
    arrs = tuple(jnp.asarray(D[k][m]) for k in ("J", "Cq", "R", "r", "J2", "R2", "r2", "Q12"))
    ww = jnp.ones(int(m.sum())) if w is None else jnp.asarray(w[m], jnp.float64)
    e0 = np.array([0.3, 0, 0.3, 0, 0, 0.3, 0.3, 0, 0.3]) if e0 is None else e0

    def fun(e):
        v, g = _VG_CONST(jnp.asarray(e), *arrs, ww)
        return float(v), np.asarray(g, np.float64)

    r = optimize.minimize(fun, e0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return r.x


def elements(M):
    """The figure's elements of corrections M (n, 5, 5): sqrt dC(z0) [um], sqrt dC(tanL), corr(z0, tanL), sqrt dC(d0) [um]
    (a negative diagonal gives a negative signed sqrt)."""
    ssq = lambda x: np.sign(x) * np.sqrt(np.abs(x))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = M[:, 2, 3] / np.sqrt(np.abs(M[:, 2, 2] * M[:, 3, 3]))
    return {"z0_um": ssq(M[:, 3, 3]) * 1e4, "tanL": ssq(M[:, 2, 2]), "corr_z0_tanL": corr, "d0_um": ssq(M[:, 4, 4]) * 1e4}


def slices(a):
    import gates as G
    D = load(FIT_FILES)
    nev = int(D["ev"].max()) + 1
    reps = G.replicas(nev=nev, n=N_BOOT_SLICE, seed=20261017)
    edges = np.arange(0.0, 2.55 + 1e-9, ETA_STEP)
    out = []
    t0 = time.time()
    for plo, phi in FIG_P_BINS:
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (D["aeta"] >= lo) & (D["aeta"] < hi) & (D["p"] >= plo) & (D["p"] < phi)
            if m.sum() < 150:
                continue
            e = fit_const(D, m)
            vals = [elements(np.asarray(_dC_const(jnp.asarray(e)))[None])]
            for r in reps:
                vals.append(elements(np.asarray(_dC_const(jnp.asarray(fit_const(D, m, r[D["ev"]].astype(float), e))))[None]))
            row = {"p": [plo, phi], "eta": [lo, hi], "n": int(m.sum())}
            for k in vals[0]:
                x = np.array([v[k][0] for v in vals])
                row[k] = float(x[0])
                row[k + "_sd"] = float(np.nanstd(x[1:]))
            out.append(row)
        print(f"[{time.time() - t0:6.1f}s] p {plo}-{phi}", flush=True)
    with open(OUT / "slices.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


def widths(a):
    """First-surface core widths per |eta| slice x |p| bin on the check file: reported, v3, the two models."""
    H = load((CHECK_FILE,))
    M, _ = load_fit()
    variants = {"reported": np.zeros_like(H["Cq"]), "v3": H["Dv3"]}
    for name, (seg, th) in M.items():
        variants[f"model_{name}"] = model_dC(H, seg, th)
    P = {k: pulls(H, x) for k, x in variants.items()}
    edges = np.arange(0.0, 2.55 + 1e-9, ETA_STEP)
    out = []
    for region, reg in (("barrel", H["surface"] <= 4), ("disc", H["surface"] > 10)):
        for plo, phi in zip(P_EDGES[:-1], P_EDGES[1:]):
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = reg & (H["aeta"] >= lo) & (H["aeta"] < hi) & (H["p"] >= plo) & (H["p"] < phi)
                if m.sum() < 50:
                    continue
                row = {"region": region, "p": [plo, phi], "eta": [lo, hi], "n": int(m.sum())}
                for k, x in P.items():
                    row[k] = [core(x[m, 0])[0], core(x[m, 1])[0]]
                out.append(row)
    summ = {}
    for region, reg in (("barrel", H["surface"] <= 4), ("disc", H["surface"] > 10)):
        summ[region] = {k: [core(x[reg, 0])[0], core(x[reg, 1])[0]] for k, x in P.items()}
        for k in P:                                       # the spread of the per-cell widths around 1 (n-weighted rms)
            cells = [r for r in out if r["region"] == region]
            for ax in (0, 1):
                w_ = np.array([c[k][ax] for c in cells]); n = np.array([c["n"] for c in cells])
                ok = np.isfinite(w_)
                summ[region][f"{k}_rms_dev_{'uv'[ax]}"] = float(np.sqrt(np.sum(n[ok] * (w_[ok] - 1) ** 2) / np.sum(n[ok])))
    with open(OUT / "widths.json", "w") as fh:
        json.dump({"cells": out, "summary": summ}, fh, indent=1)
    print(json.dumps(summ, indent=1))
    return 0


def seed_model_eta(seg, theta):
    """The eta-keyed model as the calibrated arm's seed correction (arms.seed with seed_cov="model")."""
    def f(ctx, a, o):
        tanl = a[:, 2]
        kpt = ORP.pt_constant(ctx["T"])
        p = kpt / np.maximum(np.abs(a[:, 0]), 1e-12) * np.sqrt(1.0 + tanl ** 2)
        Dd = {"aeta": np.abs(np.arcsinh(tanl)), "p": p}
        sidx = segment_index(Dd, seg)
        cen = np.array([segment_centre(Dd, seg, k) for k in range(len(seg["edges"]) - 1)])
        return np.asarray(dC_of(jnp.asarray(theta), jnp.asarray(sidx), jnp.asarray(Dd["aeta"] - cen[sidx]), jnp.asarray(p)))
    return f


def discs(a):
    """The ID1-ID4 pattern before / after on the check file: later disc visits (own primaries, clean history) of the disc
    arm at disc scale 1, v3's seed against the eta-keyed model's; core widths per disc x axis x pT bin, and the barrel's."""
    import run_c1b as RC
    M, _ = load_fit()
    seg, th = M["eta"]
    geo = OP.load_geometry(CM.GEOMETRY)
    path_f = CM.input_path("AAAA", CHECK_FILE)
    cfg = CM.producer_config(path_f)
    res = {}
    for lab, arm, tools in (("v3", AR.inn_disc(cfg), AR.Tools(cfg, disc=AR.DiscTools())),
                            ("model", AR.inn_disc(cfg, seed_cov="model"),
                             AR.Tools(cfg, disc=AR.DiscTools(), seed_model=seed_model_eta(seg, th)))):
        rows = []
        for start, A in CM.iterate(path_f, AR.branches(path_f), chunk=10):
            ctx = AR.prepare(A, geo)
            path, _ = AR.visit_plan(ctx, geo, cfg, "AAAA", arm)
            dg = AR.disc_going(path)
            out = AR.run_arm(ctx, geo, cfg, "AAAA", arm, tools, own=True)
            R = DG.mark_primary(out["own"])
            m = R["primary"] & R["clean"]
            o = R["o"][m]
            rows.append({"surface": R["surface"][m], "step": R["step"][m], "dg": dg[o], "pu": R["pu"][m], "pv": R["pv"][m],
                         "ptbin": np.clip(np.searchsorted(RC.PT_BINS, ctx["T"]["pt"][o], side="right") - 1, 0, 4)})
        Rr = {k: np.concatenate([x[k] for x in rows]) for k in rows[0]}
        e = {}
        for s in (1, 2, 3, 4, 11, 12, 13, 14):
            for first in (True, False):
                mm = (Rr["surface"] == s) & ((Rr["step"] == 0) == first) & (Rr["dg"] if s > 10 else ~Rr["dg"])
                e[f"{s}|{'first' if first else 'later'}"] = {"n": int(mm.sum()), "u": core(Rr["pu"][mm])[0],
                                                              "v": core(Rr["pv"][mm])[0],
                                                              "v_by_pt": [core(Rr["pv"][mm & (Rr["ptbin"] == k)])[0] for k in range(5)]}
        res[lab] = e
        print(lab, {k: (v["n"], round(v["u"], 3), round(v["v"], 3)) for k, v in e.items() if v["n"] >= 200}, flush=True)
    with open(OUT / "discs.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return 0


# ---------------------------------------------------------------- plots
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
C_MODEL, C_V3, C_MEAS, C_PS = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"   # the reference palette's slots 1, 2, 3, 7
ELEMENTS = (("z0_um", "√ΔC(z0) [µm]"), ("tanL", "√ΔC(tanλ)"), ("corr_z0_tanL", "corr(z0, tanλ) of ΔC"),
            ("d0_um", "√ΔC(d0) [µm]"))


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _edges(ax, seg):
    for e in seg["edges"][1:-1]:
        ax.axvline(e, color=INK2, ls="--", lw=0.8)


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight")


def plots(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": INK, "axes.titlesize": 9, "legend.fontsize": 8})
    D = load(FIT_FILES)
    M, F = load_fit()
    seg, th = M["eta"]
    cal = AR.Tools(None, calibration=AR.CALIBRATION, angle_table=None).cal
    with open(OUT / "slices.json") as fh:
        SL = json.load(fh)
    grid = np.linspace(0.0, 2.5, 251)

    # 1. the main figure: rows |p| bins, columns the elements
    fig, axs = plt.subplots(len(FIG_P_BINS), len(ELEMENTS), figsize=(15, 9.5), sharex=True)
    for i, (plo, phi) in enumerate(FIG_P_BINS):
        pm = 0.5 * (plo + phi)
        eta_max = float(np.arccosh(max(phi / 2.0, 1.0)))            # pT >= 2 GeV: the |eta| a track of this |p| reaches
        g = grid[grid <= eta_max]
        mod = elements(model_curve(seg, th, g, pm))
        fine = np.arange(0.0, eta_max + 0.05, 0.05)
        v3x, v3y = [], {k: [] for k, _ in ELEMENTS}
        for lo, hi in zip(fine[:-1], fine[1:]):
            m = (D["aeta"] >= lo) & (D["aeta"] < hi)
            if m.sum() < 20:
                continue
            e3 = elements(v3_deficit_at(D, cal, pm, m).mean(0)[None])
            v3x.append(lo)
            for k, _ in ELEMENTS:
                v3y[k].append(e3[k][0])
        pts = [r for r in SL if r["p"] == [plo, phi]]
        for j, (k, lab) in enumerate(ELEMENTS):
            ax = axs[i, j]
            _style(ax)
            _edges(ax, seg)
            ax.plot(g, mod[k], color=C_MODEL, lw=2, label="smooth model A(η)+B(η)/p², at p_mid")
            if v3x:
                ax.step(v3x + [v3x[-1] + 0.05], v3y[k] + [v3y[k][-1]], where="post", color=C_V3, lw=1.5,
                        label="v3 binned, at pT = p_mid / cosh η")
            if pts:
                x = [0.5 * (r["eta"][0] + r["eta"][1]) for r in pts]
                ax.errorbar(x, [r[k] for r in pts], yerr=[r[k + "_sd"] for r in pts], fmt="o", ms=4, color=C_MEAS,
                            ecolor=C_MEAS, elinewidth=1, capsize=2, label="per-slice fit (two surfaces), bootstrap sd")
            if k == "corr_z0_tanL":
                ax.set_ylim(-1.05, 1.05)
            ax.set_xlim(0, 2.55)
            if i == 0:
                ax.set_title(lab, color=INK)
            if j == 0:
                ax.set_ylabel(f"|p| {plo:g}-{phi:g} GeV\n" + lab)
            else:
                ax.set_ylabel(lab)
            if i == len(FIG_P_BINS) - 1:
                ax.set_xlabel("|η| (OT track)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "dashed: the η segment edges from the OT stub pattern; curves end where pT reaches 2 GeV; "
             "ΔC is the correction added to the OT track's reported covariance", ha="center", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _save(fig, "seed_correction_vs_eta")
    plt.close(fig)

    # 2. the OT stub pattern against |eta|
    e = np.arange(0.0, 2.55, 0.05)
    b = np.digitize(D["aeta"], e) - 1
    xs = e[:-1] + 0.025
    stat = lambda key, f=np.mean: np.array([f(D[key][b == k]) if (b == k).sum() >= 20 else np.nan for k in range(len(e) - 1)])
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for key, lab, c in (("n_ps", "PS stubs", C_MODEL), ("n_2s", "2S stubs", C_V3), ("n_barrel_stubs", "barrel stubs", C_MEAS),
                        ("n_disc_stubs", "disc stubs", C_PS)):
        axs[0].plot(xs, stat(key), color=c, lw=1.8, label=lab)
    axs[0].set_ylabel("stubs per OT track (mean)")
    axs[0].legend(frameon=False)
    sz = np.array([np.median(np.sqrt(D["Crep"][b == k, 3, 3])) * 1e4 if (b == k).sum() >= 20 else np.nan for k in range(len(e) - 1)])
    stl = np.array([np.median(np.sqrt(D["Crep"][b == k, 2, 2])) if (b == k).sum() >= 20 else np.nan for k in range(len(e) - 1)])
    axs[1].plot(xs, sz, color=C_MODEL, lw=1.8)
    axs[1].set_ylabel("OT fit σ(z0), median [µm]")
    axs[2].plot(xs, stl, color=C_MODEL, lw=1.8)
    axs[2].set_ylabel("OT fit σ(tanλ), median")
    for cls, lab, c in (((0, 3), "≤ 2 PS", C_MODEL), ((3, 4), "3 PS", C_V3), ((4, 5), "4 PS", C_MEAS), ((5, 99), "≥ 5 PS", C_PS)):
        fr = np.array([np.mean((D["n_ps"][b == k] >= cls[0]) & (D["n_ps"][b == k] < cls[1])) if (b == k).sum() >= 20 else np.nan
                       for k in range(len(e) - 1)])
        axs[3].plot(xs, fr, color=c, lw=1.8, label=lab)
    axs[3].set_ylabel("fraction of OT tracks")
    axs[3].legend(frameon=False)
    for ax in axs:
        _style(ax)
        _edges(ax, seg)
        ax.set_xlabel("|η| (OT track)")
    fig.suptitle("OT stub pattern against |η| (canon_v4 AAAA, fit files 1-2); dashed: the chosen segment edges", color=INK)
    fig.tight_layout()
    _save(fig, "stub_pattern_vs_eta")
    plt.close(fig)

    # 3. A(eta) and B(eta): sqrt of the diagonal, both keyings' eta model only (the PS model has no single curve)
    names = (("rInv", 0, 1e0, "1/cm"), ("phi0", 1, 1e0, "rad"), ("tanλ", 2, 1e0, ""), ("z0", 3, 1e4, "µm"), ("d0", 4, 1e4, "µm"))
    fig, axs = plt.subplots(2, 5, figsize=(17, 6), sharex=True)
    big = 1e6
    MA = model_curve(seg, th, grid, big)                 # p -> infinity: A alone
    MB = (model_curve(seg, th, grid, 1.0) - MA)          # p = 1 GeV: A + B; B = the difference (GeV^2 units)
    for j, (nm, k, sc, un) in enumerate(names):
        for i, (Mx, lab) in enumerate(((MA, "√A"), (MB, "√B [× GeV]"))):
            ax = axs[i, j]
            _style(ax)
            _edges(ax, seg)
            ax.plot(grid, np.sqrt(np.maximum(Mx[:, k, k], 0)) * sc, color=C_MODEL, lw=2)
            ax.set_ylabel(f"{lab} ({nm}){' [' + un + ']' if un else ''}")
            if i == 1:
                ax.set_xlabel("|η| (OT track)")
    fig.suptitle("The fitted smooth correction: A(η) (flat in p) and B(η) (× 1/p²), square roots of the diagonal", color=INK)
    fig.tight_layout()
    _save(fig, "seed_A_B_vs_eta")
    plt.close(fig)

    # 4. first-surface core widths, held-out file: reported / v3 / the eta model, per |p| bin
    with open(OUT / "widths.json") as fh:
        W = json.load(fh)["cells"]
    pbins = [list(x) for x in zip(P_EDGES[:-1], P_EDGES[1:]) if any(c["p"][0] == x[0] for c in W)]
    fig, axs = plt.subplots(len(pbins), 4, figsize=(16, 2.4 * len(pbins)), sharex=True, sharey=True)
    for i, pb in enumerate(pbins):
        for j, (region, ax_i, axn) in enumerate((("barrel", 0, "u"), ("barrel", 1, "v"), ("disc", 0, "u"), ("disc", 1, "v"))):
            ax = axs[i, j]
            _style(ax)
            _edges(ax, seg)
            ax.axhline(1.0, color=INK2, lw=0.8)
            cells = [c for c in W if c["region"] == region and c["p"][0] == pb[0]]
            x = [0.5 * (c["eta"][0] + c["eta"][1]) for c in cells]
            for key, c, lab in (("reported", GRID, "reported"), ("v3", C_V3, "v3 binned"), ("model_eta", C_MODEL, "smooth model (η segments)"),
                                ("model_ps", C_PS, "smooth model (PS classes)")):
                ax.plot(x, [cc[key][ax_i] for cc in cells], "o-", color=c if key != "reported" else INK2, ms=3, lw=1.2,
                        alpha=1.0 if key != "reported" else 0.5, label=lab)
            ax.set_ylim(0.5, 1.6)
            if i == 0:
                ax.set_title(f"{region} first surface, {axn}")
            if j == 0:
                ax.set_ylabel(f"|p| {pb[0]:g}-{pb[1]:g} GeV\ncore width")
            if i == len(pbins) - 1:
                ax.set_xlabel("|η| (OT track)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, f"held-out fit file {CHECK_FILE}; core width = MAD / 0.6745 of the innovation pull", ha="center",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(fig, "first_surface_widths")
    plt.close(fig)

    # 5. the ID1-ID4 pattern before / after (later visits; the barrel and first visits beside)
    with open(OUT / "discs.json") as fh:
        DS = json.load(fh)
    keys = [k for k in ("4|first", "3|later", "2|later", "1|later", "14|first", "13|first", "12|first", "11|first",
                        "14|later", "13|later", "12|later", "11|later") if DS["v3"][k]["n"] >= 200]
    lab = {f"{s}|{w}": f"{'IL' if s <= 4 else 'ID'}{s if s <= 4 else s - 10} {w}" for s in (1, 2, 3, 4, 11, 12, 13, 14)
           for w in ("first", "later")}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, axn in zip(axs, ("u", "v")):
        _style(ax)
        ax.axhline(1.0, color=INK2, lw=0.8)
        xs_ = np.arange(len(keys))
        ax.plot(xs_ - 0.1, [DS["v3"][k][axn] for k in keys], "o", color=C_V3, ms=7, label="v3 binned seed")
        ax.plot(xs_ + 0.1, [DS["model"][k][axn] for k in keys], "s", color=C_MODEL, ms=7, label="smooth model seed")
        ax.set_xticks(xs_, [lab[k] for k in keys], rotation=45, ha="right")
        ax.set_title(f"{axn}: pull core width (calibrated disc arm, scale 1, file {CHECK_FILE})")
    axs[0].set_ylabel("core width")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "disc_pattern_before_after")
    plt.close(fig)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("collect", "fit", "slices", "widths", "discs", "plots"))
    ap.add_argument("--nev", type=int, default=None)
    a = ap.parse_args(argv)
    return {"collect": collect, "fit": fit, "slices": slices, "widths": widths, "discs": discs, "plots": plots}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
