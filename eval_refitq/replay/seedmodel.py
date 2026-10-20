"""The calibrated arms' seed covariance correction: the two-surface smooth model (c1b/PLAN.md §8 A16), barrel and discs.

The OT track's reported covariance C plus

    dC(eta, p) = S [ A(eta) + B(eta) / p_eff^2 ] S,          S = diag of typical reported sigmas,

block-diagonal in the transverse helix parameters (rInv, phi0, d0) and the polar ones (tanL, z0). A (flat in p) and B
(the scattering inside the OT) are each D R D per block: D = diag(exp l_i), R the correlation matrix from partial
correlations tanh(x_j) (positive definite by construction); every l_i and x_j is linear in |eta| within a segment. The
segments are |eta| ranges whose edges follow the OT stub pattern ("eta") or PS-stub classes ("ps"); p = |p| of the OT
track, p_eff = p clipped into the populated |p| range of the track's |eta| slice (the minimum-count rule: no free
extrapolation into unpopulated corners).

The estimator, per OT track whose TP is prompt: its first two own-cluster surfaces with no update (truth only picks the
cluster), the seed propagated with the calibrated material between the OT and them (C1b's path-ordered material for
disc-going tracks, scale 1), v3's cluster errors R; the joint innovation r = (r1, r2) ~ N(0, V) with
V11 = J1 (C + Q + dC) J1^T + R1, V12 = J1 (C + Q + dC) J2^T, V22 = J2 (C + Q + Q12 + dC) J2^T + R2, maximising the
clipped Gaussian likelihood sum [log det V + min(r^T V^-1 r, 12.5 per coordinate)].

Jobs (python seedmodel.py <job>): collect (rows per sample, file and scenario), select (both keyings on the selection
files, held-out -2 log L on files 9-10, the H->tautau check), final (the chosen keying on all fit files, + H->tautau if it
passed; writes the model file), slices (per-slice two-surface fits, event bootstrap), plots.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

import arms as AR
import common as CM
import diagnostics as DG
from common import OP, ORP

jax.config.update("jax_enable_x64", True)

OUT = CM.OUT / "seedmodel"
MODEL_FILE = OUT / "seed_model_v1.json"              # the smooth seed the arms load (Tools.seed_model)
HTT = CM.CANON / "htt"
HTT_FILES = (1, 2, 3, 4, 5, 7, 8)                   # canon_v4's H->tautau files (no file role in WP-C)
SELECT_FIT, SELECT_HELD = (1, 2, 3, 5, 7, 8), (9, 10)  # A16: the keying is chosen on held-out files 9-10
TRANS, POLAR = (0, 1, 4), (2, 3)                    # helix parameters: rInv, phi0, tanL, z0, d0
SCALE = np.array([3e-5, 2e-3, 4e-3, 0.15, 0.03])     # typical reported sigmas: rInv [1/cm], phi0, tanL, z0 [cm], d0 [cm]
PROMPT_D0_CM = 0.01
CLIP_PER_DIM = 12.5
SEG_ETA = {"kind": "eta", "edges": (0.0, 0.9, 1.4, 1.75, 2.25, 2.6)}
SEG_PS = {"kind": "ps", "edges": (0, 3, 4, 5, 99)}
ETA_CELL = 0.15
P_CELL = (2.0, 3.0, 4.5, 7.0, 11.0, 17.0, 26.0, 40.0, np.inf)
N_MIN = 200
N_PER_BLOCK = {"T": (3, 3), "L": (2, 1)}             # (log sigmas, partial correlations)
NPAR_SEG = 2 * 2 * sum(a + b for a, b in N_PER_BLOCK.values())   # (A, B) x (constant, slope) x 9 = 36
MAXITER = 20000
FIG_P_BINS = ((2.0, 3.0), (5.0, 6.0), (10.0, 15.0))
N_BOOT_SLICE = 20
NEV_FILE = 1000                                      # the event-identity stride per file (H->tautau files hold > 100)


# ---------------------------------------------------------------- collection
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
    return {"v": v, "J": np.asarray(pj["J"])[:, :2, :], "r": np.stack([meas[:, 0] - pj["u"], meas[:, 1] - pj["v"]], 1),
            "R": sig[:, :2], "C": st.C[v].copy()}


def two_surface_rows(ctx, geo, cfg, scen, arm, tools, ev_id):
    """One chunk: per OT track whose TP is prompt, its first two visited surfaces with an own primary cluster, with no
    update anywhere: per surface the innovation r, its projection J, the cluster error R and the propagated covariance;
    the reported C, v3's deficit and the OT features. has2 False when the track has a single own surface."""
    kpt = ORP.pt_constant(ctx["T"])
    tp = ctx["trk_tp"]
    ok = (np.trace(ctx["C"], axis1=1, axis2=2) > 0) & (tp >= 0)
    ok[ok] &= np.abs(ctx["TP"]["d0"][tp[ok]]) < PROMPT_D0_CM
    o = np.flatnonzero(ok)
    if not len(o):
        return None
    path, seq = AR.visit_plan(ctx, geo, cfg, scen, arm)
    a, Crep = AR.seed(ctx, cfg, arm, tools, o)
    pt = kpt / np.maximum(np.abs(a[:, 0]), 1e-12)
    Dv3 = ORP.seed_cov(tools.cal, Crep, a[:, 2], ctx["n_disk_stubs"][o], ctx["T"]["nStubs"][o], pt) - Crep
    dg = AR.disc_going(path)[o]
    st = AR.State(a.copy(), Crep.copy(), o.copy(), seq[o], dg=dg)
    got = np.zeros(len(o), np.int8)
    rec = {1: {}, 2: {}}
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
                rec[g][int(v)] = {"S": S, **{kk: x[kk][k] for kk in ("J", "r", "R", "C")}}
                got[v] = g
    v1 = np.array(sorted(rec[1]), np.int64)
    if not len(v1):
        return None
    f1 = [rec[1][v] for v in v1]
    f2 = [rec[2].get(int(v)) for v in v1]
    trk = o[v1]
    tanl = a[v1, 2]
    out = {"ev": ev_id[ctx["tev"][trk]], "scen": np.full(len(v1), CM.SCENARIOS.index(scen), np.int8), "dg": dg[v1],
           "a": a[v1], "Crep": Crep[v1], "Dv3": Dv3[v1],
           "surface": np.array([x["S"] for x in f1], np.int16), "J": np.stack([x["J"] for x in f1]),
           "r": np.stack([x["r"] for x in f1]), "R": np.stack([x["R"] for x in f1]), "Cq": np.stack([x["C"] for x in f1]),
           "has2": np.array([x is not None for x in f2]), "surface2": np.array([x["S"] if x else -1 for x in f2], np.int16),
           "J2": np.stack([x["J"] if x else np.zeros((2, 5)) for x in f2]),
           "r2": np.stack([x["r"] if x else np.zeros(2) for x in f2]),
           "R2": np.stack([x["R"] if x else np.ones(2) for x in f2]),
           "Q12": np.stack([x["C"] - y["C"] if x else np.zeros((5, 5)) for x, y in zip(f2, f1)]),
           "eta": np.arcsinh(tanl), "pt": pt[v1], "p": pt[v1] * np.sqrt(1.0 + tanl ** 2),
           "n_stubs": ctx["T"]["nStubs"][trk].astype(np.int16), "n_disk": ctx["n_disk_stubs"][trk].astype(np.int16)}
    for k in ("n_ps", "n_2s", "n_barrel_stubs", "n_disc_stubs"):
        out[k] = ctx[k][trk]
    return out


def sample_path(sample, scen, f):
    return CM.input_path(scen, f) if sample == "ttbar" else HTT / f"spstudy_{scen}_htt_f{f:02d}.root"


def rows_file(sample, scen, f):
    return OUT / "rows" / f"{sample}_{scen}_f{f:02d}.npz"


def collect(a):
    """Rows per sample x scenario x file (TTbar: the fit files; H->tautau: its files), canon_v4."""
    os.makedirs(OUT / "rows", exist_ok=True)
    geo = OP.load_geometry(CM.GEOMETRY)
    todo = [("ttbar", f) for f in CM.FIT_FILES] + [("htt", f) for f in HTT_FILES]
    for sample, f in todo:
        assert sample == "htt" or f in CM.FIT_FILES
        for scen in a.scen:
            t0 = time.time()
            path = sample_path(sample, scen, f)
            cfg = CM.producer_config(path)
            assert CM.mask_of(cfg) == scen
            ev_id = (100 * (sample == "htt") + f) * NEV_FILE + CM.event_rank(path)     # identity across scenarios
            tools = AR.Tools(cfg, disc=AR.DiscTools())
            arm = AR.inn_disc(cfg, seed_cov="reported")              # the seed as reported; the material as calibrated
            parts = []
            for start, A in CM.iterate(path, AR.branches(path), chunk=10, nev=a.nev):
                ctx = AR.prepare(A, geo)
                r = two_surface_rows(ctx, geo, cfg, scen, arm, tools, ev_id[start:])
                if r is not None:
                    parts.append(r)
            rows = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
            np.savez_compressed(rows_file(sample, scen, f), **rows)
            print(f"[{time.time() - t0:6.1f}s] {sample} {scen} f{f:02d}: {len(rows['eta'])} tracks", flush=True)
    return 0


def load(sample, files, scens=CM.SCENARIOS):
    zs = [np.load(rows_file(sample, s, f)) for f in files for s in scens]
    out = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    out["aeta"] = np.abs(out["eta"])
    return out


def merge(*Ds):
    return {k: np.concatenate([D[k] for D in Ds]) for k in Ds[0]}


# ---------------------------------------------------------------- the populated domain (the minimum-count rule)
def domain(D, n_min=N_MIN):
    """Per |eta| cell (ETA_CELL wide): the populated |p| range [p_lo, p_hi] (cells with >= n_min two-surface AAAA
    tracks, one copy of each OT track); the last populated |eta| cell."""
    m = D["has2"] & (D["scen"] == CM.SCENARIOS.index("AAAA"))
    ne = int(np.ceil(2.6 / ETA_CELL))
    eb = np.minimum((D["aeta"][m] / ETA_CELL).astype(int), ne - 1)
    pb = np.clip(np.searchsorted(P_CELL, D["p"][m], side="right") - 1, 0, len(P_CELL) - 2)
    cnt = np.zeros((ne, len(P_CELL) - 1), int)
    np.add.at(cnt, (eb, pb), 1)
    lo, hi = np.full(ne, np.nan), np.full(ne, np.nan)
    for i in range(ne):
        ok = np.flatnonzero(cnt[i] >= n_min)
        if len(ok):
            lo[i], hi[i] = P_CELL[ok[0]], P_CELL[ok[-1] + 1]
    last = int(np.flatnonzero(np.isfinite(lo))[-1])
    for i in range(ne):                                  # an unpopulated |eta| cell takes its nearest populated one's
        if not np.isfinite(lo[i]):
            j = min(np.flatnonzero(np.isfinite(lo)), key=lambda k: abs(k - i))
            lo[i], hi[i] = lo[j], hi[j]
    return {"eta_cell": ETA_CELL, "p_lo": lo.tolist(), "p_hi": hi.tolist(), "eta_last": (last + 1) * ETA_CELL,
            "counts": cnt.tolist(), "p_cells": [float(x) for x in P_CELL], "n_min": n_min}


def p_eff(dom, aeta, p):
    """p clipped into the populated |p| range of the track's |eta| cell, and whether the track was outside."""
    i = np.minimum((np.minimum(aeta, dom["eta_last"] - 1e-9) / dom["eta_cell"]).astype(int), len(dom["p_lo"]) - 1)
    lo, hi = np.asarray(dom["p_lo"])[i], np.asarray(dom["p_hi"])[i]
    pe = np.clip(p, lo, hi)
    return pe, (pe != p) | (aeta >= dom["eta_last"])


# ---------------------------------------------------------------- the model
def segment_index(seg, aeta, n_ps):
    x = aeta if seg["kind"] == "eta" else n_ps
    return np.clip(np.searchsorted(seg["edges"], x, side="right") - 1, 0, len(seg["edges"]) - 2)


def segment_x(seg, aeta, sidx):
    """|eta| relative to the segment's centre (PS classes: one line per class over all |eta|, centred at 1.2)."""
    if seg["kind"] == "ps":
        return aeta - 1.2
    e = np.asarray(seg["edges"], np.float64)
    return aeta - 0.5 * (e[sidx] + np.minimum(e[sidx + 1], 2.6))


def _corr(z, n):
    """The correlation matrix of partial correlations z (..., n(n-1)/2), positive definite."""
    if n == 2:
        one = jnp.ones_like(z[..., 0])
        return jnp.stack([jnp.stack([one, z[..., 0]], -1), jnp.stack([z[..., 0], one], -1)], -2)
    z12, z13, z23 = z[..., 0], z[..., 1], z[..., 2]
    zero, one = jnp.zeros_like(z12), jnp.ones_like(z12)
    L = jnp.stack([jnp.stack([one, zero, zero], -1),
                   jnp.stack([z12, jnp.sqrt(1 - z12 ** 2), zero], -1),
                   jnp.stack([z13, z23 * jnp.sqrt(1 - z13 ** 2), jnp.sqrt((1 - z13 ** 2) * (1 - z23 ** 2))], -1)], -2)
    return L @ jnp.swapaxes(L, -1, -2)


Z_MAX, L_MAX = 0.999, 30.0                          # |partial correlation| and |log sigma| bounds: finite gradients


def _block_cov(ent, nl, nz):
    s = jnp.exp(jnp.clip(ent[..., :nl], -L_MAX, L_MAX))
    R = _corr(Z_MAX * jnp.tanh(ent[..., nl:nl + nz]), nl)
    return s[..., :, None] * R * s[..., None, :]


def dC_of(theta, sidx, x, pe):
    """The correction per track (n, 5, 5) in helix units; theta (nseg, NPAR_SEG)."""
    th = theta[sidx]
    half = NPAR_SEG // 2
    ent = th[:, :half] + th[:, half:] * x[:, None]
    (tl, tz), (ll, lz) = N_PER_BLOCK["T"], N_PER_BLOCK["L"]
    nb = tl + tz + ll + lz
    blocks = {}
    for i, ab in enumerate(("A", "B")):
        e = ent[:, i * nb:(i + 1) * nb]
        blocks[ab] = (_block_cov(e[:, :tl + tz], tl, tz), _block_cov(e[:, tl + tz:], ll, lz))
    inv_p2 = (1.0 / pe ** 2)[:, None, None]
    MT = blocks["A"][0] + blocks["B"][0] * inv_p2
    ML = blocks["A"][1] + blocks["B"][1] * inv_p2
    M = jnp.zeros((len(pe), 5, 5))
    ti, li = np.array(TRANS), np.array(POLAR)
    M = M.at[:, ti[:, None], ti[None, :]].set(MT).at[:, li[:, None], li[None, :]].set(ML)
    s = jnp.asarray(SCALE)
    return M * s[None, :, None] * s[None, None, :]


def _inv2(M):
    det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    inv = jnp.stack([jnp.stack([M[:, 1, 1], -M[:, 0, 1]], -1), jnp.stack([-M[:, 1, 0], M[:, 0, 0]], -1)], -2)
    return inv / det[:, None, None], det


def joint_nll(C1, J1, R1, r1, J2, R2, r2, Q12, w):
    """Clipped Gaussian -2 log L of the two surfaces' innovations, through 2x2 blocks (a track without a second
    surface carries J2 = 0, r2 = 0, R2 = 1: a constant second block)."""
    V11 = jnp.einsum("nij,njk,nlk->nil", J1, C1, J1) + jax.vmap(jnp.diag)(R1 ** 2)
    V12 = jnp.einsum("nij,njk,nlk->nil", J1, C1, J2)
    V22 = jnp.einsum("nij,njk,nlk->nil", J2, C1 + Q12, J2) + jax.vmap(jnp.diag)(R2 ** 2)
    I11, d11 = _inv2(V11)
    Wm = jnp.einsum("nji,njk->nik", V12, I11)                    # V21 V11^-1
    Sc = V22 - jnp.einsum("nij,njk->nik", Wm, V12)
    Isc, dsc = _inv2(Sc)
    e = r2 - jnp.einsum("nij,nj->ni", Wm, r1)
    q = jnp.einsum("ni,nij,nj->n", r1, I11, r1) + jnp.einsum("ni,nij,nj->n", e, Isc, e)
    dof = 2.0 + 2.0 * jnp.any(J2 != 0, axis=(1, 2))
    return jnp.sum(w * (jnp.log(jnp.maximum(d11, 1e-300)) + jnp.log(jnp.maximum(dsc, 1e-300))
                        + jnp.minimum(q, CLIP_PER_DIM * dof)))


def _nll(theta_flat, nseg, sidx, x, pe, J, Cq, R, r, J2, R2, r2, Q12, w):
    return joint_nll(Cq + dC_of(theta_flat.reshape(nseg, NPAR_SEG), sidx, x, pe), J, R, r, J2, R2, r2, Q12, w)


_VG = jax.jit(jax.value_and_grad(_nll), static_argnums=(1,))


def model_inputs(D, seg, dom, m=None):
    m = np.ones(len(D["eta"]), bool) if m is None else m
    sidx = segment_index(seg, D["aeta"][m], D["n_ps"][m])
    x = segment_x(seg, D["aeta"][m], sidx)
    pe, _ = p_eff(dom, D["aeta"][m], D["p"][m])
    return sidx, x, pe


def start_theta(nseg):
    th = np.zeros((nseg, NPAR_SEG))
    (tl, tz), (ll, lz) = N_PER_BLOCK["T"], N_PER_BLOCK["L"]
    nb = tl + tz + ll + lz
    for i, l0 in enumerate((np.log(0.5), np.log(0.5 * 5.0))):          # B starts as A at p = 5 GeV
        th[:, i * nb + np.r_[0:tl, tl + tz:tl + tz + ll]] = l0
    return th


def fit_model(D, seg, dom, m=None, w=None, theta0=None):
    """Maximum clipped likelihood on rows m (weights w). Returns theta, -2 log L, converged, iterations, message."""
    m = np.ones(len(D["eta"]), bool) if m is None else m
    nseg = len(seg["edges"]) - 1
    sidx, x, pe = model_inputs(D, seg, dom, m)
    arrs = (jnp.asarray(sidx), jnp.asarray(x), jnp.asarray(pe),
            *(jnp.asarray(D[k][m]) for k in ("J", "Cq", "R", "r", "J2", "R2", "r2", "Q12")))
    ww = jnp.ones(int(m.sum())) if w is None else jnp.asarray(w[m], jnp.float64)
    th0 = start_theta(nseg) if theta0 is None else np.asarray(theta0)

    def fun(t):
        v, g = _VG(jnp.asarray(t), nseg, *arrs, ww)
        return float(v), np.asarray(g, np.float64)

    r = optimize.minimize(fun, th0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": MAXITER, "maxcor": 30})
    return r.x.reshape(nseg, NPAR_SEG), float(r.fun), bool(r.success), int(r.nit), str(r.message)


def nll_per_track(D, extra, m=None):
    m = np.ones(len(D["eta"]), bool) if m is None else m
    args = [jnp.asarray(D[k][m]) for k in ("J", "R", "r", "J2", "R2", "r2", "Q12")]
    return float(joint_nll(jnp.asarray(D["Cq"][m] + extra), *args, jnp.ones(int(m.sum())))) / int(m.sum())


def model_dC(D, seg, dom, theta, m=None):
    return np.asarray(dC_of(jnp.asarray(theta), *(jnp.asarray(v) for v in model_inputs(D, seg, dom, m))))


class SeedModel:
    """The model file: the arms' seed correction (arms.seed with seed_cov="smooth")."""

    def __init__(self, path=MODEL_FILE):
        with open(path) as fh:
            self.doc = json.load(fh)
        self.path = str(path)
        self.seg = {**self.doc["segments"], "edges": tuple(self.doc["segments"]["edges"])}
        self.dom = self.doc["domain"]
        self.theta = np.asarray(self.doc["theta"])

    def dC(self, ctx, a, o):
        tanl = a[:, 2]
        kpt = CM.pt_constant(ctx)
        p = kpt / np.maximum(np.abs(a[:, 0]), 1e-12) * np.sqrt(1.0 + tanl ** 2)
        aeta = np.abs(np.arcsinh(tanl))
        n_ps = ctx["n_ps"][o] if self.seg["kind"] == "ps" else np.zeros(len(o), int)
        sidx = segment_index(self.seg, aeta, n_ps)
        pe, _ = p_eff(self.dom, aeta, p)
        return np.asarray(dC_of(jnp.asarray(self.theta), jnp.asarray(sidx), jnp.asarray(segment_x(self.seg, aeta, sidx)),
                                jnp.asarray(pe)))

    def curve(self, aeta, p, n_ps):
        """dC at |eta| and n_ps (arrays; the PS keying's class) and |p| (scalar); with the out-of-domain flag."""
        pe, out = p_eff(self.dom, aeta, np.full(len(aeta), p))
        sidx = segment_index(self.seg, aeta, n_ps)
        return np.asarray(dC_of(jnp.asarray(self.theta), jnp.asarray(sidx), jnp.asarray(segment_x(self.seg, aeta, sidx)),
                                jnp.asarray(pe))), out


# ---------------------------------------------------------------- A17's width
G25 = 0.9999988920803029                             # E[min(z^2, 25)] of a unit Gaussian


def var_width(z):
    z = z[np.isfinite(z)]
    return float(np.sqrt(np.mean(np.minimum(z * z, 25.0)) / G25)) if len(z) >= 30 else np.nan


S_GRID = np.arange(0.4, 3.0 + 1e-9, 0.002)           # A18: the clipped-likelihood scale's grid


def clip_scale(z, ev, reps):
    """A18: per pull sample z (events ev), the scale s minimising sum [2 log s + min(z^2 / s^2, 25)] (the 1-dim clipped
    Gaussian likelihood; a unit Gaussian gives 1), on S_GRID with a parabolic refinement, and its event-bootstrap sd
    (reps: replica weights per event). Per event and grid point the sums are cumulative over |z|, so every replica is
    one matrix product."""
    ok = np.isfinite(z)
    z, ev = np.abs(z[ok]), ev[ok].astype(np.int64)
    if len(z) < 30:
        return np.nan, np.nan
    nev = reps.shape[1]
    k0 = np.searchsorted(S_GRID, z / 5.0, side="right")          # the first grid point where |z| < 5 s
    G = len(S_GRID)
    H2, H1 = np.zeros((G + 1, nev)), np.zeros((G + 1, nev))
    np.add.at(H2, (k0, ev), z * z)
    np.add.at(H1, (k0, ev), 1.0)
    T, Cn = np.cumsum(H2, 0)[:G], np.cumsum(H1, 0)[:G]          # per grid point and event: sum z^2, count, |z| < 5 s
    Ne = np.bincount(ev, minlength=nev).astype(float)
    s2 = (S_GRID ** 2)[:, None]

    def best(w):
        f = (2.0 * np.log(S_GRID)) * (w @ Ne) + (T / s2 + 25.0 * (Ne[None, :] - Cn)) @ w
        i = int(np.clip(np.argmin(f), 1, G - 2))
        a, b, c = f[i - 1], f[i], f[i + 1]
        d = a - 2 * b + c
        return float(S_GRID[i] + (0.5 * (a - c) / d * (S_GRID[1] - S_GRID[0]) if d > 0 else 0.0))

    return best(np.ones(nev)), float(np.std([best(w.astype(float)) for w in reps]))


def pulls_first(D, extra):
    C = D["Cq"] + extra
    V = np.einsum("nij,njk,nlk->nil", D["J"], C, D["J"])
    return D["r"] / np.sqrt(np.stack([V[:, 0, 0] + D["R"][:, 0] ** 2, V[:, 1, 1] + D["R"][:, 1] ** 2], 1))


def region_widths(D, extra):
    P = pulls_first(D, extra)
    return {f"{reg}_{ax}": var_width(P[m, i]) for reg, m in (("barrel", D["surface"] <= 4), ("disc", D["surface"] > 10))
            for i, ax in enumerate(("u", "v"))}


# ---------------------------------------------------------------- jobs
def select(a):
    """A16: both keyings on the selection files, -2 log L per track on files 9-10; the H->tautau check with the chosen
    keying's selection-file model. Writes seedmodel/select.json."""
    Dfit, Dheld = load("ttbar", SELECT_FIT), load("ttbar", SELECT_HELD)
    dom = domain(Dfit, a.n_min)
    res = {"fit_files": SELECT_FIT, "held_files": SELECT_HELD, "n_fit": len(Dfit["eta"]), "n_held": len(Dheld["eta"]),
           "keyings": {}}
    for name, seg in (("eta", SEG_ETA), ("ps", SEG_PS)):
        t0 = time.time()
        th, f2, ok, nit, msg = fit_model(Dfit, seg, dom)
        res["keyings"][name] = {"theta": th.tolist(), "m2lnL": f2, "converged": ok, "iterations": nit, "message": msg,
                                "held_nll": nll_per_track(Dheld, model_dC(Dheld, seg, dom, th)), "n_par": int(th.size),
                                "seconds": round(time.time() - t0, 1)}
        print(f"{name}: converged {ok} ({nit} it, {msg}), held-out -2lnL/track {res['keyings'][name]['held_nll']:.4f}",
              flush=True)
    res["held_nll_v3"] = nll_per_track(Dheld, Dheld["Dv3"])
    res["held_nll_reported"] = nll_per_track(Dheld, np.zeros_like(Dheld["Cq"]))
    res["chosen"] = min(res["keyings"], key=lambda k: res["keyings"][k]["held_nll"])
    seg = SEG_ETA if res["chosen"] == "eta" else SEG_PS
    th = np.asarray(res["keyings"][res["chosen"]]["theta"])
    H = load("htt", HTT_FILES)
    hn = nll_per_track(H, model_dC(H, seg, dom, th))
    wh, wt = region_widths(H, model_dC(H, seg, dom, th)), region_widths(Dheld, model_dC(Dheld, seg, dom, th))
    dn = hn - res["keyings"][res["chosen"]]["held_nll"]
    dw = {k: wh[k] - wt[k] for k in wh}
    res["htt"] = {"n": len(H["eta"]), "nll": hn, "nll_minus_ttbar_held": dn, "widths": wh, "ttbar_held_widths": wt,
                  "width_diff": dw, "include": bool(abs(dn) <= 0.1 and all(abs(x) <= 0.05 for x in dw.values()))}
    print(json.dumps({"chosen": res["chosen"], "v3": res["held_nll_v3"], "htt": {k: res["htt"][k] for k in
                                                                                 ("nll_minus_ttbar_held", "width_diff", "include")}},
                     indent=1), flush=True)
    with open(OUT / "select.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return 0


def final(a):
    """A16: the chosen keying on all TTbar fit files (+ H->tautau if it passed); continued until converged; writes the
    model file."""
    with open(OUT / "select.json") as fh:
        S = json.load(fh)
    seg = SEG_ETA if S["chosen"] == "eta" else SEG_PS
    D = load("ttbar", CM.FIT_FILES)
    samples = ["ttbar"]
    if S["htt"]["include"]:
        D = merge(D, load("htt", HTT_FILES))
        samples.append("htt")
    dom = domain(D, a.n_min)
    t0 = time.time()
    th, f2, ok, nit, msg = fit_model(D, seg, dom, theta0=np.asarray(S["keyings"][S["chosen"]]["theta"]))
    runs = [{"iterations": nit, "converged": ok, "message": msg, "m2lnL": f2}]
    while not ok and len(runs) < 4:                       # A16: continue until converged
        th, f2, ok, nit, msg = fit_model(D, seg, dom, theta0=th)
        runs.append({"iterations": nit, "converged": ok, "message": msg, "m2lnL": f2})
    _, out = p_eff(dom, D["aeta"], D["p"])
    doc = {"schema": "wpc-seedmodel/1", "plan": "c1b/PLAN.md §8 A16", "segments": {"kind": seg["kind"], "edges": list(seg["edges"])},
           "scale": SCALE.tolist(), "trans": TRANS, "polar": POLAR, "theta": th.tolist(), "domain": dom,
           "fit": {"samples": samples, "ttbar_files": CM.FIT_FILES, "htt_files": HTT_FILES if "htt" in samples else [],
                   "scenarios": CM.SCENARIOS, "n_tracks": len(D["eta"]), "runs": runs, "converged": ok,
                   "out_of_domain_share": float(out.mean()), "seconds": round(time.time() - t0, 1)},
           "material_scale": 1.0, "prompt_d0_cm": PROMPT_D0_CM}
    with open(MODEL_FILE, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(json.dumps(doc["fit"], indent=1), flush=True)
    return 0


def _dC_const(e):
    (tl, tz), (ll, lz) = N_PER_BLOCK["T"], N_PER_BLOCK["L"]
    MT, ML = _block_cov(e[:tl + tz], tl, tz), _block_cov(e[tl + tz:], ll, lz)
    ti, li = np.array(TRANS), np.array(POLAR)
    M = jnp.zeros((5, 5)).at[ti[:, None], ti[None, :]].set(MT).at[li[:, None], li[None, :]].set(ML)
    s = jnp.asarray(SCALE)
    return M * s[:, None] * s[None, :]


def _nll_const(e, J, Cq, R, r, J2, R2, r2, Q12, w):
    return joint_nll(Cq + _dC_const(e)[None], J, R, r, J2, R2, r2, Q12, w)


_VG_CONST = jax.jit(jax.value_and_grad(_nll_const))


def fit_const(D, m, w=None, e0=None):
    arrs = tuple(jnp.asarray(D[k][m]) for k in ("J", "Cq", "R", "r", "J2", "R2", "r2", "Q12"))
    ww = jnp.ones(int(m.sum())) if w is None else jnp.asarray(w[m], jnp.float64)
    e0 = np.r_[np.full(3, np.log(0.5)), np.zeros(3), np.full(2, np.log(0.5)), 0.0] if e0 is None else e0

    def fun(e):
        v, g = _VG_CONST(jnp.asarray(e), *arrs, ww)
        return float(v), np.asarray(g, np.float64)

    return optimize.minimize(fun, e0, jac=True, method="L-BFGS-B", options={"maxiter": 3000}).x


def elements(M):
    """The figure's elements of corrections M (n, 5, 5): sqrt dC(z0) [um], sqrt dC(tanL), corr(z0, tanL), sqrt dC(d0) [um]
    (a negative diagonal, possible only for v3's deficits, gives a negative signed sqrt)."""
    ssq = lambda x: np.sign(x) * np.sqrt(np.abs(x))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = M[:, 2, 3] / np.sqrt(np.abs(M[:, 2, 2] * M[:, 3, 3]))
    return {"z0_um": ssq(M[:, 3, 3]) * 1e4, "tanL": ssq(M[:, 2, 2]), "corr_z0_tanL": corr, "d0_um": ssq(M[:, 4, 4]) * 1e4}


def final_data():
    with open(MODEL_FILE) as fh:
        F = json.load(fh)["fit"]
    D = load("ttbar", CM.FIT_FILES)
    return merge(D, load("htt", HTT_FILES)) if "htt" in F["samples"] else D


def slices(a):
    """Per |eta| slice x figure |p| bin: a constant correction from the two-surface fit, with an event bootstrap by
    identity (A14). Writes seedmodel/slices.json."""
    import gates as G
    D = final_data()
    uev, inv = np.unique(D["ev"], return_inverse=True)
    reps = G.replicas(nev=len(uev), n=N_BOOT_SLICE, seed=20261017)
    edges = np.arange(0.0, 2.55 + 1e-9, ETA_CELL)
    out = []
    t0 = time.time()
    for plo, phi in FIG_P_BINS:
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (D["aeta"] >= lo) & (D["aeta"] < hi) & (D["p"] >= plo) & (D["p"] < phi)
            if (m & (D["scen"] == 0)).sum() < a.n_min:
                continue
            e = fit_const(D, m)
            vals = [elements(np.asarray(_dC_const(jnp.asarray(e)))[None])]
            for r in reps:
                vals.append(elements(np.asarray(_dC_const(jnp.asarray(fit_const(D, m, r[inv].astype(float), e))))[None]))
            row = {"p": [plo, phi], "eta": [lo, hi], "n": int(m.sum()), "n_tracks": int((m & (D["scen"] == 0)).sum())}
            for k in vals[0]:
                x = np.array([v[k][0] for v in vals])
                row[k], row[k + "_sd"] = float(x[0]), float(np.nanstd(x[1:]))
            out.append(row)
        print(f"[{time.time() - t0:6.1f}s] p {plo}-{phi}", flush=True)
    with open(OUT / "slices.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


# ---------------------------------------------------------------- plots
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
C_MODEL, C_V3, C_MEAS = "#2a78d6", "#eb6834", "#1baf7a"     # the reference palette's slots 1-3
ELEMENTS = (("z0_um", "√ΔC(z0) [µm]"), ("tanL", "√ΔC(tanλ)"), ("corr_z0_tanL", "corr(z0, tanλ) of ΔC"),
            ("d0_um", "√ΔC(d0) [µm]"))


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def plots(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": INK, "axes.titlesize": 9, "legend.fontsize": 8})
    SM = SeedModel()
    D = final_data()
    cal = AR.Tools(None, calibration=AR.CALIBRATION, angle_table=None).cal
    with open(OUT / "slices.json") as fh:
        SL = json.load(fh)
    grid = np.linspace(0.0, 2.5, 251)
    fig, axs = plt.subplots(len(FIG_P_BINS), len(ELEMENTS), figsize=(15, 9.5), sharex=True)
    for i, (plo, phi) in enumerate(FIG_P_BINS):
        pm = 0.5 * (plo + phi)
        eta_max = float(np.arccosh(max(phi / 2.0, 1.0)))
        g = grid[grid <= eta_max]
        M, out = SM.curve(g, pm, modal_ps(D, g))
        mod = elements(M)
        fine = np.arange(0.0, eta_max + 0.05, 0.05)
        v3x, v3y = [], {k: [] for k, _ in ELEMENTS}
        for lo, hi in zip(fine[:-1], fine[1:]):
            m = (D["aeta"] >= lo) & (D["aeta"] < hi)
            if m.sum() < 20:
                continue
            pt = pm / np.cosh(D["eta"][m])
            e3 = elements(ORP.seed_cov(cal, np.zeros((int(m.sum()), 5, 5)), D["a"][m, 2], D["n_disk"][m], D["n_stubs"][m],
                                       pt).mean(0)[None])
            v3x.append(lo)
            for k, _ in ELEMENTS:
                v3y[k].append(e3[k][0])
        pts = [r for r in SL if r["p"] == [plo, phi]]
        for j, (k, lab) in enumerate(ELEMENTS):
            ax = axs[i, j]
            _style(ax)
            edges = SM.seg["edges"][1:-1] if SM.seg["kind"] == "eta" else \
                g[1:][np.diff(segment_index(SM.seg, g, modal_ps(D, g))) != 0]
            for e in edges:
                ax.axvline(e, color=INK2, ls="--", lw=0.8)
            for lo_, hi_ in _runs(g, out):
                ax.axvspan(lo_, hi_, color=GRID, alpha=0.6, lw=0)
            ax.plot(g, mod[k], color=C_MODEL, lw=2, label="smooth model A(η)+B(η)/p_eff², at p_mid")
            if v3x:
                ax.step(v3x + [v3x[-1] + 0.05], v3y[k] + [v3y[k][-1]], where="post", color=C_V3, lw=1.5,
                        label="v3 binned, at pT = p_mid / cosh η")
            # a slice fit with no constraint on this element (a central value or an sd beyond 3x the model's and v3's
            # range in the panel) is left out of the panel and counted in its corner
            ref = np.nanmax(np.abs(np.r_[mod[k], v3y[k]])) if k != "corr_z0_tanL" else 1.0
            good = [r for r in pts if np.isfinite(r[k]) and np.isfinite(r[k + "_sd"]) and abs(r[k]) <= 3 * ref
                    and r[k + "_sd"] <= 3 * ref]
            if good:
                ax.errorbar([0.5 * (r["eta"][0] + r["eta"][1]) for r in good], [r[k] for r in good],
                            yerr=[r[k + "_sd"] for r in good], fmt="o", ms=4, color=C_MEAS, ecolor=C_MEAS, elinewidth=1,
                            capsize=2, label="per-slice two-surface fit, bootstrap sd")
            if len(good) < len(pts):
                ax.text(0.98, 0.04, f"{len(pts) - len(good)} unconstrained slice(s) not drawn", transform=ax.transAxes,
                        ha="right", color=INK2, fontsize=7)
            if k == "corr_z0_tanL":
                ax.set_ylim(-1.05, 1.05)
            ax.set_xlim(0, 2.55)
            if i == 0:
                ax.set_title(lab, color=INK)
            ax.set_ylabel((f"|p| {plo:g}-{phi:g} GeV\n" if j == 0 else "") + lab)
            if i == len(FIG_P_BINS) - 1:
                ax.set_xlabel("|η| (OT track)")
    h, l = axs[-1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, ("dashed: segment edges" if SM.seg["kind"] == "eta" else
                          "dashed: where the modal PS class changes") + "; shaded: outside the populated domain (p frozen to "
             "the nearest populated cell, A16); curves end where pT reaches 2 GeV; ΔC adds to the OT track's reported "
             "covariance", ha="center", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"seed_correction_vs_eta.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 0


def modal_ps(D, aeta, half=0.05):
    """The most common PS-stub count of the tracks within +-half of each |eta| (the PS keying's curve)."""
    out = np.zeros(len(aeta), int)
    for i, e in enumerate(aeta):
        m = np.abs(D["aeta"] - e) < half
        out[i] = np.bincount(D["n_ps"][m].astype(int)).argmax() if m.any() else 0
    return out


def _runs(x, flag):
    """[lo, hi] intervals of x where flag is set."""
    out, start = [], None
    for xi, f in zip(x, flag):
        if f and start is None:
            start = xi
        if not f and start is not None:
            out.append((start, xi))
            start = None
    if start is not None:
        out.append((start, x[-1]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("collect", "select", "final", "slices", "plots"))
    ap.add_argument("--scen", nargs="+", default=list(CM.SCENARIOS))
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--n-min", type=int, default=N_MIN, help="the minimum-count rule (smoke runs only lower it)")
    a = ap.parse_args(argv)
    return {"collect": collect, "select": select, "final": final, "slices": slices, "plots": plots}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
