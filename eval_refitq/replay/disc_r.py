"""C1b: the disc position errors from same-surface module overlaps (c1b/PLAN.md rev 4 §1 item 2, §8 A1, A2, A6, A8, A10).

A particle crossing two overlapping modules of one surface leaves two own class-T clusters. Each cluster's truth
residual (measured position minus the TP's production helix on its own module) carries the cluster error plus the
scattering offset from the vertex to the surface. Mapped to global through each module's local axes and differenced,
the common offset cancels and, along d in (e_phi, e_2) (e_2 = e_r on a disc, z on a barrel layer),

    D_d = sum_j s_j [(d.u_j) x_u,j + (d.v_j) x_v,j] + leftover,      s_1 = +1, s_2 = -1,

with x_u,j, x_v,j the cluster's local residuals (a disc module's u is 1-10 deg from e_phi, so the rotation mixes u and
v) and the leftover the scattering between the two clusters, a Gaussian of variance b^2 g_d / p^2 (A1: g_d the pair's
own geometric factor, b one scale per surface).

A8: per size bin and axis the single-cluster residual is a symmetric core-plus-tail shape about a centre mu (the
module-local offset, which enters through the signed projections): a two-Gaussian mixture (core s, tail fraction eps,
tail width kappa s), and as the systematic family a Student-t written as a 4-node Gaussian scale mixture. With a
cluster's u and v independent, D_d is a Gaussian mixture (16 components for the two-Gaussian family), so the likelihood
is exact, and a pair whose clusters sit in different size bins gets the convolution of the two shapes. One unbinned
likelihood per surface (composite over the two directions) fits every bin's shape and b; sparse bins (fewer than
MIN_PARAM clusters) share one shape per axis and are reported NaN (A6). The table width is read off each fitted single
shape with v3's definition, MAD / 0.6745 about the median. Errors: the event bootstrap (weighted refits).

IL1 is the method check only (D0): the same fit on IL1's pairs against IL1's single-cluster truth width where the
beam pipe's scattering is negligible (the plateau rule), reweighted to the pairs' |cot| mix. The truth residual of an
IL1 cluster carries the beam pipe's Highland scattering (beam_pipe_sigma), which the plateau rule keeps out.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize, special, stats

from common import OP, ORP
from ngtagger.spdefs import angles as SA

jax.config.update("jax_enable_x64", True)

MAD_K = 0.6745
UM = 1e4                                            # cm -> um: the fit runs in um
P_MIN = {"disc": 10.0, "barrel": 2.0}               # GeV: A2 on the discs; A6 on IL1 (its overlapping ladders are a
                                                    # few mm apart, so its leftover at 2 GeV is below a disc's at 10)
MIN_PARAM = 400                                     # clusters (after the p cut) a size bin needs for its own shape (A6)
OVERLAP_MAX_CM = 0.1                                # a geometric overlap: each cluster within 1 mm of its helix
                                                    # crossing, and the two residuals within 1 mm of each other
PT_MIN, D0_MAX_CM = 2.0, 0.005
B_PRIME_MAX = 0.02                                  # cm GeV: a 1.9 cm lever arm gives ~0.003-0.006
# the shapes (A8)
EPS_MAX, KAPPA_MAX = 0.5, 30.0                      # the two-Gaussian family: tail fraction, tail / core width
NU_MIN, NU_MAX = 1.5, 60.0                          # the Student-t family's degrees of freedom
T_Q = (0.125, 0.375, 0.625, 0.875)                  # its 4 equal-weight nodes: quantiles of the Gamma(nu/2, nu/2) precision
_LOGNU = np.linspace(np.log(NU_MIN), np.log(NU_MAX), 400)
_T_LAM = np.array([[stats.gamma.ppf(q, a=np.exp(x) / 2, scale=2 / np.exp(x)) for q in T_Q] for x in _LOGNU])
# parameters per size bin: mix2 (log s, eps, kappa, mu); t (log s, nu, mu); splitN: a core split into two Gaussians at
# +-delta (flat-topped for delta ~ g) plus N Gaussian tails: (log g, log delta, tail fraction, [tail split], kappas, mu)
FAMILIES = {"mix2": 4, "t": 3, "split1": 5, "split2": 7}
# the beam pipe (A8): D121's central beryllium tube (Geometry/CMSCommonData beampipe/2030/v3: r 2.17-2.25 cm, 0.8 mm;
# X0(Be) = 35.28 cm); tkLayout's is 0.228 % X0 at r = 2.3 cm (arms.BEAM_PIPE)
BEAM_PIPE_R_CM = 0.5 * (2.17 + 2.25)
BEAM_PIPE_X0 = 0.08 / 35.28
HIGHLAND_GEV = 0.0136


def size_bin(size, edges):
    return np.clip(np.searchsorted(edges, size, side="right") - 1, 0, len(edges) - 2)


def v3_width(tables, axis, size, cot):
    """v3's R (cm) at each cluster's own (size, |cot|) cell (refit_calibration_fit.apply_R's lookup)."""
    t = tables[axis]
    W = np.asarray(t["width_cm"])
    return W[size_bin(size, t["size_edges"]), size_bin(np.abs(cot), t["cot_edges"])]


def wquantile(x, w, q):
    """The weighted q quantile (midpoint interpolation: the ordinary median for equal weights)."""
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    if not len(x) or c[-1] <= 0:
        return np.nan
    return float(np.interp(q * c[-1], c - 0.5 * w, x))


def wmad_width(x, w=None):
    """v3's robust width: MAD / 0.6745 about the median (weighted)."""
    w = np.ones(len(x)) if w is None else np.asarray(w, np.float64)
    m = wquantile(x, w, 0.5)
    return wquantile(np.abs(x - m), w, 0.5) / MAD_K


def beam_pipe_sigma(p, eta, r, cot_a, cot_b):
    """The Highland scattering offset (cm) the beam pipe puts on a prompt track's truth residual at radius r, per
    local axis: theta0 over the path from the pipe to the cluster, mapped onto the module plane (a perpendicular
    offset delta lands at delta sqrt(1 + cot^2) along the axis)."""
    x = BEAM_PIPE_X0 * np.cosh(eta)
    th = HIGHLAND_GEV / p * np.sqrt(x) * (1.0 + 0.038 * np.log(x))
    L = np.maximum(r - BEAM_PIPE_R_CM, 0.0) * np.cosh(eta)
    return th * L * np.sqrt(1.0 + cot_a ** 2), th * L * np.sqrt(1.0 + cot_b ** 2)


# ------------------------------------------------------------------ clusters and pairs
def own_clusters(ctx, geo, surfaces):
    """Per (TP, module) on the surfaces: the own class-T cluster nearest the TP's production helix (prompt TPs, pT >= 2
    GeV, |d0| < 50 um), within 1 mm of it. Returns the cluster index, TP, module, surface, local residuals (du, dv)
    and the TP's local cots on the module."""
    K, TP = ctx["K"], ctx["TP"]
    kpt = ORP.pt_constant(ctx["T"])
    mi, tp = K["mi"], K["tp_global"]
    ok = (mi >= 0) & (tp >= 0) & SA.class_T(K["tpEmuClass"])
    ok &= np.isin(geo["surface"][np.maximum(mi, 0)], surfaces)
    tpr = np.where(ok, tp, 0)
    ok &= (TP["pt"][tpr] >= PT_MIN) & (np.abs(TP["d0"][tpr]) < D0_MAX_CM)
    ci = np.flatnonzero(ok)
    if not len(ci):
        return None
    t, m = tp[ci], mi[ci]
    S = geo["surface"][m]
    at = np.stack([TP["charge"][t] * kpt / TP["pt"][t], TP["phi0"][t], np.sinh(TP["eta"][t]), TP["z0"][t],
                   TP["d0"][t]], 1)
    s0 = np.full(len(ci), np.nan)
    for s in np.unique(S):
        k = S == s
        s0[k] = OP.surface_s(at[k], geo, int(s))
    pj = OP.project_to_modules(at, s0, geo, m, refine=np.ones(len(ci), bool))
    du = K["localX"][ci] - pj["u"]
    dv = K["localY"][ci] - pj["v"]
    good = pj["ok"] & (np.hypot(du, dv) < OVERLAP_MAX_CM)
    ci, t, m, S, du, dv = ci[good], t[good], m[good], S[good], du[good], dv[good]
    d = np.hypot(du, dv)
    o = np.lexsort((d, m, t))
    first = np.r_[True, (t[o][1:] != t[o][:-1]) | (m[o][1:] != m[o][:-1])]
    o = o[first]
    return {"ci": ci[o], "t": t[o], "m": m[o], "S": S[o], "du": du[o], "dv": dv[o],
            "cotA": K["tpLocalCotAlpha"][ci[o]], "cotB": K["tpLocalCotBeta"][ci[o]]}


def clusters_of_chunk(ctx, geo, surface, start=0):
    """The own clusters of one surface with their truth residuals, for the truth width against p (IL1, A8): ev, p,
    pT, eta, sizes, the TP's local cots, (du, dv), the transverse radius and the beam pipe's expected offsets."""
    C = own_clusters(ctx, geo, [surface])
    if C is None:
        return None
    K, TP, t, m = ctx["K"], ctx["TP"], C["t"], C["m"]
    pos = geo["origin"][m] + C["du"][:, None] * geo["ex"][m] + C["dv"][:, None] * geo["ey"][m]
    r = np.hypot(pos[:, 0], pos[:, 1])
    p = TP["pt"][t] * np.cosh(TP["eta"][t])
    bu, bv = beam_pipe_sigma(p, TP["eta"][t], r, C["cotA"], C["cotB"])
    return {"ev": K["event"][C["ci"]].astype(np.int64) + start, "p": p, "pt": TP["pt"][t], "eta": TP["eta"][t],
            "sx": K["sizeX"][C["ci"]].astype(np.float64), "sy": K["sizeY"][C["ci"]].astype(np.float64),
            "cotA": C["cotA"], "cotB": C["cotB"], "du": C["du"], "dv": C["dv"], "r": r, "bp_u": bu, "bp_v": bv}


def pairs_of_chunk(ctx, geo, surfaces, start=0):
    """Overlap pairs of one chunk: every two modules of the same surface whose own clusters (own_clusters) of one TP
    pass the overlap test form a pair. Returns per pair: surface, ev, p, eta, sizes, |truth cot| per cluster, the
    local residuals (du, dv), each module's axes in the surface plane, the common directions, the leftover's
    geometric factors and the facings."""
    C = own_clusters(ctx, geo, surfaces)
    if C is None:
        return None
    K, TP = ctx["K"], ctx["TP"]
    ci, t, m, S, du, dv, cotA, cotB = (C[k] for k in ("ci", "t", "m", "S", "du", "dv", "cotA", "cotB"))
    key = t * 64 + S
    o = np.argsort(key, kind="stable")
    key_o = key[o]
    i1, i2 = [], []
    starts = np.flatnonzero(np.r_[True, key_o[1:] != key_o[:-1]])
    ends = np.r_[starts[1:], len(o)]
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue
        idx = o[a:b]
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                i1.append(idx[x])
                i2.append(idx[y])
    if not i1:
        return None
    i1, i2 = np.array(i1), np.array(i2)
    g1 = du[i1, None] * geo["ex"][m[i1]] + dv[i1, None] * geo["ey"][m[i1]]
    g2 = du[i2, None] * geo["ex"][m[i2]] + dv[i2, None] * geo["ey"][m[i2]]
    pos = 0.5 * (geo["origin"][m[i1]] + geo["origin"][m[i2]])
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], 1)
    disc = np.isin(S[i1], [s for s in np.unique(S) if OP.SPN.is_disc_surface(int(s))])
    # the second direction: radial on a disc, z on a barrel layer (the barrel's v)
    e_2 = np.where(disc[:, None], np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], 1),
                   np.tile([0.0, 0.0, 1.0], (len(phi), 1)))
    D = g1 - g2
    keep = (np.abs((D * e_phi).sum(1)) < OVERLAP_MAX_CM) & (np.abs((D * e_2).sum(1)) < OVERLAP_MAX_CM)
    i1, i2, e_phi, e_2 = i1[keep], i2[keep], e_phi[keep], e_2[keep]
    gpos = [geo["origin"][m[ii]] + K["localX"][ci[ii], None] * geo["ex"][m[ii]]
            + K["localY"][ci[ii], None] * geo["ey"][m[ii]] for ii in (i1, i2)]
    tl = np.abs(np.sinh(TP["eta"][t[i1]]))
    sec = np.sqrt(1.0 + tl * tl)
    # the leftover scattering's geometric factor per direction: path between the two module planes L, the
    # module's thickness path factor, the in-plane projection of a kink (A1)
    disc_ = disc[keep]
    dz = np.abs(gpos[0][:, 2] - gpos[1][:, 2])
    dr = np.abs(np.hypot(gpos[0][:, 0], gpos[0][:, 1]) - np.hypot(gpos[1][:, 0], gpos[1][:, 1]))
    L = np.where(disc_, dz * sec / np.maximum(tl, 1e-6), dr * sec)
    xfac = np.where(disc_, sec / np.maximum(tl, 1e-6), sec)
    proj2 = np.where(disc_, sec * sec / np.maximum(tl * tl, 1e-12), sec * sec)     # 1 / cos^2 or 1 / sin^2
    out = {"surface": S[i1], "ev": K["event"][ci[i1]].astype(np.int64) + start,
           "p": TP["pt"][t[i1]] * np.cosh(TP["eta"][t[i1]]), "eta": np.abs(TP["eta"][t[i1]]),
           "e_phi": e_phi, "e_2": e_2, "g_phi": L * L * xfac, "g_2": L * L * xfac * proj2}
    for j, ii in (("1", i1), ("2", i2)):
        out[f"sx{j}"] = K["sizeX"][ci[ii]].astype(np.float64)
        out[f"sy{j}"] = K["sizeY"][ci[ii]].astype(np.float64)
        out[f"cotA{j}"], out[f"cotB{j}"] = cotA[ii], cotB[ii]
        out[f"du{j}"], out[f"dv{j}"] = du[ii], dv[ii]
        out[f"ux{j}"], out[f"vx{j}"] = geo["ex"][m[ii]], geo["ey"][m[ii]]
        out[f"facing{j}"] = np.sign(geo["ez"][m[ii]][:, 2]).astype(np.int8)
    return out


def differences(P, rotation=True, drop_facing=None, du=None, dv=None):
    """(D_phi, D_2) per pair (cm) and the signed projections of each cluster's local axes on the two directions,
    co[(d, axis, j)], cluster 2's with its minus sign. rotation=False: the no-rotation known-bad (u on e_phi and v
    on e_2 exactly, with the facing's sign, no mixing). drop_facing = +-1: the known-bad with that facing's u sign
    dropped. du, dv: {"1": ..., "2": ...} residuals replacing the pairs' own (the toys)."""
    co = {}
    G = []
    for j, sg in (("1", 1.0), ("2", -1.0)):
        ux, vx = P[f"ux{j}"].copy(), P[f"vx{j}"]
        if drop_facing is not None:
            ux = np.where((P[f"facing{j}"] == drop_facing)[:, None], -ux, ux)
        xu = P[f"du{j}"] if du is None else du[j]
        xv = P[f"dv{j}"] if dv is None else dv[j]
        G.append(xu[:, None] * ux + xv[:, None] * vx)
        for d, e in (("phi", P["e_phi"]), ("2", P["e_2"])):
            pu, pv = (ux * e).sum(1), (vx * e).sum(1)
            if not rotation:
                pu = np.sign(pu) if d == "phi" else np.zeros_like(pu)
                pv = np.zeros_like(pv) if d == "phi" else np.sign(pv)
            co[(d, "u", j)], co[(d, "v", j)] = sg * pu, sg * pv
    D = G[0] - G[1]
    return {"phi": (D * P["e_phi"]).sum(1), "2": (D * P["e_2"]).sum(1)}, co


# ------------------------------------------------------------------ the shape model (A8)
def _kappa(x):
    return 1.0 + (KAPPA_MAX - 1.0) * jax.nn.sigmoid(x)


def _components(family, par):
    """Per size bin (rows of par): the single shape's Gaussian components (weights, variances in um^2, locations in
    um, symmetric about the centre) and the centre (um)."""
    s = jnp.exp(par[:, 0])
    if family == "mix2":
        eps = EPS_MAX * jax.nn.sigmoid(par[:, 1])
        kap = _kappa(par[:, 2])
        var = jnp.stack([s * s, (kap * s) ** 2], 1)
        return jnp.stack([1.0 - eps, eps], 1), var, jnp.zeros_like(var), par[:, 3]
    if family == "t":
        lnu = np.log(NU_MIN) + (np.log(NU_MAX) - np.log(NU_MIN)) * jax.nn.sigmoid(par[:, 1])
        lam = jnp.stack([jnp.interp(lnu, _LOGNU, _T_LAM[:, k]) for k in range(len(T_Q))], 1)
        return jnp.full(lam.shape, 1.0 / len(T_Q)), (s * s)[:, None] / lam, jnp.zeros_like(lam), par[:, 2]
    delta = jnp.exp(par[:, 1])
    etot = EPS_MAX * jax.nn.sigmoid(par[:, 2])
    core = jnp.sqrt(s * s + delta * delta)                         # the split core's standard deviation
    if family == "split1":
        fr, kaps, mu = [jnp.ones_like(s)], [_kappa(par[:, 3])], par[:, 4]
    else:
        f1 = jax.nn.sigmoid(par[:, 3])
        k1 = _kappa(par[:, 4])
        fr, kaps, mu = [f1, 1.0 - f1], [k1, k1 * _kappa(par[:, 5])], par[:, 6]
    w = jnp.stack([0.5 * (1.0 - etot), 0.5 * (1.0 - etot)] + [etot * f for f in fr], 1)
    var = jnp.stack([s * s, s * s] + [(k * core) ** 2 for k in kaps], 1)
    loc = jnp.stack([-delta, delta] + [jnp.zeros_like(s) for _ in kaps], 1)
    return w, var, loc, mu


def _nll(theta, family, nbu, nbv, D, A, bins, lo, w, fixed_lb):
    """Minus the weighted log-likelihood of the pair differences (composite over the two directions)."""
    k = FAMILIES[family]
    pu = theta[:nbu * k].reshape(nbu, k)
    pv = theta[nbu * k:(nbu + nbv) * k].reshape(nbv, k)
    lb = theta[-1] if fixed_lb is None else fixed_lb
    comp = {"u": _components(family, pu), "v": _components(family, pv)}
    terms = (("u", 0), ("v", 1), ("u", 2), ("v", 3))               # (u1, v1, u2, v2): bins[t], A[d, t]
    total = 0.0
    n = D.shape[1]
    for d in range(2):
        mean = jnp.zeros((n, 1))
        logw = jnp.zeros((n, 1))
        var = (jnp.exp(2.0 * lb) * lo[d])[:, None]
        for ax, tt in terms:
            cw, cv, cl, mu = comp[ax]
            ib, a = bins[tt], A[d, tt]
            mean = (mean[:, :, None] + (a[:, None] * (mu[ib][:, None] + cl[ib]))[:, None, :]).reshape(n, -1)
            logw = (logw[:, :, None] + jnp.log(cw[ib])[:, None, :]).reshape(n, -1)
            var = (var[:, :, None] + (a * a)[:, None, None] * cv[ib][:, None, :]).reshape(n, -1)
        var = var + 1e-6
        r2 = (D[d][:, None] - mean) ** 2
        lp = jax.scipy.special.logsumexp(logw - 0.5 * jnp.log(2.0 * np.pi * var) - 0.5 * r2 / var, axis=1)
        total = total + jnp.sum(w * lp)
    return -total


_VG = jax.jit(jax.value_and_grad(_nll), static_argnums=(1, 2, 3))


def single_width(family, par):
    """MAD / 0.6745 (um) of each size bin's fitted single shape (par rows), about its centre (the median of a symmetric
    mixture), solved on the mixture's distribution function: sum_k w_k [Phi((m - l_k) / s_k) - Phi((-m - l_k) / s_k)]
    = 1/2."""
    cw, cv, cl, _ = (np.asarray(x) for x in _components(family, jnp.asarray(par)))
    sd = np.sqrt(cv)
    lo, hi = np.zeros(len(par)), 10.0 * (sd + np.abs(cl)).max(1)
    for _ in range(80):
        m = 0.5 * (lo + hi)
        f = np.sum(cw * (stats.norm.cdf((m[:, None] - cl) / sd) - stats.norm.cdf((-m[:, None] - cl) / sd)), 1) - 0.5
        lo, hi = np.where(f < 0, m, lo), np.where(f < 0, hi, m)
    return 0.5 * (lo + hi) / MAD_K


class ShapeFit:
    """One surface's pairs (sel, after the surface's p cut) ready for repeated unbinned fits of the single-cluster
    shapes: bootstrap weights, toy residuals, a fixed leftover scale. family: one of FAMILIES."""

    def __init__(self, P, tables, sel, family="mix2", rotation=True, drop_facing=None):
        self.family = family
        disc = OP.SPN.is_disc_surface(int(P["surface"][sel][0]))
        sel = sel & (P["p"] >= P_MIN["disc" if disc else "barrel"])
        self.idx = np.flatnonzero(sel)
        self.Q = Q = {k: v[sel] for k, v in P.items()}
        self.rotation, self.drop_facing = rotation, drop_facing
        self.edges = {"u": tables["u"]["size_edges"], "v": tables["v"]["size_edges"]}
        self.raw, self.cnt, self.own, self.par_bin = {}, {}, {}, {}
        for ax, sk in (("u", "sx"), ("v", "sy")):
            e = self.edges[ax]
            nb = len(e) - 1
            raw = [size_bin(Q[f"{sk}{j}"], e) for j in ("1", "2")]
            cnt = np.bincount(np.concatenate(raw), minlength=nb)
            own = cnt >= MIN_PARAM                       # A6: a bin with fewer clusters shares the sparse shape
            pooled = [np.where(own[b], b, nb) for b in raw]
            used = np.unique(np.concatenate(pooled))
            remap = np.full(nb + 1, -1)
            remap[used] = np.arange(len(used))
            self.raw[ax], self.cnt[ax], self.own[ax] = raw, cnt, own
            self.par_bin[ax] = used                      # raw bin (nb = the sparse shape) of each parameter row
            pooled = [remap[b] for b in pooled]
            setattr(self, f"_b{ax}", pooled)
        self.nbu, self.nbv = len(self.par_bin["u"]), len(self.par_bin["v"])
        self.bins = jnp.asarray(np.stack([self._bu[0], self._bv[0], self._bu[1], self._bv[1]]))
        D, co = differences(Q, rotation, drop_facing)
        self.D = self._D(D)
        self.A = jnp.asarray(np.stack([np.stack([co[(d, "u", "1")], co[(d, "v", "1")], co[(d, "u", "2")],
                                                 co[(d, "v", "2")]]) for d in ("phi", "2")]))
        self.lo = jnp.asarray(np.stack([Q["g_phi"], Q["g_2"]]) / Q["p"] ** 2 * UM * UM)
        self.n = len(self.idx)

    @staticmethod
    def _D(D):
        return jnp.asarray(np.stack([D["phi"], D["2"]]) * UM)

    def start(self, widths=None):
        """Starting parameters: each bin's core at its width (um; default 5 um in u, 20 in v), a 10 % tail of 3x
        (split: a core of half Gaussian, half split, and a 5 % tail)."""
        x3 = special.logit(2.0 / (KAPPA_MAX - 1.0))                  # kappa 3
        rows = []
        for ax in ("u", "v"):
            for rb in self.par_bin[ax]:
                s = (widths[ax][rb] if widths is not None and rb < len(widths[ax]) and np.isfinite(widths[ax][rb])
                     else (5.0 if ax == "u" else 20.0))
                rows.append({"mix2": [np.log(s), special.logit(0.1 / EPS_MAX), x3, 0.0],
                             "t": [np.log(s), special.logit((np.log(5.0) - np.log(NU_MIN))
                                                            / (np.log(NU_MAX) - np.log(NU_MIN))), 0.0],
                             "split1": [np.log(0.7 * s), np.log(0.7 * s), special.logit(0.05 / EPS_MAX), x3, 0.0],
                             "split2": [np.log(0.7 * s), np.log(0.7 * s), special.logit(0.05 / EPS_MAX), 0.0, x3, x3,
                                        0.0]}[self.family])
        return np.concatenate([np.concatenate(rows), [np.log(0.01)]])

    def fit(self, weights=None, theta0=None, D=None, fixed_b=None, start_widths=None):
        """One maximum-likelihood fit. weights: per pair (the bootstrap); D: replacement differences (the toys);
        fixed_b: the leftover scale held fixed. Returns the result dict (widths in cm, NaN for sparse bins)."""
        w = jnp.ones(self.n) if weights is None else jnp.asarray(weights, jnp.float64)
        Dj = self.D if D is None else D
        th0 = self.start(start_widths) if theta0 is None else np.asarray(theta0, np.float64)
        flb = None if fixed_b is None else float(np.log(max(fixed_b, 1e-12)))
        if flb is not None:
            th0 = th0[:-1]
        args = (self.family, self.nbu, self.nbv, Dj, self.A, self.bins, self.lo, w, flb)

        def fun(th):
            v, g = _VG(jnp.asarray(th if flb is None else np.r_[th, 0.0]), *args)
            g = np.asarray(g, np.float64)
            return float(v), (g if flb is None else g[:-1])

        r = optimize.minimize(fun, th0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "maxfun": 6000})
        th = r.x if flb is None else np.r_[r.x, flb]
        return self.result(th, r)

    def result(self, th, r=None):
        k = FAMILIES[self.family]
        out = {"theta": th, "converged": bool(r.success) if r is not None else None,
               "nll": float(r.fun) if r is not None else None, "n_pairs": self.n}
        off = 0
        for ax, nb in (("u", self.nbu), ("v", self.nbv)):
            par = th[off * k:(off + nb) * k].reshape(nb, k)
            off += nb
            W = single_width(self.family, par) / UM
            e_all = self._shape(par)
            nraw = len(self.edges[ax]) - 1
            R = np.full(nraw, np.nan)
            shape = {kk: np.full(nraw, np.nan) for kk in e_all}
            sparse = None
            for i, rb in enumerate(self.par_bin[ax]):
                e = {"R": W[i], **{kk: v[i] for kk, v in e_all.items()}}
                if rb == nraw:
                    sparse = e
                    continue
                R[rb] = e["R"]
                for kk in shape:
                    shape[kk][rb] = e[kk]
            out[f"R_{ax}"], out[f"shape_{ax}"], out[f"sparse_{ax}"], out[f"n_{ax}"] = R, shape, sparse, self.cnt[ax]
        b = float(np.exp(th[-1]))
        out["b"] = b
        out["B_prime"] = np.array([b * np.sqrt(np.median(self.Q["g_phi"])), b * np.sqrt(np.median(self.Q["g_2"]))])
        return out

    def _shape(self, par):
        """Per parameter row: the fitted shape's summary (cm for lengths): the core width, the tail fraction and the
        first tail's width over the core, the split half-distance or the t's degrees of freedom, the centre."""
        cw, cv, cl, mu = (np.asarray(x) for x in _components(self.family, jnp.asarray(par)))
        out = {"mu": mu / UM}
        if self.family == "t":
            out["s"] = np.exp(par[:, 0]) / UM
            out["nu"] = np.exp(np.log(NU_MIN) + (np.log(NU_MAX) - np.log(NU_MIN)) * special.expit(par[:, 1]))
            return out
        ncore = 1 if self.family == "mix2" else 2
        core = np.sqrt(cv[:, 0] + cl[:, 0] ** 2)
        out["s"] = core / UM
        out["tail_frac"] = cw[:, ncore:].sum(1)
        out["tail_rel"] = np.sqrt(cv[:, ncore]) / core
        if ncore == 2:
            out["delta"] = np.abs(cl[:, 1]) / UM
        return out

    # -------------------------------------------------------------- toys (T0)
    def toy_D(self, planted, shape, rng, b_common=0.1, b_prime=0.004):
        """Toy differences on these pairs: each cluster's residual drawn from the planted shape (draw_unit) scaled to the planted width per raw size bin (cm, MAD / 0.6745), plus a
        common scattering offset (b_common / p, cancels) and the leftover with the pairs' own geometric factors
        (B'_phi = b_prime cm GeV at the median pair)."""
        Q, n = self.Q, self.n
        common = (b_common / Q["p"])[:, None] * rng.standard_normal((n, 2))
        gc = common[:, 0, None] * Q["e_phi"] + common[:, 1, None] * Q["e_2"]
        du, dv = {}, {}
        for j, jj in (("1", 0), ("2", 1)):
            x = {}
            for ax in ("u", "v"):
                R = planted[ax][self.raw[ax][jj]]
                x[ax] = R * draw_unit(shape, rng, n)
            du[j] = x["u"] + (gc * Q[f"ux{j}"]).sum(1)
            dv[j] = x["v"] + (gc * Q[f"vx{j}"]).sum(1)
        b_scale = b_prime / np.sqrt(np.median(Q["g_phi"]))
        rel = (b_scale / Q["p"])[:, None] * np.sqrt(np.stack([Q["g_phi"], Q["g_2"]], 1)) * rng.standard_normal((n, 2))
        g = rel[:, 0, None] * Q["e_phi"] + rel[:, 1, None] * Q["e_2"]
        du["2"] = du["2"] + (g * Q["ux2"]).sum(1)
        dv["2"] = dv["2"] + (g * Q["vx2"]).sum(1)
        D, _ = differences(Q, True, None, du, dv)              # the truth always has the rotation
        return self._D(D)


PLANT_MIX2 = (0.15, 4.0)                    # the two-Gaussian plant: tail fraction, tail / core width
PLANT_BOX = (0.2, 0.01, 5.0)                # the flat-topped plant: a box of half-width 1 smeared by N(0, 0.2), with a
                                            # 1 % tail of 5x its sd (the real disc size-1 pair quantiles, A8 review)


def _box_cdf_sym(m, g):
    """P(|X| < m) for X uniform on (-1, 1) plus N(0, g^2)."""
    def F(x):
        return 0.5 * ((x + 1) * stats.norm.cdf((x + 1) / g) - (x - 1) * stats.norm.cdf((x - 1) / g)
                      + g * (stats.norm.pdf((x + 1) / g) - stats.norm.pdf((x - 1) / g)))
    return F(m) - F(-m)


def _unit_mad(shape):
    """The MAD of a planted shape at unit scale."""
    if shape == "t3":
        return float(stats.t.ppf(0.75, 3))
    if shape == "box":
        g, eps, kap = PLANT_BOX
        sT = kap * np.sqrt(1.0 / 3.0 + g * g)
        return optimize.brentq(lambda m: (1 - eps) * _box_cdf_sym(m, g) + eps * (2 * stats.norm.cdf(m / sT) - 1) - 0.5,
                               1e-3, 10.0)
    eps, kap = PLANT_MIX2
    return optimize.brentq(lambda m: (1 - eps) * (2 * stats.norm.cdf(m) - 1) + eps * (2 * stats.norm.cdf(m / kap) - 1)
                           - 0.5, 0.1, 10.0)


def draw_unit(shape, rng, n):
    """n draws of a planted shape ("mix2", "t3", "box") with MAD / 0.6745 = 1."""
    if shape == "t3":
        x = rng.standard_t(3, n)
    elif shape == "box":
        g, eps, kap = PLANT_BOX
        x = rng.uniform(-1.0, 1.0, n) + g * rng.standard_normal(n)
        t = rng.random(n) < eps
        x[t] = kap * np.sqrt(1.0 / 3.0 + g * g) * rng.standard_normal(int(t.sum()))
    else:
        eps, kap = PLANT_MIX2
        x = rng.standard_normal(n) * np.where(rng.random(n) < eps, kap, 1.0)
    return x * MAD_K / _unit_mad(shape)


def planted_widths(tables):
    """The planted R per size bin (only a plausible scale): v3's IL1 widths, the median over each size row's
    |cot| cells."""
    return {a: np.median(np.asarray(tables[a]["width_cm"]), axis=1) for a in ("u", "v")}


# ------------------------------------------------------------------ references
def truth_width(P, tables, sel, axis):
    """The pair clusters' own truth-residual R per size bin (MAD / 0.6745 about the median), both clusters of every
    pair: D2's scattering-contaminated known-bad (the round-0 estimator)."""
    e = tables[axis]["size_edges"]
    s = np.concatenate([P[f"s{'x' if axis == 'u' else 'y'}{j}"][sel] for j in ("1", "2")])
    r = np.concatenate([P[f"d{axis}{j}"][sel] for j in ("1", "2")])
    b = size_bin(s, e)
    out = np.full(len(e) - 1, np.nan)
    for i in range(len(e) - 1):
        if np.sum(b == i) >= 20:
            out[i] = wmad_width(r[b == i])
    return out, np.bincount(b, minlength=len(e) - 1)


def v3_at_cells(P, tables, sel, axis):
    """v3's R at each pair cluster's own (size, |truth cot|) cell, averaged per size bin (pair-weighted)."""
    e = tables[axis]["size_edges"]
    s = np.concatenate([P[f"s{'x' if axis == 'u' else 'y'}{j}"][sel] for j in ("1", "2")])
    cot = np.concatenate([P[f"cot{'A' if axis == 'u' else 'B'}{j}"][sel] for j in ("1", "2")])
    w = v3_width(tables, axis, s, cot)
    b = size_bin(s, e)
    n = np.bincount(b, minlength=len(e) - 1)
    return np.bincount(b, weights=w, minlength=len(e) - 1) / np.maximum(n, 1), n


AX_KEYS = {"u": ("sx", "cotA", "du", "bp_u"), "v": ("sy", "cotB", "dv", "bp_v")}


def cell_reweighted_width(C, sel, pair_cells, tables, axis, w_ev=None):
    """D0's reference per size bin: MAD / 0.6745 of the truth residuals of the clusters C[sel], reweighted within
    each size bin to the pair clusters' |cot| cell mix (pair_cells: (size bins, cot cells) counts). w_ev: event
    weights (the bootstrap). Also returns the unmatched share (pair mass in cells with no reference cluster)."""
    sk, ck, rk, _ = AX_KEYS[axis]
    t = tables[axis]
    sb = size_bin(C[sk][sel], t["size_edges"])
    cb = size_bin(np.abs(C[ck][sel]), t["cot_edges"])
    x = C[rk][sel]
    we = np.ones(len(x)) if w_ev is None else w_ev[C["ev"][sel]].astype(np.float64)
    nb = len(t["size_edges"]) - 1
    out, unmatched = np.full(nb, np.nan), np.zeros(nb)
    for i in range(nb):
        m = sb == i
        tot = pair_cells[i].sum()
        if tot == 0 or not m.any():
            continue
        ref = np.bincount(cb[m], weights=we[m], minlength=pair_cells.shape[1])
        f = np.where(ref > 0, pair_cells[i] / tot / np.maximum(ref, 1e-12), 0.0)
        unmatched[i] = float(pair_cells[i][ref == 0].sum() / tot)
        out[i] = wmad_width(x[m], we[m] * f[cb[m]])
    return out, unmatched


def pair_cells(P, sel, tables, axis, w=None):
    """The pair clusters' (size bin, |cot| cell) counts (w: per-pair weights)."""
    sk, ck = ("sx", "cotA") if axis == "u" else ("sy", "cotB")
    t = tables[axis]
    ww = np.ones(int(np.sum(sel))) if w is None else np.asarray(w, np.float64)
    sb = np.concatenate([size_bin(P[f"{sk}{j}"][sel], t["size_edges"]) for j in ("1", "2")])
    cb = np.concatenate([size_bin(np.abs(P[f"{ck}{j}"][sel]), t["cot_edges"]) for j in ("1", "2")])
    nb, nc = len(t["size_edges"]) - 1, len(t["cot_edges"]) - 1
    return np.bincount(sb * nc + cb, weights=np.r_[ww, ww], minlength=nb * nc).reshape(nb, nc)


def highland_inflation(x, sig, rng, draws=20):
    """The width a residual sample x would have with each cluster's Gaussian offset sig added, over its own width:
    MAD / 0.6745 of (x_i + N(0, sig_i)), draws per cluster, over that of x."""
    y = np.repeat(x, draws) + rng.standard_normal(len(x) * draws) * np.repeat(sig, draws)
    return wmad_width(y) / wmad_width(x)


def highland_expected(x_plateau, sig, rng, draws=20):
    """The Highland expectation of a p bin's width: the plateau residuals (resampled) with each of the bin's
    clusters' beam-pipe offset added."""
    pick = rng.integers(0, len(x_plateau), len(sig) * draws)
    y = x_plateau[pick] + rng.standard_normal(len(sig) * draws) * np.repeat(sig, draws)
    return wmad_width(y)
