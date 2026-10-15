"""C1b: the disc position errors from same-surface module overlaps (c1b/PLAN.md rev 4 §1 item 2).

A particle crossing two overlapping modules of one surface leaves two own class-T clusters. Each cluster's
truth residual (measured position minus the TP's production helix on its own module) carries the cluster
error plus the scattering offset from the vertex to the surface; mapped to global through each module's
local axes and differenced, the common scattering offset cancels and

    var(D_d) = sum_i [(d.u_i)^2 R_u(bX_i)^2 + (d.v_i)^2 R_v(bY_i)^2] + b^2 g_d / p^2,   d in (e_phi, e_2),

with u_i, v_i the modules' local axes (a disc module's u is 1-10 deg from e_phi, so R_v leaks into the
e_phi difference unless the rotation is in the model), e_2 = e_r on a disc and z on a barrel layer, and
b^2 g_d / p^2 the small leftover scattering between the two clusters: g_d is the pair's own geometric
factor (the path between the two module planes squared, the module's path factor for the thickness, and
the in-plane projection of the kink for direction d; it carries the eta dependence), b one scale per
surface. The fit is linear in (R_u^2 per size-X bin, R_v^2 per size-Y bin, b^2), one surface at a time;
each cell (direction, size-bin pair, p bin, eta slice, rotation bin) contributes its
robust width (MAD / 0.6745 about the cell's own median, v3's estimator) against the cell mean of the
pairs' coefficients. Errors: event bootstrap. Checks: the toy closure in the real pair geometry (T0), the
IL1 closure against the same clusters' truth-residual widths (D0), the disc ratio to v3's R at the pairs'
own (size, |cot|) cells (D2), the same- vs opposite-facing agreement and the effective leftover
B'_d = b sqrt(median g_d) < 0.02 cm GeV.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from common import OP, ORP
from ngtagger.spdefs import angles as SA

MAD_K = 0.6745
UM = 1e4                                            # cm -> um
P_EDGES = (10.0, 15.0, 20.0, 30.0, 50.0, np.inf)   # GeV, the cells' momentum bins; p < 10 GeV not used
ETA_SLICES = 3                                      # the cells' |eta| terciles (the leftover's g varies with eta)
OVERLAP_MAX_CM = 0.1                                # a geometric overlap: each cluster within 1 mm of its helix
                                                    # crossing, and the two residuals within 1 mm of each other
PT_MIN, D0_MAX_CM = 2.0, 0.005
B_PRIME_MAX = 0.02                                  # cm GeV: a 1.9 cm lever arm gives ~0.003-0.006
MIN_CELL = 20                                       # pairs per cell for a robust width
IRLS_ITER = 4                                       # reweighting passes: the weights use the model's variance
ROT_EDGES = (0.005, 0.02)                           # cells' rotation bins on the summed cross-axis coefficient
                                                    # (sin^2 of 4 and 8 deg): similar variances within a cell


def size_bin(size, edges):
    return np.clip(np.searchsorted(edges, size, side="right") - 1, 0, len(edges) - 2)


def v3_width(tables, axis, size, cot):
    """v3's R (cm) at each cluster's own (size, |cot|) cell (refit_calibration_fit.apply_R's lookup)."""
    t = tables[axis]
    W = np.asarray(t["width_cm"])
    return W[size_bin(size, t["size_edges"]), size_bin(np.abs(cot), t["cot_edges"])]


# ------------------------------------------------------------------ pairs
def pairs_of_chunk(ctx, geo, surfaces, start=0):
    """Overlap pairs of one chunk: per (TP, surface), the own class-T cluster nearest the TP's helix on each
    module; every two modules of the same surface whose clusters pass the overlap test form a pair.
    Returns per pair: surface, ev, p, eta, sizes, |truth cot| per cluster, the local residuals (du, dv),
    each module's axes in the surface plane, the common directions and the facings."""
    K, TP = ctx["K"], ctx["TP"]
    kpt = ORP.pt_constant(ctx["T"])
    mi, tp = K["mi"], K["tp_global"]
    ok = (mi >= 0) & (tp >= 0) & SA.class_T(K["tpEmuClass"])
    ok &= np.isin(geo["surface"][np.maximum(mi, 0)], surfaces)
    tpr = np.where(ok, tp, 0)
    pt, d0 = TP["pt"][tpr], TP["d0"][tpr]
    ok &= (pt >= PT_MIN) & (np.abs(d0) < D0_MAX_CM)
    ci = np.flatnonzero(ok)
    if not len(ci):
        return None
    t = tp[ci]
    m = mi[ci]
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
    cotA, cotB = K["tpLocalCotAlpha"][ci], K["tpLocalCotBeta"][ci]
    # one cluster per (TP, module): the one nearest the helix
    d = np.hypot(du, dv)
    o = np.lexsort((d, m, t))
    first = np.r_[True, (t[o][1:] != t[o][:-1]) | (m[o][1:] != m[o][:-1])]
    o = o[first]
    ci, t, m, S, du, dv, cotA, cotB = ci[o], t[o], m[o], S[o], du[o], dv[o], cotA[o], cotB[o]
    # pairs: same TP and surface, different modules
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
    i1, i2, g1, g2, e_phi, e_2 = i1[keep], i2[keep], g1[keep], g2[keep], e_phi[keep], e_2[keep]
    gpos = [geo["origin"][m[ii]] + K["localX"][ci[ii], None] * geo["ex"][m[ii]]
            + K["localY"][ci[ii], None] * geo["ey"][m[ii]] for ii in (i1, i2)]
    tl = np.abs(np.sinh(TP["eta"][t[i1]]))
    sec = np.sqrt(1.0 + tl * tl)
    # the leftover scattering's geometric factor per direction: path between the two module planes L, the
    # module's thickness path factor, the in-plane projection of a kink (PLAN rev 4 amendment A1)
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


# ------------------------------------------------------------------ the fit
def differences(P, rotation=True, drop_facing=None):
    """(D_phi, D_2) per pair and the design coefficients: the projections of each cluster's local axes on the
    two directions. rotation=False: the no-rotation known-bad (u on e_phi and v on e_2 exactly, no mixing).
    drop_facing = +-1: the known-bad with that facing's u sign dropped (its u axis flipped)."""
    co = {}
    G = []
    for j in ("1", "2"):
        ux, vx = P[f"ux{j}"].copy(), P[f"vx{j}"]
        if drop_facing is not None:
            ux = np.where((P[f"facing{j}"] == drop_facing)[:, None], -ux, ux)
        G.append(P[f"du{j}"][:, None] * ux + P[f"dv{j}"][:, None] * vx)
        for d, e in (("phi", P["e_phi"]), ("2", P["e_2"])):
            if rotation:
                co[(d, "u", j)], co[(d, "v", j)] = (ux * e).sum(1) ** 2, (vx * e).sum(1) ** 2
            else:
                co[(d, "u", j)] = np.full(len(e), 1.0 if d == "phi" else 0.0)
                co[(d, "v", j)] = np.full(len(e), 0.0 if d == "phi" else 1.0)
    D = G[0] - G[1]
    return {"phi": (D * P["e_phi"]).sum(1), "2": (D * P["e_2"]).sum(1)}, co


def fit_surface(P, tables, sel=None, rotation=True, drop_facing=None):
    """The joint u-v fit on one surface's pairs (sel). Returns R_u per size-X bin, R_v per size-Y bin
    (cm, NaN where unconstrained), B' per direction x eta slice (cm GeV), the cell medians, the cells used."""
    if sel is None:
        sel = np.ones(len(P["p"]), bool)
    sel = sel & (P["p"] >= P_EDGES[0])               # low-p pairs: the leftover scattering spreads within a cell
    Q = {k: v[sel] for k, v in P.items()}
    if not len(Q["p"]):
        return empty_fit(tables, sel)
    eu, ev_ = tables["u"]["size_edges"], tables["v"]["size_edges"]
    nu, nv = len(eu) - 1, len(ev_) - 1
    bx = [size_bin(Q[f"sx{j}"], eu) for j in ("1", "2")]
    by = [size_bin(Q[f"sy{j}"], ev_) for j in ("1", "2")]
    eta_cut = np.quantile(Q["eta"], np.linspace(0, 1, ETA_SLICES + 1)[1:-1]) if len(Q["eta"]) else []
    es = np.searchsorted(eta_cut, Q["eta"])
    pb = np.clip(np.searchsorted(P_EDGES, Q["p"], side="right") - 1, 0, len(P_EDGES) - 2)
    Dv, co = differences(Q, rotation, drop_facing)
    Dv = {d: x * UM for d, x in Dv.items()}                       # the fit runs in um (well-scaled parameters)
    npar = nu + nv + 1
    rows, obs, wts, meds = [], [], [], []
    for di, d in enumerate(("phi", "2")):
        dom = (bx if d == "phi" else by)
        lo, hi = np.minimum(dom[0], dom[1]), np.maximum(dom[0], dom[1])
        other = "v" if d == "phi" else "u"
        leak = co[(d, other, "1")] + co[(d, other, "2")]          # the rotation's cross-axis coefficient
        rb = np.searchsorted(ROT_EDGES, leak, side="right")
        cell = (((lo * 16 + hi) * 8 + pb) * 8 + es) * 8 + rb
        for c in np.unique(cell):
            k = cell == c
            n = int(k.sum())
            if n < MIN_CELL:
                continue
            x = Dv[d][k]
            med = np.median(x)
            w = np.median(np.abs(x - med)) / MAD_K * n / (n - 0.8)      # the small-sample MAD factor
            if not w > 0:
                continue
            A = np.zeros(npar)
            for j, jj in enumerate(("1", "2")):
                np.add.at(A, bx[j][k], co[(d, "u", jj)][k] / n)
                np.add.at(A, nu + by[j][k], co[(d, "v", jj)][k] / n)
            A[nu + nv] = np.mean(Q[f"g_{d}"][k] / Q["p"][k] ** 2) * UM * UM
            rows.append(A)
            obs.append(w * w)
            wts.append(n)
            meds.append(med)
    if not rows:
        return empty_fit(tables, sel)
    A, y, n_c = np.array(rows), np.array(obs), np.array(wts, np.float64)
    used = A.any(0)
    # sd(w^2) ~ model variance / sqrt(n): weighting by the observed w^2 would bias the widths low, so the
    # weights use the fitted model's variance, iterated from equal relative weights
    pred = y.copy()
    for _ in range(IRLS_ITER):
        W = np.sqrt(n_c) / np.maximum(pred, 1e-6 * np.median(y))
        r = lsq_linear(A[:, used] * W[:, None], y * W, bounds=(0, np.inf))
        pred = A[:, used] @ r.x
    th = np.full(npar, np.nan)
    th[used] = r.x
    bb = np.sqrt(th[nu + nv])
    return {"R_u": np.sqrt(th[:nu]) / UM, "R_v": np.sqrt(th[nu:nu + nv]) / UM, "b": bb,
            "B_prime": np.array([bb * np.sqrt(np.median(Q["g_phi"])), bb * np.sqrt(np.median(Q["g_2"]))]),
            "median_abs_max": float(np.max(np.abs(meds))) / UM,
            "n_cells": len(rows), "n_pairs": int(sel.sum()),
            "n_u": np.bincount(np.concatenate(bx), minlength=nu), "n_v": np.bincount(np.concatenate(by), minlength=nv)}


def empty_fit(tables, sel):
    """A fit with nothing constrained (no cell with enough pairs): NaN widths."""
    nu, nv = len(tables["u"]["size_edges"]) - 1, len(tables["v"]["size_edges"]) - 1
    return {"R_u": np.full(nu, np.nan), "R_v": np.full(nv, np.nan), "b": np.nan, "B_prime": np.full(2, np.nan),
            "median_abs_max": np.nan, "n_cells": 0, "n_pairs": int(np.sum(sel)), "n_u": np.zeros(nu, int),
            "n_v": np.zeros(nv, int)}


def bootstrap(P, tables, surface, reps, rotation=True, drop_facing=None):
    """The fit on one surface and its event-bootstrap replicas (rep: (n_rep, nev) multinomial weights)."""
    base = P["surface"] == surface
    out = [fit_surface(P, tables, base, rotation, drop_facing)]
    ev = P["ev"].astype(np.int64)
    for w in reps:
        k = np.repeat(np.flatnonzero(base), w[ev[base]])
        Q = {kk: v[k] for kk, v in P.items()}
        out.append(fit_surface(Q, tables, None, rotation, drop_facing))
    return out


# ------------------------------------------------------------------ references
def truth_width(P, tables, sel, axis):
    """The same clusters' truth-residual R per size bin (v3's estimator: MAD / 0.6745 about the median),
    pooling both clusters of every pair (IL1: scattering-free)."""
    e = tables[axis]["size_edges"]
    s = np.concatenate([P[f"s{'x' if axis == 'u' else 'y'}{j}"][sel] for j in ("1", "2")])
    r = np.concatenate([P[f"d{axis}{j}"][sel] for j in ("1", "2")])
    b = size_bin(s, e)
    out = np.full(len(e) - 1, np.nan)
    for i in range(len(e) - 1):
        x = r[b == i]
        if len(x) >= MIN_CELL:
            out[i] = np.median(np.abs(x - np.median(x))) / MAD_K
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


# ------------------------------------------------------------------ the toy closure (T0)
def toy(P, tables, sel, planted, rng, b_common=0.1, b_prime=0.004, rotation=True):
    """One toy replica on the real pairs (sel): each cluster's residual replaced by the planted per-size-bin
    (R_u, R_v) draw plus a common scattering offset (B = b_common / p, cancels) and a leftover relative
    offset with the pairs' own geometric factors (B'_phi = b_prime cm GeV at the median pair); returns
    the fit."""
    Q = {k: v[sel].copy() for k, v in P.items()}
    n = len(Q["p"])
    common = (b_common / Q["p"])[:, None] * rng.standard_normal((n, 2))
    for j in ("1", "2"):
        Ru = planted["u"][size_bin(Q[f"sx{j}"], tables["u"]["size_edges"])]
        Rv = planted["v"][size_bin(Q[f"sy{j}"], tables["v"]["size_edges"])]
        gc = common[:, 0, None] * Q["e_phi"] + common[:, 1, None] * Q["e_2"]          # the common offset, global
        Q[f"du{j}"] = rng.standard_normal(n) * Ru + (gc * Q[f"ux{j}"]).sum(1)
        Q[f"dv{j}"] = rng.standard_normal(n) * Rv + (gc * Q[f"vx{j}"]).sum(1)
    b_scale = b_prime / np.sqrt(np.median(Q["g_phi"]))          # the planted b: B'_phi = b_prime at the median
    rel = (b_scale / Q["p"])[:, None] * np.sqrt(np.stack([Q["g_phi"], Q["g_2"]], 1)) * rng.standard_normal((n, 2))
    g = rel[:, 0, None] * Q["e_phi"] + rel[:, 1, None] * Q["e_2"]
    Q["du2"] = Q["du2"] + (g * Q["ux2"]).sum(1)
    Q["dv2"] = Q["dv2"] + (g * Q["vx2"]).sum(1)
    return fit_surface(Q, tables, None, rotation)


def planted_widths(tables):
    """The planted R per size bin: v3's IL1 widths, the median over the |cot| cells of each size row."""
    return {a: np.median(np.asarray(tables[a]["width_cm"]), axis=1) for a in ("u", "v")}
