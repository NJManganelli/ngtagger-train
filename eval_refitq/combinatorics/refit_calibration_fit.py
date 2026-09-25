"""Fit the pieces of an honest refit covariance, in the order they can be isolated.

Input: refit_calibration.extract() records (target cluster vs the UN-UPDATED
OT-only projection, perfect tracks). Run with the ngtagger-train pixi env
(scipy).

MODEL for the covariance of the no-update target residual (u, v) at layer k:

    Sigma_k = J (D C D) J^T  +  sum_s theta_s^2 J (jT_s jT_s^T + jL_s jL_s^T) J^T  +  R

  D       diag(d_rInv, d_phi, d_tanL, d_z0, d_d0): the OT seed covariance scale,
          correlations preserved.                                   (item 1)
  s       every scatterer the extrapolation from the OT to layer k crosses: the
          OT->IT gap plane at r_gap, and each IT layer OUTSIDE k (a kink at the
          layer's own radius has no lever arm on its own position). The kink
          parametrisation is the producer's Q: a kink theta at radius r_s is
          jT = (0, 1, 0, 0, -r_s), jL = (0, 0, sec^2, -r_s sec^2, 0).
          theta_s from Highland, 13.6 MeV / p * sqrt(X) (1 + 0.038 ln X),
          X = (x/X0)_s * sqrt(1 + tanL^2) (path length through a barrel
          cylinder), p = pT sqrt(1 + tanL^2).                          (item 3)
  R       the cluster's own position errors, calibrated separately.   (item 2)

Only physical parameters are fitted: five seed scales, x/X0 of an IT layer (one
value for all four), x/X0 and radius of the gap plane. No per-layer scale.
"""
from __future__ import annotations

import numpy as np

KINK_COEF = 0.0136          # GeV, Highland
IT_R = None                 # filled from the data: median module radius per layer


def load(path):
    D = dict(np.load(path))
    global IT_R
    IT_R = {L: float(np.median(D["r_mod"][D["layer"] == L])) for L in (1, 2, 3, 4)}
    return D


def rs(x):
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan, np.nan, 0
    m = np.median(x)
    return float(1.4826 * np.median(np.abs(x - m))), float(np.std(x)), len(x)


def highland_theta2(xx0, p, tanl):
    X = np.maximum(xx0 * np.sqrt(1.0 + tanl ** 2), 1e-12)
    th = KINK_COEF / p * np.sqrt(X) * (1.0 + 0.038 * np.log(X))
    return np.clip(th, 0, None) ** 2


AZIMUTHAL_SEC = True   # the producer's Q omits it; see kink_cov


def kink_cov(J, r_s, tanl, theta2):
    """theta^2 J (jT jT^T + jL jL^T) J^T for a kink at radius r_s, (n,4,4).

    A spatial kink theta perpendicular to the track changes the AZIMUTH by
    theta / cos(lambda) -- the transverse momentum is only p cos(lambda) -- and
    the dip by theta, i.e. tanL by theta sec^2(lambda). The producer's Q uses
    jT = (0, 1, 0, 0, -r_s), which is wrong TWICE: it omits the sec(lambda)
    (transverse scattering low by sqrt(1 + tanL^2), 1.9x at |tanL| = 1.6), and
    its d0 term has the WRONG SIGN. With the TTTrack convention, POCA =
    (d0 sin phi, -d0 cos phi), rotating phi by dphi about the point at r_s needs
    d0 -> d0 + r_s dphi to keep passing through it. Verified numerically: with
    -r_s a kink at IL4 moves the IL4 crossing itself by a median 33 um and IL2
    by 23.6 um against 9.6 um of lever arm; with +r_s, 0.06 um and 9.8 um. The
    longitudinal term (-r_s sec^2 in z0) is correct.
    """
    n = len(J)
    sec2 = 1.0 + tanl ** 2
    sec = np.sqrt(sec2) if AZIMUTHAL_SEC else np.ones(n)
    jT = np.zeros((n, 5)); jT[:, 1] = sec; jT[:, 4] = +r_s * sec
    jL = np.zeros((n, 5)); jL[:, 2] = sec2; jL[:, 3] = -r_s * sec2
    a = np.einsum("nij,nj->ni", J, jT)
    b = np.einsum("nij,nj->ni", J, jL)
    return theta2[:, None, None] * (a[:, :, None] * a[:, None, :] + b[:, :, None] * b[:, None, :])


def model_cov(D, dscale, xx0_it=0.0, xx0_gap=0.0, r_gap=19.0, rscale=(1, 1, 1, 1)):
    """Sigma (n,4,4) of the no-update residual under a trial model."""
    J, C, lay = D["J"], D["C"], D["layer"]
    tanl = D["tanL"].astype(np.float64)
    p = D["pt"].astype(np.float64) * np.sqrt(1.0 + tanl ** 2)
    Dd = np.asarray(dscale, np.float64)
    Cs = C * Dd[None, :, None] * Dd[None, None, :]
    S = np.einsum("nij,njk,nlk->nil", J, Cs, J)
    if xx0_gap > 0:
        S += kink_cov(J, np.full(len(J), r_gap), tanl, highland_theta2(xx0_gap, p, tanl))
    if xx0_it > 0:
        th2 = highland_theta2(xx0_it, p, tanl)
        for Ls in (2, 3, 4):                  # scatterer layer; affects layers inside it
            m = lay < Ls
            if m.any():
                S[m] += kink_cov(J[m], np.full(m.sum(), IT_R[Ls]), tanl[m], th2[m])
    sig = D["sig"] * np.asarray(rscale)[None, :]
    S[:, [0, 1, 2, 3], [0, 1, 2, 3]] += np.nan_to_num(sig) ** 2
    return S


def pos_pulls(D, S):
    """Per-coordinate pulls and the 2D Mahalanobis distance in (u, v)."""
    r = D["res"][:, :2]
    pu = r[:, 0] / np.sqrt(S[:, 0, 0])
    pv = r[:, 1] / np.sqrt(S[:, 1, 1])
    Suv = S[:, :2, :2]
    det = Suv[:, 0, 0] * Suv[:, 1, 1] - Suv[:, 0, 1] ** 2
    m2 = (Suv[:, 1, 1] * r[:, 0] ** 2 - 2 * Suv[:, 0, 1] * r[:, 0] * r[:, 1]
          + Suv[:, 0, 0] * r[:, 1] ** 2) / det
    return pu, pv, m2


def nll(D, S, sel, clip=None):
    """Gaussian NLL of the (u, v) residuals; optionally clip m2 (robustness)."""
    r = D["res"][sel, :2]
    Suv = S[sel][:, :2, :2]
    det = Suv[:, 0, 0] * Suv[:, 1, 1] - Suv[:, 0, 1] ** 2
    m2 = (Suv[:, 1, 1] * r[:, 0] ** 2 - 2 * Suv[:, 0, 1] * r[:, 0] * r[:, 1]
          + Suv[:, 0, 0] * r[:, 1] ** 2) / det
    if clip is not None:
        m2 = np.minimum(m2, clip)
    return 0.5 * float(np.sum(m2 + np.log(np.maximum(det, 1e-300))))


PT_BINS = [2, 3, 5, 10, 20, 1e9]
TANL_BINS = [0, 0.4, 0.8, 1.2, 1.6, 10]


def pull_table(D, S, base=None):
    """Robust pull widths per layer x pT and per layer x |tanL|, plus 4-sigma containment."""
    base = np.ones(len(D["layer"]), bool) if base is None else base
    pu, pv, m2 = pos_pulls(D, S)
    out = {}
    for L in (4, 3, 2, 1):
        ml = base & (D["layer"] == L)
        row = {"n": int(ml.sum()), "u": rs(pu[ml])[:2], "v": rs(pv[ml])[:2],
               "cont4": float(np.mean(m2[ml] < 16)) if ml.any() else np.nan}
        for lab, key, edges in (("pt", "pt", PT_BINS), ("tanL", "tanL", TANL_BINS)):
            x = np.abs(D[key]) if key == "tanL" else D[key]
            row[lab] = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mb = ml & (x >= lo) & (x < hi)
                row[lab].append((lo, rs(pu[mb])[0], rs(pv[mb])[0], int(mb.sum()),
                                 float(np.mean(m2[mb] < 16)) if mb.any() else np.nan))
        out[f"IL{L}"] = row
    return out


def print_table(tab, title):
    print(f"--- {title}")
    for L, row in tab.items():
        print(f"  {L}: n {row['n']:6d}  pull u MAD/RMS {row['u'][0]:.3f}/{row['u'][1]:.2f}  "
              f"v {row['v'][0]:.3f}/{row['v'][1]:.2f}  4sig-cont {row['cont4']:.4f}")
        print("       pT  " + "  ".join(f"[{lo:g},{'inf' if hi > 1e8 else f'{hi:g}'}) u{mu:.2f} v{mv:.2f} c{c:.3f}"
                                for (lo, mu, mv, n, c), hi in zip(row["pt"], PT_BINS[1:])))
        print("     |tanL| " + "  ".join(f"[{lo:g},{hi:g}) u{mu:.2f} v{mv:.2f} c{c:.3f}"
                                for (lo, mu, mv, n, c), hi in zip(row["tanL"], TANL_BINS[1:])))


# ---- item 2: the cluster's own errors against truth ------------------------------
BETA_EDGES = [0, 0.3, 0.6, 1.07, 1.5, 2.5, 6.0]   # 1.07 = PixelAV coverage edge


def angle_R_table(D, ptmin=5.0):
    """(measured - true) / sig for cotAlpha and cotBeta, by layer x |true cotBeta|.

    Truth is the TP helix at the module; scattering moves the direction by
    ~1e-4 above 5 GeV, far below sig ~0.02, so this measures R itself.
    """
    out = {}
    tr = D["truth"]
    for k, nm, has in ((2, "cotAlpha", D["hasA"]), (3, "cotBeta", D["hasB"])):
        pull = (D["meas"][:, k] - tr[:, k]) / D["sig"][:, k]
        ok = has & (D["pt"] >= ptmin) & np.isfinite(pull) & (D["sig"][:, k] > 0)
        rows = {}
        for L in (1, 2, 3, 4):
            ml = ok & (D["layer"] == L)
            b = np.abs(tr[:, 3])
            rows[f"IL{L}"] = [(lo, hi) + rs(pull[ml & (b >= lo) & (b < hi)])
                              for lo, hi in zip(BETA_EDGES[:-1], BETA_EDGES[1:])]
        out[nm] = rows
    return out


def position_R_table(D, pt_bins=(5, 10, 20, 50, 1e9)):
    """(measured - true) / sig for u and v, by layer x pT. The truth helix has no
    scattering, so the width falls with p toward the CPE-only value; the
    highest bin (or the 1/p^2 extrapolation) is what R should reproduce."""
    tr = D["truth"]
    out = {}
    for k, nm in ((0, "u"), (1, "v")):
        pull = (D["meas"][:, k] - tr[:, k]) / D["sig"][:, k]
        rows = {}
        for L in (1, 2, 3, 4):
            ml = (D["layer"] == L) & np.isfinite(pull)
            rows[f"IL{L}"] = [(lo, hi) + rs(pull[ml & (D["pt"] >= lo) & (D["pt"] < hi)])
                              for lo, hi in zip(pt_bins[:-1], pt_bins[1:])]
        out[nm] = rows
    return out


# ---- item 2: calibrated measurement errors R -------------------------------------
# The published errors are near-constant (u 2.5-3.5 um, v 12.5 um) and blind to
# the cluster: against truth at IL1 (prompt, pT > 5 GeV, scattering < 1 um),
# sizeX >= 5 clusters sit 100-200 sigma away, sizeY 1-2 fragments of long
# clusters 17-28 sigma, and even well-formed ones 1.1-2.7x their error. So R is
# a LOOKUP, not a scale: position error by (cluster size, |predicted cot|),
# angle-error SCALE by |predicted cotBeta| (the PixelAV coverage edge at 1.07
# is where beta goes 1.7x). Built at IL1 where the truth helix is clean; bins
# use the PREDICTED angle (online-available, not fooled by broken clusters).
R_BINS = {
    "u": {"size": [1, 2, 3, 4, 5, 99], "cot": [0.0, 0.05, 0.1, 0.2, 0.4, 99.0]},
    "v": {"size": [1, 2, 3, 5, 7, 9, 12, 99], "cot": [0.0, 0.3, 0.6, 1.07, 1.5, 2.5, 99.0]},
    "a": {"cot": [0.0, 0.3, 0.6, 1.07, 1.5, 2.5, 99.0]},
    "b": {"cot": [0.0, 0.3, 0.6, 1.07, 1.5, 2.5, 99.0]},
}
R_MIN_N = 40


def _bin(x, edges):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def primary_mask(D):
    """Per (track, layer), the target-TP cluster NEAREST the true crossing.

    The target TP can own several clusters on a layer that are not its
    crossing: delta-ray deposits (linked back to the parent particle) and
    second passages, measured mm to cm from the helix. They are not what a
    refit should include and they must not set R. Distance in pitch units.
    """
    d = np.hypot((D["meas"][:, 0] - D["truth"][:, 0]) / 0.0025,
                 (D["meas"][:, 1] - D["truth"][:, 1]) / 0.0100)
    key = D["track"].astype(np.int64) * 8 + D["layer"].astype(np.int64)
    o = np.lexsort((d, key))
    first = np.r_[True, key[o][1:] != key[o][:-1]]
    m = np.zeros(len(key), bool)
    m[o[first]] = True
    return m


def build_R_tables(D, layer=1, ptmin=5.0, d0max=0.005):
    """Robust (MAD) widths of the truth residual, per bin, PRIMARY clusters only.
    Position tables are ABSOLUTE widths [cm]; angle tables are SCALES on the
    published sigma."""
    tr = D["truth"]
    base = (D["layer"] == layer) & (D["pt"] >= ptmin) & (np.abs(D["d0"]) < d0max) & primary_mask(D)
    res = D["meas"] - tr
    out = {"built_from": f"IL{layer}, pT >= {ptmin} GeV, |tp_d0| < {d0max * 1e4:.0f} um, perfect tracks"}
    for k, nm, szk, cotk in ((0, "u", "sizeX", 2), (1, "v", "sizeY", 3)):
        e_s, e_c = R_BINS[nm]["size"], R_BINS[nm]["cot"]
        tab = np.full((len(e_s) - 1, len(e_c) - 1), np.nan)
        cnt = np.zeros_like(tab, int)
        bs, bc = _bin(D[szk], e_s), _bin(np.abs(tr[:, cotk]), e_c)
        for i in range(len(e_s) - 1):
            for j in range(len(e_c) - 1):
                m = base & (bs == i) & (bc == j)
                cnt[i, j] = m.sum()
                if m.sum() >= R_MIN_N:
                    tab[i, j] = rs(res[m, k])[0]
        # sparse cells: fall back to the size row's pooled width
        for i in range(len(e_s) - 1):
            m = base & (bs == i)
            row = rs(res[m, k])[0] if m.sum() >= R_MIN_N else np.nanmax(tab)
            tab[i] = np.where(np.isfinite(tab[i]), tab[i], row)
        out[nm] = {"size_edges": e_s, "cot_edges": e_c, "width_cm": tab.tolist(), "n": cnt.tolist()}
    for k, nm, has in ((2, "a", D["hasA"]), (3, "b", D["hasB"])):
        e_c = R_BINS[nm]["cot"]
        bc = _bin(np.abs(tr[:, 3]), e_c)
        pull = res[:, k] / D["sig"][:, k]
        sc, cnt = [], []
        for j in range(len(e_c) - 1):
            m = base & has & (bc == j) & np.isfinite(pull)
            cnt.append(int(m.sum()))
            sc.append(rs(pull[m])[0] if m.sum() >= R_MIN_N else np.nan)
        sc = np.array(sc)
        sc = np.where(np.isfinite(sc), sc, np.nanmax(sc))
        out[nm] = {"cot_edges": e_c, "scale": sc.tolist(), "n": cnt}
    return out


def apply_R(tables, sizeX, sizeY, cotA_pred, cotB_pred, sig):
    """Calibrated (sigX, sigY, sigA, sigB) from the lookups; sig is the published (n,4)."""
    out = sig.astype(np.float64).copy()
    for k, nm, sz, cot in ((0, "u", sizeX, cotA_pred), (1, "v", sizeY, cotB_pred)):
        t = tables[nm]
        W = np.asarray(t["width_cm"])
        out[:, k] = W[_bin(sz, t["size_edges"]), _bin(np.abs(cot), t["cot_edges"])]
    for k, nm in ((2, "a"), (3, "b")):
        t = tables[nm]
        out[:, k] = sig[:, k] * np.asarray(t["scale"])[_bin(np.abs(cotB_pred), t["cot_edges"])]
    return out


# ---- item 1: the OT seed covariance scale D --------------------------------------
def _pair_terms(D, sel):
    """T[n, i, j, a, b] = J_ai C_ij J_bj over position rows a, b in (u, v), so a
    trial Sigma = sum_ij d_i d_j T_ij + (extra) is a cheap contraction."""
    J = D["J"][sel][:, :2, :]
    C = D["C"][sel]
    return np.einsum("nai,nij,nbj->nijab", J, C, J)


def fit_seed_scale(D, sel, R_uv, extra=None, clip=25.0, x0=None):
    """Maximum likelihood (u, v) fit of log d_i; m2 clipped at `clip` so the
    residual tails (secondary-like clusters, merges) cannot drive the scales."""
    from scipy.optimize import minimize
    T = _pair_terms(D, sel)
    r = D["res"][sel, :2]
    add = np.zeros((sel.sum(), 2, 2))
    add[:, 0, 0] += R_uv[:, 0] ** 2
    add[:, 1, 1] += R_uv[:, 1] ** 2
    if extra is not None:
        add += extra

    def f(ld):
        d = np.exp(ld)
        S = np.einsum("i,j,nijab->nab", d, d, T) + add
        det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] ** 2
        m2 = (S[:, 1, 1] * r[:, 0] ** 2 - 2 * S[:, 0, 1] * r[:, 0] * r[:, 1] + S[:, 0, 0] * r[:, 1] ** 2) / det
        return 0.5 * np.sum(np.minimum(m2, clip) + np.log(np.maximum(det, 1e-300)))
    res = minimize(f, np.zeros(5) if x0 is None else np.log(x0), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-3})
    return np.exp(res.x), res


# ---- items 1 + 3 jointly: seed floor F and the material ----------------------------
# A multiplicative scale on the OT covariance fixes high pT and over-inflates low
# pT (measured: IL4 u pull 0.97 at > 20 GeV but 0.59 at 2-3 GeV): the OT fit
# lacks a pT-INDEPENDENT floor. The floor (flat in pT) and scattering (1/p^2)
# separate through the pT dependence, so they are fitted together.
def joint_terms(D, sel, r_gap):
    J = D["J"][sel][:, :2, :]
    tanl = D["tanL"][sel].astype(np.float64)
    lay = D["layer"][sel]
    base = np.einsum("nai,nij,nbj->nab", J, D["C"][sel], J)
    fl = np.einsum("nai,nbi->niab", J, J)                     # floor shapes, per parameter
    Jf = D["J"][sel]
    unit = np.ones(len(tanl))
    gap = kink_cov(Jf, np.full(len(tanl), r_gap), tanl, unit)[:, :2, :2]
    it = np.zeros_like(gap)                                   # sum over outer IT layers
    for Ls in (2, 3, 4):
        m = lay < Ls
        if m.any():
            it[m] += kink_cov(Jf[m], np.full(m.sum(), IT_R[Ls]), tanl[m], unit[m])[:, :2, :2]
    p = D["pt"][sel].astype(np.float64) * np.sqrt(1.0 + tanl ** 2)
    return {"base": base, "floor": fl, "gap": gap, "it": it, "p": p, "tanl": tanl,
            "r": D["res"][sel, :2]}


FLOOR_REF = np.array([1e-5, 1e-3, 1e-3, 0.03, 0.02])   # conditioning units per parameter


def joint_cov(t, params, R_uv):
    f = np.exp(params[:5]) * FLOOR_REF
    x_it, x_gap = np.exp(params[5]), np.exp(params[6])
    S = t["base"] + np.einsum("i,niab->nab", f ** 2, t["floor"])
    S = S + highland_theta2(x_gap, t["p"], t["tanl"])[:, None, None] * t["gap"]
    S = S + highland_theta2(x_it, t["p"], t["tanl"])[:, None, None] * t["it"]
    S[:, 0, 0] += R_uv[:, 0] ** 2
    S[:, 1, 1] += R_uv[:, 1] ** 2
    return S


def fit_joint(D, sel, R_uv, r_gap, clip=25.0, x0=None):
    from scipy.optimize import minimize
    t = joint_terms(D, sel, r_gap)
    r = t["r"]

    def f(pp):
        S = joint_cov(t, pp, R_uv)
        det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] ** 2
        m2 = (S[:, 1, 1] * r[:, 0] ** 2 - 2 * S[:, 0, 1] * r[:, 0] * r[:, 1] + S[:, 0, 0] * r[:, 1] ** 2) / det
        return 0.5 * np.sum(np.minimum(m2, clip) + np.log(np.maximum(det, 1e-300)))
    p0 = np.r_[np.zeros(5), np.log(0.01), np.log(0.02)] if x0 is None else x0
    res = minimize(f, p0, method="Powell", options={"maxiter": 20000, "xtol": 1e-3, "ftol": 1e-4})
    return res, t


def joint_model_full(D, params, r_gap, R_all):
    """Full 4x4 Sigma for every record, for pull tables (same model as fit_joint)."""
    f = np.exp(params[:5]) * FLOOR_REF
    x_it, x_gap = np.exp(params[5]), np.exp(params[6])
    J, tanl = D["J"], D["tanL"].astype(np.float64)
    p = D["pt"].astype(np.float64) * np.sqrt(1.0 + tanl ** 2)
    S = np.einsum("nij,njk,nlk->nil", J, D["C"] + np.diag(f ** 2)[None], J)
    S += kink_cov(J, np.full(len(J), r_gap), tanl, highland_theta2(x_gap, p, tanl))
    th2 = highland_theta2(x_it, p, tanl)
    for Ls in (2, 3, 4):
        m = D["layer"] < Ls
        S[m] += kink_cov(J[m], np.full(m.sum(), IT_R[Ls]), tanl[m], th2[m])
    S[:, [0, 1, 2, 3], [0, 1, 2, 3]] += np.nan_to_num(R_all) ** 2
    return S


# ---- v2: per-layer material, seed floor per OT-track group --------------------------
# v1 (one x/X0 for every IT layer, one floor set) over-covers IL3 (u pull 0.73) and
# under-covers IL1 (1.07-1.18): one material value put too much at IL4. v1 also
# leaves v rising with |tanL| at IL4, where only the seed acts, and u differing
# between OT tracks with and without disk stubs. So:
#   * x/X0 per IT layer IL2, IL3, IL4 (IL1's own kink never enters an
#     extrapolation). Identifiable here because these are NO-UPDATE
#     extrapolations: there is no visit order to confuse with material, which
#     is what forbade per-layer Q scales in the KF.
#   * a seed floor per OT-track group, since the OT fit's quality depends on
#     |eta| and on which layers/disks it used.
FLOOR_GROUPS = (("barrel |tanL|<0.8", False, 0.0, 0.8), ("barrel 0.8-1.2", False, 0.8, 1.2),
                ("barrel >=1.2", False, 1.2, 99.0), ("disk |tanL|<1.6", True, 0.0, 1.6),
                ("disk >=1.6", True, 1.6, 99.0))
N_GROUP = len(FLOOR_GROUPS)
MAT_LAYERS = (2, 3, 4)


def floor_group(D, sel=None):
    t = np.abs(D["tanL"] if sel is None else D["tanL"][sel])
    disk = (D["n_disk_stubs"] if sel is None else D["n_disk_stubs"][sel]) > 0
    g = np.full(len(t), -1, np.int64)
    for i, (_, dk, lo, hi) in enumerate(FLOOR_GROUPS):
        g[(disk == dk) & (t >= lo) & (t < hi)] = i
    return g


def joint_terms_v2(D, sel, r_gap):
    J = D["J"][sel][:, :2, :]
    Jf = D["J"][sel]
    tanl = D["tanL"][sel].astype(np.float64)
    lay = D["layer"][sel]
    unit = np.ones(len(tanl))
    it = np.zeros((len(tanl), len(MAT_LAYERS), 2, 2))
    for k, Ls in enumerate(MAT_LAYERS):
        m = lay < Ls
        if m.any():
            it[m, k] = kink_cov(Jf[m], np.full(m.sum(), IT_R[Ls]), tanl[m], unit[m])[:, :2, :2]
    return {"base": np.einsum("nai,nij,nbj->nab", J, D["C"][sel], J),
            "floor": np.einsum("nai,nbi->niab", J, J),
            "gap": kink_cov(Jf, np.full(len(tanl), r_gap), tanl, unit)[:, :2, :2],
            "it": it, "g": floor_group(D, sel), "tanl": tanl,
            "p": D["pt"][sel].astype(np.float64) * np.sqrt(1.0 + tanl ** 2),
            "r": D["res"][sel, :2]}


def unpack_v2(params):
    fl = (np.exp(params[:5 * N_GROUP]).reshape(N_GROUP, 5)) * FLOOR_REF[None]
    xx0 = np.exp(params[5 * N_GROUP:5 * N_GROUP + len(MAT_LAYERS)])
    return fl, xx0, float(np.exp(params[-1]))


def joint_cov_v2(t, params, R_uv):
    fl, xx0, xgap = unpack_v2(params)
    S = t["base"] + np.einsum("ni,niab->nab", fl[t["g"]] ** 2, t["floor"])
    S = S + highland_theta2(xgap, t["p"], t["tanl"])[:, None, None] * t["gap"]
    for k in range(len(MAT_LAYERS)):
        S = S + highland_theta2(xx0[k], t["p"], t["tanl"])[:, None, None] * t["it"][:, k]
    S[:, 0, 0] += R_uv[:, 0] ** 2
    S[:, 1, 1] += R_uv[:, 1] ** 2
    return S


def fit_joint_v2(D, sel, R_uv, r_gap, clip=25.0, x0=None):
    from scipy.optimize import minimize
    sel = sel & (floor_group(D) >= 0)
    t = joint_terms_v2(D, sel, r_gap)
    r = t["r"]
    Ruv = R_uv[sel] if len(R_uv) == len(D["layer"]) else R_uv

    def f(pp):
        S = joint_cov_v2(t, pp, Ruv)
        det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] ** 2
        m2 = (S[:, 1, 1] * r[:, 0] ** 2 - 2 * S[:, 0, 1] * r[:, 0] * r[:, 1] + S[:, 0, 0] * r[:, 1] ** 2) / det
        return 0.5 * np.sum(np.minimum(m2, clip) + np.log(np.maximum(det, 1e-300)))
    p0 = (np.r_[np.zeros(5 * N_GROUP), np.log(np.full(len(MAT_LAYERS), 0.02)), np.log(0.01)]
          if x0 is None else x0)
    res = minimize(f, p0, method="Powell", options={"maxiter": 60000, "xtol": 1e-3, "ftol": 1e-5})
    return res


def model_full_v2(D, params, r_gap, R_all):
    fl, xx0, xgap = unpack_v2(params)
    g = floor_group(D)
    J, tanl = D["J"], D["tanL"].astype(np.float64)
    p = D["pt"].astype(np.float64) * np.sqrt(1.0 + tanl ** 2)
    F2 = np.zeros((len(J), 5, 5))
    ok = g >= 0
    for i in range(5):
        F2[ok, i, i] = fl[g[ok], i] ** 2
    S = np.einsum("nij,njk,nlk->nil", J, D["C"] + F2, J)
    S += kink_cov(J, np.full(len(J), r_gap), tanl, highland_theta2(xgap, p, tanl))
    for k, Ls in enumerate(MAT_LAYERS):
        m = D["layer"] < Ls
        S[m] += kink_cov(J[m], np.full(m.sum(), IT_R[Ls]), tanl[m], highland_theta2(xx0[k], p[m], tanl[m]))
    S[:, [0, 1, 2, 3], [0, 1, 2, 3]] += np.nan_to_num(R_all) ** 2
    return S


# ---- v3: fit to the KF INNOVATION likelihood (oracle clusters, visit order) --------
# No-update residuals cannot tell a seed error from a material kink at IL1: both
# are an IL1-only excess (v2 put 23% X0 "at IL2" to absorb it). Updates CAN: an
# IT update removes seed error, while a kink is charged again after it. So the
# parameters are fitted to every innovation of a LINEAR Kalman filter run over
# each perfect track's primary target clusters in visit order IL4 -> IL1, with
# the recorded residual-vs-seed and Jacobian (H is a property of the module and
# barely moves with the state). Scattering is charged along the PHYSICAL
# trajectory: before predicting layer L, the kinks of every IT layer j with
# L < j <= (previously visited layer), and the gap before the first -- whether
# or not the previous layer had a target to update on.
def pack_tracks(D, sel, R_all):
    """Per-track arrays over layer slots 1..4 (index L-1): res, H, R, masks."""
    tk = D["track"][sel]
    ut, inv = np.unique(tk, return_inverse=True)
    n = len(ut)
    lay = D["layer"][sel].astype(np.int64) - 1
    P = {"n": n, "have": np.zeros((n, 4), bool), "res": np.zeros((n, 4, 4)),
         "H": np.zeros((n, 4, 4, 5)), "R": np.ones((n, 4, 4)), "dimok": np.zeros((n, 4, 4), bool)}
    ii = np.flatnonzero(sel)
    P["have"][inv, lay] = True
    P["res"][inv, lay] = np.nan_to_num(D["res"][ii])
    P["H"][inv, lay] = D["J"][ii]
    P["R"][inv, lay] = np.nan_to_num(R_all[ii], nan=1.0)
    ok = np.isfinite(D["res"][ii]) & np.isfinite(R_all[ii]) & (R_all[ii] > 0)
    ok[:, 2] &= D["hasA"][ii]
    ok[:, 3] &= D["hasB"][ii]
    P["dimok"][inv, lay] = ok
    first = np.zeros(len(ut), np.int64)
    first[inv[::-1]] = ii[::-1]
    P["C"] = D["C"][first]
    P["tanl"] = D["tanL"][first].astype(np.float64)
    P["p"] = D["pt"][first].astype(np.float64) * np.sqrt(1.0 + P["tanl"] ** 2)
    P["g"] = floor_group(D)[first]
    keep = P["g"] >= 0
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in P.items()} | {"n": int(keep.sum())}


def _kink_param_cov(r_s, tanl, theta2):
    sec2 = 1.0 + tanl ** 2
    sec = np.sqrt(sec2)
    n = len(tanl)
    jT = np.zeros((n, 5)); jT[:, 1] = sec; jT[:, 4] = +r_s * sec     # sign: see kink_cov
    jL = np.zeros((n, 5)); jL[:, 2] = sec2; jL[:, 3] = -r_s * sec2
    return theta2[:, None, None] * (jT[:, :, None] * jT[:, None, :] + jL[:, :, None] * jL[:, None, :])


def kf_innovations(P, params, r_gap, clip=25.0, return_pulls=False):
    """Total (u, v) innovation NLL of the linear oracle KF; optionally the pulls."""
    fl, xx0, xgap = unpack_v2(params)
    n = P["n"]
    X = {2: xx0[0], 3: xx0[1], 4: xx0[2]}
    Cst = P["C"].copy()
    Cst[:, np.arange(5), np.arange(5)] += fl[P["g"]] ** 2
    dlt = np.zeros((n, 5))
    Lv = 5
    tot = 0.0
    pulls = []
    for L in (4, 3, 2, 1):
        if Lv == 5 and xgap > 0:
            Cst += _kink_param_cov(np.full(n, r_gap), P["tanl"], highland_theta2(xgap, P["p"], P["tanl"]))
        for j in range(L + 1, min(Lv, 4) + 1):
            Cst += _kink_param_cov(np.full(n, IT_R[j]), P["tanl"], highland_theta2(X.get(j, 0.0), P["p"], P["tanl"]))
        Lv = L
        h = P["have"][:, L - 1]
        if not h.any():
            continue
        H = P["H"][h, L - 1]
        y = P["res"][h, L - 1] - np.einsum("nij,nj->ni", H, dlt[h])
        S = np.einsum("nij,njk,nlk->nil", H, Cst[h], H)
        S[:, np.arange(4), np.arange(4)] += P["R"][h, L - 1] ** 2
        Suv = S[:, :2, :2]
        det = Suv[:, 0, 0] * Suv[:, 1, 1] - Suv[:, 0, 1] ** 2
        m2 = (Suv[:, 1, 1] * y[:, 0] ** 2 - 2 * Suv[:, 0, 1] * y[:, 0] * y[:, 1] + Suv[:, 0, 0] * y[:, 1] ** 2) / det
        tot += 0.5 * np.sum(np.minimum(m2, clip) + np.log(np.maximum(det, 1e-300)))
        if return_pulls:
            pulls.append({"L": L, "pu": y[:, 0] / np.sqrt(S[:, 0, 0]), "pv": y[:, 1] / np.sqrt(S[:, 1, 1]),
                          "m2": m2, "pt": P["p"][h] / np.sqrt(1 + P["tanl"][h] ** 2), "tanl": P["tanl"][h],
                          "su": np.sqrt(S[:, 0, 0])})
        # update on the dimensions this cluster measures (angles only where present)
        dm = P["dimok"][h, L - 1]
        Sm = S.copy()
        Sm[~np.repeat(dm[:, :, None], 4, 2) | ~np.repeat(dm[:, None, :], 4, 1)] = 0.0
        Sm[:, np.arange(4), np.arange(4)] = np.where(dm, Sm[:, np.arange(4), np.arange(4)], 1.0)
        ym = np.where(dm, y, 0.0)
        Hm = np.where(dm[:, :, None], H, 0.0)
        CHt = np.einsum("nij,nkj->nik", Cst[h], Hm)
        Kg = np.einsum("nik,nkl->nil", CHt, np.linalg.inv(Sm))
        dlt[h] += np.einsum("nik,nk->ni", Kg, ym)
        Cst[h] -= np.einsum("nik,njk->nij", Kg, CHt)
    return (tot, pulls) if return_pulls else tot


def fit_innovations(P, r_gap, x0=None, clip=25.0):
    from scipy.optimize import minimize
    p0 = (np.r_[np.zeros(5 * N_GROUP), np.log(np.full(len(MAT_LAYERS), 0.02)), np.log(0.01)]
          if x0 is None else x0)
    return minimize(lambda pp: kf_innovations(P, pp, r_gap, clip), p0, method="Powell",
                    options={"maxiter": 60000, "xtol": 1e-3, "ftol": 1e-5})


# ---- v4: PHYSICAL material from tkLayout, fitted seed floors ---------------------
# Phase-2 BASELINE material (tkLayout www/OT807_IT944: modulesMaterials_* for one
# module crossing, allVolumesMaterials_* for full-phi cylinders spanning z = 0),
# x/X0 at normal incidence. NOT a SmartPixels design (tkLayout's SPIX geometries
# are a trigger-efficiency kludge of the baseline). Every plane is a scatterer at
# its own radius; IL1's module is omitted because its kink never enters an
# extrapolation onto IL1..IL4. Limitation: the two support cylinders end at
# |z| = 20.1 / 22.7 cm; forward tracks that miss them cross flanges/disks that
# are not modelled here, and every track is charged the cylinders.
TKL_PLANES = (   # (radius cm, x/X0, what)
    (4.17, 0.0009, "TBPX HV lines outside L1"),
    ("IL2", 0.0187, "TBPX L2 module (incl. ladder cooling)"),
    (7.29, 0.0010, "TBPX HV lines outside L2"),
    ("IL3", 0.0148, "TBPX L3 module"),
    (11.68, 0.0008, "TBPX HV lines outside L3"),
    ("IL4", 0.0148, "TBPX L4 module"),
    (15.84, 0.0009, "TBPX HV lines outside L4"),
    (16.24, 0.0120, "TBPX external cylinder"),
    (20.95, 0.0117, "ITST, IT support tube"),
    (21.30, 0.0049, "TBPS innermost cylinder"),
)
OT_INNER_R = 22.0      # everything inside this is charged before the first IT layer


def tkl_planes():
    return [(IT_R[int(r[2])] if isinstance(r, str) else r, x, w) for r, x, w in TKL_PLANES]


G_REF = np.array([1e-5, 1e-3, 1e-3, 0.03, 0.03])   # 1/p^2 seed term units (at p = 1 GeV)


def kf_innovations_v4(P, params, clip=25.0, return_pulls=False, mat_scale=None, g_seed=None):
    """Linear oracle KF innovation NLL with tkLayout material; params = 25 log
    floors (+ 1 log global material scale unless mat_scale is given).

    g_seed (v5): per-parameter sigma of a seed term falling as 1/p -- the OT
    fit's own scattering, if it under-reports it -- added as diag(g^2)/p^2."""
    fl = np.exp(params[:5 * N_GROUP]).reshape(N_GROUP, 5) * FLOOR_REF[None]
    s = float(np.exp(params[5 * N_GROUP])) if mat_scale is None else mat_scale
    planes = tkl_planes()
    n = P["n"]
    Cst = P["C"].copy()
    Cst[:, np.arange(5), np.arange(5)] += fl[P["g"]] ** 2
    if g_seed is not None:
        Cst[:, np.arange(5), np.arange(5)] += (np.asarray(g_seed)[None] ** 2) / (P["p"][:, None] ** 2)
    dlt = np.zeros((n, 5))
    r_prev = OT_INNER_R
    tot, pulls = 0.0, []
    for L in (4, 3, 2, 1):
        rL = IT_R[L]
        for r_s, x, _ in planes:          # physically crossed between layer L and the last visit
            if rL < r_s <= r_prev + 1e-6:
                Cst += _kink_param_cov(np.full(n, r_s), P["tanl"], highland_theta2(s * x, P["p"], P["tanl"]))
        r_prev = rL
        h = P["have"][:, L - 1]
        if not h.any():
            continue
        H = P["H"][h, L - 1]
        y = P["res"][h, L - 1] - np.einsum("nij,nj->ni", H, dlt[h])
        S = np.einsum("nij,njk,nlk->nil", H, Cst[h], H)
        S[:, np.arange(4), np.arange(4)] += P["R"][h, L - 1] ** 2
        Suv = S[:, :2, :2]
        det = Suv[:, 0, 0] * Suv[:, 1, 1] - Suv[:, 0, 1] ** 2
        m2 = (Suv[:, 1, 1] * y[:, 0] ** 2 - 2 * Suv[:, 0, 1] * y[:, 0] * y[:, 1] + Suv[:, 0, 0] * y[:, 1] ** 2) / det
        tot += 0.5 * np.sum(np.minimum(m2, clip) + np.log(np.maximum(det, 1e-300)))
        if return_pulls:
            pulls.append({"L": L, "pu": y[:, 0] / np.sqrt(S[:, 0, 0]), "pv": y[:, 1] / np.sqrt(S[:, 1, 1]),
                          "m2": m2, "pt": P["p"][h] / np.sqrt(1 + P["tanl"][h] ** 2), "tanl": P["tanl"][h],
                          "su": np.sqrt(S[:, 0, 0]), "upd": L < 4})
        dm = P["dimok"][h, L - 1]
        Sm = S.copy()
        Sm[~np.repeat(dm[:, :, None], 4, 2) | ~np.repeat(dm[:, None, :], 4, 1)] = 0.0
        Sm[:, np.arange(4), np.arange(4)] = np.where(dm, Sm[:, np.arange(4), np.arange(4)], 1.0)
        ym = np.where(dm, y, 0.0)
        Hm = np.where(dm[:, :, None], H, 0.0)
        CHt = np.einsum("nij,nkj->nik", Cst[h], Hm)
        Kg = np.einsum("nik,nkl->nil", CHt, np.linalg.inv(Sm))
        dlt[h] += np.einsum("nik,nk->ni", Kg, ym)
        Cst[h] -= np.einsum("nik,njk->nij", Kg, CHt)
    return (tot, pulls) if return_pulls else tot


def fit_v4(P, fit_scale=True, x0=None, clip=25.0):
    from scipy.optimize import minimize
    k = 5 * N_GROUP + (1 if fit_scale else 0)
    p0 = np.zeros(k) if x0 is None else x0
    f = (lambda pp: kf_innovations_v4(P, pp, clip)) if fit_scale else \
        (lambda pp: kf_innovations_v4(P, np.r_[pp, 0.0], clip, mat_scale=1.0))
    return minimize(f, p0, method="L-BFGS-B", options={"maxiter": 400, "eps": 1e-4, "ftol": 1e-10})


def innovation_table(pulls):
    """Innovation pull widths per layer (visit order) x pT and |tanL|, and 4-sigma containment."""
    out = {}
    for d in pulls:
        row = {"n": int(len(d["pu"])), "u": rs(d["pu"])[:2], "v": rs(d["pv"])[:2],
               "cont4": float(np.mean(d["m2"] < 16)), "su_um": float(np.median(d["su"]) * 1e4), "pt": [], "tanL": []}
        for lab, x, edges in (("pt", d["pt"], PT_BINS), ("tanL", np.abs(d["tanl"]), TANL_BINS)):
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (x >= lo) & (x < hi)
                row[lab].append((lo, rs(d["pu"][m])[0], rs(d["pv"][m])[0], int(m.sum()),
                                 float(np.mean(d["m2"][m] < 16)) if m.any() else np.nan))
        out[f"IL{d['L']}"] = row
    return out


def fit_v5(P, fit_scale=True, x0=None, clip=25.0):
    """v5 = v4 + a 1/p seed term (5 params). Layout: 25 log floors, [1 log scale], 5 log g."""
    from scipy.optimize import minimize
    nf = 5 * N_GROUP

    def unpack(pp):
        sc = pp[nf] if fit_scale else 0.0
        g = np.exp(pp[nf + (1 if fit_scale else 0):]) * G_REF
        return np.r_[pp[:nf], sc], g
    def f(pp):
        base, g = unpack(pp)
        return kf_innovations_v4(P, base, clip, mat_scale=None if fit_scale else 1.0, g_seed=g)
    k = nf + (1 if fit_scale else 0) + 5
    p0 = np.zeros(k) if x0 is None else x0
    res = minimize(f, p0, method="L-BFGS-B", options={"maxiter": 400, "eps": 1e-4, "ftol": 1e-10})
    base, g = unpack(res.x)
    return res, base, g


def export_refit_calibration(seed_json, r_tables_json, out, material_scale=1.0):
    """Write the v3 refit calibration (ot_refit_projection.load_calibration): the
    hybrid-seed deficit tables of results/refit_calibration/run_fit_v10h.py, the
    tkLayout planes at `material_scale`, and the cluster R tables. The ~1.37x
    scattering excess is NOT folded in (memory: smartpixels-scattering-excess-mystery)."""
    import json
    S = json.load(open(seed_json))
    out_d = {"format": "v3", "seed_deficit": S["seed"], "pt_bins": S["pt_bins"],
             "stub_min": 4, "stub_max": 6,
             "floor_groups": [list(g) for g in FLOOR_GROUPS],
             "planes": [[r, x, w] for r, x, w in TKL_PLANES],
             "ot_inner_r_cm": OT_INNER_R, "material_scale": float(material_scale),
             "R_tables": json.load(open(r_tables_json)),
             "provenance": {"seed": str(seed_json), "R_tables": str(r_tables_json)}}
    with open(out, "w") as fh:
        json.dump(out_d, fh, indent=1)
    return out
