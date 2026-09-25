"""Broken-line (GBL-style) refit of an OT seed with inner-tracker hits.

THE FIT. Linearised about the seed helix a_s (TTTrack order rInv, phi0, tanL, z0,
d0). Unknowns per track: da (5, the helix correction of the OUTERMOST segment,
i.e. the OT seed's own helix) and one pair of kink angles (thetaT, thetaL) per
scatterer. The particle travels beamspot -> IL1 -> ... -> IL4 -> OT, so the helix
of the segment a hit sits on is the OT helix plus every kink BETWEEN the hit and
the OT:

    res_m = H_m (da + sum_{s outside hit m} J_s theta_s) + eps_m,   eps ~ N(0, R_m^2)
    da ~ N(0, C_seed)   (the OT fit as an external prior)
    theta_s ~ N(0, theta0_s^2 I2)   (Highland, the projected space angle)

with J_s the kink Jacobian of refit_calibration_fit._kink_param_cov:
jT = (0, sec, 0, 0, +r_s sec), jL = (0, 0, sec^2, -r_s sec^2, 0). This is exactly
the model the refit KF of refit_calibration_fit.kf_innovations_v4 assumes (the
kink covariances there are the process noise). All unknowns are solved at once
from the normal equations; the estimate of the INNERMOST segment is
da + sum_s J_s theta_s.

EQUIVALENCE. For a linear Gaussian model this is the same estimator as the
Kalman filter followed by the Rauch-Tung-Striebel smoother: the KF's final state
equals the innermost-segment estimate here, and the smoothed residuals are what
the RTS pass would give at every hit. What the joint solution adds is the
smoothed residual and its variance at EVERY hit (pull = r / sqrt(R^2 - H Cov H^T),
which equals the pull of the residual with that hit excluded), and one global
chi2 with ndf = number of measurements (priors count as measurements of the
unknowns, so they cancel in ndf).

Blobel's broken lines [V. Blobel, NIM A 566 (2006) 14] and the General Broken
Lines refit [C. Kleinwort, NIM A 673 (2012) 107] parametrise the same model by
OFFSETS at the scatterers (the kink is then the second difference of offsets),
which makes the normal matrix band-diagonal and O(n) to solve; the explicit
kink-angle parametrisation used here is an equivalent reparametrisation (a
linear change of variables), chosen because it maps one-to-one onto the refit
KF's process noise and onto the material model, and because with <= 4 hits and
~10 scatterers per track a dense batched solve is cheap.

Two harnesses (see main):
  oracle   the refit-calibration sample (target-TP clusters on every layer):
           KF-vs-GBL agreement on a toy and on data, smoothed pulls, chi2
           probability, parameter resolution against the truth at the POCA.
  cmssw    the AAAA producer's OWN accepted hits (refit hit table), refit by GBL
           with the producer's Q and errors (reproduction) and with Highland
           material (physics), against the producer's refit and the truth.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refit_calibration_fit as F   # noqa: E402

HP = ("rInv", "phi", "tanL", "z0", "d0")


def kink_jacobian(r_s, tanl):
    """(n, S, 5, 2): d(helix)/d(thetaT, thetaL) for kinks at radii r_s (n, S)."""
    sec2 = 1.0 + tanl[:, None] ** 2
    sec = np.sqrt(sec2)
    J = np.zeros(r_s.shape + (5, 2))
    J[..., 1, 0] = sec
    J[..., 4, 0] = r_s * sec
    J[..., 2, 1] = sec2
    J[..., 3, 1] = -r_s * sec2
    return J


def _design(H, pos_hit, pos_scat, Jk, act_scat):
    """G (n, M, 5, K): the helix of hit m's segment as a function of the unknowns."""
    n, M = pos_hit.shape
    S = pos_scat.shape[1]
    G = np.zeros((n, M, 5, 5 + 2 * S))
    G[:, :, np.arange(5), np.arange(5)] = 1.0
    on = (pos_scat[:, None, :] > pos_hit[:, :, None]) & act_scat[:, None, :]      # (n, M, S)
    for s in range(S):
        G[:, :, :, 5 + 2 * s:7 + 2 * s] = np.where(on[:, :, s, None, None], Jk[:, None, s], 0.0)
    return G


def gbl_fit(H, y, R, w, C0, pos_hit, r_scat, pos_scat, th2, tanl):
    """Joint broken-line solution.

    H (n,M,4,5) Jacobians at the seed; y (n,M,4) = meas - proj(seed); R (n,M,4)
    measurement sigmas; w (n,M,4) bool, the dimensions used (the KF's mask);
    C0 (n,5,5) seed prior; pos_hit (n,M) and pos_scat (n,S) an ordering
    coordinate along the track (radius, or layer rank) -- scatterer s acts on hit
    m iff pos_scat > pos_hit; r_scat (n,S) the kink radius used in J; th2 (n,S)
    theta0^2, <= 0 marks an absent scatterer.

    Returns the innermost-segment correction `da_in` (n,5) and its covariance,
    per-hit smoothed residuals and pulls (NaN where unused), chi2 and ndf.
    """
    n, M = pos_hit.shape
    S = pos_scat.shape[1]
    K = 5 + 2 * S
    act = th2 > 0
    Jk = kink_jacobian(r_scat, tanl)
    G = _design(H, pos_hit, pos_scat, Jk, act)
    A = np.einsum("nmij,nmjk->nmik", H, G)                             # (n,M,4,K)
    W = np.where(w, 1.0 / np.where(w, R, 1.0) ** 2, 0.0)
    y0 = np.where(w, y, 0.0)
    # scale the unknowns to unit prior width: the seed variances span ~7 orders
    sd = np.sqrt(np.diagonal(C0, axis1=1, axis2=2))
    scale = np.concatenate([sd, np.repeat(np.sqrt(np.where(act, th2, 1.0)), 2, axis=1)], 1)
    As = A * scale[:, None, None, :]
    Np = np.zeros((n, K, K))
    Np[:, :5, :5] = np.linalg.inv(C0 / np.einsum("ni,nj->nij", sd, sd))
    Np[:, np.arange(5, K), np.arange(5, K)] = 1.0
    Nm = Np + np.einsum("nmik,nmi,nmil->nkl", As, W, As)
    b = np.einsum("nmik,nmi,nmi->nk", As, W, y0)
    Cs = np.linalg.inv(Nm)
    xs = np.einsum("nkl,nl->nk", Cs, b)
    x = xs * scale
    Cov = Cs * scale[:, :, None] * scale[:, None, :]
    Gin = np.zeros((n, 5, K))
    Gin[:, np.arange(5), np.arange(5)] = 1.0
    for s in range(S):
        Gin[:, :, 5 + 2 * s:7 + 2 * s] = np.where(act[:, s, None, None], Jk[:, s], 0.0)
    da_in = np.einsum("nik,nk->ni", Gin, x)
    C_in = np.einsum("nik,nkl,njl->nij", Gin, Cov, Gin)
    r = y0 - np.einsum("nmik,nk->nmi", A, x)
    Vfit = np.einsum("nmik,nkl,nmil->nmi", A, Cov, A)
    Vr = np.where(w, R ** 2 - Vfit, np.nan)
    with np.errstate(invalid="ignore"):
        pull = np.where(w, r / np.sqrt(Vr), np.nan)
    chi2 = (np.einsum("nmi,nmi,nmi->n", r, W, r)
            + np.einsum("nk,nkl,nl->n", xs, Np, xs))
    return {"da_in": da_in, "C_in": C_in, "x": x, "Cov": Cov, "G": G, "res": np.where(w, r, np.nan),
            "pull": pull, "chi2": chi2, "ndf": w.sum((1, 2))}


def kf_reference(H, y, R, w, C0, pos_hit, r_scat, pos_scat, th2, tanl):
    """The refit KF of refit_calibration_fit.kf_innovations_v4, same recursion
    (kink covariance added as process noise before each hit, vector update on
    the used dimensions), for the same arguments as gbl_fit; hits are visited in
    DECREASING pos_hit (outside-in). Returns the final state, its covariance and
    the innovation pulls, so that it can be checked against kf_innovations_v4
    and compared with gbl_fit (which kf_innovations_v4 cannot: it returns only
    the likelihood and pulls)."""
    n, M = pos_hit.shape
    act = th2 > 0
    Jk = kink_jacobian(r_scat, tanl)
    Ks = th2[:, :, None, None] * np.einsum("nsik,nsjk->nsij", Jk, Jk)
    order = np.argsort(-pos_hit, axis=1, kind="stable")
    ar = np.arange(n)
    dlt = np.zeros((n, 5))
    Cst = C0.copy()
    done = np.full((n, pos_scat.shape[1]), False)
    pulls = np.full((n, M, 4), np.nan)
    m2 = np.full((n, M), np.nan)
    for k in range(M):
        m = order[:, k]
        ph = pos_hit[ar, m]
        add = act & ~done & (pos_scat > ph[:, None])
        Cst = Cst + np.einsum("ns,nsij->nij", add.astype(float), Ks)
        done |= add
        Hm, ym, Rm, wm = H[ar, m], y[ar, m], R[ar, m], w[ar, m]
        v = ym - np.einsum("nij,nj->ni", Hm, dlt)
        S = np.einsum("nij,njk,nlk->nil", Hm, Cst, Hm)
        S[:, np.arange(4), np.arange(4)] += Rm ** 2
        with np.errstate(invalid="ignore"):
            pulls[ar, m] = np.where(wm, v / np.sqrt(np.diagonal(S, axis1=1, axis2=2)), np.nan)
        mask2 = wm[:, :, None] & wm[:, None, :]
        Sm = np.where(mask2, S, 0.0)
        Sm[:, np.arange(4), np.arange(4)] = np.where(wm, Sm[:, np.arange(4), np.arange(4)], 1.0)
        Hw = np.where(wm[:, :, None], Hm, 0.0)
        CHt = np.einsum("nij,nkj->nik", Cst, Hw)
        Si = np.linalg.inv(Sm)
        vw = np.where(wm, v, 0.0)
        m2[ar, m] = np.where(wm.any(1), np.einsum("ni,nij,nj->n", vw, Si, vw), np.nan)
        Kg = np.einsum("nik,nkl->nil", CHt, Si)
        dlt = dlt + np.einsum("nik,nk->ni", Kg, np.where(wm, v, 0.0))
        Cst = Cst - np.einsum("nik,njk->nij", Kg, CHt)
    rest = act & ~done                                   # kinks inside the innermost hit
    Cst = Cst + np.einsum("ns,nsij->nij", rest.astype(float), Ks)
    return {"da_in": dlt, "C_in": Cst, "pull": pulls, "m2": m2}


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------
def rs(x):
    """Robust sigma (1.4826 MAD), RMS, n, over finite entries."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan, np.nan, len(x)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))), float(np.sqrt(np.mean(x ** 2))), len(x)


def chi2_prob(chi2, ndf):
    from scipy.stats import chi2 as C2
    return np.where(ndf > 0, C2.sf(chi2, np.maximum(ndf, 1)), np.nan)


def prob_table(p):
    p = p[np.isfinite(p)]
    dec = np.histogram(p, bins=np.linspace(0, 1, 11))[0] / max(len(p), 1)
    return (f"P(chi2) deciles (flat = 0.100): " + " ".join(f"{d:.3f}" for d in dec)
            + f"  | P<0.01 {np.mean(p < 0.01):.4f}  P<0.001 {np.mean(p < 0.001):.4f}")


def truth_at_poca(N):
    """Truth helix at the POCA from the nano L1TTrack tp_* columns (TTTrack order)."""
    from ngtagger.truth_helix import KPT_CMSSW
    return np.stack([N["tp_charge"] * KPT_CMSSW / N["tp_pt"], N["tp_phi0"], N["tp_tanL"],
                     N["tp_z0"], N["tp_d0"]], 1)


def param_diff(a, b):
    d = a - b
    d[:, 1] = np.angle(np.exp(1j * d[:, 1]))
    return d


UNIT = {"rInv": (1e4, "1e-4/cm"), "phi": (1e3, "mrad"), "tanL": (1e3, "1e-3"), "z0": (1e4, "um"), "d0": (1e4, "um")}
PTB = [2, 3, 5, 10, 20, 1e9]


def resolution_table(log, lab, est, truth, pt, sel=None):
    """Robust sigma of (estimate - truth) per parameter and pT bin, in physical units."""
    sel = np.ones(len(pt), bool) if sel is None else sel
    hdr = "   ".join(f"{p}[{UNIT[p][1]}]" for p in HP)
    log(f"   {lab:34s} {hdr}")
    for lo, hi in zip(PTB[:-1], PTB[1:]):
        m = sel & (pt >= lo) & (pt < hi)
        row = []
        for name, e in est.items():
            d = param_diff(e[m], truth[m])
            row.append(f"{name}: " + " ".join(f"{rs(d[:, i])[0] * UNIT[p][0]:7.2f}" for i, p in enumerate(HP)))
        log(f"   pT [{lo:>2g},{hi:>4g}) n {m.sum():6d} | " + " | ".join(row))


# ----------------------------------------------------------------------------
# (a) oracle harness: the refit-calibration sample
# ----------------------------------------------------------------------------
def empirical_seed(D, Pall, dres, Kall, log):
    """Robust covariance of (TTTrack - truth at the POCA) minus the IT/gap kinks,
    per floor group x pT bin. Copied from results/refit_calibration/run_fit_v10.py
    (same construction, so the two studies use one seed model)."""
    pt_all = Pall["p"] / np.sqrt(1 + Pall["tanl"] ** 2)
    E = {}
    for g in range(F.N_GROUP):
        for b in range(len(PTB) - 1):
            m = (Pall["g"] == g) & (pt_all >= PTB[b]) & (pt_all < PTB[b + 1])
            if m.sum() < 150:
                E[(g, b)] = None
                continue
            x = dres[Pall["ids"][m]]
            med = np.median(x, 0)
            mad = 1.4826 * np.median(np.abs(x - med), 0)
            keep = np.all(np.abs(x - med) < 4 * mad, 1)
            Ce = np.cov(x[keep].T) - Kall[m][keep].mean(0)
            sdv = np.sqrt(np.maximum(np.diag(Ce), 1e-30))
            wv, V = np.linalg.eigh(Ce / np.outer(sdv, sdv))
            E[(g, b)] = ((V * np.maximum(wv, 1e-4)) @ V.T) * np.outer(sdv, sdv)
    for (g, b), v in list(E.items()):
        if v is None:
            near = sorted((abs(bb - b), bb) for (gg, bb), vv in E.items() if gg == g and vv is not None)
            E[(g, b)] = E[(g, near[0][1])]
    return E


def load_hybrid(path):
    """Per-bin seed deficit of the HYBRID seed (results/refit_calibration/run_fit_v10h.py):
    robust cov(TTTrack - truth@POCA) - IT kinks x1 - bin-mean reported C, PSD in
    correlation space, per (floor group, OT stubs 4/5/6+, pT bin). The track's
    own reported C is kept and this deficit added to it."""
    J = json.load(open(path))
    E = {tuple(int(v) for v in k.split(",")): np.array(M) for k, M in J["seed"].items()}
    return E, J["pt_bins"]


def hybrid_seed(E, ptb, C, g, n_stubs, pt):
    ns = np.clip(n_stubs, 4, 6) - 4
    b = np.clip(np.searchsorted(ptb, pt, side="right") - 1, 0, len(ptb) - 2)
    return C + np.stack([E[(gg, nn, bb)] for gg, nn, bb in zip(g, ns, b)])


def seed_lookup(E, P):
    pt = P["p"] / np.sqrt(1 + P["tanl"] ** 2)
    b = np.clip(np.searchsorted(PTB, pt, side="right") - 1, 0, len(PTB) - 2)
    return np.stack([E[(g, bb)] for g, bb in zip(P["g"], b)])


def oracle(args, log):
    import uproot
    import awkward as ak
    base = Path(args.calib_dir)
    D = F.load(str(base / "calib_perfect_ttbar_pu200_tp.npz"))
    prim = F.primary_mask(D)
    T = json.load(open(base / "R_tables_tp.json"))
    tr = D["truth"]
    Rcal = F.apply_R(T, D["sizeX"], D["sizeY"], tr[:, 2], tr[:, 3], D["sig"])

    def pack(sel):
        P = F.pack_tracks(D, sel, Rcal)
        tk = D["track"][sel]
        ut, inv = np.unique(tk, return_inverse=True)
        ii = np.flatnonzero(sel)
        first = np.zeros(len(ut), np.int64)
        first[inv[::-1]] = ii[::-1]
        kept = F.floor_group(D)[first] >= 0
        P["ids"] = ut[kept]
        P["n_stubs"] = D["n_stubs"][first][kept]
        assert len(P["ids"]) == P["n"]
        return P

    files = sorted(glob.glob(args.nano))
    cols = list(HP) + ["tp_pt", "tp_phi0", "tp_tanL", "tp_z0", "tp_d0", "tp_charge"]
    A = uproot.concatenate([f"{f}:Events" for f in files], [f"L1TTrack_{c}" for c in cols])
    N = {c: ak.to_numpy(ak.flatten(A[f"L1TTrack_{c}"])).astype(float) for c in cols}
    if np.mean(np.abs(N["tanL"][D["track"]] - D["tanL"]) < 1e-5) < 0.999:
        raise SystemExit("npz track index does not join the nano L1TTrack table (tanL mismatch)")
    seedpar = np.stack([N[p] for p in HP], 1)
    truth = truth_at_poca(N)
    dres = param_diff(seedpar, truth)

    planes = F.tkl_planes()
    INNER = [(2.25, 0.00227), (F.IT_R[1], 0.0185)]           # beam pipe (0.8 mm Be), IL1 module

    def material(P, extra=()):
        r = np.array([p[0] for p in planes] + [p[0] for p in extra])
        x = np.array([p[1] for p in planes] + [p[1] for p in extra])
        n = P["n"]
        th2 = np.stack([F.highland_theta2(np.full(n, xx), P["p"], P["tanl"]) for xx in x], 1)
        return np.broadcast_to(r, (n, len(r))).copy(), th2

    def kinks_cov(P, pl):
        K = np.zeros((P["n"], 5, 5))
        for r_s, x in pl:
            K += F._kink_param_cov(np.full(P["n"], r_s), P["tanl"], F.highland_theta2(np.full(P["n"], x), P["p"], P["tanl"]))
        return K

    tracks = np.unique(D["track"][prim & (D["pt"] >= 2)])
    Pall = pack(prim & np.isin(D["track"], tracks))
    E = empirical_seed(D, Pall, dres, kinks_cov(Pall, INNER + [(r, x) for r, x, _ in planes]), log)
    fit_tr = np.random.default_rng(2).choice(tracks, 40000, replace=False)
    sample = np.random.default_rng(7).choice(np.setdiff1d(tracks, fit_tr), args.n_tracks, replace=False)
    P = pack(prim & np.isin(D["track"], sample))
    n = P["n"]
    pos_hit = np.broadcast_to(np.array([F.IT_R[L] for L in (1, 2, 3, 4)]), (n, 4)).copy()
    w = P["have"][:, :, None] & P["dimok"]
    r_sc, th2 = material(P)
    Eh, ptb_h = load_hybrid(Path(args.calib_dir) / "fit_v10h.json")
    seeds = {"hybrid seed (reported C + per-bin deficit)":
             hybrid_seed(Eh, ptb_h, P["C"], P["g"], P["n_stubs"], P["p"] / np.sqrt(1 + P["tanl"] ** 2))}
    if not args.hybrid_only:
        seeds |= {"empirical seed": seed_lookup(E, P), "TTTrack cov": P["C"]}
    K_irr = kinks_cov(P, INNER)

    log("== (a1) toy: residuals drawn from the model itself (empirical seed, Highland x1, calibrated R)")
    rng = np.random.default_rng(11)

    def draw(C):
        wv, V = np.linalg.eigh(C)
        return np.einsum("nij,nj->ni", V * np.sqrt(np.maximum(wv, 0))[:, None, :], rng.standard_normal((len(C), 5)))

    C0 = seed_lookup(E, P)
    e = draw(C0)
    Jk = kink_jacobian(r_sc, P["tanl"])
    th = rng.standard_normal(th2.shape + (2,)) * np.sqrt(th2)[..., None]
    ytoy = np.zeros_like(P["res"])
    for m in range(4):
        dp = np.einsum("nsij,nsj->ni", Jk * (r_sc > pos_hit[:, m:m + 1])[..., None, None], th)
        ytoy[:, m] = np.einsum("nij,nj->ni", P["H"][:, m], dp - e) + P["R"][:, m] * rng.standard_normal((n, 4))
    g = gbl_fit(P["H"], ytoy, P["R"], w, C0, pos_hit, r_sc, r_sc, th2, P["tanl"])
    k = kf_reference(P["H"], ytoy, P["R"], w, C0, pos_hit, r_sc, r_sc, th2, P["tanl"])
    sd = np.sqrt(np.diagonal(g["C_in"], axis1=1, axis2=2))
    log("   max |KF - GBL| / sigma(GBL), innermost segment: "
        + " ".join(f"{p} {np.max(np.abs(k['da_in'][:, i] - g['da_in'][:, i]) / sd[:, i]):.1e}" for i, p in enumerate(HP))
        + f" | covariance max rel diff {np.max(np.abs(k['C_in'] - g['C_in']) / np.sqrt(np.einsum('nii,njj->nij', g['C_in'], g['C_in']))):.1e}")
    pin = (g["da_in"] - (np.einsum("nsij,nsj->ni", Jk, th) - e)) / sd
    log("   GBL innermost-segment pulls vs the toy truth (MAD): " + " ".join(f"{p} {rs(pin[:, i])[0]:.3f}" for i, p in enumerate(HP)))
    for m, L in enumerate((1, 2, 3, 4)):
        log(f"   IL{L} smoothed pull MAD u {rs(g['pull'][:, m, 0])[0]:.3f} v {rs(g['pull'][:, m, 1])[0]:.3f} "
            f"| RMS u {rs(g['pull'][:, m, 0])[1]:.3f} v {rs(g['pull'][:, m, 1])[1]:.3f}")
    log("   " + prob_table(chi2_prob(g["chi2"], g["ndf"])))

    pt = P["p"] / np.sqrt(1 + P["tanl"] ** 2)
    tru = truth[P["ids"]]
    a_s = seedpar[P["ids"]]
    for lab, C0 in seeds.items():
        log(f"== (a2-a5) data, seed = {lab}, Highland x1 tkLayout planes, calibrated R (R_tables_tp)")
        g = gbl_fit(P["H"], P["res"], P["R"], w, C0, pos_hit, r_sc, r_sc, th2, P["tanl"])
        k = kf_reference(P["H"], P["res"], P["R"], w, C0, pos_hit, r_sc, r_sc, th2, P["tanl"])
        sd = np.sqrt(np.diagonal(g["C_in"], axis1=1, axis2=2))
        log("   max |KF - GBL| / sigma: " + " ".join(
            f"{p} {np.max(np.abs(k['da_in'][:, i] - g['da_in'][:, i]) / sd[:, i]):.1e}" for i, p in enumerate(HP)))
        # the replica is the KF of kf_innovations_v4: its innovation pulls must match
        P2 = dict(P); P2["C"] = C0
        _, fp = F.kf_innovations_v4(P2, np.r_[np.full(5 * F.N_GROUP, np.log(1e-9)), 0.0], return_pulls=True, mat_scale=1.0)
        dmax = max(float(np.nanmax(np.abs(d["pu"] - k["pull"][P["have"][:, d["L"] - 1], d["L"] - 1, 0]))) for d in fp)
        log(f"   kf_reference vs kf_innovations_v4 innovation pulls: max |diff| {dmax:.1e} (floors 1e-9 there)")
        for m, L in enumerate((1, 2, 3, 4)):
            pu, pv = g["pull"][:, m, 0], g["pull"][:, m, 1]
            log(f"   IL{L} smoothed pull: MAD u {rs(pu)[0]:.3f} v {rs(pv)[0]:.3f} | RMS u {rs(pu)[1]:.2f} v {rs(pv)[1]:.2f} "
                f"| |pull|>4 u {np.nanmean(np.abs(pu) > 4):.4f} v {np.nanmean(np.abs(pv) > 4):.4f}")
        c2n = g["chi2"] / np.maximum(g["ndf"], 1)
        log(f"   chi2/ndf: median {np.median(c2n):.3f} mean {np.mean(c2n):.3f} (ndf median {np.median(g['ndf']):.0f})")
        log("   " + prob_table(chi2_prob(g["chi2"], g["ndf"])))
        est = {"seed": a_s, "KF": a_s + k["da_in"], "GBL": a_s + g["da_in"]}
        resolution_table(log, "robust sigma (estimate - truth@POCA)", est, tru, pt)
        pred = np.sqrt(np.diagonal(g["C_in"] + K_irr, axis1=1, axis2=2))
        pg = param_diff(est["GBL"], tru) / pred
        irr = np.sqrt(np.median(np.diagonal(K_irr, axis1=1, axis2=2), 0))
        log("   GBL pull vs truth, sigma = sqrt(C_in + beam-pipe + IL1-module kinks) (MAD): "
            + " ".join(f"{p} {rs(pg[:, i])[0]:.3f}" for i, p in enumerate(HP)))
        log("   irreducible (beam pipe + IL1 module scattering, median): "
            + " ".join(f"{p} {irr[i] * UNIT[p][0]:.2f} {UNIT[p][1]}" for i, p in enumerate(HP)))


# ----------------------------------------------------------------------------
# (b) the CMSSW producer's own hits
# ----------------------------------------------------------------------------
HIT = "L1TSmartPixelsRefitHitDigiRefitAAAA"
VTRK = "L1TSmartPixelsTrackDigiRefitAAAA"
HIT_COLS = ["trackIdx", "layer", "detId", "hitAccepted", "selClusterIdx", "selHitClass", "hasAlpha",
            "recoLocalX", "recoLocalY", "recoCotAlpha", "sigX", "sigY", "sigAlpha",
            "projSeedLocalX", "projSeedLocalY", "projSeedCotAlpha", "recoSizeX", "recoSizeY"]
VTRK_COLS = list(HP) + ["spixRefitPerformed", "spixNAcceptedHits"]
TP_COLS = ["genuine", "tp_pt", "tp_phi0", "tp_tanL", "tp_z0", "tp_d0", "tp_charge"]


def cmssw(args, log):
    """GBL on the producer's own accepted hits (one per crossing, as its KF used them)."""
    import awkward as ak
    import uproot
    import ot_projection as OP
    import ot_refit_projection as RP
    from ngtagger.truth_helix import KPT_CMSSW
    from sklearn.metrics import roc_auc_score
    geo = OP.load_geometry(args.geometry)
    F.IT_R = {L: geo["R"][L] for L in (1, 2, 3, 4)}
    planes = F.tkl_planes()
    # order the scatterers by layer rank, not radius: a hit's radius can sit either
    # side of its own layer's mean module radius, and a module kink must never act
    # on a hit of its own layer. Module planes rank as their layer, passive planes
    # between layers as L + 0.5.
    rank = np.array([next((float(L) for L in (2, 3, 4) if w.startswith(f"TBPX L{L} module")),
                          1.5 + sum(r > F.IT_R[L] for L in (2, 3, 4)))
                     for r, _, w in planes])
    Rtab = json.load(open(Path(args.calib_dir) / "R_tables_tp.json"))
    files = sorted(glob.glob(args.nano))
    need = ([f"L1TTrack_{c}" for c in OP.TRK_COLS + TP_COLS] + [f"{HIT}_{c}" for c in HIT_COLS]
            + [f"{VTRK}_{c}" for c in VTRK_COLS] + ["L1TTrackStub_trackIdx", "L1TTrackStub_isBarrel",
                                                    "run", "luminosityBlock", "event"])
    Eh, ptb_h = load_hybrid(Path(args.calib_dir) / "fit_v10h.json")
    CONF = (("HL calR", "HL calR hybrid") if args.hybrid_only
            else ("prodQ", "HL prodR", "HL calR", "HL calR hybrid"))
    NIT = args.iterations
    out = {k: [] for k in ["pt", "clean", "nacc", "seed", "cmssw", "truth", "hitcls", "proj_du", "proj_dv",
                           "counts", "sd_lin", "nstub_check"]}
    TRK = {k: [] for k in ("event", "id_run", "id_luminosityBlock", "id_event", "track", "pt", "chi2", "ndf",
                           "prob", "pull_max", "pull_2nd", "pull_layer", "kf_chi2_sum", "kf_pull_max",
                           "n_acc_same_tp", "n_acc_other_tp", "n_acc_noise", "gbl_par", "gbl_sigma",
                           "res_gbl", "res_cmssw", "res_seed")}
    for c in CONF:
        for it in range(NIT + 1):
            out[f"est {c} {it}"] = []
        out[f"prob {c}"] = []
        out[f"hitpull {c}"] = []

    def project(states, L_h, mi_h):
        u = np.full(len(states), np.nan); v = u.copy(); cA = u.copy()
        J = np.full((len(states), 4, 5), np.nan)
        rad = u.copy()
        for L in (1, 2, 3, 4):
            i = np.flatnonzero(L_h == L)
            if not len(i):
                continue
            pj = OP.project_with_cov(states[i], np.zeros((len(i), 5, 5)), OP.cylinder_s(states[i], geo["R"][L]),
                                     geo, mi_h[i], jac="central")
            u[i], v[i], cA[i], J[i] = pj["u"], pj["v"], pj["cotA"], pj["J"]
            gp = geo["origin"][mi_h[i]] + pj["u"][:, None] * geo["ex"][mi_h[i]] + pj["v"][:, None] * geo["ey"][mi_h[i]]
            rad[i] = np.hypot(gp[:, 0], gp[:, 1])
        return u, v, cA, J, rad

    done = 0
    n_fallback = 0
    for A in uproot.iterate([f"{f}:Events" for f in files], need, step_size=args.chunk):
        if args.nev is not None and done >= args.nev:
            break
        T = OP._flat(A, "L1TTrack", OP.TRK_COLS + TP_COLS)
        V = OP._flat(A, VTRK, VTRK_COLS)
        Hh = OP._flat(A, HIT, HIT_COLS)
        if len(V["rInv"]) != len(T["pt"]):
            raise SystemExit(f"{VTRK} is not row-aligned with L1TTrack")
        a, C = OP.track_state(T)
        toff = np.concatenate([[0], np.cumsum(T["_n"])])
        gt = toff[Hh["event"]] + Hh["trackIdx"].astype(np.int64)
        acc = (Hh["hitAccepted"] > 0) & (Hh["selClusterIdx"] >= 0)
        mi = OP.module_index(geo, Hh["detId"])
        ia = np.flatnonzero(acc & (mi >= 0))
        u0, v0, cA0, J0, rad0 = project(a[gt[ia]], Hh["layer"][ia], mi[ia])
        ok0 = np.isfinite(u0) & np.isfinite(J0).all((1, 2))
        stored = Hh["projSeedLocalX"][ia] > OP.SENTINEL
        out["proj_du"].append((u0 - Hh["projSeedLocalX"][ia])[ok0 & stored])
        out["proj_dv"].append((v0 - Hh["projSeedLocalY"][ia])[ok0 & stored])
        out["counts"].append(np.array([int(acc.sum()), int((~ok0).sum() + (acc & (mi < 0)).sum()),
                                       int((ok0 & ~stored).sum())]))
        ia, u0, v0, cA0, J0, rad0 = ia[ok0], u0[ok0], v0[ok0], cA0[ok0], J0[ok0], rad0[ok0]
        tsel = np.unique(gt[ia])
        tsel = tsel[(V["spixRefitPerformed"][tsel] > 0) & (T["genuine"][tsel] > 0) & (T["tp_pt"][tsel] > 0)]
        keep = np.isin(gt[ia], tsel)
        ia, u0, v0, cA0, J0, rad0 = ia[keep], u0[keep], v0[keep], cA0[keep], J0[keep], rad0[keep]
        if not len(tsel):
            done += len(A)
            continue
        o = np.lexsort((-rad0, -Hh["layer"][ia], gt[ia]))                     # the producer's visitation order
        ia, u0, v0, cA0, J0, rad0 = ia[o], u0[o], v0[o], cA0[o], J0[o], rad0[o]
        tt = gt[ia]
        pos_t = np.searchsorted(tsel, tt)
        first = np.r_[0, np.flatnonzero(np.diff(pos_t)) + 1]
        slot = np.arange(len(ia)) - np.repeat(first, np.diff(np.r_[first, len(ia)]))
        n, M = len(tsel), int(slot.max()) + 1
        L_h, mi_h = Hh["layer"][ia], mi[ia]
        meas = np.stack([Hh["recoLocalX"][ia], Hh["recoLocalY"][ia], Hh["recoCotAlpha"][ia]], 1)

        def fill(u, v, cA, J, shift):
            """Per-track arrays for the fit; `shift` = G x at the linearisation point."""
            Hm = np.zeros((n, M, 4, 5)); y = np.zeros((n, M, 4))
            Hm[pos_t, slot] = J
            h0 = np.stack([u, v, cA], 1)
            y[pos_t, slot, :3] = meas - h0 + np.einsum("hij,hj->hi", J[:, :3], shift)
            return Hm, y

        Rp = np.ones((n, M, 4)); Rc = np.ones((n, M, 4)); w = np.zeros((n, M, 4), bool)
        lay = np.zeros((n, M), int); rr = np.zeros((n, M)); cls = np.full((n, M), -9)
        Rp[pos_t, slot, 0], Rp[pos_t, slot, 1], Rp[pos_t, slot, 2] = Hh["sigX"][ia], Hh["sigY"][ia], Hh["sigAlpha"][ia]
        sig4 = np.stack([Hh["sigX"][ia], Hh["sigY"][ia], Hh["sigAlpha"][ia], np.ones(len(ia))], 1)
        # calibrated R looked up at the SEED-predicted cotAlpha (no truth); the cotBeta
        # scale is never used, the producer does not update on beta
        Rcal = F.apply_R(Rtab, Hh["recoSizeX"][ia].astype(float), Hh["recoSizeY"][ia].astype(float),
                         cA0, np.zeros(len(ia)), sig4)
        Rc[pos_t, slot, :3] = Rcal[:, :3]
        w[pos_t, slot, 0] = w[pos_t, slot, 1] = True
        w[pos_t, slot, 2] = Hh["hasAlpha"][ia] > 0                              # useAngles = "alpha"
        lay[pos_t, slot] = L_h
        rr[pos_t, slot] = rad0
        cls[pos_t, slot] = Hh["selHitClass"][ia]
        valid = np.zeros((n, M), bool); valid[pos_t, slot] = True
        tanl = a[tsel, 2]
        pT = KPT_CMSSW / np.abs(a[tsel, 0])
        p = pT * np.sqrt(1 + tanl ** 2)
        C0 = C[tsel]
        # hybrid seed: floor group as F.floor_group (|tanL| band, any disk stub), OT stubs
        # from L1TTrack_nStubs (checked against the stub table the calibration counted)
        St = OP._flat(A, "L1TTrackStub", ["trackIdx", "isBarrel"])
        gst = toff[St["event"]] + St["trackIdx"].astype(np.int64)
        n_disk = np.bincount(gst, weights=St["isBarrel"] == 0, minlength=len(T["pt"]))[tsel]
        n_st = np.bincount(gst, minlength=len(T["pt"]))[tsel]
        out["nstub_check"].append(np.array([len(tsel), int((n_st != T["nStubs"][tsel]).sum())]))
        grp = F.floor_group({"tanL": tanl, "n_disk_stubs": n_disk})
        has_grp = grp >= 0
        C0h = C0.copy()
        C0h[has_grp] = hybrid_seed(Eh, ptb_h, C0[has_grp], grp[has_grp], T["nStubs"][tsel][has_grp].astype(int),
                                   pT[has_grp])
        # producer Q: a kink at every accepted hit but the innermost, acting on the hits after
        # it, theta0 = k / p (natural form), times the layers crossed to the next hit
        pos_q = np.where(valid, lay + rr / 1000.0, -1.0)
        nxt_l = np.concatenate([lay[:, 1:], np.zeros((n, 1), int)], 1)
        has_next = np.concatenate([valid[:, 1:], np.zeros((n, 1), bool)], 1)
        th2_q = np.where(valid & has_next, (RP.MULT_SCATT_TERM / p[:, None]) ** 2 * np.maximum(1, lay - nxt_l), 0.0)
        # Highland x1 on the tkLayout planes, layer-ranked
        S = len(planes)
        r_pl = np.broadcast_to(np.array([pl[0] for pl in planes]), (n, S)).copy()
        th2_h = np.stack([F.highland_theta2(np.full(n, pl[1]), p, tanl) for pl in planes], 1)
        rk = np.broadcast_to(rank, (n, S)).copy()
        pos_l = np.where(valid, lay.astype(float), -1.0)
        cfg = {"prodQ": (Rp, pos_q, rr, pos_q - 1e-9, th2_q, C0), "HL prodR": (Rp, pos_l, r_pl, rk, th2_h, C0),
               "HL calR": (Rc, pos_l, r_pl, rk, th2_h, C0), "HL calR hybrid": (Rc, pos_l, r_pl, rk, th2_h, C0h)}
        for c in CONF:
            R_, ph, r_s, ps, th2, Cs0 = cfg[c]
            Hm, y = fill(u0, v0, cA0, J0, np.zeros((len(ia), 5)))
            g = gbl_fit(Hm, y, R_, w, Cs0, ph, r_s, ps, th2, tanl)
            out[f"est {c} 0"].append(a[tsel] + g["da_in"])
            if c == "HL calR":
                out["sd_lin"].append(np.sqrt(np.diagonal(g["C_in"], axis1=1, axis2=2)))
            J_it, uvc = J0, (u0, v0, cA0)
            for it in range(1, NIT + 1):
                # Gauss-Newton: re-project every hit at the helix of ITS OWN segment
                shift = np.einsum("hik,hk->hi", g["G"][pos_t, slot], g["x"][pos_t])
                u, v, cA, J, _ = project(a[tt] + shift, L_h, mi_h)
                bad = ~(np.isfinite(u) & np.isfinite(J).all((1, 2)))
                n_fallback += int(bad.sum())
                u = np.where(bad, uvc[0], u); v = np.where(bad, uvc[1], v); cA = np.where(bad, uvc[2], cA)
                J = np.where(bad[:, None, None], J_it, J)
                Hm, y = fill(u, v, cA, J, shift)
                g = gbl_fit(Hm, y, R_, w, Cs0, ph, r_s, ps, th2, tanl)
                out[f"est {c} {it}"].append(a[tsel] + g["da_in"])
                J_it, uvc = J, (u, v, cA)
            out[f"prob {c}"].append(chi2_prob(g["chi2"], g["ndf"]))
            out[f"hitpull {c}"].append(np.nanmax(np.abs(g["pull"][valid][:, :2]), 1))
            if c == "HL calR hybrid":
                # forward-only analogue: the KF innovations on the SAME final linearisation
                k = kf_reference(Hm, y, R_, w, Cs0, ph, r_s, ps, th2, tanl)
                hp = np.where(valid, np.max(np.where(w, np.abs(np.nan_to_num(g["pull"], nan=0.0)), -np.inf), 2), -np.inf)
                srt = -np.sort(-hp, 1)
                pl = np.full((n, 4, 3), np.nan)
                # one crossing per layer is recorded: the first visited (outermost radius)
                for j in range(M - 1, -1, -1):
                    vj = valid[:, j]
                    pl[vj, lay[vj, j] - 1] = g["pull"][vj, j, :3]
                truth_t = truth_at_poca({cc: T[cc][tsel] for cc in TP_COLS})
                ev_loc = T["event"][tsel]
                TRK["event"].append(done + ev_loc)
                for b_ in ("run", "luminosityBlock", "event"):
                    TRK[f"id_{b_}"].append(ak.to_numpy(A[b_])[ev_loc])
                TRK["track"].append(tsel - toff[ev_loc])
                TRK["pt"].append(pT); TRK["chi2"].append(g["chi2"]); TRK["ndf"].append(g["ndf"])
                TRK["prob"].append(chi2_prob(g["chi2"], g["ndf"]))
                TRK["pull_max"].append(srt[:, 0])
                TRK["pull_2nd"].append(srt[:, 1] if M > 1 else np.full(n, -np.inf))
                TRK["pull_layer"].append(pl)
                TRK["kf_chi2_sum"].append(np.nansum(k["m2"], 1))
                TRK["kf_pull_max"].append(np.nanmax(np.where(w, np.abs(k["pull"]), -np.inf), (1, 2)))
                for key, cc in (("n_acc_same_tp", 0), ("n_acc_other_tp", 1), ("n_acc_noise", 2)):
                    TRK[key].append((valid & (cls == cc)).sum(1))
                TRK["gbl_par"].append(a[tsel] + g["da_in"])
                TRK["gbl_sigma"].append(np.sqrt(np.diagonal(g["C_in"], axis1=1, axis2=2)))
                TRK["res_gbl"].append(param_diff(a[tsel] + g["da_in"], truth_t))
                TRK["res_cmssw"].append(param_diff(np.stack([V[cc][tsel] for cc in HP], 1), truth_t))
                TRK["res_seed"].append(param_diff(a[tsel], truth_t))
        out["pt"].append(pT); out["nacc"].append(valid.sum(1))
        out["clean"].append(np.all(np.where(valid, cls == 0, True), 1))
        out["seed"].append(a[tsel]); out["cmssw"].append(np.stack([V[c][tsel] for c in HP], 1))
        out["truth"].append(truth_at_poca({c: T[c][tsel] for c in TP_COLS}))
        out["hitcls"].append(cls[valid])
        done += len(A)
        log(f"    cmssw harness: {done} events")
    cnt = np.sum(out.pop("counts"), 0)
    nsc = np.sum(out.pop("nstub_check"), 0)
    O = {k: np.concatenate(v) for k, v in out.items()}
    log(f"== L1TTrack_nStubs vs the stub-table count (the calibration's n_stubs): {nsc[1]} of {nsc[0]} tracks differ")
    if args.tracks_out:
        np.savez_compressed(args.tracks_out, **{k: np.concatenate(v) for k, v in TRK.items()},
                            param_order=np.array(HP), layer_order=np.array(["IL1", "IL2", "IL3", "IL4"]),
                            pull_dims=np.array(["u", "v", "cotAlpha"]))
        log(f"   wrote {args.tracks_out} ({sum(len(v) for v in TRK['chi2'])} tracks)")
    log(f"== (b0) accepted hits {cnt[0]}: not refittable here {cnt[1]} ({cnt[1] / max(cnt[0], 1):.4%}; module outside "
        f"the barrel geometry, seed projection misses the module, or a non-finite Jacobian column); "
        f"no stored producer seed projection (projSeedLocalX = -999) {cnt[2]} ({cnt[2] / max(cnt[0], 1):.2%})")
    for k, lab in (("proj_du", "u"), ("proj_dv", "v")):
        d = np.abs(O[k]) * 1e4
        log(f"   seed projection vs projSeedLocal{'X' if lab == 'u' else 'Y'}: |d{lab}| median {np.median(d):.4f} um, "
            f"p99 {np.percentile(d, 99):.3f} um, max {d.max():.2f} um, within 1 um {np.mean(d < 1):.5f} (n {len(d)})")
    log(f"   Gauss-Newton re-projections that failed and kept the previous linearisation: {n_fallback}")
    pt, clean = O["pt"], O["clean"]
    log(f"== (b1) GBL with the producer's Q, errors and dims vs the producer's own refit "
        f"({len(pt)} refit genuine-seed tracks, {clean.mean():.3f} with only same-TP accepted hits)")
    ds = param_diff(O["cmssw"], O["seed"])
    for it in (range(NIT + 1) if "prodQ" in CONF else ()):
        dq = param_diff(O[f"est prodQ {it}"], O["cmssw"])
        log(f"   iteration {it} ({'linearised at the seed' if it == 0 else 're-linearised'}): " + "; ".join(
            f"{p} rsig(GBL-CMSSW) {rs(dq[:, i])[0] * UNIT[p][0]:.3f} {UNIT[p][1]} "
            f"[<5% of the producer's shift: {np.mean(np.abs(dq[:, i]) < 0.05 * np.abs(ds[:, i])):.3f}]"
            for i, p in enumerate(HP)))
    log("   producer's own shift rsig(CMSSW - seed): " + " ".join(
        f"{p} {rs(ds[:, i])[0] * UNIT[p][0]:.3f} {UNIT[p][1]}" for i, p in enumerate(HP)))
    log("== (b1') linearisation: iterated minus seed-linearised GBL (Highland, calibrated R), in units of the fit sigma")
    for it in range(1, NIT + 1):
        d = param_diff(O[f"est HL calR {it}"], O[f"est HL calR {it - 1}"]) / O["sd_lin"]
        log(f"   iteration {it} - {it - 1}: " + " ".join(
            f"{p} rsig {rs(d[:, i])[0]:.3f} p99|.| {np.percentile(np.abs(d[:, i]), 99):.2f}" for i, p in enumerate(HP)))
    est = {"seed": O["seed"], "CMSSW": O["cmssw"]}
    for c in CONF:
        est[f"GBL {c} lin"] = O[f"est {c} 0"]
        est[f"GBL {c} it{NIT}"] = O[f"est {c} {NIT}"]
    for lab, sel in (("all", np.ones(len(pt), bool)), ("clean (all accepted hits same TP)", clean),
                     ("contaminated (>= 1 other-TP/noise hit)", ~clean)):
        log(f"== (b2) resolution vs truth@POCA, {lab}")
        resolution_table(log, "robust sigma (estimate - truth)", est, O["truth"], pt, sel)
    m2 = O["nacc"] >= 2
    for c in CONF:
        pr = O[f"prob {c}"][m2]
        ok = np.isfinite(pr)
        auc = roc_auc_score((~clean[m2])[ok], -pr[ok])
        log(f"== (b3) chi2 probability as a contamination flag, GBL {c} it{NIT} (tracks with >= 2 accepted hits): "
            f"AUC {auc:.3f}; flagged at P<0.01: clean {np.mean(pr[ok & clean[m2]] < 0.01):.3f} "
            f"contaminated {np.mean(pr[ok & ~clean[m2]] < 0.01):.3f}")
        hc, hp = O["hitcls"], O[f"hitpull {c}"]
        for k, lab in ((0, "same-TP"), (1, "other-TP"), (2, "noise")):
            m = hc == k
            if m.sum():
                log(f"   per-hit max(|smoothed pull u|,|v|), {lab} hits: n {m.sum():6d} median {np.median(hp[m]):.2f} "
                    f"frac > 3: {np.mean(hp[m] > 3):.3f}, > 5: {np.mean(hp[m] > 5):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--harness", choices=("oracle", "cmssw", "both"), default="both")
    ap.add_argument("--calib-dir", default=str(Path(__file__).parent / "results" / "refit_calibration"))
    ap.add_argument("--nano", default="/Users/nmangane/smartpixels/cmssw/work/otstub_arm/itot_tp_f*_100ev.root")
    ap.add_argument("--geometry", default="/Users/nmangane/smartpixels/cmssw/work/otstub_arm/spix_module_geometry_D121.json")
    ap.add_argument("--n-tracks", type=int, default=40000, help="oracle sample size (tracks)")
    ap.add_argument("--nev", type=int, default=None, help="cmssw harness event cap")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=2, help="Gauss-Newton re-linearisations (cmssw harness)")
    ap.add_argument("--hybrid-only", action="store_true",
                    help="oracle: fit only the hybrid seed; cmssw: only the Highland + calibrated-R "
                         "configurations (reported C and hybrid)")
    ap.add_argument("--tracks-out", default=None, help="cmssw: per-track npz of the hybrid-seed GBL")
    ap.add_argument("-o", "--log", default=None)
    a = ap.parse_args()
    fh = open(a.log, "w") if a.log else None

    def log(s):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n"); fh.flush()
    if a.harness in ("oracle", "both"):
        oracle(a, log)
    if a.harness in ("cmssw", "both"):
        cmssw(a, log)


if __name__ == "__main__":
    main()
