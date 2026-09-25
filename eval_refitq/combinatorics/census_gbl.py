"""Smoothed refits of census tracks in the census KF's own measurement model.

THE CENSUS FIT (kf_emulation.fit_tracks, as tp_findability runs it for every
exported track row): tmtt::KFParamsComb emulated in a global frame, positions
only (use_angles=False), inner to outer, diffuse prior, NO process noise:

    phi(r) = phi0 - d0/r - c kappa r  (+ 3rd-order helix terms)     r-phi, 3 params
    z(r)   = z0 + cot r               (+ 3rd-order term)             r-z,   2 params

Scattering is folded into each hit's azimuth VARIANCE, (0.00075/pT)^2, i.e.
treated as independent per hit. With Q = 0 the forward filter's final state is
the global least-squares solution, and its chi2 is the sum of innovation
(predicted-residual) chi2 -- the only residual information a forward-only
filter has.

TWO REFITS ON EXACTLY THE SAME HITS, ERRORS AND PRIORS:

  forward_filter  a replica of kf_run's position recursion that also returns
                  every innovation pull. Verified against kf_run: final state
                  and chi2 to rounding.
  gbl_fit         a broken-line fit (Blobel NIM A 566 (2006) 14; Kleinwort
                  NIM A 673 (2012) 107) in explicit-kink form: the 5 helix
                  parameters of the INNERMOST segment plus one transverse and
                  one longitudinal kink per scatterer, with N(0, theta0^2)
                  priors, solved jointly over all hits. The per-hit (0.00075/pT)
                  term is REMOVED from the variances, because the kinks carry
                  the scattering with its correlation -- a kink deflects every
                  outer hit coherently, which the per-hit term cannot express.
                  Kink priors: Highland (refit_calibration_fit.highland_theta2)
                  on the tkLayout IT planes x1 (validated fit-free in the refit
                  calibration) and the CMSSW D121 scan's OT module/support
                  material (material_xcheck.json, |eta| < 0.4, phi-averaged).
                  kf_emulation has no kink model to borrow; this is the stated
                  choice. With the kinks switched off and the per-hit term kept,
                  gbl_fit reproduces kf_run(ho=False) exactly -- the equivalence
                  check -- and its smoothed residuals are then the
                  smoother of the census fit itself.

Kink geometry in this frame. A transverse kink of projected angle theta at
radius r_s turns the azimuth direction by sec(lambda) theta, which adds
sec*theta*(1 - r_s/r) to phi(r) beyond r_s: a phi0 shift plus a d0 shift of
sec*theta*r_s (TTTrack d0). A longitudinal kink adds sec^2*theta*(r - r_s) to
z(r). Module material acts only on hits of OUTER layers (ordered by layer
radius, never by the hit's own radius, so a module never kinks its own hit);
a scatterer inside a track's innermost hit is dropped, so the parameters are
those of the innermost segment.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2 as _chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M      # noqa: E402
import kf_emulation as KF               # noqa: E402
import refit_calibration_fit as RF      # noqa: E402

LAYERS = np.asarray(KF.LAYER_ORDER)     # 1..4 IT, 11..16 OT barrel
# (position, x/X0, kind): kind "mod" = module material of that layer (acts on
# outer layers), "cyl" = passive cylinder at that radius
IT_MOD = {1: 0.0185, 2: 0.0187, 3: 0.0148, 4: 0.0148}
OT_MOD = {11: 0.0522, 12: 0.0447, 13: 0.0436, 14: 0.0190, 15: 0.0190, 16: 0.0190}
CYL = [(4.17, 0.0009), (7.29, 0.0010), (11.68, 0.0008), (15.84, 0.0009),
       (16.24, 0.0120), (20.95, 0.0117), (21.30, 0.0049), (57.3, 0.0049), (66.6, 0.0049)]
PRIOR_PREC_RPHI = np.array([1.0, 1.0 / 100.0])       # x0, x1 (x4 from d0 prior)
PRIOR_PREC_RZ = np.array([1.0 / 100.0, 1.0 / 1.0e4])  # x2, x3


def census_inputs(U, Q, trip, gidx, d0_prior_cm=KF.D0_PRIOR_CM):
    """The arrays kf_emulation.fit_tracks hands kf_run, plus the angle data.

    Same functions, same order of operations as fit_tracks(use_angles=False),
    so the census fit is reproduced exactly (checked by the caller).
    """
    T = KF.gather_hits(U, Q, gidx, KF.LAYER_ORDER)
    lay = np.tile(LAYERS, (T["R"].shape[0], 1))
    is_ot = lay > 10
    ga, gb = trip[0], trip[1]
    dr = U["globalR"][gb] - U["globalR"][ga]
    safe = np.where(np.abs(dr) > 0.5, dr, 1e9)
    kap0 = np.abs(M.wrap(U["globalPhi"][ga] - U["globalPhi"][gb]) / (M.C_BEND * safe))[:, None]
    pt0 = np.broadcast_to(1.0 / np.maximum(kap0, 1e-3), T["R"].shape)
    o0, o1 = KF.ot_variances(np.where(is_ot, lay - 10, 1), T["R"], pt0, z=T["Z"])
    i0, i1 = KF.it_variances(T["R"], T["SX"], T["SY"], pt0)
    # the resolution part alone: the scattering term evaluated at infinite pT
    o0r, _ = KF.ot_variances(np.where(is_ot, lay - 10, 1), T["R"], np.full_like(T["R"], 1e12), z=T["Z"])
    i0r, _ = KF.it_variances(T["R"], T["SX"], T["SY"], np.full_like(T["R"], 1e12))
    ref = np.where(T["VALID"], T["PH"], 0.0)
    first = np.argmax(T["VALID"], axis=1)
    phi_ref = ref[np.arange(len(first)), first]
    return {"T": T, "R": T["R"], "m0": M.wrap(T["PH"] - phi_ref[:, None]), "m1": T["Z"],
            "v0": np.where(is_ot, o0, i0), "v1": np.where(is_ot, o1, i1),
            "v0_res": np.where(is_ot, o0r, i0r), "valid": T["VALID"], "is_ot": is_ot,
            "phi_ref": phi_ref, "d0_prior_cm": d0_prior_cm}


def forward_filter(inp, ho=True):
    """kf_run's position recursion (Q = 0, inner to outer), returning the final
    state AND every innovation pull (resid / sqrt(R)), r-phi and r-z."""
    R_, m0, m1, v0, v1, valid = (inp[k] for k in ("R", "m0", "m1", "v0", "v1", "valid"))
    nt, nl = R_.shape
    x0 = np.zeros(nt); x1 = np.zeros(nt); x4 = np.zeros(nt); x2 = np.zeros(nt); x3 = np.zeros(nt)
    C00 = np.full(nt, 1.0); C01 = np.zeros(nt); C04 = np.zeros(nt); C11 = np.full(nt, 100.0)
    C14 = np.zeros(nt); C44 = np.full(nt, inp["d0_prior_cm"] ** 2 / 3.0)
    C22 = np.full(nt, 100.0); C23 = np.zeros(nt); C33 = np.full(nt, 1.0e4)
    ok = valid.sum(axis=1) >= 2
    pp = np.zeros((nt, nl)); pz = np.zeros((nt, nl))
    c2p = np.zeros(nt); c2z = np.zeros(nt)
    for L in range(nl):
        use = valid[:, L] & ok
        if not use.any():
            continue
        r = np.where(use, np.maximum(R_[:, L], 1e-3), 1.0)
        g = -1.0 / r
        hor = ((1.0 / 6.0) * np.power(x0 * r, 3) + (1.0 / 6.0) * np.power(x4 * g, 3)) if ho else 0.0
        res = np.where(use, m0[:, L] - (x0 * r + x1 + x4 * g + hor), 0.0)
        S0 = C00 * r + C01 + C04 * g; S1 = C01 * r + C11 + C14 * g; S4 = C04 * r + C14 + C44 * g
        Rv = np.where(use, r * S0 + S1 + g * S4 + v0[:, L], 1.0)
        Rv = np.where(np.abs(Rv) > 1e-30, Rv, 1e-30)
        pp[:, L] = np.where(use, res / np.sqrt(Rv), 0.0)
        c2p += np.where(use, res * res / Rv, 0.0)
        K0, K1, K4 = S0 / Rv, S1 / Rv, S4 / Rv
        x0 = np.where(use, x0 + res * K0, x0); x1 = np.where(use, x1 + res * K1, x1)
        x4 = np.where(use, x4 + res * K4, x4)
        C00 = np.where(use, C00 - K0 * S0, C00); C01 = np.where(use, C01 - K0 * S1, C01)
        C04 = np.where(use, C04 - K0 * S4, C04); C11 = np.where(use, C11 - K1 * S1, C11)
        C14 = np.where(use, C14 - K1 * S4, C14); C44 = np.where(use, C44 - K4 * S4, C44)
        hoz = (1.0 / 6.0) * r * np.power(x0 * r, 2) * x2 if ho else 0.0
        rz = np.where(use, m1[:, L] - (x2 * r + x3 + hoz), 0.0)
        S2 = C22 * r + C23; S3 = C23 * r + C33
        Rz = np.where(use, r * S2 + S3 + v1[:, L], 1.0)
        Rz = np.where(np.abs(Rz) > 1e-30, Rz, 1e-30)
        pz[:, L] = np.where(use, rz / np.sqrt(Rz), 0.0)
        c2z += np.where(use, rz * rz / Rz, 0.0)
        K2, K3 = S2 / Rz, S3 / Rz
        x2 = np.where(use, x2 + rz * K2, x2); x3 = np.where(use, x3 + rz * K3, x3)
        C22 = np.where(use, C22 - K2 * S2, C22); C23 = np.where(use, C23 - K2 * S3, C23)
        C33 = np.where(use, C33 - K3 * S3, C33)
    return {"x": np.stack([x0, x1, x4, x2, x3], 1), "chi2_rphi": c2p, "chi2_rz": c2z,
            "pull_rphi": pp, "pull_rz": pz, "ok": ok}


def _layer_radius(U):
    return {int(L): float(np.median(U["globalR"][U["layer"] == L])) for L in LAYERS
            if (U["layer"] == L).any()}


def scatterers(rmed):
    """(position used for ordering, radius, x/X0, is_module_of_layer or -1)."""
    out = [(rmed[L] + 1e-4, rmed[L], x, L) for L, x in {**IT_MOD, **OT_MOD}.items() if L in rmed]
    out += [(r, r, x, -1) for r, x in CYL]
    return sorted(out)


def gbl_fit(inp, x_start, rmed, kinks=True, ho=True, n_iter=2):
    """Joint (helix of the innermost segment + kinks) fit on the census hits.

    kinks=False keeps the census per-hit scattering variance and no kinks, which
    is exactly the least-squares problem kf_run solves: the equivalence check.
    Returns the final state, smoothed per-hit pulls, kink pulls and chi2/ndf.
    """
    R_, m0, m1, valid = inp["R"], inp["m0"], inp["m1"], inp["valid"]
    nt, nl = R_.shape
    r = np.maximum(R_, 1e-3)
    v0 = inp["v0_res"] if kinks else inp["v0"]
    v1 = inp["v1"]
    w0 = np.where(valid, 1.0 / v0, 0.0); w1 = np.where(valid, 1.0 / v1, 0.0)
    x = x_start.copy()
    cot = x[:, 3]
    sec2 = 1.0 + cot * cot; sec = np.sqrt(sec2)
    pt = 1.0 / np.maximum(np.abs(x[:, 0] / M.C_BEND), 1e-6)
    p = pt * sec
    sc = scatterers(rmed) if kinks else []
    S = len(sc)
    lay_pos = np.array([rmed.get(int(L), np.inf) for L in LAYERS])
    # act[t, l, s]: scatterer s deflects the hit on layer l of track t
    if S:
        spos = np.array([s[0] for s in sc]); srad = np.array([s[1] for s in sc])
        act = (lay_pos[None, :, None] > spos[None, None, :]) & valid[:, :, None]
        inner = np.where(valid, lay_pos[None, :], np.inf).min(axis=1)
        act &= (spos[None, None, :] > inner[:, None, None])
        th2 = np.stack([RF.highland_theta2(np.full(nt, s[2]), p, cot) for s in sc], 1)
        th2 = np.maximum(th2, 1e-30)
        dT = sec[:, None, None] * (1.0 - srad[None, None, :] / r[:, :, None]) * act
        dL = sec2[:, None, None] * (r[:, :, None] - srad[None, None, :]) * act
    x4p = 3.0 / inp["d0_prior_cm"] ** 2
    kT = np.zeros((nt, S)); kL = np.zeros((nt, S))
    for it in range(n_iter):
        x0, x1, x4, x2, x3 = x.T
        g = -1.0 / r
        hor = ((1.0 / 6.0) * (x0[:, None] * r) ** 3 + (1.0 / 6.0) * (x4[:, None] * g) ** 3) if ho else 0.0
        pred0 = x0[:, None] * r + x1[:, None] + x4[:, None] * g + hor
        J0 = np.stack([r + (0.5 * (x0[:, None] * r) ** 2 * r if ho else 0.0), np.ones_like(r),
                       g + (0.5 * (x4[:, None] * g) ** 2 * g if ho else 0.0)], 2)
        hoz = (1.0 / 6.0) * r * (x0[:, None] * r) ** 2 * x2[:, None] if ho else 0.0
        pred1 = x2[:, None] * r + x3[:, None] + hoz
        J1 = np.stack([r + ((1.0 / 6.0) * r * (x0[:, None] * r) ** 2 if ho else 0.0), np.ones_like(r)], 2)
        if S:
            J0 = np.concatenate([J0, dT], 2); J1 = np.concatenate([J1, dL], 2)
        if S:
            pred0 = pred0 + np.einsum("tls,ts->tl", dT, kT)
            pred1 = pred1 + np.einsum("tls,ts->tl", dL, kL)
        res0 = np.where(valid, m0 - pred0, 0.0)
        res1 = np.where(valid, m1 - pred1, 0.0)
        sol0, cov0 = _solve(J0, w0, res0, np.r_[PRIOR_PREC_RPHI, x4p], x[:, [0, 1, 2]], th2 if S else None, kT)
        sol1, cov1 = _solve(J1, w1, res1, PRIOR_PREC_RZ, x[:, [3, 4]], th2 if S else None, kL)
        x = x + np.stack([sol0[:, 0], sol0[:, 1], sol0[:, 2], sol1[:, 0], sol1[:, 1]], 1)
        kT = kT + sol0[:, 3:]
        kL = kL + sol1[:, 2:]
    # final residuals and covariances at the solution
    x0, x1, x4, x2, x3 = x.T
    g = -1.0 / r
    hor = ((1.0 / 6.0) * (x0[:, None] * r) ** 3 + (1.0 / 6.0) * (x4[:, None] * g) ** 3) if ho else 0.0
    e0 = m0 - (x0[:, None] * r + x1[:, None] + x4[:, None] * g + hor)
    hoz = (1.0 / 6.0) * r * (x0[:, None] * r) ** 2 * x2[:, None] if ho else 0.0
    e1 = m1 - (x2[:, None] * r + x3[:, None] + hoz)
    if S:
        e0 = e0 - np.einsum("tls,ts->tl", dT, kT)
        e1 = e1 - np.einsum("tls,ts->tl", dL, kL)
    ve0 = np.where(valid, v0 - np.einsum("tla,tab,tlb->tl", J0, cov0, J0), 1.0)
    ve1 = np.where(valid, v1 - np.einsum("tla,tab,tlb->tl", J1, cov1, J1), 1.0)
    p0 = np.where(valid, e0 / np.sqrt(np.maximum(ve0, 1e-30)), 0.0)
    p1 = np.where(valid, e1 / np.sqrt(np.maximum(ve1, 1e-30)), 0.0)
    c2 = (np.where(valid, e0 * e0 * w0, 0.0).sum(1) + np.where(valid, e1 * e1 * w1, 0.0).sum(1))
    c2p = np.where(valid, e0 * e0 * w0, 0.0).sum(1)
    c2z = np.where(valid, e1 * e1 * w1, 0.0).sum(1)
    out = {"x": x, "pull_rphi": p0, "pull_rz": p1}
    if S:
        activ = act.any(axis=1)
        vk0 = th2 - np.diagonal(cov0[:, 3:, 3:], axis1=1, axis2=2)
        vk1 = th2 - np.diagonal(cov1[:, 2:, 2:], axis1=1, axis2=2)
        kp = np.maximum(np.abs(kT) / np.sqrt(np.maximum(vk0, 1e-30)), np.abs(kL) / np.sqrt(np.maximum(vk1, 1e-30)))
        out["max_kink_pull"] = np.where(activ, kp, 0.0).max(1)
        kprior = np.where(activ, kT ** 2 / th2 + kL ** 2 / th2, 0.0).sum(1)
        c2 = c2 + kprior
        c2p = c2p + np.where(activ, kT ** 2 / th2, 0.0).sum(1)
        c2z = c2z + np.where(activ, kL ** 2 / th2, 0.0).sum(1)
        out["kT"], out["kL"], out["dT"], out["dL"], out["act"] = kT, kL, dT, dL, act
        out["srad"] = np.array([s[1] for s in sc])
    n = valid.sum(1)
    out.update({"chi2": c2, "chi2_rphi": c2p, "chi2_rz": c2z, "ndf": np.maximum(2 * n - 5, 1),
                "ndf_rphi": np.maximum(n - 3, 1), "ndf_rz": np.maximum(n - 2, 1)})
    out["prob"] = _chi2.sf(c2, out["ndf"])
    return out


def _solve(J, w, res, prior_prec_x, x_now, th2, k_now):
    """Batched Gauss-Newton step: minimise sum w (res - J d)^2 + priors.

    Parameter priors are on the ABSOLUTE state (mean 0, kf_run's diffuse prior),
    so the step carries -prec * x_now; likewise -k_now / theta0^2 for the kinks.
    """
    nt, nl, k = J.shape
    nx = len(prior_prec_x)
    A = np.einsum("tla,tl,tlb->tab", J, w, J)
    b = np.einsum("tla,tl,tl->ta", J, w, res)
    A[:, np.arange(nx), np.arange(nx)] += prior_prec_x[None, :]
    b[:, :nx] -= prior_prec_x[None, :] * x_now
    if th2 is not None:
        A[:, np.arange(nx, k), np.arange(nx, k)] += 1.0 / th2
        b[:, nx:] -= k_now / th2
    cov = np.linalg.inv(A)
    return np.einsum("tab,tb->ta", cov, b), cov


def angle_pulls_local(inp, fit_x, gbl):
    """SmartPixels angle pulls against the LOCAL smoothed direction at each IT
    hit (kinks inside the hit turn it), same measurement variance and layer
    scales as kf_emulation.angle_pulls. alpha measures x0 + d0_eff/r^2 with
    d0_eff = d0 + sum sec*theta*r_s over the kinks inside; beta measures the
    local cot = cot + sum sec^2*theta_L."""
    T = inp["T"]
    r = np.maximum(T["R"], 1e-3)
    sa, sb = KF._layer_scales(T["layers"])
    x0, x4, x2 = fit_x[:, 0], fit_x[:, 2], fit_x[:, 3]
    d0_eff = np.broadcast_to(x4[:, None], r.shape).copy()
    cot_loc = np.broadcast_to(x2[:, None], r.shape).copy()
    if "kT" in gbl:
        sec = np.sqrt(1.0 + x2 * x2)
        d0_eff = d0_eff + sec[:, None] * np.einsum("tls,ts,s->tl", gbl["act"], gbl["kT"], gbl["srad"])
        cot_loc = cot_loc + (1.0 + x2 * x2)[:, None] * np.einsum("tls,ts->tl", gbl["act"], gbl["kL"])
    pa = ((-M.C_BEND * T["KA"]) - (x0[:, None] + d0_eff / (r * r))) / (M.C_BEND * np.maximum(T["SKA"], 1e-9) * sa)
    pb = (T["CT"] - cot_loc) / (np.maximum(T["SCT"], 1e-9) * sb)
    use = T["VALID"] & T["ANG"]
    return np.where(use, pa, 0.0), np.where(use, pb, 0.0), use


def features(inp, fwd, gbl, pa, pb, use_ang):
    """Per-track GBL (smoothed) and forward-only features."""
    valid, is_ot = inp["valid"], inp["is_ot"]
    hp = np.maximum(np.abs(gbl["pull_rphi"]), np.abs(gbl["pull_rz"]))
    hp = np.where(valid, hp, 0.0)
    srt = np.sort(hp, 1)[:, ::-1]
    fp = np.where(valid, np.maximum(np.abs(fwd["pull_rphi"]), np.abs(fwd["pull_rz"])), 0.0)
    fsrt = np.sort(fp, 1)[:, ::-1]
    ap = np.where(use_ang, np.maximum(np.abs(pa), np.abs(pb)), 0.0)
    asrt = np.sort(ap, 1)[:, ::-1]
    nang = use_ang.sum(1)
    return {
        "gbl_chi2_ndf": gbl["chi2"] / gbl["ndf"],
        "gbl_chi2_rphi_ndf": gbl["chi2_rphi"] / gbl["ndf_rphi"],
        "gbl_chi2_rz_ndf": gbl["chi2_rz"] / gbl["ndf_rz"],
        "gbl_logprob": np.log10(np.maximum(gbl["prob"], 1e-300)),
        "gbl_max_pull": srt[:, 0], "gbl_second_pull": srt[:, 1],
        "gbl_max_pull_it": np.where(valid & ~is_ot, hp, 0.0).max(1),
        "gbl_max_pull_ot": np.where(valid & is_ot, hp, 0.0).max(1),
        "gbl_max_kink_pull": gbl.get("max_kink_pull", np.zeros(len(hp))),
        "gbl_chi2_angle_per_cl": np.where(use_ang, pa * pa + pb * pb, 0.0).sum(1) / np.maximum(nang, 1),
        "gbl_max_angle_pull": asrt[:, 0], "gbl_second_angle_pull": asrt[:, 1],
        "fwd_max_innov_pull": fsrt[:, 0], "fwd_second_innov_pull": fsrt[:, 1],
        "fwd_max_innov_pull_it": np.where(valid & ~is_ot, fp, 0.0).max(1),
        "fwd_max_innov_pull_ot": np.where(valid & is_ot, fp, 0.0).max(1),
    }


FEATURE_NAMES = ("gbl_chi2_ndf", "gbl_chi2_rphi_ndf", "gbl_chi2_rz_ndf", "gbl_logprob",
                 "gbl_max_pull", "gbl_second_pull", "gbl_max_pull_it", "gbl_max_pull_ot",
                 "gbl_max_kink_pull", "gbl_chi2_angle_per_cl", "gbl_max_angle_pull",
                 "gbl_second_angle_pull", "fwd_max_innov_pull", "fwd_second_innov_pull",
                 "fwd_max_innov_pull_it", "fwd_max_innov_pull_ot")
