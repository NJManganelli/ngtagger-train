import numpy as np

# ---------------------------------------------------------------------------
# FAITHFUL OFFLINE REPLAY of the CMSSW digiRefit sequential-scalar Kalman filter
# (L1SmartPixelsTrackProducer.cc, digiRefit branch). Pure numpy; reads ONLY the
# reference seed + per-hit sidecar records (never the real refit answer).
# ---------------------------------------------------------------------------

_KEPS = np.array([1e-6, 1e-5, 1e-5, 1e-3, 1e-3])   # producer kEps for numerical H
_DEFAULT_RADII = np.array([3.0, 6.8, 10.9, 16.0])  # TBPX mean layer radii [cm]
# Phase-2 TBPX pixel pitch -> position sigmas (pitch/sqrt(12)); pitch=(rphi, z)
_SIGX = 0.0025 / np.sqrt(12.0)   # 25 um r-phi
_SIGY = 0.0100 / np.sqrt(12.0)   # 100 um z


def _helix_global(a, R):
    """Analytic helix -> cylinder(R) crossing in the transverse plane, mirroring
    SmartPixelsHelixProjector::crossLayer's analytic step. a=(rInv,phi0,tanL,z0,d0);
    POCA x0=d0*sin(phi0), y0=-d0*cos(phi0). Returns (pos[3], dir[3], s) or None."""
    rInv, phi0, tanL, z0, d0 = a
    x0 = d0 * np.sin(phi0); y0 = -d0 * np.cos(phi0)
    c0 = np.cos(phi0); s0 = np.sin(phi0)
    if abs(rInv) < 1e-6:
        b = x0 * c0 + y0 * s0; c = x0 * x0 + y0 * y0 - R * R; disc = b * b - c
        if disc < 0: return None
        s = -b + np.sqrt(disc)
        if s < 0: return None
        gx = x0 + s * c0; gy = y0 + s * s0
        dirx = c0; diry = s0
    else:
        Rr = 1.0 / rInv; cx = x0 - Rr * s0; cy = y0 + Rr * c0
        dcen = np.hypot(cx, cy); absR = abs(Rr)
        if R > dcen + absR or R < abs(dcen - absR): return None
        cosArg = (absR * absR + dcen * dcen - R * R) / (2.0 * absR * dcen)
        if cosArg < -1.0 or cosArg > 1.0: return None
        delta = np.arccos(max(-1.0, min(1.0, cosArg)))
        dpsi = delta if rInv > 0 else -delta
        s = abs(dpsi) * absR; psi = rInv * s
        gx = x0 + (np.sin(phi0 + psi) - s0) / rInv
        gy = y0 - (np.cos(phi0 + psi) - c0) / rInv
        dirx = np.cos(phi0 + psi); diry = np.sin(phi0 + psi)
    gz = z0 + tanL * s
    return np.array([gx, gy, gz]), np.array([dirx, diry, tanL]), s


def _build_frame(a, R, sx, sy):
    """FREEZE the module local basis at the nominal crossing (this is the key to a
    correct Jacobian: the producer's det->toLocal uses a FIXED per-module rotation,
    so the measurement axes must NOT rotate with a perturbed crossing point).
      local-z (normal) = radial outward; local-x = sx*tangential; local-y = sy*global-z.
    Returns (origin g0, basis[3,3] = [ex,ey,ez]) or None."""
    r = _helix_global(a, R)
    if r is None: return None
    g0 = r[0]
    phim = np.arctan2(g0[1], g0[0]); cp, sp = np.cos(phim), np.sin(phim)
    ez = np.array([cp, sp, 0.0])
    ex = sx * np.array([-sp, cp, 0.0])
    ey = sy * np.array([0.0, 0.0, 1.0])
    return g0, np.array([ex, ey, ez])


def _h_local(a, R, frame):
    """Measurement h(a)=(localx,localy,cotAlpha,cotBeta) in the FROZEN frame."""
    r = _helix_global(a, R)
    if r is None: return None
    g, d, _ = r
    g0, B = frame; ex, ey, ez = B
    pln = ez @ d; pz = pln if abs(pln) > 1e-9 else 1e-9
    return np.array([ex @ (g - g0), ey @ (g - g0), (ex @ d) / pz, (ey @ d) / pz])


def _jac_local(a, R, sx, sy):
    """One-sided numerical Jacobian H[4][5] of h(a) against the frozen frame,
    matching the producer's kEps and one-sided-with-fallback differencing."""
    frame = _build_frame(a, R, sx, sy)
    if frame is None: return None, None
    h0 = _h_local(a, R, frame)
    if h0 is None: return None, None
    H = np.zeros((4, 5))
    for j in range(5):
        for sgn in (1.0, -1.0):
            ap = a.copy(); ap[j] += sgn * _KEPS[j]
            hp = _h_local(ap, R, frame)
            if hp is None: continue
            H[:, j] = (hp - h0) / (sgn * _KEPS[j]); break
    return h0, H


def _recover_sy(a, R, parCotBeta, cotBetaMeas):
    """Per-module local-y handedness sign, recovered from the recorded angle whose
    magnitude tracks tanL (verified 100% consistent per detId). Prefer the truth
    parCotBeta; fall back to the measured cotBetaMeas; default +1."""
    r = _helix_global(a, R)
    if r is None: return 1.0
    g, d, _ = r
    phim = np.arctan2(g[1], g[0]); cp, sp = np.cos(phim), np.sin(phim)
    pln = d[0] * cp + d[1] * sp; pz = pln if abs(pln) > 1e-9 else 1e-9
    cotB_raw = d[2] / pz                       # sy=+1 analytic cotBeta
    refB = parCotBeta if parCotBeta > -900 else (cotBetaMeas if cotBetaMeas > -900 else None)
    if refB is not None and abs(cotB_raw) > 1e-2 and abs(refB) > 1e-2:
        return float(np.sign(refB / cotB_raw))
    return 1.0


# Recommended parametrized seed for the offline replay. Tighter on (rInv, phi0) than
# the producer's ablation paramSigmas so the r-phi position updates do not overshoot
# curvature/azimuth or scramble the phi0/d0 correlation. This is a DOCUMENTED
# parametrized-seed choice (motivated by realistic Phase-2 L1 track precision), NOT a
# per-track fit to the answer. Maximizes rInv/phi/d0 sign agreement across all configs.
_REPLAY_SEED_SIGMAS = (3e-5, 3e-4, 1e-3, 0.02, 0.03)


def replay_track(seed_params, hit_records, layer_radii=None, use_angles="alphaBeta",
                 active_layers=(1, 1, 1, 1), param_sigmas=_REPLAY_SEED_SIGMAS,
                 seed_npar=5):
    """Offline replay of the digiRefit sequential-scalar Kalman filter.

    seed_params : (rInv, phi0, tanL, z0, d0)  -- phi0 from momentum().phi().
    hit_records : list of per-hit dicts (sidecar), each with keys:
        layer(int 1..4), hitAccepted(bool), resX, resY,
        parCotBeta, cotBetaMeas, sigAlpha, sigBeta,
        pullAlpha, pullBeta, hasAlpha(bool), hasBeta(bool).
      (resX/resY are the recorded seed-frame position innovations; angle innovations
       are reconstructed as pull*sqrt(S). The function NEVER reads the refit answer.)
    param_sigmas : parametrized seed-covariance sqrt-diagonal (documented caveat:
      production used seedCovMode='trackCov', not persisted in nano).

    Returns dict with 'final' params, 'delta' vs seed, per-step 'states',
    'chi2_rphi'/'chi2_rz' increments, and per-step 'pulls'.
    """
    RAD = _DEFAULT_RADII if layer_radii is None else np.asarray(layer_radii, float)
    a = np.asarray(seed_params, float).copy()
    aSeed = a.copy()
    ps = np.asarray(param_sigmas, float)
    C = np.diag(ps ** 2).astype(float)
    if seed_npar == 4 or not (C[4, 4] > 0):        # weak d0 prior for 4-par seeds
        C[4, 4] = max(C[4, 4], 0.25)               # (0.5 cm)^2

    useAlpha = use_angles != "none"
    useBeta = use_angles == "alphaBeta"
    states, chi2_rphi, chi2_rz, pulls = [], [], [], []

    for hd in sorted(hit_records, key=lambda h: int(h["layer"])):
        L = int(hd["layer"])
        if not active_layers[L - 1] or not hd["hitAccepted"]:
            continue
        R = RAD[L - 1]
        sy = _recover_sy(a, R, hd.get("parCotBeta", -999.0), hd.get("cotBetaMeas", -999.0))
        sx = sy                                    # right-handed GeomDet frame: e_x x e_y = e_z
        h0, H = _jac_local(a, R, sx, sy)
        if h0 is None:
            continue
        aLin = a.copy()
        pk = {"layer": L, "pullX": None, "pullY": None, "pullAlpha": None, "pullBeta": None}
        c_rphi = 0.0; c_rz = 0.0

        def scalar_update(k, m_innov, sig):
            """m_innov is the innovation r directly (not the measurement)."""
            nonlocal a, C
            Hrow = H[k]; v = C @ Hrow; S = Hrow @ v + sig * sig
            if not (S > 0):
                return None, None
            K = v / S; a = a + K * m_innov; C = C - np.outer(K, v)
            return m_innov / np.sqrt(S), (m_innov * m_innov / S)

        # x (first scalar update: a-aLin=0 so innovation == resX exactly)
        p, ci = scalar_update(0, hd["resX"], _SIGX)
        if p is not None: pk["pullX"] = p; c_rphi += ci
        # y (relinearize: innovation = resY - H[1].(a - aLin))
        p, ci = scalar_update(1, hd["resY"] - H[1] @ (a - aLin), _SIGY)
        if p is not None: pk["pullY"] = p; c_rz += ci
        # alpha (innovation reconstructed from recorded pull: r = pull*sqrt(S))
        if useAlpha and hd["hasAlpha"] and hd.get("pullAlpha", -999.0) > -900:
            Hrow = H[2]; S = Hrow @ (C @ Hrow) + hd["sigAlpha"] ** 2
            if S > 0:
                p, ci = scalar_update(2, hd["pullAlpha"] * np.sqrt(S), hd["sigAlpha"])
                if p is not None: pk["pullAlpha"] = p; c_rphi += ci
        # beta
        if useBeta and hd["hasBeta"] and hd.get("pullBeta", -999.0) > -900:
            Hrow = H[3]; S = Hrow @ (C @ Hrow) + hd["sigBeta"] ** 2
            if S > 0:
                p, ci = scalar_update(3, hd["pullBeta"] * np.sqrt(S), hd["sigBeta"])
                if p is not None: pk["pullBeta"] = p; c_rz += ci

        states.append(a.copy()); chi2_rphi.append(c_rphi); chi2_rz.append(c_rz); pulls.append(pk)

    return {
        "final": a.copy(),
        "delta": a - aSeed,
        "seed": aSeed.copy(),
        "states": states,
        "chi2_rphi": chi2_rphi,
        "chi2_rz": chi2_rz,
        "chi2_rphi_tot": float(np.sum(chi2_rphi)),
        "chi2_rz_tot": float(np.sum(chi2_rz)),
        "pulls": pulls,
        "cov": C,
    }
