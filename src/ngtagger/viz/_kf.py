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
    # Mirror the CMSSW producer's SIGN FIX (SmartPixelsHelixProjector::crossLayer):
    # the stored rInv follows the CMSSW TTTrack convention, which curls opposite to
    # this (sin(phi0+psi))/rInv parametrization. The producer negates the signed
    # curvature at crossLayer entry so the IT crossing lands on the correct (real-
    # digi) side; this replay must negate identically to stay faithful to it (and to
    # match the OT stubs / drawn helix). z0/tanL and arc length are unaffected.
    rInv = -rInv
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


# ---------------------------------------------------------------------------
# Config-wide FULL-HIT-SET chi2 (viz round 4). Evaluate ONE track state against a
# FIXED hit set: the linked OT BARREL stubs + the accepted SmartPixels hits on
# the config's ACTIVE layers. Every displayed variant of a (track, config) -- the
# OT-only L1Track, each intermediate KF state, the final refit -- is charged
# against the SAME set with the SAME per-hit sigmas, so the reduced chi2 column
# is directly comparable across rows (the OT-only track is now "charged" for the
# smartpix hits its fit never saw).
#   * IT hits: the KF measurement model's LOCAL-X (r-phi) channel with _SIGX
#     (frozen local frame at the state's predicted crossing). The hit GLOBAL
#     positions are reconstructed ONCE per track by
#     :func:`reconstruct_hit_globals` (the recorded resX/resY are innovations vs
#     the RUNNING AAAA/alphaBeta production state, so the reconstruction replays
#     that pass and places hit i at its PRE-update predicted crossing + res); the
#     same fixed positions are then shared by every evaluated state and config.
#     The IT LOCAL-Y (z) channel is EXCLUDED from the comparator: the hit z can
#     only be reconstructed through the replay's z trajectory, which is the
#     documented ILLUSTRATIVE direction (trackCov correlations not persisted), so
#     charging it at the 29 um _SIGY weight would swamp the column with
#     reconstruction noise (verified numerically: with IT-y included the produced
#     refit shows chi2 GROWTH on most tracks whose d0 demonstrably moved toward
#     truth; with the transverse-faithful channels the produced refit lowers the
#     full-set chi2 on ~90% of d0-improved tracks). OT z IS included -- stub z is
#     real data and the state's z extrapolation is honest.
#   * OT stubs: extrapolate the helix to the stub cylinder radius and take the
#     (r-phi, z) residuals with an approximate Phase-2 OT resolution model -- PS
#     modules (barrel L1-3): 100 um r-phi pitch + 1.467 mm macro-pixel z; 2S
#     (L4-6): 90 um pitch + 5.025 cm strip length -- with the r-phi sigma
#     inflated by a simple multiple-scattering lever-arm term _OT_MS_COEFF*r/pt.
#     ENDCAP stubs are EXCLUDED (their disk/strip measurement model is not
#     persisted here; the overview panel likewise draws only barrel stubs). The
#     L1 fit's internal weights are not persisted in nano, so this is a
#     COMPARATOR model, not the fit's own chi2; identical sigmas for all rows
#     means the row RANKING never depends on the sigma choice.
# ---------------------------------------------------------------------------
_OT_PS_SIG_RPHI = 0.0100 / np.sqrt(12.0)   # PS 100 um pitch      [cm]
_OT_PS_SIG_Z = 0.1467 / np.sqrt(12.0)      # PS 1.467 mm macropix [cm]
_OT_2S_SIG_RPHI = 0.0090 / np.sqrt(12.0)   # 2S 90 um pitch       [cm]
_OT_2S_SIG_Z = 5.025 / np.sqrt(12.0)       # 2S 5.025 cm strip    [cm]
_OT_MS_COEFF = 0.002                       # MS r-phi inflation: coeff*r[cm]/pt [cm]


def _wrap_phi(dphi):
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def _ot_stub_sigmas(stub, pt):
    """(sig_rphi, sig_z) for one OT barrel stub: PS (layers 1-3) vs 2S (4-6),
    with the MS lever-arm inflation on r-phi."""
    ps = int(stub.get("layer", 1)) <= 3
    sig_rphi = _OT_PS_SIG_RPHI if ps else _OT_2S_SIG_RPHI
    sig_z = _OT_PS_SIG_Z if ps else _OT_2S_SIG_Z
    r = float(stub.get("r", np.hypot(stub["x"], stub["y"])))
    ms = _OT_MS_COEFF * r / max(abs(pt), 1.0)
    return float(np.hypot(sig_rphi, ms)), sig_z


def _cyl_residuals(a, sx, sy_, sz, r_surf):
    """(res_rphi, res_z) of the point (sx, sy_, sz) at cylinder radius r_surf vs
    the helix extrapolated to that radius -> (r*dphi, dz). If the helix never
    reaches the radius (soft curler), fall back to the transverse closest-
    approach point on the helix circle so the chi2 stays finite/comparable."""
    rInv, phi0, tanL, z0, d0 = a
    phi_p = np.arctan2(sy_, sx)
    cr = _helix_global(np.asarray(a, float), r_surf)
    if cr is not None:
        g, _, _ = cr
        return (r_surf * _wrap_phi(np.arctan2(g[1], g[0]) - phi_p), g[2] - sz)
    # fallback: transverse closest approach on the (sign-fixed) helix circle
    rf = -rInv
    x0 = d0 * np.sin(phi0); y0 = -d0 * np.cos(phi0)
    if abs(rf) < 1e-6:          # straight line: project the point onto the line
        dx, dy = np.cos(phi0), np.sin(phi0)
        s = (sx - x0) * dx + (sy_ - y0) * dy
        perp = (sx - x0) * (-dy) + (sy_ - y0) * dx
        return perp, z0 + tanL * max(s, 0.0) - sz
    R = 1.0 / rf
    cx = x0 - R * np.sin(phi0); cy = y0 + R * np.cos(phi0)
    dist = max(np.hypot(sx - cx, sy_ - cy), 1e-9)
    res_perp = dist - abs(R)                           # distance to the circle
    ang_p = np.arctan2(sy_ - cy, sx - cx)
    ang_0 = np.arctan2(y0 - cy, x0 - cx)
    s = abs(_wrap_phi(ang_p - ang_0)) * abs(R)         # arc to the closest point
    return res_perp, z0 + tanL * s - sz


def reconstruct_hit_globals(seed_params, hit_records, layer_radii=None,
                            param_sigmas=None):
    """Reconstruct each accepted IT hit's GLOBAL position from the sidecar.

    The recorded (resX, resY) are innovations vs the RUNNING refit state of the
    AAAA/alphaBeta production pass (the sidecar's pass), in the frozen local frame
    of that state's predicted crossing -- NOT vs the seed. So: replay that pass
    and place the i-th accepted hit at (pre-update predicted crossing) +
    resX*ex + resY*ey. The reconstruction is per-track and config-independent:
    the SAME fixed positions feed every displayed (config, angle) variant.

    Returns dict layer -> np.array([x, y, z]) for each localizable accepted hit.
    """
    RAD = _DEFAULT_RADII if layer_radii is None else np.asarray(layer_radii, float)
    ps = _REPLAY_SEED_SIGMAS if param_sigmas is None else param_sigmas
    rep = replay_track(seed_params, hit_records, layer_radii=RAD,
                       use_angles="alphaBeta", active_layers=(1, 1, 1, 1),
                       param_sigmas=ps, seed_npar=5)
    traj = rep["traj_states"]
    out = {}
    k = 0                                   # pre-update stage index into traj
    for hd in sorted(hit_records, key=lambda h: int(h["layer"])):
        L = int(hd["layer"])
        if not hd["hitAccepted"]:
            continue
        a_pre = np.asarray(traj[min(k, len(traj) - 1)], float)
        R = RAD[L - 1]
        sy = _recover_sy(a_pre, R, hd.get("parCotBeta", -999.0),
                         hd.get("cotBetaMeas", -999.0))
        frame = _build_frame(a_pre, R, sy, sy)
        if frame is None:                   # unlocalizable: skipped identically
            continue                        # by the replay -> stage not consumed
        g0, B = frame
        out[L] = g0 + hd["resX"] * B[0] + hd["resY"] * B[1]
        k += 1
    return out


def hitset_chi2(params, stubs, hit_globals, active_layers=(1, 1, 1, 1),
                layer_radii=None, pt=None):
    """Chi2 of ONE track state against the FIXED config-wide full hit set.

    params      : the 5-par state to evaluate (rInv, phi0, tanL, z0, d0).
    stubs       : list of real OT stub dicts (x/y/z, r, layer, isBarrel); only
                  BARREL stubs are counted (endcap excluded, see block comment).
    hit_globals : dict layer -> global IT hit position from
                  :func:`reconstruct_hit_globals` (computed ONCE per track so the
                  hit set is literally identical for every evaluated state);
                  only ACTIVE layers are counted.
    pt          : seed pt [GeV] for the OT multiple-scattering sigma inflation.

    Returns dict(chi2, chi2_ot, chi2_it, n_meas): n_meas = 2 per counted OT stub
    (r-phi + z) + 1 per counted IT hit (local-x only; the IT z channel is
    excluded, see the block comment), identical for every state of a given
    (track, config) -- the caller divides by the common ndof = n_meas - 5.
    """
    RAD = _DEFAULT_RADII if layer_radii is None else np.asarray(layer_radii, float)
    a = np.asarray(params, float)
    pt = 2.0 if pt is None else float(pt)

    chi2_ot = 0.0
    n_meas = 0
    for stub in stubs:
        if not stub.get("isBarrel", True):
            continue
        r_stub = float(stub.get("r", np.hypot(stub["x"], stub["y"])))
        res1, res2 = _cyl_residuals(a, float(stub["x"]), float(stub["y"]),
                                    float(stub["z"]), r_stub)
        sig1, sig2 = _ot_stub_sigmas(stub, pt)
        chi2_ot += (res1 / sig1) ** 2 + (res2 / sig2) ** 2
        n_meas += 2

    chi2_it = 0.0
    for L in sorted(hit_globals):
        if not active_layers[L - 1]:
            continue
        g_hit = np.asarray(hit_globals[L], float)
        R = RAD[L - 1]
        frame = _build_frame(a, R, 1.0, 1.0)   # sign of the axes is chi2-neutral
        if frame is not None:
            g0, B = frame
            rx = B[0] @ (g_hit - g0)
        else:                                  # state misses the layer: fall back
            rx, _ = _cyl_residuals(a, g_hit[0], g_hit[1], g_hit[2],
                                   float(np.hypot(g_hit[0], g_hit[1])))
        chi2_it += (rx / _SIGX) ** 2           # local-x only (IT z excluded)
        n_meas += 1

    return dict(chi2=chi2_ot + chi2_it, chi2_ot=chi2_ot, chi2_it=chi2_it,
                n_meas=n_meas)


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
    'chi2_rphi'/'chi2_rz' increments, and per-step 'pulls'. Also returns the full
    COVARIANCE-EVOLUTION trajectory used by the animated replay: 'traj_states'
    (state at seed, after-L1, ..., final -- length = n_updates+1, one entry per KF
    stage) and 'traj_sigmas' (the matching sqrt-diagonal per-param 1sigma for the 5
    params, same order as the state (rInv, phi0, tanL, z0, d0)). 'traj_covs' holds
    the full 5x5 covariance at each stage. This is the literal
    "full covariance matrix computations from L1Track level, first update, ..." so the
    viz can show each param's sigma SHRINK as each IT hit is folded in.
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

    # Covariance-evolution trajectory: stage 0 = the seed (L1Track level), then one
    # entry after every accepted IT update. The seed sigma is the parametrized-seed
    # sqrt-diagonal (documented caveat); each update shrinks it as C = C - K v^T.
    def _sigma_diag(cov):
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))

    traj_states = [aSeed.copy()]
    traj_covs = [C.copy()]
    traj_sigmas = [_sigma_diag(C)]

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
        traj_states.append(a.copy()); traj_covs.append(C.copy()); traj_sigmas.append(_sigma_diag(C))

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
        # covariance-evolution trajectory (seed -> after each accepted update)
        "traj_states": traj_states,
        "traj_sigmas": traj_sigmas,
        "traj_covs": traj_covs,
    }
