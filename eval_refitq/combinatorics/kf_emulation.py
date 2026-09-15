"""Emulation of the prompt hybrid Kalman filter, run on SmartPixels seeds.

WHICH KALMAN FILTER, because there are two and they are not the same. The prompt
hybrid chain fits with tmtt::KFParamsComb -- HybridFit.cc:181 constructs it
directly -- NOT with L1Trigger/TrackFindingTracklet/src/KalmanFilter.cc, which
belongs to the newer TrackerTFP path. An earlier version of this file emulated
the latter; the differences are in the measurement model and they matter.

WHAT KFParamsComb DOES, read off L1Trigger/TrackFindingTMTT:

  * DIFFUSE PRIOR, then one update per stub (KFParamsComb.cc:52-64). There is no
    exact two-point seed. The priors are
        inv2R 0.0314*invPtToInvR , phi0 0.0102 , z0 5.0 , tanL 0.5 , d0 1.0 cm
    and d0Sigma = 1.0 cm is why D0_PRIOR_CM below is 1.0 and should stay there:
    tuning it to reproduce a resolution would be fitting the prior to cover for
    a defect elsewhere.

  * NO PROCESS NOISE. KFbase.cc:462 is explicit -- "Get scattering contribution
    to helix parameter covariance (currently zero)". Scattering lives in the
    measurement variance instead.

  * BARREL MEASUREMENT VARIANCE (KFParamsComb::matrixV), which with the shipped
    higher-order flags (KalmanHOtilted False, KalmanHOfw True) reduces to

        vphi = (sigmaPerp / r)^2 + (MULT_SCATT_TERM / pT)^2
        vz   =  sigmaPar^2

    with sigmaPerp = stripPitch/sqrt(12) and sigmaPar = stripLength/sqrt(12). No
    cluster-width factor, no additive phi term, no tilt correction: the tilted
    branch is off and kalmanHOfw pins vz to sigmaPar^2 regardless.

    The SensorModule model this file used before -- clusterWidth * pitch / r plus
    an additive addPhiUncertainty -- belongs to the OTHER filter. It made the
    r-phi variance 5x too large at r = 100 cm and, being pT-independent where the
    scattering term is not, mis-weighted high-pT stubs -- precisely where
    sigma(kappa) is determined.

FIVE PARAMETERS. The samples are produced with promptHnpar=5, so the prompt
collection carries a real fitted d0 and a 5x5 covariance; KFbase pins d0 only on
nHelixPar==4. The model is the one the exact three-point solve also uses:

    phi(r) = phi0 + d0 / r - c * kappa * r        (r-phi, 3 parameters)
    z(r)   = z0   + cot * r                       (r-z,   2 parameters)

so the state is (x0, x1, x4 | x2, x3) = (-c*kappa, phi0, d0 | cot, z0) and the
two halves stay decoupled -- a transverse impact parameter does not touch r-z to
first order.

SMARTPIXELS CLUSTERS enter by the same form, with the nano's measured per-cluster
CPE sigmas playing sigmaPerp and sigmaPar, which they already are. The scattering
constant is the OT's own, applied to a system it was not tuned for; that is the
one borrowed piece.

Radii are ABSOLUTE, not offsets from a reference: the 1/r term has to be d0/r to
mean d0. The fixed-point digitisation is not emulated at all.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402

# ---- L1Trigger/TrackFindingTMTT + TrackerDTC Setup_cfi.py ----------------
CBC_PITCH, CBC_LENGTH = 0.009, 5.025        # 2S strip pitch / length, cm
MPA_PITCH, MPA_LENGTH = 0.01, 0.1467        # PS pixel pitch / length, cm
INV_ROOT12 = 1.0 / np.sqrt(12.0)
MULT_SCATT_TERM = 0.00075                   # KalmanMultiScattTerm, rad*GeV
PS_LAYERS = (1, 2, 3)                       # OT barrel 1-3 are PS, 4-6 are 2S


def ot_variances(layer, r, pt):
    """(vphi, vz) for OT barrel stubs, following KFParamsComb::matrixV."""
    ps = np.isin(layer, PS_LAYERS)
    sigma_perp = np.where(ps, MPA_PITCH, CBC_PITCH) * INV_ROOT12
    sigma_par = np.where(ps, MPA_LENGTH, CBC_LENGTH) * INV_ROOT12
    scat = MULT_SCATT_TERM / np.maximum(np.abs(pt), 1e-3)
    return ((sigma_perp / np.maximum(r, 1e-3)) ** 2 + scat ** 2,
            np.broadcast_to(sigma_par ** 2, np.shape(r)).copy())


def it_variances(r, sigX, sigY, pt):
    """(vphi, vz) for SmartPixels clusters, by the same form.

    sigX and sigY are the nano's measured per-cluster CPE sigmas and are already
    the Gaussian sigmas that sigmaPerp and sigmaPar are, so they substitute
    directly. Only the scattering constant is borrowed from the OT.
    """
    scat = MULT_SCATT_TERM / np.maximum(np.abs(pt), 1e-3)
    return ((np.maximum(sigX, 1e-6) / np.maximum(r, 1e-3)) ** 2 + scat ** 2,
            np.maximum(sigY, 1e-6) ** 2)


D0_PRIOR_CM = 1.0        # uniform half-range for the d0 prior; see kf_run


def kf_run(H, m0, m1, v0, v1, valid, ma=None, va=None, mb=None, vb=None,
           has_angle=None, d0_prior_cm=D0_PRIOR_CM):
    """Vectorised FIVE-parameter KF over ntrack x nlayer arrays, inner to outer.

    THE MODEL, and it is the same one the exact three-point solve uses:

        phi(r) = phi0 + d0 / r - c * kappa * r        (r-phi, 3 parameters)
        z(r)   = z0   + cot * r                       (r-z,   2 parameters)

    so the state is (x0, x1, x4 | x2, x3) = (-c*kappa, phi0, d0 | cot, z0) and
    the two halves stay DECOUPLED exactly as in CMSSW -- a transverse impact
    parameter does not touch r-z to first order. The r-phi half is 3x3 and the
    r-z half 2x2; there are no cross terms between them.

    RELATION TO THE OT FILTER. KalmanFilter.cc carries an x4 that enters the
    residual as x4/H (line 101), but update() propagates only x0..x3, so the
    filter that actually runs there is 4-parameter and prompt. This is the
    5-parameter version that slot anticipates, which is the one SmartPixels
    needs: with d0 fixed at zero a real impact parameter is absorbed into
    curvature at 10.34 per cm for an IL1+IL2 pair, which is the coupling these
    studies spent their time removing.

    Radii are ABSOLUTE, not offsets from a reference radius: the 1/r term has to
    be d0/r to mean d0. CMSSW subtracts a reference radius purely to keep its
    fixed-point ranges small, which does not change the fit and is not emulated
    here -- nothing else about the digitisation is either.

    THE D0 PRIOR IS A REAL CHOICE, not a formality. Two seed clusters give two
    azimuths for three r-phi parameters, so d0 starts at 0 with a uniform prior
    of half-range d0_prior_cm (variance d0^2/3, the same uniform convention
    State.cc uses for its measurement variances) and the third and later layers
    determine it. Too tight a prior pulls d0 toward zero and re-couples curvature
    to displacement; too wide degrades kappa on prompt tracks, which are 95% of
    the sample. It is exposed so it can be scanned rather than assumed.

    ma/va and mb/vb are the OPTIONAL per-cluster SmartPixels angle measurements.
    An OT stub has neither, which has_angle records.

      alpha measures  x0 - x4 / r^2 , NOT x0 alone. From the same helix model the
        local direction gives  half = d0/r + c*kappa*r , so the alpha-implied
        curvature kappa_alpha = half/(c*r) equals kappa + d0/(c*r^2). Treating it
        as a clean measurement of curvature is exactly the d0-into-kappa
        confusion the 5-parameter fit exists to avoid; written properly it
        constrains a DIFFERENT combination than the positions do, which is where
        its displaced power comes from.
      beta measures x2 directly, since a transverse d0 leaves r-z alone.

    Because the filter is linear and carries no process noise, these updates
    commute with the position update and with each other, so their order is not
    a tuning choice.

    Returns (kappa, phi0, d0, cot, z0, var_kappa, var_d0, var_cot, var_z0,
             nhit, ok).
    """
    nt, nl = H.shape
    ANG = has_angle is not None and ma is not None
    # ---- DIFFUSE PRIOR, NOT A TWO-POINT SEED -----------------------------
    # calcSeeds' exact two-point initialisation is valid only for the
    # 4-parameter fit, where two azimuths determine two r-phi parameters. With
    # d0 free it is WRONG, and silently: it fixes x0 and x1 as though the track
    # passed through both seed clusters with d0 = 0, and reports C04 = C14 = 0,
    # i.e. that curvature is uncorrelated with an impact parameter it in fact
    # absorbed. Measured against noiseless tracks that left kappa off by up to
    # 0.24 and d0 by 4.5 mm. Starting diffuse and letting every hit -- the two
    # seed clusters included -- enter as an update builds those correlations
    # correctly and reduces to exact least squares.
    #
    # The priors are wide enough to be uninformative over each parameter's
    # physical range but not so wide as to cost precision: x0 = -c*kappa is
    # under 0.003 at pT > 2 GeV, phi0 is an angle, z0 sits inside the luminous
    # region. d0 keeps its explicit, tunable prior.
    x0 = np.zeros(nt); x1 = np.zeros(nt); x4 = np.zeros(nt)
    x2 = np.zeros(nt); x3 = np.zeros(nt)
    C00 = np.full(nt, 1.0); C01 = np.zeros(nt); C04 = np.zeros(nt)
    C11 = np.full(nt, 100.0); C14 = np.zeros(nt)
    C44 = np.full(nt, d0_prior_cm ** 2 / 3.0)
    C22 = np.full(nt, 100.0); C23 = np.zeros(nt); C33 = np.full(nt, 1.0e4)
    nseen = np.zeros(nt, np.int32)
    chi2_rphi = np.zeros(nt); chi2_rz = np.zeros(nt)
    chi2_ang = np.zeros(nt); nang = np.zeros(nt, np.int32)
    ok = valid.sum(axis=1) >= 2

    def upd3(h0, h1, h4, resid, v, use, chi2=None):
        """r-phi update against a measurement row (h0, h1, h4).

        chi2, when given, accumulates r^2 / R -- the same increment
        KFbase::adjustChi2 adds. R is scalar here because every measurement is
        one-dimensional, so no matrix inverse is needed.
        """
        nonlocal x0, x1, x4, C00, C01, C04, C11, C14, C44
        S0 = C00 * h0 + C01 * h1 + C04 * h4
        S1 = C01 * h0 + C11 * h1 + C14 * h4
        S4 = C04 * h0 + C14 * h1 + C44 * h4
        R = np.where(use, h0 * S0 + h1 * S1 + h4 * S4 + v, 1.0)
        R = np.where(np.abs(R) > 1e-30, R, 1e-30)
        if chi2 is not None:
            chi2 += np.where(use, resid * resid / R, 0.0)
        K0, K1, K4 = S0 / R, S1 / R, S4 / R
        x0 = np.where(use, x0 + resid * K0, x0)
        x1 = np.where(use, x1 + resid * K1, x1)
        x4 = np.where(use, x4 + resid * K4, x4)
        C00 = np.where(use, C00 - K0 * S0, C00)
        C01 = np.where(use, C01 - K0 * S1, C01)
        C04 = np.where(use, C04 - K0 * S4, C04)
        C11 = np.where(use, C11 - K1 * S1, C11)
        C14 = np.where(use, C14 - K1 * S4, C14)
        C44 = np.where(use, C44 - K4 * S4, C44)

    def upd2(h2, h3, resid, v, use, chi2=None):
        """r-z update against a measurement row (h2, h3)."""
        nonlocal x2, x3, C22, C23, C33
        S2 = C22 * h2 + C23 * h3
        S3 = C23 * h2 + C33 * h3
        R = np.where(use, h2 * S2 + h3 * S3 + v, 1.0)
        R = np.where(np.abs(R) > 1e-30, R, 1e-30)
        if chi2 is not None:
            chi2 += np.where(use, resid * resid / R, 0.0)
        K2, K3 = S2 / R, S3 / R
        x2 = np.where(use, x2 + resid * K2, x2)
        x3 = np.where(use, x3 + resid * K3, x3)
        C22 = np.where(use, C22 - K2 * S2, C22)
        C23 = np.where(use, C23 - K2 * S3, C23)
        C33 = np.where(use, C33 - K3 * S3, C33)

    # ---- positions ---------------------------------------------------------
    for L in range(nl):
        use = valid[:, L] & ok
        if not use.any():
            continue
        r = np.where(use, np.maximum(H[:, L], 1e-3), 1.0)
        g = 1.0 / r
        pred = x0 * r + x1 + x4 * g
        upd3(r, np.ones(nt), g, np.where(use, m0[:, L] - pred, 0.0), v0[:, L],
             use, chi2_rphi)
        upd2(r, np.ones(nt), np.where(use, m1[:, L] - (x2 * r + x3), 0.0),
             v1[:, L], use, chi2_rz)
        nseen = np.where(use, nseen + 1, nseen)
    # ---- SmartPixels angles, on every instrumented layer including the seed --
    if ANG:
        for L in range(nl):
            ua = valid[:, L] & has_angle[:, L] & ok
            if not ua.any():
                continue
            r = np.where(ua, np.maximum(H[:, L], 1e-3), 1.0)
            h4 = -1.0 / (r * r)
            pred = x0 + x4 * h4
            upd3(np.ones(nt), np.zeros(nt), h4,
                 np.where(ua, ma[:, L] - pred, 0.0), va[:, L], ua, chi2_ang)
            upd2(np.ones(nt), np.zeros(nt),
                 np.where(ua, mb[:, L] - x2, 0.0), vb[:, L], ua, chi2_ang)
            nang = np.where(ua, nang + 1, nang)
    kappa = -x0 / M.C_BEND
    return (kappa, x1, x4, x2, x3,
            C00 / (M.C_BEND ** 2), C44, C22, C33, nseen, ok,
            chi2_rphi, chi2_rz, chi2_ang, nang)


# ==========================================================================
# self-test
# ==========================================================================
def _selftest():
    """Pin the filter against two independent references.

    (1) NOISELESS RECOVERY. Hits generated exactly on the model must come back
        with the generating parameters, which catches every sign and algebra
        error at once.
    (2) AGREEMENT WITH THE EXACT THREE-POINT SOLVE. With three hits, three r-phi
        measurements determine three r-phi parameters, so the KF with a wide d0
        prior must reproduce solve3 to numerical precision. That is the same
        closed form the displaced triplet already uses, so the two independent
        implementations check each other.
    """
    rng = np.random.default_rng(7)
    n = 4000
    r = np.tile(np.array([3.0, 6.8, 10.2, 16.0, 23.0, 35.0]), (n, 1))
    kap = rng.uniform(-0.5, 0.5, n)
    phi0 = rng.uniform(-0.3, 0.3, n)
    d0 = rng.uniform(-0.2, 0.2, n)
    cot = rng.uniform(-1.5, 1.5, n)
    z0 = rng.uniform(-12.0, 12.0, n)
    phi = (phi0[:, None] + d0[:, None] / r
           - M.C_BEND * kap[:, None] * r)
    z = z0[:, None] + cot[:, None] * r
    v0 = np.full_like(r, (20e-4 / 10.0) ** 2)
    v1 = np.full_like(r, 0.02 ** 2)
    valid = np.ones(r.shape, bool)
    out = kf_run(r, phi, z, v0, v1, valid, d0_prior_cm=5.0)
    kf_kap, kf_phi0, kf_d0, kf_cot, kf_z0 = out[:5]
    assert len(out) == 15
    bad = [f"{nm}: max |err| {np.abs(a - b).max():.3e}"
           for nm, a, b in (("kappa", kf_kap, kap), ("phi0", kf_phi0, phi0),
                            ("d0", kf_d0, d0), ("cot", kf_cot, cot),
                            ("z0", kf_z0, z0))
           if np.abs(a - b).max() > 1e-6]
    print("(1) noiseless recovery, 6 layers: " + ("FAIL " + "; ".join(bad)
                                                  if bad else "exact"))
    # (2) three hits against solve3
    r3, phi3, z3 = r[:, :3], phi[:, :3], z[:, :3]
    out3 = kf_run(r3, phi3, z3, v0[:, :3], v1[:, :3],
                  np.ones(r3.shape, bool), d0_prior_cm=1e3)
    s_phi0, s_d0, s_kap, s_ok = M.solve3(r3[:, 0], phi3[:, 0], r3[:, 1],
                                         phi3[:, 1], r3[:, 2], phi3[:, 2])
    m = s_ok
    dk = np.abs(out3[0][m] - s_kap[m]).max()
    dd = np.abs(out3[2][m] - s_d0[m]).max()
    dp = np.abs(out3[1][m] - s_phi0[m]).max()
    print(f"(2) 3 hits vs solve3 ({int(m.sum())} solvable): "
          f"max |dkappa| {dk:.3e}, |dd0| {dd:.3e} cm, |dphi0| {dp:.3e} rad -> "
          + ("agree" if max(dk, dd, dp) < 1e-6 else "DISAGREE"))
    # (3) angles: alpha on a displaced track must not bias kappa
    half = d0[:, None] / r + M.C_BEND * kap[:, None] * r
    kap_a = half / (M.C_BEND * r)
    ma = -M.C_BEND * kap_a
    va = np.full_like(r, (M.C_BEND * 0.03) ** 2)
    mb = np.tile(cot[:, None], (1, r.shape[1]))
    vb = np.full_like(r, 0.02 ** 2)
    oa = kf_run(r, phi, z, v0, v1, valid, ma, va, mb, vb,
                np.ones(r.shape, bool), d0_prior_cm=5.0)
    e = max(np.abs(oa[0] - kap).max(), np.abs(oa[2] - d0).max())
    print(f"(3) with noiseless alpha/beta, max |err| kappa,d0: {e:.3e} -> "
          + ("exact" if e < 1e-6 else "BIASED"))


_MAIN = None


# ==========================================================================
# assembling a track: the seed's three clusters plus one projection per layer
# ==========================================================================
LAYER_ORDER = (1, 2, 3, 4, 11, 12, 13, 14, 15, 16)


def collect_track_hits(U, Q, trip, layers=LAYER_ORDER, nsig=4.0):
    """For each seed triple, gather at most one hit per layer.

    The seed's own three clusters are taken as given. Every OTHER instrumented
    layer is then PROJECTED to along the seed's own three-point helix, and the
    best hit in that layer is attached -- best meaning smallest combined
    (dphi, dz) pull, so a layer contributes nothing rather than a wrong hit when
    nothing is compatible. This is the track-following the OT finder does after
    its seed, and it is what turns a 3-point seed into a fittable track.

    Returns the per-layer arrays kf_run wants, shaped (ntrack, nlayer).
    """
    ga, gb, gc = trip
    nt = len(ga)
    nl = len(layers)
    R = np.zeros((nt, nl)); PH = np.zeros((nt, nl)); Z = np.zeros((nt, nl))
    SX = np.zeros((nt, nl)); SY = np.zeros((nt, nl))
    KA = np.zeros((nt, nl)); SKA = np.zeros((nt, nl))
    CT = np.zeros((nt, nl)); SCT = np.zeros((nt, nl))
    VALID = np.zeros((nt, nl), bool)
    ANG = np.zeros((nt, nl), bool)
    GIDX = np.full((nt, nl), -1, np.int64)
    lpos = {L: i for i, L in enumerate(layers)}

    def place(g):
        lay = U["layer"][g]
        for L in np.unique(lay):
            if L not in lpos:
                continue
            j = lpos[int(L)]
            m = lay == L
            rowi = np.flatnonzero(m)
            GIDX[rowi, j] = g[m]
            VALID[rowi, j] = True
    for g in (ga, gb, gc):
        place(g)
    # seed helix from the exact three-point solve, which is what the seed has
    phi0, d0, kap, ok3 = M.solve3(U["globalR"][ga], U["globalPhi"][ga],
                                  U["globalR"][gb], U["globalPhi"][gb],
                                  U["globalR"][gc], U["globalPhi"][gc])
    dr = U["globalR"][gb] - U["globalR"][ga]
    cot = (U["globalZ"][gb] - U["globalZ"][ga]) / np.where(np.abs(dr) > 0.5, dr, 1e9)
    z0 = U["globalZ"][ga] - U["globalR"][ga] * cot
    ev = U["event"][ga]
    for L in layers:
        j = lpos[L]
        need = ~VALID[:, j]
        if not need.any():
            continue
        sel = np.flatnonzero(U["layer"] == L)
        if len(sel) < 1:
            continue
        # candidates in the same event, nearest in predicted z then arbitrated
        # on the combined pull
        key_c = U["event"][sel].astype(np.int64)
        o = np.lexsort((U["globalZ"][sel], key_c))
        sel, key_c = sel[o], key_c[o]
        zc, rc, pc = U["globalZ"][sel], U["globalR"][sel], U["globalPhi"][sel]
        sg = np.maximum(U["sigY"][sel], 1e-6)
        pt_typ = 1.0 / max(float(np.median(np.abs(kap))), 1e-3)
        if L > 10:
            w0, _w1 = ot_variances(np.full(len(sel), L - 10), rc, pt_typ)
        else:
            w0, _w1 = it_variances(rc, U["sigX"][sel], U["sigY"][sel], pt_typ)
        sphi = np.sqrt(np.maximum(w0, 1e-12))
        rows = np.flatnonzero(need)
        rmed = float(np.median(rc))
        zpred = z0[rows] + rmed * cot[rows]
        ppred = M.wrap(phi0[rows] + d0[rows] / rmed - M.C_BEND * kap[rows] * rmed)
        lo = np.searchsorted(key_c, ev[rows])
        hi = np.searchsorted(key_c, ev[rows], side="right")
        best = np.full(len(rows), -1, np.int64)
        bp = np.full(len(rows), np.inf)
        # per-track scan over its own event's clusters in this layer; the
        # occupancy per (event, layer) is small enough that this stays cheap
        for t in range(len(rows)):
            a, b = lo[t], hi[t]
            if b <= a:
                continue
            dz = (zc[a:b] - (z0[rows[t]] + rc[a:b] * cot[rows[t]])) / sg[a:b]
            dp = M.wrap(pc[a:b] - M.wrap(phi0[rows[t]] + d0[rows[t]] / rc[a:b]
                                         - M.C_BEND * kap[rows[t]] * rc[a:b]))
            # The azimuth tolerance is the layer's OWN uncertainty, not a flat
            # number. A hardcoded 5 mrad accepted ~8 of 10 layers and degraded
            # sigma(d0) from 37 to 53 um by attaching wrong hits -- a projection
            # that accepts everything is not a projection.
            pull = np.hypot(dz, dp / sphi[a:b])
            k = int(np.argmin(pull))
            if pull[k] < nsig * np.sqrt(2.0):
                best[t] = sel[a + k]
                bp[t] = pull[k]
        got = best >= 0
        GIDX[rows[got], j] = best[got]
        VALID[rows[got], j] = True
    return gather_hits(U, Q, GIDX, layers)


def gather_hits(U, Q, GIDX, layers):
    """Per-hit quantities for a (ntrack, nlayer) table of global cluster indices.

    -1 means the track has nothing on that layer. Shared by the standalone
    projection path and by the seed follow stage, which has already chosen its
    hits and must not choose different ones here.
    """
    nt, nl = GIDX.shape
    z = lambda: np.zeros((nt, nl))
    R, PH, Z, SX, SY = z(), z(), z(), z(), z()
    KA, SKA, CT, SCT = z(), z(), z(), z()
    VALID = GIDX >= 0
    ANG = np.zeros((nt, nl), bool)
    for j in range(nl):
        m = VALID[:, j]
        if not m.any():
            continue
        gg = GIDX[m, j]
        R[m, j] = U["globalR"][gg]
        PH[m, j] = U["globalPhi"][gg]
        Z[m, j] = U["globalZ"][gg]
        SY[m, j] = np.maximum(U["sigY"][gg], 1e-6)
        SX[m, j] = np.maximum(U.get("sigX", U["sigY"])[gg], 1e-6)
        isit = U["layer"][gg] <= 4
        ANG[np.flatnonzero(m)[isit], j] = True
        KA[m, j] = Q["kap_a"][gg]
        SKA[m, j] = Q["s_kap"][gg]
        CT[m, j] = U["globalClusterCotTheta"][gg]
        SCT[m, j] = U["sigGlobalClusterCotTheta"][gg]
    return dict(R=R, PH=PH, Z=Z, SX=SX, SY=SY, KA=KA, SKA=SKA, CT=CT, SCT=SCT,
                VALID=VALID, ANG=ANG, GIDX=GIDX, layers=np.array(layers))


def hits_from_seed(U, out, layers=None):
    """(ntrack, nlayer) cluster-index table from a seed_arity.run_seed result.

    The follow stage already chose one hit per layer under the projection
    windows; re-projecting here would both duplicate the work and risk choosing
    differently, so the fit is done on exactly the track the seeding built.
    """
    layers = LAYER_ORDER if layers is None else layers
    lpos = {L: i for i, L in enumerate(layers)}
    gA, gB, gC = out["_gA"], out["_gB"], out.get("_gC")
    GIDX = np.full((len(gA), len(layers)), -1, np.int64)
    for g in ([gA, gB] + ([gC] if gC is not None else [])):
        lay = U["layer"][g]
        for L in np.unique(lay):
            if int(L) in lpos:
                m = lay == L
                GIDX[np.flatnonzero(m), lpos[int(L)]] = g[m]
    for L, (rows, gc) in out.get("_hits", {}).items():
        if L in lpos and len(rows):
            GIDX[rows, lpos[L]] = gc
    return GIDX


    return dict(R=R, PH=PH, Z=Z, SX=SX, SY=SY, KA=KA, SKA=SKA, CT=CT, SCT=SCT,
                VALID=VALID, ANG=ANG, GIDX=GIDX, layers=np.array(layers))


def fit_tracks(U, Q, trip, use_angles=True, alpha_scale=1.0, beta_scale=1.0,
               d0_prior_cm=D0_PRIOR_CM, layers=LAYER_ORDER, gidx=None):
    """Assemble each seed's track and fit it. Returns fitted parameters + hits.

    gidx, when given, is the hit table the seeding already built; otherwise the
    track is assembled here by projecting the seed helix.
    """
    T = (gather_hits(U, Q, gidx, layers) if gidx is not None
         else collect_track_hits(U, Q, trip, layers))
    lay = np.tile(np.asarray(layers), (T["R"].shape[0], 1))
    is_ot = lay > 10
    # A pT estimate is needed before the fit because the scattering term scales
    # as 1/pT. KFParamsComb takes it "from input track candidate as more
    # stable" (matrixV, first line); the seed pair's two-point curvature is the
    # equivalent here.
    ga, gb = trip[0], trip[1]
    dr = U["globalR"][gb] - U["globalR"][ga]
    safe = np.where(np.abs(dr) > 0.5, dr, 1e9)
    kap0 = np.abs(M.wrap(U["globalPhi"][ga] - U["globalPhi"][gb])
                  / (M.C_BEND * safe))[:, None]
    pt0 = np.broadcast_to(1.0 / np.maximum(kap0, 1e-3), T["R"].shape)
    o0, o1 = ot_variances(np.where(is_ot, lay - 10, 1), T["R"], pt0)
    i0, i1 = it_variances(T["R"], T["SX"], T["SY"], pt0)
    v0 = np.where(is_ot, o0, i0)
    v1 = np.where(is_ot, o1, i1)
    ma = va = mb = vb = ang = None
    if use_angles:
        # alpha and beta enter as measurements of x0 - x4/r^2 and of x2; the
        # scales are the retuned weights, applied to the VARIANCE.
        ma = -M.C_BEND * T["KA"]
        va = (M.C_BEND * np.maximum(T["SKA"], 1e-9) * alpha_scale) ** 2
        mb = T["CT"]
        vb = (np.maximum(T["SCT"], 1e-9) * beta_scale) ** 2
        ang = T["ANG"]
    # azimuth relative to each track's own first hit, so no 2pi wrap enters
    ref = np.where(T["VALID"], T["PH"], 0.0)
    first = np.argmax(T["VALID"], axis=1)
    phi_ref = ref[np.arange(len(first)), first]
    m0 = M.wrap(T["PH"] - phi_ref[:, None])
    out = kf_run(T["R"], m0, T["Z"], v0, v1, T["VALID"], ma, va, mb, vb, ang,
                 d0_prior_cm)
    (kappa, phi0, d0, cot, z0, vk, vd, vc, vz, nhit, ok,
     c2p, c2z, c2a, nang) = out
    fit = {"kappa": kappa, "phi0": M.wrap(phi0 + phi_ref), "d0": d0,
           "cot": cot, "z0": z0, "var_kappa": vk, "var_d0": vd,
           "var_cot": vc, "var_z0": vz, "nhit": nhit, "ok": ok, "hits": T,
           "chi2_rphi": c2p, "chi2_rz": c2z}
    if use_angles:
        # angles were consumed by the fit, so their residuals are no longer an
        # independent test; report what the updates contributed
        fit["chi2_angle"], fit["n_angle"] = c2a, nang
    else:
        fit["chi2_angle"], fit["n_angle"] = angle_chi2(T, fit, alpha_scale,
                                                       beta_scale)
    return fit


def angle_chi2(T, fit, alpha_scale=1.0, beta_scale=1.0):
    """SmartPixels angle consistency for a track fitted on POSITIONS ONLY.

    THIS IS THE ANGLES' JOB. Feeding alpha and beta into the fit buys nothing --
    measured: sigma(kappa) 0.0093 with them against 0.0093 without, at any
    weighting, because sigma(kappa_alpha) is 0.64 at IL1 where the positions
    already deliver 0.019. What they CAN do is say whether a cluster's own
    direction agrees with the trajectory the positions found, which is a test of
    whether the cluster belongs on the track at all. That is precisely the role
    the OT gives stub bend: bendchi2 is a feature of its quality MVA, not a
    measurement in its KF.

    alpha predicts x0 - x4 / r^2 and beta predicts x2, the same rows the fit
    would have used. The pull uses the MEASUREMENT variance only and neglects
    the fitted state's own uncertainty, which makes the chi2 conservative -- the
    same simplification bend chi2 makes, and harmless for a discriminant that
    only has to be monotonic.
    """
    r = np.maximum(T["R"], 1e-3)
    x0 = -M.C_BEND * fit["kappa"][:, None]
    pred_a = x0 - fit["d0"][:, None] / (r * r)
    res_a = (-M.C_BEND * T["KA"]) - pred_a
    var_a = (M.C_BEND * np.maximum(T["SKA"], 1e-9) * alpha_scale) ** 2
    res_b = T["CT"] - fit["cot"][:, None]
    var_b = (np.maximum(T["SCT"], 1e-9) * beta_scale) ** 2
    use = T["VALID"] & T["ANG"]
    c2 = (np.where(use, res_a * res_a / var_a, 0.0).sum(axis=1)
          + np.where(use, res_b * res_b / var_b, 0.0).sum(axis=1))
    return c2, use.sum(axis=1).astype(np.int32)


# ---- acceptance, mirroring KFParamsComb::isGoodState ---------------------
# All cuts are indexed by the number of layers on the track, from
# L1Trigger/TrackFindingTMTT/python/TMTrackProducer_Defaults_cfi.py. They LOOSEN
# with layer count because chi2 grows with the degrees of freedom.
KF_CHISQ_CUT5 = (999., 999., 10., 30., 80., 120., 160.)     # KFLayerVsChiSq5
KF_Z0_CUT5 = (999., 999., 25.5, 25.5, 25.5, 25.5, 25.5)     # KFLayerVsZ0Cut5, cm
KF_D0_CUT5 = (999., 999., 999., 10., 10., 10., 10.)         # KFLayerVsD0Cut5, cm
KF_PT_TOLER = (999., 999., 0.1, 0.1, 0.05, 0.05, 0.05)      # KFLayerVsPtToler
CHI2_RPHI_SCALE = 8.0                                        # KalmanChi2RphiScale


def good_state(fit, ptmin):
    """The real acceptance: scaled chi2, |z0|, |d0| and pT, all per layer count.

    chi2scaled = chi2rphi / 8 + chi2rz is the quantity cut on, not chi2 itself:
    the r-phi component is scaled down to keep electrons, which radiate.
    """
    n = np.clip(fit["nhit"], 0, len(KF_CHISQ_CUT5) - 1)
    chi2s = fit["chi2_rphi"] / CHI2_RPHI_SCALE + fit["chi2_rz"]
    pt = 1.0 / np.maximum(np.abs(fit["kappa"]), 1e-6)
    keep = fit["ok"] & (chi2s <= np.asarray(KF_CHISQ_CUT5)[n])
    keep &= np.abs(fit["z0"]) <= np.asarray(KF_Z0_CUT5)[n]
    keep &= np.abs(fit["d0"]) <= np.asarray(KF_D0_CUT5)[n]
    keep &= pt >= ptmin - np.asarray(KF_PT_TOLER)[n]
    return keep, chi2s


def rank_score(fit, chi2s, w_angle=0.0):
    """Rank candidates the way the real accumulator does, optionally using angles.

    KalmanFilter.cc's accumulator sorts on the number of CONSISTENT layers
    first, then consistent PS layers, keeping one state per track id. Layers
    dominate; chi2 breaks ties. w_angle folds in the angle chi2 per angle-bearing
    cluster, which is the knob for testing whether the SmartPixels directions
    add ranking power the positions do not already have.
    """
    per = chi2s / np.maximum(fit["nhit"], 1)
    score = fit["nhit"].astype(np.float64) - 0.02 * per
    if w_angle:
        score -= w_angle * fit["chi2_angle"] / np.maximum(fit["n_angle"], 1)
    return score



def truth_residuals(U, trip, fit, TP):
    """(dkappa, dd0, dcot, dz0, mask) against the TrackingParticle truth."""
    ga, gb, gc = trip
    ta = U["tpIdx"][ga]
    real = (ta >= 0) & (ta == U["tpIdx"][gb]) & (ta == U["tpIdx"][gc]) & fit["ok"]
    kk = M.tp_key(U["event"][ga], np.maximum(ta, 0))
    p = np.clip(np.searchsorted(TP["key"], kk), 0, max(len(TP["key"]) - 1, 0))
    real &= (len(TP["key"]) > 0) & (TP["key"][p] == kk)
    if not real.any():
        z = np.zeros(0)
        return z, z, z, z, real
    q = p[real]
    # |kappa|, NOT kappa: the nano carries no TrackingParticle charge, so truth
    # is 1/pT and unsigned. Comparing a signed fit against it gives a residual of
    # -2/pT for every negative track, which reads as sigma(kappa) = 0.45 on a
    # sample whose kappa only spans +-0.5 -- i.e. as a total loss of curvature
    # resolution, when the fit is in fact fine.
    return (np.abs(fit["kappa"][real]) - TP["kappa"][q],
            fit["d0"][real] - TP["d0"][q],
            fit["cot"][real] - TP["cot"][q],
            fit["z0"][real] - TP["z0"][q], real)


def tp_truth_table(U):
    """Per-TrackingParticle truth helix, keyed and sorted for searchsorted.

    Built from IT cluster rows only: an OT stub row carries no tpEta/tpPhi/tpV*,
    so reading truth off the inner hit gives nonsense for every OT-only seed.
    """
    m = (U["layer"] <= 4) & (U["tpIdx"] >= 0)
    k = M.tp_key(U["event"][m], U["tpIdx"][m])
    o = np.argsort(k, kind="stable")
    k = k[o]
    f = np.r_[True, k[1:] != k[:-1]] if len(k) else np.zeros(0, bool)
    sel = np.flatnonzero(m)[o][f]
    ph = U["tpPhi"][sel]
    return {"key": k[f],
            "kappa": 1.0 / np.maximum(U["tpPt"][sel], 1e-6),
            "d0": -U["tpVx"][sel] * np.sin(ph) + U["tpVy"][sel] * np.cos(ph),
            "cot": np.sinh(U["tpEta"][sel]), "z0": U["tpVz"][sel]}


def rs(x):
    if len(x) < 30:
        return float("nan")
    q = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q[1] - q[0]))


def retune(U, Q, trip, TP, scales=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0), **kw):
    """Scan the alpha and beta variance scales and report what each buys.

    THE REFIT'S OWN WEIGHTS ARE NOT TRANSFERABLE. It weights alpha and beta by
    the sigmas its network reports, tuned for per-cluster angle recovery on a
    different population; here they enter a 5-parameter helix fit alongside
    positions from two detector systems. Scanning the scale applied to those
    variances is what says whether they are over- or under-trusted, and the
    minimum of the resolution curve is the retuned value. A scale above 1 means
    the reported sigmas are too small.
    """
    rows = []
    base = fit_tracks(U, Q, trip, use_angles=False, **kw)
    dk, dd, dc, dz, _ = truth_residuals(U, trip, base, TP)
    rows.append({"alpha_scale": None, "beta_scale": None, "n": int(len(dk)),
                 "sig_kappa": rs(dk), "sig_d0_um": 1e4 * rs(dd),
                 "sig_cot": rs(dc), "sig_z0_um": 1e4 * rs(dz)})
    for s in scales:
        f = fit_tracks(U, Q, trip, True, s, s, **kw)
        dk, dd, dc, dz, _ = truth_residuals(U, trip, f, TP)
        rows.append({"alpha_scale": s, "beta_scale": s, "n": int(len(dk)),
                     "sig_kappa": rs(dk), "sig_d0_um": 1e4 * rs(dd),
                     "sig_cot": rs(dc), "sig_z0_um": 1e4 * rs(dz)})
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=20)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--seed", default="IL1+IL2>IL3")
    ap.add_argument("--max-tracks", type=int, default=3000)
    ap.add_argument("--d0-prior-cm", type=float, default=D0_PRIOR_CM)
    a = ap.parse_args()
    import tp_findability as TF
    cal = calibrate_for(a.input, a.ptmin)
    la, rest = a.seed.split("+"); lb, lc = rest.split(">")
    LA, LB, LC = TF.CODE[la], TF.CODE[lb], TF.CODE[lc]
    Us, Qs, trips = [], [], []
    for (U, Q), ev in TF.unified_chunks(a.input, a.nev, 4, cal):
        o = M.it_pair_seed(U, Q, np.arange(len(U["layer"])), LA, LB, LC,
                           a.ptmin, True, False, 0.0)
        if "_trip" not in o:
            continue
        Us.append(U); Qs.append(Q); trips.append(o["_trip"])
    if not trips:
        raise SystemExit("no triples")
    print(f"{a.seed}: {sum(len(t[0]) for t in trips):,} candidate triples "
          f"over {a.nev} events")
    allrows = []
    for U, Q, trip in zip(Us, Qs, trips):
        if len(trip[0]) > a.max_tracks:
            k = np.random.default_rng(0).choice(len(trip[0]), a.max_tracks, False)
            trip = tuple(x[k] for x in trip)
        TP = tp_truth_table(U)
        allrows.append(retune(U, Q, trip, TP, d0_prior_cm=a.d0_prior_cm))
    keys = ["n", "sig_kappa", "sig_d0_um", "sig_cot", "sig_z0_um"]
    print(f"\n{'alpha/beta scale':>17}{'n':>8}{'sig(kappa)':>12}"
          f"{'sig(d0) um':>12}{'sig(cot)':>11}{'sig(z0) um':>12}")
    for i in range(len(allrows[0])):
        lab = allrows[0][i]["alpha_scale"]
        n = sum(r[i]["n"] for r in allrows)
        vals = [np.median([r[i][k] for r in allrows if np.isfinite(r[i][k])])
                for k in keys[1:]]
        print(f"{'positions only' if lab is None else f'x{lab:g}':>17}{n:>8,}"
              f"{vals[0]:>12.4f}{vals[1]:>12.0f}{vals[2]:>11.4f}{vals[3]:>12.0f}")


def calibrate_for(spec, ptmin, nev=50):
    import tp_findability as TF
    c = TF.calibrate(spec, nev, ptmin)
    return {int(k): float(v) for k, v in c["sigz_ot"].items()}


if __name__ == "__main__":
    import sys as _s
    if "--selftest" in _s.argv:
        _selftest()
    else:
        main()
