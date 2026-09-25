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

    phi(r) = phi0 - d0 / r - c * kappa * r        (r-phi, 3 parameters)
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


# Tilt correction for the z variance of a TILTED PS module, from
# KFParamsComb::matrixV (TrackFindingTMTT/src/KFParamsComb.cc:105-117):
#     scaleTilted = approxB(z, r);  vz = b * scaleTilted^2
# with approxB = bApprox_gradient*|z|/r + bApprox_intercept for a tilted barrel
# module and 1 for a flat one (TrackFindingTMTT/src/Stub.cc:352-358). The
# constants are TMTT's fitted defaults (src/Settings.cc:22-23).
B_APPROX_GRADIENT = 0.886454
B_APPROX_INTERCEPT = 0.504148


def approx_b(z, r):
    """The tilted-module bend/resolution factor, floored at the flat value.

    WHICH MODULES ARE TILTED IS A MODULE PROPERTY WE DO NOT CARRY. CMSSW gates
    this on Stub::tiltedBarrel(); our ntuple has only (r, z). Taking
    max(1, approxB) puts the crossover at |z|/r = 0.559, where the formula
    itself passes through the flat-module value of 1 -- so the central barrel
    keeps vz unscaled and the scaling turns on where the tilted region begins.
    That is an approximation to a discrete geometric fact, not the fact itself.

    MEASURED, sigma(z_stub - z_track) on genuine PS stubs of real tracks with
    pT > 3, ratio of the highest |z|/r bin to the central one against the
    predicted max(1, approxB)^2 = 3.98:
        OT L1  1.66 measured   OT L2  2.07   OT L3  4.71
    so the form is right and L3 matches, while L1 and L2 are over-predicted by
    roughly 2x. The measurement is an upper bound -- it contains the track's own
    extrapolation error, which also grows with |z|/r -- so it cannot settle the
    discrepancy on its own. We follow CMSSW rather than the measurement here,
    because the point is to emulate the KF that exists, and record the gap.
    """
    return np.maximum(1.0, B_APPROX_GRADIENT * np.abs(z) / np.maximum(r, 1e-3)
                      + B_APPROX_INTERCEPT)


def ot_variances(layer, r, pt, z=None):
    """(vphi, vz) for OT barrel stubs, following KFParamsComb::matrixV."""
    ps = np.isin(layer, PS_LAYERS)
    sigma_perp = np.where(ps, MPA_PITCH, CBC_PITCH) * INV_ROOT12
    sigma_par = np.where(ps, MPA_LENGTH, CBC_LENGTH) * INV_ROOT12
    scat = MULT_SCATT_TERM / np.maximum(np.abs(pt), 1e-3)
    vz = np.broadcast_to(sigma_par ** 2, np.shape(r)).copy()
    if z is not None:
        # PS ONLY. The 2S branch of matrixV has no tilt term at all, and the
        # comment there says the non-radial strip effect is neglected for PS.
        vz = np.where(ps, vz * approx_b(z, r) ** 2, vz)
    return ((sigma_perp / np.maximum(r, 1e-3)) ** 2 + scat ** 2, vz)


def it_variances(r, sigX, sigY, pt):
    """(vphi, vz) for SmartPixels clusters, by the same form.

    sigX and sigY are the nano's measured per-cluster CPE sigmas and are already
    the Gaussian sigmas that sigmaPerp and sigmaPar are, so they substitute
    directly. Only the scattering constant is borrowed from the OT.
    """
    scat = MULT_SCATT_TERM / np.maximum(np.abs(pt), 1e-3)
    return ((np.maximum(sigX, 1e-6) / np.maximum(r, 1e-3)) ** 2 + scat ** 2,
            np.maximum(sigY, 1e-6) ** 2)


# ---- process noise ------------------------------------------------------
# Q = 0 in both CMSSW and this file by default (KFbase.cc:462, "Get scattering
# contribution to helix parameter covariance (currently zero)"). Scattering is
# instead folded into the MEASUREMENT variance, which treats it as INDEPENDENT
# per hit. A real scatter is CORRELATED: it deflects every downstream layer
# coherently. That approximation is harmless within one system, whose hits span
# a short radial range, and is not harmless across r = 3 -> 108 cm.
#
# WHERE A KINK LIVES. With phi(r) = phi0 - d0/r - c*kappa*r (TTTrack d0) the
# basis functions are 1, 1/r and r. A kink of angle delta at radius r_s adds
#     delta * (r - r_s)/r  =  delta * 1  -  delta * r_s * (1/r)
# which is EXACTLY a phi0 shift of +delta together with a d0 shift of
# +delta*r_s (the 1/r coefficient is -d0), with NO kappa component to first order. So a 1 mrad scatter at
# r_s = 20 cm is indistinguishable from 200 um of impact parameter. With Q = 0
# the OT hits pin phi0 and kappa -- 1/r is nearly flat at large radius, so they
# carry almost no d0 information -- and the IT azimuths must then be explained
# by d0 alone. d0 is the sink for inter-system scattering by construction, which
# is why a matched refit measured WORSE d0 (40 um) than the standalone IT
# mini-track (29 um).
#
# Q therefore enters in the (phi0, d0) and (cot, z0) planes with the correlation
# a kink implies, and nothing in kappa:
#     Q[phi0,phi0] = s^2 , Q[d0,d0] = s^2 r_s^2 , Q[phi0,d0] = +s^2 r_s
# MS_SCALE is the per-step scattering angle at 1 GeV, scaling as 1/pT after the
# PDG small-angle form. There is no material map here, so it is a PARAMETER to
# be scanned, not a derived number; IT_OT_SCALE multiplies it for the single
# step that crosses between the systems, where the support and services sit.
MS_SCALE = 0.0             # rad*GeV per step; 0 reproduces CMSSW exactly
IT_OT_SCALE = 1.0          # extra factor on the IT -> OT crossing


# higher-order helix terms, mirroring the hybrid's own flags
# d0 IS CMSSW's TTTrack d0 EVERYWHERE HERE: d0 = x0 sin(phi) - y0 cos(phi), the
# POCA being (d0 sin phi, -d0 cos phi) -- which is MINUS reco::TrackBase::dxy().
# In that convention a hit at radius r sits at azimuth phi(r) = phi0 - d0/r - c
# kappa r. (Until 2026-09-23 this file computed in the dxy convention and flipped
# the sign once at output via D0_SIGN = -1; the rewrite is the same arithmetic
# with the sign carried inside, bit for bit -- every d0 product flips in pairs --
# so no seeding, fitting, cache or exported number changed. It was homogenised
# because a formula derived here in the dxy convention was once carried into the
# CMSSW producer, which works in TTTrack d0, and got the wrong sign there.)
# Measured check of the convention: this file's truth d0 against L1TTrack's
# tp_d0 for the same particles, median difference 0.000 um, robust sigma 0.1 um.

HO_HELIX = True      # kalmanHOhelixExp_ = true  (Settings.cc:60)
HO_D0 = True         # the (d0/r)^3 term, 5-parameter only (KFbase.cc:610)
D0_PRIOR_CM = 1.0        # uniform half-range for the d0 prior; see kf_run


def kf_run(H, m0, m1, v0, v1, valid, ma=None, va=None, mb=None, vb=None,
           has_angle=None, d0_prior_cm=D0_PRIOR_CM, ms_scale=MS_SCALE,
           it_ot_scale=IT_OT_SCALE, pt_for_ms=None, layers=None,
           reverse=False, ho=None):
    """Vectorised FIVE-parameter KF over ntrack x nlayer arrays, inner to outer.

    THE MODEL, and it is the same one the exact three-point solve uses:

        phi(r) = phi0 - d0 / r - c * kappa * r        (r-phi, 3 parameters)
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

      alpha measures  x0 + x4 / r^2 , NOT x0 alone. From the same helix model the
        local direction gives  half = -d0/r + c*kappa*r , so the alpha-implied
        curvature kappa_alpha = half/(c*r) equals kappa - d0/(c*r^2). Treating it
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
    # r_prev tracks the radius of each track's last used layer, which is where a
    # kink between that layer and this one effectively sits.
    # DIRECTION MATTERS ONLY ONCE Q IS NONZERO. With Q = 0 the forward filter's
    # terminal state IS the exact global least-squares solution, so order is
    # irrelevant -- which is why everything worked before. With Q > 0 the
    # terminal state is the estimate AT THE LAST LAYER PROCESSED. Going inner to
    # outer that is the OT, so Q inflates the IT-determined d0 covariance right
    # before hits arrive that carry no d0 information. Reversed, the terminal
    # state lives at the innermost layer, where d0 is actually measured.
    # The proper remedy is a smoother (Frühwirth NIM A262 (1987) 444, sec. 4);
    # this is the cheap test of whether direction is the whole story.
    r_prev = np.zeros(nt)
    seen_any = np.zeros(nt, bool)
    lay_arr = None if layers is None else np.asarray(layers)
    seq = range(nl - 1, -1, -1) if reverse else range(nl)
    for L in seq:
        use = valid[:, L] & ok
        if not use.any():
            continue
        r = np.where(use, np.maximum(H[:, L], 1e-3), 1.0)
        if ms_scale and pt_for_ms is not None:
            step = use & seen_any
            if step.any():
                sc = ms_scale
                if lay_arr is not None:
                    # the crossing step joins an IT radius to an OT one, in
                    # whichever direction the filter is running
                    here_ot = lay_arr[L] > 10
                    cross = np.where(here_ot, (r_prev < 20.0) & (r_prev > 0),
                                     r_prev > 20.0)
                    sc = np.where(cross, ms_scale * it_ot_scale, ms_scale)
                sd = np.where(step, sc / np.maximum(np.abs(pt_for_ms), 1e-3), 0.0)
                q = sd * sd
                rs_ = np.where(step, r_prev, 0.0)
                C11 = C11 + q                      # phi0
                C44 = C44 + q * rs_ * rs_          # d0
                C14 = C14 + q * rs_                # correlation a kink implies (TTTrack d0: +r_s)
                C22 = C22 + q                      # cot
                C33 = C33 + q * rs_ * rs_          # z0
                C23 = C23 - q * rs_
        g = -1.0 / r                                # d(phi)/d(d0) in TTTrack's d0 convention
        # THIRD-ORDER CIRCLE EXPANSION, which the real KF applies and we used
        # not to. KFbase::residual adds (1/6)(r*inv2R)^3 in r-phi, (1/6)(d0/r)^3
        # for the 5-parameter fit, and -(1/6) r (r*inv2R)^2 * tanL in r-z
        # (KFbase.cc:603,610,606), active because kalmanHOhelixExp_ is true and
        # kalmanHOfw_ false in the hybrid Settings DEFAULT constructor
        # (Settings.cc:60,63) -- NOT the values in TMTrackProducer_Defaults_cfi,
        # which HybridFit never reads (HybridFit.cc:55-58 default-constructs).
        #
        # Leaving them out cost 1151 um of d0 at pT = 2 falling as 1/pT^3. The
        # r-phi term is ODD in charge, so it broadens symmetrically with no mean
        # offset; the r-z term is EVEN, so it biases z0. Both are properties of
        # the helix, not of the outer tracker, so they apply to every fit here.
        # x0 = -c*kappa, so r*inv2R == -x0*r up to the sign convention.
        use_ho = HO_HELIX if ho is None else ho
        ho_rphi = ((1.0 / 6.0) * np.power(x0 * r, 3) if use_ho else 0.0)
        if use_ho and HO_D0:
            ho_rphi = ho_rphi + (1.0 / 6.0) * np.power(x4 * g, 3)
        pred = x0 * r + x1 + x4 * g + ho_rphi
        upd3(r, np.ones(nt), g, np.where(use, m0[:, L] - pred, 0.0), v0[:, L],
             use, chi2_rphi)
        # deltaS = (1/6) r (r*inv2R)^2 ; the path-length correction enters r-z
        # through the dip angle
        # SIGN, from the code rather than from reasoning: KFbase.cc:632 applies
        # `delta += correction` where delta is the RESIDUAL, so
        # `correction[1] -= deltaS*tanL` (line 606) means the PREDICTION gains
        # +deltaS*tanL. Physically right too: the helix arc length exceeds r, so
        # z grows faster than cot*r. Getting this backwards made sigma(z0) 37%
        # worse while d0 improved, which is how it was caught.
        ho_rz = ((1.0 / 6.0) * r * np.power(x0 * r, 2) * x2 if use_ho else 0.0)
        upd2(r, np.ones(nt), np.where(use, m1[:, L] - (x2 * r + x3 + ho_rz), 0.0),
             v1[:, L], use, chi2_rz)
        nseen = np.where(use, nseen + 1, nseen)
        r_prev = np.where(use, r, r_prev)
        seen_any = seen_any | use
    # ---- SmartPixels angles, on every instrumented layer including the seed --
    if ANG:
        for L in range(nl):
            ua = valid[:, L] & has_angle[:, L] & ok
            if not ua.any():
                continue
            r = np.where(ua, np.maximum(H[:, L], 1e-3), 1.0)
            h4 = 1.0 / (r * r)                     # alpha term: x0 + d0 / r^2 (TTTrack d0)
            pred = x0 + x4 * h4
            upd3(np.ones(nt), np.zeros(nt), h4,
                 np.where(ua, ma[:, L] - pred, 0.0), va[:, L], ua, chi2_ang)
            upd2(np.ones(nt), np.zeros(nt),
                 np.where(ua, mb[:, L] - x2, 0.0), vb[:, L], ua, chi2_ang)
            nang = np.where(ua, nang + 1, nang)
    kappa = -x0 / M.C_BEND
    return (kappa, x1, x4, x2, x3,
            C00 / (M.C_BEND ** 2), C44, C22, C33, nseen, ok,
            chi2_rphi, chi2_rz, chi2_ang, nang, C11)


# ==========================================================================
# self-test
# ==========================================================================
def _selftest():
    """Pin the filter against two independent references.

    NOTE ON THE MODEL. Test (1) generates hits from the SAME helix the filter
    assumes, third-order terms included. Test (2) compares against solve3,
    which is a first-order closed form, so it generates first-order hits and
    runs the filter with the higher-order terms disabled -- otherwise the two
    would disagree by construction rather than by error.

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
    # THE GENERATOR MUST USE THE SAME HELIX THE FIT DOES. It used to be purely
    # first order, which is precisely why the missing third-order terms went
    # unnoticed for so long: a first-order generator can never reveal them.
    x0 = -M.C_BEND * kap[:, None]
    phi = phi0[:, None] - d0[:, None] / r + x0 * r
    zz = z0[:, None] + cot[:, None] * r
    if HO_HELIX:
        phi = phi + (1.0 / 6.0) * np.power(x0 * r, 3)
        if HO_D0:
            phi = phi + (1.0 / 6.0) * np.power(-d0[:, None] / r, 3)
        zz = zz + (1.0 / 6.0) * r * np.power(x0 * r, 2) * cot[:, None]
    z = zz
    # first-order hit set, for the two tests whose reference is first order
    phi_lin = phi0[:, None] - d0[:, None] / r + x0 * r
    z_lin = z0[:, None] + cot[:, None] * r
    v0 = np.full_like(r, (20e-4 / 10.0) ** 2)
    v1 = np.full_like(r, 0.02 ** 2)
    valid = np.ones(r.shape, bool)
    # (1a) the LINEAR core must be exact: first-order hits, higher-order off.
    out = kf_run(r, phi_lin, z_lin, v0, v1, valid, d0_prior_cm=5.0, ho=False)
    bad = [f"{nm}: max |err| {np.abs(a - b).max():.3e}"
           for nm, a, b in (("kappa", out[0], kap), ("phi0", out[1], phi0),
                            ("d0", out[2], d0), ("cot", out[3], cot),
                            ("z0", out[4], z0))
           if np.abs(a - b).max() > 1e-6]
    assert len(out) == 16
    print("(1a) linear core, first-order hits, ho=False: "
          + ("FAIL " + "; ".join(bad) if bad else "exact"))
    # (1b) with the third-order terms on BOTH sides the filter is a
    # linearisation, so recovery is close but not exact. The residual is
    # dominated by the (d0/r)^3 term at the innermost radius -- verified: with
    # d0 = 0 it is a flat 1.2e-4 in kappa (the filter's own numerical floor,
    # independent of kappa), and with kappa = 0 it grows as d0^3, reaching
    # 13.6 um at d0 = 4 mm against an analytic 12 um. Bounds below are set from
    # those measurements, not from wishful thinking.
    outh = kf_run(r, phi, z, v0, v1, valid, d0_prior_cm=5.0, ho=True)
    lim = {"kappa": 3e-3, "phi0": 1e-3, "d0": 2e-3, "cot": 1e-3, "z0": 5e-3}
    res = {nm: float(np.abs(a - b).max())
           for nm, a, b in (("kappa", outh[0], kap), ("phi0", outh[1], phi0),
                            ("d0", outh[2], d0), ("cot", outh[3], cot),
                            ("z0", outh[4], z0))}
    over = [f"{k} {v:.2e} > {lim[k]:.0e}" for k, v in res.items() if v > lim[k]]
    print("(1b) third-order on both sides, linearisation residual: "
          + ", ".join(f"{k} {v:.2e}" for k, v in res.items())
          + ("  -> OVER BOUND: " + "; ".join(over) if over else "  -> within bounds"))
    # (2) three hits against solve3
    r3, phi3, z3 = r[:, :3], phi_lin[:, :3], z_lin[:, :3]
    out3 = kf_run(r3, phi3, z3, v0[:, :3], v1[:, :3],
                  np.ones(r3.shape, bool), d0_prior_cm=1e3, ho=False)
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
    half = -d0[:, None] / r + M.C_BEND * kap[:, None] * r
    kap_a = half / (M.C_BEND * r)
    ma = -M.C_BEND * kap_a
    va = np.full_like(r, (M.C_BEND * 0.03) ** 2)
    mb = np.tile(cot[:, None], (1, r.shape[1]))
    vb = np.full_like(r, 0.02 ** 2)
    oa = kf_run(r, phi_lin, z_lin, v0, v1, valid, ma, va, mb, vb,
                np.ones(r.shape, bool), d0_prior_cm=5.0, ho=False)
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
            w0, _w1 = ot_variances(np.full(len(sel), L - 10), rc, pt_typ, z=zc)
        else:
            w0, _w1 = it_variances(rc, U["sigX"][sel], U["sigY"][sel], pt_typ)
        sphi = np.sqrt(np.maximum(w0, 1e-12))
        rows = np.flatnonzero(need)
        rmed = float(np.median(rc))
        zpred = z0[rows] + rmed * cot[rows]
        ppred = M.wrap(phi0[rows] - d0[rows] / rmed - M.C_BEND * kap[rows] * rmed)
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
            dp = M.wrap(pc[a:b] - M.wrap(phi0[rows[t]] - d0[rows[t]] / rc[a:b]
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
               d0_prior_cm=D0_PRIOR_CM, layers=LAYER_ORDER, gidx=None,
               pt_hint=None, ms_scale=MS_SCALE, it_ot_scale=IT_OT_SCALE,
               reverse=False):
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
    # equivalent here. pt_hint overrides it, which the matched IT+OT refit needs:
    # its IT pair spans at most 13 cm and gives a curvature far worse than the
    # OT track it is being merged with.
    if pt_hint is not None:
        pt0 = np.broadcast_to(np.asarray(pt_hint, float)[:, None], T["R"].shape)
    else:
        ga, gb = trip[0], trip[1]
        dr = U["globalR"][gb] - U["globalR"][ga]
        safe = np.where(np.abs(dr) > 0.5, dr, 1e9)
        kap0 = np.abs(M.wrap(U["globalPhi"][ga] - U["globalPhi"][gb])
                      / (M.C_BEND * safe))[:, None]
        pt0 = np.broadcast_to(1.0 / np.maximum(kap0, 1e-3), T["R"].shape)
    o0, o1 = ot_variances(np.where(is_ot, lay - 10, 1), T["R"], pt0, z=T["Z"])
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
                 d0_prior_cm, ms_scale, it_ot_scale,
                 pt_for_ms=pt0[:, 0], layers=layers, reverse=reverse)
    (kappa, phi0, d0, cot, z0, vk, vd, vc, vz, nhit, ok,
     c2p, c2z, c2a, nang, vphi) = out
    fit = {"kappa": kappa, "phi0": M.wrap(phi0 + phi_ref), "d0": d0,
           "cot": cot, "z0": z0, "var_kappa": vk, "var_d0": vd,
           "var_cot": vc, "var_z0": vz, "var_phi0": vphi,
           "nhit": nhit, "ok": ok, "hits": T,
           "chi2_rphi": c2p, "chi2_rz": c2z}
    if use_angles:
        # angles were consumed by the fit, so their residuals are no longer an
        # independent test; report what the updates contributed
        fit["chi2_angle"], fit["n_angle"] = c2a, nang
    else:
        fit["chi2_angle"], fit["n_angle"] = angle_chi2(T, fit, alpha_scale,
                                                       beta_scale)
    return fit


# MEASURED per-layer pull widths of the SmartPixels angles against TRUTH
# (tpGlobalClusterPhi / tpGlobalClusterCotTheta, 60 events, tpPt > 2), and
# independently against the fitted track, which agree: alpha is calibrated to
# within 8% and beta reads about 20% optimistic on every layer. These scale the
# reported sigmas so an angle pull is a pull.
#
# This supersedes an earlier inference that the sigmas were ~6.6x optimistic,
# taken from a MEAN chi2_angle per cluster of 44-50. That mean was arithmetic,
# not calibration: with ~2% of surviving tracks contaminated and carrying angle
# pulls near 40, those few dominate the average. The clean-track pull is 1.0-1.26.
ALPHA_PULL_SCALE = {1: 1.089, 2: 1.045, 3: 1.079, 4: 0.922}
BETA_PULL_SCALE = {1: 1.231, 2: 1.191, 3: 1.124, 4: 1.131}


def _layer_scales(layers):
    a = np.array([ALPHA_PULL_SCALE.get(int(L), 1.0) for L in layers])
    b = np.array([BETA_PULL_SCALE.get(int(L), 1.0) for L in layers])
    return a[None, :], b[None, :]


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

    alpha predicts x0 + x4 / r^2 and beta predicts x2, the same rows the fit
    would have used. The pull uses the MEASUREMENT variance only and neglects
    the fitted state's own uncertainty, which makes the chi2 conservative -- the
    same simplification bend chi2 makes, and harmless for a discriminant that
    only has to be monotonic.
    """
    pa, pb = angle_pulls(T, fit, alpha_scale, beta_scale)
    use = T["VALID"] & T["ANG"]
    c2 = (np.where(use, pa * pa, 0.0).sum(axis=1)
          + np.where(use, pb * pb, 0.0).sum(axis=1))
    return c2, use.sum(axis=1).astype(np.int32)


def angle_pulls(T, fit, alpha_scale=1.0, beta_scale=1.0):
    """Per-(track, layer) alpha and beta pulls against the fitted trajectory.

    Kept separate from the summed chi2 because WHICH layer disagrees is the
    interesting part: four IT layers give eight direction pulls, each localising
    a suspect hit, where one summed bend-chi2 cannot.
    """
    r = np.maximum(T["R"], 1e-3)
    sa, sb = _layer_scales(T["layers"])
    x0 = -M.C_BEND * fit["kappa"][:, None]
    pred_a = x0 + fit["d0"][:, None] / (r * r)
    res_a = (-M.C_BEND * T["KA"]) - pred_a
    sig_a = M.C_BEND * np.maximum(T["SKA"], 1e-9) * sa * alpha_scale
    res_b = T["CT"] - fit["cot"][:, None]
    sig_b = np.maximum(T["SCT"], 1e-9) * sb * beta_scale
    return res_a / sig_a, res_b / sig_b


ANGLE_COLS = ("track_row", "layer", "res_a", "pred_a", "res_b", "pred_b",
              "sig_a", "sig_b", "hit_wrong", "n_wrong")


def angle_rows(U, T, fit, trip, keep_idx, alpha_scale=1.0, beta_scale=1.0):
    """Per-CLUSTER angle residuals for the IT hits of selected tracks.

    THE RESIDUAL, NOT THE PULL. angle_pulls divides by sigma immediately, which
    is what a chi2 wants but hides the two things a sensor study needs: how big
    the disagreement is in angle, and what the track angle AT THAT SENSOR was.
    A cluster's direction is a local measurement, so plotting the residual
    against the local track angle is what shows whether the cluster's angle
    reconstruction is biased where the track is steep.

    `hit_wrong` is per HIT -- this cluster belongs to a TP other than the
    track's majority owner -- which is a sharper split than the track's total
    wrong-hit count, because it separates the contaminating hit from its
    innocent neighbours on the same track.

    One row per (track, IT layer with a cluster). Returns float32, ordered by
    track then layer.
    """
    r = np.maximum(T["R"], 1e-3)
    lay = np.asarray(T["layers"])[None, :]
    use = T["VALID"] & T["ANG"] & (lay <= 4)
    sa, sb = _layer_scales(T["layers"])
    x0 = -M.C_BEND * fit["kappa"][:, None]
    pred_a = x0 + fit["d0"][:, None] / (r * r)
    res_a = (-M.C_BEND * T["KA"]) - pred_a
    sig_a = M.C_BEND * np.maximum(T["SKA"], 1e-9) * sa * alpha_scale
    pred_b = np.broadcast_to(fit["cot"][:, None], T["CT"].shape)
    res_b = T["CT"] - pred_b
    sig_b = np.maximum(T["SCT"], 1e-9) * sb * beta_scale
    # ownership, per hit, against the same majority owner the residuals use
    G = T["GIDX"]
    on = G >= 0
    tp_of = np.where(on, U["tpIdx"][np.clip(G, 0, None)], -1)
    own, _ = majority_owner(tp_of, on)
    wrong = on & (tp_of != own[:, None]) | (own[:, None] < 0)
    n_wrong = (on & ~((tp_of == own[:, None]) & (own[:, None] >= 0))).sum(axis=1)
    ti, li = np.nonzero(use)
    if not len(ti):
        return np.zeros((0, len(ANGLE_COLS)), np.float32)
    cols = {
        "track_row": keep_idx[ti].astype(np.float64),
        "layer": np.asarray(T["layers"])[li].astype(np.float64),
        "res_a": res_a[ti, li], "pred_a": pred_a[ti, li],
        "res_b": res_b[ti, li], "pred_b": pred_b[ti, li],
        "sig_a": sig_a[ti, li], "sig_b": sig_b[ti, li],
        "hit_wrong": wrong[ti, li].astype(np.float64),
        "n_wrong": n_wrong[ti].astype(np.float64),
    }
    return np.stack([cols[c] for c in ANGLE_COLS], axis=1).astype(np.float32)


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
    # |kappa| against the unsigned truth 1/pT: a signed residual is -2/pT for
    # every wrong-charge fit, which would swamp sigma(kappa) with charge
    # misassignment rather than measure curvature resolution.
    return (np.abs(fit["kappa"][real]) - TP["kappa"][q],
            fit["d0"][real] - TP["d0"][q],
            fit["cot"][real] - TP["cot"][q],
            fit["z0"][real] - TP["z0"][q], real)


def majority_owner(tp, valid):
    """Per track: the TP holding the most attached hits, and how many.

    tp < 0 (a cluster belonging to no TrackingParticle) never wins, so a track
    built entirely from unassociated clusters correctly gets no owner rather
    than being called clean.

    WHY THE OWNER AND NOT THE SEED. Truth used to be read off the seed's inner
    cluster. Measured over 200 events, 45.8% of accepted tracks have a majority
    owner that is NOT the seed's TP, and referencing the residual to the seed
    reports sigma(d0) = 1440 um where the owner-referenced value is 372 um in
    the 2-wrong-hit population: the residual was being taken against a particle
    that owns a minority of the hits.
    """
    v = np.where(valid & (tp >= 0), tp, -1)
    S = np.sort(v, axis=1)[:, ::-1]          # real values first, -1 trailing
    cur_v = S[:, 0].copy()
    cur_n = (S[:, 0] >= 0).astype(np.int64)
    best_v, best_n = cur_v.copy(), cur_n.copy()
    for j in range(1, S.shape[1]):
        col = S[:, j]
        real = col >= 0
        same = real & (col == cur_v)
        cur_n = np.where(same, cur_n + 1, np.where(real, 1, 0))
        cur_v = np.where(same, cur_v, col)
        take = cur_n > best_n
        best_n = np.where(take, cur_n, best_n)
        best_v = np.where(take, cur_v, best_v)
    return best_v, best_n


def tp_truth_table(U):
    """Per-TrackingParticle truth helix, keyed and sorted for searchsorted.

    One entry per TP with a hit in U and an L1TTP row (charged, pT >= 1 GeV),
    read from the per-hit L1TTP join (M.attach_tp_truth) and so the same for IT
    and OT hits. phi0, d0 [cm], z0 [cm] and cot = tanL are at the POCA to the
    beamline, the perigee this fit's phi(r) = phi0 - d0/r - c*kappa*r uses.
    kappa is UNSIGNED 1/pT [1/GeV]: every consumer compares it with |fitted
    kappa|, so a fit with the wrong charge is scored on curvature magnitude only.
    """
    m = (U["tpIdx"] >= 0) & np.isfinite(U["tp_pt"])
    k = M.tp_key(U["event"][m], U["tpIdx"][m])
    o = np.argsort(k, kind="stable")
    k = k[o]
    f = np.r_[True, k[1:] != k[:-1]] if len(k) else np.zeros(0, bool)
    sel = np.flatnonzero(m)[o][f]
    return {"key": k[f], "kappa": 1.0 / U["tp_pt"][sel],
            "d0": U["tp_d0"][sel], "phi0": U["tp_phi0"][sel],
            "cot": U["tp_tanL"][sel], "z0": U["tp_z0"][sel]}


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





# ==========================================================================
# per-track features for a track-quality MVA
# ==========================================================================
# ONE TABLE SERVES BOTH the interactive page and the quality MVA. The page needs
# identity and fitted kinematics to bin on; the MVA needs the chi2 and pull
# columns; both need the truth labels. Keeping two exports in step was not going
# to survive, so columns are selected by NAME from a single row set.
#
# Ordering is deliberate: identity, then fitted, then quality, then truth. The
# exporter quantises per column, and the truth block is dropped entirely for any
# payload that must not carry MC information.
TRACK_COLS = (
    # --- identity ---
    "seed_idx", "arity", "sysclass", "event", "tp_key",
    # --- fitted parameters, what the page bins on ---
    "inv_pt", "phi0", "eta", "cot", "z0", "d0",
    # --- quality: the OT TQ MVA's own features first ---
    "nhit", "n_miss_interior", "chi2_rphi_per_layer", "chi2_rz_per_layer",
    "chi2_scaled",
    # --- quality: what the combined scenario adds ---
    "n_it", "n_ot", "n_angle", "chi2_angle_per_cl", "max_angle_pull",
    "second_angle_pull", "max_pos_pull", "chi2_rphi_it", "chi2_rphi_ot",
    "d0_over_sigma", "rank_score",
    # --- truth ---
    "n_wrong", "n_own", "n_it_right", "n_it_wrong", "n_ot_right",
    "n_ot_wrong", "n_ot_combinatoric", "n_ot_unknown",
    "own_is_seed", "own_tpidx", "sig_kf_d0", "is_clean",
    "tp_pt", "tp_eta", "tp_d0", "tp_z0", "tp_phi0",
    # residuals for all FIVE fitted parameters; phi0 had none until now, so a
    # per-parameter quality study could not include it
    "d_kappa", "d_phi0", "d_d0", "d_cot", "d_z0",
)
N_TRACK_COLS = len(TRACK_COLS)
# columns the quality MVA trains on; the rest are identity, fitted kinematics or
# truth and must not be fed to it
MVA_FEATURES = tuple(c for c in TRACK_COLS if c not in (
    "seed_idx", "arity", "sysclass", "event", "tp_key", "phi0", "eta",
    "n_wrong", "n_own", "n_it_right", "n_it_wrong", "n_ot_right",
    "n_ot_wrong", "n_ot_combinatoric", "n_ot_unknown",
    "own_is_seed", "own_tpidx", "is_clean", "tp_pt", "tp_eta",
    "tp_d0",
    "tp_z0", "tp_phi0",
    "d_kappa", "d_phi0", "d_d0", "d_cot", "d_z0"))
SYS_IT, SYS_OT, SYS_MIX = 0, 1, 2


def numerics_key():
    """Every knob that changes the NUMBERS without changing a column or a config.

    THE CACHE KEY NEEDS THIS. It hashes format, inputs, config and TRACK_COLS --
    and none of those move when a KF coefficient or an OT match window changes.
    So a census re-run after the third-order helix terms were added RESUMED the
    pre-correction cache, exited in seconds and looked successful, and the site
    shipped residuals from the uncorrected fit for a day. The two HO caches even
    hashed IDENTICALLY (452a0529f3828a76) and were only kept apart by living in
    different directories.

    Add a knob here rather than remembering to bump FORMAT_VERSION: forgetting
    this dict costs a silently wrong dataset, forgetting a version bump costs
    the same, but this one is checked by the self-test below.
    """
    return {"b_approx": [float(B_APPROX_GRADIENT), float(B_APPROX_INTERCEPT)],
            "ho_helix": bool(HO_HELIX), "ho_d0": bool(HO_D0),
            "ms_scale": float(MS_SCALE), "it_ot_scale": float(IT_OT_SCALE),
            "mult_scatt_term": float(MULT_SCATT_TERM),
            "chi2_rphi_scale": float(CHI2_RPHI_SCALE),
            "kf_pt_toler": [float(x) for x in KF_PT_TOLER],
            "alpha_pull": {str(k): float(v)
                           for k, v in sorted(ALPHA_PULL_SCALE.items())},
            "beta_pull": {str(k): float(v)
                          for k, v in sorted(BETA_PULL_SCALE.items())},
            "ot": M.ot_numerics_key()}


def sysclass_of(layers):
    """0 = IT-only seed, 1 = OT-only, 2 = spans both."""
    it = any(L <= 4 for L in layers)
    ot = any(L > 10 for L in layers)
    return SYS_MIX if (it and ot) else (SYS_IT if it else SYS_OT)


def track_rows(U, Q, T, fit, trip, TP, seed_idx, arity, sysclass, rank,
               alpha_scale=1.0, beta_scale=1.0):
    """One row per track, ordered by TRACK_COLS. float32 throughout.

    tp_key is -1 for a fake, which is what lets the exporter sort the table by
    TP and keep fakes in a contiguous block: the per-menu duplicate removal in
    the browser is then a linear scan over the sorted region, and fake queries
    skip it entirely.
    """
    lay = np.asarray(T["layers"])[None, :]
    use = T["VALID"]
    nhit = use.sum(axis=1).astype(np.float64)
    is_it = use & (lay <= 4)
    is_ot = use & (lay > 10)
    pa, pb = angle_pulls(T, fit, alpha_scale, beta_scale)
    uang = use & T["ANG"]
    apull = np.where(uang, np.maximum(np.abs(pa), np.abs(pb)), 0.0)
    srt = np.sort(apull, axis=1)[:, ::-1]
    r = np.maximum(T["R"], 1e-3)
    dz = T["Z"] - (fit["cot"][:, None] * r + fit["z0"][:, None])
    ppos = np.where(use, np.abs(dz) / np.maximum(T["SY"], 1e-6), 0.0)
    idx = np.arange(use.shape[1])[None, :]
    lo = np.where(use, idx, 99).min(axis=1)
    hi = np.where(use, idx, -1).max(axis=1)
    span = (idx >= lo[:, None]) & (idx <= hi[:, None])
    n_miss = (span & ~use).sum(axis=1).astype(np.float64)
    c2a, nang = angle_chi2(T, fit, alpha_scale, beta_scale)
    chi2s = fit["chi2_rphi"] / CHI2_RPHI_SCALE + fit["chi2_rz"]
    # ---- truth, referenced to the MAJORITY OWNER of the hits -------------
    ga = trip[0]
    G = T["GIDX"]
    on = G >= 0
    tp_of = np.where(on, U["tpIdx"][np.clip(G, 0, None)], -1)
    own, n_own = majority_owner(tp_of, on)
    right = on & (tp_of == own[:, None]) & (own[:, None] >= 0)
    n_wrong = (on & ~right).sum(axis=1).astype(np.float64)
    # A WRONG OT STUB IS NOT ONE THING. flg is 1 genuine / 2 combinatoric /
    # 3 unknown from the CMS per-stub flags (see tp_findability._unify), so the
    # three failure modes separate: a genuine stub owned by another particle is
    # a mis-association, a combinatoric stub is this particle's cluster merged
    # with another's and still carries usable position, and an unknown stub is
    # noise. The four counters below partition n_ot exactly, and their non-right
    # members plus n_it_wrong reproduce n_wrong -- asserted below on real data.
    flg = np.where(on, U["ot_flag"][np.clip(G, 0, None)], 0)
    ot_gen = is_ot & (flg == 1)
    kk = M.tp_key(U["event"][ga], np.maximum(own, 0))
    p = np.clip(np.searchsorted(TP["key"], kk), 0, max(len(TP["key"]) - 1, 0))
    # an owner outside L1TTP (neutral or pT < 1 GeV) has no truth helix and
    # gets no residual
    hit = (len(TP["key"]) > 0) & (TP["key"][p] == kk) & (own >= 0)
    nan = np.full(len(ga), np.nan)
    cols = {
        "seed_idx": np.full(len(ga), float(seed_idx)),
        "arity": np.full(len(ga), float(arity)),
        "sysclass": np.full(len(ga), float(sysclass)),
        "event": U["event"][ga].astype(np.float64),
        "tp_key": np.where(hit, kk, -1.0),
        "inv_pt": np.abs(fit["kappa"]), "phi0": fit["phi0"],
        "eta": np.arcsinh(np.clip(fit["cot"], -30, 30)),
        "cot": fit["cot"], "z0": fit["z0"], "d0": fit["d0"],
        "nhit": nhit, "n_miss_interior": n_miss,
        "chi2_rphi_per_layer": fit["chi2_rphi"] / np.maximum(nhit, 1),
        "chi2_rz_per_layer": fit["chi2_rz"] / np.maximum(nhit, 1),
        "chi2_scaled": chi2s,
        "n_it": is_it.sum(axis=1).astype(np.float64),
        "n_ot": is_ot.sum(axis=1).astype(np.float64),
        "n_angle": nang.astype(np.float64),
        "chi2_angle_per_cl": c2a / np.maximum(nang, 1),
        "max_angle_pull": srt[:, 0],
        "second_angle_pull": srt[:, 1] if srt.shape[1] > 1 else srt[:, 0],
        "max_pos_pull": ppos.max(axis=1),
        "chi2_rphi_it": fit["chi2_rphi"] * is_it.sum(axis=1) / np.maximum(nhit, 1),
        "chi2_rphi_ot": fit["chi2_rphi"] * is_ot.sum(axis=1) / np.maximum(nhit, 1),
        "d0_over_sigma": np.abs(fit["d0"]) / np.sqrt(np.maximum(fit["var_d0"], 1e-12)),
        "rank_score": np.asarray(rank, float),
        "n_wrong": n_wrong,
        "n_own": n_own.astype(np.float64),
        # the same contamination split by system, because d0 is bought by
        # correct IT hits and curvature by correct OT hits: a single count
        # explains 44% of the variance of log sigma(d0) where the four
        # per-system counts explain 80%
        "n_it_right": (is_it & right).sum(axis=1).astype(np.float64),
        "n_it_wrong": (is_it & ~right).sum(axis=1).astype(np.float64),
        "n_ot_right": (ot_gen & right).sum(axis=1).astype(np.float64),
        "n_ot_wrong": (ot_gen & ~right).sum(axis=1).astype(np.float64),
        "n_ot_combinatoric": (is_ot & (flg == 2)).sum(axis=1).astype(np.float64),
        "n_ot_unknown": (is_ot & (flg == 3)).sum(axis=1).astype(np.float64),
        # 0 when the seed cluster belongs to a TP other than the owner, which
        # is what the seed-referenced convention used to hide
        "own_is_seed": (own == U["tpIdx"][ga]).astype(np.float64),
        # THE EXACT OWNER IDENTITY, and the reason it is stored as a bare tpIdx
        # rather than read off tp_key: the packed (event << 20) | tpIdx needs 30
        # bits, these rows are float32, and at event 999 keys 128 apart collapse
        # onto one value. tpIdx alone is < 2**20 and survives exactly, and the
        # event is already a column, so the exporter can rebuild the key in
        # int64 and match each track to its TrackingParticle's row. That link is
        # what the DELIVERED efficiency needs.
        "own_tpidx": own.astype(np.float64),
        # the fit's CLAIMED error, not a resolution: the covariance never sees
        # the residuals, and is measured 1.6-3.4x optimistic on clean tracks
        "sig_kf_d0": np.sqrt(np.maximum(fit["var_d0"], 1e-30)),
        "is_clean": ((own >= 0) & (n_wrong == 0)).astype(np.float64),
        "tp_pt": np.where(hit, 1.0 / np.maximum(TP["kappa"][p], 1e-6), nan),
        "tp_eta": np.where(hit, np.arcsinh(np.clip(TP["cot"][p], -30, 30)), nan),
        "tp_d0": np.where(hit, TP["d0"][p], nan),
        "tp_z0": np.where(hit, TP["z0"][p], nan),
        "tp_phi0": np.where(hit, TP["phi0"][p], nan),
        # wrapped into [-pi, pi): an unwrapped azimuth difference puts a 2pi
        # jump in the middle of the residual distribution
        "d_phi0": np.where(hit, np.arctan2(
            np.sin(fit["phi0"] - TP["phi0"][p]),
            np.cos(fit["phi0"] - TP["phi0"][p])), nan),
        "d_kappa": np.where(hit, np.abs(fit["kappa"]) - TP["kappa"][p], nan),
        "d_d0": np.where(hit, fit["d0"] - TP["d0"][p], nan),
        "d_cot": np.where(hit, fit["cot"] - TP["cot"][p], nan),
        "d_z0": np.where(hit, fit["z0"] - TP["z0"][p], nan),
    }
    # THE OT PARTITION MUST BE EXACT, and this is the cheapest place to know it.
    # Every valid OT hit has to land in exactly one of right / wrong /
    # combinatoric / unknown, and every non-right hit in the whole track has to
    # add up to n_wrong. A silent failure here would mean ot_flag and VALID
    # disagree -- a gather bug -- and would show up only as a slightly wrong
    # contamination axis on the page, which is undetectable by eye.
    bad = np.abs(cols["n_it_wrong"] + cols["n_ot_wrong"]
                 + cols["n_ot_combinatoric"] + cols["n_ot_unknown"] - n_wrong)
    if bad.size and bad.max() > 0:
        raise AssertionError(
            f"OT truth partition broken on {int((bad > 0).sum())} of {bad.size} "
            f"tracks (max discrepancy {bad.max():.0f} hits): ot_flag does not "
            f"cover every valid OT hit")
    n_ot_parts = (cols["n_ot_right"] + cols["n_ot_wrong"]
                  + cols["n_ot_combinatoric"] + cols["n_ot_unknown"])
    bad2 = np.abs(n_ot_parts - cols["n_ot"])
    if bad2.size and bad2.max() > 0:
        raise AssertionError(
            f"OT counters do not sum to n_ot on {int((bad2 > 0).sum())} tracks")
    return np.stack([cols[c] for c in TRACK_COLS], axis=1).astype(np.float32)


if __name__ == "__main__":
    import sys as _s
    if "--selftest" in _s.argv:
        _selftest()
    else:
        main()


# ---- DEFERRED: hardware-style bit encoding -------------------------------
# DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h encodes an L1 track
# in 96 bits: uniform steps for the kinematics (stepD0 = 1/256 cm over 13 bits,
# kZ0Size = 12, stepTanL = 1/4096) and NON-UNIFORM predefined bin tables for the
# chi2 quantities --
#     chi2RPhiBins (4 bits) 0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 10, 15, 20, 35, 60, 200
#     chi2RZBins   (4 bits) 0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8, 10, 20, 50
#     bendChi2Bins (3 bits) 0, 0.75, 1, 1.5, 2.25, 3.5, 5, 20
# -- fine where the discrimination is, coarse out to a saturating top bin.
#
# NOT adopted yet, deliberately: the track word is still being evolved upstream,
# so encoding to it now would bake in a moving target. float32 stays the default
# and the encoding belongs behind an option.
#
# WHAT THAT OPTION WOULD NEED, when it is built: a configurable bit count per
# column, and for any column feeding the MVA an ALGORITHM to choose the edges
# rather than a hand-written table. Quantile edges are the wrong default for a
# learned feature -- they maximise the encoded variable's own entropy, not its
# information about the label. Optimal k-bin discretisation against a binary
# target is exactly solvable by dynamic programming in O(n*k) after sorting,
# maximising mutual information (equivalently minimising the AUC lost to
# quantisation), and that is what should be used for the chi2 and pull columns.
# The physics-motivated tables above are a reasonable prior to seed it with.
