"""Exact port of the prompt hybrid Kalman filter as CMSSW runs it for L1TTrack.

WHAT IS PORTED, line for line (CMSSW_20_1_0_pre1):
  L1Trigger/TrackFindingTracklet/src/HybridFit.cc       the tmtt::Stub list, the seed state, the eta
                                                          region, the output conversion and floor
  L1Trigger/TrackFindingTMTT/src/KFbase.cc              doKF (branching, skips, per-layer cap, final
                                                          state), kalmanUpdate, residual (HO terms),
                                                          adjustState, adjustChi2, kalmanLayer
  L1Trigger/TrackFindingTMTT/src/KFParamsComb.cc        seedX, seedC, vectorM, matrixV, matrixH,
                                                          trackParams, isGoodState
  L1Trigger/TrackFindingTMTT/src/KalmanState.cc         nStubLayers, chi2scaled, hit pattern
  L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h  float storage, consistentSector
  L1Trigger/TrackFindingTracklet/src/Tracklet.cc, Track.{h,cc}   integer clamps, bin centres, phi0

FLOAT32. CMSSW casts to float at fixed points, and the output is floored to the tracklet LSBs, so a
double-only port would differ in the last bins. Reproduced here at the same points:
  tmtt::Stub phi / r / z / sigmaPerp / sigmaPar (Stub.h:120-122, 202-204, 252-254);
  the seed (std::pair<float, float>, HybridFit.cc:147-148; L1track3D float accessors);
  inside residual(): inv2R, tanL, z0, d0, deltaS and the products stub->r() * inv2R, deltaS * tanL;
  inside matrixV(): a, b, r2 (float products), approxB, scaleTilted;
  the fitted parameters and chi2 as L1fittedTrack / KFTrackletTrack store them (float).
Everything else is double, as in CMSSW.

SCOPE. Barrel and disk stubs, every KF eta region (the layer map and the ambiguous "maybe" layers of
KFbase.h; the endcap branches of matrixV and residual, with the 2S disk alpha that HybridFit passes as
-L1TStub::alpha(pitch)). The closure judges KF regions 0-3; the disk path serves WP-F and the canon_v4
disc-column check. The seeds here are the barrel ones (exacttracklet); disk and overlap seeding
(exacttrackletdisk / exacttrackletOverlap) are not ported.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np

f32 = np.float32

# ---- constants -------------------------------------------------------------
# B: the float32 value of the field at the origin that L1FPGATrackProducer reads
# (MagneticField::inTesla returns a float vector; beginRun). Pinned on canon_v4 production tracks:
# TTTrack pT = 2.99792458e-3 * B / |rinv| reproduces the stored pT exactly on 71 % of tracks with
# this value, 33-35 % with either float neighbour (the rest is the float perp() of the momentum).
B_FIELD = float(f32(3.81120228767395))
C_TRK = 0.299792458                       # trklet::Settings c_
C_TMTT = 1.0e8 * 299.792458               # tmtt::Settings cSpeed(): CLHEP c_light [mm/ns] -> cm/s
N_SECTOR = 9
PT_MIN = 2.0                              # trklet::Settings ptmin_
RCRIT = 55.0                              # trklet::Settings rcrit_
RMAXDISK = 120.0
ZLENGTH = 120.0
NBITS_RINV, NBITS_T, NBITS_Z0, NBITS_D0 = 14, 15, 11, 13
MAXT, MAXD0 = 32.0, 10.0
NPHIBITS_L6 = 17                          # nphibitsstub(N_LAYER - 1)

# tmtt::Settings default constructor (the hybrid's), Settings.cc:13-75
ETA_REGIONS = tuple(float(f32(v)) for v in (-2.4, -2.08, -1.68, -1.26, -0.90, -0.62, -0.41, -0.20, 0.0,
                                             0.20, 0.41, 0.62, 0.90, 1.26, 1.68, 2.08, 2.4))
CHOSEN_R_OF_Z = 50.0
CHOSEN_R_OF_PHI = 55.0
HOUGH_MIN_PT = 2.0
B_APPROX_GRADIENT, B_APPROX_INTERCEPT = 0.886454, 0.504148
KF_MIN_STUBS, KF_MAX_STUBS = 4, 6
KF_MAX_SKIP_HARD, KF_MAX_SKIP_EASY, KF_MAX_STUBS_EASY = 1, 2, 10
KF_MAX_STUBS_PER_LAYER = 4
KF_MULT_SCATT = 0.00075
KF_CHI2_RPHI_SCALE = 8
KF_PT_TOLER = (999., 999., 0.1, 0.1, 0.05, 0.05, 0.05)
KF_D0_CUT5 = (999., 999., 999., 10., 10., 10., 10.)
KF_Z0_CUT5 = (999., 999., 25.5, 25.5, 25.5, 25.5, 25.5)
KF_CHI2_CUT5 = (999., 999., 10., 30., 80., 120., 160.)
N_KF_LAYER = 7
INVALID_KF_LAYER = N_KF_LAYER
# KFbase.h layerMap_[kfEtaReg][layerIdReduced] = (barrel KF layer, endcap KF layer); 7 = invalid
_LAYER_MAP = (
    ((7, 7), (0, 7), (1, 7), (5, 7), (4, 7), (3, 7), (7, 7), (2, 7)),
    ((7, 7), (0, 7), (1, 7), (5, 7), (4, 7), (3, 7), (7, 7), (2, 7)),
    ((7, 7), (0, 7), (1, 7), (5, 7), (4, 7), (3, 7), (7, 7), (2, 7)),
    ((7, 7), (0, 7), (1, 7), (5, 7), (4, 7), (3, 7), (7, 7), (2, 7)),
    ((7, 7), (0, 7), (1, 7), (5, 4), (4, 5), (3, 6), (7, 7), (2, 7)),
    ((7, 7), (0, 7), (1, 7), (7, 3), (7, 4), (2, 5), (7, 6), (2, 6)),
    ((7, 7), (0, 7), (1, 7), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6)),
    ((7, 7), (0, 7), (7, 7), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5)),
)
# KFbase::kalmanAmbiguousLayer (kfUseMaybeLayers true): ambiguityMap[kfEtaReg][kfLayer]
_AMBIGUOUS = {(5, 2), (6, 0), (6, 1), (7, 0)}


def layer_id_reduced(layer_id):
    """TrackerModule::calcLayerIdReduced: barrel 1-6, disk 11-15 (+z) / 21-25 (-z) -> 1..7."""
    lay = layer_id if layer_id < 20 else layer_id - 10
    lay = {6: 11, 5: 12, 4: 13, 3: 15}.get(lay, lay)
    if lay > 10:
        lay -= 8
    if lay < 1 or lay > 7:
        raise ValueError(f"reduced layer id out of range for layerId {layer_id}")
    return lay

# trklet::Settings strip pitch / length (Settings.h:1017-1021), as HybridFit hands them to tmtt::Stub
PITCH = {True: 0.01, False: 0.009}
LENGTH = {True: 0.1467, False: 5.025}
INV_ROOT12 = f32(math.sqrt(1.0 / 12.0))   # tmtt::Stub::invRoot12 (const float)

INV2R, PHI0, T, Z0, D0 = range(5)         # KFbase.h PAR_IDS
PHI, Z = 0, 1


def _cround(x):
    """std::round: half away from zero."""
    return math.copysign(math.floor(abs(x) + 0.5), x)


def reduce_phi(x):
    """reco::reducePhiRange<double> (DataFormats/Math/interface/deltaPhi.h)."""
    if abs(x) <= math.pi:
        return x
    n = _cround(x * (1.0 / (2.0 * math.pi)))
    return x - n * (2.0 * math.pi)


def delta_phi(a, b):
    """reco::deltaPhi with a double argument: double arithmetic."""
    return reduce_phi(a - b)


def delta_phi_f32(a, b):
    """reco::deltaPhi(float, float): float arithmetic throughout."""
    x = f32(f32(a) - f32(b))
    if abs(x) <= f32(math.pi):
        return x
    n = f32(_cround(float(f32(x * f32(1.0 / (2.0 * math.pi))))))
    return f32(x - f32(n * f32(2.0 * math.pi)))


@dataclass(frozen=True)
class Constants:
    """The B-dependent tracklet quantities (Settings.h), computed once per field value."""
    b: float = B_FIELD

    @property
    def rinvmax(self):
        return 0.01 * C_TRK * self.b / PT_MIN

    @property
    def dphisector(self):
        return 2 * math.pi / N_SECTOR

    @property
    def dphisector_hg(self):
        return 2 * math.pi / N_SECTOR + self.rinvmax * max(RCRIT - 21.8, 112.7 - RCRIT)

    @property
    def krinvpars(self):
        shift = math.ceil(-math.log2(0.02 * RMAXDISK / ((1 << NBITS_RINV) * self.dphisector_hg)))
        return self.dphisector_hg / RMAXDISK / (1 << shift)

    @property
    def kphi0pars(self):
        return 2 * self.dphisector_hg / (1 << NPHIBITS_L6)

    @property
    def ktpars(self):
        return MAXT / (1 << NBITS_T)

    @property
    def kz0pars(self):
        return 0.5 * ZLENGTH / (1 << NBITS_Z0)

    @property
    def kd0pars(self):
        return 2 * MAXD0 / (1 << NBITS_D0)

    @property
    def b_tmtt(self):
        """tmtt::Settings::magneticField() is a float (setMagneticField(float))."""
        return float(f32(self.b))

    @property
    def inv_pt_to_inv_r(self):
        return self.b_tmtt * C_TMTT / 1.0e13

    @property
    def inv_pt_to_dphi(self):
        return self.b_tmtt * C_TMTT / 2.0e13


K = Constants()


# ---- stubs -------------------------------------------------------------------
@dataclass
class KFStub:
    """tmtt::Stub as HybridFit builds it from an L1TStub."""
    idx: int                 # caller's stub id (e.g. the L1TOTStub row)
    phi: np.float32
    r: np.float32
    z: np.float32
    layer: int               # tmtt layerId: barrel 1..6; disk |d| + 10 (z > 0) or + 20 (z < 0)
    ps: bool
    tilted: bool
    barrel: bool = True
    alpha: np.float32 = f32(0.0)   # -L1TStub::alpha(pitch); 0 except on 2S disk stubs

    @property
    def reduced(self):
        return layer_id_reduced(self.layer)

    @property
    def sigma_perp(self):
        return f32(INV_ROOT12 * f32(PITCH[self.ps]))

    @property
    def sigma_par(self):
        return f32(INV_ROOT12 * f32(LENGTH[self.ps]))


def l1tstub_alpha(strip, flipped, x, y, z, ps):
    """L1TStub::alpha(stripPitch(ps)) (L1TStub.cc:79-89): 0 on PS; on 2S
    +-((int)strip - 509.5) * pitch * flip / r^2, the sign set by z and the flip flag."""
    if ps:
        return 0.0
    flip = -1 if flipped else 1
    r2 = x * x + y * y
    v = (int(strip) - 509.5) * PITCH[False] * flip / r2
    return v if z > 0.0 else -v


def make_stub(idx, x, y, z, layer, ps, tilted, barrel=True, strip=None, flipped=False):
    """From the float stub position (stubPosTT; the nano x/y/z): L1TStub holds the floats as doubles,
    r = hypot(x, y), phi = atan2(y, x); tmtt::Stub stores phi, r, z as float.

    layer: barrel 1..6, or the disk number 1..5 when barrel is False (HybridFit.cc:79-91 turns it into
    |d| + 10 for z > 0, + 20 for z < 0). strip / flipped: L1TOTStub innerStrip / isFlipped, needed for
    2S disk stubs (alpha); alpha stays 0 when strip is None."""
    xd, yd, zd = float(f32(x)), float(f32(y)), float(f32(z))
    lid = int(layer) if barrel else abs(int(layer)) + (10 if zd > 0 else 20)
    alpha = 0.0 if strip is None else l1tstub_alpha(strip, flipped, xd, yd, zd, bool(ps))
    return KFStub(int(idx), f32(math.atan2(yd, xd)), f32(math.hypot(xd, yd)), f32(zd), lid,
                  bool(ps), bool(tilted), bool(barrel), f32(-alpha))


# ---- the seed (tracklet) -------------------------------------------------------
def sector_phimin(isector, k=K):
    """Sector.cc:47-57 (also Track::phi0's offset)."""
    dphi = 2 * math.pi / N_SECTOR
    dphi_hg = 0.5 * k.dphisector_hg - math.pi / N_SECTOR
    phimin = isector * dphi - dphi_hg
    phimax = phimin + dphi + 2 * dphi_hg
    phimin -= math.pi / N_SECTOR
    phimax -= math.pi / N_SECTOR
    phimin, phimax = reduce_phi(phimin), reduce_phi(phimax)
    if phimin > phimax:
        phimin -= 2 * math.pi
    return phimin


def exact_tracklet(r1, z1, phi1, r2, z2, phi2, isector, k=K):
    """TrackletCalculatorBase::exacttracklet (barrel seeding): (rinv, phi0 rel. to phimin, t, z0)."""
    deltaphi = reduce_phi(phi1 - phi2)
    dist = math.sqrt(r2 * r2 + r1 * r1 - 2 * r1 * r2 * math.cos(deltaphi))
    rinv = 2 * math.sin(deltaphi) / dist
    if r1 > r2:
        rinv = -rinv
    phi1tmp = phi1 - sector_phimin(isector, k)
    phi0 = reduce_phi(phi1tmp + math.asin(0.5 * r1 * rinv))
    rhopsi1 = 2 * math.asin(0.5 * r1 * rinv) / rinv
    rhopsi2 = 2 * math.asin(0.5 * r2 * rinv) / rinv
    t = (z1 - z2) / (rhopsi1 - rhopsi2)
    z0 = z1 - t * rhopsi1
    return rinv, phi0, t, z0


@dataclass
class Seed:
    """The tmtt::L1track3D HybridFit builds (floats), plus the sector and the KF eta region."""
    q_over_pt: np.float32
    phi0: np.float32
    z0: np.float32
    tanl: np.float32
    isector: int
    eta_reg: int
    n_input: int             # number of input stubs (easy / hard)

    @property
    def pt(self):
        """L1track3D::pt(): 1 / (1e-6 + |q/pT|), float."""
        return f32(f32(1.0) / (f32(1.0e-6) + abs(self.q_over_pt)))


def make_seed(rinv, phi0_rel, t, z0, isector, n_input, k=K):
    """HybridFit.cc:132-169: the seed state from the tracklet's (approx = exact) parameters."""
    kfphi0 = reduce_phi(phi0_rel + isector * k.dphisector - 0.5 * k.dphisector_hg)
    q = f32(rinv / (0.01 * C_TRK * k.b))
    kfz0, kft = f32(z0), f32(t)
    kfzref = f32(float(kfz0) + CHOSEN_R_OF_Z * float(kft))
    eta_reg = 0
    for i in range(1, len(ETA_REGIONS) - 1):
        zref_max = f32(CHOSEN_R_OF_Z) / math.tan(2.0 * math.atan(math.exp(-ETA_REGIONS[i])))
        if kfzref > zref_max:
            eta_reg = i
    return Seed(q, f32(kfphi0), kfz0, kft, int(isector), eta_reg, int(n_input))


def kf_eta_region(eta_reg, n_eta=16):
    """KFbase::kalmanLayer: the KF's eta sector (small in the barrel)."""
    return n_eta // 2 - 1 - eta_reg if eta_reg < n_eta // 2 else eta_reg - n_eta // 2


def kalman_layer(seed, stub):
    """KFbase::kalmanLayer (kfUseMaybeLayers true, so no region-6 decrement)."""
    pair = _LAYER_MAP[kf_eta_region(seed.eta_reg)][stub.reduced]
    return pair[0] if stub.barrel else pair[1]


def ambiguous_layer(seed, kf_layer):
    """KFbase::kalmanAmbiguousLayer."""
    return kf_layer < N_KF_LAYER and (kf_eta_region(seed.eta_reg), kf_layer) in _AMBIGUOUS


def sector_phi(isector):
    """KFbase::sectorPhi (numPhiNonants = numPhiSectors = 9, so phiCentreSec0 = 0)."""
    phi_centre_sec0 = float(f32(-math.pi / f32(9) + math.pi / f32(9)))
    return 2.0 * math.pi * float(f32(isector)) / float(f32(9)) + phi_centre_sec0


# ---- the filter -----------------------------------------------------------------
@dataclass
class State:
    x: np.ndarray            # (5,) double
    c: np.ndarray            # (5, 5) double
    klayer: int
    nskip: int
    chi2rphi: float
    chi2rz: float
    stubs: tuple = ()
    hit_pattern: int = 0

    @property
    def n_stub_layers(self):
        return 1 + self.klayer - self.nskip

    @property
    def chi2scaled(self):
        return self.chi2rphi / KF_CHI2_RPHI_SCALE + self.chi2rz


def seed_x(seed):
    x = np.zeros(5)
    x[INV2R] = K.inv_pt_to_inv_r * float(seed.q_over_pt) / 2
    x[PHI0] = delta_phi(float(seed.phi0) - sector_phi(seed.isector), 0.0)
    x[Z0] = float(seed.z0)
    x[T] = float(seed.tanl)
    x[D0] = 0.0
    return x


def seed_c():
    inv2r_sigma = f32(0.0314 * (K.inv_pt_to_inv_r / 2))
    c = np.zeros((5, 5))
    c[INV2R, INV2R] = float(inv2r_sigma) ** 2
    c[PHI0, PHI0] = float(f32(0.0102)) ** 2
    c[Z0, Z0] = float(f32(5.0)) ** 2
    c[T, T] = float(f32(0.5)) ** 2
    c[D0, D0] = float(f32(1.0)) ** 2
    return c


def matrix_h(stub):
    r = float(stub.r)
    h = np.zeros((2, 5))
    h[PHI, INV2R] = -r
    h[PHI, PHI0] = 1.0
    h[PHI, D0] = -1.0 / r
    h[Z, Z0] = 1.0
    h[Z, T] = r
    return h


def approx_b(z, r):
    """KFbase::approxB(float z, float r) -> float."""
    return f32(B_APPROX_GRADIENT * float(abs(f32(z))) / float(f32(r)) + B_APPROX_INTERCEPT)


def matrix_v(stub, state, seed):
    """KFParamsComb::matrixV (kalmanHOtilted, useApproxB, kalmanHOprojZcorr = kalmanHOalpha = 1, not
    kalmanHOfw): barrel tilted / flat, and endcap with vz = b tanL^2 (the 2S beta terms need the HO
    switches at 2, so they vanish here)."""
    a = float(f32(stub.sigma_perp * stub.sigma_perp))
    b = float(f32(stub.sigma_par * stub.sigma_par))
    r2 = float(f32(stub.r * stub.r))
    invr2 = 1.0 / r2
    sigma_scat = KF_MULT_SCATT / float(seed.pt)
    vphi = a * invr2 + sigma_scat * sigma_scat
    if not stub.barrel:
        tanl = state.x[T]
        vz = b * (tanl * tanl)
    elif stub.tilted:
        scale = approx_b(stub.z, stub.r)
        vz = b * float(f32(scale * scale))
    else:
        vz = b
    v = np.zeros((2, 2))
    v[PHI, PHI], v[Z, Z] = vphi, vz
    return v


def residual(stub, x, seed):
    """KFbase::residual with kalmanHOhelixExp (barrel; 5 parameters)."""
    vd = np.array([delta_phi(float(stub.phi), sector_phi(seed.isector)), float(stub.z)])
    delta = vd - matrix_h(stub) @ x
    inv2r = f32(K.inv_pt_to_inv_r * 0.5 * float(seed.q_over_pt))
    tanl = f32(x[T])
    corr = float(f32(stub.r * inv2r))
    c0 = (1.0 / 6.0) * corr ** 3
    delta_s = f32((1.0 / 6.0) * float(stub.r) * corr ** 2)
    c1 = -float(f32(delta_s * tanl))
    d0 = f32(x[D0])
    c0 += (1.0 / 6.0) * float(f32(d0 / stub.r)) ** 3
    if not stub.barrel and not stub.ps:
        # 2S endcap: inside -> outside tracking, rShift in float (KFbase.cc:590-630)
        z0 = f32(x[Z0])
        r_shift = f32(f32(f32(stub.z - z0) / tanl) - stub.r)
        r_shift = f32(r_shift - delta_s)
        c0 += float(f32(inv2r * r_shift))                 # kalmanHOprojZcorr == 1
        c0 += float(f32(stub.alpha * r_shift))            # kalmanHOalpha == 1
    delta = delta + np.array([c0, c1])
    delta[0] = delta_phi(delta[0], 0.0)
    return delta


def kalman_update(state, stub, layer, nskip, seed):
    """KFbase::kalmanUpdate (F = 1, no process noise)."""
    x, c = state.x, state.c
    delta = residual(stub, x, seed)
    h = matrix_h(stub)
    v = matrix_v(stub, state, seed)
    r = v + h @ c @ h.T
    det = r[0, 0] * r[1, 1] - r[0, 1] * r[1, 0]
    rinv = np.linalg.inv(r) if det > 0 else 9.9e9 * np.eye(2)
    kgain = c @ h.T @ rinv
    xn = x + kgain @ delta
    cn = (np.eye(5) - kgain @ h) @ c
    d_rphi = delta[PHI] * delta[PHI] * rinv[PHI, PHI] + 2 * delta[PHI] * delta[Z] * rinv[PHI, Z]
    d_rz = delta[Z] * delta[Z] * rinv[Z, Z]
    return State(xn, cn, layer, nskip, state.chi2rphi + d_rphi, state.chi2rz + d_rz,
                 state.stubs + (stub,), state.hit_pattern | (1 << layer))


def track_params(state, seed):
    x = state.x
    return (2.0 * x[INV2R] / K.inv_pt_to_inv_r, delta_phi(x[PHI0] + sector_phi(seed.isector), 0.0),
            x[Z0], x[T], x[D0])


def is_good_state(state, seed):
    n = state.n_stub_layers
    q, _, z0, _, _ = track_params(state, seed)
    pt = abs(1.0 / q)
    good = True
    if abs(z0) > KF_Z0_CUT5[n]:
        good = False
    if pt < HOUGH_MIN_PT - KF_PT_TOLER[n]:
        good = False
    if abs(state.x[D0]) > KF_D0_CUT5[n]:
        good = False
    if state.chi2scaled > KF_CHI2_CUT5[n]:
        good = False
    return good


def initial_state(seed):
    return State(seed_x(seed), seed_c(), -1, 0, 0.0, 0.0)


def do_kf(seed, stubs):
    """KFbase::doKF. stubs: the input list (tmtt::Stub order = HybridFit's trackstublist order).
    Returns the finished State or None."""
    order = sorted(range(len(stubs)), key=lambda i: stubs[i].layer)   # sort by layerId (stable here)
    layer_stubs = {}
    for i in order:
        s = stubs[i]
        kl = kalman_layer(seed, s)
        if kl == INVALID_KF_LAYER:
            continue
        lst = layer_stubs.setdefault(kl, [])
        if len(lst) < KF_MAX_STUBS_PER_LAYER:
            lst.append(s)
        else:
            lst[-1] = s                                       # "FW keeps the last stub"
    best_by_n = {}
    prev = [initial_state(seed)]
    max_iter = max(6, KF_MAX_STUBS)
    easy = seed.n_input < KF_MAX_STUBS_EASY
    max_skip = KF_MAX_SKIP_EASY if easy else KF_MAX_SKIP_HARD
    for iteration in range(max_iter):
        new_states = []
        for st in prev:
            layer = st.klayer + 1
            nskip = st.nskip
            n_amb = 0
            while ambiguous_layer(seed, layer) and not layer_stubs.get(layer):
                layer += 1
                n_amb += 1
            this_lay = layer_stubs.get(layer, [])
            next_lay, n_amb_next = [], 0
            if nskip < max_skip:                       # tested before the ambiguous skips are added
                if ambiguous_layer(seed, layer + 1) and not layer_stubs.get(layer + 1):
                    next_lay, n_amb_next = layer_stubs.get(layer + 2, []), 1
                else:
                    next_lay = layer_stubs.get(layer + 1, [])
            nskip += n_amb
            nxt, nxt_skip = [], []
            for s in this_lay:
                ns = kalman_update(st, s, layer, nskip, seed)
                if is_good_state(ns, seed):
                    nxt.append(ns)
            for s in next_lay:
                ns = kalman_update(st, s, layer + 1 + n_amb_next, nskip + 1 + n_amb_next, seed)
                if is_good_state(ns, seed):
                    nxt_skip.append(ns)
            nxt.sort(key=lambda q: q.chi2scaled)
            nxt_skip.sort(key=lambda q: q.chi2scaled)
            new_states += nxt + nxt_skip
        new_states.sort(key=lambda q: q.chi2scaled * (q.nskip + 1))
        nstubs = iteration + 1
        if nstubs >= KF_MIN_STUBS and new_states:
            best_by_n[nstubs] = new_states[0]
        if nstubs == KF_MAX_STUBS:
            break
        prev = new_states
    return best_by_n[max(best_by_n)] if best_by_n else None


def forced_fit(seed, stubs, max_skip=KF_MAX_SKIP_EASY):
    """The sequential filter over exactly these stubs (one per KF layer) in KF-layer order, with doKF's
    layer and skip bookkeeping: what doKF's chosen state computed if these were its stubs.

    Returns (State or None, list of per-step gate results, reason). None with a reason when doKF could
    not have produced this sequence (a gap it cannot step over, a stub in an invalid KF layer)."""
    by_layer = {}
    for s in stubs:
        kl = kalman_layer(seed, s)
        if kl == INVALID_KF_LAYER:
            return None, [], "stub in an invalid KF layer"
        if kl in by_layer:
            return None, [], "two stubs in one KF layer"
        by_layer[kl] = s
    st = initial_state(seed)
    gates = []
    for kl in sorted(by_layer):
        layer, nskip = st.klayer + 1, st.nskip
        nskip_before = nskip
        while ambiguous_layer(seed, layer) and layer not in by_layer:
            layer += 1
            nskip += 1
        if kl == layer:
            st = kalman_update(st, by_layer[kl], kl, nskip, seed)
        else:
            if nskip_before >= max_skip:
                return None, gates, "skip budget"
            nxt, extra = layer + 1, 0
            if ambiguous_layer(seed, nxt) and nxt not in by_layer:
                nxt, extra = nxt + 1, 1
            if kl != nxt:
                return None, gates, "gap doKF cannot step over"
            st = kalman_update(st, by_layer[kl], kl, nskip + 1 + extra, seed)
        gates.append(is_good_state(st, seed))
    return st, gates, ""


# ---- output: L1fittedTrack (floats), consistentSector, HybridFit floor, Track ---------------
@dataclass
class FitOutput:
    accepted: bool
    q_over_pt: np.float32 = f32(0)
    phi0: np.float32 = f32(0)
    z0: np.float32 = f32(0)
    tanl: np.float32 = f32(0)
    d0: np.float32 = f32(0)
    chi2rphi: np.float32 = f32(0)
    chi2rz: np.float32 = f32(0)
    consistent_sector: bool = False
    ipars: tuple = ()                 # (irinv, iphi0, id0, it, iz0) after Tracklet's clamps
    track: tuple = ()                 # TTTrack (rinv, phi0, tanL, z0, d0)
    stubs: tuple = ()
    hit_pattern: int = 0
    nskip: int = 0


def fitted_output(state, seed, k=K):
    if state is None:
        return FitOutput(False)
    q, phi0, z0, tanl, d0 = track_params(state, seed)
    q, phi0, z0, tanl, d0 = f32(q), f32(phi0), f32(z0), f32(tanl), f32(d0)
    chi2rphi, chi2rz = f32(state.chi2rphi), f32(state.chi2rz)
    # consistentSector (hybrid), L1fittedTrack.h:349-355, phiAtChosenR(false) :274-283
    phi_centre = f32(2.0 * math.pi * seed.isector / 9)
    half = f32(math.pi / 9)
    phi_at = f32(delta_phi(float(phi0) - ((k.inv_pt_to_dphi * CHOSEN_R_OF_PHI) * float(q))
                           - float(d0) / CHOSEN_R_OF_PHI, 0.0))
    inside = bool(abs(delta_phi_f32(phi_at, phi_centre)) < half)
    out = FitOutput(True, q, phi0, z0, tanl, d0, chi2rphi, chi2rz, inside,
                    stubs=state.stubs, hit_pattern=state.hit_pattern, nskip=state.nskip)
    if not inside:
        return out
    # HybridFit.cc:198-221
    phi0fit = reduce_phi(float(phi0) - seed.isector * 2 * math.pi / N_SECTOR + 0.5 * k.dphisector_hg)
    rinvfit = 0.01 * C_TRK * k.b * float(q)
    irinv = math.floor(rinvfit / k.krinvpars)
    iphi0 = math.floor(phi0fit / k.kphi0pars)
    it = math.floor(float(tanl) / k.ktpars)
    iz0 = math.floor(float(z0) / k.kz0pars)
    id0 = math.floor(float(d0) / k.kd0pars)
    # Tracklet::setFitPars clamps (Tracklet.cc:560-576)
    irinv = min(irinv, 1 << 14)
    if irinv <= -(1 << 14):
        irinv = -(1 << 14) + 1
    half_z0 = 1 << (NBITS_Z0 - 1)
    if iz0 >= half_z0:
        iz0 = half_z0 - 1
    if iz0 <= -half_z0:
        iz0 = 1 - half_z0
    out.ipars = (irinv, iphi0, id0, it, iz0)
    # Track.h:70-73, Track.cc:38-53 -> TTTrack (floats in the nano)
    out.track = ((irinv + 0.5) * k.krinvpars,
                 reduce_phi(iphi0 * k.kphi0pars + sector_phimin(seed.isector, k)),
                 (it + 0.5) * k.ktpars, (iz0 + 0.5) * k.kz0pars, (id0 + 0.5) * k.kd0pars)
    return out
