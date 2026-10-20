"""Known-answer tests of the WP-D OT chain emulation (eval_refitq/combinatorics: ot_dtc, ot_chain, cmssw_kf):
the DTC frame decoding, the integer tracklet calculator and its gates, the integer projections, the match
engine's neighbour-bin rule, the improving chain and the track conversion; then, on the
canon_v4 10-event test production when it is present, the decoding against the DTC's own positions and the
chain against CMSSW's no-DR tracks (a regression floor at the registered identity)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_refitq" / "combinatorics"))
import cmssw_kf as KF   # noqa: E402
import ot_chain as OC   # noqa: E402
import ot_dtc as D      # noqa: E402

TEST_PROD = Path("/Volumes/WDMac/smartpixels_scratch/canon_v4_test/spstudy_AAAA_f01_10ev_at35436445850.root")
MODECMP_NODR = Path("/Users/nmangane/smartpixels/cmssw/work/modecmp/HYBRID_np5_noDR_f01_100ev.root")
# the chain's no-DR identity on the test events (identical stub sets / CMSSW no-DR tracks, barrel seeds and
# barrel-only tracks, every KF region), registered 2026-10-15 at 0.959; a drop below this floor is a regression
NODR_IDENTITY_FLOOR = 0.95


# ---- DTC frames ----------------------------------------------------------------------------------
def _frame(mt, layer_code, bend, alpha, phi, z, r, nd):
    """A 64-bit DTC word from its fields (bit 0 valid, then layer code, bend, alpha, phi, z, r, negDisk)."""
    w, s = 1, 1
    for v, n in ((layer_code, 2), (bend, D.W_BEND[mt]), (alpha, D.W_ALPHA[mt]), (phi, D.W_PHI[mt]),
                 (z, D.W_Z[mt]), (r, D.W_R[mt]), (nd, D.W_ND[mt])):
        w |= (v & ((1 << n) - 1)) << s if n else 0
        s += n
    return w >> 32, w & 0xFFFFFFFF


@pytest.mark.parametrize("mt", [D.BARREL_PS, D.BARREL_2S, D.DISK_PS, D.DISK_2S])
def test_dtc_decode_roundtrip(mt):
    rng = np.random.default_rng(mt)
    for _ in range(50):
        lc, bend = int(rng.integers(0, 4)), int(rng.integers(0, 1 << D.W_BEND[mt]))
        alpha = int(rng.integers(-(1 << max(D.W_ALPHA[mt] - 1, 0)), 1 << max(D.W_ALPHA[mt] - 1, 0))) if D.W_ALPHA[mt] else 0
        phi = int(rng.integers(0, 1 << D.W_PHI[mt]))
        z = int(rng.integers(-(1 << (D.W_Z[mt] - 1)), 1 << (D.W_Z[mt] - 1)))
        barrel = mt < D.DISK_PS
        r = int(rng.integers(-(1 << (D.W_R[mt] - 1)), 1 << (D.W_R[mt] - 1))) if barrel else int(rng.integers(0, 1 << D.W_R[mt]))
        nd = 0 if barrel else int(rng.integers(0, 2))
        hi, lo = _frame(mt, lc, bend, alpha, phi, z, r, nd)
        got = [int(v[0]) for v in D.decode([hi], [lo], [mt])]
        assert got == [lc, bend, alpha, phi, z, r, nd]


# ---- the integer tracklet calculator -------------------------------------------------------------
def _helix_copies(rinv, phi0g, z0, t, sector, layers=(1, 2), dir_=(3, -2)):
    """Copies of two barrel stubs on a helix (global phi0g), digitised as the DTC words: r on the layer's
    integer grid, phi relative to the sector's phimin in kphi(layer), z in kz(layer)."""
    rows = []
    for L, ir in zip(layers, dir_):
        r = D.rmean(L) + ir * D.kr_barrel()
        s = math.asin(0.5 * r * rinv)
        phi = phi0g - s
        z = z0 + t * 2 * s / rinv
        phirel = KF.reduce_phi(phi - KF.sector_phimin(sector))
        rows.append((L, ir, int(math.floor(phirel / D.kphi(L))), int(math.floor(z / D.kz(L) + 0.5)), r, phi, z))
    a = lambda i: np.array([q[i] for q in rows])
    n = len(rows)
    return OC.Copies(np.arange(n), np.full(n, sector), np.arange(n), a(0), a(0) <= 3, np.zeros(n, bool),
                     a(2), a(3), a(1), np.zeros(n, np.int64), a(2), a(4), a(4) * np.cos(a(5)), a(4) * np.sin(a(5)), a(6))


HELICES = [(sgn * 0.01 * KF.C_TRK * KF.K.b / pt, z0, t)
           for pt in (2.0, 5.0, 30.0) for sgn in (1, -1) for z0, t in ((3.0, 0.4), (-8.0, -0.9), (0.5, 0.05))]


@pytest.mark.parametrize("rinv,z0,t", HELICES)
def test_calc_pars_reproduces_the_helix(rinv, z0, t):
    sector = 4
    phi0g = KF.sector_phimin(sector) + 0.55 * KF.K.dphisector_hg
    cp = _helix_copies(rinv, phi0g, z0, t, sector)
    irinv, iphi0, iz0, it = OC.calc_pars(cp, 0, 1, 0)
    k = KF.K
    assert abs(irinv * k.krinvpars - rinv) < 4 * k.krinvpars + 0.01 * abs(rinv)
    assert abs(iz0 * k.kz0pars - z0) < 0.15                     # the stubs' own z digitisation
    assert abs(it * k.ktpars - t) < 0.01
    assert abs(KF.reduce_phi(iphi0 * k.kphi0pars + KF.sector_phimin(sector) - phi0g)) < 2e-4


def test_tc_gates_edges():
    k = KF.K
    mid = int((0.5 * k.dphisector_hg) / k.kphi0pars)        # phicrit at the sector centre
    z0lsb = 4096 / 120.0                                      # iz0 per cm (n_z0 12 bits over 120 cm)
    for seed, ok_z, bad_z in ((0, 14.9, 15.1), (2, 22.4, 22.6), (3, 22.4, 22.6)):
        assert OC.tc_gates(0, mid, int(ok_z * z0lsb), seed)
        assert not OC.tc_gates(0, mid, int(bad_z * z0lsb), seed)
    cut = 0.01 * KF.C_TRK * k.b / OC.RINVCUT_PT * (120.0 * (1 << OC.N_RINV)) / k.dphisector_hg
    assert OC.tc_gates(int(cut) - 1, mid, 0, 0) and not OC.tc_gates(int(cut) + 1, mid, 0, 0)
    assert not OC.tc_gates(0, -1, 0, 0)                       # phi0 below the sector
    assert not OC.tc_gates(0, int(k.dphisector_hg / k.kphi0pars), 0, 0)   # phicrit outside the window


@pytest.mark.parametrize("rinv,z0,t", HELICES)
def test_projection_to_the_outer_layers(rinv, z0, t):
    """ProjectionCalculator keeps asin to third order (x + x^3 / 6, x = r rinv / 2): against that series of
    the tracklet's own integer parameters the projection agrees to its digitisation; against the exact
    helix it is off by the fifth-order term (up to ~4e-4 rad at 2 GeV in L6)."""
    sector = 4
    k = KF.K
    phi0g = KF.sector_phimin(sector) + 0.55 * k.dphisector_hg
    ipars = OC.calc_pars(_helix_copies(rinv, phi0g, z0, t, sector), 0, 1, 0)
    rv, ph0, zz0, tt = ipars[0] * k.krinvpars, ipars[1] * k.kphi0pars, ipars[2] * k.kz0pars, ipars[3] * k.ktpars
    for L in (3, 4, 5, 6):
        pj = OC.proj_layer(ipars, L)
        r = D.rmean(L)
        x = 0.5 * r * rv
        z = zz0 + tt * r * (1 + x * x / 6)
        if abs(z) > 110:
            continue
        assert pj is not None
        assert abs(pj[0] * D.kphi(L) - (ph0 - x - x ** 3 / 6)) < 4 * D.kphi(1)
        assert abs(pj[1] * D.kz(L) - z) < 2 * D.kz(L)
        assert abs(KF.reduce_phi(pj[0] * D.kphi(L) + KF.sector_phimin(sector) - (phi0g - math.asin(0.5 * r * rinv)))) < 6e-4
        assert pj[2] == -(ipars[0] >> 4) and pj[3] == ipars[3] >> 4


# ---- the match engine and processor --------------------------------------------------------------
def test_match_engine_neighbour_bin_never_matches():
    """MatchEngineUnit::step moves the projection's fine phi by a VM bin with the opposite sign to the
    geometry: a stub one fine unit across the bin edge is 15 units away as the engine computes it."""
    # projection near the upper edge (shift +1, fine 7); its own bin is use.second 0, the upper neighbour 1
    assert 6 - OC.meu_finephi(7, 0, 1) == -1                    # own bin, fine 6: |dphi| 1, passes
    assert 0 - OC.meu_finephi(7, 1, 1) == -15                   # next bin, fine 0: geometrically +1
    # projection near the lower edge (shift -1, fine 0): the lower neighbour is use.second 0
    assert 7 - OC.meu_finephi(0, 0, -1) == 15                   # lower bin, fine 7: geometrically -1
    assert 1 - OC.meu_finephi(0, 1, -1) == 1                    # own bin
    assert OC.meu_finephi(5, 0, 0) == 5


def _chain(dphi, dz, cut_phi, cut_z):
    """The kept positions of one tracklet's candidates in arrival order (ot_chain.improving)."""
    kept, bp, bz = [], cut_phi, cut_z
    for i, (p, z) in enumerate(zip(dphi, dz)):
        ok, bp, bz = OC.improving(p, z, bp, bz)
        if ok:
            kept.append(i)
    return kept


def test_improving_chain():
    # worse then better: both kept; better then worse: one; the z window is [-best, best)
    assert _chain([5, 2], [3, 1], 10, 10) == [0, 1]
    assert _chain([2, 5], [1, 3], 10, 10) == [0]
    assert _chain([0, 0], [10, -10], 10, 10) == [1]
    assert _chain([11], [0], 10, 10) == []
    assert _chain([3, -3, 2], [4, 3, 3], 10, 10) == [0, 1]   # ties in phi pass; dz 3 < best 3 fails
    assert _chain([5, 1], [0, 0], 10, 10) == [0]             # a dz 0 match closes the z window


def test_track_from_ipars_clamps_and_bin_centres():
    k = KF.K
    ip, tr = KF.track_from_ipars(1 << 20, 100, 0, -3, 5000, 2)
    assert ip == ((1 << 14), 100, 0, -3, (1 << (KF.NBITS_Z0 - 1)) - 1)
    assert tr[0] == pytest.approx(((1 << 14) + 0.5) * k.krinvpars)
    assert tr[2] == pytest.approx(-2.5 * k.ktpars)
    assert tr[1] == pytest.approx(KF.reduce_phi(100 * k.kphi0pars + KF.sector_phimin(2)))
    assert KF.track_from_ipars(-(1 << 20), 0, 0, 0, -5000, 0)[0][::4] == (-(1 << 14) + 1, 1 - (1 << (KF.NBITS_Z0 - 1)))


# ---- data-backed (canon_v4 test production) -------------------------------------------------------
needs_test = pytest.mark.skipif(not TEST_PROD.exists(), reason="canon_v4 test production not present")


@needs_test
def test_decode_matches_the_dtc_positions():
    import awkward as ak
    import uproot

    t = uproot.open(f"{TEST_PROD}:Events")
    A = t.arrays([k for k in t.keys() if k.startswith(("L1TOTStub_", "L1TOTStubDTC_"))], entry_stop=3)
    for e in range(len(A)):
        g = lambda k: ak.to_numpy(A[k][e])
        row = g("L1TOTStubDTC_otStubIdx").astype(np.int64)
        bar = g("L1TOTStub_isBarrel")[row].astype(bool)
        ps = g("L1TOTStub_isPS")[row].astype(bool)
        mt = D.module_type(bar, ps)
        lc, bend, al, iphi, iz, ir, nd = D.decode(g("L1TOTStubDTC_frameHi"), g("L1TOTStubDTC_frameLo"), mt)
        lay = g("L1TOTStub_layer")[row].astype(np.int64)
        q = bar
        phi, z, r = D.barrel_physical(lay[q], iphi[q], iz[q], ir[q], g("L1TOTStubDTC_region")[q].astype(np.int64))
        dphi = (phi - g("L1TOTStubDTC_phiDTC")[q] + np.pi) % (2 * np.pi) - np.pi
        assert np.abs(dphi).max() < 1e-5
        assert np.abs(z - g("L1TOTStubDTC_zDTC")[q]).max() < 1e-4
        assert np.abs(r - g("L1TOTStubDTC_rDTC")[q]).max() < 1e-4


@needs_test
@pytest.mark.skipif(not MODECMP_NODR.exists(), reason="modecmp no-DR file not present")
def test_chain_matches_cmssw_no_dr_tracks():
    """The emulated no-DR tracks' stub sets against CMSSW's on the same 10 events (barrel seeds, barrel-only
    tracks); CMSSW's tracks are identified by their stub positions (the modecmp file predates the stub links).
    The floor was registered on format 35's chain, which this runs (the discs and WP-Disc corrections off; the
    test production also predates innerStrip, which the discs need)."""
    import awkward as ak
    import uproot

    OC.set_discs(False, fixes=False)
    try:
        _chain_matches_cmssw_no_dr_tracks(ak, uproot)
    finally:
        OC.set_discs(True)


def _chain_matches_cmssw_no_dr_tracks(ak, uproot):

    t = uproot.open(f"{TEST_PROD}:Events")
    evs = t["event"].array(library="np")
    st = t.arrays([k for k in t.keys() if k.startswith(("L1TOTStub_", "L1TOTStubDTC_"))], library="ak")
    m = uproot.open(f"{MODECMP_NODR}:Events")
    mev = m["event"].array(library="np")
    mt = m.arrays(["L1TTrack_seedType"], library="ak")
    ms = m.arrays(["L1TTrackStub_trackIdx", "L1TTrackStub_x", "L1TTrackStub_y", "L1TTrackStub_z",
                   "L1TTrackStub_isBarrel"], library="ak")
    n_cms = n_same = 0
    for e in range(len(evs)):
        S = {k[10:]: ak.to_numpy(st[k][e]) for k in st.fields if k.startswith("L1TOTStub_")}
        Dt = {k[13:]: ak.to_numpy(st[k][e]) for k in st.fields if k.startswith("L1TOTStubDTC_")}
        nodr, _ = OC.run_event(S, Dt, do_dr=False)
        pos = {(float(a), float(b), float(c)): i for i, (a, b, c) in enumerate(zip(S["x"], S["y"], S["z"]))}
        j = int(np.nonzero(mev == evs[e])[0][0])
        sd = ak.to_numpy(mt["L1TTrack_seedType"][j])
        tix = ak.to_numpy(ms["L1TTrackStub_trackIdx"][j])
        X, Y, Z, B = (ak.to_numpy(ms["L1TTrackStub_" + c][j]) for c in ("x", "y", "z", "isBarrel"))
        ref = {}
        for i in range(len(sd)):
            q = tix == i
            if sd[i] > 3 or not B[q].all():
                continue
            key = tuple(sorted(pos.get((float(a), float(b), float(c)), -1) for a, b, c in zip(X[q], Y[q], Z[q])))
            ref[key] = ref.get(key, 0) + 1
        emu = {}
        for tr in nodr:
            if S["isBarrel"][list(tr.rows)].all():
                k = tuple(sorted(tr.rows))
                emu[k] = emu.get(k, 0) + 1
        n_cms += sum(ref.values())
        n_same += sum(min(v, emu.get(k, 0)) for k, v in ref.items())
    assert n_same / n_cms >= NODR_IDENTITY_FLOOR
