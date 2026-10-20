"""Known-answer tests of the WP-Disc additions to the OT chain emulation (eval_refitq/combinatorics: ot_dtc,
ot_chain, cmssw_kf): the disc stub decoding, VMRouterCM's hidden inner-stub cuts, the wired projection list and
its format-35 switch, the disc projection's validity and the added-layer rule, the side-aware disc codes, and
std::sort's order of equal keys against libstdc++; then, with the f01 closure siblings present, C1 and C1b with
the discs on (regression floors at the development values of 2026-10-18)."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_refitq" / "combinatorics"))
import cmssw_kf as KF   # noqa: E402
import ot_chain as OC   # noqa: E402
import ot_dtc as D      # noqa: E402

CLOSURE = Path("/Volumes/WDMac/smartpixels_scratch/canon_v4/closure")
NODR = CLOSURE / "spstudy_AAAA_f01_100ev_noDR.root"
FAKEFIT = CLOSURE / "spstudy_AAAA_f01_10ev_fakefit.root"


@pytest.fixture(autouse=True)
def _discs_on():
    OC.set_discs(True)
    yield
    OC.set_discs(True)


# ---- std::sort -----------------------------------------------------------------------------------
# keys and the order libstdc++'s std::sort leaves their indices in (g++ 14, comp: key <), 2026-10-18
STD_SORT_REF = [
    ("1 2 2 1 0 0 1 2 1 2 0 0 2 0 0 2 0", "16 14 13 4 5 11 10 8 0 6 3 9 7 12 2 15 1"),
    ("0 2 0 1 1 2 2 2 0 1 0 1 2 1 2 2 1 2 2 0 0 2 1 2 0 2 2 1 1 1 2 0 1 1 2 2 0 2 2 2",
     "0 2 24 36 8 20 10 19 31 27 28 29 22 32 16 33 13 11 9 4 3 34 35 30 37 38 39 26 25 23 21 18 17 15 14 12 7 6 5 1"),
    ("2 2 1 4 1 2 0 3 2 3 3 1 2 1 2 1 1 2 4 4 0 2 4 4 1 2 4 0 4 3 3 2 3 1 1 1 2 2 0 3",
     "38 20 27 6 4 35 34 33 11 2 13 15 16 24 36 37 31 25 21 1 0 5 17 14 12 8 30 39 7 9 32 10 29 19 28 26 23 3 22 18"),
]


@pytest.mark.parametrize("keys,want", STD_SORT_REF)
def test_std_sort_orders_ties_as_libstdcxx(keys, want):
    k = [float(x) for x in keys.split()]
    got = list(range(len(k)))
    KF.std_sort(got, key=k.__getitem__)
    assert got == [int(x) for x in want.split()]
    OC.set_discs(False, fixes=False)                      # format 35: Python's stable sort
    got = list(range(len(k)))
    KF.std_sort(got, key=k.__getitem__)
    assert got == sorted(range(len(k)), key=k.__getitem__)


# ---- disc stubs ----------------------------------------------------------------------------------
def test_disc_rvalue_and_position():
    # Stub::rvalue(): a PS r above 10 counts from 2^8 kr; a 2S ring index (< 10) is kept
    assert D.disc_rvalue(np.array([11, 10, 3, 200]), np.ones(4, bool)).tolist() == [267, 10, 3, 456]
    disk, neg, ps = np.array([1, 1, 3]), np.array([False, True, False]), np.array([True, True, False])
    _, z, r = D.disc_physical(disk, neg, ps, np.array([100, 100, 100]), np.array([5, 5, -2]),
                              np.array([1000, 1000, 3]), np.array([0, 0, 0]))
    assert z.tolist() == pytest.approx([5 * D.kz(1) + D.zmean(1), 5 * D.kz(1) - D.zmean(1), -2 * D.kz(1) + D.zmean(3)])
    assert r[:2].tolist() == pytest.approx([1000 * D.kr()] * 2)
    assert r[2] == pytest.approx(D.RDSS_OUTER_MOD[1] + D.HALFSTRIP)   # ring 3 of D3: outer table, second module, +


def test_disc_codes_are_side_aware():
    # the chain's codes 7-11 are discs 1-5 with the side as its own flag; the KF's tmtt layerId carries the side
    assert [OC.layer_name(c) for c in (1, 6, 7, 11)] == ["L1", "L6", "D1", "D5"]
    plus = KF.make_stub(0, 30.0, 0.0, 131.0, 1, True, False, barrel=False)
    minus = KF.make_stub(1, 30.0, 0.0, -131.0, 1, True, False, barrel=False)
    assert (plus.layer, minus.layer) == (11, 21) and plus.reduced == minus.reduced


# ---- VMRouterCM's hidden inner-stub cuts ---------------------------------------------------------
def _cp(**cols):
    return SimpleNamespace(**{k: np.asarray(v) for k, v in cols.items()})


def test_vmr_hidden_cut_d1_d3_radius():
    rmax = OC.VMR_RMAX_D1D3 / D.kr()
    ir = [math.floor(rmax), math.floor(rmax) + 1, 2 * D.N_DSS_MOD - 1, 2 * D.N_DSS_MOD]
    cp = _cp(ir=ir, ps=[1, 1, 0, 1], iz=[0] * 4, z=[0.0] * 4)
    for seed in (4, 5):                                   # D1D2 inner D1, D3D4 inner D3
        assert OC._inner_ok(cp, np.arange(4), seed).tolist() == [True, False, False, True]


def test_vmr_hidden_cuts_l1_l2_z():
    k1, k2 = D.kz(1), D.kz(2)
    lo70, hi70 = math.ceil(70 / k1) - 1, math.ceil(70 / k1)
    lo95, hi95 = math.floor(95 / k1), math.floor(95 / k1) + 1
    cp = _cp(ir=[0] * 4, ps=[1] * 4, iz=[lo70, -hi70, lo95, -hi95], z=[0.0] * 4)
    assert OC._inner_ok(cp, np.arange(2), 6).tolist() == [False, True]          # L1D1: L1 |z| >= 70 cm
    assert OC._inner_ok(cp, np.arange(2, 4), 0).tolist() == [True, False]       # L1L2: L1 |z| <= 95 cm
    lo50, hi50 = math.ceil(50 / k2) - 1, math.ceil(50 / k2)
    cp2 = _cp(ir=[0] * 2, ps=[1] * 2, iz=[lo50, hi50], z=[50.5, 50.5])
    assert OC._inner_ok(cp2, np.arange(2), 7).tolist() == [False, True]         # L2D1: L2 |z| >= 50 cm only
    assert OC._inner_ok(cp2, np.arange(2), 1).tolist() == [False, False]        # L2L3 also needs |z| >= 52 cm


# ---- the wired projections -----------------------------------------------------------------------
def test_wired_targets_and_the_format35_switch():
    assert OC.active_seeds() == tuple(range(8))
    assert OC.targets(1) == (1, 4, 5, 7, 8, 9, 10)          # L2L3 -> L6 is not wired
    assert OC.targets(6) == (8, 9, 10, 11) and OC.targets(4) == (1, 2, 9, 10, 11)
    OC.set_discs(False, fixes=False)
    assert OC.active_seeds() == (0, 1, 2, 3) and not KF.STD_SORT
    assert OC.targets(1) == (1, 4, 5, 6) and OC.targets(0) == (3, 4, 5, 6)


def _ipars(t, z0=0.0):
    k = KF.K
    return 0, int((0.55 * k.dphisector_hg) / k.kphi0pars), int(round(z0 / k.kz0pars)), int(round(t / k.ktpars))


def test_added_layer_rule_and_disc_validity():
    ip = _ipars(1.14, z0=-5.0)                            # reaches L6 (z ~ 118.5 cm) and D1 (r ~ 119.5 cm)
    assert OC.proj_layer(ip, 6) is not None and OC.proj_disk(ip, 1) is not None
    assert sorted(OC.tracklet_projections(ip, 0)) == [3, 4, 5, 6]   # L1L2: D1 <-> L6, D2 <-> L5, D3 <-> L4
    assert sorted(OC.tracklet_projections(_ipars(1.6), 0)) == [3, 4, 7, 8]   # L5, L6 out of reach: D1, D2
    assert OC.proj_disk(_ipars(0.5), 1) is None           # r at D1 beyond rmaxdisk
    tcut = int(1.0 / KF.K.ktpars + 1)
    ip_low = (0, ip[1], 0, tcut)
    assert OC.proj_disk(ip_low, 1) is None                # |t| must exceed 1 / ktpars + 1
    OC.set_discs(False)
    assert all(L <= 6 for L in OC.tracklet_projections(ip, 0))


# ---- C1 / C1b on the f01 closure siblings (development values, 2026-10-18) -----------------------
@pytest.mark.skipif(not NODR.exists(), reason="f01 no-DR closure sibling not present")
def test_c1_identity_every_seed_and_region():
    import closure_vs_cmssw as CV
    rep, _ = CV.c1(str(NODR), nev=10)
    assert rep["judged"] >= 5600 and rep["identical"] == rep["judged"]
    assert set(rep["by_kf_region"]) == {str(r) for r in range(8)}
    assert set(rep["by_seed"]) == set(OC.SEED_NAME.values())


@pytest.mark.skipif(not (NODR.exists() and FAKEFIT.exists()), reason="f01 Fakefit / no-DR siblings not present")
def test_c1b_forward_lists_and_kf_on_cmssw_lists():
    import closure_vs_cmssw as CV
    rep, _ = CV.c1b(str(FAKEFIT), str(NODR))
    fw, br = rep["groups"]["forward"], rep["groups"]["barrel"]
    assert fw["n"] >= 3700 and fw["fraction"] >= 0.99 and fw["class"] != "MATERIAL"
    assert (br["identical"], br["n"]) == CV.C1B_ACCEPTED       # the barrel group is WP-D's
    assert rep["kf_both_accept_different"] == 0 and rep["kf_census_accepts_cmssw_rejects"] == 0
    assert rep["kf_census_rejects_cmssw_accepts"] == 0 and rep["tracklet_pars_identical"] == rep["same_tracklet"]
