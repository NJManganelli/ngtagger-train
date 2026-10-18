"""Unit tests of the WP-D closure's bookkeeping and statistics (eval_refitq/combinatorics/closure_vs_cmssw):
the shared record rule, the event exclusion, the no-DR alignment, bit-level branch comparison, the joint
power at another sample size and the known-bad classification."""
import math
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_refitq" / "combinatorics"))
import closure_vs_cmssw as CV   # noqa: E402
import tracklet_topology_cost as M  # noqa: E402


def _stubs(layers, xyz, tp, inner=None, outer=None):
    """A one-event L1TOTStub table: barrel stubs on `layers` at `xyz`, credited to `tp` (their own and both
    clusters' owner unless given)."""
    n = len(layers)
    tp = np.asarray(tp, np.int32)
    return {"layer": np.asarray(layers), "isBarrel": np.ones(n, bool), "x": np.array([p[0] for p in xyz]),
            "y": np.array([p[1] for p in xyz]), "z": np.array([p[2] for p in xyz]), "tpIdx": tp,
            "tpIdxInner": tp if inner is None else np.asarray(inner, np.int32),
            "tpIdxOuter": tp if outer is None else np.asarray(outer, np.int32), "event": np.zeros(n, np.int64)}


def _line(rs, z0=1.0, cot=0.2, curl=1e-3):
    """Stub positions on a slightly curved track (exacttracklet needs rinv != 0, as any real track has)."""
    return [(r * math.cos(curl * r), r * math.sin(curl * r), z0 + r * cot) for r in rs]


RS = (23.0, 36.0, 51.0, 68.0, 87.0, 108.0)


def test_record_rule_seed_tp_owner_and_region():
    L = _stubs([1, 2, 3, 4], _line(RS[:4]), [7, 7, 7, 9])
    rec = dict(zip(CV.RECORD_COLS, CV.track_record(L, 0, 0, [0, 1, 2, 3])))   # L1L2 seed: stubs 0, 1
    assert rec["tp_key"] == int(M.tp_key(np.array([0]), np.array([7]))[0]) and rec["real"] == 1
    assert rec["own_key"] == rec["tp_key"] and rec["n_wrong"] == 1 and rec["nstub"] == 4
    assert 0 <= rec["kfreg"] <= CV.MAX_KF_REGION_BARREL                              # cot 0.2: central
    # a fake seed (the two seed stubs credit different TPs) is not seed-truthful
    L2 = _stubs([1, 2, 3, 4], _line(RS[:4]), [7, 8, 7, 7])
    rec2 = dict(zip(CV.RECORD_COLS, CV.track_record(L2, 0, 0, [0, 1, 2, 3])))
    assert rec2["tp_key"] == -1 and rec2["real"] == 0 and rec2["n_wrong"] == 1
    # no stub in a seed layer: not judged (region -1), on either side alike
    rec3 = dict(zip(CV.RECORD_COLS, CV.track_record(L, 0, 0, [1, 2, 3])))
    assert rec3["kfreg"] == -1 and len(CV._judged(np.array([CV.track_record(L, 0, 0, [1, 2, 3])]))) == 0


def test_record_rule_uses_the_final_seed_layer_stub():
    # two stubs in L1 (rows 0 and 4): the record uses the fitted one the track holds
    L = _stubs([1, 2, 3, 4, 1], _line(RS[:4]) + [(23.1, 0.5, 5.0)], [7, 7, 7, 7, 8])
    assert dict(zip(CV.RECORD_COLS, CV.track_record(L, 0, 0, [4, 1, 2, 3])))["real"] == 0
    assert dict(zip(CV.RECORD_COLS, CV.track_record(L, 0, 0, [0, 1, 2, 3])))["real"] == 1


def test_event_mask():
    k = CV._event_mask(12, "0-9")
    assert k.tolist() == [False] * 10 + [True] * 2
    assert CV._event_mask(5, None).all() and CV._event_mask(5, "1-1,3-4").tolist() == [True, False, True, False, False]


def test_align_nodr_is_an_ordered_subsequence():
    cands = [(0, 1, [1, 2, 3, 4]), (0, 1, [1, 2, 5, 6]), (2, 1, [7, 8, 9]), (0, 2, [1, 2, 3])]
    T = {"rInv": np.zeros(2), "seedType": np.array([0, 2]), "phiSector": np.array([1, 1])}
    S = {"trackIdx": np.array([0, 0, 0, 1, 1, 1]), "otStubIdx": np.array([1, 2, 5, 7, 8, 9])}
    match, left = CV._align_nodr(cands, (T, S))
    assert match == {1: 0, 2: 1} and left == 0
    T2 = {"rInv": np.zeros(1), "seedType": np.array([3]), "phiSector": np.array([1])}
    assert CV._align_nodr(cands, (T2, {"trackIdx": np.array([0]), "otStubIdx": np.array([1])}))[1] == 1


def test_same_compares_bits():
    a = ak.Array([[0.0, 1.0], [np.nan]])
    assert CV._same(a, ak.Array([[0.0, 1.0], [np.nan]]))
    assert not CV._same(a, ak.Array([[-0.0, 1.0], [np.nan]]))
    assert not CV._same(a, ak.Array([[0.0], [1.0, np.nan]]))


def test_round_mantissa_matches_nanoaod():
    x = np.array([1.0 + 2 ** -13, 1.0 + 3 * 2 ** -14, -2.5, 0.0], np.float32)
    r = CV._round_mantissa(x, 12)
    assert r[0] == np.float32(1.0 + 2 ** -12) and r[2] == np.float32(-2.5) and r[3] == 0.0


def test_joint_power_scales_with_the_sample():
    rng = np.random.default_rng(1)
    stats = {"C3": {"d_eff": 0.002, "disc": 0.004}}
    boot = {"C3": {"d_eff": 0.002 + rng.normal(0, 0.005, 400), "disc": 0.004 + np.abs(rng.normal(0, 0.006, 400))}}
    p1, p9 = CV._joint_power(stats, boot), CV._joint_power(stats, boot, 9.0)
    assert 0.0 <= p1 < p9 <= 1.0 and p9 > 0.9
    # a statistic centred outside its margin: more events do not help
    bad = {"C3": {"d_eff": 0.03}}
    assert CV._joint_power(bad, {"C3": {"d_eff": 0.03 + rng.normal(0, 0.005, 400)}}, 9.0) == 0.0


def test_known_bad_classes():
    assert CV._kb_class(1.5, False) == "fails, decision-scale"
    assert CV._kb_class(3.0, False) == "fails"
    assert CV._kb_class(1.2, True) == "gate defect"
    assert CV._kb_class(0.4, True) == "below resolution"
    assert CV._min_sample() == min(n for n in range(1, 2000) if CV._wilson_lower(n, n) >= CV.C1B_MATCH_MIN)
    assert math.isclose(CV._wilson_lower(0, 0), 0.0)


def test_match_list_gate_physicist_acceptance():
    # the measured agreement (plan sec. 11) is accepted; below it the 99 % rule decides; above 99 % it passes
    lo, acc, ok = CV.match_list_gate(2446, 2482)
    assert lo < CV.C1B_MATCH_MIN and acc and ok
    assert CV.match_list_gate(2445, 2482)[1:] == (False, False)
    lo, acc, ok = CV.match_list_gate(2482, 2482)
    assert lo >= CV.C1B_MATCH_MIN and not acc and ok


def test_injected_known_bads_shift_by_their_margins():
    rng = np.random.default_rng(0)
    den = np.ones(1000, bool)
    f_cen = rng.random(1000) < 0.8
    f_cms = f_cen.copy()
    f = CV._inject_found(f_cen, f_cms, den, "d_eff", 1.5, CV.MARGIN["d_eff"], rng)
    assert (f.sum() - f_cen.sum()) / 1000 == pytest.approx(-0.015, abs=1e-3)
    f = CV._inject_found(f_cen, f_cms, den, "disc", 1.5, CV.MARGIN["disc"], rng)
    assert f.sum() == f_cen.sum() and (f ^ f_cms).sum() / 1000 == pytest.approx(0.03, abs=2e-3)
    assert CV._injection(["C2 L1L2: d_eff=1.5", "C3 delivered: d_dup=10"], "C2 L1L2") == {"d_eff": 1.5}


# ---- WP-Disc (format 36) ---------------------------------------------------------------------------
def test_tp_kf_index_on_the_kf_edges():
    eta = np.array([0.0, 0.19, -0.2, 0.89, 0.9, 1.67, 1.68, -2.39, 2.4, np.nan])
    assert CV.tp_kf_index(eta).tolist() == [0, 0, 1, 3, 4, 5, 6, 7, 8, 8]
    assert [CV.BANDS[b] for b in ("barrel", "overlap", "forward")] == [(0, 3), (4, 5), (6, 7)]


def test_record_rule_disc_seed():
    """A D1D2 track (L1TOTStub disc codes 11, 12): its seed stubs are its disc stubs, its KF region forward."""
    rs = [40.0, 45.0, 52.0, 60.0]
    xyz = [(r, 0.001 * r * r, 0.5 + 3.0 * r) for r in rs]           # cot 3: |eta| ~ 1.8
    L = _stubs([11, 12, 13, 14], xyz, [5, 5, 5, 5])
    L["isBarrel"][:] = False
    rec = dict(zip(CV.RECORD_COLS, CV.track_record(L, 0, 4, [0, 1, 2, 3])))
    assert rec["real"] == 1 and rec["tp_key"] >= 0 and rec["kfreg"] in (6, 7)


def test_g0_array_rule():
    a = np.arange(14).reshape(2, 7)
    assert CV._g0_array("hit_o0", a, a.copy()) == ""
    wide = np.c_[a, np.full((2, 5), -1)]
    assert CV._g0_array("hit_o0", a, wide) == ""
    wide[1, 9] = 3
    assert CV._g0_array("hit_o0", a, wide) == "disc columns filled"
    assert CV._g0_array("cen_hit_otd", None, np.zeros(4, np.uint8)) == ""
    assert CV._g0_array("cen_hit_otd", None, np.ones(4, np.uint8)) == "new array"
    assert CV._g0_array("cen_pt", np.ones(3), np.r_[1.0, 1.0, 2.0]) == "values differ"
