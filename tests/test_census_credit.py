"""Unit tests of the census's hit credit and findable denominator (definition sheet truth.stub,
census.findable; WP-D0a, decisions of 2026-10-13 and 2026-10-14): an OT stub is ONE hit for each
TP it is credited to, however many of its clusters credit that TP; an IT cluster is one hit; two
or more credited hits of one TP on one module are ONE hit, the IT and OT modules kept apart; a TP
is findable when charged with credited hits on >= 4 modules, counting modules, not layers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_refitq" / "combinatorics"))
import kf_emulation as KF           # noqa: E402
import tp_findability as TF        # noqa: E402
import tracklet_topology_cost as M  # noqa: E402

A, B, C, N = 5, 7, 9, 11   # TP rows in L1TTP: A, B, C charged; N neutral (a photon)


def _unified(it, ot):
    """A minimal unified hit table: IT clusters (layer, tpIdx[, detId]) then OT stubs (layer,
    tpIdx, tpIdxInner, tpIdxOuter[, detId]), all in event 0, plus the event's L1TTP rows. A hit
    given no detId sits on a module of its own."""
    it = [h if len(h) == 3 else (*h, 1000 + i) for i, h in enumerate(it)]
    ot = [h if len(h) == 5 else (*h, 2000 + i) for i, h in enumerate(ot)]
    it_l, it_tp, it_det = (np.array(x) for x in zip(*it)) if it else (np.zeros(0, int),) * 3
    ot_l, ot_p, ot_i, ot_o, ot_det = (np.array(x) for x in zip(*ot))
    tps = np.r_[M.credited_tps(it_tp), M.credited_tps(ot_p, ot_i, ot_o)]
    n = len(it_l) + len(ot_l)
    ntp = 12
    L = {"key": M.tp_key(np.zeros(ntp, np.int64), np.arange(ntp)),
         "pt": np.full(ntp, 3.0, np.float32), "eta": np.zeros(ntp, np.float32),
         "phi": np.zeros(ntp, np.float32), "d0": np.zeros(ntp, np.float32),
         "z0": np.zeros(ntp, np.float32), "vx": np.zeros(ntp, np.float32),
         "vy": np.zeros(ntp, np.float32), "charge": np.ones(ntp, np.float32),
         "tanL": np.zeros(ntp, np.float32), "phi0": np.zeros(ntp, np.float32)}
    L["charge"][N] = 0.0
    U = {"layer": np.r_[it_l, np.asarray(ot_l) + 10], "event": np.zeros(n, np.int64),
         "tpIdx": np.r_[it_tp, ot_p].astype(np.int32), "tps": tps,
         "module": np.r_[TF.module_key(0, it_det), TF.module_key(1, ot_det)],
         "tpPt": np.full(n, 3.0, np.float32), "l1ttp": L}
    M.attach_tp_truth(U, L)
    return U


def _row(cen, tp):
    return int(np.flatnonzero(cen["key"] == M.tp_key(np.array([0]), np.array([tp]))[0])[0])


def test_a_stub_credited_through_both_clusters_is_one_hit():
    # one stub: genuine for A (tpIdx = inner = outer = A); one combinatoric stub with A in both clusters
    U = _unified([], [(1, A, A, A), (2, -1, A, A)])
    cen = TF.chunk_census(U)
    r = _row(cen, A)
    assert cen["n_mod_ot"][r] == 2 and cen["n_mod_it"][r] == 0   # two stubs, not four clusters


def test_a_stub_counts_once_for_each_of_two_tps():
    # a combinatoric stub with A's inner cluster and B's outer cluster: one hit for A, one for B
    U = _unified([], [(3, -1, A, B)])
    cen = TF.chunk_census(U)
    assert cen["n_mod_ot"][_row(cen, A)] == 1 and cen["n_mod_ot"][_row(cen, B)] == 1


def test_one_hit_per_module_modules_not_layers_and_the_threshold():
    # A: two clusters on IT module 100 (one hit) + one on module 101 of the same layer (an overlap)
    #    + one in IL2 + one stub on OT module 100 (the same rawId number, another system) = 4 modules
    #    on 3 layers -> findable
    # B: two stubs on OT module 300 (one hit) + two IT clusters = 4 hits on 3 modules -> not findable
    # N: 4 modules but neutral -> not findable
    U = _unified([(1, A, 100), (1, A, 100), (1, A, 101), (2, A, 200), (1, B, 400), (2, B, 401),
                  (1, N), (2, N), (3, N), (4, N)],
                 [(1, A, A, A, 100), (1, B, B, B, 300), (1, -1, B, B, 300)])
    cen = TF.chunk_census(U)
    rA, rB, rN = _row(cen, A), _row(cen, B), _row(cen, N)
    assert (cen["n_mod_it"][rA], cen["n_mod_ot"][rA]) == (3, 1)
    assert bin(int(cen["hit_it"][rA])).count("1") + bin(int(cen["hit_ot"][rA])).count("1") == 3
    assert (cen["n_mod_it"][rB], cen["n_mod_ot"][rB]) == (2, 1)
    f = TF.findable(cen)
    assert f[rA] and not f[rB] and not f[rN]
    assert TF.FINDABLE_MIN_MODULES == 4


def test_unknown_stub_and_unmatched_cluster_credit_nobody():
    U = _unified([(1, -1)], [(4, -1, -1, -1), (1, C, C, C)])
    cen = TF.chunk_census(U)
    assert list(cen["key"]) == [M.tp_key(np.array([0]), np.array([C]))[0]]
    assert cen["n_mod_it"][0] == 0 and cen["n_mod_ot"][0] == 1


def test_a_tied_owner_is_no_owner():
    # majority_owner, as CMSSW (truth spec sec. 4; census format 32): a tie gives no owner and n_own 0
    tp = np.array([[A, B, -1, -1],          # 1-1 tie
                   [A, A, B, -1],           # A
                   [A, B, A, B],            # 2-2 tie
                   [-1, -1, -1, -1],        # nothing credited
                   [-3, -3, B, -1]])        # an owner outside L1TTP still wins
    own, n_own, n_top = KF.majority_owner(tp, np.ones(tp.shape, bool))
    assert own.tolist() == [-1, A, -1, -1, -3]
    assert n_own.tolist() == [0, 2, 0, 0, 2]
    assert n_top.tolist() == [2, 1, 2, 0, 1]
    # a stub voting for both of its clusters' TPs: A on one stub, B on one, both on a third -> tie
    tps = np.array([[[A, -1, -1], [B, -1, -1], [A, B, -1]]])
    own, n_own, owned, right, n_top = KF.credited_owner(tps, np.ones((1, 3), bool))
    assert own[0] == -1 and n_own[0] == 0 and not owned[0] and not right.any() and n_top[0] == 2


def test_a_seed_credited_to_two_tps_has_no_seed_tp_but_is_truthful():
    # two stubs, each credited to A and B through its clusters: truthful, but no unique seed TP
    U = _unified([], [(1, -1, A, B), (2, -1, B, A), (3, A, A, A)])
    tp, real = M.seed_tp(U, np.array([0, 0]), np.array([1, 2]))
    assert real.tolist() == [True, True] and tp.tolist() == [-1, A]
