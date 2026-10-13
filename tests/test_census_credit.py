"""Unit tests of the census's hit credit and findable denominator (definition sheet truth.stub,
census.findable; WP-D0a, decision of 2026-10-13): an OT stub is ONE hit for each TP it is
credited to, however many of its clusters credit that TP; an IT cluster is one hit; a TP is
findable when charged with >= 4 credited hits, counting hits, not layers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_refitq" / "combinatorics"))
import tp_findability as TF        # noqa: E402
import tracklet_topology_cost as M  # noqa: E402

A, B, C, N = 5, 7, 9, 11   # TP rows in L1TTP: A, B, C charged; N neutral (a photon)


def _unified(it, ot):
    """A minimal unified hit table: IT clusters (layer, tpIdx) then OT stubs (layer,
    tpIdx, tpIdxInner, tpIdxOuter), all in event 0, plus the event's L1TTP rows."""
    it_l, it_tp = (np.array(x) for x in zip(*it)) if it else (np.zeros(0, int), np.zeros(0, int))
    ot_l, ot_p, ot_i, ot_o = (np.array(x) for x in zip(*ot))
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
    assert cen["n_hit_ot"][r] == 2 and cen["n_hit_it"][r] == 0   # two stubs, not four clusters


def test_a_stub_counts_once_for_each_of_two_tps():
    # a combinatoric stub with A's inner cluster and B's outer cluster: one hit for A, one for B
    U = _unified([], [(3, -1, A, B)])
    cen = TF.chunk_census(U)
    assert cen["n_hit_ot"][_row(cen, A)] == 1 and cen["n_hit_ot"][_row(cen, B)] == 1


def test_hits_not_layers_and_the_threshold():
    # A: two overlapping IT clusters in IL1 + one in IL2 + one stub = 4 hits on 3 layers -> findable
    # B: 3 hits -> not findable; N: 4 hits but neutral -> not findable
    U = _unified([(1, A), (1, A), (2, A), (1, B), (2, B), (1, N), (2, N), (3, N), (4, N)],
                 [(1, A, A, A), (1, B, B, B)])
    cen = TF.chunk_census(U)
    rA, rB, rN = _row(cen, A), _row(cen, B), _row(cen, N)
    assert (cen["n_hit_it"][rA], cen["n_hit_ot"][rA]) == (3, 1)
    assert bin(int(cen["hit_it"][rA])).count("1") + bin(int(cen["hit_ot"][rA])).count("1") == 3
    f = TF.findable(cen)
    assert f[rA] and not f[rB] and not f[rN]
    assert TF.FINDABLE_MIN_HITS == 4


def test_unknown_stub_and_unmatched_cluster_credit_nobody():
    U = _unified([(1, -1)], [(4, -1, -1, -1), (1, C, C, C)])
    cen = TF.chunk_census(U)
    assert list(cen["key"]) == [M.tp_key(np.array([0]), np.array([C]))[0]]
    assert cen["n_hit_it"][0] == 0 and cen["n_hit_ot"][0] == 1
