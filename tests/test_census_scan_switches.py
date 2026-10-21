"""Known-answer tests of the census-scan switches (census_scan/PLAN.md revision 1): S1's placement (the physicist's
ruling: beamline pointing only at pair formation), the angle regions, and the null control's permutation."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_refitq" / "combinatorics"))
import tp_findability as TF        # noqa: E402
import tracklet_topology_cost as M  # noqa: E402
from ngtagger.spdefs import angles as SD  # noqa: E402


class _Veto(Exception):
    pass


def _project_inputs(n_target=3):
    D = {"layer": np.r_[np.ones(2, np.int8), np.full(2, 2, np.int8), np.full(n_target, 3, np.int8)],
         "globalR": np.r_[3.0, 3.0, 7.0, 7.0, np.full(n_target, 11.0)], "globalZ": np.zeros(4 + n_target),
         "globalPhi": np.zeros(4 + n_target), "sigY": np.full(4 + n_target, 0.002),
         "event": np.zeros(4 + n_target, np.int64)}
    idx = {1: np.array([0, 1]), 2: np.array([2, 3]), "r_med": {1: 3.0, 2: 7.0}}
    e = np.zeros(0, np.int64)
    return D, idx, e


@pytest.mark.parametrize("s1_at,vetoed", [("pairs", False), ("all", True)])
def test_s1_acts_on_targets_only_when_all(monkeypatch, s1_at, vetoed):
    """At s1_at pairs the projection's target set is the layer's full cluster set (S1 never called there); at all,
    S1 vetoes the target layer as before."""
    def spy(*a, **k):
        raise _Veto()
    monkeypatch.setattr(M, "alpha_veto", spy)
    D, idx, e = _project_inputs()
    ang = M.AngleUse(M.DEFAULT_STAGES, s1_at=s1_at)
    out = {}
    if vetoed:
        with pytest.raises(_Veto):
            M.it_project(D, {}, out, e, e, np.zeros(0), np.zeros(0), np.zeros(0), idx, 1, 2, 3, 2.0, use_angles=ang)
    else:
        try:
            M.it_project(D, {}, out, e, e, np.zeros(0), np.zeros(0), np.zeros(0), idx, 1, 2, 3, 2.0,
                         use_angles=ang)
        except _Veto:
            pytest.fail("S1 acted on a target layer at s1_at pairs")
        except Exception:
            pass                                   # the empty pair set may stop later stages; in_C is set first
        assert out["in_C"] == 3


def test_angle_use_key_and_legacy_default():
    a = M.AngleUse(M.DEFAULT_STAGES)
    assert (a.s1_at, a.regions) == ("pairs", "both") and a.key()["s1_at"] == "pairs"
    k = a.key()
    del k["s1_at"], k["regions"]                     # a key written before the switches existed
    assert M.AngleUse.from_key(k).s1_at == "all"
    with pytest.raises(ValueError):
        M.AngleUse(M.DEFAULT_STAGES, s1_at="follow")


def test_angle_regions_keep_positions_and_drop_angles():
    U = {"layer": np.array([1, 2, 5, 6, 12])}
    Q = {"dir_ok": np.array([True, True, True, True, True]), "ldir_ok": np.array([False, False, True, True, False])}
    Qb = TF.angle_regions(U, Q, "barrel")             # the discs' angles off
    assert Qb["dir_ok"].tolist() == [True, True, True, True, True]
    assert Qb["ldir_ok"].tolist() == [False] * 5 and Qb["ang_off"].tolist() == [False, False, True, True, False]
    Qd = TF.angle_regions(U, Q, "disc")               # the barrel's angles off
    assert Qd["dir_ok"].tolist() == [False, False, True, True, True]
    assert Qd["ldir_ok"].tolist() == [False, False, True, True, False]
    assert Qd["ang_off"].tolist() == [True, True, False, False, False]
    # S1 passes an off region's clusters whatever their statistic (as with every stage off)
    D = {"layer": U["layer"]}
    Qs = {"kap_a": np.full(5, 99.0), "s_kap": np.full(5, 1e-3), "ovf_a": np.zeros(5, bool), **Qd}
    kept = M.alpha_veto(D, Qs, np.arange(5), 0.5, False)
    assert set(kept.tolist()) >= {0, 1}


def test_null_permutation():
    """Per event and surface the usable clusters' (alpha, cot, local cots) are a permutation of the original ones;
    positions, sigmas, flags and truth stay with the cluster; unusable clusters are untouched; the seed fixes it."""
    rng = np.random.default_rng(3)
    n = 40
    I = {"event": np.repeat([10, 11], n // 2), "layer": np.tile(np.r_[np.ones(10, np.int8), np.full(10, 5, np.int8)], 2),
         "globalPhi": rng.uniform(-3, 3, n), "globalClusterPhi": rng.uniform(-3, 3, n),
         "globalClusterCotTheta": rng.normal(0, 2, n), "localCotAlpha": rng.normal(0, 0.3, n),
         "localCotBeta": rng.normal(0, 1, n), "hasAlpha": np.ones(n, bool), "hasBeta": np.ones(n, bool),
         "sigGlobalClusterPhi": rng.uniform(0.01, 0.02, n)}
    I["hasAlpha"][[3, 25]] = False                    # two unusable clusters
    ref = {k: np.asarray(v).copy() for k, v in I.items()}
    alpha0 = SD.alpha_of(ref["globalPhi"], ref["globalClusterPhi"])
    TF.permute_angles(I, 10)
    alpha1 = SD.alpha_of(I["globalPhi"], I["globalClusterPhi"])
    for e in (10, 11):
        for L in (1, 5):
            g = (I["event"] == e) & (I["layer"] == L) & ref["hasAlpha"]
            assert np.allclose(np.sort(alpha1[g]), np.sort(alpha0[g]))
            assert np.array_equal(np.sort(I["globalClusterCotTheta"][g]), np.sort(ref["globalClusterCotTheta"][g]))
    for c in ("globalPhi", "sigGlobalClusterPhi", "hasAlpha", "event", "layer"):
        assert np.array_equal(I[c], ref[c])
    for i in (3, 25):
        assert I["globalClusterCotTheta"][i] == ref["globalClusterCotTheta"][i]
    assert not np.allclose(alpha1, alpha0)
    J = {k: v.copy() for k, v in ref.items()}
    TF.permute_angles(J, 10)
    assert np.array_equal(J["globalClusterCotTheta"], I["globalClusterCotTheta"])
