"""Unit tests of the census angle model (WP-D0b, census format 33): the cluster correction is
spdefs' table pull for any prediction, k = 1 is the identity, the knob moves class T only and each
angle by its own k, the per-angle stage switches, the cut-file refusals and the S6 per-n fit."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_refitq" / "combinatorics"))
import census_angles as CA          # noqa: E402
import angle_cuts as AC             # noqa: E402
import tracklet_topology_cost as M  # noqa: E402
import kf_emulation as KF         # noqa: E402
from ngtagger.spdefs import angles as SD  # noqa: E402

RNG = np.random.default_rng(20261014)


def _clusters(n=400):
    t = RNG.normal(0.0, 0.2, n)
    lam_t = RNG.normal(0.0, 0.5, n)
    return {"half_e": t + RNG.normal(0, 0.05, n), "half_t": t, "lam_e": lam_t + RNG.normal(0, 0.1, n),
            "lam_t": lam_t, "is_T": RNG.random(n) < 0.7, "sa_pub": np.full(n, 0.05),
            "sb_pub": np.full(n, 0.1), "ca": RNG.normal(0, 0.2, n), "wa": RNG.uniform(0.8, 1.3, n),
            "cb": RNG.normal(0, 0.2, n), "wb": RNG.uniform(0.8, 1.3, n)}


@pytest.mark.parametrize("k,mode", [(1.0, "fixed"), (0.5, "fixed"), (0.5, "scaled"), (0.0, "scaled")])
def test_correction_is_the_table_pull(k, mode):
    C = _clusters()
    pred = RNG.normal(0, 0.2, len(C["half_e"]))
    a2, s2 = CA.corrected("alpha", C["half_e"], C["half_t"], C["is_T"], C["sa_pub"], C["ca"], C["wa"], k, mode)
    a_k = SD.improve("alpha", C["half_e"], C["half_t"], C["is_T"], k)[0]
    ref = SD.table_pull(SD.residual("alpha", a_k, pred), C["sa_pub"], SD.sigma_k(C["sa_pub"], k, mode),
                        k, C["ca"], C["wa"])
    assert np.allclose(SD.residual("alpha", a2, pred) / s2, ref, rtol=1e-12, atol=1e-12)


def test_k1_moves_nothing_and_class_c_never_moves():
    C = _clusters()
    h, sh, lam, sl = CA.corrected_of(C, 1.0, 1.0, "fixed")
    assert np.allclose(h, SD.WRAP["alpha"](C["half_e"] - C["ca"] * C["sa_pub"]))
    h2, _, lam2, _ = CA.corrected_of(C, 0.3, 0.7, "scaled")
    c = ~C["is_T"]
    # class C keeps its emitted angle (only the k-scaled centre moves it)
    assert np.allclose(h2[c], SD.WRAP["alpha"](C["half_e"][c] - 0.3 * C["ca"][c] * C["sa_pub"][c]))
    # each angle by its own k: at k_alpha = 0 class-T alpha sits on truth (minus 0 centre)
    h0, _, _, _ = CA.corrected_of(C, 0.0, 1.0, "fixed")
    assert np.allclose(h0[C["is_T"]], C["half_t"][C["is_T"]])
    _, sh1, _, sl1 = CA.corrected_of(C, 1.0, 0.0, "scaled")
    assert np.allclose(sh1, C["wa"] * C["sa_pub"])            # alpha sigma untouched at k_alpha = 1
    assert np.allclose(sl1, C["wb"] * 0.05 * C["sb_pub"])     # beta's floored at k_beta = 0


def test_scaled_k0_is_floored():
    C = _clusters()
    _, sh, _, sl = CA.corrected_of(C, 0.0, 0.0, "scaled")
    assert np.allclose(sh, C["wa"] * 0.05 * C["sa_pub"])
    assert np.allclose(sl, C["wb"] * 0.05 * C["sb_pub"])


def test_per_angle_switches():
    ang = M.AngleUse(M.ANGLE_STAGES, angles={"attach": ["alpha"], "kf": []},
                     keys={"proj_arb": "emitted"})
    assert ang("attach") and ang.uses("attach", "alpha") and not ang.uses("attach", "beta")
    assert ang.subset("attach") == "a" and ang.subset("proj_window") == "ab"
    assert not ang("kf")                      # a per-angle stage with no angle is off
    assert ang.uses("veto", "alpha") and not ang.uses("veto", "beta")
    assert ang.uses("pair_z0", "beta") and not ang.uses("pair_z0", "alpha")
    back = M.AngleUse.from_key(json.loads(json.dumps(ang.key())))
    assert back.key() == ang.key()
    parsed = M.AngleUse.parse("all", "", "S3=alpha,S5=", "S4a=emitted")
    assert parsed.key() == ang.key()
    with pytest.raises(ValueError):
        M.AngleUse(angles={"veto": ["alpha"]})          # S1 is alpha-only by construction
    with pytest.raises(ValueError):
        M.AngleUse(keys={"kf": "true"})


def _doc(s5="ab", key="emitted", sub="ab"):
    return {"schema": CA.CUT_SCHEMA,
            "identity": {"table_sha256": "x", "k_alpha": 1.0, "k_beta": 1.0, "mode": "fixed",
                         "bench": None, "d0_prior_cm": 1.0, "level": CA.LEVEL, "floors": [2.0]},
            "s1": {"bins": [[2.0, 3]], "thr": [2.5]}, "s2a": {"bins": [], "thr": []},
            "s2z": {"bins": [], "thr": []},
            "win": {"bins": [[0, 10203, 3, 1, 2.0]], "thr": {"ab": [[3.0, 4.0]]}},
            "s6": {"bins": [["AAAA", s5, key, sub, 4]], "thr": [20.0]}}


class _Model:
    table_sha = "x"
    mode = "fixed"

    def calib_k(self):
        return (1.0, 1.0)


def _run(**kw):
    r = {"bench": None, "d0_prior_cm": 1.0, "level": CA.LEVEL, "floors": [2.0],
         "s6": ("ab", "emitted", "ab"), "win_subsets": ["ab"], "builds": ["AAAA"]}
    r.update(kw)
    return r


def test_cut_file_refuses_a_configuration_it_was_not_made_for():
    C = CA.CutSet(_doc())
    C.check(_Model(), _run())
    for bad in ({"s6": ("off", "emitted", "ab")}, {"win_subsets": ["a"]}, {"floors": [1.0, 2.0]},
                {"d0_prior_cm": 0.5}, {"bench": [8, 8]}):
        with pytest.raises(SystemExit):
            C.check(_Model(), _run(**bad))
    m = _Model()
    m.table_sha = "y"
    with pytest.raises(SystemExit):
        C.check(m, _run())
    assert C.s1_thr(2.0, np.array([3, -1])).tolist() == [2.5, np.inf]
    qa, qb = C.win_thr("ab", 0, 10203, 3, 2.0, np.array([1, 1]))
    assert qa.tolist() == [3.0, 3.0] and qb.tolist() == [4.0, 4.0]
    assert C.s6_thr("AAAA", "ab", "emitted", "ab", np.array([0, 4])).tolist() == [np.inf, 20.0]


def test_census_identity_is_accepted_by_its_cut_file():
    """The census's own run identity (tp_findability.cut_identity), not a hand-made one: the S6 build
    keys are the fit's (one mask -> its name; was 'A+A+A+A', which refused every cut file)."""
    import tp_findability as TF
    ang = M.AngleUse.parse("all")
    run = TF.cut_identity(ang, None, None, 2.0, None, ["AAAA", "AAAI", "IAAA"])
    assert run["builds"] == ["AAAA", "AAAI", "IAAA"]
    assert run["s6"] == ("ab", "emitted", "ab")
    C = CA.CutSet(_doc())
    C.check(_Model(), run)
    with pytest.raises(SystemExit):
        C.check(_Model(), TF.cut_identity(M.AngleUse.parse("all", off="kf"), None, None, 2.0, None, ["AAAA"]))


def test_s5_and_s6_are_off_by_default_and_switches():
    """Physicist, 2026-10-15: every stage but S5 (angles in the fit) and S6 (the angle cut; a
    feature only, its statistic still exported) by default; 'all' or naming kf / accept turns them on."""
    import tp_findability as TF
    assert M.DEFAULT_STAGES == frozenset(M.ANGLE_STAGES) - {"kf", "accept"}
    ang = M.AngleUse.parse()
    assert ang.on == M.DEFAULT_STAGES and not ang("kf") and not ang("accept")
    assert ang("dr") and ang("mva")                    # S7 and S8 read the S6 statistic
    assert M.AngleUse.parse("all")("kf") and M.AngleUse.parse("all")("accept")
    assert M.AngleUse.parse("default,kf,accept").on == frozenset(M.ANGLE_STAGES)
    assert M.AngleUse.parse(off="dr").on == M.DEFAULT_STAGES - {"dr"}
    assert TF.cut_identity(ang, None, None, 2.0, None, ["AAAA"])["s6"] is None   # no S6 thresholds read
    run = TF.cut_identity(M.AngleUse.parse("default,accept"), None, None, 2.0, None, ["AAAA"])
    assert run["s6"] == ("off", "emitted", "ab")
    assert {"n_angle_meas", "chi2_angle_per_meas"} <= set(KF.TRACK_COLS)
    assert not {"n_angle_meas", "chi2_angle_per_meas"} & set(KF.MVA_FEATURES)

def test_s6_per_n_keeps_99_percent_per_populated_n():
    n = np.repeat([2, 4, 6, 8], 4000)
    c2 = RNG.chisquare(n) * 1.2
    tp = np.arange(len(n))
    thr, grp, ntp = AC._s6_cell(c2, n, tp, None, CA.LEVEL, 8)
    for v in (2, 4, 6, 8):
        eff = np.mean(c2[n == v] <= thr[v])
        assert 0.985 <= eff <= 0.995, (v, eff)
    # sparse odd n borrow a populated neighbour's shape, scaled
    assert all(np.isfinite(thr[v]) for v in (1, 3, 5, 7))


def test_joint_window_statistic_keeps_99_percent_jointly():
    m = 20000
    pa, pb = RNG.standard_t(5, m), RNG.standard_t(5, m)
    ui = np.zeros(m, np.int64)
    thr, sstar = AC._fit_joint([(0,)], ui, pa, pb, np.arange(m), None, np.zeros(1, np.int64), CA.LEVEL)
    qa, qb = thr["ab"][0]
    eff = np.mean((np.abs(pa) <= qa) & (np.abs(pb) <= qb))
    assert 0.988 <= eff <= 0.992
    assert np.mean(np.abs(pa) <= qa) > 0.99 and np.mean(np.abs(pb) <= qb) > 0.99


def test_tp_weights_count_each_particle_once():
    w = AC.tp_weights(np.array([7, 7, 7, 9]), np.array([0, 0, 1, 0]))
    assert w.tolist() == [0.5, 0.5, 1.0, 1.0]
