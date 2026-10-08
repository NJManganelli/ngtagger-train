"""Unit tests of the definition sheet's reference implementations (ngtagger.definitions);
each with the known-bad it must catch."""
import numpy as np
import pytest

from ngtagger import definitions as D
from ngtagger import truth_helix as TH
from ngtagger.definitions import estimators as E
from ngtagger.definitions import helix as H
from ngtagger.definitions import matched_wp as MW
from ngtagger.definitions import puppi_dxy as PD


def test_sheet_is_complete():
    assert D.check() == []
    ids = set(D.items())
    for need in ("truth.delivery", "truth.census_fake", "truth.genealogy", "truth.stub", "truth.class_T",
                 "frame.d0", "frame.phi0", "frame.module_orientation", "angle.k_knob", "angle.error_table",
                 "angle.cut_level", "fom.matched_wp", "fom.harm", "fom.first_visited_surface",
                 "fom.attachment", "estimator.core", "estimator.sentinels", "census.ot_truncated"):
        assert need in ids, need
    assert D.item("angle.cut_level")["params"]["level"] == 0.99
    assert D.item("angle.k_knob")["params"]["scaled_floor"] == 0.05
    assert D.item("fom.harm")["params"] == {"d0_mm": 0.5, "z0_mm": 1.0}


def _residuals_cm(n=5000, seed=3):
    rng = np.random.default_rng(seed)
    return np.r_[rng.normal(0, 0.05, n), rng.normal(0, 0.6, n // 5)]       # cm, with a 6 mm tail


def test_core_is_unit_free_and_the_pull_table_core_is_not():
    x = _residuals_cm()
    assert E.scale_invariance(x)["rel_spread"] < 1e-9
    # known-bad: the census pull table's |x| < 1e3 cut and absolute tolerance (2026-10-09 finding)
    assert E.scale_invariance(x, estimator=E.core_pull_table)["rel_spread"] > 0.01


def test_core_sentinels_need_a_validity_mask():
    x = _residuals_cm()
    with_sentinel = np.r_[x, np.full(300, -999.0)]
    valid = with_sentinel != -999.0
    assert E.core(with_sentinel, valid)["rms0"] == pytest.approx(E.core(x)["rms0"], rel=1e-12)


def test_core_matches_the_pull_table_on_pulls():
    rng = np.random.default_rng(1)
    p = np.r_[rng.normal(0, 1.1, 20000), rng.standard_cauchy(500)]
    assert E.core(p)["rms0"] == pytest.approx(E.core_pull_table(p)["rms0"], rel=1e-6)


def test_tie_split_matches_the_target_exactly_on_discrete_scores():
    rng = np.random.default_rng(5)
    sb = np.round(rng.beta(2, 5, 3000) * 255) / 255          # 8-bit scores: many ties
    ss = np.round(rng.beta(5, 2, 1000) * 255) / 255
    for target in (0.4898, 0.10, 0.01, 0.0, 1.0):
        w = rng.poisson(1.0, len(sb)).astype(float)
        p = MW.matched_point(ss, sb, target, w_b=w)
        assert p["realised_background"] == pytest.approx(target, abs=1e-12)
    # known-bad: dropping the tied jets (the old `thresholds` / `frac_above`) misses the target
    t, _ = MW.tie_split_threshold(sb, 0.4898)
    assert abs(MW.accepted_fraction(sb, t, 0.0) - 0.4898) > 1e-3


def test_puppi_dxy_code3_interval_is_asymmetric():
    d0_um = np.array([-80.0, -50.0, -39.0, 0.0, 39.0, 78.0, 78.2])
    code, known = PD.code_of_d0(d0_um * 1e-4)
    assert known.all()
    assert code.tolist() == [7, 7, 3, 3, 3, 3, 7]
    lo, hi = PD.d0_interval_um(3)
    assert (lo, hi) == (0.0, 78.125)                         # in |hwD0| terms; signed: [-39.06, 78.13)


def test_helix_from_poca_meets_the_production_point():
    rng = np.random.default_rng(2)
    n = 500
    pt, q = rng.uniform(1, 20, n), rng.choice([-1.0, 1.0], n)
    phi = rng.uniform(-np.pi, np.pi, n)
    vx, vy = rng.normal(0, 0.5, n), rng.normal(0, 0.5, n)
    phi0 = TH.poca_phi0(phi, pt, q, vx, vy)
    # d0 of the circle through (vx, vy) with azimuth phi, in the sheet's convention
    R = pt / H.KPT_CMSSW
    xc, yc = vx + q * R * np.sin(phi), vy - q * R * np.cos(phi)              # centre (dphi/ds = -q/R)
    d0 = q * (np.hypot(xc, yc) - R)
    r = np.hypot(vx, vy)
    out = (r > 0.3) & (vx * np.cos(phi) + vy * np.sin(phi) > 0)     # produced moving outward
    x, y, _, _ = H.at_radius(r[out], pt[out], q[out], phi0[out], d0[out])
    assert np.nanmax(np.hypot(x - vx[out], y - vy[out])) < 1e-6
    # known-bad: the production azimuth in place of phi0 misses by more than a micron
    xb, yb, _, _ = H.at_radius(r[out], pt[out], q[out], phi[out], d0[out])
    assert np.nanmax(np.hypot(xb - vx[out], yb - vy[out])) > 1e-4
