"""Known-answer tests of the P0.3 table recipe and test rule (eval_refitq/p03_angles/table.py; amendment
1d): the table entries (median or 25 %-trimmed-mean centre, quantiles at the rank ceil(L (n + 1)), the
weighting width q95 / 1.96), the one-sided family-wise threshold, the exact coverage null with a FITTED
centre (stats review 01, F9) on a Gaussian and on a bimodal (64 / 35 %) toy, missing rows, and the
known-bad perturbation hook."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_refitq" / "p03_angles"))
import table as T  # noqa: E402

from ngtagger.spdefs import angles as SA  # noqa: E402


def test_bin_estimates_centre_quantiles_and_weighting_width():
    from scipy.stats import trim_mean

    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(0.3, 1.0, 900), rng.normal(6.0, 2.0, 101)]
    for centre in SA.CENTRES:
        e = SA.bin_estimates(x, centre=centre)
        c = np.median(x) if centre == "median" else trim_mean(x, 0.25)
        assert e["centre"] == pytest.approx(c, rel=1e-12)
        d = np.sort(np.abs(x - c))
        for lv in SA.LEVELS:
            k = int(min(np.ceil(lv * (len(x) + 1) - 1e-9), len(x)))
            assert e["q"][f"{lv:g}"] == d[k - 1]
        assert e["width"] == pytest.approx(e["q"]["0.95"] / 1.96, rel=1e-12)
        assert {"clip_centre", "core_width", "mad", "out3", "full_rms0"} <= set(e)
    assert SA.bin_estimates(x[:199]) is None
    # integer (bootstrap) weights repeat the values
    w = rng.integers(0, 3, len(x))
    assert SA.bin_estimates(x, w=w, diagnostics=False) == SA.bin_estimates(np.repeat(x, w), diagnostics=False)


def test_one_sided_threshold():
    from scipy.stats import norm
    assert T.t_one_sided(1214) == pytest.approx(norm.isf(1 - 0.99 ** (1 / 1214)), rel=1e-12)
    assert 4.25 < T.t_one_sided(1214) < 4.35 < T.t_family(1214)


@pytest.mark.parametrize("centre", SA.CENTRES)
@pytest.mark.parametrize("bimodal", [False, True])
def test_exact_null_holds_with_a_fitted_centre(bimodal, centre):
    """The fitted centre makes fit and test scores not exactly exchangeable (F9); the effect on the
    mid-p z must stay small: mean |.| < 0.08, sd within 0.08 of 1, at n_fit 744 / n_test 2232."""
    rng = np.random.default_rng(17 + bimodal)
    nf, nt = 744, 2232

    def draw(n):
        if not bimodal:
            return rng.standard_normal(n)
        k = rng.random(n) < 0.64
        return np.where(k, rng.normal(0, 0.15, n), rng.normal(5.6, 0.8, n))
    zs = {lv: [] for lv in SA.LEVELS}
    for _ in range(1200):
        f, t = draw(nf), draw(nt)
        e = SA.bin_estimates(f, centre=centre, diagnostics=False)
        d = np.abs(t - e["centre"])
        for lv in SA.LEVELS:
            m = np.sum(d > e["q"][f"{lv:g}"])
            e0 = SA.expected_coverage(nf, lv)
            zs[lv].append(float(SA.coverage_z(m, nt, e0, SA.exchangeable_coverage_variance(nf, nt, lv))))
    for lv, z in zs.items():
        z = np.asarray(z)
        assert abs(z.mean()) < 0.08 and abs(z.std() - 1) < 0.08, (lv, z.mean(), z.std())


def test_coverage_null_is_the_beta_binomial_of_the_rank():
    """k = min(ceil(L (n + 1)), n); the coverage null is the beta-binomial with that k."""
    from scipy import stats
    nf, nt, lv = 744, 2232, 0.99
    k = SA.quantile_order(nf, lv)
    d = stats.betabinom(nt, nf - k + 1, k)
    e0 = SA.expected_coverage(nf, lv)
    v = SA.exchangeable_coverage_variance(nf, nt, lv)
    up = d.sf(20) + 0.5 * d.pmf(20)
    assert SA.coverage_z(20, nt, e0, v) == pytest.approx(-stats.norm.isf(up) if up < 0.5 else stats.norm.isf(1 - up),
                                                         rel=1e-9, abs=1e-9)


def test_missing_row_is_not_uncovered():
    """Amendment 1c (N1): test clusters looked up in a row without an estimate are left out of the
    coverage; clusters with no row at all count as outside."""
    rng = np.random.default_rng(9)
    p = rng.standard_normal(300)
    K = {"trows": [{"surface": 1}], "rows": [{}, {}], "ts": np.array([0]), "te": np.array([300]),
         "test_lrow": np.r_[np.zeros(100, int), np.ones(100, int), -np.ones(100, int)],
         "test_p": {"alpha": p, "beta": p}}
    par = np.full((3, 2, 2 + len(SA.LEVELS)), np.nan)
    par[0, :, 0], par[0, :, 1], par[0, :, 2:] = 0.0, 1.0, 10.0          # row 0: covers everything
    st = T._test_stats(K, par, None)                                     # row 1: no estimate; -1: no row
    assert st[0, 0, T.ST_N] == 200                                       # 100 (row 0) + 100 (no row); row 1 missing
    assert st[0, 0, T.ST_COV] == pytest.approx(0.5)                      # the no-row clusters are outside
    assert np.isnan(st[0, 0, 0])                                         # < 200 clusters with an estimate: no centre


def test_after_table_centre_uses_the_table_estimator():
    rng = np.random.default_rng(12)
    p = np.r_[rng.normal(0.0, 1.0, 700), rng.normal(3.0, 0.5, 300)]
    K = {"trows": [{"surface": 1}], "rows": [{}], "ts": np.array([0]), "te": np.array([len(p)]),
         "test_lrow": np.zeros(len(p), int), "test_p": {"alpha": p, "beta": p}}
    par = np.full((2, 2, 2 + len(SA.LEVELS)), np.nan)
    par[0, :, 0], par[0, :, 1], par[0, :, 2:] = 0.2, 2.0, 50.0
    for centre in SA.CENTRES:
        T.G["centre"] = centre
        st = T._test_stats(K, par, None)
        assert st[0, 0, 0] == pytest.approx(SA.centre_of((p - 0.2) / 2.0, centre), rel=1e-12)
    T.G["centre"] = SA.CENTRE


def test_known_bad_perturbation_hook():
    """KB-q95 multiplies q95 and the width by 0.9, KB-c moves the centre by 0.5 widths, in the plateau-edge
    rows only (|key| range overlapping [0.50, 0.67))."""
    rng = np.random.default_rng(2)
    rows = [{"lo": 0.30, "hi": 0.40}, {"lo": 0.40, "hi": 0.55}, {"lo": -0.70, "hi": -0.60}, {"lo": 0.67, "hi": 0.80}]
    assert T.plateau_edge_rows(rows).tolist() == [False, True, True, False]
    x = rng.standard_normal(4 * 400)
    K = {"rows": rows, "fs": np.arange(4) * 400, "fe": np.arange(1, 5) * 400, "fit_p": {"alpha": x, "beta": x},
         "pert": None, "pert_rows": None}
    ref = T._fit_params(K, None)
    m = np.r_[T.plateau_edge_rows(rows), False]
    got = T._fit_params({**K, "pert": {"q95_scale": 0.9}, "pert_rows": m}, None)
    assert np.allclose(got[m, :, 1], 0.9 * ref[m, :, 1]) and np.allclose(got[m, :, 2 + T.I95], 0.9 * ref[m, :, 2 + T.I95])
    others = np.delete(np.arange(2 + len(SA.LEVELS)), [1, 2 + T.I95])
    assert np.array_equal(got[m][:, :, others], ref[m][:, :, others], equal_nan=True)
    assert np.array_equal(got[~m], ref[~m], equal_nan=True)
    got = T._fit_params({**K, "pert": {"centre_shift": 0.5}, "pert_rows": m}, None)
    assert np.allclose(got[m, :, 0], ref[m, :, 0] + 0.5 * ref[m, :, 1])
    assert np.array_equal(got[m][:, :, 1:], ref[m][:, :, 1:]) and np.array_equal(got[~m], ref[~m], equal_nan=True)
