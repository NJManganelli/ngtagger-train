"""Known-answer tests of the P0.3 table test rule (eval_refitq/p03_angles/table.py; amendment 1b):
the bimodality criterion of the stability flag, the one-sided family-wise threshold, the scale
equivariance the width known-bads rely on, and the exact coverage null with a FITTED centre
(stats review 01, F9), on a Gaussian and on a bimodal (64 / 35 %) toy."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_refitq" / "p03_angles"))
import table as T  # noqa: E402

from ngtagger.spdefs import angles as SA  # noqa: E402


def test_bimodal_criterion_false_positive_rate_and_power():
    rng = np.random.default_rng(3)
    fp_gauss = np.mean([T.em_two_gaussians(rng.normal(1.0, 0.03, 250))["bimodal"] for _ in range(200)])
    fp_gamma = np.mean([T.em_two_gaussians(0.5 + 0.02 * rng.gamma(4.0, size=250))["bimodal"] for _ in range(200)])
    fp_t3 = np.mean([T.em_two_gaussians(1.0 + 0.02 * rng.standard_t(3, size=250))["bimodal"] for _ in range(200)])
    assert fp_gauss <= 0.02 and fp_gamma <= 0.05 and fp_t3 <= 0.05
    # the D3 row: fitted widths 0.27-0.31 in ~65 % of re-splits, 0.48-0.50 in ~35 %
    x = np.r_[rng.normal(0.29, 0.012, 163), rng.normal(0.488, 0.008, 87)]
    assert T.em_two_gaussians(x)["bimodal"]
    # a harmless 3 % basin split is not "bimodal" (separation below 5 % of the median)
    x = np.r_[rng.normal(1.0, 0.004, 200), rng.normal(1.03, 0.004, 50)]
    assert not T.em_two_gaussians(x)["bimodal"]


def test_one_sided_threshold():
    from scipy.stats import norm
    assert T.t_one_sided(1214) == pytest.approx(norm.isf(1 - 0.99 ** (1 / 1214)), rel=1e-12)
    assert 4.25 < T.t_one_sided(1214) < 4.35 < T.t_family(1214)


def test_clip_estimates_are_scale_equivariant():
    rng = np.random.default_rng(5)
    x = np.r_[rng.normal(0, 0.15, 640), rng.normal(5.6, 0.8, 350)]
    for s in (1 / 1.2, 1.5, 3.7):
        a, b = SA.core_w(x), SA.core_w(x / s)
        assert b["mean"] == pytest.approx(a["mean"] / s, rel=1e-6, abs=1e-9)
        assert b["rms0"] == pytest.approx(a["rms0"] / s, rel=1e-6)


@pytest.mark.parametrize("bimodal", [False, True])
def test_exact_null_holds_with_a_fitted_centre(bimodal):
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
        e = SA.bin_estimates(f)
        d = np.abs(t - e["centre"])
        for lv in SA.LEVELS:
            m = np.sum(d > e["q"][f"{lv:g}"])
            e0 = SA.expected_coverage(nf, lv)
            zs[lv].append(float(SA.coverage_z(m, nt, e0, SA.exchangeable_coverage_variance(nf, nt, lv))))
    for lv, z in zs.items():
        z = np.asarray(z)
        assert abs(z.mean()) < 0.08 and abs(z.std() - 1) < 0.08, (lv, z.mean(), z.std())


def test_rank_n_plus_1_reaches_the_level_and_is_exact():
    """Recipe option (a): k = min(ceil(L (n + 1)), n) gives E[coverage] >= L whenever reachable, and the
    coverage null is the beta-binomial with that k."""
    from scipy import stats
    for n in (200, 300, 332, 333, 744, 5000):
        for lv in SA.LEVELS:
            k = SA.quantile_order(n, lv, "n+1")
            reachable = np.ceil(lv * (n + 1) - 1e-9) <= n
            assert (SA.expected_coverage(n, lv, "n+1") >= lv - 1e-12) == reachable
            assert SA.expected_coverage(n, lv, "n+1") >= SA.expected_coverage(n, lv, "n")
    assert not np.ceil(0.997 * 333 - 1e-9) <= 332 and np.ceil(0.997 * 334 - 1e-9) <= 333
    nf, nt, lv = 744, 2232, 0.99
    k = SA.quantile_order(nf, lv, "n+1")
    d = stats.betabinom(nt, nf - k + 1, k)
    e0 = SA.expected_coverage(nf, lv, "n+1")
    v = SA.exchangeable_coverage_variance(nf, nt, lv, "n+1")
    up = d.sf(20) + 0.5 * d.pmf(20)
    assert SA.coverage_z(20, nt, e0, v) == pytest.approx(-stats.norm.isf(up) if up < 0.5 else stats.norm.isf(1 - up),
                                                         rel=1e-9, abs=1e-9)


def test_missing_row_is_not_uncovered():
    """Amendment 1c (N1): test clusters looked up in a row without an estimate are left out of the
    coverage; clusters with no row at all count as outside."""
    T.G["rank"] = "n"
    rng = np.random.default_rng(9)
    p = rng.standard_normal(300)
    K = {"trows": [{"surface": 1}], "rows": [{}, {}], "ts": np.array([0]), "te": np.array([300]),
         "test_lrow": np.r_[np.zeros(100, int), np.ones(100, int), -np.ones(100, int)],
         "test_p": {"alpha": p, "beta": p}}
    par = np.full((3, 2, 2 + len(SA.LEVELS)), np.nan)
    par[0, :, 0], par[0, :, 1], par[0, :, 2:] = 0.0, 1.0, 10.0          # row 0: covers everything
    st = T._test_stats(K, par, None)                                     # row 1: no estimate; -1: no row
    assert st[0, 0, 2] == 200                                            # 100 (row 0) + 100 (no row); row 1 missing
    assert st[0, 0, 3] == pytest.approx(0.5)                             # the no-row clusters are outside
