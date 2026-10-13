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
