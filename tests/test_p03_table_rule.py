"""Known-answer tests of the P0.3 table recipe and test rule (eval_refitq/p03_angles/table.py; amendment
1e): the table entries (25 %-trimmed-mean centre, the 95 %-trimmed-mean weighting width / 0.7168,
quantiles at the rank ceil(L (n + 1))), the one-sided family-wise threshold, the exact coverage null with
a FITTED centre (stats review 01, F9) on a Gaussian and on a bimodal (64 / 35 %) toy, missing rows, the
after-table statistics, the known-bad perturbation hook and the materiality verdict."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_refitq" / "p03_angles"))
import table as T  # noqa: E402

from ngtagger.spdefs import angles as SA  # noqa: E402


def test_bin_estimates_centre_quantiles_and_weighting_width():
    from scipy.stats import norm, trim_mean

    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(0.3, 1.0, 900), rng.normal(6.0, 2.0, 101)]
    e = SA.bin_estimates(x)
    c = trim_mean(x, 0.25)
    assert e["centre"] == pytest.approx(c, rel=1e-12)
    d = np.sort(np.abs(x - c))
    for lv in SA.LEVELS:
        k = int(min(np.ceil(lv * (len(x) + 1) - 1e-9), len(x)))
        assert e["q"][f"{lv:g}"] == d[k - 1]
    t = 0.95 * len(x)
    m = int(np.floor(t))
    assert e["width"] == pytest.approx((d[:m].sum() + (t - m) * d[m]) / t / 0.7168, rel=1e-12)
    assert {"clip_centre", "core_width", "mad", "out3", "full_rms0"} <= set(e)
    assert SA.bin_estimates(x[:199]) is None
    # integer (bootstrap) weights repeat the values
    w = rng.integers(0, 3, len(x))
    assert SA.bin_estimates(x, w=w, diagnostics=False) == SA.bin_estimates(np.repeat(x, w), diagnostics=False)
    # the divisor is the unit-Gaussian value: the width of a large Gaussian sample is 1
    c95 = norm.isf(0.025)
    assert 2 * (norm.pdf(0) - norm.pdf(c95)) / 0.95 == pytest.approx(0.7168, abs=5e-5)
    g = rng.standard_normal(400000)
    assert SA.bin_estimates(3.0 * g + 1.0, diagnostics=False)["width"] == pytest.approx(3.0, rel=0.005)


def test_width_does_not_jump_where_q95_does():
    """Statistics review 02 R4-2: with a separate population of 5 % far away (the worst case: its edge at the
    95 % point), q95 jumps between fits by a factor ~15; the 95 % trimmed mean moves only by the far mass
    that crosses its cut-off times the distance (relative sd ~0.07 here)."""
    rng = np.random.default_rng(8)
    w, q = [], []
    for _ in range(40):
        n = 3500
        far = rng.random(n) < 0.05
        x = np.where(far, rng.normal(27.0, 2.0, n), rng.normal(0.0, 0.5, n))
        e = SA.bin_estimates(x, diagnostics=False)
        w.append(e["width"])
        q.append(e["q"]["0.95"])
    assert np.std(q) / np.median(q) > 0.5 and np.std(w) / np.median(w) < 0.15


def test_one_sided_threshold():
    from scipy.stats import norm
    assert T.t_one_sided(1214) == pytest.approx(norm.isf(1 - 0.99 ** (1 / 1214)), rel=1e-12)
    assert 4.25 < T.t_one_sided(1214) < 4.35 < T.t_family(1214)


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
        e = SA.bin_estimates(f, diagnostics=False)
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
    assert np.isnan(st[0, 0, :2]).all()                                  # < 200 clusters with an estimate: no centre


def test_after_table_statistics_use_the_table_estimators():
    rng = np.random.default_rng(12)
    p = np.r_[rng.normal(0.0, 1.0, 700), rng.normal(3.0, 0.5, 300)]
    K = {"trows": [{"surface": 1}], "rows": [{}], "ts": np.array([0]), "te": np.array([len(p)]),
         "test_lrow": np.zeros(len(p), int), "test_p": {"alpha": p, "beta": p}}
    par = np.full((2, 2, 2 + len(SA.LEVELS)), np.nan)
    par[0, :, 0], par[0, :, 1], par[0, :, 2:] = 0.2, 2.0, 50.0
    st = T._test_stats(K, par, None)
    a = (p - 0.2) / 2.0
    ca = SA.centre_of(a)
    assert st[0, 0, 0] == pytest.approx(ca, rel=1e-12)
    assert st[0, 0, 1] == pytest.approx(SA.weighting_width(np.abs(a - ca)), rel=1e-12)
    # a table fitted on the same values leaves after-table centre 0 and width 1
    e = SA.bin_estimates(p, diagnostics=False)
    par[0, :, 0], par[0, :, 1] = e["centre"], e["width"]
    st = T._test_stats(K, par, None)
    assert st[0, 0, 0] == pytest.approx(0.0, abs=1e-12) and st[0, 0, 1] == pytest.approx(1.0, rel=1e-12)


def test_known_bad_perturbation_hook():
    """KB-q95 multiplies q95 by 0.9, KB-w divides the width by 1.15, KB-c moves the centre by 0.5 widths, in
    the plateau-edge rows only (|key| range overlapping [0.50, 0.67))."""
    rng = np.random.default_rng(2)
    rows = [{"lo": 0.30, "hi": 0.40}, {"lo": 0.40, "hi": 0.55}, {"lo": -0.70, "hi": -0.60}, {"lo": 0.67, "hi": 0.80}]
    assert T.plateau_edge_rows(rows).tolist() == [False, True, True, False]
    x = rng.standard_normal(4 * 400)
    K = {"rows": rows, "fs": np.arange(4) * 400, "fe": np.arange(1, 5) * 400, "fit_p": {"alpha": x, "beta": x},
         "pert": None, "pert_rows": None}
    ref = T._fit_params(K, None)
    m = np.r_[T.plateau_edge_rows(rows), False]
    for pert, idx, f in (({"q95_scale": 0.9}, 2 + T.I95, 0.9), ({"width_scale": 1 / 1.15}, 1, 1 / 1.15)):
        got = T._fit_params({**K, "pert": pert, "pert_rows": m}, None)
        assert np.allclose(got[m, :, idx], f * ref[m, :, idx])
        others = np.delete(np.arange(2 + len(SA.LEVELS)), [idx])
        assert np.array_equal(got[m][:, :, others], ref[m][:, :, others], equal_nan=True)
        assert np.array_equal(got[~m], ref[~m], equal_nan=True)
    got = T._fit_params({**K, "pert": {"centre_shift": 0.5}, "pert_rows": m}, None)
    assert np.allclose(got[m, :, 0], ref[m, :, 0] + 0.5 * ref[m, :, 1])
    assert np.array_equal(got[m][:, :, 1:], ref[m][:, :, 1:]) and np.array_equal(got[~m], ref[~m], equal_nan=True)


def test_materiality_verdict():
    """Amendment 1e: a (iii) cell failing at 95-99 % or failing (i) / (ii) bins holding > 2 % of a surface's
    test clusters or a false-veto rate above 5.7 % is MATERIAL; anything else is MINOR."""
    K = {"trows": [{"surface": 1}, {"surface": 1}, {"surface": 11}], "test_n": np.array([985, 15, 500])}
    P = {"keys": {"emitted": K}}
    judged = {"emitted": np.array([True, True, True])}
    nl = len(SA.LEVELS)
    zk = {"emitted": np.zeros((3, 2, nl))}
    uk = {"emitted": np.zeros((3, 2))}

    def res(cells=(), chi=(), fv=None, ok=True):
        r = {"pass": ok, "iii": {"fail_cells": list(cells), "power_below_0p8": []}, "ii": {"chi2_fail": list(chi)}}
        if fv is not None:
            r["false_veto"] = {"fail_i_ii_rate": fv}
        return r
    assert T.materiality(P, res(fv=0.02), zk, uk, judged, 4.0, 4.7)["verdict"] == "pass"
    assert T.materiality(P, res(["emitted/barrel/beta/0.997"], ok=False), zk, uk, judged, 4.0, 4.7)["verdict"] == "minor"
    assert T.materiality(P, res(["emitted/barrel/beta/0.99"], ok=False), zk, uk, judged, 4.0, 4.7)["verdict"] == "material"
    assert T.materiality(P, res(chi=["true/0.95"], ok=False), zk, uk, judged, 4.0, 4.7)["verdict"] == "minor"
    assert T.materiality(P, res(fv=0.06), zk, uk, judged, 4.0, 4.7)["verdict"] == "material"
    u = {"emitted": np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 0.0]])}        # bin 1: 15 of 1000 IL1 clusters
    assert T.materiality(P, res(ok=False), zk, u, judged, 4.0, 4.7)["verdict"] == "minor"
    z = np.zeros((3, 2, nl))
    z[0, 1, 2] = -6.0                                                      # bin 0: 985 of 1000 IL1 clusters
    mt = T.materiality(P, res(ok=False), {"emitted": z}, uk, judged, 4.0, 4.7)
    assert mt["verdict"] == "material" and mt["failing_bin_shares"]["emitted/IL1"]["test_cluster_share"] == 0.985


def test_iii_skips_only_empty_cells():
    """Amendment 2 (R6): a (iii) cell with zero test clusters is skipped; a NaN in a populated cell fails."""
    C = {"k": np.full((2, 2, len(SA.LEVELS)), np.nan)}
    C["k"][1] = 0.999
    sd = {"k": np.zeros((2, 2, len(SA.LEVELS)))}
    n = {"k": np.array([[0, 0], [100, 100]])}
    assert T._iii_fail_cells(C, sd, T.GATED_LEVELS, n) == []
    n["k"][0, 1] = 5                                                         # populated, NaN: a failure
    assert T._iii_fail_cells(C, sd, T.GATED_LEVELS, n) == [f"k/barrel/beta/{lv:g}" for lv in T.GATED_LEVELS]


def test_module_frame_binning_kinds():
    """amendment 2: the module frame's canonical rows pool the facings (k_loc key), facing_split uses the
    geometric facing, global_rows the global key and facing."""
    C = {"surface": np.array([11, 11, 11, 11]), "facing": np.zeros(4, np.int8), "pooled_facings": True,
         "facing_geo": np.array([1, -1, 1, -1], np.int8), "facing_g": np.array([-1, 1, -1, 1], np.int8),
         "kv": {"true_local": np.array([0.05, 0.05, -0.35, 0.95])}, "kv_g": {"true_local": np.array([0.05, -0.05, -0.35, -0.95])}}
    fac, fb = T.fine_index("canonical", C, "true_local")
    assert fac.tolist() == [0, 0, 0, 0] and T.facings("canonical", 11, C) == (0,)
    assert fb.tolist() == SA.fine_bin(C["surface"], C["kv"]["true_local"]).tolist()
    fac, _ = T.fine_index("facing_split", C, "true_local")
    assert fac.tolist() == [1, -1, 1, -1] and T.facings("facing_split", 11, C) == (-1, 1)
    fac, fb = T.fine_index("global_rows", C, "true_local")
    assert fac.tolist() == [-1, 1, -1, 1] and fb.tolist() == SA.fine_bin(C["surface"], C["kv_g"]["true_local"]).tolist()
