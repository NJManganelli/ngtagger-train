"""Unit tests of the shared angle module (ngtagger.spdefs.angles; plan C.2 P8): identity at the
neutral setting, non-null, a discriminating toy, and only-its-own-quantity, each with the
known-bad it must catch. The data-level checks are ngtagger.spdefs.checks."""
import numpy as np
import pytest

from ngtagger.spdefs import angles as SA


def _toy_cols(n=20000, seed=7, disc=False):
    rng = np.random.default_rng(seed)
    surf = np.full(n, 11 if disc else 1)
    cot_t = rng.uniform(-3, 3, n) if not disc else rng.choice([-1, 1], n) * rng.uniform(1.3, 9, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    half_t = rng.normal(0, 0.05, n)
    sig_a, sig_c = np.full(n, 0.07), np.full(n, 0.2)
    lam_t = np.arctan(cot_t)
    sig_l = sig_c / (1 + cot_t ** 2)
    # a signed centre: emitted cot pulled toward 0 (-0.3 sigma x sign), width 1.2
    lam_e = lam_t + sig_l * (-0.3 * np.sign(cot_t) + 1.2 * rng.normal(size=n))
    half_e = half_t + sig_a * rng.normal(size=n)
    return {"surface": surf, "globalPhi": phi, "globalClusterPhi": phi - half_e, "tpGlobalClusterPhi": phi - half_t,
            "globalClusterCotTheta": np.tan(lam_e), "tpGlobalClusterCotTheta": cot_t,
            "sigGlobalClusterPhi": sig_a, "sigGlobalClusterCotTheta": sig_l * (1 + np.tan(lam_e) ** 2),
            "tpEmuClass": rng.choice([0, 1], n, p=[0.3, 0.7])}


def test_signed_edges_and_fine_bins():
    se = SA.signed_edges(SA.BARREL_ABS_EDGES)
    assert len(se) == 23 and se[0] == -np.inf and se[-1] == np.inf and se[11] == 0
    assert SA.N_FINE[1] == 22 and SA.N_FINE[11] == 20
    fb = SA.fine_bin(np.array([1, 1, 11, 11]), np.array([-np.inf, np.inf, 0.05, -0.05]))
    assert fb.tolist() == [0, 21, 10, 9]


def test_merge_rule_keeps_the_plateau_edges():
    bounds = [SA.fine_bin_bounds(1, b) for b in range(22)]
    counts = np.full(22, 1000)
    counts[[0, 21]] = 50          # sparse ends merge inward
    counts[[10, 11]] = 150        # the two bins at 0 merge across 0, away from 0.5
    g = SA.merge_bins(bounds, counts)
    assert [0, 1] in g and [20, 21] in g and [10, 11] in g
    assert all(len(x) == 1 for x in g if x not in ([0, 1], [20, 21], [10, 11]))


def test_knob_identity_non_null_and_class_c():
    c = _toy_cols()
    T = SA.class_T(c["tpEmuClass"])
    for a in SA.ANGLES:
        e, t = SA.emitted(a, c), SA.truth(a, c)
        assert np.array_equal(SA.improve(a, e, t, T, 1.0)[0], e)                 # (i) identity
        a0, moved = SA.improve(a, e, t, T, 0.0)
        assert np.allclose(SA.residual(a, a0, t)[moved], 0, atol=1e-12)       # (ii) non-null
        assert np.array_equal(a0[~T], e[~T])                                    # class C untouched
        with pytest.raises(ValueError):
            SA.improve(a, e, t, T, 1.2)                                        # k > 1 refused


def test_sigma_modes_and_floor():
    s = np.array([0.1, 0.2])
    assert np.array_equal(SA.sigma_k(s, 1.0, "scaled"), s) and np.array_equal(SA.sigma_k(s, 0.3, "fixed"), s)
    assert np.allclose(SA.sigma_k(s, 0.3, "scaled"), 0.3 * s)
    assert np.allclose(SA.sigma_k(s, 0.0, "scaled"), SA.K_FLOOR * s)
    with pytest.raises(ValueError):
        SA.sigma_k(s, 0.5, "other")


def test_knob_changes_only_its_own_angle():
    c = _toy_cols()
    T = SA.class_T(c["tpEmuClass"])
    ra = SA.knob("alpha", c, T, 0.3, "scaled")
    rb = SA.knob("beta", c, T, 1.0, "fixed")
    assert np.array_equal(rb["a"], SA.emitted("beta", c))     # beta untouched when only alpha moves
    assert not np.array_equal(ra["a"], SA.emitted("alpha", c))


def _table_from_toy(c, key="true"):
    """A one-surface table fitted on the toy, as the study does it."""
    T = SA.class_T(c["tpEmuClass"])
    cot = c["tpGlobalClusterCotTheta"] if key == "true" else c["globalClusterCotTheta"]
    kv = SA.incidence_key(c["surface"], cot)
    fb = SA.fine_bin(c["surface"], kv)
    s = int(c["surface"][0])
    pulls = {a: SA.residual(a, SA.emitted(a, c), SA.truth(a, c)) / SA.sigma_of(a, c) for a in SA.ANGLES}
    bounds = [SA.fine_bin_bounds(s, b) for b in range(SA.N_FINE[s])]
    rows = []
    for grp in SA.merge_bins(bounds, np.bincount(fb[T], minlength=SA.N_FINE[s])):
        m = T & np.isin(fb, grp)
        rows.append({"surface": s, "facing": 0, "lo": bounds[grp[0]][0], "hi": bounds[grp[-1]][1], "fine": grp,
                     "n_fit": int(m.sum()), **{a: SA.bin_estimates(pulls[a][m]) for a in SA.ANGLES}})
    doc = SA.new_table_doc({key: rows}, (np.array([], np.uint32), np.array([], np.int8)), {"toy": True}, "toy", "toy")
    return SA.AngleTable(doc), pulls, kv, T


def test_table_round_trip_and_discriminating_toy(tmp_path):
    c = _toy_cols()
    tab, pulls, kv, T = _table_from_toy(c)
    p = tmp_path / "t.json"
    tab.save(str(p))
    tab2 = SA.AngleTable.load(str(p))
    rows = tab2.rows_of("true", c["surface"], kv, np.zeros(len(kv), np.int8))
    assert np.array_equal(rows, tab.rows_of("true", c["surface"], kv, np.zeros(len(kv), np.int8)))
    after = (pulls["beta"] - tab2.get("true", "beta", "centre", rows)) / tab2.get("true", "beta", "width", rows)
    pos, neg = T & (kv > 0.5), T & (kv < -0.5)
    # (iii) the signed table removes the sign-dependent centre; a folded lookup (|key|) does not
    assert abs(np.median(after[pos])) < 0.05 and abs(np.median(after[neg])) < 0.05
    rows_f = tab2.rows_of("true", c["surface"], np.abs(kv), np.zeros(len(kv), np.int8))
    bad = (pulls["beta"] - tab2.get("true", "beta", "centre", rows_f)) / tab2.get("true", "beta", "width", rows_f)
    assert abs(np.median(bad[neg])) > 0.3
    # the quantile covers its level in-sample
    q99 = tab2.get("true", "beta", 0.99, rows)
    cen = tab2.get("true", "beta", "centre", rows)
    assert np.mean(np.abs(pulls["beta"][T] - cen[T]) <= q99[T]) == pytest.approx(0.99, abs=0.003)


def test_table_pull_centre_scales_with_k():
    r, s, c, w = np.array([0.0]), np.array([0.1]), np.array([0.4]), np.array([1.2])
    for mode in SA.K_MODES:
        assert SA.table_pull(r, s, SA.sigma_k(s, 0.0, mode), 0.0, c, w)[0] == 0.0
    # known-bad: the k = 1 centre at k = 0
    assert (r - c * s)[0] / (w * s)[0] != 0.0


def test_consumer_quantile_matches_numpy_inverted_cdf():
    rng = np.random.default_rng(3)
    x, b = rng.normal(size=5000), rng.integers(0, 3, 5000)
    q = SA.consumer_quantile(x, b, levels=(0.5, 0.99), n_bins=3)
    for i in range(3):
        assert np.allclose(q[i], np.quantile(x[b == i], (0.5, 0.99), method="inverted_cdf"))
    w = rng.integers(0, 3, 5000)
    qw = SA.consumer_quantile(x, b, levels=(0.9,), w=w, n_bins=3)
    for i in range(3):
        assert qw[i, 0] == pytest.approx(np.quantile(np.repeat(x[b == i], w[b == i]), 0.9, method="inverted_cdf"))
    # fixed mode calibrates at k = 1, scaled at k
    seen = []
    SA.consumer_quantile_at_k(lambda k: seen.append(k) or x, 0.3, "fixed", b, None)
    SA.consumer_quantile_at_k(lambda k: seen.append(k) or x, 0.3, "scaled", b, None)
    assert seen == [1.0, 0.3]


def test_local_conversion_signs():
    # barrel: cotBeta = s cot theta sqrt(1 + cotA^2); disc: cotBeta = s tan theta (no alpha)
    cot_t, ca = np.array([0.8, -1.5]), np.array([0.2, -0.1])
    for s in (1, -1):
        cb = s * cot_t * np.sqrt(1 + ca ** 2)
        assert np.allclose(SA.key_from_local(np.array([2, 2]), ca, cb, np.array([s, s])), cot_t)
        cbd = s * (1 / cot_t)
        assert np.allclose(SA.key_from_local(np.array([12, 12]), np.zeros(2), cbd, np.array([s, s])), 1 / cot_t)
    assert SA.to_local("beta", 0.3, np.int8(-1), 1) == -0.3 and SA.to_local("beta", 0.3, np.int8(-1), 11) == 0.3
    assert np.isnan(SA.to_local("alpha", 0.3, np.int8(1), 11)) and SA.to_local("alpha", 0.3, np.int8(-1), 2) == 0.3


# ---------------------------------------------------------------- coverage of a fitted quantile (amendment 1)
def test_quantile_order_and_expected_coverage_match_numpy():
    for n in (200, 201, 744, 5082):
        for lv in SA.LEVELS:
            k = SA.quantile_order(n, lv)
            assert k == np.quantile(np.arange(1, n + 1), lv, method="inverted_cdf")
            assert SA.expected_coverage(n, lv) == pytest.approx(k / (n + 1))
    # the inverted-CDF shortfall: up to 1 / (n + 1) below the level (200 values, 99 %: 198 / 201)
    assert SA.expected_coverage(200, 0.99) == pytest.approx(198 / 201)


def test_coverage_z_is_the_exact_order_statistic_null():
    """identity: with the exchangeable variance, coverage_z is the mid-p of beta-binomial(n_t, n_f - k + 1, k)."""
    from scipy import stats

    nf, nt, lv = 744, 2381, 0.997
    e0 = SA.expected_coverage(nf, lv)
    var = SA.exchangeable_coverage_variance(nf, nt, lv)
    k = SA.quantile_order(nf, lv)
    d = stats.betabinom(nt, nf - k + 1, k)
    for m in (0, 3, 10, 25):
        up = d.sf(m) + 0.5 * d.pmf(m)
        ref = stats.norm.isf(1 - up) if up > 0.5 else -stats.norm.isf(up)
        assert SA.coverage_z(m, nt, e0, var) == pytest.approx(ref, rel=1e-9, abs=1e-9)


def test_coverage_z_sign_inflation_and_calibration():
    nf, nt, lv = 1000, 3000, 0.99
    e0 = SA.expected_coverage(nf, lv)
    v0 = SA.exchangeable_coverage_variance(nf, nt, lv)
    mean_m = nt * (1 - e0)
    # sign: more exceedances than expected = under-coverage = negative z
    assert SA.coverage_z(mean_m * 3, nt, e0, v0) < -3 < 3 < SA.coverage_z(0, nt, e0, v0)
    # a larger (clustered) variance can only shrink |z|
    assert abs(SA.coverage_z(mean_m * 3, nt, e0, 3 * v0)) < abs(SA.coverage_z(mean_m * 3, nt, e0, v0))
    # discriminating toy: exchangeable values -> z ~ N(0, 1); the Gaussian z with the same sd is skewed
    rng = np.random.default_rng(11)
    z, zg = [], []
    for _ in range(3000):
        x, y = np.abs(rng.standard_normal(nf)), np.abs(rng.standard_normal(nt))
        m = np.sum(y > np.quantile(x, lv, method="inverted_cdf"))
        z.append(SA.coverage_z(m, nt, e0, v0))
        zg.append(((1 - m / nt) - e0) / np.sqrt(v0))
    z, zg = np.array(z), np.array(zg)
    assert abs(z.mean()) < 0.06 and abs(z.std() - 1) < 0.05
    from scipy.stats import skew
    assert abs(skew(z)) < 0.15 < abs(skew(zg))
