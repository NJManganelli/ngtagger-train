"""Unit tests of the SmartPixels output-bits study's new functions (eval_refitq/spixbits) and replay switches."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "eval_refitq"
for p in ("spixbits", "replay", "combinatorics"):
    sys.path.insert(0, str(ROOT / p))

EM = pytest.importorskip("emulate")
import registry as REG          # noqa: E402
import run_scan as RS           # noqa: E402


def test_hash_is_deterministic_and_position_sensitive():
    n = 1000
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=n).astype(np.float32), rng.normal(size=n).astype(np.float32)
    args = (np.full(n, 1), np.full(n, 7), np.arange(n), np.full(n, 303042565), x, y)
    h1, h2 = EM.cluster_hash(*args), EM.cluster_hash(*args)
    assert np.array_equal(h1, h2) and len(np.unique(h1)) == n
    h3 = EM.cluster_hash(*args[:4], np.nextafter(x, np.float32(9)), y)
    assert (h3 != h1).all()
    u = EM.uniform(h1, 11)
    assert 0 <= u.min() and u.max() < 1 and abs(u.mean() - 0.5) < 0.05
    assert abs(np.corrcoef(u, EM.uniform(h1, 12))[0, 1]) < 0.1
    g = EM.normal(EM.cluster_hash(np.zeros(50000), np.zeros(50000), np.arange(50000), np.zeros(50000),
                                  np.zeros(50000, np.float32), np.zeros(50000, np.float32)), 15, 16)
    assert abs(g.std() - 1) < 0.02 and abs(g.mean()) < 0.02


@pytest.mark.parametrize("step", [0.06, 0.015, 0.00094])
def test_position_quantizer_is_uniform(step):
    rng = np.random.default_rng(1)
    x, ph = rng.uniform(-0.8, 0.8, 100000), rng.random(100000)
    q = EM.quantize_position(x, -0.84, step, ph)
    assert np.abs(q - x).max() <= step / 2 * (1 + 1e-9)
    assert abs(np.sqrt(np.mean((q - x) ** 2)) / step / 0.288675 - 1) < 0.02
    comb = np.round(x / 0.0025) * 0.0025                       # positions on a pixel comb: still uniform
    qc = EM.quantize_position(comb, -0.84, step, ph)
    assert abs(np.sqrt(np.mean((qc - comb) ** 2)) / step / 0.288675 - 1) < 0.02


def test_angle_quantizer_codes_and_flags():
    v = np.array([-2.0, -1.0, -0.999, 0.0, 0.999, 1.0, 5.0])
    q, f = EM.quantize_angle(v, np.full(7, -1.0), np.full(7, 1.0), 2)          # 2 value codes of width 1
    assert f.tolist() == [-1, 0, 0, 0, 0, 1, 1]
    assert q.tolist() == [-1.0, -0.5, -0.5, 0.5, 0.5, 1.0, 1.0]
    q, f = EM.quantize_angle(np.linspace(-0.99, 0.99, 1000), np.full(1000, -1.0), np.full(1000, 1.0), 4)
    assert len(np.unique(q)) == 14 and (f == 0).all()


def test_registry_names_both_frames():
    doc = REG.build()
    names = [a["name"] for a in doc["arms"]]
    assert len(names) == len(set(names)) and "implemented_now" in names
    by = {a["name"]: a for a in doc["arms"]}
    a = by["alloc_6604"]
    assert a["spec"]["send"] == ["beta"] and a["spec"]["step_x_um"] == 37.5 and a["spec"]["bits_beta"] == 4
    assert "alpha 0" in a["smartpixels_frame"] and "code beta 0" in a["code_frame"] and "code alpha 4b" in a["code_frame"]
    assert by["alloc_4444_wholesensor"]["spec"]["step_y_um"] == 600.0
    pri = [a["name"] for a in doc["arms"] if a["group"] == "priority"]
    assert len(pri) == 8


def test_units_priority_first_and_stable():
    reg = REG.build()
    u = RS.units_of(reg, (5, 7), ("AAAA", "IAAA"))
    assert [x["group"] for x in u[:4]] == ["priority"] * 4
    assert len({x["name"] for x in u}) == len(u)
    assert sorted(a for x in u if x["file"] == 5 and x["scen"] == "AAAA" for a in x["arms"]) == \
        sorted(a["name"] for a in reg["arms"])
    assert all(len(x["arms"]) <= RS.ARMS_PER_UNIT for x in u)


def test_build_arm_frames_and_defaults():
    AR = pytest.importorskip("arms")
    cfg = {"useAngles": "alphaBeta"}
    by = {a["name"]: a["spec"] for a in REG.build()["arms"]}
    now = RS.build_arm(AR, cfg, "implemented_now", by["implemented_now"])
    assert now == AR.v3(cfg, name="implemented_now")                     # every switch at today's value
    b = RS.build_arm(AR, cfg, "alloc_6604", by["alloc_6604"])
    assert b.uses("rank", "alpha", "barrel") and not b.uses("rank", "beta", "barrel")   # SmartPixels beta = code alpha
    assert b.dims_key("rank") == "xya" and b.flag_mode == "onesided" and b.pos_err == "rtables_sizeblind"
    assert b.pos_var_add == pytest.approx(((37.5e-4) ** 2 / 12,) * 2)
    y = RS.build_arm(AR, cfg, "down_y_4b", by["down_y_4b"])
    assert y.pos_var_add[0] == pytest.approx((150e-4) ** 2 / 12) and y.pos_var_add[1] == 0.0    # SmartPixels y = localX
    assert y.flag_mode == "off"
    assert not RS.build_arm(AR, cfg, "c", by["alloc_4444_v3window"]).window_var_add


def test_flag_cost_is_one_sided():
    AR = pytest.importorskip("arms")
    n = 4
    K = {"sb_flagAlpha": np.array([1, 1, -1, 0], np.int8), "sb_flagBeta": np.zeros(n, np.int8),
         "sb_loAlpha": np.full(n, -1.0), "sb_hiAlpha": np.full(n, 1.0), "sb_loBeta": np.full(n, -1.0),
         "sb_hiBeta": np.full(n, 1.0)}
    tools = SimpleNamespace(linktable=SimpleNamespace(lookup=lambda K, ci, a, b, with_qvar=True: (
        np.zeros((len(ci), 2)), np.full((len(ci), 2), 0.5 if not with_qvar else 99.0))))
    geo = {"surface": np.array([1])}
    arm = AR.Arm("t", flag_mode="onesided", ang_err="linktable")
    pred = np.array([0.0, 1.5, 0.0, 0.0])                # overflow: inside by 1 -> (1 / 0.5)^2; beyond the edge -> 0
    cost, nfl = AR.flag_cost(arm, "rank", {"K": K}, geo, np.arange(n), np.zeros(n, int), pred, np.zeros(n), tools)
    assert cost.tolist() == [4.0, 0.0, 4.0, 0.0] and nfl.tolist() == [1, 1, 1, 0]
    arm2 = AR.Arm("t", flag_mode="onesided", ang_err="linktable", use=AR.use_set(("rank", "update"), ("beta",)))
    cost2, nfl2 = AR.flag_cost(arm2, "rank", {"K": K}, geo, np.arange(n), np.zeros(n, int), pred, np.zeros(n), tools)
    assert not cost2.any() and not nfl2.any()            # an angle the arm does not use costs nothing


def test_sizeblind_R_lookup_and_not_dominated_by_rare_cells():
    AR = pytest.importorskip("arms")
    sb = json.load(open(ROOT / "spixbits" / "sizeblind_R_v3.json"))
    tools = SimpleNamespace(sizeblind=sb)
    r = AR.sizeblind_R(tools, np.array([0.15, -0.15, 5.0]), np.array([0.1, 2.0, -0.1]))
    assert r[0, 0] == r[1, 0] == sb["u"]["width_cm"][2] and r[2, 0] == sb["u"]["width_cm"][4]
    assert r[0, 1] == r[2, 1] == sb["v"]["width_cm"][0] and r[1, 1] == sb["v"]["width_cm"][4]
    cal = ROOT / "combinatorics" / "results" / "refit_calibration" / "refit_calibration_v3.json"
    if not cal.exists():
        pytest.skip("v3 calibration file not in the tree")
    R = json.load(open(cal))["R_tables"]
    for nm in "uv":                       # between the smallest cell width and the count-weighted 84 % cell width:
        W, N = np.array(R[nm]["width_cm"]), np.array(R[nm]["n"], float)      # a 1-2 % cell cannot dominate it
        for j, w in enumerate(sb[nm]["width_cm"]):
            o = np.argsort(W[:, j])
            cum = np.cumsum(N[o, j]) / N[:, j].sum()
            hi = W[o, j][np.searchsorted(cum, 0.84)]
            lo = W[N[:, j] > 0, j].min()
            assert lo <= w <= hi * 1.05, (nm, j, lo, w, hi)
        assert sb[nm]["n"] == N.sum(0).astype(int).tolist()
    assert max(sb["u"]["width_cm"]) < 8e-4                                   # r-phi: 2-7 um, never 11-17 um


def test_units_mix_scenarios_per_worker():
    reg = REG.build()
    u = RS.units_of(reg, (5, 7, 8, 9, 10), ("AAAA", "AAAI", "IAAA"))
    assert [x["group"] for x in u[:15]] == ["priority"] * 15
    for i in range(3):
        mine = [x["scen"] for x in u[i::3]]
        counts = [mine.count(s) for s in ("AAAA", "AAAI", "IAAA")]
        assert max(counts) - min(counts) <= 2, counts


def test_link_lookup_signs_and_quantization_variance(tmp_path):
    t = {"edges": [-9, 0, 9], "mean": [0.1, 0.3], "width": [0.2, 0.4], "n": [500, 500]}
    pool = {"range": {"A": [-1, 1], "B": [-1, 1]}, "table": {"net": {"A": t, "B": t}, "today": {"A": t, "B": t}},
            "qvar": {k: {a: {str(b): 0.09 for b in range(2, 8)} for a in "AB"} for k in ("link", "global")}}
    doc = {"links": {"10110": {"pool": 1, "sA": 1.0, "sB": -1.0}}, "pools": [pool, pool],
           "global_range": {"A": [-2, 2], "B": [-1, 1]}}
    f = tmp_path / "lt.json"
    f.write_text(json.dumps(doc))
    LT = EM.LinkTables(f)
    pool_, sA, sB = LT.link_index(np.array([10110, 999]))
    assert pool_.tolist() == [1, 0] and sB.tolist() == [-1.0, 1.0]
    K = {"sb_pool": pool_, "sb_sA": sA, "sb_sB": sB}
    look = EM.LinkLookup(LT, "net", None, 4)
    mean, sig = look.lookup(K, np.array([0, 1]), np.array([0.5, -0.5]), np.array([0.5, 0.5]))
    # the tables are in the link's own frame (no sign flip); code alpha = SmartPixels beta carries the 4-bit variance
    assert mean[0, 0] == pytest.approx(0.3) and sig[0, 0] == pytest.approx(np.hypot(0.4, 0.3))
    assert mean[1, 0] == pytest.approx(0.1) and sig[1, 1] == pytest.approx(0.4)      # alpha float: no quantization term
    _, w = look.lookup(K, np.array([0]), np.array([0.5]), np.array([0.5]), with_qvar=False)
    assert w[0, 0] == pytest.approx(0.4)                 # the one-sided cost's sigma: the table width alone


def test_collection_switch_renames_only_the_track_tables():
    CM = pytest.importorskip("common")
    assert CM.COLLECTION == "prompt" and CM.file_branch("L1TTrack_pt") == "L1TTrack_pt"
    CM.set_collection("extended")
    try:
        assert CM.file_branch("L1TTrack_pt") == "L1TExtTrack_pt"
        assert CM.file_branch("L1TTrackStub_trackIdx") == "L1TExtTrackStub_trackIdx"
        assert CM.file_branch("L1TSmartPixelsTrack_cov_d0_d0") == "L1TSmartPixelsExtTrack_cov_d0_d0"
        assert CM.file_branch("L1TSmartPixelsRefitHit_hitAccepted") == "L1TSmartPixelsExtRefitHit_hitAccepted"
        assert CM.file_branch("L1TSmartPixelsCluster_localX") == "L1TSmartPixelsCluster_localX"
        assert CM.file_branch("L1TTP_pt") == "L1TTP_pt" and CM.file_branch("run") == "run"
    finally:
        CM.set_collection("prompt")
    with pytest.raises(ValueError):
        CM.set_collection("other")


def test_heads_reader_strict_split_and_mantissa_rounding(tmp_path):
    HD = pytest.importorskip("heads")
    tree = {"feature": [0, -2, -2], "threshold": [1.0, 0.0, 0.0], "value": [0.0, -1.0, 2.0], "children_left": [1, -1, -1],
            "children_right": [2, -1, -1]}
    model = {"splitting_convention": "<", "n_features": 38, "init_predict": [0.5], "max_depth": 1, "trees": [[tree]]}
    for w in HD.WINDOWS:
        d = tmp_path / "SmartPixels" / "AAAA" / w
        d.mkdir(parents=True)
        for h in HD.HEADS:
            (d / f"{h}.json").write_text(json.dumps(model))
    H = HD.Heads(tmp_path, "AAAA")
    X = np.zeros((3, 38), np.float32)
    X[:, 0] = [0.5, 1.0, -999.0]                       # x < 1 goes left; a value ON the threshold goes right
    m = H.margins(X, np.array([0.0, 0.03, 0.0]))
    assert m[:, 0].tolist() == [-0.5, 2.5, -0.5] and m.shape == (3, 6)
    r = HD.round_mantissa(np.array([1.0, 1.0 + 2 ** -13, 1.0 + 2 ** -12, 0.0, 3.14159], np.float32))
    assert r[0] == 1.0 and r[2] == 1.0 + 2 ** -12 and r[3] == 0.0 and abs(r[4] / 3.14159 - 1) < 2 ** -12
    assert r[1] in (1.0, 1.0 + 2 ** -12)


def test_parametrization_pieces():
    PM = pytest.importorskip("param_model")
    q = np.linspace(-1, 1, len(PM.KNOTS))
    u = np.array([0.0, 0.01, 0.5, 0.99, 0.999])
    v = PM.eval_inv_cdf(np.tile(q, (5, 1)), u)
    assert v[0] == v[1] == -1 and v[3] == v[4] == 1 and abs(v[2]) < 1e-12         # clamped ends, linear between
    D1 = PM.draws(np.ones(5), np.ones(5), np.arange(5), np.zeros(5))
    D2 = PM.draws(np.ones(5), np.ones(5), np.arange(5), np.zeros(5))
    assert all(np.array_equal(D1[k], D2[k]) for k in D1) and len(np.unique(D1["u_rank"])) == 5
    assert set(D1) == {"u_state", "u_rank", "u_miss", "n_cov", "n_score", "n_z0", "n_tanL", "n_rInv", "n_phi0"}
    assert D1["n_cov"].shape == (5, 5) and D1["n_score"].shape == D1["u_miss"].shape == (5, len(PM.SCORES))
    C = {"ieta": np.array([0, 0, 1, 1, 2] * 100), "window": np.array([0, 1] * 250)}
    x = np.arange(500.0)
    tab, level = PM.pooled_table(C, np.ones(500, bool), lambda m: float(x[m].mean()), 150, PM.KERNEL_AXES, ("window", "ieta"))
    assert tab.shape == (3, 2) and level[0, 0] == 1 and tab[0, 0] == tab[0, 1]       # thin cell: window pooled first
    assert PM.unnatural("spixChi2IncXTot", PM.natural("spixChi2IncXTot", np.array([-999.0, 2.0])))[0] == -999.0
    assert PM.unnatural("spixChi2IncXTot", PM.natural("spixChi2IncXTot", np.array([2.0])))[0] == pytest.approx(2.0)
    # the heads' inputs rebuilt from emulated parameters: a track whose refit does not run carries the OT track
    today = {"ot_a": np.array([[1e-3, 0.1, 0.5, 1.0, 0.0]] * 2), "ot_inputs": np.full((2, 13), 0.25, np.float32),
             "inputs": np.zeros((2, 25))}
    a = np.array([[2e-3, 0.1, 0.6, 1.5, 0.01]] * 2)
    counts = {c: np.array([1.0, 0.0]) if c == "spixRefitPerformed" else np.array([3.0, 0.0]) for c in PM.COUNTS}
    values = {c: np.array([2.4, PM.MISSING]) for c in PM.SCORES}
    X, aa = PM.assemble_inputs(today, a, counts, values, kpt=0.01)
    assert X.shape == (2, 25) and np.array_equal(aa[1], today["ot_a"][1]) and np.array_equal(aa[0], a[0])
    assert X[0, 0] == pytest.approx(0.2) and X[1, 0] == pytest.approx(0.25) and X[1, 4] == 0.0       # 1/pT and its change
    assert X[0, 2] == pytest.approx(0.5) and X[0, 3] == pytest.approx(0.1) and X[1, 2] == 0.0
    assert X[0, PM.IN["h_nangle"]] == 2.0 and X[1, PM.IN["h_nangle"]] == 0.0 and X[0, 14] == PM.MISSING
    assert PM.mask_bits([5]).tolist() == [[1, 0, 1] + [0] * 13]
    # hit-choice donors: among the nearest pool tracks of the track's own class, picked by the draw
    P = {"cls": np.array([0, 0, 0, 0, 1, 1, 1, 1] * 10), "feat": np.arange(80, dtype=np.float32)[:, None], "scale": np.ones(1, np.float32)}
    pick = PM.nearest_donors(P, np.array([0, 1, 1]), np.array([[1.2], [1.2], [77.0]]), np.array([0.0, 0.0, 0.99]))
    assert pick[0] == 1 and pick[1] == 4 and P["cls"][pick].tolist() == [0, 1, 1] and pick[2] in (76, 77, 78, 79) and pick[2] != 77


def test_tree_maps_fit_draw_and_shipped_arrays():
    PM = pytest.importorskip("param_model")
    SA = pytest.importorskip("spixbits_apply")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    n = 4000
    Z = rng.normal(size=(n, 3))
    Y = np.column_stack([2 * (Z[:, 0] > 0) + 0.1 * rng.normal(size=n), Z[:, 1] + 0.5 * np.abs(Z[:, 2]) * rng.normal(size=n)])
    Y[rng.random(n) < 0.1, 1] = np.nan                                        # missing values are left out of their column
    B = PM.fit_block(Z, Y, np.ones(n, bool), rng.integers(0, 2, n), [False, True])
    got = PM.draw_block(B, Z, rng.normal(size=(n, 2)))
    assert abs(np.std(got[:, 0] - 2 * (Z[:, 0] > 0)) - 0.1) < 0.03            # the residual's size is reproduced ...
    wide, narrow = np.abs(Z[:, 2]) > 1, np.abs(Z[:, 2]) < 0.3
    assert np.std(got[wide, 1] - Z[wide, 1]) > 2 * np.std(got[narrow, 1] - Z[narrow, 1])      # ... and follows the track
    assert np.allclose(B["chol"] @ B["chol"].T, np.eye(2), atol=0.1)          # independent residuals: no correlation
    # the shipped arrays, read by the helper, give the fitted model
    t = B["cols"][0]["mean"][0]
    base, n_nodes, feat, thr, left, right, val = t.arrays()
    m = {"base": np.array([base]), "tree_start": np.array([0, len(n_nodes)]), "node_start": np.cumsum(np.r_[0, n_nodes]),
         "feature": feat, "threshold": thr, "left": left, "right": right, "value": val}
    assert np.allclose(SA._raw(m, 0, Z), t.raw(Z), atol=1e-5)
    const = PM.Trees(const=1.5)
    assert const.raw(Z[:3]).tolist() == [1.5] * 3 and len(const.arrays()[1]) == 0


def test_helper_reads_head_models(tmp_path):
    SA = pytest.importorskip("spixbits_apply")
    tree = {"feature": [0, -2, -2], "threshold": [0.5, 0.0, 0.0], "value": [0.0, -1.0, 2.0], "children_left": [1, -1, -1],
            "children_right": [2, -1, -1]}
    for w in SA.WINDOWS:
        d = tmp_path / "SmartPixels" / "AAAA" / w
        d.mkdir(parents=True)
        for h in SA.HEADS:
            (d / f"{h}.json").write_text(json.dumps({"splitting_convention": "<", "n_features": 38, "max_depth": 1,
                                                      "init_predict": [0.5], "trees": [[tree], [tree]]}))
    H = SA.load_heads(tmp_path, "AAAA")
    X = np.zeros((3, 38), np.float32)
    X[:, 0] = [0.0, 0.5, -999.0]                              # strict "<": 0.5 goes right; missing (-999) goes left
    m = SA.head_margins(H, X, np.array([0.0, 0.0, 0.05]))
    assert m.shape == (3, 6) and m[:, 0].tolist() == [-1.5, 4.5, -1.5]


def test_helper_full_covariance_keeps_correlations():
    SA = pytest.importorskip("spixbits_apply")
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 5, 5))
    C = A @ np.transpose(A, (0, 2, 1)) + 0.1 * np.eye(5)
    new = np.einsum("nii->ni", C) * rng.uniform(0.5, 2.0, (4, 5))
    C2 = SA.full_covariance(C, new)
    assert np.allclose(np.einsum("nii->ni", C2), new) and (np.linalg.eigvalsh(C2) > 0).all()
    d, d2 = np.sqrt(np.einsum("nii->ni", C)), np.sqrt(new)
    assert np.allclose(C2 / (d2[:, :, None] * d2[:, None, :]), C / (d[:, :, None] * d[:, None, :]))
    assert SA.to_p_clean(SA.to_margin(0.9)) == pytest.approx(0.9)


def test_pt_from_bending_angle_on_ideal_modules():
    # a d0 = 0 track of known pT crossing a barrel module (both facings, tilted) and a disc module (both facings)
    k = EM.KPT_GEV_CM
    for pt_true, q in ((2.0, 1), (5.0, -1), (30.0, 1)):
        for r, z in ((3.0, 4.0), (11.0, -20.0)):
            R = pt_true / k
            psi = q * np.arcsin(r / (2 * R))
            phi = 0.7
            rh, ph = np.array([np.cos(phi), np.sin(phi), 0.0]), np.array([-np.sin(phi), np.cos(phi), 0.0])
            d = rh * np.cos(psi) + ph * np.sin(psi) + np.array([0, 0, z / r])
            for tilt in (0.0, 0.2):
                for facing in (1.0, -1.0):
                    ezb = facing * (rh * np.cos(tilt) + ph * np.sin(tilt))            # barrel: normal about radial
                    exb = facing * (-rh * np.sin(tilt) + ph * np.cos(tilt))
                    for ex, ez in ((exb, ezb), (ph * facing, np.array([0, 0, facing]))):   # barrel module, disc module
                        cot = (d @ ex) / (d @ ez)
                        got = EM.pt_from_bending_angle(np.array([cot]), np.array([phi]), np.array([r]), np.array([z]),
                                                       ex[None, :], ez[None, :])[0]
                        assert got == pytest.approx(pt_true, rel=1e-6), (pt_true, q, r, tilt, facing)
    assert EM.pt_from_bending_angle(np.array([np.nan]), np.array([0.0]), np.array([3.0]), np.array([0.0]),
                                    np.array([[0, 1.0, 0]]), np.array([[1.0, 0, 0]]))[0] == 0.0
