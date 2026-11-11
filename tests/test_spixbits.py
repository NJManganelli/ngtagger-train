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
    tools = SimpleNamespace(linktable=SimpleNamespace(lookup=lambda K, ci, a, b: (np.zeros((len(ci), 2)),
                                                                              np.full((len(ci), 2), 0.5))))
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
    mean, sig = EM.LinkLookup(LT, "net", None, 4).lookup(K, np.array([0, 1]), np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    # code alpha = SmartPixels beta: link 0 has sign -1 -> pool-frame cot -0.5 -> first interval, mean back in the link frame
    assert mean[0, 0] == pytest.approx(-0.1) and sig[0, 0] == pytest.approx(np.hypot(0.2, 0.3))
    assert mean[1, 0] == pytest.approx(0.3) and sig[1, 1] == pytest.approx(0.4)      # alpha float: no quantization term
