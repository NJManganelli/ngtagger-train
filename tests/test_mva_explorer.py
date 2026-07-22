"""MVA-explorer tests: quantization round-trip, the generic correctionlib
ingester (every schema-v2 node type on tiny synthetic files), the structured
smear preset path, the tkquality/tagger table exporters (synthetic inputs
only — the big-nano integration is gated on file existence), the prediction
dump round-trip, and the JS compute core vs python references (macOS JXA)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

SITE_SRC = os.path.join(os.path.dirname(__file__), "..",
                        "src", "ngtagger", "viz", "mva_explorer", "site_src")


# ------------------------------------------------------------- quantization

def test_quantize_roundtrip():
    from ngtagger.viz.mva_explorer.quantize import (
        dequantize_log10_int16, quantize_log10_int16)

    rng = np.random.default_rng(7)
    v = 10.0 ** rng.uniform(-4, 4, size=1000)
    q = quantize_log10_int16(v)
    assert q.dtype == np.dtype("<i2")
    back = dequantize_log10_int16(q)
    # SCALE=3000 => half-step 1/6000 decade ~ 0.0384% relative error
    assert np.max(np.abs(back / v - 1)) < 4.0e-4


def test_choose_block_fallback():
    from ngtagger.viz.mva_explorer.quantize import choose_block

    b, tag = choose_block(np.array([0.5, 2.0, 30.0]))
    assert tag == "log10_i16" and len(b) == 3 * 2
    b, tag = choose_block(np.array([-0.5, 2.0]))
    assert tag == "f32" and len(b) == 2 * 4
    with pytest.raises(ValueError):
        choose_block(np.array([1.0, np.nan]))


def test_canonical_config_order():
    from ngtagger.viz.mva_explorer import canonical_config_order

    assert canonical_config_order() == [
        "0000", "1000", "0100", "0010", "0001",
        "1100", "1010", "1001", "0110", "0101", "0011",
        "1110", "1101", "1011", "0111", "1111"]


# --------------------------------------------------- correctionlib ingester

def _cset(corrections, compound=None):
    out = {"schema_version": 2, "corrections": corrections}
    if compound:
        out["compound_corrections"] = compound
    return out


def _inp(name, typ="real"):
    return {"name": name, "type": typ}


def _corr(name, inputs, data, output="w"):
    return {"name": name, "version": 1, "inputs": inputs,
            "output": {"name": output, "type": "real"}, "data": data}


def _synthetic_cset():
    """One correction per node type + a hashprng + a compound stack."""
    binning = {"nodetype": "binning", "input": "x",
               "edges": [0.0, 1.0, 2.0, 4.0], "flow": "clamp",
               "content": [1.5, 2.5, 3.5]}
    multibin = {"nodetype": "multibinning", "inputs": ["x", "y"],
                "edges": [[0.0, 1.0, 2.0], [0.0, 0.5, 1.0, 2.0]],
                "flow": "clamp",
                "content": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    category = {"nodetype": "category", "input": "layer",
                "content": [{"key": 1, "value": binning},
                            {"key": 2, "value": 2.0}]}
    formula = {"nodetype": "formula", "expression": "x*x + 1.0",
               "parser": "TFormula", "variables": ["x"]}
    formularef = {"nodetype": "formularef", "index": 0,
                  "parameters": [2.0, 0.5]}
    transform = {"nodetype": "transform", "input": "x",
                 "rule": {"nodetype": "formula", "expression": "abs(x)",
                          "parser": "TFormula", "variables": ["x"]},
                 "content": binning}
    prng = {"nodetype": "hashprng", "inputs": ["x", "y"],
            "distribution": "stdnormal"}
    corrections = [
        _corr("c_binning", [_inp("x")], binning),
        _corr("c_multibinning", [_inp("x"), _inp("y")], multibin),
        _corr("c_category", [_inp("layer", "int"), _inp("x")], category),
        _corr("c_formula", [_inp("x")], formula),
        dict(_corr("c_formularef", [_inp("x")], formularef),
             generic_formulas=[{"nodetype": "formula",
                                "expression": "[0]*x + [1]",
                                "parser": "TFormula", "variables": ["x"]}]),
        _corr("c_transform", [_inp("x")], transform),
        _corr("c_signed", [_inp("x")],
              {"nodetype": "binning", "input": "x",
               "edges": [0.0, 1.0, 2.0], "flow": "clamp",
               "content": [-1.0, 1.0]}),
        _corr("c_inactive_input", [_inp("x"), _inp("unused")], binning),
        _corr("c_prng", [_inp("x"), _inp("y")], prng),
        # envelope-shaped smear factorization: sigma/bias pair + stdnormal prng
        _corr("c_env_sigma", [_inp("x")], binning),
        _corr("c_env_bias", [_inp("x")],
              {"nodetype": "binning", "input": "x",
               "edges": [0.0, 1.0, 2.0, 4.0], "flow": "clamp",
               "content": [-0.1, 0.0, 0.1]}),
        _corr("c_flatprng", [_inp("x"), _inp("y")],
              {"nodetype": "hashprng", "inputs": ["x", "y"],
               "distribution": "stdflat"}),
    ]
    compound = [{
        # matches the envelope shape (sigma-like = c_binning, no *_bias partner)
        "name": "comp_with_prng",
        "inputs": [_inp("x"), _inp("y")],
        "output": {"name": "w", "type": "real"},
        "inputs_update": [], "input_op": "*", "output_op": "*",
        "stack": ["c_binning", "c_prng"],
    }, {
        # matches the envelope shape with a *_sigma/*_bias pair
        "name": "comp_env",
        "inputs": [_inp("x"), _inp("y")],
        "output": {"name": "w", "type": "real"},
        "inputs_update": [], "input_op": "*", "output_op": "*",
        "stack": ["c_env_sigma", "c_prng"],
    }, {
        # stdflat prng: NOT envelope-shaped -> stays on the skip path
        "name": "comp_with_flatprng",
        "inputs": [_inp("x"), _inp("y")],
        "output": {"name": "w", "type": "real"},
        "inputs_update": [], "input_op": "*", "output_op": "*",
        "stack": ["c_binning", "c_flatprng"],
    }]
    return _cset(corrections, compound)


@pytest.fixture
def synthetic_json(tmp_path):
    path = tmp_path / "synth.json"
    path.write_text(json.dumps(_synthetic_cset()))
    return str(path)


def test_introspection(synthetic_json):
    from ngtagger.viz.mva_explorer.correctionlib_ingest import (
        build_axes, has_hashprng, load_cset_json)

    cj = load_cset_json(synthetic_json)
    by = {c["name"]: c for c in cj["corrections"]}

    axes, fixed = build_axes(by["c_binning"], cj)
    assert [a["name"] for a in axes] == ["x"]
    assert axes[0]["edges"] == [0.0, 1.0, 2.0, 4.0]

    axes, _ = build_axes(by["c_multibinning"], cj)
    assert [a["name"] for a in axes] == ["x", "y"]
    assert axes[1]["edges"] == [0.0, 0.5, 1.0, 2.0]

    axes, _ = build_axes(by["c_category"], cj)
    assert axes[0] == {"name": "layer", "kind": "cat", "categories": [1, 2]}
    assert axes[1]["kind"] == "real"

    # formula-only input -> auto linspace, override respected
    axes, _ = build_axes(by["c_formula"], cj)
    assert axes[0]["auto_linspace"] is True
    axes, _ = build_axes(by["c_formula"], cj, {"x": (0.0, 2.0, 4)})
    assert axes[0]["edges"] == [0.0, 0.5, 1.0, 1.5, 2.0]

    # formularef resolves the referenced generic formula's variables
    axes, _ = build_axes(by["c_formularef"], cj)
    assert [a["name"] for a in axes] == ["x"]

    # transform marks the input
    axes, _ = build_axes(by["c_transform"], cj)
    assert axes[0]["transformed"] is True

    # declared-but-unused input is fixed out of the grid
    axes, fixed = build_axes(by["c_inactive_input"], cj)
    assert [a["name"] for a in axes] == ["x"] and fixed == {"unused": 0.0}

    assert has_hashprng(by["c_prng"])
    assert not has_hashprng(by["c_binning"])


def test_generic_export_matches_correctionlib(synthetic_json, tmp_path):
    correctionlib = pytest.importorskip("correctionlib")
    from ngtagger.viz.mva_explorer.correctionlib_ingest import export_generic_dataset
    from ngtagger.viz.mva_explorer.quantize import dequantize_log10_int16

    out = str(tmp_path / "site")
    meta = export_generic_dataset(synthetic_json, "reg_synth", "synthetic", out)

    names = [c["name"] for c in meta["corrections"]]
    assert "c_prng" not in names
    assert set(names) == {"c_binning", "c_multibinning", "c_category",
                          "c_formula", "c_formularef", "c_transform",
                          "c_signed", "c_inactive_input",
                          "c_env_sigma", "c_env_bias"}
    skipped = {s["name"] for s in meta["skipped"]}
    assert skipped == {"c_prng", "c_flatprng", "comp_with_flatprng"}
    # envelope-shaped compounds are recorded (not skipped), with bias association
    envs = {e["name"]: e for e in meta["envelopes"]}
    assert set(envs) == {"comp_with_prng", "comp_env"}
    assert envs["comp_with_prng"]["sigma"] == "c_binning"
    assert envs["comp_with_prng"]["bias"] is None
    assert envs["comp_env"]["sigma"] == "c_env_sigma"
    assert envs["comp_env"]["bias"] == "c_env_bias"
    assert "N(bias, sigma)" in envs["comp_env"]["label"]

    blob = open(os.path.join(out, meta["file"]), "rb").read()
    cset = correctionlib.CorrectionSet.from_file(synthetic_json)

    for cm in meta["corrections"]:
        n = cm["n_values"]
        if cm["quant"] == "log10_i16":
            q = np.frombuffer(blob, "<i2", count=n, offset=cm["byte_offset"])
            vals = dequantize_log10_int16(q, cm["scale"])
            rtol = 4e-4
        else:
            vals = np.frombuffer(blob, "<f4", count=n, offset=cm["byte_offset"])
            rtol = 1e-6
        vals = vals.reshape(cm["shape"])
        # direct correctionlib evaluation on the same grid
        pts = []
        for ax in cm["axes"]:
            pts.append(np.asarray(ax["categories"] if ax["kind"] == "cat"
                                  else ax["centers"]))
        mesh = np.meshgrid(*pts, indexing="ij") if pts else []
        flat = {ax["name"]: m.ravel() for ax, m in zip(cm["axes"], mesh)}
        npts = int(np.prod(cm["shape"]))
        corr_json = next(c for c in json.load(open(synthetic_json))["corrections"]
                         if c["name"] == cm["name"])
        args = []
        for inp in corr_json["inputs"]:
            if inp["name"] in flat:
                a = flat[inp["name"]]
                args.append(a.astype(np.int64) if inp["type"] == "int" else a)
            else:
                args.append(np.full(npts, cm["fixed_inputs"][inp["name"]]))
        ref = np.asarray(cset[cm["name"]].evaluate(*args)).reshape(cm["shape"])
        assert np.allclose(vals, ref, rtol=rtol), cm["name"]

    # byte offsets must be 4-byte aligned for JS typed-array views
    for cm in meta["corrections"]:
        assert cm["byte_offset"] % 4 == 0 or cm["quant"] == "log10_i16"
        if cm["quant"] == "f32":
            assert cm["byte_offset"] % 4 == 0


def test_category_values(synthetic_json, tmp_path):
    pytest.importorskip("correctionlib")
    from ngtagger.viz.mva_explorer.correctionlib_ingest import export_generic_dataset
    from ngtagger.viz.mva_explorer.quantize import dequantize_log10_int16

    out = str(tmp_path / "site")
    meta = export_generic_dataset(synthetic_json, "reg_synth", "synthetic", out,
                                  include=["c_category"])
    cm = meta["corrections"][0]
    assert cm["shape"] == [2, 3]  # 2 layers x 3 x-bins
    blob = open(os.path.join(out, meta["file"]), "rb").read()
    q = np.frombuffer(blob, "<i2", count=6, offset=cm["byte_offset"]).reshape(2, 3)
    vals = dequantize_log10_int16(q, cm["scale"])
    assert np.allclose(vals[0], [1.5, 2.5, 3.5], rtol=4e-4)  # layer 1: binning
    assert np.allclose(vals[1], 2.0, rtol=4e-4)              # layer 2: scalar


def _mini_smear_cset(configs, params):
    """Tiny tkLayout-like payload: {p}_smear_{c} + {p}_relative_smear_{c}."""
    edges_pt = [2.0, 5.0, 10.0]
    edges_eta = [0.0, 1.0, 2.4]
    corrections = []
    rng = np.random.default_rng(3)
    for p in params:
        base = None
        for c in configs:
            vals = (0.01 + rng.random((2, 2))).round(6)
            if base is None:
                base = vals.copy()
            for kind, v in (("smear", vals), ("relative_smear", vals / base)):
                content = [{"nodetype": "binning", "input": "eta_tp",
                            "edges": edges_eta, "flow": "clamp",
                            "content": list(map(float, row))} for row in v]
                data = {"nodetype": "binning", "input": "pt_tp",
                        "edges": edges_pt, "flow": "clamp", "content": content}
                corrections.append(_corr(f"{p}_{kind}_{c}",
                                         [_inp("pt_tp"), _inp("eta_tp")], data))
    return _cset(corrections)


def test_structured_smear_export(tmp_path):
    pytest.importorskip("correctionlib")
    from ngtagger.viz.mva_explorer import canonical_config_order
    from ngtagger.viz.mva_explorer.correctionlib_ingest import export_structured_smear
    from ngtagger.viz.mva_explorer.quantize import dequantize_log10_int16

    configs = canonical_config_order()
    params = ["d0", "pt"]
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(_mini_smear_cset(configs, params)))
    out = str(tmp_path / "site")
    meta = export_structured_smear(str(path), "reg_mini", "mini", out,
                                   params=params)
    assert meta["configs"] == configs           # canonical combinatoric order
    assert meta["shape"] == [2, 16, 2, 2, 2]    # kinds x cfg x par x pt x eta
    q = np.fromfile(os.path.join(out, meta["file"]), "<i2").reshape(meta["shape"])
    vals = dequantize_log10_int16(q, meta["scale"])
    # relative kind of config 0000 must be exactly 1
    ik = meta["kinds"].index("relative")
    ic = meta["configs"].index("0000")
    assert np.allclose(vals[ik, ic], 1.0, rtol=4e-4)
    # ratio identity: relative == sigma(cfg)/sigma(0000) on the shared grid
    isig = meta["kinds"].index("sigma")
    ratio = vals[isig] / vals[isig, ic:ic + 1]
    assert np.allclose(vals[ik], ratio, rtol=2e-3)


# ------------------------------------------------------ tkquality exporter

def _synth_refit_tables(n_events=8, tracks_per_event=12, seed=1):
    import awkward as ak

    rng = np.random.default_rng(seed)
    ref_ev, var_ev, hit_ev = [], [], []
    for _ in range(n_events):
        ref_rows, var_rows, hit_rows = [], [], []
        for it in range(tracks_per_event):
            g = bool(rng.random() < 0.7)
            ref_rows.append({
                "hwTanl": int(rng.integers(0, 2 ** 12)),
                "hwZ0": int(rng.integers(0, 2 ** 10)),
                "hwBendChi2": int(rng.integers(0, 8)),
                "hwChi2RPhi": int(rng.integers(0, 16)),
                "hwChi2RZ": int(rng.integers(0, 16)),
                "hitPattern": int(rng.integers(1, 2 ** 7)),
                "nStubs": int(rng.integers(4, 7)),
                "hwRinv": 0, "hwPhi": 0, "hwD0": 0,
                "rInv": 0.01, "phi": float(rng.uniform(-3, 3)),
                "tanL": 0.5, "z0": float(rng.uniform(-10, 10)),
                "d0": float(rng.normal(0, 0.1)),
                "pt": float(2 + 40 * rng.random()),
                "eta": float(rng.uniform(-2.4, 2.4)),
                "trkMVA1": float(rng.random()),
                "genuine": g, "looselyGenuine": g, "combinatoric": not g,
                "unknown": False, "tpPt": 10.0, "tpFromHardInteraction": g,
            })
            var_rows.append({
                "spxRefitPerformed": bool(it % 4 != 3),  # some passthrough
                "spxSeedCovOK": True, "spxNCrossings": 2,
                "spxNAcceptedHits": 2, "spxLayerHitMask": 3,
                "spxMaxWindowMult": 2, "spxAnyWindowTruncated": False,
                "spxNKFUpdates": 2,
                "spxChi2IncRPhiTot": float(rng.random() * 10),
                "spxChi2IncRZTot": float(rng.random() * 10),
                "rInv": 0.01, "phi": 0.1, "tanL": 0.5, "z0": 1.0, "d0": 0.0,
            })
            for lyr in (1, 2):
                hit_rows.append({
                    "trackIdx": it, "layer": lyr, "windowMult": 1,
                    "windowTruncated": False, "hasAlpha": True, "hasBeta": True,
                    "resX": 0.01, "resY": 0.01, "pullX": 0.5, "pullY": 0.5,
                    "pullAlpha": 0.5, "pullBeta": 0.5,
                    "sigAlpha": 0.1, "sigBeta": 0.1,
                    "chi2IncRPhi": 1.0, "chi2IncRZ": 1.0,
                })
        ref_ev.append(ref_rows); var_ev.append(var_rows); hit_ev.append(hit_rows)
    return ak.Array(ref_ev), ak.Array(var_ev), ak.Array(hit_ev)


def _tiny_conifer_json(path, n_features=24):
    """Single stump: margin = 0.3 + (x[20] <= 5 ? -0.8 : 0.9)."""
    model = {
        "n_classes": 2, "n_trees": 1, "n_features": n_features,
        "init_predict": [0.3], "norm": 1.0, "splitting_convention": "<=",
        "trees": [[{"feature": [20, -2, -2], "threshold": [5.0, 0.0, 0.0],
                    "children_left": [1, -1, -1], "children_right": [2, -1, -1],
                    "value": [0.0, -0.8, 0.9]}]],
    }
    with open(path, "w") as f:
        json.dump(model, f)
    return model


def test_tkq_rows_synthetic(tmp_path):
    from ngtagger.viz.mva_explorer.tkquality_export import COLUMNS, tkq_rows

    ref, var, hits = _synth_refit_tables()
    cj = str(tmp_path / "conifer.json")
    _tiny_conifer_json(cj)
    rows = tkq_rows(ref, var, hits, "AAAA", cj)
    assert rows.shape[1] == len(COLUMNS)
    n_refit = int(np.sum(np.asarray(
        [v for ev in var["spxRefitPerformed"].tolist() for v in ev])))
    assert len(rows) == n_refit
    score = rows[:, COLUMNS.index("score")]
    assert np.all((score > 0) & (score < 1))
    # stump: margin only depends on nStubs (feature 20) <= 5
    nstub = rows[:, COLUMNS.index("nstub")]
    exp = 1 / (1 + np.exp(-(0.3 + np.where(nstub <= 5, -0.8, 0.9))))
    assert np.allclose(score, exp, atol=1e-6)
    assert set(np.unique(rows[:, COLUMNS.index("label")])) <= {0.0, 1.0}
    assert np.all(rows[:, COLUMNS.index("abs_eta")] >= 0)
    assert np.all(rows[:, COLUMNS.index("abs_d0")] >= 0)


@pytest.mark.skipif(
    not os.path.exists("/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/"
                       "nano_fat_1111_coopt_file1.root"),
    reason="fat coherent nanos not available")
def test_tkq_export_integration(tmp_path):
    from ngtagger.viz.mva_explorer.tkquality_export import export_tkquality

    out = str(tmp_path / "site")
    meta = export_tkquality(out, n_files=1, max_events=20)
    assert [g["view"] for g in meta["groups"]] == ["1111", "1100"]
    rows = np.fromfile(os.path.join(out, "tkq.bin"), "<f4")
    assert len(rows) == sum(g["n_rows"] for g in meta["groups"]) * len(meta["columns"])


# ------------------------------------------------ prediction dump + tagger

def _make_dump(path, n=50, seed=1, charge=False, rng=None):
    from ngtagger.data.labels import CHARGE_CLASS_LABELS, CLASS_LABELS
    from ngtagger.train.prediction_dump import dump_predictions

    rng = rng or np.random.default_rng(seed)
    probs = rng.random((n, 8))
    probs /= probs.sum(1, keepdims=True)
    kwargs = {}
    if charge:
        cp = rng.random((n, 3))
        cp /= cp.sum(1, keepdims=True)
        kwargs = {"charge_probs": cp,
                  "charge_true": rng.integers(0, 3, n),
                  "charge_labels": list(CHARGE_CLASS_LABELS)}
    dump_predictions(
        path, class_probs=probs, class_labels=list(CLASS_LABELS),
        y_true=rng.integers(0, 8, n),
        kinematics={"pt": 15 + 100 * rng.random(n),
                    "abs_eta": 2.4 * rng.random(n),
                    "phi": rng.uniform(-3.14, 3.14, n),
                    "nconst": rng.integers(1, 16, n)},
        meta={"seed": seed}, **kwargs)


def test_prediction_dump_roundtrip(tmp_path):
    from ngtagger.train.prediction_dump import load_predictions

    p = str(tmp_path / "d.npz")
    _make_dump(p, n=20, charge=True)
    d = load_predictions(p)
    assert d["class_probs"].shape == (20, 8)
    assert d["charge_probs"].shape == (20, 3)
    assert set(d["kinematics"]) == {"pt", "abs_eta", "phi", "nconst"}
    assert d["meta"]["seed"] == 1
    assert len(d["class_labels"]) == 8


def test_tagger_export(tmp_path):
    from ngtagger.viz.mva_explorer.tagger_export import export_tagger

    dumps = tmp_path / "dumps"
    dumps.mkdir()
    rng = np.random.default_rng(9)
    for cell, charge in [("1111__baseline", False), ("1100__baseline", False),
                         ("0000__baseline", False),
                         ("1111__both__chargehead", True)]:
        for seed in (1, 2):
            _make_dump(str(dumps / f"{cell}__s{seed}.npz"),
                       n=30, seed=seed, charge=charge, rng=rng)
    out = str(tmp_path / "site")
    meta = export_tagger(out, dumps_dir=str(dumps))
    assert len(meta["groups"]) == 8
    # canonical view order: 1111 first, 0000 last
    views = [g["view"] for g in meta["groups"]]
    assert views.index("1111") < views.index("1100") < views.index("0000")
    ch = [g for g in meta["groups"] if g["cell"] == "1111__both__chargehead"]
    assert all(g["has_charge"] for g in ch)
    assert all(not g["has_charge"] for g in meta["groups"]
               if g["cell"].endswith("__baseline"))
    data = np.fromfile(os.path.join(out, "tagger.bin"), "<f4")
    ncol = len(meta["columns"])
    total = sum(g["n_rows"] for g in meta["groups"])
    assert len(data) == total * ncol
    mat = data.reshape(total, ncol)
    # id probs sum to 1; charge columns are -1 where absent
    idcols = [meta["columns"].index(f"prob_{c}") for c in meta["class_names"]]
    assert np.allclose(mat[:, idcols].sum(1), 1.0, atol=1e-5)
    qcol = meta["columns"].index("prob_qminus")
    nochg = [g for g in meta["groups"] if not g["has_charge"]][0]
    sl = slice(nochg["row_offset"], nochg["row_offset"] + nochg["n_rows"])
    assert np.all(mat[sl, qcol] == -1.0)


# ------------------------------------------------------------ JS core (JXA)

@pytest.mark.skipif(sys.platform != "darwin", reason="JXA needs macOS osascript")
def test_js_core_matches_python(tmp_path):
    sys.path.insert(0, SITE_SRC)
    try:
        from make_core_testdata import generate
    finally:
        sys.path.pop(0)

    fixture = str(tmp_path / "core_testdata.json")
    generate(fixture)

    core = open(os.path.join(SITE_SRC, "explorer_core.js")).read()
    harness = open(os.path.join(SITE_SRC, "test_core_jxa.js")).read()
    script = tmp_path / "run.js"
    script.write_text(core + "\n" +
                      harness.replace("__TESTDATA__", fixture))
    res = subprocess.run(["osascript", "-l", "JavaScript", str(script)],
                         capture_output=True, text=True, timeout=120)
    print(res.stderr)
    assert res.returncode == 0, res.stderr
    assert "ALL PASS" in (res.stdout + res.stderr), res.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="JXA needs macOS osascript")
def test_site_smoke_jxa(tmp_path):
    """Load the rendered site data exactly as explorer.html would (typed
    arrays at byte offsets) and run the compute core on it.  Gated on the
    regenerable site dir existing (python -m ngtagger.viz.mva_explorer ...)."""
    site = os.path.join(os.path.dirname(__file__), "..", "eval_mva_explorer", "site")
    site = os.path.abspath(site)
    if not os.path.exists(os.path.join(site, "manifest.json")):
        pytest.skip("no rendered site (run python -m ngtagger.viz.mva_explorer export-all)")
    core = open(os.path.join(SITE_SRC, "explorer_core.js")).read()
    smoke = open(os.path.join(SITE_SRC, "smoke_site_jxa.js")).read()
    script = tmp_path / "smoke.js"
    script.write_text(core + "\n" + smoke.replace("__SITEDIR__", site))
    res = subprocess.run(["osascript", "-l", "JavaScript", str(script)],
                         capture_output=True, text=True, timeout=600)
    print(res.stdout, res.stderr)
    assert res.returncode == 0, res.stderr
    assert "SMOKE PASS" in (res.stdout + res.stderr), res.stdout + res.stderr
