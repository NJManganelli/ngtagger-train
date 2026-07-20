"""Charge-classifier head scaffold tests (model-space study B.2.3): the
partonFlavour -> 3-class map, gen matching, the engineered Q_kappa baseline,
and a synthetic charge-correlated training where the head must learn the
sign (with Q_kappa evaluated on the same sample as the benchmark)."""

import os

import awkward as ak
import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from ngtagger.data.features import FEATURE_GROUPS  # noqa: E402
from ngtagger.data.labels import (  # noqa: E402
    CHARGE_CLASS_LABELS,
    label_jet_charge,
    parton_charge_class,
)
from ngtagger.eval.charge_baseline import (  # noqa: E402
    evaluate_charge_baseline,
    jet_charge_from_features,
    jet_charge_kappa,
)

BASELINE = FEATURE_GROUPS["baseline"]


def test_parton_charge_class_map():
    # full documented pdgId -> class table (0: q-, 1: neutral, 2: q+)
    pdg = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 21, 0]
    want = [0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 1, 1]
    got = ak.to_numpy(parton_charge_class(ak.Array(pdg)))
    assert got.tolist() == want
    assert len(CHARGE_CLASS_LABELS) == 3


def test_label_jet_charge_matching():
    # event 0: two jets, matched to b (q-) and ubar (q-? no: -2 ubar -> q-)
    # event 1: one jet matched to gluon, one unmatched -> both neutral
    jets = ak.Array([
        {"eta": [0.0, 1.0], "phi": [0.0, 1.0], "pt": [50.0, 40.0]},
        {"eta": [0.5, 2.0], "phi": [-1.0, 2.5], "pt": [30.0, 20.0]},
    ])
    gen = {"GenJet": ak.Array([
        {"eta": [0.05, 1.02], "phi": [0.02, 0.98], "partonFlavour": [5, -2]},
        {"eta": [0.52], "phi": [-1.03], "partonFlavour": [21]},
    ])}
    got = ak.to_numpy(ak.flatten(label_jet_charge(jets, gen, max_dr=0.4)))
    assert got.tolist() == [0, 0, 1, 1]  # b -> q-, ubar -> q-, gluon/unmatched -> neutral


def test_jet_charge_kappa_hand_computed():
    charge = np.array([[1.0, -1.0, 0.0]])
    pt = np.array([[4.0, 1.0, 0.0]])
    # sum_pow (study formula): (1*2 - 1*1) / (5)^0.5
    got = jet_charge_kappa(charge, pt, kappa=0.5, norm="sum_pow")
    assert np.isclose(got[0], 1.0 / np.sqrt(5.0))
    # pow_sum (Field-Feynman): (2 - 1) / (2 + 1)
    got = jet_charge_kappa(charge, pt, kappa=0.5, norm="pow_sum")
    assert np.isclose(got[0], 1.0 / 3.0)
    # kappa=1 sum_pow: (4 - 1)/5
    got = jet_charge_kappa(charge, pt, kappa=1.0, norm="sum_pow")
    assert np.isclose(got[0], 3.0 / 5.0)
    # empty jet -> 0, and jagged awkward input agrees with numpy
    assert jet_charge_kappa(np.zeros((1, 3)), np.zeros((1, 3)))[0] == 0.0
    jag = jet_charge_kappa(ak.Array([[1.0, -1.0, 0.0], []]), ak.Array([[4.0, 1.0, 0.0], []]))
    assert np.isclose(jag[0], 1.0 / np.sqrt(5.0)) and jag[1] == 0.0
    with pytest.raises(ValueError, match="norm"):
        jet_charge_kappa(charge, pt, norm="nope")


def _synthetic_charge_tensor(n=900, n_const=16, seed=7):
    """Charge-correlated toy: class q+ jets are mostly + tracks, q- mostly -,
    neutral jets neutral hadrons. Returns (X, feature_names, class ids)."""
    rng = np.random.default_rng(seed)
    names = list(BASELINE)
    idx = {k: names.index(k) for k in names}
    cls = rng.integers(0, 3, n)  # 0: q-, 1: neutral, 2: q+
    X = np.zeros((n, n_const, len(names)), dtype="float32")
    X[:, :, idx["pt"]] = rng.uniform(1.0, 3.0, (n, n_const))
    X[:, :, idx["isfilled"]] = 1.0
    for j in range(n):
        for c in range(n_const):
            if cls[j] == 1 or rng.random() < 0.2:
                X[j, c, idx["isNeutralHadron"]] = 1.0
            else:
                # 85% of tracks carry the jet's sign
                sign_is_plus = (cls[j] == 2) == (rng.random() < 0.85)
                col = "isChargedHadronPlus" if sign_is_plus else "isChargedHadronMinus"
                X[j, c, idx[col]] = 1.0
    return X, names, cls


def test_qkappa_baseline_on_synthetic():
    X, names, cls = _synthetic_charge_tensor()
    q = jet_charge_from_features(X, names, kappa=0.5)
    res = evaluate_charge_baseline(q, cls)
    assert res["auc_pm"] > 0.95  # strongly charge-correlated toy
    assert res["mean_q_per_class"][2] > 0 > res["mean_q_per_class"][0]
    assert abs(res["mean_q_per_class"][1]) < 0.1
    assert sum(res["n_per_class"]) == len(cls)


def test_jet_charge_from_features_missing_columns():
    with pytest.raises(ValueError, match="missing baseline charge columns"):
        jet_charge_from_features(np.zeros((1, 2, 2)), ["pt", "deta"])


def test_charge_head_learns_sign(tmp_path):
    """The 3-class head on a tiny DeepSetHGQ2 learns the charge-correlated
    toy well above chance (1/3) and above random-head level."""
    pytest.importorskip("hgq")
    import yaml

    from ngtagger.models.base import ModelRegistry

    X, names, cls = _synthetic_charge_tensor()
    y8 = np.eye(8)[np.random.default_rng(0).integers(0, 8, len(X))]
    pt = np.ones(len(X), dtype="float32")
    yq = np.eye(3)[cls]

    cfg = {
        "model": "DeepSetHGQ2",
        "run_config": {"verbose": 0},
        "model_config": {"name": "tinyq", "conv1d_layers": [8], "conv1d_parallelisation_factor": [16],
                         "classification_layers": [8], "regression_layers": [4],
                         "charge_layers": [8], "beta": 1e-8},
        "training_config": {"validation_split": 0.2, "epochs": 30, "batch_size": 128,
                            "learning_rate": 0.02, "loss_weights": [0.1, 0.1, 1.0],
                            "EarlyStopping_patience": 30},
        "firmware_config": {"project_name": "tinyq"},
    }
    p = tmp_path / "tinyq.yaml"
    p.write_text(yaml.safe_dump(cfg))

    model = ModelRegistry.create("DeepSetHGQ2", str(tmp_path / "outq"))
    model.load_yaml(str(p))
    model.class_labels = [f"c{i}" for i in range(8)]
    model.feature_names = names
    model.build((16, len(names)), 8)
    model.compile()
    model.fit(X, y8, pt, validation_split=0.2, seed=1, y_charge=yq)

    preds = model.predict(X)
    assert len(preds) == 3 and preds[2].shape == (len(X), 3)
    acc = float(np.mean(preds[2].argmax(axis=1) == cls))
    assert acc > 0.7, acc


def test_charge_head_requires_labels(tmp_path):
    pytest.importorskip("hgq")
    import yaml

    from ngtagger.models.base import ModelRegistry

    cfg = {
        "model": "DeepSetHGQ2",
        "run_config": {"verbose": 0},
        "model_config": {"name": "t", "conv1d_layers": [4], "conv1d_parallelisation_factor": [16],
                         "classification_layers": [8], "regression_layers": [4],
                         "charge_layers": [4], "beta": 1e-8},
        "training_config": {"epochs": 1, "batch_size": 128, "loss_weights": [1.0, 1.0, 1.0]},
        "firmware_config": {"project_name": "t"},
    }
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(cfg))
    model = ModelRegistry.create("DeepSetHGQ2", str(tmp_path / "out"))
    model.load_yaml(str(p))
    model.build((16, 20), 8)
    model.compile()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 16, 20)).astype("float32")
    with pytest.raises(ValueError, match="charge head"):
        model.fit(X, np.eye(8)[rng.integers(0, 8, 64)], np.ones(64, dtype="float32"), seed=0)


def test_contrastive_rejects_charge_head(tmp_path):
    from ngtagger.models.base import ModelRegistry

    model = ModelRegistry.create("DeepSetContrastive", str(tmp_path / "out"))
    model.config = {"model_config": {"charge_layers": [4]}}
    with pytest.raises(NotImplementedError, match="charge head"):
        model.build((16, 20), 8)


def test_charge_labels_from_nano(jet_nano_path):
    """Real-data smoke (runs only when NGTAGGER_TEST_NANO points at a JET
    nano with GenJet_partonFlavour; a track-only nano is skipped cleanly)."""
    from ngtagger.train.trainer import prepare_dataset

    ds = prepare_dataset([jet_nano_path], max_events=200)
    assert ds["charge_train"].shape[1] == 3
    assert set(ds["charge_train"].argmax(axis=1)) <= {0, 1, 2}
