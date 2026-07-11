"""Smoke tests: synthetic-data model training and feature building.
Data-reading tests require an actual L1PFTrkNano file (skipped if absent,
set NGTAGGER_TEST_NANO to point at one)."""

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

N_CONST, N_FEAT, N_CLASSES = 16, 20, 8


def _synthetic(n=512):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, N_CONST, N_FEAT)).astype("float32")
    y = np.eye(N_CLASSES)[rng.integers(0, N_CLASSES, n)]
    pt = rng.uniform(0.3, 2.0, n).astype("float32")
    return X, y, pt


@pytest.fixture
def tiny_config(tmp_path):
    import yaml

    cfg = {
        "model": "DeepSetHGQ2",
        "run_config": {"verbose": 0},
        "model_config": {"name": "tiny", "conv1d_layers": [4], "conv1d_parallelisation_factor": [16],
                         "classification_layers": [8], "regression_layers": [4], "beta": 1e-8},
        "quantization_config": {"pt_output_quantization": [16, 6]},
        "training_config": {"weight_method": "onlyclass", "validation_split": 0.2, "epochs": 1,
                            "batch_size": 128, "learning_rate": 0.01, "loss_weights": [1.0, 1.0]},
        "firmware_config": {"project_name": "tiny_test"},
    }
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_deepset_hgq2_smoke(tiny_config, tmp_path):
    pytest.importorskip("hgq")
    from ngtagger.models.base import ModelRegistry

    X, y, pt = _synthetic()
    model = ModelRegistry.create("DeepSetHGQ2", str(tmp_path / "out"))
    model.load_yaml(tiny_config)
    model.class_labels = [f"c{i}" for i in range(N_CLASSES)]
    model.feature_names = [f"f{i}" for i in range(N_FEAT)]
    model.build((N_CONST, N_FEAT), N_CLASSES)
    model.compile()
    model.fit(X, y, pt, validation_split=0.2, seed=1)
    assert model.best_val_loss() < np.inf
    model.save()
    assert (tmp_path / "out" / "model.keras").exists()


def test_contrastive_smoke(tmp_path):
    pytest.importorskip("hgq")
    import yaml

    from ngtagger.models.base import ModelRegistry

    cfg = {
        "model": "DeepSetContrastive",
        "run_config": {"verbose": 0},
        "model_config": {"name": "tinyc", "conv1d_layers": [4], "classification_layers": [8],
                         "regression_layers": [4], "projection_dims": 4, "temperature": 0.5,
                         "beta": 1e-8},
        "quantization_config": {"pt_output_quantization": [16, 6]},
        "training_config": {"weight_method": "onlyclass", "validation_split": 0.2,
                            "embedding_epochs": 1, "finetuning_epochs": 1, "batch_size": 128,
                            "learning_rate": 0.01, "embedding_lr": 0.01, "loss_weights": [1.0, 1.0]},
        "firmware_config": {"project_name": "tinyc_test"},
    }
    p = tmp_path / "tinyc.yaml"
    p.write_text(yaml.safe_dump(cfg))

    X, y, pt = _synthetic()
    model = ModelRegistry.create("DeepSetContrastive", str(tmp_path / "outc"))
    model.load_yaml(str(p))
    model.class_labels = [f"c{i}" for i in range(N_CLASSES)]
    model.feature_names = [f"f{i}" for i in range(N_FEAT)]
    model.build((N_CONST, N_FEAT), N_CLASSES)
    model.compile()
    model.fit(X, y, pt, validation_split=0.2, seed=1)
    assert model.best_val_loss() < np.inf


def test_class_pt_weights():
    from ngtagger.train.trainer import class_pt_weights

    rng = np.random.default_rng(0)
    y = np.eye(N_CLASSES)[rng.integers(0, N_CLASSES, 1000)]
    pt = rng.uniform(10, 200, 1000)
    w = class_pt_weights(y, pt, "onlyclass")
    assert w.shape == (1000,)
    assert (w > 0).all()


@pytest.mark.skipif(not os.environ.get("NGTAGGER_TEST_NANO"), reason="no nano test file")
def test_nano_pipeline():
    from ngtagger.train.trainer import prepare_dataset

    ds = prepare_dataset([os.environ["NGTAGGER_TEST_NANO"]], max_events=200)
    assert ds["X_train"].shape[1:] == (16, len(ds["feature_names"]))
    assert ds["y_train"].shape[1] == 8
