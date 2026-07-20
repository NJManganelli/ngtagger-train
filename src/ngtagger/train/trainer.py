"""Single training run: L1PFTrkNano files -> tensors -> model fit, with
mlflow tracking. Class/pt re-balancing weights ported from upstream
TrainTagger train_weights."""

from __future__ import annotations

import os

import awkward as ak
import numpy as np

from ngtagger.data.features import build_features
from ngtagger.data.labels import CLASS_LABELS, label_jets
from ngtagger.data.nano import load_jets
from ngtagger.models.base import ModelRegistry, TagModel

PT_BINS = np.array([15, 17, 19, 22, 25, 30, 35, 40, 45, 50, 60, 76, 97, 122, 154, np.inf])


def class_pt_weights(y: np.ndarray, reco_pt: np.ndarray, method: str = "onlyclass") -> np.ndarray:
    """Re-balance classes (and optionally flatten in pt), following upstream."""
    if method == "none":
        return np.ones(len(y))
    bins = np.array([0.0, np.inf]) if method == "onlyclass" else PT_BINS
    weights = np.ones(len(y))
    labels = y.argmax(axis=1)
    counts = np.zeros((y.shape[1], len(bins) - 1))
    for c in range(y.shape[1]):
        counts[c], _ = np.histogram(reco_pt[labels == c], bins=bins)
    ref = counts[0]  # weight all classes to the b class, as upstream
    binidx = np.clip(np.digitize(reco_pt, bins) - 1, 0, len(bins) - 2)
    for c in range(y.shape[1]):
        sel = labels == c
        w = np.where(counts[c][binidx[sel]] > 0, ref[binidx[sel]] / counts[c][binidx[sel]], 1.0)
        weights[sel] = w
    return weights


def prepare_dataset(files: list[str], n_const: int = 16, feature_groups: list[str] | None = None,
                    max_events: int | None = None, test_fraction: float = 0.1, seed: int = 0,
                    tables: dict | None = None, gen_match_dr: float = 0.4):
    """nano files -> train/test numpy tensors. No intermediate formats.

    tables: optional overrides for the nano table names (jet_table, link_table,
    cand_table, track_table, cluster_table) — e.g. the SC8 pipeline points
    jet_table/link_table at the SC8 NG collections. gen_match_dr follows the
    jet radius (0.4 for SC4, 0.8 for SC8).
    """
    jets, constituents, gen = load_jets(files, n_const=n_const, feature_groups=feature_groups,
                                        max_events=max_events, **(tables or {}))
    label, target_pt, target_pt_phys, keep = label_jets(jets, gen, max_dr=gen_match_dr)

    X, feature_names = build_features(jets, constituents, n_const=n_const, feature_groups=feature_groups)
    flat_label = ak.to_numpy(ak.flatten(label))
    flat_keep = ak.to_numpy(ak.flatten(keep))
    flat_pt_phys = ak.to_numpy(ak.flatten(target_pt_phys))
    flat_reco_pt = ak.to_numpy(ak.flatten(jets.pt))

    X = X[flat_keep]
    y = np.eye(len(CLASS_LABELS))[flat_label[flat_keep]]
    pt_target = target_pt[flat_keep]
    truth_pt = flat_pt_phys[flat_keep]
    reco_pt = flat_reco_pt[flat_keep]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_fraction)
    test, train = idx[:n_test], idx[n_test:]
    return {
        "X_train": X[train], "y_train": y[train], "pt_train": pt_target[train],
        "truth_pt_train": truth_pt[train], "reco_pt_train": reco_pt[train],
        "X_test": X[test], "y_test": y[test], "pt_test": pt_target[test],
        "truth_pt_test": truth_pt[test], "reco_pt_test": reco_pt[test],
        "feature_names": feature_names, "class_labels": CLASS_LABELS,
    }


def run_training(config_path: str, files: list[str], output_dir: str, seed: int = 0,
                 max_events: int | None = None, mlflow_run_name: str | None = None,
                 dataset: dict | None = None) -> TagModel:
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = ModelRegistry.create(config["model"], output_dir)
    model.load_yaml(config_path)

    dc = config.get("data_config", {})
    tables = {k: dc[k] for k in ("jet_table", "link_table", "cand_table",
                                 "track_table", "cluster_table") if k in dc}
    ds = dataset or prepare_dataset(
        files,
        n_const=dc.get("n_constituents", 16),
        feature_groups=dc.get("feature_groups", ["baseline"]),
        max_events=max_events,
        seed=seed,
        tables=tables,
        gen_match_dr=dc.get("gen_match_dr", 0.4),
    )
    model.class_labels = list(ds["class_labels"])
    model.feature_names = list(ds["feature_names"])

    weights = class_pt_weights(ds["y_train"], ds["reco_pt_train"],
                               model.training_config.get("weight_method", "onlyclass"))

    try:
        import mlflow

        mlflow.set_experiment(config.get("experiment", "ngtagger"))
        run_ctx = mlflow.start_run(run_name=mlflow_run_name, nested=mlflow.active_run() is not None)
    except Exception:
        mlflow = None
        run_ctx = _NullCtx()

    with run_ctx:
        if mlflow:
            mlflow.log_params({
                "model": config["model"], "seed": seed,
                "n_train": len(ds["X_train"]),
                **{f"train.{k}": v for k, v in model.training_config.items() if np.isscalar(v)},
            })
        model.build(ds["X_train"].shape[1:], len(model.class_labels))
        model.compile()
        model.fit(ds["X_train"], ds["y_train"], ds["pt_train"],
                  sample_weight=weights,
                  validation_split=model.training_config.get("validation_split", 0.1),
                  seed=seed)
        model.save()
        if mlflow:
            mlflow.log_metric("best_val_loss", model.best_val_loss())
            mlflow.log_artifacts(model.output_dir)
    return model


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
