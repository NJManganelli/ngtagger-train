"""Displaced-vertex tagger retraining from L1TrkNano.

Rebuilds the GTT DisplacedVertexProducer conifer GBDT
(dispVertTaggerEmulationConiferShifted13p8.json, ap_fixed<13,8> emulation)
with the exact 20-feature set documented in DisplacedVertexProducer_cfi:

  per track pair (firstTrk = higher pt), gathered from the L1TExtTrack
  table via firstIndexTrk/secondIndexTrk:
    pt, eta, phi, d0, z0, chi2rz (chi2ZRed), bendchi2 (chi2BendRed),
    MVA (trkMVA1)
  per vertex (from the L1DispVertex table): d_T, R_T, cos_T, del_Z

Labels: L1DispVertex_isReal (computed in-producer from the extended-track
truth association). The stock production score (L1DispVertex_score) lives in
the same table, enabling direct stock-vs-retrained comparisons.
"""

from __future__ import annotations

import json
import os

import awkward as ak
import numpy as np
import uproot

TRACK_FEATURES = ["pt", "eta", "phi", "d0", "z0", "chi2ZRed", "chi2BendRed", "trkMVA1"]
VERTEX_FEATURES = ["d_T", "R_T", "cos_T", "del_Z"]

# documented ordering: interleaved (firstTrk, secondTrk) per quantity, then dv_*
FEATURE_NAMES = [f"trkExt_{f}_firstTrk" if i == 0 else f"trkExt_{f}"
                 for f in TRACK_FEATURES for i in range(2)] + [f"dv_{v}" for v in VERTEX_FEATURES]


def load_dispvtx_data(files: list[str], vertex_table: str = "L1DispVertex",
                      track_table: str = "L1TExtTrack", max_events: int | None = None):
    branches = [f"{vertex_table}_*"] + [f"{track_table}_{b}" for b in TRACK_FEATURES]
    events = uproot.concatenate([f"{f}:Events" for f in files], filter_name=branches, how="zip")
    if max_events is not None:
        events = events[:max_events]
    return events[vertex_table], events[track_table]


def build_dispvtx_dataset(vertices: ak.Array, tracks: ak.Array):
    """Gather the 20 GBDT features; returns (X, y, stock_score)."""
    first = vertices.firstIndexTrk
    second = vertices.secondIndexTrk

    cols = []
    for f in TRACK_FEATURES:
        for idx in (first, second):
            valid = idx >= 0
            safe = ak.where(valid, idx, 0)
            vals = ak.where(valid, tracks[f][safe], 0.0)
            cols.append(ak.to_numpy(ak.flatten(vals)).astype(np.float32))
    for v in VERTEX_FEATURES:
        cols.append(ak.to_numpy(ak.flatten(vertices[v])).astype(np.float32))

    X = np.stack(cols, axis=1)
    y = ak.to_numpy(ak.flatten(vertices.isReal)).astype(np.int64)
    stock = ak.to_numpy(ak.flatten(vertices.score)).astype(np.float32)
    return X, y, stock


def train_dispvtx(files: list[str], output_dir: str, max_events: int | None = None,
                  test_fraction: float = 0.2, seed: int = 0, xgb_params: dict | None = None,
                  dataset=None):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    os.makedirs(output_dir, exist_ok=True)
    if dataset is None:
        vertices, tracks = load_dispvtx_data(files, max_events=max_events)
        X, y, stock = build_dispvtx_dataset(vertices, tracks)
    else:
        X, y, stock = dataset

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_fraction)
    test, train = idx[:n_test], idx[n_test:]

    params = {
        "n_estimators": 100, "max_depth": 4, "learning_rate": 0.2,
        "objective": "binary:logistic", "eval_metric": "auc",
        "early_stopping_rounds": 10, **(xgb_params or {}),
    }
    model = xgb.XGBClassifier(**params, random_state=seed)
    model.fit(X[train], y[train], eval_set=[(X[test], y[test])], verbose=False)

    new_score = model.predict_proba(X[test])[:, 1]
    metrics = compare_dispvtx_scores(y[test], stock[test], new_score)
    print(f"dispvtx GBDT: {metrics}")

    model.save_model(os.path.join(output_dir, "dispvtx_xgb.json"))
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump({"features": FEATURE_NAMES, "metrics": metrics,
                   "params": {k: v for k, v in params.items()}}, f, indent=2)

    try:
        import mlflow

        mlflow.set_experiment("ngtagger-dispvtx")
        with mlflow.start_run(run_name="dispvtx"):
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            mlflow.log_artifacts(output_dir)
    except Exception:
        pass
    return model, metrics


def compare_dispvtx_scores(y_true, stock_score, new_score):
    """Stock-production conifer score vs retrained score, both against the
    isReal truth label."""
    from sklearn.metrics import roc_auc_score

    out = {}
    if len(np.unique(y_true)) > 1:
        out["stock_auc"] = float(roc_auc_score(y_true, stock_score))
        out["new_auc"] = float(roc_auc_score(y_true, new_score))
    ranks_stock = np.argsort(np.argsort(stock_score))
    ranks_new = np.argsort(np.argsort(new_score))
    out["rank_correlation"] = float(np.corrcoef(ranks_stock, ranks_new)[0, 1])
    return out


def export_conifer(model_dir: str, output_dir: str | None = None):
    """Export to a conifer json matching the deployed model format. The
    emulation uses ap_fixed<13,8,AP_RND_CONV,AP_SAT> in- and outputs
    (DisplacedVertexProducer), so bit-exact deployment needs the shifted/
    scaled feature convention of the 'Shifted13p8' training."""
    import conifer
    import xgboost as xgb

    output_dir = output_dir or model_dir
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "dispvtx_xgb.json"))
    cfg = conifer.backends.cpp.auto_config()
    cfg["OutputDir"] = os.path.join(output_dir, "conifer_cpp")
    cnf = conifer.converters.convert_from_xgboost(booster, cfg)
    cnf.save(os.path.join(output_dir, "dispvtx_conifer.json"))
    print(f"conifer model written to {output_dir}/dispvtx_conifer.json "
          f"(deployable via DisplacedVertexProducer.model FileInPath)")
    return cnf
