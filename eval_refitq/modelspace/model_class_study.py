"""Part A.2: model-class exploration for the refit track-quality MVA.

Models (all on identical split seeds; paired deltas vs xgb_spec baseline):
  xgb_spec        60x3 trees, spec17 raw (the deployed-contract baseline)
  xgb_spec24      60x3 trees, spec17_guard + classic 7 TRKQ hw features
  xgb_deep24      200 trees depth 4, lr 0.1, subsample/colsample 0.8
  xgb_deeper24    400 trees depth 6, lr 0.05, subsample/colsample 0.8
  xgb_bag5_24     5-model bagging ensemble of the 60x3 (seed-varied)
  mlp24           sklearn MLP (32,32), standardized, nan->0
  deepset_hits    tiny per-layer-slot DeepSet (phi 16-16, masked sum pool,
                  concat 18 track-level globals, rho 32) - the structurally
                  right model for the variable-length hit set

Hardware mapping is done in the study doc, not here.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelspace_common import (  # noqa: E402
    SEEDS, load_dataset, paired_auc_deltas, perlayer_matrix, spec_guarded, split,
)
from perlayer_study import _tierA_block  # noqa: E402

HIT_COLS_SLOT = ["accepted", "windowMult", "resX", "resY", "pullX", "pullY",
                 "pullAlpha", "pullBeta", "chi2IncRPhi", "chi2IncRZ", "crossed"]


def _xgb_auc(X, y, seed, params):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    train, test = split(len(X), seed)
    p = dict(params)
    n_pos = max(int(y[train].sum()), 1)
    n_neg = max(int((y[train] == 0).sum()), 1)
    p.setdefault("scale_pos_weight", n_neg / n_pos)
    m = xgb.XGBClassifier(**p, random_state=seed)
    m.fit(X[train], y[train], eval_set=[(X[test], y[test])], verbose=False)
    return float(roc_auc_score(y[test], m.predict_proba(X[test])[:, 1]))


def _xgb_bag_auc(X, y, seed, params, n_bag=5):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    train, test = split(len(X), seed)
    p = dict(params)
    n_pos = max(int(y[train].sum()), 1)
    n_neg = max(int((y[train] == 0).sum()), 1)
    p.setdefault("scale_pos_weight", n_neg / n_pos)
    p.setdefault("subsample", 0.8)
    p.setdefault("colsample_bytree", 0.8)
    probs = np.zeros(len(test))
    for b in range(n_bag):
        m = xgb.XGBClassifier(**p, random_state=1000 * seed + b)
        m.fit(X[train], y[train], eval_set=[(X[test], y[test])], verbose=False)
        probs += m.predict_proba(X[test])[:, 1]
    return float(roc_auc_score(y[test], probs / n_bag))


def _mlp_auc(X, y, seed):
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    train, test = split(len(X), seed)
    Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sc = StandardScaler().fit(Xc[train])
    Zt, Zs = sc.transform(Xc[train]), sc.transform(Xc[test])
    m = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=400,
                      early_stopping=True, n_iter_no_change=15,
                      random_state=seed)
    # class re-weight via sample duplication is overkill; MLP handles the
    # 2.8% minority acceptably with early stopping on val AUC-proxy loss
    m.fit(Zt, y[train])
    return float(roc_auc_score(y[test], m.predict_proba(Zs)[:, 1]))


def _deepset_auc(Xslots, Xglob, y, seed):
    import keras
    from keras import layers
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    train, test = split(len(y), seed)
    ns, nf = Xslots.shape[1], Xslots.shape[2]
    # standardize per feature over filled slots; nan->0 after scaling
    Xs = np.nan_to_num(Xslots, nan=0.0)
    flat = Xs.reshape(-1, nf)
    sc = StandardScaler().fit(flat)
    Xs = sc.transform(flat).reshape(-1, ns, nf).astype(np.float32)
    scg = StandardScaler().fit(np.nan_to_num(Xglob[train], nan=0.0))
    Xg = scg.transform(np.nan_to_num(Xglob, nan=0.0)).astype(np.float32)

    keras.utils.set_random_seed(seed)
    in_s = layers.Input(shape=(ns, nf))
    in_g = layers.Input(shape=(Xg.shape[1],))
    h = layers.Dense(16, activation="relu")(in_s)
    h = layers.Dense(16, activation="relu")(h)
    pooled = layers.GlobalAveragePooling1D()(h)
    z = layers.Concatenate()([pooled, in_g])
    z = layers.Dense(32, activation="relu")(z)
    out = layers.Dense(1, activation="sigmoid")(z)
    m = keras.Model([in_s, in_g], out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    n_pos = max(int(y[train].sum()), 1)
    n_neg = max(int((y[train] == 0).sum()), 1)
    m.fit([Xs[train], Xg[train]], y[train],
          validation_split=0.15, epochs=60, batch_size=512,
          class_weight={0: n_neg and (len(train) / (2 * n_neg)), 1: len(train) / (2 * n_pos)},
          callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
          verbose=0)
    p = m.predict([Xs[test], Xg[test]], batch_size=4096, verbose=0)[:, 0]
    return float(roc_auc_score(y[test], p))


def main(config: str = "AAAA"):
    ds = load_dataset(config)
    X_spec, y, names = ds["X_spec"], ds["y"], ds["spec_names"]
    Xg17 = spec_guarded(X_spec, names)
    Xa = _tierA_block(config)[0][ds["aux"]["refit_mask"]]
    X24 = np.concatenate([Xg17, Xa], axis=1)

    # slot tensor for the DeepSet: (N, 4 layers, len(HIT_COLS_SLOT))
    pl = ds["per_layer"]
    slot_m, _slot_names = perlayer_matrix(pl, HIT_COLS_SLOT)
    Xslots = slot_m.reshape(len(y), 4, len(HIT_COLS_SLOT))
    # globals: counters + kicks + seedTrkMVA1 (spec ix 0-4, 11-16) + tierA
    glob_ix = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16]
    Xglob = np.concatenate([Xg17[:, glob_ix], Xa], axis=1)

    small = {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.15,
             "objective": "binary:logistic", "eval_metric": "auc",
             "early_stopping_rounds": 15}
    deep = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
            "objective": "binary:logistic", "eval_metric": "auc",
            "early_stopping_rounds": 20, "subsample": 0.8, "colsample_bytree": 0.8}
    deeper = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
              "objective": "binary:logistic", "eval_metric": "auc",
              "early_stopping_rounds": 30, "subsample": 0.8, "colsample_bytree": 0.8}

    runs = {
        "xgb_spec": lambda s: _xgb_auc(X_spec, y, s, small),
        "xgb_spec24": lambda s: _xgb_auc(X24, y, s, small),
        "xgb_deep24": lambda s: _xgb_auc(X24, y, s, deep),
        "xgb_deeper24": lambda s: _xgb_auc(X24, y, s, deeper),
        "xgb_bag5_24": lambda s: _xgb_bag_auc(X24, y, s, small),
        "mlp24": lambda s: _mlp_auc(X24, y, s),
        "deepset_hits": lambda s: _deepset_auc(Xslots, Xglob, y, s),
    }

    aucs = {k: [] for k in runs}
    for seed in SEEDS:
        for k, fn in runs.items():
            aucs[k].append(fn(seed))
        print(f"seed {seed}: " + "  ".join(f"{k}={aucs[k][-1]:.4f}" for k in runs))

    stats = paired_auc_deltas(aucs, "xgb_spec")
    out = {"config": config, "n_rows": int(len(y)), "n_pos": int(y.sum()),
           "n_neg": int((y == 0).sum()), "seeds": list(SEEDS),
           "aucs": aucs, "paired_vs_xgb_spec": stats}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"model_class_results_{config}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    print("\npaired deltas vs xgb_spec:")
    for k, s in stats.items():
        print(f"  {k:14s} auc={s['auc_mean']:.4f}+-{s['auc_std']:.4f}  "
              f"delta={s['delta_mean']:+.4f}+-{s['delta_std']:.4f}  "
              f"improved {s['n_seeds_improved']}/{len(SEEDS)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAAA")
