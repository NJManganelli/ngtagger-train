"""Part A.1: per-layer feature critique of REFIT_BDT_FEATURES v0.

Question under test: the spec pools pull^2 and chi2 increments over layers,
but the innermost layer is 'golden' (extrapolation q68 0.041 -> 0.899 cm
L1->L4; hit-level |pullBeta| fake-AUC 0.735 at L1 falling outward). Do
per-layer features beat the pooled sums?

Feature sets (all include the same split seeds; paired deltas quoted):
  spec17            producer contract v0 (baseline)
  spec17_guard      spec17 with chi2 totals log1p-clipped at 2e6 (§6b proxy)
  spec17+pullL      + per-layer pulls (pullX/Y/Alpha/Beta x L1..L4)
  spec17+occL       + per-layer window multiplicity
  spec17+chi2L      + per-layer chi2 increments (guarded log)
  spec17+fullL      + everything per-layer (pulls, occ, chi2, res, accepted)
  perlayer_subst    pooled sums REMOVED, per-layer blocks in their place
  spec17+tierA      + the classic 7 TRKQ hw features (hitPattern cross-info)

Also reports: xgb gain importances (fullL, seed 0), logistic-regression
layer weights on standardized per-layer pull^2 (the direct answer to 'how
should layers be weighted'), and mutual information of per-layer |pulls|.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelspace_common import (  # noqa: E402
    SEEDS, load_dataset, log1p_guard, paired_auc_deltas, perlayer_matrix,
    spec_guarded, split,
)

from ngtagger.train.refitquality import _tier_a_features  # noqa: E402
import awkward as ak  # noqa: E402


def _tierA_block(config: str):
    """Classic 7 TRKQ features for the same refit-track rows."""
    from ngtagger.train.refitquality import load_refit_tables, _REF_HW, _REF_FLOAT, _REF_TRUTH
    from modelspace_common import NANO
    ref, var, hits = load_refit_tables([NANO], config)
    ref_flat = {b: ak.to_numpy(ak.flatten(ref[b])) for b in (_REF_HW + _REF_FLOAT + _REF_TRUTH)}
    Xa, names = _tier_a_features(ref_flat)
    return Xa, names


def _fit_auc(X, y, seed, params=None):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    train, test = split(len(X), seed)
    p = {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.15,
         "objective": "binary:logistic", "eval_metric": "auc",
         "early_stopping_rounds": 15}
    p.update(params or {})
    n_pos = max(int(y[train].sum()), 1)
    n_neg = max(int((y[train] == 0).sum()), 1)
    p.setdefault("scale_pos_weight", n_neg / n_pos)
    m = xgb.XGBClassifier(**p, random_state=seed)
    m.fit(X[train], y[train], eval_set=[(X[test], y[test])], verbose=False)
    return float(roc_auc_score(y[test], m.predict_proba(X[test])[:, 1])), m


def main(config: str = "AAAA"):
    ds = load_dataset(config)
    X_spec, y, names = ds["X_spec"], ds["y"], ds["spec_names"]
    pl = ds["per_layer"]

    Xg = spec_guarded(X_spec, names)
    pull_m, pull_n = perlayer_matrix(pl, ["pullX", "pullY", "pullAlpha", "pullBeta"])
    occ_m, occ_n = perlayer_matrix(pl, ["windowMult"])
    chi_m, chi_n = perlayer_matrix(pl, ["chi2IncRPhi", "chi2IncRZ"])
    full_m, full_n = perlayer_matrix(
        pl, ["accepted", "windowMult", "resX", "resY", "pullX", "pullY",
             "pullAlpha", "pullBeta", "chi2IncRPhi", "chi2IncRZ"])

    Xa_all, tierA_names = _tierA_block(config)
    Xa = Xa_all[ds["aux"]["refit_mask"]]

    # substitution set: drop pooled sums (5..10) keep counters/kicks/seedMVA
    keep_ix = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16]
    X_subst = np.concatenate([X_spec[:, keep_ix], full_m], axis=1)
    subst_names = [names[i] for i in keep_ix] + full_n

    sets = {
        "spec17": (X_spec, names),
        "spec17_guard": (Xg, names),
        "spec17+pullL": (np.concatenate([Xg, pull_m], axis=1), names + pull_n),
        "spec17+occL": (np.concatenate([Xg, occ_m], axis=1), names + occ_n),
        "spec17+chi2L": (np.concatenate([Xg, chi_m], axis=1), names + chi_n),
        "spec17+fullL": (np.concatenate([Xg, full_m], axis=1), names + full_n),
        "perlayer_subst": (X_subst, subst_names),
        "spec17+tierA": (np.concatenate([Xg, Xa], axis=1), names + tierA_names),
    }

    aucs = {k: [] for k in sets}
    for seed in SEEDS:
        for k, (X, _) in sets.items():
            a, _m = _fit_auc(X, y, seed)
            aucs[k].append(a)
        print(f"seed {seed}: " + "  ".join(f"{k}={aucs[k][-1]:.4f}" for k in sets))

    stats = paired_auc_deltas(aucs, "spec17")

    # --- importances on the fullL model (seed 0) ---
    _, m_full = _fit_auc(sets["spec17+fullL"][0], y, 0)
    imp = m_full.get_booster().get_score(importance_type="gain")
    full_names = sets["spec17+fullL"][1]
    gain = {full_names[int(k[1:])]: float(v) for k, v in imp.items()}
    gain_sorted = dict(sorted(gain.items(), key=lambda kv: -kv[1])[:25])

    # --- logistic layer weights on per-layer pull^2 (standardized) ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    p2 = np.nan_to_num(pull_m, nan=0.0) ** 2
    Z = StandardScaler().fit_transform(p2)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Z, y)
    logit_w = dict(zip(pull_n, [float(w) for w in lr.coef_[0]]))

    # --- mutual information of per-layer |pull| ---
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(np.nan_to_num(np.abs(pull_m), nan=0.0), y,
                             discrete_features=False, random_state=0)
    mi_d = dict(zip(pull_n, [float(x) for x in mi]))

    out = {
        "config": config, "n_rows": int(len(y)), "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()), "seeds": list(SEEDS),
        "aucs": aucs, "paired_vs_spec17": stats,
        "xgb_gain_top25_fullL_seed0": gain_sorted,
        "logistic_layer_weights_pull2": logit_w,
        "mutual_info_abs_pull": mi_d,
        "notes": [
            "paired deltas share split seeds; delta_std is the honest error",
            "chi2 features log1p-clipped at 2e6 everywhere except raw spec17",
            "pull sign convention: raw pulls (can be negative); trees split on sign too",
        ],
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"perlayer_results_{config}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    print("\npaired deltas vs spec17 (mean +- std over seeds):")
    for k, s in stats.items():
        print(f"  {k:18s} auc={s['auc_mean']:.4f}+-{s['auc_std']:.4f}  "
              f"delta={s['delta_mean']:+.4f}+-{s['delta_std']:.4f}  "
              f"improved {s['n_seeds_improved']}/{len(SEEDS)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAAA")
