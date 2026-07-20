"""Part A.1c / Part B.1: transmitted-subset information retention at track level.

The Part-B design question 'how much can the trkquality/refitq MVA score proxy
for the raw refit features across the hardware boundary' is exactly TS0 vs
TS1 vs TS2 (RefitSidecarSpec §3). The candidate/jet tables needed for the
jet-level version are absent from this nano, but the track-level version
bounds it: train the in-producer scorer on the TRAIN split, then train a
downstream consumer BDT on the TEST-side features at each transmission tier
and compare fake-rejection AUC.

Tiers (per spec §3, quantizers reproduced bit-exactly):
  TS0   score only (+ seedTrkMVA1, which crosses in the track word anyway)
  TS1   score + unpacked 16-bit compact word
        (layerHitMask 4b | q(chi2RPhiTot) 4b | q(chi2RZTot) 4b | occ 3b)
  TS1b  score + compact word + PER-LAYER 2-bit occupancy (proposed layout
        variant: spends the 4+ spare/reserved bits on occ_L1..L4)
  TS2   score + full spec17 floats (upper bound, studies-only)

The downstream consumer here is a small BDT (proxy for the tagger's ability
to exploit per-constituent refit info).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelspace_common import (  # noqa: E402
    SEEDS, load_dataset, paired_auc_deltas, perlayer_matrix, split,
)


def q_chi2(c: np.ndarray) -> np.ndarray:
    """spec §3: q(c) = clamp(round(2 * log2(1 + c)), 0, 15)."""
    c = np.clip(c.astype(np.float64), 0.0, None)
    return np.clip(np.round(2.0 * np.log2(1.0 + c)), 0, 15).astype(np.float32)


def q_occ(m: np.ndarray, bits: int = 3) -> np.ndarray:
    """spec §3: occ = clamp(floor(log2(1 + maxWindowMult)), 0, 2^bits - 1)."""
    m = np.clip(m.astype(np.float64), 0.0, None)
    return np.clip(np.floor(np.log2(1.0 + m)), 0, (1 << bits) - 1).astype(np.float32)


def _train_scorer(X17, y, train_idx, seed):
    import xgboost as xgb

    p = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.2,
         "objective": "binary:logistic", "eval_metric": "auc"}
    n_pos = max(int(y[train_idx].sum()), 1)
    n_neg = max(int((y[train_idx] == 0).sum()), 1)
    p["scale_pos_weight"] = n_neg / n_pos
    m = xgb.XGBClassifier(**p, random_state=seed)
    m.fit(X17[train_idx], y[train_idx], verbose=False)
    return m


def _consumer_auc(X, y, train_idx, test_idx, seed):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    p = {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.15,
         "objective": "binary:logistic", "eval_metric": "auc",
         "early_stopping_rounds": 15}
    n_pos = max(int(y[train_idx].sum()), 1)
    n_neg = max(int((y[train_idx] == 0).sum()), 1)
    p["scale_pos_weight"] = n_neg / n_pos
    m = xgb.XGBClassifier(**p, random_state=seed)
    m.fit(X[train_idx], y[train_idx], eval_set=[(X[test_idx], y[test_idx])], verbose=False)
    return float(roc_auc_score(y[test_idx], m.predict_proba(X[test_idx])[:, 1]))


def main(config: str = "AAAA"):
    from sklearn.metrics import roc_auc_score

    ds = load_dataset(config)
    X17, y, names = ds["X_spec"], ds["y"], ds["spec_names"]
    i_mask = names.index("layerHitMask")
    i_occ = names.index("maxWindowMult")
    i_rphi = names.index("chi2IncRPhiTot")
    i_rz = names.index("chi2IncRZTot")
    i_seedmva = names.index("seedTrkMVA1")

    occL, _ = perlayer_matrix(ds["per_layer"], ["windowMult"])
    occL2b = np.stack([q_occ(np.nan_to_num(occL[:, k], nan=0.0), bits=2)
                       for k in range(4)], axis=1)

    aucs = {k: [] for k in ("score_alone", "TS0", "TS1", "TS1b", "TS2")}
    for seed in SEEDS:
        train_idx, test_idx = split(len(y), seed)
        scorer = _train_scorer(X17, y, train_idx, seed)
        score = scorer.predict_proba(X17)[:, 1].astype(np.float32)

        aucs["score_alone"].append(float(roc_auc_score(y[test_idx], score[test_idx])))

        seedmva = X17[:, [i_seedmva]]
        ts0 = np.concatenate([score[:, None], seedmva], axis=1)
        compact = np.stack([
            X17[:, i_mask],                # 4 bits, already integer-valued
            q_chi2(X17[:, i_rphi]),        # 4 bits
            q_chi2(X17[:, i_rz]),          # 4 bits
            q_occ(X17[:, i_occ]),          # 3 bits
        ], axis=1)
        ts1 = np.concatenate([ts0, compact], axis=1)
        ts1b = np.concatenate([ts1, occL2b], axis=1)
        ts2 = np.concatenate([ts0, X17], axis=1)

        aucs["TS0"].append(_consumer_auc(ts0, y, train_idx, test_idx, seed))
        aucs["TS1"].append(_consumer_auc(ts1, y, train_idx, test_idx, seed))
        aucs["TS1b"].append(_consumer_auc(ts1b, y, train_idx, test_idx, seed))
        aucs["TS2"].append(_consumer_auc(ts2, y, train_idx, test_idx, seed))
        print(f"seed {seed}: " + "  ".join(f"{k}={aucs[k][-1]:.4f}" for k in aucs))

    stats = paired_auc_deltas(aucs, "TS0")
    out = {"config": config, "n_rows": int(len(y)), "n_pos": int(y.sum()),
           "n_neg": int((y == 0).sum()), "seeds": list(SEEDS),
           "aucs": aucs, "paired_vs_TS0": stats,
           "notes": [
               "scorer = spec17 60x3 BDT trained on the train split only",
               "quantizers reproduce RefitSidecarSpec §3 bit-exactly",
               "TS1b = proposed variant: +4x2-bit per-layer occupancy",
               "track-level bound for the jet-level TS question (no cand tables in this nano)",
           ]}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"transmitted_subset_results_{config}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    for k, s in stats.items():
        print(f"  {k:12s} auc={s['auc_mean']:.4f}+-{s['auc_std']:.4f}  "
              f"delta={s['delta_mean']:+.4f}+-{s['delta_std']:.4f}  "
              f"improved {s['n_seeds_improved']}/{len(SEEDS)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAAA")
