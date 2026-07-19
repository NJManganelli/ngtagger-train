"""Refit-aware track-quality BDT study on the SmartPixels digiRefit nano tables.

This is the offline counterpart to the Tier-2 digiRefit producer
(L1Trigger/Phase3SmartPixels): it trains an xgboost genuine-vs-fake classifier
using progressively richer feature sets that ablate what the BDT is allowed to
*see* of the refit, holding the labels (reference L1TTrack truth) and the
train/test split fixed via the 1:1 row alignment between the reference tracks
and every digiRefit variant table.

Tier / config matrix (13 trainings):
  Tier A  (config-independent): the classic 7 TRKQ_FEATURES built from the
          REFERENCE L1TTrack hw track-word columns exactly as trkquality.py.
  Tier B  (per config): A + refit info WITHOUT angles (crossing/hit counts,
          layer mask, window multiplicity/truncation, per-hit position pulls
          and residuals aggregated per track, and the refit-kick deltas
          variant-minus-reference on rInv/phi/tanl/z0/d0). chi2 totals are
          NOT used in B (they mix position + angle terms).
  Tier C  (per config): B + bending-angle (alpha) features, and the
          spxChi2IncRPhiTot total is now allowed.
  Tier D  (per config): C + beta features and spxChi2IncRZTot.

IMPORTANT caveat: the production ran useAngles=alphaBeta for ALL variants, so
the Kalman filter itself always used the full (alpha+beta) angle payload. The
tiers below ablate only the BDT *input features*, NOT the refit physics. The
refit tracks in every table already carry the full-angle correction; the A/B/C/D
progression measures how much of that already-baked-in improvement a
downstream quality BDT can recover from increasingly angle-aware inputs.
"""

from __future__ import annotations

import json
import os

import awkward as ak
import numpy as np
import uproot

# Reuse the exact reference-track feature machinery (two's-complement decode,
# missed-interior-layer counting, bit widths) so Tier A is byte-identical to
# the deployed TrackQuality GBDT training.
from ngtagger.train.trkquality import (
    K_TANL_SIZE,
    K_Z0_SIZE,
    TRKQ_FEATURES,
    nlaymiss_interior,
    twos_complement,
)

SMARTPIXELS_CONFIGS = ("AIII", "AAII", "AAAI", "AAAA")
# activeSP bitmask per config (L1-only up to all four TBPX layers)
CONFIG_ACTIVESP = {"AIII": "1000", "AAII": "1100", "AAAI": "1110", "AAAA": "1111"}
TIERS = ("A", "B", "C", "D")

_SENTINEL = -900.0  # values <= this are the -999 fill; test with (x > _SENTINEL)

# reference-track branches needed for Tier A + labels + kick reference
_REF_HW = ["hwTanl", "hwZ0", "hwBendChi2", "hwChi2RPhi", "hwChi2RZ", "hitPattern",
           "nStubs", "hwRinv", "hwPhi", "hwD0"]
_REF_FLOAT = ["rInv", "phi", "tanL", "z0", "d0", "pt", "eta"]
_REF_TRUTH = ["genuine", "looselyGenuine", "combinatoric", "unknown",
              "tpPt", "tpFromHardInteraction"]

# per-track extension columns from the variant track table (tier B core)
_VAR_EXT = ["spxRefitPerformed", "spxSeedCovOK", "spxNCrossings", "spxNAcceptedHits",
            "spxLayerHitMask", "spxMaxWindowMult", "spxAnyWindowTruncated", "spxNKFUpdates",
            "spxChi2IncRPhiTot", "spxChi2IncRZTot"]
# variant refit float parameters (for the variant-minus-reference kick deltas)
_VAR_FLOAT = ["rInv", "phi", "tanL", "z0", "d0"]
# per-hit link columns aggregated per track
_HIT_COLS = ["trackIdx", "layer", "windowMult", "windowTruncated", "hasAlpha", "hasBeta",
             "resX", "resY", "pullX", "pullY", "pullAlpha", "pullBeta",
             "sigAlpha", "sigBeta", "chi2IncRPhi", "chi2IncRZ"]


def _log1p_clip(x: np.ndarray) -> np.ndarray:
    """Compress the heavy chi2 tails (numerical-Jacobian blowups reach ~1e9):
    clip negatives to 0 then log1p. Keeps monotonicity, tames the range for
    the trees, and is finite for the passthrough 0 values."""
    return np.log1p(np.clip(x.astype(np.float64), 0.0, None)).astype(np.float32)


def load_refit_tables(files: list[str], config: str, track_table: str = "L1TTrack",
                      max_events: int | None = None):
    """Read the reference track table, the matching variant track table, and
    the per-hit link table for one config. Returns three flat awkward arrays
    (reference tracks, variant tracks, per-hit rows) all from the same events.

    Reference and variant track tables are 1:1 row aligned (verified 17324 ==
    17324 in the study sample); the per-hit table's trackIdx indexes into that
    shared per-event row order.
    """
    if config not in SMARTPIXELS_CONFIGS:
        raise ValueError(f"unknown config {config!r}; known: {SMARTPIXELS_CONFIGS}")
    ext = "Ext" if "Ext" in track_table else ""
    var_tbl = f"L1TSmartPixels{ext}TrackDigiRefit{config}"
    hit_tbl = f"L1TSmartPixels{ext}RefitHitDigiRefit{config}"

    def _load(prefix, cols):
        br = [f"{prefix}_{b}" for b in cols]
        arrs = uproot.concatenate([f"{f}:Events" for f in files], filter_name=br)
        if max_events is not None:
            arrs = arrs[:max_events]
        # rezip into a record keyed by the bare column name (drop the prefix)
        return ak.zip({b: arrs[f"{prefix}_{b}"] for b in cols}, depth_limit=1)

    ref = _load(track_table, _REF_HW + _REF_FLOAT + _REF_TRUTH)
    var = _load(var_tbl, _VAR_EXT + _VAR_FLOAT)
    hits = _load(hit_tbl, _HIT_COLS)
    return ref, var, hits


def _aggregate_hits_per_track(hits_flat: dict, n_tracks: int) -> dict:
    """Aggregate the per-hit link rows down to one record per (global) track
    row, sentinel-excluded. Returns dict of length-n_tracks numpy arrays.

    Aggregations: sum(pullX^2), sum(pullY^2), max|resX|, max|resY| (position,
    always valid); sum(pullAlpha^2), n(hasAlpha), mean(sigAlpha) and the beta
    analogues (angle, sentinel-excluded); n_hits. trackIdx is the LOCAL
    (per-event) index, so the caller must pass GLOBAL indices already offset.
    """
    idx = hits_flat["_gidx"].astype(np.int64)
    out = {}

    def _sum_sq(col, valid=None):
        v = hits_flat[col].astype(np.float64)
        m = np.ones(len(v), bool) if valid is None else valid
        acc = np.zeros(n_tracks, np.float64)
        np.add.at(acc, idx[m], v[m] ** 2)
        return acc

    def _count(mask):
        acc = np.zeros(n_tracks, np.float64)
        np.add.at(acc, idx[mask], 1.0)
        return acc

    def _max_abs(col):
        v = np.abs(hits_flat[col].astype(np.float64))
        acc = np.zeros(n_tracks, np.float64)
        np.maximum.at(acc, idx, v)
        return acc

    def _mean(col, valid):
        v = hits_flat[col].astype(np.float64)
        s = np.zeros(n_tracks, np.float64)
        c = np.zeros(n_tracks, np.float64)
        np.add.at(s, idx[valid], v[valid])
        np.add.at(c, idx[valid], 1.0)
        return np.divide(s, c, out=np.zeros_like(s), where=c > 0)

    has_a = hits_flat["hasAlpha"].astype(bool) & (hits_flat["pullAlpha"] > _SENTINEL)
    has_b = hits_flat["hasBeta"].astype(bool) & (hits_flat["pullBeta"] > _SENTINEL)
    sig_a_ok = hits_flat["sigAlpha"] > _SENTINEL
    sig_b_ok = hits_flat["sigBeta"] > _SENTINEL

    out["hit_nhits"] = _count(np.ones(len(idx), bool))
    out["hit_sumPullX2"] = _sum_sq("pullX")
    out["hit_sumPullY2"] = _sum_sq("pullY")
    out["hit_maxAbsResX"] = _max_abs("resX")
    out["hit_maxAbsResY"] = _max_abs("resY")
    out["hit_sumChi2RPhi"] = np.zeros(n_tracks)  # filled below (log later)
    acc = np.zeros(n_tracks, np.float64)
    np.add.at(acc, idx, np.clip(hits_flat["chi2IncRPhi"].astype(np.float64), 0, None))
    out["hit_sumChi2RPhi"] = acc
    # angle (alpha) aggregates
    out["hit_sumPullAlpha2"] = _sum_sq("pullAlpha", valid=has_a)
    out["hit_nHasAlpha"] = _count(has_a)
    out["hit_meanSigAlpha"] = _mean("sigAlpha", sig_a_ok)
    # beta aggregates
    out["hit_sumPullBeta2"] = _sum_sq("pullBeta", valid=has_b)
    out["hit_nHasBeta"] = _count(has_b)
    out["hit_meanSigBeta"] = _mean("sigBeta", sig_b_ok)
    return out


def _tier_a_features(ref_flat: dict):
    """The classic 7 TRKQ_FEATURES from the reference hw track word."""
    return np.stack([
        twos_complement(ref_flat["hwTanl"], K_TANL_SIZE).astype(np.float32),
        twos_complement(ref_flat["hwZ0"], K_Z0_SIZE).astype(np.float32),
        ref_flat["hwBendChi2"].astype(np.float32),
        ref_flat["nStubs"].astype(np.float32),
        nlaymiss_interior(ref_flat["hitPattern"]).astype(np.float32),
        ref_flat["hwChi2RPhi"].astype(np.float32),
        ref_flat["hwChi2RZ"].astype(np.float32),
    ], axis=1), list(TRKQ_FEATURES)


def build_refitq_dataset(ref, var, hits, tier: str, config: str, label: str = "genuine",
                         require_truth: bool = True):
    """Build (X, y, feature_names, info) for one tier/config cell.

    ref/var are 1:1 row-aligned track tables; hits is the per-hit link table
    with a per-event trackIdx into that shared row order. Flattening is done
    event-major so a global track offset can be applied to trackIdx.
    """
    tier = tier.upper()
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; known {TIERS}")

    # per-event counts to build global offsets for the hit trackIdx
    counts = ak.to_numpy(ak.num(ref["genuine"]))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n_tracks = int(offsets[-1])

    ref_flat = {b: ak.to_numpy(ak.flatten(ref[b])) for b in (_REF_HW + _REF_FLOAT + _REF_TRUTH)}
    var_flat = {b: ak.to_numpy(ak.flatten(var[b])) for b in (_VAR_EXT + _VAR_FLOAT)}

    y = ref_flat[label].astype(np.int64)
    if require_truth and ref_flat["unknown"].all():
        raise RuntimeError(
            "all reference tracks are truth-'unknown': the TTTrackAssociator did "
            "not run; produce the sample with the withGen track-truth sequence.")
    if require_truth and y.sum() == 0:
        raise RuntimeError(
            f"no positive ('{label}') reference tracks: refit-quality training "
            "needs a genuine/fake mix (truth-required mode fails loudly).")

    X, names = _tier_a_features(ref_flat)
    if tier == "A":
        info = {"n_tracks": n_tracks, "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}
        return X.astype(np.float32), y, names, info

    # --- Tier B and up need the refit info ---
    # global-index the hit rows
    hit_flat = {b: ak.to_numpy(ak.flatten(hits[b])) for b in _HIT_COLS}
    ev_of_hit = np.repeat(np.arange(len(counts)), ak.to_numpy(ak.num(hits["trackIdx"])))
    hit_flat["_gidx"] = hit_flat["trackIdx"].astype(np.int64) + offsets[ev_of_hit]
    agg = _aggregate_hits_per_track(hit_flat, n_tracks)

    # per-track extension columns (tier B, sans chi2 totals)
    ext_cols = {
        "spxNCrossings": var_flat["spxNCrossings"].astype(np.float32),
        "spxNAcceptedHits": var_flat["spxNAcceptedHits"].astype(np.float32),
        "spxLayerHitMask": var_flat["spxLayerHitMask"].astype(np.float32),
        "spxMaxWindowMult": var_flat["spxMaxWindowMult"].astype(np.float32),
        "spxAnyWindowTruncated": var_flat["spxAnyWindowTruncated"].astype(np.float32),
        "spxNKFUpdates": var_flat["spxNKFUpdates"].astype(np.float32),
    }
    # refit kicks: variant-minus-reference on the shared helix parameters
    kicks = {
        "dRinv": (var_flat["rInv"] - ref_flat["rInv"]).astype(np.float32),
        "dPhi": (var_flat["phi"] - ref_flat["phi"]).astype(np.float32),
        "dTanl": (var_flat["tanL"] - ref_flat["tanL"]).astype(np.float32),
        "dZ0": (var_flat["z0"] - ref_flat["z0"]).astype(np.float32),
        "dD0": (var_flat["d0"] - ref_flat["d0"]).astype(np.float32),
    }
    b_hit = {
        "hit_nhits": agg["hit_nhits"].astype(np.float32),
        "hit_sumPullX2": agg["hit_sumPullX2"].astype(np.float32),
        "hit_sumPullY2": agg["hit_sumPullY2"].astype(np.float32),
        "hit_maxAbsResX": agg["hit_maxAbsResX"].astype(np.float32),
        "hit_maxAbsResY": agg["hit_maxAbsResY"].astype(np.float32),
    }
    b_block = {**ext_cols, **kicks, **b_hit}

    c_block = {
        "spxChi2IncRPhiTot": _log1p_clip(var_flat["spxChi2IncRPhiTot"]),
        "hit_sumPullAlpha2": agg["hit_sumPullAlpha2"].astype(np.float32),
        "hit_nHasAlpha": agg["hit_nHasAlpha"].astype(np.float32),
        "hit_meanSigAlpha": agg["hit_meanSigAlpha"].astype(np.float32),
    }
    d_block = {
        "spxChi2IncRZTot": _log1p_clip(var_flat["spxChi2IncRZTot"]),
        "hit_sumPullBeta2": agg["hit_sumPullBeta2"].astype(np.float32),
        "hit_nHasBeta": agg["hit_nHasBeta"].astype(np.float32),
        "hit_meanSigBeta": agg["hit_meanSigBeta"].astype(np.float32),
    }

    blocks = dict(b_block)
    if tier in ("C", "D"):
        blocks.update(c_block)
    if tier == "D":
        blocks.update(d_block)

    add_names = list(blocks.keys())
    add = np.stack([blocks[n] for n in add_names], axis=1).astype(np.float32)
    X = np.concatenate([X, add], axis=1)
    names = names + add_names
    info = {"n_tracks": n_tracks, "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}
    return X, y, names, info


def _split(n: int, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_fraction)
    return idx[n_test:], idx[:n_test]  # train, test


def _xgb_params(user: dict | None):
    # small trees for ~17k rows with a few-hundred-fake minority class
    params = {
        "n_estimators": 80,
        "max_depth": 3,
        "learning_rate": 0.15,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "early_stopping_rounds": 15,
    }
    params.update(user or {})
    return params


def train_one(ref, var, hits, tier: str, config: str, output_dir: str,
              label: str = "genuine", test_fraction: float = 0.2, seed: int = 0,
              xgb_params: dict | None = None, log_mlflow: bool = True,
              scale_pos_weight: bool = True):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    os.makedirs(output_dir, exist_ok=True)
    X, y, names, info = build_refitq_dataset(ref, var, hits, tier, config, label=label)
    train, test = _split(len(X), test_fraction, seed)

    params = _xgb_params(xgb_params)
    if scale_pos_weight:
        n_pos = max(int(y[train].sum()), 1)
        n_neg = max(int((y[train] == 0).sum()), 1)
        params.setdefault("scale_pos_weight", n_neg / n_pos)

    model = xgb.XGBClassifier(**params, random_state=seed)
    model.fit(X[train], y[train], eval_set=[(X[test], y[test])], verbose=False)
    proba = model.predict_proba(X[test])[:, 1]
    # AUC is on genuine-vs-fake; here positive=label, so the fake-rejection AUC
    # is 1-AUC symmetric. Report the standard label-positive AUC.
    auc = float(roc_auc_score(y[test], proba))

    tag = "A" if tier == "A" else f"{tier}-{config}"
    print(f"refitq[{tag}]: test AUC = {auc:.4f}  "
          f"(nfeat={len(names)}, n_train={len(train)}, n_test={len(test)}, "
          f"pos={info['n_pos']}, neg={info['n_neg']})")

    model.save_model(os.path.join(output_dir, f"refitq_{tag}_xgb.json"))
    meta = {"tier": tier, "config": (None if tier == "A" else config),
            "activeSP": (None if tier == "A" else CONFIG_ACTIVESP[config]),
            "label": label, "features": names, "n_features": len(names),
            "test_auc": auc, "n_pos": info["n_pos"], "n_neg": info["n_neg"],
            "n_train": len(train), "n_test": len(test), "seed": seed,
            "params": {k: v for k, v in params.items()},
            "caveat": ("production used useAngles=alphaBeta for ALL variants; tiers "
                       "ablate BDT INPUT FEATURES only, not the refit physics.")}
    with open(os.path.join(output_dir, f"refitq_{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if log_mlflow:
        try:
            import mlflow

            mlflow.set_experiment("ngtagger-refitquality")
            with mlflow.start_run(run_name=f"refitq-{tag}"):
                mlflow.log_params(params)
                mlflow.log_param("tier", tier)
                mlflow.log_param("config", meta["config"])
                mlflow.log_param("n_features", len(names))
                mlflow.log_metric("test_auc", auc)
                mlflow.log_metric("n_pos", info["n_pos"])
                mlflow.log_metric("n_neg", info["n_neg"])
                mlflow.log_artifact(os.path.join(output_dir, f"refitq_{tag}_meta.json"))
        except Exception:
            pass

    return model, auc, meta


def auc_std_over_seeds(ref, var, hits, tier: str, config: str, label: str = "genuine",
                       seeds=(0, 1, 2, 3, 4), test_fraction: float = 0.2,
                       xgb_params: dict | None = None):
    """AUC mean/std over several deterministic split seeds for one cell, to
    quantify the (large, few-hundred-fake) statistical uncertainty."""
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    X, y, _, _ = build_refitq_dataset(ref, var, hits, tier, config, label=label)
    aucs = []
    for s in seeds:
        train, test = _split(len(X), test_fraction, s)
        params = _xgb_params(xgb_params)
        n_pos = max(int(y[train].sum()), 1)
        n_neg = max(int((y[train] == 0).sum()), 1)
        params.setdefault("scale_pos_weight", n_neg / n_pos)
        m = xgb.XGBClassifier(**params, random_state=s)
        m.fit(X[train], y[train], eval_set=[(X[test], y[test])], verbose=False)
        aucs.append(float(roc_auc_score(y[test], m.predict_proba(X[test])[:, 1])))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def train_matrix(files: list[str], output_dir: str, track_table: str = "L1TTrack",
                 label: str = "genuine", max_events: int | None = None,
                 test_fraction: float = 0.2, seed: int = 0,
                 configs=SMARTPIXELS_CONFIGS, xgb_params: dict | None = None):
    """Train the full 1 + 3x4 = 13-cell matrix and return a nested AUC dict:
    {'A': auc, 'B': {config: auc, ...}, 'C': {...}, 'D': {...}}."""
    os.makedirs(output_dir, exist_ok=True)
    results = {"A": None, "B": {}, "C": {}, "D": {}}
    meta_all = {}

    # Tier A is config-independent: train once off the first config's tables
    ref0, var0, hits0 = load_refit_tables(files, configs[0], track_table, max_events)
    _, auc_a, meta_a = train_one(ref0, var0, hits0, "A", configs[0], output_dir,
                                 label=label, test_fraction=test_fraction, seed=seed,
                                 xgb_params=xgb_params)
    results["A"] = auc_a
    meta_all["A"] = meta_a

    for cfg in configs:
        ref, var, hits = load_refit_tables(files, cfg, track_table, max_events)
        for tier in ("B", "C", "D"):
            _, auc, meta = train_one(ref, var, hits, tier, cfg, output_dir,
                                     label=label, test_fraction=test_fraction, seed=seed,
                                     xgb_params=xgb_params)
            results[tier][cfg] = auc
            meta_all[f"{tier}-{cfg}"] = meta

    with open(os.path.join(output_dir, "auc_matrix.json"), "w") as f:
        json.dump({"results": results, "meta": meta_all,
                   "label": label, "configs": list(configs)}, f, indent=2)
    print("\nAUC matrix (rows=tier, cols=config):")
    print(f"  A (baseline): {results['A']:.4f}")
    header = "       " + "  ".join(f"{c:>7}" for c in configs)
    print(header)
    for tier in ("B", "C", "D"):
        row = f"  {tier}:  " + "  ".join(f"{results[tier][c]:7.4f}" for c in configs)
        print(row)
    return results, meta_all


def export_conifer(model_dir: str, tag: str, output_dir: str | None = None, backend: str = "cpp"):
    """Stub call-through to the conifer exporter used for the deployed
    TrackQuality GBDT; converts one trained refit-quality xgboost model to a
    conifer json (same FileInPath-deployable format)."""
    import conifer
    import xgboost as xgb

    output_dir = output_dir or model_dir
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, f"refitq_{tag}_xgb.json"))
    cfg = (conifer.backends.cpp.auto_config() if backend == "cpp"
           else conifer.backends.xilinxhls.auto_config())
    cfg["OutputDir"] = os.path.join(output_dir, f"conifer_{tag}_{backend}")
    cnf = conifer.converters.convert_from_xgboost(booster, cfg)
    cnf.save(os.path.join(output_dir, f"refitq_{tag}_conifer.json"))
    print(f"conifer model written to {output_dir}/refitq_{tag}_conifer.json")
    return cnf
