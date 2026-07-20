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

# ---------------------------------------------------------------------------
# SPEC-ORDER feature contract (RefitSidecarSpec.md §6a, REFIT_BDT_FEATURES v0)
#
# This is DISTINCT from the tier A/B/C/D study above. The producer
# (L1SmartPixelsTrackProducer) computes THESE 17 features in-flight per refit
# track and evaluates the conifer BDT on them; the conifer JSON's n_features
# must equal 17 or the producer throws at load. train-refitquality
# --export-conifer builds this exact matrix and exports the deployable model.
#
# Producer semantics matched exactly (see the feature-vector block in
# L1SmartPixelsTrackProducer.cc):
#  - features 0-4 : per-track sidecar counters (nCrossings, nAcceptedHits,
#                   layerHitMask, maxWindowMult, anyWindowTruncated 0/1)
#  - features 5-8 : sum of pull^2 over APPLIED scalar updates (sentinel-gated:
#                   the producer only accumulates where the update was applied;
#                   in nano those pulls are non-sentinel, so we sum pull^2 over
#                   hit rows with pull > -900)
#  - features 9-10: chi2IncRPhiTot / chi2IncRZTot (post-guard; sentinel -> 0;
#                   passthrough tracks carry 0 - but passthrough tracks are
#                   never scored, so we train/parity on refit tracks only)
#  - features 11-15: refit-minus-seed deltas on (rInv, phi, tanl, z0, d0) =
#                    variant track minus reference track, 1:1 row aligned
#  - feature 16   : seedTrkMVA1 = the INPUT (reference) track's trkMVA1 column
#
# Features are RAW (no log/clip transform - unlike the tier-study chi2 columns);
# the KF numerical guards (spec §6b) bound the chi2 at source on post-guard
# productions. NOTE: this PU nano was produced PRE-guard, so its chi2 columns
# still contain the unguarded tail; we train on the data as-is (parity is about
# the evaluation math, not training-sample perfection).
REFIT_SPEC_FEATURES = [
    "nCrossings", "nAcceptedHits", "layerHitMask", "maxWindowMult",
    "anyWindowTruncated", "sumPullX2", "sumPullY2", "sumPullAlpha2",
    "sumPullBeta2", "chi2IncRPhiTot", "chi2IncRZTot",
    "dRinv", "dPhi", "dTanl", "dZ0", "dD0", "seedTrkMVA1",
]
# REFIT_BDT_FEATURES v1 (spec §6a v1): the v0 17 plus the classic-7 TrackQuality
# hw features of the INPUT (reference) track at indices 17-23, decoded EXACTLY as
# trkquality.py does on the raw nano hw columns (two's-complement for signed
# fields, raw bin index for the binned chi2 quantities, nStubs, interior-missed
# layers over the hit pattern). Mirrors the producer's in-flight v1 assembly
# bit-for-bit (the producer calls getTanlBits()/getZ0Bits()/... = these nano
# columns and the same decode helpers). Order == TRKQ_FEATURES.
REFIT_SPEC_FEATURES_V1 = REFIT_SPEC_FEATURES + list(TRKQ_FEATURES)  # 17 + 7 = 24
REFIT_SPEC_VERSION = 0    # REFIT_BDT_FEATURES v0 (spec §6a); 17 features
REFIT_SPEC_VERSION_V1 = 1  # REFIT_BDT_FEATURES v1 (spec §6a v1); 24 features
_SPEC_NFEAT = {0: 17, 1: 24}

_SENTINEL = -900.0  # values <= this are the -999 fill; test with (x > _SENTINEL)

# reference-track branches needed for Tier A + labels + kick reference
_REF_HW = ["hwTanl", "hwZ0", "hwBendChi2", "hwChi2RPhi", "hwChi2RZ", "hitPattern",
           "nStubs", "hwRinv", "hwPhi", "hwD0"]
_REF_FLOAT = ["rInv", "phi", "tanL", "z0", "d0", "pt", "eta", "trkMVA1"]
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


# ===========================================================================
# SPEC-ORDER path: producer-contract 17-feature model + conifer export + parity
# ===========================================================================

def build_spec_dataset(ref, var, hits, config: str, label: str = "genuine",
                       refit_only: bool = True, require_truth: bool = True,
                       seed_npar: int = 5, track_npar: int | None = None,
                       spec_version: int = 0):
    """Build the SPEC-ORDER (RefitSidecarSpec §6a) feature matrix that the
    producer evaluates in-flight. spec_version=0 -> the 17-feature v0 vector;
    spec_version=1 -> the 24-feature v1 vector (v0 + the classic-7 TrackQuality
    hw features of the reference track, indices 17-23, decoded exactly as
    trkquality.py). Returns (X float32 [N,nfeat], y int64 [N], names, aux) where
    aux carries the refit-performed mask and truth columns.

    Row order is the shared reference/variant per-event flatten order; the
    per-hit link table's trackIdx indexes into it (global-offset applied). When
    refit_only=True, rows are restricted to spxRefitPerformed==1 tracks (the
    producer never scores passthrough tracks - their trkMVA1 passes through).

    seed_npar / track_npar: the producer seeds the KF d0 as
        aSeed[4] = (seedNPar==5 && track.nFitPars()==5) ? track.d0 : 0.0
    so feature 15 (dD0 = refit_d0 - aSeed[4]) subtracts 0.0 for 4-par seeds.
    track_npar defaults to 4 for the prompt L1TTrack collection and 5 for the
    Ext collection (nFitPars is not persisted in nano). Getting this wrong puts
    a constant d0-LSB offset on feature 15 that flips near-threshold tracks.
    """
    if track_npar is None:
        track_npar = 4  # prompt L1TTrack default; callers pass 5 for Ext tables
    use_ref_d0_as_seed = (seed_npar == 5 and track_npar == 5)
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

    # --- per-hit sum-of-squared-pull (features 5-8), applied-gated exactly as
    #     the producer: accumulate pull^2 only where the pull is non-sentinel. ---
    hit_flat = {b: ak.to_numpy(ak.flatten(hits[b])) for b in _HIT_COLS}
    ev_of_hit = np.repeat(np.arange(len(counts)), ak.to_numpy(ak.num(hits["trackIdx"])))
    gidx = hit_flat["trackIdx"].astype(np.int64) + offsets[ev_of_hit]

    def _sum_sq_applied(col):
        v = hit_flat[col].astype(np.float64)
        m = v > _SENTINEL  # non-sentinel == applied update in the producer
        acc = np.zeros(n_tracks, np.float64)
        np.add.at(acc, gidx[m], v[m] ** 2)
        return acc

    sumPullX2 = _sum_sq_applied("pullX")
    sumPullY2 = _sum_sq_applied("pullY")
    sumPullAlpha2 = _sum_sq_applied("pullAlpha")
    sumPullBeta2 = _sum_sq_applied("pullBeta")

    # chi2 totals: producer already maps sentinel -> 0 and passthrough -> 0.
    chi2rphi = var_flat["spxChi2IncRPhiTot"].astype(np.float64)
    chi2rz = var_flat["spxChi2IncRZTot"].astype(np.float64)
    chi2rphi = np.where(chi2rphi > _SENTINEL, chi2rphi, 0.0)
    chi2rz = np.where(chi2rz > _SENTINEL, chi2rz, 0.0)

    cols = [
        var_flat["spxNCrossings"].astype(np.float32),          # 0
        var_flat["spxNAcceptedHits"].astype(np.float32),       # 1
        var_flat["spxLayerHitMask"].astype(np.float32),        # 2
        var_flat["spxMaxWindowMult"].astype(np.float32),       # 3
        var_flat["spxAnyWindowTruncated"].astype(np.float32),  # 4
        sumPullX2.astype(np.float32),                          # 5
        sumPullY2.astype(np.float32),                          # 6
        sumPullAlpha2.astype(np.float32),                     # 7
        sumPullBeta2.astype(np.float32),                      # 8
        chi2rphi.astype(np.float32),                          # 9
        chi2rz.astype(np.float32),                            # 10
        (var_flat["rInv"] - ref_flat["rInv"]).astype(np.float32),  # 11 dRinv
        (var_flat["phi"] - ref_flat["phi"]).astype(np.float32),    # 12 dPhi
        (var_flat["tanL"] - ref_flat["tanL"]).astype(np.float32),  # 13 dTanl
        (var_flat["z0"] - ref_flat["z0"]).astype(np.float32),      # 14 dZ0
        # 15 dD0 = refit_d0 - aSeed[4]; aSeed[4] = ref.d0 only for a 5-par seed
        # (seedNPar==5 && track 5-par), else 0.0 - matches the producer seeding.
        (var_flat["d0"] - (ref_flat["d0"] if use_ref_d0_as_seed
                           else np.zeros_like(ref_flat["d0"]))).astype(np.float32),  # 15 dD0
        ref_flat["trkMVA1"].astype(np.float32),               # 16 seedTrkMVA1
    ]
    X = np.stack(cols, axis=1).astype(np.float32)
    names = list(REFIT_SPEC_FEATURES)

    # v1 (spec §6a v1): append the reference track's classic-7 TRKQ_FEATURES at
    # indices 17-23, decoded EXACTLY as trkquality.py (same helpers, imported at
    # top). This is bit-for-bit the producer's in-flight v1 vector: the producer
    # reads getTanlBits()/getZ0Bits()/getBendChi2Bits()/getChi2RPhiBits()/
    # getChi2RZBits() = these raw nano hw columns, nStubs = getStubRefs().size(),
    # and hitPattern() for the interior-missed-layer count.
    if spec_version == 1:
        v1cols = [
            twos_complement(ref_flat["hwTanl"], K_TANL_SIZE).astype(np.float32),  # 17 tanl
            twos_complement(ref_flat["hwZ0"], K_Z0_SIZE).astype(np.float32),      # 18 z0_scaled
            ref_flat["hwBendChi2"].astype(np.float32),                            # 19 bendchi2_bin
            ref_flat["nStubs"].astype(np.float32),                                # 20 nstub
            nlaymiss_interior(ref_flat["hitPattern"]).astype(np.float32),         # 21 nlaymiss_interior
            ref_flat["hwChi2RPhi"].astype(np.float32),                            # 22 chi2rphi_bin
            ref_flat["hwChi2RZ"].astype(np.float32),                              # 23 chi2rz_bin
        ]
        X = np.concatenate([X, np.stack(v1cols, axis=1).astype(np.float32)], axis=1)
        names = list(REFIT_SPEC_FEATURES_V1)
    elif spec_version != 0:
        raise ValueError(f"build_spec_dataset: spec_version must be 0 or 1, got {spec_version}")

    refit_mask = (var_flat["spxRefitPerformed"].astype(np.int64) == 1)
    aux = {"refit_mask": refit_mask, "n_tracks": n_tracks,
           "tpFromHardInteraction": ref_flat["tpFromHardInteraction"],
           "genuine": ref_flat["genuine"].astype(np.int64),
           "unknown": ref_flat["unknown"].astype(np.int64),
           "pt": ref_flat["pt"], "eta": ref_flat["eta"]}

    if refit_only:
        X = X[refit_mask]
        y = y[refit_mask]
        for k in ("tpFromHardInteraction", "genuine", "unknown", "pt", "eta"):
            aux[k] = aux[k][refit_mask]
        aux["n_selected"] = int(refit_mask.sum())
    return X, y, names, aux


def conifer_json_walk(model_json: dict, X: np.ndarray) -> np.ndarray:
    """Faithful float32 reproduction of conifer.h v1.7 BDT::decision_function
    for n_classes<=2 (collapsed to 1), useAddTree=false, T=U=float - i.e. the
    exact math the producer runs via conifer::BDT<float,float>. This is the
    parity reference for the in-producer score.

    Per conifer.h: split(x[feature], threshold); comparison True -> children_left
    else children_right; leaf when feature==-2; value accumulated in float32
    seeded from float32(init_predict[0]); result *= float32(norm).
    """
    conv = model_json.get("splitting_convention", "<=")
    if conv == "<":
        cmp = lambda a, b: a < b
    elif conv == "<=":
        cmp = lambda a, b: a <= b
    else:
        raise ValueError(f"unsupported splitting_convention {conv!r}")
    init = np.float32(model_json["init_predict"][0])
    norm = np.float32(model_json["norm"])
    trees = model_json["trees"]  # outer: tree; inner: class (use class 0)

    Xf = np.ascontiguousarray(X, dtype=np.float32)
    out = np.empty(len(Xf), np.float32)
    for r in range(len(Xf)):
        x = Xf[r]
        acc = np.float32(init)
        for t in trees:
            sub = t[0]
            feat = sub["feature"]
            cl = sub["children_left"]
            cr = sub["children_right"]
            thr = sub["threshold"]
            val = sub["value"]
            i = 0
            while feat[i] != -2:
                comparison = cmp(x[feat[i]], np.float32(thr[i]))
                i = cl[i] if comparison else cr[i]
            acc = np.float32(acc + np.float32(val[i]))
        out[r] = np.float32(acc * norm)
    return out


def _spec_xgb_params(user: dict | None):
    # trkquality-like params; small binary model on ~15k refit rows with a
    # few-hundred-fake minority class (statistically weak by design).
    params = {
        "n_estimators": 60,
        "max_depth": 3,
        "learning_rate": 0.2,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "early_stopping_rounds": 10,
    }
    params.update(user or {})
    return params


def train_refitq_spec(files: list[str], output_dir: str, config: str = "AAAA",
                      track_table: str = "L1TTrack", label: str = "genuine",
                      max_events: int | None = None, test_fraction: float = 0.2,
                      seed: int = 0, xgb_params: dict | None = None,
                      export_conifer: bool = True,
                      conifer_name: str = "refitq_conifer_v0",
                      provenance: str = "", scale_pos_weight: bool = True,
                      spec_version: int = 0,
                      seed_npar: int = 5, track_npar: int | None = None):
    """Train the SPEC-ORDER refit-quality BDT and export it in conifer JSON
    matching the producer contract. spec_version=0 -> the 17-feature v0 vector;
    spec_version=1 -> the 24-feature v1 vector (v0 + the classic-7 TrackQuality hw
    features of the reference track). The exported n_features (17 or 24) is
    validated against the version; the producer accepts both and selects the
    assembly by the model's n_features.

    Writes: <conifer_name>.json (deployable conifer model, raw-margin/logit),
    <conifer_name>_meta.json (feature order + spec version + provenance),
    refitq_spec_xgb.json (the raw xgboost model). Returns
    (model, conifer_json_path, meta).
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    if spec_version not in _SPEC_NFEAT:
        raise ValueError(f"spec_version must be 0 or 1, got {spec_version}")
    nfeat_expected = _SPEC_NFEAT[spec_version]
    os.makedirs(output_dir, exist_ok=True)
    ref, var, hits = load_refit_tables(files, config, track_table, max_events)
    X, y, names, aux = build_spec_dataset(ref, var, hits, config, label=label,
                                          seed_npar=seed_npar, track_npar=track_npar,
                                          spec_version=spec_version)
    assert len(names) == nfeat_expected, f"spec feature count {len(names)} != {nfeat_expected}"

    train, test = _split(len(X), test_fraction, seed)
    params = _spec_xgb_params(xgb_params)
    if scale_pos_weight:
        n_pos = max(int(y[train].sum()), 1)
        n_neg = max(int((y[train] == 0).sum()), 1)
        params.setdefault("scale_pos_weight", n_neg / n_pos)

    model = xgb.XGBClassifier(**params, random_state=seed)
    model.fit(X[train], y[train], eval_set=[(X[test], y[test])], verbose=False)
    proba = model.predict_proba(X[test])[:, 1]
    auc = float(roc_auc_score(y[test], proba)) if (y[test].sum() and (y[test] == 0).sum()) else float("nan")

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"refitq[SPEC v{spec_version}]: test AUC = {auc:.4f}  "
          f"(config={config}, nfeat={nfeat_expected}, n_refit={len(X)}, "
          f"n_train={len(train)}, n_test={len(test)}, pos={n_pos}, neg={n_neg})")

    xgb_path = os.path.join(output_dir, "refitq_spec_xgb.json")
    model.save_model(xgb_path)

    conifer_json_path = None
    if export_conifer:
        import conifer

        booster = model.get_booster()
        cfg = conifer.backends.cpp.auto_config()
        cfg["OutputDir"] = os.path.join(output_dir, f"{conifer_name}_prj")
        cnf = conifer.converters.convert_from_xgboost(booster, cfg)
        conifer_json_path = os.path.join(output_dir, f"{conifer_name}.json")
        cnf.save(conifer_json_path)
        # validate the exported JSON against the producer contract at export time
        with open(conifer_json_path) as f:
            cj = json.load(f)
        if int(cj.get("n_features", -1)) != nfeat_expected:
            raise RuntimeError(
                f"exported conifer n_features={cj.get('n_features')} != {nfeat_expected} "
                f"(spec v{spec_version}); producer would reject this model.")
        # offline-margin self-check: python conifer walk vs the raw BOOSTER
        # margin (which is what conifer converts and what conifer.h/the producer
        # compute). NB: the sklearn XGBClassifier.predict(output_margin=True)
        # wrapper disagrees with the booster on xgboost>=2.0 - the booster margin
        # is the correct conifer reference, verified to match the walk to 0.0.
        walk = conifer_json_walk(cj, X[test])
        dtest = xgb.DMatrix(X[test])
        booster_margin = booster.predict(dtest, output_margin=True).astype(np.float32)
        max_walk_diff = float(np.max(np.abs(walk - booster_margin))) if len(X[test]) else 0.0
        print(f"  conifer written: {conifer_json_path}  "
              f"(n_trees={cj['n_trees']}, init_predict={cj['init_predict']}, "
              f"conv={cj.get('splitting_convention')})")
        print(f"  margin self-check max|walk - booster_margin| = {max_walk_diff:.3e} (float32)")

    meta = {
        "spec": "RefitSidecarSpec.md REFIT_BDT_FEATURES", "spec_version": spec_version,
        "n_features": nfeat_expected, "features": names, "config": config, "activeSP": CONFIG_ACTIVESP[config],
        "label": label, "track_table": track_table, "margin_semantics": "raw_logit_margin",
        "conifer_decision_function": "sum(tree_values)+init_predict, then *norm (float32)",
        "test_auc": auc, "n_refit_tracks": len(X), "n_pos": n_pos, "n_neg": n_neg,
        "n_train": len(train), "n_test": len(test), "seed": seed,
        "params": {k: v for k, v in params.items()},
        "input_files": [os.path.basename(f) for f in files],
        "provenance": provenance,
        "caveat": ("PRE-guard PU nano: chi2 columns carry the unguarded numerical tail. "
                   "Trained as-is; parity is about evaluation math, not sample perfection. "
                   "Trained on refit tracks only (passthrough tracks are never scored)."),
    }
    meta_path = os.path.join(output_dir, f"{conifer_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  metadata written: {meta_path}")

    return model, conifer_json_path, meta
