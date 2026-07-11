"""Track-quality GBDT retraining from L1TrkNano track tables.

Reproduces the CMSSW TrackerTFP TrackQuality GBDT
(L1_TrackQuality_GBDT_emulation_digitized.json) training with the exact
feature set of the deployed model:

    tanl, z0_scaled, bendchi2_bin, nstub, nlaymiss_interior,
    chi2rphi_bin, chi2rz_bin

built from the hardware track-word columns of the L1TTrack/L1TExtTrack
tables (hwTanl, hwZ0, hwBendChi2, hwChi2RPhi, hwChi2RZ, hitPattern) plus the
nStubs and genuine/fake truth columns added by L1TrackTruthTableProducer
(withGen flavors; requires the TTTrackAssociator to have run upstream).

Model: xgboost binary classifier (genuine vs fake), exported to a conifer
model json for the CMSSW FileInPath deployment and optionally to HLS.
"""

from __future__ import annotations

import json
import os

import awkward as ak
import numpy as np
import uproot

TRKQ_FEATURES = ["tanl", "z0_scaled", "bendchi2_bin", "nstub", "nlaymiss_interior",
                 "chi2rphi_bin", "chi2rz_bin"]

_BRANCHES = ["hwTanl", "hwZ0", "hwBendChi2", "hwChi2RPhi", "hwChi2RZ", "hitPattern",
             "nStubs", "genuine", "looselyGenuine", "unknown", "tpPt", "tpFromHardInteraction",
             "pt", "eta"]


def nlaymiss_interior(hitpattern: np.ndarray, width: int = 7) -> np.ndarray:
    """Number of missed interior layers: zero bits strictly between the
    lowest and highest set bits of the hit pattern."""
    hp = hitpattern.astype(np.int64)
    out = np.zeros(len(hp), dtype=np.int64)
    nonzero = hp > 0
    if not nonzero.any():
        return out
    bits = ((hp[:, None] >> np.arange(width)[None, :]) & 1).astype(bool)
    first = np.argmax(bits, axis=1)
    last = width - 1 - np.argmax(bits[:, ::-1], axis=1)
    for i in np.where(nonzero)[0]:
        interior = bits[i, first[i]:last[i] + 1]
        out[i] = int((~interior).sum())
    return out


def twos_complement(bits: np.ndarray, width: int) -> np.ndarray:
    """Decode raw track-word bit fields into signed integers (the track word
    stores tanl and z0 as two's-complement; the nano hw columns are the raw
    unsigned bits)."""
    v = bits.astype(np.int64)
    return np.where(v >= (1 << (width - 1)), v - (1 << width), v)


# TTTrack_TrackWord bit widths (DataFormats/L1TrackTrigger/TTTrack_TrackWord.h)
K_TANL_SIZE = 16
K_Z0_SIZE = 12


def load_tracks(files: list[str], track_table: str = "L1TTrack",
                max_events: int | None = None) -> ak.Array:
    branches = [f"{track_table}_{b}" for b in _BRANCHES]
    events = uproot.concatenate([f"{f}:Events" for f in files], filter_name=branches, how="zip")
    if max_events is not None:
        events = events[:max_events]
    return events[track_table]


def build_trkq_dataset(tracks: ak.Array, label: str = "genuine",
                       require_truth: bool = True):
    """Flatten tracks and build the 7 GBDT features + genuine/fake label."""
    flat = {f: ak.to_numpy(ak.flatten(tracks[f])) for f in _BRANCHES}

    if require_truth and not (flat["genuine"].any() or flat["unknown"].all()):
        pass  # leave validation to the caller; unknown-only files mean no associator

    X = np.stack([
        twos_complement(flat["hwTanl"], K_TANL_SIZE).astype(np.float32),
        twos_complement(flat["hwZ0"], K_Z0_SIZE).astype(np.float32),
        flat["hwBendChi2"].astype(np.float32),
        flat["nStubs"].astype(np.float32),
        nlaymiss_interior(flat["hitPattern"]).astype(np.float32),
        flat["hwChi2RPhi"].astype(np.float32),
        flat["hwChi2RZ"].astype(np.float32),
    ], axis=1)
    y = flat[label].astype(np.int64)

    if require_truth and flat["unknown"].all():
        raise RuntimeError(
            "all tracks are truth-'unknown': the TTTrackAssociator did not run "
            "in the nano production workflow; produce the sample with the "
            "track-truth sequence enabled (withGen L1TrkNano flavors)."
        )
    return X, y, flat


def train_trkquality(files: list[str], output_dir: str, track_table: str = "L1TTrack",
                     label: str = "genuine", max_events: int | None = None,
                     test_fraction: float = 0.2, seed: int = 0,
                     xgb_params: dict | None = None):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    os.makedirs(output_dir, exist_ok=True)
    tracks = load_tracks(files, track_table=track_table, max_events=max_events)
    X, y, _ = build_trkq_dataset(tracks, label=label)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_fraction)
    test, train = idx[:n_test], idx[n_test:]

    params = {
        "n_estimators": 60,
        "max_depth": 3,
        "learning_rate": 0.2,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "early_stopping_rounds": 10,
        **(xgb_params or {}),
    }
    model = xgb.XGBClassifier(**params, random_state=seed)
    model.fit(X[train], y[train], eval_set=[(X[test], y[test])], verbose=False)

    auc = roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])
    print(f"trkquality GBDT: test AUC = {auc:.4f} "
          f"(n_train={len(train)}, n_test={len(test)}, genuine fraction={y.mean():.3f})")

    model.save_model(os.path.join(output_dir, "trkquality_xgb.json"))
    meta = {"features": TRKQ_FEATURES, "label": label, "track_table": track_table,
            "test_auc": float(auc), "params": {k: v for k, v in params.items()}}
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    try:
        import mlflow

        mlflow.set_experiment("ngtagger-trkquality")
        with mlflow.start_run(run_name=f"trkq-{track_table}"):
            mlflow.log_params(params)
            mlflow.log_metric("test_auc", auc)
            mlflow.log_artifacts(output_dir)
    except Exception:
        pass

    return model, auc


def export_conifer(model_dir: str, output_dir: str | None = None, backend: str = "cpp"):
    """Convert the trained xgboost model to a conifer model (json + optional
    HLS project), matching the CMSSW TrackQuality FileInPath deployment
    format (conifer GBDT json)."""
    import conifer
    import xgboost as xgb

    output_dir = output_dir or model_dir
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "trkquality_xgb.json"))

    if backend == "cpp":
        cfg = conifer.backends.cpp.auto_config()
    else:
        cfg = conifer.backends.xilinxhls.auto_config()
    cfg["OutputDir"] = os.path.join(output_dir, f"conifer_{backend}")

    cnf_model = conifer.converters.convert_from_xgboost(booster, cfg)
    cnf_model.save(os.path.join(output_dir, "trkquality_conifer.json"))
    print(f"conifer model written to {output_dir}/trkquality_conifer.json "
          f"(deployable via TrackQuality_params.Model FileInPath)")
    return cnf_model
