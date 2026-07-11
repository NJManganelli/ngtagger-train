"""End-to-end NNVtx + track-association network training from L1PFTrkNano.

Rebuilds the CMSSW GTT E2E vertexing chain (L1Trigger/VertexFinder
"NNEmulation" + L1TTrackMatch association network) in Keras 3:

  weight NN   : per-track MLP on (pt, trkMVA1, |eta|) [+ extra features]
                -> track weight  (upstream: NNVtx_WeightModelGraph.pb)
  pattern NN  : Conv1D over the weighted, differentiable soft z0 histogram
                -> PV z0 via soft-argmax (upstream: NNVtx_PatternModelGraph.pb)
  assoc NN    : per-track MLP on (|dz|, eta-resolution bin, pt, trkMVA1)
                [+ extra features] -> P(track from PV)
                (upstream: NNVtx_AssociationModelGraph.pb)

Trained jointly (end to end): huber(z0_pred, GenVtx_z) + BCE(assoc,
tpFromHardInteraction). Inputs are the stock L1PFTrkNano track columns; the
`extra_features` option appends new quantities (branch names or computed
features from the COMPUTED_FEATURES registry) to both MLPs for exploration.

The stock production scores live in the same files (L1Vertex_z0 / sumPt from
l1tVertexFinderEmulator, trkMVA1) enabling direct stock-vs-retrained
comparisons (see compare_vertex_scores and tests/test_nnvtx.py).
"""

from __future__ import annotations

import json
import os

import awkward as ak
import numpy as np
import uproot

# fastHisto binning from l1tVertexProducer_cfi (min, max, width) [cm]
HISTO_MIN, HISTO_MAX, HISTO_WIDTH = -20.46, 20.46, 0.16
N_BINS = int(round((HISTO_MAX - HISTO_MIN) / HISTO_WIDTH))  # 255 -> firmware uses 256-ish
MAX_TRACKS = 250

BASE_BRANCHES = ["pt", "eta", "z0", "trkMVA1", "tpFromHardInteraction", "genuine", "hitPattern",
                 "chi2XYRed", "chi2ZRed", "chi2BendRed", "nStubs"]

# registry of computable extra quantities (option to explore new inputs)
COMPUTED_FEATURES = {
    "abs_eta": lambda t: abs(t["eta"]),
    "log_pt": lambda t: np.log(np.maximum(t["pt"], 1e-3)),
    "nlaymiss_interior": lambda t: _nlaymiss(t["hitPattern"]),
}


def _nlaymiss(hitpattern):
    from ngtagger.train.trkquality import nlaymiss_interior

    counts = ak.num(hitpattern, axis=1)
    flat = nlaymiss_interior(ak.to_numpy(ak.flatten(hitpattern)))
    return ak.unflatten(flat, counts)


def load_vtx_data(files: list[str], track_table: str = "L1TTrack",
                  extra_features: list[str] | None = None,
                  max_events: int | None = None):
    """Read tracks + truth PV + stock vertex from L1PFTrkNano(withGen)."""
    branches = [f"{track_table}_{b}" for b in BASE_BRANCHES]
    branches += ["GenVtx_z", "L1Vertex_z0", "L1Vertex_sumPt"]
    events = uproot.concatenate([f"{f}:Events" for f in files], filter_name=branches, how="zip")
    if max_events is not None:
        events = events[:max_events]

    tracks = events[track_table]
    feats = {b: tracks[b] for b in BASE_BRANCHES}
    for name in extra_features or []:
        if name in COMPUTED_FEATURES:
            feats[name] = COMPUTED_FEATURES[name](feats)
        elif name not in feats:
            feats[name] = tracks[name]

    gen_z = ak.to_numpy(events["GenVtx"].z if "GenVtx" in events.fields else events["GenVtx_z"])
    stock_z0 = events["L1Vertex"].z0 if "L1Vertex" in events.fields else None
    stock_z0 = ak.to_numpy(ak.firsts(stock_z0)) if stock_z0 is not None else None
    return feats, gen_z, stock_z0


def to_padded(feats: dict, feature_names: list[str], max_tracks: int = MAX_TRACKS):
    """(event, track) jagged -> padded (n_events, max_tracks, n_feat) + mask."""
    cols = []
    for name in feature_names:
        arr = ak.fill_none(ak.pad_none(feats[name], max_tracks, axis=1, clip=True), 0.0)
        cols.append(ak.to_numpy(arr).astype(np.float32))
    X = np.stack(cols, axis=-1)
    mask = ak.to_numpy(
        ak.fill_none(ak.pad_none(ak.ones_like(feats["z0"]), max_tracks, axis=1, clip=True), 0.0)
    ).astype(np.float32)
    return X, mask


def fast_histo_z0(z0: np.ndarray, pt: np.ndarray, mask: np.ndarray,
                  window_bins: int = 3, max_track_pt: float = 50.0):
    """Numpy reference of the fastHisto baseline: pt-weighted (capped) z0
    histogram, best sliding window, pt-weighted mean inside the window."""
    n_events = z0.shape[0]
    edges = np.linspace(HISTO_MIN, HISTO_MAX, N_BINS + 1)
    out = np.zeros(n_events)
    w = np.minimum(pt, max_track_pt) * mask
    for i in range(n_events):
        hist, _ = np.histogram(z0[i], bins=edges, weights=w[i])
        windows = np.convolve(hist, np.ones(window_bins), mode="valid")
        b = int(np.argmax(windows))
        sel = (z0[i] >= edges[b]) & (z0[i] < edges[b + window_bins]) & (w[i] > 0)
        out[i] = np.average(z0[i][sel], weights=w[i][sel]) if sel.any() else 0.5 * (edges[b] + edges[b + window_bins])
    return out


# --------------------------------------------------------------------------
# E2E model
# --------------------------------------------------------------------------
def build_e2e_model(n_weight_feats: int, n_assoc_extra: int, max_tracks: int = MAX_TRACKS,
                    weight_layers=(10, 10), pattern_filters=(16, 16), assoc_layers=(20, 20),
                    soft_sigma: float = HISTO_WIDTH):
    """Weight NN + differentiable soft histogram + pattern NN + assoc NN,
    joined end-to-end. Inputs: track features (n_weight_feats, first column
    MUST be z0 is NOT included there - z0 enters via a dedicated input), mask."""
    import keras
    from keras import layers, ops

    z0_in = keras.Input((max_tracks,), name="trk_z0")
    feat_in = keras.Input((max_tracks, n_weight_feats), name="trk_features")
    mask_in = keras.Input((max_tracks,), name="trk_mask")

    # ---- weight network (per track)
    w = feat_in
    for i, width in enumerate(weight_layers):
        w = layers.Dense(width, activation="relu", name=f"weight_dense_{i}")(w)
    trk_weight = layers.Dense(1, activation="relu", name="weight_out")(w)
    trk_weight = layers.Lambda(lambda t: t[0][..., 0] * t[1], name="masked_weight")([trk_weight, mask_in])

    # ---- differentiable soft histogram over z0
    centers = np.linspace(HISTO_MIN + HISTO_WIDTH / 2, HISTO_MAX - HISTO_WIDTH / 2, N_BINS).astype("float32")

    def soft_hist(args):
        z0, wgt = args
        d = ops.expand_dims(z0, -1) - centers[None, None, :]           # (B, T, N_BINS)
        k = ops.exp(-0.5 * (d / soft_sigma) ** 2)
        return ops.sum(ops.expand_dims(wgt, -1) * k, axis=1)           # (B, N_BINS)

    hist = layers.Lambda(soft_hist, name="soft_histogram")([z0_in, trk_weight])
    hist = layers.Reshape((N_BINS, 1))(hist)

    # ---- pattern network -> soft-argmax z0
    p = hist
    for i, f in enumerate(pattern_filters):
        p = layers.Conv1D(f, 3, padding="same", activation="relu", name=f"pattern_conv_{i}")(p)
    logits = layers.Conv1D(1, 3, padding="same", name="pattern_logits")(p)
    logits = layers.Reshape((N_BINS,))(logits)
    probs = layers.Softmax(name="pattern_softmax")(logits)
    z0_pred = layers.Lambda(lambda pr: ops.sum(pr * centers[None, :], axis=-1, keepdims=True),
                            name="pv_z0")(probs)

    # ---- association network (per track, conditioned on the found vertex)
    dz = layers.Lambda(lambda t: ops.abs(t[0] - t[1]), name="abs_dz")([z0_in, z0_pred])
    assoc_in = layers.Concatenate(name="assoc_features")(
        [layers.Reshape((max_tracks, 1))(dz), feat_in]
    )
    a = assoc_in
    for i, width in enumerate(assoc_layers):
        a = layers.Dense(width, activation="relu", name=f"assoc_dense_{i}")(a)
    assoc = layers.Dense(1, activation="sigmoid", name="assoc_raw")(a)
    assoc = layers.Lambda(lambda t: t[0][..., 0] * t[1], name="assoc_out")([assoc, mask_in])

    model = keras.Model([z0_in, feat_in, mask_in], [z0_pred, assoc], name="e2e_nnvtx")
    _ = n_assoc_extra  # extras are already concatenated inside feat_in
    return model


def train_nnvtx(files: list[str], output_dir: str, track_table: str = "L1TTrack",
                extra_features: list[str] | None = None, max_events: int | None = None,
                epochs: int = 30, batch_size: int = 64, learning_rate: float = 1e-3,
                max_tracks: int = MAX_TRACKS, seed: int = 0, dataset=None):
    import keras

    keras.utils.set_random_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    weight_features = ["pt", "trkMVA1", "abs_eta"] + list(extra_features or [])

    if dataset is None:
        feats, gen_z, stock_z0 = load_vtx_data(files, track_table=track_table,
                                               extra_features=["abs_eta"] + list(extra_features or []),
                                               max_events=max_events)
        X, mask = to_padded(feats, weight_features, max_tracks)
        z0_trk, _ = to_padded(feats, ["z0"], max_tracks)
        z0_trk = z0_trk[..., 0]
        y_assoc, _ = to_padded(feats, ["tpFromHardInteraction"], max_tracks)
        y_assoc = y_assoc[..., 0]
    else:
        X, mask, z0_trk, y_assoc, gen_z, stock_z0 = dataset

    model = build_e2e_model(X.shape[-1], 0, max_tracks=max_tracks)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss={"pv_z0": keras.losses.Huber(delta=0.5), "assoc_out": keras.losses.BinaryCrossentropy()},
        loss_weights={"pv_z0": 1.0, "assoc_out": 5.0},
    )
    history = model.fit(
        {"trk_z0": z0_trk, "trk_features": X, "trk_mask": mask},
        {"pv_z0": np.asarray(gen_z, dtype=np.float32).reshape(-1, 1), "assoc_out": y_assoc},
        epochs=epochs, batch_size=batch_size, validation_split=0.15,
        callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
        verbose=2,
    )

    model.save(os.path.join(output_dir, "e2e_nnvtx.keras"))
    meta = {"weight_features": weight_features, "track_table": track_table,
            "max_tracks": max_tracks, "n_bins": N_BINS,
            "best_val_loss": float(np.min(history.history["val_loss"]))}
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    try:
        import mlflow

        mlflow.set_experiment("ngtagger-nnvtx")
        with mlflow.start_run(run_name=f"nnvtx-{track_table}"):
            mlflow.log_params({"extra_features": ",".join(extra_features or []), "epochs": epochs})
            mlflow.log_metric("best_val_loss", meta["best_val_loss"])
            mlflow.log_artifacts(output_dir)
    except Exception:
        pass
    return model, history


def compare_vertex_scores(gen_z, stock_z0, new_z0, y_assoc=None, assoc_pred=None,
                          mask=None, match_window: float = 0.15):
    """Stock-production vs newly-computed vertex (and association) metrics."""
    out = {}
    for name, z in (("stock", stock_z0), ("new", new_z0)):
        if z is None:
            continue
        res = np.asarray(z) - np.asarray(gen_z)
        out[f"{name}_res_mean"] = float(res.mean())
        out[f"{name}_res_std"] = float(res.std())
        out[f"{name}_efficiency"] = float((np.abs(res) < match_window).mean())
    if stock_z0 is not None and new_z0 is not None:
        out["stock_vs_new_std"] = float((np.asarray(stock_z0) - np.asarray(new_z0)).std())
    if y_assoc is not None and assoc_pred is not None:
        from sklearn.metrics import roc_auc_score

        sel = (mask > 0).ravel() if mask is not None else slice(None)
        out["assoc_auc"] = float(roc_auc_score(np.ravel(y_assoc)[sel], np.ravel(assoc_pred)[sel]))
    return out


def export_frozen_graphs(model_dir: str):
    """Export the three sub-networks as frozen TF graphs matching the CMSSW
    deployment artifacts (NNVtx_WeightModelGraph.pb / PatternModelGraph.pb /
    AssociationModelGraph.pb). Requires the TF backend; sub-network
    extraction relies on the layer naming of build_e2e_model.

    NOTE: the deployed graphs are the *quantised* (QKeras-era) variants with
    fixed input scalings; retraining for deployment additionally needs the
    digitised-input conventions of VertexFinder::NNVtxEmulation (GTT-word
    inputs), mirroring the trkquality deployment-parity discussion.
    """
    raise NotImplementedError(
        "frozen-graph export pending the digitised-input (GTT word) training mode"
    )
