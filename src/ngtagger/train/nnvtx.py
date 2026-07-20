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


VTX_KERNELS = ("flat", "triangular", "gaussian", "epanechnikov")


def make_kernel(kernel: str = "flat", window_bins: int = 3, sigma_bins: float = 1.0,
                kernel_array=None):
    """Peak-finder kernel weights for fast_histo_z0 / fast_histo_vtx.

    flat          ones(window_bins)  -- emulator boxcar (default)
    triangular    1 - |i|/(hw+1)     over i in [-hw, hw], hw = (window_bins-1)//2
    epanechnikov  1 - (i/(hw+1))^2   same support
    gaussian      exp(-i^2/2 sigma_bins^2), support 2*ceil(3 sigma)+1 (>= window_bins)
    kernel_array  arbitrary user weights (overrides `kernel`)

    Normalised to max=1 (normalisation cannot affect the arg-max selection)."""
    if kernel_array is not None:
        k = np.asarray(kernel_array, dtype=float)
        if k.ndim != 1 or k.size == 0 or not np.any(k > 0):
            raise ValueError("kernel_array must be a non-empty 1D array with a positive entry")
        return k / k.max()
    if kernel == "flat":
        return np.ones(window_bins)
    hw = (window_bins - 1) // 2
    if kernel == "gaussian":
        hw = max(hw, int(np.ceil(3.0 * sigma_bins)))
    i = np.arange(-hw, hw + 1, dtype=float)
    if kernel == "gaussian":
        return np.exp(-0.5 * (i / sigma_bins) ** 2)
    if kernel == "triangular":
        return 1.0 - np.abs(i) / (hw + 1.0)
    if kernel == "epanechnikov":
        return 1.0 - (i / (hw + 1.0)) ** 2
    raise ValueError(f"unknown kernel '{kernel}'; known: {VTX_KERNELS} or kernel_array")


def _select_window(hist: np.ndarray, kern: np.ndarray, window_bins: int) -> int:
    """Kernel-weighted arg-max window selection: correlate the histogram with
    the kernel, return the first bin of the window_bins-wide centroid window
    centred on the winning kernel position. For the flat kernel this reduces
    bit-identically to the original boxcar arg-max (lo == argmax index)."""
    scores = np.convolve(hist, kern[::-1], mode="valid")  # correlation semantics
    b = int(np.argmax(scores))
    lo = b + (len(kern) - 1) // 2 - (window_bins - 1) // 2
    return min(max(lo, 0), len(hist) - window_bins)


def fast_histo_z0(z0: np.ndarray, pt: np.ndarray, mask: np.ndarray,
                  window_bins: int = 3, max_track_pt: float = 50.0,
                  kernel: str = "flat", sigma_bins: float = 1.0, kernel_array=None):
    """Numpy reference of the fastHisto baseline: pt-weighted (capped) z0
    histogram, best sliding window, pt-weighted mean inside the window.

    `kernel` selects the peak-finder weighting (see make_kernel); the default
    "flat" is bit-identical to the emulator boxcar. The kernel enters ONLY the
    arg-max window selection: the z estimate stays the plain pt-weighted
    centroid of the window (emulator convention -- a kernel-weighted centroid
    would pull z toward the kernel centre and break stock comparisons)."""
    n_events = z0.shape[0]
    edges = np.linspace(HISTO_MIN, HISTO_MAX, N_BINS + 1)
    kern = make_kernel(kernel, window_bins, sigma_bins, kernel_array)
    out = np.zeros(n_events)
    w = np.minimum(pt, max_track_pt) * mask
    for i in range(n_events):
        hist, _ = np.histogram(z0[i], bins=edges, weights=w[i])
        lo = _select_window(hist, kern, window_bins)
        sel = (z0[i] >= edges[lo]) & (z0[i] < edges[lo + window_bins]) & (w[i] > 0)
        out[i] = np.average(z0[i][sel], weights=w[i][sel]) if sel.any() else 0.5 * (edges[lo] + edges[lo + window_bins])
    return out


# --------------------------------------------------------------------------
# Vertex transverse-position (dx, dy) estimation in the fastHisto PV window
# --------------------------------------------------------------------------
# Sign convention (VERIFIED against DataFormats/L1TrackTrigger/interface/
# TTTrack.h: thePOCA_(d0*sin(phi0), -d0*cos(phi0), z0), and the beamspot-
# correction comment d0 - (XB*sin(phi) - YB*cos(phi)); the nano d0/phi
# branches are Var("d0()")/Var("phi()") on the same TTTrack):
#     d0_i = x_v sin(phi_i) - y_v cos(phi_i) + noise    (prompt track from
# vertex (x_v, y_v); curvature negligible at vertex scale). NOTE this is the
# NEGATIVE of offline TrackBase::dxy().
VTX_RESULT_FIELDS = ("z0_pv", "dx", "dy", "sigma_dx", "sigma_dy", "dxsig", "dysig",
                     "n_window", "sum_w", "phi_condition", "d0_scatter")


def fast_histo_vtx(z0: np.ndarray, pt: np.ndarray, mask: np.ndarray,
                   d0: np.ndarray, phi: np.ndarray,
                   window_bins: int = 3, max_track_pt: float = 50.0,
                   kernel: str = "flat", sigma_bins: float = 1.0, kernel_array=None,
                   estimator: str = "lsq", min_condition: float = 1e-6,
                   d0_gate: float | None = None):
    """fastHisto extended with a PV-window (dx, dy) estimate.

    d0_gate (cm, optional): tracks with |d0| > d0_gate are excluded from the
    (dx, dy) accumulators only (never from the z histogram, which stays
    emulator-faithful). The transverse solve assumes PROMPT tracks
    (curvature ignored at vertex scale); L1 extended-track collections in a
    jet environment also carry displaced/loose tracks with |d0| up to
    O(cm) -- see eval_refitq/vtxdxy/realdata_smoke -- that violate the prompt
    approximation and inflate the fit scatter. A gate of ~0.1-0.2 cm restores
    a beam-spot-scale estimate. None = no gate (all tracks, faithful to the
    raw accumulator picture).

    Parallel per-z-bin accumulators alongside the pt-weighted z histogram:
    w, w*d0*s, w*d0*c, w*s^2, w*c^2, w*s*c, w*d0^2, n  (s=sin(phi), c=cos(phi),
    w = capped-pt weight). At the found peak window the sums are aggregated
    and, with the convention d0 = x_v s - y_v c (see above), minimising
    Sum w (d0 - x_v s + y_v c)^2 gives the weighted least-squares normal
    equations (symmetric form after negating the second row):

        [  S_ss  -S_sc ] [x_v]   [ +S_wds ]        S_ab  = sum w a b
        [ -S_sc   S_cc ] [y_v] = [ -S_wdc ]        S_wds = sum w d0 s, etc.

    solved as x_v = (S_cc S_wds - S_sc S_wdc)/det,
             y_v = (S_sc S_wds - S_ss S_wdc)/det,  det = S_ss S_cc - S_sc^2.
    (Noiseless closure is exact -- proven in test_convention_exact_recovery.)

    estimator="isotropic" is the cheap variant x_v = +2 S_wds/S_w,
    y_v = -2 S_wdc/S_w, exact only for phi-isotropic weighted coverage
    (S_ss = S_cc = S_w/2, S_sc = 0); its uncertainties reuse the residual
    formula with the isotropic normal matrix.

    Uncertainties: residual scatter of d0 about the fit inside the window,
    computed FROM THE SUMS (no second track pass):
        S_wrr = S_wdd - 2 x S_wds + 2 y S_wdc + x^2 S_ss + y^2 S_cc - 2 x y S_sc
        sigma^2_d0 = S_wrr / (n - 2),   Cov(x,y) = sigma^2_d0 * A^-1
    (pragmatic: assumes sigma_d0,i ~ 1/sqrt(w_i) up to a common scale; needs
    n >= 3 for the scatter, n >= 2 and phi_condition >= min_condition for the
    solve; otherwise NaN -- degraded inputs must be visible, not silent).

    d0 MUST come from a 5-parameter (Extended) track collection (nano
    L1TExtTrack): the prompt 4-par tracks have d0 pinned to 0 and carry no
    (dx, dy) information. Use vertex_dxy_features for the loud guard.

    Returns a dict of (n_events,) float64 arrays, fields VTX_RESULT_FIELDS.
    z0_pv is computed exactly as fast_histo_z0 (same histogram, selection and
    centroid); phi_condition = det(A)/(S_ss*S_cc) in (0, 1], 1 = balanced
    coverage; d0_scatter = sqrt(S_wrr/S_w) (weighted RMS of the d0 residuals)."""
    if estimator not in ("lsq", "isotropic"):
        raise ValueError(f"unknown estimator '{estimator}' (lsq | isotropic)")
    n_events = z0.shape[0]
    edges = np.linspace(HISTO_MIN, HISTO_MAX, N_BINS + 1)
    kern = make_kernel(kernel, window_bins, sigma_bins, kernel_array)
    w = np.minimum(pt, max_track_pt) * mask
    s, c = np.sin(phi), np.cos(phi)

    out = {f: np.full(n_events, np.nan) for f in VTX_RESULT_FIELDS}
    out["n_window"] = np.zeros(n_events)
    out["sum_w"] = np.zeros(n_events)
    for i in range(n_events):
        zi, wi, d0i, si, ci = z0[i], w[i], d0[i], s[i], c[i]
        hist, _ = np.histogram(zi, bins=edges, weights=wi)
        lo = _select_window(hist, kern, window_bins)

        # transverse-solve weight: pt weight, optionally prompt-track gated
        # (gate never touches the z histogram / window selection above)
        wg = wi if d0_gate is None else wi * (np.abs(d0i) <= d0_gate)

        # parallel per-bin accumulator histograms, aggregated over the window
        sums = []
        for vals in (wg, wg * d0i * si, wg * d0i * ci, wg * si * si, wg * ci * ci,
                     wg * si * ci, wg * d0i * d0i, (wg > 0).astype(float)):
            acc, _ = np.histogram(zi, bins=edges, weights=vals)
            sums.append(acc[lo:lo + window_bins].sum())
        Sw, Swds, Swdc, Sss, Scc, Ssc, Swdd, Sn = sums
        n = int(round(Sn))
        out["n_window"][i], out["sum_w"][i] = n, Sw

        # z centroid: identical to fast_histo_z0 (value-based selection)
        sel = (zi >= edges[lo]) & (zi < edges[lo + window_bins]) & (wi > 0)
        out["z0_pv"][i] = (np.average(zi[sel], weights=wi[sel]) if sel.any()
                           else 0.5 * (edges[lo] + edges[lo + window_bins]))
        if Sw <= 0:
            continue

        det = Sss * Scc - Ssc * Ssc
        cond = det / (Sss * Scc) if (Sss > 0 and Scc > 0) else 0.0
        out["phi_condition"][i] = cond
        if estimator == "isotropic":
            dx, dy = 2.0 * Swds / Sw, -2.0 * Swdc / Sw
            inv_xx = inv_yy = 2.0 / Sw  # isotropic A = (Sw/2) * I
        else:
            if n < 2 or det <= 0 or cond < min_condition:
                continue  # degenerate phi coverage: no transverse solve
            dx = (Scc * Swds - Ssc * Swdc) / det
            dy = (Ssc * Swds - Sss * Swdc) / det
            inv_xx, inv_yy = Scc / det, Sss / det
        out["dx"][i], out["dy"][i] = dx, dy

        s_wrr = max(Swdd - 2 * dx * Swds + 2 * dy * Swdc
                    + dx * dx * Sss + dy * dy * Scc - 2 * dx * dy * Ssc, 0.0)
        out["d0_scatter"][i] = np.sqrt(s_wrr / Sw)
        if n >= 3:
            s2 = s_wrr / (n - 2)
            out["sigma_dx"][i] = np.sqrt(s2 * inv_xx)
            out["sigma_dy"][i] = np.sqrt(s2 * inv_yy)
            with np.errstate(divide="ignore", invalid="ignore"):
                out["dxsig"][i] = dx / out["sigma_dx"][i]
                out["dysig"][i] = dy / out["sigma_dy"][i]
    return out


def vertex_dxy_features(tracks, pt_field: str = "pt", z0_field: str = "z0",
                        d0_field: str = "d0", phi_field: str = "phi",
                        sanitize: bool = False, **fast_histo_kwargs):
    """Per-event vertex (dx, dy) feature columns from a jagged nano track
    table (feature group "vertexdxy"; track_table must be the 5-parameter
    extended collection, e.g. L1TExtTrack -- an identically-zero d0 column
    means the prompt 4-par collection was passed and raises loudly instead of
    silently yielding dx=dy=0).

    Returns {vtx_dx, vtx_dy, vtx_dxsig, vtx_dysig} as (n_events,) float32.
    sanitize=True maps non-finite values (no solve / zero scatter) to 0.0 for
    direct use as model inputs; keep False for studies (NaN stays visible)."""
    for fld in (pt_field, z0_field, d0_field, phi_field):
        if fld not in tracks.fields:
            raise KeyError(
                f"track table lacks '{fld}' (needed by vertexdxy; extended-track "
                f"table required, e.g. L1TExtTrack). Available: {sorted(tracks.fields)}")
    if not ak.any(np.abs(tracks[d0_field]) > 0):
        raise ValueError(
            "track-table d0 is identically zero: the prompt 4-parameter collection "
            "carries no (dx, dy) information; use the extended table (L1TExtTrack)")
    counts = ak.num(tracks[z0_field], axis=1)
    max_trk = max(int(ak.max(counts)), 1)

    def _pad(field, fill=0.0):
        return ak.to_numpy(ak.fill_none(
            ak.pad_none(tracks[field], max_trk, axis=1, clip=True), fill)).astype(np.float64)

    mask = ak.to_numpy(ak.fill_none(
        ak.pad_none(ak.ones_like(tracks[z0_field]), max_trk, axis=1, clip=True), 0.0)
    ).astype(np.float64)
    res = fast_histo_vtx(_pad(z0_field), _pad(pt_field), mask,
                         _pad(d0_field), _pad(phi_field), **fast_histo_kwargs)
    out = {"vtx_dx": res["dx"], "vtx_dy": res["dy"],
           "vtx_dxsig": res["dxsig"], "vtx_dysig": res["dysig"]}
    if sanitize:
        out = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in out.items()}
    return {k: v.astype(np.float32) for k, v in out.items()}


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
