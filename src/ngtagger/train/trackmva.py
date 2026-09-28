"""Per-track resolution-quality MVAs for SmartPixels refit tracks (vertexing /
tagger inputs), trained directly from nano.

Promoted from the vertexing pilot (eval_vertex/track_mvas.py +
eval_vertex/build_dataset.py; mem:smartpixels-vertexing-pilot). Same targets,
features, model family and cross-fitting as the pilot, so results close
against it; what is new is the nano reader (event-batched, either key
spelling), the frozen-model bundle with metadata, and the loader/scorer used by
``ngtagger.data.trackmva_feature``.

Track families (all 1:1 row aligned per event, validated on read):

  prompt  L1TTrack    + L1TSmartPixelsTrackDigiRefit<cfg>    + ...RefitHitDigiRefit<cfg>
  ext     L1TExtTrack + L1TSmartPixelsExtTrackDigiRefit<cfg> + ...ExtRefitHitDigiRefit<cfg>

The base table (L1TTrack / L1TExtTrack) supplies the OT-only fit, its KF
covariance and the TP truth (genuine, tp_pt, tp_d0, tp_z0: the OT associator's
TP, also the refit row's majority owner since the IT hits never outvote the
>= 4 OT stubs); L1TTP is not needed. Tracks without a refit carry the baseline
parameters in the refit table and are kept (spixRefitPerformed is a feature).

Collections (``coll``): "refit" (the default and the deliverable) uses base +
refit + per-hit features; "base" uses the base table alone (the pilot's
OT-only reference). The refit collection is split into two windows by the
FITTED refit |d0| <= / > 200 um (D0_SPLIT), each with its own models; "base"
has one window ("all": at ~580 um base d0 resolution a 200 um split on the
fitted d0 is meaningless).

Heads (per window; OUTPUTS order):
  clean    P(track clean): base = genuine; refit = genuine AND no wrong IT hit
           (an accepted IT hit whose cluster tpIdx differs from spixMatchedTpIdx)
  d0acc    P(|d0 - d0_TP| below the 68.3 pct of the CLEAN matched population)
  z0acc    same for z0
  log_sd0  log sigma_hat(d0/um), sigma_hat = max((q84.1 - q15.9)/2, 1 um) from a
           pair of quantile HGBs (plus the median) regressing d0 - d0_TP
  log_sz0  same for z0
d0acc/z0acc/sd0/sz0 are trained on TP-matched tracks only (tp_pt > 0); clean
on all tracks of the window.

WITHHELD FEATURES (the rule, keep it): every d0-derived quantity (d0, |d0|,
d0/sigma, sigma_KF(d0), the refit-minus-base d0 shift) is withheld from EVERY
head. They buy AUC by penalising genuinely displaced tracks (the "trap" of
docs/track-resolution-quality-study.md: a d0 head that learns |d0| large =>
bad resolution kills b/tau daughters, and cleanliness drops them for free).
``trap_ablation`` measures that on nano (d0acc with vs without them). The z0
VALUE is withheld too: a vertexing input must not encode where the track sits
along the beam (it would let the vertex model learn the pile-up z profile
instead of the track quality). The only use of d0 is routing a track to its
window by the fitted |d0|.

Cross-fitting (event level, never track level). Events of `train_only` files
(e.g. the files that trained the cluster-angle CNN, whose IT angles are
in-sample) get fold -1: they are in every training set and never evaluated.
The rest are permuted (seed) into K folds. Modes:
  none    no out-of-sample scores; only the frozen model is fitted
  outer   fold k scored by the model trained on folds != k (+ train-only)
  nested  the pilot's scheme: for every outer fold k, scores[k] holds fold-k
          tracks scored by the model trained on folds != k, fold-j tracks
          (j != k) scored by the model trained on folds not in {k, j}, and
          train-only tracks scored by the model trained on the test-eligible
          folds != k. So a downstream model trained on folds != k and
          evaluated on fold k only ever sees out-of-sample scores.
The per-track report (``evaluate``) always uses the outer (fully
out-of-sample) scores. The FROZEN model is a separate fit on all events of
the window (train-only included); its own performance is by construction the
cross-validated one, not re-measured.

Model family: sklearn HistGradientBoosting (classifier; quantile regressor),
max_iter=200, learning_rate=0.1, random_state=1, exactly the pilot's, so the
closure is exact on identical inputs. A conifer/xgboost port (fixed-point,
FPGA-deployable, e.g. for the free 6-bit hwMVAOther word: 3-bit P(clean) +
3-bit sigma_hat(d0)) is a FOLLOW-UP: the HGB models are the float reference.

Frozen bundle (``save_bundle`` / ``load_bundle``) in the output directory:
  trackmva_<family>_<cfg>.joblib      {coll: {window: {head: estimator}}}
  trackmva_<family>_<cfg>_meta.json   feature order per coll, spec version,
                                      head definitions and thresholds, windows,
                                      training files (+size/mtime), event split,
                                      HGB params, package versions, git commit,
                                      sha256 of the joblib
  trackmva_<family>_<cfg>_oof.npz     cross-fitted scores + track keys
  trackmva_<family>_<cfg>_report.json the out-of-sample per-track report
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import awkward as ak
import numpy as np

from ngtagger.data.spix_nano import CLUSTER_TABLE, OT_RENAMES, branch_name, branch_names, tp_idx
from ngtagger.train.trkquality import nlaymiss_interior

SPEC_VERSION = 1
N_FOLDS = 5
D0_SPLIT = 0.02            # cm, on the FITTED refit d0 (refit windows)
SIGMA_FLOOR_UM = 1.0       # sigma_hat floor
CLEAN_PCT = 68.3           # d0acc / z0acc target: percentile of the clean |residual|
QUANTILES = (0.15865, 0.5, 0.84135)
OUTPUTS = ["clean", "d0acc", "z0acc", "log_sd0", "log_sz0"]
# fit_predict columns: OUTPUTS + the d0 quantiles (evaluation of the width head)
EVAL_COLUMNS = OUTPUTS + ["q16_d0", "q50_d0", "q84_d0"]
HGB_PARAMS = {"max_iter": 200, "learning_rate": 0.1}
SEED = 1                   # model random_state (the pilot's)
COLLS = ("refit", "base")

FAMILIES = {
    "prompt": {"base": "L1TTrack", "refit": "L1TSmartPixelsTrackDigiRefit{cfg}",
               "hits": "L1TSmartPixelsRefitHitDigiRefit{cfg}"},
    "ext": {"base": "L1TExtTrack", "refit": "L1TSmartPixelsExtTrackDigiRefit{cfg}",
            "hits": "L1TSmartPixelsExtRefitHitDigiRefit{cfg}"},
}

# online (model-input) columns
BASE_COLS = ["pt", "eta", "tanL", "z0", "d0", "trkMVA1", "chi2XYRed", "chi2ZRed",
             "chi2BendRed", "nStubs", "hitPattern",
             "cov_rInv_rInv", "cov_phi_phi", "cov_tanL_tanL", "cov_z0_z0", "cov_d0_d0"]
REFIT_COLS = ["pt", "tanL", "z0", "d0", "spixRefitPerformed", "spixAnyWindowTruncated",
              "spixNCrossings", "spixNAcceptedHits", "spixLayerHitMask", "spixMaxWindowMult",
              "spixChi2IncXTot", "spixChi2IncYTot", "spixChi2IncAlphaTot", "spixChi2ITAtOT",
              "spixChi2ITAtRefit", "spixShiftChi2", "spixLogDetRatio"]
HIT_COLS = ["hitAccepted", "trackIdx", "hasAlpha", "windowMult", "clusterMerged",
            "pullX", "pullY", "pullAlpha", "chi2IncX", "chi2IncY", "chi2IncAlpha"]
# truth (labels / bookkeeping only; never a model input)
BASE_TRUTH = ["genuine", "tpFromHardInteraction", "tp_pt", "tp_d0", "tp_z0"]
REFIT_TRUTH = ["spixMatchedTpIdx"]
HIT_TRUTH = ["selClusterIdx"]
EVENT_ID = ["run", "luminosityBlock", "event"]

REFIT_SENTINEL_COLS = ["spixRefitPerformed", "spixAnyWindowTruncated", "spixNCrossings",
                       "spixNAcceptedHits", "spixLayerHitMask", "spixMaxWindowMult",
                       "spixChi2IncXTot", "spixChi2IncYTot", "spixChi2IncAlphaTot",
                       "spixChi2ITAtOT", "spixChi2ITAtRefit", "spixShiftChi2",
                       "spixLogDetRatio"]
HIT_FEATURES = ["h_nangle", "h_chi2_angle_per_cl", "h_chi2_pos_per_hit", "h_mean_wmult",
                "h_nmerged", "h_max_angle_pull", "h_second_angle_pull", "h_max_pos_pull"]

HEAD_DEFINITIONS = {
    "clean": "P(clean): base = genuine; refit = genuine AND n_it_wrong == 0 "
             "(accepted IT hit with cluster tpIdx != spixMatchedTpIdx)",
    "d0acc": f"P(|d0 - tp_d0| <= {CLEAN_PCT} pct of |d0 - tp_d0| over clean matched training tracks)",
    "z0acc": f"P(|z0 - tp_z0| <= {CLEAN_PCT} pct of |z0 - tp_z0| over clean matched training tracks)",
    "log_sd0": f"log(max(0.5*(q84.1 - q15.9), {SIGMA_FLOOR_UM})) of d0 - tp_d0 in um "
               "(quantile HGB pair)",
    "log_sz0": f"log(max(0.5*(q84.1 - q15.9), {SIGMA_FLOOR_UM})) of z0 - tp_z0 in um "
               "(quantile HGB pair)",
}
WITHHELD = ("every d0-derived quantity (d0, |d0|, d0/sigma, sigma_KF(d0), refit-base d0 shift) "
            "and the z0 value: d0 features buy AUC by penalising displaced tracks (the trap); "
            "z0 would encode where the track sits along the beam. |d0_refit| only routes "
            "refit tracks to a window.")


def table_names(family: str, config: str) -> dict:
    if family not in FAMILIES:
        raise ValueError(f"family {family!r} not in {sorted(FAMILIES)}")
    return {k: v.format(cfg=config) for k, v in FAMILIES[family].items()}


# ---------------------------------------------------------------------------
# nano -> flat per-track table
# ---------------------------------------------------------------------------
def nano_branches(keys, family: str = "prompt", config: str = "AAAA",
                  with_truth: bool = True, with_event_id: bool = True) -> list[str]:
    """The branches ``tracks_from_batch`` needs, resolved on a file whose
    branch names are `keys` (TP links by row or, on older nanos, by key)."""
    tn = table_names(family, config)
    keys = set(keys)
    br = [branch_name(keys, tn["base"], c) for c in BASE_COLS]
    br += [branch_name(keys, tn["refit"], c) for c in REFIT_COLS]
    br += [branch_name(keys, tn["hits"], c) for c in HIT_COLS]
    if with_truth:
        br += [branch_name(keys, tn["base"], c) for c in BASE_TRUTH]
        br += branch_names(keys, tn["refit"], REFIT_TRUTH)
        br += [branch_name(keys, tn["hits"], c) for c in HIT_TRUTH]
        br += branch_names(keys, CLUSTER_TABLE, ["tpIdx"])
    if with_event_id:
        br += [c for c in EVENT_ID if c in keys]
    return list(dict.fromkeys(br))


def _flat(A, name, dtype=np.float64):
    return ak.to_numpy(ak.flatten(A[name])).astype(dtype)


def _second_largest(values, group, n_groups):
    """Per group, the second-largest value (0 when the group has < 2 rows):
    the largest after removing ONE occurrence of the maximum."""
    out = np.zeros(n_groups)
    if len(values) == 0:
        return out
    order = np.lexsort((-values, group))
    g, v = group[order], values[order]
    first = np.r_[True, g[1:] != g[:-1]]
    second = np.r_[False, first[:-1]] & ~first      # 2nd row of a group of >= 2
    out[g[second]] = v[second]
    return out


def hit_aggregates(hits: dict, g: np.ndarray, n_tracks: int) -> dict:
    """Per-track aggregates of the ACCEPTED IT hits (online quantities).
    `hits` maps HIT_COLS to flat arrays; `g` is each hit's global track row."""
    acc = hits["hitAccepted"] > 0
    gi = g[acc]
    a = {c: v[acc] for c, v in hits.items()}
    N = n_tracks
    out = {}
    nacc = np.bincount(gi, minlength=N).astype(np.float64)
    out["h_nangle"] = np.bincount(gi, weights=a["hasAlpha"], minlength=N)
    out["h_chi2_angle_per_cl"] = (np.bincount(gi, weights=a["chi2IncAlpha"] * a["hasAlpha"], minlength=N)
                                  / np.maximum(out["h_nangle"], 1))
    out["h_chi2_pos_per_hit"] = (np.bincount(gi, weights=a["chi2IncX"] + a["chi2IncY"], minlength=N)
                                 / np.maximum(nacc, 1))
    out["h_mean_wmult"] = np.bincount(gi, weights=a["windowMult"], minlength=N) / np.maximum(nacc, 1)
    out["h_nmerged"] = np.bincount(gi, weights=a["clusterMerged"], minlength=N)
    pa = np.where(a["hasAlpha"] > 0, np.abs(a["pullAlpha"]), 0.0)
    pp = np.maximum(np.abs(a["pullX"]), np.abs(a["pullY"]))
    mx_a = np.zeros(N)
    mx_p = np.zeros(N)
    np.maximum.at(mx_a, gi, pa)
    np.maximum.at(mx_p, gi, pp)
    out["h_max_angle_pull"] = mx_a
    out["h_second_angle_pull"] = _second_largest(pa, gi, N)
    out["h_max_pos_pull"] = mx_p
    return out


def tracks_from_batch(A, family: str = "prompt", config: str = "AAAA",
                      with_truth: bool = True):
    """Flatten one uproot batch (read with ``nano_branches``) to per-track
    numpy columns. Returns (T, ntrk): T keys b_<col>, r_<col>, h_<agg>,
    b_nmiss_interior and, with truth, the truth columns plus t_n_it_wrong;
    ntrk = tracks per event. Raises if base and refit tables are not 1:1."""
    tn = table_names(family, config)
    f = A.fields
    bn = lambda tab, c: branch_name(f, tab, c)  # noqa: E731
    ntrk = ak.to_numpy(ak.num(A[bn(tn["base"], "pt")])).astype(np.int64)
    nref = ak.to_numpy(ak.num(A[bn(tn["refit"], "pt")]))
    if not np.array_equal(ntrk, nref):
        raise RuntimeError(f"{tn['base']} and {tn['refit']} are not 1:1 row aligned "
                           f"({int((ntrk != nref).sum())} events differ)")
    N = int(ntrk.sum())
    off = np.concatenate([[0], np.cumsum(ntrk)[:-1]]).astype(np.int64)
    T = {"b_" + c: _flat(A, bn(tn["base"], c)) for c in BASE_COLS}
    T.update({"r_" + c: _flat(A, bn(tn["refit"], c)) for c in REFIT_COLS})
    T["b_nmiss_interior"] = nlaymiss_interior(T["b_hitPattern"].astype(np.int64)).astype(np.float64)

    tidx = A[bn(tn["hits"], "trackIdx")]
    nh = ak.to_numpy(ak.num(tidx))
    evi = np.repeat(np.arange(len(nh)), nh)
    g = off[evi] + ak.to_numpy(ak.flatten(tidx)).astype(np.int64)
    if len(g) and (g.min() < 0 or np.any(g >= (off + ntrk)[evi])):
        raise RuntimeError(f"{tn['hits']}.trackIdx outside the per-event track rows")
    hits = {c: _flat(A, bn(tn["hits"], c)) for c in HIT_COLS}
    T.update(hit_aggregates(hits, g, N))

    if with_truth:
        for c in BASE_TRUTH:
            T["b_" + c] = _flat(A, bn(tn["base"], c))
        ttp = ak.to_numpy(ak.flatten(tp_idx(A, tn["refit"], "spixMatchedTpIdx"))).astype(np.int64)
        ctp_j = tp_idx(A, CLUSTER_TABLE)
        nc = ak.to_numpy(ak.num(ctp_j))
        coff = np.concatenate([[0], np.cumsum(nc)[:-1]]).astype(np.int64)
        ctp = ak.to_numpy(ak.flatten(ctp_j)).astype(np.int64)
        if np.any(ttp == -2) or np.any(ctp == -2):
            raise RuntimeError("TrackingParticle links unavailable (tpIdx -2, e.g. no L1TTP table in the "
                               "nano): n_it_wrong cannot be computed")
        sel = _flat(A, bn(tn["hits"], "selClusterIdx"), np.int64)
        if np.any(sel >= nc[evi]):
            raise RuntimeError("selClusterIdx outside the per-event cluster rows")
        htp = np.where(sel >= 0, ctp[np.where(sel >= 0, coff[evi] + sel, 0)] if len(ctp) else -2, -2)
        wrong = (htp != ttp[g]) & (hits["hitAccepted"] > 0)
        T["t_n_it_wrong"] = np.bincount(g[wrong], minlength=N).astype(np.float64)
    return T, ntrk


def build_dataset(files, family: str = "prompt", config: str = "AAAA",
                  step_events: int = 25, max_events: int | None = None,
                  with_truth: bool = True, log=print):
    """Read `files` in order, `step_events` events at a time (bounded RSS),
    into (T, E): T per-track columns (+ 'evt', the global event index), E
    per-event bookkeeping (file index, entry, run/lumi/event, ntrk)."""
    import uproot

    Ts, Es = [], []
    n_done = 0
    for fi, fn in enumerate(files):
        t = uproot.open(fn)["Events"]
        br = nano_branches(t.keys(), family, config, with_truth)
        stop = t.num_entries if max_events is None else min(t.num_entries, max_events - n_done)
        for start in range(0, max(stop, 0), step_events):
            A = t.arrays(br, entry_start=start, entry_stop=min(start + step_events, stop))
            T, ntrk = tracks_from_batch(A, family, config, with_truth)
            E = {"file": np.full(len(ntrk), fi), "entry": np.arange(start, start + len(ntrk)),
                 "ntrk": ntrk}
            for c in EVENT_ID:
                E[c] = ak.to_numpy(A[c]).astype(np.int64) if c in A.fields else np.full(len(ntrk), -1)
            Ts.append(T)
            Es.append(E)
            del A
        n_done += max(stop, 0)
        log(f"{os.path.basename(fn)}: {max(stop, 0)} ev")
        if max_events is not None and n_done >= max_events:
            break
    T = {k: np.concatenate([t[k] for t in Ts]) for k in Ts[0]}
    E = {k: np.concatenate([e[k] for e in Es]) for k in Es[0]}
    E["evt"] = np.arange(len(E["ntrk"]))
    T["evt"] = np.repeat(E["evt"], E["ntrk"])
    return T, E


# ---------------------------------------------------------------------------
# features / labels / windows (the pilot's definitions)
# ---------------------------------------------------------------------------
def _nan_sentinel(x):
    x = np.asarray(x, float).copy()
    x[x <= -998] = np.nan
    return x


def features(T, coll: str = "refit") -> dict:
    """Online per-track features in model order (no truth, no d0, no z0 value)."""
    inv = lambda pt: 1.0 / np.maximum(pt, 1e-3)  # noqa: E731
    F = {
        "inv_pt": inv(T["b_pt"]), "tanL": T["b_tanL"], "abs_eta": np.abs(T["b_eta"]),
        "nStubs": T["b_nStubs"], "nmiss_interior": T["b_nmiss_interior"],
        "chi2XYRed": T["b_chi2XYRed"], "chi2ZRed": T["b_chi2ZRed"], "chi2BendRed": T["b_chi2BendRed"],
        "trkMVA1": T["b_trkMVA1"],
        "sig_kf_rinv": np.sqrt(np.abs(T["b_cov_rInv_rInv"])), "sig_kf_phi": np.sqrt(np.abs(T["b_cov_phi_phi"])),
        "sig_kf_tanl": np.sqrt(np.abs(T["b_cov_tanL_tanL"])), "sig_kf_z0": np.sqrt(np.abs(T["b_cov_z0_z0"])),
    }
    if coll == "refit":
        F.update({
            "r_inv_pt": inv(T["r_pt"]), "r_tanL": T["r_tanL"],
            "dz0_refit": np.abs(T["r_z0"] - T["b_z0"]), "dtanl_refit": np.abs(T["r_tanL"] - T["b_tanL"]),
            "dinvpt_refit": inv(T["r_pt"]) - inv(T["b_pt"]),
        })
        for c in REFIT_SENTINEL_COLS:
            F[c] = _nan_sentinel(T["r_" + c])
        for c in HIT_FEATURES:
            F[c] = T[c]
    elif coll != "base":
        raise ValueError(f"coll {coll!r} not in {COLLS}")
    return F


def feature_matrix(T, coll: str):
    F = features(T, coll)
    return np.column_stack([F[n] for n in F]).astype(np.float64), list(F)


def d0_features(T, coll: str) -> dict:
    """The WITHHELD d0-derived features (trap ablation only, never a head input)."""
    p = "r_" if coll == "refit" else "b_"
    sk = np.sqrt(np.abs(T["b_cov_d0_d0"]))
    G = {"abs_d0": np.abs(T[p + "d0"]), "sig_kf_d0": sk, "d0_over_sigma": np.abs(T[p + "d0"]) / sk}
    if coll == "refit":
        G["dd0_refit"] = np.abs(T["r_d0"] - T["b_d0"])
    return G


def truth(T, coll: str):
    """(matched, clean, d0 residual um, z0 residual um); residuals NaN if unmatched."""
    p = "r_" if coll == "refit" else "b_"
    matched = T["b_tp_pt"] > 0
    clean = (T["b_genuine"] > 0) & ((T["t_n_it_wrong"] == 0) if coll == "refit" else True)
    rd0 = np.where(matched, T[p + "d0"] - T["b_tp_d0"], np.nan) * 1e4
    rz0 = np.where(matched, T[p + "z0"] - T["b_tp_z0"], np.nan) * 1e4
    return matched, np.asarray(clean, bool), rd0, rz0


def window_names(coll: str) -> list[str]:
    return ["prompt", "displaced"] if coll == "refit" else ["all"]


def windows(T, coll: str) -> dict:
    """{window: boolean track mask}. Refit: fitted |d0_refit| <= / > D0_SPLIT."""
    if coll == "base":
        return {"all": np.ones(len(T["b_pt"]), bool)}
    ad = np.abs(T["r_d0"])
    return {"prompt": ad <= D0_SPLIT, "displaced": ad > D0_SPLIT}


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------
def train_only_mask(files, train_only) -> np.ndarray:
    """Per-file boolean. `train_only` entries match a file by full path,
    basename, or (the pilot's form) a substring of the basename such as
    '_f01_'. Every entry must match at least one file."""
    names = [os.path.basename(str(f)) for f in files]
    m = np.zeros(len(files), bool)
    for t in train_only or ():
        hit = np.array([str(f) == str(t) or n == os.path.basename(str(t)) or str(t) in n
                        for f, n in zip(files, names)])
        if not hit.any():
            raise ValueError(f"train_only entry {t!r} matches none of the input files")
        m |= hit
    return m


def event_folds(file_idx, train_only_files, n_folds: int = N_FOLDS, seed: int = 0) -> np.ndarray:
    """Event -> fold 0..n_folds-1, or -1 for events of train-only files.
    `train_only_files` is the per-file boolean mask (``train_only_mask``)."""
    to = np.asarray(train_only_files, bool)[np.asarray(file_idx)]
    f = np.full(len(to), -1)
    idx = np.flatnonzero(~to)
    rng = np.random.default_rng(seed)
    f[idx[rng.permutation(len(idx))]] = np.arange(len(idx)) % n_folds
    return f


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def _hgb_cls(X, y, seed, params):
    from sklearn.ensemble import HistGradientBoostingClassifier as GB
    return GB(random_state=seed, **params).fit(X, y)


def _hgb_q(X, y, q, seed, params):
    from sklearn.ensemble import HistGradientBoostingRegressor as GB
    return GB(loss="quantile", quantile=q, random_state=seed, **params).fit(X, y)


def fit_heads(X, lab, tr, seed: int = SEED, params: dict | None = None) -> dict:
    """Fit every head on rows `tr`. Returns {'clean', 'd0acc', 'z0acc':
    classifier; 'd0_q', 'z0_q': {quantile: regressor}; 'thresholds_um':
    {'d0acc', 'z0acc'}}."""
    params = dict(HGB_PARAMS if params is None else params)
    matched, clean, rd0, rz0 = lab
    trm = tr[matched[tr]]
    cl_m = trm[clean[trm]]
    M = {"clean": _hgb_cls(X[tr], clean[tr], seed, params), "thresholds_um": {}}
    for name, r in (("d0acc", rd0), ("z0acc", rz0)):
        thr = float(np.percentile(np.abs(r[cl_m]), CLEAN_PCT))
        M["thresholds_um"][name] = thr
        M[name] = _hgb_cls(X[trm], np.abs(r[trm]) <= thr, seed, params)
    for name, r in (("d0_q", rd0), ("z0_q", rz0)):
        M[name] = {q: _hgb_q(X[trm], r[trm], q, seed, params) for q in QUANTILES}
    return M


def predict_heads(M: dict, X) -> np.ndarray:
    """(n, len(EVAL_COLUMNS)) array: OUTPUTS then the three d0 quantiles."""
    out = np.full((len(X), len(EVAL_COLUMNS)), np.nan)
    if len(X) == 0:
        return out
    for j, h in enumerate(("clean", "d0acc", "z0acc")):
        out[:, j] = M[h].predict_proba(X)[:, 1]
    for j, h in ((3, "d0_q"), (4, "z0_q")):
        qs = {q: m.predict(X) for q, m in M[h].items()}
        s = np.maximum(0.5 * (qs[QUANTILES[2]] - qs[QUANTILES[0]]), SIGMA_FLOOR_UM)
        out[:, j] = np.log(s)
        if h == "d0_q":
            out[:, 5], out[:, 6], out[:, 7] = qs[QUANTILES[0]], qs[QUANTILES[1]], qs[QUANTILES[2]]
    return out


def fit_predict(X, lab, tr, sc, seed: int = SEED, params: dict | None = None):
    return predict_heads(fit_heads(X, lab, tr, seed, params), X[sc])


def crossfit(T, coll: str, track_folds, mode: str = "nested", n_folds: int = N_FOLDS,
             seed: int = SEED, params: dict | None = None, log=print):
    """Cross-fitted scores (see the module docstring). Returns scores of shape
    (n_folds, n_tracks, len(EVAL_COLUMNS)) for 'nested'; (1, n_tracks, ...)
    holding the outer scores for 'outer'."""
    if mode not in ("outer", "nested"):
        raise ValueError(f"crossfit mode {mode!r}")
    X, _ = feature_matrix(T, coll)
    lab = truth(T, coll)
    tf = np.asarray(track_folds)
    W = windows(T, coll)
    nk = n_folds if mode == "nested" else 1
    scores = np.full((nk, len(tf), len(EVAL_COLUMNS)), np.nan)
    for k in range(n_folds):
        t0 = time.perf_counter()
        slot = k if mode == "nested" else 0
        for win in W.values():
            tr = np.flatnonzero(win & (tf != k))
            sc = np.flatnonzero(win & (tf == k))
            scores[slot, sc] = fit_predict(X, lab, tr, sc, seed, params)
            if mode == "outer":
                continue
            # training-only events: scored by a model trained on the test-eligible folds != k
            sc = np.flatnonzero(win & (tf == -1))
            if len(sc):
                tr = np.flatnonzero(win & (tf >= 0) & (tf != k))
                scores[slot, sc] = fit_predict(X, lab, tr, sc, seed, params)
            for j in range(n_folds):
                if j == k:
                    continue
                tr = np.flatnonzero(win & (tf != k) & (tf != j))
                sc = np.flatnonzero(win & (tf == j))
                scores[slot, sc] = fit_predict(X, lab, tr, sc, seed, params)
        log(f"[{coll}] {mode} fold {k} done ({time.perf_counter() - t0:.0f}s)")
    return scores


def outer_scores(scores, track_folds, mode: str):
    """The fully out-of-sample score of every test-eligible track."""
    if mode == "outer":
        return scores[0].copy()
    tf = np.asarray(track_folds)
    out = np.full(scores.shape[1:], np.nan)
    for k in range(scores.shape[0]):
        out[tf == k] = scores[k, tf == k]
    return out


# ---------------------------------------------------------------------------
# evaluation (the pilot's report)
# ---------------------------------------------------------------------------
def sig68(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return float("nan")
    q = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q[1] - q[0]))


def _nll(r, s):
    s = np.maximum(s, 1e-6)
    return float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * (r / s) ** 2))


def evaluate(T, coll: str, outer, track_folds) -> dict:
    """Per-track performance of the OUTER (fully out-of-sample) scores on the
    test-eligible tracks (fold >= 0), per window."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    matched, clean, rd0, rz0 = truth(T, coll)
    tested = np.asarray(track_folds) >= 0
    tp_d0 = np.abs(T["b_tp_d0"]) * 1e4
    res = {}
    for wname, w in windows(T, coll).items():
        win = w & tested
        R = {"n_tracks": int(win.sum()), "n_matched": int((win & matched).sum()),
             "clean_frac": float(clean[win].mean())}
        R["auc_clean"] = float(roc_auc_score(clean[win], outer[win, 0]))
        m = win & matched
        cl = m & clean
        for j, name, r in ((1, "d0acc", rd0), (2, "z0acc", rz0)):
            thr = np.percentile(np.abs(r[cl]), CLEAN_PCT)
            R[f"auc_{name}"] = float(roc_auc_score(np.abs(r[m]) <= thr, outer[m, j]))
            R[f"{name}_target_um"] = float(thr)
            s = outer[m, j]
            R[f"{name}_sigma68_at_keep"] = {
                str(f): sig68(r[m][s >= np.percentile(s, 100 * (1 - f))]) for f in (1.0, 0.5, 0.2, 0.1)}
        s = np.exp(outer[m, 3])
        rr = rd0[m]
        q16, q50, q84 = outer[m, 5], outer[m, 6], outer[m, 7]
        R["sd0_coverage_68"] = float(np.mean((rr >= q16) & (rr <= q84)))
        pull_c = (rr - q50) / s
        dec = np.percentile(s, np.linspace(0, 100, 11))
        R["sd0_pull68_by_decile"] = [sig68(pull_c[(s >= dec[i]) & (s <= dec[i + 1])]) for i in range(10)]
        R["sd0_pull68_all"] = sig68(pull_c)
        R["sd0_pred_p10_p50_p90_um"] = [float(v) for v in np.percentile(s, [10, 50, 90])]
        R["sd0_rank_sigma68_best_frac"] = {str(f): sig68(rr[s <= np.percentile(s, 100 * f)])
                                           for f in (0.1, 0.2, 0.5, 1.0)}
        R["spearman_sd0_absres"] = float(spearmanr(s, np.abs(rr - q50)).statistic)
        glob_s = sig68(rr - q50)
        R["nll_sd0"] = _nll(rr - q50, s)
        R["nll_global"] = _nll(rr - q50, np.full_like(s, glob_s))
        if coll == "base":
            R["nll_kf"] = _nll(rr - q50, np.sqrt(np.abs(T["b_cov_d0_d0"][m])) * 1e4)
        sz = np.exp(outer[m, 4])
        R["sz0_rank_sigma68_best_frac"] = {str(f): sig68(rz0[m][sz <= np.percentile(sz, 100 * f)])
                                           for f in (0.1, 0.2, 0.5, 1.0)}
        s1 = outer[m, 1]
        keep = s1 >= np.percentile(s1, 20)
        good = clean[m]
        bands = [(0, 50), (50, 200), (200, 1000), (1000, 1e9)]
        R["d0acc_keep80_retention_by_true_d0"] = {}
        for lo, hi in bands:
            sel = good & (tp_d0[m] >= lo) & (tp_d0[m] < hi)
            R["d0acc_keep80_retention_by_true_d0"][f"{lo}-{hi if hi < 1e9 else 'inf'}um"] = \
                [float(keep[sel].mean()) if sel.sum() else None, int(sel.sum())]
        res[wname] = R
    return res


def trap_ablation(T, coll: str, track_folds, n_folds: int = N_FOLDS, seed: int = SEED,
                  params: dict | None = None, log=print) -> dict:
    """d0acc with vs without the withheld d0-derived features (K-fold
    out-of-sample): documents why they are withheld. Not a deployable head."""
    from sklearn.metrics import roc_auc_score

    params = dict(HGB_PARAMS if params is None else params)
    matched, clean, rd0, _ = truth(T, coll)
    X0, _ = feature_matrix(T, coll)
    X1 = np.column_stack([X0] + list(d0_features(T, coll).values()))
    tf = np.asarray(track_folds)
    tp_d0 = np.abs(T["b_tp_d0"]) * 1e4
    out = {}
    for wname, w in windows(T, coll).items():
        m = w & (tf >= 0) & matched
        s0 = np.full(len(tf), np.nan)
        s1 = np.full(len(tf), np.nan)
        for k in range(n_folds):
            tr = np.flatnonzero(w & matched & (tf != k))
            te = np.flatnonzero(m & (tf == k))
            thr = np.percentile(np.abs(rd0[tr][clean[tr]]), CLEAN_PCT)
            y = np.abs(rd0[tr]) <= thr
            s0[te] = _hgb_cls(X0[tr], y, seed, params).predict_proba(X0[te])[:, 1]
            s1[te] = _hgb_cls(X1[tr], y, seed, params).predict_proba(X1[te])[:, 1]
        thr = np.percentile(np.abs(rd0[m & clean]), CLEAN_PCT)
        y = np.abs(rd0[m]) <= thr
        rec = {}
        good = clean[m]
        far = good & (tp_d0[m] > 200)
        for nm, s in (("without_d0", s0[m]), ("with_d0", s1[m])):
            keep = s >= np.percentile(s, 20)
            rec[nm] = {"auc": float(roc_auc_score(y, s)),
                       "keep80_good_retention_true_d0_gt200um": float(keep[far].mean()) if far.sum() else None,
                       "n_good_true_d0_gt200um": int(far.sum())}
        out[wname] = rec
        log(f"[{coll}/{wname}] trap ablation: {rec}")
    return out


# ---------------------------------------------------------------------------
# frozen bundle
# ---------------------------------------------------------------------------
def bundle_stem(family: str, config: str) -> str:
    return f"trackmva_{family}_{config}"


def _package_versions() -> dict:
    import importlib.metadata as md

    v = {"python": platform.python_version(), "platform": platform.platform()}
    for p in ("numpy", "scikit-learn", "scipy", "awkward", "uproot", "joblib"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    return v


def _git_state() -> dict:
    # the ngtagger package directory; "src_dirty" = uncommitted changes under it
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        head = subprocess.run(["git", "-C", pkg, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", pkg, "status", "--porcelain", "--", "."],
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": head, "src_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "src_dirty": None}


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fit_frozen(T, colls=("refit",), seed: int = SEED, params: dict | None = None, log=print):
    """Final models on ALL events (train-only included), per coll and window.
    Returns ({coll: {window: heads}}, {coll: {window: {n_train, thresholds_um}}})."""
    models, info = {}, {}
    for coll in colls:
        X, _ = feature_matrix(T, coll)
        lab = truth(T, coll)
        models[coll], info[coll] = {}, {}
        for wname, w in windows(T, coll).items():
            t0 = time.perf_counter()
            tr = np.flatnonzero(w)
            M = fit_heads(X, lab, tr, seed, params)
            info[coll][wname] = {"n_train_tracks": int(len(tr)),
                                 "n_train_matched": int(lab[0][tr].sum()),
                                 "thresholds_um": M.pop("thresholds_um")}
            models[coll][wname] = M
            log(f"[{coll}/{wname}] frozen fit on {len(tr)} tracks ({time.perf_counter() - t0:.0f}s)")
    return models, info


def save_bundle(output_dir, models, meta: dict, family: str, config: str) -> str:
    """Write <stem>.joblib + <stem>_meta.json (with the joblib sha256)."""
    import joblib

    os.makedirs(output_dir, exist_ok=True)
    stem = bundle_stem(family, config)
    jpath = os.path.join(output_dir, stem + ".joblib")
    joblib.dump(models, jpath, compress=3)
    meta = dict(meta, model_file=os.path.basename(jpath), model_sha256=_sha256(jpath))
    mpath = os.path.join(output_dir, stem + "_meta.json")
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=1)
    return mpath


class TrackMVA:
    """A loaded frozen bundle. ``predict(T)`` scores flat per-track columns
    (as ``tracks_from_batch`` returns them, truth not needed)."""

    def __init__(self, models: dict, meta: dict):
        self.models = models
        self.meta = meta
        self.family = meta["family"]
        self.config = meta["config"]
        self.tables = meta["tables"]

    @property
    def colls(self):
        return list(self.models)

    def predict(self, T, coll: str = "refit", quantiles: bool = False) -> dict:
        """{head: per-track float array} (+ q16_d0/q50_d0/q84_d0 with
        `quantiles`); each track scored by the model of its window."""
        if coll not in self.models:
            raise KeyError(f"bundle has no {coll!r} models (has {self.colls})")
        X, names = feature_matrix(T, coll)
        # bundles frozen before the OT rename name OT-track features by their "seed" spelling
        old_to_new = {o: n for n, o in OT_RENAMES.items()}
        if names != [old_to_new.get(n, n) for n in self.meta["features"][coll]]:
            raise RuntimeError(f"feature order mismatch for {coll}: bundle {self.meta['features'][coll]} "
                               f"vs code {names}")
        out = np.full((len(X), len(EVAL_COLUMNS)), np.nan)
        for wname, w in windows(T, coll).items():
            idx = np.flatnonzero(w)
            out[idx] = predict_heads(self.models[coll][wname], X[idx])
        cols = EVAL_COLUMNS if quantiles else OUTPUTS
        return {c: out[:, EVAL_COLUMNS.index(c)] for c in cols}


def load_bundle(path, family: str | None = None, config: str | None = None,
                verify: bool = True) -> TrackMVA:
    """Load a bundle from its directory (with `family`/`config` if it holds
    several) or from its _meta.json path. Checks the sha256, the spec version
    and warns when the scikit-learn version differs from the training one
    (pickled HGB models are only guaranteed on the same version)."""
    import warnings

    import joblib

    if os.path.isdir(path):
        if family and config:
            mpath = os.path.join(path, bundle_stem(family, config) + "_meta.json")
        else:
            cands = sorted(f for f in os.listdir(path) if f.startswith("trackmva_") and f.endswith("_meta.json"))
            if len(cands) != 1:
                raise ValueError(f"{path}: {len(cands)} bundles ({cands}); pass family and config")
            mpath = os.path.join(path, cands[0])
    else:
        mpath = path
    with open(mpath) as f:
        meta = json.load(f)
    if meta.get("spec_version") != SPEC_VERSION:
        raise RuntimeError(f"{mpath}: spec_version {meta.get('spec_version')} != code {SPEC_VERSION}")
    jpath = os.path.join(os.path.dirname(mpath), meta["model_file"])
    if verify and _sha256(jpath) != meta["model_sha256"]:
        raise RuntimeError(f"{jpath}: sha256 does not match its metadata")
    have = _package_versions().get("scikit-learn")
    if have != meta["package_versions"].get("scikit-learn"):
        warnings.warn(f"trackmva bundle trained with scikit-learn "
                      f"{meta['package_versions'].get('scikit-learn')}, loading with {have}")
    return TrackMVA(joblib.load(jpath), meta)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _file_info(files, to_mask):
    out = []
    for f, to in zip(files, to_mask):
        st = os.stat(f)
        out.append({"path": os.path.abspath(f), "basename": os.path.basename(f), "size": st.st_size,
                    "mtime": int(st.st_mtime), "train_only": bool(to)})
    return out


def train_trackmva(files, output_dir, family: str = "prompt", config: str = "AAAA",
                   colls=("refit",), train_only=(), n_folds: int = N_FOLDS,
                   crossfit_mode: str = "nested", fold_seed: int = 0, seed: int = SEED,
                   hgb_params: dict | None = None, step_events: int = 25,
                   max_events: int | None = None, trap: bool = False,
                   dataset_cache: str | None = None, freeze: bool = True, log=print):
    """Build the dataset from nano, cross-fit + evaluate, fit and save the
    frozen bundle. `files` is the full ordered input list; `train_only`
    selects the subset whose events are training-only (see ``train_only_mask``).
    Returns the path of the report JSON."""
    files = [str(f) for f in files]
    colls = list(colls)
    for c in colls:
        if c not in COLLS:
            raise ValueError(f"coll {c!r} not in {COLLS}")
    params = dict(HGB_PARAMS if hgb_params is None else hgb_params)
    to_mask = train_only_mask(files, train_only)
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, bundle_stem(family, config))

    if dataset_cache and os.path.exists(dataset_cache):
        d = np.load(dataset_cache, allow_pickle=False)
        if [str(x) for x in d["files"]] != files:
            raise RuntimeError(f"{dataset_cache} was built from other files")
        T = {k[2:]: d[k] for k in d.files if k.startswith("T_")}
        E = {k[2:]: d[k] for k in d.files if k.startswith("E_")}
        log(f"dataset from cache {dataset_cache}")
    else:
        T, E = build_dataset(files, family, config, step_events, max_events, True, log)
        if dataset_cache:
            np.savez_compressed(dataset_cache, **{"T_" + k: v for k, v in T.items()},
                                **{"E_" + k: v for k, v in E.items()}, files=np.array(files))
    folds_ev = event_folds(E["file"], to_mask, n_folds, fold_seed)
    tf = folds_ev[T["evt"]]
    log(f"{len(E['evt'])} events ({int((folds_ev >= 0).sum())} test-eligible), {len(tf)} tracks")

    report = {"family": family, "config": config, "n_events": int(len(E["evt"])),
              "n_tracks": int(len(tf)), "n_folds": n_folds, "crossfit": crossfit_mode,
              "n_test_events": int((folds_ev >= 0).sum()),
              "train_only_files": [os.path.basename(f) for f, t in zip(files, to_mask) if t]}
    oof = {"folds_ev": folds_ev, "track_evt": T["evt"].astype(np.int64),
           "track_row": (np.arange(len(tf)) - np.repeat(np.concatenate([[0], np.cumsum(E["ntrk"])[:-1]]),
                                                         E["ntrk"])).astype(np.int64),
           "ev_file": E["file"], "ev_entry": E["entry"], "outputs": np.array(OUTPUTS),
           "files": np.array(files)}
    for coll in colls:
        rep = {"features": list(features(T, coll))}
        if crossfit_mode != "none":
            scores = crossfit(T, coll, tf, crossfit_mode, n_folds, seed, params, log)
            outer = outer_scores(scores, tf, crossfit_mode)
            rep["perf"] = evaluate(T, coll, outer, tf)
            oof[f"{coll}_scores"] = scores[:, :, :len(OUTPUTS)].astype(np.float32)
            oof[f"{coll}_outer"] = outer.astype(np.float32)
            log(json.dumps({w: {k: v for k, v in r.items() if k.startswith(("auc", "sd0_cov", "sd0_pull68_all"))}
                            for w, r in rep["perf"].items()}))
        if trap:
            rep["trap_ablation"] = trap_ablation(T, coll, tf, n_folds, seed, params, log)
        report[coll] = rep
    if crossfit_mode != "none":
        np.savez_compressed(stem + "_oof.npz", **oof)

    if freeze:
        models, info = fit_frozen(T, colls, seed, params, log)
        meta = {
            "what": "SmartPixels per-track resolution-quality MVAs (ngtagger.train.trackmva)",
            "spec_version": SPEC_VERSION, "family": family, "config": config,
            "tables": table_names(family, config), "colls": colls,
            "features": {c: report[c]["features"] for c in colls},
            "outputs": OUTPUTS, "head_definitions": HEAD_DEFINITIONS, "withheld_features": WITHHELD,
            "windows": {c: window_names(c) for c in colls},
            "window_rule": f"refit: |d0_refit| <= {D0_SPLIT} cm -> prompt, else displaced; base: all",
            "sigma_floor_um": SIGMA_FLOOR_UM, "quantiles": list(QUANTILES),
            "model_family": "sklearn HistGradientBoostingClassifier / Regressor(loss=quantile)",
            "hgb_params": params, "random_state": seed, "frozen_fit": info,
            "training_files": _file_info(files, to_mask), "max_events": max_events,
            "event_split": {"n_folds": n_folds, "fold_seed": fold_seed, "crossfit": crossfit_mode,
                            "fold_by_file": {os.path.basename(f): folds_ev[E["file"] == i].tolist()
                                             for i, f in enumerate(files)},
                            "event_id_by_file": {os.path.basename(f): np.column_stack(
                                [E[c][E["file"] == i] for c in EVENT_ID]).tolist()
                                for i, f in enumerate(files)}},
            "oof_scores_file": os.path.basename(stem + "_oof.npz") if crossfit_mode != "none" else None,
            "package_versions": _package_versions(), "git": _git_state(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv,
        }
        mpath = save_bundle(output_dir, models, meta, family, config)
        log(f"frozen bundle: {mpath}")
    rpath = stem + "_report.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=1)
    log(f"report: {rpath}")
    return rpath
