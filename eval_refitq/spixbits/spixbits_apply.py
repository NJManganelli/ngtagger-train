"""Reference helper: apply a shipped SmartPixels quantization scenario to today's extended-collection tracks through
the correctionlib file of the detector scenario (spixbits_<AAAA|AAAI|IAAA>.json.gz) and the head models shipped beside
it. Needs numpy, scipy, correctionlib >= 2.6.

    import spixbits_apply as SA
    cset, meta = SA.load("spixbits_AAAA.json.gz")
    heads = SA.load_heads("heads/retrained", "AAAA")          # or "heads/deployed"
    out = SA.apply(cset, meta, "alloc_4444", heads, a=a, cov_diag=cov_diag, refit=refit, clean_margin=m, inputs=inputs,
                   ot_inputs=ot_inputs, ot_a=ot_a, ot_cov_diag=ot_cov_diag, truth=truth, ids=(run, lumi, event, track))

Inputs, one entry per track of L1TSmartPixelsExtTrack as produced today (flattened over events):
  a            (n, 5) rInv [1/cm], phi0 [rad], tanL, z0 [cm], d0 [cm]
  cov_diag     (n, 5) the diagonal of the track covariance, same order and units
  refit        (n,)   spixRefitPerformed
  clean_margin (n,)   today's DEPLOYED clean head as a margin (to_margin(head_clean)): a cell variable only
  inputs       (n, 25) the heads' inputs 14-38 of today's track, in the order of meta["head_inputs_14_38"]
                    (-999 = missing); meta = the second value of load()
  ot_inputs    (n, 13) the heads' inputs 1-13 (from the OT track; no quantization scenario changes them)
  ot_a, ot_cov_diag  (n, 5) the OT track's parameters and covariance diagonal (a track whose refit does not run carries them)
  truth        (n, 5) the OT track's particle in the same order (otTp_rInv, otTp_phi0, otTp_tanL, otTp_z0, otTp_d0), NaN
               where the track has none (otTpIdx < 0)
  ids          (run, lumi, event, track): track = the index along axis 1 of the event's track collection
Output: {"a", "cov_diag", "inputs" (n, 25: the emulated head inputs 14-38), "margins" (n, 6: the head models evaluated on
the emulated inputs: clean as a margin, then log sigma_hat of d0, z0, tanL, 1/R, phi), "changed"}. The draws depend on
the track's identity only, never on the quantization scenario, so scenarios stay comparable track by track. pT and
eta follow from the new rInv and tanL (pT = kpt / |rInv|); full_covariance() rebuilds the matrix with today's
correlations. Hardware words (hw*) and spixCompactWord are NOT recomputed: do not read them beside emulated values.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

PARAMS = ("rInv", "phi0", "tanL", "z0", "d0")
HEADS = ("clean", "d0", "z0", "tanl", "rinv", "phi")
WINDOWS = ("prompt", "displaced")
COUNTS = ("spixRefitPerformed", "spixNCrossings", "spixNAcceptedHits", "spixLayerHitMask", "spixMaxWindowMult")
LINEAR = ("spixLogDetRatio", "h_nangle", "h_mean_wmult")
PARTNER = {"z0": "d0", "tanL": "z0", "rInv": "d0", "phi0": "d0"}
D0_SPLIT, MISSING = 0.02, -999.0


def load_heads(directory, scenario):
    """{(window, head): trees} of one detector scenario from <directory>/SmartPixels/<scenario>/<window>/<head>.json
    (conifer JSON; a split goes left when x < threshold)."""
    out = {}
    for w in WINDOWS:
        for h in HEADS:
            m = json.load(open(Path(directory) / "SmartPixels" / scenario / w / f"{h}.json"))
            assert m["splitting_convention"] == "<" and m["n_features"] == 38
            trees = [t[0] for t in m["trees"]]
            n = max(len(t["feature"]) for t in trees)
            pad = lambda key, fill, dt: np.array([list(t[key]) + [fill] * (n - len(t[key])) for t in trees], dt)
            out[(w, h)] = {"init": float(m["init_predict"][0]), "feature": pad("feature", -2, np.int64),
                           "threshold": pad("threshold", 0.0, np.float32), "value": pad("value", 0.0, np.float64),
                           "left": pad("children_left", -1, np.int64), "right": pad("children_right", -1, np.int64),
                           "depth": int(m["max_depth"])}
    return out


def _margin(M, X):
    out = np.full(len(X), M["init"])
    rows = np.arange(len(X))
    for k in range(len(M["feature"])):
        node = np.zeros(len(X), np.int64)
        for _ in range(M["depth"] + 1):
            f = M["feature"][k][node]
            go_left = X[rows, np.maximum(f, 0)] < M["threshold"][k][node]
            node = np.where(f < 0, node, np.where(go_left, M["left"][k][node], M["right"][k][node]))
        out += M["value"][k][node]
    return out


def head_margins(heads, X, d0):
    """(n, 6) head outputs on the 38 inputs X (float32), the window chosen by |d0| <= 0.02 cm."""
    X = np.asarray(X, np.float32)
    prompt = np.abs(d0) <= D0_SPLIT
    out = np.full((len(X), len(HEADS)), np.nan)
    for w, sel in (("prompt", prompt), ("displaced", ~prompt)):
        idx = np.flatnonzero(sel)
        if len(idx):
            for i, h in enumerate(HEADS):
                out[idx, i] = _margin(heads[(w, h)], X[idx])
    return out


def load(path):
    """(the CorrectionSet, its metadata dict) of spixbits_<detector scenario>.json.gz."""
    import gzip
    import correctionlib
    with gzip.open(path, "rt") as fh:
        text = fh.read()
    return correctionlib.CorrectionSet.from_string(text), json.loads(json.loads(text)["description"])


def to_margin(p_clean):
    """head_clean (a probability) as the margin the tables use."""
    p = np.clip(np.asarray(p_clean, np.float64), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def to_p_clean(margin):
    return 1.0 / (1.0 + np.exp(-np.asarray(margin, np.float64)))


def full_covariance(cov_today, cov_diag_new):
    """(n, 5, 5): today's correlations with the new diagonal (positive definite whenever today's matrix is)."""
    d0 = np.sqrt(np.maximum(np.einsum("nii->ni", cov_today), 1e-300))
    d1 = np.sqrt(np.maximum(cov_diag_new, 0.0))
    return cov_today / (d0[:, :, None] * d0[:, None, :]) * (d1[:, :, None] * d1[:, None, :])


def _natural(name, x):
    x = np.asarray(x, np.float64)
    y = x.copy() if name in LINEAR else np.log(np.maximum(x, 0.0) + 1e-3)
    return np.where(x <= -998, np.nan, y)


def _unnatural(name, y):
    x = np.asarray(y, np.float64).copy() if name in LINEAR else np.maximum(np.exp(y) - 1e-3, 0.0)
    return np.where(np.isfinite(y), x, MISSING)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def apply(cset, M, allocation, heads, a, cov_diag, refit, clean_margin, inputs, ot_inputs, ot_a, ot_cov_diag, truth, ids=None,
          draws=None):
    kpt, names = M["kpt_GeV_cm"], M["head_inputs_14_38"]
    IN = {c: i for i, c in enumerate(names)}
    scores = names[11:14] + names[15:]                 # the mapped inputs: chi2-like (without spixChi2ITAtOT) and aggregates
    a = np.asarray(a, np.float64)
    cov_diag = np.asarray(cov_diag, np.float64)
    inputs = np.asarray(inputs, np.float64)
    ot_a = np.asarray(ot_a, np.float64)
    refit = np.asarray(refit).astype(bool)
    if draws is None:
        run, lumi, event, track = (np.asarray(x).astype(np.int64) for x in ids)
        uni = lambda k: cset["uniform"].evaluate(run, lumi, event, track, k)
        nor = lambda k: cset["normal"].evaluate(run, lumi, event, track, k)
        draws = {"u_state": uni(1), "u_rank": uni(2), "u_score": uni(3), "u_cov": uni(4), "n_z0": nor(11), "n_tanL": nor(12), "n_rInv": nor(13),
                 "n_phi0": nor(14)}
    pt = kpt / np.maximum(np.abs(a[:, 0]), 1e-12)
    abs_eta = np.abs(np.arcsinh(a[:, 2]))
    window = np.where(refit, (np.abs(a[:, 4]) > D0_SPLIT).astype(np.int64), 2)
    w2 = np.minimum(window, 1)
    matched = np.isfinite(truth).all(1).astype(np.int64)
    mult, n_hits = inputs[:, IN["spixMaxWindowMult"]], inputs[:, IN["spixNAcceptedHits"]]
    mask = inputs[:, IN["spixLayerHitMask"]].astype(np.int64)
    sg = np.sqrt(np.maximum(cov_diag, 1e-30))
    with np.errstate(invalid="ignore"):
        pull = np.nan_to_num(np.maximum(np.abs(a[:, 4] - truth[:, 4]) / sg[:, 4], np.abs(a[:, 3] - truth[:, 3]) / sg[:, 3]), nan=0.0)
    changed = draws["u_state"] < cset["p_changed"].evaluate(allocation, matched, window, abs_eta, pt, mult, np.asarray(clean_margin, np.float64), pull)
    # ---- precision part
    z = {"d0": ndtri(np.clip(draws["u_rank"], 1e-12, 1 - 1e-12))}
    for p in ("z0", "tanL", "rInv", "phi0"):
        r = cset["rho"].evaluate(allocation, p, abs_eta, w2)
        z[p] = r * z[PARTNER[p]] + np.sqrt(1 - r * r) * draws["n_" + p]
    new_a, new_cov = a.copy(), cov_diag.copy()
    for i, p in enumerate(PARAMS):
        e = cset["eps"].evaluate(allocation, p, abs_eta, w2, ndtr(z[p]))
        lam = cset["lambda"].evaluate(allocation, p, w2, abs_eta, pt)
        co = [cset["cov_scale"].evaluate(allocation, p, c, w2, abs_eta, pt, mask) for c in ("a", "b", "sigma")]
        lr = co[0] + co[1] * np.log(sg[:, i])
        if M.get("cov_scatter", True):
            lr = lr + co[2] * cset["cov_eta"].evaluate(allocation, p, draws["u_cov"])
        cov_i = cov_diag[:, i] * np.exp(np.clip(lr, np.log(0.2), np.log(1.0e4)))
        unit = np.sqrt(np.abs(cov_i - cov_diag[:, i]) + (0.05 * sg[:, i]) ** 2)       # the sigma the scenario adds
        new_a[:, i] += np.where(refit, lam * unit * e, 0.0)
        new_cov[:, i] = np.where(refit, cov_i, new_cov[:, i])
    w_arm = (np.abs(new_a[:, 4]) > D0_SPLIT).astype(np.int64)
    tail = np.abs(2 * draws["u_rank"] - 1)
    X = np.zeros((len(a), 25))
    for s in scores:
        t = _natural(s, inputs[:, IN[s]])
        co = [cset["score_map"].evaluate(allocation, s, c, w_arm, n_hits, abs_eta) for c in ("a", "b", "sigma")]
        eta = cset["score_eta"].evaluate(allocation, s, tail, draws["u_score"])
        X[:, IN[s]] = _unnatural(s, np.where(refit, co[0] + co[1] * t + co[2] * eta, t))
    for c in COUNTS:
        X[:, IN[c]] = inputs[:, IN[c]]
    # ---- hit-choice part: everything from one donor
    ch = np.flatnonzero(changed)
    if len(ch):
        don = lambda q: cset["donor"].evaluate(allocation, q, matched[ch], window[ch], abs_eta[ch], pull[ch], draws["u_rank"][ch])
        has = matched[ch] > 0
        for i, p in enumerate(PARAMS):
            new_a[ch, i] = np.where(has, np.where(has, truth[ch, i], 0.0) + sg[ch, i] * don(f"matched_ratio_{p}"),
                                    a[ch, i] + sg[ch, i] * don(f"delta_ratio_{p}"))
            new_cov[ch, i] = cov_diag[ch, i] * don(f"cov_ratio_{p}")
        for c in list(COUNTS) + scores:
            X[ch, IN[c]] = don(c)
    new_a[:, 1] = _wrap(new_a[:, 1])
    # ---- the heads' inputs that follow from the parameters; a track whose refit does not run carries the OT track
    ran = X[:, IN["spixRefitPerformed"]] > 0
    new_a = np.where(ran[:, None], new_a, ot_a)
    new_cov[~ran] = np.asarray(ot_cov_diag, np.float64)[~ran]
    ot_inv = np.asarray(ot_inputs)[:, 0].astype(np.float64)
    X[:, 0] = np.where(ran, 1.0 / np.maximum(kpt / np.maximum(np.abs(new_a[:, 0]), 1e-12), 1e-3), ot_inv)
    X[:, 1] = new_a[:, 2]
    X[:, 2] = np.abs(new_a[:, 3] - ot_a[:, 3])
    X[:, 3] = np.abs(new_a[:, 2] - ot_a[:, 2])
    X[:, 4] = X[:, 0] - ot_inv
    X[:, IN["spixAnyWindowTruncated"]] = inputs[:, IN["spixAnyWindowTruncated"]]
    X[:, IN["spixChi2ITAtOT"]] = MISSING                # always missing in refit v3
    X[:, IN["h_nangle"]] = np.clip(np.rint(X[:, IN["h_nangle"]]), 0, np.maximum(X[:, IN["spixNAcceptedHits"]], 0))
    X = X.astype(np.float32)
    margins = head_margins(heads, np.hstack([np.asarray(ot_inputs, np.float32), X]), new_a[:, 4])
    return {"a": new_a, "cov_diag": new_cov, "inputs": X, "margins": margins, "changed": changed}
