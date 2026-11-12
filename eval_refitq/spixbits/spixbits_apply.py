"""Reference helper: apply a shipped SmartPixels quantization scenario to today's extended-collection tracks through
the files of the detector scenario: the correctionlib file spixbits_<AAAA|AAAI|IAAA>.json.gz (shares, parameter noise),
spixbits_<...>_maps.npz (the tree maps of the covariance and of the heads' inputs; the donor pool) and the head models. Needs numpy, scipy,
correctionlib >= 2.6.

    import spixbits_apply as SA
    cset, meta = SA.load("spixbits_AAAA.json.gz")
    maps = SA.load_maps("spixbits_AAAA_maps.npz")
    heads = SA.load_heads("heads/retrained", "AAAA")          # or "heads/deployed"
    out = SA.apply(cset, meta, maps, "alloc_4444", heads, a=a, cov_diag=cov_diag, refit=refit, clean_margin=m, inputs=inputs,
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
LINEAR = ("spixLogDetRatio", "h_nangle", "h_mean_wmult")
PARTNER = {"z0": "d0", "tanL": "z0", "rInv": "d0", "phi0": "d0"}
D0_SPLIT, MISSING = 0.02, -999.0
N_COV, N_BITS = 5, 16                                 # covariance columns; layer-mask bits given to the maps


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


def _unnatural(name, y):
    x = np.asarray(y, np.float64).copy() if name in LINEAR else np.maximum(np.exp(y) - 1e-3, 0.0)
    return np.where(np.isfinite(y), x, MISSING)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def load_maps(path):
    """The tree maps of one detector scenario (spixbits_<detector scenario>_maps.npz)."""
    return np.load(path)


def _raw(m, k, Z):
    """Model k of an allocation's maps on the features Z: base + the sum over its trees (x <= threshold goes left)."""
    s, e = int(m["tree_start"][k]), int(m["tree_start"][k + 1])
    out = np.full(len(Z), float(m["base"][k]))
    if e == s:
        return out
    first = m["node_start"][s:e].astype(np.int64)[:, None]             # the first node of each tree
    for lo in range(0, len(Z), 20000):
        z = Z[lo:lo + 20000]
        rows = np.arange(len(z))[None, :]
        node = np.broadcast_to(first, (e - s, len(z))).copy()
        for _ in range(64):
            f = m["feature"][node]
            if (f < 0).all():
                break
            nxt = first + np.where(z[rows, np.maximum(f, 0)] <= m["threshold"][node], m["left"][node], m["right"][node])
            node = np.where(f < 0, node, nxt)
        out[lo:lo + 20000] += m["value"][node].sum(0, dtype=np.float64)
    return out


def _inv_cdf(q, knots, u):
    u = np.clip(u, knots[0], knots[-1])
    k = np.clip(np.searchsorted(knots, u, side="right") - 1, 0, len(knots) - 2)
    t = (u - knots[k]) / (knots[k + 1] - knots[k])
    return q[k] + t * (q[k + 1] - q[k])


def _block(m, name, first, Z, normals, knots):
    """The columns of a block of maps: mean of the two mean models + residual scale x standardized residual at the
    copula's draw, kept within the fitted range. Models of column j: first + 3 j (mean), + 1 (mean), + 2 (ln scale)."""
    u = ndtr(normals @ m[name + "_chol"].T)
    out = []
    for j in range(normals.shape[1]):
        k = first + 3 * j
        y = 0.5 * (_raw(m, k, Z) + _raw(m, k + 1, Z)) + np.exp(_raw(m, k + 2, Z)) * _inv_cdf(m[name + "_q"][j], knots, u[:, j])
        out.append(np.clip(y, m[name + "_lo"][j], m[name + "_hi"][j]))
    return np.column_stack(out)


def _bits(mask):
    return ((np.asarray(mask).astype(np.int64)[:, None] >> np.arange(N_BITS)) & 1).astype(np.float64)


def apply(cset, M, maps, allocation, heads, a, cov_diag, refit, clean_margin, inputs, ot_inputs, ot_a, ot_cov_diag, truth, ids=None,
          draws=None):
    kpt, names, knots = M["kpt_GeV_cm"], M["head_inputs_14_38"], np.asarray(M["knots"])
    IN = {c: i for i, c in enumerate(names)}
    scores = M["mapped_inputs"]
    a = np.asarray(a, np.float64)
    cov_diag = np.asarray(cov_diag, np.float64)
    inputs = np.asarray(inputs, np.float64)
    ot_inputs = np.asarray(ot_inputs, np.float64)
    ot_a = np.asarray(ot_a, np.float64)
    ot_cov_diag = np.asarray(ot_cov_diag, np.float64)
    refit = np.asarray(refit).astype(bool)
    if draws is None:
        run, lumi, event, track = (np.asarray(x).astype(np.int64) for x in ids)
        uni = lambda k: cset["uniform"].evaluate(run, lumi, event, track, k)
        nor = lambda k: cset["normal"].evaluate(run, lumi, event, track, k)
        draws = {"u_state": uni(1), "u_rank": uni(2), "n_z0": nor(11), "n_tanL": nor(12), "n_rInv": nor(13), "n_phi0": nor(14),
                 "n_cov": np.stack([nor(21 + i) for i in range(N_COV)], 1),
                 "n_score": np.stack([nor(41 + j) for j in range(len(scores))], 1),
                 "u_miss": np.stack([uni(61 + j) for j in range(len(scores))], 1)}
    m = {key: maps[f"{allocation}/{key}"] for key in ("base", "tree_start", "node_start", "feature", "threshold", "left", "right", "value",
                                                      "cov_q", "cov_lo", "cov_hi", "cov_chol", "score_q", "score_lo", "score_hi", "score_chol")}
    pt = kpt / np.maximum(np.abs(a[:, 0]), 1e-12)
    abs_eta = np.abs(np.arcsinh(a[:, 2]))
    window = np.where(refit, (np.abs(a[:, 4]) > D0_SPLIT).astype(np.int64), 2)
    w2 = np.minimum(window, 1)
    matched = np.isfinite(truth).all(1).astype(np.int64)
    sg = np.sqrt(np.maximum(cov_diag, 1e-30))
    with np.errstate(invalid="ignore"):
        pull = np.nan_to_num(np.maximum(np.abs(a[:, 4] - truth[:, 4]) / sg[:, 4], np.abs(a[:, 3] - truth[:, 3]) / sg[:, 3]), nan=0.0)
    changed = draws["u_state"] < cset["p_changed"].evaluate(allocation, matched, window, abs_eta, pt, inputs[:, IN["spixMaxWindowMult"]],
                                                           np.asarray(clean_margin, np.float64), pull)
    # ---- the structure (refit flag, crossings, accepted hits, layer mask, largest window multiplicity): today's, or the
    # donor's in the hit-choice part
    X = np.zeros((len(a), 25))
    X[:, 5:11] = inputs[:, 5:11]
    ch = np.flatnonzero(changed)
    if len(ch):
        # the donor: one of the k nearest pool tracks of the same class (particle or not x today's window), by the rank draw
        from scipy.spatial import cKDTree
        k, cls = int(M["k_donors"]), (matched * 3 + window)[ch]
        F = np.column_stack([np.log(pt), abs_eta, np.log(sg[:, 4]), np.log(sg[:, 3]), np.log1p(pull),
                             np.minimum(inputs[:, IN["spixMaxWindowMult"]], 8.0), inputs[:, IN["spixNAcceptedHits"]],
                             np.asarray(clean_margin, np.float64)])[ch]
        pool = {key: maps[f"{allocation}/pool_{key}"] for key in ("cls", "feat", "scale", "matched_ratio", "delta_ratio", "counts")}
        scale = pool["scale"].astype(np.float64)
        pick = np.zeros(len(ch), np.int64)
        for c in np.unique(cls):
            r = np.flatnonzero(cls == c)
            p = np.flatnonzero(pool["cls"] == c)
            if len(p) < 10 * k:
                p = np.arange(len(pool["cls"]))
            _, idx = cKDTree(pool["feat"][p].astype(np.float64) / scale).query(F[r] / scale, k=k)
            pick[r] = p[idx[np.arange(len(r)), np.minimum((draws["u_rank"][ch][r] * k).astype(int), k - 1)]]
        X[ch, 5], X[ch, 7:11] = pool["counts"][pick, 0], pool["counts"][pick, 1:]
    ran = X[:, IN["spixRefitPerformed"]] > 0
    # ---- covariance diagonal: today's x exp(map)
    lc_today = np.log(np.maximum(cov_diag, 1e-30))
    Zc = np.column_stack([ot_inputs, inputs, lc_today, a, _bits(inputs[:, IN["spixLayerHitMask"]]), X[:, 5:11],
                          _bits(X[:, IN["spixLayerHitMask"]])])
    new_cov = np.where(ran[:, None], cov_diag * np.exp(_block(m, "cov", 0, Zc, draws["n_cov"], knots)), ot_cov_diag)
    # ---- parameters: the precision part, then the donor's residual in the hit-choice part
    z = {"d0": ndtri(np.clip(draws["u_rank"], 1e-12, 1 - 1e-12))}
    for p in ("z0", "tanL", "rInv", "phi0"):
        r = cset["rho"].evaluate(allocation, p, abs_eta, w2)
        z[p] = r * z[PARTNER[p]] + np.sqrt(1 - r * r) * draws["n_" + p]
    new_a = a.copy()
    for i, p in enumerate(PARAMS):
        e = cset["eps"].evaluate(allocation, p, abs_eta, w2, ndtr(z[p]))
        lam = cset["lambda"].evaluate(allocation, p, w2, abs_eta, pt)
        unit = np.sqrt(np.abs(new_cov[:, i] - cov_diag[:, i]) + (0.05 * sg[:, i]) ** 2)       # the sigma the scenario adds
        new_a[:, i] += np.where(refit, lam * unit * e, 0.0)
    if len(ch):
        has = matched[ch] > 0
        new_a[ch] = np.where(has[:, None], np.where(has[:, None], truth[ch], 0.0) + sg[ch] * pool["matched_ratio"][pick],
                             a[ch] + sg[ch] * pool["delta_ratio"][pick])
    new_a[:, 1] = _wrap(new_a[:, 1])
    new_a = np.where(ran[:, None], new_a, ot_a)            # a track whose refit does not run carries the OT track
    # ---- the heads' mapped inputs
    d_ot, d_today = new_a - ot_a, new_a - a
    d_ot[:, 1], d_today[:, 1] = _wrap(d_ot[:, 1]), _wrap(d_today[:, 1])
    lc = np.log(np.maximum(new_cov, 1e-30))
    Zs = np.column_stack([Zc, d_ot, np.abs(d_today) / sg, lc, lc - lc_today, new_a, np.abs(d_ot) / np.sqrt(np.maximum(ot_cov_diag, 1e-30))])
    Y = _block(m, "score", 3 * N_COV, Zs, draws["n_score"], knots)
    for j, s in enumerate(scores):
        p_miss = 1.0 / (1.0 + np.exp(-_raw(m, 3 * (N_COV + len(scores)) + j, Zs)))
        X[:, IN[s]] = _unnatural(s, np.where((draws["u_miss"][:, j] < p_miss) | ~ran, np.nan, Y[:, j]))
    # ---- the heads' inputs that follow from the parameters
    ot_inv = ot_inputs[:, 0]
    X[:, 0] = np.where(ran, 1.0 / np.maximum(kpt / np.maximum(np.abs(new_a[:, 0]), 1e-12), 1e-3), ot_inv)
    X[:, 1] = new_a[:, 2]
    X[:, 2] = np.abs(new_a[:, 3] - ot_a[:, 3])
    X[:, 3] = np.abs(new_a[:, 2] - ot_a[:, 2])
    X[:, 4] = X[:, 0] - ot_inv
    X[:, IN["spixChi2ITAtOT"]] = MISSING                # always missing in refit v3
    X[:, IN["h_nangle"]] = np.clip(np.rint(X[:, IN["h_nangle"]]), 0, np.maximum(X[:, IN["spixNAcceptedHits"]], 0))
    X = X.astype(np.float32)
    margins = head_margins(heads, np.hstack([ot_inputs.astype(np.float32), X]), new_a[:, 4])
    return {"a": new_a, "cov_diag": new_cov, "inputs": X, "margins": margins, "changed": changed}
