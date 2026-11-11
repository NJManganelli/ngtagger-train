"""Reference helper: apply a shipped SmartPixels bit allocation to today's extended-collection tracks through the
correctionlib file of the scenario (spixbits_<AAAA|AAAI|IAAA>.json.gz). Needs numpy, scipy, correctionlib >= 2.6.

    import correctionlib, spixbits_apply as SA
    cset = correctionlib.CorrectionSet.from_file("spixbits_AAAA.json.gz")
    out = SA.apply(cset, "alloc_4444", a=a, cov_diag=cov_diag, refit=refit, margins=margins, columns=columns,
                   truth=truth, kpt=SA.meta(cset)["kpt_GeV_cm"], ids=(run, lumi, event, track))

Inputs, one entry per track of L1TSmartPixelsExtTrack as produced today (flattened over events):
  a          (n, 5) rInv [1/cm], phi0 [rad], tanL, z0 [cm], d0 [cm]
  cov_diag   (n, 5) the diagonal of the track covariance, same order and units
  refit      (n,)   spixRefitPerformed
  margins    (n, 6) the deployed heads: clean as a MARGIN (logit of head_clean: see to_margin), then head_d0, head_z0,
                    head_tanl, head_rinv, head_phi (log sigma_hat) as stored
  columns    {name: (n,)} for spixRefitPerformed, spixNCrossings, spixNAcceptedHits, spixLayerHitMask, spixMaxWindowMult,
             spixChi2IncXTot, spixChi2IncYTot, spixChi2IncAlphaTot, spixChi2ITAtRefit, spixShiftChi2, spixLogDetRatio
  truth      (n, 5) the OT track's particle in the same order (otTp_rInv, otTp_phi0, otTp_tanL, otTp_z0, otTp_d0), NaN
             where the track has none (otTpIdx < 0)
  ids        (run, lumi, event, track): track = the index along axis 1 of the event's track collection
Output: {"a", "cov_diag", "margins", "changed", and the columns}: the emulated allocation. The draws depend on the
track's identity only, never on the allocation, so allocations stay comparable track by track. pT and eta follow from
the new rInv and tanL (pT = kpt / |rInv|); full_covariance() rebuilds the matrix with today's correlations. Hardware
words (hw*) and spixCompactWord are NOT recomputed: do not read them beside emulated values.
"""
import json

import numpy as np
from scipy.special import ndtr, ndtri

PARAMS = ("rInv", "phi0", "tanL", "z0", "d0")
HEADS = ("clean", "d0", "z0", "tanl", "rinv", "phi")
COUNTS = ("spixRefitPerformed", "spixNCrossings", "spixNAcceptedHits", "spixLayerHitMask", "spixMaxWindowMult")
CHI2 = ("spixChi2IncXTot", "spixChi2IncYTot", "spixChi2IncAlphaTot", "spixChi2ITAtRefit", "spixShiftChi2", "spixLogDetRatio")
SCORES = tuple(f"head_{h}" for h in HEADS) + CHI2
PARTNER = {"z0": "d0", "tanL": "z0", "rInv": "d0", "phi0": "d0"}
D0_SPLIT, MISSING = 0.02, -999.0


def meta(cset):
    return json.loads(cset.description)


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
    y = x.copy() if (name.startswith("head_") or name == "spixLogDetRatio") else np.log(np.maximum(x, 0.0) + 1e-3)
    return np.where(x <= -998, np.nan, y)


def _unnatural(name, y):
    x = np.asarray(y, np.float64).copy() if (name.startswith("head_") or name == "spixLogDetRatio") else np.maximum(np.exp(y) - 1e-3, 0.0)
    return np.where(np.isfinite(y), x, MISSING)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def apply(cset, allocation, a, cov_diag, refit, margins, columns, truth, kpt, ids=None, draws=None):
    a = np.asarray(a, np.float64)
    cov_diag = np.asarray(cov_diag, np.float64)
    margins = np.asarray(margins, np.float64)
    refit = np.asarray(refit).astype(bool)
    n = len(a)
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
    mult = np.asarray(columns["spixMaxWindowMult"], np.float64)
    n_hits = np.asarray(columns["spixNAcceptedHits"], np.float64)
    sg = np.sqrt(np.maximum(cov_diag, 1e-30))
    with np.errstate(invalid="ignore"):
        pull = np.nan_to_num(np.maximum(np.abs(a[:, 4] - truth[:, 4]) / sg[:, 4], np.abs(a[:, 3] - truth[:, 3]) / sg[:, 3]), nan=0.0)
    changed = draws["u_state"] < cset["p_changed"].evaluate(allocation, matched, window, abs_eta, pt, mult, margins[:, 0], pull)
    # ---- same clusters
    z = {"d0": ndtri(np.clip(draws["u_rank"], 1e-12, 1 - 1e-12))}
    for p in ("z0", "tanL", "rInv", "phi0"):
        r = cset["rho"].evaluate(allocation, p, abs_eta, w2)
        z[p] = r * z[PARTNER[p]] + np.sqrt(1 - r * r) * draws["n_" + p]
    new_a, new_cov = a.copy(), cov_diag.copy()
    for i, p in enumerate(PARAMS):
        e = cset["eps"].evaluate(allocation, p, abs_eta, w2, ndtr(z[p]))
        lam = cset["lambda"].evaluate(allocation, p, w2, abs_eta, pt)
        lr = cset["cov_scale"].evaluate(allocation, p, "a", w2, abs_eta, pt) \
            + cset["cov_scale"].evaluate(allocation, p, "b", w2, abs_eta, pt) * np.log(sg[:, i]) \
            + cset["cov_scale"].evaluate(allocation, p, "sigma", w2, abs_eta, pt) * cset["cov_eta"].evaluate(allocation, p, draws["u_cov"])
        cov_i = cov_diag[:, i] * np.exp(np.clip(lr, np.log(0.2), np.log(1.0e4)))
        unit = np.sqrt(np.abs(cov_i - cov_diag[:, i]) + (0.05 * sg[:, i]) ** 2)       # the sigma the allocation adds
        new_a[:, i] += np.where(refit, lam * unit * e, 0.0)
        new_cov[:, i] = np.where(refit, cov_i, new_cov[:, i])
    w_arm = (np.abs(new_a[:, 4]) > D0_SPLIT).astype(np.int64)
    tail = np.abs(2 * draws["u_rank"] - 1)
    today = {f"head_{h}": _natural("head_", margins[:, i]) for i, h in enumerate(HEADS)}
    today.update({c: _natural(c, columns[c]) for c in CHI2})
    sc = {}
    for s in SCORES:
        co = [cset["score_map"].evaluate(allocation, s, c, w_arm, n_hits, abs_eta) for c in ("a", "b", "sigma")]
        eta = cset["score_eta"].evaluate(allocation, s, tail, draws["u_score"])
        sc[s] = np.where(refit, co[0] + co[1] * today[s] + co[2] * eta, today[s])
    counts = {c: np.asarray(columns[c], np.float64).copy() for c in COUNTS}
    # ---- changed: the donor's values
    ch = np.flatnonzero(changed)
    if len(ch):
        don = lambda q: cset["donor"].evaluate(allocation, q, matched[ch], window[ch], abs_eta[ch], pull[ch], draws["u_rank"][ch])
        for i, p in enumerate(PARAMS):
            has = matched[ch] > 0
            new_a[ch, i] = np.where(has, np.where(has, truth[ch, i], 0.0) + sg[ch, i] * don(f"matched_ratio_{p}"),
                                    a[ch, i] + sg[ch, i] * don(f"delta_ratio_{p}"))
            new_cov[ch, i] = cov_diag[ch, i] * don(f"cov_ratio_{p}")
        for i, h in enumerate(HEADS):
            sc[f"head_{h}"][ch] = margins[ch, i] + don(f"d_margin_{h}")
        for c in COUNTS:
            counts[c][ch] = don(c)
        for c in CHI2:
            sc[c][ch] = _natural(c, don(c))
    new_a[:, 1] = _wrap(new_a[:, 1])
    out = {"a": new_a, "cov_diag": new_cov, "margins": np.stack([sc[f"head_{h}"] for h in HEADS], 1), "changed": changed}
    out.update(counts)
    out.update({c: _unnatural(c, sc[c]) for c in CHI2})
    return out
