"""The B.1 estimator (definition sheet v1, item `estimator.core`): the one width estimator
of the study (plan App. B.1, C.2 P11).

Algorithm (the census pull table's, site_v2/scripts/emu_pulls_table.py `core`, made
unit-free -- see "Units" below):
  1. keep the finite values the CALLER marks valid; fewer than 200 -> no estimate;
  2. centre seed = the centre of the most populated of 400 bins spanning the 0.5-99.5 %
     quantiles; width seed = the 68.27 % quantile of |x - centre|;
  3. iterate {window = |x - c| < 3 w; c = mean(window); w = std(window) (ddof 0)} until both
     move by less than 1e-5 of the current width, at most 50 times;
  4. core = RMS ABOUT ZERO of the final window (a bias counts against the core); also the
     window's mean and std, the fraction outside it and the full RMS about zero of all
     valid values. MAD is a shape flag only, never the width.

Units. The estimator must give the same physical width whatever unit the values are in:
core(k x) = k core(x). It therefore never cuts on magnitude. Sentinels are removed by the
caller with an explicit validity mask (`valid`), defined per column in the definition sheet
(item `estimator.sentinels`). The census table's original "drop |x| >= 1e3" and its absolute
1e-5 convergence tolerance were pull-specific: on residuals in micrometres the first cut
every track beyond 1 mm (an OT z0 core of 554 um instead of 4786 um, found by the P0.2 audit
on 2026-10-09), and in centimetres the second stops the iteration early. `core_pull_table`
keeps the original for reproducing old pull tables and as the known-bad of the unit test.
"""
from __future__ import annotations

import numpy as np

CORE_MIN_N = 200
CORE_BINS = 400
CORE_NSIG = 3.0
CORE_SEED_Q = 68.27
CORE_MAXIT = 50
CORE_REL_TOL = 1e-5


def core(x, valid=None, nit: int = CORE_MAXIT) -> dict | None:
    """B.1 core of `x` over the entries `valid` marks (default: every finite value)."""
    x = np.asarray(x, dtype=np.float64)
    keep = np.isfinite(x) if valid is None else (np.asarray(valid, bool) & np.isfinite(x))
    x = x[keep]
    if len(x) < CORE_MIN_N:
        return None
    lo, hi = np.percentile(x, [0.5, 99.5])
    if not hi > lo:
        return {"n": int(len(x)), "mean": float(lo), "sigma": 0.0, "rms0": float(abs(lo)), "out3": 0.0,
                "full_rms0": float(np.sqrt(np.mean(x ** 2)))}
    h, e = np.histogram(x, bins=CORE_BINS, range=(lo, hi))
    c = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    w = np.percentile(np.abs(x - c), CORE_SEED_Q)
    for _ in range(nit):
        s = x[np.abs(x - c) < CORE_NSIG * w]
        if len(s) == 0:
            break
        c2, w2 = s.mean(), s.std()
        if abs(c2 - c) < CORE_REL_TOL * w and abs(w2 - w) < CORE_REL_TOL * w:
            break
        c, w = c2, w2
    s = np.abs(x - c) < CORE_NSIG * w
    return {"n": int(len(x)), "mean": float(c), "sigma": float(w),
            "rms0": float(np.sqrt(np.mean(x[s] ** 2))) if s.any() else float("nan"),
            "out3": float(1 - s.mean()), "full_rms0": float(np.sqrt(np.mean(x ** 2)))}


def core_pull_table(x, nit: int = CORE_MAXIT) -> dict | None:
    """The census pull table's original `core` (PULLS ONLY: it drops |x| >= 1e3 and uses an
    absolute 1e-5 tolerance). Kept to reproduce old pull tables; never for residuals."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.abs(x) < 1e3]
    if len(x) < CORE_MIN_N:
        return None
    lo, hi = np.percentile(x, [0.5, 99.5])
    h, e = np.histogram(x, bins=CORE_BINS, range=(lo, hi))
    c = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    w = np.percentile(np.abs(x - c), CORE_SEED_Q)
    for _ in range(nit):
        s = x[np.abs(x - c) < 3 * w]
        c2, w2 = s.mean(), s.std()
        if abs(c2 - c) < 1e-5 and abs(w2 - w) < 1e-5:
            break
        c, w = c2, w2
    s = np.abs(x - c) < 3 * w
    xf = x[np.abs(x) < 50]
    return {"n": int(len(x)), "mean": float(c), "sigma": float(w),
            "rms0": float(np.sqrt(np.mean(x[s] ** 2))), "out3": float(1 - s.mean()),
            "full_rms0": float(np.sqrt(np.mean(xf ** 2))), "beyond50": float(1 - len(xf) / len(x))}


def scale_invariance(x, valid=None, scales=(1e-4, 1.0, 1e4), estimator=core) -> dict:
    """Relative spread of estimator(k x).rms0 / k over the scales (0 for a unit-free
    estimator). The unit test of the estimator item, and its known-bad (core_pull_table)."""
    vals = []
    for k in scales:
        r = estimator(np.asarray(x, np.float64) * k) if valid is None or estimator is core_pull_table \
            else estimator(np.asarray(x, np.float64) * k, valid)
        vals.append(np.nan if r is None else r["rms0"] / k)
    vals = np.asarray(vals)
    return {"scales": list(scales), "rms0_over_k": vals.tolist(),
            "rel_spread": float((np.nanmax(vals) - np.nanmin(vals)) / np.nanmax(np.abs(vals)))}
