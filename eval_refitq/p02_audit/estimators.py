"""P0.2 estimators: the B.1 core and event-level bootstrap helpers.

B.1 core (plan App. B.1 / C.2 P11): the RMS about zero of a mode-seeded iterative 3-sigma clip. This is the
algorithm of the census pull table (site_v2/scripts/emu_pulls_table.py ``core``): seed the centre at the mode of
a 400-bin histogram over the 0.5-99.5% quantile range and the width at the 68.27% quantile of |x - mode|;
iterate mean / std over |x - c| < 3 s; return the RMS about zero of the kept values and the fraction outside.
No shared module holds it yet (WP-A1 creates the definition sheet), so it is written here once, with
frequency weights so that every bootstrap replica re-fits it (C.2 P7), on pre-sorted values (O(n) per replica).

Bootstrap: event-level, multinomial counts over the events (``replica_weights``); one weight matrix is shared
by every arm and scenario of a run, so differences are paired.
"""
from __future__ import annotations

import numpy as np

CORE_MIN_N = 200        # the census table's minimum
CORE_BINS = 400
CORE_NSIG = 3.0
CORE_SEED_Q = 0.6827
CORE_MAXIT = 50


def replica_weights(n_events, n_rep, seed):
    """(n_rep, n_events) float matrix of multinomial event counts (sum n_events per row)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_events, np.full(n_events, 1.0 / n_events), size=n_rep).astype(np.float64)


class SortedSample:
    """Values sorted once, with their event indices, for repeated weighted core fits."""

    def __init__(self, x, ev):
        x = np.asarray(x, np.float64)
        ok = np.isfinite(x)
        o = np.argsort(x[ok], kind="stable")
        self.xs = x[ok][o]
        self.es = np.asarray(ev)[ok][o]
        self.n = len(self.xs)

    def core(self, w=None):
        return core_sorted(self.xs, np.ones(self.n) if w is None else w)

    def core_reps(self, W):
        """core for every replica row of the event-weight matrix W -> (R, 4) array."""
        out = np.full((W.shape[0], 4), np.nan)
        if self.n == 0:
            return out
        for r in range(W.shape[0]):
            out[r] = core_sorted(self.xs, W[r, self.es])
        return out


def _seg(c, i0, i1):
    """sum over sorted positions i0 .. i1-1 from a cumulative sum c."""
    if i1 <= i0:
        return 0.0
    return c[i1 - 1] - (c[i0 - 1] if i0 > 0 else 0.0)


def core_sorted(xs, w):
    """B.1 core of sorted values xs with frequency weights w.

    Returns (core_rms_about_zero, centre, sigma, outside_fraction); NaNs when the total weight (the
    resampled track count; the plain count when unweighted) is below CORE_MIN_N."""
    nan4 = (np.nan, np.nan, np.nan, np.nan)
    if len(xs) == 0 or w.sum() < CORE_MIN_N:
        return nan4
    cw = np.cumsum(w)
    W = cw[-1]
    cwx = np.cumsum(w * xs)
    cwx2 = np.cumsum(w * xs * xs)

    def wq(q):
        i = int(np.searchsorted(cw, q * W, side="left"))
        return xs[min(i, len(xs) - 1)]

    def window(a, b):
        # strictly inside (a, b), as the census' |x - c| < 3 s
        i0 = int(np.searchsorted(xs, a, side="right"))
        i1 = int(np.searchsorted(xs, b, side="left"))
        return _seg(cw, i0, i1), _seg(cwx, i0, i1), _seg(cwx2, i0, i1)

    lo, hi = wq(0.005), wq(0.995)
    if not hi > lo:
        hi = lo + max(abs(lo), 1.0) * 1e-6
    idx = np.floor((xs - lo) / (hi - lo) * CORE_BINS).astype(np.int64)
    idx[xs == hi] = CORE_BINS - 1          # np.histogram's closed last bin
    inr = (idx >= 0) & (idx < CORE_BINS)
    h = np.bincount(idx[inr], weights=w[inr], minlength=CORE_BINS)
    j = int(np.argmax(h))
    c = lo + (j + 0.5) * (hi - lo) / CORE_BINS

    # seed width: weighted 68.27% quantile of |x - c| (bisection on the window half-width)
    tlo, thi = 0.0, max(abs(xs[0] - c), abs(xs[-1] - c)) * 1.000001 + 1e-12
    for _ in range(60):
        t = 0.5 * (tlo + thi)
        if window(c - t, c + t)[0] >= CORE_SEED_Q * W:
            thi = t
        else:
            tlo = t
    s = thi
    for _ in range(CORE_MAXIT):
        sw, sx, sx2 = window(c - CORE_NSIG * s, c + CORE_NSIG * s)
        if sw <= 0:
            break
        c2 = sx / sw
        s2 = float(np.sqrt(max(sx2 / sw - c2 * c2, 0.0)))
        done = abs(c2 - c) <= 1e-7 * max(s, 1e-30) and abs(s2 - s) <= 1e-7 * max(s, 1e-30)
        c, s = c2, s2
        if done or s <= 0:
            break
    sw, sx, sx2 = window(c - CORE_NSIG * s, c + CORE_NSIG * s)
    if sw <= 0:
        return nan4
    return float(np.sqrt(sx2 / sw)), float(c), float(s), float(1.0 - sw / W)


def per_event(ev, vals, n_events):
    """per-event sums of vals (bool or float) -> float array (n_events,)."""
    return np.bincount(ev, weights=np.asarray(vals, np.float64), minlength=n_events)


def ratio(num_e, den_e, W):
    """sum(num) / sum(den): nominal and per replica."""
    d = den_e.sum()
    v = num_e.sum() / d if d > 0 else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = (W @ num_e) / (W @ den_e)
    return v, reps


def summ(v, reps):
    """{'v', 'se', 'var'} from a nominal value and its replicas (NaN replicas dropped)."""
    reps = np.asarray(reps, np.float64)
    good = reps[np.isfinite(reps)]
    se = float(np.std(good, ddof=1)) if len(good) > 2 else float("nan")
    out = {"v": None if v is None or not np.isfinite(v) else float(v),
           "se": se if np.isfinite(se) else None}
    out["var"] = None if out["se"] is None else out["se"] ** 2
    if len(good) < len(reps):
        out["n_rep_valid"] = int(len(good))
    return out
