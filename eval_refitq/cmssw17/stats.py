"""Statistics of the CMSSW_17 comparison (plan 3.4.5): event-level Poisson bootstrap, weighted summaries recomputed in
every replica, permutation references for the shape distances, verdicts, Holm.

Objects (tracks, hits, clusters, jets) are correlated within an event, so every resampling is by whole events:
- the bootstrap gives each event a Poisson(1) weight per replica, from a hash of (event uid, replica), so every
  quantity of a side sees the same replica weights;
- the permutation reference reassigns whole events between the two sides.
"""
from __future__ import annotations

import math

import numpy as np

_POIS_CDF = np.cumsum([math.exp(-1) / math.factorial(k) for k in range(13)])
GOLD = np.uint64(0x9E3779B97F4A7C15)


def splitmix64(x):
    x = np.asarray(x, np.uint64)
    with np.errstate(over="ignore"):
        x = x + GOLD
        z = x
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def event_uid(side_tag: int, file_no, run, lumi, event):
    """A 64-bit identity per event (side, file, run, lumi, event)."""
    with np.errstate(over="ignore"):
        h = splitmix64(np.uint64(side_tag) * np.uint64(1_000_003) + np.asarray(file_no, np.uint64))
        h = splitmix64(h ^ np.asarray(run, np.uint64))
        h = splitmix64(h ^ np.asarray(lumi, np.uint64))
        return splitmix64(h ^ np.asarray(event, np.uint64))


def poisson_weights(uid, r: int):
    """Poisson(1) bootstrap weight of each event (uid) in replica r; replica -1 is the nominal sample (all 1)."""
    if r < 0:
        return np.ones(len(uid))
    with np.errstate(over="ignore"):
        h = splitmix64(np.asarray(uid, np.uint64) ^ (np.uint64(r + 1) * np.uint64(0xD1B54A32D192ED03)))
    u = (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return np.searchsorted(_POIS_CDF, u, side="right").astype(np.float64)


# ---------------------------------------------------------------- weighted summaries --------------------------------
def wquantile_sorted(xs, ws, q):
    """Weighted quantile of sorted values xs with weights ws."""
    cw = np.cumsum(ws)
    if cw[-1] <= 0:
        return np.nan
    return float(xs[min(np.searchsorted(cw, q * cw[-1], side="left"), len(xs) - 1)])


class Resid:
    """Resolution summaries of residuals about zero (plan 3.4.5): the core RMS (iterative 3-sigma clip about zero,
    started from 1.4826 x the median |x|, at most 10 iterations), the share beyond 3 core RMS, the full RMS and the
    mean within the core (the full mean is dominated by the tails). Built once per cell: |x| sorted, so each replica
    costs two cumulative sums."""

    def __init__(self, x):
        x = np.asarray(x, np.float64)
        x = x[np.isfinite(x)]
        self.o = np.argsort(np.abs(x), kind="stable")
        self.ax = np.abs(x)[self.o]
        self.x2 = self.ax ** 2
        self.x = x[self.o]
        self.n = len(x)

    def summaries(self, w):
        """w: weights in the ORIGINAL order of the finite x (the caller keeps them aligned)."""
        if self.n == 0:
            return dict(core=np.nan, outside=np.nan, full=np.nan, mean=np.nan, wsum=0.0)
        ws = w[self.o]
        cw = np.cumsum(ws)
        tot = cw[-1]
        if tot <= 0:
            return dict(core=np.nan, outside=np.nan, full=np.nan, mean=np.nan, wsum=0.0)
        cx2 = np.cumsum(ws * self.x2)
        med = self.ax[min(np.searchsorted(cw, 0.5 * tot), self.n - 1)]
        s = 1.4826 * med
        for _ in range(10):
            if s <= 0:
                break
            k = np.searchsorted(self.ax, 3 * s, side="left")
            if k == 0:
                break
            snew = math.sqrt(cx2[k - 1] / cw[k - 1]) if cw[k - 1] > 0 else s
            if abs(snew - s) <= 1e-7 * s:
                s = snew
                break
            s = snew
        k = np.searchsorted(self.ax, 3 * s, side="right")
        outside = 1.0 - (cw[k - 1] / tot if k > 0 else 0.0)
        # the mean within the core (|x| < 3 core RMS): the bias of the core, robust against the tails
        mcore = float(np.sum(ws[:k] * self.x[:k]) / cw[k - 1]) if k > 0 and cw[k - 1] > 0 else np.nan
        return dict(core=s, outside=outside, full=math.sqrt(cx2[-1] / tot), mean=mcore, wsum=float(tot))


def wmean(x, w):
    sw = w.sum()
    return float((w * x).sum() / sw) if sw > 0 else np.nan


def wfrac(cond, w):
    sw = w.sum()
    return float((w * cond).sum() / sw) if sw > 0 else np.nan


# ---------------------------------------------------------------- shapes ---------------------------------------------
def ks_distance(x, side):
    """Two-sample KS distance between x[side] and x[~side] (unweighted)."""
    o = np.argsort(x, kind="stable")
    xs, sd = x[o], side[o]
    na, nb = sd.sum(), (~sd).sum()
    if na == 0 or nb == 0:
        return np.nan
    ca = np.cumsum(sd) / na
    cb = np.cumsum(~sd) / nb
    last = np.r_[xs[1:] != xs[:-1], True]           # evaluate at the end of tie groups
    return float(np.max(np.abs(ca - cb)[last]))


class KSPerm:
    """KS distance and its event-permutation reference for one cell. Objects carry their event index into a pooled
    event list; a permutation reassigns events between the sides keeping the event counts."""

    def __init__(self, x, ev, side_of_event):
        x = np.asarray(x, np.float64)
        ok = np.isfinite(x)
        x, ev = x[ok], np.asarray(ev)[ok]
        self.o = np.argsort(x, kind="stable")
        self.xs = x[self.o]
        self.ev = ev[self.o]
        self.last = np.r_[self.xs[1:] != self.xs[:-1], True] if len(x) else np.zeros(0, bool)
        self.side_ev = np.asarray(side_of_event, bool)

    def distance(self, side_ev=None):
        sd = (self.side_ev if side_ev is None else side_ev)[self.ev]
        na, nb = sd.sum(), (~sd).sum()
        if na == 0 or nb == 0:
            return np.nan
        d = np.abs(np.cumsum(sd) / na - np.cumsum(~sd) / nb)
        return float(np.max(d[self.last]))

    def permutations(self, n_perm, seed):
        rng = np.random.default_rng(seed)
        out = np.empty(n_perm)
        for p in range(n_perm):
            out[p] = self.distance(rng.permutation(self.side_ev))
        return out


def tv_distance(cat, w_a, w_b, cats):
    pa = np.array([w_a[cat == c].sum() for c in cats])
    pb = np.array([w_b[cat == c].sum() for c in cats])
    if pa.sum() <= 0 or pb.sum() <= 0:
        return np.nan
    return 0.5 * float(np.abs(pa / pa.sum() - pb / pb.sum()).sum())


# ---------------------------------------------------------------- verdicts --------------------------------------------
def phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def p_exceed(effect, se, bound):
    """One-sided p of |effect| > bound (Gaussian, bootstrap error)."""
    if not np.isfinite(effect) or not np.isfinite(se) or se <= 0:
        return 1.0 if not (np.isfinite(effect) and abs(effect) > bound) else 0.0
    return 1.0 - phi((abs(effect) - bound) / se)


def agrees(effect, se, tol):
    """The 90 % interval of the effect lies inside +-tol (an equivalence test at 5 % per side)."""
    if not np.isfinite(effect) or not np.isfinite(se):
        return False
    return abs(effect) + 1.645 * se <= tol


def holm(pvals, alpha):
    """Holm step-down: a boolean array of rejections."""
    p = np.asarray(pvals, np.float64)
    m = len(p)
    rej = np.zeros(m, bool)
    if m == 0:
        return rej
    order = np.argsort(p)
    for i, j in enumerate(order):
        if p[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej
