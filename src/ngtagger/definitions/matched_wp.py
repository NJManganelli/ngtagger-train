"""Matched working points with tie splitting (definition sheet v1, item `fom.matched_wp`;
plan App. C.2 P6).

Every arm is compared at EXACTLY the same background fraction. On discrete scores (the
stock tagger's 8-bit outputs) many jets share the threshold value; they are accepted with
the fraction p that makes the weighted background fraction equal the target (a randomised
threshold, i.e. ROC interpolation between the two neighbouring discrete points). Signal at
the threshold value is accepted with the same p. Thresholds are recomputed in every
bootstrap replica (pass that replica's weights). The realised background fraction is
reported beside every efficiency.
"""
from __future__ import annotations

import numpy as np


def tie_split_threshold(score_b, target: float, w_b=None) -> tuple[float, float]:
    """(t, p): accept score > t fully and score == t with fraction p, so that the weighted
    background fraction accepted equals `target` exactly (0 <= target <= 1)."""
    s = np.asarray(score_b, dtype=np.float64)
    w = np.ones_like(s) if w_b is None else np.asarray(w_b, dtype=np.float64)
    if s.size == 0 or w.sum() <= 0:
        return float("nan"), float("nan")
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"target {target} outside [0, 1]")
    u, inv = np.unique(s, return_inverse=True)          # ascending distinct values
    wu = np.bincount(inv, weights=w, minlength=len(u))
    total = wu.sum()
    want = target * total
    above = total - np.cumsum(wu)                         # weight strictly above u[i]
    i = int(np.searchsorted(-(above + wu), -want, side="right")) - 1   # last i with above+at >= want
    i = min(max(i, 0), len(u) - 1)
    at = wu[i]
    p = 0.0 if at <= 0 else (want - above[i]) / at
    return float(u[i]), float(min(max(p, 0.0), 1.0))


def accepted_fraction(score, t: float, p: float, w=None) -> float:
    s = np.asarray(score, dtype=np.float64)
    w = np.ones_like(s) if w is None else np.asarray(w, dtype=np.float64)
    tot = w.sum()
    if tot <= 0:
        return float("nan")
    return float(((w * (s > t)).sum() + p * (w * (s == t)).sum()) / tot)


def matched_point(score_s, score_b, target: float, w_s=None, w_b=None) -> dict:
    """Signal efficiency at background fraction `target`, ties split; with the realised
    background fraction (equal to the target up to rounding) and the threshold."""
    t, p = tie_split_threshold(score_b, target, w_b)
    return {"target": target, "threshold": t, "tie_fraction": p,
            "efficiency": accepted_fraction(score_s, t, p, w_s),
            "realised_background": accepted_fraction(score_b, t, p, w_b)}


def matched_points_replicas(score_s, score_b, target: float, W_s, W_b) -> dict:
    """matched_point in every bootstrap replica: W_s, W_b are (n_rep, n) weight matrices
    (event-level multinomial counts mapped to jets). Thresholds re-derived per replica."""
    eff, real = np.empty(len(W_s)), np.empty(len(W_s))
    for r in range(len(W_s)):
        m = matched_point(score_s, score_b, target, W_s[r], W_b[r])
        eff[r], real[r] = m["efficiency"], m["realised_background"]
    return {"efficiency": eff, "realised_background": real}
