"""E5 (PLAN §4, §10 PD4 (a), (b), (f); PD10 (e)): the mirror PM against refit v2, step by step.

  compare      PM's per-step record against CMSSW's recorded crossings (mapped to visit steps with
               ORP._row_steps): who accepted which cluster; each track's FIRST disagreement (later ones
               cascade from it)
  pick         the producer's box selection on GIVEN projections (CMSSW's recorded ones, perturbed or
               not): box_candidates + the producer metric over the cluster sigma (capped angles)
  near_tie     a first disagreement is a near-tie when one of the four (+-du, +-dv) corner perturbations of
               CMSSW's recorded projection, at the surface's p99.9 offsets, reproduces PM's pick
  model        file 4: CMSSW's recorded projections perturbed by draws from the tail offsets (PM's input
               precision, per surface and earlier-max-pull bin); each track's first flip
  flip_effect  CMSSW's own selections refitted with PM's arithmetic (refit_forced), with and without the
               first flipped pick substituted: the flip's effect on harm and on the wrong-hit counts
"""
from __future__ import annotations

import numpy as np

import arms as AR
import fom as FM
from common import OP, ORP

PULL_BINS = (3.0, 5.0, 10.0, 30.0)


def compare(arm_rec, cm, tracks):
    """Per (track, step): disagree when exactly one side accepted, or both accepted different clusters.
    Returns (disagree (ntrk, ns), first_step per track (-1 none))."""
    a_acc, a_ci = arm_rec["acc"], arm_rec["ci"]
    dis = (a_acc != cm["acc"]) | (a_acc & cm["acc"] & (a_ci != cm["ci"]))
    dis &= tracks[:, None]
    first = np.where(dis.any(1), np.argmax(dis, 1), -1)
    return dis, first


def pick(ctx, cfg, ev, det, surf, u, v, ca, cb):
    """The producer's choice (cluster row, -1 none) on the given projections (one per crossing)."""
    K = ctx["K"]
    xi, ci, order, _ = AR.box_candidates(ctx, cfg, ev, det, surf, u, v)
    use_a, use_b = cfg["useAngles"] in ORP.USE_ANGLES, cfg["useAngles"] == "alphaBeta"
    val = (((K["localX"][ci] - u[xi]) / np.maximum(K["sigX"][ci], 1e-12)) ** 2
           + ((K["localY"][ci] - v[xi]) / np.maximum(K["sigY"][ci], 1e-12)) ** 2)
    if use_a:
        val = val + np.where(K["hasAlpha"][ci] > 0, ((K["localCotAlpha"][ci] - ca[xi]) / np.maximum(K["sigAlpha"][ci], 1e-12)) ** 2, 0.0)
    if use_b:
        val = val + np.where(K["hasBeta"][ci] > 0, ((K["localCotBeta"][ci] - cb[xi]) / np.maximum(K["sigBeta"][ci], 1e-12)) ** 2, 0.0)
    best, _ = AR._ranked({"x": xi, "val": val, "order": order}, len(u))
    return np.where(best >= 0, ci[np.maximum(best, 0)], -1)


def pull_bin(emp):
    return np.digitize(emp, PULL_BINS)


class OffsetModel:
    """The tail offsets (tail_<scen>.npz, PM's input precision) per (surface, earlier-max-pull bin)."""

    def __init__(self, path):
        z = np.load(path)
        self.pool = {}
        ok = np.isfinite(z["du"]) & np.isfinite(z["dv"])
        for S in np.unique(z["surface"]):
            for b in range(len(PULL_BINS) + 1):
                m = ok & (z["surface"] == S) & (z["pull_bin"] == b)
                self.pool[(int(S), b)] = np.stack([z["du"][m], z["dv"][m]], 1)
        self.p999 = {}
        for (S, b), x in self.pool.items():
            if len(x):
                self.p999[(S, b)] = np.percentile(np.abs(x), 99.9, axis=0)

    def draw(self, surf, b, rng):
        out = np.zeros((len(surf), 2))
        for key in set(zip(surf.tolist(), b.tolist())):
            m = (surf == key[0]) & (b == key[1])
            pool = self.pool.get(key)
            if pool is None or not len(pool):
                pool = self.pool.get((key[0], 0), np.zeros((1, 2)))
            out[m] = pool[rng.integers(0, len(pool), m.sum())]
        return out

    def corners(self, surf, b):
        out = np.zeros((len(surf), 2))
        for i, key in enumerate(zip(surf.tolist(), b.tolist())):
            out[i] = self.p999.get(key, self.p999.get((key[0], 0), np.zeros(2)))
        return out


def earlier_max_pull_rows(hit):
    """Per CMSSW row: the largest |position pull| of the track's earlier accepted updates (hit rows are in
    visit order per track); computed from the record's own pulls."""
    p = np.where(hit["acc"], np.nan_to_num(np.nanmax(np.abs(np.stack([hit["pullX"], hit["pullY"]], 1)), 1)), 0.0)
    out = np.zeros(len(p))
    t = hit["t"]
    cur, prev_t = 0.0, None
    for i in range(len(p)):
        if t[i] != prev_t:
            cur, prev_t = 0.0, t[i]
        out[i] = cur
        cur = max(cur, p[i])
    return out


def flip_effect(arm, ctx, geo, cfg, tools, path, seq, base_mi, base_ci, tracks, step, new_ci, new_mi, kpt):
    """Refit `tracks` on CMSSW's selections (base_mi / base_ci per step) and on the same with the pick at
    `step` replaced by new_ci (-1: no hit) on module new_mi. Returns (d0, z0) of both refits."""
    sel_ci = base_ci[tracks].copy()
    sel_mi = base_mi[tracks].copy()
    base = AR.refit_forced(arm, ctx, geo, cfg, tools, tracks, seq[tracks], sel_mi, sel_ci, path, kpt)
    k = np.arange(len(tracks))
    sel_ci[k, step] = new_ci
    sel_mi[k, step] = np.where(new_ci >= 0, new_mi, sel_mi[k, step])
    flip = AR.refit_forced(arm, ctx, geo, cfg, tools, tracks, seq[tracks], sel_mi, sel_ci, path, kpt)
    return (base.a[:, 4], base.a[:, 3], base.nUpd), (flip.a[:, 4], flip.a[:, 3], flip.nUpd)


def counts_of_flip(ctx, truth, tracks, base, flip, ot_d0, ot_z0, ev, nev, step_surface, old_ci, new_ci):
    """Per-event shift counts of a set of first flips (first-flip basis): H num (prompt population,
    either-harm of the flip refit minus the base refit) per bin, P num / den per surface (wrong / accepted
    at the flipped crossing, new minus old)."""
    out = {}
    hb = FM.harm(base[0], base[1], ot_d0[tracks], ot_z0[tracks], {k: v[tracks] for k, v in truth.items()})
    hf = FM.harm(flip[0], flip[1], ot_d0[tracks], ot_z0[tracks], {k: v[tracks] for k, v in truth.items()})
    pr = truth["pop"][tracks] & (truth["origin"][tracks] == 0)
    aeta, pt = np.abs(truth["eta"][tracks]), truth["pt"][tracks]
    e = ev[tracks]
    for (elo, ehi), (plo, phi) in FM.H_BINS:
        m = pr & (aeta >= elo) & (aeta < ehi) & (pt >= plo) & (pt < phi)
        out[f"H/{FM.h_key((elo, ehi), (plo, phi))}/num"] = np.bincount(
            e[m], weights=(hf["either"][m].astype(float) - hb["either"][m]), minlength=nev)
    tp = ctx["trk_tp"][tracks]
    ktp = ctx["K"]["tp_global"]
    wrong = lambda c, tpm: (c >= 0) & ((ktp[np.maximum(c, 0)] < 0) | (ktp[np.maximum(c, 0)] != tpm))
    popm = truth["pop"][tracks]
    for S in FM.SURF:
        m = popm & (step_surface == S)
        out[f"P/{FM.NAME[S]}/num"] = np.bincount(e[m], weights=(wrong(new_ci[m], tp[m]).astype(float)
                                                                - wrong(old_ci[m], tp[m])), minlength=nev)
        out[f"P/{FM.NAME[S]}/den"] = np.bincount(e[m], weights=((new_ci[m] >= 0).astype(float) - (old_ci[m] >= 0)),
                                                 minlength=nev)
    return out
