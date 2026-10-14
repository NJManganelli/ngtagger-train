"""Bit-identity checks of the engine against ORP (PLAN §4 E2 / E3, §2 E6 identities).

The strongest statement available is "the same state after every Kalman update": both ORP and the engine
are run with a recorder wrapped around their update function, which stores the updated tracks' states
(a, C) after each call, in call order. Identity = every recorded array equal bit for bit.

  orp_cal_identity   ORP.run_chunk's calibrated `innovation` arm vs the engine's ORP-cal arm (E2 / E3)
  forced_identity    ORP.replay_against_refit vs engine.forced at its default options (E6)
  box_identity       ORP.selection_closure's pick vs arms.box_candidates' best on the recorded projections
"""
from __future__ import annotations

import contextlib

import numpy as np

import arms as AR
import common as CM
import engine as E
from common import OP, ORP


@contextlib.contextmanager
def recording(module, name, log):
    """Wrap module.<name>(a, C, t, ...) to append (t, a[t], C[t]) after each call."""
    orig = getattr(module, name)

    def wrapped(a, C, t, *args, **kw):
        r = orig(a, C, t, *args, **kw)
        log.append((np.asarray(t).copy(), a[t].copy(), C[t].copy()))
        return r
    setattr(module, name, wrapped)
    try:
        yield log
    finally:
        setattr(module, name, orig)


def compare_logs(la, lb, ids_b=None):
    """{n_calls, identical, first_mismatch, max_abs_diff}: the per-call (t, a, C) records. ids_b maps
    the second log's track indices to the first's (the engine's virtual tracks to the chunk's tracks)."""
    out = {"n_calls_a": len(la), "n_calls_b": len(lb), "identical": len(la) == len(lb), "first_mismatch": None,
           "max_abs_diff": 0.0}
    for i, (x, y) in enumerate(zip(la, lb)):
        tb = y[0] if ids_b is None else ids_b[y[0]]
        same_t = np.array_equal(x[0], tb)
        same = same_t and np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2])
        if same_t:
            out["max_abs_diff"] = max(out["max_abs_diff"], float(np.max(np.abs(x[1] - y[1]), initial=0.0)))
        if not same and out["first_mismatch"] is None:
            out["first_mismatch"] = i
            out["identical"] = False
    return out


def orp_cal_identity(A, geo, cfg, mask, tools):
    """E2 / E3: ORP's calibrated `innovation` arm and the engine's ORP-cal arm, on ORP's track set
    (track class >= 0), every updated state compared."""
    ctx = AR.prepare(A, geo)
    la, lb = [], []
    use = ctx["cls"] >= 0
    with recording(ORP, "kalman_update", la):
        ORP.run_chunk(A, geo, mask, cfg["useAngles"], ("innovation",), 4.0, False, tools.cal, CM.REFIT_HIT, cfg)
    with recording(E, "kf_update", lb):
        AR.run_arm(ctx, geo, cfg, mask, AR.orp_cal(cfg), tools, tracks=use)
    return compare_logs(la, lb, ids_b=np.flatnonzero(use))


def forced_identity(A, geo, cfg, mask):
    """E6: ORP's forced-hit replay and engine.forced (defaults), every updated state compared."""
    ctx = OP.prepare_chunk(A, geo)
    H = E.hit_table(A)
    la, lb = [], []
    with recording(ORP, "kalman_update", la):
        ORP.replay_against_refit(ctx, geo, ORP.pt_constant(ctx["T"]), CM.REFIT_HIT, cfg,
                                 ORP.surface_path(ctx["a"], geo), mask)
    with recording(E, "kf_update", lb):
        E.forced(ctx, H, geo, cfg, mask)
    return compare_logs(la, lb)


def box_identity(A, geo, cfg):
    """E6: the producer's box selection on CMSSW's recorded projections, ORP.selection_closure vs
    arms.box_candidates + the producer metric (published errors, capped angles)."""
    ctx = AR.prepare(A, geo)
    H = OP._flat(A, CM.REFIT_HIT, list(dict.fromkeys(OP.REFIT_COLS + ORP.REPLAY_COLS)))
    K = ctx["K"]
    coff = ctx["coff"]
    sc = ORP.selection_closure(H, K, coff, cfg)
    surf = H["layer"].astype(np.int64)
    xi, ci, order, trunc = AR.box_candidates(ctx, cfg, H["event"], H["detId"], surf,
                                             H["projLocalX"].astype(np.float64), H["projLocalY"].astype(np.float64))
    use_a, use_b = cfg["useAngles"] in ORP.USE_ANGLES, cfg["useAngles"] == "alphaBeta"
    pa = H["projCotAlpha"][xi].astype(np.float64)
    pb = H["projCotBeta"][xi].astype(np.float64)
    dx = K["localX"][ci].astype(np.float64) - H["projLocalX"][xi]
    dy = K["localY"][ci].astype(np.float64) - H["projLocalY"][xi]
    val = (dx / np.maximum(K["sigX"][ci], 1e-12)) ** 2 + (dy / np.maximum(K["sigY"][ci], 1e-12)) ** 2
    if use_a:
        val = val + np.where(K["hasAlpha"][ci] > 0, ((K["localCotAlpha"][ci] - pa) / np.maximum(K["sigAlpha"][ci], 1e-12)) ** 2, 0.0)
    if use_b:
        val = val + np.where(K["hasBeta"][ci] > 0, ((K["localCotBeta"][ci] - pb) / np.maximum(K["sigBeta"][ci], 1e-12)) ** 2, 0.0)
    best, _ = AR._ranked({"x": xi, "val": val, "order": order}, len(surf))
    pick = np.where(best >= 0, ci[np.maximum(best, 0)], -1)
    return {"n": int(len(surf)), "same_pick": float(np.mean(pick == sc["best"])),
            "same_truncated": float(np.mean(trunc == sc["truncated"])),
            "same_mult": float(np.mean(np.bincount(xi, minlength=len(surf)) == sc["mult"]))}
