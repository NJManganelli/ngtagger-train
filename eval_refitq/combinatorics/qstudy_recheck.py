"""Re-check of the refit's process-noise (Q) scale study with the Q sign fixed.

The settled Q study (2026-09-05, scale 0.00075 adopted, per-layer scales not
fitted because the residual miss followed VISIT DEPTH) was measured with the
producer's Q, whose transverse d0 term has the wrong sign for TTTrack d0
(refit_calibration_fit.kink_cov). This replays the producer's OWN refit on the
production files -- its selected clusters, its alpha-only update, its Newton /
Jacobian scheme, its rule that Q is charged only after the first update -- and
swaps ONLY the Q formula:

  producer    jT = (0, 1, 0, 0, -r_s), jL = (0, 0, sec^2, -r_s sec^2, 0), theta = k / pT
  corrected   jT = (0, 1, 0, 0, +r_s), jL = (0, 0, sec,  -r_s sec,  0), theta = k / pT
              (k/pT is TMTT's TRANSVERSE deflection; the dip changes by that times
              cos(lambda), so dtanL = sec^2 * cos * k/pT = sec * k/pT)

and measures what the old study measured: the pull r/sqrt(S) of the CORRECT
selected cluster (selHitClass == 0) in local x and y, per layer (visit order)
and per pT bin, as robust width (MAD) and RMS.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ot_projection as OP          # noqa: E402
import ot_refit_projection as RP    # noqa: E402

PT_BINS = [2, 3, 4, 6, 10, 1e9]
COLS = ["selClusterIdx", "hitAccepted", "selHitClass"]


def q_producer_form(a, C, t, L, prevL, prevR, kpt, k, form):
    if not len(t) or k <= 0:
        return
    pt = kpt / np.maximum(np.abs(a[t, 0]), 1e-12)
    var = (k / pt) ** 2 * np.maximum(1, np.abs(L - prevL[t]))
    rs, tanl = prevR[t], a[t, 2]
    sec2 = 1.0 + tanl ** 2
    jT = np.zeros((len(t), 5)); jT[:, 1] = 1.0
    jL = np.zeros((len(t), 5))
    if form == "producer":
        jT[:, 4] = -rs
        jL[:, 2] = sec2; jL[:, 3] = -rs * sec2
    else:
        sec = np.sqrt(sec2)
        jT[:, 4] = +rs
        jL[:, 2] = sec; jL[:, 3] = -rs * sec
    C[t] += var[:, None, None] * (jT[:, :, None] * jT[:, None, :] + jL[:, :, None] * jL[:, None, :])


def replay_pulls(A, geo, k, form):
    ctx = OP.prepare_chunk(A, geo)
    K = ctx["K"]
    H = OP._flat(A, OP.REFIT, OP.REFIT_COLS + COLS)
    toff = np.concatenate([[0], np.cumsum(ak.to_numpy(ak.num(A["L1TTrack_pt"])))])
    coff = np.concatenate([[0], np.cumsum(ak.to_numpy(ak.num(A["L1TSmartPixelsCluster_detId"])))])
    gt = toff[H["event"]] + H["trackIdx"].astype(np.int64)
    kpt = RP.pt_constant(ctx["T"])
    a, C = ctx["a"].copy(), ctx["C"].copy()
    n = ctx["ntrk"]
    st = {"nUpd": np.zeros(n, np.int64), "prevR": np.full(n, -1.0), "prevL": np.zeros(n, np.int64)}
    out = []
    for L in (4, 3, 2, 1):
        idx = np.flatnonzero(H["layer"] == L)
        mi = OP.module_index(geo, H["detId"][idx])
        ok = mi >= 0
        idx, mi = idx[ok], mi[ok]
        t = gt[idx]
        q = t[(st["nUpd"][t] > 0) & (st["prevR"][t] > 0)]
        q_producer_form(a, C, q, L, st["prevL"], st["prevR"], kpt, k, form)
        pj = OP.project_with_cov(a[t], C[t], OP.cylinder_s(a[t], geo["R"][L]), geo, mi,
                                 jac="cmssw", R=L)
        acc = (H["hitAccepted"][idx] > 0) & (H["selClusterIdx"][idx] >= 0) & np.isfinite(pj["u"])
        u = np.flatnonzero(acc)
        if not len(u):
            continue
        ci = coff[H["event"][idx][u]] + H["selClusterIdx"][idx][u].astype(np.int64)
        Jz = np.where(np.isfinite(pj["J"][u]).all(1, keepdims=True)
                      & (np.abs(np.nan_to_num(pj["J"][u], nan=np.inf)) <= RP.JACOBIAN_MAX_ABS).all(1, keepdims=True),
                      np.nan_to_num(pj["J"][u]), 0.0)
        S = np.einsum("nij,njk,nlk->nil", Jz, C[t[u]], Jz)
        sx, sy = K["sigX"][ci].astype(np.float64), K["sigY"][ci].astype(np.float64)
        pu = (K["localX"][ci] - pj["u"][u]) / np.sqrt(S[:, 0, 0] + sx ** 2)
        pv = (K["localY"][ci] - pj["v"][u]) / np.sqrt(S[:, 1, 1] + sy ** 2)
        good = H["selHitClass"][idx][u] == 0
        out.append({"L": np.full(len(u), L), "pu": pu, "pv": pv, "good": good,
                    "pt": ctx["T"]["pt"][t[u]]})
        h0 = np.stack([pj["u"][u], pj["v"][u], pj["cotA"][u], pj["cotB"][u]], 1)
        meas = np.stack([K["localX"][ci], K["localY"][ci], K["localCotAlpha"][ci],
                         K["localCotBeta"][ci]], 1).astype(np.float64)
        sig = np.stack([sx, sy, K["sigAlpha"][ci], K["sigBeta"][ci]], 1).astype(np.float64)
        ud = np.stack([np.ones(len(u), bool), np.ones(len(u), bool), K["hasAlpha"][ci] > 0,
                       np.zeros(len(u), bool)], 1)
        RP.kalman_update(a, C, t[u], pj["J"][u], h0, meas, sig, ud)
        gp = geo["origin"][mi[u]] + pj["u"][u][:, None] * geo["ex"][mi[u]] + pj["v"][u][:, None] * geo["ey"][mi[u]]
        st["prevR"][t[u]] = np.hypot(gp[:, 0], gp[:, 1])
        st["nUpd"][t[u]] += 1
        st["prevL"][t[u]] = L
    return {k_: np.concatenate([d[k_] for d in out]) for k_ in out[0]}


def rs(x):
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan, np.nan
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m)), np.std(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputs", nargs="+", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("-n", "--nev", type=int, default=200)
    a = ap.parse_args()
    files = [f for g in a.inputs for f in (sorted(glob.glob(g)) or [g])]
    geo = OP.load_geometry(a.geometry)
    extra = [f"{OP.REFIT}_{c}" for c in COLS]
    chunks = []
    done = 0
    for A in uproot.iterate([f"{f}:Events" for f in files], OP.branches(True) + extra, step_size=20):
        chunks.append(A)
        done += len(A)
        if done >= a.nev:
            break
    for form in ("producer", "corrected"):
        for k in (0.0, 0.00075, 0.0015, 0.003):
            R = [replay_pulls(A, geo, k, form) for A in chunks]
            R = {k_: np.concatenate([r[k_] for r in R]) for k_ in R[0]}
            line = []
            for L in (4, 3, 2, 1):
                m = R["good"] & (R["L"] == L)
                wu, ru = rs(R["pu"][m])
                wb = [rs(R["pu"][m & (R["pt"] >= lo) & (R["pt"] < hi)])[0] for lo, hi in zip(PT_BINS[:-1], PT_BINS[1:])]
                line.append(f"L{L} x {wu:.3f}/{ru:.2f} span {np.nanmax(wb) - np.nanmin(wb):.3f}")
            print(f"{form:9s} k={k:.5f}: " + " | ".join(line), flush=True)
            if k == 0.00075:
                for L in (3, 2, 1):
                    m = R["good"] & (R["L"] == L)
                    wb = [rs(R["pu"][m & (R["pt"] >= lo) & (R["pt"] < hi)])[0] for lo, hi in zip(PT_BINS[:-1], PT_BINS[1:])]
                    print(f"      L{L} x-pull MAD by pT {PT_BINS[:-1]}: " + " ".join(f"{w:.2f}" for w in wb), flush=True)


if __name__ == "__main__":
    main()
