"""Own-cluster diagnostics (PLAN §1 items 5 and 7, §4 E1, §10 PD1).

At every visited surface, after Q and before the update, the arm's state is projected onto every
module that holds a cluster of the track's own TP (L1TTrack_tpIdx, truth.sp_track), whether or not the
arm selects it. Per own cluster:

  table_pop     the table's criterion: class T (the TP helix within 1 cm), a usable direction, TP pT >= 1
  e1            the signed innovation-normalised angle pull (m' - h) / sqrt(sigma'^2 + (J C J^T)_aa), the
                E1(a) statistic (sigma' the arm's angle error: w sigma_local on the barrel with the table)
  t1            the table-only pull (m' - h) / sigma' (E1's attribution: table vs covariance)
  z             |e1|, the window cut's statistic
  pu, pv        the signed innovation position pulls (E2 / E3's reported cores)
  g_<dims>      the innovation chi2 over positions + angle dims (xy, xya, xyb, xyab): the hit gate's
  rank_chi2     the innovation chi2 over positions + the arm's ranking angles: PD1's primary choice
  helix_d       distance of the cluster from the TP's truth helix crossing of the module (PD1's report)
  clean         every earlier accepted hit of the track is its own TP's (PD8)
  keybin        the signed predicted-key bin (spdefs.fine_bin) per angle; orient the module orientation

Truth (tpIdx, class, the truth helix) defines WHICH clusters are the TP's own and is reported; it never
chooses between them: the own primary on a module is the lowest rank_chi2 (PD1).
"""
from __future__ import annotations

import numpy as np

import arms as AR
import engine as E
from common import OP
from ngtagger.spdefs import angles as SA


def own_rows(arm, st, j, t, S, ctx, geo, tools, kpt):
    K = ctx["K"]
    tp = ctx["trk_tp"][st.o[t]]
    sel = (K["tp_global"] >= 0) & (K["mi"] >= 0)
    sel &= np.asarray(geo["surface"][np.maximum(K["mi"], 0)] == S)
    cl = np.flatnonzero(sel)
    if not len(cl) or not (tp >= 0).any():
        return {}
    # (virtual track, own cluster) pairs: join on the global TP row
    o = np.argsort(K["tp_global"][cl], kind="stable")
    keys = K["tp_global"][cl][o]
    lo, hi = np.searchsorted(keys, tp, "left"), np.searchsorted(keys, tp, "right")
    n = np.where(tp >= 0, hi - lo, 0)
    if not n.sum():
        return {}
    v = np.repeat(t, n)
    ci = cl[o][np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)]
    mi = K["mi"][ci]
    pj = AR.project(arm, st.a[v], st.C[v], S, geo, mi, kpt)
    ok = np.isfinite(pj["u"])
    v, ci, mi = v[ok], ci[ok], mi[ok]
    h0 = np.stack([pj["u"][ok], pj["v"][ok], pj["cotA"][ok], pj["cotB"][ok]], 1)
    Jz = E.guard_zeroed(pj["J"][ok])
    Sz = np.einsum("nij,njk,nlk->nil", Jz, st.C[v], Jz)
    meas, sig, has = AR.measurements(ctx, geo, ci, mi, h0[:, 2], h0[:, 3], arm, tools)
    r = meas - h0
    out = {"v": v, "o": st.o[v], "step": np.full(len(v), j, np.int16), "first": np.full(len(v), j == 0),
           "surface": np.full(len(v), S, np.int16), "ci": ci, "mi": mi}
    # PD8: a clean history = every earlier accepted hit of the (virtual) track is its own TP's
    acc_prev = st.rec["acc"][v][:, :j]
    tp_prev = K["tp_global"][np.maximum(st.rec["ci"][v][:, :j], 0)]
    out["clean"] = ~(acc_prev & (tp_prev != ctx["trk_tp"][st.o[v]][:, None])).any(1)
    e1 = np.full((len(v), 2), np.nan)
    t1 = np.full((len(v), 2), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in (0, 1):
            e1[:, i] = r[:, 2 + i] / np.sqrt(sig[:, 2 + i] ** 2 + Sz[:, 2 + i, 2 + i])
            t1[:, i] = r[:, 2 + i] / sig[:, 2 + i]
    e1 = np.where(has, e1, np.nan)
    t1 = np.where(has, t1, np.nan)
    out.update(e1=e1, t1=t1, z=np.abs(e1), has=has)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["pu"] = r[:, 0] / np.sqrt(sig[:, 0] ** 2 + Sz[:, 0, 0])          # innovation position pulls
        out["pv"] = r[:, 1] / np.sqrt(sig[:, 1] ** 2 + Sz[:, 1, 1])
    out["ev"] = ctx["tev"][st.o[v]]
    for dims, d in (("xy", (False, False)), ("xya", (True, False)), ("xyb", (False, True)), ("xyab", (True, True))):
        dm = np.concatenate([np.ones((len(v), 2), bool), has & np.array(d)[None, :]], 1)
        out[f"g_{dims}"] = AR.innovation_chi2(r, Sz, sig, dm)
    rd = np.concatenate([np.ones((len(v), 2), bool), AR.angle_dims(arm, "rank", geo["surface"][mi], has)], 1)
    out["rank_chi2"] = AR.innovation_chi2(r, Sz, sig, rd)
    cols = {c: K[c][ci] for c in ("tpEmuClass", "hasAlpha", "hasBeta", "localCotAlpha", "localCotBeta",
                                  "sigGlobalClusterPhi", "sigGlobalClusterCotTheta", "globalClusterPhi",
                                  "globalClusterCotTheta")}
    tpr = K["tp_global"][ci]
    out["classT"] = SA.class_T(K["tpEmuClass"][ci])
    out["table_pop"] = out["classT"] & SA.usable(cols) & (ctx["TP"]["pt"][tpr] >= SA.PT_MIN)
    orient = tools.angles.table.orientation(geo["detId"][mi]) if tools.angles else np.zeros(len(v), np.int8)
    key = SA.key_from_local(np.full(len(v), S), h0[:, 2], h0[:, 3], orient)
    kb = SA.fine_bin(np.full(len(v), S), key)
    out["keybin"] = np.stack([kb, kb], 1)
    out["orient"] = orient
    # C1b: the disc rows' module-frame key bin and the module's facing (barrel rows: -1 / 0)
    out["dkeybin"] = SA.fine_bin(np.full(len(v), S), SA.local_incidence_key(h0[:, 2], h0[:, 3])) \
        if OP.SPN.is_disc_surface(S) else np.full(len(v), -1)
    out["facing"] = np.sign(geo["ez"][mi][:, 2]).astype(np.int8)
    # the TP's truth helix on the cluster's module (TTTrack conventions: rInv = q kpt / pT)
    TP = ctx["TP"]
    at = np.stack([TP["charge"][tpr] * kpt / np.maximum(TP["pt"][tpr], 1e-9), TP["phi0"][tpr],
                   np.sinh(TP["eta"][tpr]), TP["z0"][tpr], TP["d0"][tpr]], 1)
    pt_ = OP.project_to_modules(at, OP.surface_s(at, geo, S), geo, mi)
    out["helix_d"] = np.where(pt_["ok"], np.hypot(meas[:, 0] - pt_["u"], meas[:, 1] - pt_["v"]), np.nan)
    out["tp_pt"] = TP["pt"][tpr]
    return out


def mark_primary(R):
    """PD1: per (virtual track, step, module) the own class-T cluster with the lowest rank_chi2 is the
    own primary; the helix-nearest choice is marked beside it for the report."""
    n = len(R["v"])
    prim = np.zeros(n, bool)
    hprim = np.zeros(n, bool)
    if not n:
        R["primary"], R["helix_primary"] = prim, hprim
        return R
    elig = R["classT"]
    grp = (R["v"].astype(np.int64) * 64 + R["step"]) * (1 << 24) + R["mi"].astype(np.int64)
    for key, out in (("rank_chi2", prim), ("helix_d", hprim)):
        val = np.where(elig & np.isfinite(R[key]), R[key], np.inf)
        o = np.lexsort((R["ci"], val, grp))
        first = np.r_[True, grp[o][1:] != grp[o][:-1]]
        pick = o[first]
        out[pick[np.isfinite(val[pick])]] = True
    R["primary"], R["helix_primary"] = prim, hprim
    R["secondary"] = R["classT"] & ~prim
    return R
