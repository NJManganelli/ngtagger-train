"""The P0.2 figures of merit for any refit output (PLAN §4 E5; sheet fom.harm, truth.genealogy).

P0.2 computes them inline in eval_refitq/p02_audit/analyze.py main(); its constants are imported here
(never copied) and the definitions follow its code line for line:

  population  OT tracks with a TTTrackAssociator TP (L1TTrack_tpIdx >= 0), TP pT >= 2 GeV, |TP eta| < 2.4
  origin      0 prompt (not b/c, r_prod < 100 um), 1 b/c daughter, 2 displaced inside IL1, 3 beyond IL1
  harm        |res_SP| - |res_OT| > 0.5 mm in d0 or > 1 mm in z0 (residual = reconstructed - TP, at the POCA)
  H           harmed ("either") fraction of prompt population tracks per |eta| x pT bin (12 bins)
  P           wrong accepted hits / accepted hits per surface, population tracks; wrong = the accepted
              cluster's dominant TP is not the track's TP (or none)

The closure test (gates.p02_closure) recomputes P0.2's per-event file-3 references from CMSSW's own
refit-v2 tracks with this module; equality certifies the transcription.
"""
from __future__ import annotations

import sys

import numpy as np

import common as CM

sys.path.insert(0, str(CM.HERE.parent / "p02_audit"))
import analyze as P02          # noqa: E402

SURF = (4, 3, 2, 1, 14, 13, 12, 11)
NAME = {s: (f"ID{s - 10}" if s > 10 else f"IL{s}") for s in SURF}
H_BINS = [(e, p) for e in P02.ETA_BINS for p in P02.PT_BINS]


def h_key(e, p):
    return f"eta{e[0]:g}-{e[1]:g}|pt{p[0]:g}-{p[1]:g}"


def track_truth(ctx):
    """Per chunk track: the TP columns at L1TTrack_tpIdx (NaN without a TP), population and origin."""
    tp = ctx["trk_tp"]
    TP = ctx["TP"]
    ok = tp >= 0
    g = lambda c: np.where(ok, TP[c][np.maximum(tp, 0)], np.nan)
    pt, eta = g("pt"), g("eta")
    r = np.hypot(g("vx"), g("vy"))
    hf = ok & ((g("genFromB") > 0) | (g("genFromC") > 0))
    pop = ok & (pt >= 2.0) & (np.abs(eta) < 2.4)
    org = np.full(len(tp), -1, np.int8)
    org[ok & ~hf & (r < P02.R_PROMPT_CM)] = 0
    org[hf] = 1
    org[ok & ~hf & (r >= P02.R_PROMPT_CM) & (r <= P02.R_IL1_CM)] = 2
    org[ok & ~hf & (r > P02.R_IL1_CM)] = 3
    return {"pt": pt, "eta": eta, "d0": g("d0"), "z0": g("z0"), "pop": pop, "origin": org,
            "evt": g("evt"), "bx": g("bx")}


def harm(d0_sp, z0_sp, d0_ot, z0_ot, truth):
    """Per track: harmed in d0, z0, either (P0.2: |res_SP| - |res_OT| > 500 / 1000 um)."""
    r = lambda x, p: (np.asarray(x, np.float64) - truth[p]) * 1e4
    hd0 = np.abs(r(d0_sp, "d0")) - np.abs(r(d0_ot, "d0")) > P02.HARM_D0_UM
    hz0 = np.abs(r(z0_sp, "z0")) - np.abs(r(z0_ot, "z0")) > P02.HARM_Z0_UM
    return {"d0": hd0, "z0": hz0, "either": hd0 | hz0}


def per_event_counts(ev, nev, truth, hm, hit_trk, hit_surface, hit_acc, hit_wrong):
    """The E5 reference counts per event (P0.2's per_ev_ref layout): H num / den per bin (prompt
    population), P num / den per surface (population tracks' accepted hits)."""
    out = {}
    pr = truth["pop"] & (truth["origin"] == 0)
    aeta, pt = np.abs(truth["eta"]), truth["pt"]
    for e, p in H_BINS:
        m = pr & (aeta >= e[0]) & (aeta < e[1]) & (pt >= p[0]) & (pt < p[1])
        k = h_key(e, p)
        out[f"H/{k}/num"] = np.bincount(ev[m], weights=hm["either"][m], minlength=nev)
        out[f"H/{k}/den"] = np.bincount(ev[m], minlength=nev).astype(float)
    hpop = truth["pop"][hit_trk]
    hev = ev[hit_trk]
    for S in SURF:
        sel = hpop & (hit_surface == S) & hit_acc
        out[f"P/{NAME[S]}/num"] = np.bincount(hev[sel], weights=hit_wrong[sel], minlength=nev)
        out[f"P/{NAME[S]}/den"] = np.bincount(hev[sel], minlength=nev).astype(float)
    return out


def hits_of_record(rec, ctx):
    """The accepted hits of an engine output (out["rec"]): track, surface, wrong (the cluster's dominant
    TP is not the track's TP, or none)."""
    t, j = np.nonzero(rec["acc"])
    ci = rec["ci"][t, j]
    ctp = ctx["K"]["tp_global"][ci]
    wrong = (ctp < 0) | (ctp != ctx["trk_tp"][t])
    return t, rec["surface"][t, j], np.ones(len(t), bool), wrong


def hits_of_cmssw(H, ctx):
    """The producer's recorded crossings: track, surface, accepted, wrong (selHitClass != 0)."""
    toff = np.concatenate([[0], np.cumsum(ctx["T"]["_n"])])
    t = toff[H["event"]] + H["trackIdx"].astype(np.int64)
    acc = (H["hitAccepted"] > 0) & (H["selClusterIdx"] >= 0)
    return t, H["layer"].astype(np.int64), acc, acc & (H["selHitClass"] != 0)
