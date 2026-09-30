"""Flatten one input set (canon_vN AAAA + the coexist OT-fed L1Vertex) into per-file numpy caches.

One row per L1TSmartPixelsTrack (== one per L1TTrack; l1TrackIdx is checked to be the row
index). Prefixes: ot_ (L1TTrack), sp_ (L1TSmartPixelsTrack), ottp_/sptp_ (L1TTP row of the
OT loosely-genuine truth / the SmartPixels weighted cluster-majority truth), ev_ (per event).
Events are ordered by event number in both files (the NANO output order differs).

  pixi run python eval_refitq/sub2gev/build_cache.py [--input-set canon_v2] [file numbers, default 1..10]
"""
from __future__ import annotations

import argparse
import os
import sys

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

OT_COLS = ["pt", "eta", "z0", "nStubs", "chi2XYRed", "chi2ZRed", "chi2BendRed", "tpIdx", "hwZ0"]
SP_COLS = ["pt", "eta", "z0", "hwZ0", "hwRinv", "l1TrackIdx", "spixRefitPerformed", "spixNAcceptedHits",
           "tpIdx", "tpNOTClusters", "tpNITClusters", "spixLayerHitMask"]
HIT_COLS = ["trackIdx", "hitAccepted", "selHitClass"]
TP_COLS = ["pt", "eta", "z0", "evt", "bx", "pdgId"]


def _flat(a, dtype=None):
    x = ak.to_numpy(ak.flatten(a))
    return x.astype(dtype) if dtype else x


def build(fnum: int):
    ev = uproot.open(common.CANON.format(fnum) + ":Events")
    br = (["event", "GenVtx_z", "L1Vertex_z0"] + [f"L1TTrack_{c}" for c in OT_COLS]
          + [f"L1TSmartPixelsTrack_{c}" for c in SP_COLS] + [f"L1TSmartPixelsRefitHit_{c}" for c in HIT_COLS]
          + [f"L1TTP_{c}" for c in TP_COLS])
    a = ev.arrays(br)
    a = a[np.argsort(ak.to_numpy(a.event))]
    x = uproot.open(common.COEXIST.format(fnum) + ":Events").arrays(["event", "L1Vertex_z0"])
    x = x[np.argsort(ak.to_numpy(x.event))]
    assert np.array_equal(ak.to_numpy(a.event), ak.to_numpy(x.event)), "canon/coexist events differ"
    nev = len(a)

    sp_idx = a.L1TSmartPixelsTrack_l1TrackIdx
    assert ak.all(sp_idx == ak.local_index(sp_idx)), "l1TrackIdx is not the row index"
    ntrk = ak.to_numpy(ak.num(sp_idx))
    out = {"evidx": np.repeat(np.arange(nev), ntrk)}
    for c in OT_COLS:
        out["ot_" + c] = _flat(a[f"L1TTrack_{c}"])
    for c in SP_COLS:
        out["sp_" + c] = _flat(a[f"L1TSmartPixelsTrack_{c}"])
    for c in ("hwZ0", "hwRinv"):  # unsigned words -> signed
        for p in ("ot_", "sp_"):
            if p + c in out:
                bits = 12 if c == "hwZ0" else 15
                v = out[p + c].astype(np.int64)
                out[p + c + "_signed"] = np.where(v >= 1 << (bits - 1), v - (1 << bits), v)

    # accepted refit hits per SP track: total and "wrong" by selHitClass (relative to the OT TP)
    h_trk = a.L1TSmartPixelsRefitHit_trackIdx
    acc = a.L1TSmartPixelsRefitHit_hitAccepted != 0
    cls = a.L1TSmartPixelsRefitHit_selHitClass
    offs = np.concatenate([[0], np.cumsum(ntrk)])[:-1]
    gidx = _flat(h_trk + offs[:, None] * ak.ones_like(h_trk), np.int64)
    ntot = int(ntrk.sum())
    accf = _flat(acc)
    clsf = _flat(cls)
    out["hit_nacc"] = np.bincount(gidx[accf], minlength=ntot)
    out["hit_nacc_other"] = np.bincount(gidx[accf & (clsf == 1)], minlength=ntot)
    out["hit_nacc_noise"] = np.bincount(gidx[accf & (clsf == 2)], minlength=ntot)

    # truth rows
    ntp = ak.num(a.L1TTP_pt)
    tpoffs = np.concatenate([[0], np.cumsum(ak.to_numpy(ntp))])[:-1]
    tp = {c: _flat(a[f"L1TTP_{c}"]) for c in TP_COLS}
    for pre, col in (("ottp_", "ot_tpIdx"), ("sptp_", "sp_tpIdx")):
        ti = out[col]
        ok = ti >= 0
        g = np.where(ok, ti + np.repeat(tpoffs, ntrk), 0)
        for c in TP_COLS:
            v = tp[c][g].astype(float)
            out[pre + c] = np.where(ok, v, np.nan)

    out["ev_event"] = ak.to_numpy(a.event)
    out["ev_genz"] = ak.to_numpy(a.GenVtx_z).astype(float)
    out["ev_l1v_sp"] = ak.to_numpy(ak.firsts(a.L1Vertex_z0)).astype(float)
    out["ev_l1v_ot"] = ak.to_numpy(ak.firsts(x.L1Vertex_z0)).astype(float)
    out["ev_file"] = np.full(nev, fnum)
    os.makedirs(os.path.dirname(common.CACHE.format(fnum)), exist_ok=True)
    np.savez_compressed(common.CACHE.format(fnum), **out)
    print(f"f{fnum:02d}: {nev} events, {ntot} tracks, "
          f"refit {out['sp_spixRefitPerformed'].mean():.3f}, SP<2 {np.sum(out['sp_pt'] < 2)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-set", choices=sorted(common.INPUT_SETS), default="canon_v1")
    ap.add_argument("files", nargs="*", type=int, default=common.FILES_ALL)
    a = ap.parse_args()
    common.configure(a.input_set)
    for f in a.files:
        build(f)
