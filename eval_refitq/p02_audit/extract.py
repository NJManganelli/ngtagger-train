#!/usr/bin/env python
"""P0.2 refit-v2 audit, step 1: read each canon_v2 input file ONCE into a compact cache.

The inputs are read-only and will be deleted after a separate check, so everything the
audit needs is copied here, per (scenario, file), into one compressed npz:

  tracks (one row per L1TTrack = per L1TSmartPixelsTrack, rows aligned, asserted):
    event key, OT and SP d0 / z0 / pT / eta / cov_d0_d0 / cov_z0_z0, refit status, the
    refit's own compatibility columns, and the OT track's truth TP (L1TTrack_tpIdx, the
    loosely-genuine TTTrackAssociator match, which is selHitClass's reference) with its
    L1TTP genealogy (genFromB/C, evt, bx, production vertex) and helix truth (d0, z0).
  hits (one row per refit crossing record, L1TSmartPixelsRefitHit, in visit order):
    the owning track row, surface, accepted, selHitClass, the selected cluster's dominant
    TP charge fraction, its emulator class (tpEmuClass), and its dominant TP's genealogy.

Every link is asserted: SP l1TrackIdx == row; cluster[selClusterIdx].detId == hit detId
and .surface == hit layer (the producer's contract); hit tpChargeFrac == cluster
tpChargeFrac; selHitClass 1 <=> cluster tpIdx != track tpIdx; L1TTP_d0[tpIdx] ==
L1TTrack_tp_d0.

    python extract.py [--scenarios AAAA AAAI IAAA] [--files 3 4 ... 10] [--chunk 25]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from ngtagger.data import spix_nano as SPN

IN_DIR = Path("/Volumes/WDMac/smartpixels_scratch/canon_v2")
OUT_DIR = Path("/Volumes/WDMac/smartpixels_scratch/p02_refit_audit")
CACHE_FMT = 1

OT_T, SP_T, HIT_T, TP_T = "L1TTrack", "L1TSmartPixelsTrack", "L1TSmartPixelsRefitHit", SPN.TP_TABLE
OT_COLS = ["d0", "z0", "pt", "eta", "phi", "cov_d0_d0", "cov_z0_z0", "nStubs", "tpIdx", "tp_d0"]
SP_COLS = ["d0", "z0", "pt", "eta", "cov_d0_d0", "cov_z0_z0", "l1TrackIdx", "spixRefitPerformed",
           "spixNAcceptedHits", "spixNCrossings", "spixLayerHitMask", "tpIdx", "spixShiftChi2",
           "spixChi2ITAtOT", "spixChi2ITAtRefit"]
TP_COLS = ["d0", "z0", "pt", "eta", "phi0", "vx", "vy", "vz", "genFromB", "genFromC", "evt", "bx",
           "pdgId", "charge", "genPartIdx"]
HIT_COLS = ["trackIdx", "layer", "hitAccepted", "selClusterIdx", "selHitClass", "tpChargeFrac",
            "clusterMerged", "detId", "windowMult", "selChi2Margin"]
CL_COLS = ["detId", "surface", "tpIdx", "tpEmuClass", "tpChargeFrac"]
# L1TTP columns carried for the selected cluster's dominant TP (the wrong hit's origin)
CLTP_COLS = ["evt", "bx", "pt", "vx", "vy", "genFromB", "genFromC"]


def input_path(scen, f):
    return IN_DIR / f"spstudy_{scen}_f{f:02d}_100ev.root"


def cache_path(scen, f):
    return OUT_DIR / "cache" / f"p02_{scen}_f{f:02d}.npz"


def sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def free_memory_pct():
    """System-wide free memory percentage from macOS memory_pressure."""
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("%"))
    return 100


def wait_for_memory(min_free=25, poll=30, log=print):
    while (p := free_memory_pct()) < min_free:
        log(f"  memory free {p}% < {min_free}%: waiting {poll}s")
        time.sleep(poll)


def flat(x, dtype=None):
    a = ak.to_numpy(ak.flatten(x))
    return a.astype(dtype) if dtype is not None else a


def gather(table_col, idx, ok):
    """table_col[idx] per event (index 0 where not ok: the caller masks those rows);
    returns the flat numpy array. Every event has L1TTP and cluster rows (PU200)."""
    return flat(table_col[ak.where(ok, idx, 0)])


def read_chunk(t, keys, start, stop):
    br = ([f"{OT_T}_{c}" for c in OT_COLS] + [f"{SP_T}_{c}" for c in SP_COLS]
          + [f"{TP_T}_{c}" for c in TP_COLS] + [f"{HIT_T}_{c}" for c in HIT_COLS]
          + SPN.cluster_branches(keys, CL_COLS, cap_angles=False)
          + ["run", "luminosityBlock", "event"])
    return t.arrays(list(dict.fromkeys(br)), entry_start=start, entry_stop=stop)


def extract_chunk(A, start):
    nev = len(A["event"])
    ntrk = ak.num(A[f"{OT_T}_d0"])
    nsp = ak.num(A[f"{SP_T}_d0"])
    assert ak.all(ntrk == nsp), "OT and SP track tables differ in length"
    li = A[f"{SP_T}_l1TrackIdx"]
    assert ak.all(li == ak.local_index(li)), "SP l1TrackIdx is not the row index"

    ntrk_np = ak.to_numpy(ntrk).astype(np.int64)
    ev_local = np.repeat(np.arange(nev), ntrk_np)
    T = {"entry": (start + ev_local).astype(np.int32)}
    for c in OT_COLS:
        T[f"ot_{c}"] = flat(A[f"{OT_T}_{c}"])
    for c in SP_COLS:
        T[f"sp_{c}"] = flat(A[f"{SP_T}_{c}"])

    # OT truth TP (loosely genuine); tpIdx <= -3 must not occur with the 1GeVLinked L1TTP
    tpi = A[f"{OT_T}_tpIdx"]
    assert not ak.any(tpi <= -3), "OT tpIdx <= -3: TP not written"
    ok = tpi >= 0
    for c in TP_COLS:
        v = gather(A[f"{TP_T}_{c}"], tpi, ok)
        T[f"tp_{c}"] = v
    m = T["ot_tpIdx"] >= 0
    dd = np.abs(T["tp_d0"][m] - T["ot_tp_d0"][m])
    assert dd.size == 0 or dd.max() < 1e-6, "L1TTP_d0[tpIdx] != L1TTrack_tp_d0"

    # hits, in visit order; global row of the owning track within this chunk
    toff = np.concatenate([[0], np.cumsum(ntrk_np)])
    nh = ak.to_numpy(ak.num(A[f"{HIT_T}_layer"])).astype(np.int64)
    hev = np.repeat(np.arange(nev), nh)
    H = {c: flat(A[f"{HIT_T}_{c}"]) for c in HIT_COLS}
    H["trk"] = (toff[hev] + H["trackIdx"]).astype(np.int64)
    assert np.all(np.diff(H["trk"]) >= 0), "hit rows not grouped by track"

    J = SPN.cluster_columns(A, CL_COLS, cap_angles=False)
    sci = A[f"{HIT_T}_selClusterIdx"]
    sel = sci >= 0
    hsel = H["selClusterIdx"] >= 0
    assert np.array_equal(hsel, H["hitAccepted"].astype(bool)), "selected != accepted"
    CL = {}
    for c in CL_COLS:
        v = gather(J[c], sci, sel)
        CL[c] = v
    assert np.array_equal(CL["detId"][hsel], H["detId"][hsel]), "cluster detId != hit detId"
    assert np.array_equal(CL["surface"][hsel], H["layer"][hsel]), "cluster surface != hit layer"
    fr_h, fr_c = H["tpChargeFrac"][hsel], CL["tpChargeFrac"][hsel]
    good = fr_c > -900
    assert np.allclose(fr_h[good], fr_c[good], atol=1e-3), "hit vs cluster tpChargeFrac"
    H["cl_tpIdx"] = np.where(hsel, CL["tpIdx"], -1).astype(np.int32)
    H["cl_emu"] = np.where(hsel, CL["tpEmuClass"], -9).astype(np.int16)
    H["cl_frac"] = np.where(hsel, CL["tpChargeFrac"], -999.0).astype(np.float32)

    # selHitClass consistency with the cluster's dominant TP vs the OT track TP
    trk_tp = T["ot_tpIdx"][H["trk"]]
    mt = hsel & (trk_tp >= 0)
    cls = H["selHitClass"]
    exp = np.where(H["cl_tpIdx"] < 0, 2, np.where(H["cl_tpIdx"] == trk_tp, 0, 1))
    nbad = int((exp[mt] != cls[mt]).sum())
    assert nbad == 0, f"selHitClass disagrees with cluster tpIdx on {nbad} hits"
    assert not np.any(H["cl_tpIdx"][hsel] <= -3), "cluster TP not written"

    # the selected cluster's dominant TP genealogy
    cl_tp_jag = J["tpIdx"][ak.where(sel, sci, 0)]
    okc = sel & (cl_tp_jag >= 0)
    for c in CLTP_COLS:
        v = gather(A[f"{TP_T}_{c}"], cl_tp_jag, okc).astype(np.float64)
        H[f"cltp_{c}"] = np.where(H["cl_tpIdx"] >= 0, v, -999).astype(
            np.float32 if c in ("pt", "vx", "vy") else np.int32)

    E = {"run": ak.to_numpy(A["run"]).astype(np.int64), "lumi": ak.to_numpy(A["luminosityBlock"]).astype(np.int64),
         "event": ak.to_numpy(A["event"]).astype(np.int64), "ntrk": ntrk_np.astype(np.int32)}
    del H["trackIdx"]
    return E, T, H


def extract_file(scen, f, chunk, log):
    path = input_path(scen, f)
    t = uproot.open(path)["Events"]
    keys = t.keys()
    nent = t.num_entries
    Es, Ts, Hs = [], [], []
    ntrk_before = 0
    for start in range(0, nent, chunk):
        stop = min(nent, start + chunk)
        A = read_chunk(t, keys, start, stop)
        E, T, H = extract_chunk(A, start)
        H["trk"] = H["trk"] + ntrk_before
        ntrk_before += len(T["entry"])
        Es.append(E); Ts.append(T); Hs.append(H)
        del A
    cat = lambda L: {k: np.concatenate([d[k] for d in L]) for k in L[0]}
    E, T, H = cat(Es), cat(Ts), cat(Hs)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    log(f"  {scen} f{f:02d}: {nent} events, {len(T['entry'])} tracks, {len(H['trk'])} crossings; peak RSS {rss:.2f} GB")
    return nent, E, T, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["AAAA", "AAAI", "IAAA"])
    ap.add_argument("--files", nargs="+", type=int, default=list(range(3, 11)))
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--checksums", default=str(OUT_DIR / "inputs.sha256"),
                    help="sha256sum output computed before reading; recomputed if absent")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    (OUT_DIR / "cache").mkdir(parents=True, exist_ok=True)
    logf = open(OUT_DIR / "extract.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    sums = {}
    if os.path.exists(args.checksums):
        for line in open(args.checksums):
            h, name = line.split()
            sums[name] = h
    log(f"extract start {time.strftime('%Y-%m-%d %H:%M:%S')} format {CACHE_FMT}")
    manifest = {}
    for scen in args.scenarios:
        for f in args.files:
            out = cache_path(scen, f)
            path = input_path(scen, f)
            st = path.stat()
            name = path.name
            if name not in sums:
                log(f"  sha256 {name}")
                sums[name] = sha256(path)
            if out.exists() and not args.force:
                log(f"  {scen} f{f:02d}: cached")
            else:
                wait_for_memory(log=log)
                nent, E, T, H = extract_file(scen, f, args.chunk, log)
                meta = json.dumps({"format": CACHE_FMT, "input": str(path), "sha256": sums[name],
                                   "size": st.st_size, "mtime": st.st_mtime, "n_events": nent,
                                   "scenario": scen, "file": f})
                np.savez_compressed(out, meta=np.array(meta),
                                    **{f"E_{k}": v for k, v in E.items()},
                                    **{f"T_{k}": v for k, v in T.items()},
                                    **{f"H_{k}": v for k, v in H.items()})
            manifest[name] = {"path": str(path), "sha256": sums[name], "size": st.st_size,
                              "mtime": st.st_mtime, "cache": str(out)}
    with open(OUT_DIR / "cache" / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    log(f"extract done {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
