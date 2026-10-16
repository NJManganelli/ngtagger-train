#!/usr/bin/env python
"""P0.3 step 1: read each AAAA file of the era (common.ERA) ONCE into a compact per-file cache (own loader:
uproot only; no census loader, plan D.3).

Per cluster (raw nano columns, float32 as stored) for:
  * every cluster linked to a TP with pT >= 1 GeV, and every cluster of a hard-scatter TP
    (the table population and every ROC signal and same-interaction category);
  * a deterministic 1-in-SUB subsample of all other clusters (pile-up TPs below 1 GeV,
    unlinked), with weight SUB (the ROC's cluster retention is computed over all clusters).
Joined from L1TTP through tpIdx (truth.cluster_tp): pT, eta, charge, the event origin
(hard scatter = evt 0 and bx 0), genFromB / genFromC, the production vertex, d0, z0.
Per event: GenVtx_z, the generator jets (pT, eta, phi, hadron flavour), and exact cluster counts
per surface x category (the subsample's check). Streaming per event (not subsampled): the
shared-cluster counts within dR < 0.1 of b-jet axes and inclusively.

    python extract.py [--files 1 ... 10] [--scenario AAAA] [--events N] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

IN_DIR = CM.IN_DIR          # the era's TTbar files (common.ERA)
OUT_DIR = CM.CACHE_DIR
CACHE_FMT = 1
SUB = 20                    # 1-in-SUB subsample of the clusters outside the kept population
KEEP_PT = 1.0               # GeV: linked clusters at or above it are all kept
CL = "L1TSmartPixelsCluster"
CL_F4 = ("globalR", "globalZ", "globalPhi", "globalClusterPhi", "globalClusterCotTheta",
         "sigGlobalClusterPhi", "sigGlobalClusterCotTheta", "tpGlobalClusterPhi", "tpGlobalClusterCotTheta",
         "localCotAlpha", "localCotBeta", "tpLocalCotAlpha", "tpLocalCotBeta", "sigAlpha", "sigBeta",
         "tpChargeFrac")
CL_INT = ("surface", "detId", "idxInDet", "hasAlpha", "hasBeta", "tpEmuClass", "tpIdx", "side")
TP_COLS = ("pt", "eta", "charge", "evt", "bx", "genFromB", "genFromC", "vx", "vy", "vz", "d0", "z0")
JET_COLS = ("pt", "eta", "phi", "hadronFlavour")
SURFACES = (1, 2, 3, 4, 11, 12, 13, 14)
# exact-count categories: 0 unlinked; then hard scatter / pile-up x TP pT bins
PT_BINS = (0.0, 0.5, 1.0, 1.5, 2.0, np.inf)
N_CAT = 1 + 2 * (len(PT_BINS) - 1)
SHARED_FRACS = (0.5, 0.7, 0.9)      # dominant-TP charge fraction below which a cluster is shared
BJET_PT, BJET_ETA, BJET_DR = 15.0, 2.4, 0.1


def input_path(scen: str, f: int) -> Path:
    return IN_DIR / f"spstudy_{scen}_f{f:02d}_100ev.root"


def splitmix(x: np.ndarray) -> np.ndarray:
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return x ^ (x >> np.uint64(31))


def sample_mask(gev, det, idx) -> np.ndarray:
    """Deterministic 1-in-SUB choice by (event, detId, idxInDet), the stable cluster key."""
    with np.errstate(over="ignore"):
        h = splitmix(gev.astype(np.uint64) * np.uint64(1_000_003) ^ det.astype(np.uint64) * np.uint64(7919)
                     ^ idx.astype(np.uint64))
    return (h % np.uint64(SUB)) == 0


def sha256_partial(path: Path, nbytes: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
        f.seek(max(0, path.stat().st_size - nbytes))
        h.update(f.read(nbytes))
    return h.hexdigest()


def extract_file(scen: str, f: int, out_dir: Path, events: int | None, step: int) -> dict:
    path = input_path(scen, f)
    t = uproot.open(path)["Events"]
    keys = set(t.keys())
    if "L1TTP_key" in keys or "L1TTP_idx" in keys:
        raise RuntimeError(f"{path}: key-linked nano refused")
    br = [f"{CL}_{c}" for c in CL_F4 + CL_INT] + [f"L1TTP_{c}" for c in TP_COLS] \
        + [f"GenJet_{c}" for c in JET_COLS] + ["GenVtx_z"]
    missing = [b for b in br if b not in keys]
    if missing:
        raise RuntimeError(f"{path}: missing branches {missing}")
    out = {k: [] for k in CL_F4 + CL_INT + tuple(f"tp_{c}" for c in TP_COLS if c not in ("evt", "bx", "genFromB", "genFromC"))
           + ("tp_hs", "tp_hf", "ev", "w")}
    ev_rows = {"ev": [], "genvtx_z": [], "jet_n": [], "counts": []}
    jets = {c: [] for c in JET_COLS}
    shared = []        # per event: [surface, (incone, inclusive), (all, T), (n, n_shared@fracs...)]
    ev0 = (f - 1) * 100
    n_seen = 0
    for A in t.iterate(br, step_size=step, entry_stop=events):
        ne = len(A)
        ncl = ak.to_numpy(ak.num(A[f"{CL}_surface"]))
        gev = np.repeat(np.arange(ev0 + n_seen, ev0 + n_seen + ne, dtype=np.int32), ncl)
        D = {c: ak.to_numpy(ak.flatten(A[f"{CL}_{c}"])) for c in CL_F4 + CL_INT}
        ntp = ak.to_numpy(ak.num(A["L1TTP_pt"]))
        toff = np.r_[0, np.cumsum(ntp)[:-1]]
        T = {c: ak.to_numpy(ak.flatten(A[f"L1TTP_{c}"])) for c in TP_COLS}
        ti = D["tpIdx"].astype(np.int64)
        linked = ti >= 0
        lev = gev - (ev0 + n_seen)
        row = np.where(linked, toff[lev] + np.clip(ti, 0, None), 0)
        if linked.any():
            assert (np.clip(ti, 0, None)[linked] < ntp[lev][linked]).all(), "tpIdx beyond L1TTP"
        tp = {c: T[c][row] for c in TP_COLS}
        hs = linked & (tp["evt"] == 0) & (tp["bx"] == 0)
        tp_pt = np.where(linked, tp["pt"], -1.0).astype(np.float32)
        keep_all = linked & ((tp_pt >= KEEP_PT) | hs)
        sub = ~keep_all & sample_mask(gev, D["detId"], D["idxInDet"])
        keep = keep_all | sub
        for c in CL_F4 + CL_INT:
            out[c].append(D[c][keep])
        for c in ("pt", "eta", "charge", "vx", "vy", "vz", "d0", "z0"):
            v = np.where(linked, tp[c], np.nan if c != "charge" else 0).astype(np.float32 if c != "charge" else np.int8)
            out[f"tp_{c}"].append(v[keep])
        out["tp_hs"].append(np.where(linked, hs.astype(np.int8), -1).astype(np.int8)[keep])
        hf = np.where(linked & tp["genFromB"].astype(bool), 1, np.where(linked & tp["genFromC"].astype(bool), 2, 0))
        out["tp_hf"].append(hf.astype(np.int8)[keep])
        out["ev"].append(gev[keep])
        out["w"].append(np.where(keep_all, 1, SUB).astype(np.uint8)[keep])
        # exact counts per event x surface x category
        cat = np.zeros(len(ti), np.int64)
        pb = np.clip(np.searchsorted(PT_BINS, tp_pt, side="right") - 1, 0, len(PT_BINS) - 2)
        cat[linked] = 1 + 2 * pb[linked] + (~hs[linked]).astype(np.int64)
        assert np.isin(D["surface"], SURFACES).all(), "unexpected surface code"
        sidx = np.searchsorted(SURFACES, D["surface"])
        flat = (lev * len(SURFACES) + sidx) * N_CAT + cat
        cnt = np.bincount(flat, minlength=ne * len(SURFACES) * N_CAT).reshape(ne, len(SURFACES), N_CAT)
        # shared clusters within dR < 0.1 of b-jet axes (from the generator vertex), and inclusive
        gz = ak.to_numpy(A["GenVtx_z"]).astype(np.float64)
        jp = {c: A[f"GenJet_{c}"] for c in JET_COLS}
        isb = (jp["hadronFlavour"] == 5) & (jp["pt"] > BJET_PT) & (abs(jp["eta"]) < BJET_ETA)
        ceta = np.arcsinh((D["globalZ"].astype(np.float64) - gz[lev]) / D["globalR"])
        cphi = D["globalPhi"].astype(np.float64)
        incone = np.zeros(len(ti), bool)
        for e in range(ne):
            be, bp = ak.to_numpy(jp["eta"][e][isb[e]]), ak.to_numpy(jp["phi"][e][isb[e]])
            if len(be) == 0:
                continue
            m = lev == e
            de = ceta[m][:, None] - be[None, :]
            dp = (cphi[m][:, None] - bp[None, :] + np.pi) % (2 * np.pi) - np.pi
            incone[m] = ((de * de + dp * dp) < BJET_DR ** 2).any(1)
        isT = D["tpEmuClass"] == 1
        frac = D["tpChargeFrac"]
        sh = np.zeros((ne, len(SURFACES), 2, 2, 1 + len(SHARED_FRACS)), np.int64)
        for ci, cm in enumerate((incone, np.ones(len(ti), bool))):
            for ti_, tm in enumerate((np.ones(len(ti), bool), isT)):
                m = cm & tm & linked
                base = (lev * len(SURFACES) + sidx)[m]
                sh[..., ci, ti_, 0] += np.bincount(base, minlength=ne * len(SURFACES)).reshape(ne, len(SURFACES))
                for k, fr in enumerate(SHARED_FRACS):
                    mm = m & (frac < fr)
                    b2 = (lev * len(SURFACES) + sidx)[mm]
                    sh[..., ci, ti_, 1 + k] += np.bincount(b2, minlength=ne * len(SURFACES)).reshape(ne, len(SURFACES))
        shared.append(sh)
        ev_rows["ev"].append(np.arange(ev0 + n_seen, ev0 + n_seen + ne, dtype=np.int32))
        ev_rows["genvtx_z"].append(gz.astype(np.float32))
        ev_rows["jet_n"].append(ak.to_numpy(ak.num(jp["pt"])).astype(np.int32))
        ev_rows["counts"].append(cnt.astype(np.int32))
        for c in JET_COLS:
            jets[c].append(ak.to_numpy(ak.flatten(jp[c])))
        n_seen += ne
        print(f"  f{f:02d}: {n_seen} events, kept {sum(len(x) for x in out['ev'])} clusters "
              f"(rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB)", flush=True)
    C = {k: np.concatenate(v) for k, v in out.items()}
    E = {f"ev_{k}": np.concatenate(v) for k, v in ev_rows.items()}
    J = {f"jet_{c}": np.concatenate(v) for c, v in jets.items()}
    S = {"shared_counts": np.concatenate(shared)}
    meta = {"fmt": CACHE_FMT, "scenario": scen, "file": f, "input": str(path), "events": n_seen,
            "input_bytes": path.stat().st_size, "input_mtime": path.stat().st_mtime,
            "input_sha256_partial": sha256_partial(path), "sub": SUB, "keep_pt": KEEP_PT,
            "surfaces": SURFACES, "pt_bins": [float(x) for x in PT_BINS], "shared_fracs": SHARED_FRACS,
            "bjet": {"pt": BJET_PT, "eta": BJET_ETA, "dr": BJET_DR},
            "shared_counts_axes": "event, surface, (in b-jet cone, inclusive), (all linked, class T), "
                                  "(n, n shared at each frac)",
            "count_categories": "0 unlinked; 1 + 2 * ptbin + (0 hard scatter, 1 pile-up)"}
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"p03_{scen}_f{f:02d}.npz"
    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **C, **E, **J, **S)
    os.replace(tmp, dst)
    return {"file": str(dst), "events": n_seen, "clusters_kept": int(len(C["ev"])),
            "clusters_total": int(E["ev_counts"].sum()), "bytes": dst.stat().st_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="AAAA")
    ap.add_argument("--files", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--events", type=int, default=None, help="per file (default: all)")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    t0 = time.time()
    res = []
    for f in a.files:
        r = extract_file(a.scenario, f, Path(a.out), a.events, a.step)
        print(json.dumps(r), flush=True)
        res.append(r)
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
