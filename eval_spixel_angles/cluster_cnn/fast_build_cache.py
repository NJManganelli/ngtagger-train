"""FAST class-C study: training arrays with the FIXED class-T definition (no pT cut) for the slim / distributional networks.

Class T = TP-linked (linkClass 2), charged, valid truth, helix within 1 cm (hxOk, hxDist < 1) -- NO pT cut (the CMSSW brief's
pT >= 0.25 was an error). Train = split 0 (capped), val = split 1 (capped 300k). Test rows are NOT cached here: the evaluation uses
cluster_cnn_v2/arrays/{barrel,disc}_test_* (all clusters of the 50 test events), exactly as the knn prototype.
Output: /Volumes/WDMac/smartpixels_scratch/emu_fast/arrays/{barrel,disc}_{train,val}_{img.npy,meta.parquet}
"""
import os, numpy as np, pandas as pd
from v2_common import load, D_ALL

OUT = "/Volumes/WDMac/smartpixels_scratch/emu_fast/arrays"; os.makedirs(OUT, exist_ok=True)
CAP = {"barrel": 3_000_000, "disc": 3_000_000}
KEEP = ["row", "event", "subdet", "layer", "disk", "side", "linkClass", "tpPt", "tpLocalCotAlpha", "tpLocalCotBeta", "hxDist",
        "tx_hx", "ty_hx", "sizeX", "sizeY", "size", "charge", "adcSum", "truncX", "truncY", "nPxLost", "globalR", "globalZ",
        "localX", "localY", "zOutward", "axXz", "axYz", "axZz", "axXr", "axYr", "axZr", "sideZ", "pu", "pv", "pw"]


def classT(m):
    return ((m.linkClass == 2) & (m.tpCharge != 0) & (m.tpLocalCotAlpha > -900) & (m.hxOk == 1) & (m.hxDist < 1.0)).values


if __name__ == "__main__":
    imgs = np.load(os.path.join(D_ALL, "images.npy"), mmap_mode="r")
    rng = np.random.default_rng(11)
    for reg, flt in [("barrel", [("subdet", "==", 1)]), ("disc", [("subdet", "==", 2), ("disk", "<=", 4)])]:
        for split, sp, cap in (("val", 1, 300_000), ("train", 0, CAP[reg])):
            m = load(flt + [("split", "==", sp)])
            m = m[classT(m)]
            if len(m) > cap:
                m = m.iloc[np.sort(rng.choice(len(m), cap, replace=False))]
            m = m[KEEP].reset_index(drop=True)
            rows = m.row.values
            assert (np.diff(rows) > 0).all()
            out = np.lib.format.open_memmap(os.path.join(OUT, f"{reg}_{split}_img.npy"), mode="w+", dtype=np.uint8,
                                            shape=(len(m),) + imgs.shape[1:])
            for i in range(0, len(m), 500_000):
                out[i:i + 500_000] = imgs[rows[i:i + 500_000]]
            out.flush(); del out
            m.to_parquet(os.path.join(OUT, f"{reg}_{split}_meta.parquet"))
            print(reg, split, len(m), "pT<0.25 frac", round(float((m.tpPt < 0.25).mean()), 4), flush=True)
