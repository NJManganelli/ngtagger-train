"""Build fixed-window ADC images + per-cluster metadata from the ClusterAdcNtuple.

Output (data/):
  images.npy   uint8 [N, WX, WY]: raw 4-bit digi ADC + 1 (0 = no pixel), window centred
               on the cluster bounding box (row = local x / 25 um, col = local y / 100 um)
  meta.parquet one row per cluster, ntuple scalars + derived targets/flags
Split: whole events, by sorted-event rank: rank % 4 == 0 -> test, rank % 8 == 1 -> val.
"""
import sys, numpy as np, pandas as pd, uproot, awkward as ak, os

WX, WY = 24, 16
SRC = sys.argv[1] if len(sys.argv) > 1 else "/Users/nmangane/smartpixels/cmssw/work/clusterntuple/clusters_pu200_200ev_numEvent200.root"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[2] if len(sys.argv) > 2 else "data")
os.makedirs(OUT, exist_ok=True)

t = uproot.open(SRC)["clusterAdcNtuple/clusters"]
N = t.num_entries
scal = [k for k in t.keys() if not k.startswith("px")]
img = np.lib.format.open_memmap(os.path.join(OUT, "images.npy"), mode="w+", dtype=np.uint8, shape=(N, WX, WY))
metas = []
i0 = 0
for arr in t.iterate(scal + ["pxRow", "pxCol", "pxAdc"], step_size=400_000, library="ak"):
    n = len(arr)
    sx = ak.to_numpy(arr["sizeX"]); sy = ak.to_numpy(arr["sizeY"])
    offx = (WX - sx) // 2; offy = (WY - sy) // 2          # may be negative (truncation)
    cnt = ak.to_numpy(ak.num(arr["pxRow"]))
    ci = np.repeat(np.arange(n), cnt)
    r = ak.to_numpy(ak.flatten(arr["pxRow"])).astype(np.int32) + offx[ci]
    c = ak.to_numpy(ak.flatten(arr["pxCol"])).astype(np.int32) + offy[ci]
    adc = ak.to_numpy(ak.flatten(arr["pxAdc"])).astype(np.int32)
    ok = (r >= 0) & (r < WX) & (c >= 0) & (c < WY)
    block = np.zeros((n, WX, WY), np.uint8)
    block[ci[ok], r[ok], c[ok]] = np.clip(adc[ok] + 1, 1, 255)
    img[i0:i0 + n] = block
    lost = np.bincount(ci[~ok], minlength=n)
    m = pd.DataFrame({k: ak.to_numpy(arr[k]) for k in scal})
    m["nPxLost"] = lost
    m["truncX"] = (sx > WX).astype(np.int8); m["truncY"] = (sy > WY).astype(np.int8)
    m["adcSum"] = np.bincount(ci, weights=adc + 1, minlength=n)
    metas.append(m)
    i0 += n
    print(i0, "/", N, flush=True)
img.flush()
meta = pd.concat(metas, ignore_index=True)

# --- targets in pixel units relative to the bounding-box centre
def rel(x, o, p, s):
    return (x - o) / p - s / 2.0
meta["tx_hx"] = rel(meta.hxLocalX, meta.originLocalX, meta.pitchX, meta.sizeX)
meta["ty_hx"] = rel(meta.hxLocalY, meta.originLocalY, meta.pitchY, meta.sizeY)
meta["tx_sh"] = rel(meta.shLocalX, meta.originLocalX, meta.pitchX, meta.sizeX)
meta["ty_sh"] = rel(meta.shLocalY, meta.originLocalY, meta.pitchY, meta.sizeY)
meta["tx_cpe"] = rel(meta.localX, meta.originLocalX, meta.pitchX, meta.sizeX)
meta["ty_cpe"] = rel(meta.localY, meta.originLocalY, meta.pitchY, meta.sizeY)
# --- event split (whole events)
ev = np.unique(meta.event.values)
rank = {e: i for i, e in enumerate(np.sort(ev))}
rk = meta.event.map(rank).values
meta["split"] = np.where(rk % 4 == 0, 2, np.where(rk % 8 == 1, 1, 0)).astype(np.int8)  # 0 train 1 val 2 test
meta.to_parquet(os.path.join(OUT, "meta.parquet"))
print("done", N, "events", len(ev), "split counts", np.bincount(meta.split))
