"""Build fixed-window ADC images + per-cluster metadata from the ClusterAdcNtuple.

  prep.py [SRC] [OUT]      OUT is relative to this directory (data/ -> WDMac symlink)

Output (OUT/):
  images.npy   uint8 [N, WX, WY]: raw 4-bit digi ADC + 1 (0 = no pixel), window centred
               on the cluster bounding box (row = local x / 25 um, col = local y / 100 um)
  meta.parquet one row per cluster, ntuple scalars + derived targets/flags; `row` is the
               image index, so readers may filter rows (e.g. by subdet) and keep the link
Split: whole events, by sorted-event rank: rank % 4 == 0 -> test, rank % 8 == 1 -> val.
Written in chunks (the disc ntuple has ~30M rows), so memory stays bounded.
"""
import sys, numpy as np, pandas as pd, uproot, awkward as ak, os, pyarrow as pa, pyarrow.parquet as pq

WX, WY = 24, 16
SRC = sys.argv[1] if len(sys.argv) > 1 else "/Users/nmangane/smartpixels/cmssw/work/clusterntuple/clusters_pu200_200ev_numEvent200.root"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[2] if len(sys.argv) > 2 else "data")
os.makedirs(OUT, exist_ok=True)

t = uproot.open(SRC)["clusterAdcNtuple/clusters"]
N = t.num_entries
scal = [k for k in t.keys() if not k.startswith("px")]
ev = np.unique(t["event"].array(library="np"))
rank = pd.Series(np.arange(len(ev)), index=np.sort(ev))
img = np.lib.format.open_memmap(os.path.join(OUT, "images.npy"), mode="w+", dtype=np.uint8, shape=(N, WX, WY))


def rel(x, o, p, s):
    """target in pixel units relative to the bounding-box centre"""
    return (x - o) / p - s / 2.0


writer = None
i0 = 0
nsplit = np.zeros(3, np.int64)
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
    m = pd.DataFrame({k: ak.to_numpy(arr[k]) for k in scal})
    m["row"] = np.arange(i0, i0 + n, dtype=np.int64)
    m["nPxLost"] = np.bincount(ci[~ok], minlength=n)
    m["truncX"] = (sx > WX).astype(np.int8); m["truncY"] = (sy > WY).astype(np.int8)
    m["adcSum"] = np.bincount(ci, weights=adc + 1, minlength=n)
    for tag, px, py in [("hx", "hxLocalX", "hxLocalY"), ("sh", "shLocalX", "shLocalY"), ("cpe", "localX", "localY")]:
        m[f"tx_{tag}"] = rel(m[px], m.originLocalX, m.pitchX, m.sizeX)
        m[f"ty_{tag}"] = rel(m[py], m.originLocalY, m.pitchY, m.sizeY)
    rk = rank.loc[m.event.values].values
    m["split"] = np.where(rk % 4 == 0, 2, np.where(rk % 8 == 1, 1, 0)).astype(np.int8)  # 0 train 1 val 2 test
    nsplit += np.bincount(m.split, minlength=3)
    tab = pa.Table.from_pandas(m, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(os.path.join(OUT, "meta.parquet"), tab.schema)
    writer.write_table(tab)
    i0 += n
    print(i0, "/", N, flush=True)
img.flush()
writer.close()
print("done", N, "events", len(ev), "split counts", nsplit)
