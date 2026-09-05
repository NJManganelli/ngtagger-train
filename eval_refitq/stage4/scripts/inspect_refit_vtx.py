"""Check refit-table columns (vs REFIT_BDT_FEATURES needs), L1Vertex, and the
1:1 row alignment of L1TExtTrack with the refit variant table."""
import sys

import awkward as ak
import numpy as np
import uproot

path = sys.argv[1]
cfg = sys.argv[2] if len(sys.argv) > 2 else "AAAA"
f = uproot.open(path)
t = f["Events"]
keys = list(t.keys())

for tbl in [f"L1TSmartPixelsExtTrackDigiRefit{cfg}", f"L1TSmartPixelsExtRefitHitDigiRefit{cfg}"]:
    cols = sorted(k for k in keys if k.startswith(tbl + "_"))
    print(f"=== {tbl} ({len(cols)}) ===")
    for c in cols:
        print("   ", c.replace(tbl + "_", ""))

print("\n=== L1Vertex columns ===")
for c in sorted(k for k in keys if k.startswith("L1Vertex_")):
    print("   ", c)

# 1:1 alignment check: L1TExtTrack row count == refit variant row count per event
arrs = t.arrays(["nL1TExtTrack", f"nL1TSmartPixelsExtTrackDigiRefit{cfg}",
                 "L1TExtTrack_pt", f"L1TSmartPixelsExtTrackDigiRefit{cfg}_spixRefitPerformed",
                 "L1Vertex_z0", "L1ExtPuppiCand_l1TrackIdx", "L1ExtPuppiCand_dxy"])
nExt = ak.to_numpy(arrs["nL1TExtTrack"])
nVar = ak.to_numpy(arrs[f"nL1TSmartPixelsExtTrackDigiRefit{cfg}"])
print("\n=== alignment ===")
print("per-event nL1TExtTrack == nRefitVar all events:", bool(np.all(nExt == nVar)))
print("total ExtTrack rows:", int(nExt.sum()), " total refit rows:", int(nVar.sum()))

perf = arrs[f"L1TSmartPixelsExtTrackDigiRefit{cfg}_spixRefitPerformed"]
perf_flat = ak.to_numpy(ak.flatten(perf))
print("refitPerformed fraction:", float(perf_flat.mean()))

# L1Vertex: first vertex is the PV in emulator convention
nv = ak.to_numpy(ak.num(arrs["L1Vertex_z0"]))
print("\nL1Vertex per-event count: min/mean/max",
      int(nv.min()), float(nv.mean()), int(nv.max()))
pv_z0 = ak.to_numpy(ak.firsts(arrs["L1Vertex_z0"]))
print("PV z0 (firsts): finite frac", float(np.isfinite(pv_z0).mean()),
      " range", np.nanmin(pv_z0), np.nanmax(pv_z0))

# constituent l1TrackIdx coverage
li = arrs["L1ExtPuppiCand_l1TrackIdx"]
li_flat = ak.to_numpy(ak.flatten(li))
print("\nL1ExtPuppiCand l1TrackIdx: frac >=0 (has track):", float((li_flat >= 0).mean()),
      " total cands", len(li_flat))
