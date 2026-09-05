"""Diagnose the constituent->track crossref: is l1TrackIdx unfilled across all
tables/files, and is there an alternative (charge-based, plain PuppiCand, or
kinematic match to L1TExtTrack)?"""
import sys

import awkward as ak
import numpy as np
import uproot

path = sys.argv[1]
f = uproot.open(path)
t = f["Events"]
keys = list(t.keys())

def stats(name):
    if name not in keys:
        print(f"  {name}: ABSENT")
        return None
    a = t[name].array()
    fl = ak.to_numpy(ak.flatten(a)) if a.ndim > 1 else ak.to_numpy(a)
    if fl.dtype.kind in "iu" or fl.dtype.kind == "f":
        print(f"  {name}: n={len(fl)} min={fl.min()} max={fl.max()} "
              f"frac>=0={float((fl>=0).mean()):.3f} unique(<=8)={np.unique(fl)[:8]}")
    return fl

print("=== crossref index columns on L1ExtPuppiCand ===")
for c in ["L1ExtPuppiCand_l1TrackIdx", "L1ExtPuppiCand_hgcClusterIdx",
          "L1ExtPuppiCand_jetIdx", "L1ExtPuppiCand_charge", "L1ExtPuppiCand_id",
          "L1ExtPuppiCand_pdgId"]:
    stats(c)

print("\n=== plain L1PuppiCand index columns (alt tier) ===")
for c in sorted(k for k in keys if k.startswith("L1PuppiCand_")):
    print("   ", c)

# Is the charged fraction of ExtPuppiCand ~ the frac that SHOULD have a track?
li = stats("L1ExtPuppiCand_l1TrackIdx")
idc = t["L1ExtPuppiCand_id"].array()
ch = t["L1ExtPuppiCand_charge"].array()
idc_f = ak.to_numpy(ak.flatten(idc))
ch_f = ak.to_numpy(ak.flatten(ch))
print("\nExtPuppiCand: charged (|charge|>0) frac:", float((np.abs(ch_f) > 0).mean()))
print("ExtPuppiCand id values (0=chHad,1=el,2=nHad,3=photon,4=muon):", np.unique(idc_f, return_counts=True))

# check L1TExtTrack size vs charged cands (could kinematically match)
nExt = ak.to_numpy(t["nL1TExtTrack"].array())
nCand = ak.to_numpy(ak.num(idc))
print("\nper-event: nExtTrack mean", float(nExt.mean()), " nExtPuppiCand mean", float(nCand.mean()))
