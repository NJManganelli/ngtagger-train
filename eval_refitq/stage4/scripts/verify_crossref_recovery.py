"""Verify the crossref recovery path: is L1PuppiCand row-aligned per-event with
L1ExtPuppiCand, is its l1TrackIdx usable, and is the dxy column filled?"""
import sys

import awkward as ak
import numpy as np
import uproot

path = sys.argv[1]
t = uproot.open(path)["Events"]
a = t.arrays([
    "nL1ExtPuppiCand", "nL1PuppiCand",
    "L1ExtPuppiCand_pt", "L1ExtPuppiCand_eta", "L1ExtPuppiCand_phi",
    "L1ExtPuppiCand_charge", "L1ExtPuppiCand_dxy", "L1ExtPuppiCand_z0",
    "L1PuppiCand_pt", "L1PuppiCand_eta", "L1PuppiCand_phi",
    "L1PuppiCand_charge", "L1PuppiCand_dxy", "L1PuppiCand_l1TrackIdx", "L1PuppiCand_z0",
    "nL1TExtTrack",
])
nE = ak.to_numpy(a["nL1ExtPuppiCand"])
nP = ak.to_numpy(a["nL1PuppiCand"])
print("per-event nExtPuppiCand == nPuppiCand all events:", bool(np.all(nE == nP)))
print("total ext", int(nE.sum()), " total plain", int(nP.sum()))

# row-for-row identity check on pt/eta/phi (are the two tables the same order?)
def flat(n): return ak.to_numpy(ak.flatten(a[n]))
if np.all(nE == nP):
    for f in ["pt", "eta", "phi", "charge", "z0"]:
        e = flat(f"L1ExtPuppiCand_{f}")
        p = flat(f"L1PuppiCand_{f}")
        same = np.allclose(e, p, atol=1e-3, rtol=1e-3)
        maxd = np.max(np.abs(e - p)) if len(e) == len(p) else -1
        print(f"  row-aligned identical {f}: {same}  maxdiff={maxd:.4g}")

# l1TrackIdx fill on plain tier
li = flat("L1PuppiCand_l1TrackIdx")
ch = flat("L1PuppiCand_charge")
print("\nL1PuppiCand l1TrackIdx frac>=0:", float((li >= 0).mean()))
print("  among charged (|q|>0) frac>=0:", float((li[np.abs(ch) > 0] >= 0).mean()))
nExtTrk = ak.to_numpy(a["nL1TExtTrack"])
print("  max l1TrackIdx:", li.max(), " max within nExtTrack range OK:",
      bool(li.max() < nExtTrk.max()))

# dxy fill status on both tiers
for tier in ["L1ExtPuppiCand", "L1PuppiCand"]:
    d = flat(f"{tier}_dxy")
    print(f"\n{tier}_dxy: nonzero frac {float((np.abs(d) > 1e-9).mean()):.3f} "
          f"range [{d.min():.4g},{d.max():.4g}] mean|dxy| {np.abs(d).mean():.4g}")
    dc = d[np.abs(ch) > 0] if tier == "L1PuppiCand" else d[np.abs(flat("L1ExtPuppiCand_charge")) > 0]
    print(f"   among charged: nonzero frac {float((np.abs(dc) > 1e-9).mean()):.3f}")
