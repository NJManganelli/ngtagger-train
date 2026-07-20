"""Probe the SmartPixels PU nano for table inventory relevant to the
model-space study: SC8 jet tables, per-hit refit tables, GenPart content,
candidate crossrefs, chi2 pathology tail."""
import json
import sys

import numpy as np
import uproot

NANO = "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/nano_pu100_TrkSmartPix_withGen.root"

tree = uproot.open(f"{NANO}:Events")
keys = list(tree.keys())

prefixes = {}
for k in keys:
    p = k.split("_")[0]
    prefixes.setdefault(p, []).append(k)

print(f"n branches: {len(keys)}, n events: {tree.num_entries}")
print("\n== table prefixes (branch counts) ==")
for p in sorted(prefixes):
    print(f"  {p:50s} {len(prefixes[p])}")

print("\n== jet-like tables ==")
for p in sorted(prefixes):
    if "Jet" in p or "jet" in p:
        print(f"  {p}: {sorted(prefixes[p])}")

print("\n== candidate tables ==")
for p in sorted(prefixes):
    if "Puppi" in p or "Cand" in p:
        print(f"  {p}: n_branches={len(prefixes[p])}")
        if "l1TrackIdx" in [b.split("_", 1)[-1] for b in prefixes[p]]:
            print("     -> has l1TrackIdx")

print("\n== GenPart branches ==")
print(sorted(prefixes.get("GenPart", [])))

print("\n== SmartPixels refit hit tables (first config) ==")
for p in sorted(prefixes):
    if "RefitHit" in p:
        print(f"  {p}: {sorted(b.split('_',1)[-1] for b in prefixes[p])}")
        break

print("\n== chi2 tail check (AAAA variant) ==")
c = tree["L1TSmartPixelsTrackDigiRefitAAAA_spxChi2IncRPhiTot"].array(library="np")
flat = np.concatenate(c) if c.dtype == object else np.asarray(c).ravel()
flat = np.asarray(uproot.open(f"{NANO}:Events")["L1TSmartPixelsTrackDigiRefitAAAA_spxChi2IncRPhiTot"].array().to_numpy() if False else [])
import awkward as ak
arr = ak.to_numpy(ak.flatten(tree["L1TSmartPixelsTrackDigiRefitAAAA_spxChi2IncRPhiTot"].array()))
print(f"  chi2IncRPhiTot: max={arr.max():.3e}, n>2e6: {(arr > 2e6).sum()}, "
      f"q99={np.quantile(arr, 0.99):.3e}, q999={np.quantile(arr, 0.999):.3e}")
