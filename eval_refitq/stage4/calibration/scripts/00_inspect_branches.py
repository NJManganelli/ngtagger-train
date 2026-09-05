"""Inspect NG-tagger-related branch names in the unified coherent nanos (bounded output)."""
import uproot

F = "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/nano_fat_1111_coopt_file1.root"
with uproot.open(F + ":Events") as t:
    keys = list(t.keys())
    ng = sorted(k for k in keys if k.startswith("L1puppiJetSC4NG_"))
    print(f"n branches total: {len(keys)}; L1puppiJetSC4NG_*: {len(ng)}")
    for k in ng:
        print(" ", k)
    other = sorted({k.split("_")[0] for k in keys if "SC4" in k})
    print("SC4-ish tables:", other)
