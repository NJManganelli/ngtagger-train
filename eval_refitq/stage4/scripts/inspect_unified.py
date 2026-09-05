"""Inspect a unified coherent nano: enumerate table prefixes, and check the
tagger-relevant tiers + refit tables + GenJet flavor labels."""
import re
import sys

import uproot

path = sys.argv[1]
f = uproot.open(path)
t = f["Events"]
keys = list(t.keys())

prefixes = {}
for k in keys:
    m = re.match(r"([A-Za-z0-9]+)_", k)
    p = m.group(1) if m else k
    prefixes.setdefault(p, [])
    prefixes[p].append(k)

print("NEVENTS", t.num_entries)
print("=== table prefixes (count) ===")
for p in sorted(prefixes):
    print(f"{len(prefixes[p]):4d}  {p}")

want = [
    "L1puppiJetSC4NG", "L1SC4NGJetCands", "L1ExtPuppiCand", "L1TExtTrack",
    "L1HGCCluster", "GenJet", "GenVisTau", "GenPart", "L1Vertex", "L1TVertex",
]
print("\n=== tagger tier presence ===")
for p in want:
    br = [k for k in keys if k.startswith(p + "_") or k == "n" + p]
    print(f"{p}: {len(br)} branches")

# GenJet flavor labels
print("\n=== GenJet branches ===")
for k in sorted([k for k in keys if k.startswith("GenJet_")]):
    print("   ", k)

# refit tables (SmartPixels)
print("\n=== SmartPixels refit tables ===")
sp = sorted(set(re.match(r"(L1TSmartPixels\w*?(?:DigiRefit|RefitHitDigiRefit)\w+?)_", k).group(1)
               for k in keys if k.startswith("L1TSmartPixels")))
for p in sp:
    n = len([k for k in keys if k.startswith(p + "_")])
    print(f"   {p}: {n}")

# Vertex branches
print("\n=== Vertex-like branches ===")
for k in sorted([k for k in keys if "ertex" in k or k.startswith("L1Vertex") or k.startswith("L1TVertex")]):
    print("   ", k)

# L1ExtPuppiCand crossref columns
print("\n=== L1ExtPuppiCand columns ===")
for k in sorted([k for k in keys if k.startswith("L1ExtPuppiCand_")]):
    print("   ", k)

# L1TExtTrack columns (for refit crossref + 5-par)
print("\n=== L1TExtTrack columns ===")
for k in sorted([k for k in keys if k.startswith("L1TExtTrack_")]):
    print("   ", k)
