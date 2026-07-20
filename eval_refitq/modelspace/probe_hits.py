"""Data-shape probe: crossings per (track,layer), accepted-hit uniqueness,
per-layer genuine/fake raw separation of pulls; feeds the per-layer feature
study design (Part A)."""
import numpy as np
import awkward as ak

from ngtagger.train.refitquality import load_refit_tables, _SENTINEL

NANO = "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/nano_pu100_TrkSmartPix_withGen.root"

ref, var, hits = load_refit_tables([NANO], "AAAA")

counts = ak.to_numpy(ak.num(ref["genuine"]))
offsets = np.concatenate([[0], np.cumsum(counts)])
n_tracks = int(offsets[-1])

h = {b: ak.to_numpy(ak.flatten(hits[b])) for b in hits.fields}
ev_of_hit = np.repeat(np.arange(len(counts)), ak.to_numpy(ak.num(hits["trackIdx"])))
gidx = h["trackIdx"].astype(np.int64) + offsets[ev_of_hit]

genuine = ak.to_numpy(ak.flatten(ref["genuine"])).astype(bool)
print(f"tracks={n_tracks} genuine={genuine.sum()} fake={(~genuine).sum()}")
print(f"hit rows={len(gidx)}")

# crossings per (track, layer)
key = gidx * 8 + h["layer"].astype(np.int64)
uniq, cnt = np.unique(key, return_counts=True)
print("crossings per (track,layer): ", dict(zip(*np.unique(cnt, return_counts=True))))

import uproot
h["hitAccepted"] = ak.to_numpy(ak.flatten(
    uproot.open(f"{NANO}:Events")["L1TSmartPixelsRefitHitDigiRefitAAAA_hitAccepted"].array()))
acc = h["hitAccepted"].astype(bool)
key_acc = key[acc]
uniq_a, cnt_a = np.unique(key_acc, return_counts=True)
print("ACCEPTED hits per (track,layer):", dict(zip(*np.unique(cnt_a, return_counts=True))))

print("\nlayer occupancy of hit rows:", dict(zip(*np.unique(h["layer"], return_counts=True))))

# per-layer raw separation: |pullX| and windowMult for genuine vs fake tracks
from sklearn.metrics import roc_auc_score
lab_of_hit = genuine[gidx]
for L in (1, 2, 3, 4):
    mL = (h["layer"] == L) & acc & (h["pullX"] > _SENTINEL)
    if mL.sum() < 50:
        continue
    for col in ("pullX", "pullY", "pullAlpha", "pullBeta"):
        v = h[col][mL]
        ok = v > _SENTINEL
        if ok.sum() < 50:
            continue
        yl = lab_of_hit[mL][ok]
        if yl.all() or (~yl).all():
            continue
        a = roc_auc_score(~yl, np.abs(v[ok]))  # fake-positive AUC on |pull|
        print(f"L{L} |{col}|: hit-level fake-AUC={a:.3f} (n={ok.sum()}, nfakehits={(~yl).sum()})")

# window multiplicity per layer, genuine vs fake (first crossing per track-layer)
for L in (1, 2, 3, 4):
    mL = h["layer"] == L
    yl = lab_of_hit[mL]
    v = h["windowMult"][mL].astype(float)
    if (~yl).sum() > 20:
        a = roc_auc_score(~yl, v)
        print(f"L{L} windowMult: hit-level fake-AUC={a:.3f} "
              f"(gen mean {v[yl].mean():.1f}, fake mean {v[~yl].mean():.1f})")

# chi2 hit-level tail
for col in ("chi2IncRPhi", "chi2IncRZ"):
    v = h[col]
    okv = v > _SENTINEL
    print(f"{col}: max={v[okv].max():.2e} n>2e6={(v[okv] > 2e6).sum()}")
