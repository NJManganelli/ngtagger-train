#!/usr/bin/env python3
"""Leg 3: CMSSW-side true/predicted local angle distributions from the payload analyzer
ntuple, re-derived from data (not the doc table). Compare against source angles under
SWAP / NO-SWAP / SWAP+FLIP mappings."""
import json, os
import numpy as np
import uproot
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUD = "/Users/nmangane/smartpixels/ngtagger-train/eval_spixel_angles/convention_audit"
ev = json.load(open(os.path.join(AUD, "evidence.json")))

f = uproot.open("/Users/nmangane/smartpixels/cmssw/work/spxsmoke/payload_v4fixed_numEvent300.root")
t = f["smartPixelsPayloadAnalyzer/crossings"]
arr = t.arrays(["layer", "trk_cotAlpha", "trk_cotBeta", "b_localy",
                "digi_parCotAlpha", "digi_parCotBeta"], library="np")
lay = arr["layer"]; ca = arr["trk_cotAlpha"]; cb = arr["trk_cotBeta"]; by = arr["b_localy"]

def qs(v):
    q = np.quantile(v, [0.01, 0.5, 0.99])
    return dict(q01=float(q[0]), med=float(q[1]), q99=float(q[2]),
                min=float(v.min()), max=float(v.max()), frac_neg=float(np.mean(v < 0)))

per_layer = {}
for L in [1, 2, 3, 4]:
    m = lay == L
    per_layer[int(L)] = dict(n=int(m.sum()), cotAlpha=qs(ca[m]), cotBeta=qs(cb[m]))

# parent (true) angles, flattened, sentinel-filtered
pca = np.concatenate([np.asarray(x, float) for x in arr["digi_parCotAlpha"]]) if len(arr["digi_parCotAlpha"]) else np.array([])
pcb = np.concatenate([np.asarray(x, float) for x in arr["digi_parCotBeta"]])
good = (pca > -900) & (pcb > -900)
pca, pcb = pca[good], pcb[good]

ev["leg3_cmssw_ranges"] = dict(
    n_crossings=int(len(lay)),
    per_layer_trk=per_layer,
    parent_true=dict(n=int(len(pca)), cotAlpha=qs(pca), cotBeta=qs(pcb)),
    b_localy=dict(min=float(by.min()), max=float(by.max()), mean=float(by.mean()),
                  frac_neg=float(np.mean(by < 0))),
    spec_doc_table_layer1=dict(cotAlpha=[-0.27, 0.52], cotBeta=[-5.0, 5.1]),
)

# source angles (eval dumps) for overlay
tt = pq.read_table("/Users/nmangane/smartpixels/ngtagger-train/2t-Conv1D_Full-2bit_optimized-vars.parquet",
                   columns=["cotAtrue", "cotBtrue"])
sA = np.asarray(tt.column("cotAtrue")); sB = np.asarray(tt.column("cotBtrue"))
# part.93 grid
p93 = pq.read_table("/Users/nmangane/smartpixels/ngtagger-train/part.93.parquet",
                    columns=["cotAlpha", "cotBeta"])
gA = np.asarray(p93.column("cotAlpha")); gB = np.asarray(p93.column("cotBeta"))

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
bins_a = np.linspace(-1.5, 1.5, 121)
axs[0].hist(np.clip(ca, -1.5, 1.5), bins=bins_a, histtype="step", density=True, color="k",
            label="CMSSW trk_cotAlpha (bending)")
axs[0].hist(np.clip(pca, -1.5, 1.5), bins=bins_a, histtype="step", density=True, color="gray", ls=":",
            label="CMSSW parent true cotAlpha")
axs[0].hist(np.clip(sB, -1.5, 1.5), bins=bins_a, histtype="step", density=True, color="C3",
            label="SWAP: source cotBtrue")
axs[0].hist(np.clip(sA, -1.5, 1.5), bins=bins_a, histtype="step", density=True, color="C0", ls="--",
            label="NO-SWAP: source cotAtrue")
axs[0].hist(np.clip(-sB, -1.5, 1.5), bins=bins_a, histtype="step", density=True, color="C2", ls="-.",
            label="SWAP+FLIP: -cotBtrue")
axs[0].set_title("spec cotAlpha axis (payload edges +-0.6)"); axs[0].legend(fontsize=7)
axs[0].axvline(-0.6, color="k", lw=.5); axs[0].axvline(0.6, color="k", lw=.5)
bins_b = np.linspace(-8, 8, 161)
axs[1].hist(np.clip(cb, -8, 8), bins=bins_b, histtype="step", density=True, color="k",
            label="CMSSW trk_cotBeta (non-bending)")
axs[1].hist(np.clip(gA, -8, 8), bins=bins_b, histtype="step", density=True, color="C3",
            label="SWAP: pixelav grid cotAlpha")
axs[1].hist(np.clip(gB, -8, 8), bins=bins_b, histtype="step", density=True, color="C0", ls="--",
            label="NO-SWAP: pixelav grid cotBeta")
axs[1].hist(np.clip(sA, -8, 8), bins=bins_b, histtype="step", density=True, color="C1", ls=":",
            label="SWAP: eval cotAtrue (restricted)")
axs[1].set_yscale("log"); axs[1].set_title("spec cotBeta axis (payload edges +-6)")
axs[1].legend(fontsize=7)
for a in axs: a.grid(alpha=.3)
fig.suptitle("Leg 3: CMSSW-side local angles vs source angles under candidate mappings")
fig.tight_layout()
fig.savefig(os.path.join(AUD, "plots", "leg3_range_overlay.png"), dpi=120)

json.dump(ev, open(os.path.join(AUD, "evidence.json"), "w"), indent=1, default=float)
print("L1 trk cotAlpha q01/med/q99:", per_layer[1]["cotAlpha"])
print("L1 trk cotBeta  q01/med/q99:", per_layer[1]["cotBeta"])
print("parent true alpha:", ev["leg3_cmssw_ranges"]["parent_true"]["cotAlpha"])
print("parent true beta :", ev["leg3_cmssw_ranges"]["parent_true"]["cotBeta"])
print("b_localy:", ev["leg3_cmssw_ranges"]["b_localy"])
EOF_MARKER_NOT_USED = None
