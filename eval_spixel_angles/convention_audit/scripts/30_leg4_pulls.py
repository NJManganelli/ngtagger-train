#!/usr/bin/env python3
"""Leg 4: end-to-end refit sidecar. Caveat derived from producer code: measured angles are
SYNTHESIZED with the payload's own sigma, so pull width ~1 is internal consistency only.
The discriminating observable is WHICH parent angle the deployed sigma varies with:
 - correct SWAP: sigAlpha varies along parCotAlpha (bending, fine axis) with the shape of
   the NN's sigma-vs-cotB profile; sigBeta varies coarsely, saturating for |parCotBeta|>~1
   (source coverage clamp).
 - transposed mapping would key sigAlpha to parCotBeta instead."""
import json, os
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUD = "/Users/nmangane/smartpixels/ngtagger-train/eval_spixel_angles/convention_audit"
ev = json.load(open(os.path.join(AUD, "evidence.json")))

f = uproot.open("/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/nano_fat_1111_coopt_file1.root")
t = f["Events"]
pre = "L1TSmartPixelsExtRefitHitDigiRefitAAAA_"
cols = ["hasAlpha", "hasBeta", "sigAlpha", "sigBeta", "pullAlpha", "pullBeta",
        "parCotAlpha", "parCotBeta", "cotAlphaMeas", "cotBetaMeas"]
a = t.arrays([pre + c for c in cols], library="np")
d = {c: np.concatenate([np.asarray(x, float) for x in a[pre + c]]) for c in cols}
n_all = len(d["sigAlpha"])
ok = (d["parCotAlpha"] > -900) & (d["parCotBeta"] > -900)
hasA = ok & (d["hasAlpha"] > 0); hasB = ok & (d["hasBeta"] > 0)

def rob(v):
    if len(v) < 20: return (np.nan, np.nan)
    q16, q50, q84 = np.quantile(v, [0.16, 0.5, 0.84])
    return (float(q50), float((q84 - q16) / 2))

pullA = d["pullAlpha"][hasA & (np.abs(d["pullAlpha"]) < 50)]
pullB = d["pullBeta"][hasB & (np.abs(d["pullBeta"]) < 50)]
mA, wA = rob(pullA); mB, wB = rob(pullB)

# sigma variation vs each parent angle
def prof(x, y, edges):
    idx = np.digitize(x, edges) - 1
    med = []
    for i in range(len(edges) - 1):
        m = idx == i
        med.append(np.median(y[m]) if m.sum() >= 20 else np.nan)
    return np.array(med)

eA = np.linspace(-1.0, 1.0, 21); cAc = 0.5 * (eA[:-1] + eA[1:])
eB = np.linspace(-6, 6, 25); cBc = 0.5 * (eB[:-1] + eB[1:])
sa_vs_a = prof(d["parCotAlpha"][hasA], d["sigAlpha"][hasA], eA)
sa_vs_b = prof(d["parCotBeta"][hasA], d["sigAlpha"][hasA], eB)
sb_vs_a = prof(d["parCotAlpha"][hasB], d["sigBeta"][hasB], eA)
sb_vs_b = prof(d["parCotBeta"][hasB], d["sigBeta"][hasB], eB)

# pull width vs |parCotBeta| (blow-up test if axes transposed)
pw = []
bedges = [0, 0.5, 1, 2, 3, 4, 6]
for i in range(len(bedges) - 1):
    m = hasB & (np.abs(d["parCotBeta"]) >= bedges[i]) & (np.abs(d["parCotBeta"]) < bedges[i + 1]) \
        & (np.abs(d["pullBeta"]) < 50)
    pw.append(rob(d["pullBeta"][m])[1])

fig, axs = plt.subplots(2, 2, figsize=(11, 7))
axs[0, 0].hist(np.clip(pullA, -6, 6), bins=80, histtype="step", label=f"pullAlpha w={wA:.2f}")
axs[0, 0].hist(np.clip(pullB, -6, 6), bins=80, histtype="step", label=f"pullBeta w={wB:.2f}")
axs[0, 0].set_yscale("log"); axs[0, 0].legend(); axs[0, 0].set_title("angle pulls (synthesis-consistency)")
axs[0, 1].plot(bedges[:-1], pw, "o-"); axs[0, 1].set_xlabel("|parCotBeta| bin lo")
axs[0, 1].set_ylabel("robust pullBeta width"); axs[0, 1].axhline(1, color="k", lw=.5)
axs[0, 1].set_title("pullBeta width vs |cotBeta| (blow-up test)")
axs[1, 0].plot(cAc, sa_vs_a, "o-", label="sigAlpha vs parCotAlpha")
axs[1, 0].plot(cBc / 6, sa_vs_b, "s--", label="sigAlpha vs parCotBeta (x/6)")
axs[1, 0].legend(fontsize=8); axs[1, 0].set_title("which axis does sigAlpha vary with?")
axs[1, 1].plot(cAc, sb_vs_a, "o-", label="sigBeta vs parCotAlpha")
axs[1, 1].plot(cBc / 6, sb_vs_b, "s--", label="sigBeta vs parCotBeta (x/6)")
axs[1, 1].legend(fontsize=8); axs[1, 1].set_title("which axis does sigBeta vary with?")
for ax in axs.flat: ax.grid(alpha=.3)
fig.suptitle("Leg 4: deployed-refit sidecar (nano_fat_1111_coopt_file1)")
fig.tight_layout()
fig.savefig(os.path.join(AUD, "plots", "leg4_pulls_and_sigma_keying.png"), dpi=120)

def rng(v):
    v = v[np.isfinite(v)]
    return [float(v.min()), float(v.max())] if len(v) else [None, None]

ev["leg4_pulls"] = dict(
    n_hits=int(n_all), n_hasAlpha=int(hasA.sum()), n_hasBeta=int(hasB.sum()),
    pullAlpha_med_width=[mA, wA], pullBeta_med_width=[mB, wB],
    pullBeta_width_vs_absCotBeta=dict(zip([f"{bedges[i]}-{bedges[i+1]}" for i in range(len(bedges)-1)],
                                          [float(x) if np.isfinite(x) else None for x in pw])),
    sigAlpha_range_vs_parCotAlpha=rng(sa_vs_a), sigAlpha_range_vs_parCotBeta=rng(sa_vs_b),
    sigBeta_range_vs_parCotAlpha=rng(sb_vs_a), sigBeta_range_vs_parCotBeta=rng(sb_vs_b),
    sigAlpha_med=float(np.median(d["sigAlpha"][hasA])), sigBeta_med=float(np.median(d["sigBeta"][hasB])),
    caveat="measured angles synthesized with payload sigma -> pull width ~1 by construction; "
           "keying (variation axis) is the discriminating observable.",
)
json.dump(ev, open(os.path.join(AUD, "evidence.json"), "w"), indent=1, default=float)
print(f"hits={n_all} hasA={hasA.sum()} hasB={hasB.sum()}")
print(f"pullAlpha med/w = {mA:.3f}/{wA:.3f}  pullBeta med/w = {mB:.3f}/{wB:.3f}")
print("pullBeta width per |cotB| bin:", [None if not np.isfinite(x) else round(x, 3) for x in pw])
print("sigAlpha med=%.4f range-vs-alpha %s range-vs-beta %s" % (np.median(d['sigAlpha'][hasA]), rng(sa_vs_a), rng(sa_vs_b)))
print("sigBeta  med=%.4f range-vs-alpha %s range-vs-beta %s" % (np.median(d['sigBeta'][hasB]), rng(sb_vs_a), rng(sb_vs_b)))
