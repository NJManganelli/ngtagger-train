#!/usr/bin/env python3
"""Leg 5a (sign test): CMSSW cluster x-extent vs trk_cotAlpha dip position (Lorentz
compensation angle, signed) vs the pixelav-side proxy (NN sigmacotB peak at cotBtrue
~ -0.15). PURE SWAP (values unchanged) predicts the same signed dip; a mirrored
response predicts the opposite sign."""
import json, os
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUD = "/Users/nmangane/smartpixels/ngtagger-train/eval_spixel_angles/convention_audit"
ev = json.load(open(os.path.join(AUD, "evidence.json")))

f = uproot.open("/Users/nmangane/smartpixels/cmssw/work/spxsmoke/payload_v4fixed_numEvent300.root")
t = f["smartPixelsPayloadAnalyzer/crossings"]
a = t.arrays(["trk_cotAlpha", "trk_cotBeta", "digi_class", "digi_localx", "digi_localy"],
             library="np")
ca = a["trk_cotAlpha"]; cb = a["trk_cotBeta"]

nx, ny, wa, wb, nd = [], [], [], [], []
for i in range(len(ca)):
    cls = np.asarray(a["digi_class"][i])
    m = cls == 0
    if m.sum() == 0:
        continue
    lx = np.asarray(a["digi_localx"][i], float)[m]
    ly = np.asarray(a["digi_localy"][i], float)[m]
    nx.append(lx.max() - lx.min()); ny.append(ly.max() - ly.min())
    wa.append(ca[i]); wb.append(cb[i]); nd.append(m.sum())
nx = np.array(nx) * 1e4  # cm -> um
ny = np.array(ny) * 1e4
wa = np.array(wa); wb = np.array(wb); nd = np.array(nd, float)

def prof(x, y, edges, minn=100):
    idx = np.digitize(x, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    cnt = np.zeros(len(edges) - 1, int)
    for i in range(len(edges) - 1):
        m = idx == i
        cnt[i] = m.sum()
        if cnt[i] >= minn:
            out[i] = np.mean(y[m])
    return out, cnt

# central beta to isolate alpha-driven x extent
mB = np.abs(wb) < 1.5
eA = np.linspace(-0.35, 0.55, 37); cA = 0.5 * (eA[:-1] + eA[1:])
xa, ca_cnt = prof(wa[mB], nx[mB], eA)
sza, _ = prof(wa[mB], nd[mB], eA)
eB = np.linspace(-6, 6, 61); cB = 0.5 * (eB[:-1] + eB[1:])
yb, _ = prof(wb, ny, eB)

# dip locations (parabola fit around min)
def dipfit(c, v, halfwin=6):
    good = np.isfinite(v)
    i0 = np.nanargmin(v)
    lo, hi = max(0, i0 - halfwin), min(len(c), i0 + halfwin + 1)
    cc, vv = c[lo:hi], v[lo:hi]
    g = np.isfinite(vv)
    p = np.polyfit(cc[g], vv[g], 2)
    return float(-p[1] / (2 * p[0])), float(c[i0])

dip_alpha_fit, dip_alpha_bin = dipfit(cA, xa)
dip_beta_fit, dip_beta_bin = dipfit(cB, yb, 8)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
axs[0].plot(cA, xa, "o-")
axs[0].axvline(dip_alpha_fit, color="C3", ls="--", label=f"dip fit {dip_alpha_fit:+.3f}")
axs[0].axvline(-0.15, color="C2", ls=":", label="pixelav proxy (cotB peak) -0.15")
axs[0].axvline(+0.15, color="gray", ls=":", label="mirrored +0.15")
axs[0].set_xlabel("trk_cotAlpha (CMSSW local)"); axs[0].set_ylabel("mean cluster x-extent [um]")
axs[0].set_title("CMSSW: x-extent vs cotAlpha (|cotBeta|<1.5)"); axs[0].legend(fontsize=8)
axs[1].plot(cB, yb, "o-")
axs[1].axvline(dip_beta_fit, color="C3", ls="--", label=f"dip fit {dip_beta_fit:+.3f}")
axs[1].set_xlabel("trk_cotBeta"); axs[1].set_ylabel("mean cluster y-extent [um]")
axs[1].set_title("control: y-extent vs cotBeta (dip ~0 expected)"); axs[1].legend(fontsize=8)
for ax in axs: ax.grid(alpha=.3)
fig.suptitle("Leg 5a: Lorentz-compensation dip sign test")
fig.tight_layout()
fig.savefig(os.path.join(AUD, "plots", "leg5a_lorentz_dip_sign.png"), dpi=120)

ev["leg5a_lorentz_sign"] = dict(
    n_crossings_used=int(len(wa)),
    cms_xextent_dip_cotAlpha=dict(fit=dip_alpha_fit, argmin_bin=dip_alpha_bin),
    cms_yextent_dip_cotBeta=dict(fit=dip_beta_fit, argmin_bin=dip_beta_bin),
    pixelav_proxy_sigmacotB_peak=-0.15,
    prediction=dict(pure_swap="cms alpha dip at ~-0.15 (same sign as pixelav cotB proxy)",
                    mirrored="cms alpha dip at ~+0.15"),
)
json.dump(ev, open(os.path.join(AUD, "evidence.json"), "w"), indent=1, default=float)
print("crossings with class-0 digis:", len(wa))
print("CMSSW x-extent dip at cotAlpha = %+0.4f (fit), argmin bin %+0.3f" % (dip_alpha_fit, dip_alpha_bin))
print("control y-extent dip at cotBeta = %+0.4f" % dip_beta_fit)
