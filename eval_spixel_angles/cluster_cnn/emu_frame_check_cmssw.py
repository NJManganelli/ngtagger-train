"""Frame check of the CMSSW emu output: is the emitted cotAlpha/cotBeta in the refit projector's convention
(SmartPixelsHelixProjector: lmom = det.toLocal(globalMom); cotAlpha = lmom.x/lmom.z, cotBeta = lmom.y/lmom.z)?

  emu_frame_check_cmssw.py NTUPLE.root [NTUPLE2.root ...]    (ClusterAdcNtuple, angleSource=emu)

As v2_emu_frame_check.py, but on the CMSSW output (pCotAlpha/pCotBeta): an independent python helix projection of the
dominant TP (vertex, momentum, charge; uniform Bz = 3.8112 T) to the cluster's global (x, y), rotated with the module
axes; for CMSSW class-T clusters with hxDist < 1 mm and pT >= 0.5, per surface group x orientation (barrel zOutward,
disc axZz) x side:
  projected - tpLocalCot (the producer truth)                      -> ~0 (frame identity)
  innovation = emitted - projected [deg of atan]                   -> centred (median = the response bias)
Output: results/emu_frame_check_cmssw.csv, figs/emu_frame_check_cmssw.pdf."""
import os, sys, numpy as np, pandas as pd, uproot, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_common import deg, smad

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["subdet", "layer", "disk", "globalX", "globalY", "globalZ", "axXx", "axXy", "axXz", "axYx", "axYy", "axYz", "axZx", "axZy",
        "axZz", "zOutward", "tpVx", "tpVy", "tpPx", "tpPy", "tpPz", "tpCharge", "tpPt", "hxDist", "emuClass", "tpLocalCotAlpha",
        "tpLocalCotBeta", "pCotAlpha", "pCotBeta", "pSatAlpha", "pSatBeta"]
m = pd.concat([uproot.open(f)["clusterAdcNtuple/clusters"].arrays(COLS, library="pd") for f in sys.argv[1:]], ignore_index=True)
m = m[(m.emuClass == 1) & (m.hxDist < 0.1) & (m.tpPt >= 0.5)].reset_index(drop=True)
BZ = 3.8112


def project(m, sense):
    pt = np.hypot(m.tpPx, m.tpPy).values; q = np.sign(m.tpCharge.values)
    R = pt / (0.299792458 * BZ) * 100.0
    ux, uy = m.tpPx.values / pt, m.tpPy.values / pt
    cx = m.tpVx.values + sense * q * R * uy; cy = m.tpVy.values - sense * q * R * ux
    dx, dy = m.globalX.values - cx, m.globalY.values - cy; dn = np.hypot(dx, dy); dx, dy = dx / dn, dy / dn
    vx, vy = sense * q * dy, -sense * q * dx
    px, py, pz = vx * pt, vy * pt, m.tpPz.values
    lx = px * m.axXx + py * m.axXy + pz * m.axXz; ly = px * m.axYx + py * m.axYy + pz * m.axYz; lz = px * m.axZx + py * m.axZy + pz * m.axZz
    return (lx / lz).values, (ly / lz).values


best = min(((s,) + project(m, s) for s in (1, -1)), key=lambda b: np.median(np.abs(deg(b[1]) - deg(m.tpLocalCotAlpha.values))))
sense, pA, pB = best
grp = np.where(m.subdet == 2, "ID1-4", np.where(m.layer == 1, "IL1", "IL2-4"))
orient = np.where(m.subdet == 1, m.zOutward.values, np.round(m.axZz.values)).astype(int)
side = np.where(m.globalZ.values > 0, 1, -1)
cells = pd.DataFrame({"surface": grp, "orient": orient, "side": side})
rows = []
for key, g in cells.groupby(["surface", "orient", "side"]).groups.items():
    ii = cells.index.get_indexer(g)
    for ang, pr, tr, em, st in (("alpha", pA, m.tpLocalCotAlpha.values, m.pCotAlpha.values, m.pSatAlpha.values),
                                ("beta", pB, m.tpLocalCotBeta.values, m.pCotBeta.values, m.pSatBeta.values)):
        p, t, e_ = pr[ii], tr[ii], em[ii]
        inn = deg(e_) - deg(p); big = np.abs(deg(p)) > 10
        rows.append(dict(surface=key[0], orient=int(key[1]), side=int(key[2]), angle=ang, n=len(ii),
                         proj_minus_truth_median_deg=float(np.median(deg(p) - deg(t))), proj_minus_truth_max_abs_deg=float(np.max(np.abs(deg(p) - deg(t)))),
                         innovation_median_deg=float(np.median(inn)), innovation_smad_deg=smad(inn),
                         sign_agree_big=float(np.mean(np.sign(e_[big]) == np.sign(p[big]))) if big.any() else np.nan,
                         saturated=int(st[ii].sum())))
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, "results", "emu_frame_check_cmssw.csv"), index=False)
pd.set_option("display.width", 250); print("helix sense", sense); print(T.round(4).to_string(index=False))
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), layout="constrained")
e = np.linspace(-40, 40, 161); PAL = ["#2a78d6", "#e8742c", "#1f9e76", "#b04a9c"]
labels = {}
for i, (ang, pr, em) in enumerate((("alpha", pA, m.pCotAlpha.values), ("beta", pB, m.pCotBeta.values))):
    for j, sname in enumerate(["IL2-4", "IL1", "ID1-4"]):
        ax = axes[i, j]
        for (orv, sd), g in cells[cells.surface == sname].groupby(["orient", "side"]).groups.items():
            ii = cells.index.get_indexer(g); r = deg(em[ii]) - deg(pr[ii])
            k = {(1, 1): 0, (1, -1): 1, (-1, 1): 2, (-1, -1): 3}[(int(orv), int(sd))]
            h, _ = np.histogram(np.clip(r, e[0] + 1e-6, e[-1] - 1e-6), bins=e)
            ax.stairs(h / max(h.sum(), 1) / np.diff(e), e, color=PAL[k], lw=1.3, label=f"orientation {int(orv):+d}, side {int(sd):+d}")
        ax.set_yscale("log"); ax.set_ylim(1e-6, 1); ax.grid(True, color="#eeeeee")
        ax.set_title(f"{sname} cot{'Alpha' if ang == 'alpha' else 'Beta'}: emitted (CMSSW emu, class T) - projected", fontsize=9)
        ax.set_xlabel("atan(emitted) - atan(projected) [deg]")
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            labels.setdefault(l_, h_)
fig.legend(list(labels.values()), list(labels.keys()), loc="outside lower center", ncol=4, fontsize=9, frameon=False)
fig.suptitle("Frame check of the CMSSW emu output against an independent helix projection (projector convention)", fontsize=10)
with PdfPages(os.path.join(HERE, "figs", "emu_frame_check_cmssw.pdf")) as pdf:
    pdf.savefig(fig); plt.close(fig)
    for ang in ("alpha", "beta"):
        tt = T[T.angle == ang].drop(columns="angle").round(4)
        f2, a2 = plt.subplots(figsize=(18, 0.34 * len(tt) + 1.4)); a2.axis("off")
        tb = a2.table(cellText=tt.values, colLabels=[c.replace("_", " ") for c in tt.columns], loc="center", cellLoc="left")
        tb.auto_set_font_size(False); tb.set_fontsize(8); a2.set_title(f"Frame check statistics - cot{ang}", fontsize=10)
        pdf.savefig(f2, bbox_inches="tight"); plt.close(f2)
