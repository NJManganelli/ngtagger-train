#!/usr/bin/env python3
"""Leg 1 (analytic) + Leg 2 (source geometry) of the convention audit.
Bounded stdout; results -> evidence.json, plots -> plots/.
"""
import json, os
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUD = "/Users/nmangane/smartpixels/ngtagger-train/eval_spixel_angles/convention_audit"
EV = os.path.join(AUD, "evidence.json")

def load_ev():
    if os.path.exists(EV):
        return json.load(open(EV))
    return {}

def save_ev(ev):
    json.dump(ev, open(EV, "w"), indent=1, default=float)

ev = load_ev()

# ---------------- Leg 1: analytic ----------------
R = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)  # x_cms=-y_pix, y_cms=-x_pix, z_cms=-z_pix
det = float(np.linalg.det(R))
# symbolic-by-numeric check of cot transform on random momenta
rng = np.random.default_rng(1)
p = rng.normal(size=(10000, 3)); p[:, 2] = np.sign(p[:, 2]) * (np.abs(p[:, 2]) + 0.1)
pc = p @ R.T
cotA_cms = pc[:, 0] / pc[:, 2]; cotB_cms = pc[:, 1] / pc[:, 2]
cotA_pix = p[:, 0] / p[:, 2]; cotB_pix = p[:, 1] / p[:, 2]
swap_err = float(max(np.abs(cotA_cms - cotB_pix).max(), np.abs(cotB_cms - cotA_pix).max()))
# counterfactual: z NOT flipped (improper reflection z_cms=+z_pix) -> sign flip appears
Rbad = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]], float)
pb = p @ Rbad.T
flip_err = float(np.abs(pb[:, 0] / pb[:, 2] + cotB_pix).max())  # cotA_cms = -cotB_pix under Rbad
ev["leg1_analytic"] = dict(
    det_R=det,
    pure_swap_max_abs_err=swap_err,
    counterfactual_no_zflip_gives_minus_cotB_err=flip_err,
    det_Rbad=float(np.linalg.det(Rbad)),
    conclusion=("det=+1 proper rotation; cotAlpha_cms==+cotBeta_pix and cotBeta_cms==+cotAlpha_pix "
                "exactly (max err ~1e-16). A sign flip only arises from an IMPROPER map (e.g. z not "
                "negated, det=-1). Distinguishers: (i) swap vs no-swap -> which pix angle is WIDE; "
                "(ii) sign flip -> mirror-asymmetry of the bending-angle distribution."),
)

# ---------------- Leg 2: source geometry ----------------
# 2a/2b already in column_stats.json; copy the decisive numbers.
cs = json.load(open(os.path.join(AUD, "column_stats.json")))
p93 = cs["part.93.parquet"]
c1d = cs["2t-Conv1D_Full-2bit_optimized-vars.parquet"]
ev["leg2a_ranges"] = dict(
    part93_cotAlpha_q01_q99=[p93["cotAlpha"]["q01"], p93["cotAlpha"]["q99"]],
    part93_cotBeta_q01_q99=[p93["cotBeta"]["q01"], p93["cotBeta"]["q99"]],
    part93_n_x_q01_q99=[p93["n_x"]["q01"], p93["n_x"]["q99"]],
    part93_n_y_q01_q99=[p93["n_y"]["q01"], p93["n_y"]["q99"]],
    part93_n_z_range=[p93["n_z"]["min"], p93["n_z"]["max"]],
    eval_cotAtrue_q01_q99=[c1d["cotAtrue"]["q01"], c1d["cotAtrue"]["q99"]],
    eval_cotBtrue_q01_q99=[c1d["cotBtrue"]["q01"], c1d["cotBtrue"]["q99"]],
    eval_cotAtrue_median=c1d["cotAtrue"]["med"],
    eval_cotBtrue_median=c1d["cotBtrue"]["med"],
    note=("pixelav cotAlpha is the WIDE axis (q01..q99 +-8.6, flat grid), cotBeta narrower "
          "(+-3.7). n_z strictly negative. In eval dumps cotAtrue is symmetric (med~0), "
          "cotBtrue is ASYMMETRIC (med=-0.176) -> Lorentz-offset bending-plane candidate."),
)
ev["leg2b_pitch"] = dict(
    x_midplane_span=[p93["x-midplane"]["min"], p93["x-midplane"]["max"]],
    y_midplane_span=[p93["y-midplane"]["min"], p93["y-midplane"]["max"]],
    y_local_span=[p93["y-local"]["min"], p93["y-local"]["max"]],
    note=("pix x-midplane spans +-100um (LONG direction), y-midplane +-25um (SHORT). The wide "
          "angle (pix cotAlpha=n_x/n_z) elongates clusters along the LONG pix-x direction, as in "
          "CMSSW local-y (global z). So x_pix<->y_cms, y_pix<->x_cms: consistent with SWAP."),
)

# 2c: NN response structure vs source angles (Conv1D file, scalar cols only)
f = "/Users/nmangane/smartpixels/ngtagger-train/2t-Conv1D_Full-2bit_optimized-vars.parquet"
cols = ["cotAtrue", "cotBtrue", "residuals_cotA", "residuals_cotB", "sigmacotA", "sigmacotB"]
t = pq.read_table(f, columns=cols)
d = {c: np.asarray(t.column(c)) for c in cols}
n = len(d["cotAtrue"])

def profile(xs, ys, edges):
    idx = np.digitize(xs, edges) - 1
    out = []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() < 50:
            out.append((np.nan, np.nan, np.nan, 0)); continue
        v = ys[m]
        q16, q50, q84 = np.quantile(v, [0.16, 0.5, 0.84])
        out.append((q50, (q84 - q16) / 2, np.median(np.abs(v - q50)), int(m.sum())))
    return np.array([o[:3] for o in out]), [o[3] for o in out]

edgesA = np.linspace(-1.1, 1.1, 23)
edgesB = np.linspace(-1.1, 1.1, 23)
cenA = 0.5 * (edgesA[:-1] + edgesA[1:]); cenB = 0.5 * (edgesB[:-1] + edgesB[1:])

# residual (pred-true) = -residuals column
resA = -d["residuals_cotA"]; resB = -d["residuals_cotB"]
pA_vs_A, _ = profile(d["cotAtrue"], resA, edgesA)   # cotA response vs cotAtrue
pB_vs_B, _ = profile(d["cotBtrue"], resB, edgesB)   # cotB response vs cotBtrue
sA_vs_A, _ = profile(d["cotAtrue"], d["sigmacotA"], edgesA)
sB_vs_B, _ = profile(d["cotBtrue"], d["sigmacotB"], edgesB)

fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
axs[0, 0].plot(cenA, pA_vs_A[:, 0], "o-", label="bias (pred-true)")
axs[0, 0].fill_between(cenA, pA_vs_A[:, 0] - pA_vs_A[:, 1], pA_vs_A[:, 0] + pA_vs_A[:, 1], alpha=.3)
axs[0, 0].set_title("source cotA: residual vs cotAtrue"); axs[0, 0].axhline(0, color="k", lw=.5)
axs[0, 1].plot(cenB, pB_vs_B[:, 0], "o-", color="C3")
axs[0, 1].fill_between(cenB, pB_vs_B[:, 0] - pB_vs_B[:, 1], pB_vs_B[:, 0] + pB_vs_B[:, 1], alpha=.3, color="C3")
axs[0, 1].set_title("source cotB: residual vs cotBtrue"); axs[0, 1].axhline(0, color="k", lw=.5)
axs[1, 0].plot(cenA, sA_vs_A[:, 0], "o-"); axs[1, 0].set_title("NN sigmacotA vs cotAtrue")
axs[1, 1].plot(cenB, sB_vs_B[:, 0], "o-", color="C3"); axs[1, 1].set_title("NN sigmacotB vs cotBtrue")
axs[1, 0].set_xlabel("cotAtrue (source units)"); axs[1, 1].set_xlabel("cotBtrue (source units)")
for a in axs.flat:
    a.grid(alpha=.3)
fig.suptitle("Leg 2c: Conv1D_Full-2bit response structure vs SOURCE angles (pix labels)")
fig.tight_layout()
fig.savefig(os.path.join(AUD, "plots", "leg2c_response_vs_source_angles.png"), dpi=120)

# 1D true-angle histograms: asymmetry evidence
fig2, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(d["cotAtrue"], bins=120, histtype="step", color="C0")
ax[0].set_title(f"cotAtrue (med={np.median(d['cotAtrue']):+.4f})")
ax[1].hist(d["cotBtrue"], bins=120, histtype="step", color="C3")
ax[1].set_title(f"cotBtrue (med={np.median(d['cotBtrue']):+.4f})")
for a in ax:
    a.axvline(0, color="k", lw=.5); a.grid(alpha=.3); a.set_yscale("log")
fig2.suptitle("Leg 2: source true-angle distributions (eval sample)")
fig2.tight_layout()
fig2.savefig(os.path.join(AUD, "plots", "leg2_source_angle_hists.png"), dpi=120)

# asymmetry quantification
def asym(v):
    q = np.quantile(v, [0.01, 0.05, 0.5, 0.95, 0.99])
    return dict(q01=float(q[0]), q05=float(q[1]), med=float(q[2]), q95=float(q[3]), q99=float(q[4]),
                mean=float(np.mean(v)), frac_neg=float(np.mean(v < 0)))
ev["leg2c_response"] = dict(
    n_rows=int(n),
    cotAtrue_asym=asym(d["cotAtrue"]),
    cotBtrue_asym=asym(d["cotBtrue"]),
    sigma_ratio_med=float(np.median(d["sigmacotB"]) / np.median(d["sigmacotA"])),
    sigmaA_edges_vs_center=[float(np.nanmean(sA_vs_A[[0, 1, -2, -1], 0])), float(np.nanmean(sA_vs_A[9:13, 0]))],
    sigmaB_edges_vs_center=[float(np.nanmean(sB_vs_B[[0, 1, -2, -1], 0])), float(np.nanmean(sB_vs_B[9:13, 0]))],
)
save_ev(ev)
print("leg1 det=", det, "swap_err=", swap_err)
print("leg2c: cotAtrue med %+0.4f frac_neg %.3f | cotBtrue med %+0.4f frac_neg %.3f" % (
    np.median(d["cotAtrue"]), np.mean(d["cotAtrue"] < 0), np.median(d["cotBtrue"]), np.mean(d["cotBtrue"] < 0)))
print("sigma med A=%.4f B=%.4f" % (np.median(d["sigmacotA"]), np.median(d["sigmacotB"])))
print("wrote evidence legs 1,2 + 2 plots")
