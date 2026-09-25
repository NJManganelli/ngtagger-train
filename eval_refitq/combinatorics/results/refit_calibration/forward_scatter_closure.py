"""Fit-free test of the Highland x material x kink-Jacobian code path.
At IL4, meas - truth (truth = the TP production helix, no scattering) is the
PHYSICAL scattering in the beam pipe and IL1..IL3 (plus R, ~2 um). Predict its
variance with the same _kink_param_cov / highland_theta2 / module Jacobian J the
KF model uses, with tkLayout/CMSSW-D121-consistent material, and compare.
Barrel |tanL| < 0.8 only (where the CMSSW scan confirms the material list)."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200.npz")
prim = F.primary_mask(D)
T = json.load(open(base + "R_tables.json")); tr = D["truth"]
Rcal = F.apply_R(T, D["sizeX"], D["sizeY"], tr[:, 2], tr[:, 3], D["sig"])
g = F.floor_group(D)
sel = prim & (D["layer"] == 4) & (g == 0) & (D["pt"] >= 2) & np.isfinite(D["meas"][:, 0]) & np.isfinite(D["truth"][:, 0])
n = int(sel.sum()); tanl = D["tanL"][sel].astype(float); pt = D["pt"][sel].astype(float); p = pt * np.sqrt(1 + tanl**2)
H = D["J"][sel]
PLANES = [(2.25, 0.00227, "beam pipe (0.8 mm Be, assumed)"), (F.IT_R[1], 0.0185, "IL1 module"), (4.17, 0.0009, "HV lines"),
          (F.IT_R[2], 0.0187, "IL2 module"), (7.29, 0.0010, "HV lines"), (F.IT_R[3], 0.0148, "IL3 module"), (11.68, 0.0008, "HV lines")]
print("IT_R (cm):", {k: round(v, 2) for k, v in F.IT_R.items()}, " KINK_COEF:", F.KINK_COEF)
var = {}
for r_s, x, nm in PLANES:
    C = F._kink_param_cov(np.full(n, r_s), tanl, F.highland_theta2(x, p, tanl))
    var[nm + f" r={r_s:.2f}"] = np.einsum("nij,njk,nlk->nil", H[:, :2], C, H[:, :2])
Vs = sum(var.values())
R = Rcal[sel]
d = D["meas"][sel] - D["truth"][sel]
print(f"barrel |tanL|<0.8, IL4, {n} primary clusters")
for k, (lab) in enumerate(("u", "v")):
    print(f"-- {lab}: pT bin | MAD(meas-truth) um | predicted sigma um (scatter, R) | pull MAD | implied x/X0 scale (pull MAD^2)")
    for lo, hi in zip(F.PT_BINS[:-1], F.PT_BINS[1:]):
        m = (pt >= lo) & (pt < hi)
        s = np.sqrt(Vs[m, k, k] + R[m, k] ** 2)
        pm = F.rs(d[m, k] / s)[0]
        print(f"   [{lo:>3g},{hi:>4g}) n {m.sum():5d} | {F.rs(d[m, k])[0]*1e4:7.1f} | {np.median(np.sqrt(Vs[m, k, k]))*1e4:6.1f} {np.median(R[m, k])*1e4:5.1f} | {pm:.2f} | {pm**2:.2f}")
m = (pt >= 2) & (pt < 5)
print("share of predicted u variance per plane, pT 2-5:", {k: round(float(np.median(v[m, 0, 0] / Vs[m, 0, 0])), 3) for k, v in var.items()})
