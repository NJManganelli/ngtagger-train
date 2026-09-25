"""Charge-signed test of the transverse excess: split meas - truth (u) at each IT
layer by the track charge (sign of the nano L1TTrack_rInv, joined through the
global track index the extract wrote; join validated on tanL)."""
import sys, glob, numpy as np, uproot, awkward as ak
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200.npz")
files = sorted(glob.glob("/Users/nmangane/smartpixels/cmssw/work/otstub_arm/itot_cov_f*_100ev.root"))
A = uproot.concatenate([f"{f}:Events" for f in files], ["L1TTrack_rInv", "L1TTrack_tanL"])
rinv = ak.to_numpy(ak.flatten(A["L1TTrack_rInv"])).astype(float)
tanl = ak.to_numpy(ak.flatten(A["L1TTrack_tanL"])).astype(float)
t = D["track"]
agree = np.mean(np.abs(tanl[t] - D["tanL"]) < 1e-5) if t.max() < len(tanl) else 0.0
print(f"join check: {len(tanl)} nano tracks, max index {t.max()}, tanL agreement {agree:.4f}")
if agree < 0.999:
    sys.exit("join failed")
q = np.sign(rinv[t])
prim = F.primary_mask(D); g = F.floor_group(D)
ok = prim & (g == 0) & (D["pt"] >= 2) & np.isfinite(D["meas"][:, 0]) & np.isfinite(D["truth"][:, 0])
du = (D["meas"][:, 0] - D["truth"][:, 0]) * 1e4
for L in (1, 2, 3, 4):
    print(f"== IL{L} (barrel |tanL|<0.8): median u residual by charge, and spread")
    for lo, hi in ((2, 3), (3, 5), (5, 10), (10, 20), (20, 1e9)):
        m = ok & (D["layer"] == L) & (D["pt"] >= lo) & (D["pt"] < hi)
        mp, mn = m & (q > 0), m & (q < 0)
        medp, medn = np.median(du[mp]), np.median(du[mn])
        resid = np.where(q[m] > 0, du[m] - medp, du[m] - medn)
        print(f"   pT [{lo:>2g},{hi:>4g}) n {m.sum():6d}  median q+ {medp:+7.2f} um  q- {medn:+7.2f} um  | MAD all {F.rs(du[m])[0]:6.1f}  within-charge {F.rs(resid)[0]:6.1f} um")
