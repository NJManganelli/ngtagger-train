"""Test 1: linearisation. The KF predicts every residual as H (a - a_seed) with H the
module Jacobian AT THE SEED. Exact counterpart: the truth helix projected onto the
same module (npz 'truth', nonlinear, same projection code) minus the seed
projection (meas - res). The mismatch  (truth - proj_seed) - H (a_true - a_seed)
is compared with the innovation width at material x1 / empirical seed (v10), per
layer and pT. a_true: nano tp_* at the POCA (tp_phi0), rInv = q KPT/pT."""
import sys, glob, json, numpy as np, uproot, awkward as ak
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
from ngtagger.truth_helix import KPT_CMSSW
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200_tp.npz")
prim = F.primary_mask(D)
files = sorted(glob.glob("/Users/nmangane/smartpixels/cmssw/work/otstub_arm/itot_tp_f*_100ev.root"))
HP = ["rInv", "phi", "tanL", "z0", "d0"]
cols = HP + ["tp_pt", "tp_phi0", "tp_tanL", "tp_z0", "tp_d0", "tp_charge"]
A = uproot.concatenate([f"{f}:Events" for f in files], [f"L1TTrack_{c}" for c in cols])
N = {c: ak.to_numpy(ak.flatten(A[f"L1TTrack_{c}"])).astype(float) for c in cols}
t = D["track"]
assert np.mean(np.abs(N["tanL"][t] - D["tanL"]) < 1e-5) > 0.999
da = np.stack([N["tp_charge"] * KPT_CMSSW / N["tp_pt"], N["tp_phi0"], N["tp_tanL"], N["tp_z0"], N["tp_d0"]], 1)[t] \
     - np.stack([N[p] for p in HP], 1)[t]
da[:, 1] = np.angle(np.exp(1j * da[:, 1]))
proj_seed = D["meas"] - D["res"]
exact = D["truth"] - proj_seed
lin = np.einsum("nij,nj->ni", D["J"], da)
mis = exact - lin
S = np.sqrt(np.einsum("nij,njk,nlk->nil", D["J"], D["C"], D["J"])[:, np.arange(4), np.arange(4)])   # seed-only width
ok = prim & (D["pt"] >= 2) & np.all(np.isfinite(exact[:, :2]), 1)
print("mismatch = exact nonlinear (truth - seed projection) - H (a_true - a_seed); compared with the seed-only width sqrt(H C H^T)")
for L in (4, 3, 2, 1):
    for lo, hi in ((2, 3), (3, 5), (5, 10), (10, 1e9)):
        m = ok & (D["layer"] == L) & (D["pt"] >= lo) & (D["pt"] < hi)
        row = f"IL{L} pT [{lo:>2g},{hi:>4g}) n {m.sum():6d}"
        for k, nm in ((0, "u"), (1, "v")):
            row += (f" | {nm}: MAD exact {F.rs(exact[m, k])[0]*1e4:7.1f} um, MAD mismatch {F.rs(mis[m, k])[0]*1e4:6.2f} um, "
                    f"|mis| 99% {np.percentile(np.abs(mis[m, k]), 99)*1e4:7.1f} um, mismatch/width MAD {F.rs(mis[m, k] / S[m, k])[0]:.3f}")
        print(row)
