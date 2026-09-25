"""OT seed covariance closure against truth AT THE POCA (tp_phi moved from the
production vertex to the POCA with the truth table's own delphi). Genuine tracks,
TP pT > 2, barrel |tanL| < 0.8. Per parameter and for the transverse position at
r = 14.4 cm (IL4) and 3 cm: u(r) ~ r dphi0 - d0 - r^2/2 drInv (TTTrack-native)."""
import sys, glob, numpy as np, uproot, awkward as ak
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
files = sorted(glob.glob("/Users/nmangane/smartpixels/cmssw/work/otstub_arm/itot_cov_f*_100ev.root"))
HP = ["rInv", "phi", "tanL", "z0", "d0"]
cov = [f"cov_{HP[i]}_{HP[j]}" for i in range(5) for j in range(i, 5)]
cols = HP + ["pt", "genuine", "tp_pt", "tp_phi", "tp_tanL", "tp_z0", "tp_d0", "tp_charge", "tp_vx", "tp_vy"] + cov
A = uproot.concatenate([f"{f}:Events" for f in files], [f"L1TTrack_{c}" for c in cols])
N = {c: ak.to_numpy(ak.flatten(A[f"L1TTrack_{c}"])).astype(float) for c in cols}
m = (N["genuine"] > 0) & (N["tp_pt"] > 2) & (np.abs(N["tanL"]) < 0.8)
N = {k: v[m] for k, v in N.items()}
kpt = float(np.median(N["pt"] * np.abs(N["rInv"])))
r2 = N["tp_charge"] * kpt / N["tp_pt"] / 2
x0p = -N["tp_vx"] - np.sin(N["tp_phi"]) / (2 * r2); y0p = -N["tp_vy"] + np.cos(N["tp_phi"]) / (2 * r2)
phi0 = np.arctan2(-r2 * x0p, r2 * y0p)            # truth phi at the POCA
tru = np.stack([N["tp_charge"] * kpt / N["tp_pt"], phi0, N["tp_tanL"], N["tp_z0"], N["tp_d0"]], 1)
trk = np.stack([N[p] for p in HP], 1)
d = trk - tru; d[:, 1] = np.angle(np.exp(1j * d[:, 1]))
C = np.zeros((len(d), 5, 5)); k = 0
for i in range(5):
    for j in range(i, 5):
        C[:, i, j] = C[:, j, i] = N[cov[k]]; k += 1
pt = N["tp_pt"]
print(f"{len(d)} genuine barrel tracks; kpt {kpt:.6f}")
for lo, hi in ((2, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)):
    s = (pt >= lo) & (pt < hi)
    row = " ".join(f"{p}:{F.rs(d[s, i] / np.sqrt(C[s, i, i]))[0]:5.2f}" for i, p in enumerate(HP))
    uu = []
    for r in (3.0, 14.4):
        h = np.array([-r * r / 2, r, 0, 0, -1.0])
        su = np.sqrt(np.einsum("j,njk,k->n", h, C[s], h))
        du = d[s] @ h
        uu.append(f"u@{r:4.1f}cm pull {F.rs(du / su)[0]:.2f} (obs {F.rs(du)[0]*1e4:5.0f} um, cov {np.median(su)*1e4:5.0f} um)")
    print(f"pT [{lo:>2g},{hi:>4g}) n {s.sum():6d} | pull MAD {row} | " + " | ".join(uu))
