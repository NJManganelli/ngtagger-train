"""Truth-helix consistency: the nano stores tp_phi at the PRODUCTION vertex while
tp_d0 / tp_z0 are at the POCA to the beamline. Recompute the turning angle
delphi (same formulas as L1TrackTruthTableProducer) and test whether moving the
truth phi to the POCA removes the transverse excess of meas - truth.
Linearised: truth_u(phi0 + dphi) = truth_u + (du/dphi) dphi, du/dphi from J."""
import sys, glob, json, numpy as np, uproot, awkward as ak
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200.npz")
files = sorted(glob.glob("/Users/nmangane/smartpixels/cmssw/work/otstub_arm/itot_cov_f*_100ev.root"))
cols = ["L1TTrack_tanL", "L1TTrack_tp_vx", "L1TTrack_tp_vy", "L1TTrack_tp_phi", "L1TTrack_tp_pt", "L1TTrack_tp_charge", "L1TTrack_pt", "L1TTrack_rInv"]
A = uproot.concatenate([f"{f}:Events" for f in files], cols)
N = {c[9:]: ak.to_numpy(ak.flatten(A[c])).astype(float) for c in cols}
t = D["track"]
assert np.mean(np.abs(N["tanL"][t] - D["tanL"]) < 1e-5) > 0.999
kpt = float(np.median(N["pt"] * np.abs(N["rInv"])))      # = c B / 100, the L1 constant
vx, vy, phi, pt, q = (N[k][t] for k in ("tp_vx", "tp_vy", "tp_phi", "tp_pt", "tp_charge"))
r2 = q * kpt / pt / 2.0
x0p = -vx - np.sin(phi) / (2 * r2); y0p = -vy + np.cos(phi) / (2 * r2)
delphi = np.angle(np.exp(1j * (phi - np.arctan2(-r2 * x0p, r2 * y0p))))
rv = np.hypot(vx, vy)
print(f"kpt {kpt:.6f}; TP vertex r: median {np.median(rv)*1e4:.0f} um, mean vx {np.median(vx)*1e4:+.1f} um vy {np.median(vy)*1e4:+.1f} um")
print(f"|delphi| (mrad): median {np.median(np.abs(delphi))*1e3:.3f}, 68% {np.percentile(np.abs(delphi),68)*1e3:.3f}, 95% {np.percentile(np.abs(delphi),95)*1e3:.3f}")
prim = F.primary_mask(D); g = F.floor_group(D)
ok = prim & (g == 0) & (D["pt"] >= 2) & np.isfinite(D["meas"][:, 0]) & np.isfinite(D["truth"][:, 0])
du = (D["meas"][:, 0] - D["truth"][:, 0]) * 1e4
dudphi = D["J"][:, 0, 1] * 1e4
for sgn, lab in ((0, "as stored"), (-1, "phi0 = phi - delphi"), (+1, "phi0 = phi + delphi")):
    corr = du - sgn * dudphi * delphi
    print(f"-- truth {lab}")
    for L in (2, 3, 4):
        s = []
        for lo, hi in ((2, 3), (3, 5), (5, 10), (10, 20), (20, 1e9)):
            m = ok & (D["layer"] == L) & (D["pt"] >= lo) & (D["pt"] < hi)
            s.append(f"[{lo:g},{hi:g}) {F.rs(corr[m])[0]:6.1f}")
        print(f"   IL{L} MAD u (um): " + "  ".join(s))
