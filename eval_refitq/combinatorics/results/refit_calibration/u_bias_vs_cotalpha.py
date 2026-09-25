"""Is the transverse (u) excess of meas - truth a COHERENT bias (depends on the
TP's local cotAlpha, i.e. on charge/pT and radius -- CPE/Lorentz-drift or field
type) rather than extra random spread? Barrel |tanL| < 0.8, primary clusters."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200.npz")
prim = F.primary_mask(D); g = F.floor_group(D)
ok = prim & (g == 0) & (D["pt"] >= 2) & np.isfinite(D["meas"][:, 0]) & np.isfinite(D["truth"][:, 0])
du = (D["meas"][:, 0] - D["truth"][:, 0]) * 1e4
ca = D["cltruth"][:, 0]
EDG = [-1.0, -0.3, -0.15, -0.08, -0.04, 0.0, 0.04, 0.08, 0.15, 0.3, 1.0]
for L in (1, 2, 3, 4):
    m0 = ok & (D["layer"] == L)
    print(f"== IL{L}: n {m0.sum()}   median u residual (um) [n] by TP local cotAlpha bin")
    ib = np.digitize(ca, EDG) - 1
    print("   cotA bins:", " ".join(f"[{a:+.2f},{b:+.2f})" for a, b in zip(EDG[:-1], EDG[1:])))
    print("   all pT   :", " ".join(f"{np.median(du[m0 & (ib == k)]):+7.1f}[{(m0 & (ib == k)).sum()}]" if (m0 & (ib == k)).sum() > 30 else "     --" for k in range(len(EDG) - 1)))
    for lo, hi in ((2, 5), (5, 10), (10, 1e9)):
        m = m0 & (D["pt"] >= lo) & (D["pt"] < hi)
        # remove the cotAlpha-binned median: what spread is left?
        resid = du[m].copy()
        for k in range(len(EDG) - 1):
            mk = ib[m] == k
            if mk.sum() > 30:
                resid[mk] -= np.median(du[m][mk])
        print(f"   pT [{lo:>2g},{hi:>4g}) n {m.sum():6d}  MAD u {F.rs(du[m])[0]:6.1f} um -> after removing cotA-binned median {F.rs(resid)[0]:6.1f} um;"
              f"  |cotA| median {np.median(np.abs(ca[m])):.3f}")
