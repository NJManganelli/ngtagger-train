"""IL4 u innovation pull (first layer, no update) vs the TP's true |d0|: does the
constant ~60 um high-pT excess come from displaced TPs fitted by the 4-parameter
(d0 = 0) OT fit?"""
import sys
sys.argv = [sys.argv[0]]
exec(open("/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/il4_decompose.py").read().split("um = 1e4")[0])
d0 = np.abs(D["d0"][sel].astype(float)) * 1e4
print("OT covariance: median sqrt(C_d0d0) =", f"{np.median(np.sqrt(np.maximum(C0[:,4,4],0)))*1e4:.2f} um;",
      "fraction with C_d0d0 == 0:", f"{np.mean(C0[:,4,4]==0):.3f}")
print("TP |d0| quantiles (um) 50/68/90/95/99:", np.round(np.percentile(d0, [50, 68, 90, 95, 99]), 1))
Sd = s_ot + s_fl + s_mat + R**2
for lab, pm in (("pT 2-5 GeV", (pt >= 2) & (pt < 5)), ("pT >= 5 GeV", pt >= 5)):
    print(f"-- {lab}")
    for lo, hi in ((0, 10), (10, 20), (20, 50), (50, 100), (100, 300), (300, 1e9)):
        m = pm & (d0 >= lo) & (d0 < hi)
        if m.sum() < 20: continue
        x = res[m] / np.sqrt(Sd[m])
        print(f"   |d0| [{lo:>4g},{hi:>4g}) um  n {m.sum():6d}  pull MAD {F.rs(x)[0]:.2f}  RMS {F.rs(x)[1]:5.2f}  |pull|>5 {np.mean(np.abs(x)>5):.3f}"
              f"  median sigma_u {np.median(np.sqrt(Sd[m]))*1e4:4.0f} um  MAD(res) {F.rs(res[m])[0]*1e4:4.0f} um")
