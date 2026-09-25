"""forward_scatter_closure.py at every IT layer: observed meas - truth spread vs
the Highland x material x kink-Jacobian prediction (barrel |tanL| < 0.8), and the
missing spread in quadrature, for u (transverse) and v (longitudinal)."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200_tp.npz")
prim = F.primary_mask(D); g = F.floor_group(D)
T = json.load(open(base + "R_tables_tp.json")); tr = D["truth"]
Rcal = F.apply_R(T, D["sizeX"], D["sizeY"], tr[:, 2], tr[:, 3], D["sig"])
PLANES = [(2.25, 0.00227), (F.IT_R[1], 0.0185), (4.17, 0.0009), (F.IT_R[2], 0.0187), (7.29, 0.0010),
          (F.IT_R[3], 0.0148), (11.68, 0.0008)]
for L in (2, 3, 4):
    sel = prim & (D["layer"] == L) & (g == 0) & (D["pt"] >= 2) & np.isfinite(D["meas"][:, 0]) & np.isfinite(D["truth"][:, 0])
    n = int(sel.sum()); tanl = D["tanL"][sel].astype(float); pt = D["pt"][sel].astype(float); p = pt * np.sqrt(1 + tanl**2)
    H = D["J"][sel][:, :2]
    V = np.zeros((n, 2, 2))
    for r_s, x in PLANES:
        if r_s < F.IT_R[L] - 0.3:
            V += np.einsum("nij,njk,nlk->nil", H, F._kink_param_cov(np.full(n, r_s), tanl, F.highland_theta2(x, p, tanl)), H)
    d = (D["meas"][sel] - D["truth"][sel])[:, :2] * 1e4
    R = Rcal[sel][:, :2] * 1e4
    print(f"== IL{L} (r {F.IT_R[L]:.2f} cm), n {n}")
    for lo, hi in zip(F.PT_BINS[:-1], F.PT_BINS[1:]):
        m = (pt >= lo) & (pt < hi)
        row = []
        for k, lab in ((0, "u"), (1, "v")):
            obs = F.rs(d[m, k])[0]
            pred = float(np.median(np.sqrt(V[m, k, k] * 1e8 + R[m, k] ** 2)))
            miss = np.sqrt(max(obs**2 - pred**2, 0.0))
            row.append(f"{lab}: obs {obs:6.1f} pred {pred:6.1f} missing {miss:5.1f}")
        print(f"   pT [{lo:>3g},{hi:>4g}) n {m.sum():5d} | " + " | ".join(row) + "  (um)")
