"""Option 2 diagnostic: split the IL4 innovation (first layer, no updates) into
   cluster term  meas - truth   vs the calibrated R, and
   seed term     truth - proj   vs H C H^T (OT covariance + v4b floors + gap material),
per pT bin, on tracks NOT used in the fits. The truth helix is the TP production
helix (no scattering), so the split is clean only at high pT, where the IL1..IL3
modules the particle crossed before IL4 deflect it little."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200.npz")
prim = F.primary_mask(D)
T = json.load(open(base + "R_tables.json")); tr = D["truth"]
Rcal = F.apply_R(T, D["sizeX"], D["sizeY"], tr[:, 2], tr[:, 3], D["sig"])
rng = np.random.default_rng(2)
tracks = np.unique(D["track"][prim & (D["pt"] >= 2)])
fit_tracks = rng.choice(tracks, 40000, replace=False)
indep = np.random.default_rng(7).choice(np.setdiff1d(tracks, fit_tracks), 40000, replace=False)
g_all = F.floor_group(D)
sel = prim & np.isin(D["track"], indep) & (D["layer"] == 4) & (g_all >= 0) & np.isfinite(D["res"][:, 0])
pl = json.load(open(base + "fit_v4b.json"))["start floors 0.3, scale 1.5"]["params_log"]
fl = np.exp(np.array(pl[:5 * F.N_GROUP])).reshape(F.N_GROUP, 5) * F.FLOOR_REF[None]
scale = float(np.exp(pl[5 * F.N_GROUP]))
n = int(sel.sum()); tanl = D["tanL"][sel].astype(float); pt = D["pt"][sel].astype(float); p = pt * np.sqrt(1 + tanl**2)
C0 = D["C"][sel].copy()
Cf = C0.copy(); Cf[:, np.arange(5), np.arange(5)] += fl[g_all[sel]] ** 2
Cm = np.zeros_like(C0)
for r_s, x, _ in F.tkl_planes():
    if F.IT_R[4] < r_s <= F.OT_INNER_R + 1e-6:
        Cm += F._kink_param_cov(np.full(n, r_s), tanl, F.highland_theta2(scale * x, p, tanl))
H = D["J"][sel]
var = lambda C: np.einsum("nj,njk,nk->n", H[:, 0], C, H[:, 0])
s_ot, s_fl, s_mat = var(C0), var(Cf) - var(C0), var(Cm)
R = Rcal[sel, 0]
res = D["res"][sel, 0]                       # meas - proj
clus = D["meas"][sel, 0] - D["truth"][sel, 0]
seed = res - clus                            # truth - proj
S = s_ot + s_fl + s_mat + R**2
um = 1e4
mad = lambda x: F.rs(x)[0]
print(f"IL4 u, independent sample: {n} primary clusters, material scale {scale:.3f}")
print("pT bin      n     | total pull  | cluster (meas-truth)/R | seed (truth-proj)/sqrt(HCH) | median sigma um: OT  floor  mat  R | req. extra um | tail>5 tot/clus/seed")
for lo, hi in zip(F.PT_BINS[:-1], F.PT_BINS[1:]):
    m = (pt >= lo) & (pt < hi)
    sd = np.sqrt(s_ot + s_fl + s_mat)
    # extra u variance (added in quadrature) that brings the total-pull MAD to 1
    lo_a, hi_a = 0.0, 0.2
    for _ in range(50):
        a = 0.5 * (lo_a + hi_a)
        (lo_a, hi_a) = (a, hi_a) if mad(res[m] / np.sqrt(S[m] + a**2)) > 1 else (lo_a, a)
    tail = lambda x: float(np.mean(np.abs(x) > 5))
    print(f"[{lo:>4g},{hi:>4g}) {m.sum():6d} | MAD {mad(res[m]/np.sqrt(S[m])):.2f}  | MAD {mad(clus[m]/R[m]):.2f}  RMS {F.rs(clus[m]/R[m])[1]:5.2f}"
          f"     | MAD {mad(seed[m]/sd[m]):.2f}  RMS {F.rs(seed[m]/sd[m])[1]:5.2f}          "
          f"| {np.median(np.sqrt(s_ot[m]))*um:5.0f} {np.median(np.sqrt(s_fl[m]))*um:5.0f} {np.median(np.sqrt(s_mat[m]))*um:5.0f} {np.median(R[m])*um:4.0f}"
          f" | {a*um:6.0f}       | {tail(res[m]/np.sqrt(S[m])):.3f}/{tail(clus[m]/R[m]):.3f}/{tail(seed[m]/sd[m]):.3f}")
print("\nseed-term MAD at pT >= 5 GeV by floor group:")
for gi, (nm, *_) in enumerate(F.FLOOR_GROUPS):
    m = (pt >= 5) & (g_all[sel] == gi)
    sd = np.sqrt(s_ot + s_fl + s_mat)
    print(f"  {nm:18s} n {m.sum():5d}  seed MAD {mad(seed[m]/sd[m]):.2f}  cluster MAD {mad(clus[m]/R[m]):.2f}  total MAD {mad(res[m]/np.sqrt(S[m])):.2f}")
