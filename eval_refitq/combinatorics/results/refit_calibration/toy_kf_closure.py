"""Machinery check: generate innovations from the model's OWN assumptions and fit
them with the same code. Per track (real H, R, masks, p, tanL from the calibration
sample): seed error e ~ N(0, C_seed); every kink plane s (tkLayout, x1) draws
dp_s ~ N(0, K_s); the true trajectory inside r_s is the OT helix + dp_s, so the
residual at layer L = H_L (sum_{r_s > r_L} dp_s - e) + R noise. kf_innovations_v4
must then give pull MAD ~1 at every layer and an ML material scale ~1."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
from scipy.optimize import minimize_scalar
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200_tp.npz")
prim = F.primary_mask(D)
T = json.load(open(base + "R_tables_tp.json")); tr = D["truth"]
Rcal = F.apply_R(T, D["sizeX"], D["sizeY"], tr[:, 2], tr[:, 3], D["sig"])
tracks = np.unique(D["track"][prim & (D["pt"] >= 2)])
P = F.pack_tracks(D, prim & np.isin(D["track"], np.random.default_rng(2).choice(tracks, 40000, replace=False)), Rcal)
rng = np.random.default_rng(11)
n = P["n"]


def draw(C):
    w, V = np.linalg.eigh(C)
    return np.einsum("nij,nj->ni", V * np.sqrt(np.maximum(w, 0))[:, None, :], rng.standard_normal((len(C), 5)))


e = draw(P["C"])
planes = F.tkl_planes()
dps = [(r_s, draw(F._kink_param_cov(np.full(n, r_s), P["tanl"], F.highland_theta2(x, P["p"], P["tanl"])))) for r_s, x, _ in planes]
res = np.zeros_like(P["res"])
for L in (1, 2, 3, 4):
    dp = sum(d for r_s, d in dps if r_s > F.IT_R[L])
    res[:, L - 1] = np.einsum("nij,nj->ni", P["H"][:, L - 1], dp - e) + P["R"][:, L - 1] * rng.standard_normal((n, 4))
Pt = dict(P); Pt["res"] = res
zero = np.r_[np.full(5 * F.N_GROUP, np.log(1e-9)), 0.0]
_, pulls = F.kf_innovations_v4(Pt, zero, return_pulls=True, mat_scale=1.0)
for d in pulls:
    print(f"toy IL{d['L']}: pull MAD u {F.rs(d['pu'])[0]:.3f} v {F.rs(d['pv'])[0]:.3f}  RMS u {F.rs(d['pu'])[1]:.3f} v {F.rs(d['pv'])[1]:.3f}  4sig {np.mean(d['m2'] < 16):.4f}")
r = minimize_scalar(lambda s: F.kf_innovations_v4(Pt, zero, mat_scale=s), bounds=(0.2, 5.0), method="bounded")
print(f"toy ML material scale: {r.x:.3f} (truth 1)")
