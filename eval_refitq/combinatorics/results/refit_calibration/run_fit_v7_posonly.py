"""v7 (block-scaled TTTrack cov + floors, material free) with POSITION-ONLY
updates: the cluster angle measurements (cotA, cotB) are dropped from the KF
update. If the material scale falls to ~1, the x2 was compensating for
over-confident angle measurements (their R, or their correlation with position)."""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
from scipy.optimize import minimize
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200.npz")
prim = F.primary_mask(D)
T = json.load(open(base + "R_tables.json")); tr = D["truth"]
Rcal = F.apply_R(T, D["sizeX"], D["sizeY"], tr[:, 2], tr[:, 3], D["sig"])
rng = np.random.default_rng(2)
tracks = np.unique(D["track"][prim & (D["pt"] >= 2)])
P = F.pack_tracks(D, prim & np.isin(D["track"], rng.choice(tracks, 40000, replace=False)), Rcal)
P["dimok"][:, :, 2:] = False
nf = 5 * F.N_GROUP


def f(q):
    s = np.array([q[-2], q[-2], q[-1], q[-1], q[-2]])
    P2 = dict(P); P2["C"] = P["C"] * s[None, :, None] * s[None, None, :]
    return F.kf_innovations_v4(P2, np.r_[np.log(np.maximum(q[:nf], 1e-9)), 0.0], mat_scale=q[nf])


t0 = time.time()
res = minimize(f, np.r_[np.full(nf, 0.3), 1.0, 1.5, 1.2], method="L-BFGS-B",
               bounds=[(0.0, 50.0)] * nf + [(0.1, 10.0)] + [(0.3, 5.0)] * 2,
               options={"maxiter": 400, "eps": 1e-3, "ftol": 1e-11})
print(f"== position-only updates, material free: nll {res.fun:.1f} ({time.time()-t0:.0f}s, {res.nfev} evals) "
      f"material {res.x[nf]:.3f}  sT {res.x[-2]:.3f}  sL {res.x[-1]:.3f}", flush=True)
json.dump({"q": res.x.tolist(), "nll": float(res.fun)}, open(base + "fit_v7_posonly.json", "w"), indent=1)
