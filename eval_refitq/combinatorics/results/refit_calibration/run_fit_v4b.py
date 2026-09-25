"""v4 with LINEAR, bounded floors (log floors lose their gradient once they get
small and cannot climb back, which is suspected of pinning phi/rInv floors at 0)."""
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
nf = 5 * F.N_GROUP
def to_log(q):                        # q = [floors / FLOOR_REF (linear), scale]
    return np.r_[np.log(np.maximum(q[:nf], 1e-9)), np.log(q[nf])]
f = lambda q: F.kf_innovations_v4(P, to_log(q))
out = {}
for lab, q0 in (("start floors 0.3, scale 1.5", np.r_[np.full(nf, 0.3), 1.5]),
                ("start floors 1.0, scale 1.0", np.r_[np.full(nf, 1.0), 1.0])):
    t0 = time.time()
    res = minimize(f, q0, method="L-BFGS-B", bounds=[(0.0, 50.0)] * nf + [(0.1, 10.0)],
                   options={"maxiter": 400, "eps": 1e-3, "ftol": 1e-11})
    fl = res.x[:nf].reshape(F.N_GROUP, 5) * F.FLOOR_REF[None]
    print(f"== {lab}: nll {res.fun:.1f} ({time.time()-t0:.0f}s, {res.nfev} evals, {res.message}) material scale {res.x[nf]:.3f}", flush=True)
    for (nm, *_), fv in zip(F.FLOOR_GROUPS, fl):
        print(f"   floor {nm:18s}: rInv {fv[0]:.1e} phi {fv[1]*1e3:.3f}mrad tanL {fv[2]:.1e} z0 {fv[3]*1e4:.0f}um d0 {fv[4]*1e4:.0f}um", flush=True)
    out[lab] = {"params_log": to_log(res.x).tolist(), "nll": float(res.fun), "scale": float(res.x[nf])}
json.dump(out, open(base + "fit_v4b.json", "w"), indent=1)
print("done", flush=True)
