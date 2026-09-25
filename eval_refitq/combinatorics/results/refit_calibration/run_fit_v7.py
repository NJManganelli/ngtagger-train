"""v7: the OT (TTTrack) covariance rescaled by block, correlations kept:
C' = S C S, S = diag(sT, sT, sL, sL, sT) over (rInv, phi, tanL, z0, d0), plus
linear bounded floors; tkLayout material x1 (validated fit-free against the
physical IT scattering, forward_scatter_closure + truth_phi_poca) or free.
Motivation: ot_seed_pulls.log -- TTTrack transverse pulls ~1.9 at all pT."""
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


def make_f(fix_scale):
    def f(q):            # q = [floors/FLOOR_REF, (material scale), sT, sL]
        sT, sL = q[-2], q[-1]
        s = np.array([sT, sT, sL, sL, sT])
        P2 = dict(P); P2["C"] = P["C"] * s[None, :, None] * s[None, None, :]
        pl = np.r_[np.log(np.maximum(q[:nf], 1e-9)), 0.0]
        return F.kf_innovations_v4(P2, pl, mat_scale=fix_scale if fix_scale else q[nf])
    return f


out = {}
for fix in (1.0, None):
    lab = f"seed block scale, material {'x1' if fix else 'scale free'}"
    q0 = np.r_[np.full(nf, 0.3), [] if fix else [1.0], 1.5, 1.2]
    bnds = [(0.0, 50.0)] * nf + ([] if fix else [(0.1, 10.0)]) + [(0.3, 5.0)] * 2
    t0 = time.time()
    res = minimize(make_f(fix), q0, method="L-BFGS-B", bounds=bnds, options={"maxiter": 400, "eps": 1e-3, "ftol": 1e-11})
    sc = fix if fix else res.x[nf]
    fl = res.x[:nf].reshape(F.N_GROUP, 5) * F.FLOOR_REF[None]
    print(f"== {lab}: nll {res.fun:.1f} ({time.time()-t0:.0f}s, {res.nfev} evals) material {sc:.3f}  sT {res.x[-2]:.3f}  sL {res.x[-1]:.3f}", flush=True)
    for (nm, *_), fv in zip(F.FLOOR_GROUPS, fl):
        print(f"   floor {nm:18s}: rInv {fv[0]:.1e} phi {fv[1]*1e3:.3f}mrad tanL {fv[2]:.1e} z0 {fv[3]*1e4:.0f}um d0 {fv[4]*1e4:.0f}um", flush=True)
    out[lab] = {"q": res.x.tolist(), "nll": float(res.fun), "scale": float(sc), "sT": float(res.x[-2]), "sL": float(res.x[-1])}
    json.dump(out, open(base + "fit_v7.json", "w"), indent=1)
print("done", flush=True)
