"""v6 = v4 (linear bounded floors, tkLayout material x scale) + a pT-INDEPENDENT
pivot term in the OT seed: constant transverse/longitudinal angle errors thT, thL
applied as a rotation about a point at radius r_piv (same Jacobian as a kink).
Hypothesis: the OT fit under-reports its own measurement error, which projected
inward is a direction error pivoting inside the OT (correlated phi0/d0), not a
diagonal floor. r_piv is scanned; the fit is run with the material scale free."""
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
TH_REF = 1e-4                        # rad
sec2 = 1.0 + P["tanl"] ** 2; sec = np.sqrt(sec2)


def pivot_cov(r, thT, thL):
    jT = np.zeros((P["n"], 5)); jT[:, 1] = sec; jT[:, 4] = r * sec
    jL = np.zeros((P["n"], 5)); jL[:, 2] = sec2; jL[:, 3] = -r * sec2
    return (thT ** 2) * jT[:, :, None] * jT[:, None, :] + (thL ** 2) * jL[:, :, None] * jL[:, None, :]


def make_f(r, fix_scale=None):
    def f(q):            # q = [floors/FLOOR_REF, (scale), thT/TH_REF, thL/TH_REF], all linear
        sc = fix_scale if fix_scale is not None else q[nf]
        thT, thL = q[-2] * TH_REF, q[-1] * TH_REF
        P2 = dict(P); P2["C"] = P["C"] + pivot_cov(r, thT, thL)
        pl = np.r_[np.log(np.maximum(q[:nf], 1e-9)), 0.0]
        return F.kf_innovations_v4(P2, pl, mat_scale=sc)
    return f


start = json.load(open(base + "fit_v4b.json"))["start floors 0.3, scale 1.5"]
fl0 = np.exp(np.array(start["params_log"][:nf]))
out = {}
for r in (30.0, 50.0, 80.0):
    for fix in (None, 1.0):
        lab = f"r_piv {r:.0f} cm, material {'scale free' if fix is None else 'x1'}"
        q0 = np.r_[np.maximum(fl0, 0.05), [] if fix else [1.5], 3.0, 3.0]
        bnds = [(0.0, 50.0)] * nf + ([] if fix else [(0.1, 10.0)]) + [(0.0, 200.0)] * 2
        t0 = time.time()
        res = minimize(make_f(r, fix), q0, method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": 400, "eps": 1e-3, "ftol": 1e-11})
        sc = fix if fix else res.x[nf]
        fl = res.x[:nf].reshape(F.N_GROUP, 5) * F.FLOOR_REF[None]
        print(f"== {lab}: nll {res.fun:.1f} ({time.time()-t0:.0f}s, {res.nfev} evals) scale {sc:.3f}"
              f"  pivot thT {res.x[-2]*TH_REF*1e3:.3f} mrad  thL {res.x[-1]*TH_REF*1e3:.3f} mrad", flush=True)
        for (nm, *_), fv in zip(F.FLOOR_GROUPS, fl):
            print(f"   floor {nm:18s}: rInv {fv[0]:.1e} phi {fv[1]*1e3:.3f}mrad tanL {fv[2]:.1e} z0 {fv[3]*1e4:.0f}um d0 {fv[4]*1e4:.0f}um", flush=True)
        out[lab] = {"q": res.x.tolist(), "nll": float(res.fun), "scale": float(sc), "r_piv": r,
                    "thT": float(res.x[-2] * TH_REF), "thL": float(res.x[-1] * TH_REF)}
        json.dump(out, open(base + "fit_v6.json", "w"), indent=1)
print("done", flush=True)
