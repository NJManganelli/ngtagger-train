"""v8 = v7 (block-scaled TTTrack cov + floors) + OT-layer kinks. The seed is the
OT fit's helix; the particle scattered in the OT modules too (TBPS L1 5.2% X0
phi-averaged at 23.4 cm -- CMSSW D121 geantino scan, material_xcheck.json,
|eta|<0.4), which the OT fit only models crudely. Two material scales: s_IT on
the (fit-free validated) IT/gap planes, s_OT on the OT module planes."""
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
OT_PLANES = [(23.4, 0.0522, "OT L1 TBPS"), (36.2, 0.0447, "OT L2 TBPS"),
             (51.5, 0.0436, "OT L3 TBPS"), (68.7, 0.0190, "OT L4 2S")]
IT_PLANES = F.tkl_planes()
F.OT_INNER_R = 70.0


def f_factory(fix_it):
    def f(q):
        s_it = 1.0 if fix_it else q[nf]
        s_ot = q[nf + (0 if fix_it else 1)]
        # fold the two scales into x/X0 and hand kf_innovations_v4 scale 1
        F.tkl_planes = lambda: [(r, s_it * x, w) for r, x, w in IT_PLANES] + [(r, s_ot * x, w) for r, x, w in OT_PLANES]
        s = np.array([q[-2], q[-2], q[-1], q[-1], q[-2]])
        P2 = dict(P); P2["C"] = P["C"] * s[None, :, None] * s[None, None, :]
        return F.kf_innovations_v4(P2, np.r_[np.log(np.maximum(q[:nf], 1e-9)), 0.0], mat_scale=1.0)
    return f


out = {}
for fix_it in (True, False):
    lab = f"OT kinks, s_OT free, s_IT {'x1' if fix_it else 'free'}"
    q0 = np.r_[np.full(nf, 0.3), [] if fix_it else [1.0], 1.0, 1.2, 1.2]
    bnds = [(0.0, 50.0)] * nf + ([] if fix_it else [(0.1, 10.0)]) + [(0.0, 10.0)] + [(0.3, 5.0)] * 2
    t0 = time.time()
    res = minimize(f_factory(fix_it), q0, method="L-BFGS-B", bounds=bnds,
                   options={"maxiter": 400, "eps": 1e-3, "ftol": 1e-11})
    s_it = 1.0 if fix_it else res.x[nf]
    s_ot = res.x[nf + (0 if fix_it else 1)]
    print(f"== {lab}: nll {res.fun:.1f} ({time.time()-t0:.0f}s, {res.nfev} evals) s_IT {s_it:.3f}  s_OT {s_ot:.3f}"
          f"  sT {res.x[-2]:.3f}  sL {res.x[-1]:.3f}", flush=True)
    out[lab] = {"q": res.x.tolist(), "nll": float(res.fun), "s_IT": float(s_it), "s_OT": float(s_ot)}
    json.dump(out, open(base + "fit_v8.json", "w"), indent=1)
print("done", flush=True)
