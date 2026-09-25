import sys, json, time, numpy as np
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200.npz")
prim = F.primary_mask(D)
T = json.load(open(base + "R_tables.json")); tr = D["truth"]
Rcal = F.apply_R(T, D["sizeX"], D["sizeY"], tr[:, 2], tr[:, 3], D["sig"])
rng = np.random.default_rng(2)
tracks = np.unique(D["track"][prim & (D["pt"] >= 2)])
pick = rng.choice(tracks, 40000, replace=False)
P = F.pack_tracks(D, prim & np.isin(D["track"], pick), Rcal)
out = {}
for lab, fit_scale in (("floors + global material scale", True), ("floors only, tkLayout material x1", False)):
    t0 = time.time()
    res = F.fit_v4(P, fit_scale=fit_scale)
    pp = res.x if fit_scale else np.r_[res.x, 0.0]
    fl = np.exp(pp[:5 * F.N_GROUP]).reshape(F.N_GROUP, 5) * F.FLOOR_REF[None]
    scale = float(np.exp(pp[5 * F.N_GROUP]))
    print(f"== {lab}: nll {res.fun:.1f} ({time.time()-t0:.0f}s, {res.nfev} evals, {res.message}) material scale {scale:.3f}", flush=True)
    for (nm, *_), f in zip(F.FLOOR_GROUPS, fl):
        print(f"   floor {nm:18s}: rInv {f[0]:.1e} phi {f[1]*1e3:.3f}mrad tanL {f[2]:.1e} z0 {f[3]*1e4:.0f}um d0 {f[4]*1e4:.0f}um", flush=True)
    out[lab] = {"params": pp.tolist(), "nll": float(res.fun), "scale": scale}
json.dump(out, open(base + "fit_v4.json", "w"), indent=1)
print("done", flush=True)
