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
best = None
for rg in (16.5, 19.5, 22.5):
    t0 = time.time()
    res = F.fit_innovations(P, rg, x0=None if best is None else best[1].x)
    fl, xx0, xg = F.unpack_v2(res.x)
    print(f"r_gap {rg}: nll {res.fun:.1f} ({time.time()-t0:.0f}s, {res.nfev} evals) x/X0 IL2 {xx0[0]*100:.2f}% "
          f"IL3 {xx0[1]*100:.2f}% IL4 {xx0[2]*100:.2f}% gap {xg*100:.2f}%", flush=True)
    if best is None or res.fun < best[1].fun:
        best = (rg, res)
rg, res = best
fl, xx0, xg = F.unpack_v2(res.x)
for (nm, *_), f in zip(F.FLOOR_GROUPS, fl):
    print(f"   floor {nm:18s}: rInv {f[0]:.1e} phi {f[1]*1e3:.3f}mrad tanL {f[2]:.1e} z0 {f[3]*1e4:.0f}um d0 {f[4]*1e4:.0f}um", flush=True)
json.dump({"model": "v3 innovation likelihood", "r_gap_cm": rg, "params": res.x.tolist(), "nll": res.fun},
          open(base + "joint_fit_v3.json", "w"), indent=1)
print("done")
