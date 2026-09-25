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
for lab, fit_scale in (("v5: floors + 1/p seed term + global material scale", True),
                       ("v5: floors + 1/p seed term, tkLayout material x1", False)):
    t0 = time.time()
    res, basep, g = F.fit_v5(P, fit_scale=fit_scale)
    fl = np.exp(basep[:5 * F.N_GROUP]).reshape(F.N_GROUP, 5) * F.FLOOR_REF[None]
    scale = float(np.exp(basep[5 * F.N_GROUP])) if fit_scale else 1.0
    print(f"== {lab}: nll {res.fun:.1f} ({time.time()-t0:.0f}s, {res.nfev} evals) material scale {scale:.3f}", flush=True)
    print("   1/p seed term sigma at p = 1 GeV: rInv {:.2e} /cm, phi {:.3f} mrad, tanL {:.2e}, z0 {:.0f} um, d0 {:.0f} um".format(
        g[0], g[1] * 1e3, g[2], g[3] * 1e4, g[4] * 1e4), flush=True)
    for (nm, *_), f in zip(F.FLOOR_GROUPS, fl):
        print(f"   floor {nm:18s}: rInv {f[0]:.1e} phi {f[1]*1e3:.3f}mrad tanL {f[2]:.1e} z0 {f[3]*1e4:.0f}um d0 {f[4]*1e4:.0f}um", flush=True)
    out[lab] = {"base_params": np.asarray(basep).tolist(), "g_seed": np.asarray(g).tolist(), "nll": float(res.fun), "scale": scale}
json.dump(out, open(base + "fit_v5.json", "w"), indent=1)
print("done", flush=True)
