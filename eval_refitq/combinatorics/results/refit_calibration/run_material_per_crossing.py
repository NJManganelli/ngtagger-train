"""Test C: is the ML-vs-MAD disagreement (Gaussian NLL wants x2.2 material, core
MAD wants ~x1.3) a SCALE MIXTURE from track-to-track material, i.e. module
OVERLAPS (the particle crosses two modules of one layer: TBPX ladders alternate
inner/outer radius)?

Overlap flag per (track, layer): the target TP owns clusters on modules at two
distinct radii (>1 mm apart) of that layer (from ALL its clusters, not only the
primary one). Expected signature, in fit order IL4 -> IL1: the innovation at L
widens when layer L+1 was an overlap (its extra module material sits between
L+1 and L); the IL4 innovation must NOT depend on the IL4 overlap (a kink at the
hit does not move the hit) -- the control.

Model 'per-crossing': IT module planes charged the SINGLE-crossing x/X0 (CMSSW D121
scan median: IL2 1.79%, IL3/IL4 1.43%) times the number of modules crossed,
instead of the phi-averaged tkLayout value for every track."""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
from scipy.optimize import minimize
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200_tp.npz")
prim = F.primary_mask(D)
T = json.load(open(base + "R_tables_tp.json")); tr = D["truth"]
Rcal = F.apply_R(T, D["sizeX"], D["sizeY"], tr[:, 2], tr[:, 3], D["sig"])

# ---- modules crossed per (track, layer), from all target-TP clusters ---------
key = D["track"].astype(np.int64) * 8 + D["layer"].astype(np.int64)
rk = np.round(D["r_mod"].astype(np.float64) * 10.0).astype(np.int64)      # 1 mm bins
pair = np.unique(np.stack([key, rk], 1), axis=0)
uk, nmod = np.unique(pair[:, 0], return_counts=True)


def modules_crossed(tracks, L):
    k = tracks.astype(np.int64) * 8 + L
    p = np.clip(np.searchsorted(uk, k), 0, len(uk) - 1)
    return np.where(uk[p] == k, nmod[p], 0)


def pack(sel):
    """F.pack_tracks plus the track ids in its order (same unique/keep logic)."""
    P = F.pack_tracks(D, sel, Rcal)
    tk = D["track"][sel]
    ut, inv = np.unique(tk, return_inverse=True)
    ii = np.flatnonzero(sel)
    first = np.zeros(len(ut), np.int64); first[inv[::-1]] = ii[::-1]
    ids = ut[F.floor_group(D)[first] >= 0]
    assert len(ids) == P["n"]
    P["ids"] = ids
    P["nmod"] = np.stack([modules_crossed(ids, L) for L in (1, 2, 3, 4)], 1)    # (n, 4)
    return P


rng = np.random.default_rng(2)
tracks = np.unique(D["track"][prim & (D["pt"] >= 2)])
fit_tr = rng.choice(tracks, 40000, replace=False)
indep = np.random.default_rng(7).choice(np.setdiff1d(tracks, fit_tr), 40000, replace=False)
P = pack(prim & np.isin(D["track"], fit_tr))
Pc = pack(prim & np.isin(D["track"], indep))
for L in (1, 2, 3, 4):
    c = Pc["nmod"][:, L - 1]
    print(f"IL{L}: tracks with a cluster {np.mean(c > 0):.3f}; of those, 2+ modules crossed {np.mean(c[c > 0] >= 2):.3f}")

SINGLE = {2: 0.0179, 3: 0.0143, 4: 0.0143}
nf = 5 * F.N_GROUP


def kf(P, fl, s_mat, s_seed, per_crossing, clip=25.0, return_pulls=False):
    planes = F.tkl_planes()
    n = P["n"]
    Cst = P["C"] * s_seed[None, :, None] * s_seed[None, None, :]
    Cst[:, np.arange(5), np.arange(5)] += fl[P["g"]] ** 2
    dlt = np.zeros((n, 5)); r_prev = F.OT_INNER_R; tot = 0.0; pulls = []
    for L in (4, 3, 2, 1):
        rL = F.IT_R[L]
        for r_s, x, w in planes:
            if rL < r_s <= r_prev + 1e-6:
                xx = np.full(n, s_mat * x)
                if per_crossing and "module" in w:
                    j = min(SINGLE, key=lambda k: abs(F.IT_R[k] - r_s))
                    xx = s_mat * SINGLE[j] * np.maximum(P["nmod"][:, j - 1], 1)
                Cst += F._kink_param_cov(np.full(n, r_s), P["tanl"], F.highland_theta2(xx, P["p"], P["tanl"]))
        r_prev = rL
        h = P["have"][:, L - 1]
        if not h.any():
            continue
        H = P["H"][h, L - 1]
        y = P["res"][h, L - 1] - np.einsum("nij,nj->ni", H, dlt[h])
        S = np.einsum("nij,njk,nlk->nil", H, Cst[h], H)
        S[:, np.arange(4), np.arange(4)] += P["R"][h, L - 1] ** 2
        Suv = S[:, :2, :2]
        det = Suv[:, 0, 0] * Suv[:, 1, 1] - Suv[:, 0, 1] ** 2
        m2 = (Suv[:, 1, 1] * y[:, 0] ** 2 - 2 * Suv[:, 0, 1] * y[:, 0] * y[:, 1] + Suv[:, 0, 0] * y[:, 1] ** 2) / det
        tot += 0.5 * np.sum(np.minimum(m2, clip) + np.log(np.maximum(det, 1e-300)))
        if return_pulls:
            pulls.append({"L": L, "h": h, "pu": y[:, 0] / np.sqrt(S[:, 0, 0]), "pv": y[:, 1] / np.sqrt(S[:, 1, 1]),
                          "m2": m2, "pt": P["p"][h] / np.sqrt(1 + P["tanl"][h] ** 2)})
        dm = P["dimok"][h, L - 1]
        Sm = S.copy()
        Sm[~np.repeat(dm[:, :, None], 4, 2) | ~np.repeat(dm[:, None, :], 4, 1)] = 0.0
        Sm[:, np.arange(4), np.arange(4)] = np.where(dm, Sm[:, np.arange(4), np.arange(4)], 1.0)
        ym = np.where(dm, y, 0.0); Hm = np.where(dm[:, :, None], H, 0.0)
        CHt = np.einsum("nij,nkj->nik", Cst[h], Hm)
        Kg = np.einsum("nik,nkl->nil", CHt, np.linalg.inv(Sm))
        dlt[h] += np.einsum("nik,nk->ni", Kg, ym)
        Cst[h] -= np.einsum("nik,njk->nij", Kg, CHt)
    return (tot, pulls) if return_pulls else tot


def split_table(lab, pulls, P):
    print(f"-- {lab}: core MAD of innovation pulls, split by modules crossed on the layer ABOVE (fit order)")
    for d in pulls:
        L = d["L"]
        above = P["nmod"][d["h"], L] if L < 4 else None       # layer L+1
        same = P["nmod"][d["h"], L - 1]
        row = f"   IL{L}: all u {F.rs(d['pu'])[0]:.3f} v {F.rs(d['pv'])[0]:.3f} 4sig {np.mean(d['m2'] < 16):.4f}"
        if above is not None:
            for nm, m in (("1 module above", above == 1), ("2+ above", above >= 2)):
                row += f" | {nm}: n {m.sum():5d} u {F.rs(d['pu'][m])[0]:.3f} v {F.rs(d['pv'][m])[0]:.3f}"
        for nm, m in (("1 module here", same == 1), ("2+ here", same >= 2)):
            row += f" | {nm}: u {F.rs(d['pu'][m])[0]:.3f}"
        print(row, flush=True)


# ---- diagnostic at the v9 material-x1 point (floors, seed scales), uniform material
v9 = json.load(open(base + "fit_v9.json"))["mixture NLL, material x1"]["q"]
fl9 = np.array(v9[:nf]).reshape(F.N_GROUP, 5) * F.FLOOR_REF[None]
s9 = np.array([v9[nf], v9[nf], v9[nf + 1], v9[nf + 1], v9[nf]])
for per in (False, True):
    _, pulls = kf(Pc, fl9, 1.0, s9, per, return_pulls=True)
    split_table(f"material x1, {'per-crossing' if per else 'phi-averaged'}", pulls, Pc)

# ---- ML refit with per-crossing material: does the material scale fall to ~1?
def nll(q):
    fl = q[:nf].reshape(F.N_GROUP, 5) * F.FLOOR_REF[None]
    return kf(P, fl, q[nf], np.array([q[-2], q[-2], q[-1], q[-1], q[-2]]), True)


t0 = time.time()
res = minimize(nll, np.r_[np.full(nf, 0.3), 1.2, 1.2, 1.2], method="L-BFGS-B",
               bounds=[(0.0, 50.0)] * nf + [(0.1, 10.0)] + [(0.3, 5.0)] * 2,
               options={"maxiter": 400, "eps": 1e-3, "ftol": 1e-11})
q = res.x
print(f"== per-crossing material, ML fit: nll {res.fun:.1f} ({time.time()-t0:.0f}s, {res.nfev} evals) "
      f"material scale {q[nf]:.3f} sT {q[-2]:.3f} sL {q[-1]:.3f}", flush=True)
fl = q[:nf].reshape(F.N_GROUP, 5) * F.FLOOR_REF[None]
_, pulls = kf(Pc, fl, q[nf], np.array([q[-2], q[-2], q[-1], q[-1], q[-2]]), True, return_pulls=True)
split_table("per-crossing, ML optimum (independent sample)", pulls, Pc)
json.dump({"q": q.tolist(), "nll": float(res.fun)}, open(base + "fit_per_crossing.json", "w"), indent=1)
print("done", flush=True)
