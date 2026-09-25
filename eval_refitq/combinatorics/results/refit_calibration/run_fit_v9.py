"""v9: v7 model (block-scaled TTTrack cov + floors + tkLayout material x s_IT) with
a TWO-COMPONENT innovation likelihood: (1-f) N(0, S) + f N(0, k^2 S), f and k
free. The Gaussian (clipped) likelihood of v4..v8 must widen its core to cover
the heavy tails (innovation RMS 2.4-11 against MAD ~1), which a material scale
does; if s_IT falls toward the robust-closure value (~1.2-1.3), the x2 was tails.
Corrected sample (itot_tp, tp_phi0 truth) and R_tables_tp.json.
The KF update itself stays Gaussian (standard); only the NLL changes."""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
from scipy.optimize import minimize
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200_tp.npz")
prim = F.primary_mask(D)
T = json.load(open(base + "R_tables_tp.json")); tr = D["truth"]
Rcal = F.apply_R(T, D["sizeX"], D["sizeY"], tr[:, 2], tr[:, 3], D["sig"])
rng = np.random.default_rng(2)
tracks = np.unique(D["track"][prim & (D["pt"] >= 2)])
fit_tr = rng.choice(tracks, 40000, replace=False)
P = F.pack_tracks(D, prim & np.isin(D["track"], fit_tr), Rcal)
indep = np.random.default_rng(7).choice(np.setdiff1d(tracks, fit_tr), 40000, replace=False)
Pc = F.pack_tracks(D, prim & np.isin(D["track"], indep), Rcal)
nf = 5 * F.N_GROUP


def kf_mixture(P, fl, s_mat, f_out, k_out, return_pulls=False):
    """kf_innovations_v4's recursion with the mixture NLL (same KF update)."""
    planes = F.tkl_planes()
    n = P["n"]
    Cst = P["C"].copy()
    Cst[:, np.arange(5), np.arange(5)] += fl[P["g"]] ** 2
    dlt = np.zeros((n, 5)); r_prev = F.OT_INNER_R; tot = 0.0; pulls = []
    lk2 = np.log(k_out ** 2)
    for L in (4, 3, 2, 1):
        rL = F.IT_R[L]
        for r_s, x, _ in planes:
            if rL < r_s <= r_prev + 1e-6:
                Cst += F._kink_param_cov(np.full(n, r_s), P["tanl"], F.highland_theta2(s_mat * x, P["p"], P["tanl"]))
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
        a = np.log1p(-f_out) - 0.5 * m2
        b = np.log(f_out) - lk2 - 0.5 * m2 / k_out ** 2
        tot += np.sum(0.5 * np.log(np.maximum(det, 1e-300)) - np.logaddexp(a, b))
        if return_pulls:
            pulls.append({"L": L, "pu": y[:, 0] / np.sqrt(S[:, 0, 0]), "pv": y[:, 1] / np.sqrt(S[:, 1, 1]),
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


def unpack(q, fix_mat):
    fl = q[:nf].reshape(F.N_GROUP, 5) * F.FLOOR_REF[None]
    i = nf
    s_mat = 1.0 if fix_mat else q[i]; i += 0 if fix_mat else 1
    sT, sL, f_out, k_out = q[i:i + 4]
    s = np.array([sT, sT, sL, sL, sT])
    return fl, s_mat, s, f_out, k_out


def nll(q, fix_mat, PP):
    fl, s_mat, s, f_out, k_out = unpack(q, fix_mat)
    P2 = dict(PP); P2["C"] = PP["C"] * s[None, :, None] * s[None, None, :]
    return kf_mixture(P2, fl, s_mat, f_out, k_out)


out = {}
for fix_mat in (False, True):
    lab = f"mixture NLL, material {'x1' if fix_mat else 'free'}"
    q0 = np.r_[np.full(nf, 0.3), [] if fix_mat else [1.2], 1.2, 1.2, 0.05, 4.0]
    bnds = [(0.0, 50.0)] * nf + ([] if fix_mat else [(0.1, 10.0)]) + [(0.3, 5.0)] * 2 + [(1e-4, 0.5), (1.5, 30.0)]
    t0 = time.time()
    res = minimize(nll, q0, args=(fix_mat, P), method="L-BFGS-B", bounds=bnds,
                   options={"maxiter": 400, "eps": 1e-3, "ftol": 1e-11})
    fl, s_mat, s, f_out, k_out = unpack(res.x, fix_mat)
    print(f"== {lab}: nll {res.fun:.1f} ({time.time()-t0:.0f}s, {res.nfev} evals) material {s_mat:.3f} "
          f"sT {s[0]:.3f} sL {s[2]:.3f}  tail fraction {f_out:.4f} width x{k_out:.2f}", flush=True)
    # robust closure on the independent sample
    P2 = dict(Pc); P2["C"] = Pc["C"] * s[None, :, None] * s[None, None, :]
    _, pulls = kf_mixture(P2, fl, s_mat, f_out, k_out, return_pulls=True)
    for d in pulls:
        row = f"   IL{d['L']}: core MAD u {F.rs(d['pu'])[0]:.3f} v {F.rs(d['pv'])[0]:.3f}  4sig cont {np.mean(d['m2'] < 16):.4f}  pT bins u:"
        for lo, hi in zip(F.PT_BINS[:-1], F.PT_BINS[1:]):
            m = (d["pt"] >= lo) & (d["pt"] < hi)
            row += f" {F.rs(d['pu'][m])[0]:.2f}"
        print(row, flush=True)
    out[lab] = {"q": res.x.tolist(), "nll": float(res.fun), "material": float(s_mat), "sT": float(s[0]),
                "sL": float(s[2]), "f_tail": float(f_out), "k_tail": float(k_out)}
    json.dump(out, open(base + "fit_v9.json", "w"), indent=1)
print("done", flush=True)
