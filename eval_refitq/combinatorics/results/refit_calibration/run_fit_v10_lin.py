"""v10 on LINEARISED residuals (see the block after dres). v10: EMPIRICAL seed covariance. The TTTrack covariance reports corr(rInv, phi0)
~ +0.975 where the truth residuals give +0.5..+0.85 (seed_correlations.log): a
near-degeneracy that makes the refit KF over-trust one (rInv, phi0) combination.
Seed here = robust covariance of (TTTrack - truth at the POCA) per floor group x
pT bin, MINUS the covariance the IT/gap kinks already add between the POCA and the
OT (beam pipe .. TBPS inner cylinder, Highland x1 -- validated fit-free), since the
innovation model charges those kinks itself. Truth: nano tp_phi0/tp_d0/tp_z0.
Then: closure at material x1 with NO fitted seed parameters, and an ML fit of the
material scale alone."""
import sys, glob, json, time, numpy as np, uproot, awkward as ak
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
from ngtagger.truth_helix import KPT_CMSSW
from scipy.optimize import minimize_scalar
base = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/"
D = F.load(base + "calib_perfect_ttbar_pu200_tp.npz")
prim = F.primary_mask(D)
T = json.load(open(base + "R_tables_tp.json")); tr = D["truth"]
Rcal = F.apply_R(T, D["sizeX"], D["sizeY"], tr[:, 2], tr[:, 3], D["sig"])

files = sorted(glob.glob("/Users/nmangane/smartpixels/cmssw/work/otstub_arm/itot_tp_f*_100ev.root"))
HP = ["rInv", "phi", "tanL", "z0", "d0"]
cols = HP + ["tp_pt", "tp_phi0", "tp_tanL", "tp_z0", "tp_d0", "tp_charge"]
A = uproot.concatenate([f"{f}:Events" for f in files], [f"L1TTrack_{c}" for c in cols])
N = {c: ak.to_numpy(ak.flatten(A[f"L1TTrack_{c}"])).astype(float) for c in cols}
assert np.mean(np.abs(N["tanL"][D["track"]] - D["tanL"]) < 1e-5) > 0.999, "track index join failed"
dres = np.stack([N[p] for p in HP], 1) - np.stack([N["tp_charge"] * KPT_CMSSW / N["tp_pt"], N["tp_phi0"],
                                                     N["tp_tanL"], N["tp_z0"], N["tp_d0"]], 1)
dres[:, 1] = np.angle(np.exp(1j * dres[:, 1]))
# LINEARISED DATA (linearity_check.py): remove the nonlinear part of every residual,
# i.e. (truth - seed projection) - H (a_true - a_seed), so the KF's linear model is exact
_mis = D["truth"] - (D["meas"] - D["res"]) - np.einsum("nij,nj->ni", D["J"], -dres[D["track"]])
D["res"] = np.where(np.isfinite(_mis), D["res"] - _mis, D["res"])


def pack(sel):
    P = F.pack_tracks(D, sel, Rcal)
    tk = D["track"][sel]
    ut, inv = np.unique(tk, return_inverse=True)
    ii = np.flatnonzero(sel)
    first = np.zeros(len(ut), np.int64); first[inv[::-1]] = ii[::-1]
    P["ids"] = ut[F.floor_group(D)[first] >= 0]
    assert len(P["ids"]) == P["n"]
    return P


# kinks between the POCA and the OT: beam pipe, IL1 module, then the tkLayout list
INNER = [(2.25, 0.00227, "beam pipe"), (F.IT_R[1], 0.0185, "IL1 module")]
def kinks_to_ot(P, s_mat=1.0):
    K = np.zeros((P["n"], 5, 5))
    for r_s, x, _ in INNER + F.tkl_planes():
        K += F._kink_param_cov(np.full(P["n"], r_s), P["tanl"], F.highland_theta2(s_mat * x, P["p"], P["tanl"]))
    return K


PTB = [2, 3, 5, 10, 20, 1e9]
tracks = np.unique(D["track"][prim & (D["pt"] >= 2)])
Pall = pack(prim & np.isin(D["track"], tracks))
pt_all = Pall["p"] / np.sqrt(1 + Pall["tanl"] ** 2)
Kall = kinks_to_ot(Pall)
E = {}
print("empirical seed (TTTrack - truth - IT kinks), sqrt diag / median reported, per floor group x pT bin:")
for g in range(F.N_GROUP):
    for b in range(len(PTB) - 1):
        m = (Pall["g"] == g) & (pt_all >= PTB[b]) & (pt_all < PTB[b + 1])
        x = dres[Pall["ids"][m]]
        if m.sum() < 150:
            E[(g, b)] = None
            continue
        med = np.median(x, 0); mad = 1.4826 * np.median(np.abs(x - med), 0)
        keep = np.all(np.abs(x - med) < 4 * mad, 1)
        Ce = np.cov(x[keep].T) - Kall[m][keep].mean(0)
        # PSD projection in correlation space: the parameters span ~7 orders of
        # magnitude in variance, so an eigenvalue floor on the raw matrix would
        # swamp rInv
        sd = np.sqrt(np.maximum(np.diag(Ce), 1e-30))
        w, V = np.linalg.eigh(Ce / np.outer(sd, sd))
        Ce = ((V * np.maximum(w, 1e-4)) @ V.T) * np.outer(sd, sd)
        E[(g, b)] = Ce
        Cr = np.median(Pall["C"][m], 0)
        print(f"  {F.FLOOR_GROUPS[g][0]:18s} pT [{PTB[b]:g},{PTB[b+1]:g}) n {m.sum():6d}: "
              + " ".join(f"{p} {np.sqrt(Ce[i,i]/Cr[i,i]):.2f}" for i, p in enumerate(HP))
              + f"  corr(rInv,phi) {Ce[0,1]/np.sqrt(Ce[0,0]*Ce[1,1]):+.2f} (reported {Cr[0,1]/np.sqrt(Cr[0,0]*Cr[1,1]):+.2f})"
              + f"  negative eigen clipped {int((w < 0).sum())}")
# fill sparse bins from the nearest populated pT bin of the same group
for (g, b), v in list(E.items()):
    if v is None:
        near = sorted((abs(bb - b), bb) for (gg, bb), vv in E.items() if gg == g and vv is not None)
        E[(g, b)] = E[(g, near[0][1])]


def with_empirical_seed(P):
    pt = P["p"] / np.sqrt(1 + P["tanl"] ** 2)
    b = np.clip(np.searchsorted(PTB, pt, side="right") - 1, 0, len(PTB) - 2)
    P2 = dict(P)
    P2["C"] = np.stack([E[(g, bb)] for g, bb in zip(P["g"], b)])
    return P2


rng = np.random.default_rng(2)
fit_tr = rng.choice(tracks, 40000, replace=False)
indep = np.random.default_rng(7).choice(np.setdiff1d(tracks, fit_tr), 40000, replace=False)
Pf = with_empirical_seed(pack(prim & np.isin(D["track"], fit_tr)))
Pc = with_empirical_seed(pack(prim & np.isin(D["track"], indep)))
zero_floors = np.r_[np.full(5 * F.N_GROUP, np.log(1e-9)), 0.0]


def closure(lab, s_mat):
    _, pulls = F.kf_innovations_v4(Pc, zero_floors, return_pulls=True, mat_scale=s_mat)
    print(f"-- {lab} (independent sample, empirical seed, no fitted seed parameters)")
    for d in pulls:
        row = f"   IL{d['L']}: core MAD u {F.rs(d['pu'])[0]:.3f} v {F.rs(d['pv'])[0]:.3f}  4sig {np.mean(d['m2'] < 16):.4f}  u by pT:"
        for lo, hi in zip(PTB[:-1], PTB[1:]):
            m = (d["pt"] >= lo) & (d["pt"] < hi)
            row += f" {F.rs(d['pu'][m])[0]:.2f}"
        row += "  v by pT:"
        for lo, hi in zip(PTB[:-1], PTB[1:]):
            m = (d["pt"] >= lo) & (d["pt"] < hi)
            row += f" {F.rs(d['pv'][m])[0]:.2f}"
        print(row, flush=True)


closure("material x1", 1.0)
t0 = time.time()
res = minimize_scalar(lambda s: F.kf_innovations_v4(Pf, zero_floors, mat_scale=s), bounds=(0.2, 5.0), method="bounded",
                      options={"xatol": 1e-3})
print(f"== ML material scale with the empirical seed: {res.x:.3f} (nll {res.fun:.1f}, {time.time()-t0:.0f}s)", flush=True)
closure(f"material x{res.x:.3f} (ML)", res.x)
json.dump({"ml_material_scale": float(res.x), "nll": float(res.fun), "pt_bins": PTB,
           "seed": {f"{g},{b}": v.tolist() for (g, b), v in E.items()}}, open(base + "fit_v10_lin.json", "w"), indent=1)
print("done", flush=True)
