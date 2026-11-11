"""The parametrization of PLAN.md section 7b (rules: reviews/stats-review-02-parametrization.md): from today's
extended-collection tracks ("implemented now") to the tracks a shipped bit allocation delivers, per scenario.

No fitted mixture. Per allocation the extended scan gives, track by track, today's result, the arm's result and the
arm's refit on today's clusters (forced). The model has two observed states:

  SAME CLUSTERS (precision part)
    parameters   q' = q + lambda * sigma_today(q) * eps_q, eps_q from the empirical inverse CDF of
                 (forced - today) / sigma_today (knots KNOTS, clamped); lambda per cell from a root find so that the 68 %
                 width of the residual to truth equals the forced refit's. d0 is drawn with u_rank; z0 through a
                 Gaussian copula with d0's draw, tanL with z0's, 1/R and phi0 with d0's (normal-score correlations).
    covariance   diagonal scaled by the cell's median forced / today ratio; today's correlations kept.
    scores       (the six deployed-head outputs and the chi2-like inputs, in their natural variable)
                 s' = a + b s + sigma_s * eta, eta from the empirical standardized residual, conditioned on
                 |2 u_rank - 1| in [0, 0.5), [0.5, 0.9), [0.9, 1]; a second map for tracks whose emulated d0 changes window.
    hits, mask   unchanged.
  CHANGED (hit-choice part; also refit flips), with the observed share p per cell
    a donor from the cell's changed tracks, picked by u_rank, supplies everything together: residuals to truth over
    today's sigma (tracks with a particle: the residual is replaced) or changes over today's sigma (tracks without),
    covariance ratios, the changes of the head outputs, and the arm's own hit counts, layer mask and chi2-like inputs.

One set of draws per track (u_state, u_rank and the normals), hashed from (run, lumi, event, track) and NOT from the
allocation: allocations stay comparable track by track.

Cells use only what today's production stores: today's reconstructed pT (2-3, 3-5, >= 5 GeV), |eta| (0-1.1, 1.1-1.8,
>= 1.8), window (prompt |d0| <= 0.02 cm, displaced, or "not refit today"), today's largest window multiplicity
(<= 1, >= 2), today's clean-score tercile, and whether the OT track has a particle. Thin cells fall back by dropping,
in this order: tercile, multiplicity, pT, |eta|.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

PARAMS = ("rInv", "phi0", "tanL", "z0", "d0")          # the helix order of the stored states
KNOTS = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99])
PT_EDGES = np.array([3.0, 5.0])
ETA_EDGES = np.array([1.1, 1.8])
D0_SPLIT = 0.02                                        # cm
TAIL_EDGES = np.array([0.5, 0.9])                      # |2 u_rank - 1| intervals of the score residual
N_DONORS = 128
MIN_P, MIN_LAMBDA, MIN_KERNEL, MIN_DONORS, MIN_MAP = 1000, 400, 2000, 200, 400
HEADS = ("clean", "d0", "z0", "tanl", "rinv", "phi")
# head inputs 14-38 as stored (ext_scan "inputs"): the columns the parametrization emits
IN = {"spixRefitPerformed": 5, "spixNCrossings": 7, "spixNAcceptedHits": 8, "spixLayerHitMask": 9, "spixMaxWindowMult": 10,
      "spixChi2IncXTot": 11, "spixChi2IncYTot": 12, "spixChi2IncAlphaTot": 13, "spixChi2ITAtRefit": 15,
      "spixShiftChi2": 16, "spixLogDetRatio": 17}
COUNTS = ("spixRefitPerformed", "spixNCrossings", "spixNAcceptedHits", "spixLayerHitMask", "spixMaxWindowMult")
CHI2 = ("spixChi2IncXTot", "spixChi2IncYTot", "spixChi2IncAlphaTot", "spixChi2ITAtRefit", "spixShiftChi2", "spixLogDetRatio")
SCORES = tuple(f"head_{h}" for h in HEADS) + CHI2
MISSING = -999.0
# copula partners: parameter -> (the parameter whose normal draw it follows)
PARTNER = {"z0": "d0", "tanL": "z0", "rInv": "d0", "phi0": "d0"}
DRAW_ORDER = ("d0", "z0", "tanL", "rInv", "phi0")
U64 = np.uint64


# ---------------------------------------------------------------- draws ----
def _mix(x):
    x = np.array(x, U64, copy=True)
    with np.errstate(over="ignore"):
        x ^= x >> U64(30)
        x *= U64(0xBF58476D1CE4E5B9)
        x ^= x >> U64(27)
        x *= U64(0x94D049BB133111EB)
        x ^= x >> U64(31)
    return x


def draws(run, lumi, event, track):
    """The per-track draws (stable, not allocation dependent): uniforms u_state, u_rank, u_score and standard
    normals n_z0, n_tanL, n_rInv, n_phi0. (The fit and the closure use these; a production may use any stable hash,
    e.g. correctionlib's hashprng, as the reference helper does.)"""
    h = np.full(len(np.atleast_1d(run)), 0x9E3779B97F4A7C15, U64)
    with np.errstate(over="ignore"):
        for i, p in enumerate((run, lumi, event, track)):
            h = _mix(h ^ (np.asarray(p).astype(U64) + U64(0x9E3779B97F4A7C15) * U64(i + 1)))
        u = lambda salt: (_mix(h ^ (U64(salt) * U64(0xD6E8FEB86659FD93))) >> U64(11)).astype(np.float64) * 2.0 ** -53
        out = {"u_state": u(1), "u_rank": u(2), "u_score": u(3)}
        for i, p in enumerate(("z0", "tanL", "rInv", "phi0")):
            out["n_" + p] = stats.norm.ppf(np.clip(u(10 + i), 1e-12, 1 - 1e-12))
    return out


# ---------------------------------------------------------------- today's tracks and cells ----
def natural(name, x):
    """A score in its natural variable: head outputs as they are (margins); chi2-like sums as log(x + 1e-3); the
    log-determinant ratio as it is. Missing (-999) stays NaN."""
    x = np.asarray(x, np.float64)
    miss = x <= -998
    if name.startswith("head_") or name == "spixLogDetRatio":
        y = x.copy()
    else:
        y = np.log(np.maximum(x, 0.0) + 1e-3)
    return np.where(miss, np.nan, y)


def unnatural(name, y):
    if name.startswith("head_") or name == "spixLogDetRatio":
        x = np.asarray(y, np.float64).copy()
    else:
        x = np.maximum(np.exp(y) - 1e-3, 0.0)
    return np.where(np.isfinite(y), x, MISSING)


def scores_of(inputs, margins):
    """{score name: natural variable} from the stored head inputs 14-38 and the six margins."""
    out = {f"head_{h}": natural("head_", margins[:, i]) for i, h in enumerate(HEADS)}
    for c in CHI2:
        out[c] = natural(c, inputs[:, IN[c]])
    return out


def cells(today, terciles):
    """Cell indices of today's tracks. today: {"a" (n,5), "refit", "inputs" (n,25), "margins" (n,6), "matched"};
    terciles: the two edges of today's clean margin."""
    a = today["a"]
    kpt = today["kpt"]
    pt = kpt / np.maximum(np.abs(a[:, 0]), 1e-12)
    eta = np.abs(np.arcsinh(a[:, 2]))
    window = np.where(today["refit"], (np.abs(a[:, 4]) > D0_SPLIT).astype(int), 2)
    return {"ipt": np.searchsorted(PT_EDGES, pt, side="right"), "ieta": np.searchsorted(ETA_EDGES, eta, side="right"),
            "window": window, "mult": (today["inputs"][:, IN["spixMaxWindowMult"]] >= 2).astype(int),
            "tercile": np.searchsorted(terciles, today["margins"][:, 0], side="right"),
            "matched": today["matched"].astype(int)}


CELL_AXES = (("matched", 2), ("window", 3), ("ieta", 3), ("ipt", 3), ("mult", 2), ("tercile", 3))
FALLBACK = ("tercile", "mult", "ipt", "ieta")         # dropped in this order when a cell is thin


def pooled_table(C, sel, value_fn, min_n, axes=CELL_AXES, drop=FALLBACK, shape_tail=()):
    """A dense table over `axes`: each cell holds value_fn(track mask) of the smallest pooling of that cell with at
    least min_n selected tracks (axes dropped in the order `drop`; the last fallback keeps what is left)."""
    names = [n for n, _ in axes]
    dims = [k for _, k in axes]
    out = np.zeros(dims + list(shape_tail))
    level = np.zeros(dims, np.int8)
    for idx in np.ndindex(*dims):
        keep = list(names)
        for step in range(len(drop) + 1):
            m = sel.copy()
            for n_, i in zip(names, idx):
                if n_ in keep:
                    m &= C[n_] == i
            if m.sum() >= min_n or step == len(drop) or not any(d in keep for d in drop[step:]):
                break
            if drop[step] in keep:
                keep.remove(drop[step])
        out[idx] = value_fn(m)
        level[idx] = len(names) - len(keep)
    return out, level


def lookup(table, C, axes):
    return table[tuple(C[n] for n, _ in axes)]


def inv_cdf(x):
    """The empirical inverse CDF of x at KNOTS."""
    x = x[np.isfinite(x)]
    return np.quantile(x, KNOTS) if len(x) >= 20 else np.zeros(len(KNOTS))


def eval_inv_cdf(q, u):
    """Piecewise-linear inverse CDF with values q (..., len(KNOTS)) at u, clamped at the end knots."""
    u = np.clip(u, KNOTS[0], KNOTS[-1])
    k = np.clip(np.searchsorted(KNOTS, u, side="right") - 1, 0, len(KNOTS) - 2)
    t = (u - KNOTS[k]) / (KNOTS[k + 1] - KNOTS[k])
    lo = np.take_along_axis(q, k[..., None], -1)[..., 0]
    hi = np.take_along_axis(q, (k + 1)[..., None], -1)[..., 0]
    return lo + t * (hi - lo)


def w68(x):
    x = np.abs(x[np.isfinite(x)])
    return float(np.quantile(x, 0.6827)) if len(x) else np.nan


def normal_score_corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return 0.0
    zx = stats.norm.ppf((stats.rankdata(x[ok]) - 0.5) / ok.sum())
    zy = stats.norm.ppf((stats.rankdata(y[ok]) - 0.5) / ok.sum())
    r = float(np.corrcoef(zx, zy)[0, 1])
    return r if abs(r) >= 0.1 else 0.0


def param_uniforms(D, rho, C):
    """The uniform draw of each parameter's eps: d0 from u_rank; the others through the Gaussian copula with their
    partner's normal draw: z = rho z_partner + sqrt(1 - rho^2) n_own."""
    z = {"d0": stats.norm.ppf(np.clip(D["u_rank"], 1e-12, 1 - 1e-12))}
    for p in DRAW_ORDER[1:]:
        r = rho[p][C["ieta"], np.minimum(C["window"], 1)]
        z[p] = r * z[PARTNER[p]] + np.sqrt(1 - r * r) * D["n_" + p]
    return {p: stats.norm.cdf(z[p]) for p in DRAW_ORDER}


KERNEL_AXES = (("ieta", 3), ("window", 2))
LAMBDA_AXES = (("window", 2), ("ieta", 3), ("ipt", 3))
MAP_AXES = (("window_arm", 2), ("ieta", 3))
DONOR_AXES = (("matched", 2), ("window", 3), ("ieta", 3))


def sigma_today(today):
    return np.sqrt(np.maximum(today["cov_diag"], 1e-30))


def wrap(dphi):
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def residual(a, truth):
    r = a - truth
    r[:, 1] = wrap(r[:, 1])
    return r


# ---------------------------------------------------------------- fit ----
def fit(today, arm, truth, D, terciles, sel):
    """The tables of one allocation from the tracks `sel` (bool). today / arm: {"a", "cov_diag", "refit", "inputs",
    "margins"} (+ arm: "same_clusters", "forced_a", "forced_cov_diag", "forced_refit"; today: "matched", "kpt");
    truth (n, 5): the particle's helix parameters (NaN without one); D: draws()."""
    C = cells(today, terciles)
    sg = sigma_today(today)
    T = {}
    changed = ~arm["same_clusters"]
    T["p"], T["p_level"] = pooled_table(C, sel, lambda m: float(changed[m].mean()) if m.any() else 0.0, MIN_P)
    # ---- precision part: tracks refit today whose forced refit ran
    both = sel & today["refit"] & arm["forced_refit"]
    dprec = (arm["forced_a"] - today["a"])
    dprec[:, 1] = wrap(dprec[:, 1])
    eps = dprec / sg
    Ck = dict(C, window=np.minimum(C["window"], 1))
    T["kernel"], T["rho"], T["lambda"], T["cov_scale"] = {}, {}, {}, {}
    for i, p in enumerate(PARAMS):
        T["kernel"][p], _ = pooled_table(Ck, both, lambda m, i=i: inv_cdf(eps[m, i]), MIN_KERNEL, KERNEL_AXES, ("window", "ieta"),
                                         (len(KNOTS),))
        ratio = arm["forced_cov_diag"][:, i] / np.maximum(today["cov_diag"][:, i], 1e-30)
        T["cov_scale"][p], _ = pooled_table(Ck, both, lambda m, r=ratio: float(np.median(r[m])) if m.any() else 1.0, MIN_LAMBDA,
                                            LAMBDA_AXES, ("ipt", "ieta"))
    for p in DRAW_ORDER[1:]:
        i, j = PARAMS.index(p), PARAMS.index(PARTNER[p])
        T["rho"][p], _ = pooled_table(Ck, both, lambda m, i=i, j=j: normal_score_corr(eps[m, i], eps[m, j]), MIN_KERNEL,
                                      KERNEL_AXES, ("window", "ieta"))
    U = param_uniforms(D, T["rho"], Ck)
    r_today, r_forced = residual(today["a"].copy(), truth), residual(arm["forced_a"].copy(), truth)
    has_truth = both & np.isfinite(truth).all(1)
    for i, p in enumerate(PARAMS):
        e = eval_inv_cdf(lookup(T["kernel"][p], Ck, KERNEL_AXES), U[p])

        def lam(m, i=i, e=e):
            if m.sum() < 30:
                return 1.0
            target = w68(r_forced[m, i])
            f = lambda L: w68(r_today[m, i] + L * sg[m, i] * e[m]) - target
            lo, hi = 0.0, 3.0
            if f(lo) >= 0:
                return 0.0
            if f(hi) <= 0:
                return hi
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
            return 0.5 * (lo + hi)
        T["lambda"][p], _ = pooled_table(Ck, has_truth, lam, MIN_LAMBDA, LAMBDA_AXES, ("ipt", "ieta"))
    # ---- scores in the same-cluster state: linear map in the natural variable + empirical residual
    same = sel & arm["same_clusters"] & today["refit"]
    s_today, s_arm = scores_of(today["inputs"], today["margins"]), scores_of(arm["inputs"], arm["margins"])
    Cm = dict(Ck, window_arm=(np.abs(arm["a"][:, 4]) > D0_SPLIT).astype(int))
    tail = np.searchsorted(TAIL_EDGES, np.abs(2 * D["u_rank"] - 1), side="right")
    T["map"], T["eta"] = {}, {}
    for s in SCORES:
        x, y = s_today[s], s_arm[s]
        ok = same & np.isfinite(x) & np.isfinite(y)

        def ols(m, x=x, y=y):
            if m.sum() < 20 or np.std(x[m]) < 1e-9:
                return np.array([float(np.mean(y[m] - x[m])) if m.any() else 0.0, 1.0, 0.0])
            b, a_ = np.polyfit(x[m], y[m], 1)
            return np.array([a_, b, float(np.std(y[m] - a_ - b * x[m]))])
        T["map"][s], _ = pooled_table(Cm, ok, ols, MIN_MAP, MAP_AXES, ("ieta",), (3,))
        co = lookup(T["map"][s], Cm, MAP_AXES)
        z = (y - co[:, 0] - co[:, 1] * x) / np.maximum(co[:, 2], 1e-9)
        T["eta"][s] = np.stack([inv_cdf(z[ok & (tail == k)]) for k in range(3)])
    # ---- changed state: donors per cell (the cell's changed tracks, a fixed random subset of N_DONORS)
    rng = np.random.default_rng(20261111)
    r_arm = residual(arm["a"].copy(), truth)
    delta = arm["a"] - today["a"]
    delta[:, 1] = wrap(delta[:, 1])
    cols = {"matched_ratio": np.where(np.isfinite(r_arm), r_arm, 0.0) / sg, "delta_ratio": delta / sg,
            "cov_ratio": arm["cov_diag"] / np.maximum(today["cov_diag"], 1e-30),
            "d_margin": arm["margins"] - today["margins"],
            "counts": np.stack([arm["inputs"][:, IN[c]] for c in COUNTS], 1),
            "chi2": np.stack([arm["inputs"][:, IN[c]] for c in CHI2], 1)}
    T["donors"] = {k: np.zeros((2, 3, 3, N_DONORS, v.shape[1])) for k, v in cols.items()}
    T["donor_n"] = np.zeros((2, 3, 3), np.int64)
    pool = sel & changed
    for idx in np.ndindex(2, 3, 3):
        keep = ["matched", "window", "ieta"]
        for step in range(3):
            m = pool.copy()
            for n_, i in zip(("matched", "window", "ieta"), idx):
                if n_ in keep:
                    m &= C[n_] == i
            if m.sum() >= MIN_DONORS or step == 2:
                break
            keep.remove(("ieta", "window")[step])
        rows = np.flatnonzero(m)
        T["donor_n"][idx] = len(rows)
        if not len(rows):
            rows = np.flatnonzero(pool)[:1] if pool.any() else np.zeros(1, np.int64)
        pick = rows[rng.integers(0, len(rows), N_DONORS)] if len(rows) > N_DONORS else np.resize(rows, N_DONORS)
        for k, v in cols.items():
            T["donors"][k][idx] = v[pick]
    return T


# ---------------------------------------------------------------- apply ----
def apply(T, today, truth, D, terciles):
    """The emulated arm for today's tracks: {"a", "cov_diag", "margins", "changed", <COUNTS and CHI2 columns>}."""
    C = cells(today, terciles)
    Ck = dict(C, window=np.minimum(C["window"], 1))
    sg = sigma_today(today)
    n = len(sg)
    changed = D["u_state"] < lookup(T["p"], C, CELL_AXES)
    refit_today = today["refit"]
    out = {"changed": changed}
    # same-cluster state
    U = param_uniforms(D, T["rho"], Ck)
    a = today["a"].astype(np.float64).copy()
    cov = today["cov_diag"].astype(np.float64).copy()
    for i, p in enumerate(PARAMS):
        e = eval_inv_cdf(lookup(T["kernel"][p], Ck, KERNEL_AXES), U[p])
        L = lookup(T["lambda"][p], Ck, LAMBDA_AXES)
        a[:, i] += np.where(refit_today, L * sg[:, i] * e, 0.0)
        cov[:, i] *= np.where(refit_today, lookup(T["cov_scale"][p], Ck, LAMBDA_AXES), 1.0)
    s_today = scores_of(today["inputs"], today["margins"])
    Cm = dict(Ck, window_arm=(np.abs(a[:, 4]) > D0_SPLIT).astype(int))
    tail = np.searchsorted(TAIL_EDGES, np.abs(2 * D["u_rank"] - 1), side="right")
    sc = {}
    for s in SCORES:
        co = lookup(T["map"][s], Cm, MAP_AXES)
        eta = eval_inv_cdf(T["eta"][s][tail], D["u_score"])
        y = co[:, 0] + co[:, 1] * s_today[s] + co[:, 2] * eta
        sc[s] = np.where(refit_today, y, s_today[s])             # a track not refit today keeps its values
    counts = {c: today["inputs"][:, IN[c]].astype(np.float64).copy() for c in COUNTS}
    # changed state: the donor's values
    ch = np.flatnonzero(changed)
    if len(ch):
        k = np.minimum((D["u_rank"][ch] * N_DONORS).astype(int), N_DONORS - 1)
        cell = (C["matched"][ch], C["window"][ch], C["ieta"][ch])
        g = lambda name: T["donors"][name][cell[0], cell[1], cell[2], k]
        has_truth = np.isfinite(truth[ch]).all(1)
        a_m = np.where(has_truth[:, None], truth[ch], 0.0) + sg[ch] * g("matched_ratio")
        a_u = today["a"][ch] + sg[ch] * g("delta_ratio")
        a[ch] = np.where(has_truth[:, None], a_m, a_u)
        cov[ch] = today["cov_diag"][ch] * g("cov_ratio")
        dm = g("d_margin")
        for i, h in enumerate(HEADS):
            sc[f"head_{h}"][ch] = today["margins"][ch, i] + dm[:, i]
        gc, gx = g("counts"), g("chi2")
        for i, c in enumerate(COUNTS):
            counts[c][ch] = gc[:, i]
        for i, c in enumerate(CHI2):
            sc[c][ch] = natural(c, gx[:, i])
    a[:, 1] = wrap(a[:, 1])
    out["a"], out["cov_diag"] = a, cov
    out["margins"] = np.stack([sc[f"head_{h}"] for h in HEADS], 1)
    out.update(counts)
    for c in CHI2:
        out[c] = unnatural(c, sc[c])
    return out
