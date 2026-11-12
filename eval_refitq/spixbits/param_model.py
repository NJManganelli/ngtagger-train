"""The parametrization of PLAN.md section 7b (rules: reviews/stats-review-02-parametrization.md): from today's
extended-collection tracks ("implemented now") to the tracks a shipped bit allocation delivers, per scenario.

No fitted mixture. Per allocation the extended scan gives, track by track, today's result, the arm's result (free)
and the arm's refit on today's clusters (forced): precision part = forced - today, hit-choice part = free - forced.

  PRECISION PART, for every track that today refits
    parameters   q' = q + lambda * s_add(q) * eps_q, with s_add the sigma the allocation adds on this track
                 (sqrt(|C_arm - C_today| + (0.05 sigma_today)^2), C_arm from the covariance model below) and eps_q from
                 the empirical inverse CDF of (forced - today) / s_add over all forced refits (knots KNOTS, clamped);
                 lambda per cell from a root find so that the 68 % width of the residual to truth equals the forced
                 refit's. d0 is drawn with u_rank; z0 through a Gaussian copula with d0's draw, tanL with z0's, 1/R and
                 phi0 with d0's (normal-score correlations).
    hits, mask   unchanged.
  HIT-CHOICE PART, with the observed share p per cell: the tracks whose free result differs materially from the forced
  one (more than CHOICE_SIGMA of today's sigma in d0 or z0, or another refit flag or layer mask). A swap between the
  particle's two clusters on overlapping modules, which is most cluster changes, is below that and stays above.
    a donor track supplies together: residuals to truth over today's sigma (tracks with a particle: the residual is
    replaced) or changes over today's sigma (tracks without), and the arm's own refit flag, hit counts, layer mask and
    largest window multiplicity. The donor is one of the K_DONORS hit-choice tracks nearest to the receiving track
    (same class: particle or not x today's window; distance in today's ln pT, |eta|, ln sigma(d0), ln sigma(z0), pull
    against the particle, window multiplicity, accepted hits, clean margin), picked by u_rank.
  BOTH PARTS: the covariance diagonal and the heads' continuous inputs are functions of the track (a Kalman covariance;
  sums over its hits), so they are emulated as functions of the track too, by regression trees (MAPS below):
    covariance   ln(C_arm / C_today) per parameter from today's track (the heads' 38 inputs, parameters, ln covariance,
                 layer-mask bits) and the emulated hit counts and layer mask;
    head inputs  the chi2-like inputs and the hit aggregates (natural variable) from the same and the emulated
                 parameters and covariance (how far the track moved, from the OT track and from today's refit);
    each with a second tree model for the size of its residual and the measured distribution of the standardized
    residual, the residuals of the columns of a block tied by a Gaussian copula; a tree classifier gives the chance
    that an input is missing. The models are trained on the arm's own (free) results; mean models on two halves of the
    events, the residuals taken out of fold.
  The six head outputs are not emulated: the head models are evaluated on the emulated inputs.

One set of draws per track (u_state, u_rank, the normals and the per-column draws of the maps), hashed from (run, lumi, event, track) and NOT from the
allocation: allocations stay comparable track by track.

Cells use only what today's production stores: today's reconstructed pT (2-3, 3-5, >= 5 GeV), |eta| (0-1.1, 1.1-1.8,
>= 1.8), window (prompt |d0| <= 0.02 cm, displaced, or "not refit today"), today's largest window multiplicity
(<= 1, 2, 3-5, >= 6), today's clean-score tercile, whether the OT track has a particle and, for the state share of a
track with a particle, today's pull class (max of the |d0| and |z0| pulls against the particle: < 1, 1-3, >= 3): a track
that is off today is the one most likely to change clusters, and without this the precision part inherits today's
worst tracks. The state share is shrunk towards coarser cells (pseudo-count 50) in the order pT, |eta|, tercile, pull,
multiplicity; the other tables fall back to pooled cells when thin.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

PARAMS = ("rInv", "phi0", "tanL", "z0", "d0")          # the helix order of the stored states
KNOTS = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99])
PT_EDGES = np.array([3.0, 5.0])
PULL_EDGES = np.array([1.0, 3.0])
SHRINK = 50.0                                          # pseudo-count of the state share's shrinkage to the coarser cell
MULT_EDGES = np.array([2.0, 3.0, 6.0])                 # today's largest window multiplicity: <= 1, 2, 3-5, >= 6
ETA_EDGES = np.array([1.1, 1.8])
N_ETA = len(ETA_EDGES) + 1
D0_SPLIT = 0.02                                        # cm
LAMBDA_MAX = 30.0
# Choices compared on the first 200 scan events of AAAA (closure failures over five allocations; README of the
# parametrization): the donor's residual over today's sigma, rescaled by the receiving track's sigma (the donor's absolute
# residual was much worse); eps in units of the sigma the allocation adds on the track (today's sigma was worse: a track's
# noise and its new covariance then do not move together); three |eta| cells (six were worse at this sample size).
# Covariance and head inputs, classifier AUC actual against emulated on the tagger's track inputs (alloc_4444, tolerance
# 0.62): head outputs from a table with a random residual 0.88; head inputs from per-cell linear maps and the real head
# models 0.77 (a deterministic covariance line: 0.78); the tree maps 0.62.


def eps_unit(sg, cov_today, cov_arm):
    """(n,) the unit of the precision-part change eps of one parameter: the sigma the allocation adds on this track,
    sqrt(|C_arm - C_today| + (0.05 sigma_today)^2), with C_arm from the covariance model."""
    return np.sqrt(np.abs(cov_arm - cov_today) + (0.05 * sg) ** 2)


MIN_P, MIN_LAMBDA, MIN_KERNEL, MIN_MAP = 1000, 150, 2000, 400   # MIN_MAP: below it a map is a constant
HEADS = ("clean", "d0", "z0", "tanl", "rinv", "phi")
# head inputs 14-38 as stored (ext_scan "inputs"): the columns the parametrization emits
IN = {"spixRefitPerformed": 5, "spixNCrossings": 7, "spixNAcceptedHits": 8, "spixLayerHitMask": 9, "spixMaxWindowMult": 10,
      "spixChi2IncXTot": 11, "spixChi2IncYTot": 12, "spixChi2IncAlphaTot": 13, "spixChi2ITAtRefit": 15,
      "spixShiftChi2": 16, "spixLogDetRatio": 17, "h_nangle": 18, "h_chi2_angle_per_cl": 19, "h_chi2_pos_per_hit": 20,
      "h_mean_wmult": 21, "h_max_angle_pull": 22, "h_second_angle_pull": 23, "h_max_pos_pull": 24}
COUNTS = ("spixRefitPerformed", "spixNCrossings", "spixNAcceptedHits", "spixLayerHitMask", "spixMaxWindowMult")
CHI2 = ("spixChi2IncXTot", "spixChi2IncYTot", "spixChi2IncAlphaTot", "spixChi2ITAtRefit", "spixShiftChi2", "spixLogDetRatio")
AGG = ("h_nangle", "h_chi2_angle_per_cl", "h_chi2_pos_per_hit", "h_mean_wmult", "h_max_angle_pull", "h_second_angle_pull",
       "h_max_pos_pull")                              # the heads' aggregates over the accepted hits
SCORES = CHI2 + AGG                                   # the mapped head inputs (natural variable)
LINEAR = ("spixLogDetRatio", "h_nangle", "h_mean_wmult")     # natural variable = the value itself; the others log(x + 1e-3)
N_BITS = 16                                           # layer-mask bits given to the maps as separate features
MAP_MEAN = dict(max_iter=100, max_depth=4, learning_rate=0.1, min_samples_leaf=40, random_state=0)
# the columns that are nearly functions of the track (the covariance diagonal; the log-determinant ratio; the mean window
# multiplicity) get larger models: out-of-fold residual of ln C(d0) 0.108 -> 0.067, of spixLogDetRatio 0.35 -> 0.22; the
# chi2-like sums and pulls are noise-dominated and gain nothing
MAP_BIG = dict(max_iter=200, max_depth=6, learning_rate=0.2, min_samples_leaf=10, random_state=0)
BIG_SCORES = ("spixLogDetRatio", "h_mean_wmult")
MAP_SCALE = dict(max_iter=40, max_depth=3, learning_rate=0.1, min_samples_leaf=100, random_state=0)
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
    """The per-track draws (stable, not allocation dependent): uniforms u_state, u_rank, u_miss (n, 13: one per mapped
    input); standard normals n_z0, n_tanL, n_rInv, n_phi0, n_cov (n, 5) and n_score (n, 13). (The fit and the closure use these; a production may use any stable hash,
    e.g. correctionlib's hashprng, as the reference helper does.)"""
    h = np.full(len(np.atleast_1d(run)), 0x9E3779B97F4A7C15, U64)
    with np.errstate(over="ignore"):
        for i, p in enumerate((run, lumi, event, track)):
            h = _mix(h ^ (np.asarray(p).astype(U64) + U64(0x9E3779B97F4A7C15) * U64(i + 1)))
        u = lambda salt: (_mix(h ^ (U64(salt) * U64(0xD6E8FEB86659FD93))) >> U64(11)).astype(np.float64) * 2.0 ** -53
        nor = lambda salt: stats.norm.ppf(np.clip(u(salt), 1e-12, 1 - 1e-12))
        out = {"u_state": u(1), "u_rank": u(2), "u_miss": np.stack([u(60 + j) for j in range(len(SCORES))], 1),
               "n_cov": np.stack([nor(20 + i) for i in range(5)], 1), "n_score": np.stack([nor(40 + j) for j in range(len(SCORES))], 1)}
        for i, p in enumerate(("z0", "tanL", "rInv", "phi0")):
            out["n_" + p] = nor(10 + i)
    return out


# ---------------------------------------------------------------- today's tracks and cells ----
def natural(name, x):
    """A mapped input in its natural variable: chi2-like sums and pulls as log(x + 1e-3); the log-determinant ratio,
    the number of hits with an angle and the mean window multiplicity as they are. Missing (-999) stays NaN."""
    x = np.asarray(x, np.float64)
    y = x.copy() if name in LINEAR else np.log(np.maximum(x, 0.0) + 1e-3)
    return np.where(x <= -998, np.nan, y)


def unnatural(name, y):
    x = np.asarray(y, np.float64).copy() if name in LINEAR else np.maximum(np.exp(y) - 1e-3, 0.0)
    return np.where(np.isfinite(y), x, MISSING)


def assemble_inputs(today, a, counts, values, kpt):
    """The heads' inputs 14-38 (n, 25) of emulated tracks: the first five follow from the emulated parameters and the OT
    track (today["ot_a"], today["ot_inputs"]); counts {COUNTS}; values {CHI2 + AGG} as stored values (-999 missing).
    A track whose refit does not run carries the OT track's parameters, as the producer writes it."""
    n = len(a)
    X = np.zeros((n, 25))
    refit = counts["spixRefitPerformed"] > 0
    aa = np.where(refit[:, None], a, today["ot_a"])
    inv_pt = 1.0 / np.maximum(kpt / np.maximum(np.abs(aa[:, 0]), 1e-12), 1e-3)
    ot_inv = today["ot_inputs"][:, 0].astype(np.float64)
    X[:, 0] = np.where(refit, inv_pt, ot_inv)
    X[:, 1] = aa[:, 2]
    X[:, 2] = np.abs(aa[:, 3] - today["ot_a"][:, 3])
    X[:, 3] = np.abs(aa[:, 2] - today["ot_a"][:, 2])
    X[:, 4] = X[:, 0] - ot_inv
    for c in COUNTS:
        X[:, IN[c]] = counts[c]
    X[:, 6] = today["inputs"][:, 6]                      # spixAnyWindowTruncated: today's value
    X[:, 14] = MISSING                                   # spixChi2ITAtOT: always missing in refit v3
    for c in SCORES:
        X[:, IN[c]] = values[c]
    X[:, IN["h_nangle"]] = np.clip(np.rint(X[:, IN["h_nangle"]]), 0, np.maximum(counts["spixNAcceptedHits"], 0))
    return X.astype(np.float32), aa


def head_margins(heads, today, inputs, d0):
    """The six head outputs: the real head models on (the OT track's 13 inputs, the 25 given inputs)."""
    return heads.margins(np.hstack([today["ot_inputs"].astype(np.float32), inputs.astype(np.float32)]), d0)


def cells(today, terciles, truth=None):
    """Cell indices of today's tracks. today: {"a" (n,5), "refit", "inputs" (n,25), "margins" (n,6), "matched"};
    terciles: the two edges of today's clean margin; truth (n, 5): the particle's parameters (NaN without one), for
    the pull class of the state share: max(|d0 pull|, |z0 pull|) of today's track against its particle in < 1, 1-3,
    >= 3 (0 for a track without a particle)."""
    a = today["a"]
    kpt = today["kpt"]
    pt = kpt / np.maximum(np.abs(a[:, 0]), 1e-12)
    eta = np.abs(np.arcsinh(a[:, 2]))
    window = np.where(today["refit"], (np.abs(a[:, 4]) > D0_SPLIT).astype(int), 2)
    return {"ipt": np.searchsorted(PT_EDGES, pt, side="right"), "ieta": np.searchsorted(ETA_EDGES, eta, side="right"),
            "window": window, "mult": np.searchsorted(MULT_EDGES, today["inputs"][:, IN["spixMaxWindowMult"]], side="right"),
            "tercile": np.searchsorted(terciles, today["margins"][:, 0], side="right"),
            "matched": today["matched"].astype(int), "pull": np.searchsorted(PULL_EDGES, today_pull(today, truth), side="right")}


def today_pull(today, truth):
    """max(|d0 pull|, |z0 pull|) of today's track against its particle; 0 without one."""
    if truth is None:
        return np.zeros(len(today["a"]))
    sg = np.sqrt(np.maximum(today["cov_diag"], 1e-30))
    with np.errstate(invalid="ignore"):
        p = np.maximum(np.abs(today["a"][:, 4] - truth[:, 4]) / sg[:, 4], np.abs(today["a"][:, 3] - truth[:, 3]) / sg[:, 3])
    return np.nan_to_num(p, nan=0.0)


def shrunk_share(C, sel, flag, axes, order, prior=SHRINK):
    """A dense table of the share of `flag` among the selected tracks per cell of `axes`, shrunk towards the coarser
    cell obtained by dropping the next axis of `order` (pseudo-count `prior`), recursively up to the global share:
    share = (k + prior * parent share) / (n + prior)."""
    names = [n for n, _ in axes]
    dims = [k for _, k in axes]
    idx = [C[n][sel] for n in names]
    y = flag[sel].astype(np.float64)
    parent = np.full(dims, y.mean() if len(y) else 0.0)
    kept = [n for n in names if n not in order]
    for nxt in list(order[::-1]):                           # from the coarsest level to the full cell
        kept.append(nxt)
        ax = [i for i, n in enumerate(names) if n in kept]
        shape = [dims[i] for i in ax]
        flat = np.ravel_multi_index([idx[i] for i in ax], shape) if ax else np.zeros(len(y), np.int64)
        size = int(np.prod(shape)) if ax else 1
        k_ = np.bincount(flat, weights=y, minlength=size).reshape(shape or [1])
        n_ = np.bincount(flat, minlength=size).reshape(shape or [1])
        view = [slice(None) if i in ax else None for i in range(len(dims))]
        parent = (k_[tuple(view)] + prior * parent) / (n_[tuple(view)] + prior)
        parent = np.broadcast_to(parent, dims).copy()
    return parent


CELL_AXES = (("matched", 2), ("window", 3), ("ieta", N_ETA), ("ipt", 3), ("mult", 4), ("tercile", 3), ("pull", 3))
# Thin cells drop axes in this order. (stats-review-02 proposed tercile, multiplicity, pT, |eta|; on the extended scan the
# largest window multiplicity is by far the strongest predictor of a track changing clusters (about 10 % at 1, 40-45 % at
# 2, 60-70 % from 3 on), so it is binned finer and kept longest.)
FALLBACK = ("ipt", "ieta", "tercile", "pull", "mult")


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


# ---------------------------------------------------------------- maps ----
class Trees:
    """One boosted tree model of scikit-learn (HistGradientBoosting; x <= threshold goes left), or a constant.
    raw(Z): the regression value, or the classifier's margin."""

    def __init__(self, sk=None, const=0.0):
        self.sk, self.const = sk, float(const)

    def raw(self, Z):
        if self.sk is None:
            return np.full(len(Z), self.const)
        return self.sk.decision_function(Z) if hasattr(self.sk, "decision_function") else self.sk.predict(Z)

    def arrays(self):
        """(base, nodes per tree, feature, threshold, left, right, value): the trees' nodes one after the other; feature
        -1 = leaf; left / right count within the tree. Thresholds stay float64: they are data values, and a rounded
        threshold sends the tracks sitting on it the other way."""
        if self.sk is None:
            z = np.zeros(0)
            return self.const, z.astype(np.int64), z.astype(np.int8), z, z.astype(np.uint8), z.astype(np.uint8), z.astype(np.float32)
        nodes = [p[0].nodes for p in self.sk._predictors]
        cat = lambda fn, dt: np.concatenate([fn(t) for t in nodes]).astype(dt)
        return (float(np.ravel(self.sk._baseline_prediction)[0]), np.array([len(t) for t in nodes], np.int64),
                cat(lambda t: np.where(t["is_leaf"], -1, t["feature_idx"].astype(np.int64)), np.int8),
                cat(lambda t: t["num_threshold"], np.float64), cat(lambda t: t["left"], np.uint8),
                cat(lambda t: t["right"], np.uint8), cat(lambda t: t["value"], np.float32))


def mask_bits(mask):
    return ((np.asarray(mask).astype(np.int64)[:, None] >> np.arange(N_BITS)) & 1).astype(np.float64)


def features_cov(today, counts):
    """The covariance maps' inputs: today's track (the heads' 38 inputs, ln covariance diagonal, parameters, layer-mask
    bits) and the emulated structure `counts` (n, 6: head inputs 19-24 = spixRefitPerformed ... spixMaxWindowMult) with
    its layer-mask bits."""
    return np.column_stack([today["ot_inputs"], today["inputs"], np.log(np.maximum(today["cov_diag"], 1e-30)), today["a"],
                            mask_bits(today["inputs"][:, IN["spixLayerHitMask"]]), counts, mask_bits(counts[:, 4])])


def features_scores(today, counts, a, cov):
    """The head-input maps' inputs: those of the covariance maps and the emulated parameters and covariance: the change
    from the OT track, the change from today's refit over today's sigma, ln covariance and its change, the parameters,
    the change from the OT track over the OT sigma."""
    d_ot = a - today["ot_a"]
    d_ot[:, 1] = wrap(d_ot[:, 1])
    d_today = a - today["a"]
    d_today[:, 1] = wrap(d_today[:, 1])
    lc, lc_today = np.log(np.maximum(cov, 1e-30)), np.log(np.maximum(today["cov_diag"], 1e-30))
    return np.column_stack([features_cov(today, counts), d_ot, np.abs(d_today) / sigma_today(today), lc, lc - lc_today, a,
                            np.abs(d_ot) / np.sqrt(np.maximum(today["ot_cov_diag"], 1e-30))])


def fit_column(Z, y, half, kw):
    """One mapped column: two mean models, each trained on one half of the events (their average is the map), the
    out-of-fold residual, a model of ln |residual| and the quantiles of the standardized residual."""
    from sklearn.ensemble import HistGradientBoostingRegressor as HR
    if len(y) < MIN_MAP or min((half == 0).sum(), (half == 1).sum()) < MIN_MAP // 4:
        med = float(np.median(y)) if len(y) else 0.0
        r = y - med
        mean, scale = [Trees(const=med)] * 2, Trees(const=float(np.log(np.median(np.abs(r)) + 1e-4)) if len(y) else 0.0)
    else:
        mean = [Trees(HR(**kw).fit(Z[half == k], y[half == k])) for k in (0, 1)]
        r = y.astype(np.float64).copy()
        for k in (0, 1):
            r[half == k] -= mean[1 - k].raw(Z[half == k])
        scale = Trees(HR(**MAP_SCALE).fit(Z, np.log(np.abs(r) + 1e-4)))
    e = r / np.exp(scale.raw(Z)) if len(y) else r
    return {"mean": mean, "scale": scale, "q": inv_cdf(e), "lo": float(y.min()) if len(y) else 0.0,
            "hi": float(y.max()) if len(y) else 0.0}, e


def fit_block(Z, Y, ok, half, big):
    """The maps of the columns of Y (NaN = missing, left out of that column; big[j]: the larger model) on the tracks `ok`, and the Cholesky factor
    of the normal-score correlation of their standardized residuals (pairwise complete, made positive definite)."""
    cols, E = [], np.full(Y.shape, np.nan)
    for j in range(Y.shape[1]):
        m = ok & np.isfinite(Y[:, j])
        c, e = fit_column(Z[m], Y[m, j], half[m], MAP_BIG if big[j] else MAP_MEAN)
        cols.append(c)
        if m.sum() >= 50:
            E[m, j] = stats.norm.ppf((stats.rankdata(e) - 0.5) / len(e))
    C = np.eye(Y.shape[1])
    for i in range(Y.shape[1]):
        for j in range(i):
            m = np.isfinite(E[:, i]) & np.isfinite(E[:, j])
            if m.sum() >= 200:
                r = np.corrcoef(E[m, i], E[m, j])[0, 1] if min(np.std(E[m, i]), np.std(E[m, j])) > 1e-9 else 0.0
                C[i, j] = C[j, i] = r if np.isfinite(r) else 0.0       # a constant column (an angle that is not sent) has none
    w, v = np.linalg.eigh(C)
    C = (v * np.maximum(w, 1e-4)) @ v.T
    d = np.sqrt(np.diag(C))
    return {"cols": cols, "chol": np.linalg.cholesky(C / d[:, None] / d[None, :])}


def draw_block(B, Z, normals):
    """(n, columns): map + residual scale x the standardized residual at the copula's draw, kept within the fitted range."""
    u = stats.norm.cdf(normals @ B["chol"].T)
    out = []
    for j, c in enumerate(B["cols"]):
        mu = 0.5 * (c["mean"][0].raw(Z) + c["mean"][1].raw(Z))
        y = mu + np.exp(c["scale"].raw(Z)) * eval_inv_cdf(np.broadcast_to(c["q"], (len(Z), len(KNOTS))), u[:, j])
        out.append(np.clip(y, c["lo"], c["hi"]))
    return np.column_stack(out)


def emulated_cov(T, today, counts, D):
    """(n, 5) covariance diagonal for the emulated structure: today's x exp(map) where the refit runs, else the OT track's."""
    lr = draw_block(T["cov_map"], features_cov(today, counts), D["n_cov"])
    return np.where(counts[:, :1] > 0, today["cov_diag"] * np.exp(lr), today["ot_cov_diag"])


def emulated_values(T, today, counts, a, cov, D):
    """{mapped input: stored value (-999 missing)} for the emulated structure, parameters and covariance."""
    Z = features_scores(today, counts, a, cov)
    Y = draw_block(T["score_map"], Z, D["n_score"])
    out = {}
    for j, c in enumerate(SCORES):
        p_miss = 1.0 / (1.0 + np.exp(-T["missing"][j].raw(Z)))
        out[c] = unnatural(c, np.where((D["u_miss"][:, j] < p_miss) | (counts[:, 0] <= 0), np.nan, Y[:, j]))
    return out


CHOICE_SIGMA = 1.0


def choice_changed(today, arm):
    """The hit-choice part of a track is material: the arm's free result differs from its refit on today's clusters
    (forced) by more than CHOICE_SIGMA of today's sigma in d0 or z0, or its refit flag or layer mask differ. A swap
    between the particle's two clusters on overlapping modules (most cluster changes) moves the track by less and
    stays in the precision part, where the track keeps its identity. (Code review round 2. Trial on 200 events of AAAA,
    five allocations: the share of tracks sent to the donor part falls from 29-43 % with "any cluster differs" to 14-35 %;
    closure failures 9 / 11 / 10 at thresholds 0 / 0.3 / 1 sigma; adding the donor's free - forced on top of the precision
    part instead of replacing the residual gave 30.)"""
    sg = sigma_today(today)
    d = np.abs(arm["a"] - arm["forced_a"]) / sg
    return (d[:, 3] > CHOICE_SIGMA) | (d[:, 4] > CHOICE_SIGMA) | (arm["refit"] != arm["forced_refit"]) \
        | (arm["inputs"][:, IN["spixLayerHitMask"]] != today["inputs"][:, IN["spixLayerHitMask"]])


KERNEL_AXES = (("ieta", N_ETA), ("window", 2))
LAMBDA_AXES = (("window", 2), ("ieta", N_ETA), ("ipt", 3))
# Donors: nearest neighbours in today's track quantities, the pull against the particle among them (a track that is off
# today and changes clusters is not, on average, repaired). Against a random donor from a cell of (particle, window, |eta|,
# pull class): classifier AUC on the tagger's inputs 0.657 -> 0.633 (alloc_4444_wholesensor, 35 % of tracks in this part),
# 0.597 -> 0.588 (alloc_4444); no width / tail finding either way (200 events).
K_DONORS, MAX_POOL = 4, 8000                           # MAX_POOL: donors kept per quantization scenario (a random subset)


def donor_features(today, truth):
    """What a hit-choice track and its donor are matched on (today's track): ln pT, |eta|, ln sigma(d0), ln sigma(z0),
    ln(1 + pull against the particle), the largest window multiplicity (capped at 8), accepted hits, clean margin."""
    a, sg = today["a"], sigma_today(today)
    return np.column_stack([np.log(today["kpt"] / np.maximum(np.abs(a[:, 0]), 1e-12)), np.abs(np.arcsinh(a[:, 2])), np.log(sg[:, 4]),
                            np.log(sg[:, 3]), np.log1p(today_pull(today, truth)), np.minimum(today["inputs"][:, IN["spixMaxWindowMult"]], 8.0),
                            today["inputs"][:, IN["spixNAcceptedHits"]], today["margins"][:, 0]])


def nearest_donors(P, cls, F, u):
    """Index into the pool P of each track's donor: among its K_DONORS nearest pool tracks of the same class (particle or
    not x today's window; every pool track when the class is thin) in the features over their pool sd, the one picked by u."""
    from scipy.spatial import cKDTree
    pick = np.zeros(len(cls), np.int64)
    for c in np.unique(cls):
        r = np.flatnonzero(cls == c)
        p = np.flatnonzero(P["cls"] == c)
        if len(p) < 10 * K_DONORS:
            p = np.arange(len(P["cls"]))
        _, idx = cKDTree(P["feat"][p].astype(np.float64) / P["scale"]).query(F[r] / P["scale"].astype(np.float64), k=K_DONORS)
        pick[r] = p[idx[np.arange(len(r)), np.minimum((u[r] * K_DONORS).astype(int), K_DONORS - 1)]]
    return pick


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
    C = cells(today, terciles, truth)
    sg = sigma_today(today)
    T = {}
    changed = choice_changed(today, arm)
    T["p"] = shrunk_share(C, sel, changed, CELL_AXES, FALLBACK)
    # ---- precision part: tracks refit today whose forced refit ran
    both = sel & today["refit"] & arm["forced_refit"]
    dprec = (arm["forced_a"] - today["a"])
    dprec[:, 1] = wrap(dprec[:, 1])
    eps = dprec / np.stack([eps_unit(sg[:, i], today["cov_diag"][:, i], arm["forced_cov_diag"][:, i]) for i in range(5)], 1)
    Ck = dict(C, window=np.minimum(C["window"], 1))
    T["kernel"], T["rho"], T["lambda"] = {}, {}, {}
    for i, p in enumerate(PARAMS):
        T["kernel"][p], _ = pooled_table(Ck, both, lambda m, i=i: inv_cdf(eps[m, i]), MIN_KERNEL, KERNEL_AXES, ("window", "ieta"),
                                         (len(KNOTS),))
    for p in DRAW_ORDER[1:]:
        i, j = PARAMS.index(p), PARAMS.index(PARTNER[p])
        T["rho"][p], _ = pooled_table(Ck, both, lambda m, i=i, j=j: normal_score_corr(eps[m, i], eps[m, j]), MIN_KERNEL,
                                      KERNEL_AXES, ("window", "ieta"))
    # ---- the maps, trained on the arm's own results (every track whose refit runs in the arm)
    from sklearn.ensemble import HistGradientBoostingClassifier as HC
    okm = sel & (arm["inputs"][:, IN["spixRefitPerformed"]] > 0)
    c_arm = arm["inputs"][:, 5:11].astype(np.float64)
    lc = lambda c: np.log(np.maximum(c, 1e-30))
    T["cov_map"] = fit_block(features_cov(today, c_arm), lc(arm["cov_diag"]) - lc(today["cov_diag"]), okm, today["half"], [True] * 5)
    Zs = features_scores(today, c_arm, arm["a"].astype(np.float64), arm["cov_diag"])
    Ys = np.column_stack([natural(c, arm["inputs"][:, IN[c]]) for c in SCORES])
    T["score_map"] = fit_block(Zs, Ys, okm, today["half"], [c in BIG_SCORES for c in SCORES])
    T["missing"] = []
    for j in range(len(SCORES)):
        miss = ~np.isfinite(Ys[okm, j])
        share = float(miss.mean()) if okm.any() else 1.0
        if okm.sum() >= MIN_MAP and min(miss.sum(), (~miss).sum()) >= 20:
            T["missing"].append(Trees(HC(**MAP_SCALE).fit(Zs[okm], miss)))
        else:
            T["missing"].append(Trees(const=float(np.log(max(share, 1e-9) / max(1 - share, 1e-9)))))
    cov_same = emulated_cov(T, today, today["inputs"][:, 5:11].astype(np.float64), D)     # the structure kept: the precision part
    U = param_uniforms(D, T["rho"], Ck)
    r_today, r_forced = residual(today["a"].copy(), truth), residual(arm["forced_a"].copy(), truth)
    has_truth = both & np.isfinite(truth).all(1)
    for i, p in enumerate(PARAMS):
        e = eval_inv_cdf(lookup(T["kernel"][p], Ck, KERNEL_AXES), U[p])
        unit_model = eps_unit(sg[:, i], today["cov_diag"][:, i], cov_same[:, i])

        def lam(m, i=i, e=e, unit_model=unit_model):
            if m.sum() < 30:
                return 1.0
            target = w68(r_forced[m, i])
            f = lambda L: w68(r_today[m, i] + L * unit_model[m] * e[m]) - target
            lo, hi = 0.0, LAMBDA_MAX
            if f(lo) >= 0:
                return 0.0
            if f(hi) <= 0:
                return hi
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
            return 0.5 * (lo + hi)
        T["lambda"][p], _ = pooled_table(Ck, has_truth, lam, MIN_LAMBDA, LAMBDA_AXES, ("ipt", "ieta"))
    # ---- hit-choice part: the donor pool (float32, as shipped)
    rng = np.random.default_rng(20261111)
    r_arm = residual(arm["a"].copy(), truth)
    delta = arm["a"] - today["a"]
    delta[:, 1] = wrap(delta[:, 1])
    cols = {"matched_ratio": np.where(np.isfinite(r_arm), r_arm, 0.0) / sg, "delta_ratio": delta / sg,
            "counts": np.stack([arm["inputs"][:, IN[c]] for c in COUNTS], 1)}
    pool = np.flatnonzero(sel & changed)
    if len(pool) > MAX_POOL:
        pool = np.sort(rng.choice(pool, MAX_POOL, replace=False))
    F = donor_features(today, truth)
    f32 = lambda x: x.astype(np.float32)
    T["pool"] = {"cls": (C["matched"] * 3 + C["window"])[pool], "feat": f32(F[pool]),
                 "scale": f32(np.maximum(F[pool].std(0), 1e-6) if len(pool) else np.ones(F.shape[1])),
                 **{k: f32(v[pool]) for k, v in cols.items()}}
    return T


# ---------------------------------------------------------------- apply ----
def apply(T, today, truth, D, terciles, heads):
    """The emulated quantization scenario for today's tracks: {"a", "cov_diag", "inputs" (n, 25: the heads' inputs
    14-38), "margins" (the real head models `heads` on the emulated inputs), "changed", and the COUNTS / CHI2 columns}."""
    C = cells(today, terciles, truth)
    Ck = dict(C, window=np.minimum(C["window"], 1))
    sg = sigma_today(today)
    changed = D["u_state"] < lookup(T["p"], C, CELL_AXES)
    refit_today = today["refit"].astype(bool)
    # the structure: today's, or the donor's in the hit-choice part
    counts = {c: today["inputs"][:, IN[c]].astype(np.float64).copy() for c in COUNTS}
    ch = np.flatnonzero(changed)
    if len(ch):
        pick = nearest_donors(T["pool"], (C["matched"] * 3 + C["window"])[ch], donor_features(today, truth)[ch], D["u_rank"][ch])
        g = lambda name: T["pool"][name][pick]
        gc = g("counts")
        for i, c in enumerate(COUNTS):
            counts[c][ch] = gc[:, i]
    c6 = today["inputs"][:, 5:11].astype(np.float64).copy()
    for c in COUNTS:
        c6[:, IN[c] - 5] = counts[c]
    cov = emulated_cov(T, today, c6, D)
    # parameters: the precision part ...
    U = param_uniforms(D, T["rho"], Ck)
    a = today["a"].astype(np.float64).copy()
    for i, p in enumerate(PARAMS):
        e = eval_inv_cdf(lookup(T["kernel"][p], Ck, KERNEL_AXES), U[p])
        L = lookup(T["lambda"][p], Ck, LAMBDA_AXES)
        a[:, i] += np.where(refit_today, L * eps_unit(sg[:, i], today["cov_diag"][:, i], cov[:, i]) * e, 0.0)
    # ... and the hit-choice part: the donor's residual
    if len(ch):
        has_truth = np.isfinite(truth[ch]).all(1)
        a_m = np.where(has_truth[:, None], truth[ch], 0.0) + sg[ch] * g("matched_ratio")
        a_u = today["a"][ch] + sg[ch] * g("delta_ratio")
        a[ch] = np.where(has_truth[:, None], a_m, a_u)
    a[:, 1] = wrap(a[:, 1])
    a = np.where(c6[:, :1] > 0, a, today["ot_a"])         # a track whose refit does not run carries the OT track
    values = emulated_values(T, today, c6, a, cov, D)
    inputs, a = assemble_inputs(today, a, counts, values, today["kpt"])
    out = {"changed": changed, "a": a, "cov_diag": cov, "inputs": inputs, "margins": head_margins(heads, today, inputs, a[:, 4])}
    out.update(counts)
    out.update({c: values[c] for c in CHI2})
    return out
