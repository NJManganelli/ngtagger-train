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
    covariance   diagonal scaled by exp(a + b ln sigma_today + s eta) per cell (a straight-line fit of the log ratio
                 forced / today against ln sigma_today with its empirical scatter: an added quantization variance weighs
                 more on a precise track; clipped to 0.2-10,000); today's correlations kept.
    scores       (the six deployed-head outputs and the chi2-like inputs, in their natural variable)
                 s' = a + b s + sigma_s * eta, fitted on the tracks without a material hit-choice part; eta from the
                 empirical standardized residual, conditioned on |2 u_rank - 1| in [0, 0.5), [0.5, 0.9), [0.9, 1].
    hits, mask   unchanged.
  HIT-CHOICE PART, with the observed share p per cell: the tracks whose free result differs materially from the forced
  one (more than CHOICE_SIGMA of today's sigma in d0 or z0, or another refit flag or layer mask). A swap between the
  particle's two clusters on overlapping modules, which is most cluster changes, is below that and stays above.
    a donor from the cell's such tracks, picked by u_rank, supplies everything together: residuals to truth over
    today's sigma (tracks with a particle: the residual is replaced) or changes over today's sigma (tracks without),
    covariance ratios, the changes of the head outputs, and the arm's own hit counts, layer mask and chi2-like inputs.

One set of draws per track (u_state, u_rank and the normals), hashed from (run, lumi, event, track) and NOT from the
allocation: allocations stay comparable track by track.

Cells use only what today's production stores: today's reconstructed pT (2-3, 3-5, >= 5 GeV), |eta| (0-1.1, 1.1-1.8,
>= 1.8), window (prompt |d0| <= 0.02 cm, displaced, or "not refit today"), today's largest window multiplicity
(<= 1, 2, 3-5, >= 6), today's clean-score tercile, whether the OT track has a particle and, for the state share of a
track with a particle, today's pull class (max of the |d0| and |z0| pulls against the particle: < 1, 1-3, >= 3): a track
that is off today is the one most likely to change clusters, and without this the same-cluster state inherits today's
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
NACC_EDGES = np.array([2.0, 3.0, 4.0])                 # today's accepted hits: <= 1, 2, 3, >= 4 (the score maps)
SHRINK = 50.0                                          # pseudo-count of the state share's shrinkage to the coarser cell
MULT_EDGES = np.array([2.0, 3.0, 6.0])                 # today's largest window multiplicity: <= 1, 2, 3-5, >= 6
ETA_EDGES = np.array([1.1, 1.8])
N_ETA = len(ETA_EDGES) + 1
D0_SPLIT = 0.02                                        # cm
TAIL_EDGES = np.array([0.5, 0.9])                      # |2 u_rank - 1| intervals of the score residual
N_DONORS = 128
LAMBDA_MAX = 30.0
# Choices compared on the first 200 scan events of AAAA (closure failures over five allocations; README of the
# parametrization): the donor's residual over today's sigma, rescaled by the receiving track's sigma (the donor's absolute
# residual was much worse); eps in units of the sigma the allocation adds on the track (today's sigma was worse: a track's
# noise and its new covariance then do not move together); three |eta| cells (six were worse at this sample size); the
# head outputs and covariance ratios of the same-cluster state from a map with an empirical residual (taking them
# together from a same-cluster donor track was worse).


def eps_unit(sg, cov_today, cov_arm):
    """(n,) the unit of the same-cluster change eps of one parameter: the sigma the allocation adds on this track,
    sqrt(|C_arm - C_today| + (0.05 sigma_today)^2), with C_arm from the covariance model."""
    return np.sqrt(np.abs(cov_arm - cov_today) + (0.05 * sg) ** 2)


COV_CLIP = (float(np.log(0.2)), float(np.log(1.0e4)))   # limits of ln(covariance ratio) in the same-cluster state
MIN_P, MIN_LAMBDA, MIN_KERNEL, MIN_DONORS, MIN_MAP = 1000, 150, 2000, 200, 400
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
SCORES = CHI2 + AGG                                   # mapped in the precision part (natural variable)
LINEAR = ("spixLogDetRatio", "h_nangle", "h_mean_wmult")     # natural variable = the value itself; the others log(x + 1e-3)
N_MASK = 12                                           # layer-mask classes of the covariance map: the 12 most frequent, and "other"
COV_SCATTER = True                                    # the covariance line's empirical scatter (False: deterministic)
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
    """The per-track draws (stable, not allocation dependent): uniforms u_state, u_rank, u_score, u_cov and standard
    normals n_z0, n_tanL, n_rInv, n_phi0. (The fit and the closure use these; a production may use any stable hash,
    e.g. correctionlib's hashprng, as the reference helper does.)"""
    h = np.full(len(np.atleast_1d(run)), 0x9E3779B97F4A7C15, U64)
    with np.errstate(over="ignore"):
        for i, p in enumerate((run, lumi, event, track)):
            h = _mix(h ^ (np.asarray(p).astype(U64) + U64(0x9E3779B97F4A7C15) * U64(i + 1)))
        u = lambda salt: (_mix(h ^ (U64(salt) * U64(0xD6E8FEB86659FD93))) >> U64(11)).astype(np.float64) * 2.0 ** -53
        out = {"u_state": u(1), "u_rank": u(2), "u_score": u(3), "u_cov": u(4)}
        for i, p in enumerate(("z0", "tanL", "rInv", "phi0")):
            out["n_" + p] = stats.norm.ppf(np.clip(u(10 + i), 1e-12, 1 - 1e-12))
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


def scores_of(inputs):
    """{name: natural variable} of the mapped inputs from the stored head inputs 14-38."""
    return {c: natural(c, inputs[:, IN[c]]) for c in SCORES}


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
            "nacc": np.searchsorted(NACC_EDGES, today["inputs"][:, IN["spixNAcceptedHits"]], side="right"),
            "maskc": mask_class(today["inputs"][:, IN["spixLayerHitMask"]], today["mask_classes"]),
            "matched": today["matched"].astype(int), "pull": np.searchsorted(PULL_EDGES, today_pull(today, truth), side="right")}


def mask_class(mask, classes):
    """Index of the layer mask among `classes` (sorted mask values), len(classes) for any other mask."""
    classes = np.asarray(classes, np.int64)
    m = np.asarray(mask).astype(np.int64)
    i = np.minimum(np.searchsorted(classes, m), len(classes) - 1)
    return np.where(classes[i] == m, i, len(classes))


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


def cov_model_of(T, p, cov_today, sg, Ck, D):
    """The precision part's covariance diagonal of parameter p: today's x exp(a + b ln sigma_today [+ s eta(u_cov)]),
    the exponent clipped to COV_CLIP; a, b, s per cell and layer-mask class; one u_cov per track for all five
    parameters (their ratios move together). COV_SCATTER False: the deterministic line alone."""
    cs = lookup(T["cov_scale"][p], Ck, COV_AXES)
    lr = cs[:, 0] + cs[:, 1] * np.log(sg)
    if COV_SCATTER:
        lr = lr + cs[:, 2] * eval_inv_cdf(np.broadcast_to(T["cov_eta"][p], (len(sg), len(KNOTS))), D["u_cov"])
    return cov_today * np.exp(np.clip(lr, COV_CLIP[0], COV_CLIP[1]))


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
# the covariance map also by today's layer mask (the hit pattern fixes most of the covariance's response)
COV_AXES = (("window", 2), ("ieta", N_ETA), ("ipt", 3), ("maskc", N_MASK + 1))
COV_DROP = ("ipt", "ieta", "maskc")
MAP_AXES = (("window_arm", 2), ("nacc", 4), ("ieta", N_ETA))
# donors are taken from the same pull class as the receiving track (a track that is off today and changes clusters is
# not, on average, repaired: without this the emulation repairs it too often and the prompt tails come out too small)
DONOR_AXES = (("matched", 2), ("window", 3), ("ieta", N_ETA), ("pull", 3))
DONOR_DROP = ("ieta", "window", "pull")                # thin donor cells pool in this order


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
    T["kernel"], T["rho"], T["lambda"], T["cov_scale"] = {}, {}, {}, {}
    for i, p in enumerate(PARAMS):
        T["kernel"][p], _ = pooled_table(Ck, both, lambda m, i=i: inv_cdf(eps[m, i]), MIN_KERNEL, KERNEL_AXES, ("window", "ieta"),
                                         (len(KNOTS),))
        lr = np.log(np.maximum(arm["forced_cov_diag"][:, i], 1e-30) / np.maximum(today["cov_diag"][:, i], 1e-30))
        ls = np.log(sg[:, i])

        def line(m, lr=lr, ls=ls):
            ok = m & np.isfinite(lr) & np.isfinite(ls)
            if ok.sum() < 30 or np.std(ls[ok]) < 1e-6:
                return np.array([float(np.median(lr[ok])) if ok.any() else 0.0, 0.0, 0.0])
            y_ = np.clip(lr[ok], COV_CLIP[0], COV_CLIP[1])
            b, a_ = np.polyfit(ls[ok], y_, 1)
            return np.array([a_, b, float(np.std(y_ - a_ - b * ls[ok]))])
        T["cov_scale"][p], _ = pooled_table(Ck, both, line, MIN_LAMBDA, COV_AXES, COV_DROP, (3,))
        co_ = lookup(T["cov_scale"][p], Ck, COV_AXES)
        zc = (np.clip(lr, COV_CLIP[0], COV_CLIP[1]) - co_[:, 0] - co_[:, 1] * ls) / np.maximum(co_[:, 2], 1e-9)
        T.setdefault("cov_eta", {})[p] = inv_cdf(zc[both & np.isfinite(zc) & (co_[:, 2] > 0)])
    for p in DRAW_ORDER[1:]:
        i, j = PARAMS.index(p), PARAMS.index(PARTNER[p])
        T["rho"][p], _ = pooled_table(Ck, both, lambda m, i=i, j=j: normal_score_corr(eps[m, i], eps[m, j]), MIN_KERNEL,
                                      KERNEL_AXES, ("window", "ieta"))
    U = param_uniforms(D, T["rho"], Ck)
    r_today, r_forced = residual(today["a"].copy(), truth), residual(arm["forced_a"].copy(), truth)
    has_truth = both & np.isfinite(truth).all(1)
    for i, p in enumerate(PARAMS):
        e = eval_inv_cdf(lookup(T["kernel"][p], Ck, KERNEL_AXES), U[p])
        cov_model = cov_model_of(T, p, today["cov_diag"][:, i], sg[:, i], Ck, D)
        unit_model = eps_unit(sg[:, i], today["cov_diag"][:, i], cov_model)

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
    # ---- scores in the same-cluster state: linear map in the natural variable + empirical residual
    same = sel & ~changed & today["refit"]
    s_today, s_arm = scores_of(today["inputs"]), scores_of(arm["inputs"])
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
        T["map"][s], _ = pooled_table(Cm, ok, ols, MIN_MAP, MAP_AXES, ("ieta", "nacc"), (3,))
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
            "counts": np.stack([arm["inputs"][:, IN[c]] for c in COUNTS], 1),
            "values": np.stack([arm["inputs"][:, IN[c]] for c in SCORES], 1)}
    dims = [k for _, k in DONOR_AXES]
    dnames = [n_ for n_, _ in DONOR_AXES]
    T["donors"] = {k: np.zeros(dims + [N_DONORS, v.shape[1]]) for k, v in cols.items()}
    T["donor_n"] = np.zeros(dims, np.int64)
    pool = sel & changed
    for idx in np.ndindex(*dims):
        keep = list(dnames)
        for step in range(len(DONOR_DROP) + 1):
            m = pool.copy()
            for n_, i in zip(dnames, idx):
                if n_ in keep:
                    m &= C[n_] == i
            if m.sum() >= MIN_DONORS or step == len(DONOR_DROP):
                break
            keep.remove(DONOR_DROP[step])
        rows = np.flatnonzero(m)
        T["donor_n"][idx] = len(rows)
        if not len(rows):
            rows = np.flatnonzero(pool)[:1] if pool.any() else np.zeros(1, np.int64)
        pick = rows[rng.integers(0, len(rows), N_DONORS)] if len(rows) > N_DONORS else np.resize(rows, N_DONORS)
        for k, v in cols.items():
            T["donors"][k][idx] = v[pick]
    return T


# ---------------------------------------------------------------- apply ----
def apply(T, today, truth, D, terciles, heads):
    """The emulated quantization scenario for today's tracks: {"a", "cov_diag", "inputs" (n, 25: the heads' inputs
    14-38), "margins" (the real head models `heads` on the emulated inputs), "changed", and the COUNTS / CHI2 columns}."""
    C = cells(today, terciles, truth)
    Ck = dict(C, window=np.minimum(C["window"], 1))
    sg = sigma_today(today)
    changed = D["u_state"] < lookup(T["p"], C, CELL_AXES)
    refit_today = today["refit"]
    out = {"changed": changed}
    # precision part
    U = param_uniforms(D, T["rho"], Ck)
    a = today["a"].astype(np.float64).copy()
    cov = today["cov_diag"].astype(np.float64).copy()
    for i, p in enumerate(PARAMS):
        e = eval_inv_cdf(lookup(T["kernel"][p], Ck, KERNEL_AXES), U[p])
        L = lookup(T["lambda"][p], Ck, LAMBDA_AXES)
        cov_new = cov_model_of(T, p, today["cov_diag"][:, i], sg[:, i], Ck, D)
        a[:, i] += np.where(refit_today, L * eps_unit(sg[:, i], today["cov_diag"][:, i], cov_new) * e, 0.0)
        cov[:, i] = np.where(refit_today, cov_new, cov[:, i])
    s_today = scores_of(today["inputs"])
    Cm = dict(Ck, window_arm=(np.abs(a[:, 4]) > D0_SPLIT).astype(int))
    tail = np.searchsorted(TAIL_EDGES, np.abs(2 * D["u_rank"] - 1), side="right")
    values = {}
    for s in SCORES:
        co = lookup(T["map"][s], Cm, MAP_AXES)
        eta = eval_inv_cdf(T["eta"][s][tail], D["u_score"])
        y = co[:, 0] + co[:, 1] * s_today[s] + co[:, 2] * eta
        values[s] = unnatural(s, np.where(refit_today, y, s_today[s]))    # a track not refit today keeps its values
    counts = {c: today["inputs"][:, IN[c]].astype(np.float64).copy() for c in COUNTS}
    # hit-choice part: everything from one donor
    ch = np.flatnonzero(changed)
    if len(ch):
        k = np.minimum((D["u_rank"][ch] * N_DONORS).astype(int), N_DONORS - 1)
        cell = tuple(C[n_][ch] for n_, _ in DONOR_AXES)
        g = lambda name: T["donors"][name][cell + (k,)]
        has_truth = np.isfinite(truth[ch]).all(1)
        a_m = np.where(has_truth[:, None], truth[ch], 0.0) + sg[ch] * g("matched_ratio")
        a_u = today["a"][ch] + sg[ch] * g("delta_ratio")
        a[ch] = np.where(has_truth[:, None], a_m, a_u)
        cov[ch] = today["cov_diag"][ch] * g("cov_ratio")
        gc, gv = g("counts"), g("values")
        for i, c in enumerate(COUNTS):
            counts[c][ch] = gc[:, i]
        for i, c in enumerate(SCORES):
            values[c][ch] = gv[:, i]
    a[:, 1] = wrap(a[:, 1])
    inputs, a = assemble_inputs(today, a, counts, values, today["kpt"])
    not_refit = counts["spixRefitPerformed"] <= 0
    cov[not_refit] = today["ot_cov_diag"][not_refit]
    out.update(a=a, cov_diag=cov, inputs=inputs, margins=head_margins(heads, today, inputs, a[:, 4]))
    out.update(counts)
    out.update({c: values[c] for c in CHI2})
    return out
