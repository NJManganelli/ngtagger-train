"""The shared angle definitions (plan D.3 P0.3, C.2 P11; definition sheet v1 items frame.angles,
frame.module_orientation, frame.incidence_key, truth.class_T, angle.k_knob, angle.error_table,
angle.cut_level).

Representation (the census's, in angle space as the emulator and the knob work):
  alpha = wrap(globalPhi - globalClusterPhi): the azimuth of the cluster's direction relative to
          the radial at the cluster, in (-pi, pi]; per-cluster uncertainty sigGlobalClusterPhi.
  beta  = lambda = atan(globalClusterCotTheta): the dip angle, in (-pi/2, pi/2); uncertainty
          sigGlobalClusterCotTheta / (1 + cot^2) at the EMITTED cot (what a consumer has).
  Residuals are wrapped in the angle's own period: 2 pi for alpha, pi for beta (a direction is
  sensed outward from the beam line, so lambda past +90 deg re-enters at -90 deg; this matters
  on the discs, whose beta residuals reach tens of degrees).
  pull = residual / sigma. The truth reference is tpGlobalClusterPhi / tpGlobalClusterCotTheta.

Incidence key (frame.incidence_key): signed cot theta on the barrel, signed 1 / cot theta
(= tan theta, sign of pz) on the discs; disc rows are split by module facing. The emitted key
uses globalClusterCotTheta, the true key tpGlobalClusterCotTheta (standing in for a prediction).

Module-local conversion (frame.module_orientation): a module's orientation sign s =
sign(localCotBeta x cot theta), the same for every cluster on the module (derived from emitted
columns only, by majority per detId); a global-frame beta quantity q is s q in the module frame
of a barrel module and -s q on a disc (local_beta_sign: the module-frame beta angle,
atan(localCotBeta), follows lambda on the barrel and theta on a disc); on a disc s is the module
facing. Alpha has no orientation flip on the barrel (every barrel
module of canon_v3 has the local x along the global sense); no disc consumer works in the
module frame today, so the disc local-alpha relation is not defined here.

Improved angles (angle.k_knob, decision 10): on class T only, angle' = truth + k (angle - truth),
0 <= k <= 1; alpha clusters whose emitted and true angles are more than 90 deg apart keep their
angle (the census's rule); class C is untouched. Uncertainty modes: `fixed` keeps sigma, `scaled`
uses max(k, 0.05) sigma (for every cluster: a real system cannot know the class). In both modes
the subtracted table centre is k x the centre in absolute angle units.

Table (angle.error_table, sheet v2.7, P0.3 amendment 1e): per surface x signed incidence bin (x facing
on the discs), for alpha and beta, in both keys: the centre (the 25 %-trimmed mean of the pulls;
centre_of), the weighting width (the mean of the smallest 95 % of |pull - centre| / 0.7168;
weighting_width) and the quantiles of |pull - centre| at 95 / 97.5 / 99 / 99.7 % with the
distribution-free rank k = min(ceil(L (n + 1)), n) (quantile_order); the B.1 core width and clipped
mean are reported diagnostics only. Bins with fewer than 200 fit clusters merge with their neighbour
away from the plateau edges.

Coverage of a fitted quantile: the exact order-statistic null of the number of new values beyond the
fitted quantile (expected_coverage, fit_coverage_variance, exchangeable_coverage_variance,
coverage_z), used by the table's coverage test and reusable by every cutting consumer that fits a
per-bin quantile (angle.cut_level, consumer_quantile: the same rank).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from ngtagger.data.spix_nano import DISC_SURFACE_OFFSET, direction_usable
from ngtagger.definitions import SHEET_VERSION, item as _sheet_item
from ngtagger.definitions.estimators import core
from ngtagger.definitions.helix import KPT_CMSSW

# ---------------------------------------------------------------- the sheet's parameters
_ET = _sheet_item("angle.error_table")["params"]
_KK = _sheet_item("angle.k_knob")["params"]
_CL = _sheet_item("angle.cut_level")["params"]

BARREL = (1, 2, 3, 4)
DISCS = (11, 12, 13, 14)
SURFACES = BARREL + DISCS
ANGLES = ("alpha", "beta")
KEYS = ("emitted", "true")
BARREL_ABS_EDGES = tuple(float(x) for x in _ET["barrel_signed_cot_edges"])
DISC_ABS_EDGES = tuple(float(x) for x in _ET["disc_signed_tan_edges_per_facing"])
LEVELS = tuple(float(x) for x in _ET["quantile_levels"])
MIN_FIT = int(_ET["min_fit_clusters_per_bin"])
PT_MIN = float(_ET["pt_min_gev"])
FIT_FILES = tuple(int(x) for x in _ET["fit_files"])
TEST_FILES = tuple(int(x) for x in _ET["test_files"])
K_FLOOR = float(_KK["scaled_floor"])
K_MODES = tuple(_KK["modes"])
CUT_LEVEL = float(_CL["level"])
if _ET["centre"] != "trimmed_mean_25":
    raise ValueError(f"angle.error_table centre {_ET['centre']!r}: this module implements trimmed_mean_25")
WIDTH_FRACTION = float(_ET["weighting_width"]["lower_fraction"])
WIDTH_DIVISOR = float(_ET["weighting_width"]["divisor"])
# |key| of the plateau edges: 27 deg (PixelAV, |cot| 0.5) and 34 deg (Phase-2, |eta| 0.62, |cot| 0.67)
# (plan B.4); the merge rule keeps them resolved. The disc incidence variable tan theta steps at
# the same values (B.4).
PLATEAU_EDGES = (0.5, 0.67)
TABLE_SCHEMA = "sp-angle-table/3"      # /3 (sheet v2.7): trimmed-mean centre and width, rank ceil(L (n + 1))

SURFACE_NAMES = {s: (f"ID{s - DISC_SURFACE_OFFSET}" if s > DISC_SURFACE_OFFSET else f"IL{s}") for s in SURFACES}


# ---------------------------------------------------------------- class T, validity
def class_T(tp_emu_class):
    """truth.class_T: tpEmuClass == 1 (the TP helix passes within 1 cm; angle = truth + 1.5 x a
    PixelAV residual). Truth: never a decision input."""
    return np.asarray(tp_emu_class) == 1


def usable(cols):
    """The cluster's global direction and its uncertainties may be used: both local angles pass
    frame.angles (has-flag and |cot| <= 12; spix_nano.direction_usable) and both sigmas are
    finite and positive. An explicit validity mask (estimator.sentinels), never a magnitude cut
    on the residual."""
    ok = np.asarray(direction_usable(cols), bool)
    for c in ("sigGlobalClusterPhi", "sigGlobalClusterCotTheta"):
        s = np.asarray(cols[c])
        ok &= np.isfinite(s) & (s > 0)
    return ok


# ---------------------------------------------------------------- representation
def wrap_2pi(a):
    return (np.asarray(a, np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_pi(a):
    return (np.asarray(a, np.float64) + 0.5 * np.pi) % np.pi - 0.5 * np.pi


WRAP = {"alpha": wrap_2pi, "beta": wrap_pi}


def alpha_of(global_phi, dir_phi):
    """alpha: azimuth of the direction relative to the radial at the cluster (the census's
    `half`), in (-pi, pi]."""
    return wrap_2pi(np.asarray(global_phi, np.float64) - np.asarray(dir_phi, np.float64))


def beta_of(cot_theta):
    """beta: the dip angle lambda = atan(cot theta)."""
    return np.arctan(np.asarray(cot_theta, np.float64))


def residual(angle, a, ref):
    """a - ref in the angle's own period."""
    return WRAP[angle](np.asarray(a, np.float64) - np.asarray(ref, np.float64))


def sigma_of(angle, cols):
    """Per-cluster published uncertainty of the angle (radians), from the emitted columns."""
    if angle == "alpha":
        return np.asarray(cols["sigGlobalClusterPhi"], np.float64)
    cot = np.asarray(cols["globalClusterCotTheta"], np.float64)
    return np.asarray(cols["sigGlobalClusterCotTheta"], np.float64) / (1.0 + cot * cot)


def emitted(angle, cols):
    if angle == "alpha":
        return alpha_of(cols["globalPhi"], cols["globalClusterPhi"])
    return beta_of(cols["globalClusterCotTheta"])


def truth(angle, cols):
    """The truth reference at the cluster (truth columns: valid only where tpIdx >= 0)."""
    if angle == "alpha":
        return alpha_of(cols["globalPhi"], cols["tpGlobalClusterPhi"])
    return beta_of(cols["tpGlobalClusterCotTheta"])


def angle_from(angle, value, cols):
    """The cot-theta / direction-azimuth columns implied by an angle value (inverse of
    emitted/truth): alpha -> direction azimuth, beta -> cot theta."""
    if angle == "alpha":
        return wrap_2pi(np.asarray(cols["globalPhi"], np.float64) - value)
    return np.tan(value)


# ---------------------------------------------------------------- incidence key
def is_disc(surface):
    return np.asarray(surface) > DISC_SURFACE_OFFSET


def incidence_key(surface, cot_theta):
    """frame.incidence_key: signed cot theta (barrel), signed 1 / cot theta (discs)."""
    cot = np.asarray(cot_theta, np.float64)
    with np.errstate(divide="ignore"):
        inv = np.where(cot == 0, np.copysign(np.inf, cot), 1.0 / np.where(cot == 0, 1.0, cot))
    return np.where(is_disc(surface), inv, cot)


def abs_edges(surface: int) -> tuple:
    return DISC_ABS_EDGES if int(surface) > DISC_SURFACE_OFFSET else BARREL_ABS_EDGES


def signed_edges(abs_e) -> np.ndarray:
    """(-inf, ..., -e1, 0, e1, ..., inf) from |edges| (0, e1, ..., inf)."""
    a = np.asarray(abs_e, np.float64)
    return np.r_[-a[:0:-1], a]


N_FINE = {s: 2 * (len(abs_edges(s)) - 1) for s in SURFACES}


def fine_bin(surface, key):
    """Index of the signed (unmerged) incidence bin, 0 .. N_FINE-1 in signed order; -1 for a
    non-finite key that is not +-inf (NaN) or an unknown surface."""
    surface = np.asarray(surface)
    key = np.asarray(key, np.float64)
    out = np.full(key.shape, -1, np.int16)
    for disc, ae in ((False, BARREL_ABS_EDGES), (True, DISC_ABS_EDGES)):
        se = signed_edges(ae)
        m = (is_disc(surface) == disc) & ~np.isnan(key) & np.isin(surface, SURFACES)
        b = np.searchsorted(se, key[m], side="right") - 1
        out[m] = np.clip(b, 0, len(se) - 2)
    return out


def fine_bin_bounds(surface: int, b: int) -> tuple:
    se = signed_edges(abs_edges(surface))
    return float(se[b]), float(se[b + 1])


# ---------------------------------------------------------------- module orientation (module frame)
def orientation_from_clusters(det_id, local_cot_beta, global_cot_theta, valid=None):
    """frame.module_orientation: (detIds, sign) with sign = the majority over the module's
    clusters of sign(localCotBeta x globalClusterCotTheta), emitted columns only (geometry, no
    truth). Modules without a decisive cluster are absent (lookup gives 0)."""
    det_id = np.asarray(det_id)
    p = np.asarray(local_cot_beta, np.float64) * np.asarray(global_cot_theta, np.float64)
    ok = np.isfinite(p) & (p != 0)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    ids, inv = np.unique(det_id[ok], return_inverse=True)
    vote = np.bincount(inv, weights=np.sign(p[ok]), minlength=len(ids))
    sign = np.where(vote > 0, 1, np.where(vote < 0, -1, 0)).astype(np.int8)
    keep = sign != 0
    return ids[keep], sign[keep]


def orientation_lookup(det_id, ids, signs):
    """The orientation sign of each cluster's module (0 when the module is unknown)."""
    det_id = np.asarray(det_id)
    ids = np.asarray(ids)
    signs = np.asarray(signs, np.int8)
    out = np.zeros(det_id.shape, np.int8)
    if len(ids) == 0:
        return out
    o = np.argsort(ids, kind="stable")
    ids, signs = ids[o], signs[o]
    pos = np.clip(np.searchsorted(ids, det_id), 0, len(ids) - 1)
    hit = ids[pos] == det_id
    out[hit] = signs[pos[hit]]
    return out


def facing(surface, orientation):
    """The table's facing split: 0 on the barrel (one row set), the orientation sign on discs."""
    return np.where(is_disc(surface), np.asarray(orientation, np.int8), 0).astype(np.int8)


def key_from_local(surface, cot_alpha, cot_beta, orientation):
    """The incidence key from module-local angles and the module's orientation sign s:
    barrel cot theta = s cotBeta / sqrt(1 + cotAlpha^2); disc 1 / cot theta =
    s sign(cotBeta) sqrt(cotAlpha^2 + cotBeta^2)."""
    ca = np.asarray(cot_alpha, np.float64)
    cb = np.asarray(cot_beta, np.float64)
    s = np.asarray(orientation, np.float64)
    barrel = s * cb / np.sqrt(1.0 + ca * ca)
    disc = s * np.sign(cb) * np.sqrt(ca * ca + cb * cb)
    return np.where(is_disc(surface), disc, barrel)


def local_beta_sign(surface, orientation):
    """d atan(localCotBeta) / d lambda for a module of orientation sign s: +s on the barrel
    (cotBeta ~ s cot theta), -s on a disc (cotBeta ~ s tan theta = s cot lambda)."""
    s = np.asarray(orientation, np.int8)
    return np.where(is_disc(surface), -s, s).astype(np.int8)


def to_local(angle, value, orientation, surface):
    """A global-frame table quantity with a sign (a centre, a residual, a pull) in the module
    frame, whose beta angle is atan(localCotBeta): beta -> local_beta_sign x value; alpha ->
    value on the barrel (no flip); alpha on a disc is not defined (NaN: no disc consumer works in
    the module frame)."""
    v = np.asarray(value, np.float64)
    if angle == "beta":
        return local_beta_sign(surface, orientation) * v
    return np.where(is_disc(surface), np.nan, v)


# ---------------------------------------------------------------- the knob k (decision 10, D11 c)
def check_k(k) -> float:
    k = float(k)
    if not (0.0 <= k <= 1.0):
        raise ValueError(f"angle improvement k must be in [0, 1] (k > 1 is refused): {k}")
    return k


def improve(angle, a, ref, is_T, k):
    """(angle', moved): angle' = ref + k (a - ref) on class T clusters (is_T, which must imply a
    valid ref); alpha clusters with |a - ref| > 90 deg keep a; k == 1 and class C return the
    input values exactly (float64)."""
    k = check_k(k)
    out = np.array(a, dtype=np.float64, copy=True)
    if k == 1.0:
        return out, np.zeros(out.shape, bool)
    d = residual(angle, out, ref)
    use = np.asarray(is_T, bool) & np.isfinite(d)
    if angle == "alpha":
        use &= np.abs(d) <= 0.5 * np.pi
    ref = np.asarray(ref, np.float64)
    out[use] = WRAP[angle](ref[use] + k * d[use])
    return out, use


def sigma_k(sigma, k, mode):
    """The consumer's per-cluster uncertainty at k: `fixed` keeps sigma, `scaled` multiplies it by
    max(k, 0.05) (every cluster). k == 1 returns sigma exactly in both modes."""
    k = check_k(k)
    s = np.array(sigma, dtype=np.float64, copy=True)
    if mode not in K_MODES:
        raise ValueError(f"sigma mode {mode!r} not in {K_MODES}")
    if mode == "scaled" and k < 1.0:
        s *= max(k, K_FLOOR)
    return s


def centre_k(centre_abs, k):
    """The subtracted centre at k, in absolute angle units: k x the k = 1 centre (both modes)."""
    return check_k(k) * np.asarray(centre_abs, np.float64)


def calibration_k(k, mode) -> float:
    """The k at which a cutting consumer measures its quantile (angle.cut_level): k itself in
    `scaled` mode (re-measured on the k-transformed sample), 1 in `fixed` mode."""
    return check_k(k) if mode == "scaled" else 1.0


def knob(angle, cols, is_T, k, mode, a=None, ref=None, sigma=None):
    """The angle, its uncertainty and the truth reference as a consumer at (k, mode) sees them:
    {"a", "sigma", "sigma_pub", "ref", "moved"}. sigma_pub is the published (k = 1) uncertainty,
    to which table centres and widths refer."""
    a = emitted(angle, cols) if a is None else a
    sig = sigma_of(angle, cols) if sigma is None else sigma
    if ref is None:
        ref = truth(angle, cols) if k < 1.0 else None
    a2, moved = improve(angle, a, ref, is_T, k) if k < 1.0 else (np.array(a, np.float64, copy=True),
                                                               np.zeros(np.shape(a), bool))
    return {"a": a2, "sigma": sigma_k(sig, k, mode), "sigma_pub": np.asarray(sig, np.float64),
            "ref": ref, "moved": moved}


def table_pull(resid, sigma_pub, sigma_cons, k, centre, width):
    """The consumer's pull after the table: (r - k c sigma_pub) / (w sigma_cons), with the table's
    centre c and width w in published-sigma units. At k = 1 this is (pull - c) / w."""
    return (np.asarray(resid, np.float64) - centre_k(np.asarray(centre) * sigma_pub, k)) / (
        np.asarray(width, np.float64) * np.asarray(sigma_cons, np.float64))


# ---------------------------------------------------------------- per-bin estimators
def wquantile(x, levels, w=None):
    """Quantiles (inverted CDF: the smallest value whose cumulative weight fraction reaches the
    level) with optional non-negative weights (bootstrap counts). NaN for an empty input."""
    x = np.asarray(x, np.float64)
    lv = np.atleast_1d(np.asarray(levels, np.float64))
    if w is None:
        w = np.ones(len(x))
    w = np.asarray(w, np.float64)
    m = w > 0
    x, w = x[m], w[m]
    if len(x) == 0:
        return np.full(lv.shape, np.nan)
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    i = np.searchsorted(cw, lv * cw[-1] * (1 - 1e-12), side="left")
    return x[o][np.clip(i, 0, len(x) - 1)]


def core_w(x, w=None):
    """The B.1 core (estimator.core) of x with integer bootstrap weights (each value repeated)."""
    x = np.asarray(x, np.float64)
    if w is not None:
        x = np.repeat(x, np.asarray(w, np.int64))
    return core(x)


def centre_of(x):
    """The table centre of a pull sample (angle.error_table): the 25 %-trimmed mean, the mean of the
    central 50 % (floor(n / 4) values dropped at each end)."""
    s = np.sort(np.asarray(x, np.float64))
    a = int(np.floor(0.25 * len(s)))
    return float(s[a:len(s) - a].mean())


def weighting_width(d):
    """The weighting width from d = |pull - centre| (angle.error_table): the mean of the smallest
    WIDTH_FRACTION (95 %) of d, i.e. of the smallest t = 0.95 n values with the boundary value weighted by
    the fractional part of t, divided by WIDTH_DIVISOR (0.7168, its value for a unit Gaussian)."""
    d = np.asarray(d, np.float64)
    t = WIDTH_FRACTION * len(d)
    m = int(np.floor(t))
    if m >= len(d):
        return float(d.mean() / WIDTH_DIVISOR)
    part = np.partition(d, m)
    return float((part[:m].sum() + (t - m) * part[m]) / t / WIDTH_DIVISOR)


def bin_estimates(pulls, levels=LEVELS, w=None, diagnostics: bool = True) -> dict | None:
    """The table entries of one bin from its class-T pulls (angle.error_table): centre = centre_of (the
    25 %-trimmed mean), width = weighting_width of |pull - centre|, q = quantiles of |pull - centre| (the
    k-th smallest, k = quantile_order(n, level)). With diagnostics: the B.1 clipped mean and core width
    of the centred pulls, the MAD / 0.6745, the fraction outside 3 widths and the full RMS (reported,
    never used). None below 200 values."""
    p = np.asarray(pulls, np.float64)
    if w is not None:
        p = np.repeat(p, np.asarray(w, np.int64))
    if len(p) < MIN_FIT:
        return None
    c = centre_of(p)
    d = np.abs(p - c)
    k = quantile_order(len(d), np.asarray(levels, np.float64)).astype(np.int64)
    qs = np.partition(d, k - 1)[k - 1]
    width = weighting_width(d)
    out = {"n": int(len(p)), "centre": c, "width": width, "q": {f"{v:g}": float(x) for v, x in zip(levels, qs)}}
    if diagnostics:
        c0 = core(p)
        cc = core(p - c0["mean"])
        out.update({"clip_centre": float(c0["mean"]), "core_width": float(cc["rms0"]),
                    "mad": float(np.median(np.abs(p - np.median(p))) / 0.6744897501960817),
                    "out3": float(np.mean(d > 3.0 * width)), "full_rms0": float(np.sqrt(np.mean(d * d)))})
    return out


# ---------------------------------------------------------------- the merge rule
def _edge_distance(lo: float, hi: float) -> float:
    """Distance of the |key| interval of a signed bin from the nearest plateau edge."""
    a, b = sorted((abs(lo), abs(hi))) if lo * hi >= 0 else (0.0, max(abs(lo), abs(hi)))
    return min(0.0 if a <= e <= b else min(abs(a - e), abs(b - e)) for e in PLATEAU_EDGES)


def merge_bins(bounds, counts, min_n=MIN_FIT):
    """angle.error_table merge rule. `bounds` [(lo, hi)] of contiguous signed bins in order and
    their fit counts; while a bin has fewer than min_n, the smallest such bin merges with its
    neighbour away from the plateau edges (the neighbour farther from 0.5 / 0.67 in |key|; on a
    tie the one at larger |key|). Returns the merged groups as lists of input indices."""
    groups = [[i] for i in range(len(bounds))]
    n = [int(c) for c in counts]
    b = [tuple(x) for x in bounds]
    while len(groups) > 1:
        small = [j for j in range(len(groups)) if n[j] < min_n]
        if not small:
            break
        j = min(small, key=lambda t: (n[t], t))
        cand = [x for x in (j - 1, j + 1) if 0 <= x < len(groups)]
        if len(cand) == 2:
            d = [_edge_distance(*b[x]) for x in cand]
            far = [max(abs(b[x][0]), abs(b[x][1])) for x in cand]
            pick = cand[0] if (d[0], far[0]) > (d[1], far[1]) else cand[1]
        else:
            pick = cand[0]
        lo_i, hi_i = min(j, pick), max(j, pick)
        groups[lo_i] = groups[lo_i] + groups[hi_i]
        n[lo_i] += n[hi_i]
        b[lo_i] = (b[lo_i][0], b[hi_i][1])
        del groups[hi_i], n[hi_i], b[hi_i]
    return groups


# ---------------------------------------------------------------- the table
class AngleTable:
    """The angle-uncertainty table (angle.error_table) with its provenance.

    doc = {"schema", "definition_sheet", "provenance": {...}, "era", "fit_sample", "pt_min",
           "levels", "orientation": {"det_id": [...], "sign": [...]},
           "keys": {key: {"rows": [ {surface, facing, lo, hi, fine: [fine bins], n_fit, flags,
                                      "alpha": {centre, width, q, E0, core_width, clip_centre, mad, ...},
                                      "beta": {...}} ]}}}
    Centres, widths and quantiles are in published-sigma units of this module's representation:
    centre = the 25 %-trimmed mean, width = the weighting width (95 %-trimmed mean of |pull - centre| /
    0.7168), q = the quantiles of |pull - centre| (angle.error_table, sheet v2.7); core_width,
    clip_centre and mad are diagnostics."""

    def __init__(self, doc: dict):
        if doc.get("schema") != TABLE_SCHEMA:
            raise ValueError(f"not an angle table ({doc.get('schema')!r} != {TABLE_SCHEMA})")
        self.doc = doc
        self.levels = tuple(float(x) for x in doc["levels"])
        self._maps = {}
        for key, kd in doc["keys"].items():
            # rowmap[surface][facing] -> int array over fine bins (-1 = no row)
            m = {}
            for i, r in enumerate(kd["rows"]):
                s, f = int(r["surface"]), int(r["facing"])
                arr = m.setdefault((s, f), np.full(N_FINE[s], -1, np.int32))
                arr[np.asarray(r["fine"], int)] = i
            self._maps[key] = m
        self._arr = {key: self._arrays(key) for key in doc["keys"]}

    def _arrays(self, key):
        rows = self.doc["keys"][key]["rows"]
        out = {}
        for ang in ANGLES:
            out[(ang, "centre")] = np.array([r[ang]["centre"] if r.get(ang) else np.nan for r in rows] + [np.nan])
            out[(ang, "width")] = np.array([r[ang]["width"] if r.get(ang) else np.nan for r in rows] + [np.nan])
            for lv in self.levels:
                out[(ang, lv)] = np.array([r[ang]["q"][f"{lv:g}"] if r.get(ang) else np.nan for r in rows] + [np.nan])
        return out

    # -- I/O
    @classmethod
    def load(cls, path: str) -> "AngleTable":
        with open(path) as f:
            return cls(json.load(f))

    def save(self, path: str) -> None:
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(self.doc, f, indent=1, default=float)
            f.write("\n")
        os.replace(tmp, path)

    @property
    def provenance(self) -> dict:
        return self.doc.get("provenance", {})

    def orientation(self, det_id):
        o = self.doc["orientation"]
        return orientation_lookup(det_id, np.asarray(o["det_id"], np.uint32), np.asarray(o["sign"], np.int8))

    # -- lookup
    def rows_of(self, key: str, surface, key_value, fac):
        """Row index per cluster in table `key` (-1 where none)."""
        surface = np.asarray(surface)
        fb = fine_bin(surface, key_value)
        fac = np.asarray(fac)
        out = np.full(fb.shape, -1, np.int32)
        for (s, f), arr in self._maps[key].items():
            m = (surface == s) & (fac == f) & (fb >= 0)
            out[m] = arr[fb[m]]
        return out

    def get(self, key: str, angle: str, what, rows):
        """centre / width (what = 'centre' | 'width') or the quantile at level `what` (a float),
        per row index (NaN for -1)."""
        a = self._arr[key][(angle, float(what) if not isinstance(what, str) else what)]
        return a[np.where(np.asarray(rows) >= 0, rows, -1)]

    def apply(self, key, angle, resid, sigma_pub, sigma_cons, k, rows):
        """The consumer pull after the table (table_pull) for the clusters' rows."""
        return table_pull(resid, sigma_pub, sigma_cons, k, self.get(key, angle, "centre", rows),
                          self.get(key, angle, "width", rows))


def new_table_doc(keys_rows: dict, orientation: tuple, provenance: dict, era: str, fit_sample: str,
                  pt_min: float = PT_MIN, levels=LEVELS) -> dict:
    ids, signs = orientation
    return {"schema": TABLE_SCHEMA, "definition_sheet": SHEET_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "representation": "alpha = wrap(globalPhi - globalClusterPhi), beta = atan(cot theta); pulls in published sigma "
                              "(sigGlobalClusterPhi; sigGlobalClusterCotTheta / (1 + cot^2) at the emitted cot)",
            "keys_definition": {"emitted": "incidence key from globalClusterCotTheta",
                                "true": "incidence key from tpGlobalClusterCotTheta (stands in for a prediction)"},
            "era": era, "fit_sample": fit_sample, "pt_min": pt_min, "levels": list(levels),
            "provenance": provenance,
            "orientation": {"det_id": [int(x) for x in ids], "sign": [int(x) for x in signs]},
            "keys": {k: {"rows": v} for k, v in keys_rows.items()}}


# ---------------------------------------------------------------- cutting consumers
def consumer_quantile(stat, bin_id, levels=LEVELS, population=None, w=None, n_bins=None):
    """Per-bin quantiles of a cutting consumer's OWN statistic over `population` (class-T
    clusters of the consumer's calibration population). Returns an array [n_bins, n_levels]
    (NaN for an empty bin). angle.cut_level: the consumer cuts at the bin's quantile at its
    level (decision 2: 99 %), with the rank of quantile_order (with weights: the smallest value
    whose cumulative weight reaches min(level x (total + 1), total))."""
    stat = np.asarray(stat, np.float64)
    bin_id = np.asarray(bin_id, np.int64)
    m = np.isfinite(stat) & (bin_id >= 0)
    if population is not None:
        m &= np.asarray(population, bool)
    if w is not None:
        w = np.asarray(w, np.float64)
        m &= w > 0
    nb = int(n_bins if n_bins is not None else (bin_id[m].max() + 1 if m.any() else 0))
    lv = np.asarray(levels, np.float64)
    out = np.full((nb, len(lv)), np.nan)
    if not m.any():
        return out
    b, s = bin_id[m], stat[m]
    ww = np.ones(len(s)) if w is None else w[m]
    o = np.lexsort((s, b))
    b, s, ww = b[o], s[o], ww[o]
    starts = np.searchsorted(b, np.arange(nb), side="left")
    ends = np.searchsorted(b, np.arange(nb), side="right")
    cw = np.cumsum(ww)
    for i in range(nb):
        a0, a1 = starts[i], ends[i]
        if a1 <= a0:
            continue
        base = cw[a0 - 1] if a0 > 0 else 0.0
        tot = cw[a1 - 1] - base
        target = np.minimum(lv * (tot + 1.0), tot)
        j = np.searchsorted(cw[a0:a1] - base, target * (1 - 1e-12), side="left")
        out[i] = s[a0:a1][np.clip(j, 0, a1 - a0 - 1)]
    return out


def consumer_quantile_at_k(stat_at_k, k, mode, bin_id, population, levels=LEVELS, w=None, n_bins=None):
    """angle.cut_level at (k, mode): the quantile of the consumer's statistic measured on the
    k-transformed sample in `scaled` mode and at k = 1 in `fixed` mode. `stat_at_k(k)` returns
    the statistic per cluster with the knob at k (and the consumer's sigma mode)."""
    return consumer_quantile(stat_at_k(calibration_k(k, mode)), bin_id, levels, population, w, n_bins)


# ---------------------------------------------------------------- coverage of a fitted quantile (amendment 1)
def quantile_order(n_fit, level):
    """k of the quantile at `level` of n_fit values (the k-th smallest): k = min(ceil(level (n + 1)), n),
    the distribution-free (conformal) rank, E[coverage] = k / (n + 1) >= level whenever
    ceil(level (n + 1)) <= n (angle.error_table, angle.cut_level; P0.3 amendment 1d; sheet v1's
    ceil(level n) is retired). Levels that n values cannot reach (at 99.7 %: n <= 332) get the maximum."""
    n = np.asarray(n_fit, np.float64)
    return np.minimum(np.ceil(level * (n + 1.0) - 1e-9), n)


def level_reachable(n_fit, level):
    """Whether n_fit values can deliver E[coverage] >= level (ceil(level (n + 1)) <= n)."""
    n = np.asarray(n_fit, np.float64)
    return np.ceil(level * (n + 1.0) - 1e-9) <= n


def expected_coverage(n_fit, level):
    """E[coverage] on new exchangeable values of the quantile fitted on n_fit values: k / (n_fit + 1)
    (order statistics; distribution-free)."""
    n = np.asarray(n_fit, np.float64)
    return quantile_order(n, level) / (n + 1.0)


def fit_coverage_variance(n_fit, level):
    """Variance of the true coverage F(q) of that quantile: Beta(k, n - k + 1), pi (1 - pi) / (n + 2)
    with pi = 1 - k / (n + 1)."""
    n = np.asarray(n_fit, np.float64)
    pi = 1.0 - expected_coverage(n, level)
    return pi * (1.0 - pi) / (n + 2.0)


def exchangeable_coverage_variance(n_fit, n_test, level):
    """Variance of the test coverage 1 - M / n_test of that quantile for exchangeable values: with
    C ~ Beta the true coverage, Var = E0 (1 - E0) / n_test + Var(C) (1 - 1 / n_test) (total variance;
    the beta-binomial(n_test, n_fit - k + 1, k) variance exactly)."""
    e0 = expected_coverage(n_fit, level)
    n_t = np.asarray(n_test, np.float64)
    return e0 * (1.0 - e0) / n_t + fit_coverage_variance(n_fit, level) * (1.0 - 1.0 / n_t)


def coverage_z(m, n_test, e0, var):
    """Signed z of an observed exceedance count m (test values beyond the fitted quantile) among
    n_test values, under the null "coverage has mean e0 and variance var": the count is
    beta-binomial(n_test, a, b) with a / (a + b) = 1 - e0 and its variance matched to var (binomial
    when var does not exceed the binomial variance). With var = exchangeable_coverage_variance this
    is the exact order-statistic null (beta-binomial(n_test, n_fit - k + 1, k)); event clustering
    enters through a larger var. Mid-p;
    z = -Phi^-1(P_mid(M <= m)): negative for under-coverage. Vectorised; NaN where undefined."""
    from scipy import stats

    m, n, e0, var = np.broadcast_arrays(*(np.asarray(x, np.float64) for x in (m, n_test, e0, var)))
    z = np.full(m.shape, np.nan)
    pi = 1.0 - e0
    ok = np.isfinite(m) & np.isfinite(n) & (n > 0) & np.isfinite(pi) & (pi > 0) & (pi < 1) & np.isfinite(var)
    if not ok.any():
        return z
    m, n, pi, var = m[ok], n[ok], pi[ok], var[ok]
    rho = var * n / (pi * (1.0 - pi))          # variance in units of the binomial variance
    bb = rho > 1.0 + 1e-9
    upper = np.empty(m.shape)                   # P(M > m) + P(M = m) / 2
    lower = np.empty(m.shape)                   # P(M < m) + P(M = m) / 2
    if (~bb).any():
        d = stats.binom(n[~bb], pi[~bb])
        pm = d.pmf(m[~bb])
        upper[~bb] = d.sf(m[~bb]) + 0.5 * pm
        lower[~bb] = d.cdf(m[~bb] - 1) + 0.5 * pm
    if bb.any():
        s = (n[bb] - rho[bb]) / (rho[bb] - 1.0)
        s = np.maximum(s, 1e-6)                 # rho >= n: maximal over-dispersion
        d = stats.betabinom(n[bb], pi[bb] * s, (1.0 - pi[bb]) * s)
        pm = d.pmf(m[bb])
        upper[bb] = d.sf(m[bb]) + 0.5 * pm
        lower[bb] = d.cdf(m[bb] - 1) + 0.5 * pm
    upper = np.clip(upper, 1e-300, 1.0)
    lower = np.clip(lower, 1e-300, 1.0)
    # z > 0: fewer exceedances than expected (over-coverage); computed on the smaller tail for precision
    z[ok] = np.where(upper < 0.5, -stats.norm.isf(upper), stats.norm.isf(lower))
    return z


# ---------------------------------------------------------------- single-cluster consumer statistics
Z_LUMI_CM = 15.0     # the census's luminous-region half-window (tracklet_topology_cost Z_LUMI) [cm]


def alpha_band(r_cm, pt_floor):
    """|alpha| of a prompt track of transverse momentum pt_floor at radius r: asin(KPT r / 2 pT)."""
    return np.arcsin(np.clip(KPT_CMSSW * np.asarray(r_cm, np.float64) / (2.0 * pt_floor), 0.0, 1.0))


def s1_statistic(a, sigma_cons, centre_abs, r_cm, pt_floor):
    """S1-like single-cluster alpha test against the pT floor: (|alpha - c| - band(r, floor)) /
    sigma (negative inside the band). `centre_abs` = the subtracted centre (centre_k x sigma_pub)."""
    return (np.abs(wrap_2pi(np.asarray(a) - centre_abs)) - alpha_band(r_cm, pt_floor)) / sigma_cons


def beta_window(r_cm, z_cm, z_lumi=Z_LUMI_CM):
    """(mid, half-length) of the lambda interval pointing to |z0| <= z_lumi from (r, z)."""
    r = np.asarray(r_cm, np.float64)
    z = np.asarray(z_cm, np.float64)
    lo, hi = np.arctan((z - z_lumi) / r), np.arctan((z + z_lumi) / r)
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def z0_window_statistic(lam, sigma_cons, centre_abs, r_cm, z_cm, z_lumi=Z_LUMI_CM):
    """Single-cluster beta test against the luminous region: (distance of lambda - c from the
    interval of directions pointing to |z0| <= z_lumi) / sigma (negative inside)."""
    mid, h = beta_window(r_cm, z_cm, z_lumi)
    return (np.abs(wrap_pi(np.asarray(lam) - centre_abs - mid)) - h) / sigma_cons
