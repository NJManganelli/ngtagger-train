"""Displacement heads, second version: the definitions shared by the head training (displacement_heads.py), the jet
tagger (tagger_sp/quant_shift.py) and the pair algorithm. Everything here is a function of the track's own tanL and z0,
the detector scenario and fixed surface positions; numpy only.

SURFACES (nominal, measured on the SmartPixels cluster table of canon_v5, cm): barrel layers 1-4 at r = 2.85, 5.95,
10.33, 14.49 with |z| <= 20.2; discs 1-4 at |z| = 25.3, 32.3, 41.2, 52.6 (each disc is two planes 0.6 cm either side
of that) with 3.1 <= r <= 16.1. Letter k of a detector scenario switches barrel layer k and disc k together.
TRAJECTORY: a straight line in (r, z) from the beam line, z(r) = z0 + tanL r (pT >= 2 GeV: the bending changes the
radius of a crossing by well under the surfaces' own spread). A surface is CROSSED when the line meets it inside the
acceptance above: a barrel layer the track passes beyond its end is not crossed, a disc the track passes inside its
inner radius or outside its outer radius is not crossed. The acceptance edges are nominal (measured: between |z| = 19
and 20.2 cm of a barrel layer about half of the crossings have a hit, beyond 20.2 cm 1 %; discs between r = 16.1 and
17 cm 14-44 %): a track near an edge can have a hit on a surface this function calls not crossed, or the reverse.
CLASSES: the crossed active surfaces of a trajectory, ordered by the radius of the crossing: r_1 < r_2 < ... The class
of a particle is the number of them inside its production radius, capped at N = the number of active letters (AAAA 4,
AAAI and IAAA 3): class 0 = produced inside the first crossed active surface, class k = between the k-th and the
(k+1)-th, class N = beyond the N-th. In the barrel (four or three crossed surfaces) these are the barrel layers; a
forward trajectory crosses up to six surfaces and the cap merges everything beyond its N-th into the last class.
A trajectory with m < N crossed surfaces has classes 0..m only.
RANGE: class k of a track is the radius interval [r_k, r_(k+1)) of that track's own crossings (r_0 = 0; beyond the last
crossing up to R_MAX). The class probabilities are turned into the 16 %, 50 % and 84 % points of the radius ("low",
"median", "high") with a stated convention for where inside a class the radius sits: the distribution of the
fractional position (r - r_k) / (r_(k+1) - r_k) of the training tracks' true radii, per class (a table of 101
quantiles shipped with the models; a class with fewer than 50 training tracks is taken flat).
PER-SURFACE INPUTS, two lists (SLOT_SETS): "full" = what the CMSSW refit-hit table gives (signed residuals in cm, signed
pulls in x and y and of both angles); "magnitudes" = what the Python replay of a quantization scenario also gives (the
absolute pulls only), used where heads are compared across quantization scenarios.
"""
from __future__ import annotations

import numpy as np

R_BARREL = np.array([2.85, 5.95, 10.33, 14.49])
Z_BARREL = 20.2
Z_DISC = np.array([25.3, 32.3, 41.2, 52.6])
R_DISC = (3.1, 16.1)
R_MAX = 60.0                  # cm: the end of the open last class (99.99 % of the matched prompt tracks are produced inside 48 cm)
N_SLOTS = 6                   # crossed active surfaces kept per track (the most a trajectory crosses)
CODES = np.array([1, 2, 3, 4, 11, 12, 13, 14])       # the refit-hit table's surface code: barrel layers, discs
QUANTILES = (0.16, 0.5, 0.84)
SHAPE_GRID = np.linspace(0.0, 1.0, 101)
SHAPE_MIN_TRACKS = 50

# per crossed surface, from the refit-hit table (none is a truth column: refuse_truth_columns)
HIT_COLUMNS = ["hitAccepted", "projResX", "projResY", "pullX", "pullY", "pullAlpha", "pullBeta", "windowMult"]
SLOT_SETS = {"full": ["res_x", "res_y", "pull_x", "pull_y", "pull_alpha", "pull_beta"],
             "magnitudes": ["abs_pull_x", "abs_pull_y", "abs_pull_alpha", "abs_pull_beta"]}
SUMMARY_NAMES = ["n_crossed", "first_without_hit", "crossed_before_first_hit", "radius_first_hit"]
M_PI, M_K0S = 0.13957, 0.497611


def slot_names(kind: str = "full") -> list[str]:
    return ["radius", "is_disc", "accepted"] + SLOT_SETS[kind] + ["window_mult"]


def input_names(kind: str = "full") -> list[str]:
    return [f"s{j + 1}_{n}" for j in range(N_SLOTS) for n in slot_names(kind)] + SUMMARY_NAMES


def hits_from_table(cols: dict, kind: str = "full") -> dict:
    """The per-hit quantities of slot_inputs from the refit-hit table's columns {column of HIT_COLUMNS: flat array}
    (-999 = none)."""
    v = lambda c: np.where(np.asarray(cols[c], np.float64) <= -998, np.nan, np.asarray(cols[c], np.float64))
    out = {"accepted": np.asarray(cols["hitAccepted"], np.float64), "window_mult": v("windowMult")}
    if kind == "full":
        out.update(res_x=v("projResX"), res_y=v("projResY"), pull_x=v("pullX"), pull_y=v("pullY"), pull_alpha=v("pullAlpha"), pull_beta=v("pullBeta"))
    else:
        out.update({f"abs_pull_{n}": np.abs(v(f"pull{n.capitalize()}")) for n in ("x", "y", "alpha", "beta")})
    return out


def k0s_pions(pdg, evt, bx, vx, vy, vz, pt, eta, phi):
    """Per tracking particle of ONE event: it is a pion of a K0S -> pi+ pi- (a pi+ and a pi- with a bit-identical
    production vertex and nothing else charged-pion at it, invariant mass within 1 MeV of the K0S mass)."""
    pdg = np.asarray(pdg)
    is_k = np.zeros(len(pdg), bool)
    pi = np.flatnonzero(np.abs(pdg) == 211)
    if len(pi) < 2:
        return is_k
    key = np.stack([np.asarray(x, np.float64)[pi] for x in (evt, bx, vx, vy, vz)], 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    two = np.flatnonzero(cnt[inv] == 2)
    if len(two):
        o = two[np.argsort(inv[two], kind="stable")]
        a, b = pi[o[0::2]], pi[o[1::2]]
        pt, eta, phi = (np.asarray(x, np.float64) for x in (pt, eta, phi))
        p4 = lambda i: (pt[i] * np.cos(phi[i]), pt[i] * np.sin(phi[i]), pt[i] * np.sinh(eta[i]), np.sqrt((pt[i] * np.cosh(eta[i])) ** 2 + M_PI ** 2))
        xa, xb = p4(a), p4(b)
        m = np.sqrt(np.maximum((xa[3] + xb[3]) ** 2 - sum((xa[k] + xb[k]) ** 2 for k in range(3)), 0.0))
        ok = (pdg[a] * pdg[b] < 0) & (np.abs(m - M_K0S) < 1e-3)
        is_k[a[ok]] = is_k[b[ok]] = True
    return is_k


def n_edges(mask: str) -> int:
    """N: the number of class boundaries (the classes are 0..N)."""
    return mask.count("A")


def hit_table(track_table: str) -> str:
    return track_table.replace("Track", "RefitHit")


def refuse_truth_columns(tree, table: str) -> None:
    """Refuse a refit-hit input column whose title marks it as truth."""
    for c in HIT_COLUMNS + ["layer", "trackIdx"]:
        if "TRUTH" in (tree[f"{table}_{c}"].title or "").upper():
            raise RuntimeError(f"{table}_{c} is a truth column: not an input")


def crossings(tanl, z0, mask: str):
    """(radii (n, N_SLOTS) ascending, NaN-padded; codes (n, N_SLOTS), 0 = none; n_crossed (n,)) of the crossed active
    surfaces of each trajectory."""
    tanl, z0 = np.asarray(tanl, np.float64), np.asarray(z0, np.float64)
    r = np.full((len(tanl), 8), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(4):
            if mask[k] != "A":
                continue
            r[np.abs(z0 + tanl * R_BARREL[k]) <= Z_BARREL, k] = R_BARREL[k]
            rd = (np.sign(tanl) * Z_DISC[k] - z0) / tanl
            ok = (rd >= R_DISC[0]) & (rd <= R_DISC[1])
            r[ok, 4 + k] = rd[ok]
    order = np.argsort(r, axis=1, kind="stable")[:, :N_SLOTS]          # NaN last
    radii = np.take_along_axis(r, order, 1)
    codes = np.where(np.isfinite(radii), CODES[order], 0)
    return radii, codes, np.isfinite(r).sum(1)


def true_class(r_prod, tanl, z0, mask: str):
    """The class of a particle produced at radius r_prod on the trajectory (tanl, z0): its own truth parameters."""
    radii, _, _ = crossings(tanl, z0, mask)
    return np.minimum((radii < np.asarray(r_prod, np.float64)[:, None]).sum(1), n_edges(mask))


def slot_inputs(mask: str, tanl, z0, hit_track, hit_layer, hits: dict, kind: str = "full"):
    """The per-surface inputs (n, len(input_names(kind))) float32, NaN = absent, and the track's crossing radii
    (n, N_SLOTS). tanl, z0: the track's (refit) parameters; hit_track: the flat index of each per-surface entry's track;
    hit_layer: its surface code; hits: {"accepted", "window_mult", the names of SLOT_SETS[kind]: flat array, NaN = none}
    (hits_from_table). Slot j is the j-th crossed active surface (inside out)."""
    n = len(tanl)
    radii, codes, n_crossed = crossings(tanl, z0, mask)
    col = np.where(np.asarray(hit_layer) < 10, np.asarray(hit_layer, np.int64) - 1, np.asarray(hit_layer, np.int64) - 7)
    entry = np.full((n, 8), -1, np.int64)
    entry[np.asarray(hit_track, np.int64), col] = np.arange(len(col))
    slot_col = np.where(codes < 10, codes - 1, codes - 7)
    e = np.where(codes > 0, np.take_along_axis(entry, np.maximum(slot_col, 0), 1), -1)      # the slot's refit-hit entry
    exists, has = codes > 0, e >= 0

    take = lambda c: np.where(has, np.asarray(hits[c], np.float64)[np.maximum(e, 0)], np.nan)
    acc = np.where(exists, np.nan_to_num(take("accepted")), np.nan)                           # crossed, no entry: no hit
    hit = acc > 0
    per = [radii, np.where(exists, (codes > 10).astype(np.float64), np.nan), acc] + [
        np.where(hit, take(c), np.nan) for c in SLOT_SETS[kind]] + [take("window_mult")]
    nc = np.minimum(n_crossed, N_SLOTS).astype(np.float64)
    j = np.arange(N_SLOTS)[None, :]
    first_miss = np.where(exists & ~hit, j, N_SLOTS).min(1)
    first_hit = np.where(hit, j, N_SLOTS).min(1)
    X = np.concatenate([np.stack(per, 2).reshape(n, -1), np.column_stack([
        nc, np.minimum(first_miss, nc), np.minimum(first_hit, nc),
        np.where(first_hit < N_SLOTS, np.take_along_axis(radii, np.minimum(first_hit, N_SLOTS - 1)[:, None], 1)[:, 0], np.nan)])], 1)
    return X.astype(np.float32), radii


def edges(radii, mask: str):
    """(n, N + 2): 0, the track's first N crossing radii (a missing one = R_MAX: an empty class), R_MAX."""
    N = n_edges(mask)
    r = np.minimum(np.where(np.isfinite(radii[:, :N]), radii[:, :N], R_MAX), R_MAX)
    return np.column_stack([np.zeros(len(r)), r, np.full(len(r), R_MAX)])


def position_in_class(r_prod, cls, E):
    """The fractional position (r - lower edge) / (upper edge - lower edge) of a radius in its class, in [0, 1]."""
    lo, hi = np.take_along_axis(E, cls[:, None], 1)[:, 0], np.take_along_axis(E, cls[:, None] + 1, 1)[:, 0]
    return np.clip((np.asarray(r_prod, np.float64) - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)


def fit_shape(r_prod, cls, radii, mask: str):
    """The convention inside a class: (N + 1, 101) quantiles of the fractional position of the given (training) tracks."""
    E = edges(radii, mask)
    u = position_in_class(r_prod, cls, E)
    out = np.tile(SHAPE_GRID, (n_edges(mask) + 1, 1))
    for k in range(len(out)):
        if (cls == k).sum() >= SHAPE_MIN_TRACKS:
            out[k] = np.quantile(u[cls == k], SHAPE_GRID)
    return out


def radius_range(P, radii, mask: str, shape, quantiles=QUANTILES):
    """The radius [cm] at each cumulative probability of `quantiles` (n, len(quantiles)): P (n, N + 1) class
    probabilities, radii the track's crossing radii, shape from fit_shape."""
    P = np.asarray(P, np.float64)
    P = P / P.sum(1, keepdims=True)
    E = edges(radii, mask)
    cum = np.concatenate([np.zeros((len(P), 1)), np.cumsum(P, 1)], 1)
    cum[:, -1] = 1.0
    out = np.empty((len(P), len(quantiles)))
    idx = np.arange(len(P))
    for i, q in enumerate(quantiles):
        k = np.clip((cum[:, 1:] <= q).sum(1), 0, P.shape[1] - 1)
        f = np.clip((q - cum[idx, k]) / np.maximum(P[idx, k], 1e-12), 0.0, 1.0)
        u = np.empty(len(P))
        for c in range(P.shape[1]):
            m = k == c
            u[m] = np.interp(f[m], SHAPE_GRID, shape[c])
        out[:, i] = E[idx, k] + u * (E[idx, k + 1] - E[idx, k])
    return out


def assemble(X_heads, abs_d0, sigma_d0, slots):
    """The second version's input matrix: the track-quality heads' inputs, |d0|, |d0| / sigma(d0), the per-surface inputs."""
    return np.column_stack([X_heads, abs_d0, abs_d0 / np.maximum(sigma_d0, 1e-6), slots]).astype(np.float32)
