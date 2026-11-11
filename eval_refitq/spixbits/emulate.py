"""Emulation of the four SmartPixels outputs per cluster (PLAN.md section 4), consumer side only.

Frames. SmartPixels x (50 um pitch) = CMS local y = the cluster column localY; SmartPixels y (12.5 um pitch) = CMS
local x = localX. SmartPixels alpha = CMS local cot beta = localCotBeta (the replay code's "beta"); SmartPixels
beta, the bending angle, = CMS local cot alpha = localCotAlpha (the code's "alpha"). Angles swap with no sign change.
In this file "A" is the SmartPixels alpha and "B" the SmartPixels beta.

Everything random is a hash of (run, lumi, event, detId, the float32 bits of the stored localX and localY), so every
arm, scenario and restart sees the same grid phases, residual draws, noise and cap choice for a cluster.

  base()      once per chunk: the hash, grid phases, the network-resolution angles (true local cot + a jointly drawn
              residual of the full-MLP quantized network, class-T clusters; class C keeps its angle), the link of every
              cluster, the cap of three clusters per 2.4 mm sub-area per event
  emulate()   per arm: a context whose cluster columns hold what the chip sends (positions quantized with the
              per-cluster grid phase or with Gaussian noise of equal variance; angles quantized per link with
              underflow / overflow codes; clusters beyond the cap removed from every window)
"""
from __future__ import annotations

import json

import awkward as ak
import numpy as np
from scipy.spatial import cKDTree

from ngtagger.data import spix_nano as SPN

SUBAREA_CM = 0.24                     # 48 x 50 um = 192 x 12.5 um = 2.4 mm
CAP_PER_SUBAREA = 3
MEAS_ANGLE_MAX = 12.0                 # today's measured-angle cap |cot| <= 12 (spix_nano's loader)
A_EDGES = np.arange(-7.25, 7.2501, 0.25)      # true cot alpha (SmartPixels) cells of the network draw (quant_bits.py)
B_EDGES = np.arange(-1.25, 1.2501, 0.05)      # true cot beta (SmartPixels)
MIN_CELL = 20
U64 = np.uint64
SALT = {"phase_localX": 11, "phase_localY": 12, "draw": 13, "cap": 14, "noise1": 15, "noise2": 16, "noise3": 17}
EXTRA_CLUSTER_COLS = ("ring", "globalZ", "globalR", "tpChargeFrac", "tpPt")


# ---------------------------------------------------------------- hashes ----
def _mix(x):
    """splitmix64's finalizer on uint64 arrays."""
    x = np.array(x, U64, copy=True)
    with np.errstate(over="ignore"):
        x ^= x >> U64(30)
        x *= U64(0xBF58476D1CE4E5B9)
        x ^= x >> U64(27)
        x *= U64(0x94D049BB133111EB)
        x ^= x >> U64(31)
    return x


def cluster_hash(run, lumi, event, det_id, local_x, local_y):
    """uint64 per cluster from (run, lumi, event, detId, float32 bits of localX and localY)."""
    parts = [np.asarray(run).astype(U64), np.asarray(lumi).astype(U64), np.asarray(event).astype(U64),
             np.asarray(det_id).astype(U64),
             np.ascontiguousarray(local_x, np.float32).view(np.uint32).astype(U64),
             np.ascontiguousarray(local_y, np.float32).view(np.uint32).astype(U64)]
    h = np.full(len(parts[3]), 0x9E3779B97F4A7C15, U64)
    with np.errstate(over="ignore"):
        for i, p in enumerate(parts):
            h = _mix(h ^ (p + U64(0x9E3779B97F4A7C15) * U64(i + 1)))
    return h


def uniform(h, salt):
    """Uniform [0, 1) per cluster, independent for each salt."""
    with np.errstate(over="ignore"):
        return (_mix(h ^ (U64(salt) * U64(0xD6E8FEB86659FD93))) >> U64(11)).astype(np.float64) * 2.0 ** -53


def normal(h, salt_a, salt_b):
    """Standard normal per cluster (Box-Muller on two hashed uniforms)."""
    u1 = np.maximum(uniform(h, salt_a), 2.0 ** -53)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * uniform(h, salt_b))


# ---------------------------------------------------------------- quantizers ----
def quantize_position(x, x0, step, phase):
    """q = x0 + (floor((x - x0) / step - phase) + phase + 1/2) step: a grid of the given step whose phase is the
    cluster's own, so that q - x is uniform on (-step/2, step/2] for every cluster."""
    return x0 + (np.floor((x - x0) / step - phase) + phase + 0.5) * step


def quantize_angle(v, lo, hi, nbits):
    """Angle code over [lo, hi) with 2^nbits - 2 value codes and an underflow and an overflow code.
    Returns (decoded value: the code's centre, or the range edge for a flag code; flag: -1 under, +1 over, 0)."""
    n = 2 ** nbits - 2
    step = (hi - lo) / n
    flag = np.where(v < lo, -1, np.where(v >= hi, 1, 0)).astype(np.int8)
    q = lo + (np.floor((v - lo) / step) + 0.5) * step
    return np.where(flag < 0, lo, np.where(flag > 0, hi, q)), flag


# ---------------------------------------------------------------- the network draw ----
class Network:
    """The network's residuals (prediction - truth), physical units, with the true-angle cells of the draw."""

    def __init__(self, path):
        z = np.load(path)
        self.rA, self.rB = z["rcotA"].astype(np.float64), z["rcotB"].astype(np.float64)
        self.rx_um, self.ry_um = z["rx"].astype(np.float64), z["ry"].astype(np.float64)
        self.nb = len(B_EDGES) - 1
        ia = np.clip(np.digitize(z["cotA_true"], A_EDGES) - 1, 0, len(A_EDGES) - 2)
        ib = np.clip(np.digitize(z["cotB_true"], B_EDGES) - 1, 0, self.nb - 1)
        cell = ia * self.nb + ib
        self.order = np.argsort(cell, kind="stable")
        self.starts = np.searchsorted(cell[self.order], np.arange((len(A_EDGES) - 1) * self.nb + 1))
        self.counts = np.diff(self.starts)
        self.pop = np.array([(i // self.nb, i % self.nb) for i in np.where(self.counts >= MIN_CELL)[0]], float)
        self.tree = cKDTree(self.pop * [1, 0.2])      # nearest populated cell, an A step weighs 5 B steps

    def draw(self, cot_a_true, cot_b_true, u):
        """Row of a network cluster in the same true-angle cell (nearest populated cell when the cell holds fewer
        than 20), chosen by the uniform u; and whether the cell was exact."""
        a = np.clip(np.digitize(cot_a_true, A_EDGES) - 1, 0, len(A_EDGES) - 2)
        b = np.clip(np.digitize(cot_b_true, B_EDGES) - 1, 0, self.nb - 1)
        c = a * self.nb + b
        exact = self.counts[c] >= MIN_CELL
        if (~exact).any():
            _, k = self.tree.query(np.column_stack([a[~exact], b[~exact]]) * [1, 0.2])
            c[~exact] = (self.pop[k, 0] * self.nb + self.pop[k, 1]).astype(int)
        return self.order[self.starts[c] + (u * self.counts[c]).astype(np.int64)], exact


# ---------------------------------------------------------------- links ----
class Links:
    """The link of a cluster: layer or disc x ring (the module's position along z in the barrel, the cluster table's
    ring on a disc) x z side x module facing (quant_bits.py's definition; the facing flips the sign of a local angle,
    so it is in the key). Code = surface * 10000 + ring * 100 + z side * 10 + facing."""

    def __init__(self, geo, geometry_json):
        with open(geometry_json) as fh:
            mods = json.load(fh)["modules"]
        from common import OP
        ring = np.full(len(geo["detId"]), -1, np.int64)
        mi = OP.module_index(geo, np.array([m["detId"] for m in mods], np.int64))
        for m, i in zip(mods, mi):
            if i >= 0 and m["subdet"] == "PXB":
                ring[i] = m["module"]
        self.barrel_ring = ring
        disc = SPN.is_disc_surface(np.asarray(geo["surface"]))
        radial = (geo["ez"][:, :2] * geo["origin"][:, :2]).sum(1) > 0
        self.facing = np.where(disc, geo["ez"][:, 2] > 0, radial).astype(np.int64)
        self.surface = np.asarray(geo["surface"]).astype(np.int64)
        self.disc = disc
        self.flip_facing = False          # planted defect: the facing bit flipped in the key

    def code(self, mi, ring_col, global_z):
        ok = mi >= 0
        m = np.maximum(mi, 0)
        ring = np.where(self.disc[m], np.asarray(ring_col).astype(np.int64), self.barrel_ring[m])
        fac = self.facing[m] ^ 1 if self.flip_facing else self.facing[m]
        c = self.surface[m] * 10000 + ring * 100 + (np.asarray(global_z) > 0).astype(np.int64) * 10 + fac
        return np.where(ok, c, -1)

    @staticmethod
    def name(code):
        s, r, z, f = code // 10000, (code // 100) % 100, (code // 10) % 10, code % 10
        return f"{'ID' + str(s - 10) if s > 10 else 'IL' + str(s)}_ring{r}_z{'+' if z else '-'}_facing{f}"


class LinkTables:
    """The fitted per-link angle ranges and angle-error tables (fit_links.py writes the file).

    Pools: a link belongs to one pool with a sign per angle (sA, sB): value in the pool frame = sign x value in the
    link's frame. Pool 0 is the fallback of links that had no fit clusters (all links pooled, signs +1).
    Per pool: range[source]["A" | "B"] = (lo, hi) in the pool frame (source "net": network-resolution angles);
    table[source]["A" | "B"] = edges of the true cot (pool frame), mean and width of (reported - true);
    qvar[range kind][angle][bits] = the measured variance of (decoded - input). global_range["A" | "B"]: one range
    for all links (link frame)."""

    def __init__(self, path):
        with open(path) as fh:
            self.doc = json.load(fh)
        d = self.doc
        codes = np.array(sorted(int(k) for k in d["links"]), np.int64)
        self.codes = codes
        self.pool_of = np.array([d["links"][str(c)]["pool"] for c in codes], np.int64)
        self.sign = {a: np.array([d["links"][str(c)]["s" + a] for c in codes], np.float64) for a in "AB"}
        self.npool = len(d["pools"])
        self.range = {a: np.array([p["range"][a] for p in d["pools"]], np.float64) for a in "AB"}
        self.global_range = {a: np.array(d["global_range"][a], np.float64) for a in "AB"}
        self.tables = {}
        for src in d["pools"][0]["table"]:
            for a in "AB":
                nmax = max(len(p["table"][src][a]["mean"]) for p in d["pools"])
                edges = np.full((self.npool, nmax - 1 if nmax > 1 else 1), np.inf)
                mean = np.zeros((self.npool, nmax))
                width = np.ones((self.npool, nmax))
                for i, p in enumerate(d["pools"]):
                    t = p["table"][src][a]
                    n = len(t["mean"])
                    edges[i, :n - 1] = t["edges"][1:-1]
                    mean[i, :n], width[i, :n] = t["mean"], t["width"]
                    mean[i, n:], width[i, n:] = t["mean"][-1], t["width"][-1]
                self.tables[(src, a)] = (edges, mean, width)

    def link_index(self, code):
        """(pool, sA, sB) per cluster code; unknown links: pool 0, signs +1."""
        i = np.searchsorted(self.codes, code)
        i = np.minimum(i, len(self.codes) - 1)
        ok = self.codes[i] == code
        return (np.where(ok, self.pool_of[i], 0), np.where(ok, self.sign["A"][i], 1.0),
                np.where(ok, self.sign["B"][i], 1.0))

    def qvar(self, kind, angle, bits):
        """Per pool, the measured variance of (decoded - input) for a bit count ("F" or None: 0)."""
        if bits in (None, "F", 0):
            return np.zeros(self.npool)
        return np.array([p["qvar"][kind][angle][str(bits)] for p in self.doc["pools"]], np.float64)


class LinkLookup:
    """The replay's angle-error mode "linktable" for one arm: per cluster the mean of (reported - true) and the
    assumed sigma, in the replay's order (code alpha = SmartPixels beta, code beta = SmartPixels alpha), from the
    cluster's link pool at the PREDICTED cot; sigma^2 = table width^2 + the measured quantization variance."""

    def __init__(self, LT, source="net", bits_A=None, bits_B=None, range_kind="link", scale=1.0, qvar_on=True):
        self.t = {a: LT.tables[(source, a)] for a in "AB"}
        self.q = {"A": LT.qvar(range_kind, "A", bits_A) if qvar_on else np.zeros(LT.npool),
                  "B": LT.qvar(range_kind, "B", bits_B) if qvar_on else np.zeros(LT.npool)}
        self.scale = scale                     # planted defects only

    def lookup(self, K, ci, cot_a_pred, cot_b_pred):
        pool = K["sb_pool"][ci]
        mean = np.zeros((len(ci), 2))
        sig = np.ones((len(ci), 2))
        for col, a, pred, s in ((0, "B", cot_a_pred, K["sb_sB"][ci]), (1, "A", cot_b_pred, K["sb_sA"][ci])):
            edges, m, w = self.t[a]
            x = s * np.nan_to_num(pred)
            k = (x[:, None] >= edges[pool]).sum(1)
            mean[:, col] = s * m[pool, k]
            sig[:, col] = np.sqrt(w[pool, k] ** 2 + self.q[a][pool]) * self.scale
        return mean, sig


# ---------------------------------------------------------------- per chunk ----
def base(ctx, geo, A, net, links, LT=None, defect=""):
    """Everything about a chunk's clusters that no arm changes (see the module doc). A: the chunk's arrays with run,
    luminosityBlock, event and the EXTRA_CLUSTER_COLS of the cluster table. defect: a planted defect ("one_draw_per_event",
    "residual_negated")."""
    K = ctx["K"]
    n = len(K["detId"])
    ev = K["event"]
    J = SPN.cluster_columns(A, list(EXTRA_CLUSTER_COLS), disc_scope="all")
    X = {c: ak.to_numpy(ak.flatten(J[c])) for c in EXTRA_CLUSTER_COLS}
    run, lumi, evt = (ak.to_numpy(A[k]) for k in ("run", "luminosityBlock", "event"))
    h = cluster_hash(run[ev], lumi[ev], evt[ev], K["detId"], K["localX"], K["localY"])
    mi = K["mi"]
    m = np.maximum(mi, 0)
    hw, hl = geo["hw"][m], geo["hl"][m]
    x, y = K["localX"].astype(np.float64), K["localY"].astype(np.float64)
    tx, ty = (np.clip(np.floor((v + h_) / SUBAREA_CM), 0, 63) for v, h_ in ((x, hw), (y, hl)))
    B = {"hash": h, "x": x, "y": y, "x0": -hw + tx * SUBAREA_CM, "y0": -hl + ty * SUBAREA_CM,
         "phase_x": uniform(h, SALT["phase_localX"]), "phase_y": uniform(h, SALT["phase_localY"]),
         "gauss_x": normal(h, SALT["noise1"], SALT["noise2"]), "gauss_y": normal(h, SALT["noise3"], SALT["noise1"] + 100)}
    # the cap: at most three clusters per sub-area per event, three kept at random by the cluster's hash
    tile = ((ev.astype(np.int64) * 8192 + m) * 64 + tx.astype(np.int64)) * 64 + ty.astype(np.int64)
    capkey = uniform(h, SALT["cap"])
    o = np.lexsort((capkey, tile))
    first = np.r_[True, tile[o][1:] != tile[o][:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    rank = np.empty(n, np.int64)
    rank[o] = np.arange(n) - start
    B["sent"] = (rank < CAP_PER_SUBAREA) | (mi < 0)
    B["n_in_subarea"] = np.empty(n, np.int64)
    cnt = np.diff(np.r_[np.flatnonzero(first), n])
    B["n_in_subarea"][o] = np.repeat(cnt, cnt)
    # angles: today's (class C keeps them), and the network-resolution ones for class T
    tA, tB = K["tpLocalCotBeta"].astype(np.float64), K["tpLocalCotAlpha"].astype(np.float64)
    T = (K["tpEmuClass"] == 1) & (np.abs(tA) < 50) & (np.abs(tB) < 50)
    B["class_T"] = T
    B["true_A"], B["true_B"] = tA, tB
    B["today_A"], B["today_B"] = K["localCotBeta"].astype(np.float64), K["localCotAlpha"].astype(np.float64)
    B["today_hasA"], B["today_hasB"] = K["hasBeta_raw"] > 0, K["hasAlpha_raw"] > 0
    u = uniform(h, SALT["draw"])
    j, exact = net.draw(tA[T], tB[T], u[T])
    if defect == "one_draw_per_event":                      # one network cluster's residual for the whole event
        evT = ev[T]
        _, fi = np.unique(evT, return_index=True)
        j = j[fi][np.searchsorted(evT[fi], evT)]
    sgn = -1.0 if defect == "residual_negated" else 1.0
    B["net_A"], B["net_B"] = B["today_A"].copy(), B["today_B"].copy()
    B["net_A"][T] = tA[T] + sgn * net.rA[j]
    B["net_B"][T] = tB[T] + sgn * net.rB[j]
    B["net_hasA"], B["net_hasB"] = B["today_hasA"] | T, B["today_hasB"] | T
    B["draw_row"] = np.full(n, -1, np.int64)
    B["draw_row"][T] = j
    B["exact_cell"] = np.zeros(n, bool)
    B["exact_cell"][T] = exact
    B["link"] = links.code(mi, X["ring"], X["globalZ"])
    B["eta"] = np.arcsinh(X["globalZ"].astype(np.float64) / np.maximum(X["globalR"].astype(np.float64), 1e-9))
    tpg = K["tp_global"]
    charge = np.where(tpg >= 0, ctx["TP"]["charge"][np.maximum(tpg, 0)], 0.0)
    B["target"] = T & (charge != 0) & (X["tpPt"] >= 2.0) & (X["tpChargeFrac"] >= 0.5)
    if LT is not None:
        B["pool"], B["sA"], B["sB"] = LT.link_index(B["link"])
    return B


def emulate(ctx, B, spec, LT):
    """The context an arm's refit reads. spec keys (all optional):
      step_x_um, step_y_um   quantization step of the SmartPixels x (-> localY) and y (-> localX); None: today's float
      noise                  True: Gaussian noise of sigma = step / sqrt(12) in place of the quantization
      angles                 "today" (untouched) | "network" (true cot + drawn network residual on class T)
      bits_alpha, bits_beta  SmartPixels alpha / beta: None = float, an int = bits per link range (2 codes are flags)
      range                  "link" (per link pool) | "global"
      cap                    True: only three clusters per sub-area per event are sent
    Returns a shallow copy of ctx with a new cluster dict; the cluster order and indices never change."""
    K = dict(ctx["K"])
    out = dict(ctx)
    out["K"] = K
    sx, sy = spec.get("step_x_um"), spec.get("step_y_um")
    for col, v, v0, ph, g, step in (("localX", B["x"], B["x0"], B["phase_x"], B["gauss_x"], sy),
                                    ("localY", B["y"], B["y0"], B["phase_y"], B["gauss_y"], sx)):
        if step:
            s_cm = step * 1e-4
            K[col] = v + g * s_cm / np.sqrt(12.0) if spec.get("noise") else quantize_position(v, v0, s_cm, ph)
    if spec.get("angles", "today") == "network":
        vA, vB, hasA, hasB = B["net_A"].copy(), B["net_B"].copy(), B["net_hasA"], B["net_hasB"]
        n = len(vA)
        flag = {"A": np.zeros(n, np.int8), "B": np.zeros(n, np.int8)}
        lo = {a: np.full(n, -np.inf) for a in "AB"}
        hi = {a: np.full(n, np.inf) for a in "AB"}
        for a, v, has, bits, s in (("A", vA, hasA, spec.get("bits_alpha"), B["sA"]),
                                   ("B", vB, hasB, spec.get("bits_beta"), B["sB"])):
            if not bits:
                continue
            if spec.get("range", "link") == "global":
                l_, h_ = np.full(n, LT.global_range[a][0]), np.full(n, LT.global_range[a][1])
            else:                                         # the pool's range mapped back into the link's frame
                r = LT.range[a][B["pool"]]
                l_, h_ = np.where(s > 0, r[:, 0], -r[:, 1]), np.where(s > 0, r[:, 1], -r[:, 0])
            q, f = quantize_angle(v, l_, h_, int(bits))
            v[has] = q[has]
            flag[a][has] = f[has]
            lo[a], hi[a] = l_, h_
        K["localCotBeta"], K["localCotAlpha"] = vA, vB
        K["hasBeta_raw"], K["hasAlpha_raw"] = hasA.astype(np.int8), hasB.astype(np.int8)
        K["hasBeta"] = (hasA & (np.abs(vA) <= MEAS_ANGLE_MAX)).astype(np.int8)
        K["hasAlpha"] = (hasB & (np.abs(vB) <= MEAS_ANGLE_MAX)).astype(np.int8)
        K["sb_flagBeta"], K["sb_flagAlpha"] = flag["A"], flag["B"]
        K["sb_loBeta"], K["sb_hiBeta"], K["sb_loAlpha"], K["sb_hiAlpha"] = lo["A"], hi["A"], lo["B"], hi["B"]
    if "pool" in B:
        K["sb_pool"], K["sb_sA"], K["sb_sB"] = B["pool"], B["sA"], B["sB"]
    if spec.get("cap"):
        kkey = np.where(B["sent"], ctx["kkey"], -1)       # a cluster that is not sent is in no window
        out["kkey"] = kkey
        out["ko"] = np.argsort(kkey, kind="stable")
    return out


def blank_unsent_information(ctx):
    """The "reads nothing the chip does not send" check: the cluster size, the published errors and the class set to
    NaN / a sentinel in the context an arm reads. The output must not change."""
    K = dict(ctx["K"])
    for c in ("sizeX", "sizeY", "sigX", "sigY", "sigAlpha", "sigBeta"):
        K[c] = np.full(len(K[c]), np.nan)
    K["tpEmuClass"] = np.full(len(K["tpEmuClass"]), -77)
    for c in ("tpLocalCotAlpha", "tpLocalCotBeta"):
        K[c] = np.full(len(K[c]), np.nan)
    return dict(ctx, K=K)
