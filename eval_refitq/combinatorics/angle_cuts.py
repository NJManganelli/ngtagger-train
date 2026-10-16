"""THE CENSUS ANGLE CUT FILE (WP-D0b; sheet angle.cut_level, decision 2): fitted from a calibration
census's records (census_angles.Recorder; tp_findability --calibrate), read by the census
(census_angles.CutSet).

    python angle_cuts.py fit CAL_CENSUS -o cuts.json [--k-alpha K --k-beta K --mode scaled]
                                         [--s6-from CENSUS_AT_K]

PER CONSUMER, PER BIN, the rank-n+1 quantile at the cut level (0.99) of the consumer's own statistic
(spdefs.consumer_quantile), on its calibration population: correct candidates whose angle-bearing IT
clusters are class T, of TPs born inside 2.5 cm (truth.genealogy codes 0-2) above the stage's pT
floor, each TP weighted once per bin. Bins (what the census can apply):
  S1   floor x the cluster's emitted-key table row             merged by spdefs.merge_bins along rows
  S2a  floor x seed pair (la, lb)                               never merged (a sparse pair reported)
  S2z  floor x seed pair x the inner cluster's row              merged along rows
  S3 / S4w  role x seed layer set x target x seed-measured pT bin x floor
                                                                 merged along pT, then across seed
                                                                 layer sets of the same arity, then
                                                                 over every seed (role, target, floor)
  S6   build x S5 state x S5 key x S6 angle set x n_meas         sparse n pooled by tail probability
S3 / S4w use ONE statistic per consumer, s = max over its angles of the per-bin empirical CDF of
|pull| (TP-weighted), cut at its quantile s*, stored as the per-angle thresholds F_a^-1(s*); a single
angle's subset is that angle's own quantile. S6 within a merged n group: each track's CDF value under
its own n (the per-n empirical CDF where n holds >= 200 TPs, else the nearest populated n' with chi2
scaled by n'/n), the group's quantile v*, mapped back to a per-n threshold F_n^-1(v*). Bins without
calibration data take their merged group's threshold.

Statistics are recomputed from the recorded components at the file's (k, mode) (census_angles'
functions, the ones the census runs); at the calibration run's own k they reproduce the recorded
statistic, which `fit` asserts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import census_angles as CA          # noqa: E402
import tracklet_topology_cost as M  # noqa: E402
from ngtagger.spdefs import angles as SD  # noqa: E402

INF = float("inf")
SUBS = ("ab", "a", "b")
N_MAX = {"ab": 8, "a": 4, "b": 4}


# ==========================================================================
# statistics from components
# ==========================================================================
def _comp(R, p=""):
    return {k: R[p + k] for k in CA.COMP_KEYS}


def _rep(R, p, ka, kb, mode):
    with np.errstate(all="ignore"):
        h, sh, lam, sl = CA.corrected_of(_comp(R, p), ka, kb, mode)
        return h, sh, CA.represent(R[p + "r"], R[p + "z"], h, sh, lam, sl)


def stat_s1(R, ka, kb, mode):
    h, sh, _ = _rep(R, "", ka, kb, mode)
    with np.errstate(all="ignore"):
        return CA.s1_stat(h, np.maximum(sh, 1e-12), R["r"], R["floor"])


def stat_pair(R, ka, kb, mode):
    """(S2a, S2z) of pair records (S2z NaN unless both clusters are IT)."""
    _, _, (kA, sA, _, _, zA, szA) = _rep(R, "A_", ka, kb, mode)
    _, _, (kB, sB, _, _, zB, szB) = _rep(R, "B_", ka, kb, mode)
    useA = R["A_dir_ok"] & (R["A_layer"] <= 4) & (R["A_layer"] >= 1)
    useB = R["B_dir_ok"] & (R["B_layer"] <= 4) & (R["B_layer"] >= 1)
    with np.errstate(all="ignore"):
        st_a = CA.s2a_stat(R["kap_pair"], kA, sA, kB, sB, useA, useB)
        both_it = (R["A_layer"] <= 4) & (R["A_layer"] >= 1) & (R["B_layer"] <= 4) & (R["B_layer"] >= 1)
        st_z = np.where(both_it, np.where(useA & useB, CA.s2z_stat(zA, szA, zB, szB), 0.0), np.nan)
    return st_a, st_z


def stat_win(R, ka, kb, mode):
    _, _, (kap, skap, cot, scot, _, _) = _rep(R, "", ka, kb, mode)
    with np.errstate(all="ignore"):
        return CA.cand_pulls(kap, skap, cot, scot, R["kap_pred"], R["cot_pred"])


def s6_parts(T, s5, key, sub):
    """(chi2, n) of S6 records under S5 state `s5` and key `key`, over angle subset `sub`."""
    c2 = np.zeros(len(T["n_a"]))
    n = np.zeros(len(c2), np.int64)
    if "a" in sub:
        c2 = c2 + T[f"c2a_{s5}_{key}"]
        n = n + T["n_a"]
    if "b" in sub:
        c2 = c2 + T[f"c2b_{s5}_{key}"]
        n = n + T["n_b"]
    return c2, n


# ==========================================================================
# populations and weights
# ==========================================================================
def _it(layer):
    return (layer >= 1) & (layer <= 4)


def population(R, kind):
    """The calibration population of a record table (class T, born inside 2.5 cm, the TP above
    the stage's floor); `kind` s1 | pair | win | trk."""
    pop = (R["gen"] >= 0) & (R["gen"] <= 2) & (R["tp_pt"] >= R["floor"])
    if kind in ("s1", "win"):
        pop &= R["is_T"].astype(bool) & R["dir_ok"].astype(bool)
    elif kind == "pair":
        for p in ("A_", "B_"):
            pop &= ~_it(R[p + "layer"]) | (R[p + "is_T"].astype(bool) & R[p + "dir_ok"].astype(bool))
    elif kind == "trk":
        pop &= R["all_T"].astype(bool)
    return pop


def class_c(R, kind):
    """The class-C reporting population: as population(), with some angle-bearing IT cluster of
    class C (reported, never fitted)."""
    pop = (R["gen"] >= 0) & (R["gen"] <= 2) & (R["tp_pt"] >= R["floor"])
    if kind in ("s1", "win"):
        return pop & ~R["is_T"].astype(bool) & R["dir_ok"].astype(bool)
    if kind == "pair":
        c = np.zeros(len(pop), bool)
        for p in ("A_", "B_"):
            c |= _it(R[p + "layer"]) & ~R[p + "is_T"].astype(bool)
        return pop & c
    return pop & ~R["all_T"].astype(bool)


def tp_weights(tp_key, unit, ev_w=None):
    """Weight 1 per TP per unit (1 / its candidates in the unit), times the event weights."""
    if not len(tp_key):
        return np.zeros(0)
    _, inv, cnt = np.unique(np.stack([np.asarray(unit, np.int64), np.asarray(tp_key, np.int64)], 1),
                            axis=0, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv.ravel()]
    return w if ev_w is None else w * ev_w


def n_tp(tp_key, unit, n_units):
    if not len(tp_key):
        return np.zeros(n_units, np.int64)
    u = np.unique(np.stack([np.asarray(unit, np.int64), np.asarray(tp_key, np.int64)], 1), axis=0)
    return np.bincount(u[:, 0], minlength=n_units)


def wquantile(x, w, level=CA.LEVEL):
    """The rank-n+1 weighted quantile of one sample (spdefs.consumer_quantile)."""
    if not len(x):
        return np.nan
    return float(SD.consumer_quantile(x, np.zeros(len(x), np.int64), levels=(level,), w=w,
                                      n_bins=1)[0, 0])


def wcdf(x_sorted, w_sorted):
    """(x, cumulative weight fraction) of a sorted weighted sample."""
    cw = np.cumsum(w_sorted)
    return x_sorted, cw / max(cw[-1], 1e-300)


def cdf_at(xs, F, v):
    """Weighted ECDF value at v: the weight fraction of the sample <= v."""
    j = np.searchsorted(xs, v, side="right")
    return np.where(j > 0, F[np.clip(j - 1, 0, len(F) - 1)], 0.0)


def cdf_inv(xs, F, p):
    """The smallest sample value whose weighted ECDF reaches p."""
    j = np.searchsorted(F, np.asarray(p) * (1 - 1e-12), side="left")
    return xs[np.clip(j, 0, len(xs) - 1)]


# ==========================================================================
# merging
# ==========================================================================
def merge_rows(units, counts, table):
    """S1 / S2z: within each (prefix, surface) the units' table rows, in signed-key order, merged
    by spdefs.merge_bins. units: list of tuples whose last element is the row; returns group id
    per unit (the smallest member index)."""
    rows = table.doc["keys"]["emitted"]["rows"]
    grp = np.arange(len(units))
    doms = {}
    for i, u in enumerate(units):
        r = rows[u[-1]]
        doms.setdefault((u[:-1], r["surface"]), []).append((r["lo"], i))
    for d, items in doms.items():
        items.sort()
        idx = [i for _, i in items]
        bounds = [(rows[units[i][-1]]["lo"], rows[units[i][-1]]["hi"]) for i in idx]
        for g in SD.merge_bins(bounds, [counts[i] for i in idx], CA.MIN_TP):
            members = [idx[j] for j in g]
            for i in members:
                grp[i] = min(members)
    return grp


def merge_axis(counts, domains, positions, start=None):
    """Merge units (or the groups of `start`) along `positions` within each domain until every
    group holds >= MIN_TP TPs: the smallest group joins its smaller neighbour. Returns group id per
    unit."""
    n_u = len(counts)
    grp = np.arange(n_u) if start is None else np.asarray(start).copy()
    gids = np.unique(grp)
    g_dom, g_pos, g_n = {}, {}, {}
    for g in gids:
        m = np.flatnonzero(grp == g)
        g_dom[g] = domains[m[0]]
        g_pos[g] = min(positions[i] for i in m)
        g_n[g] = int(sum(counts[i] for i in m))
    by_dom = {}
    for g in gids:
        by_dom.setdefault(g_dom[g], []).append(g)
    remap = {}
    for d, gs in by_dom.items():
        gs.sort(key=lambda g: g_pos[g])
        groups = [[g] for g in gs]
        tot = [g_n[g] for g in gs]
        while len(groups) > 1:
            j = int(np.argmin(tot))
            if tot[j] >= CA.MIN_TP:
                break
            if j == 0:
                nb = 1
            elif j == len(groups) - 1:
                nb = j - 1
            else:
                nb = j - 1 if tot[j - 1] <= tot[j + 1] else j + 1
            a, b = sorted((j, nb))
            groups[a] += groups[b]
            tot[a] += tot[b]
            del groups[b], tot[b]
        for g_list in groups:
            for g in g_list:
                remap[g] = min(g_list)
    return np.array([remap[g] for g in grp])


# ==========================================================================
# the menu's bins
# ==========================================================================
def menu_units(man, table):
    """Every bin the census of manifest `man` can apply, per consumer."""
    cfg = man["config"]
    ptmin = float(cfg["ptmin"])
    ipt = cfg.get("it_ptmin")
    floors = sorted({ptmin} | ({float(ipt)} if ipt is not None else set()))
    rows = table.doc["keys"]["emitted"]["rows"]
    rows_of = {L: [i for i, r in enumerate(rows) if r["surface"] == L and r["facing"] == 0]
               for L in (1, 2, 3, 4)}
    U = {"s1": [], "s2a": set(), "s2z": set(), "win": set(), "s6": []}
    for fl in floors:
        for L in (1, 2, 3, 4):
            U["s1"] += [(fl, i) for i in rows_of[L]]
    builds_of = {}
    for m, es in man["builds"].items():
        for e in es:
            builds_of.setdefault(e, []).append(m)
    for e, (lay, soft, follow) in enumerate(zip(man["seeds"], man["seed_soft"], man["seed_follow"])):
        fl = float(ipt) if (soft and ipt is not None) else ptmin
        la, lb = lay[0], lay[1]
        if la <= 4 or lb <= 4:
            U["s2a"].add((fl, la, lb))
        if la <= 4 and lb <= 4:
            for i in rows_of[la]:
                U["s2z"].add((fl, la, lb, i))
        code = M.seed_code(lay, soft)
        for pb in range(CA.N_PTBIN):
            if len(lay) == 3 and lay[2] <= 4:
                U["win"].add((0, code, lay[2], pb, fl))
            for L in follow:
                if L <= 4:
                    U["win"].add((1, code, L, pb, fl))
    bkeys = sorted({"+".join(sorted(v)) for v in builds_of.values()})
    for b in bkeys:
        for s5, key in CA.S5_REFITS:
            for sub in SUBS:
                for n in range(1, N_MAX[sub] + 1):
                    U["s6"].append((b, s5, key, sub, n))
    return {k: sorted(v) if isinstance(v, set) else v for k, v in U.items()}, floors


def _unit_index(units, keys):
    """Index of each record's bin key in `units` (-1 when the bin is not in the menu)."""
    pos = {u: i for i, u in enumerate(units)}
    return np.array([pos.get(k, -1) for k in keys], np.int64)


# ==========================================================================
# fitting
# ==========================================================================
def _fit_simple(units, unit_idx, stat, tp, ev_w, grp, level, pool_of=None):
    """Per unit, the group's weighted quantile of `stat` (records with unit_idx >= 0). A group
    without any calibration record takes the pooled quantile of its pool (pool_of(unit): e.g. the
    floor), so that no bin is left without a threshold; returns (thr, fallback unit indices)."""
    m = (unit_idx >= 0) & np.isfinite(stat)
    g = grp[unit_idx[m]]
    w = tp_weights(tp[m], unit_idx[m], None if ev_w is None else ev_w[m])
    thr_g = {}
    for gid in np.unique(grp):
        mm = g == gid
        thr_g[gid] = wquantile(stat[m][mm], w[mm], level) if mm.any() else np.nan
    thr = np.array([thr_g[grp[i]] for i in range(len(units))])
    fb = np.flatnonzero(~np.isfinite(thr))
    if len(fb):
        pools = np.array([pool_of(u) if pool_of else 0 for u in units], dtype=object)
        um = unit_idx[m]
        for i in fb:
            pm = np.array([pools[j] == pools[i] for j in um], bool) if len(um) else np.zeros(0, bool)
            thr[i] = wquantile(stat[m][pm], w[pm], level) if pm.any() else wquantile(stat[m], w, level)
    return thr, fb


def _fit_joint(units, unit_idx, pa, pb, tp, ev_w, grp, level):
    """S3 / S4w: per unit and angle subset, the per-angle thresholds of the joint statistic."""
    m = (unit_idx >= 0) & np.isfinite(pa) & np.isfinite(pb)
    g = grp[unit_idx[m]]
    w = tp_weights(tp[m], unit_idx[m], None if ev_w is None else ev_w[m])
    A, B = np.abs(pa[m]), np.abs(pb[m])
    out = {s: {} for s in SUBS}
    sstar = {}
    for gid in np.unique(grp):
        mm = g == gid
        if not mm.any():
            for s in SUBS:
                out[s][gid] = (np.nan, np.nan)
            continue
        a, b, ww = A[mm], B[mm], w[mm]
        oa, ob = np.argsort(a, kind="stable"), np.argsort(b, kind="stable")
        xa, Fa = wcdf(a[oa], ww[oa])
        xb, Fb = wcdf(b[ob], ww[ob])
        s = np.maximum(cdf_at(xa, Fa, a), cdf_at(xb, Fb, b))
        ss = wquantile(s, ww, level)
        sstar[gid] = ss
        out["ab"][gid] = (float(cdf_inv(xa, Fa, ss)), float(cdf_inv(xb, Fb, ss)))
        out["a"][gid] = (wquantile(a, ww, level), INF)
        out["b"][gid] = (INF, wquantile(b, ww, level))
    return ({s: [out[s][grp[i]] for i in range(len(units))] for s in SUBS},
            [sstar.get(grp[i], np.nan) for i in range(len(units))])


def _s6_cell(c2, n, tp, ev_w, level, n_max):
    """One S6 cell (build x S5 state x key x subset): per n in 1..n_max the chi2 threshold, the n
    groups, and the TPs per n."""
    ok = n > 0
    c2, n, tp = c2[ok], n[ok], tp[ok]
    ew = None if ev_w is None else ev_w[ok]
    ntp = n_tp(tp, n, n_max + 1)
    pop = [v for v in range(1, n_max + 1) if ntp[v] >= CA.MIN_TP]
    if not pop:
        return None
    w = tp_weights(tp, n, ew)
    F = {}
    for v in pop:
        mm = n == v
        o = np.argsort(c2[mm], kind="stable")
        F[v] = wcdf(c2[mm][o], w[mm][o])

    def ref(v):
        """(populated n', scale n'/v) whose CDF stands in for n = v."""
        if v in F:
            return v, 1.0
        nb = min(pop, key=lambda u: (abs(u - v), -u))
        return nb, nb / v
    grp = merge_axis([int(ntp[v]) for v in range(1, n_max + 1)],
                     [0] * n_max, list(range(1, n_max + 1)))
    vstar = {}
    for gid in np.unique(grp):
        members = [v for v in range(1, n_max + 1) if grp[v - 1] == gid]
        mm = np.isin(n, members)
        if not mm.any():
            continue
        vals = np.empty(int(mm.sum()))
        for v in members:
            sel = n[mm] == v
            if sel.any():
                r_, sc = ref(v)
                vals[sel] = cdf_at(F[r_][0], F[r_][1], c2[mm][sel] * sc)
        vstar[gid] = wquantile(vals, w[mm], level)
    thr = {}
    for v in range(1, n_max + 1):
        gid = grp[v - 1]
        vs = vstar.get(gid)
        if vs is None:                       # an n group without records: its nearest populated n
            vs = vstar[grp[ref(v)[0] - 1]]
        r_, sc = ref(v)
        thr[v] = float(cdf_inv(F[r_][0], F[r_][1], vs)) / sc
    return thr, grp, ntp


def statistics(R, k=(1.0, 1.0), mode="fixed"):
    """The cutting consumers' statistics of records R at (k, mode), computed once (fit, evaluate and
    the bootstrap reuse them)."""
    return {"s1": stat_s1(R["s1"], *k, mode), "pair": stat_pair(R["pair"], *k, mode),
            "win": stat_win(R["win"], *k, mode)}


def fit(R, man, model, k=(1.0, 1.0), mode="fixed", level=CA.LEVEL, ev_w=None, s6_R=None,
        only=None, stats=None):
    """The cut file's per-consumer thresholds from records R (census_angles.load_records) of a
    calibration census with manifest `man`. ev_w: per-event weights (the bootstrap). s6_R: the S6
    records to use (a calibration census at the file's k, scaled mode); default R's. only: restrict
    to {consumer: config} for the bootstrap (win subset, S6 cells). stats: statistics(R, k, mode),
    when already computed. Returns the cut-file body."""
    units, floors = menu_units(man, model.table)
    S = statistics(R, k, mode) if stats is None else stats
    out = {}

    def evw(T):
        return None if ev_w is None else ev_w[T["event"]]

    def want(c):
        return only is None or c in only
    if want("s1"):
        T = R["s1"]
        st = S["s1"]
        pop = population(T, "s1")
        ui = _unit_index(units["s1"], list(zip(T["floor"].tolist(), T["row"].tolist())))
        ui = np.where(pop, ui, -1)
        cnt = n_tp(T["tp_key"][ui >= 0], ui[ui >= 0], len(units["s1"]))
        grp = merge_rows(units["s1"], cnt, model.table)
        thr, fb = _fit_simple(units["s1"], ui, st, T["tp_key"], evw(T), grp, level, lambda u: u[0])
        out["s1"] = {"bins": units["s1"], "thr": thr.tolist(), "n_tp": cnt.tolist(),
                     "group": grp.tolist(), "fallback": fb.tolist()}
    if want("s2a") or want("s2z"):
        T = R["pair"]
        st_a, st_z = S["pair"]
        pop = population(T, "pair")
        if want("s2a"):
            ui = _unit_index(units["s2a"], list(zip(T["floor"].tolist(), T["la"].tolist(),
                                                    T["lb"].tolist())))
            ui = np.where(pop, ui, -1)
            cnt = n_tp(T["tp_key"][ui >= 0], ui[ui >= 0], len(units["s2a"]))
            grp = np.arange(len(units["s2a"]))
            thr, fb = _fit_simple(units["s2a"], ui, st_a, T["tp_key"], evw(T), grp, level,
                                  lambda u: u[0])
            out["s2a"] = {"bins": units["s2a"], "thr": thr.tolist(), "n_tp": cnt.tolist(),
                          "group": grp.tolist(), "fallback": fb.tolist()}
        if want("s2z"):
            keys = list(zip(T["floor"].tolist(), T["la"].tolist(), T["lb"].tolist(),
                            T["A_row"].tolist()))
            ui = _unit_index(units["s2z"], keys)
            ui = np.where(pop & np.isfinite(st_z), ui, -1)
            cnt = n_tp(T["tp_key"][ui >= 0], ui[ui >= 0], len(units["s2z"]))
            grp = merge_rows(units["s2z"], cnt, model.table)
            thr, fb = _fit_simple(units["s2z"], ui, st_z, T["tp_key"], evw(T), grp, level,
                                  lambda u: (u[0], u[1], u[2]))
            out["s2z"] = {"bins": units["s2z"], "thr": thr.tolist(), "n_tp": cnt.tolist(),
                          "group": grp.tolist(), "fallback": fb.tolist()}
    if want("win"):
        T = R["win"]
        pa, pb = S["win"]
        pop = population(T, "win")
        keys = list(zip(T["role"].tolist(), T["seed"].tolist(), T["lc"].tolist(),
                        T["ptbin"].tolist(), T["floor"].tolist()))
        U_ = units["win"]
        ui = np.where(pop, _unit_index(U_, keys), -1)
        cnt = n_tp(T["tp_key"][ui >= 0], ui[ui >= 0], len(U_))
        # merge along pT within (role, seed, target, floor); then across seed layer sets of one
        # arity within (role, target, floor, arity); then over every seed (role, target, floor)
        arity = [len(str(u[1] % 10 ** 7)) // 2 + (len(str(u[1] % 10 ** 7)) % 2) for u in U_]
        g1 = merge_axis(cnt, [(u[0], u[1], u[2], u[4]) for u in U_], [u[3] for u in U_])
        g2 = merge_axis(cnt, [(u[0], u[2], u[4], a) for u, a in zip(U_, arity)],
                        [u[1] for u in U_], start=g1)
        grp = merge_axis(cnt, [(u[0], u[2], u[4]) for u in U_], [a * 10 ** 9 + u[1] for u, a
                                                                in zip(U_, arity)], start=g2)
        thr, sstar = _fit_joint(U_, ui, pa, pb, T["tp_key"], evw(T), grp, level)
        # a merged group without any calibration record (a target the menu never reached in the
        # calibration sample) takes the pooled thresholds of its role and floor
        fb = [i for i in range(len(U_)) if not np.isfinite(thr["ab"][i][0])]
        if fb:
            m = (ui >= 0) & np.isfinite(pa) & np.isfinite(pb)
            for i in fb:
                pm = m & np.isin(ui, [j for j, u in enumerate(U_) if (u[0], u[4]) == (U_[i][0], U_[i][4])])
                if not pm.any():
                    pm = m
                pt, _ = _fit_joint([(0,)], np.where(pm, 0, -1), pa, pb, T["tp_key"], evw(T),
                                   np.zeros(1, np.int64), level)
                for s_ in SUBS:
                    thr[s_][i] = pt[s_][0]
        if only is not None and "win" in only:
            thr = {s: thr[s] for s in only["win"]}
        out["win"] = {"bins": U_, "thr": {s: [list(x) for x in v] for s, v in thr.items()},
                      "s_star": sstar, "n_tp": cnt.tolist(), "group": grp.tolist(), "fallback": fb}
    if want("s6"):
        T = (R if s6_R is None else s6_R)["trk"]
        pop = population(T, "trk")
        ew = evw(T)
        bins, thr, ntps, groups = [], [], [], []
        cells = sorted({(u[0], u[1], u[2], u[3]) for u in units["s6"]})
        if only is not None and "s6" in only:
            cells = [c for c in cells if c in only["s6"]]
        for (b, s5, key, sub) in cells:
            sel = pop & (T["build"] == b) & (T[f"ok_{s5}_{key}"] if f"ok_{s5}_{key}" in T
                                              else np.ones(len(pop), bool))
            c2, n = s6_parts(T, s5, key, sub)
            res = _s6_cell(c2[sel], n[sel], T["tp_key"][sel],
                           None if ew is None else ew[sel], level, N_MAX[sub])
            if res is None:
                raise SystemExit(f"S6 cell {(b, s5, key, sub)}: no n with >= {CA.MIN_TP} TPs")
            t, grp, ntp = res
            for v in range(1, N_MAX[sub] + 1):
                bins.append((b, s5, key, sub, v))
                thr.append(t[v])
                ntps.append(int(ntp[v]))
                groups.append(int(grp[v - 1]))
        out["s6"] = {"bins": bins, "thr": thr, "n_tp": ntps, "group": groups}
    return out, floors


def evaluate(R, doc, k=(1.0, 1.0), mode="fixed"):
    """Per consumer: (unit index, statistic, pass) of the records under cut-file body `doc`, at the
    file's k (records' statistics recomputed from components)."""
    ka, kb = k
    res = {}
    if "s1" in doc and "s1" in R:
        T = R["s1"]
        st = stat_s1(T, ka, kb, mode)
        ui = _unit_index([tuple(u) for u in doc["s1"]["bins"]],
                         list(zip(T["floor"].tolist(), T["row"].tolist())))
        thr = np.where(ui >= 0, np.asarray(doc["s1"]["thr"])[np.maximum(ui, 0)], np.inf)
        res["s1"] = (ui, st, st <= thr)
    if "pair" in R:
        T = R["pair"]
        st_a, st_z = stat_pair(T, ka, kb, mode)
        if "s2a" in doc:
            ui = _unit_index([tuple(u) for u in doc["s2a"]["bins"]],
                             list(zip(T["floor"].tolist(), T["la"].tolist(), T["lb"].tolist())))
            thr = np.where(ui >= 0, np.asarray(doc["s2a"]["thr"])[np.maximum(ui, 0)], np.inf)
            res["s2a"] = (ui, st_a, st_a <= thr)
        if "s2z" in doc:
            ui = _unit_index([tuple(u) for u in doc["s2z"]["bins"]],
                             list(zip(T["floor"].tolist(), T["la"].tolist(), T["lb"].tolist(),
                                      T["A_row"].tolist())))
            ui = np.where(np.isfinite(st_z), ui, -1)
            thr = np.where(ui >= 0, np.asarray(doc["s2z"]["thr"])[np.maximum(ui, 0)], np.inf)
            res["s2z"] = (ui, st_z, ~np.isfinite(st_z) | (st_z <= thr))
    if "win" in doc and "win" in R:
        T = R["win"]
        pa, pb = stat_win(T, ka, kb, mode)
        ui = _unit_index([tuple(u) for u in doc["win"]["bins"]],
                         list(zip(T["role"].tolist(), T["seed"].tolist(), T["lc"].tolist(),
                                  T["ptbin"].tolist(), T["floor"].tolist())))
        for sub, q in doc["win"]["thr"].items():
            q = np.asarray(q, np.float64)
            qa = np.where(ui >= 0, q[np.maximum(ui, 0), 0], np.inf)
            qb = np.where(ui >= 0, q[np.maximum(ui, 0), 1], np.inf)
            res[f"win_{sub}"] = (ui, np.maximum(np.abs(pa) / qa, np.abs(pb) / qb),
                                 (np.abs(pa) <= qa) & (np.abs(pb) <= qb))
    return res


def save(path, body, floors, R_man, model, k, mode, cal_dir, s6_dir=None, level=CA.LEVEL):
    """Write a cut file (atomic) with its identity."""
    cfg = R_man["config"]
    doc = {"schema": CA.CUT_SCHEMA,
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "identity": {"table_path": model.table_path, "table_sha256": model.table_sha,
                        "table_registry": model.registry, "level": level,
                        "k_alpha": float(k[0]), "k_beta": float(k[1]), "mode": mode,
                        "bench": cfg.get("bench"),
                        "d0_prior_cm": float(cfg["kf_opts"].get("d0_prior_cm",
                                                                CA.KF.D0_PRIOR_CM)),
                        "floors": floors, "pos_arb": cfg["pos_arb"],
                        "calibration": {"census": cal_dir, "s6_census": s6_dir or cal_dir,
                                        "inputs": [r["path"] for r in R_man["inputs"]],
                                        "format": R_man["format"]},
                        "seeds": R_man["seeds"], "seed_soft": R_man["seed_soft"],
                        "seed_follow": R_man["seed_follow"], "builds": R_man["builds"]},
            "rule": ("per bin, the rank-n+1 quantile (k = min(ceil(L (n + 1)), n)) at the cut level of "
                     "the consumer's own statistic on correct class-T candidates of TPs born inside "
                     "2.5 cm above the stage's floor, each TP weighted once per bin (angle_cuts)"),
            "notes": {"discs": CA.DISC_NOTE, "class_c": CA.CLASS_C_NOTE},
            **body}
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("census")
    f.add_argument("-o", "--out", required=True)
    f.add_argument("--k-alpha", type=float, default=1.0)
    f.add_argument("--k-beta", type=float, default=1.0)
    f.add_argument("--mode", default="fixed", choices=list(SD.K_MODES))
    f.add_argument("--s6-from", default=None,
                   help="the calibration census at the file's k (scaled mode) for S6")
    a = ap.parse_args()
    import tp_findability as TF
    d = TF.census_dir(a.census)
    man = json.load(open(os.path.join(d, "manifest.json")))
    if man["config"].get("angle_cuts") != "open":
        raise SystemExit(f"{d} is not a calibration census (angle_cuts {man['config'].get('angle_cuts')!r})")
    am = man["config"]["angle_model"]
    model = CA.Model(am["table_path"], am["k_alpha"], am["k_beta"], am["mode"])
    if model.table_sha != am["table_sha256"]:
        raise SystemExit("the table changed since the calibration census ran")
    R = CA.load_records(d)
    # at the census's own k the recomputed statistics are the recorded ones
    kc = (am["k_alpha"], am["k_beta"])
    for kind, fn, cols in (("s1", stat_s1, ("stat",)), ("pair", stat_pair, ("stat_a", "stat_z")),
                           ("win", stat_win, ("pa", "pb"))):
        got = fn(R[kind], *kc, am["mode"])
        got = got if isinstance(got, tuple) else (got,)
        for g_, c in zip(got, cols):
            ref = R[kind][c]
            both = np.isfinite(g_) & np.isfinite(ref)
            if (np.isfinite(g_) != np.isfinite(ref)).any() or \
                    np.max(np.abs(g_[both] - ref[both]), initial=0.0) > 1e-9 * max(1.0, np.max(np.abs(ref[both]), initial=1.0)):
                raise SystemExit(f"{kind}.{c}: the recomputed statistic differs from the recorded one")
    s6R = None
    s6d = None
    if a.s6_from:
        s6d = TF.census_dir(a.s6_from)
        # one cut file from two calibration censuses: one format
        TF.one_format({d: man, s6d: json.load(open(os.path.join(s6d, "manifest.json")))})
        s6R = CA.load_records(s6d)
    k = (a.k_alpha, a.k_beta)
    body, floors = fit(R, man, model, k, a.mode, s6_R=s6R)
    save(a.out, body, floors, man, model, k, a.mode, d, s6d)
    for c, v in body.items():
        thr = v["thr"] if c != "win" else [x[0] for x in v["thr"]["ab"]]
        fin = [t for t in thr if np.isfinite(t)]
        print(f"{c}: {len(v['bins'])} bins, {len(np.unique(v['group']))} groups, "
              f"threshold median {np.median(fin):.3g} (min {min(fin):.3g}, max {max(fin):.3g})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
