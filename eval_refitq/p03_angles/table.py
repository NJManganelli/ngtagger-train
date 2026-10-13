#!/usr/bin/env python
"""P0.3 step 2: the angle-uncertainty table (plan D.3 D10(b), D.0; decisions 2, 3, 11), under the
rule of pre-registration amendment 1 (preregistration.txt; PROPOSED 2026-10-13, pending approval).

Recipe (angle.error_table): class-T clusters with a usable direction and TP pT >= 1 GeV; per
surface (IL1-IL4, ID1-ID4) x signed incidence bin (signed cot theta barrel; signed 1/cot theta
discs, per module facing), merged to >= 200 fit clusters away from the plateau edges; per bin and
angle (alpha, beta) the centre (clipped mean of the B.1 clip), the core width (B.1 core of the
centred pulls) and the quantiles of |pull - centre| at 95 / 97.5 / 99 / 99.7 %, in both keys
(emitted, true). pT check on the fit sample (1-2 vs >= 2 GeV, D.0 agreement rule): a bin failing
in both signs is refitted at pT >= 2 GeV and flagged.

Test (amendment 1), on the test sample, per bin of the same binning:
 (i)  adequacy, for the rows whose centre and core width weighting consumers use (barrel alpha
      and beta, disc alpha): the after-table clipped mean and B.1 core width within 0 +- 0.25 and
      1 +- 0.1, widened by W_i sigma, W_i = t(N_i) the family-wise threshold for the N_i = 2 x
      (judged bin-angles) statistics, sigma = the larger of the event-level BOOTSTRAP spread of the
      whole fit-and-test chain (fit and test events resampled independently, every fitted step
      re-fitted per replica) and its RE-SPLIT (permutation) spread (the bootstrap stays in the basin
      of a bistable mode-seeded clip). Disc beta rows (no Gaussian core; weighting consumers use the
      full response, decision 3) are reported, not judged by (i);
 (ii) coverage: per bin, angle and level, the number of test clusters beyond the fitted quantile
      is judged against its null distribution "the table is right": beta-binomial with mean
      1 - E0, E0 = k / (n_fit + 1) (the order-statistic expectation of the inverted-CDF quantile),
      and variance max(event-PERMUTATION variance, exchangeable-cluster variance); the permutation
      re-splits the pooled fit + test events at the fit and test sizes and re-fits every step. The
      mid-p tail probability gives a signed z; pass iff no |z| > t(N) (N = every z-score read) and
      per level and key chi2 p > 0.01 / (levels x keys).
 False-veto rate: the same rule applied to every permutation replica (each a known-good split).

    python table.py diagnosis  [--rep 250] [--proc 10]   amendment-1 rule on the split where the v1 rule
                                                         failed (fit 3-4, test 5-10): diagnosis only
    python table.py known-good [--rep 250] [--proc 10]   amendment-1 demonstration: fit 9-10, test 3-8
    python table.py known-bad  [--rep 250] [--proc 10]   the four known-bad tables on the same split
    python table.py gated      [--rep 250] [--proc 10]   fit 1-2, test 3-10; refuses unless the
                                                         amendment-1 known-good passed, every known-bad
                                                         failed and $P03_OUT/amendment1_approval.txt exists
    python table.py fit-only   [--rep 250] [--proc 10]   the recipe on files 1-2 without the test (only
                                                         when the gated test cannot run): UNTESTED table
Outputs: $P03_OUT/p03_table_<mode>_a1.json (the v1-rule results of 2026-10-09 stay as they are).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

from ngtagger.spcheck.core import t_family  # noqa: E402
from ngtagger.spdefs import angles as SA  # noqa: E402

RULE = "P0.3 pre-registration amendment 1 (2026-10-13)"
SEED = 20261013
LEVELS = SA.LEVELS
BAND_W, BAND_C = 0.1, 0.25
WIDEN_V1 = 2.0          # the v1 rule's per-bin widening of (i): reported for comparison only
REGISTRY_SCALES = Path.home() / "smartpixels" / "sp_registry" / "results" / "appa.census.angle-scales.json"
DEMO_SPLIT = ((9, 10), (3, 4, 5, 6, 7, 8))
SPLITS = {"diagnosis": ((3, 4), (5, 6, 7, 8, 9, 10)), "known-good": DEMO_SPLIT, "known-bad": DEMO_SPLIT,
          "gated": (SA.FIT_FILES, SA.TEST_FILES), "fit-only": (SA.FIT_FILES, SA.FIT_FILES)}
TABLE_FILE = "angle_table_v1a1_canon_v3.json"
UNTESTED_FILE = "angle_table_v1a1_canon_v3_UNTESTED.json"
APPROVAL_FILE = "amendment1_approval.txt"
KNOWN_BADS = {
    "folded": "the folded |key| table (signs pooled; the round-4 table's binning), read by |key|",
    "cotbinned_disc": "disc rows binned like the barrel in signed cot theta (barrel edges; 75-98% of disc clusters in "
                      "|cot| > 2.5), per facing",
    "pooled_disc": "disc rows with the two module facings pooled",
    "wrong_key": "the table read in the wrong key (the true-key table looked up with the emitted key, and vice versa)",
    "test_scale_1p05": "input perturbation (amendment 1): every test-sample pull x 1.05, a 5 % resolution difference "
                       "between fit and test events (half the (i) tolerance; a production-batch difference of that size "
                       "would bias every cut's efficiency): the rule must detect it",
}
TEST_SCALE = {"test_scale_1p05": 1.05}


def result_name(mode: str) -> str:
    return f"p03_table_{mode}_a1.json"


# ---------------------------------------------------------------- binnings
def _fold(fb, n):
    h = n // 2
    return np.where(fb >= h, fb - h, h - 1 - fb)


def fine_index(kind: str, C: dict, cot: np.ndarray):
    """(facing used, fine bin index) of each cluster under binning `kind`, from its cot theta."""
    s = C["surface"]
    disc = SA.is_disc(s)
    fac = np.where(disc & (kind == "pooled_disc"), 0, C["facing"]).astype(np.int8)
    if kind == "cotbinned_disc":
        se = SA.signed_edges(SA.BARREL_ABS_EDGES)
        fb_cot = np.clip(np.searchsorted(se, cot, side="right") - 1, 0, len(se) - 2).astype(np.int16)
        fb = np.where(disc, fb_cot, SA.fine_bin(s, SA.incidence_key(s, cot)))
    else:
        fb = SA.fine_bin(s, SA.incidence_key(s, cot))
        if kind == "folded":
            for sv in SA.SURFACES:
                m = (s == sv) & (fb >= 0)
                fb[m] = _fold(fb[m], SA.N_FINE[sv])
    return fac, fb.astype(np.int16)


def n_fine(kind: str, s: int) -> int:
    if kind == "folded":
        return SA.N_FINE[s] // 2
    if kind == "cotbinned_disc" and s in SA.DISCS:
        return SA.N_FINE[1]
    return SA.N_FINE[s]


def bounds(kind: str, s: int) -> list:
    if kind == "folded":
        a = SA.abs_edges(s)
        return [(a[i], a[i + 1]) for i in range(len(a) - 1)]
    if kind == "cotbinned_disc" and s in SA.DISCS:
        return [SA.fine_bin_bounds(1, b) for b in range(SA.N_FINE[1])]
    return [SA.fine_bin_bounds(s, b) for b in range(SA.N_FINE[s])]


def facings(kind: str, s: int) -> tuple:
    return (0,) if (s in SA.BARREL or kind == "pooled_disc") else (-1, 1)


def build_rows(kind: str, C: dict, key: str):
    """Rows of a table of binning `kind` in `key` from the fit clusters C (merge rule on their
    counts). Returns (rows meta, rowmap {(s, f): array over fine bins -> row id})."""
    fac, fb = fine_index(kind, C, C["cotv"][key])
    rows, rowmap = [], {}
    for s in SA.SURFACES:
        for f in facings(kind, s):
            g = (C["surface"] == s) & (fac == f) & (fb >= 0)
            if not g.any():
                continue
            cnt = np.bincount(fb[g], minlength=n_fine(kind, s))
            bd = bounds(kind, s)
            arr = np.full(n_fine(kind, s), -1, np.int32)
            for grp in SA.merge_bins(bd, cnt):
                arr[grp] = len(rows)
                rows.append({"surface": int(s), "facing": int(f), "lo": float(bd[grp[0]][0]),
                             "hi": float(bd[grp[-1]][1]), "fine": [int(x) for x in grp],
                             "n_fit": int(cnt[grp].sum()), "binning": kind, "key": key})
            rowmap[(s, f)] = arr
    return rows, rowmap


def row_of(kind: str, rowmap: dict, C: dict, key: str) -> np.ndarray:
    fac, fb = fine_index(kind, C, C["cotv"][key])
    out = np.full(len(fb), -1, np.int32)
    for (s, f), arr in rowmap.items():
        m = (C["surface"] == s) & (fac == f) & (fb >= 0)
        out[m] = arr[fb[m]]
    return out


def mirror_row(rows: list, rowmap: dict, i: int) -> int:
    r = rows[i]
    if r["binning"] == "folded":
        return i
    n = len(rowmap[(r["surface"], r["facing"])])
    return int(rowmap[(r["surface"], r["facing"])][n - 1 - r["fine"][0]])


# ---------------------------------------------------------------- data
def load_split(files) -> dict:
    C = CM.load(files, CM.ANGLE_COLS, mask_fn=CM.table_population)
    assert (C["w"] == 1).all(), "the table population must be unsubsampled"
    return C


def event_index(ev: np.ndarray):
    u, inv = np.unique(ev, return_inverse=True)
    return u, inv.astype(np.int32)


def fit_rows(kind: str, rowmap: dict, C: dict, key: str, flagged) -> np.ndarray:
    """Fit-row id per cluster; clusters below 2 GeV in a pT-flagged row are left out (-1)."""
    r = row_of(kind, rowmap, C, key)
    if flagged:
        r = np.where(np.isin(r, list(flagged)) & (C["tp_pt"] < 2.0), -1, r)
    return r


def pooled_sample(fit: dict, test: dict) -> dict:
    """The fit and test clusters as one sample (for the permutation), with the columns the chain uses."""
    out = {c: np.concatenate([fit[c], test[c]]) for c in ("ev", "surface", "facing", "tp_pt")}
    for d in ("pull", "cotv"):
        out[d] = {k: np.concatenate([fit[d][k], test[d][k]]) for k in fit[d]}
    return out


# ---------------------------------------------------------------- the replica engine (fork workers)
G: dict = {}


def _sorted_slices(row: np.ndarray, n: int):
    o = np.argsort(row, kind="stable")
    r = row[o]
    return o, np.searchsorted(r, np.arange(n), "left"), np.searchsorted(r, np.arange(n), "right")


def prepare(fit: dict, test: dict, spec: dict, flags: dict | None = None, pooled: bool = False) -> dict:
    """spec = {test_key: {"kind", "lookup_key"}}; the table's binning `kind` is built in test_key
    on the fit sample and read with lookup_key; the test bins are the canonical rows of test_key on
    the same fit sample. flags = {test_key: set of row ids refitted at pT >= 2} (canonical tables).
    pooled=False: the chain's fit side is the fit sample, its test side the test sample (bootstrap).
    pooled=True: both sides hold the pooled fit + test clusters and a replica assigns each event to
    one side (permutation); the nominal assignment reproduces the actual split.
    Also per test bin and level, from the ACTUAL split: E0 (the expected coverage of a correct table,
    mixing the looked-up rows' k / (n_fit + 1); the level for clusters without a usable row) and the
    exchangeable-cluster variance of the fitted quantiles' coverage (sum over looked-up rows of the
    squared share x the Beta variance)."""
    P = {"keys": {}, "pooled": pooled}
    uf, _ = event_index(fit["ev"])
    if pooled:
        A = pooled_sample(fit, test)
        ua, eva = event_index(A["ev"])
        P["n_ev"], P["n_ev_fit"] = len(ua), len(uf)
        P["nominal_fit_events"] = np.isin(ua, uf)
        fside, tside, evf, evt = A, A, eva, eva
    else:
        ut, evt = event_index(test["ev"])
        _, evf = event_index(fit["ev"])
        P["n_ev_fit"], P["n_ev_test"] = len(uf), len(ut)
        fside, tside = fit, test
    lv = np.asarray(LEVELS)
    for key, sp in spec.items():
        rows, rowmap = build_rows(sp["kind"], fit, key)
        trows, trowmap = build_rows("canonical", fit, key)  # the test binning
        fl = (flags or {}).get(key, set())
        frow = fit_rows(sp["kind"], rowmap, fside, key, fl)
        lrow = row_of(sp["kind"], rowmap, tside, sp["lookup_key"])
        tbin = row_of("canonical", trowmap, tside, key)
        fo, fs, fe = _sorted_slices(frow, len(rows))
        to, ts, te = _sorted_slices(tbin, len(trows))
        # the actual split: fit counts per row, test clusters per bin and their looked-up rows
        nf_row = np.bincount(np.maximum(fit_rows(sp["kind"], rowmap, fit, key, fl), -1) + 1,
                             minlength=len(rows) + 1)[1:].astype(np.float64)
        lr_a = row_of(sp["kind"], rowmap, test, sp["lookup_key"])
        tb_a = row_of("canonical", trowmap, test, key)
        usable_row = nf_row >= SA.MIN_FIT
        e_row = np.where(usable_row[:, None], SA.expected_coverage(nf_row[:, None], lv[None, :]), lv[None, :])
        v_row = np.where(usable_row[:, None], SA.fit_coverage_variance(nf_row[:, None], lv[None, :]), 0.0)
        e_row = np.vstack([e_row, lv[None, :]])          # index -1: no row -> the level, no fit variance
        v_row = np.vstack([v_row, np.zeros((1, len(lv)))])
        E0 = np.full((len(trows), len(lv)), np.nan)
        Vfit = np.full((len(trows), len(lv)), np.nan)
        n_test = np.bincount(tb_a[tb_a >= 0], minlength=len(trows))
        for i in range(len(trows)):
            m = tb_a == i
            if not m.any():
                continue
            rr, cnt = np.unique(lr_a[m], return_counts=True)
            sh = cnt / cnt.sum()
            E0[i] = (sh[:, None] * e_row[rr]).sum(0)
            Vfit[i] = (sh[:, None] ** 2 * v_row[rr]).sum(0)
        P["keys"][key] = {
            "rows": rows, "trows": trows, "trowmap": trowmap, "flags": sorted(int(x) for x in fl), "kind": sp["kind"],
            "lookup_key": sp["lookup_key"],
            "fit_ev": evf[fo], "fit_p": {a: fside["pull"][a][fo] for a in SA.ANGLES}, "fs": fs, "fe": fe,
            "test_ev": evt[to], "test_p": {a: tside["pull"][a][to] for a in SA.ANGLES},
            "test_lrow": lrow[to], "ts": ts, "te": te, "n_lookup_missing": int(np.sum((lr_a < 0) & (tb_a >= 0))),
            "test_n": n_test.astype(int), "nf_row": nf_row, "E0": E0, "Vfit": Vfit}
    return P


def _fit_params(K, wf):
    nr = len(K["rows"])
    par = np.full((nr + 1, len(SA.ANGLES), 2 + len(LEVELS)), np.nan)      # centre, width, q...
    for i in range(nr):
        a0, a1 = K["fs"][i], K["fe"][i]
        if a1 <= a0:
            continue
        w = None if wf is None else wf[a0:a1]
        for j, a in enumerate(SA.ANGLES):
            e = SA.bin_estimates(K["fit_p"][a][a0:a1], LEVELS, w)
            if e is None:
                continue
            par[i, j, 0], par[i, j, 1] = e["centre"], e["width"]
            par[i, j, 2:] = [e["q"][f"{lv:g}"] for lv in LEVELS]
    return par


def _test_stats(K, par, wt):
    """Per test bin and angle: after-table clipped mean and B.1 core width, the test count, and
    the coverage at each level (clusters without a usable row count as outside)."""
    nt = len(K["trows"])
    out = np.full((nt, len(SA.ANGLES), 3 + len(LEVELS)), np.nan)        # centre_after, width_after, n, cov...
    lr = np.where(K["test_lrow"] >= 0, K["test_lrow"], len(K["rows"]))
    for j, a in enumerate(SA.ANGLES):
        p = K["test_p"][a]
        c, w = par[lr, j, 0], par[lr, j, 1]
        after = (p - c) / w
        dev = np.abs(p - c)
        inside = [dev <= par[lr, j, 2 + li] for li in range(len(LEVELS))]
        for i in range(nt):
            a0, a1 = K["ts"][i], K["te"][i]
            if a1 <= a0:
                continue
            ww = None if wt is None else wt[a0:a1]
            cc = SA.core_w(after[a0:a1], ww)
            if cc is not None:
                out[i, j, 0], out[i, j, 1] = cc["mean"], cc["rms0"]
            tot = (a1 - a0) if ww is None else ww.sum()
            out[i, j, 2] = tot
            for li in range(len(LEVELS)):
                ins = inside[li][a0:a1]
                out[i, j, 3 + li] = (ins.sum() if ww is None else (ww * ins).sum()) / tot if tot > 0 else np.nan
    return out


def _weights(b: int):
    """(fit weights, test weights) per event for replica b: None = unweighted (bootstrap nominal)."""
    P, scheme = G["P"], G["scheme"]
    if scheme == "bootstrap":
        if b == 0:
            return None, None
        rng = np.random.default_rng([SEED, b])
        return (rng.multinomial(P["n_ev_fit"], np.full(P["n_ev_fit"], 1.0 / P["n_ev_fit"])),
                rng.multinomial(P["n_ev_test"], np.full(P["n_ev_test"], 1.0 / P["n_ev_test"])))
    if b == 0:
        fm = P["nominal_fit_events"]
    else:
        rng = np.random.default_rng([SEED, 2, b])
        fm = np.zeros(P["n_ev"], bool)
        fm[rng.choice(P["n_ev"], P["n_ev_fit"], replace=False)] = True
    return fm.astype(np.int64), (~fm).astype(np.int64)


def _replica(b: int):
    P = G["P"]
    cf, ct = _weights(b)
    res = {}
    for key, K in P["keys"].items():
        par = _fit_params(K, None if cf is None else cf[K["fit_ev"]])
        st = _test_stats(K, par, None if ct is None else ct[K["test_ev"]])
        res[key] = (par, st)
    return b, res


def run_chain(P: dict, n_rep: int, n_proc: int, label: str, scheme: str) -> dict:
    """Nominal (b = 0) and n_rep replicas of the fit-and-test chain: scheme 'bootstrap' (fit and test
    events resampled independently) or 'permutation' (the pooled events re-split at the fit and test
    sizes; P from prepare(..., pooled=True))."""
    assert scheme in ("bootstrap", "permutation") and (scheme == "permutation") == P["pooled"]
    G["P"], G["scheme"] = P, scheme
    t0 = time.time()
    out = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_proc) as pool:
        for i, (b, res) in enumerate(pool.imap_unordered(_replica, range(n_rep + 1), chunksize=2)):
            out[b] = res
            if (i + 1) % 25 == 0 or i == n_rep:
                print(f"  [{label} {scheme}] {i + 1}/{n_rep + 1} replicas, {time.time() - t0:.0f} s", flush=True)
    stack = {}
    for key in P["keys"]:
        stack[key] = {"par": np.stack([out[b][key][0] for b in range(n_rep + 1)]),
                      "st": np.stack([out[b][key][1] for b in range(n_rep + 1)])}
    return stack


# ---------------------------------------------------------------- the rule (amendment 1)
def _v_iid(K, n_t):
    """Exchangeable-cluster variance of the coverage per test bin, angle, level (n_t [nt, angle]):
    the looked-up rows' fit-quantile variance x (1 - 1 / n_t) + E0 (1 - E0) / n_t."""
    e0 = K["E0"][:, None, :]
    n = n_t[:, :, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return K["Vfit"][:, None, :] * (1.0 - 1.0 / n) + e0 * (1.0 - e0) / n


def _coverage_z(K, st, var_perm):
    """Signed coverage z per test bin, angle, level (SA.coverage_z) from chain statistics st
    [nt, angle, 3 + levels]; the null variance is max(var_perm, exchangeable-cluster variance)."""
    n_t = st[:, :, 2]
    cov = st[:, :, 3:]
    m = np.rint(n_t[:, :, None] * (1.0 - cov))
    var = np.fmax(var_perm, _v_iid(K, n_t))
    return SA.coverage_z(m, n_t[:, :, None], K["E0"][:, None, :], var)


def judged_by_i(surface, angle) -> bool:
    """(i) judges the rows whose clipped-mean centre and B.1 core width weighting consumers use:
    barrel alpha and beta, disc alpha. Disc beta has no Gaussian core (plan D.3; sheet v1
    angle.error_table: its weighting consumers use the full response), so it is reported only."""
    return not (angle == "beta" and int(surface) in SA.DISCS)


def i_scope(K) -> np.ndarray:
    """[test bin, angle] mask of the bins (i) judges."""
    return np.array([[judged_by_i(tr["surface"], a) for a in SA.ANGLES] for tr in K["trows"]], bool).reshape(-1, len(SA.ANGLES))


def _band_excess(nom, sd):
    """Per test bin and angle: how many sd the after-table centre / width lie outside the decision
    band, u = max((|centre| - 0.25) / sd_c, (|width - 1| - 0.1) / sd_w); (i) fails a bin when u > W."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.fmax((np.abs(nom[..., 0]) - BAND_C) / sd[..., 0], (np.abs(nom[..., 1] - 1.0) - BAND_W) / sd[..., 1])


def _band_fraction(nom, sd, w):
    """Per test bin and angle: max(|centre| / (0.25 + w sd_c), |width - 1| / (0.1 + w sd_w)) (reported; > 1
    exactly when the excess u > w)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.fmax(np.abs(nom[..., 0]) / (BAND_C + w * sd[..., 0]),
                       np.abs(nom[..., 1] - 1.0) / (BAND_W + w * sd[..., 1]))


def _decide(zk: dict, judged: dict, uk: dict, w_i: float):
    """The rule on one set of statistics: zk[key] [nt, angle, level] coverage z, judged[key] [nt]
    (bins with >= 200 test clusters), uk[key] [nt, angle] band excess u (NaN outside the (i) scope),
    w_i the (i) threshold. Returns (pass_i, pass_ii, N, t, max|z|, n beyond t, chi2 results,
    failing chi2, (i) failures)."""
    from scipy.stats import chi2

    zs, chis = [], {}
    n_i_fail = 0
    for key, z in zk.items():
        jm = judged[key]
        n_i_fail += int(np.sum(uk[key][jm] > w_i))
        zz = z[jm]
        for li, lv in enumerate(LEVELS):
            v = zz[:, :, li].ravel()
            v = v[np.isfinite(v)]
            chis[(key, lv)] = v
            zs.append(v)
    zall = np.concatenate(zs) if zs else np.zeros(0)
    N = len(zall)
    t = t_family(N) if N else float("nan")
    n_beyond = int(np.sum(np.abs(zall) > t))
    n_chi = len(chis)
    chi_res = {f"{k}/{lv:g}": {"chi2": float(np.sum(v ** 2)), "ndf": int(len(v)),
                               "p": float(chi2.sf(np.sum(v ** 2), len(v)))} for (k, lv), v in chis.items()}
    chi_fail = sorted(k for k, v in chi_res.items() if v["p"] <= 0.01 / max(n_chi, 1))
    return (n_i_fail == 0, (n_beyond == 0) and not chi_fail and N > 0, N, t,
            float(np.max(np.abs(zall))) if N else None, n_beyond, chi_res, chi_fail, n_i_fail)


def judge(P: dict, boot: dict, perm: dict, min_n: int = SA.MIN_FIT, false_veto: bool = True) -> dict:
    """(i) adequacy band with bootstrap errors; (ii) coverage against the exact-shape null with the
    permutation variance; over every key and angle. Also: the same rule on every permutation replica
    (false-veto rate), and, reported, the coverage z against the nominal level instead of E0."""
    from scipy.stats import beta as beta_d

    zk, judged, uk, sdi, sdb, zlev, varp, scope, viid = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for key, K in P["keys"].items():
        sb, sp = boot[key]["st"], perm[key]["st"]
        nom = sb[0]
        if not np.allclose(nom, sp[0], equal_nan=True, rtol=0, atol=1e-12):
            raise RuntimeError(f"{key}: the permutation chain's nominal split differs from the bootstrap's")
        sdb[key] = np.nanstd(sb[1:], axis=0, ddof=1)
        # (i) error: the larger of the bootstrap and the re-split (permutation) spread of the statistic
        sdi[key] = np.fmax(sdb[key][..., :2], np.nanstd(sp[1:, :, :, :2], axis=0, ddof=1))
        varp[key] = np.nanvar(sp[1:, :, :, 3:], axis=0, ddof=1)
        viid[key] = _v_iid(K, nom[:, :, 2])
        zk[key] = _coverage_z(K, nom, varp[key])
        judged[key] = K["test_n"] >= min_n
        scope[key] = i_scope(K)
        uk[key] = np.where(scope[key], _band_excess(nom, sdi[key]), np.nan)
        # reported alternative: the nominal level as the reference (no order-statistic offset)
        Kl = {**K, "E0": np.broadcast_to(np.asarray(LEVELS), K["E0"].shape)}
        zlev[key] = _coverage_z(Kl, nom, varp[key])
    n_i = int(sum(np.isfinite(uk[k][judged[k]]).sum() for k in uk)) * 2      # centre and width per bin-angle
    w_i = t_family(n_i)
    details_i, reported_db = [], []
    for key, K in P["keys"].items():
        nom, sd = boot[key]["st"][0], sdi[key]
        raw = _band_fraction(nom, sd, w_i)
        for i, tr in enumerate(K["trows"]):
            if not judged[key][i]:
                continue
            for j, a in enumerate(SA.ANGLES):
                tag = f"{key}/{a}/{SA.SURFACE_NAMES[tr['surface']]}/f{tr['facing']:+d}/[{tr['lo']:+.2f},{tr['hi']:+.2f})"
                item = (float(raw[i, j]), tag, float(nom[i, j, 0]), float(sd[i, j, 0]), float(nom[i, j, 1]), float(sd[i, j, 1]),
                        float(sdb[key][i, j, 0]), float(sdb[key][i, j, 1]))
                (details_i if scope[key][i, j] else reported_db).append(item)
    pass_i, pass_ii, N, t, maxz, nb, chi_res, chi_fail, n_i_fail = _decide(zk, judged, uk, w_i)
    # reported: the v1 recipe of (i) within the same scope (bootstrap sd only, widening 2)
    v1_fail = int(sum(np.sum(np.where(scope[k], _band_excess(boot[k]["st"][0], sdb[k]), np.nan)[judged[k]] > WIDEN_V1)
                      for k in uk))
    zlist = []
    for key, K in P["keys"].items():
        for i, tr in enumerate(K["trows"]):
            if not judged[key][i]:
                continue
            for j, a in enumerate(SA.ANGLES):
                for li, lv in enumerate(LEVELS):
                    z = zk[key][i, j, li]
                    if np.isfinite(z):
                        tag = f"{key}/{a}/{SA.SURFACE_NAMES[tr['surface']]}/f{tr['facing']:+d}/[{tr['lo']:+.2f},{tr['hi']:+.2f})"
                        st0 = boot[key]["st"][0]
                        zlist.append({"bin": tag, "level": lv, "coverage": float(st0[i, j, 3 + li]),
                                      "E0": float(K["E0"][i, li]), "n_test": int(K["test_n"][i]),
                                      "null_sd": float(np.sqrt(np.fmax(varp[key][i, j, li], viid[key][i, j, li]))),
                                      "perm_over_iid_var": float(varp[key][i, j, li] / viid[key][i, j, li]),
                                      "z": float(z), "z_vs_level": float(zlev[key][i, j, li])})
    zarr = np.array([x["z"] for x in zlist])
    worst_i = sorted(details_i, reverse=True)
    zl = np.concatenate([zlev[k][judged[k]].ravel() for k in zlev])
    zl = zl[np.isfinite(zl)]
    res = {
        "rule": RULE,
        "i": {"bins": len(details_i), "N_i": n_i, "W_i": w_i, "fail": n_i_fail,
              "worst_fraction_of_widened_band": worst_i[0][0] if worst_i else None,
              "worst": [{"bin": x[1], "fraction_of_band": x[0], "centre": x[2], "sd_c": x[3], "width": x[4],
                         "sd_w": x[5], "sd_c_boot": x[6], "sd_w_boot": x[7]} for x in worst_i[:8]],
              "failures": [{"bin": x[1], "centre": x[2], "sd_c": x[3], "width": x[4], "sd_w": x[5]}
                           for x in worst_i if x[0] > 1.0][:50],
              "reported_v1_recipe_in_scope_fail": v1_fail,
              "tolerance_pct_10_50_90": {
                  "width": np.percentile([BAND_W + w_i * x[5] for x in details_i], [10, 50, 90]).round(3).tolist(),
                  "centre": np.percentile([BAND_C + w_i * x[3] for x in details_i], [10, 50, 90]).round(3).tolist(),
                  "width_v1": np.percentile([BAND_W + WIDEN_V1 * x[7] for x in details_i], [10, 50, 90]).round(3).tolist(),
                  "centre_v1": np.percentile([BAND_C + WIDEN_V1 * x[6] for x in details_i], [10, 50, 90]).round(3).tolist(),
                  "definition": "half-width of the band a bin's after-table width / centre must leave to fail (i), over "
                                "the judged bin-angles; _v1: the v1 recipe (2 x bootstrap sd)"},
              "reported_disc_beta": {"bins": len(reported_db), "outside_band": int(sum(x[0] > 1.0 for x in reported_db)),
                                     "worst": [{"bin": x[1], "fraction_of_band": x[0], "centre": x[2], "sd_c": x[3],
                                                "width": x[4], "sd_w": x[5]}
                                               for x in sorted(reported_db, reverse=True)[:8]]},
              "n_a": int(sum((~judged[k]).sum() * len(SA.ANGLES) for k in judged))},
        "ii": {"N": N, "t_N": t, "max_abs_z": maxz, "n_beyond_t": nb, "n_beyond_3p5": int(np.sum(np.abs(zarr) > 3.5)),
               "beyond": sorted([x for x in zlist if abs(x["z"]) > t], key=lambda x: -abs(x["z"]))[:30],
               "largest": sorted(zlist, key=lambda x: -abs(x["z"]))[:12],
               "chi2": chi_res, "chi2_threshold_p": 0.01 / max(len(chi_res), 1), "chi2_fail": chi_fail,
               "per_level_max_abs_z": {f"{lv:g}": float(np.max(np.abs([x["z"] for x in zlist if x["level"] == lv])))
                                       for lv in LEVELS if any(x["level"] == lv for x in zlist)},
               "z_mean": float(zarr.mean()) if N else None, "z_sd": float(zarr.std()) if N else None,
               "null_sd_over_iid_median": None,
               "reported_vs_level": {"max_abs_z": float(np.max(np.abs(zl))) if len(zl) else None,
                                     "n_beyond_t": int(np.sum(np.abs(zl) > t)), "mean_z": float(zl.mean()) if len(zl) else None}},
        "lookup_missing": {k: P["keys"][k]["n_lookup_missing"] for k in P["keys"]},
    }
    ratio = []
    for key in P["keys"]:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio.append((varp[key] / viid[key])[judged[key]])
    rr = np.concatenate([x.reshape(-1, len(LEVELS)) for x in ratio])
    res["ii"]["null_sd_over_iid_median"] = {f"{lv:g}": float(np.nanmedian(np.sqrt(rr[:, li]))) for li, lv in enumerate(LEVELS)}
    res["ii"]["perm_var_over_iid_pct"] = {f"{lv:g}": np.nanpercentile(rr[:, li], [5, 50, 95, 99]).round(3).tolist()
                                          for li, lv in enumerate(LEVELS)}
    res["pass_i"], res["pass_ii"] = pass_i, pass_ii
    res["pass"] = pass_i and pass_ii
    res["per_region"] = region_summary(P, zk, {k: np.where(scope[k], _band_fraction(boot[k]["st"][0], sdi[k], w_i), np.nan)
                                               for k in uk}, judged)
    res["_z"] = {k: v.tolist() for k, v in zk.items()}
    if false_veto:
        res["false_veto"] = false_veto_rate(P, perm, judged, sdb, sdi, w_i, scope)
        lo, hi = beta_d.ppf(0.025, res["false_veto"]["fail_any"], res["false_veto"]["n_replicas"] -
                            res["false_veto"]["fail_any"] + 1), \
            beta_d.ppf(0.975, res["false_veto"]["fail_any"] + 1, res["false_veto"]["n_replicas"] - res["false_veto"]["fail_any"])
        res["false_veto"]["rate_95cl"] = [float(np.nan_to_num(lo)), float(hi)]
    return res


def false_veto_rate(P, perm, judged, sdb, sdi, w_i, scope) -> dict:
    """The rule applied to every permutation replica as if it were the observed split (each is a
    known-good split of the pooled events): (ii) with the leave-one-out permutation variance and the
    nominal E0, (i) with the observed split's errors and threshold. Returns counts, the max|z| and
    max-excess spreads, and for comparison the (i) false-veto rate under the v1 recipe (bootstrap sd,
    widening 2) and the two half-way variants."""
    keys = list(P["keys"])
    R = perm[keys[0]]["st"].shape[0] - 1
    fail_i = fail_ii = fail_any = 0
    maxz, minp, max_u = [], [], []
    alt = {"boot_sd_W2": [], "max_sd_W2": [], "boot_sd_Wi": []}
    tally = {}
    zrep = {}
    for key in keys:
        K = P["keys"][key]
        st = perm[key]["st"][1:]
        cov = st[:, :, :, 3:]
        s1 = np.nansum(cov, axis=0)
        s2 = np.nansum(cov ** 2, axis=0)
        loo_mean = (s1[None] - cov) / (R - 1)
        loo_var = (s2[None] - cov ** 2 - (R - 1) * loo_mean ** 2) / (R - 2)
        zrep[key] = np.stack([_coverage_z(K, st[b], loo_var[b]) for b in range(R)])
    for b in range(R):
        zb = {k: zrep[k][b] for k in keys}
        ub = {k: np.where(scope[k], _band_excess(perm[k]["st"][1 + b], sdi[k]), np.nan) for k in keys}
        pi, pii, N, t, mz, nb, chi_res, chi_fail, _ = _decide(zb, judged, ub, w_i)
        u_b, ubb = [], []
        for k in keys:
            u = np.where(judged[k][:, None], ub[k], np.nan)
            u_b.append(np.nanmax(u))
            ubb.append(np.nanmax(np.where(scope[k] & judged[k][:, None], _band_excess(perm[k]["st"][1 + b], sdb[k]), np.nan)))
            for i, j in zip(*np.nonzero(u > w_i)):
                tr = P["keys"][k]["trows"][i]
                tag = f"{k}/{SA.ANGLES[j]}/{SA.SURFACE_NAMES[tr['surface']]}/f{tr['facing']:+d}/[{tr['lo']:+.2f},{tr['hi']:+.2f})"
                tally[tag] = tally.get(tag, 0) + 1
        max_u.append(max(u_b))
        alt["boot_sd_W2"].append(max(ubb) > WIDEN_V1)
        alt["max_sd_W2"].append(max(u_b) > WIDEN_V1)
        alt["boot_sd_Wi"].append(max(ubb) > w_i)
        fail_i += not pi
        fail_ii += not pii
        fail_any += not (pi and pii)
        maxz.append(mz)
        minp.append(min(v["p"] for v in chi_res.values()))
    return {"n_replicas": R, "fail_i": fail_i, "fail_ii": fail_ii, "fail_any": fail_any,
            "rate": fail_any / R, "max_abs_z_pct_5_50_95_99": np.percentile(maxz, [5, 50, 95, 99]).round(2).tolist(),
            "min_chi2_p_pct_1_5_50": np.percentile(minp, [1, 5, 50]).tolist(),
            "i_max_excess_sigma_pct_5_50_95_99": np.percentile(max_u, [5, 50, 95, 99]).round(2).tolist(),
            "i_false_veto_alternatives": {k: int(np.sum(v)) for k, v in alt.items()},
            "i_fail_tally": dict(sorted(tally.items(), key=lambda x: -x[1])[:20]),
            "definition": "each permutation replica judged by the amendment-1 rule as if observed: (ii) leave-one-out "
                          "permutation variance, nominal E0, same N and t(N) rule, chi2 per level x key; (i) the replica's "
                          "after-table centre / width outside the band by more than W_i x max(bootstrap, re-split) sd "
                          "of the observed split"}


def region_summary(P, zk, frac, judged):
    """Worst |z| and worst band fraction per (key, barrel / disc), as the demonstration must show both."""
    out = {}
    for key, K in P["keys"].items():
        for region, surfs in (("barrel", SA.BARREL), ("discs", SA.DISCS)):
            idx = [i for i, tr in enumerate(K["trows"]) if tr["surface"] in surfs and judged[key][i]]
            if not idx:
                continue
            out[f"{key}/{region}"] = {"bins": len(idx), "max_abs_z": float(np.nanmax(np.abs(zk[key][idx]))),
                                      "worst_band_fraction": float(np.nanmax(frac[key][idx]))}
    return out


# ---------------------------------------------------------------- pT check (fit sample)
def _pt_replica(b: int):
    PT = G["PT"]
    cf = None
    if b:
        rng = np.random.default_rng([SEED, 7, b])
        cf = rng.multinomial(PT["n_ev"], np.full(PT["n_ev"], 1.0 / PT["n_ev"]))
    out = {}
    for key, K in PT["keys"].items():
        res = np.full((len(K["rows"]), len(SA.ANGLES), 2), np.nan)
        for i, sl in enumerate(K["slices"]):
            if sl is None:
                continue
            for j, a in enumerate(SA.ANGLES):
                e = [SA.bin_estimates(K["p"][a][x], LEVELS, None if cf is None else cf[K["ev"][x]]) for x in sl]
                if e[0] is not None and e[1] is not None:
                    res[i, j] = (e[0]["centre"] - e[1]["centre"], e[0]["width"] - e[1]["width"])
        out[key] = res
    return b, out


def pt_check(fit: dict, n_rep: int, n_proc: int, min_n: int = SA.MIN_FIT):
    """Per canonical row, angle and key: centre and width at pT 1-2 vs >= 2 GeV with
    the paired event bootstrap of the fit sample; the D.0 agreement rule (no |z| beyond t(N) over
    every z-score of both keys; chi2 per key reported). Rows failing in both signs (the row and its
    mirror) are refitted at pT >= 2 GeV and flagged; a failure in one sign only is reported."""
    from scipy.stats import chi2

    uf, evf = event_index(fit["ev"])
    hi = fit["tp_pt"] >= 2.0
    PT = {"n_ev": len(uf), "keys": {}}
    for key in SA.KEYS:
        rows, rowmap = build_rows("canonical", fit, key)
        r = row_of("canonical", rowmap, fit, key)
        o = np.lexsort((hi, r))
        rs, hs = r[o], hi[o]
        slices = []
        for i in range(len(rows)):
            a0, a1 = np.searchsorted(rs, i, "left"), np.searchsorted(rs, i, "right")
            k = a0 + np.searchsorted(hs[a0:a1], True, "left")
            slices.append(None if (k - a0 < min_n or a1 - k < min_n) else (slice(a0, k), slice(k, a1)))
        PT["keys"][key] = {"rows": rows, "rowmap": rowmap, "slices": slices, "ev": evf[o],
                           "p": {a: fit["pull"][a][o] for a in SA.ANGLES}}
    G["PT"] = PT
    with mp.get_context("fork").Pool(n_proc) as pool:
        got = dict(pool.imap_unordered(_pt_replica, range(n_rep + 1), chunksize=4))
    out, flags, zall = {}, {}, []
    for key, K in PT["keys"].items():
        st = np.stack([got[b][key] for b in range(n_rep + 1)])
        sd = np.nanstd(st[1:], axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = st[0] / sd
        tests = [{"row": i, "angle": a, "what": nm, "diff": float(st[0, i, j, q]), "sd": float(sd[i, j, q]),
                  "z": float(z[i, j, q])}
                 for i in range(len(K["rows"])) for j, a in enumerate(SA.ANGLES)
                 for q, nm in enumerate(("centre", "width")) if np.isfinite(z[i, j, q])]
        out[key] = {"tests": tests}
        zall += [t_["z"] for t_ in tests]
    N = len(zall)
    t = t_family(N)
    for key, K in PT["keys"].items():
        rows, rowmap = K["rows"], K["rowmap"]
        failing = {x["row"] for x in out[key]["tests"] if abs(x["z"]) > t}
        both = {i for i in failing if mirror_row(rows, rowmap, i) in failing}
        flags[key] = both
        zk = np.array([x["z"] for x in out[key]["tests"]])
        out[key]["summary"] = {"N_all_keys": N, "t_N": t, "rows_tested": len({x["row"] for x in out[key]["tests"]}),
                               "rows": len(rows), "failing_rows": sorted(failing),
                               "failing_both_signs_refit_pt2": sorted(both),
                               "failing_one_sign_reported": sorted(failing - both),
                               "chi2": float(np.sum(zk ** 2)), "ndf": int(len(zk)),
                               "p": float(chi2.sf(np.sum(zk ** 2), len(zk))) if len(zk) else None,
                               "max_abs_z": float(np.max(np.abs(zk))) if len(zk) else None}
        out[key]["rows"] = [{k: v for k, v in r_.items() if k in ("surface", "facing", "lo", "hi")} for r_ in rows]
    return {"flags": {k: sorted(v) for k, v in flags.items()}, "detail": out}, flags


# ---------------------------------------------------------------- reported quantities
def census_scales() -> dict:
    """Today's census per-layer scales, from the registry (appa.census.angle-scales)."""
    e = json.load(open(REGISTRY_SCALES))
    v = {x["name"].split("_")[0].lower(): [float(t) for t in x["value"].split("/")] for x in e["values"]}
    return {"alpha": dict(zip(SA.BARREL, v["alpha"])), "beta": dict(zip(SA.BARREL, v["beta"]))}


def before_after(P: dict, boot: dict, judgement: dict, test: dict, scales: dict) -> dict:
    """Per test bin (nominal): pulls before (published sigma; and today's census sigma on the
    barrel: per-layer scale, no centre) and after the table; the coverage with its
    E0 and z; today's 3-census-sigma class-T efficiency; the table's half-width at each level in
    today's census sigma."""
    out = {}
    for key, K in P["keys"].items():
        trow = row_of("canonical", K["trowmap"], test, key)
        nom = boot[key]["st"][0]
        sd = np.nanstd(boot[key]["st"][1:], axis=0, ddof=1)
        par = boot[key]["par"][0]
        psd = np.nanstd(boot[key]["par"][1:], axis=0, ddof=1)
        zk = np.asarray(judgement["_z"][key])
        rows = []
        for i, tr in enumerate(K["trows"]):
            m = trow == i
            ent = {"surface": SA.SURFACE_NAMES[tr["surface"]], "facing": tr["facing"], "lo": tr["lo"], "hi": tr["hi"],
                   "n_fit": tr["n_fit"], "n_test": int(m.sum())}
            for j, a in enumerate(SA.ANGLES):
                p = test["pull"][a][m]
                b = SA.core_w(p) if m.sum() >= SA.MIN_FIT else None
                d = {"judged_by_i": judged_by_i(tr["surface"], a),
                     "before": None if b is None else {"centre": b["mean"], "width": b["rms0"], "out3": b["out3"]},
                     "after": {"centre": nom[i, j, 0], "sd_centre": sd[i, j, 0], "width": nom[i, j, 1],
                               "sd_width": sd[i, j, 1]},
                     "coverage": {f"{lv:g}": {"cov": nom[i, j, 3 + li], "sd_bootstrap": sd[i, j, 3 + li],
                                              "E0": K["E0"][i, li], "z": zk[i, j, li]}
                                  for li, lv in enumerate(LEVELS)},
                     "table": {"centre": par[i, j, 0], "sd_centre": psd[i, j, 0], "width": par[i, j, 1],
                               "sd_width": psd[i, j, 1],
                               "q": {f"{lv:g}": par[i, j, 2 + li] for li, lv in enumerate(LEVELS)},
                               "sd_q": {f"{lv:g}": psd[i, j, 2 + li] for li, lv in enumerate(LEVELS)}}}
                if tr["surface"] in SA.BARREL and m.sum():
                    sc = scales[a][tr["surface"]]
                    pc = p / sc
                    bc = SA.core_w(pc) if m.sum() >= SA.MIN_FIT else None
                    d["before_census"] = None if bc is None else {"centre": bc["mean"], "width": bc["rms0"]}
                    d["census_3sigma_eff"] = float(np.mean(np.abs(pc) <= 3.0))
                    d["half_width_census_sigma"] = {f"{lv:g}": par[i, j, 2 + li] / sc for li, lv in enumerate(LEVELS)}
                ent[a] = d
            rows.append(ent)
        out[key] = rows
    return out


def table_doc(P: dict, stack: dict, orient, flags, prov: dict) -> dict:
    keys_rows = {}
    for key, K in P["keys"].items():
        par = stack[key]["par"][0]
        psd = np.nanstd(stack[key]["par"][1:], axis=0, ddof=1)
        rows = []
        for i, r in enumerate(K["rows"]):
            ent = {k: r[k] for k in ("surface", "facing", "lo", "hi", "fine", "n_fit")}
            ent["flags"] = ["refit_pt2"] if i in flags.get(key, set()) else []
            for j, a in enumerate(SA.ANGLES):
                full = SA.bin_estimates(K["fit_p"][a][K["fs"][i]:K["fe"][i]], LEVELS)
                if full is None:
                    ent[a] = None
                    continue
                ent[a] = {"centre": par[i, j, 0], "width": par[i, j, 1],
                          "q": {f"{lv:g}": par[i, j, 2 + li] for li, lv in enumerate(LEVELS)},
                          "sd_centre": psd[i, j, 0], "sd_width": psd[i, j, 1],
                          "sd_q": {f"{lv:g}": psd[i, j, 2 + li] for li, lv in enumerate(LEVELS)},
                          "mad": full["mad"], "out3": full["out3"], "full_rms0": full["full_rms0"], "n": full["n"]}
                if not judged_by_i(r["surface"], a):
                    ent[a]["note"] = ("disc beta: no Gaussian core; the core width is not a weighting width (weighting "
                                      "consumers use the full response, decision 3); cutting consumers use the quantiles")
            rows.append(ent)
        keys_rows[key] = rows
    doc = SA.new_table_doc(keys_rows, orient, prov, era="canon_v3 (150X D121 PU200 TTbar, AAAA)",
                           fit_sample=f"canon_v3 AAAA files {list(SA.FIT_FILES)}, class T, pT >= {SA.PT_MIN:g} GeV "
                                      "(the class-C angles of these files are cluster-CNN training data; class T is "
                                      "truth + 1.5 x a PixelAV residual and is not)")
    doc["recipe"] = "angle.error_table (sheet v1); tested under " + RULE
    return doc


# ---------------------------------------------------------------- driver
def run(mode: str, n_rep: int, n_proc: int, out_dir: Path) -> dict:
    fit_files, test_files = SPLITS[mode]
    t0 = time.time()
    orient = CM.orientation_map(fit_files)
    fit = CM.derive(load_split(fit_files), orient)
    test = CM.derive(load_split(test_files), orient)
    print(f"[{mode}] fit files {fit_files}: {len(fit['ev'])} clusters; test files {test_files}: "
          f"{len(test['ev'])} clusters; orientation of {len(orient[0])} modules ({time.time() - t0:.0f} s)", flush=True)
    res = {"mode": mode, "rule": RULE, "fit_files": list(fit_files), "test_files": list(test_files), "n_rep": n_rep,
           "seed": SEED, "n_fit": int(len(fit["ev"])), "n_test": int(len(test["ev"])), "code": CM.git_state(),
           "inputs": CM.inputs_provenance(sorted(set(fit_files) | set(test_files))),
           "rule_parameters": {"band_width": BAND_W, "band_centre": BAND_C, "i_widening": "W_i = t(N_i) (family-wise 1 %, N_i = 2 x judged bin-angles) x max(bootstrap, re-split sd)", "levels": LEVELS,
                               "min_n": SA.MIN_FIT, "i_scope": "barrel alpha and beta, disc alpha (disc beta reported)",
                               "N_definition": "levels x test bins x angles x keys, after merges",
                               "coverage_null": "beta-binomial, mean 1 - E0 (E0 = k / (n_fit + 1)), variance max(event "
                                                "permutation, exchangeable clusters); mid-p; z = -Phi^-1(P_mid(M <= m))",
                               "chi2": "per level and key over bins and angles; p > 0.01 / (levels x keys)"}}
    if mode in ("diagnosis", "known-good", "gated", "fit-only"):
        ptc, flags = pt_check(fit, n_rep, n_proc)
        res["pt_check"] = ptc
        print(f"[{mode}] pT check: refit at pT >= 2 {ptc['flags']}", flush=True)
        spec = {k: {"kind": "canonical", "lookup_key": k} for k in SA.KEYS}
        P = prepare(fit, test, spec, flags)
        boot = run_chain(P, n_rep, n_proc, mode, "bootstrap")
        if mode != "fit-only":
            perm = run_chain(prepare(fit, test, spec, flags, pooled=True), n_rep, n_proc, mode, "permutation")
            res["judgement"] = judge(P, boot, perm)
            res["bins"] = before_after(P, boot, res["judgement"], test, census_scales())
        if mode in ("gated", "fit-only"):
            prov = {"code": res["code"], "inputs": res["inputs"], "fit_files": list(fit_files), "rule": RULE,
                    "job_id": os.environ.get("SP_JOB_ID"), "n_rep_bootstrap": n_rep,
                    "pt_check_refit_rows": ptc["flags"],
                    "test": f"files 3-10; see {result_name('gated')}" if mode == "gated" else
                    "NOT TESTED: the gated test did not run (amendment 1 not approved, its known-good demonstration "
                    f"failed or a known-bad passed; see {result_name('known-good')}, {result_name('known-bad')})"}
            doc = table_doc(P, boot, orient, flags, prov)
            if mode == "gated":
                doc["test_result"] = {"pass": res["judgement"]["pass"], "pass_i": res["judgement"]["pass_i"],
                                      "pass_ii": res["judgement"]["pass_ii"]}
            else:
                doc["test_result"] = {"pass": None, "status": "untested"}
            tab = SA.AngleTable(doc)
            fn = TABLE_FILE if mode == "gated" else UNTESTED_FILE
            tab.save(str(out_dir / fn))
            res["table_file"] = str(out_dir / fn)
    else:
        res["known_bads"] = {}
        for name, desc in KNOWN_BADS.items():
            tst = test
            if name == "wrong_key":
                spec = {"true": {"kind": "canonical", "lookup_key": "emitted"},
                        "emitted": {"kind": "canonical", "lookup_key": "true"}}
            elif name in TEST_SCALE:
                spec = {k: {"kind": "canonical", "lookup_key": k} for k in SA.KEYS}
                tst = {**test, "pull": {a: v * TEST_SCALE[name] for a, v in test["pull"].items()}}
            else:
                spec = {k: {"kind": name, "lookup_key": k} for k in SA.KEYS}
            P = prepare(fit, tst, spec)
            boot = run_chain(P, n_rep, n_proc, f"known-bad {name}", "bootstrap")
            perm = run_chain(prepare(fit, tst, spec, pooled=True), n_rep, n_proc, f"known-bad {name}", "permutation")
            j = judge(P, boot, perm, false_veto=False)
            j.pop("_z", None)
            res["known_bads"][name] = {"description": desc, "judgement": j, "fails_as_required": not j["pass"]}
            print(f"[known-bad {name}] pass={j['pass']} (i fail {j['i']['fail']}/{j['i']['bins']}, ii beyond t "
                  f"{j['ii']['n_beyond_t']}, max|z| {j['ii']['max_abs_z']:.1f}, chi2 fail {j['ii']['chi2_fail']})",
                  flush=True)
        res["all_known_bads_fail"] = all(v["fails_as_required"] for v in res["known_bads"].values())
    if "judgement" in res:
        res["judgement"].pop("_z", None)
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def gated_refusal(out_dir: Path) -> str | None:
    """Why the gated test may not run yet (None when it may)."""
    try:
        kg = json.load(open(out_dir / result_name("known-good")))
        kb = json.load(open(out_dir / result_name("known-bad")))
    except FileNotFoundError as e:
        return f"the gated test needs the amendment-1 known-good and known-bad results first ({e})"
    if kg.get("rule") != RULE or kb.get("rule") != RULE:
        return "the known-good / known-bad results are not from the amendment-1 rule"
    if not kg["judgement"]["pass"] or not kb["all_known_bads_fail"] or kg["n_rep"] < 250 or kb["n_rep"] < 250:
        return ("the amendment-1 known-good demonstration did not pass, or a known-bad did not fail, or < 250 replicas: "
                "the rule goes back for redesign (D.0 step 4)")
    if not (out_dir / APPROVAL_FILE).exists():
        return (f"amendment 1 is not approved: {out_dir / APPROVAL_FILE} does not exist (written when the physicist "
                "approves the rule change)")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=tuple(SPLITS))
    ap.add_argument("--rep", type=int, default=250)
    ap.add_argument("--proc", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(CM.OUT_DIR))
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.mode == "gated":
        why = gated_refusal(out_dir)
        if why:
            print(f"refused: {why}")
            return 66
    res = run(a.mode, a.rep, a.proc, out_dir)
    CM.jdump(res, out_dir / result_name(a.mode))
    if a.mode == "known-bad":
        print("every known-bad fails" if res["all_known_bads_fail"] else "SOME KNOWN-BAD PASSES")
        return 0 if res["all_known_bads_fail"] else 3
    if a.mode == "fit-only":
        print(f"wrote the UNTESTED table {res['table_file']}")
        return 0
    j = res["judgement"]
    fv = j.get("false_veto", {})
    print(f"[{a.mode}] PASS={j['pass']}: (i) {j['i']['fail']} of {j['i']['bins']} bins outside the widened band "
          f"(worst {j['i']['worst_fraction_of_widened_band']}); (ii) N={j['ii']['N']} t(N)={j['ii']['t_N']:.2f} "
          f"max|z|={j['ii']['max_abs_z']:.2f}, beyond t: {j['ii']['n_beyond_t']}, chi2 fail: {j['ii']['chi2_fail']}; "
          f"false-veto {fv.get('fail_any')}/{fv.get('n_replicas')} permutation replicas")
    return 0 if j["pass"] else 4


if __name__ == "__main__":
    sys.exit(main())
