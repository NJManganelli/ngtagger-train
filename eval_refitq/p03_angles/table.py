#!/usr/bin/env python
"""P0.3 step 2: the angle-uncertainty table (plan D.3 D10(b), D.0; decisions 2, 3, 11), under the
rule of pre-registration amendment 1e (preregistration.txt: amendments 1-1d, and the physicist's
approval of 2026-10-14 of statistics review 02's round-4 recipe).

Recipe (angle.error_table, sheet v2.7): class-T clusters with a usable direction and TP pT >= 1 GeV; per
surface (IL1-IL4, ID1-ID4) x signed incidence bin (signed cot theta barrel; signed 1/cot theta discs, per
module facing), merged to >= 200 fit clusters away from the plateau edges; per bin and angle (alpha, beta)
the centre (the 25 %-trimmed mean of the pulls), the weighting width (the mean of the smallest 95 % of
|pull - centre| / 0.7168) and the quantiles of |pull - centre| at 95 / 97.5 / 99 / 99.7 % with the rank
k = min(ceil(L (n + 1)), n), in both keys (emitted, true). The B.1 clipped mean and core width, the MAD, the
fraction beyond 3 widths and the full RMS are reported diagnostics. pT check on the fit sample (1-2 vs
>= 2 GeV): a bin failing in both signs is refitted at pT >= 2 GeV and flagged.

Errors: the event-level BOOTSTRAP of the whole fit-and-test chain and its RE-SPLIT (permutation: the
pooled fit + test events re-split at the fit and test sizes); every fitted step re-fitted per replica;
250 each. A replica row without an estimate (pseudo-fit < 200 clusters) is MISSING, not uncovered.

Test, on the test sample, per bin of the same binning (every bin with >= 200 test clusters):
 (i)   adequacy: the after-table centre and width (the table's estimators on (pull - centre) / width)
       within 0 +- 0.25 and 1 +- 0.1, widened by W_i x max(bootstrap, re-split sd), W_i the one-sided
       Sidak threshold t1(N_i), N_i = 2 x the judged bin-angles;
 (ii)  coverage against the exact order-statistic null (beta-binomial, mean 1 - E0, E0 = k / (n_fit + 1),
       variance max(re-split, exchangeable)); mid-p z; no |z| > t(N); chi2 per level x key p > 0.01 / 8;
 (iii) pooled coverage per key x region x angle x level against the NOMINAL level: the one-sided 95 %
       upper bound of the shortfall (exact pooled exchangeable sd) within 0.1 (1 - L) (0.045 % at 99.7 %),
       at the fixed levels 95 / 97.5 / 99 / 99.7 %; its pass probability on the re-splits is reported.
 False-veto: the (i) + (ii) rate and the per-level (iii) pass probability over the re-splits.
 Materiality (amendment 1e, for the gated run): MATERIAL = a pooled (iii) cell failing at 95-99 %, or bins
 failing (i) or (ii) that together hold > 2 % of a surface's test clusters (per key), or a (i) + (ii)
 false-veto rate above 5.7 % on the run's own re-splits; anything else is MINOR: the table ships with the
 deviations listed in its header and the registry.
 Reported, no effect on the rule: the re-split and bootstrap sd of every row's centre and width.

    python table.py diagnosis  [--rep 250] [--proc 10]   fit 3-4 / test 5-10
    python table.py known-good [--rep 250] [--proc 10]   demonstration: fit 7-8, test 3-6, 9-10
    python table.py known-bad  [--rep 250] [--proc 10]   known-bads (a)-(e), KB-q95, KB-c, KB-w on the same split
    python table.py gated      ...   fit 1-2, test 3-10; refuses unless the amendment-1e known-good passed, every
                                     known-bad failed and $P03_OUT/amendment1e_approval.txt exists
    python table.py fit-only   ...   the recipe on files 1-2 without the test: UNTESTED table
Exit: 0 pass; 5 MINOR deviations only (the table ships, deviations listed); 4 MATERIAL; 3 a known-bad passed.
Outputs: $P03_OUT/p03_table_<mode>_a1e.json; the demonstration's fitted table in $P03_OUT/amend1e/.
Earlier results (v1, amendments 1-1d) stay as they are.
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

RULE = "P0.3 pre-registration amendment 1e (2026-10-14)"
SEED = 20261014          # amendment 1b; replica streams are keyed by (seed, scheme, fit files, replica)
LEVELS = SA.LEVELS
I95 = LEVELS.index(0.95)
BAND_W, BAND_C = 0.1, 0.25
REGISTRY_SCALES = Path.home() / "smartpixels" / "sp_registry" / "results" / "appa.census.angle-scales.json"
DEMO_SPLIT = ((7, 8), (3, 4, 5, 6, 9, 10))   # amendment 1b (amendment 1: fit 9-10, test 3-8)
SPLITS = {"diagnosis": ((3, 4), (5, 6, 7, 8, 9, 10)), "known-good": DEMO_SPLIT, "known-bad": DEMO_SPLIT,
          "gated": (SA.FIT_FILES, SA.TEST_FILES), "fit-only": (SA.FIT_FILES, SA.FIT_FILES)}
TABLE_FILE = "angle_table_v1a1e_canon_v3.json"
UNTESTED_FILE = "angle_table_v1a1e_canon_v3_UNTESTED.json"
APPROVAL_FILE = "amendment1e_approval.txt"
AMEND_DIR = "amend1e"
KNOWN_BADS = {
    "folded": "the folded |key| table (signs pooled; the round-4 table's binning), read by |key|",
    "cotbinned_disc": "disc rows binned like the barrel in signed cot theta (barrel edges; 75-98% of disc clusters in "
                      "|cot| > 2.5), per facing",
    "pooled_disc": "disc rows with the two module facings pooled",
    "wrong_key": "the table read in the wrong key (the true-key table looked up with the emitted key, and vice versa)",
    "test_scale_1p05": "input perturbation (amendment 1): every test-sample pull x 1.05, a 5 % resolution difference "
                       "between fit and test events (a production-batch difference of that size would bias every "
                       "cut's efficiency): the rule must detect it",
}
TEST_SCALE = {"test_scale_1p05": 1.05}
PERTURBED = {   # amendments 1d / 1e: defects of the fitted table, applied in every replica to the plateau-edge rows
    "kb_q95_plateau_x0p9": {
        "pert": {"q95_scale": 0.9},
        "must_fail": "ii_iii",
        "description": "cut-quantile error at decision scale (amendment 1d, KB-q95): q95 multiplied by 0.9 in every row "
                       "whose |key| range overlaps the plateau edges [0.50, 0.67), both angles, both keys, all surfaces "
                       "(expected 95 % coverage ~92 %); centres, widths and the other quantiles untouched: must fail "
                       "through (ii) / (iii)"},
    "kb_centre_plateau_plus0p5": {
        "pert": {"centre_shift": 0.5},
        "must_fail": "i",
        "description": "centre error at decision scale (amendment 1d, KB-c): the centre moved by + 0.5 weighting "
                       "widths in the same rows (twice the (i) band); widths and quantiles untouched: must fail "
                       "through (i)"},
    "kb_width_plateau_div1p15": {
        "pert": {"width_scale": 1.0 / 1.15},
        "must_fail": "i",
        "description": "weighting-width error at decision scale (amendment 1e, KB-w; statistics review 02 R4-2): the "
                       "width divided by 1.15 in the same rows (after-table width ~1.15; Kalman weight 32 % too large); "
                       "centres and quantiles untouched, so (ii) and (iii) cannot see it: must fail through (i), per-row "
                       "power reported"},
}
MATERIAL_LEVELS = (0.95, 0.975, 0.99)     # amendment 1e: a failing (iii) cell at these levels is MATERIAL
MATERIAL_SHARE = 0.02                     # ... and (i) / (ii) failing bins holding > 2 % of a surface's test clusters
FALSE_VETO_MAX = 0.057                    # ... and a (i) + (ii) false-veto rate above 5.7 % on the run's re-splits


def result_name(mode: str) -> str:
    return f"p03_table_{mode}_a1e.json"


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
    """spec = {test_key: {"kind", "lookup_key"[, "pert"]}}; the table's binning `kind` is built in test_key
    on the fit sample and read with lookup_key; the test bins are the canonical rows of test_key on
    the same fit sample; "pert" (amendment 1d / 1e known-bads) perturbs the fitted plateau-edge rows in every
    replica (_fit_params). flags = {test_key: set of row ids refitted at pT >= 2} (canonical tables).
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
        e_row = np.where(usable_row[:, None], SA.expected_coverage(nf_row[:, None], lv[None, :]), np.nan)
        v_row = np.where(usable_row[:, None], SA.fit_coverage_variance(nf_row[:, None], lv[None, :]), np.nan)
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
            keep = np.isfinite(e_row[rr, 0])                # rows without an estimate: their clusters are missing
            if not keep.any():
                continue
            rr, cnt = rr[keep], cnt[keep]
            sh = cnt / cnt.sum()
            E0[i] = (sh[:, None] * e_row[rr]).sum(0)
            Vfit[i] = (sh[:, None] ** 2 * v_row[rr]).sum(0)
        pert = sp.get("pert")
        P["keys"][key] = {
            "rows": rows, "trows": trows, "trowmap": trowmap, "flags": sorted(int(x) for x in fl), "kind": sp["kind"],
            "lookup_key": sp["lookup_key"],
            "fit_ev": evf[fo], "fit_p": {a: fside["pull"][a][fo] for a in SA.ANGLES}, "fs": fs, "fe": fe,
            "test_ev": evt[to], "test_p": {a: tside["pull"][a][to] for a in SA.ANGLES},
            "test_lrow": lrow[to], "ts": ts, "te": te, "n_lookup_missing": int(np.sum((lr_a < 0) & (tb_a >= 0))),
            "test_n": n_test.astype(int), "nf_row": nf_row, "E0": E0, "Vfit": Vfit, "pert": pert,
            "pert_rows": np.r_[plateau_edge_rows(rows), False] if pert else None}
    return P


def _fit_params(K, wf):
    """[row (+ the no-row slot), angle, centre / width / q at each level] of one replica's fit; a known-bad
    "pert" multiplies q95 (q95_scale) or the width (width_scale), or moves the centre by a multiple of the
    width (centre_shift), in the plateau-edge rows."""
    nr = len(K["rows"])
    par = np.full((nr + 1, len(SA.ANGLES), 2 + len(LEVELS)), np.nan)      # centre, width, q...
    for i in range(nr):
        a0, a1 = K["fs"][i], K["fe"][i]
        if a1 <= a0:
            continue
        w = None if wf is None else wf[a0:a1]
        for j, a in enumerate(SA.ANGLES):
            e = SA.bin_estimates(K["fit_p"][a][a0:a1], LEVELS, w, diagnostics=False)
            if e is None:
                continue
            par[i, j, 0], par[i, j, 1] = e["centre"], e["width"]
            par[i, j, 2:] = [e["q"][f"{lv:g}"] for lv in LEVELS]
    pert, m = K["pert"], K["pert_rows"]
    if pert:
        if "q95_scale" in pert:
            par[m, :, 2 + I95] *= pert["q95_scale"]
        if "width_scale" in pert:
            par[m, :, 1] *= pert["width_scale"]
        if "centre_shift" in pert:
            par[m, :, 0] += pert["centre_shift"] * par[m, :, 1]
    return par


def _test_stats(K, par, wt):
    """Per test bin and angle: the after-table centre and width (the table's estimators on the after-table
    pulls a = (pull - centre) / width over the bin's clusters that have an estimate, >= 200 of them: centre
    = SA.centre_of(a), width = SA.weighting_width(|a - centre|)), the test count and the coverage at each
    level. A test cluster looked up in a table row that has no estimate in this replica (its pseudo-fit fell
    below 200 clusters) is MISSING: it is left out of the count and the coverage (amendment 1c, N1). A
    cluster for which the table has no row at all (a binning defect of a known-bad table) counts as outside."""
    nt = len(K["trows"])
    out = np.full((nt, len(SA.ANGLES), 3 + len(LEVELS)), np.nan)        # centre_after, width_after, n, cov...
    has_row = K["test_lrow"] >= 0
    lr = np.where(has_row, K["test_lrow"], len(K["rows"]))
    for j, a in enumerate(SA.ANGLES):
        p = K["test_p"][a]
        c, w = par[lr, j, 0], par[lr, j, 1]
        valid = ~has_row | np.isfinite(par[lr, j, 2:]).all(-1)
        after = (p - c) / w
        dev = np.abs(p - c)
        inside = [dev <= par[lr, j, 2 + li] for li in range(len(LEVELS))]
        for i in range(nt):
            a0, a1 = K["ts"][i], K["te"][i]
            if a1 <= a0:
                continue
            ww = None if wt is None else wt[a0:a1]
            x = after[a0:a1]
            fin = np.isfinite(x)
            xs = x[fin] if ww is None else np.repeat(x[fin], ww[fin])
            if len(xs) >= SA.MIN_FIT:
                ca = SA.centre_of(xs)
                out[i, j, 0], out[i, j, 1] = ca, SA.weighting_width(np.abs(xs - ca))
            v = valid[a0:a1]
            tot = v.sum() if ww is None else (ww * v).sum()
            out[i, j, ST_N] = tot
            for li in range(len(LEVELS)):
                ins = inside[li][a0:a1] & v
                out[i, j, ST_COV + li] = (ins.sum() if ww is None else (ww * ins).sum()) / tot if tot > 0 else np.nan
    return out


def _weights(b: int):
    """(fit weights, test weights) per event for replica b: None = unweighted (bootstrap nominal)."""
    P, scheme = G["P"], G["scheme"]
    if scheme == "bootstrap":
        if b == 0:
            return None, None
        rng = np.random.default_rng([G["seed"], 1, *G["fit_files"], b])
        return (rng.multinomial(P["n_ev_fit"], np.full(P["n_ev_fit"], 1.0 / P["n_ev_fit"])),
                rng.multinomial(P["n_ev_test"], np.full(P["n_ev_test"], 1.0 / P["n_ev_test"])))
    if b == 0:
        fm = P["nominal_fit_events"]
    else:
        rng = np.random.default_rng([G["seed"], 2, *G["fit_files"], b])
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


# ---------------------------------------------------------------- the rule (amendments 1-1e)
POOLED_REL_MARGIN = 0.1   # (iii): margin m(L) = 0.1 (1 - L) in coverage (efficiency) units
MARGIN_997 = 0.00045      # (iii) at 99.7 %: 0.045 % (amendment 1c)
GATED_LEVELS = (0.95, 0.975, 0.99, 0.997)    # (iii) levels fixed in the pre-registration (amendments 1c, 1d)
POOLED_Z = 1.6448536269514722   # one-sided 95 %
POWER_MIN = 0.8           # (iii) power below this at a gated level: a MINOR deviation (amendment 1e; never un-gates)
REGIONS = (("barrel", SA.BARREL), ("discs", SA.DISCS))
ST_N, ST_COV = 2, 3       # chain statistics st[..., bin, angle, :]: centre_after, width_after, n, coverage per level


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
    n_t = st[:, :, ST_N]
    cov = st[:, :, ST_COV:]
    m = np.rint(n_t[:, :, None] * (1.0 - cov))
    var = np.fmax(var_perm, _v_iid(K, n_t))
    return SA.coverage_z(m, n_t[:, :, None], K["E0"][:, None, :], var)


def iii_margin(lv: float) -> float:
    return MARGIN_997 if lv == 0.997 else POOLED_REL_MARGIN * (1 - lv)


def t_one_sided(n: int, alpha: float = 0.01) -> float:
    """One-sided Sidak family-wise threshold: n correct one-sided statistics exceed it with total probability alpha."""
    from scipy.stats import norm

    n = max(int(n), 1)
    return float(norm.isf(1.0 - (1.0 - alpha) ** (1.0 / n)))


def _band_excess(nom, sd):
    """Per test bin and angle: how many sd the after-table centre / width lie outside the decision band,
    u = max((|centre| - 0.25) / sd_c, (|width - 1| - 0.1) / sd_w), from nom [..., (centre, width)] and sd
    [..., (sd_c, sd_w)]; (i) fails a bin when u > W_i."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.fmax((np.abs(nom[..., 0]) - BAND_C) / sd[..., 0], (np.abs(nom[..., 1] - 1.0) - BAND_W) / sd[..., 1])


def _band_fraction(nom, sd, w):
    """max(|centre| / (0.25 + w sd_c), |width - 1| / (0.1 + w sd_w)) (reported; > 1 exactly when u > w)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.fmax(np.abs(nom[..., 0]) / (BAND_C + w * sd[..., 0]),
                       np.abs(nom[..., 1] - 1.0) / (BAND_W + w * sd[..., 1]))


def _region_index(K: dict) -> np.ndarray:
    return np.array([0 if tr["surface"] in SA.BARREL else 1 for tr in K["trows"]], np.int64)


def pooled_coverage(K: dict, st: np.ndarray):
    """Cluster-weighted pooled coverage per (replica, region, angle, level) from chain statistics
    st [..., nt, angle, 3 + levels] (all test bins with clusters). Returns (C, n)."""
    reg = _region_index(K)
    n = np.nan_to_num(st[..., ST_N])
    cov = np.nan_to_num(st[..., ST_COV:])
    out_c = np.zeros(st.shape[:-3] + (2, len(SA.ANGLES), len(LEVELS)))
    out_n = np.zeros(st.shape[:-3] + (2, len(SA.ANGLES)))
    for r in (0, 1):
        m = reg == r
        nn = n[..., m, :].sum(-2)
        out_n[..., r, :] = nn
        with np.errstate(invalid="ignore", divide="ignore"):
            out_c[..., r, :, :] = (n[..., m, :, None] * cov[..., m, :, :]).sum(-3) / nn[..., None]
    return out_c, out_n


def _decide(zk: dict, judged: dict, uk: dict, w_i: float):
    """(i) and (ii) on one set of statistics: zk[key] [nt, angle, level] coverage z, judged[key] [nt]
    (bins with >= 200 test clusters), uk[key] [nt, angle] band excess u of the after-table centre and
    width, w_i the (i) threshold. Returns (pass_i, pass_ii, N, t, max|z|, n beyond t, chi2 results,
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


def _iii_fail_cells(C: dict, sd: dict, levels) -> list:
    """(iii) failing cells at `levels`: one-sided 95 % upper bound of the shortfall L - C beyond iii_margin(L)."""
    out = []
    for key in C:
        for r, (rn, _) in enumerate(REGIONS):
            for j, a in enumerate(SA.ANGLES):
                for li, lv in enumerate(LEVELS):
                    if lv not in levels:
                        continue
                    ub = (lv - C[key][r, j, li]) + POOLED_Z * sd[key][r, j, li]
                    if not ub <= iii_margin(lv):
                        out.append(f"{key}/{rn}/{a}/{lv:g}")
    return out


def pooled_exact_sd(K: dict, n_t: np.ndarray, viid: np.ndarray) -> np.ndarray:
    """Exact pooled exchangeable sd of the cell coverage [region, angle, level]:
    sqrt(sum_i n_i^2 V_exch,i) / sum_i n_i over the cell's test bins (amendment 1c, N1 / N8)."""
    reg = _region_index(K)
    n = np.nan_to_num(n_t)
    v = np.where(np.isfinite(viid), viid, 0.0)
    out = np.zeros((2, len(SA.ANGLES), len(LEVELS)))
    for r in (0, 1):
        m = reg == r
        with np.errstate(invalid="ignore", divide="ignore"):
            out[r] = np.sqrt((n[m][:, :, None] ** 2 * v[m]).sum(0)) / n[m].sum(0)[:, None]
    return out


def _tag(key, a, tr):
    return f"{key}/{a}/{SA.SURFACE_NAMES[tr['surface']]}/f{tr['facing']:+d}/[{tr['lo']:+.2f},{tr['hi']:+.2f})"


def stability_report(P: dict, boot: dict, perm: dict) -> dict:
    """Reported only (amendment 1d: no flag, no effect on the rule): per table row and angle, the sd of the
    fitted centre and weighting width over the bootstrap and over the re-splits (pseudo-fit samples of
    the fit size drawn without replacement from the pooled events, their spread scaled by
    fpc = 1 / sqrt(1 - n_ev_fit / n_ev_pool) to a fresh fit's), in units of the row's width."""
    fpc = 1.0 / np.sqrt(1.0 - P["n_ev_fit"] / (P["n_ev_fit"] + P["n_ev_test"]))
    items = []
    for key, K in P["keys"].items():
        bp, pp = boot[key]["par"][:, :-1], perm[key]["par"][:, :-1]
        w0 = bp[0, :, :, 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            sdc_b, sdc_r = np.nanstd(bp[1:, :, :, 0], axis=0, ddof=1), fpc * np.nanstd(pp[1:, :, :, 0], axis=0, ddof=1)
            sdw_b, sdw_r = np.nanstd(bp[1:, :, :, 1], axis=0, ddof=1), fpc * np.nanstd(pp[1:, :, :, 1], axis=0, ddof=1)
        miss = np.mean(~np.isfinite(pp[1:, :, :, 1]), axis=0)
        for i, r in enumerate(K["rows"]):
            for j, a in enumerate(SA.ANGLES):
                if not np.isfinite(w0[i, j]):
                    continue
                items.append({"row": _tag(key, a, r), "n_fit": int(K["nf_row"][i]), "width": float(w0[i, j]),
                              "centre_sd_over_width": float(np.fmax(sdc_b[i, j], sdc_r[i, j]) / w0[i, j]),
                              "centre_sd_resplit_over_boot": float(sdc_r[i, j] / sdc_b[i, j]),
                              "width_rel_sd": float(np.fmax(sdw_b[i, j], sdw_r[i, j]) / w0[i, j]),
                              "width_sd_resplit_over_boot": float(sdw_r[i, j] / sdw_b[i, j]),
                              "resplits_missing": float(miss[i, j])})
    out = {"definition": stability_report.__doc__.split("\n")[0] + " ... (see stability_report)", "fpc": float(fpc),
           "row_angles": len(items)}
    for nm, lines in (("centre_sd_over_width", (0.0625, 0.125)), ("width_rel_sd", (0.05, 0.25))):
        x = np.array([it[nm] for it in items])
        out[nm] = {"pct_50_90_99_max": np.nanpercentile(x, [50, 90, 99, 100]).round(4).tolist(),
                   **{f"rows_above_{t:g}": int(np.sum(x > t)) for t in lines},
                   "worst": sorted(items, key=lambda it: -np.nan_to_num(it[nm]))[:15]}
    return out


def judge(P: dict, boot: dict, perm: dict, min_n: int = SA.MIN_FIT, false_veto: bool = True) -> dict:
    """Amendment 1e. (i) the after-table centre and width against 0 +- 0.25 and 1 +- 0.1 on every bin with
    >= min_n test clusters, error max(bootstrap, re-split sd), one-sided family-wise threshold t1(N_i),
    N_i = 2 x the judged bin-angles; (ii) coverage against the exact-shape null with the permutation
    variance (amendment 1); (iii) pooled signed coverage per key x region x angle x level against the
    NOMINAL level, one-sided 95 % upper bound of the shortfall (exact pooled exchangeable sd) within
    iii_margin(L), at GATED_LEVELS; its pass probability on the re-splits is reported. Also: the
    false-veto rate of (i) + (ii) and the per-level (iii) pass probability on the re-splits (leave-one-out
    errors), the materiality verdict (materiality), the rows that cannot reach a level, the stability
    report, per-cell summaries."""
    from scipy.stats import beta as beta_d

    zk, judged, uk, sdi, sdb, zlev, varp, viid = ({} for _ in range(8))
    for key, K in P["keys"].items():
        sb, sp = boot[key]["st"], perm[key]["st"]
        nom = sb[0]
        if not np.allclose(nom, sp[0], equal_nan=True, rtol=0, atol=1e-12):
            raise RuntimeError(f"{key}: the permutation chain's nominal split differs from the bootstrap's")
        sdb[key] = np.nanstd(sb[1:], axis=0, ddof=1)
        sdi[key] = np.fmax(sdb[key][..., :2], np.nanstd(sp[1:, :, :, :2], axis=0, ddof=1))
        varp[key] = np.nanvar(sp[1:, :, :, ST_COV:], axis=0, ddof=1)
        viid[key] = _v_iid(K, nom[:, :, ST_N])
        zk[key] = _coverage_z(K, nom, varp[key])
        judged[key] = K["test_n"] >= min_n
        uk[key] = _band_excess(nom, sdi[key])
        Kl = {**K, "E0": np.broadcast_to(np.asarray(LEVELS), K["E0"].shape)}
        zlev[key] = _coverage_z(Kl, nom, varp[key])
    n_i = 2 * int(sum(np.isfinite(uk[k][judged[k]]).sum() for k in uk))      # centre and width per judged bin-angle
    w_i = t_one_sided(n_i)
    pass_i, pass_ii, N, t, maxz, nb, chi_res, chi_fail, n_i_fail = _decide(zk, judged, uk, w_i)

    # (iii) pooled coverage against the nominal level
    Cn, Csd, Csd_boot, Cn_n, Cperm, E0p = {}, {}, {}, {}, {}, {}
    for key, K in P["keys"].items():
        cb, nb_ = pooled_coverage(K, boot[key]["st"])
        cp, _ = pooled_coverage(K, perm[key]["st"])
        Cn[key], Cn_n[key] = cb[0], nb_[0]
        Csd_boot[key] = np.nanstd(cb[1:], axis=0, ddof=1)
        Csd[key] = pooled_exact_sd(K, boot[key]["st"][0][..., ST_N], viid[key])
        Cperm[key] = cp[1:]
        reg = _region_index(K)
        n0 = np.nan_to_num(boot[key]["st"][0][..., ST_N])
        e0 = np.nan_to_num(K["E0"])
        E0p[key] = np.stack([(n0[reg == r][:, :, None] * e0[reg == r][:, None, :]).sum(0) / n0[reg == r].sum(0)[:, None]
                             for r in (0, 1)])
    R = perm[next(iter(P["keys"]))]["st"].shape[0] - 1
    power = {}
    for li, lv in enumerate(LEVELS):
        ok = np.ones(R, bool)
        for key in Cn:
            ub = (lv - Cperm[key][:, :, :, li]) + POOLED_Z * Csd[key][None, :, :, li]
            ok &= np.all(ub <= iii_margin(lv), axis=(1, 2))
        power[f"{lv:g}"] = float(ok.mean())
    low_power = [lv for lv in GATED_LEVELS if power[f"{lv:g}"] < POWER_MIN]
    iii_fail = _iii_fail_cells(Cn, Csd, GATED_LEVELS)
    cells = []
    for key, K in P["keys"].items():
        reg = _region_index(K)
        for r, (rn, _) in enumerate(REGIONS):
            for j, a in enumerate(SA.ANGLES):
                for li, lv in enumerate(LEVELS):
                    zz = zk[key][(reg == r) & judged[key], j, li]
                    zz = zz[np.isfinite(zz)]
                    c, sd = float(Cn[key][r, j, li]), float(Csd[key][r, j, li])
                    mg = iii_margin(lv)
                    cells.append({"cell": f"{key}/{rn}/{a}/{lv:g}", "coverage": c, "E0_pooled": float(E0p[key][r, j, li]),
                                  "sd": sd, "sd_bootstrap": float(Csd_boot[key][r, j, li]), "n_test": float(Cn_n[key][r, j]),
                                  "z_vs_level": (c - lv) / sd if sd > 0 else None,
                                  "shortfall_upper_95": (lv - c) + POOLED_Z * sd,
                                  "overcoverage_upper_95": (c - lv) + POOLED_Z * sd,
                                  "margin": mg, "gated": lv in GATED_LEVELS, "pass": bool((lv - c) + POOLED_Z * sd <= mg),
                                  "max_abs_z_bins": float(np.max(np.abs(zz))) if len(zz) else None})
    pass_iii = not iii_fail

    # reports: (i) details, unreachable rows, z list
    details_i = []
    for key, K in P["keys"].items():
        nom, sd = boot[key]["st"][0], sdi[key]
        raw = _band_fraction(nom, sd, w_i)
        for i, tr in enumerate(K["trows"]):
            if not judged[key][i]:
                continue
            for j, a in enumerate(SA.ANGLES):
                details_i.append({"bin": _tag(key, a, tr), "fraction_of_band": float(raw[i, j]), "u": float(uk[key][i, j]),
                                  "centre": float(nom[i, j, 0]), "sd_c": float(sd[i, j, 0]),
                                  "width": float(nom[i, j, 1]), "sd_w": float(sd[i, j, 1]),
                                  "sd_c_boot": float(sdb[key][i, j, 0]), "sd_w_boot": float(sdb[key][i, j, 1])})
    worst_i = sorted(details_i, key=lambda x: -np.nan_to_num(x["fraction_of_band"]))
    unreachable = []
    for key, K in P["keys"].items():
        for i, r in enumerate(K["rows"]):
            n = K["nf_row"][i]
            for lv in LEVELS:
                if n >= SA.MIN_FIT and not SA.level_reachable(n, lv):
                    unreachable.append({"row": _tag(key, "both", r), "level": lv, "n_fit": int(n),
                                        "E0": float(SA.expected_coverage(n, lv))})
    zlist = []
    for key, K in P["keys"].items():
        st0 = boot[key]["st"][0]
        for i, tr in enumerate(K["trows"]):
            if not judged[key][i]:
                continue
            for j, a in enumerate(SA.ANGLES):
                for li, lv in enumerate(LEVELS):
                    z = zk[key][i, j, li]
                    if np.isfinite(z):
                        zlist.append({"bin": _tag(key, a, tr), "level": lv, "coverage": float(st0[i, j, ST_COV + li]),
                                      "E0": float(K["E0"][i, li]), "n_test": int(K["test_n"][i]),
                                      "null_sd": float(np.sqrt(np.fmax(varp[key][i, j, li], viid[key][i, j, li]))),
                                      "perm_over_iid_var": float(varp[key][i, j, li] / viid[key][i, j, li]),
                                      "z": float(z), "z_vs_level": float(zlev[key][i, j, li])})
    zarr = np.array([x["z"] for x in zlist])
    zl = np.concatenate([zlev[k][judged[k]].ravel() for k in zlev])
    zl = zl[np.isfinite(zl)]
    res = {
        "rule": RULE,
        "i": {"bins": len(details_i), "N_i": n_i, "W_i": w_i, "fail": n_i_fail,
              "worst_fraction_of_widened_band": worst_i[0]["fraction_of_band"] if worst_i else None,
              "worst": worst_i[:8], "failures": [x for x in worst_i if x["fraction_of_band"] > 1.0][:60],
              "tolerance_pct_10_50_90": {
                  "centre": np.nanpercentile([BAND_C + w_i * x["sd_c"] for x in details_i], [10, 50, 90]).round(3).tolist(),
                  "width": np.nanpercentile([BAND_W + w_i * x["sd_w"] for x in details_i], [10, 50, 90]).round(3).tolist(),
                  "width_80pct_power": np.nanpercentile([BAND_W + (w_i + 0.8416) * x["sd_w"] for x in details_i],
                                                        [10, 50, 90]).round(3).tolist(),
                  "definition": "half-width a judged bin's after-table centre / width must leave to fail (i) (50 % power; "
                                "80 % with +0.84 sd), over the judged bin-angles, in weighting widths"},
              "n_a": int(sum((~judged[k]).sum() * len(SA.ANGLES) for k in judged))},
        "ii": {"N": N, "t_N": t, "max_abs_z": maxz, "n_beyond_t": nb, "n_beyond_3p5": int(np.sum(np.abs(zarr) > 3.5)),
               "beyond": sorted([x for x in zlist if abs(x["z"]) > t], key=lambda x: -abs(x["z"]))[:30],
               "largest": sorted(zlist, key=lambda x: -abs(x["z"]))[:12],
               "chi2": chi_res, "chi2_threshold_p": 0.01 / max(len(chi_res), 1), "chi2_fail": chi_fail,
               "per_level_max_abs_z": {f"{lv:g}": float(np.max(np.abs([x["z"] for x in zlist if x["level"] == lv])))
                                       for lv in LEVELS if any(x["level"] == lv for x in zlist)},
               "z_mean": float(zarr.mean()) if N else None, "z_sd": float(zarr.std()) if N else None,
               "reported_vs_level": {"max_abs_z": float(np.max(np.abs(zl))) if len(zl) else None,
                                     "n_beyond_t": int(np.sum(np.abs(zl) > t)), "mean_z": float(zl.mean()) if len(zl) else None}},
        "iii": {"margin_rel": POOLED_REL_MARGIN, "margins": {f"{lv:g}": iii_margin(lv) for lv in LEVELS},
                "power_by_level": power, "gated_levels": list(GATED_LEVELS), "power_below_0p8": low_power,
                "fail_cells": iii_fail, "cells": cells, "sd": "exact pooled exchangeable sd (bootstrap sd reported)"},
        "quantile_unreachable_rows": unreachable,
        "stability": stability_report(P, boot, perm),
        "lookup_missing": {k: P["keys"][k]["n_lookup_missing"] for k in P["keys"]},
    }
    ratio = []
    for key in P["keys"]:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio.append((varp[key] / viid[key])[judged[key]])
    rr = np.concatenate([x.reshape(-1, len(LEVELS)) for x in ratio])
    res["ii"]["perm_var_over_iid_pct"] = {f"{lv:g}": np.nanpercentile(rr[:, li], [5, 50, 95, 99]).round(3).tolist()
                                          for li, lv in enumerate(LEVELS)}
    res["pass_i"], res["pass_ii"], res["pass_iii"] = pass_i, pass_ii, pass_iii
    res["pass"] = pass_i and pass_ii and pass_iii
    res["per_region"] = region_summary(P, zk, {k: _band_fraction(boot[k]["st"][0], sdi[k], w_i) for k in uk}, judged)
    res["_z"] = {k: v.tolist() for k, v in zk.items()}
    res["_u"] = {k: v.tolist() for k, v in uk.items()}
    if false_veto:
        fv = false_veto_rate(P, perm, judged, sdb, w_i, Csd)
        for nm in ("fail_any", "fail_i_ii"):
            lo = beta_d.ppf(0.025, fv[nm], R - fv[nm] + 1)
            hi = beta_d.ppf(0.975, fv[nm] + 1, R - fv[nm])
            fv[f"{nm}_rate"] = fv[nm] / R
            fv[f"{nm}_95cl"] = [float(np.nan_to_num(lo)), float(hi)]
        fv["rate_95cl"] = fv["fail_any_95cl"]
        res["false_veto"] = fv
    res["materiality"] = materiality(P, res, zk, uk, judged, w_i, t)
    return res


def materiality(P: dict, res: dict, zk: dict, uk: dict, judged: dict, w_i: float, t: float) -> dict:
    """Amendment 1e: what counts for the gated run. MATERIAL: a pooled (iii) cell failing at 95 / 97.5 /
    99 %; or, per key and surface, the judged bins failing (i) or (ii) (either angle, any level) holding
    > 2 % of the surface's test clusters; or the (i) + (ii) false-veto rate on the run's own re-splits above
    5.7 % (when computed). MINOR: any other deviation (a (iii) cell at 99.7 %, a chi2 failure, failing bins
    below the share, (iii) power below 0.8): the table ships with them listed. verdict: pass / minor /
    material."""
    material, minor = [], []
    for c in res["iii"]["fail_cells"]:
        (material if float(c.rsplit("/", 1)[1]) in MATERIAL_LEVELS else minor).append(f"(iii) cell {c} fails")
    shares = {}
    for key, K in P["keys"].items():
        fail_i = np.any(np.asarray(uk[key]) > w_i, axis=1)
        fail_ii = np.any(np.abs(np.nan_to_num(zk[key])) > t, axis=(1, 2))
        fbin = judged[key] & (fail_i | fail_ii)
        n_t = np.asarray(K["test_n"], float)
        for s_ in SA.SURFACES:
            m = np.array([tr["surface"] == s_ for tr in K["trows"]])
            if not m.any() or not fbin[m].any():
                continue
            sh = float(n_t[m & fbin].sum() / n_t[m].sum())
            nm = f"{key}/{SA.SURFACE_NAMES[s_]}"
            shares[nm] = {"failing_bins": int((m & fbin).sum()), "failing_i": int((m & fbin & fail_i).sum()),
                          "failing_ii": int((m & fbin & fail_ii).sum()), "test_cluster_share": sh}
            msg = (f"{nm}: {int((m & fbin).sum())} bins failing (i) / (ii) hold {100 * sh:.2f} % of the surface's "
                   "test clusters")
            (material if sh > MATERIAL_SHARE else minor).append(msg)
    minor += [f"(ii) chi2 {c} fails" for c in res["ii"]["chi2_fail"]]
    minor += [f"(iii) pass probability below {POWER_MIN} at {lv:g}" for lv in res["iii"]["power_below_0p8"]]
    fv = res.get("false_veto")
    if fv is not None:
        msg = f"(i) + (ii) false-veto rate {100 * fv['fail_i_ii_rate']:.1f} % on the run's re-splits"
        if fv["fail_i_ii_rate"] > FALSE_VETO_MAX:
            material.append(msg)
    verdict = "material" if material else ("minor" if (minor or not res["pass"]) else "pass")
    return {"verdict": verdict, "material": material, "minor": minor, "failing_bin_shares": shares,
            "false_veto_judged": fv is not None,
            "definition": "amendment 1e: MATERIAL = a (iii) cell failing at 95-99 %, or (i) / (ii) failing bins holding "
                          "> 2 % of a surface's test clusters (per key), or a (i) + (ii) false-veto rate > 5.7 % on the "
                          "run's own re-splits; anything else MINOR (the table ships, deviations listed)"}


def false_veto_rate(P, perm, judged, sdb, w_i, Csd) -> dict:
    """The rule applied to every re-split as if it were the observed split (each is a known-good
    split of the pooled events), with leave-one-out errors: (i) the replica's after-table centre and
    width against the band widened by w_i x max(bootstrap sd of the observed split, re-split sd WITHOUT
    this replica); (ii) leave-one-out permutation variance, nominal E0; (iii) the replica's pooled coverage
    against the level with the observed split's exact pooled sd, at the gated levels."""
    keys = list(P["keys"])
    R = perm[keys[0]]["st"].shape[0] - 1
    fail = {"i": 0, "ii": 0, "i_ii": 0, "iii": 0, "any": 0}
    iii_pass_by_level = {f"{lv:g}": 0 for lv in GATED_LEVELS}
    maxz, minp, max_u = [], [], []
    tally = {}
    zrep, sdloo, Crep = {}, {}, {}
    for key in keys:
        K = P["keys"][key]
        st = perm[key]["st"][1:]
        cov = st[:, :, :, ST_COV:]
        s1, s2 = np.nansum(cov, axis=0), np.nansum(cov ** 2, axis=0)
        loo_mean = (s1[None] - cov) / (R - 1)
        loo_var = (s2[None] - cov ** 2 - (R - 1) * loo_mean ** 2) / (R - 2)
        zrep[key] = np.stack([_coverage_z(K, st[b], loo_var[b]) for b in range(R)])
        x = st[:, :, :, :2]
        t1, t2 = np.nansum(x, axis=0), np.nansum(x ** 2, axis=0)
        lm = (t1[None] - x) / (R - 1)
        lv = np.clip((t2[None] - x ** 2 - (R - 1) * lm ** 2) / (R - 2), 0, None)
        sdloo[key] = np.fmax(sdb[key][None, ..., :2], np.sqrt(lv))
        Crep[key], _ = pooled_coverage(K, perm[key]["st"][1:])
    for b in range(R):
        zb = {k: zrep[k][b] for k in keys}
        ub = {k: _band_excess(perm[k]["st"][1 + b], sdloo[k][b]) for k in keys}
        pi, pii, N, t, mz, nb, chi_res, chi_fail, _ = _decide(zb, judged, ub, w_i)
        piii = not _iii_fail_cells({k: Crep[k][b] for k in keys}, Csd, GATED_LEVELS)
        for lv in GATED_LEVELS:
            iii_pass_by_level[f"{lv:g}"] += not _iii_fail_cells({k: Crep[k][b] for k in keys}, Csd, [lv])
        u_b = []
        for k in keys:
            u = np.where(judged[k][:, None], ub[k], np.nan)
            u_b.append(np.nanmax(u))
            for i, j in zip(*np.nonzero(u > w_i)):
                tag = _tag(k, SA.ANGLES[j], P["keys"][k]["trows"][i])
                tally[tag] = tally.get(tag, 0) + 1
        max_u.append(max(u_b))
        fail["i"] += not pi
        fail["ii"] += not pii
        fail["i_ii"] += not (pi and pii)
        fail["iii"] += not piii
        fail["any"] += not (pi and pii and piii)
        maxz.append(mz)
        minp.append(min(v["p"] for v in chi_res.values()))
    return {"n_replicas": R, "fail_i": fail["i"], "fail_ii": fail["ii"], "fail_i_ii": fail["i_ii"], "fail_iii": fail["iii"],
            "fail_any": fail["any"], "iii_pass_probability_by_level": {k: v / R for k, v in iii_pass_by_level.items()},
            "rate": fail["any"] / R, "max_abs_z_pct_5_50_95_99": np.percentile(maxz, [5, 50, 95, 99]).round(2).tolist(),
            "min_chi2_p_pct_1_5_50": np.percentile(minp, [1, 5, 50]).tolist(),
            "i_max_excess_sigma_pct_5_50_95_99": np.percentile(max_u, [5, 50, 95, 99]).round(2).tolist(),
            "i_fail_tally": dict(sorted(tally.items(), key=lambda x: -x[1])[:20]),
            "definition": "each re-split judged by the amendment-1e rule as if observed, leave-one-out errors (see "
                          "false_veto_rate)"}


def region_summary(P, zk, frac, judged):
    """Worst |z| and worst band fraction per (key, barrel / disc), as the demonstration must show both."""
    out = {}
    for key, K in P["keys"].items():
        for region, surfs in REGIONS:
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
        rng = np.random.default_rng([G["seed"], 7, *G["fit_files"], b])
        cf = rng.multinomial(PT["n_ev"], np.full(PT["n_ev"], 1.0 / PT["n_ev"]))
    out = {}
    for key, K in PT["keys"].items():
        res = np.full((len(K["rows"]), len(SA.ANGLES), 2), np.nan)
        for i, sl in enumerate(K["slices"]):
            if sl is None:
                continue
            for j, a in enumerate(SA.ANGLES):
                e = [SA.bin_estimates(K["p"][a][x], LEVELS, None if cf is None else cf[K["ev"][x]], diagnostics=False)
                     for x in sl]
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
    """Per test bin (nominal): the pulls before the table (published sigma; and today's census sigma on
    the barrel: per-layer scale, no centre), with the table's estimators (centre, weighting width) and the
    B.1 core width as a diagnostic; after the table, the after-table centre and width (the (i)
    statistics); the coverage with its E0 and z; today's 3-census-sigma class-T efficiency; the table's
    half-width at each level in today's census sigma."""
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
                b = SA.bin_estimates(p, LEVELS)
                d = {"before": None if b is None else {"centre": b["centre"], "width": b["width"],
                                                       "core_width": b["core_width"], "out3": b["out3"]},
                     "after": {"centre": nom[i, j, 0], "sd_centre": sd[i, j, 0], "width": nom[i, j, 1],
                               "sd_width": sd[i, j, 1]},
                     "coverage": {f"{lv:g}": {"cov": nom[i, j, ST_COV + li], "sd_bootstrap": sd[i, j, ST_COV + li],
                                              "E0": K["E0"][i, li], "z": zk[i, j, li]}
                                  for li, lv in enumerate(LEVELS)},
                     "table": {"centre": par[i, j, 0], "sd_centre": psd[i, j, 0], "width": par[i, j, 1],
                               "sd_width": psd[i, j, 1],
                               "q": {f"{lv:g}": par[i, j, 2 + li] for li, lv in enumerate(LEVELS)},
                               "sd_q": {f"{lv:g}": psd[i, j, 2 + li] for li, lv in enumerate(LEVELS)}}}
                if tr["surface"] in SA.BARREL and m.sum():
                    sc = scales[a][tr["surface"]]
                    pc = p / sc
                    bc = SA.bin_estimates(pc, LEVELS)
                    d["before_census"] = None if bc is None else {"centre": bc["centre"], "width": bc["width"],
                                                                  "core_width": bc["core_width"]}
                    d["census_3sigma_eff"] = float(np.mean(np.abs(pc) <= 3.0))
                    d["half_width_census_sigma"] = {f"{lv:g}": par[i, j, 2 + li] / sc for li, lv in enumerate(LEVELS)}
                ent[a] = d
            rows.append(ent)
        out[key] = rows
    return out


def table_doc(P: dict, stack: dict, orient, flags, prov: dict) -> dict:
    """The table document (angle.error_table, sheet v2.7). Per row and angle: the centre, the weighting
    width and the quantiles (with their bootstrap sd), n_fit (the clusters the row was fitted
    on), E0 per level (k / (n_fit + 1)) and q_meets_level_in_expectation (False at a level the row's
    n_fit cannot reach: at 99.7 %, n_fit <= 332), and the reported diagnostics (B.1 clipped mean and core
    width, MAD, fraction beyond 3 widths, full RMS)."""
    keys_rows = {}
    for key, K in P["keys"].items():
        par = stack[key]["par"][0]
        psd = np.nanstd(stack[key]["par"][1:], axis=0, ddof=1)
        rows = []
        for i, r in enumerate(K["rows"]):
            ent = {k: r[k] for k in ("surface", "facing", "lo", "hi", "fine", "n_fit")}
            n = K["nf_row"][i]
            ent["n_fit_used"] = int(n)
            ent["flags"] = ["refit_pt2"] if i in flags.get(key, set()) else []
            for j, a in enumerate(SA.ANGLES):
                full = SA.bin_estimates(K["fit_p"][a][K["fs"][i]:K["fe"][i]], LEVELS)
                if full is None:
                    ent[a] = None
                    continue
                ent[a] = {"centre": par[i, j, 0], "width": par[i, j, 1],
                          "q": {f"{lv:g}": par[i, j, 2 + li] for li, lv in enumerate(LEVELS)},
                          "E0": {f"{lv:g}": float(SA.expected_coverage(n, lv)) for lv in LEVELS},
                          "q_meets_level_in_expectation": {f"{lv:g}": bool(SA.level_reachable(n, lv)) for lv in LEVELS},
                          "sd_centre": psd[i, j, 0], "sd_width": psd[i, j, 1],
                          "sd_q": {f"{lv:g}": psd[i, j, 2 + li] for li, lv in enumerate(LEVELS)},
                          "n": full["n"], "diagnostics": {k: full[k] for k in
                                                          ("clip_centre", "core_width", "mad", "out3", "full_rms0")}}
            rows.append(ent)
        keys_rows[key] = rows
    doc = SA.new_table_doc(keys_rows, orient, prov, era="canon_v3 (150X D121 PU200 TTbar, AAAA)",
                           fit_sample=prov.get("fit_sample_text") or (
                               f"canon_v3 AAAA files {list(SA.FIT_FILES)}, class T, pT >= {SA.PT_MIN:g} GeV (the class-C "
                               "angles of these files are cluster-CNN training data; class T is truth + 1.5 x a PixelAV "
                               "residual and is not)"))
    doc["recipe"] = f"angle.error_table (sheet {SA.SHEET_VERSION}); tested under {RULE}"
    return doc


def plateau_edge_rows(rows: list) -> np.ndarray:
    """Rows whose |key| range overlaps the plateau edges [0.50, 0.67) (KB-q95, KB-c, KB-w)."""
    out = []
    for r in rows:
        lo, hi = r["lo"], r["hi"]
        a, b = (min(abs(lo), abs(hi)), max(abs(lo), abs(hi))) if lo * hi >= 0 else (0.0, max(abs(lo), abs(hi)))
        out.append(b > 0.50 and a < 0.67)
    return np.array(out, bool)


def _strip(j: dict) -> dict:
    for k in ("_z", "_u"):
        j.pop(k, None)
    return j


def perturbed_bins(P: dict, j: dict) -> dict:
    """Per perturbed known-bad: over the judged test bins whose table rows were perturbed (canonical
    tables: bin = row), how many fail (i) and how many have a coverage |z| > t(N) at 95 %."""
    n, n_i, n_z95, z95 = 0, 0, 0, []
    for key, K in P["keys"].items():
        m = K["pert_rows"][:-1] & (K["test_n"] >= SA.MIN_FIT)
        u = np.asarray(j["_u"][key])[m]
        z = np.asarray(j["_z"][key])[m][:, :, I95]
        n += int(m.sum()) * len(SA.ANGLES)
        n_i += int(np.sum(u > j["i"]["W_i"]))
        n_z95 += int(np.sum(np.abs(z) > j["ii"]["t_N"]))
        z95 += z[np.isfinite(z)].tolist()
    return {"perturbed_judged_bin_angles": n, "failing_i": n_i, "per_row_power_i": n_i / n if n else None,
            "z95_beyond_t": n_z95, "z95_median": float(np.median(z95)) if z95 else None}


# ---------------------------------------------------------------- driver
def run(mode: str, n_rep: int, n_proc: int, out_dir: Path, seed: int = SEED) -> dict:
    fit_files, test_files = SPLITS[mode]
    t0 = time.time()
    orient = CM.orientation_map(fit_files)
    fit = CM.derive(load_split(fit_files), orient)
    test = CM.derive(load_split(test_files), orient)
    print(f"[{mode}] fit files {fit_files}: {len(fit['ev'])} clusters; test files {test_files}: "
          f"{len(test['ev'])} clusters; orientation of {len(orient[0])} modules ({time.time() - t0:.0f} s)", flush=True)
    G["seed"], G["fit_files"] = seed, tuple(int(x) for x in fit_files)
    res = {"mode": mode, "rule": RULE, "fit_files": list(fit_files),
           "test_files": list(test_files), "n_rep": n_rep, "seed": seed, "n_fit": int(len(fit["ev"])),
           "n_test": int(len(test["ev"])), "code": CM.git_state(), "definition_sheet": SA.SHEET_VERSION,
           "inputs": CM.inputs_provenance(sorted(set(fit_files) | set(test_files))),
           "rule_parameters": {
               "band_width": BAND_W, "band_centre": BAND_C, "levels": LEVELS, "min_n": SA.MIN_FIT,
               "centre": "25 %-trimmed mean of the pulls",
               "weighting_width": f"mean of the smallest {SA.WIDTH_FRACTION * 100:g} % of |pull - centre| / "
                                  f"{SA.WIDTH_DIVISOR:g}",
               "quantile_rank": "k = min(ceil(L (n + 1)), n)",
               "i": "every bin with >= 200 test clusters: fail when the after-table centre / width is outside 0 +- 0.25 "
                    "/ 1 +- 0.1 by > W_i x max(bootstrap, re-split sd), W_i = one-sided Sidak t1(N_i) at 1 %, N_i = 2 x "
                    "judged bin-angles",
               "ii": "beta-binomial null, mean 1 - E0 (E0 = k / (n_fit + 1)), variance max(event permutation, "
                     "exchangeable clusters); mid-p; z = -Phi^-1(P_mid(M <= m)); no |z| > t(N); chi2 per level x key",
               "iii": "pooled coverage per key x region x angle x level against the nominal level; one-sided 95 % "
                      "upper bound of the shortfall (exact pooled exchangeable sd) <= 0.1 (1 - L) (0.045 % at 99.7 %); "
                      "gated levels fixed: 95 / 97.5 / 99 / 99.7 %; power < 0.8 reported",
               "materiality": {"iii_levels": list(MATERIAL_LEVELS), "failing_bin_share": MATERIAL_SHARE,
                               "false_veto_max": FALSE_VETO_MAX},
               "seed": "replica streams [seed, scheme, fit files..., replica]"}}
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
        prov = {"code": res["code"], "inputs": res["inputs"], "fit_files": list(fit_files), "rule": RULE,
                "job_id": os.environ.get("SP_JOB_ID"), "n_rep_bootstrap": n_rep, "pt_check_refit_rows": ptc["flags"]}
        if mode in ("gated", "fit-only"):
            prov["test"] = (f"files 3-10; see {result_name('gated')}" if mode == "gated" else
                            "NOT TESTED: the gated test did not run (amendment 1e not approved, its known-good "
                            f"demonstration failed or a known-bad passed; see {result_name('known-good')}, "
                            f"{result_name('known-bad')})")
            doc = table_doc(P, boot, orient, flags, prov)
            if mode == "gated":
                j = res["judgement"]
                doc["test_result"] = {k: j[k] for k in ("pass", "pass_i", "pass_ii", "pass_iii")}
                doc["test_result"]["verdict"] = j["materiality"]["verdict"]
                doc["test_result"]["deviations"] = {"material": j["materiality"]["material"],
                                                    "minor": j["materiality"]["minor"]}
                doc["test_result"]["false_veto"] = {k: j["false_veto"][k] for k in
                                                    ("fail_i_ii_rate", "fail_i_ii_95cl", "iii_pass_probability_by_level")}
            else:
                doc["test_result"] = {"pass": None, "status": "untested"}
            fn = TABLE_FILE if mode == "gated" else UNTESTED_FILE
            SA.AngleTable(doc).save(str(out_dir / fn))
            res["table_file"] = str(out_dir / fn)
        else:      # the demonstration's own fit, for the width comparison and the packet
            prov["fit_sample_text"] = (f"DEMONSTRATION FIT ONLY (not for use): canon_v3 AAAA files {list(fit_files)}, "
                                       f"class T, pT >= {SA.PT_MIN:g} GeV")
            prov["test"] = f"files {list(test_files)}; see {result_name(mode)}"
            doc = table_doc(P, boot, orient, flags, prov)
            (out_dir / AMEND_DIR).mkdir(exist_ok=True)
            fn = out_dir / AMEND_DIR / f"angle_table_{mode}_fit_a1e.json"
            SA.AngleTable(doc).save(str(fn))
            res["demo_table_file"] = str(fn)
    else:
        res["known_bads"] = {}
        canonical = {k: {"kind": "canonical", "lookup_key": k} for k in SA.KEYS}
        for name, desc in KNOWN_BADS.items():
            tst = test
            if name == "wrong_key":
                spec = {"true": {"kind": "canonical", "lookup_key": "emitted"},
                        "emitted": {"kind": "canonical", "lookup_key": "true"}}
            elif name in TEST_SCALE:
                spec = canonical
                tst = {**test, "pull": {a: v * TEST_SCALE[name] for a, v in test["pull"].items()}}
            else:
                spec = {k: {"kind": name, "lookup_key": k} for k in SA.KEYS}
            P = prepare(fit, tst, spec)
            boot = run_chain(P, n_rep, n_proc, f"known-bad {name}", "bootstrap")
            perm = run_chain(prepare(fit, tst, spec, pooled=True), n_rep, n_proc, f"known-bad {name}", "permutation")
            j = _strip(judge(P, boot, perm, false_veto=False))
            res["known_bads"][name] = {"description": desc, "judgement": j, "fails_as_required": not j["pass"],
                                       "materiality_verdict": j["materiality"]["verdict"]}
            print(f"[known-bad {name}] pass={j['pass']} (i fail {j['i']['fail']}/{j['i']['bins']}, ii beyond t "
                  f"{j['ii']['n_beyond_t']}, max|z| {j['ii']['max_abs_z']:.1f}, chi2 fail {len(j['ii']['chi2_fail'])}, "
                  f"iii fail {len(j['iii']['fail_cells'])}); materiality {j['materiality']['verdict']}", flush=True)
        # the canonical table on the same split: the reference for (e)
        P = prepare(fit, test, canonical)
        boot = run_chain(P, n_rep, n_proc, "canonical reference", "bootstrap")
        perm = run_chain(prepare(fit, test, canonical, pooled=True), n_rep, n_proc, "canonical reference", "permutation")
        jc = _strip(judge(P, boot, perm, false_veto=False))
        ref = {c["cell"]: c for c in jc["iii"]["cells"]}
        je = res["known_bads"]["test_scale_1p05"]["judgement"]
        res["known_bads"]["test_scale_1p05"]["per_cell_vs_canonical"] = [
            {"cell": c["cell"], "coverage_shift": c["coverage"] - ref[c["cell"]]["coverage"],
             "max_abs_z_bins": c["max_abs_z_bins"], "max_abs_z_bins_canonical": ref[c["cell"]]["max_abs_z_bins"],
             "shortfall_upper_95": c["shortfall_upper_95"], "margin": c["margin"], "pass": c["pass"]}
            for c in je["iii"]["cells"]]
        # KB-q95, KB-c and KB-w: the canonical table with its plateau-edge rows perturbed in every replica
        for name, kb in PERTURBED.items():
            spec = {k: {**canonical[k], "pert": kb["pert"]} for k in SA.KEYS}
            P = prepare(fit, test, spec)
            boot = run_chain(P, n_rep, n_proc, f"known-bad {name}", "bootstrap")
            perm = run_chain(prepare(fit, test, spec, pooled=True), n_rep, n_proc, f"known-bad {name}", "permutation")
            j = judge(P, boot, perm, false_veto=False)
            pb = perturbed_bins(P, j)
            j = _strip(j)
            through = (not j["pass_i"]) if kb["must_fail"] == "i" else not (j["pass_ii"] and j["pass_iii"])
            res["known_bads"][name] = {"description": kb["description"], "judgement": j, "perturbed_bins": pb,
                                       "must_fail_through": kb["must_fail"], "fails_through_required_test": through,
                                       "fails_as_required": bool(through),
                                       "materiality_verdict": j["materiality"]["verdict"]}
            print(f"[known-bad {name}] pass={j['pass']} (i {j['pass_i']}, ii {j['pass_ii']}, iii {j['pass_iii']}); "
                  f"perturbed judged bin-angles {pb['perturbed_judged_bin_angles']}: {pb['failing_i']} fail (i), "
                  f"{pb['z95_beyond_t']} with |z95| > t(N) (median z95 {pb['z95_median']}); iii fail cells "
                  f"{j['iii']['fail_cells']}; materiality {j['materiality']['verdict']}", flush=True)
        res["canonical_reference"] = {"pass": jc["pass"], "i": {k: jc["i"][k] for k in ("bins", "fail", "W_i")},
                                      "iii": jc["iii"], "materiality": jc["materiality"]}
        res["all_known_bads_fail"] = all(v["fails_as_required"] for v in res["known_bads"].values())
    if "judgement" in res:
        _strip(res["judgement"])
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def gated_refusal(out_dir: Path) -> str | None:
    """Why the gated test may not run yet (None when it may): the amendment-1e known-good and known-bad
    results must exist and behave as required, and the approval file must exist."""
    try:
        kg = json.load(open(out_dir / result_name("known-good")))
        kb = json.load(open(out_dir / result_name("known-bad")))
    except FileNotFoundError as e:
        return f"the gated test needs the amendment-1e known-good and known-bad results first ({e})"
    if kg.get("rule") != RULE or kb.get("rule") != RULE:
        return "the known-good / known-bad results are not from the amendment-1e rule"
    if kg["judgement"]["materiality"]["verdict"] != "pass" or not kb["all_known_bads_fail"] or kg["n_rep"] < 250 \
            or kb["n_rep"] < 250:
        return ("the amendment-1e known-good demonstration did not pass, or a known-bad did not fail, or < 250 "
                "replicas: the rule goes back for redesign (D.0 step 4)")
    ap = out_dir / APPROVAL_FILE
    if not ap.exists():
        return f"amendment 1e is not approved: {ap} does not exist (it quotes the physicist's approval and its date)"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=tuple(SPLITS))
    ap.add_argument("--rep", type=int, default=250)
    ap.add_argument("--proc", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(CM.OUT_DIR))
    ap.add_argument("--seed", type=int, default=SEED, help="replica seed (smoke tests: never the demonstration's)")
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.mode == "gated":
        why = gated_refusal(out_dir)
        if why:
            print(f"refused: {why}")
            return 66
    res = run(a.mode, a.rep, a.proc, out_dir, a.seed)
    CM.jdump(res, out_dir / result_name(a.mode))
    if a.mode == "known-bad":
        print("every known-bad fails" if res["all_known_bads_fail"] else "SOME KNOWN-BAD PASSES")
        return 0 if res["all_known_bads_fail"] else 3
    if a.mode == "fit-only":
        print(f"wrote the UNTESTED table {res['table_file']}")
        return 0
    j = res["judgement"]
    fv = j.get("false_veto", {})
    s = j["stability"]
    mt = j["materiality"]
    print(f"[{a.mode}] PASS={j['pass']} VERDICT={mt['verdict']}: (i) {j['i']['fail']} of {j['i']['bins']} bins outside "
          f"the widened band (worst {j['i']['worst_fraction_of_widened_band']}); (ii) N={j['ii']['N']} "
          f"t(N)={j['ii']['t_N']:.2f} max|z|={j['ii']['max_abs_z']:.2f}, beyond t: {j['ii']['n_beyond_t']}, chi2 fail: "
          f"{j['ii']['chi2_fail']}; (iii) fail cells {j['iii']['fail_cells']} (power {j['iii']['power_by_level']}); "
          f"(i)+(ii) false-veto {fv.get('fail_i_ii')}/{fv.get('n_replicas')} ((i) {fv.get('fail_i')}, (ii) {fv.get('fail_ii')}); "
          f"(iii) pass probability {fv.get('iii_pass_probability_by_level')}; unreachable rows "
          f"{len(j['quantile_unreachable_rows'])}; centre sd / width > 0.125 in "
          f"{s['centre_sd_over_width']['rows_above_0.125']} rows, width rel sd > 0.25 in "
          f"{s['width_rel_sd']['rows_above_0.25']} rows; material {mt['material']}; minor {mt['minor']}")
    return {"pass": 0, "minor": 5, "material": 4}[mt["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
