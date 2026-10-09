#!/usr/bin/env python
"""P0.3 step 2: the angle-uncertainty table (plan D.3 D10(b), D.0; decisions 2, 3, 11).

Recipe (angle.error_table): class-T clusters with a usable direction and TP pT >= 1 GeV; per
surface (IL1-IL4, ID1-ID4) x signed incidence bin (signed cot theta barrel; signed 1/cot theta
discs, per module facing), merged to >= 200 fit clusters away from the plateau edges; per bin and
angle (alpha, beta) the centre (clipped mean), core width (B.1 core of the centred pulls) and the
quantiles of |pull - centre| at 95 / 97.5 / 99 / 99.7 %; in both keys (emitted, true). pT check on
the fit sample (1-2 vs >= 2 GeV, D.0 agreement rule): a bin failing in both signs is refitted at
pT >= 2 GeV and flagged.

Test (D.3): on the test sample, per bin of the same binning, (i) the pull centre and width after the
table within 0 +- 0.25 and 1 +- 0.1, widened by 2 combined sigma; (ii) each quantile's coverage
against its level: no z beyond t(N) for the N z-scores read (levels x bins x angles x keys), and
per level and key chi2 p > 0.01 / (number of chi2 tests). Errors: event-level bootstrap of the
whole fit-and-test chain (fit and test events resampled, every fitted step re-fitted per replica).

    python table.py known-good [--rep 250] [--proc 10]   rule demonstration: fit files 3-4, test 5-10
    python table.py known-bad  [--rep 250] [--proc 10]   the four known-bad tables on the same split
    python table.py gated      [--rep 250] [--proc 10]   fit 1-2, test 3-10; refuses unless the
                                                         known-good passed and every known-bad failed
    python table.py fit-only   [--rep 250] [--proc 10]   the recipe on files 1-2 without the test (only
                                                         when the gated test cannot run): UNTESTED table
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

SEED = 20261009
LEVELS = SA.LEVELS
BAND_W, BAND_C, WIDEN = 0.1, 0.25, 2.0
REGISTRY_SCALES = Path.home() / "smartpixels" / "sp_registry" / "results" / "appa.census.angle-scales.json"
SPLITS = {"known-good": ((3, 4), (5, 6, 7, 8, 9, 10)), "known-bad": ((3, 4), (5, 6, 7, 8, 9, 10)),
          "gated": (SA.FIT_FILES, SA.TEST_FILES), "fit-only": (SA.FIT_FILES, SA.FIT_FILES)}
TABLE_FILE = "angle_table_v1_canon_v3.json"
UNTESTED_FILE = "angle_table_v1_canon_v3_UNTESTED.json"
KNOWN_BADS = {
    "folded": "the folded |key| table (signs pooled; the round-4 table's binning), read by |key|",
    "cotbinned_disc": "disc rows binned like the barrel in signed cot theta (barrel edges; 75-98% of disc clusters in "
                      "|cot| > 2.5), per facing",
    "pooled_disc": "disc rows with the two module facings pooled",
    "wrong_key": "the table read in the wrong key (the true-key table looked up with the emitted key, and vice versa)",
}


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


# ---------------------------------------------------------------- the replica engine (fork workers)
G: dict = {}


def _sorted_slices(row: np.ndarray, n: int):
    o = np.argsort(row, kind="stable")
    r = row[o]
    return o, np.searchsorted(r, np.arange(n), "left"), np.searchsorted(r, np.arange(n), "right")


def prepare(fit: dict, test: dict, spec: dict, flags: dict | None = None) -> dict:
    """spec = {test_key: {"kind", "lookup_key"}}; the table's binning `kind` is fitted in test_key
    and read with lookup_key; the test bins are the canonical rows of test_key on the same fit
    sample. flags = {test_key: set of row ids refitted at pT >= 2} (canonical tables only)."""
    P = {"keys": {}}
    uf, evf = event_index(fit["ev"])
    ut, evt = event_index(test["ev"])
    P["n_ev_fit"], P["n_ev_test"] = len(uf), len(ut)
    for key, sp in spec.items():
        rows, rowmap = build_rows(sp["kind"], fit, key)
        trows, trowmap = build_rows("canonical", fit, key)  # the test binning
        frow = row_of(sp["kind"], rowmap, fit, key)
        fl = (flags or {}).get(key, set())
        if fl:
            frow = np.where(np.isin(frow, list(fl)) & (fit["tp_pt"] < 2.0), -1, frow)
        lrow = row_of(sp["kind"], rowmap, test, sp["lookup_key"])
        tbin = row_of("canonical", trowmap, test, key)
        fo, fs, fe = _sorted_slices(frow, len(rows))
        to, ts, te = _sorted_slices(tbin, len(trows))
        P["keys"][key] = {
            "rows": rows, "trows": trows, "trowmap": trowmap, "flags": sorted(int(x) for x in fl), "kind": sp["kind"],
            "lookup_key": sp["lookup_key"],
            "fit_ev": evf[fo], "fit_p": {a: fit["pull"][a][fo] for a in SA.ANGLES}, "fs": fs, "fe": fe,
            "test_ev": evt[to], "test_p": {a: test["pull"][a][to] for a in SA.ANGLES},
            "test_lrow": lrow[to], "ts": ts, "te": te, "n_lookup_missing": int(np.sum((lrow < 0) & (tbin >= 0))),
            "test_n": (te - ts).astype(int)}
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


def _replica(b: int):
    P = G["P"]
    if b == 0:
        cf = ct = None
    else:
        rng = np.random.default_rng([SEED, b])
        cf = rng.multinomial(P["n_ev_fit"], np.full(P["n_ev_fit"], 1.0 / P["n_ev_fit"]))
        ct = rng.multinomial(P["n_ev_test"], np.full(P["n_ev_test"], 1.0 / P["n_ev_test"]))
    res = {}
    for key, K in P["keys"].items():
        par = _fit_params(K, None if cf is None else cf[K["fit_ev"]])
        st = _test_stats(K, par, None if ct is None else ct[K["test_ev"]])
        res[key] = (par, st)
    return b, res


def run_chain(P: dict, n_rep: int, n_proc: int, label: str) -> dict:
    """Nominal (b = 0) and n_rep bootstrap replicas of the fit-and-test chain."""
    G["P"] = P
    t0 = time.time()
    out = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_proc) as pool:
        for i, (b, res) in enumerate(pool.imap_unordered(_replica, range(n_rep + 1), chunksize=2)):
            out[b] = res
            if (i + 1) % 25 == 0 or i == n_rep:
                print(f"  [{label}] {i + 1}/{n_rep + 1} replicas, {time.time() - t0:.0f} s", flush=True)
    stack = {}
    for key in P["keys"]:
        stack[key] = {"par": np.stack([out[b][key][0] for b in range(n_rep + 1)]),
                      "st": np.stack([out[b][key][1] for b in range(n_rep + 1)])}
    return stack


# ---------------------------------------------------------------- the rule
def judge(P: dict, stack: dict, min_n: int = SA.MIN_FIT) -> dict:
    """(i) adequacy band and (ii) coverage family-wise rule over every key and angle."""
    from scipy.stats import chi2

    zs, chis, bins_i, n_i_fail, details = [], {}, 0, 0, {}
    per_key = {}
    for key, S in stack.items():
        st = S["st"]
        nom, rep = st[0], st[1:]
        sd = np.nanstd(rep, axis=0, ddof=1)
        K = P["keys"][key]
        dk = {"i_fail": [], "i_na": 0, "ii_z": [], "ii_na": 0}
        for i, tr in enumerate(K["trows"]):
            n_test = K["test_n"][i]
            for j, a in enumerate(SA.ANGLES):
                tag = f"{key}/{a}/{SA.SURFACE_NAMES[tr['surface']]}/f{tr['facing']:+d}/[{tr['lo']:+.2f},{tr['hi']:+.2f})"
                c, w = nom[i, j, 0], nom[i, j, 1]
                if n_test < min_n or not np.isfinite(c):
                    dk["i_na"] += 1
                else:
                    bins_i += 1
                    sc, sw = sd[i, j, 0], sd[i, j, 1]
                    fail_c = abs(c) > BAND_C + WIDEN * sc
                    fail_w = abs(w - 1) > BAND_W + WIDEN * sw
                    if fail_c or fail_w:
                        n_i_fail += 1
                        dk["i_fail"].append({"bin": tag, "centre": c, "sd_c": sc, "width": w, "sd_w": sw})
                    details.setdefault("i_worst", []).append(
                        (max(abs(c) / (BAND_C + WIDEN * sc), abs(w - 1) / (BAND_W + WIDEN * sw)), tag))
                for li, lv in enumerate(LEVELS):
                    cov, s = nom[i, j, 3 + li], sd[i, j, 3 + li]
                    if n_test < min_n or not np.isfinite(cov) or not (s > 0):
                        dk["ii_na"] += 1
                        continue
                    z = (cov - lv) / s
                    zs.append((z, tag, lv, cov, s))
                    chis.setdefault((key, lv), []).append(z)
        per_key[key] = dk
    N = len(zs)
    t = t_family(N) if N else float("nan")
    zarr = np.array([z[0] for z in zs])
    beyond = [z for z in zs if abs(z[0]) > t]
    n_chi = len(chis)
    chi_res = {f"{k}/{lv:g}": {"chi2": float(np.sum(np.square(v))), "ndf": len(v),
                               "p": float(chi2.sf(np.sum(np.square(v)), len(v)))} for (k, lv), v in chis.items()}
    chi_fail = {k: v for k, v in chi_res.items() if v["p"] <= 0.01 / n_chi}
    worst_i = sorted(details.get("i_worst", []), reverse=True)[:5]
    res = {
        "i": {"bins": bins_i, "fail": n_i_fail, "worst_fraction_of_widened_band": worst_i[0][0] if worst_i else None,
              "worst": [{"bin": t_, "fraction_of_band": f} for f, t_ in worst_i],
              "failures": [x for k in per_key.values() for x in k["i_fail"]][:50],
              "n_a": sum(k["i_na"] for k in per_key.values())},
        "ii": {"N": N, "t_N": t, "max_abs_z": float(np.max(np.abs(zarr))) if N else None,
               "n_beyond_t": len(beyond), "n_beyond_3p5": int(np.sum(np.abs(zarr) > 3.5)) if N else 0,
               "beyond": [{"bin": b[1], "level": b[2], "coverage": b[3], "sd": b[4], "z": b[0]}
                          for b in sorted(beyond, key=lambda x: -abs(x[0]))[:30]],
               "chi2": chi_res, "chi2_threshold_p": 0.01 / max(n_chi, 1), "chi2_fail": sorted(chi_fail),
               "n_a": sum(k["ii_na"] for k in per_key.values()),
               "per_level_max_abs_z": {f"{lv:g}": float(np.max(np.abs([z[0] for z in zs if z[2] == lv])))
                                       for lv in LEVELS if any(z[2] == lv for z in zs)}},
        "lookup_missing": {k: P["keys"][k]["n_lookup_missing"] for k in P["keys"]},
    }
    res["pass_i"] = n_i_fail == 0
    res["pass_ii"] = (len(beyond) == 0) and not chi_fail and N > 0
    res["pass"] = res["pass_i"] and res["pass_ii"]
    res["per_region"] = region_summary(P, stack, min_n)
    return res


def region_summary(P, stack, min_n):
    """Pass/fail and the worst |z| per (key, barrel / disc), as the demonstration must show both."""
    out = {}
    for key, S in stack.items():
        nom, sd = S["st"][0], np.nanstd(S["st"][1:], axis=0, ddof=1)
        for region, surfs in (("barrel", SA.BARREL), ("discs", SA.DISCS)):
            idx = [i for i, tr in enumerate(P["keys"][key]["trows"]) if tr["surface"] in surfs
                   and P["keys"][key]["test_n"][i] >= min_n]
            if not idx:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (nom[idx][:, :, 3:] - np.array(LEVELS)) / sd[idx][:, :, 3:]
            z = np.where(sd[idx][:, :, 3:] > 0, z, np.nan)
            band = np.maximum(np.abs(nom[idx][:, :, 0]) / (BAND_C + WIDEN * sd[idx][:, :, 0]),
                              np.abs(nom[idx][:, :, 1] - 1) / (BAND_W + WIDEN * sd[idx][:, :, 1]))
            out[f"{key}/{region}"] = {"bins": len(idx), "max_abs_z": float(np.nanmax(np.abs(z))),
                                      "worst_band_fraction": float(np.nanmax(band))}
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
    """Per canonical row, angle and key: centre and width at pT 1-2 vs >= 2 GeV with the paired
    event bootstrap of the fit sample; the D.0 agreement rule (no |z| beyond t(N) over every
    z-score of both keys; chi2 per key reported). Rows failing in both signs (the row and its
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


def before_after(P: dict, stack: dict, test: dict, scales: dict) -> dict:
    """Per test bin (nominal): pulls before (published sigma; and today's census sigma on the
    barrel: per-layer scale, no centre) and after the table; today's 3-census-sigma class-T
    efficiency; the table's half-width at each level in today's census sigma."""
    out = {}
    for key, K in P["keys"].items():
        trow = row_of("canonical", K["trowmap"], test, key)
        nom = stack[key]["st"][0]
        sd = np.nanstd(stack[key]["st"][1:], axis=0, ddof=1)
        par = stack[key]["par"][0]
        psd = np.nanstd(stack[key]["par"][1:], axis=0, ddof=1)
        rows = []
        for i, tr in enumerate(K["trows"]):
            m = trow == i
            ent = {"surface": SA.SURFACE_NAMES[tr["surface"]], "facing": tr["facing"], "lo": tr["lo"], "hi": tr["hi"],
                   "n_fit": tr["n_fit"], "n_test": int(m.sum())}
            for j, a in enumerate(SA.ANGLES):
                p = test["pull"][a][m]
                b = SA.core_w(p) if m.sum() >= SA.MIN_FIT else None
                d = {"before": None if b is None else {"centre": b["mean"], "width": b["rms0"], "out3": b["out3"]},
                     "after": {"centre": nom[i, j, 0], "sd_centre": sd[i, j, 0], "width": nom[i, j, 1],
                               "sd_width": sd[i, j, 1]},
                     "coverage": {f"{lv:g}": {"cov": nom[i, j, 3 + li], "sd": sd[i, j, 3 + li]}
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
            rows.append(ent)
        keys_rows[key] = rows
    return SA.new_table_doc(keys_rows, orient, prov, era="canon_v3 (150X D121 PU200 TTbar, AAAA)",
                            fit_sample=f"canon_v3 AAAA files {list(SA.FIT_FILES)}, class T, pT >= {SA.PT_MIN:g} GeV "
                                       "(the class-C angles of these files are cluster-CNN training data; class T is "
                                       "truth + 1.5 x a PixelAV residual and is not)")


# ---------------------------------------------------------------- driver
def run(mode: str, n_rep: int, n_proc: int, out_dir: Path) -> dict:
    fit_files, test_files = SPLITS[mode]
    t0 = time.time()
    orient = CM.orientation_map(fit_files)
    fit = CM.derive(load_split(fit_files), orient)
    test = CM.derive(load_split(test_files), orient)
    print(f"[{mode}] fit files {fit_files}: {len(fit['ev'])} clusters; test files {test_files}: "
          f"{len(test['ev'])} clusters; orientation of {len(orient[0])} modules ({time.time() - t0:.0f} s)", flush=True)
    res = {"mode": mode, "fit_files": list(fit_files), "test_files": list(test_files), "n_rep": n_rep,
           "n_fit": int(len(fit["ev"])), "n_test": int(len(test["ev"])), "code": CM.git_state(),
           "inputs": CM.inputs_provenance(sorted(set(fit_files) | set(test_files))),
           "rule": {"band_width": BAND_W, "band_centre": BAND_C, "widen_sigma": WIDEN, "levels": LEVELS,
                    "min_n": SA.MIN_FIT, "N_definition": "levels x test bins x angles x keys, after merges",
                    "chi2": "per level and key over bins and angles; p > 0.01 / (levels x keys)"}}
    if mode in ("known-good", "gated", "fit-only"):
        ptc, flags = pt_check(fit, n_rep, n_proc)
        res["pt_check"] = ptc
        print(f"[{mode}] pT check: refit at pT >= 2 {ptc['flags']}", flush=True)
        P = prepare(fit, test, {k: {"kind": "canonical", "lookup_key": k} for k in SA.KEYS}, flags)
        stack = run_chain(P, n_rep, n_proc, mode)
        if mode != "fit-only":
            res["judgement"] = judge(P, stack)
            res["bins"] = before_after(P, stack, test, census_scales())
        if mode in ("gated", "fit-only"):
            prov = {"code": res["code"], "inputs": res["inputs"], "fit_files": list(fit_files),
                    "job_id": os.environ.get("SP_JOB_ID"), "n_rep_bootstrap": n_rep,
                    "pt_check_refit_rows": ptc["flags"],
                    "test": "files 3-10; see p03_table_gated.json" if mode == "gated" else
                    "NOT TESTED: the gated test did not run (its known-good demonstration or a known-bad did not "
                    "behave as required; see p03_table_known-good.json, p03_table_known-bad.json)"}
            doc = table_doc(P, stack, orient, flags, prov)
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
            if name == "wrong_key":
                spec = {"true": {"kind": "canonical", "lookup_key": "emitted"},
                        "emitted": {"kind": "canonical", "lookup_key": "true"}}
            else:
                spec = {k: {"kind": name, "lookup_key": k} for k in SA.KEYS}
            P = prepare(fit, test, spec)
            stack = run_chain(P, n_rep, n_proc, f"known-bad {name}")
            j = judge(P, stack)
            res["known_bads"][name] = {"description": desc, "judgement": j, "fails_as_required": not j["pass"]}
            print(f"[known-bad {name}] pass={j['pass']} (i fail {j['i']['fail']}/{j['i']['bins']}, ii beyond t "
                  f"{j['ii']['n_beyond_t']}, max|z| {j['ii']['max_abs_z']:.1f}, chi2 fail {j['ii']['chi2_fail']})",
                  flush=True)
        res["all_known_bads_fail"] = all(v["fails_as_required"] for v in res["known_bads"].values())
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("known-good", "known-bad", "gated", "fit-only"))
    ap.add_argument("--rep", type=int, default=250)
    ap.add_argument("--proc", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(CM.OUT_DIR))
    ap.add_argument("--force", action="store_true", help="run the gated test without the prerequisite results")
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.mode == "gated" and not a.force:
        try:
            kg = json.load(open(out_dir / "p03_table_known-good.json"))
            kb = json.load(open(out_dir / "p03_table_known-bad.json"))
        except FileNotFoundError as e:
            print(f"refused: the gated test needs the known-good and known-bad results first ({e})")
            return 66
        if not kg["judgement"]["pass"] or not kb["all_known_bads_fail"] or kg["n_rep"] < 250 or kb["n_rep"] < 250:
            print("refused: the rule's known-good demonstration did not pass, or a known-bad did not fail, or < 250 "
                  "replicas: the rule is redesigned before use (D.0 step 4)")
            return 66
    res = run(a.mode, a.rep, a.proc, out_dir)
    CM.jdump(res, out_dir / f"p03_table_{a.mode}.json")
    if a.mode == "known-bad":
        print("every known-bad fails" if res["all_known_bads_fail"] else "SOME KNOWN-BAD PASSES")
        return 0 if res["all_known_bads_fail"] else 3
    if a.mode == "fit-only":
        print(f"wrote the UNTESTED table {res['table_file']}")
        return 0
    j = res["judgement"]
    print(f"[{a.mode}] PASS={j['pass']}: (i) {j['i']['fail']} of {j['i']['bins']} bins outside the widened band "
          f"(worst {j['i']['worst_fraction_of_widened_band']}); (ii) N={j['ii']['N']} t(N)={j['ii']['t_N']:.2f} "
          f"max|z|={j['ii']['max_abs_z']:.2f}, beyond t: {j['ii']['n_beyond_t']}, chi2 fail: {j['ii']['chi2_fail']}")
    return 0 if j["pass"] else 4


if __name__ == "__main__":
    sys.exit(main())
