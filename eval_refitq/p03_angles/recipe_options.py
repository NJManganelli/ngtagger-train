#!/usr/bin/env python
"""P0.3 amendment 1b, item B9: two recipe options MEASURED for the physicist, not adopted.

(a) Quantile rank k' = ceil(L (n_fit + 1)) instead of k = ceil(L n_fit): E[coverage] = k' / (n_fit + 1) >= L
    (the sample maximum when k' > n_fit). Measured per key x region x angle x level: the cluster-weighted
    change of the window half-width q (|pull - centre| quantile), the expected coverage before / after, and
    the pooled test coverage before / after.
(b) Disc beta: the signed equal-tailed quantiles [Q((1-L)/2), Q((1+L)/2)] of the pull with the median as
    location, instead of |pull - centre| <= q about the B.1 clipped mean (bistable in five true-key rows).
    Measured per disc-beta row: the window width before / after, its spread over the four 200-event file-pair
    fits (3-4, 5-6, 7-8, 9-10), the test coverage of the (b) window; the two-component fit (unbinned two-
    Gaussian ML: narrow fraction, centres, widths) per file pair as a later full-response input; and the
    cluster retention of the single-cluster disc beta cut (roc.py's z0-window statistic) at 99 % signal
    efficiency with the median instead of the clipped mean as the subtracted centre.
Data: the amendment-1b demonstration split only (fit files 7-8, test 3-6 and 9-10; the four file pairs of
3-10 for the stability spread). Files 1-2 are not read.

    python recipe_options.py [--table $P03_OUT/amend1b/angle_table_known-good_fit_a1b.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import table as T  # noqa: E402

from ngtagger.spdefs import angles as SA  # noqa: E402

FIT, TEST = (7, 8), (3, 4, 5, 6, 9, 10)
PAIRS = ((3, 4), (5, 6), (7, 8), (9, 10))
LV = np.asarray(SA.LEVELS)


def rows_and_index(fit, C, key):
    rows, rowmap = T.build_rows("canonical", fit, key)
    return rows, T.row_of("canonical", rowmap, C, key)


def option_a(fit, test):
    """Window half-width and coverage with k vs k' per key x region x angle x level."""
    out = {}
    for key in SA.KEYS:
        rows, rf = rows_and_index(fit, fit, key)
        _, rt = rows_and_index(fit, test, key)
        acc = {}
        for i, r in enumerate(rows):
            reg = "barrel" if r["surface"] in SA.BARREL else "discs"
            mf, mt = rf == i, rt == i
            n = int(mf.sum())
            if n < SA.MIN_FIT:
                continue
            for a in SA.ANGLES:
                e = SA.bin_estimates(fit["pull"][a][mf])
                d = np.sort(np.abs(fit["pull"][a][mf] - e["centre"]))
                dt = np.abs(test["pull"][a][mt] - e["centre"])
                for li, lv in enumerate(LV):
                    k = int(SA.quantile_order(n, lv))
                    k2 = int(min(np.ceil(lv * (n + 1) - 1e-9), n))
                    q, q2 = d[k - 1], d[k2 - 1]
                    c = acc.setdefault((reg, a, lv), {"w": 0.0, "dq": 0.0, "e0": 0.0, "e02": 0.0, "cov": 0.0, "cov2": 0.0,
                                                      "nt": 0.0, "rows_changed": 0, "rows": 0, "rows_at_max": 0})
                    w = float(mt.sum())
                    c["w"] += w
                    c["dq"] += w * (q2 / q - 1.0)
                    c["e0"] += w * k / (n + 1)
                    c["e02"] += w * k2 / (n + 1)
                    c["cov"] += float(np.sum(dt <= q))
                    c["cov2"] += float(np.sum(dt <= q2))
                    c["nt"] += w
                    c["rows"] += 1
                    c["rows_changed"] += int(k2 != k)
                    c["rows_at_max"] += int(k2 == n and np.ceil(lv * (n + 1) - 1e-9) > n)
        out[key] = {f"{reg}/{a}/{lv:g}": {"window_change_pct": 100 * c["dq"] / c["w"], "E0_mean": c["e0"] / c["w"],
                                         "E0_mean_option_a": c["e02"] / c["w"], "test_coverage": c["cov"] / c["nt"],
                                         "test_coverage_option_a": c["cov2"] / c["nt"], "rows": c["rows"],
                                         "rows_changed": c["rows_changed"], "rows_where_level_unreachable": c["rows_at_max"]}
                    for (reg, a, lv), c in acc.items()}
    return out


def eq_tailed(p, lv):
    s = np.sort(p)
    n = len(s)
    lo = s[max(int(np.ceil((1 - lv) / 2 * n - 1e-9)) - 1, 0)]
    hi = s[min(int(np.ceil((1 + lv) / 2 * n - 1e-9)) - 1, n - 1)]
    return lo, hi


def option_b(pairs_data, fit, test):
    """Disc beta rows (rows built on the demonstration fit): window widths, stability over the four
    file-pair fits, test coverage of the (b) window, two-component fits."""
    out = {}
    for key in SA.KEYS:
        rows, rf = rows_and_index(fit, fit, key)
        _, rt = rows_and_index(fit, test, key)
        rp = {pr: rows_and_index(fit, C, key)[1] for pr, C in pairs_data.items()}
        res_rows = []
        for i, r in enumerate(rows):
            if r["surface"] not in SA.DISCS:
                continue
            mf, mt = rf == i, rt == i
            if mf.sum() < SA.MIN_FIT:
                continue
            p, pt_ = fit["pull"]["beta"][mf], test["pull"]["beta"][mt]
            e = SA.bin_estimates(p)
            ent = {"row": T._tag(key, "beta", r), "n_fit": int(mf.sum()), "n_test": int(mt.sum()),
                   "clip_centre": e["centre"], "median": float(np.median(p)), "levels": {}}
            for li, lv in enumerate(LV):
                q = e["q"][f"{lv:g}"]
                lo, hi = eq_tailed(p, lv)
                ent["levels"][f"{lv:g}"] = {
                    "width_current": 2 * q, "width_option_b": float(hi - lo), "window_option_b": [float(lo), float(hi)],
                    "test_cov_current": float(np.mean(np.abs(pt_ - e["centre"]) <= q)),
                    "test_cov_option_b": float(np.mean((pt_ >= lo) & (pt_ <= hi)))}
            # stability over the four 200-event file-pair fits
            wc, wb, cc, med, comp = [], [], [], [], []
            for pr, C in pairs_data.items():
                m = rp[pr] == i
                if m.sum() < SA.MIN_FIT:
                    continue
                pp = C["pull"]["beta"][m]
                ee = SA.bin_estimates(pp)
                wc.append(2 * ee["q"]["0.99"])
                lo, hi = eq_tailed(pp, 0.99)
                wb.append(float(hi - lo))
                cc.append(ee["centre"])
                med.append(float(np.median(pp)))
                g = T.em_two_gaussians(np.clip(pp, -50, 50))
                if "weights" in g:
                    o = np.argsort(g["sds"])
                    comp.append({"pair": list(pr), "narrow_fraction": g["weights"][o[0]], "narrow_centre": g["means"][o[0]],
                                 "narrow_width": g["sds"][o[0]], "broad_centre": g["means"][o[1]],
                                 "broad_width": g["sds"][o[1]]})
            ent["stability_99"] = {"width_current_min_max": [min(wc), max(wc)] if wc else None,
                                   "width_option_b_min_max": [min(wb), max(wb)] if wb else None,
                                   "clip_centre_min_max": [min(cc), max(cc)] if cc else None,
                                   "median_min_max": [min(med), max(med)] if med else None}
            ent["two_component_fits"] = comp
            res_rows.append(ent)
        out[key] = res_rows
    return out


def summarise_b(rows):
    s = {}
    for key, rr in rows.items():
        for lv in SA.LEVELS:
            k = f"{lv:g}"
            wcur = np.array([x["levels"][k]["width_current"] for x in rr])
            wb = np.array([x["levels"][k]["width_option_b"] for x in rr])
            nt = np.array([x["n_test"] for x in rr], float)
            s[f"{key}/{k}"] = {"rows": len(rr), "width_ratio_b_over_current_weighted": float(np.sum(nt * wb / wcur) / nt.sum()),
                               "width_ratio_pct_10_50_90": np.percentile(wb / wcur, [10, 50, 90]).round(3).tolist(),
                               "test_cov_current": float(np.sum(nt * [x["levels"][k]["test_cov_current"] for x in rr]) / nt.sum()),
                               "test_cov_option_b": float(np.sum(nt * [x["levels"][k]["test_cov_option_b"] for x in rr]) / nt.sum())}
        spread_c = [x["stability_99"]["width_current_min_max"] for x in rr if x["stability_99"]["width_current_min_max"]]
        spread_b = [x["stability_99"]["width_option_b_min_max"] for x in rr if x["stability_99"]["width_option_b_min_max"]]
        rc = np.array([b / a for a, b in spread_c])
        rb = np.array([b / a for a, b in spread_b])
        s[f"{key}/stability_99"] = {"max_over_min_width_current_pct_50_90_max": np.percentile(rc, [50, 90, 100]).round(3).tolist(),
                                    "max_over_min_width_option_b_pct_50_90_max": np.percentile(rb, [50, 90, 100]).round(3).tolist()}
    return s


def retention(table_path: str, fit_rows_median: dict):
    """Disc retention at 99 % signal efficiency of roc.py's beta and both cuts (floor 2 GeV, k = 1, fixed),
    calibration on the demonstration fit files, test on its test files; centres: the table's (clipped mean)
    and the medians (option b) in the disc beta rows of the emitted key."""
    import copy

    import roc as R

    out = {}
    base = SA.AngleTable.load(table_path)
    doc = copy.deepcopy(base.doc)
    for i, r in enumerate(doc["keys"]["emitted"]["rows"]):
        if r["surface"] in SA.DISCS and r.get("beta") and i in fit_rows_median:
            r["beta"]["centre"] = fit_rows_median[i]
    tabs = {"current_clipped_mean": base, "option_b_median": SA.AngleTable(doc)}
    cal0 = CM.load(FIT, R.ROC_COLS, mask_fn=CM.table_population)
    test0 = CM.load(TEST, R.ROC_COLS)
    u_test, ev_test = np.unique(test0["ev"], return_inverse=True)
    rng = np.random.default_rng([20261014, 99])
    reps = rng.multinomial(len(u_test), np.full(len(u_test), 1.0 / len(u_test)), size=100).astype(np.float64)
    for name, tab in tabs.items():
        cal = R.prep({k: v.copy() for k, v in cal0.items()}, tab)
        test = R.prep({k: v.copy() for k, v in test0.items()}, tab)
        n_rows = len(tab.doc["keys"]["emitted"]["rows"])
        floor = 2.0
        cat = R.categories(test, floor)
        idx_e = (ev_test.astype(np.int64) * len(SA.SURFACES) + test["sidx"]) * R.N_CAT + cat
        cal_pop = cal["T"] & (cal["born"] <= R.BORN_CM) & (cal["tp_pt"] >= floor)
        csa, csb, crow = R.statistics(cal, tab, floor, 1.0, "fixed")
        sa, sb, row = R.statistics(test, tab, floor, 1.0, "fixed")
        Ta = R.thresholds(csa, crow, cal_pop, n_rows)
        Tb = R.thresholds(csb, crow, cal_pop, n_rows)
        for cut in ("beta", "both"):
            pc = R.pass_counts(sa, sb, row, Ta, Tb, cut, test["w"].astype(np.float64), idx_e,
                               len(u_test) * len(SA.SURFACES) * R.N_CAT)
            pe = pc.reshape(R.N_THR, len(u_test), len(SA.SURFACES), R.N_CAT)
            S0 = R.summarise(pe.sum(1))
            SB = R.summarise(np.einsum("re,tesc->rtsc", reps, pe, optimize=True))
            for g in ("discs", "barrel"):
                for sig in ("eff_sig_all", "eff_sig_T"):
                    out[f"{name}/{cut}/{g}/{sig}"] = {
                        "retention_at_99": float(R.at_eff(S0[g][sig], S0[g]["retention"])[0]),
                        "sd_test": float(np.nanstd(R.at_eff(SB[g][sig], SB[g]["retention"]), ddof=1))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(CM.OUT_DIR / "amend1b" / "angle_table_known-good_fit_a1b.json"))
    ap.add_argument("--out", default=str(CM.OUT_DIR / "amend1b" / "recipe_options.json"))
    a = ap.parse_args(argv)
    t0 = time.time()
    orient = CM.orientation_map(FIT)
    fit = CM.derive(T.load_split(FIT), orient)
    test = CM.derive(T.load_split(TEST), orient)
    pairs = {pr: (fit if pr == FIT else CM.derive(T.load_split(pr), orient)) for pr in PAIRS}
    print(f"loaded ({time.time() - t0:.0f} s)", flush=True)
    res = {"definition": __doc__, "fit_files": FIT, "test_files": TEST, "pairs": PAIRS}
    res["option_a"] = option_a(fit, test)
    print(f"option (a) done ({time.time() - t0:.0f} s)", flush=True)
    rows_b = option_b(pairs, fit, test)
    res["option_b_rows"] = rows_b
    res["option_b_summary"] = summarise_b(rows_b)
    print(f"option (b) windows done ({time.time() - t0:.0f} s)", flush=True)
    rows, rf = rows_and_index(fit, fit, "emitted")
    med = {i: float(np.median(fit["pull"]["beta"][rf == i])) for i, r in enumerate(rows)
           if r["surface"] in SA.DISCS and (rf == i).sum() >= SA.MIN_FIT}
    res["option_b_retention"] = retention(a.table, med)
    print(f"retention done ({time.time() - t0:.0f} s)", flush=True)
    CM.jdump(res, Path(a.out))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
