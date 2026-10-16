#!/usr/bin/env python
"""P0.3 step 3: the cluster-level angle ROC (plan D.3; decisions 2, 10).

Single-cluster cuts, as a cutting consumer without a prediction (emitted-key table):
  S1   alpha vs the pT floor: (|alpha' - k c sigma_pub| - asin(KPT r / 2 floor)) / sigma'
  beta vs the luminous region: (distance of lambda' - k c sigma_pub from the directions to |z0| <= 15 cm) / sigma'
  both (each at its own per-bin quantile).
Each cut is placed at the per-bin quantile of its OWN statistic on class-T calibration clusters
(TPs born inside 2.5 cm, pT >= floor, every interaction; files 1-2), per surface x signed
emitted-key bin (x facing) = the table's emitted-key rows (angle.cut_level). The curve scans the
level (0.5 ... 1) and then loosens beyond the calibration maximum; the four table levels are
points on it. k in {1, 0.67, 0} x {fixed, scaled} through ngtagger.spdefs (the quantile kept at
k = 1 in fixed mode, re-measured at k in scaled mode). Measured on files 3-10; retention over all
clusters (the 1-in-20 subsample weighted by 20).

Errors: event bootstrap (250). At k = 1 the whole chain (calibration re-measured per replica);
at k < 1 the test side only (calibration fixed; stated).

    python roc.py [--table .../angle_table_v1_<era>.json] [--rep 250] [--proc 10]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

from ngtagger.spdefs import angles as SA  # noqa: E402

SEED = 20261010
FLOORS = (2.0, 1.5, 1.0)
SETTINGS = ((1.0, "fixed"), (0.67, "fixed"), (0.67, "scaled"), (0.0, "fixed"), (0.0, "scaled"))
CUTS = ("S1", "beta", "both")
L_GRID = (0.5, 0.8, 0.9, 0.95, 0.97, 0.975, 0.98, 0.985, 0.99, 0.9925, 0.995, 0.997, 0.998, 0.999, 0.9995, 1.0)
DELTAS = (0.5, 1, 2, 4, 8, 16, 64, np.inf)
N_THR = len(L_GRID) + len(DELTAS)
BORN_CM = 2.5
CATS = ("sig_prompt", "sig_hf", "pu_siglike", "pu_other", "same_lt0p5", "same_0p5_floor", "same_secondary",
        "unlinked")
CAT_DESC = {"sig_prompt": "hard scatter, pT >= floor, born inside 2.5 cm, not b/c daughter",
            "sig_hf": "hard scatter, pT >= floor, born inside 2.5 cm, b or c daughter",
            "pu_siglike": "other interaction (pile-up / out of time), pT >= floor, born inside 2.5 cm",
            "pu_other": "other interaction, pT < floor or born beyond 2.5 cm",
            "same_lt0p5": "same (hard-scatter) interaction, pT < 0.5 GeV",
            "same_0p5_floor": "same interaction, 0.5 GeV <= pT < floor (sub-threshold, signal-like)",
            "same_secondary": "same interaction, pT >= floor, born beyond 2.5 cm",
            "unlinked": "no TP (tpIdx < 0)"}
N_CAT = 2 * len(CATS)                      # x class (C = 0, T = 1)
ETA_EDGES = (0.0, 0.9, 1.4, 2.0, 2.6, 4.5)
PLATEAU_DEG = (27.0, 34.0)
ROC_COLS = ("surface", "detId", "ev", "w", "tpEmuClass", "tpIdx", "hasAlpha", "hasBeta", "globalR", "globalZ",
            "globalPhi", "globalClusterPhi", "globalClusterCotTheta", "sigGlobalClusterPhi",
            "sigGlobalClusterCotTheta", "tpGlobalClusterPhi", "tpGlobalClusterCotTheta", "localCotAlpha",
            "localCotBeta", "tp_pt", "tp_hs", "tp_hf", "tp_vx", "tp_vy")
G: dict = {}


# ---------------------------------------------------------------- per-cluster quantities
def prep(C: dict, tab: SA.AngleTable) -> dict:
    C["ok"] = SA.usable(C)
    C["T"] = SA.class_T(C["tpEmuClass"]) & C["ok"]
    C["linked"] = C["tpIdx"] >= 0
    C["facing"] = SA.facing(C["surface"], tab.orientation(C["detId"]))
    C["born"] = np.where(C["linked"], np.hypot(C["tp_vx"], C["tp_vy"]), np.inf)
    C["sig"] = {a: SA.sigma_of(a, C) for a in SA.ANGLES}
    C["em"] = {a: SA.emitted(a, C) for a in SA.ANGLES}
    C["tr"] = {a: np.where(C["linked"], SA.truth(a, C), np.nan) for a in SA.ANGLES}
    C["r"] = np.asarray(C["globalR"], np.float64)
    C["z"] = np.asarray(C["globalZ"], np.float64)
    C["sidx"] = np.searchsorted(SA.SURFACES, C["surface"]).astype(np.int8)
    return C


def categories(C: dict, floor: float) -> np.ndarray:
    L, hs, pt = C["linked"], C["tp_hs"] == 1, C["tp_pt"]
    inside = C["born"] <= BORN_CM
    hf = C["tp_hf"] > 0
    cat = np.full(len(pt), CATS.index("unlinked"), np.int8)
    cat[L & ~hs & (pt >= floor) & inside] = CATS.index("pu_siglike")
    cat[L & ~hs & ~((pt >= floor) & inside)] = CATS.index("pu_other")
    cat[L & hs & (pt < 0.5)] = CATS.index("same_lt0p5")
    cat[L & hs & (pt >= 0.5) & (pt < floor)] = CATS.index("same_0p5_floor")
    cat[L & hs & (pt >= floor) & ~inside] = CATS.index("same_secondary")
    cat[L & hs & (pt >= floor) & inside & ~hf] = CATS.index("sig_prompt")
    cat[L & hs & (pt >= floor) & inside & hf] = CATS.index("sig_hf")
    return (2 * cat + C["T"].astype(np.int8)).astype(np.int8)


def statistics(C: dict, tab: SA.AngleTable, floor: float, k: float, mode: str):
    """(s_alpha, s_beta, row): the consumers' statistics at (k, mode) and the emitted-key table row."""
    ka = {a: SA.knob(a, C, C["T"], k, mode, a=C["em"][a], ref=C["tr"][a] if k < 1 else None, sigma=C["sig"][a])
          for a in SA.ANGLES}
    cot = np.tan(ka["beta"]["a"])
    row = tab.rows_of("emitted", C["surface"], SA.incidence_key(C["surface"], cot), C["facing"])
    cen = {a: SA.centre_k(np.nan_to_num(tab.get("emitted", a, "centre", row)) * C["sig"][a], k) for a in SA.ANGLES}
    with np.errstate(invalid="ignore", divide="ignore"):
        sa = SA.s1_statistic(ka["alpha"]["a"], ka["alpha"]["sigma"], cen["alpha"], C["r"], floor)
        sb = SA.z0_window_statistic(ka["beta"]["a"], ka["beta"]["sigma"], cen["beta"], C["r"], C["z"])
    bad = ~C["ok"] | (row < 0)
    sa[bad | ~np.isfinite(sa)] = np.inf
    sb[bad | ~np.isfinite(sb)] = np.inf
    return sa, sb, row


def thresholds(s, row, pop, n_rows, w=None):
    """[n_rows + 1, N_THR]: per-row calibration quantiles on the level grid, then the
    calibration maximum loosened by DELTAS (consumer sigma); the last row (no row) passes all."""
    q = SA.consumer_quantile(s, row, L_GRID, pop & np.isfinite(s), w, n_rows)
    out = np.empty((n_rows + 1, N_THR))
    out[:n_rows, :len(L_GRID)] = q
    out[:n_rows, len(L_GRID):] = q[:, -1:] + np.array(DELTAS)[None, :]
    out[np.isnan(out)] = np.inf                      # rows without calibration clusters: no cut
    out[n_rows] = np.inf
    return out


def pass_counts(sa, sb, row, Ta, Tb, cut, w, idx, n_idx):
    """Weighted counts [N_THR, n_idx] of the clusters passing each threshold."""
    rr = np.where(row >= 0, row, Ta.shape[0] - 1)
    out = np.empty((N_THR, n_idx))
    for i in range(N_THR):
        if cut == "S1":
            p = sa <= Ta[rr, i]
        elif cut == "beta":
            p = sb <= Tb[rr, i]
        else:
            p = (sa <= Ta[rr, i]) & (sb <= Tb[rr, i])
        out[i] = np.bincount(idx[p], weights=w[p], minlength=n_idx)
    return out


# ---------------------------------------------------------------- full chain at k = 1 (workers)
def _chain_replica(b: int):
    S = G["S"]
    if b == 0:
        cfc = None
        wt = S["w_test"]
    else:
        rng = np.random.default_rng([SEED, b])
        cfc = rng.multinomial(S["n_ev_cal"], np.full(S["n_ev_cal"], 1.0 / S["n_ev_cal"]))[S["ev_cal"]]
        ct = rng.multinomial(S["n_ev_test"], np.full(S["n_ev_test"], 1.0 / S["n_ev_test"]))
        wt = S["w_test"] * ct[S["ev_test"]]
    res = {}
    for key, X in S["combos"].items():
        Ta = thresholds(X["cal_sa"], X["cal_row"], X["cal_pop"], S["n_rows"], cfc)
        Tb = thresholds(X["cal_sb"], X["cal_row"], X["cal_pop"], S["n_rows"], cfc)
        res[key] = pass_counts(X["sa"], X["sb"], X["row"], Ta, Tb, X["cut"], wt, X["idx"], S["n_idx"])
    return b, res


def summarise(cnt: np.ndarray) -> dict:
    """cnt [..., N_THR, n_surf, N_CAT] (weighted pass counts; the last threshold passes all)."""
    tot = cnt[..., -1:, :, :]
    out = {}
    sig = [2 * CATS.index("sig_prompt") + c for c in (0, 1)] + [2 * CATS.index("sig_hf") + c for c in (0, 1)]
    sigT = [2 * CATS.index("sig_prompt") + 1, 2 * CATS.index("sig_hf") + 1]
    allint = sig + [2 * CATS.index("pu_siglike"), 2 * CATS.index("pu_siglike") + 1]
    groups = {"barrel": [SA.SURFACES.index(s) for s in SA.BARREL], "discs": [SA.SURFACES.index(s) for s in SA.DISCS]}
    groups.update({SA.SURFACE_NAMES[s]: [i] for i, s in enumerate(SA.SURFACES)})
    for g, si in groups.items():
        c = cnt[..., si, :].sum(-2)
        t = tot[..., si, :].sum(-2)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[g] = {"eff_sig_all": c[..., sig].sum(-1) / t[..., sig].sum(-1),
                      "eff_sig_T": c[..., sigT].sum(-1) / t[..., sigT].sum(-1),
                      "eff_sig_prompt": c[..., sig[:2]].sum(-1) / t[..., sig[:2]].sum(-1),
                      "eff_sig_hf": c[..., sig[2:]].sum(-1) / t[..., sig[2:]].sum(-1),
                      "eff_allint": c[..., allint].sum(-1) / t[..., allint].sum(-1),
                      "retention": c.sum(-1) / t.sum(-1),
                      "cat_retention": {cn: c[..., [2 * i, 2 * i + 1]].sum(-1) / t[..., [2 * i, 2 * i + 1]].sum(-1)
                                        for i, cn in enumerate(CATS)},
                      "bg_rejected_classC_share": _classc_share(c, t)}
    return out


def _classc_share(c, t):
    bg = [i for i, cn in enumerate(CATS) if not cn.startswith("sig")]
    rej_c = sum(t[..., 2 * i] - c[..., 2 * i] for i in bg)
    rej = sum(t[..., 2 * i] + t[..., 2 * i + 1] - c[..., 2 * i] - c[..., 2 * i + 1] for i in bg)
    with np.errstate(invalid="ignore", divide="ignore"):
        return rej_c / rej


def at_eff(eff: np.ndarray, ret: np.ndarray, target: float = 0.99) -> np.ndarray:
    """Retention at signal efficiency `target`, linear between the scan points ([..., N_THR])."""
    eff = np.atleast_2d(eff)
    ret = np.atleast_2d(ret)
    out = np.full(eff.shape[0], np.nan)
    for i in range(eff.shape[0]):
        e, r = eff[i], ret[i]
        j = np.where(e >= target)[0]
        if len(j) == 0:
            continue
        j = j[0]
        if j == 0:
            out[i] = r[0]
        else:
            f = (target - e[j - 1]) / (e[j] - e[j - 1]) if e[j] > e[j - 1] else 1.0
            out[i] = r[j - 1] + f * (r[j] - r[j - 1])
    return out


def to_lists(x):
    if isinstance(x, dict):
        return {k: to_lists(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ---------------------------------------------------------------- driver
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(CM.OUT_DIR / CM.table_name("angle_table_v1")))
    ap.add_argument("--rep", type=int, default=250)
    ap.add_argument("--proc", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--cal-files", type=int, nargs="*", default=list(SA.FIT_FILES))
    ap.add_argument("--test-files", type=int, nargs="*", default=list(SA.TEST_FILES))
    ap.add_argument("--out", default=str(CM.OUT_DIR / "p03_roc.json"))
    a = ap.parse_args(argv)
    t0 = time.time()
    tab = SA.AngleTable.load(a.table)
    n_rows = len(tab.doc["keys"]["emitted"]["rows"])
    cal = prep(CM.load(a.cal_files, ROC_COLS, mask_fn=CM.table_population), tab)
    test = prep(CM.load(a.test_files, ROC_COLS), tab)
    u_cal, ev_cal = np.unique(cal["ev"], return_inverse=True)
    u_test, ev_test = np.unique(test["ev"], return_inverse=True)
    w_test = test["w"].astype(np.float64)
    print(f"calibration {len(cal['ev'])} class-T clusters ({len(u_cal)} events); test {len(test['ev'])} clusters "
          f"({w_test.sum():.0f} weighted, {len(u_test)} events); {time.time() - t0:.0f} s", flush=True)
    # eta x plateau bins of the test clusters (observables: position eta from the origin, emitted incidence)
    eta = np.abs(np.arcsinh(test["z"] / test["r"]))
    ebin = np.clip(np.searchsorted(ETA_EDGES, eta, "right") - 1, 0, len(ETA_EDGES) - 2)
    inc = np.degrees(np.arctan(np.abs(SA.incidence_key(test["surface"], test["globalClusterCotTheta"]))))
    pbin = np.searchsorted(PLATEAU_DEG, inc, "right")
    n_ev = len(u_test)
    res = {"definitions": {"floors": FLOORS, "settings": SETTINGS, "cuts": CUTS, "level_grid": L_GRID,
                           "deltas_beyond_max": [float(x) for x in DELTAS], "categories": CAT_DESC,
                           "eta_edges": ETA_EDGES, "plateau_deg": PLATEAU_DEG, "born_cm": BORN_CM,
                           "z_lumi_cm": SA.Z_LUMI_CM, "table": a.table,
                           "table_provenance": tab.provenance, "cal_files": a.cal_files, "test_files": a.test_files,
                           "subsample_weight": int(test["w"].max()), "code": CM.git_state(), "n_rep": a.rep},
           "combos": {}}
    chain = {"n_rows": n_rows, "n_ev_cal": len(u_cal), "ev_cal": ev_cal, "n_ev_test": n_ev, "ev_test": ev_test,
             "w_test": w_test, "n_idx": len(SA.SURFACES) * N_CAT, "combos": {}}
    rng = np.random.default_rng([SEED, 0])
    reps = rng.multinomial(n_ev, np.full(n_ev, 1.0 / n_ev), size=a.rep).astype(np.float64)
    for floor in FLOORS:
        cat = categories(test, floor)
        idx_s = (test["sidx"].astype(np.int64) * N_CAT + cat).astype(np.int64)
        idx_e = (ev_test.astype(np.int64) * len(SA.SURFACES) + test["sidx"]) * N_CAT + cat
        cal_pop = cal["T"] & (cal["born"] <= BORN_CM) & (cal["tp_pt"] >= floor)
        for k, mode in SETTINGS:
            kc = SA.calibration_k(k, mode)
            csa, csb, crow = statistics(cal, tab, floor, kc, mode)
            sa, sb, row = statistics(test, tab, floor, k, mode)
            Ta = thresholds(csa, crow, cal_pop, n_rows)
            Tb = thresholds(csb, crow, cal_pop, n_rows)
            for cut in CUTS:
                key = f"F{floor:g}/k{k:g}/{mode}/{cut}"
                pc = pass_counts(sa, sb, row, Ta, Tb, cut, w_test, idx_e, n_ev * len(SA.SURFACES) * N_CAT)
                pe = pc.reshape(N_THR, n_ev, len(SA.SURFACES), N_CAT)
                nom = pe.sum(1)
                boot = np.einsum("re,tesc->rtsc", reps, pe, optimize=True)
                S0, SB = summarise(nom), summarise(boot)
                comb = {"nominal": S0, "test_side_sd": {}}
                for g in S0:
                    for fld in ("eff_sig_all", "eff_sig_T", "retention", "eff_allint"):
                        comb["test_side_sd"].setdefault(g, {})[fld] = np.nanstd(SB[g][fld], axis=0, ddof=1)
                    comb.setdefault("ret_at_99", {})[g] = {
                        "sig_all": float(at_eff(S0[g]["eff_sig_all"], S0[g]["retention"])[0]),
                        "sig_all_sd_test": float(np.nanstd(at_eff(SB[g]["eff_sig_all"], SB[g]["retention"]), ddof=1)),
                        "sig_T": float(at_eff(S0[g]["eff_sig_T"], S0[g]["retention"])[0]),
                        "sig_T_sd_test": float(np.nanstd(at_eff(SB[g]["eff_sig_T"], SB[g]["retention"]), ddof=1))}
                if k == 1.0:       # fine bins at the four table levels
                    li = [L_GRID.index(lv) for lv in SA.LEVELS]
                    rr = np.where(row >= 0, row, n_rows)
                    cls = test["T"].astype(np.int64)
                    fine = {}
                    for lv, i in zip(SA.LEVELS, li):
                        p = (sa <= Ta[rr, i]) if cut == "S1" else (sb <= Tb[rr, i]) if cut == "beta" else \
                            ((sa <= Ta[rr, i]) & (sb <= Tb[rr, i]))
                        fidx = (((test["sidx"].astype(np.int64) * (len(ETA_EDGES) - 1) + ebin) * 3 + pbin) * 2 + cls) \
                            * len(CATS) + cat // 2
                        nb = len(SA.SURFACES) * (len(ETA_EDGES) - 1) * 3 * 2 * len(CATS)
                        num = np.bincount(fidx[p], weights=w_test[p], minlength=nb)
                        den = np.bincount(fidx, weights=w_test, minlength=nb)
                        fine[f"{lv:g}"] = {"pass": num.reshape(len(SA.SURFACES), -1, 3, 2, len(CATS)),
                                           "all": den.reshape(len(SA.SURFACES), -1, 3, 2, len(CATS))}
                    comb["fine_bins_k1"] = fine
                    chain["combos"][key] = {"cal_sa": csa, "cal_sb": csb, "cal_row": crow, "cal_pop": cal_pop,
                                            "sa": sa, "sb": sb, "row": row, "cut": cut, "idx": idx_s}
                res["combos"][key] = comb
                print(f"  {key}: retention at 99% signal eff (all / class T) barrel "
                      f"{comb['ret_at_99']['barrel']['sig_all']:.3f} / {comb['ret_at_99']['barrel']['sig_T']:.3f}, "
                      f"discs {comb['ret_at_99']['discs']['sig_all']:.3f} / {comb['ret_at_99']['discs']['sig_T']:.3f} "
                      f"({time.time() - t0:.0f} s)", flush=True)
            # absolute angle sigma at this k (class T, k-transformed residual; consumer sigma), per surface
            if floor == FLOORS[0]:
                for a_ in SA.ANGLES:
                    r_ = SA.knob(a_, test, test["T"], k, mode, a=test["em"][a_], ref=test["tr"][a_] if k < 1 else None,
                                 sigma=test["sig"][a_])
                    for i, s in enumerate(SA.SURFACES):
                        m = test["T"] & (test["surface"] == s) & (test["tp_pt"] >= SA.PT_MIN)
                        resid = np.degrees(SA.residual(a_, r_["a"][m], test["tr"][a_][m]))
                        c = SA.core_w(resid)
                        res.setdefault("abs_sigma_deg", {})[f"k{k:g}/{mode}/{a_}/{SA.SURFACE_NAMES[s]}"] = {
                            "core": None if c is None else c["rms0"], "full_rms": None if c is None else c["full_rms0"],
                            "consumer_sigma_median": float(np.degrees(np.median(r_["sigma"][m])))}
        del cat
    # full chain at k = 1
    G["S"] = chain
    t1 = time.time()
    with mp.get_context("fork").Pool(a.proc) as pool:
        got = {}
        for i, (b, r_) in enumerate(pool.imap_unordered(_chain_replica, range(a.rep + 1), chunksize=2)):
            got[b] = r_
            if (i + 1) % 25 == 0:
                print(f"  full chain {i + 1}/{a.rep + 1} ({time.time() - t1:.0f} s)", flush=True)
    for key in chain["combos"]:
        st = np.stack([got[b][key].reshape(N_THR, len(SA.SURFACES), N_CAT) for b in range(a.rep + 1)])
        S = summarise(st[1:])
        S0 = summarise(st[0])
        comb = res["combos"][key]
        comb["full_chain_sd"] = {g: {fld: np.nanstd(S[g][fld], axis=0, ddof=1)
                                     for fld in ("eff_sig_all", "eff_sig_T", "retention", "eff_allint")} for g in S}
        comb["full_chain_nominal_check"] = {g: float(np.nanmax(np.abs(S0[g]["retention"] - comb["nominal"][g]["retention"])))
                                            for g in S0}
        for g in S:
            comb["ret_at_99"][g]["sig_all_sd_chain"] = float(np.nanstd(at_eff(S[g]["eff_sig_all"], S[g]["retention"]), ddof=1))
            comb["ret_at_99"][g]["sig_T_sd_chain"] = float(np.nanstd(at_eff(S[g]["eff_sig_T"], S[g]["retention"]), ddof=1))
        for d in [SA.SURFACE_NAMES[s] for s in SA.DISCS] + ["discs"]:
            for sig in ("sig_all", "sig_T"):
                fld = "eff_sig_all" if sig == "sig_all" else "eff_sig_T"
                rd = at_eff(S[d][fld], S[d]["retention"])
                rb = at_eff(S["barrel"][fld], S["barrel"]["retention"])
                nom = comb["ret_at_99"][d][sig] / comb["ret_at_99"]["barrel"][sig]
                comb.setdefault("disc_over_barrel_at_99", {})[f"{d}/{sig}"] = {
                    "ratio": nom, "sd": float(np.nanstd(rd / rb, ddof=1)),
                    "rejection_ratio": (1 - comb["ret_at_99"][d][sig]) / (1 - comb["ret_at_99"]["barrel"][sig]),
                    "rejection_ratio_sd": float(np.nanstd((1 - rd) / (1 - rb), ddof=1))}
    res["wall_s"] = round(time.time() - t0, 1)
    CM.jdump(to_lists(res), Path(a.out))
    print(f"wrote {a.out} ({res['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
