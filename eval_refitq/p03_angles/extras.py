#!/usr/bin/env python
"""P0.3 step 4: the other D.3 outputs.

  * the response p(emitted | true) per surface (x facing on the discs), class T, pT >= 1 GeV, fit
    files 1-2 (the input of the later full-response model): (a) in the table's representation, the
    residual (in published sigma and in degrees) per fine true-key bin; (b) in the module frame,
    atan(local cot) emitted vs true, alpha and beta, in degrees;
  * the shared-cluster fraction within dR < 0.1 of b-jet axes (generator b jets, pT > 15 GeV,
    |eta| < 2.4), with the inclusive and the class-T fractions; charge-fraction thresholds 0.5 / 0.7
    / 0.9 (0.9 is P0.2's; the sheet's threshold is provisional);
  * the class tell: AUC of the single-cluster S1 and beta-window statistics between class-T and
    class-C clusters of signal-like TPs (pT >= 2, born inside 2.5 cm), per surface, at each k / mode;
  * the plan's B.4 priors recomputed where they are cluster-level.

    python extras.py [--table ...] [--rep 250]
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
import roc as RC  # noqa: E402

from ngtagger.spdefs import angles as SA  # noqa: E402

SEED = 20261011
RES_PULL = np.linspace(-8, 8, 161)
RES_DEG = {"alpha": np.linspace(-30, 30, 121), "beta": np.linspace(-90, 90, 181)}
LOCAL_DEG = np.linspace(-90, 90, 361)
COLS = RC.ROC_COLS + ("tpLocalCotAlpha", "tpLocalCotBeta", "tp_d0", "tp_eta", "tp_vz")


# ---------------------------------------------------------------- response
def response(fit: dict) -> tuple[dict, dict]:
    T = fit["T"] & (fit["tp_pt"] >= SA.PT_MIN)
    arrays, meta = {}, {"pull_edges": RES_PULL.tolist(), "deg_edges": {a: v.tolist() for a, v in RES_DEG.items()},
                        "local_deg_edges": LOCAL_DEG.tolist(), "groups": []}
    kv = SA.incidence_key(fit["surface"], fit["tpGlobalClusterCotTheta"])
    fb = SA.fine_bin(fit["surface"], kv)
    for s in SA.SURFACES:
        for f in ((0,) if s in SA.BARREL else (-1, 1)):
            g = T & (fit["surface"] == s) & (fit["facing"] == f)
            tag = f"{SA.SURFACE_NAMES[s]}_f{f:+d}"
            meta["groups"].append({"tag": tag, "surface": s, "facing": f, "n": int(g.sum())})
            for a in SA.ANGLES:
                r = SA.residual(a, fit["em"][a][g], fit["tr"][a][g])
                p = r / fit["sig"][a][g]
                h, _, _ = np.histogram2d(fb[g], np.clip(p, RES_PULL[0], RES_PULL[-1] - 1e-9),
                                         bins=[np.arange(SA.N_FINE[s] + 1) - 0.5, RES_PULL])
                arrays[f"{tag}/{a}/pull_vs_truekey"] = h.astype(np.int32)
                h, _, _ = np.histogram2d(fb[g], np.clip(np.degrees(r), RES_DEG[a][0], RES_DEG[a][-1] - 1e-9),
                                         bins=[np.arange(SA.N_FINE[s] + 1) - 0.5, RES_DEG[a]])
                arrays[f"{tag}/{a}/deg_vs_truekey"] = h.astype(np.int32)
                le = {"alpha": "localCotAlpha", "beta": "localCotBeta"}[a]
                xt = np.degrees(np.arctan(fit[f"tp{le[0].upper()}{le[1:]}"][g].astype(np.float64)))
                xe = np.degrees(np.arctan(fit[le][g].astype(np.float64)))
                h, _, _ = np.histogram2d(xt, xe, bins=[LOCAL_DEG, LOCAL_DEG])
                arrays[f"{tag}/{a}/local_emitted_vs_true_deg"] = h.astype(np.int32)
    return arrays, meta


# ---------------------------------------------------------------- shared clusters near b jets
def shared_fraction(files, n_rep, rng) -> dict:
    E = CM.load_events(files)
    sc = E["shared_counts"].astype(np.float64)       # ev, surface, (cone, inclusive), (all, T), (n, shared@fracs)
    fr = CM.cache_meta(files[0])["shared_fracs"]
    out = {"fracs": fr, "bjets": {"n_events_with_bjet": int(np.sum(sc[:, :, 0, 0, 0].sum(1) > 0))}}
    groups = {"barrel": [SA.SURFACES.index(s) for s in SA.BARREL], "discs": [SA.SURFACES.index(s) for s in SA.DISCS]}
    groups.update({SA.SURFACE_NAMES[s]: [i] for i, s in enumerate(SA.SURFACES)})
    n_ev = sc.shape[0]
    reps = rng.multinomial(n_ev, np.full(n_ev, 1.0 / n_ev), size=n_rep).astype(np.float64)
    for g, si in groups.items():
        x = sc[:, si].sum(1)                          # ev, cone/incl, all/T, counts
        nom = x.sum(0)
        bt = np.einsum("re,eabc->rabc", reps, x)
        res = {}
        for ci, cn in enumerate(("bjet_cone", "inclusive")):
            for ti, tn in enumerate(("linked", "classT")):
                for k, f in enumerate(fr):
                    v = nom[ci, ti, 1 + k] / nom[ci, ti, 0]
                    sd = np.std(bt[:, ci, ti, 1 + k] / bt[:, ci, ti, 0], ddof=1)
                    res[f"{cn}/{tn}/shared<{f:g}"] = {"value": float(v), "sd": float(sd), "n": float(nom[ci, ti, 0])}
        out[g] = res
    return out


# ---------------------------------------------------------------- class tell
def class_tell(test: dict, tab) -> dict:
    from scipy.stats import mannwhitneyu

    out = {}
    pop = test["linked"] & (test["tp_pt"] >= 2.0) & (test["born"] <= RC.BORN_CM) & test["ok"]
    isT = test["T"]
    for k, mode in RC.SETTINGS:
        sa, sb, _ = RC.statistics(test, tab, 2.0, k, mode)
        for name, s in (("S1", sa), ("beta", sb)):
            for i, sv in enumerate(SA.SURFACES):
                m = pop & (test["surface"] == sv) & np.isfinite(s)
                t, c = s[m & isT], s[m & ~isT]
                if len(t) < 50 or len(c) < 50:
                    continue
                u = mannwhitneyu(c, t).statistic
                out[f"k{k:g}/{mode}/{name}/{SA.SURFACE_NAMES[sv]}"] = float(u / (len(t) * len(c)))
    return out


# ---------------------------------------------------------------- priors
def priors(fit: dict, test: dict, tab, table_doc: dict) -> dict:
    P = {}
    T = test["T"] & (test["tp_pt"] >= SA.PT_MIN)
    # alpha core / full RMS per surface (deg, mrad); the curvature term at 2 GeV at the median radius
    al = {}
    for s in SA.SURFACES:
        m = T & (test["surface"] == s)
        r = np.degrees(SA.residual("alpha", test["em"]["alpha"][m], test["tr"]["alpha"][m]))
        c = SA.core_w(r)
        rmed = float(np.median(test["r"][test["surface"] == s]))
        al[SA.SURFACE_NAMES[s]] = {"core_deg": c["rms0"], "full_rms_deg": c["full_rms0"], "core_mrad": np.radians(c["rms0"]) * 1e3,
                                   "median_r_cm": rmed,
                                   "curvature_term_2GeV_deg": float(np.degrees(SA.alpha_band(rmed, 2.0)))}
    P["alpha"] = {"per_surface": al,
                  "prior": {"core_deg": "3.9-4.7", "full_rms_deg": "6.6-7.0", "curvature_2GeV_IL1_IL4_deg": "0.49 / 2.4",
                            "core_mrad_kf_weight": "68-82"}}
    # the pT below which S1 (floor 2 GeV, 99 % level, k = 1) vetoes: 50 % efficiency point per surface
    sa, _, row = RC.statistics(test, tab, 2.0, 1.0, "fixed")
    csa, _, crow = RC.statistics(fit, tab, 2.0, 1.0, "fixed")
    n_rows = len(tab.doc["keys"]["emitted"]["rows"])
    Ta = RC.thresholds(csa, crow, fit["T"] & (fit["born"] <= RC.BORN_CM) & (fit["tp_pt"] >= 2.0), n_rows)
    i99 = RC.L_GRID.index(0.99)
    passed = sa <= Ta[np.where(row >= 0, row, n_rows), i99]
    ptb = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0])
    veto = {}
    for s in SA.SURFACES:
        m = test["T"] & (test["surface"] == s) & (test["born"] <= RC.BORN_CM)
        eff = []
        for lo, hi in zip(ptb[:-1], ptb[1:]):
            mm = m & (test["tp_pt"] >= lo) & (test["tp_pt"] < hi)
            w = test["w"][mm].astype(float)
            eff.append(float(np.sum(w * passed[mm]) / w.sum()) if w.sum() else np.nan)
        eff = np.array(eff)
        mid = 0.5 * (ptb[:-1] + ptb[1:])
        j = np.where(eff >= 0.5)[0]
        p50 = float(np.interp(0.5, eff[j[0] - 1:j[0] + 1], mid[j[0] - 1:j[0] + 1])) if len(j) and j[0] > 0 else None
        veto[SA.SURFACE_NAMES[s]] = {"pt_mid": mid.tolist(), "eff": eff.tolist(), "pt_at_50pct": p50}
    P["alpha_veto"] = {"per_surface": veto, "prior": "alpha vetoes pT below ~0.3-0.5 GeV, mainly at IL3/IL4",
                       "definition": "S1 at the 2 GeV floor, per-bin 99 % level, k = 1; class-T clusters of TPs born "
                                     "inside 2.5 cm; efficiency vs TP pT; the pT where it crosses 50 %"}
    # alpha and displacement: b / c daughters, pT >= 2, |eta| < 1.1, born inside 2.5 cm (unique TPs)
    hf = test["linked"] & (test["tp_hf"] > 0) & (test["tp_pt"] >= 2.0) & (np.abs(test["tp_eta"]) < 1.1) \
        & (test["born"] <= RC.BORN_CM)
    key = test["ev"].astype(np.int64)[hf] * 1_000_000 + test["tpIdx"][hf]
    _, first = np.unique(key, return_index=True)
    d0 = np.abs(test["tp_d0"][hf][first]) * 1e4          # um
    il1 = al["IL1"]
    P["alpha_displacement"] = {
        "n_tp": int(len(d0)), "median_abs_d0_um": float(np.median(d0)),
        "p75_p90_abs_d0_um": [float(np.percentile(d0, 75)), float(np.percentile(d0, 90))],
        "frac_le_0p2mm": float(np.mean(d0 <= 200)), "frac_0p2_1mm": float(np.mean((d0 > 200) & (d0 <= 1000))),
        "frac_gt_1mm": float(np.mean(d0 > 1000)),
        "dalpha_IL1_deg_at_p75_p90": [float(np.degrees(np.percentile(d0, q) * 1e-4 / il1["median_r_cm"])) for q in (75, 90)],
        "dalpha_IL1_in_core_at_p75_p90": [float(np.degrees(np.percentile(d0, q) * 1e-4 / il1["median_r_cm"]) / il1["core_deg"])
                                          for q in (75, 90)],
        "prior": {"median_d0_um": 112, "p75_p90_um": "276 / 529", "dalpha_IL1_deg": "0.5 / 1.0",
                  "in_core": "0.1-0.26", "fractions": "64 / 33 / 2.3 %"}}
    # beta response on and near the plateau (IL1): slope of the mean emitted cot vs true cot, |true cot| < 1
    m1 = T & (test["surface"] == 1)
    ct, ce = test["tpGlobalClusterCotTheta"][m1].astype(np.float64), test["globalClusterCotTheta"][m1].astype(np.float64)
    resp = {}
    for x0 in (0.45, 0.55, 0.86):
        mm = np.abs(np.abs(ct) - x0) < 0.025
        resp[f"{x0:g}"] = float(np.mean(ce[mm] * np.sign(ct[mm])))
    edges = np.linspace(0.1, 1.0, 19)
    mids, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mm = (np.abs(ct) >= lo) & (np.abs(ct) < hi)
        mids.append(0.5 * (lo + hi))
        means.append(float(np.mean(ce[mm] * np.sign(ct[mm]))))
    slope = float(np.polyfit(mids, means, 1)[0])
    P["beta_response"] = {"IL1_mean_emitted_at_true": resp, "IL1_slope_0p1_1": slope, "profile": {"mid": mids, "mean": means},
                          "per_surface_slope": {}, "prior": {"slope": "0.74-0.88", "IL1_means": "0.33 / 0.44 / 0.76"}}
    for s in SA.BARREL:
        m = T & (test["surface"] == s)
        ct, ce = test["tpGlobalClusterCotTheta"][m].astype(np.float64), test["globalClusterCotTheta"][m].astype(np.float64)
        mm = (np.abs(ct) > 0.1) & (np.abs(ct) < 1.0)
        P["beta_response"]["per_surface_slope"][SA.SURFACE_NAMES[s]] = float(np.polyfit(ct[mm], ce[mm], 1)[0])
    # plateau steps: the beta width per row of the true-key table, barrel |key| and disc tan theta
    steps = {}
    for r in table_doc["keys"]["true"]["rows"]:
        if r.get("beta") is None:
            continue
        steps.setdefault(f"{SA.SURFACE_NAMES[r['surface']]}_f{r['facing']:+d}", []).append(
            [r["lo"], r["hi"], r["beta"]["width"], r["beta"]["centre"]])
    P["plateau_steps"] = {"true_key_beta_width_rows": steps,
                          "prior": "barrel beta width steps at |cot| 0.5, 0.55, 0.6, 0.67, 0.8; discs at tan theta ~0.5 "
                                   "(ID1: 1.08 at 0.4-0.5, 0.63 at 0.5-0.6, census-scaled units)"}
    # beta pointing: the luminous region and the other-interaction fraction near the hard-scatter vertex
    E = CM.load_events(list(SA.TEST_FILES))
    gz = E["ev_genvtx_z"].astype(np.float64)
    sz = float(np.std(gz, ddof=1))
    from scipy.special import erf

    ev_z = dict(zip(E["ev_ev"].tolist(), gz.tolist()))
    pu = test["linked"] & (test["tp_hs"] == 0) & (test["born"] <= RC.BORN_CM)
    dz = np.abs(test["tp_vz"][pu] - np.array([ev_z[e] for e in test["ev"][pu]]))
    w = test["w"][pu].astype(float)
    P["beta_pointing"] = {"sigma_z_hard_scatter_cm": sz,
                          "predicted_within_1_2cm": [float(erf(1 / (2 * sz))), float(erf(2 / (2 * sz)))],
                          "measured_other_interaction_clusters_within_1_2cm": [float(np.sum(w * (dz < 1)) / w.sum()),
                                                                               float(np.sum(w * (dz < 2)) / w.sum())],
                          "prior": {"sigma_z": "4.1-4.25", "predicted": "13 % / 27 %", "measured_wrong_hits": "11 % / 20-23 %"}}
    # disc beta pointing: sigma(z0) = r dlambda / sin^2 theta, predicted from the disc class-T lambda core,
    # and measured as the core of the z0 residual, at |eta| 1.6-1.8, r 9-11 cm
    disc = {}
    for s in SA.DISCS:
        m = T & (test["surface"] == s)
        eta = np.abs(np.arcsinh(test["z"] / test["r"]))
        mm = m & (eta > 1.6) & (eta < 1.8) & (test["r"] > 9) & (test["r"] < 11)
        if mm.sum() < SA.MIN_FIT:
            continue
        dl = SA.core_w(SA.residual("beta", test["em"]["beta"][m], test["tr"]["beta"][m]))["rms0"]
        th = 2 * np.arctan(np.exp(-1.7))
        pred = 10.0 * dl / np.sin(th) ** 2
        z0r = -test["r"][mm] * (test["globalClusterCotTheta"][mm].astype(float) - test["tpGlobalClusterCotTheta"][mm])
        c = SA.core_w(z0r)
        disc[SA.SURFACE_NAMES[s]] = {"lambda_core_deg": float(np.degrees(dl)), "predicted_sigma_z0_cm": float(pred),
                                     "measured_z0_core_cm": c["rms0"], "measured_z0_full_rms_cm": c["full_rms0"],
                                     "n": int(mm.sum()),
                                     "barrel_like_4deg_prediction_cm": float(10.0 * np.radians(4) / np.sin(th) ** 2)}
    P["disc_beta_pointing"] = {"per_disc": disc, "prior": "sigma(z0) ~ 6 cm at |eta| 1.7, r = 10 cm with dbeta = 4 deg"}
    # disc alpha: bending incidence at 2 GeV vs the per-cluster resolution (local cot alpha)
    da = {}
    for s in SA.DISCS:
        m = T & (test["surface"] == s)
        res_ = test["localCotAlpha"][m].astype(float) - test["tpLocalCotAlpha"][m]
        hi = m & (test["tp_pt"] >= 2.0) & (test["tp_pt"] < 2.5)
        da[SA.SURFACE_NAMES[s]] = {"cot_alpha_resolution_core": SA.core_w(res_)["rms0"],
                                   "true_cot_alpha_median_abs_2GeV": float(np.median(np.abs(test["tpLocalCotAlpha"][hi])))}
    P["disc_alpha"] = {"per_disc": da, "prior": "bending incidence 0.002-0.01 in cot alpha at 2 GeV vs resolution 0.02-0.04"}
    P["not_recomputed"] = {"d0_vs_pt": "track-level (refit replay planes): the WP-C entry",
                           "dxy_code": "PF/PUPPI word: the downstream package",
                           "beta_pair_reduction": "pair-level inference: the census pair gates (WP-D0b)"}
    return P


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(CM.OUT_DIR / "angle_table_v1_canon_v3.json"))
    ap.add_argument("--rep", type=int, default=250)
    ap.add_argument("--out", default=str(CM.OUT_DIR))
    a = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    tab = SA.AngleTable.load(a.table)
    out = Path(a.out)
    fit = RC.prep(CM.load(SA.FIT_FILES, COLS, mask_fn=CM.table_population), tab)
    arrays, meta = response(fit)
    np.savez_compressed(out / "p03_response.npz", meta=json.dumps(meta), **arrays)
    print(f"response: {len(arrays)} histograms ({time.time() - t0:.0f} s)", flush=True)
    test = RC.prep(CM.load(SA.TEST_FILES, COLS), tab)
    res = {"table": a.table, "table_provenance": tab.provenance, "code": CM.git_state(),
           "response_file": str(out / "p03_response.npz"), "response_groups": meta["groups"]}
    res["shared"] = shared_fraction(list(SA.TEST_FILES), a.rep, rng)
    print(f"shared fractions ({time.time() - t0:.0f} s)", flush=True)
    res["class_tell_auc"] = class_tell(test, tab)
    print(f"class tell ({time.time() - t0:.0f} s)", flush=True)
    res["priors"] = priors(fit, test, tab, tab.doc)
    res["wall_s"] = round(time.time() - t0, 1)
    CM.jdump(res, out / "p03_extras.json")
    print(f"wrote {out / 'p03_extras.json'} ({res['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
