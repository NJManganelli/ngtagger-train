"""Fit the learning curves err(N) = err_inf + a*N^-b (and floorless
err = a*N^-b), bootstrap the exponent over seeds, compute gain(N) =
AUC_1111 - AUC_0000 (paired by seed), and extrapolate N_cross where
AUC_1111(N) exceeds the embedded@0000 absolute bar.

Reads:  calibration/learning_curve.json, paired_gaps.json, embedded_eval.json
Writes: calibration/calibration_summary.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
FRACTIONS = [(1, 16), (1, 8), (1, 4), (1, 2), (1, 1)]
SEEDS = [1, 2, 3]
VIEWS = ["1111", "0000"]
N_BOOT = 2000


def collect(runs, view):
    """-> list of (N, [aucs by seed], fraction)."""
    pts = []
    for num, den in FRACTIONS:
        recs = [runs[f"{view}__f{num}of{den}__s{s}"] for s in SEEDS]
        pts.append({"fraction": num / den, "N": recs[0]["n_train"],
                    "aucs": [r["macro_auc"] for r in recs],
                    "b_aucs": [r["per_flavor_auc"]["b"] for r in recs],
                    "train_class_counts": recs[0]["train_class_counts"]})
    return pts


def fit_curve(Ns, errs, floor=True):
    Ns, errs = np.asarray(Ns, float), np.asarray(errs, float)
    if floor:
        f = lambda N, e0, a, b: e0 + a * N ** (-b)
        p0 = [max(errs.min() * 0.8, 1e-3), 1.0, 0.5]
        bounds = ([0, 1e-6, 0.01], [0.5, 1e3, 2.0])
    else:
        f = lambda N, a, b: a * N ** (-b)
        p0 = [1.0, 0.2]
        bounds = ([1e-6, 0.001], [1e3, 2.0])
    popt, _ = curve_fit(f, Ns, errs, p0=p0, bounds=bounds, maxfev=20000)
    pred = f(Ns, *popt)
    rms = float(np.sqrt(np.mean((pred - errs) ** 2)))
    return popt, rms


def boot_fits(pts, floor=True, n_boot=N_BOOT, seed=17):
    """Bootstrap over seeds within each fraction; refit each replica."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        Ns, errs = [], []
        for p in pts:
            pick = rng.integers(0, len(p["aucs"]), len(p["aucs"]))
            for j in pick:
                Ns.append(p["N"])
                errs.append(1 - p["aucs"][j])
        try:
            popt, _ = fit_curve(Ns, errs, floor=floor)
            out.append(popt)
        except Exception:
            pass
    return np.array(out)


def n_cross_from_fit(popt, bar, floor=True):
    """Smallest N with AUC(N) = 1 - err(N) >= bar; None if unreachable."""
    if floor:
        e0, a, b = popt
        target_err = 1 - bar
        if target_err <= e0:
            return None
        return float((a / (target_err - e0)) ** (1 / b))
    a, b = popt
    return float((a / (1 - bar)) ** (1 / b))


def q(x, lo=16, hi=84):
    x = np.asarray(x, float)
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def main():
    runs = json.load(open(os.path.join(CAL, "learning_curve.json")))["runs"]
    gaps = json.load(open(os.path.join(CAL, "paired_gaps.json")))
    emb = json.load(open(os.path.join(CAL, "embedded_eval.json")))
    bar = gaps["headline"]["absolute_bar_embedded_0000_test"]
    bar_std = gaps["headline"]["absolute_bar_boot_std"]
    bar_all = gaps["headline"]["absolute_bar_embedded_0000_alljets"]

    summary = {"inputs": {"bar_embedded_0000_test": bar,
                          "bar_boot_std": bar_std,
                          "bar_embedded_0000_alljets": bar_all},
               "views": {}, "gain_vs_N": [], "n_cross": {}}

    pts_by_view = {}
    for view in VIEWS:
        pts = collect(runs, view)
        pts_by_view[view] = pts
        Ns = [p["N"] for p in pts for _ in p["aucs"]]
        errs = [1 - a for p in pts for a in p["aucs"]]
        b_errs = [1 - a for p in pts for a in p["b_aucs"]]

        v = {"points": [{"fraction": p["fraction"], "N": p["N"],
                         "aucs": p["aucs"],
                         "auc_mean": float(np.mean(p["aucs"])),
                         "auc_std": float(np.std(p["aucs"])),
                         "b_auc_mean": float(np.mean(p["b_aucs"])),
                         "train_class_counts": p["train_class_counts"]}
                        for p in pts]}
        for floor, tag in [(True, "with_floor"), (False, "no_floor")]:
            popt, rms = fit_curve(Ns, errs, floor=floor)
            boots = boot_fits(pts, floor=floor)
            bcol = popt.tolist()
            rec = {"params": bcol, "rms": rms,
                   "param_names": (["err_inf", "a", "b"] if floor else ["a", "b"]),
                   "b_exponent": float(popt[-1]),
                   "b_boot_std": float(np.std(boots[:, -1])),
                   "b_boot_q16_84": q(boots[:, -1])}
            if floor:
                rec["err_inf"] = float(popt[0])
                rec["auc_inf"] = float(1 - popt[0])
                rec["auc_inf_boot_q16_84"] = q(1 - boots[:, 0])
            v[f"fit_macro_{tag}"] = rec
            v[f"_boots_{tag}"] = boots  # kept in-memory for n_cross
        # b-class fit (with floor)
        try:
            popt_b, rms_b = fit_curve(Ns, b_errs, floor=True)
            v["fit_bclass_with_floor"] = {"params": popt_b.tolist(), "rms": rms_b,
                                          "b_exponent": float(popt_b[-1])}
        except Exception as e:
            v["fit_bclass_with_floor"] = {"error": str(e)}
        summary["views"][view] = v

    # joint shared-exponent fit: err_v(N) = e0_v + a_v * N^-b (b shared)
    def joint_model(X, e1, a1, e2, a2, b):
        N, which = X
        return np.where(which == 0, e1 + a1 * N ** (-b), e2 + a2 * N ** (-b))
    Ns, errs, which = [], [], []
    for vi, view in enumerate(VIEWS):
        for p in pts_by_view[view]:
            for a in p["aucs"]:
                Ns.append(p["N"]); errs.append(1 - a); which.append(vi)
    Ns, errs, which = map(np.array, (Ns, errs, which))
    pj, _ = curve_fit(joint_model, (Ns, which), errs,
                      p0=[0.2, 1, 0.2, 1, 0.5],
                      bounds=([0, 1e-6, 0, 1e-6, 0.01], [0.5, 1e3, 0.5, 1e3, 2.0]),
                      maxfev=40000)
    summary["joint_fit_shared_b"] = {
        "param_names": ["err_inf_1111", "a_1111", "err_inf_0000", "a_0000", "b_shared"],
        "params": pj.tolist(), "b_shared": float(pj[4])}

    # gain(N), paired by seed
    for p1, p0 in zip(pts_by_view["1111"], pts_by_view["0000"]):
        d = [a1 - a0 for a1, a0 in zip(p1["aucs"], p0["aucs"])]
        summary["gain_vs_N"].append({
            "fraction": p1["fraction"],
            "N_1111": p1["N"], "N_0000": p0["N"],
            "gain_mean": float(np.mean(d)),
            "gain_sem": float(np.std(d, ddof=1) / np.sqrt(len(d))),
            "per_seed": d})

    # N_cross: extrapolate the 1111 macro fit to the embedded@0000 bar
    v1 = summary["views"]["1111"]
    for tag in ["with_floor", "no_floor"]:
        popt = np.array(v1[f"fit_macro_{tag}"]["params"])
        boots = v1[f"_boots_{tag}"]
        floor = tag == "with_floor"
        for bar_name, bar_val in [("bar_test", bar), ("bar_alljets", bar_all),
                                  ("bar_test_minus_1sigma", bar - bar_std)]:
            nc = n_cross_from_fit(popt, bar_val, floor=floor)
            ncs = [n_cross_from_fit(p, bar_val, floor=floor) for p in boots]
            finite = np.array([x for x in ncs if x is not None and np.isfinite(x)])
            frac_reach = len(finite) / len(ncs) if len(ncs) else 0.0
            rec = {"bar": bar_val, "n_cross_jets_central": nc,
                   "fraction_of_boot_fits_reaching_bar": frac_reach,
                   "n_cross_jets_q16_84": (q(finite) if len(finite) > 10 else None),
                   "n_cross_jets_median": (float(np.median(finite)) if len(finite) else None)}
            summary["n_cross"][f"{tag}__{bar_name}"] = rec
        summary["n_cross"][tag] = summary["n_cross"][f"{tag}__bar_test"]
    for view in VIEWS:  # drop in-memory boot arrays before writing
        for tag in ["with_floor", "no_floor"]:
            summary["views"][view].pop(f"_boots_{tag}", None)

    # convert to events / per-class using measured rates (view 1111)
    jpe = emb["views"]["1111"]["labeled_jets_per_event"]
    counts = emb["views"]["1111"]["class_counts_all"]
    ntot = sum(counts.values())
    frac_train = 0.8
    conv = {"labeled_jets_per_event": jpe, "train_fraction": frac_train,
            "train_jets_per_event": jpe * frac_train,
            "class_fractions": {c: n / ntot for c, n in counts.items()}}
    for tag, rec in summary["n_cross"].items():
        for key in ["n_cross_jets_central", "n_cross_jets_median"]:
            nj = rec.get(key)
            if nj:
                rec[key.replace("jets", "events")] = nj / (jpe * frac_train)
        if rec.get("n_cross_jets_central"):
            rec["n_cross_per_class_train_jets"] = {
                c: rec["n_cross_jets_central"] * f
                for c, f in conv["class_fractions"].items()}
    summary["conversions"] = conv

    with open(os.path.join(CAL, "calibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for view in VIEWS:
        fm = summary["views"][view]["fit_macro_with_floor"]
        nf = summary["views"][view]["fit_macro_no_floor"]
        print(f"[{view}] b(floor)={fm['b_exponent']:.3f}+-{fm['b_boot_std']:.3f} "
              f"auc_inf={fm.get('auc_inf', float('nan')):.4f} | "
              f"b(no floor)={nf['b_exponent']:.3f}+-{nf['b_boot_std']:.3f}", flush=True)
    print("[joint] b_shared=%.3f" % summary["joint_fit_shared_b"]["b_shared"])
    for g in summary["gain_vs_N"]:
        print(f"  gain @N~{g['N_1111']}: {g['gain_mean']:+.4f}+-{g['gain_sem']:.4f}")
    for tag, rec in summary["n_cross"].items():
        print(f"[n_cross {tag}] central={rec['n_cross_jets_central']} "
              f"median={rec['n_cross_jets_median']} "
              f"reach_frac={rec['fraction_of_boot_fits_reaching_bar']:.2f}")


if __name__ == "__main__":
    main()
