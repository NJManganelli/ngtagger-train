"""Trial screen and choice per cell (tagger training policy, decision 49), on VALIDATION jets only.

  python eval_refitq/tagger_sp/trial_select.py           -> OUT/selection.json
  prints "TOPUP <mask> <variant>" for every cell that needs more trials (run_all.sh reads them)

Inputs: train.py's prediction files OUT/preds/<mask>_<variant>_s<seed>.npz of the current build's samples
(dev files with a tag suffix and stale files of another sample set are ignored); only their validation part
(val_prob, val_label, val_closs) is read, never the test part.

Per trial:
  FAILED     the validation loss or outputs are not finite, or the trial is DEGENERATE on the validation
             jets (the d0 study's screen, d0_analyze.screen, applied to validation): more than 1 % of the
             jets share one output vector, or a b vs udsg working-point threshold (the menu-matched fake
             rate, 10 %, 1 %) sits on a tie block holding >= 1 % of the validation udsg jets (a "tie" or a
             shared vector needs at least two jets).
  OUTLIER    judged only when the cell has >= MIN_FOR_OUTLIERS non-failed trials: validation
             classification loss more than OUTLIER_Z robust SDs (1.4826 x MAD) above the median of the
             non-failed trials, or validation b vs udsg AUC more than OUTLIER_Z robust SDs below theirs.
  KEPT       neither.
Per cell: DEPLOYED = the N_DEPLOY kept trials with the lowest validation classification loss
(class-weighted cross-entropy, train.py's val_closs); the analysis averages their softmax outputs
(ensemble) and also reports the single best. A cell with fewer than N_DEPLOY kept trials whose seeds are
all below TOPUP_SEEDS[0] needs a TOP-UP (seeds TOPUP_SEEDS, once).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as AZ  # noqa: E402
import common as C  # noqa: E402

N_DEPLOY = 3
FIRST_SEEDS = list(range(10))
TOPUP_SEEDS = list(range(10, 20))
DEGEN_FRAC = 0.01
MIN_FOR_OUTLIERS = 5
OUTLIER_Z = 3.5


def selection_path() -> str:
    return os.path.join(C.OUT, "selection.json")


def load_val(pdir):
    """{(variant, mask): {seed: {val_prob, val_label, val_closs}}}."""
    P = {}
    for f in sorted(glob.glob(os.path.join(pdir, "*.npz"))):
        m = re.fullmatch(r"([AI]{4})_(\w+?)_s(\d+)\.npz", os.path.basename(f))
        if not m or m.group(2) not in C.VARIANTS:
            continue
        d = np.load(f)
        if [str(x) for x in d["samples"]] != C.built_samples(m.group(1)):
            print(f"  ignored (stale: samples {[str(x) for x in d['samples']]}): {os.path.basename(f)}")
            continue
        P.setdefault((m.group(2), m.group(1)), {})[int(m.group(3))] = {
            "prob": d["val_prob"].astype(np.float64), "label": d["val_label"], "closs": float(d["val_closs"])}
    return P


def screen(d, targets):
    """Degeneracy screen of one trial on its validation jets (d0_analyze.screen's rule)."""
    from ngtagger.definitions.matched_wp import tie_split_threshold

    p, lab = d["prob"], d["label"]
    out = {"finite": bool(np.isfinite(p).all() and np.isfinite(d["closs"]))}
    if not out["finite"]:
        out.update(degenerate=True, why="non-finite outputs or loss")
        return out
    _, cnt = np.unique(np.round(p, 6), axis=0, return_counts=True)
    D = AZ.disc(p, ["b"], ["b", "light", "gluon"])
    b, u = lab == AZ.CI["b"], np.isin(lab, [AZ.CI["light"], AZ.CI["gluon"]])
    out["max_identical_output_frac"] = float(cnt.max() / len(p)) if cnt.max() > 1 else 0.0
    out["auc_b_udsg"] = float(AZ.wauc(D[b | u], b[b | u], np.ones((1, int((b | u).sum()))))[0])
    out["tie_block_udsg_frac"] = {}
    for name, x in targets.items():
        t, _ = tie_split_threshold(D[u], x)
        k = int(np.sum(D[u] == t))
        out["tie_block_udsg_frac"][name] = k / int(u.sum()) if k > 1 else 0.0    # one jet is not a tie
    why = []
    if out["max_identical_output_frac"] > DEGEN_FRAC:
        why.append(f"{out['max_identical_output_frac']:.1%} of the jets share one output vector")
    why += [f"the {k} threshold sits on a tie block of {v:.1%} of the udsg jets"
            for k, v in out["tie_block_udsg_frac"].items() if v >= DEGEN_FRAC]
    out["degenerate"] = bool(why)
    out["why"] = "; ".join(why)
    return out


def robust_z(v):
    v = np.asarray(v, float)
    med = np.median(v)
    s = 1.4826 * np.median(np.abs(v - med))
    return (v - med) / s if s > 0 else np.zeros_like(v)


def select_cell(trials, targets):
    seeds = sorted(trials)
    T = {s: {"val_closs": trials[s]["closs"], **screen(trials[s], targets)} for s in seeds}
    ok = [s for s in seeds if not T[s]["degenerate"]]
    for s in seeds:
        T[s]["status"] = "failed" if T[s]["degenerate"] else "kept"
    if len(ok) >= MIN_FOR_OUTLIERS:
        zl = robust_z([T[s]["val_closs"] for s in ok])
        za = robust_z([T[s]["auc_b_udsg"] for s in ok])
        for s, a, b in zip(ok, zl, za):
            T[s]["z_val_closs"], T[s]["z_val_auc"] = float(a), float(b)
            if a > OUTLIER_Z or b < -OUTLIER_Z:
                T[s]["status"] = "outlier"
                T[s]["why"] = (f"validation loss z {a:+.1f}" if a > OUTLIER_Z else "") + \
                    ("; " if a > OUTLIER_Z and b < -OUTLIER_Z else "") + \
                    (f"validation b vs udsg AUC z {b:+.1f}" if b < -OUTLIER_Z else "")
    kept = sorted([s for s in seeds if T[s]["status"] == "kept"], key=lambda s: (T[s]["val_closs"], s))
    deployed = kept[:N_DEPLOY]
    for s in deployed:
        T[s]["deployed"] = True
    return {"trials": T, "launched": len(seeds), "seeds": seeds,
            "failed": sum(T[s]["status"] == "failed" for s in seeds),
            "outliers": sum(T[s]["status"] == "outlier" for s in seeds),
            "kept": len(kept), "kept_seeds": kept, "deployed_seeds": deployed,
            "best_seed": deployed[0] if deployed else None,
            "topup_needed": bool(len(kept) < N_DEPLOY and max(seeds) < TOPUP_SEEDS[0]),
            "flag": None if len(deployed) == N_DEPLOY else
            ("NO KEPT TRIAL" if not deployed else f"only {len(deployed)} trial(s) kept")}


def main():
    fr_menu, _ = AZ.stock_reference()
    targets = {"menu": fr_menu, "10%": 0.10, "1%": 0.01}
    P = load_val(os.path.join(C.OUT, "preds"))
    if not P:
        raise SystemExit("no predictions")
    S = {f"{v}@{m}": select_cell(P[(v, m)], targets) for (v, m) in sorted(P)}
    out = {"what": "trial screen and choice on validation jets (trial_select.py docstring)",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "rules": {"n_deploy": N_DEPLOY, "degenerate_frac": DEGEN_FRAC, "min_for_outliers": MIN_FOR_OUTLIERS,
                     "outlier_z": OUTLIER_Z, "first_seeds": FIRST_SEEDS, "topup_seeds": TOPUP_SEEDS,
                     "screen_targets_udsg_fake_rate": targets},
           "cells": S}
    with open(selection_path(), "w") as f:
        json.dump(out, f, indent=1)
    for c, s in S.items():
        print(f"{c:18s} launched {s['launched']:2d}  failed {s['failed']:2d}  outliers {s['outliers']}  "
              f"kept {s['kept']:2d}  deployed {s['deployed_seeds']}" + (f"  [{s['flag']}]" if s["flag"] else ""))
    for c, s in S.items():
        if s["topup_needed"]:
            v, m = c.split("@")
            print(f"TOPUP {m} {v}")


if __name__ == "__main__":
    main()
