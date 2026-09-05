"""Paired comparisons of the embedded (central) NG tagger vs our scratch
stage-4 models on IDENTICAL test jets.

The embedded dumps (01_embedded_eval) store `test_order`, the exact permutation
prepare_dataset used, so embedded probs[test_order] is row-aligned with the
stage-4 pred_dumps rows (verified there via kin_pt / y_true equality).

Outputs calibration/paired_gaps.json:
  per view x seed: scratch macro, embedded macro (same jets), paired delta with
  bootstrap std; plus the headline gap0 (embedded@0000 - scratch@0000, best
  seed) and scratch@1111 vs embedded@0000 bar (unpaired, quadrature errors).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
STAGE4 = os.path.dirname(CAL)
sys.path.insert(0, HERE)
from importlib import import_module  # noqa: E402

emb_mod = import_module("01_embedded_eval".replace("/", "."))  # noqa: F401

# reuse helpers
per_class_auc = emb_mod.per_class_auc
CLASS_LABELS = emb_mod.CLASS_LABELS
N_BOOT = 1000


def paired_boot(y, pA, pB, n_boot=N_BOOT, seed=13):
    rng = np.random.default_rng(seed)
    n = len(y)
    dm = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        _, mA = per_class_auc(y[idx], pA[idx])
        _, mB = per_class_auc(y[idx], pB[idx])
        if mA is not None and mB is not None:
            dm.append(mA - mB)
    return float(np.std(dm))


def main():
    out = {"views": {}}
    for view in ["1111", "1100", "0000"]:
        with np.load(os.path.join(CAL, "dumps", f"embedded_{view}.npz"),
                     allow_pickle=True) as f:
            y, probs, order = f["y"], f["probs"], f["test_order"]
        y_te, emb_te = y[order], probs[order]
        _, emb_macro = per_class_auc(y_te, emb_te)
        v = {"embedded_test_macro": emb_macro, "seeds": {}}
        best = None
        for s in (1, 2, 3):
            dump = os.path.join(STAGE4, "pred_dumps", f"{view}__baseline__s{s}.npz")
            with np.load(dump, allow_pickle=True) as f:
                sp, sy = f["class_probs"].astype(np.float64), f["y_true"]
            assert (sy == y_te).all(), f"alignment broke for {view} s{s}"
            _, sc_macro = per_class_auc(y_te, sp)
            delta = emb_macro - sc_macro
            rec = {"scratch_macro": sc_macro, "delta_embedded_minus_scratch": delta,
                   "delta_boot_std": paired_boot(y_te, emb_te, sp)}
            v["seeds"][str(s)] = rec
            if best is None or sc_macro > best[1]:
                best = (s, sc_macro, rec)
        v["best_seed"] = best[0]
        v["best_scratch_macro"] = best[1]
        v["gap_embedded_minus_best_scratch"] = best[2]["delta_embedded_minus_scratch"]
        v["gap_boot_std"] = best[2]["delta_boot_std"]
        # per-class embedded vs best-scratch on the same jets
        with np.load(os.path.join(STAGE4, "pred_dumps",
                                  f"{view}__baseline__s{best[0]}.npz"),
                     allow_pickle=True) as f:
            sp = f["class_probs"].astype(np.float64)
        emb_pc, _ = per_class_auc(y_te, emb_te)
        sc_pc, _ = per_class_auc(y_te, sp)
        v["per_class"] = {c: {"embedded": emb_pc[c]["auc"],
                              "embedded_se_hm": emb_pc[c]["se_hm"],
                              "scratch_best": sc_pc[c]["auc"],
                              "scratch_se_hm": sc_pc[c]["se_hm"],
                              "n_pos": emb_pc[c]["n_pos"]}
                         for c in CLASS_LABELS}
        out["views"][view] = v
        print(f"[{view}] embedded(test)={emb_macro:.4f} "
              f"best scratch={best[1]:.4f} gap={best[2]['delta_embedded_minus_scratch']:+.4f}"
              f"+-{best[2]['delta_boot_std']:.4f}", flush=True)

    # headline numbers
    g0 = out["views"]["0000"]
    bar = g0["embedded_test_macro"]
    s1111 = out["views"]["1111"]
    # unpaired: different test sets -> quadrature of bootstrap stds
    emb_eval = json.load(open(os.path.join(CAL, "embedded_eval.json")))
    bar_std = emb_eval["views"]["0000"]["embedded_auc_test"]["macro_boot_std"]
    # scratch 1111 macro boot std: bootstrap the best dump
    with np.load(os.path.join(CAL, "dumps", "embedded_1111.npz"), allow_pickle=True) as f:
        y_te = f["y"][f["test_order"]]
    with np.load(os.path.join(STAGE4, "pred_dumps",
                              f"1111__baseline__s{s1111['best_seed']}.npz"),
                 allow_pickle=True) as f:
        sp = f["class_probs"].astype(np.float64)
    scratch_std = emb_mod.boot_macro(y_te, sp)
    out["headline"] = {
        "absolute_bar_embedded_0000_test": bar,
        "absolute_bar_boot_std": bar_std,
        "absolute_bar_embedded_0000_alljets": emb_eval["views"]["0000"]["embedded_auc_all"]["macro"],
        "gap0_embedded0000_minus_scratch0000": g0["gap_embedded_minus_best_scratch"],
        "gap0_boot_std": g0["gap_boot_std"],
        "scratch_1111_best_macro": s1111["best_scratch_macro"],
        "scratch_1111_boot_std": scratch_std,
        "dist_scratch1111_to_bar": bar - s1111["best_scratch_macro"],
        "dist_std_quadrature": float(np.hypot(bar_std, scratch_std)),
    }
    with open(os.path.join(CAL, "paired_gaps.json"), "w") as f:
        json.dump(out, f, indent=2)
    h = out["headline"]
    print(f"[headline] bar={h['absolute_bar_embedded_0000_test']:.4f} "
          f"gap0={h['gap0_embedded0000_minus_scratch0000']:.4f}"
          f"+-{h['gap0_boot_std']:.4f} "
          f"dist(1111->bar)={h['dist_scratch1111_to_bar']:.4f}"
          f"+-{h['dist_std_quadrature']:.4f}", flush=True)


if __name__ == "__main__":
    main()
