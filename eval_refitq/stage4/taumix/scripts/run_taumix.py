"""Tau-mixing study: mix 2 QQToHToTauTau (htt) unified nanos into the stage-4
jet-tagger training and quantify the tau-performance gain, per view.

Protocol (import, don't copy): labeling/features/loading via the ngtagger
pipeline (load_jets, label_jets, build_features), training/eval via
run_matrix (build_model, per_flavor_auc), embedded-score alignment via the
calibration study's SCORE_FIELDS/hanley_mcneil_se (index-level exact).

Split: combined (10 ttbar + 2 htt) jets; the ttbar block keeps the ORIGINAL
stage-4 test membership (seed 12345, test_fraction 0.2, verified against the
stage-4 prediction dumps) so the continuity re-score on the original test set
has ZERO train/test leakage; the htt block gets a STRATIFIED-BY-CLASS 20%
split with a fixed seed. Train/test orders are then globally shuffled (fixed
seed) so ttbar/htt jets are fully interleaved and ordering cannot leak.

Cells per view (1111, 0000), baseline features:
  <view>__baseline_taumix       train = full mixed train        (3 seeds)
  <view>__baseline_ttonly_ctrl  train = mixed train, ttbar only (3 seeds)
plus 1111__refitbdt_taumix (the calibration hint that refit-BDT lifts taus).
All evaluated on the SAME mixed test set + the original stage-4 test subset
+ the htt-only test subset; embedded NG tagger scored on the same sets.

Incremental saves to taumix_summary.json; per-(cell,seed) test probabilities
to dumps/ for the ROC overlay plot.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import awkward as ak
import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
TAUMIX = os.path.dirname(HERE)
STAGE4 = os.path.dirname(TAUMIX)
sys.path.insert(0, os.path.join(STAGE4, "scripts"))
import run_matrix  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "embedded_eval", os.path.join(STAGE4, "calibration", "scripts", "01_embedded_eval.py"))
embedded_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(embedded_eval)
SCORE_FIELDS = embedded_eval.SCORE_FIELDS
hanley_mcneil_se = embedded_eval.hanley_mcneil_se

from ngtagger.data.features import build_features  # noqa: E402
from ngtagger.data.labels import CLASS_LABELS, label_jets  # noqa: E402
from ngtagger.data.nano import load_jets  # noqa: E402
from ngtagger.train.trainer import class_pt_weights  # noqa: E402

NANO = run_matrix.NANO
MODELS = run_matrix.MODELS
SUMMARY = os.path.join(TAUMIX, "taumix_summary.json")
DUMPS = os.path.join(TAUMIX, "dumps")

HTT_GLOBS = {"1111": "nano_fat_1111_coopt_htt{}.root",
             "0000": "nano_fat_0000_baseline_htt{}.root"}
ORIG_SEED, ORIG_TEST_FRACTION = 12345, 0.2   # stage-4 split (run_matrix)
TAUMIX_SEED = 20260722                        # htt stratified split + shuffles
SEEDS = [1, 2, 3]
EPOCHS = 40
MIN_HTT_MB = 4  # 50-event unified nanos are ~half the 100-event ~16MB


def htt_files(view):
    fs = [os.path.join(NANO, HTT_GLOBS[view].format(i)) for i in (1, 2)]
    return [f for f in fs
            if os.path.exists(f) and os.path.getsize(f) > MIN_HTT_MB << 20]


def load_summary():
    if os.path.exists(SUMMARY):
        with open(SUMMARY) as f:
            return json.load(f)
    return {"study": "taumix", "cells": {}, "meta": {
        "class_labels": list(CLASS_LABELS),
        "seeds": SEEDS, "epochs": EPOCHS,
        "orig_split": {"seed": ORIG_SEED, "test_fraction": ORIG_TEST_FRACTION},
        "taumix_seed": TAUMIX_SEED,
        "split_note": (
            "ttbar block keeps the ORIGINAL stage-4 test membership (no leakage "
            "into the continuity re-score); htt block stratified-by-class 20%; "
            "train/test orders globally shuffled (ttbar+htt fully interleaved)."),
        "provenance_note": (
            "htt nanos produced with the NEW HashPRNG-compound angle-synthesis "
            "producer (A/B-validated statistically identical to old code); the 10 "
            "ttbar unified nanos were made with the OLD code. Mixing is defensible "
            "(per-layer residual KS p=0.12-0.74) but the training set is "
            "cross-provenance. Also: the deployed l1tPh3SmartPixelsNano_cff lost "
            "the unified-nano helpers in the migration; production used the Jul-21 "
            "superset installed as l1tPh3SmartPixelsNanoFat_cff (new module, "
            "deployment untouched)."),
    }}


def save_summary(summary):
    tmp = SUMMARY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, SUMMARY)


# ---------------------------------------------------------------- data ------
def load_view(view, groups):
    """Combined ttbar+htt load, labeled, with features + embedded scores +
    per-jet file origin. Row order == uproot event-major order (exact)."""
    tt = run_matrix.files_for(view)
    ht = htt_files(view)
    files = tt + ht
    cfg = run_matrix.VIEWS[view]["cfg"]
    bdt = f"{MODELS}/refitq_{cfg}_conifer.json" if (cfg and "refitbdt" in groups) else None
    jets, constituents, gen = load_jets(
        files, n_const=16, feature_groups=groups,
        refit_config=(cfg if "refitbdt" in groups else None), refit_bdt_json=bdt)
    label, _tpt, _tptp, keep = label_jets(jets, gen, max_dr=0.4)
    X, feature_names = build_features(jets, constituents, n_const=16,
                                      feature_groups=groups)
    # per-jet file index (uproot concatenate preserves file order)
    file_idx, nev = [], []
    for fi, f in enumerate(files):
        with uproot.open(f + ":Events") as t:
            n = t.num_entries
        file_idx.append(np.full(n, fi))
        nev.append(n)
    file_idx = np.concatenate(file_idx)
    nj = ak.to_numpy(ak.num(jets.pt, axis=1))
    assert len(nj) == len(file_idx), "event count mismatch"
    jet_file = np.repeat(file_idx, nj)

    flat = lambda a: ak.to_numpy(ak.flatten(a))
    kmask = flat(keep).astype(bool)
    y = flat(label)[kmask].astype(int)
    probs_emb = np.stack([flat(jets[SCORE_FIELDS[c]]) for c in CLASS_LABELS],
                         axis=1)[kmask].astype(np.float64)
    return {
        "X": np.asarray(X)[kmask], "y": y, "probs_emb": probs_emb,
        "pt": flat(jets.pt)[kmask], "file": jet_file[kmask],
        "is_htt": jet_file[kmask] >= len(tt),
        "n_tt_files": len(tt), "n_htt_files": len(ht),
        "n_htt_events": int(sum(nev[len(tt):])),
        "feature_names": feature_names,
    }


def class_counts(y):
    return {c: int((y == i).sum()) for i, c in enumerate(CLASS_LABELS)}


def make_split(d):
    """(train_idx, test_idx, verify) — see module docstring."""
    y, is_htt = d["y"], d["is_htt"]
    n = len(y)
    tt_idx = np.where(~is_htt)[0]      # ttbar block comes first, in file order
    assert (tt_idx == np.arange(len(tt_idx))).all(), "ttbar block not leading"
    n_tt = len(tt_idx)
    # original stage-4 split over the ttbar block
    rng0 = np.random.default_rng(ORIG_SEED)
    perm = rng0.permutation(n_tt)
    orig_test = perm[:int(n_tt * ORIG_TEST_FRACTION)]
    test_mask = np.zeros(n, bool)
    test_mask[orig_test] = True
    # stratified 20% over the htt block
    rng = np.random.default_rng(TAUMIX_SEED)
    for c in range(len(CLASS_LABELS)):
        idx = np.where(is_htt & (y == c))[0]
        if len(idx) == 0:
            continue
        p = rng.permutation(idx)
        test_mask[p[:int(round(len(idx) * ORIG_TEST_FRACTION))]] = True
    train_idx = rng.permutation(np.where(~test_mask)[0])
    test_idx = rng.permutation(np.where(test_mask)[0])
    return train_idx, test_idx, orig_test


def verify_orig_split(view, d, orig_test):
    """Cross-check reconstructed original test split vs the stage-4 dump."""
    dump = os.path.join(STAGE4, "pred_dumps", f"{view}__baseline__s1.npz")
    with np.load(dump, allow_pickle=True) as f:
        kin_pt, y_true = f["kin_pt"], f["y_true"]
    ok_n = len(kin_pt) == len(orig_test)
    ok_pt = ok_n and bool(np.allclose(
        kin_pt, d["pt"][orig_test].astype(np.float32), atol=1e-4))
    ok_y = ok_n and bool((y_true == d["y"][orig_test]).all())
    return {"n_dump": int(len(kin_pt)), "n_ours": int(len(orig_test)),
            "pt_match": ok_pt, "y_match": ok_y}


# ---------------------------------------------------------------- eval ------
def eval_probs(y, probs):
    out, vals = {}, []
    for i, c in enumerate(CLASS_LABELS):
        pos = y == i
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            out[c] = {"auc": None, "se_hm": None, "n_pos": n_pos}
            continue
        from sklearn.metrics import roc_auc_score
        a = float(roc_auc_score(pos, probs[:, i]))
        out[c] = {"auc": a, "se_hm": hanley_mcneil_se(a, n_pos, n_neg),
                  "n_pos": n_pos}
        vals.append(a)
    return {"macro": (float(np.mean(vals)) if vals else None), "per_class": out}


def eval_sets(y, probs, sets):
    return {name: eval_probs(y[idx], probs[idx]) for name, idx in sets.items()}


def agg(per_seed, setname):
    ms = [r[setname]["macro"] for r in per_seed if r[setname]["macro"] is not None]
    out = {"macro_mean": float(np.mean(ms)), "macro_std": float(np.std(ms))}
    for c in CLASS_LABELS:
        a = [r[setname]["per_class"][c]["auc"] for r in per_seed
             if r[setname]["per_class"][c]["auc"] is not None]
        out[c] = {"mean": (float(np.mean(a)) if a else None),
                  "std": (float(np.std(a)) if a else None),
                  "se_hm": per_seed[0][setname]["per_class"][c]["se_hm"],
                  "n_pos": per_seed[0][setname]["per_class"][c]["n_pos"]}
    return out


# ---------------------------------------------------------------- cells -----
def run_cell(summary, key, d, train_idx, test_idx, orig_test, restrict_ttbar):
    import keras

    Xtr_all, ytr_all = d["X"][train_idx], d["y"][train_idx]
    if restrict_ttbar:
        sel = ~d["is_htt"][train_idx]
        Xtr, ytr_i = Xtr_all[sel], ytr_all[sel]
    else:
        Xtr, ytr_i = Xtr_all, ytr_all
    ytr = np.eye(len(CLASS_LABELS))[ytr_i]
    yte = d["y"][test_idx]
    Xte = d["X"][test_idx]
    # eval index sets, positions WITHIN test_idx
    pos_of = {g: p for p, g in enumerate(test_idx)}
    orig_pos = np.array([pos_of[g] for g in orig_test if g in pos_of])
    assert len(orig_pos) == len(orig_test), "orig test not fully inside mixed test"
    htt_pos = np.where(d["is_htt"][test_idx])[0]
    sets = {"mixed_test": np.arange(len(test_idx)),
            "orig_ttbar_test": orig_pos, "htt_test": htt_pos}

    per_seed = []
    for s in SEEDS:
        m = run_matrix.build_model(Xtr.shape[1:], ytr.shape[1], s)
        cb = [keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                            monitor="val_loss")]
        m.fit(Xtr, ytr, validation_split=0.15, epochs=EPOCHS, batch_size=256,
              verbose=0, callbacks=cb)
        prob = m.predict(Xte, batch_size=1024, verbose=0)
        np.savez_compressed(
            os.path.join(DUMPS, f"{key}__s{s}.npz"),
            probs=prob.astype(np.float32), y=yte,
            in_orig=np.isin(np.arange(len(test_idx)), orig_pos),
            is_htt=d["is_htt"][test_idx],
            class_labels=np.asarray(CLASS_LABELS, dtype=object))
        rec = {"seed": s, **eval_sets(yte, prob, sets)}
        per_seed.append(rec)
        keras.backend.clear_session()

    payload = {
        "n_train": int(len(Xtr)), "n_test": int(len(test_idx)),
        "restrict_ttbar": restrict_ttbar,
        "n_features": int(Xtr.shape[-1]), "feature_names": d["feature_names"],
        "per_seed": per_seed,
        "agg": {sn: agg(per_seed, sn)
                for sn in ("mixed_test", "orig_ttbar_test", "htt_test")},
    }
    summary["cells"][key] = payload
    save_summary(summary)
    print(f"[saved] {key}: mixed macro={payload['agg']['mixed_test']['macro_mean']:.4f}"
          f"±{payload['agg']['mixed_test']['macro_std']:.4f}", flush=True)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    summary = load_summary()
    summary.setdefault("inventory", {})
    summary.setdefault("split", {})
    summary.setdefault("class_weights", {})
    summary.setdefault("embedded", {})

    for view in ["1111", "0000"]:
        if only not in ("all", view) and not only.startswith(view):
            continue
        groups_list = [["baseline"]]
        if view == "1111":
            groups_list.append(["baseline", "refitbdt"])
        base_loaded = None
        for groups in groups_list:
            tag = "baseline" if groups == ["baseline"] else "refitbdt"
            mixed_key = f"{view}__{tag}_taumix"
            ctrl_key = f"{view}__{tag}_ttonly_ctrl"
            want_ctrl = tag == "baseline"
            done = mixed_key in summary["cells"] and (
                not want_ctrl or ctrl_key in summary["cells"])
            if done and "--force" not in sys.argv:
                print(f"[skip] {view}/{tag} present", flush=True)
                continue
            print(f"[load] {view} groups={groups}", flush=True)
            d = load_view(view, groups)
            train_idx, test_idx, orig_test = make_split(d)

            if tag == "baseline":
                base_loaded = (d, train_idx, test_idx, orig_test)
                ver = verify_orig_split(view, d, orig_test)
                is_htt, y = d["is_htt"], d["y"]
                inv = {
                    "n_tt_files": d["n_tt_files"], "n_htt_files": d["n_htt_files"],
                    "n_htt_events": d["n_htt_events"],
                    "htt_class_counts": class_counts(y[is_htt]),
                    "ttbar_class_counts": class_counts(y[~is_htt]),
                    "htt_labeled_jets": int(is_htt.sum()),
                    "htt_tau_jets": int(((y == 4) | (y == 5))[is_htt].sum()),
                    "htt_taus_per_event": float(((y == 4) | (y == 5))[is_htt].sum()
                                                / max(d["n_htt_events"], 1)),
                }
                summary["inventory"][view] = inv
                te_mask = np.zeros(len(y), bool)
                te_mask[test_idx] = True
                summary["split"][view] = {
                    "orig_split_verification": ver,
                    "per_class_train": class_counts(y[train_idx]),
                    "per_class_test": class_counts(y[test_idx]),
                    "per_class_test_htt_part": class_counts(y[test_idx][is_htt[test_idx]]),
                    "test_fraction_per_class": {
                        c: round(float(class_counts(y[test_idx])[c]
                                       / max(class_counts(y)[c], 1)), 3)
                        for c in CLASS_LABELS},
                }
                # class/pt weighting check (trainer's onlyclass method)
                ytr_oh = np.eye(len(CLASS_LABELS))[y[train_idx]]
                w = class_pt_weights(ytr_oh, d["pt"][train_idx], "onlyclass")
                tt_sel = ~is_htt[train_idx]
                w_tt = class_pt_weights(np.eye(len(CLASS_LABELS))[y[train_idx][tt_sel]],
                                        d["pt"][train_idx][tt_sel], "onlyclass")
                summary["class_weights"][view] = {
                    "mixed_train": {c: round(float(np.mean(w[y[train_idx] == i])), 3)
                                    for i, c in enumerate(CLASS_LABELS)
                                    if (y[train_idx] == i).any()},
                    "ttbar_only_train": {c: round(float(np.mean(w_tt[y[train_idx][tt_sel] == i])), 3)
                                         for i, c in enumerate(CLASS_LABELS)
                                         if (y[train_idx][tt_sel] == i).any()},
                }
                # embedded NG tagger on the same sets
                pos_of = {g: p for p, g in enumerate(test_idx)}
                orig_pos = np.array([pos_of[g] for g in orig_test])
                htt_pos = np.where(is_htt[test_idx])[0]
                yte, pe = y[test_idx], d["probs_emb"][test_idx]
                summary["embedded"][view] = eval_sets(yte, pe, {
                    "mixed_test": np.arange(len(test_idx)),
                    "orig_ttbar_test": orig_pos, "htt_test": htt_pos})
                np.savez_compressed(
                    os.path.join(DUMPS, f"embedded_{view}.npz"),
                    y=yte, probs=pe, is_htt=is_htt[test_idx],
                    in_orig=np.isin(np.arange(len(test_idx)), orig_pos),
                    class_labels=np.asarray(CLASS_LABELS, dtype=object))
                save_summary(summary)
                print(f"[inv] {view}: htt taus={inv['htt_tau_jets']} "
                      f"({inv['htt_taus_per_event']:.2f}/evt) verify={ver}", flush=True)
                if not (ver["pt_match"] and ver["y_match"]):
                    print(f"[FATAL] {view}: original-split verification failed", flush=True)
                    return

            if mixed_key not in summary["cells"] or "--force" in sys.argv:
                run_cell(summary, mixed_key, d, train_idx, test_idx, orig_test,
                         restrict_ttbar=False)
            if want_ctrl and (ctrl_key not in summary["cells"] or "--force" in sys.argv):
                run_cell(summary, ctrl_key, d, train_idx, test_idx, orig_test,
                         restrict_ttbar=True)
    print("[taumix] done", flush=True)


if __name__ == "__main__":
    main()
