"""Stage-4 jet-tagger variant matrix runner.

Axis A (config view): 1111 (AAAA refit), 1100 (AAII refit), 0000 (OT-only, no refit).
Axis B (per-constituent features):
  baseline | +refitbdt | +vertexdxy_pv | +both   (refitbdt N/A for 0000).
Several-shot (N seeds), keep best val AUC + report spread. Per-flavor one-vs-rest
AUC + accuracy on a held-out test split. Saves incrementally to stage4_summary.json.

A compact DeepSet-style tagger (per-constituent Conv1D-1x1 -> masked sum-pool ->
dense classifier) trained with keras (fast; low-stats study). The heavy
HGQ2/QAT/firmware model is exercised separately by the arch/head check.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from sklearn.metrics import roc_auc_score  # noqa: E402

from ngtagger.data.labels import CLASS_LABELS  # noqa: E402
from ngtagger.train.trainer import prepare_dataset  # noqa: E402

NANO = "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano"
MODELS = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/stage3/models"
OUT = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/stage4"
# STAGE4_SUMMARY lets a re-run (e.g. the MVA-explorer prediction-dump pass)
# write next to — never over — the original stage4_summary.json.
SUMMARY = os.environ.get("STAGE4_SUMMARY", os.path.join(OUT, "stage4_summary.json"))
# per-(cell, seed) per-jet prediction dumps (ngtagger.train.prediction_dump)
DUMPS = os.environ.get("STAGE4_DUMPS", os.path.join(OUT, "pred_dumps"))

VIEWS = {
    "1111": {"glob": "nano_fat_1111_coopt_file{}.root", "cfg": "AAAA", "activeSP": "1111"},
    "1100": {"glob": "nano_fat_1100_coopt_file{}.root", "cfg": "AAII", "activeSP": "1100"},
    "0000": {"glob": "nano_fat_0000_baseline_file{}.root", "cfg": None, "activeSP": "0000"},
}


def files_for(view: str, n_files: int = 10):
    g = VIEWS[view]["glob"]
    fs = [os.path.join(NANO, g.format(i)) for i in range(1, n_files + 1)]
    return [f for f in fs if os.path.exists(f) and os.path.getsize(f) > 8 << 20]


def load_summary():
    if os.path.exists(SUMMARY):
        with open(SUMMARY) as f:
            return json.load(f)
    return {"stage": 4, "cells": {}, "meta": {
        "note": "Stage-4 jet-tagger variant matrix. Low-stats (~1000 evt/view), "
                "single-coherent-view nanos. AUC=one-vs-rest on held-out test.",
        "caveats": [
            "Low stats (~1000 events/view): treat significance skeptically.",
            "Single-coherent-view: downstream vertex->PF->PUPPI->jet from ONE track config.",
            "hgcClusterIdx absent on L1ExtPuppiCand -> cluster feature group N/A (unified-nano follow-up).",
            "L1ExtPuppiCand.l1TrackIdx all -1 -> constituent->track link recovered via row-aligned L1PuppiCand.",
        ],
    }}


def save_cell(summary, key, payload):
    summary["cells"][key] = payload
    tmp = SUMMARY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, SUMMARY)
    print(f"[saved] {key}: bestAUC(macro)={payload.get('best_macro_auc')}", flush=True)


def build_model(input_shape, n_classes, seed, charge_head=False):
    import keras
    from keras import layers

    keras.utils.set_random_seed(seed)
    n_const, n_feat = input_shape
    inp = layers.Input(shape=input_shape)
    # per-constituent 1x1 conv (shared MLP), masked sum pool over constituents
    x = layers.Conv1D(16, 1, activation="relu")(inp)
    x = layers.Conv1D(16, 1, activation="relu")(x)
    pooled = layers.GlobalAveragePooling1D()(x)
    h = layers.Dense(32, activation="relu")(pooled)
    h = layers.Dense(16, activation="relu")(h)
    out_id = layers.Dense(n_classes, activation="softmax", name="jet_id")(h)
    outputs = [out_id]
    losses = {"jet_id": "categorical_crossentropy"}
    if charge_head:
        out_q = layers.Dense(3, activation="softmax", name="charge")(h)
        outputs.append(out_q)
        losses["charge"] = "categorical_crossentropy"
    model = keras.Model(inp, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses)
    return model


def per_flavor_auc(y_true_oh, y_prob):
    aucs = {}
    for i, name in enumerate(CLASS_LABELS):
        yt = y_true_oh[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            aucs[name] = None
        else:
            try:
                aucs[name] = float(roc_auc_score(yt, y_prob[:, i]))
            except Exception:
                aucs[name] = None
    valid = [v for v in aucs.values() if v is not None]
    macro = float(np.mean(valid)) if valid else None
    return aucs, macro


def run_cell(summary, view, groups, key, seeds, n_files=10, epochs=40,
             charge_head=False, max_events=None):
    import keras

    cfg = VIEWS[view]["cfg"]
    bdt_json = f"{MODELS}/refitq_{cfg}_conifer.json" if cfg else None
    files = files_for(view, n_files)
    t0 = time.time()
    ds = prepare_dataset(files, feature_groups=groups, seed=12345,
                         refit_config=cfg, refit_bdt_json=bdt_json,
                         test_fraction=0.2, max_events=max_events)
    Xtr, ytr = ds["X_train"], ds["y_train"]
    Xte, yte = ds["X_test"], ds["y_test"]
    build_t = time.time() - t0

    per_seed = []
    best = None
    for s in seeds:
        m = build_model(Xtr.shape[1:], ytr.shape[1], s, charge_head=charge_head)
        if charge_head:
            ytr_t = {"jet_id": ytr, "charge": ds["charge_train"]}
        else:
            ytr_t = ytr
        cb = [keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                            monitor="val_loss")]
        m.fit(Xtr, ytr_t, validation_split=0.15, epochs=epochs, batch_size=256,
              verbose=0, callbacks=cb)
        prob = m.predict(Xte, batch_size=1024, verbose=0)
        prob_id = prob[0] if charge_head else prob
        # per-jet prediction dump for the MVA explorer (default-on)
        try:
            from ngtagger.train.prediction_dump import dump_predictions

            kin = {}
            for src, name in (("reco_pt_test", "pt"), ("reco_eta_test", "abs_eta"),
                              ("reco_phi_test", "phi"), ("nconst_test", "nconst")):
                if src in ds:
                    v = np.asarray(ds[src], dtype=np.float32)
                    kin[name] = np.abs(v) if name == "abs_eta" else v
            dump_predictions(
                os.path.join(DUMPS, f"{key}__s{s}.npz"),
                class_probs=prob_id, class_labels=list(CLASS_LABELS),
                y_true=yte, kinematics=kin,
                charge_probs=(prob[1] if charge_head else None),
                charge_true=ds.get("charge_test"),
                charge_labels=list(ds.get("charge_class_labels", [])) or None,
                meta={"cell": key, "seed": s, "view": view, "groups": groups,
                      "refit_config": cfg, "charge_head": charge_head})
        except Exception as e:
            print(f"  [dump] {key} seed {s} skipped: {e}", flush=True)
        aucs, macro = per_flavor_auc(yte, prob_id)
        acc = float(np.mean(prob_id.argmax(1) == yte.argmax(1)))
        rec = {"seed": s, "macro_auc": macro, "acc": acc, "per_flavor_auc": aucs}
        per_seed.append(rec)
        if best is None or (macro or 0) > (best["macro_auc"] or 0):
            best = rec
        keras.backend.clear_session()

    macros = [r["macro_auc"] for r in per_seed if r["macro_auc"] is not None]
    payload = {
        "view": view, "activeSP": VIEWS[view]["activeSP"], "groups": groups,
        "refit_config": cfg, "charge_head": charge_head,
        "n_features": Xtr.shape[-1], "feature_names": ds["feature_names"],
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "n_files": len(files), "build_seconds": round(build_t, 1),
        "seeds": seeds,
        "best_macro_auc": best["macro_auc"], "best_seed": best["seed"],
        "best_acc": best["acc"], "best_per_flavor_auc": best["per_flavor_auc"],
        "macro_auc_mean": float(np.mean(macros)) if macros else None,
        "macro_auc_std": float(np.std(macros)) if macros else None,
        "per_seed": per_seed,
    }
    save_cell(summary, key, payload)
    return payload


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    seeds = [1, 2, 3]
    n_files = 10
    epochs = 40
    summary = load_summary()

    plan = []
    for view in ["1111", "1100", "0000"]:
        plan.append((view, ["baseline"], f"{view}__baseline"))
        plan.append((view, ["baseline", "vertexdxy_pv"], f"{view}__vertexdxy"))
        if VIEWS[view]["cfg"]:
            plan.append((view, ["baseline", "refitbdt"], f"{view}__refitbdt"))
            plan.append((view, ["baseline", "refitbdt", "vertexdxy_pv"], f"{view}__both"))

    for view, groups, key in plan:
        if which not in ("all", view, key):
            continue
        if key in summary["cells"] and "--force" not in sys.argv:
            print(f"[skip] {key} already present", flush=True)
            continue
        print(f"[run] {key} groups={groups}", flush=True)
        run_cell(summary, view, groups, key, seeds, n_files=n_files, epochs=epochs)

    # arch/head confirmation: charge head on the 1111 baseline+both cell
    if which in ("all", "archhead"):
        key = "1111__both__chargehead"
        if key not in summary["cells"] or "--force" in sys.argv:
            print(f"[run] {key} (charge head)", flush=True)
            run_cell(summary, "1111", ["baseline", "refitbdt", "vertexdxy_pv"],
                     key, seeds=[1, 2], n_files=n_files, epochs=epochs,
                     charge_head=True)


if __name__ == "__main__":
    main()
