"""Study 2: learning curve by training-set subsampling, views 1111 & 0000,
baseline feature group, fractions {1/16, 1/8, 1/4, 1/2, 1} x seeds {1,2,3},
FIXED full Stage-4 test split (prepare_dataset seed 12345, test_fraction 0.2).

Reuses the stage-4 runner's model + protocol (import run_matrix: build_model,
per_flavor_auc, files_for, VIEWS/MODELS). Epoch budget scales as 40/frac
(capped 320) so small-N runs see a comparable number of optimizer steps; at
frac=1 the protocol is exactly stage-4 (epochs=40). EarlyStopping identical.

Subsampling is stratified by class (>=1 jet/class) with a per-(fraction,seed)
rng, so rare classes never vanish.

Incremental output: calibration/learning_curve.json
Prediction dumps:   calibration/dumps/lc_<view>_f<i>of16_s<seed>.npz
Usage: 02_learning_curve.py [1111|0000|all]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(CAL), "scripts"))
import run_matrix  # noqa: E402

from ngtagger.data.labels import CLASS_LABELS  # noqa: E402
from ngtagger.train.prediction_dump import dump_predictions  # noqa: E402
from ngtagger.train.trainer import prepare_dataset  # noqa: E402

OUT_JSON = os.path.join(CAL, "learning_curve.json")
DUMPS = os.path.join(CAL, "dumps")
FRACTIONS = [(1, 16), (1, 8), (1, 4), (1, 2), (1, 1)]
SEEDS = [1, 2, 3]


def load_results():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            return json.load(f)
    return {"meta": {"fractions": FRACTIONS, "seeds": SEEDS,
                     "protocol": "stage-4 model/fit; epochs=min(320,40/frac); "
                                 "EarlyStopping(val_loss, patience=8); "
                                 "fixed full test split (seed 12345, 0.2)"},
            "runs": {}}


def save_run(res, key, payload):
    res["runs"][key] = payload
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, OUT_JSON)
    print(f"[saved] {key}: macro={payload['macro_auc']:.4f} "
          f"n_train={payload['n_train']}", flush=True)


def stratified_subsample(y_onehot: np.ndarray, frac: float, rng) -> np.ndarray:
    labels = y_onehot.argmax(1)
    take = []
    for c in range(y_onehot.shape[1]):
        idx = np.flatnonzero(labels == c)
        if len(idx) == 0:
            continue
        k = max(1, int(round(frac * len(idx))))
        take.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    out = np.concatenate(take)
    rng.shuffle(out)
    return out


def run_view(res, view: str):
    import keras

    cfg = run_matrix.VIEWS[view]["cfg"]
    bdt_json = f"{run_matrix.MODELS}/refitq_{cfg}_conifer.json" if cfg else None
    files = run_matrix.files_for(view)
    t0 = time.time()
    ds = prepare_dataset(files, feature_groups=["baseline"], seed=12345,
                         refit_config=cfg, refit_bdt_json=bdt_json,
                         test_fraction=0.2)
    print(f"[{view}] dataset built in {time.time()-t0:.0f}s "
          f"n_train={len(ds['X_train'])} n_test={len(ds['X_test'])}", flush=True)
    Xtr_full, ytr_full = ds["X_train"], ds["y_train"]
    Xte, yte = ds["X_test"], ds["y_test"]

    for num, den in FRACTIONS:
        frac = num / den
        for s in SEEDS:
            key = f"{view}__f{num}of{den}__s{s}"
            if key in res["runs"]:
                print(f"[skip] {key}", flush=True)
                continue
            sub_rng = np.random.default_rng(100000 * den + 1000 * num + s)
            if frac >= 1.0:
                sel = np.arange(len(Xtr_full))
            else:
                sel = stratified_subsample(ytr_full, frac, sub_rng)
            Xtr, ytr = Xtr_full[sel], ytr_full[sel]
            counts = {c: int((ytr.argmax(1) == i).sum())
                      for i, c in enumerate(CLASS_LABELS)}
            epochs = min(320, int(round(40 / frac)))
            t1 = time.time()
            m = run_matrix.build_model(Xtr.shape[1:], ytr.shape[1], s)
            cb = [keras.callbacks.EarlyStopping(patience=8,
                                                restore_best_weights=True,
                                                monitor="val_loss")]
            hist = m.fit(Xtr, ytr, validation_split=0.15, epochs=epochs,
                         batch_size=256, verbose=0, callbacks=cb)
            prob = m.predict(Xte, batch_size=1024, verbose=0)
            aucs, macro = run_matrix.per_flavor_auc(yte, prob)
            acc = float(np.mean(prob.argmax(1) == yte.argmax(1)))
            kin = {}
            for src, name in (("reco_pt_test", "pt"), ("reco_eta_test", "abs_eta"),
                              ("reco_phi_test", "phi"), ("nconst_test", "nconst")):
                if src in ds:
                    v = np.asarray(ds[src], dtype=np.float32)
                    kin[name] = np.abs(v) if name == "abs_eta" else v
            dump_predictions(
                os.path.join(DUMPS, f"lc_{view}_f{num}of{den}_s{s}.npz"),
                class_probs=prob, class_labels=list(CLASS_LABELS),
                y_true=yte, kinematics=kin,
                meta={"study": "calibration_learning_curve", "view": view,
                      "fraction": frac, "seed": s, "n_train": int(len(Xtr))})
            save_run(res, key, {
                "view": view, "fraction": frac, "seed": s,
                "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                "train_class_counts": counts,
                "epochs_budget": epochs, "epochs_ran": len(hist.history["loss"]),
                "macro_auc": macro, "acc": acc, "per_flavor_auc": aucs,
                "fit_seconds": round(time.time() - t1, 1),
            })
            keras.backend.clear_session()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = load_results()
    for view in ["1111", "0000"]:
        if which in ("all", view):
            run_view(res, view)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
