"""Export the per-track track-quality heads for CMSSW (heads_cmssw/SPEC.md): conifer models and a parity file.

  TAGGER_SP_OUT=<heads_cmssw>/work python eval_refitq/tagger_sp/export_heads.py train    # reference heads (lambda 0)
  TAGGER_SP_OUT=<heads_cmssw>/work python eval_refitq/tagger_sp/export_heads.py models   # conifer JSON + MANIFEST
  TAGGER_SP_OUT=<heads_cmssw>/work python eval_refitq/tagger_sp/export_heads.py parity   # AAAA file 4, 5 events

The heads are joint_heads.py's (its features, targets, windows and training events). "train" fits the uncoupled
reference (lambda = 0: six independent heads in one model) of every mask and window; "models" splits each
window's six-output booster into six single-output boosters (conifer's multi-output conversion sums all trees into
one output) and writes them as conifer JSON, checked against the booster with conifer's C++ backend; "parity"
writes, for the first 5 events (entry order) of canon_v5 AAAA file 4, every track's feature vector and every
head's output for the OT collection (the OT-only heads, trained on IIII) and the SmartPixels collection (the
AAAA refit heads), keyed by (run, lumi, event, collection, track index along axis 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import joint_heads as J  # noqa: E402

DEST = os.environ.get("HEADS_CMSSW", "/Volumes/WDMac/smartpixels_scratch/heads_cmssw")
COLL_NAME = {"base": "OT", "refit": "SmartPixels"}
TABLE = {"base": "L1TExtTrack", "refit": "L1TSmartPixelsExtTrack"}
PARITY = ("AAAA", 4, 5)                    # mask, TTbar file number, events


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def windows_of(coll):
    from ngtagger.train import trackmva as tm

    return tm.window_names(coll)


def stage_train(masks):
    J.GRID = [0.0]                         # the reference only (settings() = [("ref", 0.0)])
    J.XGB["nthread"] = int(os.environ.get("NTHREAD", "2"))
    J.stage_extra(masks)
    J.stage_fit(masks)


def booster(mask, window):
    import xgboost as xgb

    b = xgb.Booster()
    b.load_model(os.path.join(J.jdir("fits", mask, window), J.sname("ref", 0.0) + ".json"))
    return b


def stage_models(masks, n_check=2000):
    import conifer
    import conifer.backends.cpp.writer as W
    import xgboost as xgb

    W._json_include = lambda: f"-I{os.environ.get('JSON_ROOT', '/opt/homebrew/include')}"
    man = {"what": "per-track track-quality heads (SPEC.md); conifer JSON, one single-output BDT per head and window",
           "evaluation": "margin = init_predict + sum of the trees' leaf values (conifer); clean: P = 1/(1+exp(-margin)); "
                         "the five width heads: margin = log sigma_hat in the head's unit (SPEC.md)",
           "heads": [h[0] for h in J.head_list()], "units": J.UNIT, "missing_value": J.MISSING,
           "xgboost": __import__("xgboost").__version__, "conifer": str(conifer.__version__), "models": {}, "checks": {}}
    for mask in masks:
        coll = J.coll_of(mask)
        T, X, y, R, k, Wn = J.mask_problem(mask)
        names = J.feature_names(T, coll)
        for w in windows_of(coll):
            meta = json.load(open(os.path.join(J.jdir("fits", mask, w), "meta.json")))
            b = booster(mask, w)
            te = Wn[w][2][:n_check]
            full = b.predict(xgb.DMatrix(X[te]), output_margin=True)
            d = os.path.join(DEST, "models", COLL_NAME[coll], mask, w)
            os.makedirs(d, exist_ok=True)
            for j, h in enumerate(man["heads"]):
                bj = J.split_booster(b, j)
                cfg = conifer.backends.cpp.auto_config()
                cfg["OutputDir"] = os.path.join(DEST, "work", "conifer_build", mask, w, h)
                cfg["Precision"] = "float"
                cm = conifer.converters.convert_from_xgboost(bj, cfg)
                cm.compile()
                pc = cm.decision_function(X[te].astype(np.float32)).ravel()
                p = os.path.join(d, f"{h}.json")
                cm.save(p)
                key = f"{COLL_NAME[coll]}/{mask}/{w}/{h}"
                man["models"][key] = {"file": os.path.relpath(p, DEST), "sha256": sha(p), "n_trees": len(cm.trees),
                                      "n_features": len(names)}
                man["checks"][key] = {"n_tracks": int(len(te)),
                                      "max_abs_conifer_vs_booster": float(np.abs(pc - full[:, j]).max())}
            man.setdefault("features", {})[COLL_NAME[coll]] = names
            man.setdefault("training", {})[f"{COLL_NAME[coll]}/{mask}/{w}"] = {
                k2: meta[k2] for k2 in ("n_train", "n_val", "n_test", "k_bend", "base", "rounds", "xgb")}
            print(f"models {COLL_NAME[coll]}/{mask}/{w}: {len(man['heads'])} heads, max |conifer - booster| "
                  f"{max(man['checks'][f'{COLL_NAME[coll]}/{mask}/{w}/{h}']['max_abs_conifer_vs_booster'] for h in man['heads']):.2e}",
                  flush=True)
    man["windows"] = {"OT": "one window (all tracks)",
                      "SmartPixels": "prompt: |d0| <= 0.02 cm, displaced: |d0| > 0.02 cm; d0 = the SmartPixels refit "
                                     "track's d0 (L1TSmartPixelsExtTrack_d0, cm)"}
    json.dump(man, open(os.path.join(DEST, "models", "MANIFEST.json"), "w"), indent=1)


def stage_parity():
    import awkward as ak
    import uproot
    import xgboost as xgb

    from ngtagger.train import trackmva as tm

    mask, n, nev = PARITY
    fn = C.path("ttbar", mask, n)
    t = uproot.open(fn)["Events"]
    br = tm.nano_branches(t.keys(), C.FAMILY, "", with_truth=False, with_event_id=True)
    A = t.arrays(br, entry_stop=nev)
    T, ntrk = tm.tracks_from_batch(A, C.FAMILY, "", with_truth=False)
    ev = np.column_stack([ak.to_numpy(A[c]).astype(np.int64) for c in ("run", "luminosityBlock", "event")])
    ent = np.repeat(np.arange(nev), ntrk)
    idx = np.concatenate([np.arange(k) for k in ntrk])
    heads = [h[0] for h in J.head_list()]
    recs, arrays = [], {}
    for coll, model_mask in (("base", C.BASELINE_MASK), ("refit", mask)):
        X = J.features(T, coll)
        names = J.feature_names(T, coll)
        win = {w: m for w, m in tm.windows(T, coll).items()}
        F = np.full((len(X), len(heads)), np.nan)
        wname = np.empty(len(X), object)
        for w, m in win.items():
            b = booster(model_mask, w)
            F[m] = b.predict(xgb.DMatrix(X[m]), output_margin=True)
            wname[m] = w
        for i in range(len(X)):
            recs.append({"run": int(ev[ent[i], 0]), "lumi": int(ev[ent[i], 1]), "event": int(ev[ent[i], 2]),
                         "collection": COLL_NAME[coll], "table": TABLE[coll], "track_index": int(idx[i]),
                         "window": wname[i], "model": f"{COLL_NAME[coll]}/{model_mask}/{wname[i]}",
                         "features": {nm: float(X[i, j]) for j, nm in enumerate(names)},
                         "margin": {h: float(F[i, j]) for j, h in enumerate(heads)},
                         "p_clean": float(1 / (1 + np.exp(-F[i, 0]))),
                         "sigma_hat": {h: float(np.exp(F[i, j])) for j, h in enumerate(heads) if j > 0}})
        arrays[COLL_NAME[coll]] = {"X": X.astype(np.float64), "margin": F, "event_id": ev[ent], "track_index": idx,
                                   "window": wname.astype(str), "feature_names": np.array(names)}
    os.makedirs(os.path.join(DEST, "parity"), exist_ok=True)
    out = {"what": "per-track parity reference for the CMSSW heads (SPEC.md)", "file": fn,
           "events": [{"entry": e, "run": int(r[0]), "lumi": int(r[1]), "event": int(r[2]), "n_tracks": int(k)}
                      for e, (r, k) in enumerate(zip(ev, ntrk))],
           "models": "models/MANIFEST.json", "heads": heads, "tracks": recs}
    json.dump(out, open(os.path.join(DEST, "parity", "parity_AAAA_f04_5ev.json"), "w"))
    for c, a in arrays.items():
        np.savez_compressed(os.path.join(DEST, "parity", f"parity_AAAA_f04_5ev_{c}.npz"), heads=np.array(heads), **a)
    print(f"parity: {len(recs)} tracks ({nev} events), {fn}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "models", "parity"])
    ap.add_argument("--masks", nargs="+", default=C.MASKS)
    a = ap.parse_args()
    if a.stage == "train":
        stage_train(a.masks)
    elif a.stage == "models":
        stage_models(a.masks)
    else:
        stage_parity()


if __name__ == "__main__":
    main()
