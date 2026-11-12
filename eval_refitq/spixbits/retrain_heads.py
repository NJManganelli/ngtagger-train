"""Retrain the six per-track quality heads the tagger consumes (heads_cmssw/SPEC.md: clean and log sigma_hat of d0, z0,
tanL, 1/R, phi; one model set per window) separately for the detector scenarios AAAA, AAAI and IAAA: 20 trials each,
the single best kept (physicist, 2026-11-11).

  TAGGER_SP_OUT=/Volumes/WDMac/smartpixels_scratch/tagger_v5 python retrain_heads.py [--masks AAAA AAAI IAAA] [--trials 20]

Pipeline: the deployed heads' own training code, eval_refitq/tagger_sp/joint_heads.py (Problem, fit at lambda = 0: the
uncoupled reference; XGBoost hist, depth 3, 100 rounds, learning rate 0.1, Fisher scoring for the width heads,
residuals winsorised at 99.5 %, missing = -999), on its own track caches: today's ("implemented now", CMSSW refit v3)
extended-collection tracks of canon_v5 TTbar files 1-10 and H->tautau files 1-5, 7, 8 of the detector scenario.
Events: the tagger study's split by event identity; training = neither test nor validation. File roles: TTbar file 4
(known-good / development) and file 6 (gate) are left out of the training.
Trials: trial 0 is the deterministic reference recipe (on the full training set it is the deployed model). Trials
1-19 add row subsampling 0.8 and column subsampling 0.8 per node with seed = trial (a hist booster without
subsampling does not depend on its seed).
Selection, per detector scenario and window, on the validation events only: the trial with the largest information
(nats per track, the heads' loss gain over a constant) summed over the five regression heads; a trial with a
non-finite validation loss is failed and discarded. The clean head comes from the same trial. All trials are recorded.
Outputs under ~/smartpixels/scratch_local/spixbits/heads_retrained/: trials.json; fits/<scenario>/<window>/trial_NN.json
(the boosters); models/SmartPixels/<scenario>/<window>/<head>.json (the best trial as conifer JSON, the layout of
heads_cmssw/models, read by heads.Heads), each checked against its booster with the independent reader.
No truth column is a model input (joint_heads uses trackmva.feature_matrix; the caches were built under
trackmva.refuse_truth_inputs).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tagger_sp"))
os.environ.setdefault("TAGGER_SP_OUT", "/Volumes/WDMac/smartpixels_scratch/tagger_v5")
import common as C          # noqa: E402  (tagger_sp/common.py)
import joint_heads as J     # noqa: E402

OUT = Path.home() / "smartpixels/scratch_local/spixbits/heads_retrained"
LEFT_OUT = {("ttbar", 4), ("ttbar", 6)}
REG = slice(1, 6)                                     # the five regression heads among the six outputs


def load(mask):
    """joint_heads.load_mask_tracks plus, per track, whether its file is left out of the training."""
    import build as B
    man = json.load(open(C.manifest_path(mask)))
    parts = []
    for si, s in enumerate(man["samples"]):
        fs = [(n, p) for s2, n, p in man["files"] if s2 == s]
        loaded = {n: B.load_tracks(os.path.join(C.cache_dir("tracks"), C.tag(s, mask, n) + ".npz")) for n, _ in fs}
        evids = {n: loaded[n][2] for n, _ in fs}
        test = C.split_table(s, evids)
        val = C.val_table(s, evids, test)
        for n, p in fs:
            T, ntrk, evid = loaded[n]
            T.update(dict(np.load(J.build_extra(s, mask, n, p))))
            ent = np.repeat(np.arange(len(ntrk)), ntrk)
            T["_test"], T["_val"] = test[n][ent], val[n][ent]
            T["_left_out"] = np.full(len(ent), (s, n) in LEFT_OUT)
            parts.append(T)
    keys = set.intersection(*(set(p) for p in parts))
    return {k: np.concatenate([p[k] for p in parts]) for k in sorted(keys)}, man


def conifer_json(booster, k, path):
    import conifer
    cfg = conifer.backends.cpp.auto_config()
    cfg["OutputDir"] = str(OUT / "conifer_build")
    cfg["Precision"] = "float"
    conifer.converters.convert_from_xgboost(J.split_booster(booster, k), cfg).save(str(path))


def main():
    import xgboost as xgb
    from ngtagger.train import trackmva as tm
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", nargs="+", default=["AAAA", "AAAI", "IAAA"])
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT / "trials.json"
    rec = json.load(open(rec_path)) if rec_path.exists() else {}
    heads = [h[0] for h in J.head_list()]
    base_xgb = dict(J.XGB)
    for mask in a.masks:
        T, man = load(mask)
        y, R, k_bend = J.targets(T, "refit")
        X = J.features(T, "refit")
        for wname, w in tm.windows(T, "refit").items():
            tr = np.flatnonzero(w & ~T["_test"] & ~T["_val"] & ~T["_left_out"])
            tr_all = np.flatnonzero(w & ~T["_test"] & ~T["_val"])
            va = np.flatnonzero(w & T["_val"] & ~T["_left_out"])
            P = J.Problem(y, R, tr)
            d = OUT / "fits" / mask / wname
            d.mkdir(parents=True, exist_ok=True)
            key = f"{mask}/{wname}"
            e = rec.setdefault(key, {"n_train": int(len(tr)), "n_train_with_files_4_6": int(len(tr_all)), "n_val": int(len(va)),
                                     "k_bend": k_bend, "base": P.base.tolist(), "heads": heads, "trials": {}})
            for t in range(a.trials):
                if str(t) in e["trials"] and (d / f"trial_{t:02d}.json").exists():
                    continue
                J.XGB.clear()
                J.XGB.update(base_xgb, seed=t, **({"subsample": 0.8, "colsample_bynode": 0.8} if t else {}))
                t0 = time.time()
                b = J.fit(X, P, tr, "A", 0.0)
                Fv = J.predict(b, X[va])
                info, loss = P.info(Fv, va), P.losses(Fv, va)
                ok = bool(np.isfinite(loss).all() and np.isfinite(Fv).all())
                b.save_model(str(d / f"trial_{t:02d}.json"))
                e["trials"][str(t)] = {"seed": t, "recipe": "reference (deterministic)" if t == 0 else "subsample 0.8, colsample_bynode 0.8",
                                       "info_val": info.tolist(), "loss_val": loss.tolist(), "failed": not ok,
                                       "score": float(info[REG].sum()) if ok else None, "fit_s": time.time() - t0}
                json.dump(rec, open(rec_path, "w"), indent=1)
                print(f"{key} trial {t:2d}: info {np.round(info, 4)} sum(regression) {info[REG].sum():.4f} {time.time() - t0:.0f}s", flush=True)
            J.XGB.clear()
            J.XGB.update(base_xgb)
            good = {int(t): v for t, v in e["trials"].items() if not v["failed"]}
            best = max(good, key=lambda t: good[t]["score"])
            sc = np.array([v["score"] for v in good.values()])
            e.update(best_trial=best, best_score=good[best]["score"], reference_score=good[0]["score"] if 0 in good else None,
                     score_spread={"min": float(sc.min()), "median": float(np.median(sc)), "max": float(sc.max()),
                                   "sd": float(sc.std())}, n_failed=len(e["trials"]) - len(good))
            # the best trial as conifer JSON, checked against the booster with the independent reader
            b = xgb.Booster()
            b.load_model(str(d / f"trial_{best:02d}.json"))
            md = OUT / "models" / "SmartPixels" / mask / wname
            md.mkdir(parents=True, exist_ok=True)
            for k, h in enumerate(heads):
                conifer_json(b, k, md / f"{h}.json")
            json.dump(rec, open(rec_path, "w"), indent=1)
            print(f"{key}: best trial {best} score {good[best]['score']:.4f} (reference {e['reference_score']:.4f}, "
                  f"spread {e['score_spread']})", flush=True)
        # the reader's check: the conifer JSON of both windows against the boosters on the validation tracks
        sys.path.insert(0, str(HERE))
        import heads as HD
        H = HD.Heads(OUT / "models", mask)
        va_all = np.flatnonzero(T["_val"])[:20000]
        mine = H.margins(X[va_all], T["r_d0"][va_all])
        ref = np.full_like(mine, np.nan)
        for wname, w in tm.windows(T, "refit").items():
            b = xgb.Booster()
            b.load_model(str(OUT / "fits" / mask / wname / f"trial_{rec[f'{mask}/{wname}']['best_trial']:02d}.json"))
            m = w[va_all]
            if m.any():
                ref[m] = b.predict(xgb.DMatrix(X[va_all][m].astype(np.float32)), output_margin=True)
        rec[f"{mask}/reader_check"] = {"tracks": int(len(va_all)), "max_abs_conifer_reader_vs_booster": float(np.nanmax(np.abs(mine - ref)))}
        rec[f"{mask}/training_files"] = [[s, n] for s, n, _ in man["files"] if (s, n) not in LEFT_OUT]
        json.dump(rec, open(rec_path, "w"), indent=1)
        print(mask, rec[f"{mask}/reader_check"], flush=True)


if __name__ == "__main__":
    main()
