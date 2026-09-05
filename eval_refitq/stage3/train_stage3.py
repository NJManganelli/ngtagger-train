#!/usr/bin/env python
"""STAGE 3: retrain the refit-quality BDT across all 15 SmartPixels configs on
the fixed/clean pG nano (post cross-layer sign-bug fix).

Reuses the established REFIT_BDT_FEATURES spec path (ngtagger.train.refitquality:
build_spec_dataset / conifer export). Runs best-of-N seeds per config, keeps the
best by validation AUC, saves the winning xgboost + conifer JSON per config for
Stage 4 (on-the-fly tagger feature).

5-par framing: trained on the prompt L1TTrack refit collection
(L1TSmartPixelsTrackDigiRefit<CFG>) = 5-par OT-only reference vs 5-par OT+IT refit.
promptHnpar=5 => seed_npar=5, track_npar=5 (ref d0 is the real seed d0).

Feature set: REFIT_BDT_FEATURES v1 (24 features) = the 17 v0 sidecar/kick/seed
features + the classic-7 TrackQuality hw features of the reference track. v1 is
the richest producer-loadable contract and what Stage 4 will evaluate per-track.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

import ngtagger.train.refitquality as rq

# --- extend the 4-config harness to all 15 co-registered configs ------------
# suffix A=active(1) I=inactive(0), 4 chars = TBPX L1 L2 L3 L4.
ALL_CONFIGS = [
    "AAAA", "AAAI", "AAIA", "AAII", "AIAA", "AIAI", "AIIA", "AIII",
    "IAAA", "IAAI", "IAIA", "IAII", "IIAA", "IIAI", "IIIA",
]
def _activesp(sfx: str) -> str:
    return "".join("1" if c == "A" else "0" for c in sfx)

rq.SMARTPIXELS_CONFIGS = tuple(ALL_CONFIGS)
rq.CONFIG_ACTIVESP = {c: _activesp(c) for c in ALL_CONFIGS}

# combinatoric report order requested: 1000,0100,0010,0001,1100,... ascending by
# integer value of the 4-bit activeSP mask, then map back to suffix.
def _report_order():
    order = []
    for v in range(1, 16):
        bits = f"{v:04b}"  # L1 L2 L3 L4
        sfx = "".join("A" if b == "1" else "I" for b in bits)
        order.append((sfx, bits))
    return order  # 0001..1111 => but requested "1000,0100,...,1111"


def best_of_n(files, config, out_dir, seeds, spec_version, test_fraction,
              seed_npar, track_npar, label):
    """Load tables once, run N seeds, keep the best model by validation AUC.
    Saves winning xgboost + conifer JSON + meta for the config. Returns dict."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    ref, var, hits = rq.load_refit_tables(files, config, "L1TTrack", None)
    X, y, names, aux = rq.build_spec_dataset(
        ref, var, hits, config, label=label, seed_npar=seed_npar,
        track_npar=track_npar, spec_version=spec_version)
    n_pos_all = int(y.sum()); n_neg_all = int((y == 0).sum())

    best = None
    per_seed = []
    for s in seeds:
        train, test = rq._split(len(X), test_fraction, s)
        params = rq._spec_xgb_params(None)
        n_pos = max(int(y[train].sum()), 1)
        n_neg = max(int((y[train] == 0).sum()), 1)
        params.setdefault("scale_pos_weight", n_neg / n_pos)
        model = xgb.XGBClassifier(**params, random_state=s)
        model.fit(X[train], y[train], eval_set=[(X[test], y[test])], verbose=False)
        proba = model.predict_proba(X[test])[:, 1]
        auc = (float(roc_auc_score(y[test], proba))
               if (y[test].sum() and (y[test] == 0).sum()) else float("nan"))
        per_seed.append((s, auc))
        if best is None or (auc == auc and auc > best["auc"]):
            best = {"auc": auc, "seed": s, "model": model,
                    "n_train": len(train), "n_test": len(test)}

    # save the winning model (xgboost + conifer) for Stage 4
    os.makedirs(out_dir, exist_ok=True)
    xgb_path = os.path.join(out_dir, f"refitq_{config}_xgb.json")
    best["model"].save_model(xgb_path)

    conifer_path = None
    margin_check = None
    try:
        import conifer
        booster = best["model"].get_booster()
        cfg = conifer.backends.cpp.auto_config()
        cfg["OutputDir"] = os.path.join(out_dir, f"conifer_{config}_prj")
        cnf = conifer.converters.convert_from_xgboost(booster, cfg)
        conifer_path = os.path.join(out_dir, f"refitq_{config}_conifer.json")
        cnf.save(conifer_path)
        with open(conifer_path) as fh:
            cj = json.load(fh)
        nfeat_expected = rq._SPEC_NFEAT[spec_version]
        if int(cj.get("n_features", -1)) != nfeat_expected:
            raise RuntimeError(f"conifer n_features {cj.get('n_features')} != {nfeat_expected}")
        # offline margin parity self-check on the winning split's test rows
        train, test = rq._split(len(X), test_fraction, best["seed"])
        walk = rq.conifer_json_walk(cj, X[test])
        bm = booster.predict(xgb.DMatrix(X[test]), output_margin=True).astype(np.float32)
        margin_check = float(np.max(np.abs(walk - bm))) if len(test) else 0.0
    except Exception as e:  # keep xgb model even if conifer export hiccups
        print(f"  [{config}] conifer export warning: {e}", file=sys.stderr)

    meta = {
        "stage": 3, "config": config, "activeSP": rq.CONFIG_ACTIVESP[config],
        "spec": "RefitSidecarSpec.md REFIT_BDT_FEATURES", "spec_version": spec_version,
        "n_features": len(names), "features": names,
        "label": label, "track_table": "L1TTrack",
        "framing": "5-par OT-only ref vs 5-par OT+IT refit (promptHnpar=5)",
        "seed_npar": seed_npar, "track_npar": track_npar,
        "margin_semantics": "raw_logit_margin",
        "best_seed": best["seed"], "best_val_auc": best["auc"],
        "per_seed_auc": {str(s): a for s, a in per_seed},
        "n_train": best["n_train"], "n_test": best["n_test"],
        "n_refit_tracks": int(len(X)), "n_pos": n_pos_all, "n_neg": n_neg_all,
        "params": {k: v for k, v in rq._spec_xgb_params(None).items()},
        "conifer_margin_selfcheck": margin_check,
        "input_files": [os.path.basename(f) for f in files],
        "xgb_path": os.path.abspath(xgb_path),
        "conifer_path": os.path.abspath(conifer_path) if conifer_path else None,
        "caveat": ("Stats-limited: 1000 events, few-hundred-fake minority class per config "
                   "scaled by refit-performed multiplicity. Best-of-N by val AUC."),
    }
    with open(os.path.join(out_dir, f"refitq_{config}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nano-dir", default="/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano")
    ap.add_argument("--pattern", default="nano_pG_PFTrkSmartPix_withGen_file{i}.root")
    ap.add_argument("--n-files", type=int, default=10)
    ap.add_argument("--out-dir", default="eval_refitq/stage3/models")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--spec-version", type=int, default=1)
    ap.add_argument("--test-fraction", type=float, default=0.25)
    ap.add_argument("--label", default="genuine")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="explicit suffix list; default = physics-priority then rest")
    args = ap.parse_args()

    files = [os.path.join(args.nano_dir, args.pattern.format(i=i))
             for i in range(1, args.n_files + 1)]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        sys.exit("no input files found")

    # priority order: physics-relevant first (1000,1100,1110,1111,0100), then rest
    priority_bits = ["1000", "1100", "1110", "1111", "0100"]
    def bits2sfx(b): return "".join("A" if x == "1" else "I" for x in b)
    if args.configs:
        run_order = args.configs
    else:
        prio = [bits2sfx(b) for b in priority_bits]
        rest = [c for c in ALL_CONFIGS if c not in prio]
        run_order = prio + rest

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"stage3: {len(files)} files, v{args.spec_version} features, seeds={args.seeds}, "
          f"{len(run_order)} configs")
    results = {}
    for i, cfg in enumerate(run_order, 1):
        t0 = time.time()
        meta = best_of_n(files, cfg, args.out_dir, args.seeds, args.spec_version,
                         args.test_fraction, seed_npar=5, track_npar=5, label=args.label)
        results[cfg] = meta
        dt = time.time() - t0
        print(f"[{i:2d}/{len(run_order)}] {cfg} ({meta['activeSP']}): "
              f"best_val_AUC={meta['best_val_auc']:.4f} seed={meta['best_seed']} "
              f"n_refit={meta['n_refit_tracks']} pos={meta['n_pos']} neg={meta['n_neg']} "
              f"margin_chk={meta['conifer_margin_selfcheck']} ({dt:.1f}s)")
        # persist rolling summary after each config
        with open(os.path.join(args.out_dir, "stage3_summary.json"), "w") as fh:
            json.dump({"n_files": len(files), "spec_version": args.spec_version,
                       "seeds": args.seeds, "test_fraction": args.test_fraction,
                       "results": {c: {"activeSP": m["activeSP"],
                                       "best_val_auc": m["best_val_auc"],
                                       "best_seed": m["best_seed"],
                                       "n_refit_tracks": m["n_refit_tracks"],
                                       "n_pos": m["n_pos"], "n_neg": m["n_neg"]}
                                   for c, m in results.items()}}, fh, indent=2)

    # final matrix in requested combinatoric order 1000,0100,0010,0001,1100,...,1111
    print("\n==== CONFIG x AUC (best-of-N) ====")
    order = ["1000", "0100", "0010", "0001", "1100", "1010", "1001",
             "0110", "0101", "0011", "1110", "1101", "1011", "0111", "1111"]
    print(f"{'activeSP':>9} {'suffix':>7} {'bestAUC':>8} {'seed':>5} {'nRefit':>7} {'nPos':>6} {'nNeg':>5}")
    for b in order:
        sfx = bits2sfx(b)
        if sfx in results:
            m = results[sfx]
            print(f"{b:>9} {sfx:>7} {m['best_val_auc']:>8.4f} {m['best_seed']:>5} "
                  f"{m['n_refit_tracks']:>7} {m['n_pos']:>6} {m['n_neg']:>5}")
    best_cfg = max(results.items(), key=lambda kv: (kv[1]["best_val_auc"]
                   if kv[1]["best_val_auc"] == kv[1]["best_val_auc"] else -1))
    print(f"\nBEST config: {best_cfg[0]} ({best_cfg[1]['activeSP']}) "
          f"val AUC {best_cfg[1]['best_val_auc']:.4f}")


if __name__ == "__main__":
    main()
