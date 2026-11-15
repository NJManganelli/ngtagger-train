"""Evaluation of the heads retrained per quantization scenario (retrain_heads_q.py), after the statistics review
(reviews/stats-review-03-heads-per-quantization.md). Nothing is retrained except the small training-sample spread.

  python heads_q_eval.py run [--scen AAAA AAAI IAAA]      one process per detector scenario
  python heads_q_eval.py one --scen AAAA [--allocs ...]
      -> eval/<detector scenario>/<quantization scenario>.json (+ .npz: per-test-event loss sums), eval/<scen>/train_spread.json

Per quantization scenario, detector scenario and window (routed by the scenario's own |d0|), on the TEST events:
  a    the scenario's own best heads (the stored model file) on its inputs
  b    the five-file "implemented now" reference heads (best trial) on the scenario's inputs
  b'   b with ONE recalibration constant per head, fitted by the heads' own loss on the scenario's training tracks:
       width heads sigma'^2 = sigma^2 + kappa (kappa >= 0) or sigma' = sigma e^c, the better on validation; clean: a
       logit offset
  bm   the mean per-track loss over the reference's 20 trials
  c    the reference on today's inputs and today's targets (today's window routing: another population)
Same problem (a, b, b', bm): the per-track loss with the scenario's own winsorisation cap; G = L(b) - L(a) =
[L(b) - L(b')] recalibration + [L(b') - L(a)] retraining proper.
Against today (a, b, b', c): the absolute per-track loss log sigma_hat + r^2 / (2 sigma_hat^2) (cross-entropy for clean)
with the COMMON cap = the larger of the scenario's and today's training caps per head and window; never information.
Stored per test event: the sums and counts of every one of these losses per head (for curve-level bootstraps).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import retrain_heads_q as RQ

HERE = Path(__file__).resolve().parent
WIN = ("prompt", "displaced")
N_TRAIN_RESAMPLES = 5


def track_losses(F, y, R, cap):
    """(n, 6) per-track loss: cross-entropy of the clean head; F + r^2 e^(-2F) / 2 for the width heads with the
    residual clipped at cap (NaN for a track without a particle)."""
    out = np.full(F.shape, np.nan)
    p = np.clip(1 / (1 + np.exp(-F[:, 0])), 1e-12, 1 - 1e-12)
    out[:, 0] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    r = np.clip(R, -cap, cap)
    out[:, 1:] = F[:, 1:] + 0.5 * r * r * np.exp(-2 * F[:, 1:])
    return out


def recalibrate(F_tr, y_tr, R_tr, F_va, y_va, R_va, cap):
    """One constant per head from the training tracks; returns fn(F) -> F' and the record of the constants."""
    fns, rec = [], []
    ce = lambda F, y: float(np.mean(track_losses(np.column_stack([F] + [np.zeros(len(F))] * 5), y, np.zeros((len(F), 5)), cap)[:, 0]))
    o = minimize_scalar(lambda d: ce(F_tr[:, 0] + d, y_tr), bounds=(-6, 6), method="bounded")
    fns.append(lambda f, d=float(o.x): f + d)
    rec.append({"form": "logit offset", "value": float(o.x)})
    for k in range(5):
        m, mv = np.isfinite(R_tr[:, k]), np.isfinite(R_va[:, k])
        r2 = np.clip(R_tr[m, k], -cap[k], cap[k]) ** 2
        s2 = np.exp(2 * F_tr[m, k + 1])
        wl = lambda v, r2_: float(np.mean(0.5 * np.log(v) + r2_ / (2 * v)))
        med = float(np.median(s2))
        o = minimize_scalar(lambda u: wl(s2 + np.exp(u), r2), bounds=(np.log(1e-6 * med), np.log(1e6 * med)), method="bounded")
        kappa = float(np.exp(o.x)) if o.fun < wl(s2, r2) else 0.0
        c = 0.5 * float(np.log(np.mean(r2 / s2)))
        r2v, s2v = np.clip(R_va[mv, k], -cap[k], cap[k]) ** 2, np.exp(2 * F_va[mv, k + 1])
        if wl(s2v + kappa, r2v) <= wl(s2v * np.exp(2 * c), r2v):
            fns.append(lambda f, kappa=kappa: 0.5 * np.log(np.exp(2 * f) + kappa))
            rec.append({"form": "added variance", "value": kappa, "sqrt": float(np.sqrt(kappa))})
        else:
            fns.append(lambda f, c=c: f + c)
            rec.append({"form": "scale", "value": float(np.exp(c))})
    return (lambda F: np.column_stack([fn(F[:, i]) for i, fn in enumerate(fns)])), rec


def event_sums(L, inv, n_ev):
    """(n_ev, 6) sums and counts of the per-track losses per test event."""
    s = np.stack([np.bincount(inv, weights=np.nan_to_num(L[:, k]), minlength=n_ev) for k in range(L.shape[1])], 1)
    c = np.stack([np.bincount(inv, weights=np.isfinite(L[:, k]).astype(float), minlength=n_ev) for k in range(L.shape[1])], 1)
    return s.astype(np.float32), c.astype(np.int32)


def reference_models(scen):
    import xgboost as xgb
    out = {}
    for w in WIN:
        d = RQ.ROOT / "reference_fits" / scen / w
        best = json.load(open(d / "best.json"))["best_trial"]
        bs = {}
        for p in sorted(d.glob("trial_*.json")):
            b = xgb.Booster()
            b.load_model(str(p))
            bs[int(p.stem.split("_")[1])] = b
        out[w] = (best, bs)
    return out


def evaluate(scen, name, S, root, files, REF, TODAY):
    import heads as HD
    import joint_heads as J
    D = RQ.problem_data(*RQ.load_tracks(root, scen, files, name), S)
    H = HD.Heads.from_npz(RQ.ROOT / "models" / f"{name}_{scen}.npz", scen)
    trec = json.load(open(RQ.ROOT / "trials" / scen / f"{name}.json"))
    ev_keys = np.unique(D["key"][D["test"]])
    rec = {"detector_scenario": scen, "quantization_scenario": name, "k_bend": D["k_bend"], "n_test_events": int(len(ev_keys)), "windows": {}}
    arrays = {"event_key": ev_keys}
    wmask = {"prompt": np.abs(D["d0"]) <= 0.02, "displaced": np.abs(D["d0"]) > 0.02}
    F_a = H.margins(D["X"], D["d0"])
    for w in WIN:
        tr, va, te = (np.flatnonzero(wmask[w] & m) for m in (~D["test"] & ~D["val"], D["val"], D["test"]))
        P = J.Problem(D["y"], D["R"], tr)
        Pt = TODAY["P"][w]
        cap_common = np.maximum(P.cap, Pt.cap)
        best, bs = REF[w]
        Fb = {k: J.predict(bs[best], D["X"][i]) for k, i in (("tr", tr), ("va", va), ("te", te))}
        recal, crec = recalibrate(Fb["tr"], D["y"][tr], D["R"][tr], Fb["va"], D["y"][va], D["R"][va], P.cap)
        y, R = D["y"][te], D["R"][te]
        F = {"a": F_a[te], "b": Fb["te"], "bp": recal(Fb["te"]), "const": np.tile(P.base, (len(te), 1))}
        inv = np.searchsorted(ev_keys, D["key"][te])
        e = {"n_train": int(len(tr)), "n_test": int(len(te)), "n_test_matched": int(np.isfinite(R[:, 0]).sum()),
             "cap_own": P.cap.tolist(), "cap_common": cap_common.tolist(), "recalibration": crec,
             "clean_fraction_test": float(y.mean()), "own": {}, "abs": {}}
        for k, f in F.items():                               # the scenario's own cap: a, b, b' on the same problem
            L = track_losses(f, y, R, P.cap)
            e["own"][k] = np.nanmean(L, 0).tolist()
            arrays[f"{w}/own/{k}/sum"], arrays[f"{w}/own/{k}/count"] = event_sums(L, inv, len(ev_keys))
        Lm = np.mean([track_losses(J.predict(b, D["X"][te]), y, R, P.cap) for b in bs.values()], 0)   # reference trial mean
        e["own"]["bm"] = np.nanmean(Lm, 0).tolist()
        arrays[f"{w}/own/bm/sum"], arrays[f"{w}/own/bm/count"] = event_sums(Lm, inv, len(ev_keys))
        e["own"]["b_trial_sd"] = np.std([np.nanmean(track_losses(J.predict(b, D["X"][te]), y, R, P.cap), 0) for b in bs.values()], 0).tolist()
        # the scenario's own 20 trials: stored as information against the same constant, so a loss difference exactly
        good = [v for v in trec["windows"][w]["trials"].values() if not v["failed"]]
        it = np.array([v["info_test"] for v in good])
        e["own"]["a_trial_mean"] = (np.array(e["own"]["a"]) + np.array(trec["windows"][w]["own_info_test"]) - it.mean(0)).tolist()
        e["own"]["a_trial_sd"] = it.std(0).tolist()
        for k in ("a", "b", "bp", "const"):                  # the common cap: against today
            L = track_losses(F[k], y, R, cap_common)
            e["abs"][k] = np.nanmean(L, 0).tolist()
            arrays[f"{w}/abs/{k}/sum"], arrays[f"{w}/abs/{k}/count"] = event_sums(L, inv, len(ev_keys))
        tt = TODAY["te"][w]                                 # c: today's tracks of this window (today's routing)
        for k, f in (("c", TODAY["F"][w]), ("const_today", np.tile(Pt.base, (len(tt), 1)))):
            L = track_losses(f, TODAY["D"]["y"][tt], TODAY["D"]["R"][tt], cap_common)
            e["abs"][k] = np.nanmean(L, 0).tolist()
            arrays[f"{w}/abs/{k}/sum"], arrays[f"{w}/abs/{k}/count"] = event_sums(L, np.searchsorted(ev_keys, TODAY["D"]["key"][tt]), len(ev_keys))
        e["n_test_today"] = int(len(tt))
        rec["windows"][w] = e
    # migration between the windows, test tracks: [origin class: prompt, b/c, other][today's window][scenario's window]
    T = RQ.load_tracks(root, scen, files, name)[0]
    te = D["test"]
    oc = np.where(T["origin"] == 0, 0, np.where(T["origin"] == 1, 1, 2))
    mig = np.zeros((3, 2, 2), int)
    np.add.at(mig, (oc[te], (np.abs(TODAY["D"]["d0"][te]) > 0.02).astype(int), (np.abs(D["d0"][te]) > 0.02).astype(int)), 1)
    rec["window_migration_test_tracks"] = mig.tolist()
    return rec, arrays


def today_context(scen, S, root, files, REF):
    import joint_heads as J
    D = RQ.problem_data(*RQ.load_tracks(root, scen, files, RQ.REFERENCE), S)
    wm = {"prompt": np.abs(D["d0"]) <= 0.02, "displaced": np.abs(D["d0"]) > 0.02}
    ctx = {"D": D, "P": {}, "te": {}, "F": {}}
    for w in WIN:
        tr, te = np.flatnonzero(wm[w] & ~D["test"] & ~D["val"]), np.flatnonzero(wm[w] & D["test"])
        ctx["P"][w], ctx["te"][w] = J.Problem(D["y"], D["R"], tr), te
        ctx["F"][w] = J.predict(REF[w][1][REF[w][0]], D["X"][te])
    return ctx


def train_spread(scen, S, root, files):
    """The training-sample term, sized once: trial 0 retrained on event-bootstrap resamples of the training events, for
    "implemented now" and 4/4/4/4; sd over the resamples of the test loss per head (own cap), both windows."""
    import joint_heads as J
    out = {}
    for name in (RQ.REFERENCE, "alloc_4444"):
        D = RQ.problem_data(*RQ.load_tracks(root, scen, files, name), S)
        for w, wm in (("prompt", np.abs(D["d0"]) <= 0.02), ("displaced", np.abs(D["d0"]) > 0.02)):
            tr, te = np.flatnonzero(wm & ~D["test"] & ~D["val"]), np.flatnonzero(wm & D["test"])
            P = J.Problem(D["y"], D["R"], tr)
            ev, inv = np.unique(D["key"][tr], return_inverse=True)
            by_ev = [np.flatnonzero(inv == i) for i in range(len(ev))]
            Ls = []
            for r in range(N_TRAIN_RESAMPLES):
                pick = np.random.default_rng([20261113, r]).integers(0, len(ev), len(ev))
                idx = np.concatenate([tr[by_ev[i]] for i in pick])
                b = J.fit(D["X"], P, idx, "A", 0.0)
                Ls.append(np.nanmean(track_losses(J.predict(b, D["X"][te]), D["y"][te], D["R"][te], P.cap), 0))
            out[f"{name}/{w}"] = {"sd_per_head": np.std(Ls, 0, ddof=1).tolist(), "sd_regression_sum": float(np.std(np.sum(Ls, 1) - np.array(Ls)[:, 0], ddof=1)),
                                  "resamples": N_TRAIN_RESAMPLES}
    return out


def one(a):
    import ext_scan as XS
    import study as ST
    XS.VERSION = RQ.VERSION
    root = XS.out_root()
    files = RQ.files()
    S = dict(np.load(RQ.ROOT / "event_split.npz"))
    names = [x for x in (a.allocs or XS.SETS[RQ.VERSION] + ([x for v in XS.ENLARGED_LATER for x in XS.SETS[v]] if RQ.VERSION == "p2x" else [])) if x not in RQ.LEFT_OUT]
    for scen in a.scen:
        out = RQ.ROOT / "eval" / scen
        out.mkdir(parents=True, exist_ok=True)
        RQ.wait_for(root / "units")
        REF = reference_models(scen)
        TODAY = today_context(scen, S, root, files, REF)
        if not (out / "train_spread.json").exists():
            json.dump(train_spread(scen, S, root, files), open(out / "train_spread.json", "w"), indent=1)
            RQ.log(scen, "training-sample spread done")
        for name in names:
            if (out / f"{name}.json").exists():
                continue
            for attempt in range(1, 6):
                RQ.wait_for(root / "units")
                try:
                    rec, arrays = evaluate(scen, name, S, root, files, REF, TODAY)
                    np.savez_compressed(out / f"{name}.tmp.npz", **arrays)
                    os.replace(out / f"{name}.tmp.npz", out / f"{name}.npz")
                    json.dump(rec, open(out / f"{name}.json.tmp", "w"), indent=1)
                    os.replace(out / f"{name}.json.tmp", out / f"{name}.json")
                    RQ.log(scen, name, "done")
                    break
                except OSError as err:
                    RQ.log(f"{scen} {name}: {err!r} (attempt {attempt}); waiting")
                    time.sleep(60)
            else:
                return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "one"))
    ap.add_argument("--scen", nargs="+", default=["AAAA", "AAAI", "IAAA"])
    ap.add_argument("--allocs", nargs="+", default=None)
    a = ap.parse_args()
    if a.cmd == "one":
        return one(a)
    logs = Path.home() / "smartpixels/scratch_local/spixbits/logs"
    jobs = [(s, subprocess.Popen([sys.executable, str(HERE / "heads_q_eval.py"), "one", "--scen", s, *(["--allocs", *a.allocs] if a.allocs else [])],
                                 stdout=open(logs / f"heads_q_eval{'' if RQ.VERSION == 'v2' else '_' + RQ.VERSION}_{s}.log", "a"), stderr=subprocess.STDOUT)) for s in a.scen]
    rc = 0
    for s, p in jobs:
        rc = max(rc, p.wait())
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
