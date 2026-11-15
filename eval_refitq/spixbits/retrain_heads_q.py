"""The six per-track quality heads retrained per QUANTIZATION scenario (physicist, 2026-11-13): for every sub-area
quantization scenario of extended scan version 2 (64; the whole-sensor one is left out) and every detector scenario,
on that scenario's own replayed tracks; and a reference trained on today's ("implemented now") tracks of the SAME
files and events, so the effect of the small training sample is separated from the effect of the quantization.

  python retrain_heads_q.py split                         the tagger study's event split for the scan files (once; WDMac)
  python retrain_heads_q.py run [--scen AAAA AAAI IAAA]   one process per detector scenario, side by side
  python retrain_heads_q.py one --scen AAAA [--allocs ...] [--trials 20]
  python retrain_heads_q.py tables                        -> results.json

Pipeline and recipe: retrain_heads.py's (tagger_sp/joint_heads.py at lambda = 0: XGBoost hist, depth 3, 100 rounds,
six heads jointly, per window; trial 0 the deterministic recipe, trials 1-19 with row subsampling 0.8 and column
subsampling 0.8 per node, seed = trial; the single best trial by validation information summed over the five
regression heads; a trial with a non-finite validation loss is failed and discarded; every trial recorded).
Inputs: the 38 head inputs of the scenario's replayed tracks (13 from the OT track + the 25 stored by ext_scan.py,
built by heads.replay_columns under trackmva.refuse_truth_inputs; no truth column is an input). Targets as
joint_heads.targets: residuals of the scenario's own track against the OT track's particle (d0, z0 [um], tanL [1e-5],
1/R [1e-6 / cm] with the bending constant measured on clean matched training tracks, phi [urad]); clean = the OT track
is genuine and no accepted hit belongs to another particle (the replay's count against the OT track's particle).
Windows by the scenario's own |d0| (<= 0.02 cm prompt).
Events: canon_v5 TTbar files 5, 7, 8, 9, 10 (500 events per detector scenario), split by the tagger study's event
identity (common.split_table / val_table: test, validation, the rest training). LIMITATION: the deployed heads and
the per-detector-scenario retraining use TTbar files 1-10 and H->tautau; no H->tautau exists per quantization scenario
and only these five files were replayed, so every model here sees about a quarter of that sample: read a scenario's
heads against the five-file reference, never against the full-sample heads.
Per scenario and window, on the TEST events (information = the heads' loss gain over a constant, nats per track;
the constant and the winsorisation from that scenario's training tracks):
  own        the scenario's own best heads on its own inputs
  reference  the "implemented now" five-file heads (their best trial) applied to the scenario's inputs
  and for "implemented now" itself, own = the reference on today's inputs.
Spreads: the sd over the 20 trials of each; and a paired event bootstrap (200) of own - reference on the test tracks.
Outputs (ROOT below): trials/<detector scenario>/<quantization scenario>.json (checkpoint: a finished file is
skipped), models/<quantization scenario>_<detector scenario>.npz (the best trial of both windows, heads.Heads.from_npz;
written through the conifer JSON and checked against the booster), reference_fits/ (the reference's 20 boosters).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tagger_sp"))
os.environ.setdefault("TAGGER_SP_OUT", "/Volumes/WDMac/smartpixels_scratch/tagger_v5")
# SPIXBITS_Q_VERSION: the per-track scan the heads are trained on: "v2" (the extended collection, a beyond-baseline track
# scenario) or "p2" (the PROMPT collection, the baseline; outputs under ..._prompt/)
# "p2x": the ENLARGED sample for the first-pass quantization scenarios of the tagger step (prompt collection): scan p2's
# five files plus scan p2x's (TTbar files 1, 2, 3 and H->tautau), i.e. every file the per-detector-scenario prompt heads
# were trained on; the tagger study's event split over all of them (TTbar files 1 and 2 are training-only there);
# outputs under ..._prompt_enlarged/. Its "implemented now" reference is trained on the same enlarged sample.
VERSION = os.environ.get("SPIXBITS_Q_VERSION", "v2")
ROOT = Path(os.environ.get("SPIXBITS_Q_ROOT", "") or Path.home() / (     # SPIXBITS_Q_ROOT: another folder (development tests)
    "smartpixels/scratch_local/spixbits/heads_retrained_per_quantization"
    + {"v2": "", "p2": "_prompt", "p2x": "_prompt_enlarged"}[VERSION]))
P2X_FILES = (1, 2, 3, 101, 102, 103, 104, 105, 107, 108)     # ext_scan.P2X_FILES (checked in `one`): H->tautau file n = 100 + n
REFERENCE = "implemented_now"
LEFT_OUT = ("alloc_4444_wholesensor",)                # physicist, 2026-11-13: no further work on the whole-sensor scenario
REG = slice(1, 6)                                     # the five regression heads among the six outputs
N_BOOT = 200
MIN_FREE_GB = 12.0
SCAN_FILES = (5, 7, 8, 9, 10)                          # study.SCAN_FILES (checked in `one`)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def wait_for(path):
    """Block until the external disk holds `path` (never fail on a lost disk)."""
    n = 0
    while not Path(path).exists():
        if n % 10 == 0:
            log("waiting for", path)
        n += 1
        time.sleep(60)


def files():
    """The files of this version (H->tautau file n as 100 + n)."""
    return list(SCAN_FILES) + (list(P2X_FILES) if VERSION == "p2x" else [])


def split():
    """(run, lumi, event) -> test / validation flag of the tagger study, for this version's files."""
    import build as B                                   # tagger_sp modules only: its `common` and the replay's share a name,
    import common as C                                  # so this command must not import the replay (study, ext_scan)
    man = json.load(open(C.manifest_path("AAAA")))
    ev, te, va, fn = [], [], [], []
    for s, off in (("ttbar", 0), ("htt", 100)):
        fs = [n for s2, n, _ in man["files"] if s2 == s]
        evids = {n: B.load_tracks(os.path.join(C.cache_dir("tracks"), C.tag(s, "AAAA", n) + ".npz"))[2] for n in fs}
        test = C.split_table(s, evids)
        val = C.val_table(s, evids, test)
        for n in fs:
            if n + off in files():
                ev.append(evids[n]), te.append(test[n]), va.append(val[n]), fn.append(np.full(len(evids[n]), n + off))
    ev = np.concatenate(ev)
    assert len(np.unique(event_key(*ev.T))) == len(ev), "an event identity appears twice (TTbar and H->tautau are joined by it)"
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(ROOT / "event_split.npz", evid=ev, fileno=np.concatenate(fn), test=np.concatenate(te), val=np.concatenate(va))
    f_, t_, v_ = np.concatenate(fn), np.concatenate(te), np.concatenate(va)
    print({int(n): (int((f_ == n).sum()), int(t_[f_ == n].sum()), int(v_[f_ == n].sum())) for n in np.unique(f_)},
          "events", len(ev), "test", int(t_.sum()), "validation", int(v_.sum()), "training", int((~t_ & ~v_).sum()))


def event_key(run, lumi, event):
    return (np.asarray(run, np.int64) * 1000003 + np.asarray(lumi, np.int64)) * 1000003 + np.asarray(event, np.int64)


def load_tracks(root, scen, files, name):
    """One quantization scenario's tracks as joint_heads wants them: X (n, 38), y clean, the residual ingredients.
    Version p2x: a file of scan p2x is read from that scan's folder, the five scan files from scan p2's."""
    import ext_scan as XS
    T, A = [], []
    for f in files:
        ver = XS.VERSION
        if VERSION == "p2x":                 # a further file: the scan (p2x or p2y) that ran this quantization scenario on it
            ver = "p2" if f not in P2X_FILES else next(v for v in ("p2x", *XS.ENLARGED_LATER) if name in XS.SETS[v])
        r = Path(XS.ROOTS[ver]) if VERSION == "p2x" else root
        wait_for(r / "truth" / f"{scen}_f{f:02d}_truth.npz")
        T.append(dict(np.load(r / "truth" / f"{scen}_f{f:02d}_truth.npz")))
        chunks = [XS.SETS[ver][i:i + XS.arms_per_unit(ver)] for i in range(0, len(XS.SETS[ver]), XS.arms_per_unit(ver))]
        for k, ch in enumerate(chunks):
            if name in ch:
                Z = np.load(r / "units" / f"{scen}_f{f:02d}_{k:02d}.npz")
                A.append({k2: Z[f"{name}__{k2}"] for k2 in ("a", "inputs", "n_wrong_hits")})
    assert len(A) == len(files), (name, scen, len(A))
    T = {k: np.concatenate([t[k] for t in T]) for k in T[0]}
    A = {k: np.concatenate([a[k] for a in A]) for k in A[0]}
    return T, A


def problem_data(T, A, S, k_bend=None):
    """X, y, R (n, 5: d0, z0, tanL, 1/R, phi residuals in joint_heads' units, NaN unmatched), flags, k_bend."""
    import joint_heads as J
    a = A["a"].astype(np.float64)                      # rInv, phi0, tanL, z0, d0
    key = event_key(T["run"], T["lumi"], T["event"])
    sk = event_key(*S["evid"].T)
    o = np.argsort(sk)
    pos = o[np.searchsorted(sk[o], key)]
    assert np.array_equal(sk[pos], key), "a scan event is missing from the tagger study's split"
    test, val = S["test"][pos], S["val"][pos]
    matched = T["matched"] & (T["tp_pt"] > 0)
    y = (T["ot_genuine"] & (A["n_wrong_hits"] == 0)).astype(float)
    q, tpt = T["tp_charge"].astype(float), np.where(T["tp_pt"] > 0, T["tp_pt"], np.nan).astype(float)
    if k_bend is None:
        m = matched & (y > 0) & ~test & ~val & (q != 0)
        k_bend = float(np.median(a[m, 0] * tpt[m] / q[m]))
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    nan = lambda x: np.where(matched, x, np.nan)
    R = np.column_stack([nan(a[:, 4] - T["tp_d0"]) * J.UNIT["d0"], nan(a[:, 3] - T["tp_z0"]) * J.UNIT["z0"],
                         nan(a[:, 2] - np.sinh(T["tp_eta"].astype(np.float64))) * J.UNIT["tanl"],
                         nan(a[:, 0] - q * k_bend / tpt) * J.UNIT["rinv"], nan(wrap(a[:, 1] - T["tp_phi0"])) * J.UNIT["phi"]])
    X = np.hstack([T["ot_inputs"], A["inputs"]]).astype(np.float32)
    return {"X": X, "y": y, "R": R, "test": test, "val": val, "key": key, "d0": a[:, 4], "k_bend": k_bend}


def boot_diff(L_own, L_ref, key):
    """Paired event bootstrap of sum over the regression heads of mean(loss_ref - loss_own) on these tracks: value, sd."""
    d = (L_ref - L_own)[:, REG]
    ev, inv = np.unique(key, return_inverse=True)
    s = np.stack([np.bincount(inv, weights=np.nan_to_num(d[:, k]), minlength=len(ev)) for k in range(d.shape[1])], 1)
    c = np.stack([np.bincount(inv, weights=np.isfinite(d[:, k]).astype(float), minlength=len(ev)) for k in range(d.shape[1])], 1)
    val = float((s.sum(0) / np.maximum(c.sum(0), 1)).sum())
    rng = np.random.default_rng(20261113)
    reps = []
    for _ in range(N_BOOT):
        w = rng.multinomial(len(ev), np.full(len(ev), 1 / len(ev)))[:, None]
        reps.append(float(((w * s).sum(0) / np.maximum((w * c).sum(0), 1)).sum()))
    return val, float(np.std(reps))


def train_one(scen, name, S, root, trials, files):
    """All trials of one quantization scenario in one detector scenario; returns the record (and writes the models)."""
    import xgboost as xgb
    import heads as HD
    import joint_heads as J
    import retrain_heads as RH
    D = problem_data(*load_tracks(root, scen, files, name), S)
    base_xgb = dict(J.XGB)
    heads = [h[0] for h in J.head_list()]
    rec = {"detector_scenario": scen, "quantization_scenario": name, "k_bend": D["k_bend"], "heads": heads, "windows": {}}
    tmp = ROOT / f"tmp_{os.getpid()}"
    best_boosters = {}
    for wname, w in (("prompt", np.abs(D["d0"]) <= 0.02), ("displaced", np.abs(D["d0"]) > 0.02)):
        tr = np.flatnonzero(w & ~D["test"] & ~D["val"])
        va, te = np.flatnonzero(w & D["val"]), np.flatnonzero(w & D["test"])
        P = J.Problem(D["y"], D["R"], tr)
        e = {"n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)), "base": P.base.tolist(), "trials": {}}
        boosters = {}
        for t in range(trials):
            J.XGB.clear()
            J.XGB.update(base_xgb, seed=t, **({"subsample": 0.8, "colsample_bynode": 0.8} if t else {}))
            b = J.fit(D["X"], P, tr, "A", 0.0)
            Fv, Ft = J.predict(b, D["X"][va]), J.predict(b, D["X"][te])
            iv, lv, it = P.info(Fv, va), P.losses(Fv, va), P.info(Ft, te)
            ok = bool(np.isfinite(lv).all() and np.isfinite(Fv).all())
            e["trials"][str(t)] = {"info_val": iv.tolist(), "info_test": it.tolist(), "failed": not ok,
                                   "score": float(iv[REG].sum()) if ok else None}
            boosters[t] = b
        J.XGB.clear()
        J.XGB.update(base_xgb)
        good = {int(t): v for t, v in e["trials"].items() if not v["failed"]}
        best = max(good, key=lambda t: good[t]["score"])
        tst = np.array([np.sum(v["info_test"][REG]) for v in good.values()])
        e.update(best_trial=best, n_failed=trials - len(good), own_info_test=good[best]["info_test"],
                 own_info_test_regression_sum=float(np.sum(good[best]["info_test"][REG])),
                 own_trial_sd_regression_sum=float(tst.std()), own_trial_sd_clean=float(np.std([v["info_test"][0] for v in good.values()])))
        best_boosters[wname] = boosters[best]
        if name == REFERENCE:                               # the reference's boosters serve every other scenario
            d = ROOT / "reference_fits" / scen / wname
            d.mkdir(parents=True, exist_ok=True)
            for t in good:
                boosters[t].save_model(str(d / f"trial_{t:02d}.json"))
            json.dump({"best_trial": best}, open(d / "best.json", "w"))
        else:                                               # the reference heads on this scenario's inputs
            d = ROOT / "reference_fits" / scen / wname
            rbest = json.load(open(d / "best.json"))["best_trial"]
            infos = {}
            for p in sorted(d.glob("trial_*.json")):
                rb = xgb.Booster()
                rb.load_model(str(p))
                Fr = J.predict(rb, D["X"][te])
                infos[int(p.stem.split("_")[1])] = P.info(Fr, te)
                if int(p.stem.split("_")[1]) == rbest:
                    F_ref = Fr
            rs = np.array([v[REG].sum() for v in infos.values()])
            diff, sd = boot_diff(P.track_losses(J.predict(boosters[best], D["X"][te]), te), P.track_losses(F_ref, te), D["key"][te])
            e.update(reference_info_test=infos[rbest].tolist(), reference_info_test_regression_sum=float(infos[rbest][REG].sum()),
                     reference_trial_sd_regression_sum=float(rs.std()), reference_trial_sd_clean=float(np.std([v[0] for v in infos.values()])),
                     own_minus_reference_regression_sum=diff, own_minus_reference_bootstrap_sd=sd,
                     own_minus_reference_clean=float(good[best]["info_test"][0] - infos[rbest][0]))
        rec["windows"][wname] = e
        md = tmp / "SmartPixels" / scen / wname             # the best trial through the conifer JSON (the validated path)
        md.mkdir(parents=True, exist_ok=True)
        RH.OUT = tmp
        for k, h in enumerate(heads):
            RH.conifer_json(boosters[best], k, md / f"{h}.json")
    H = HD.Heads(tmp, scen)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    mp = ROOT / "models" / f"{name}_{scen}.npz"
    H.save_npz(str(mp) + ".tmp.npz")
    os.replace(str(mp) + ".tmp.npz", mp)
    H2 = HD.Heads.from_npz(mp, scen)                        # the stored file against the boosters, on the test tracks
    te_all = np.flatnonzero(D["test"])[:20000]
    mine = H2.margins(D["X"][te_all], D["d0"][te_all])
    ref = np.full_like(mine, np.nan)
    for wname, m in (("prompt", np.abs(D["d0"][te_all]) <= 0.02), ("displaced", np.abs(D["d0"][te_all]) > 0.02)):
        if m.any():
            ref[m] = J.predict(best_boosters[wname], D["X"][te_all][m])
    rec["model_file_vs_booster_max_abs"] = float(np.nanmax(np.abs(mine - ref)))
    shutil.rmtree(tmp, ignore_errors=True)
    return rec


def one(a):
    import ext_scan as XS                               # the replay first: joint_heads then sees the replay's `common`
    import study as ST                                  # under that name, which the functions used here never touch
    assert tuple(ST.SCAN_FILES) == SCAN_FILES and XS.P2X_FILES == P2X_FILES
    XS.VERSION = VERSION
    root = XS.out_root()
    S = dict(np.load(ROOT / "event_split.npz"))
    default = XS.SETS[VERSION] + ([x for v in XS.ENLARGED_LATER for x in XS.SETS[v]] if VERSION == "p2x" else [])
    names = [REFERENCE] + [x for x in (a.allocs or default) if x not in LEFT_OUT and x != REFERENCE]
    for scen in a.scen:
        out = ROOT / "trials" / scen
        out.mkdir(parents=True, exist_ok=True)
        for name in names:
            if (out / f"{name}.json").exists():
                continue
            if shutil.disk_usage(Path.home()).free / 2 ** 30 < MIN_FREE_GB:
                log(f"STOP: free space on the internal disk below {MIN_FREE_GB} GB")
                return 3
            for attempt in range(1, 6):
                wait_for(root / "units")
                try:
                    t0 = time.time()
                    rec = train_one(scen, name, S, root, a.trials, files())
                    rec["seconds"] = time.time() - t0
                    json.dump(rec, open(out / f"{name}.json.tmp", "w"), indent=1)
                    os.replace(out / f"{name}.json.tmp", out / f"{name}.json")
                    w = rec["windows"]
                    log(scen, name, "own", {k: round(v["own_info_test_regression_sum"], 3) for k, v in w.items()},
                        "reference", {k: round(v.get("reference_info_test_regression_sum", float("nan")), 3) for k, v in w.items()},
                        f"{rec['seconds']:.0f} s")
                    break
                except OSError as err:                      # the external disk went away mid-read: wait and repeat
                    log(f"{scen} {name}: {err!r} (attempt {attempt}); waiting")
                    time.sleep(60)
            else:
                return 1
    return 0


def run(a):
    logs = Path.home() / "smartpixels/scratch_local/spixbits/logs"
    jobs = [(s, subprocess.Popen([sys.executable, str(HERE / "retrain_heads_q.py"), "one", "--scen", s, "--trials", str(a.trials),
                                  *(["--allocs", *a.allocs] if a.allocs else [])],
                                 stdout=open(logs / f"heads_q{'' if VERSION == 'v2' else '_' + VERSION}_{s}.log", "a"), stderr=subprocess.STDOUT)) for s in a.scen]
    rc = 0
    for s, p in jobs:
        code = p.wait()
        log("detector scenario", s, "exit", code)
        rc = max(rc, code)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("split", "run", "one"))
    ap.add_argument("--scen", nargs="+", default=["AAAA", "AAAI", "IAAA"])
    ap.add_argument("--allocs", nargs="+", default=None)
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "split":
        return split()
    return run(a) if a.cmd == "run" else one(a)


if __name__ == "__main__":
    raise SystemExit(main())
