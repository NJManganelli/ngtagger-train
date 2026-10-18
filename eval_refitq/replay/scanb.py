"""WP-C scan, Part B (wpc/scan/PLAN.md §3).

  dcheck  B-pre's known-good on file 4: the d0 pull core (MAD / 0.6745 of (d0 - d0_TP) / sigma(d0)) of disc-going
          against barrel-only prompt tracks per pT bin, INN-D (K = 1), per scenario; they must agree within
          0.1 + t1(N) x the combined bootstrap sigma (N = pT bins x scenarios), else MATERIAL. Reported beside it: the
          pre-fix rule (inner_material_rule="radius", the known-bad). Output: WPC_OUT/scan/bpre/dcheck_f04.json

Every job runs through ops/spheavy --frozen --on-busy wait.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import arms as AR
import common as CM
import fom as FM
import gates as G
from common import OP

OUT = CM.OUT / "scan"
C1B = CM.OUT / "c1b"
PT_BINS = (2.0, 3.0, 5.0, 10.0, np.inf)


def _core(x):
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x))) / 0.6745) if len(x) >= 30 else np.nan


def dcheck(a):
    with open(C1B / "scale" / "scalefit_summary.json") as fh:
        scale = float(json.load(fh)["scale"])
    geo = OP.load_geometry(CM.GEOMETRY)
    rows = {"model": [], "radius": []}
    t0 = time.time()
    for scen in a.scen:
        path = CM.input_path(scen, CM.PRED_FILE)
        cfg = CM.producer_config(path)
        tools = AR.Tools(cfg, disc=AR.DiscTools(disc_scale=scale))
        arms = {"model": AR.inn_disc(cfg), "radius": AR.inn_disc(cfg, name="INN-D-radius", inner_material_rule="radius")}
        for start, A in CM.iterate(path, AR.branches(path), chunk=a.chunk, nev=a.nev):
            ctx = AR.prepare(A, geo)
            tr = FM.track_truth(ctx)
            for lab, arm in arms.items():
                out = AR.run_arm(ctx, geo, cfg, scen, arm, tools)
                path_, _ = AR.visit_plan(ctx, geo, cfg, scen, arm)
                ok = out["refit"] & tr["pop"] & (tr["origin"] == 0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    pull = (out["a"][:, 4] - tr["d0"]) / np.sqrt(out["C"][:, 4, 4])
                rows[lab].append({"scen": np.full(ok.sum(), CM.SCENARIOS.index(scen)), "ev": (ctx["tev"] + start)[ok],
                                  "dg": AR.disc_going(path_)[ok], "pt": tr["pt"][ok], "pull": pull[ok]})
        print(f"[{time.time() - t0:7.1f}s] {scen}", flush=True)
    reps = G.replicas(nev=100)
    res = {"plan": "wpc/scan/PLAN.md §3 B-pre", "material": [], "bins": {}}
    nb = 0
    for lab in rows:
        R = CM.concat(rows[lab])
        for si, scen in enumerate(CM.SCENARIOS):
            for lo, hi in zip(PT_BINS[:-1], PT_BINS[1:]):
                e = {}
                for cls, m_cls in (("disc_going", R["dg"]), ("barrel_only", ~R["dg"])):
                    m = (R["scen"] == si) & m_cls & (R["pt"] >= lo) & (R["pt"] < hi)
                    x, ev = R["pull"][m], R["ev"][m].astype(np.int64)
                    B = [_core(np.repeat(x, w[ev])) for w in reps[:100]]
                    e[cls] = {"n": int(m.sum()), "core": _core(x), "sd": float(np.nanstd(B))}
                res["bins"][f"{lab}|{scen}|{lo:g}-{hi:g}"] = e
                nb += lab == "model" and np.isfinite(e["disc_going"]["core"]) and np.isfinite(e["barrel_only"]["core"])
    t = G.t_one(max(nb, 1))
    res["t1"] = t
    for k, e in res["bins"].items():
        dg, bo = e["disc_going"], e["barrel_only"]
        if not (np.isfinite(dg["core"]) and np.isfinite(bo["core"])):
            continue
        e["agrees"] = bool(abs(dg["core"] - bo["core"]) <= 0.1 + t * np.hypot(dg["sd"], bo["sd"]))
        if k.startswith("model|") and not e["agrees"]:
            res["material"].append(f"B-pre {k}: d0 pull core disc-going {dg['core']:.3f} vs barrel-only {bo['core']:.3f}")
    os.makedirs(OUT / "bpre", exist_ok=True)
    with open(OUT / "bpre" / "dcheck_f04.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"material": res["material"]}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("dcheck",))
    ap.add_argument("--scen", nargs="+", default=list(CM.SCENARIOS))
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10)
    a = ap.parse_args(argv)
    return {"dcheck": dcheck}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
