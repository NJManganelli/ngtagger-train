"""WP-C scan, Part A (wpc/scan/PLAN.md §2 A4): E7, the skip-loss of the two-layer hit choice (tree2, PD14), with the discs.

  run    one file, every scenario: INN-D with K = 1 (own clusters); INN-D tree2; its known-bads, P at the 95 % gate cut
         (the decision-scale one) and the tail-probability rule (the gross one); on file 4 also tree2 with the wide
         window (gate99, reported for the physicist). Output: WPC_OUT/scan/tree2/<scen>_f<NN>.npz
  judge  --role known-good (file 4): the pooled skip-loss <= 1.5 % per scenario (else MATERIAL), the known-bads' shifts
         in units of the 1.5 % margin, the gate's power at file 4's value, the wide window's report;
         --role gate (file 6): pooled, file 6's one-sided 95 % bound <= 1.5 % (else MATERIAL); per first-visited
         surface x pT bin, file 6 - file 4 <= t1(N) x the combined sigma, one-sided (MINOR, as in the entry).
         Output: WPC_OUT/scan/tree2/e7_f<NN>.json

Every job runs through ops/spheavy --frozen --on-busy wait.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy import stats

import arms as AR
import common as CM
import cuts as CU
import gatedata as GD
import gates as G
from common import OP

OUT = CM.OUT / "scan" / "tree2"
C1B = CM.OUT / "c1b"
C1B_CUTS = C1B / "cuts" / "wpc_cuts_c1b_v1.json"
MARGIN = G.E7_LIMIT                                    # 1.5 %


def _cuts95(cuts, key):
    """The c1b v1 cut file with the gate table of `key` refitted at the 95 % level on the cut fit's own clusters (the
    population of the 99 % table: primary, table population, clean history)."""
    z = np.load(C1B / "cuts" / "c1b_cut_rows.npz")
    R = {k[5:]: z[k] for k in z.files if k.startswith("own__")}
    m = R["primary"] & R["table_pop"] & R["clean"]
    old = CU.LEVEL
    CU.LEVEL = 0.95
    try:
        tab = CU._fit_table(R[f"g_{key.split('|')[1]}"][m], R["surface"][m], R["first"][m])
    finally:
        CU.LEVEL = old
    doc = json.loads(json.dumps(cuts.doc))
    doc["cuts"][key] = tab
    return CU.Cuts(doc)


def run(a):
    os.makedirs(OUT, exist_ok=True)
    with open(C1B / "scale" / "scalefit_summary.json") as fh:
        scale = float(json.load(fh)["scale"])
    geo = OP.load_geometry(CM.GEOMETRY)
    cuts = CU.Cuts.load(str(C1B_CUTS))
    for f in a.files:
        for scen in a.scen:
            path = CM.input_path(scen, f)
            cfg = CM.producer_config(path)
            assert CM.mask_of(cfg) == scen
            disc = AR.DiscTools(disc_scale=scale)
            tools = AR.Tools(cfg, cuts=cuts, disc=disc)
            k1 = AR.inn_disc(cfg)
            key = CU.table_key("gate", k1.dims_key("rank"), k1.cov_model)
            tools95 = AR.Tools(cfg, cuts=_cuts95(cuts, key), disc=disc)
            runs = {"K1": (k1, tools), "T2": (AR.inn_disc(cfg, name="T2", K="tree2"), tools),
                    "T2-p95": (AR.inn_disc(cfg, name="T2-p95", K="tree2"), tools95),
                    "T2-tailprob": (AR.inn_disc(cfg, name="T2-tailprob", K="tree2", k_score="tailprob"), tools)}
            if f == CM.PRED_FILE:
                runs["T2-gate99"] = (AR.inn_disc(cfg, name="T2-gate99", K="tree2", window_cut="gate99"), tools)
            parts, tr_off, t0 = {}, 0, time.time()
            for start, A in CM.iterate(path, AR.branches(path), chunk=a.chunk, nev=a.nev):
                ctx = AR.prepare(A, geo)
                for name, (arm, tl) in runs.items():
                    out = AR.run_arm(ctx, geo, cfg, scen, arm, tl, own=(name == "K1"))
                    tb = GD.track_block(out, ctx, start)
                    tb["o"] = np.arange(ctx["ntrk"]) + tr_off
                    parts.setdefault(name, []).append(tb)
                    if name == "K1":
                        parts.setdefault("K1/own", []).append(GD.own_block(out, ctx, start, tr_off)[0])
                parts.setdefault("trk", []).append({"ot_pt": ctx["T"]["pt"].astype(np.float64), "ev": ctx["tev"] + start,
                                                    "o": np.arange(ctx["ntrk"]) + tr_off})
                tr_off += ctx["ntrk"]
            data = {k: CM.concat(v) for k, v in parts.items()}
            np.savez_compressed(OUT / f"{scen}_f{f:02d}.npz",
                                **{f"{k}__{fld}": v for k, d in data.items() for fld, v in d.items()})
            print(f"[{time.time() - t0:7.1f}s] {scen} f{f:02d}: {tr_off} tracks", flush=True)
    return 0


def _load(scen, f):
    z = np.load(OUT / f"{scen}_f{f:02d}.npz")
    d = {}
    for k in z.files:
        run_, fld = k.split("__", 1)
        d.setdefault(run_, {})[fld] = z[k]
    return d


def _e7(d, arm):
    """gates.e7 with K = 1 against `arm` (its committed first-step choice in `branch`)."""
    return G.e7({"INN": d["K1"], "INN-K2": d[arm], "INN/own": d["K1/own"], "cmssw_trk": d["trk"]})


def _window_report(d):
    """The first visit step of K = 1, tree2 and tree2 with the wide window: the wrong share of accepted hits and the
    share of tracks with an own accepted hit under K = 1 that end with none."""
    own1 = d["K1"]["first_own"].astype(bool)
    rep = {}
    for arm in ("K1", "T2", "T2-gate99"):
        if arm not in d:
            continue
        w, o = d[arm]["first_wrong"].astype(bool), d[arm]["first_own"].astype(bool)
        rep[arm] = {"wrong_share": float(w.sum() / max((w | o).sum(), 1)),
                    "own_lost": float((own1 & ~o & ~w).sum() / max(own1.sum(), 1))}
    return rep


def judge(a):
    res = {"plan": "wpc/scan/PLAN.md §2 A4", "role": a.role, "material": [], "minor": [], "scenarios": {}}
    for scen in a.scen:
        if a.role == "known-good":
            d = _load(scen, CM.PRED_FILE)
            r = _e7(d, "T2")
            kb95, kbt = _e7(d, "T2-p95")["pooled"], _e7(d, "T2-tailprob")["pooled"]
            p, sd = r["pooled"]["loss"], r["pooled"]["sd"]
            p_pass = float(stats.norm.cdf((MARGIN - p) / max(sd, 1e-12) + 1.645))    # file 6 true = file 4's value
            res["scenarios"][scen] = {
                "e7": r, "pooled_ok": p <= MARGIN,
                "known_bad_p95": {**kb95, "shift_in_margins": (kb95["loss"] - p) / MARGIN, "fails": kb95["loss"] > MARGIN},
                "known_bad_tailprob": {**kbt, "shift_in_margins": (kbt["loss"] - p) / MARGIN, "fails": kbt["loss"] > MARGIN},
                "gate_pass_probability_at_file4": p_pass, "window_report": _window_report(d)}
            if p > MARGIN:
                res["material"].append(f"E7 {scen}: the known-good skip-loss {p:.4f} > 1.5 % (the design failed)")
        else:
            r4 = _e7(_load(scen, CM.PRED_FILE), "T2")
            r6 = _e7(_load(scen, 6), "T2")
            res["scenarios"][scen] = {"e7": r6, "file4": r4, "common_bins": [k for k in r6["bins"] if k in r4["bins"]]}
    if a.role == "gate":
        n = sum(len(e["common_bins"]) for e in res["scenarios"].values())
        t = G.t_one(max(n, 1))
        for scen, e in res["scenarios"].items():
            r6, r4 = e["e7"], e["file4"]
            e["bin_fails"] = [k for k in e["common_bins"] if r6["bins"][k]["loss"] - r4["bins"][k]["loss"]
                              > t * np.hypot(r6["bins"][k]["sd"], r4["bins"][k]["sd"])]
            e["t1"] = t
            if r6["pooled"]["lower95"] > MARGIN:
                res["material"].append(f"E7 {scen}: the gated skip-loss lower bound {r6['pooled']['lower95']:.4f} > 1.5 %")
            if e["bin_fails"]:
                res["minor"].append(f"E7 {scen}: bins with an excess loss over file 4: {e['bin_fails']}")
    f = CM.PRED_FILE if a.role == "known-good" else 6
    with open(OUT / f"e7_f{f:02d}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({"material": res["material"], "minor": res["minor"]}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("run", "judge"))
    ap.add_argument("--files", type=int, nargs="+", default=[CM.PRED_FILE])
    ap.add_argument("--scen", nargs="+", default=list(CM.SCENARIOS))
    ap.add_argument("--role", choices=("known-good", "gate"), default="known-good")
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10)
    a = ap.parse_args(argv)
    return {"run": run, "judge": judge}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
