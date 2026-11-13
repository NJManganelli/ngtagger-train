"""The joint classifier closure of the parametrization (version 2) with the heads retrained per QUANTIZATION scenario.

  python param_closure_q.py [--scen AAAA AAAI IAAA] [--allocs network_float alloc_4444 alloc_6604 alloc_8800]
      -> checks/param_closure_v2_qheads_<detector scenario>.json

Per quantization scenario: the cross-fitted emulation of param_closure.py (unchanged), then the head outputs of the real
scenario, of the emulation and of today's tracks from that scenario's own heads (heads_retrained_per_quantization/models)
and, beside them, from the detector scenario's heads (heads_retrained/models); classifier real against emulated on the
tagger's track inputs, tolerance as in param_closure.py.
The per-quantization-scenario heads were trained on training events of the same five files the parametrization closes
on: on a training event a head has seen the real scenario's version of the very track whose emulated copy it scores.
So the test that counts is on the tagger study's TEST events only (122; statistics review 3, MEDIUM 7; the classifier's
sd is larger there); the same test on all 500 events is given beside it, labelled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import param_closure as PC
import param_model as PM

ALLOCS = ("network_float", "alloc_4444", "alloc_6604", "alloc_8800")


def main():
    import ext_scan as XS
    import heads as HD
    import retrain_heads_q as RQ
    import study as ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", nargs="+", default=list(ST.SCENARIOS))
    ap.add_argument("--allocs", nargs="+", default=list(ALLOCS))
    ap.add_argument("--parallel", action="store_true", help="one process per detector scenario, side by side")
    a = ap.parse_args()
    if a.parallel:
        import os
        import subprocess
        import sys
        env = dict(os.environ, OMP_NUM_THREADS="4")
        jobs = [subprocess.Popen([sys.executable, __file__, "--scen", s, "--allocs", *a.allocs], env=env,
                                 stdout=open(ST.LOCAL / "logs" / f"param_qheads_{s}.log", "w"), stderr=subprocess.STDOUT) for s in a.scen]
        return max(j.wait() for j in jobs)
    XS.VERSION = "v2"
    root = XS.out_root()
    S = dict(np.load(RQ.ROOT / "event_split.npz"))
    held = RQ.event_key(*S["evid"][S["test"]].T)
    for scen in a.scen:
        RQ.wait_for(root / "units")
        T, today, _, truth, meta = PC.load(root, scen, list(ST.SCAN_FILES))
        D = PM.draws(T["run"], T["lumi"], T["event"], T["track"])
        fold = (PM._mix(T["event_key"].astype(np.uint64)) % np.uint64(2)).astype(int)
        terc = np.quantile(today["margins"][today["refit"], 0], [1 / 3, 2 / 3])
        Hd = HD.Heads(ST.LOCAL / "heads_retrained" / "models", scen)
        ev = np.unique(T["event_key"], return_inverse=True)[1]
        subsets = {"all_events": np.ones(len(ev), bool),
                   "test_events_only": np.isin(RQ.event_key(T["run"], T["lumi"], T["event"]), held)}
        out = ST.LOCAL / "checks" / f"param_closure_v2_qheads_{scen}.json"
        res = json.load(open(out)) if out.exists() else {"scenario": scen, "tracks": int(len(ev)),
                                                         "tracks_per_subset": {k: int(v.sum()) for k, v in subsets.items()},
                                                         "events_per_subset": {k: int(len(np.unique(ev[v]))) for k, v in subsets.items()},
                                                         "allocations": {}}
        for al in a.allocs:
            if al in res["allocations"]:
                continue
            arm = PC.load_arm(root, scen, list(ST.SCAN_FILES), al)
            em = PC.crossfit(today, arm, truth, D, terc, fold, Hd)
            sets = {"quantization_scenario_heads": HD.Heads.from_npz(RQ.ROOT / "models" / f"{al}_{scen}.npz", scen),
                    "detector_scenario_heads": Hd}
            c = {}
            for hname, H in sets.items():
                wh = lambda x: dict(x, margins=PM.head_margins(H, today, x["inputs"], x["a"][:, 4]))
                Xa, Xe, X0 = (PC.features(T, today, wh(x), "tagger") for x in (arm, em, today))
                c[hname] = {}
                for sname, m in subsets.items():
                    auc_e, sd_e = PC.classifier_auc(Xa[m], Xe[m], ev[m])
                    auc_0, _ = PC.classifier_auc(Xa[m], X0[m], ev[m])
                    tol = max(0.25 * (auc_0 - 0.5), 0.01, 3 * sd_e)
                    c[hname][sname] = {"auc_actual_vs_emulated": auc_e, "sd": sd_e, "auc_actual_vs_today": auc_0,
                                       "tolerance": tol, "pass": bool(auc_e - 0.5 <= tol)}
                    print(scen, al, hname, sname, round(auc_e, 3), "tolerance", round(0.5 + tol, 3), flush=True)
            res["allocations"][al] = c
            json.dump(res, open(str(out) + ".tmp", "w"), indent=1)
            Path(str(out) + ".tmp").replace(out)


if __name__ == "__main__":
    raise SystemExit(main())
