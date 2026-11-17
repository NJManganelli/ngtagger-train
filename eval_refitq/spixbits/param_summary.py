"""The closure of the parametrization in a few numbers per detector scenario (for the README of the deliverable).

  python param_summary.py [--version v2] [--dir DIR] -> prints a table; writes param_closure_<version>_summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(res):
    core = [c for a in res["allocations"].values() for c in a["core"].values()]
    rel = np.array([(c["w68_emulated_um"] - c["w68_actual_um"]) / c["w68_actual_um"] for c in core])
    tail = [c for a in res["allocations"].values() for c in a["prompt_tail"].values()]
    e2e = [c for a in res["allocations"].values() for c in a["end_to_end"].values()]
    out = {"quantization_scenarios": len(res["allocations"]), "tracks": res["tracks"], "events": res["events"],
           "width_cells": len(core), "width_cells_passing": int(sum(c["pass"] for c in core)),
           "width_within_5pct": float(np.mean(np.abs(rel) < 0.05)), "width_within_10pct": float(np.mean(np.abs(rel) < 0.10)),
           "width_median_relative_difference": float(np.median(rel)), "width_largest_relative_difference": float(np.abs(rel).max()),
           "prompt_tail_checks": len(tail), "prompt_tail_passing": int(sum(c["pass"] for c in tail)),
           "end_to_end_checks": len(e2e), "end_to_end_passing": int(sum(c["pass"] for c in e2e)),
           "scenarios_with_every_check_passing": int(sum(not a["material"] for a in res["allocations"].values())),
           "changed_share_actual_emulated": {n: [round(a["changed_share"]["actual"], 3), round(a["changed_share"]["emulated"], 3)]
                                             for n, a in res["allocations"].items() if "classifier" in a},
           "classifier": {}, "kappa_actual_emulated": {}, "head_output_median_shift": {}}
    for n, a in res["allocations"].items():
        if "classifier" in a:
            out["classifier"][n] = {f"{h}/{w}": {"auc": round(c["auc_actual_vs_emulated"], 3), "tolerance": round(0.5 + c["tolerance"], 3),
                                                 "auc_actual_vs_today": round(c["auc_actual_vs_today"], 3), "pass": c["pass"]}
                                    for h, d in a["classifier"].items() for w, c in d.items()}
            out["kappa_actual_emulated"][n] = [round(a["kappa_actual"]["kappa"], 2), round(a["kappa_emulated"]["kappa"], 2)]
    for h in ("retrained", "deployed"):                      # head outputs: largest |median emulated - median actual| per head
        if not all(h in a["head_outputs_q16_q50_q84"] for a in res["allocations"].values()):
            continue                                         # the prompt collection has no deployed heads
        d = {}
        for a in res["allocations"].values():
            for name, q in a["head_outputs_q16_q50_q84"][h].items():
                d[name] = max(d.get(name, 0.0), abs(q["emulated"][1] - q["actual"][1]))
        out["head_output_median_shift"][h] = {k: round(v, 3) for k, v in d.items()}
    return out


def main():
    import study as ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    ap.add_argument("--dir", default=str(ST.LOCAL / "checks"))
    a = ap.parse_args()
    allout = {}
    for scen in ST.SCENARIOS:
        p = Path(a.dir) / f"param_closure_{a.version}_{scen}.json"
        if p.exists():
            allout[scen] = summarize(json.load(open(p)))
    json.dump(allout, open(Path(a.dir) / f"param_closure_{a.version}_summary.json", "w"), indent=1)
    print(json.dumps(allout, indent=1))


if __name__ == "__main__":
    main()
