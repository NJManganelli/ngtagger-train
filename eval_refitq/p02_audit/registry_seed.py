#!/usr/bin/env python
"""P0.2 refit-v2 audit, step 4: the registration input for the WP-A1 results registry.

Writes p02_registry_seed.json ({"common": ..., "results": [...]}, the format of
``python -m ngtagger.spops.registry seed FILE``; JSON is valid YAML) from p02_results.json: the headline numbers
per scenario (H per bin, harm by origin, cores, P per surface, first-wrong-hit surface shares, the benefit
baselines, the fallback working point and cost), each value with its bootstrap error. It never writes into a
registry; when the registry module is importable, every entry is checked with its validate_entry.

    python registry_seed.py [--json .../p02_results.json] [--out .../p02_registry_seed.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_DIR = Path("/Volumes/WDMac/smartpixels_scratch/p02_refit_audit")
SCEN = ("AAAA", "AAAI", "IAAA")
ETA = ("0-1.1", "1.1-1.8", "1.8-2.4")
PT = ("2-3", "3-5", "5-10", "10-inf")


def val(name, d, unit, scale=1.0, note=None):
    out = {"name": name, "value": None if d.get("v") is None else round(d["v"] * scale, 6),
           "error": None if d.get("se") is None else round(d["se"] * scale, 6), "unit": unit}
    if note:
        out["note"] = note
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(OUT_DIR / "p02_results.json"))
    ap.add_argument("--out", default=str(OUT_DIR / "p02_registry_seed.json"))
    a = ap.parse_args()
    R = json.load(open(a.json))
    P = R["provenance"]
    common = {
        "status": "refit-v2",
        "bias": P["bias"],
        "provenance": {"known": True, "commits": {"ngtagger-train": P["code_commit"]},
                       "tree_state": "dirty (shared tree; audit files committed)", "patch": P["tree_patch_file"],
                       "patch_sha256": P["tree_patch_sha256"], "results_json": a.json,
                       "inputs": [{"declared": k, **v} for k, v in P["inputs"].items()],
                       "build": P["input_build"], "prereg": str(OUT_DIR / "preregistration.txt"),
                       "bootstrap": R["bootstrap"]},
        "definitions": R["definitions"],
    }
    res = []
    for s in SCEN:
        S, sid = R["scenarios"][s], s.lower()
        H = S["reference_E5"]["H"]
        res.append({"id": f"p02.h-prompt.{sid}", "title": f"Refit v2 {s}: harmed fraction H of prompt tracks per "
                    "true |eta| x pT bin (d0 > 0.5 mm or z0 > 1 mm worse than OT)",
                    "values": [val(f"H|eta{e}|pt{p}|{k}", H[f"eta{e}|pt{p}"][k], "fraction")
                               for e in ETA for p in PT for k in ("either", "d0", "z0")]})
        res.append({"id": f"p02.harm-origin.{sid}", "title": f"Refit v2 {s}: harm by track origin (all |eta|)",
                    "values": [val(f"{c}|{k}", R["t2"][c]["harm"][s][k], "fraction")
                               for c in ("prompt|incl", "hf|eta0-2.4|pt2-5", "hf|eta0-2.4|pt5-inf",
                                         "displaced_inner|incl", "beyond_il1|incl", "all|incl")
                               for k in ("either", "d0", "z0")]})
        res.append({"id": f"p02.core-prompt.{sid}", "title": f"Refit v2 {s}: d0 / z0 B.1 core of prompt tracks "
                    "per true |eta| (all pT), SmartPixels and OT",
                    "values": [val(f"{arm}|{p}|eta{e}", R["t2"][f"prompt|eta{e}|incl"]["arms"][arm if arm == "OT" else s][p]["core"], "um")
                               for arm in ("SP", "OT") for p in ("d0", "z0") for e in ETA]})
        res.append({"id": f"p02.p-surface.{sid}", "title": f"Refit v2 {s}: wrong-hit rate P per surface "
                    "(wrong accepted / accepted hits, truth-matched tracks)",
                    "values": [val(f"P|{x}", d, "fraction") for x, d in S["reference_E5"]["P"].items()]
                    + [val(f"P_prompt|{x}", d, "fraction") for x, d in S["reference_E5"]["P_prompt"].items()]})
        A = S["attribution"]["prompt"]["either"]
        res.append({"id": f"p02.first-wrong.{sid}", "title": f"Refit v2 {s}: harmed prompt tracks: wrong-hit "
                    "presence, first-wrong-hit surface, class and origin",
                    "values": [val("with_wrong_hit", A["with_wrong_hit"], "fraction")]
                    + [val(f"first_wrong_surface|{x}", d, "fraction") for x, d in A["first_wrong_surface_share"].items()]
                    + [val(f"first_wrong_class|{x}", d, "fraction") for x, d in A["first_wrong_class_share"].items()]
                    + [val(f"first_wrong_origin|{x}", d, "fraction") for x, d in A["first_wrong_origin_share"].items()]
                    + [val("first_wrong_tp_pt_lt_0p5", A["first_wrong_tp_pt_lt_0p5"], "fraction")]})
        B = S["benefit"]
        res.append({"id": f"p02.benefit.{sid}", "title": f"Refit v2 {s}: benefit baselines (prompt): correct-hit "
                    f"attachment at {B['innermost_active_surface']}, fraction within |d0 - true| < 100 um",
                    "values": [val(f"attach_innermost|{k}", d, "fraction") for k, d in B["attach_innermost_active"].items()
                               if k.startswith("prompt")]
                    + [val(f"within_100um_SP|{k}", d["SP"], "fraction") for k, d in B["within_100um_d0"].items()
                       if k.startswith("prompt")]})
        F = S["fallback"]
        wp = F["wp"]
        res.append({"id": f"p02.fallback.{sid}", "title": f"Refit v2 {s}: fallback to OT parameters at 1.2% unharmed "
                    "revert (files 3-6), quoted on files 7-10",
                    "values": [val("threshold_chi2", F["threshold_chi2"], "chi2 (2 dof)")]
                    + [val(f"{o}|{k}", wp[o]["test_files_7-10"][k], "fraction") for o in wp
                       for k in ("harmed_revert_fraction", "unharmed_revert_fraction", "H_before", "H_after")]
                    + [val(f"{o}|{p}|FB_minus_SP|{k}", F["cost"][o][p]["FB_minus_SP"][k], "fraction")
                       for o in F["cost"] for p in ("d0", "z0") for k in F["cost"][o][p]["FB_minus_SP"]
                       if k != "core"]})
    doc = {"common": common, "results": res}
    try:
        from ngtagger.spops.registry import validate_entry
        errs = [x for e in res for x in validate_entry({**common, **e})]
        print("validated against ngtagger.spops.registry.validate_entry:", "OK" if not errs else errs)
    except ImportError as ex:
        print("registry module not importable, not validated:", ex)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"wrote {a.out}: {len(res)} entries, {sum(len(e['values']) for e in res)} values")


if __name__ == "__main__":
    main()
