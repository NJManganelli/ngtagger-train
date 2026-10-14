"""Registry entries of the WP-C entry package (src/ngtagger/spops/registry.py; working rule 6).

  tail      wpc.c1a.tail          the tail localisation (tail_summary.json), job dir of the variants job
  cuts      wpc.c1a.cuts          the 99 % cut file (cuts/wpc_cuts_v1.json), job dir of the cut fit
  entry     wpc.entry.<gate>      one entry per gate from entry/entry_summary.json, job dir of the gate data

Every entry carries the launch-time provenance of its job (--job-dir), the bias line and a status.
"""
from __future__ import annotations

import argparse
import json
import os

import common as CM
from ngtagger.spops import registry as R

BY = "WP-C entry (cmssw-l1t-expert, Claude Opus 5.5)"


def _reg(entry, job_dir, overwrite):
    p = R.register(entry, job_dir=job_dir, overwrite=overwrite, by=BY)
    print("registered", entry["id"], "->", p)


def tail(job_dir, overwrite):
    with open(CM.OUT / "tail" / "tail_summary.json") as fh:
        S = json.load(fh)
    vals = []
    for scen, s in S["scenarios"].items():
        for var in ("f64", "f32fma", "f32fma-cluster"):
            v = s.get("variants", {}).get(var)
            if not v:
                continue
            for surf, e in v["per_surface"].items():
                vals.append({"name": f"{scen} {var} {surf} exact u / v / u_ot",
                             "value": f"{e['exact_u']:.3f} / {e['exact_v']:.3f} / {e['exact_u_ot']:.3f}"})
                vals.append({"name": f"{scen} {var} {surf} p99 |du| / |dv|",
                             "value": f"{e['du_um']['p99']:.2f} / {e['dv_um']['p99']:.2f}", "unit": "um"})
        h1 = s.get("ensemble_H1")
        if h1:
            vals.append({"name": f"{scen} H1 ensemble outside fraction u / v (limit)",
                         "value": f"{h1['record']['u']['frac_outside']:.4f} / {h1['record']['v']['frac_outside']:.4f} "
                                  f"({h1['record']['u']['limit']:.4f})"})
    _reg({"id": "wpc.c1a.tail", "title": "WP-C C1a item 1: where the forced replay departs from CMSSW (canon_v3 f04)",
          "status": "valid", "values": vals,
          "bias": "none for the gates: the offsets enter E4 / E5 as measured; the attribution is a diagnosis",
          "sources": [str(CM.OUT / "tail" / "tail_summary.json")]}, job_dir, overwrite)


def cuts(job_dir, overwrite):
    with open(CM.OUT / "cuts" / "wpc_cuts_v1.json") as fh:
        C = json.load(fh)
    vals = [{"name": k, "value": v["*"]["value"], "note": f"n = {v['*']['n']} (pooled; per-bin values in the file)"}
            for k, v in C["cuts"].items()]
    _reg({"id": "wpc.c1a.cuts", "title": "WP-C C1a item 5: the replay's 99 % cuts (files 1-2, PD1 + PD8)",
          "status": "in-sample", "status_note": "fitted on canon_v3 files 1-2, also the angle table's fit sample",
          "values": vals, "bias": "none known; populations with a clean history (PD8), truth only for ownership (PD1)",
          "sources": [str(CM.OUT / "cuts" / "wpc_cuts_v1.json")]}, job_dir, overwrite)


def entry(job_dir, overwrite):
    with open(CM.OUT / "entry" / "entry_summary.json") as fh:
        S = json.load(fh)
    material = S["material"]
    base = {"bias": "refit-v2 mirror and replay arms; numbers are the entry gate's, not physics results",
            "sources": [str(CM.OUT / "entry" / "entry_summary.json")]}
    gates = {"E1a": ("E1(a)", "the table import in INN (P0.3 test (i) form)"), "E1b": ("E1(b)", "the 99 % cuts' coverage"),
             "E23": ("E2/E3", "the ORP-cal identity"), "E4": ("E4 ", "the forced replay's final parameters vs CMSSW"),
             "E5": ("E5", "the mirror PM vs refit v2"), "E7": ("E7", "K's skip-loss on real data")}
    for g, (prefix, title) in gates.items():
        mat = [m for m in material if m.startswith(prefix)]
        vals = [{"name": scen, "value": json.dumps(v)[:2000]} for scen, v in S[g].items()]
        _reg({**base, "id": f"wpc.entry.{g.lower()}", "title": f"WP-C entry gate {g}: {title}",
              "status": "valid" if not mat else "void", "status_note": "; ".join(mat) if mat else "",
              "values": vals or [{"name": "none", "value": "-"}]}, job_dir, overwrite)
    _reg({**base, "id": "wpc.entry.known-bads", "title": "WP-C entry gate: every known-bad's verdict",
          "status": "valid", "values": [{"name": k, "value": json.dumps(v)[:2000]} for k, v in S["known_bad"].items()]},
         job_dir, overwrite)
    _reg({**base, "id": "wpc.entry.verdict", "title": "WP-C entry gate: MATERIAL / MINOR verdict",
          "status": "valid", "values": [{"name": "material", "value": "; ".join(material) or "none"},
                                        {"name": "minor", "value": "; ".join(S["minor"]) or "none"},
                                        {"name": "blinded", "value": json.dumps(S["blinded"]) if S["blinded"] else "none"}]},
         job_dir, overwrite)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("tail", "cuts", "entry"))
    ap.add_argument("--job-dir", required=True, help="the launcher's job directory (launch-time provenance)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    {"tail": tail, "cuts": cuts, "entry": entry}[a.what](a.job_dir, a.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
