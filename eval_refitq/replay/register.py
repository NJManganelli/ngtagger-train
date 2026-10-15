"""Registry entries of the WP-C entry package (src/ngtagger/spops/registry.py; working rule 6).

  tail      wpc.c1a.tail          the tail localisation (tail_summary.json), job dir of the variants job
  cuts      wpc.c1a.cuts-<ver>    the 99 % cut file (common.CUTS), job dir of the cut fit; supersedes the previous
                                  version (v2: rr per scenario, PD15; v3: the smoothed leave-one-out, PD17)
  entry     wpc.entry.<gate>      one entry per gate from entry/entry_summary.json, job dir of the gate data
  final     wpc.entry.e1b-v2, wpc.entry.final
                                  the final verdict (entry/entry_verdict.json, verdict.py) with PD15's rr check, job
                                  dir of the rr check; the statuses of the earlier entries follow PD15 / PD16
  loocheck  wpc.c1a.rr-smoothed   PD17's file-4 validation of the smoothed leave-one-out (entry/loocheck_summary.json)
                                  and the end-to-end smoke, job dir of the check

Every entry carries the launch-time provenance of its job (--job-dir), the bias line and a status.
"""
from __future__ import annotations

import argparse
import json
import os

import common as CM
import verdict as VD
from ngtagger.spops import registry as R

BY = "WP-C entry (cmssw-l1t-expert, Claude Opus 5.5)"

CUT_VERSIONS = {"v2": "rr per scenario, PD15", "v3": "rr per scenario on the smoothed leave-one-out, PD15 + PD17"}
SUPERSEDES = {"v2": ("wpc.c1a.cuts", "the rr cuts pooled over scenarios biased AAAA against IAAA; refitted per scenario "
                                     "(PD15); the gate and window tables are unchanged"),
              "v3": ("wpc.c1a.cuts-v2", "the rr cuts' leave-one-out used the final (innermost) state and missed the "
                                        "scattering to the inner hits; refitted on the smoothed prediction (PD17); the "
                                        "gate and window tables are unchanged")}
GATE_TITLES = {"E1a": "the table import in INN (P0.3 test (i) form)", "E1b": "the 99 % cuts' coverage",
               "E23": "the ORP-cal identity", "E4": "the forced replay's final parameters vs CMSSW",
               "E5": "the mirror PM vs refit v2", "E7": "K's skip-loss on real data"}


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
    with open(CM.CUTS) as fh:
        C = json.load(fh)
    vals = [{"name": k, "value": v["*"]["value"], "note": f"n = {v['*']['n']} (pooled; per-bin values in the file)"}
            for k, v in C["cuts"].items()]
    ver = CM.CUTS.stem.rsplit("_", 1)[1]
    rid, (prev, note) = f"wpc.c1a.cuts-{ver}", SUPERSEDES[ver]
    _reg({"id": rid, "title": f"WP-C C1a item 5: the replay's 99 % cuts ({ver}; files 1-2, PD1 + PD8; {CUT_VERSIONS[ver]})",
          "status": "in-sample", "status_note": "fitted on canon_v3 files 1-2, also the angle table's fit sample",
          "values": vals, "bias": "none known; populations with a clean history (PD8), truth only for ownership (PD1)",
          "sources": [str(CM.CUTS)]}, job_dir, overwrite)
    R.set_status(prev, "superseded", note, by=BY, superseded_by=rid)


def entry(job_dir, overwrite):
    with open(CM.OUT / "entry" / "entry_summary.json") as fh:
        S = json.load(fh)
    material = S["material"]
    base = {"bias": "refit-v2 mirror and replay arms; numbers are the entry gate's, not physics results",
            "sources": [str(CM.OUT / "entry" / "entry_summary.json")]}
    gates = {g: (p, GATE_TITLES[g]) for g, p in {**VD.GATE_PREFIX, "E7": "E7"}.items()}
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


def final(job_dir, overwrite):
    entry_dir = CM.OUT / "entry"
    with open(entry_dir / "entry_verdict.json") as fh:
        V = json.load(fh)
    with open(entry_dir / "rrcheck_summary.json") as fh:
        Q = json.load(fh)
    with open(entry_dir / "entry_summary.json") as fh:
        S = json.load(fh)
    base = {"bias": "refit-v2 mirror and replay arms; numbers are the entry gate's, not physics results",
            "sources": [str(entry_dir / "entry_verdict.json"), str(entry_dir / "rrcheck_summary.json"),
                        str(entry_dir / "entry_summary.json")]}
    e1b = V["gates"]["E1b"]
    vals = [{"name": f"{scen} gate / window, file 3", "value": json.dumps({k: r["pooled"] for k, r in e["file3"].items()
                                                                           if not k.startswith("rr|")})[:2000]}
            for scen, e in S["E1b"].items()]
    vals += [{"name": f"{scen} rr (PD15), file {lab}", "value": json.dumps({k: {"pooled": r["pooled"], "max_abs_z": r["max_abs_z"],
                                                                             "t": r["t"]} for k, r in q[key].items()
                                                                         if k.startswith("rr|")})[:2000]}
             for scen, q in Q["scenarios"].items() for lab, key in (("4 (known-good)", "file4_known_good"), ("5", "file5"))]
    _reg({**base, "id": "wpc.entry.e1b-v2", "title": "WP-C entry gate E1b: the 99 % cuts' coverage (rr per scenario, PD15)",
          "status": "valid" if not e1b["material"] else "void", "status_note": "; ".join(e1b["material"] + e1b["minor"]),
          "values": vals}, job_dir, overwrite)
    _reg({**base, "id": "wpc.entry.final", "title": "WP-C entry gate: the final verdict (PD13, PD15, PD16)",
          "status": "valid", "values": [{"name": k, "value": "; ".join(V[k]) or "none"}
                                        for k in ("material", "minor", "accepted", "not_in_verdict")]
          + [{"name": "entry_passes", "value": str(V["entry_passes"])}]}, job_dir, overwrite)
    R.set_status("wpc.entry.e1b", "superseded", "the rr cuts are per scenario (PD15)", by=BY, superseded_by="wpc.entry.e1b-v2")
    R.set_status("wpc.entry.verdict", "superseded", "the final verdict follows PD15 / PD16", by=BY,
                 superseded_by="wpc.entry.final")
    R.set_status("wpc.entry.e7", "not-used", VD.K_SET_ASIDE, by=BY)
    acc = V["gates"]["E1a"]["accepted"]
    R.set_status("wpc.entry.e1a", "valid" if not V["gates"]["E1a"]["material"] else "void", "; ".join(acc), by=BY)
    R.set_status("wpc.entry.known-bads", "valid", "accepted (PD16): " + "; ".join(f"{k}: {w}" for k, w in VD.WEAK_KNOWN_BADS.items())
                 + "; E7's known-bad not judged (blinded, PD13)", by=BY)


def loocheck(job_dir, overwrite):
    p = CM.OUT / "entry" / "loocheck_summary.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = [{"name": f"{scen} after (smoothed, v3) rr {dims} per class",
             "value": json.dumps({b["bin"]: [b["n_test"], round(b["coverage"], 4), round(b["z"], 2)]
                                  for b in sc["classes"]["after"][dims]["bins"]})[:2000]}
            for scen, sc in Q["scenarios"].items() for dims in ("xy", "xyab")]
    vals += [{"name": f"{scen} smoke", "value": json.dumps(s)} for scen, s in Q["smoke"].items()]
    _reg({"id": "wpc.c1a.rr-smoothed", "title": "WP-C: the smoothed leave-one-out's coverage per remaining inner hits (PD17)",
          "status": "valid" if not Q["material"] else "void",
          "status_note": "validated on file 4 only (files 3 and 5 were seen for this cut); " + ("; ".join(Q["material"] + Q["minor"]) or "no failure"),
          "values": vals, "bias": "file-4 validation, not a gate", "sources": [str(p)]}, job_dir, overwrite)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("tail", "cuts", "entry", "final", "loocheck"))
    ap.add_argument("--job-dir", required=True, help="the launcher's job directory (launch-time provenance)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    {"tail": tail, "cuts": cuts, "entry": entry, "final": final, "loocheck": loocheck}[a.what](a.job_dir, a.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
