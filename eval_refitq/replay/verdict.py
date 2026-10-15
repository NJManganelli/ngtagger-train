"""The entry gate's final verdict (PLAN §5 with PD13, PD15, PD16): composes the blinded gate report
(entry/entry_summary.json, report.py --blind-e7-f03), the per-scenario rr check (entry/rrcheck_summary.json,
run_entry.py rrcheck) and the physicist's closing decisions (PD16); writes entry/entry_verdict.json.

  E1(a)         file 3, as reported; a false-veto rate above 5.7 % is accepted as statistical (PD16)
  E1(b)         gate and window: file 3 (their pooled cuts are unchanged by PD15); rr: PD15 (file 4
                known-good, file 5 gated); over-coverage bins are MINOR (the safe direction, PD16)
  E2/E3, E4, E5 file 3, as reported
  E7            not part of the verdict: K is set aside (K = 1); its file-4 result is the reason (PD16)
  known-bads    PM-nocap, PM-notrunc and INN-kb-angleunits accepted with the power statement (PD16);
                E7's is not judged (blinded, PD13)
"""
from __future__ import annotations

import argparse
import json
import os

from scipy import stats

import common as CM
import gates as G

GATE_PREFIX = {"E1a": "E1(a)", "E1b": "E1(b)", "E23": "E2/E3", "E4": "E4 ", "E5": "E5"}
WEAK_KNOWN_BADS = ("E1a/INN-kb-angleunits", "E5/PM-nocap", "E5/PM-notrunc")
POWER = ("the planted errors change ≤ 0.06 % of tracks; the gate cannot verify these features at 100 events, and they "
         "cannot move H or P measurably")
K_SET_ASIDE = ("K is set aside (entry configuration K = 1, PD16): K = 2 + skip failed its file-4 known-good; "
               "its file-3 part stays blinded (PD13)")
FALSE_VETO_MAX = 0.057


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--entry", default=str(CM.OUT / "entry"))
    a = ap.parse_args(argv)
    src = [os.path.join(a.entry, "entry_summary.json"), os.path.join(a.entry, "rrcheck_summary.json")]
    S, Q = (json.load(open(p)) for p in src)
    V = {"plan": "PLAN §5, §10 PD13, PD15, PD16", "sources": src, "material": [], "minor": [], "accepted": [],
         "not_in_verdict": []}
    for m in S["material"] + S["minor"]:
        if m.startswith("E1(b)") or (m.startswith("E1(a)") and "false-veto" in m):
            continue                                    # E1(b) and the false veto: below
        if m.startswith("E7"):
            V["not_in_verdict"].append(f"{m}. {K_SET_ASIDE}")
        elif m.startswith("known-bad") and any(k in m for k in WEAK_KNOWN_BADS):
            V["accepted"].append(f"{m}: {POWER} (PD16)")
        else:
            V["material" if m in S["material"] else "minor"].append(m)
    for scen, e in S["E1a"].items():
        fv = e["false_veto"]
        if fv["rate"] > FALSE_VETO_MAX:
            p = float(stats.binom.sf(fv["vetoed"] - 1, fv["resamples"], 0.05))
            V["accepted"].append(f"E1(a) {scen}: false-veto rate {fv['rate']:.3f} > 5.7 % ({fv['vetoed']} / "
                                 f"{fv['resamples']}), accepted as statistical, P(X >= {fv['vetoed']} | 5 %) = {p:.2f} (PD16)")
    for scen, e in S["E1b"].items():
        for k, r in e["file3"].items():
            if k.startswith("rr|"):
                continue                                # replaced by PD15's per-scenario cut
            mat, minor = G.e1b_verdict(scen, k, r, safe_direction_minor=True)
            V["material"] += mat
            V["minor"] += minor
    V["material"] += Q["material"]
    V["minor"] += Q["minor"]
    V["gates"] = {g: {"material": [m for m in V["material"] if m.startswith(p)],
                      "minor": [m for m in V["minor"] if m.startswith(p)],
                      "accepted": [m for m in V["accepted"] if m.startswith(p)]} for g, p in GATE_PREFIX.items()}
    V["known_bads_accepted"] = [m for m in V["accepted"] if m.startswith("known-bad")]
    V["entry_passes"] = not V["material"]
    with open(os.path.join(a.entry, "entry_verdict.json"), "w") as fh:
        json.dump(V, fh, indent=1)
    print(json.dumps({k: V[k] for k in ("entry_passes", "material", "minor", "accepted", "not_in_verdict")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
