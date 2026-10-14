"""The entry gate's verdict (PLAN §4, §5, §10): every gate from gatedata.py's outputs, its known-good and
known-bads, the MATERIAL / MINOR classification; writes WPC_OUT/entry/entry_summary.json.

Run after the gate-data jobs (files 3 and 4, the known-bads) and the tail jobs.

--blind-e7-f03: K = 2 + skip failed its file-4 known-good E7 (MATERIAL, 2026-10-14); its file-3 numbers are left
unopened so that a revised K can still be gated on file 3. File 3's K runs (INN-K2, the E7 known-bad
INN-K2-tailprob) are not read, so no E7 file-3 statistic is computed, printed or written; E7 is judged on its
file-4 known-good only, and the E7 known-bad is recorded as blinded.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import common as CM
import cuts as CU
import gates as G
from ngtagger.spdefs import angles as SA


BLIND_RUNS = ("INN-K2", "INN-K2-tailprob")
BLIND_REASON = ("K = 2 + skip failed its file-4 known-good E7 (MATERIAL); its file-3 numbers are left unopened so "
                "that a revised K can still be gated on file 3")


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def refit_975(fit_npz, cov="calibrated"):
    """E1(b)'s known-bad (2): the gate xy cut refitted at the 97.5 % level on the fit rows."""
    z = np.load(fit_npz)
    R = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(cov + "__")}
    m = R["primary"] & R["table_pop"] & R["clean"]
    old = CU.LEVEL
    CU.LEVEL = 0.975
    try:
        tab = CU._fit_table(R["g_xy"][m], R["surface"][m], R["first"][m])
    finally:
        CU.LEVEL = old
    return {"schema": CU.SCHEMA, "level": 0.975, "cuts": {f"gate|xy|{cov}": tab}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--entry", default=str(CM.OUT / "entry"))
    ap.add_argument("--scen", nargs="+", default=list(CM.SCENARIOS))
    ap.add_argument("--blind-e7-f03", action="store_true",
                    help="do not read file 3's K runs: E7 on its file-4 known-good only (see the module docstring)")
    a = ap.parse_args(argv)
    blind = BLIND_RUNS if a.blind_e7_f03 else ()
    cuts = CU.Cuts.load(str(CM.OUT / "cuts" / "wpc_cuts_v1.json"))
    table = SA.AngleTable.load(os.environ.get("WPC_ANGLE_TABLE",
                                              "/Volumes/WDMac/smartpixels_scratch/p03_angles/angle_table_v1a1e_canon_v3.json"))
    S = {"p02_closure": {}, "E1a": {}, "E1b": {}, "E23": {}, "E4": {}, "E5": {}, "E7": {}, "known_bad": {},
         "material": [], "minor": [], "blinded": {"file3_runs": list(blind), "reason": BLIND_REASON} if blind else {}}
    for scen in a.scen:
        for f in (CM.GATED_FILE, CM.PRED_FILE):
            S["p02_closure"][f"{scen}_f{f:02d}"] = G.p02_closure(scen, f)
        d3, d4 = G.load(scen, 3, a.entry, skip=blind), G.load(scen, 4, a.entry)
        with open(os.path.join(a.entry, f"{scen}_f03.json")) as fh:
            meta3 = json.load(fh)
        # E1(a)
        b3 = G.e1a_bins(d3["INN/own"], table)
        j3 = G.e1a_judge(b3)
        mat = G.e1a_material(j3, d3["INN/own"])
        b4 = G.e1a_bins(d4["INN/own"], table)
        j4 = G.e1a_judge(b4)
        fv = G.e1a_false_veto(b4, j4)
        tonly = G.e1a_judge(G.e1a_bins(d3["INN/own"], table, stat="t1"))
        S["E1a"][scen] = {"W": j3["W"], "N": j3["N"], "n_fail": j3["n_fail"], "material_bins": mat,
                          "failing": [r for r in j3["bins"] if r["fail"]], "known_good_file4_fail": j4["n_fail"],
                          "false_veto": fv, "table_only_n_fail": tonly["n_fail"]}
        if mat:
            S["material"].append(f"E1(a) {scen}: {len(mat)} failing bins hold > 2 % of their surface")
        elif j3["n_fail"]:
            S["minor"].append(f"E1(a) {scen}: {j3['n_fail']} failing bins, each < 2 % of its surface")
        if fv["rate"] > 0.057:
            S["material"].append(f"E1(a) {scen}: false-veto rate {fv['rate']:.3f} > 5.7 %")
        # E1(b)
        r3 = G.e1b_all(d3["INN/own"], d3["INN/loo"], cuts)
        r4 = G.e1b_all(d4["INN/own"], d4["INN/loo"], cuts)
        S["E1b"][scen] = {"file3": r3, "file4_known_good": r4}
        for k, r in r3.items():
            bad_bins = [b for b in r["bins"] if np.isfinite(b["z"]) and abs(b["z"]) > r["t"]]
            if not r["pooled"]["pass"]:
                S["material"].append(f"E1(b) {scen} {k}: pooled shortfall bound {r['pooled']['shortfall_bound95']:.4f} > 0.005")
            if bad_bins:
                big = [b for b in bad_bins if b["n_test"] > 0.02 * r["pooled"]["n"]]
                (S["material"] if big else S["minor"]).append(f"E1(b) {scen} {k}: {len(bad_bins)} bins beyond t(N)")
        # E2 / E3
        e = G.e23(meta3["identity_chunks"], d3["INN/own"], d3["INN-payload/own"])
        S["E23"][scen] = e
        if not e["identical_every_chunk"]:
            S["material"].append(f"E2/E3 {scen}: the ORP-cal identity fails")
        # E4
        kbf = d3.get("forced-kb-shift")
        e = G.e4(d3, d4, kbf)
        S["E4"][scen] = e
        for p, r in e["params"].items():
            if not r["pass"]:
                (S["material"] if p in ("d0", "z0") else S["minor"]).append(f"E4 {scen} {p} fails")
        # E5
        e = G.e5(d3, d4, scen)
        e["iv"] = G.e5_iv(d3, d4)
        e["pm_b"] = G.pmb_identity(d3)
        S["E5"][scen] = e
        n_cross = sum(v["n"] for v in e["i_rates"].values())
        for Sx, r in e["i_rates"].items():
            if not r["pass"]:
                (S["material"] if r["n"] > 0.02 * n_cross else S["minor"]).append(f"E5(i) {scen} {Sx} rate above prediction")
        if not e["i_near_tie"]["pass"]:
            S["material"].append(f"E5(i) {scen}: disagreements outside the near-tie window exceed the bound")
        for k, r in e["ii_iii_a"].items():
            if not r["pass"]:
                S["material"].append(f"E5(ii)-(iii)a {scen} {k}: pooled first-flip shift differs from the prediction")
        for k in ("H", "P"):
            r = e["ii_iii_b"].get(k)
            if r and not r["pass"]:
                note = " (the predicted, common-mode near-tie shift)" if e["ii_iii_a"][k]["pass"] else ""
                S["material"].append(f"E5(ii)-(iii)b {scen} {k}: |PM - v2| bound {r['bound95']:.4f} > {r['margin']}{note}")
        for p, r in e["iv"]["params"].items():
            if not r["pass"]:
                (S["material"] if p in ("d0", "z0") else S["minor"]).append(f"E5(iv) {scen} {p} fails")
        if not e["pm_b"]["pass"]:
            S["material"].append(f"E5 {scen}: PM-B not identical to PM on barrel-only tracks")
        # E7
        r4_ = G.e7(d4)
        if r4_["pooled"]["loss"] > G.E7_LIMIT:
            S["material"].append(f"E7 {scen}: the known-good skip-loss {r4_['pooled']['loss']:.4f} > 1.5 % (design)")
        if blind:
            S["E7"][scen] = {"file4": r4_, "file3": "blinded"}
        else:
            r3_ = G.e7(d3)
            c = G.e7_compare(r3_, r4_)
            S["E7"][scen] = {"file3": r3_, "file4": r4_, "compare": c}
            if not c["file3_ok"]:
                S["material"].append(f"E7 {scen}: the gated skip-loss lower bound {c['file3_lower95']:.4f} > 1.5 %")
            if c["bin_fails"]:
                S["minor"].append(f"E7 {scen}: bins differing from file 4: {c['bin_fails']}")
        # known-bads (AAAA)
        if scen == "AAAA":
            kb = {}
            for nm in ("INN-kb-payload", "INN-kb-folded", "INN-kb-globalsign", "INN-kb-angleunits"):
                jj = G.e1a_judge(G.e1a_bins(d3[f"{nm}/own"], table))
                kb[f"E1a/{nm}"] = {"n_fail": jj["n_fail"], "fails": jj["n_fail"] > 0}
            kb["E1b"] = G.e1b_known_bads(d3["INN/own"], d3["INN/loo"],
                                        refit_975(str(CM.OUT / "cuts" / "cutfit_own_rows.npz")))
            for nm in ("swap", "halfmat"):
                kb[f"E23/{nm}"] = {"fails": not bool(np.all(d3[f"ident-kb-{nm}"]["identical"]))}
            kb["E4/shift"] = {"fails": any(r[st].get("known_bad_fails") for r in S["E4"][scen]["params"].values()
                                           for st in ("p50", "p99", "beyond_p999"))}
            for nm in ("PM-nocap", "PM-notrunc", "PM-norecharge", "PM-ellipse"):
                kb[f"E5/{nm}"] = G.e5_mirror_kb(d3, d4, nm)
            if not blind:
                kb["E7/tailprob"] = G.e7_kb(d3, d3["INN-K2-tailprob"])
                kb["E7/tailprob"]["fails"] = kb["E7/tailprob"]["loss"] > G.E7_LIMIT
            S["known_bad"] = kb
            for k, v in kb.items():
                vals = v.values() if k == "E1b" else [v]
                for x in vals:
                    if not x.get("fails", True):
                        S["material"].append(f"known-bad {k} does not fail its gate")
    with open(os.path.join(a.entry, "entry_summary.json"), "w") as fh:
        json.dump(clean(S), fh, indent=1)
    print(json.dumps({"material": S["material"], "minor": S["minor"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
