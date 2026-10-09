#!/usr/bin/env python
"""P0.3 step 6: register the results (ngtagger.spops.registry), each with errors, provenance (the
launcher's launch-time record when run inside a launched job: $SP_JOB_DIR) and status.

    python registry_seed.py [--dry-run] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

from ngtagger.spdefs import angles as SA  # noqa: E402
from ngtagger.spops import registry as R  # noqa: E402

BIAS = ("emulated class-T angles (truth + 1.5 x a PixelAV residual; the PixelAV response on the Phase-2 geometry and "
        "the 1.5 factor taken as given): the table and the ROC describe the emulator, not a sensor; class-C angles are "
        "not improved by k (scaled-mode rejection is an upper bound)")


def v(name, value, error=None, unit=None, note=None):
    d = {"name": name, "value": value}
    if error is not None:
        d["error"] = error
    if unit:
        d["unit"] = unit
    if note:
        d["note"] = note
    return d


def load(name):
    p = CM.OUT_DIR / name
    return json.load(open(p)) if p.exists() else None


def entries() -> list:
    E = []
    common_prov = {"outputs_dir": str(CM.OUT_DIR), "preregistration": "eval_refitq/p03_angles/preregistration.txt"}
    chk = load("spdefs_checks_run.json")
    chkb = load("spdefs_checks_known_bad.json")
    if chk and chkb:
        E.append({"id": "p03.spdefs.checks", "title": "Shared angle module: known-answer checks (k knob, frame sign, key)",
                  "status": "valid", "bias": "code checks on canon_v3 AAAA f03, 5 events",
                  "values": [v(f"{k} known-good", x["status"], note=x["why"][:200]) for k, x in chk["items"].items()]
                  + [v(f"{k} known-bad", x["status"], note=x["why"][:200]) for k, x in chkb["items"].items()],
                  "provenance": {**common_prov, "files": ["spdefs_checks_run.json", "spdefs_checks_known_bad.json"]}})
    kg, kb, gt, fo = (load(f"p03_table_{m}.json") for m in ("known-good", "known-bad", "gated", "fit-only"))
    for tag, r, title in (("known-good", kg, "rule demonstration on a files-3-10 split (fit 3-4, test 5-10)"),
                          ("gated", gt, "test of the table fitted on files 1-2, on files 3-10 (judged once)")):
        if not r:
            continue
        j = r["judgement"]
        vals = [v("pass", j["pass"]), v("(i) pass", j["pass_i"]), v("(ii) pass", j["pass_ii"]),
                v("(i) bins outside the widened band", j["i"]["fail"], note=f"of {j['i']['bins']}"),
                v("(i) worst fraction of the widened band", j["i"]["worst_fraction_of_widened_band"]),
                v("(ii) N z-scores", j["ii"]["N"]), v("(ii) t(N)", j["ii"]["t_N"], unit="sigma"),
                v("(ii) max |z|", j["ii"]["max_abs_z"], unit="sigma"), v("(ii) z beyond t(N)", j["ii"]["n_beyond_t"]),
                v("(ii) chi2 tests failing", ", ".join(j["ii"]["chi2_fail"]) or "none")]
        vals += [v(f"(ii) max |z| at {lv}", z, unit="sigma") for lv, z in j["ii"]["per_level_max_abs_z"].items()]
        vals += [v(f"{k} max |z| / worst band fraction", f"{x['max_abs_z']:.2f} / {x['worst_band_fraction']:.2f}")
                 for k, x in j["per_region"].items()]
        vals += [v(f"(ii) chi2 p {k}", x["p"]) for k, x in j["ii"]["chi2"].items()]
        E.append({"id": f"p03.table.{tag}", "title": f"Angle-uncertainty table: {title}", "status": "valid", "bias": BIAS,
                  "values": vals, "provenance": {**common_prov, "file": f"p03_table_{tag}.json", "n_rep": r["n_rep"],
                                                 "inputs": r["inputs"], "code": r["code"]},
                  "status_note": None})
    if kb:
        E.append({"id": "p03.table.known-bads", "title": "Angle-uncertainty table: the four known-bad tables under the "
                  "same rule (fit 3-4, test 5-10)", "status": "valid", "bias": BIAS,
                  "values": [v(f"{n} fails", x["fails_as_required"],
                               note=f"(i) {x['judgement']['i']['fail']} bins; (ii) beyond t {x['judgement']['ii']['n_beyond_t']}, "
                                    f"max |z| {x['judgement']['ii']['max_abs_z']:.1f}; chi2 failing "
                                    f"{len(x['judgement']['ii']['chi2_fail'])} of 8") for n, x in kb["known_bads"].items()],
                  "provenance": {**common_prov, "file": "p03_table_known-bad.json", "n_rep": kb["n_rep"]}})
    tab_res = gt or fo
    if tab_res:
        tested = gt is not None
        passed = tested and gt["judgement"]["pass"]
        tf = tab_res["table_file"]
        doc = json.load(open(tf))
        vals = []
        for key in SA.KEYS:
            for s_ in SA.SURFACES:
                for a in SA.ANGLES:
                    rows = [r for r in doc["keys"][key]["rows"] if r["surface"] == s_ and r.get(a)]
                    if not rows:
                        continue
                    w = [r[a]["width"] for r in rows]
                    c = [r[a]["centre"] for r in rows]
                    q = [r[a]["q"]["0.99"] for r in rows]
                    nm = f"{key}/{a}/{SA.SURFACE_NAMES[s_]}"
                    vals += [v(f"{nm} rows", len(rows)),
                             v(f"{nm} width min / median / max", f"{min(w):.3f} / {np.median(w):.3f} / {max(w):.3f}",
                               unit="published sigma"),
                             v(f"{nm} max |centre|", round(float(np.max(np.abs(c))), 3), unit="published sigma"),
                             v(f"{nm} q99 min / median / max", f"{min(q):.3f} / {np.median(q):.3f} / {max(q):.3f}",
                               unit="published sigma")]
        vals.append(v("table file sha256", CM.file_digest(Path(tf)), note=tf))
        st = "valid" if passed else "exploratory"
        note = None if passed else ("the table failed its pre-declared test; see p03.table.gated" if tested else
                                    "UNTESTED: the gated test did not run (the rule's known-good demonstration or a "
                                    "known-bad did not behave as required); see p03.table.known-good / known-bads")
        E.append({"id": "p03.table.v1", "title": "Angle-uncertainty table v1 (canon_v3, fit files 1-2, pT >= 1 GeV)",
                  "status": st, "status_note": note, "bias": BIAS, "values": vals,
                  "provenance": {**common_prov, "table_file": tf, "fit_files": list(SA.FIT_FILES),
                                 "pt_check_refit_rows": tab_res["pt_check"]["flags"]}})
        pc = tab_res["pt_check"]["detail"]
        E.append({"id": "p03.table.pt-check", "title": "Angle table pT check (1-2 vs >= 2 GeV, fit sample)",
                  "status": "valid", "bias": BIAS,
                  "values": [v(f"{k} {f}", json.dumps(pc[k]["summary"][f])) for k in SA.KEYS
                             for f in ("t_N", "max_abs_z", "failing_rows", "failing_both_signs_refit_pt2",
                                       "failing_one_sign_reported", "p")],
                  "provenance": common_prov})
    roc = load("p03_roc.json")
    if roc:
        validated = bool(gt and gt["judgement"]["pass"])
        st = "valid" if validated else "exploratory"
        note = None if validated else "the cut centres come from an angle table that did not pass (or did not take) its test"
        vals = []
        for key, c in roc["combos"].items():
            if "/k1/" not in key and not key.startswith("F2/"):
                continue
            for g in ["barrel", "discs"] + [SA.SURFACE_NAMES[s] for s in SA.SURFACES]:
                ra = c["ret_at_99"][g]
                err_a = ra.get("sig_all_sd_chain", ra["sig_all_sd_test"])
                err_t = ra.get("sig_T_sd_chain", ra["sig_T_sd_test"])
                vals.append(v(f"{key}/{g} retention at 99% (all signal)", round(ra["sig_all"], 4), round(err_a, 4), "fraction"))
                vals.append(v(f"{key}/{g} retention at 99% (class-T signal)", round(ra["sig_T"], 4), round(err_t, 4), "fraction"))
            for d, x in c.get("disc_over_barrel_at_99", {}).items():
                vals.append(v(f"{key}/{d} disc/barrel retention ratio at 99%", round(x["ratio"], 4), round(x["sd"], 4)))
                vals.append(v(f"{key}/{d} disc/barrel rejection ratio at 99%", round(x["rejection_ratio"], 4),
                              round(x["rejection_ratio_sd"], 4)))
        E.append({"id": "p03.roc.cluster-angle", "title": "Cluster-level angle ROC: retention at 99% signal efficiency",
                  "status": st, "status_note": note, "bias": BIAS + "; single-cluster cuts without a track prediction",
                  "values": vals,
                  "provenance": {**common_prov, "file": "p03_roc.json", "errors": "k = 1: full chain (calibration "
                                 "re-measured per replica); k < 1: test side only"}})
    ex = load("p03_extras.json")
    if ex:
        sh = ex["shared"]
        E.append({"id": "p03.shared-bjet", "title": "Shared-cluster fraction within dR < 0.1 of b-jet axes (files 3-10)",
                  "status": "valid", "bias": "generator b jets (pT > 15 GeV, |eta| < 2.4); shared threshold provisional",
                  "values": [v(f"{g} {k}", round(x["value"], 5), round(x["sd"], 5), "fraction")
                             for g in ("barrel", "discs") for k, x in sh[g].items()], "provenance": common_prov})
        E.append({"id": "p03.class-tell-auc", "title": "Class tell: AUC of the single-cluster statistics, class T vs C",
                  "status": "valid", "bias": BIAS, "values": [v(k, round(x, 4)) for k, x in ex["class_tell_auc"].items()],
                  "provenance": common_prov})
        E.append({"id": "p03.response", "title": "Class-T response p(emitted | true) per surface (input of the full "
                  "response model)", "status": "valid", "bias": BIAS,
                  "values": [v(f"{g['tag']} clusters", g["n"]) for g in ex["response_groups"]],
                  "provenance": {**common_prov, "file": ex["response_file"], "fit_files": list(SA.FIT_FILES)}})
        P = ex["priors"]
        pv = [v(f"alpha core {s}", round(x["core_deg"], 3), unit="deg") for s, x in P["alpha"]["per_surface"].items()]
        pv += [v(f"alpha full RMS {s}", round(x["full_rms_deg"], 3), unit="deg") for s, x in P["alpha"]["per_surface"].items()]
        pv += [v(f"alpha curvature term at 2 GeV {s}", round(x["curvature_term_2GeV_deg"], 3), unit="deg")
               for s, x in P["alpha"]["per_surface"].items()]
        pv += [v(f"S1 50% pT {s}", x["pt_at_50pct"], unit="GeV") for s, x in P["alpha_veto"]["per_surface"].items()]
        ad = P["alpha_displacement"]
        pv += [v("b/c daughters median |d0|", ad["median_abs_d0_um"], unit="um"),
               v("b/c daughters |d0| p75 / p90", json.dumps(ad["p75_p90_abs_d0_um"]), unit="um"),
               v("b/c daughters fractions <= 0.2 / 0.2-1 / > 1 mm", json.dumps([ad["frac_le_0p2mm"], ad["frac_0p2_1mm"],
                                                                               ad["frac_gt_1mm"]])),
               v("delta alpha at IL1, p75 / p90 (core units)", json.dumps(ad["dalpha_IL1_in_core_at_p75_p90"]))]
        br = P["beta_response"]
        pv += [v("IL1 mean emitted cot at true 0.45 / 0.55 / 0.86", json.dumps(br["IL1_mean_emitted_at_true"])),
               v("beta response slope per layer", json.dumps(br["per_surface_slope"]))]
        bp = P["beta_pointing"]
        pv += [v("luminous region sigma_z", round(bp["sigma_z_hard_scatter_cm"], 3), unit="cm"),
               v("other interaction within 1 / 2 cm: predicted", json.dumps(bp["predicted_within_1_2cm"])),
               v("other interaction within 1 / 2 cm: measured", json.dumps(bp["measured_other_interaction_clusters_within_1_2cm"]))]
        pv += [v(f"disc beta pointing {d}", json.dumps({k: round(x, 3) if isinstance(x, float) else x for k, x in y.items()}))
               for d, y in P["disc_beta_pointing"]["per_disc"].items()]
        pv += [v(f"disc alpha {d}", json.dumps({k: round(x, 4) for k, x in y.items()})) for d, y in P["disc_alpha"]["per_disc"].items()]
        E.append({"id": "p03.priors", "title": "Plan B.4 priors recomputed at cluster level (canon_v3 files 3-10)",
                  "status": "valid", "bias": BIAS, "values": pv,
                  "provenance": {**common_prov, "not_recomputed": P["not_recomputed"]}})
    for e in E:
        if e.get("status_note") is None:
            e.pop("status_note", None)
        e.setdefault("definition_sheet", SA.SHEET_VERSION)
        e.setdefault("sources", ["plan.md D.3 (P0.3)", "report.md decisions 2, 3, 10, 11"])
    return E


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    E = entries()
    bad = 0
    for e in E:
        errs = R.validate_entry({**e, "provenance": {**e.get("provenance", {}), "known": True}})
        if errs:
            bad += 1
            print("INVALID", errs)
            continue
        if a.dry_run:
            print(f"ok {e['id']} ({len(e['values'])} values, {e['status']})")
            continue
        e["provenance"] = {**e.get("provenance", {}), "known": True}
        p = R.register(e, overwrite=a.overwrite, by="P0.3 (Claude Opus 5.5)")
        print(f"registered {e['id']} -> {p}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
