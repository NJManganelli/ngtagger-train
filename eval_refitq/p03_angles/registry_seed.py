#!/usr/bin/env python
"""P0.3 step 6: register the results (ngtagger.spops.registry), each with errors, provenance (the
launcher's launch-time record when run inside a launched job: $SP_JOB_DIR) and status.

    python registry_seed.py [--dry-run] [--overwrite] [--only ID_PREFIX]

The table results are those of the amendment-1e rule (p03_table_<mode>_a1e.json, ids p03.table.a1e.*, the table
p03.table.v1a1e) and of amendment 2, the module-frame disc rows under the same rule (p03_table_<mode>_a2.json, ids
p03.table.a2.*, the table p03.table.v1a1e-disclocal); earlier entries (v1 rule, amendments 1-1d) stay registered as
they are. Also the D10(c)
record: the two-component fits of the disc-beta rows (amendment 1b, item B9; $P03_OUT/amend1b/recipe_options.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import table as T  # noqa: E402

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


def OSU_fmt(x):
    return "-" if x is None else f"{x:.3f}"


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
    frames = (("global", "the amendment-1e rule", "Angle-uncertainty table", "p03.table.v1a1e",
               "Angle-uncertainty table tested under amendment 1e (canon_v3, fit files 1-2, pT >= 1 GeV)", "published sigma"),
              ("module", "amendment-2 (the 1e rule, module-frame disc rows)", "Angle table, module-frame disc rows",
               "p03.table.v1a1e-disclocal", "Angle-uncertainty table, module-frame disc rows (angle.error_table_disc_local, "
               "amendment 2; canon_v3, fit files 1-2, pT >= 1 GeV; keys emitted_local / true_local, facings pooled)",
               "local sigma (cot units)"))
    for frame, lab, head, tab_id, tab_title, unit in frames:
        T.set_frame(frame)
        ftag = T.FRAME["tag"]
        dg, kg, kb, gt = (load(T.result_name(m)) for m in ("diagnosis", "known-good", "known-bad", "gated"))
        for tag, r, title in (("diagnosis", dg, f"{lab} on fit 3-4 / test 5-10 (diagnosis)"),
                              ("known-good", kg, f"{lab} demonstration (fit 7-8, test 3-6 and 9-10)"),
                              ("gated", gt, "test of the table fitted on files 1-2, on files 3-10 (judged once)")):
            if not r:
                continue
            j = r["judgement"]
            fv = j.get("false_veto", {})
            stab = j["stability"]
            mt = j["materiality"]
            vals = [v("verdict (amendment 1e rule: pass / minor / material)", mt["verdict"]),
                    v("MATERIAL deviations", "; ".join(mt["material"]) or "none"),
                    v("MINOR deviations (the table ships with them listed)", "; ".join(mt["minor"]) or "none"),
                    v("pass", j["pass"]), v("(i) pass", j["pass_i"]), v("(ii) pass", j["pass_ii"]),
                    v("(iii) pass", j["pass_iii"]),
                    v("(i) bins outside the widened band (after-table centre or width)", j["i"]["fail"],
                      note=f"of {j['i']['bins']} judged bin-angles"),
                    v("(i) one-sided family-wise widening W_i", round(j["i"]["W_i"], 3), unit="sigma",
                      note=f"N_i = {j['i']['N_i']}"),
                    v("(i) worst fraction of the widened band", j["i"]["worst_fraction_of_widened_band"]),
                    v("(ii) N z-scores", j["ii"]["N"]), v("(ii) t(N)", j["ii"]["t_N"], unit="sigma"),
                    v("(ii) max |z|", j["ii"]["max_abs_z"], unit="sigma"), v("(ii) z beyond t(N)", j["ii"]["n_beyond_t"]),
                    v("(ii) chi2 tests failing", ", ".join(j["ii"]["chi2_fail"]) or "none"),
                    v("(iii) gated levels (fixed)", json.dumps(j["iii"]["gated_levels"])),
                    v("(iii) power by level (diagnostic)", json.dumps(j["iii"]["power_by_level"])),
                    v("(iii) power < 0.8 at a gated level (minor)", json.dumps(j["iii"]["power_below_0p8"])),
                    v("(iii) failing cells", ", ".join(j["iii"]["fail_cells"]) or "none"),
                    v("rows that cannot reach a level (row x level)", len(j["quantile_unreachable_rows"]),
                      note="E0 = n_fit / (n_fit + 1) < L; at 99.7 %: n_fit <= 332"),
                    v("reported: rows with centre sd / width > 0.125", stab["centre_sd_over_width"]["rows_above_0.125"],
                      note=f"of {stab['row_angles']} row-angles; max(bootstrap, fpc-scaled re-split) sd; no effect on the rule"),
                    v("reported: rows with width relative sd > 0.25", stab["width_rel_sd"]["rows_above_0.25"],
                      note=f"of {stab['row_angles']}; pct 50/90/99/max {stab['width_rel_sd']['pct_50_90_99_max']}")]
            if fv:
                vals += [v("(i)+(ii) false-veto over re-splits", fv["fail_i_ii"],
                           note=f"of {fv['n_replicas']}; rate {fv['fail_i_ii_rate']:.3f}, 95% CL "
                                f"{fv['fail_i_ii_95cl'][0]:.3f}-{fv['fail_i_ii_95cl'][1]:.3f}"),
                         v("(iii) pass probability over re-splits by gated level", json.dumps(fv["iii_pass_probability_by_level"]))]
            vals += [v(f"(ii) chi2 p {k}", x["p"]) for k, x in j["ii"]["chi2"].items()]
            E.append({"id": f"p03.table.{ftag}.{tag}", "title": f"{head}: {title}", "status": "valid",
                      "bias": BIAS, "values": vals, "definition_sheet": r.get("definition_sheet"),
                      "provenance": {**common_prov, "file": T.result_name(tag), "n_rep": r["n_rep"],
                                     "seed": r.get("seed"), "definition_sheet": r.get("definition_sheet"),
                                     "inputs": r["inputs"], "code": r["code"]},
                      "status_note": None})
        if kb:
            def kb_note(n, x):
                j = x["judgement"]
                extra = ""
                if "perturbed_bins" in x:
                    pb = x["perturbed_bins"]
                    extra = (f"; perturbed judged bin-angles {pb['perturbed_judged_bin_angles']}: {pb['failing_i']} fail (i) "
                         f"(per-row power {pb['per_row_power_i']:.2f}), "
                             f"{pb['z95_beyond_t']} with |z95| > t(N); must fail through {x['must_fail_through']}")
                return (f"(i) {j['i']['fail']} bins; (ii) beyond t {j['ii']['n_beyond_t']}, max |z| {j['ii']['max_abs_z']:.1f}; "
                        f"chi2 failing {len(j['ii']['chi2_fail'])} of 8; (iii) failing cells {len(j['iii']['fail_cells'])}; "
                        f"materiality {x['materiality_verdict']}" + extra)
            vals = [v(f"{n} fails as required", x["fails_as_required"], note=kb_note(n, x)) for n, x in kb["known_bads"].items()]
            fs = kb.get("reports", {}).get("facing_split")
            if fs:
                vals += [v("REPORTED (b) per-facing split: verdict", fs["verdict"],
                           note="facing pairs: " + "; ".join(f"{k} chi2 {x['chi2']:.1f}/{x['ndf']}, median |dc|/w "
                                                             f"{OSU_fmt(x['median_abs_dcentre_over_width'])}, w ratio "
                                                             f"{x['width_ratio_pct_10_50_90']}" for k, x in fs["pairs"].items()))]
            E.append({"id": f"p03.table.{ftag}.known-bads", "title": f"{head}: known-bads under the {lab} (fit 7-8, "
                      "test 3-6 and 9-10)",
                      "status": "valid", "bias": BIAS, "values": vals, "definition_sheet": kb.get("definition_sheet"),
                      "provenance": {**common_prov, "file": T.result_name("known-bad"), "n_rep": kb["n_rep"],
                                     "seed": kb.get("seed")}})
        gt, fo = load(T.result_name("gated")), load(T.result_name("fit-only"))
        tab_res = gt or fo
        if tab_res:
            tested = gt is not None
            verdict = gt["judgement"]["materiality"]["verdict"] if tested else None
            passed = verdict in ("pass", "minor")
            tf = tab_res["table_file"]
            doc = json.load(open(tf))
            vals = []
            for key in T.FRAME["keys"]:
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
                                 v(f"{nm} weighting width min / median / max", f"{min(w):.3f} / {np.median(w):.3f} / {max(w):.3f}",
                                   unit=unit),
                                 v(f"{nm} max |centre|", round(float(np.max(np.abs(c))), 3), unit=unit),
                                 v(f"{nm} q99 min / median / max", f"{min(q):.3f} / {np.median(q):.3f} / {max(q):.3f}",
                                   unit=unit)]
            vals.append(v("table file sha256", CM.file_digest(Path(tf)), note=tf))
            if tested:
                mt = gt["judgement"]["materiality"]
                vals += [v("gated verdict (amendment 1e rule)", verdict), v("MATERIAL deviations", "; ".join(mt["material"]) or "none"),
                         v("MINOR deviations (listed in the table header)", "; ".join(mt["minor"]) or "none")]
            st = "valid" if passed else "exploratory"
            note = (None if verdict == "pass" else
                    "ships with MINOR deviations (amendment 1e), listed in the table header and above" if verdict == "minor" else
                    f"a MATERIAL failure of the pre-declared test: back to the physicist; see p03.table.{ftag}.gated" if tested
                    else f"UNTESTED: the gated test did not run ({lab} not approved, or its known-good demonstration or a "
                    f"known-bad did not behave as required); see p03.table.{ftag}.*")
            E.append({"id": tab_id, "title": tab_title,
                      "status": st, "status_note": note, "bias": BIAS, "values": vals,
                      "definition_sheet": (gt or tab_res).get("definition_sheet"),
                      "provenance": {**common_prov, "table_file": tf, "fit_files": list(CM.FIT_FILES),
                                     "pt_check_refit_rows": tab_res["pt_check"]["flags"]}})
            pc = tab_res["pt_check"]["detail"]
            E.append({"id": f"p03.table.{ftag}.pt-check", "title": f"{head}: pT check (1-2 vs >= 2 GeV, fit sample)",
                      "status": "valid", "bias": BIAS, "definition_sheet": tab_res.get("definition_sheet"),
                      "values": [v(f"{k} {f}", json.dumps(pc[k]["summary"][f])) for k in T.FRAME["keys"]
                                 for f in ("t_N", "max_abs_z", "failing_rows", "failing_both_signs_refit_pt2",
                                           "failing_one_sign_reported", "p")],
                      "provenance": common_prov})
    T.set_frame("global")
    gt = load(T.result_name("gated"))
    ro = load("amend1b/recipe_options.json")
    if ro:
        vals = []
        for key, rows in ro["option_b_rows"].items():
            for r in rows:
                comp = r.get("two_component_fits") or []
                if not comp:
                    continue
                f = np.array([c["narrow_fraction"] for c in comp])
                vals.append(v(f"{r['row']} narrow fraction (mean over file pairs)", round(float(f.mean()), 4),
                              round(float(f.std(ddof=1)), 4) if len(f) > 1 else None,
                              note="per pair (narrow centre / width, broad centre / width) [published sigma]: " + "; ".join(
                                  f"{c['pair'][0]}-{c['pair'][1]}: {c['narrow_centre']:+.2f}/{c['narrow_width']:.2f}, "
                                  f"{c['broad_centre']:+.2f}/{c['broad_width']:.2f}" for c in comp)))
        E.append({"id": "p03.d10c.disc-beta-two-component", "title": "D10(c) record: two-Gaussian ML fits of the "
                  "disc-beta class-T pulls per table row and file pair (input of a later full response model)",
                  "status": "valid", "bias": BIAS + "; unbinned two-Gaussian EM fit of pulls clipped to +-50; error = sd over "
                  "the four 2-file pairs", "values": vals, "definition_sheet": "v2.1",
                  "provenance": {**common_prov, "file": "amend1b/recipe_options.json",
                                 "file_sha256": CM.file_digest(CM.OUT_DIR / "amend1b" / "recipe_options.json"),
                                 "code": "eval_refitq/p03_angles/recipe_options.py at commit 716d482 (removed in "
                                         "amendment 1d; reproduce from that commit)",
                                 "fit_files": ro["fit_files"], "pairs": ro["pairs"]}})
    roc = load("p03_roc.json")
    if roc:
        validated = bool(gt and gt["judgement"]["materiality"]["verdict"] in ("pass", "minor"))
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
                  "provenance": {**common_prov, "file": ex["response_file"], "fit_files": list(CM.FIT_FILES)}})
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
    ap.add_argument("--only", help="register only the entries whose id starts with this prefix")
    a = ap.parse_args(argv)
    if CM.ERA != "canon_v3":
        # the entries describe the canon_v3 runs (registry provenance, kept as it is): another era's results
        # need their own ids first, or --overwrite would replace the canon_v3 records
        raise SystemExit(f"registry_seed holds the canon_v3 P0.3 entries; era {CM.ERA} has no registry ids yet")
    E = [e for e in entries() if not a.only or e["id"].startswith(a.only)]
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
