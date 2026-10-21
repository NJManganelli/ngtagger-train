"""Registry entries of the WP-C entry package (src/ngtagger/spops/registry.py; working rule 6).

  tail      wpc.c1a.tail          the tail localisation (tail_summary.json), job dir of the variants job
  cuts      wpc.c1a.cuts-<ver>    a 99 % cut file (--ver; default common.CUTS), job dir of its cut fit; supersedes the
                                  previous version (v2: rr per scenario, PD15; v3: the smoothed leave-one-out, PD17;
                                  v5: PD19's rr population, PD20; v6: the fit files, PD21); v4 (the wide window, PD18)
                                  is registered not-used
  windowcheck wpc.c1a.window-gate99  PD18's file-4 validation of the wide window (entry/windowcheck_summary.json)
  cutcheck  wpc.c1a.cutcheck-<ver>   PD20's file-4 check of a cut file (entry/cutcheck_wpc_cuts_<ver>.json)
  c1b-il1   wpc.c1b.il1-truth          C1b A8 / A11: IL1's truth width, reference and findings (c1b/overlap/il1truth_summary.json)
  c1b-j1    wpc.c1b.j1-shape-fit       C1b J1: the disc shape fits, T0, D0, D2 (c1b/overlap/overlap_summary.json)
  c1b-j2    wpc.c1b.j2-seed-test-smooth  C1b J2 of the smooth seed, A17 (c1b/seed/seedtest_summary.json)
  c1b-seed  wpc.c1b.seed-model-v1      C1b A16: the smooth seed model (seedmodel/seed_model_v1.json, select.json)
  c1b-j4    wpc.c1b.j4-cut-fit         C1b J4: the cut fit (c1b/cuts/cutfit_summary.json)
  c1b-j5    wpc.c1b.j5-known-good-a20  C1b J5 under A20: file 4 with the known-bads (c1b/checks/checks_f04.json);
                                       supersedes wpc.c1b.j5-known-good (the A19 run, kept at c1b/checks/a19/)
  c1b-j6    wpc.c1b.j6-gate            C1b J6 under A20: file 6, the gate (c1b/checks/checks_f06.json)
  c1b-betasym wpc.c1b.beta-symmetry    C1b after J5: the disc beta symmetry figure (c1b/beta_symmetry.json)
  scan-tree2-kg wpc.scan.tree2-e7-known-good  scan A4: E7 of tree2 on file 4 (scan/tree2/e7_f04.json)
  scan-nonprompt wpc.scan.nonprompt-first-visit  scan B4: non-prompt first-visit calibration (scan/nonprompt_f04.json)
  scan-go   wpc.scan.go-table          scan B3: the GO / NO-GO table, 10 files (job dir of the last scan job)
  v3-closure wpc.v3.closure-<tag>       the v3 CMSSW closure, replay side (closure/closure_<tag>.json; --ver the tag;
                                        --job-dir: the port's closure job directory)
  scan-truthmod wpc.scan.truth-modules  the window's truth-target modules ON vs OFF, file 4 (scan/truthmod_f04.json)
  scan-bpre wpc.scan.bpre-d0-pull       scan B-pre: d0 pull cores, disc-going vs barrel-only, file 4
  c1b-verdict wpc.c1b.verdict          C1b closed with the discs ON and its known limitations; updates the J2 / J3 / J6
                                       statuses (job dir of J6)
  c1b-j3    wpc.c1b.j3-material-scale-smooth  C1b J3 with the smooth seed, A18 (c1b/scale/scalefit_summary.json)
  pseed     wpc.pseed.canon-v5-<scen>  the parallel seed study on canon_v5 (parallel_seed/results/match_<scen>.json and
                                  extras_<scen>.json; --ver the scenario); --job-dir: the scenario's matching job
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
from pathlib import Path

import common as CM
import verdict as VD
from ngtagger.spops import registry as R

BY = "WP-C entry (cmssw-l1t-expert, Claude Opus 5.5)"

CUT_VERSIONS = {"v2": "rr per scenario, PD15", "v3": "rr per scenario on the smoothed leave-one-out, PD15 + PD17",
                "v4": "the window as the 99 % hit cut, PD18, with PD19's rr population",
                "v5": "the 4-sigma window with PD19's rr population, PD20",
                "v6": "the 4-sigma window with PD19's rr population, on the fit files, PD20 + PD21"}
SUPERSEDES = {"v2": ("wpc.c1a.cuts", "the rr cuts pooled over scenarios biased AAAA against IAAA; refitted per scenario "
                                     "(PD15); the gate and window tables are unchanged"),
              "v3": ("wpc.c1a.cuts-v2", "the rr cuts' leave-one-out used the final (innermost) state and missed the "
                                        "scattering to the inner hits; refitted on the smoothed prediction (PD17); the "
                                        "gate and window tables are unchanged"),
              "v5": ("wpc.c1a.cuts-v3", "the rr population excludes a track's only SmartPixels cluster (PD19), with the "
                                        "4-sigma window (PD20); the gate and window tables are unchanged (no bin outside "
                                        "v3's intervals)"),
              "v6": ("wpc.c1a.cuts-v5", "refitted on every file but the known-good and the gate (PD21)")}
NOT_USED = {"v4": "fitted for the wide window (PD18), which failed its validation; the window went back to 4 sigma (PD20)"}
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


def cuts(job_dir, overwrite, ver=None):
    path = CM.CUTS if ver is None else CM.OUT / "cuts" / f"wpc_cuts_{ver}.json"
    ver = path.stem.rsplit("_", 1)[1]
    with open(path) as fh:
        C = json.load(fh)
    vals = [{"name": k, "value": v["*"]["value"], "note": f"n = {v['*']['n']} (pooled; per-bin values in the file)"}
            for k, v in C["cuts"].items()]
    files = sorted({int(f.rsplit("_f", 1)[1][:2]) for f in C["provenance"]["files"]})
    rid = f"wpc.c1a.cuts-{ver}"
    status, note = (("not-used", NOT_USED[ver]) if ver in NOT_USED else
                    ("in-sample", f"fitted on canon_v3 files {files}; files 1-2 are also the angle table's fit sample"))
    _reg({"id": rid, "title": f"WP-C C1a item 5: the replay's 99 % cuts ({ver}; files {files}, PD1 + PD8; "
                              f"{CUT_VERSIONS[ver]})",
          "status": status, "status_note": note, "values": vals,
          "bias": "none known; populations with a clean history (PD8), truth only for ownership (PD1)",
          "sources": [str(path)]}, job_dir, overwrite)
    if ver in SUPERSEDES:
        prev, why = SUPERSEDES[ver]
        R.set_status(prev, "superseded", why, by=BY, superseded_by=rid)


def windowcheck(job_dir, overwrite):
    p = CM.OUT / "entry" / "windowcheck_summary.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = [{"name": f"{scen} wide / 4-sigma", "value": json.dumps(
        {"wrong_share": [round(sc[k]["wrong_hits"] / max(sc[k]["accepted"], 1), 4) for k in ("wide", "nsig4")],
         "ncand_mean": [round(sc[k]["ncand"]["mean"], 2) for k in ("wide", "nsig4")],
         "loss_nohit": [round(sc[k]["loss_nohit"], 4) for k in ("wide", "nsig4")],
         "own_gained": sc["own_gained"], "wrong_gained": sc["wrong_gained"]})} for scen, sc in Q["scenarios"].items()]
    vals += [{"name": f"{scen} end-to-end", "value": json.dumps(e)} for scen, e in Q["end_to_end"].items()]
    _reg({"id": "wpc.c1a.window-gate99", "title": "WP-C PD18: the window as the 99 % hit cut against the 4-sigma ellipse (file 4)",
          "status": "valid", "status_note": "PD18 failed its pre-declared rule (wrong clusters gained far exceed correct "
                                            "ones); back to 4 sigma (PD20). " + "; ".join(Q["material"]),
          "values": vals, "bias": "file-4 validation, not a gate", "sources": [str(p)]}, job_dir, overwrite)


def cutcheck(job_dir, overwrite, ver=None):
    ver = ver or CM.CUTS.stem.rsplit("_", 1)[1]
    p = CM.OUT / "entry" / f"cutcheck_wpc_cuts_{ver}.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = [{"name": f"{scen} pooled coverage", "value": json.dumps(
        {k: [round(r["pooled"]["coverage"], 4), round(r["pooled"]["shortfall_bound95"], 4)]
         for k, r in sc["consumers"].items()})[:2000]} for scen, sc in Q["scenarios"].items()]
    vals += [{"name": f"{scen} end-to-end", "value": json.dumps(e)} for scen, e in Q["end_to_end"].items()]
    _reg({"id": f"wpc.c1a.cutcheck-{ver}", "title": f"WP-C PD20: the {ver} cuts' coverage and the end-to-end run (file 4)",
          "status": "valid" if not Q["material"] else "void",
          "status_note": "; ".join(Q["material"] + Q["minor"]) or "no failure",
          "values": vals, "bias": "file-4 check, not a gate", "sources": [str(p)]}, job_dir, overwrite)


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


def c1b_il1(job_dir, overwrite):
    p = CM.OUT / "c1b" / "overlap" / "il1truth_summary.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = [{"name": f"reference {ax}", "value": json.dumps([[r["size_bin"], r["n"], r["width_cm"], r["judged"]]
                                                            for r in Q["reference"][ax]])} for ax in ("u", "v")]
    vals += [{"name": f"v3 barrel tables over the reference {ax}", "value": json.dumps(Q["barrel_finding"][ax]["summary"])}
             for ax in ("u", "v")]
    vals += [{"name": f"RMS^2 / width^2 {ax} (occupancy-weighted)", "value": Q["mad_vs_rms"][ax]["variance_ratio"]}
             for ax in ("u", "v")]
    vals += [{"name": "u-v joint tail ratio", "value": Q["uv_independence"]["ratio"]},
             {"name": "count-only power", "value": json.dumps(Q["count_power"])}]
    _reg({"id": "wpc.c1b.il1-truth", "title": "WP-C C1b: IL1's truth width against p, its reference by predicted beam-pipe "
                                              "offset, and the findings (A8, A11)",
          "status": "in-sample", "status_note": "canon_v3 AAAA fit files (A10); findings reported only; " + "; ".join(Q["minor"]),
          "values": vals, "bias": "the reference's fixed point can run away in wide-cluster bins (u size 4)",
          "sources": [str(p)]}, job_dir, overwrite)


def c1b_j1(job_dir, overwrite):
    p = CM.OUT / "c1b" / "overlap" / "overlap_summary.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = [{"name": f"surface {s} R {ax} (um)", "value": json.dumps([round(x * 1e4, 2) if x == x else None
                                                                      for x in e["central"][f"R_{ax}"]])}
            for s, e in Q["fits"].items() for ax in ("u", "v")]
    vals += [{"name": f"D0 {ax}", "value": json.dumps({k: Q["D0"][ax][k] for k in ("pooled_ratio", "bounds95", "half_width95")})}
             for ax in ("u", "v")]
    vals += [{"name": f"D2 {s} ratio {ax}", "value": json.dumps([round(x, 3) if x == x else None for x in e[ax]["ratio"]])}
             for s, e in Q["D2"].items() for ax in ("u", "v")]
    _reg({"id": "wpc.c1b.j1-shape-fit", "title": "WP-C C1b J1: the disc position errors from overlap pairs, split-core shape "
                                                 "fits (A8, A11, A12)",
          "status": "exploratory", "status_note": "J1 stopped on its pre-declared rules (decision pending): "
                                                  + "; ".join(Q["material"] + Q["decisions"]),
          "values": vals, "bias": "the split-core family reads heavy-tailed shapes ~6 % wide (T0); size >= 2 bins not "
                                  "identifiable from overlaps", "sources": [str(p)]}, job_dir, overwrite)


def _width_vals(Q, keys):
    return [{"name": f"{k} {key}", "value": json.dumps({b: [e["n"], round(e["pu"], 3), round(e["pv"], 3)]
                                                         for b, e in Q[key].items() if e["n"] >= 200})[:2000]}
            for k, key in keys]


def c1b_j2(job_dir, overwrite):
    """J2 of the current seed (the A16 smooth seed, judged by A17); the A15 run (v3's seed, the relative MAD rule) was
    registered as wpc.c1b.j2-seed-test and stands."""
    p = CM.OUT / "c1b" / "seed" / "seedtest_summary.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = [{"name": "MATERIAL", "value": "; ".join(Q["material"]) or "none"},
            {"name": "D3d failing (reported)", "value": "; ".join(Q["D3d_fail"])[:2000] or "none"}]
    vals += _width_vals(Q, (("D3a (group|pT): n, u, v width (A17)", "D3a"), ("D3d barrel-only (surface|first|pT)", "D3d")))
    _reg({"id": "wpc.c1b.j2-seed-test-smooth",
          "title": "WP-C C1b J2: the smooth seed (A16) at material scale 1, judged absolutely on the clipped variance (A17)",
          "status": "in-sample", "status_note": "the fit files; " + ("stopped: the discs off (A16-A17)" if Q["material"]
                                                                     else "passes"),
          "values": vals, "bias": "the clipped variance counts the 2-6 % pull tails beyond 3 sigma that no seed or cluster "
                                  "model here describes; barrel-only tracks fail the same rule (D3d)",
          "sources": [str(p)]}, job_dir, overwrite)


def c1b_seed(job_dir, overwrite):
    p, q = CM.OUT / "seedmodel" / "seed_model_v1.json", CM.OUT / "seedmodel" / "select.json"
    with open(p) as fh:
        M = json.load(fh)
    with open(q) as fh:
        S = json.load(fh)
    vals = [{"name": "keying (held-out -2 log L per track: eta / ps / v3)",
             "value": f"{S['chosen']} ({S['keyings']['eta']['held_nll']:.4f} / {S['keyings']['ps']['held_nll']:.4f} / "
                      f"{S['held_nll_v3']:.4f})"},
            {"name": "H->tautau check", "value": json.dumps({k: S["htt"][k] for k in ("nll_minus_ttbar_held", "width_diff",
                                                                                     "include")})},
            {"name": "fit", "value": json.dumps(M["fit"])[:2000]},
            {"name": "parameters", "value": str(len(M["theta"]) * len(M["theta"][0]))}]
    _reg({"id": "wpc.c1b.seed-model-v1", "title": "WP-C C1b A16: the smooth two-surface seed model (barrel and discs)",
          "status": "in-sample", "status_note": "fitted on canon_v4 TTbar fit files and H->tautau x 3 scenarios",
          "values": vals, "bias": "material at scale 1 (an under-charge would sit in B); the clipped-likelihood fit leaves "
                                  "the pulls' clipped variance above 1 where the tails are heavy",
          "sources": [str(p), str(q)]}, job_dir, overwrite)


def c1b_j3(job_dir, overwrite):
    """J3 of the current seed (the A16 smooth seed, judged by A18); the A15 run (v3's seed) was registered as
    wpc.c1b.j3-material-scale and stands."""
    p = CM.OUT / "c1b" / "scale" / "scalefit_summary.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = [{"name": "scale (second-on / third-on)", "value": f"{Q['scale']:.3f} +- {Q['scale_sd']:.3f} / {Q['third_on']:.3f}"},
            {"name": "NLL over the grid", "value": json.dumps({"grid": Q["grid"], **Q["nll"]})},
            {"name": "MATERIAL", "value": "; ".join(Q["material"]) or "none"},
            {"name": "tail fractions beyond 3 sigma", "value": json.dumps(Q["tail_fractions"])}]
    vals += _width_vals(Q, (("D3a re-check (group|pT): n, u, v scale", "D3a"), ("D3b (disc|pT)", "D3b"),
                            ("D3c (surface|first|pT)", "D3c"), ("D3d barrel-only (surface|first|pT)", "D3d")))
    _reg({"id": "wpc.c1b.j3-material-scale-smooth",
          "title": "WP-C C1b J3: the disc material scale with the smooth seed, judged on the clipped-likelihood scale (A16, A18)",
          "status": "in-sample", "status_note": "the fit files; " + ("stopped: " + "; ".join(Q["material"]) if Q["material"]
                                                                     else "passes"),
          "values": vals, "bias": "a scale above 2 is physically suspect; the barrel-only tracks' IL1 / IL2 later visits "
                                  "show the same excess as D3c (D3d)", "sources": [str(p)]}, job_dir, overwrite)


def c1b_j4(job_dir, overwrite):
    p = CM.OUT / "c1b" / "cuts" / "cutfit_summary.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = [{"name": "D5 (barrel-only tracks identical, discs on / off)", "value": json.dumps(Q["D5"])},
            {"name": "disc material scale", "value": Q["disc_scale"]},
            {"name": "MATERIAL", "value": "; ".join(Q["material"]) or "none"},
            {"name": "cut file", "value": str(CM.OUT / "c1b" / "cuts" / "wpc_cuts_c1b_v1.json")}]
    _reg({"id": "wpc.c1b.j4-cut-fit", "title": "WP-C C1b J4: the cut fit with the disc, barrel-visit and barrel-only bins "
                                               "(smooth seed, scale 2.67; A9, A10, A16-A19)",
          "status": "in-sample", "status_note": "the fit files", "values": vals,
          "bias": "the disc material scale 2.67 is physically suspect (A19)", "sources": [str(p)]}, job_dir, overwrite)


def _c1b_checks(f, rid, title, bias, job_dir, overwrite):
    """J5 / J6 under A16-A20 (the A19 run of J5 is registered as wpc.c1b.j5-known-good; its output is kept at
    c1b/checks/a19/)."""
    p = CM.OUT / "c1b" / "checks" / f"checks_f{f:02d}.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = [{"name": "MATERIAL", "value": "; ".join(Q["material"]) or "none"},
            {"name": "MINOR", "value": "; ".join(Q["minor"]) or "none"},
            {"name": "D1 failing bins", "value": json.dumps([{k: b[k] for k in ("angle", "surface", "keybin", "n", "width")}
                                                            for b in Q["D1"]["pooled"]["bins"] if b["fail"]])},
            {"name": "D1 exempt bins (A20, not judged)", "value": json.dumps(Q["D1"]["exempt"])},
            {"name": "D5", "value": json.dumps(Q["D5"])}]
    if "known_bads" in Q:
        vals += [{"name": "known-bads (fails?)", "value": json.dumps(Q["known_bads"])},
                 {"name": "D1 false veto", "value": json.dumps(Q["D1"]["false_veto"])},
                 {"name": "power statements", "value": json.dumps(Q["power"])}]
    _reg({"id": rid, "title": title, "status": "valid" if not Q["material"] else "void",
          "status_note": "; ".join(Q["material"]) or "no failure", "values": vals, "bias": bias,
          "sources": [str(p), Q["d1_exempt_file"]]}, job_dir, overwrite)


def c1b_j5(job_dir, overwrite):
    _c1b_checks(4, "wpc.c1b.j5-known-good-a20", "WP-C C1b J5: file 4, the final configuration with the known-bads "
                "(A16-A20)", "file 4, the rehearsal; D3 on A18 / A19's statistic; D1 without A20's exempt bins",
                job_dir, overwrite)
    R.set_status("wpc.c1b.j5-known-good", "superseded", "rerun under A20 (the D1 exemption; the radius-only known-bad "
                 "as a power statement); this run's output is kept at c1b/checks/a19/checks_f04.json", by=BY,
                 superseded_by="wpc.c1b.j5-known-good-a20")


def c1b_j6(job_dir, overwrite):
    _c1b_checks(6, "wpc.c1b.j6-gate", "WP-C C1b J6: file 6, the gate, the final configuration (A16-A20)",
                "file 6, judged once; D3 on A18 / A19's statistic; D1 without A20's exempt bins", job_dir, overwrite)


C1B_LIMITATIONS = [
    "J6 (file 6): D3b ID1 later visits at pT 10-20 GeV, u: predicted errors too large, scale 0.700 +- 0.043 (n 343); "
    "file 4 0.76, fit files 0.87; safe for efficiency, some wrong-cluster rejection lost in that bin",
    "D1 exemption (A20): beta in the innermost populated disc angle bin, 9.5-11.3 deg (|eta| 2.3-2.5), both signs, ID1-ID4: "
    "widths 1.11-1.27; a stand-in artefact of the shared response's small-|beta| plateau; the table's beta width, not the "
    "prediction",
    "the disc material scale 2.67 +- 0.06: physically suspect (unmodelled forward material, or standing in for the barrel "
    "excess)",
    "the barrel's shared IL1 / IL2 later-visit excess: errors 9-19 % too small for every track (barrel-only and disc-going)",
    "tails: 1.6-5.8 % of the innovation pulls beyond 3 sigma per family; core-width judging does not cover them",
    "the seed model above |eta| 2.25 at |p| 10-15 GeV rises faster than the data (sqrt dC(z0) ~12 mm vs ~7.5 mm at |eta| 2.5)",
    "power: D3 does not catch a 1.3x material scale, nor radius-only disc material (D3b shift median 0.02, max 0.13)",
]


def c1b_verdict(job_dir, overwrite):
    """C1b's verdict (the physicist's decision, 2026-10-18): closed with the discs ON, with the known limitations; the
    statuses of J2 / J3 (re-judged under A18 / A19) and J6 follow."""
    out = CM.OUT / "c1b"
    with open(out / "checks" / "checks_f04.json") as fh:
        J5 = json.load(fh)
    with open(out / "checks" / "checks_f06.json") as fh:
        J6 = json.load(fh)
    with open(out / "scale" / "scalefit_summary.json") as fh:
        S = json.load(fh)
    vals = [{"name": "verdict", "value": "closed, discs ON (INN-D: smooth seed v1, path-ordered disc material at the fitted "
                                         "scale, v3's position errors, the module-frame disc angle entries, cuts c1b v1)"},
            {"name": "disc material scale", "value": f"{S['scale']:.3f} +- {S['scale_sd']:.3f}"},
            {"name": "J5 (file 4) MATERIAL", "value": "; ".join(J5["material"]) or "none"},
            {"name": "J6 (file 6) MATERIAL, accepted as a known limitation", "value": "; ".join(J6["material"]) or "none"},
            {"name": "waivers", "value": "A18 (clipped-likelihood scale), A19 (D3c relative to D3d), A20 (D1 beta edge "
                                         "exemption; radius-only known-bad as a power statement)"}]
    vals += [{"name": f"known limitation {i + 1}", "value": t} for i, t in enumerate(C1B_LIMITATIONS)]
    _reg({"id": "wpc.c1b.verdict", "title": "WP-C C1b: the replay's disc calibration, verdict (discs ON)",
          "status": "valid", "status_note": "closed by the physicist's decision of 2026-10-18 with the known limitations "
                                            "listed; the gate's one failure is limitation 1",
          "values": vals, "bias": "the angles are placeholders (the shared response); see the known limitations",
          "sources": [str(out / "checks" / "checks_f04.json"), str(out / "checks" / "checks_f06.json"),
                      str(out / "scale" / "scalefit_summary.json"), str(out / "PLAN.md")]}, job_dir, overwrite)
    R.set_status("wpc.c1b.j6-gate", "valid", "one MATERIAL (D3b ID1 later, pT 10-20 GeV, u 0.700) accepted as a known "
                 "limitation; the package closed with the discs ON (wpc.c1b.verdict)", by=BY)
    R.set_status("wpc.c1b.j2-seed-test-smooth", "in-sample", "re-judged under A18 (c1b/seed/seedtest_a18.json): D3a "
                 "passes on the clipped-likelihood scale; the A17 stop is withdrawn", by=BY)
    R.set_status("wpc.c1b.j3-material-scale-smooth", "in-sample", "re-judged under A19 (c1b/scale/scalefit_a19.json): "
                 "passes; D3c judged relative to D3d; second-on 2.671 / third-on 2.705 accepted", by=BY)


def scan_tree2_kg(job_dir, overwrite):
    """Scan Part A: E7 of tree2 (PD14 with the discs) on its file-4 known-good (wpc/scan/tree2/e7_f04.json)."""
    p = CM.OUT / "scan" / "tree2" / "e7_f04.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = [{"name": "MATERIAL", "value": "; ".join(Q["material"]) or "none"}]
    for scen, e in Q["scenarios"].items():
        pl = e["e7"]["pooled"]
        vals += [{"name": f"{scen} pooled skip-loss (sd, n)", "value": f"{pl['loss']:.4f} ({pl['sd']:.4f}, {pl['n']})"},
                 {"name": f"{scen} known-bads: P at 95 % / tail probability (shift in 1.5 % margins)",
                  "value": f"{e['known_bad_p95']['loss']:.4f} ({e['known_bad_p95']['shift_in_margins']:.2f}) / "
                           f"{e['known_bad_tailprob']['loss']:.4f} ({e['known_bad_tailprob']['shift_in_margins']:.1f})"},
                 {"name": f"{scen} wrong-hit rejection gained; first-step window report",
                  "value": json.dumps({"rejection": e["e7"]["wrong_hit_rejection_gained"], "window": e["window_report"]})}]
    _reg({"id": "wpc.scan.tree2-e7-known-good", "title": "WP-C scan A4: E7 skip-loss of the two-layer hit choice on "
                                                         "file 4 (known-good)",
          "status": "valid", "status_note": "; ".join(Q["material"]) or "passes", "values": vals,
          "bias": "file 4, the known-good; P from c1b v1 (fit files)", "sources": [str(p)]}, job_dir, overwrite)


def scan_bpre(job_dir, overwrite):
    """Scan B-pre: the d0 pull core of disc-going against barrel-only prompt tracks on file 4, with the fixed inner
    material and the pre-fix rule (the known-bad) (wpc/scan/bpre/dcheck_f04.json)."""
    p = CM.OUT / "scan" / "bpre" / "dcheck_f04.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = [{"name": "MATERIAL", "value": "; ".join(Q["material"]) or "none"},
            {"name": "known-bad (pre-fix rule) bins disagreeing",
             "value": json.dumps([k for k, e in Q["bins"].items() if k.startswith("radius|") and e.get("agrees") is False])}]
    vals += [{"name": f"{k} d0 pull core disc-going / barrel-only",
              "value": f"{e['disc_going']['core']:.3f} +- {e['disc_going']['sd']:.3f} / "
                       f"{e['barrel_only']['core']:.3f} +- {e['barrel_only']['sd']:.3f}"}
             for k, e in Q["bins"].items() if "agrees" in e]
    _reg({"id": "wpc.scan.bpre-d0-pull", "title": "WP-C scan B-pre: the inner material of disc-going tracks, d0 pull "
                                                  "cores on file 4",
          "status": "valid", "status_note": "; ".join(Q["material"]) or "passes; the pre-fix rule fails in IAAA",
          "values": vals, "bias": "file 4; prompt tracks only", "sources": [str(p)]}, job_dir, overwrite)


def scan_nonprompt(job_dir, overwrite):
    """Scan B4 step 3: the first-visit calibration of non-prompt particles on file 4 (wpc/scan/nonprompt_f04.json)."""
    p = CM.OUT / "scan" / "nonprompt_f04.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = [{"name": "known limitations", "value": "; ".join(Q["known_limitations"]) or "none"}]
    vals += [{"name": f"{e['scen']} {e['region']} {e['class']} {e['axis']}", "value": f"{e['scale']:.3f} +- {e['sd']:.3f} "
              f"(n {e['n']})"} for e in Q["bins"] if e["n"] >= 30]
    _reg({"id": "wpc.scan.nonprompt-first-visit", "title": "WP-C scan B4: the first-visit innovation calibration of "
                                                           "heavy-flavour daughters and particles born beyond IL1 (file 4)",
          "status": "valid", "status_note": "failures are known limitations of the prompt-fitted seed model",
          "values": vals, "bias": "file 4; INN-D with K = 1", "sources": [str(p)]}, job_dir, overwrite)


def scan_go(job_dir, overwrite):
    """Scan B3: the GO / NO-GO table for refit v3 on the 10 files (wpc/scan/go_table_01_..._10.json), with the dry run,
    the fit-file and the out-of-fit tables as sources."""
    out = CM.OUT / "scan"
    p = out / "go_table_01_02_03_04_05_06_07_08_09_10.json"
    with open(p) as fh:
        Q = json.load(fh)
    m = Q["main"]
    vals = [{"name": "verdict (INN-D alphaBeta - PM alphaBeta, pre-declared)", "value": "GO" if m["go"] else "NO-GO"},
            {"name": "reasons", "value": "; ".join(m["why"]) or "none"},
            {"name": "MATERIAL", "value": "; ".join(Q["material"]) or "none"},
            {"name": "controls (null NO-GO without gain; positive GO)",
             "value": json.dumps({"null_go": Q["control_null"]["go"], "null_no_gain": Q["control_null"]["no_gain"],
                                  "positive_go": Q["control_positive"]["go"]})}]
    for scen, by in m["diffs"].items():
        for cls, e in by.items():
            vals.append({"name": f"{scen} {cls}: INN-D / PM-ab, difference +- sd",
                         "value": json.dumps({k: [round(v["a"], 4), round(v["b"], 4), round(v["diff"], 4), round(v["sd"], 4)]
                                              for k, v in e.items()})})
    for c, e in Q["contrasts"].items():
        vals.append({"name": f"{c} [{e['family']}]: attachment / H / P difference (AAAA, AAAI, IAAA x barrel-only, "
                             "disc-going)",
                     "value": json.dumps({s: {cls: [round(x[k]["diff"], 4) for k in ("att", "H", "P")] for cls, x in by.items()}
                                          for s, by in e["diffs"].items()})})
    srcs = [str(p)] + [str(out / f) for f in ("go_table_04.json", "go_table_01_02_03_05_07_08_09_10.json",
                                              "go_table_04_06.json")]
    _reg({"id": "wpc.scan.go-table", "title": "WP-C scan: the GO / NO-GO table for refit v3 (canon_v4, 10 files, tree2)",
          "status": "in-sample", "status_note": ("NO-GO" if not m["go"] else "GO") + " by the pre-declared rule; in-sample "
                                                "for INN-D's calibrations (accepted: placeholder angles); files 4 + 6 "
                                                "agree", "values": vals,
          "bias": "in-sample for INN-D's seed model, material scale and cuts; the HF IP-significance proxy also moves "
                  "with the covariance calibration (PM's reported sigma(d0) is small)", "sources": srcs},
         job_dir, overwrite)


def scan_truthmod(job_dir, overwrite):
    """The v3 port's question (2026-10-20): the ellipse window's truth-target modules, ON against OFF, on file 4
    (wpc/scan/truthmod_f04.json)."""
    p = CM.OUT / "scan" / "truthmod_f04.json"
    with open(p) as fh:
        Q = json.load(fh)
    vals = []
    for scen, by in Q["scenarios"].items():
        for cls, e in by.items():
            vals.append({"name": f"{scen} {cls}: ON - OFF (att, H, P); ON - PM / OFF - PM attachment; target-only share "
                                 "of accepted hits",
                         "value": json.dumps({"on_off": [e["ON-OFF"][k]["diff"] for k in ("att", "H", "P")],
                                              "on_pm": e["ON-PM"]["att"]["diff"], "off_pm": e["OFF-PM"]["att"]["diff"],
                                              "tgt_only_accepted": e["tgt_only_share_of_accepted"]})})
    _reg({"id": "wpc.scan.truth-modules", "title": "WP-C: the ellipse window's truth-target modules, ON vs OFF (file 4)",
          "status": "valid", "status_note": "no effect: identical attachment, H and P; no accepted target-only cluster",
          "values": vals, "bias": "file 4 only; INN-D with K = tree2", "sources": [str(p)]}, job_dir, overwrite)


V3_PAYLOAD_NOTE = {
    "AAAA_f04_10ev": "the closure nano was built with the CMSSW payload's first byte version (99a036cd...); the values "
                     "are identical to the second's (data/refit_v3/v1/HISTORY.md in the arm tree)",
    **{f"{s}_f03_100ev": "built with the CMSSW payload's second byte version (skip_price.json sha256 07eec3fc...)"
       for s in CM.SCENARIOS}}


def v3_closure(job_dir, overwrite, ver=None):
    """The v3 CMSSW closure, the replay side, on the port's closure nano (wpc/closure/closure_<tag>.json; --ver = the
    tag, default AAAA_f04_10ev)."""
    tag = ver or "AAAA_f04_10ev"
    p = CM.OUT / "closure" / f"closure_{tag}.json"
    with open(p) as fh:
        Q = json.load(fh)
    s, f = Q["selection"], Q["forced"]
    vals = [{"name": "inputs / geometry / kpt", "value": f"{Q['inputs']} / {Q['geometry']} / {Q['kpt']}"},
            {"name": "paired tracks (unpaired)", "value": f"{Q['paired_tracks']} ({Q['unpaired_tracks']})"},
            {"name": "selection: crossings agreeing", "value": f"{s['agreeing']} of {s['crossings']}"},
            {"name": "selection: first disagreements beyond eps / near-ties (eps, cap)",
             "value": f"{s['hard']} / {s['near_ties']} ({s['eps']:.3g}, {s['cap']:.1f})"},
            {"name": "score check: agreeing crossings with committing scores differing by more than eps (checked; max)",
             "value": f"{s['score_over_eps']} ({s['score_checked_crossings']}; {s['max_dscore_agreeing']:.3g})"},
            {"name": "forced: max |da|/sigma pre / post / final",
             "value": f"{f['max_da_pre']:.3g} / {f['max_da_post']:.3g} / {f['max_da_final']:.3g}"},
            {"name": "forced: max |dC|/sqrt(CiiCjj) pre / post / final",
             "value": f"{f['max_dC_pre']:.3g} / {f['max_dC_post']:.3g} / {f['max_dC_final']:.3g}"},
            {"name": "forced: tracks over 1e-6", "value": f"{f['n_over_tolerance']} of {f['tracks']}"},
            {"name": "CMSSW payload", "value": V3_PAYLOAD_NOTE.get(tag, "not recorded")}]
    ok = s["passes"] and f["passes"]
    _reg({"id": f"wpc.v3.closure-{tag.lower().replace('_', '-')}",
          "title": f"WP-C: the v3 CMSSW closure, replay side ({tag})", "status": "valid",
          "status_note": ("passes" if ok else "fails") + " v3cmssw/PLAN.md rev 1 §3 items 4-5", "values": vals,
          "bias": "the closure arm reads the dump's double seed helix and the float32 module geometry (CMSSW's)",
          "sources": [str(p), Q["nano"], Q["dump"]]}, job_dir, overwrite)


def c1b_betasym(job_dir, overwrite):
    """The read-only disc beta symmetry figure (after J5): the innermost populated key bins, k < 0 (J5's failing bin)
    and its mirror k > 0, per disc."""
    p = CM.OUT / "c1b" / "beta_symmetry.json"
    with open(p) as fh:
        Q = json.load(fh)

    def w(t):
        return f"{t['width']:.3f} +- {t['sd_w']:.3f} (n {t['n']})" if t else "-"
    vals = []
    for S, d in Q.items():
        for b, lab in ((8, "k in (-0.2, -0.1)"), (11, "k in (0.1, 0.2)")):
            vals.append({"name": f"disc {S}, {lab}: pooled / row at the true key / same sign / per facing x side",
                         "value": " / ".join([w(d["pooled"][b]), w(d["true_key"][b]), w(d["same_sign"][b])]) + "; " +
                                  json.dumps({k: w(v[b]) for k, v in d["series"].items()})})
        vals.append({"name": f"disc {S}: wrong-sign / row-change fraction per key bin",
                     "value": json.dumps({"wrong_sign": d["wrong_sign_fraction"], "row_change": d["row_change_fraction"]})})
    _reg({"id": "wpc.c1b.beta-symmetry", "title": "WP-C C1b: disc beta pull symmetry (facing x z side, k vs -k, the "
                                                  "true-key row) after J5",
          "status": "in-sample", "status_note": "read-only diagnostic on the fit files (no file 4 or 6); no judgement",
          "values": vals, "bias": "the width is D1's (the table's weighting width), bootstrap sd over 100 event replicas",
          "sources": [str(p), str(CM.OUT / "c1b" / "beta_symmetry_rows.npz")]}, job_dir, overwrite)


PSEED = CM.OUT.parent / "parallel_seed"
PSEED_CONTEXT = ("context, not arms: the 16 Sep run (AAAA, 400 events, barrel only, OT tracks re-found by the census, the "
                 "old truth rules): 76.5 IT mini-tracks and 405 OT tracks per event, 62.8 % matched, 97.2 % right; the "
                 "4-layer barrel dev test (canon_v5 AAAA file 1, 4 events, today's rules): 114 IT mini-tracks per event, "
                 "91 % owned, 69 % matched, 100 % right, 95 % per particle")


def pseed(job_dir, overwrite, ver=None):
    """The parallel seed study on canon_v5, one scenario (--ver; parallel_seed/PLAN.md): IT-only mini-tracks (3 IT layers
    minimum, barrel and discs) matched one-to-one to CMSSW's L1TTracks at the 16 Sep default (phi0 + z0 + cot,
    chi2 < 25)."""
    scen = ver or "AAAA"
    p, q = PSEED / "results" / f"match_{scen}.json", PSEED / "results" / f"extras_{scen}.json"
    with open(p) as fh:
        Q = json.load(fh)
    with open(q) as fh:
        X = json.load(fh)
    D, P, H = Q["default"], X["particles"], X["hits"]
    if P["with_both"] != D["per_particle"]["particles"] or X["n_events"] != Q["n_events"]:
        raise SystemExit(f"{scen}: the extras and the matching pass disagree on their particles or events")
    pp = D["per_particle"]["matched_right"]
    n = D["per_particle"]["particles"]

    def fr(t):
        return f"{100 * t['value']:.2f} +- {100 * t['sd']:.2f} % (of {t['den']})"

    def bins(lab):
        return "; ".join(f"[{b['lo']:g}, {'inf' if b['hi'] is None else format(b['hi'], 'g')}): matched "
                         f"{100 * b['match']['value']:.2f} +- {100 * b['match']['sd']:.2f} %, right "
                         f"{100 * b['right']['value']:.1f} +- {100 * b['right']['sd']:.1f} %" for b in D[lab])
    vals = [{"name": "events", "value": Q["n_events"]},
            {"name": "per particle: particles owning both an IT mini-track and an OT track that are matched to each "
                     "other", "value": round(pp, 4), "error": round((pp * (1 - pp) / n) ** 0.5, 4),
             "note": f"of {n} particles ({P['with_both_per_event']:.1f} per event); binomial sd (no event bootstrap)"},
            {"name": "per particle: particles with an OT track that also own an IT mini-track",
             "value": round(P["both_share_of_ot"], 4),
             "note": f"{P['with_ot_track_per_event']:.2f} particles with an OT track per event"},
            {"name": "per particle: particles with an OT track that are matched to their own IT mini-track",
             "value": round(pp * P["both_share_of_ot"], 4)},
            {"name": "IT mini-tracks per event", "value": round(Q["it_per_event"], 1),
             "note": f"{100 * Q['it_owned']:.1f} % owned by a particle"},
            {"name": "IT mini-tracks with exactly 3 hits", "value": round(H["3"]["share"], 4),
             "note": f"{100 * H['3']['owned']:.1f} % of them owned; {100 * H['3']['share_of_fakes']:.1f} % of all "
                     f"unowned IT mini-tracks have 3 hits; 4 hits {100 * H['4']['share']:.1f} % "
                     f"({100 * H['4']['owned']:.0f} % owned), 5 or more {100 * H['5+']['share']:.1f} % "
                     f"({100 * H['5+']['owned']:.0f} % owned)"},
            {"name": "OT tracks per event (CMSSW L1TTracks)", "value": round(Q["ot_per_event"], 2),
             "note": f"{100 * Q['ot_owned']:.1f} % with a truth match"},
            {"name": "per track: IT mini-tracks matched", "value": round(D["match"]["value"], 5),
             "error": round(D["match"]["sd"], 5),
             "note": f"the ceiling is OT / IT tracks = {100 * Q['ot_per_event'] / Q['it_per_event']:.2f} %; "
                     f"{100 * D['match']['value'] * Q['it_per_event'] / Q['ot_per_event']:.1f} % of OT tracks get an IT "
                     "partner"},
            {"name": "per track: matches to the right OT track", "value": round(D["right"]["value"], 4),
             "error": round(D["right"]["sd"], 4)},
            {"name": "per track against pT (GeV; the matched OT track's, else the IT fit's)", "value": bins("pt")},
            {"name": "per track against |eta| (the IT fit's)", "value": bins("abs_eta")},
            {"name": "scan: matched / right per parameter set and chi2 cut",
             "value": json.dumps({k: [round(v["match_fraction"], 4), round(v["right_fraction"], 4)]
                                  for k, v in Q["scan"].items()})},
            {"name": "the refit's agreement with the census's exported helix (max)", "value": Q["refit_vs_export_max"]},
            {"name": "context", "value": "16 Sep run and 4-layer barrel dev test", "note": PSEED_CONTEXT}]
    jobs = sorted(d.name for d in Path(job_dir).parent.glob(f"*-pseed-census-{scen}-f*"))
    _reg({"id": f"wpc.pseed.canon-v5-{scen.lower()}",
          "title": f"Parallel seed study on canon_v5 {scen}: IT-only mini-tracks (3 IT layers minimum, barrel and discs) "
                   "matched to CMSSW's L1TTracks",
          "status": "valid", "status_note": "parallel_seed/PLAN.md as approved, with the 3-layer decision (2026-10-21)",
          "values": vals,
          "bias": "descriptive: the angles are placeholders and class C the CNN stand-in; canon_v5 TTbar files 1-2 (200 "
                  "events); the OT side is CMSSW's L1TTracks (the 16 Sep run re-found OT tracks in the census, the one "
                  "change of input); greedy one-to-one match, no refit on the union of hits (the OT tracks carry no "
                  "census stub indices); with 3 IT layers the IT side is mostly 3-hit fakes",
          "provenance": {"census_jobs": jobs, "extras": "eval_refitq/combinatorics/arch_parallel_extras.py at 03b28f9, "
                                                        "run directly (read-only, light)"},
          "sources": [str(p), str(q), str(PSEED / "PLAN.md")] + Q["census"] + Q["input"]}, job_dir, overwrite)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("tail", "cuts", "entry", "final", "loocheck", "windowcheck", "cutcheck", "c1b-il1",
                                     "c1b-j1", "c1b-j2", "c1b-j3", "c1b-seed", "c1b-j4", "c1b-j5",
                                     "c1b-j6", "c1b-verdict", "c1b-betasym", "scan-tree2-kg",
                                     "scan-bpre", "scan-nonprompt", "scan-go",
                                     "scan-truthmod", "v3-closure", "pseed"))
    ap.add_argument("--ver", help="cuts / cutcheck: the cut file's version (default: the one in use); v3-closure: the tag; "
                                    "pseed: the scenario")
    ap.add_argument("--job-dir", required=True, help="the launcher's job directory (launch-time provenance)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.what in ("cuts", "cutcheck", "v3-closure", "pseed"):
        {"cuts": cuts, "cutcheck": cutcheck, "v3-closure": v3_closure, "pseed": pseed}[a.what](a.job_dir, a.overwrite,
                                                                                              a.ver)
    else:
        {"tail": tail, "entry": entry, "final": final, "loocheck": loocheck, "windowcheck": windowcheck,
         "c1b-il1": c1b_il1, "c1b-j1": c1b_j1, "c1b-j2": c1b_j2, "c1b-j3": c1b_j3,
         "c1b-seed": c1b_seed, "c1b-j4": c1b_j4, "c1b-j5": c1b_j5,
         "c1b-j6": c1b_j6, "c1b-verdict": c1b_verdict, "c1b-betasym": c1b_betasym,
         "scan-tree2-kg": scan_tree2_kg, "scan-bpre": scan_bpre, "scan-nonprompt": scan_nonprompt,
         "scan-go": scan_go, "scan-truthmod": scan_truthmod}[a.what](a.job_dir, a.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
