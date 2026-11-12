"""Fit the shipped parametrization on ALL scan events and write it as correctionlib JSON, one file per scenario
(PLAN.md section 7b): doc-archive/spix-bits-refit-study/parametrization/spixbits_<scenario>.json.gz, with the
reference helper spixbits_apply.py and a consistency check (helper through correctionlib == the fit code's numpy
evaluation, the same draws given to both).

  python param_export.py [--scen AAAA AAAI IAAA] [--root DIR] [--out DIR] [--allocs ...]

Corrections in a file (inputs in SmartPixels-study units: cm, GeV, 1/cm; `allocation` is one of the shipped names,
anything else raises):
  uniform, normal   the per-track draws: hashprng of (run, lumi, event, track, draw); draw 1 = state, 2 = rank,
                    3 = score, 4 = covariance (uniform); 11-14 = z0, tanL, 1/R, phi0 (normal)
  p_changed         share of tracks with a material hit-choice part against today
  eps               precision part: inverse CDF of the parameter change in units of the sigma the scenario adds
  lambda, cov_scale, cov_eta, rho   its scale, the covariance-diagonal ratio (line, residual), the copula correlations
  score_map, score_eta       precision part: map of the heads' chi2-like inputs and hit aggregates (coefficients, residual)
  donor             hit-choice part: the donor's values, picked by the rank draw
The six head outputs are not tabulated: the helper evaluates the head models shipped beside the files (heads/retrained:
the best of 20 per detector scenario; heads/deployed) on the emulated inputs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

import param_model as PM

OUT = Path.home() / "smartpixels/doc-archive/spix-bits-refit-study/parametrization"
HD_NAMES = ["r_inv_pt", "r_tanL", "dz0_refit", "dtanl_refit", "dinvpt_refit", "spixRefitPerformed", "spixAnyWindowTruncated",
            "spixNCrossings", "spixNAcceptedHits", "spixLayerHitMask", "spixMaxWindowMult", "spixChi2IncXTot", "spixChi2IncYTot",
            "spixChi2IncAlphaTot", "spixChi2ITAtOT", "spixChi2ITAtRefit", "spixShiftChi2", "spixLogDetRatio", "h_nangle",
            "h_chi2_angle_per_cl", "h_chi2_pos_per_hit", "h_mean_wmult", "h_max_angle_pull", "h_second_angle_pull", "h_max_pos_pull"]
INF = "inf"
V = {"allocation": ("string", "shipped allocation name"), "parameter": ("string", "rInv, phi0, tanL, z0 or d0"),
     "matched": ("int", "1 if the OT track has a tracking particle (otTpIdx >= 0), else 0"),
     "window": ("int", "today's window: 0 prompt (|d0| <= 0.02 cm), 1 displaced, 2 not refit today"),
     "window_arm": ("int", "window of the EMULATED d0: 0 prompt, 1 displaced"),
     "abs_eta": ("real", "|eta| of today's refit track (asinh of tanL)"), "pt": ("real", "today's refit pT [GeV]"),
     "mult": ("real", "today's spixMaxWindowMult"), "n_hits": ("real", "today's spixNAcceptedHits"),
     "pull": ("real", "max(|d0 - truth d0| / sigma_d0, |z0 - truth z0| / sigma_z0) of today's track; 0 for a track without a particle"), "clean_margin": ("real", "today's clean head as a margin: logit of head_clean"),
     "u": ("real", "a uniform draw in [0, 1)"), "tail": ("real", "|2 u_rank - 1|"),
     "score": ("string", "a mapped head input: a chi2-like spix* column or an h_* aggregate over the accepted hits"),
     "layer_mask": ("int", "today's spixLayerHitMask"),
     "coefficient": ("string", "a, b (or sigma for score_map)"), "quantity": ("string", "donor quantity"),
     "run": ("int", "run"), "lumi": ("int", "luminosity block"), "event": ("int", "event"),
     "track": ("int", "index of the track along axis 1 of the event's L1TSmartPixelsExtTrack collection"),
     "draw": ("int", "which draw (see the file description)")}


def var(n):
    return {"name": n, "type": V[n][0], "description": V[n][1]}


def binning(name, edges, content, flow="clamp"):
    return {"nodetype": "binning", "input": name, "edges": [float(e) if e != INF else INF for e in edges], "content": content, "flow": flow}


def category(name, items, default=None):
    node = {"nodetype": "category", "input": name, "content": [{"key": k, "value": v} for k, v in items]}
    if default is not None:
        node["default"] = default
    return node


def ints(name, n, fn):
    return category(name, [(i, fn(i)) for i in range(n)])


def eta_bins(fn):
    return binning("abs_eta", [0.0, *PM.ETA_EDGES, INF], [fn(i) for i in range(3)])


def pt_bins(fn):
    return binning("pt", [0.0, *PM.PT_EDGES, INF], [fn(i) for i in range(3)])


def inv_cdf_node(q):
    """Piecewise-linear inverse CDF in u with constant ends (the clamp)."""
    k = PM.KNOTS
    content = [float(q[0])]
    for i in range(len(k) - 1):
        slope = (q[i + 1] - q[i]) / (k[i + 1] - k[i])
        content.append({"nodetype": "formula", "expression": "[0]+[1]*x", "parser": "TFormula", "variables": ["u"],
                        "parameters": [float(q[i] - slope * k[i]), float(slope)]})
    content.append(float(q[-1]))
    return binning("u", [0.0, *k, 1.0], content)


def correction(name, desc, inputs, data, out="value"):
    return {"name": name, "description": desc, "version": 1, "inputs": [var(i) for i in inputs],
            "output": {"name": out, "type": "real"}, "data": data}


def build(tables, terc, meta, mask_classes):
    """The CorrectionSet dict from {allocation: tables of param_model.fit}."""
    al = list(tables)
    per = lambda fn: category("allocation", [(a, fn(tables[a])) for a in al])
    ids = ["run", "lumi", "event", "track", "draw"]
    cs = [correction("uniform", "uniform [0, 1) draw per track", ids, {"nodetype": "hashprng", "inputs": ids, "distribution": "stdflat"}),
          correction("normal", "standard normal draw per track", ids, {"nodetype": "hashprng", "inputs": ids, "distribution": "stdnormal"})]
    terc_edges = ["-inf", float(terc[0]), float(terc[1]), INF]
    cs.append(correction("p_changed", "share of tracks whose accepted clusters or refit flag differ from today's",
                         ["allocation", "matched", "window", "abs_eta", "pt", "mult", "clean_margin", "pull"],
                         per(lambda T: ints("matched", 2, lambda m: ints("window", 3, lambda w: eta_bins(lambda e: pt_bins(
                             lambda p: binning("mult", [0.0, *PM.MULT_EDGES, INF], [binning("clean_margin", terc_edges, [binning("pull", [0.0, *PM.PULL_EDGES, INF],
                                                                                                                         [float(T["p"][m, w, e, p, k, t, q]) for q in range(3)])
                                                                                                                 for t in range(3)])
                                                                                for k in range(4)]))))))))
    par = lambda fn: category("parameter", [(p, fn(p)) for p in PM.PARAMS])
    cs.append(correction("eps", "same-cluster change of a parameter in units of the sigma the allocation adds on the track, "
                                "sqrt(|C_arm - C_today| + (0.05 sigma_today)^2) with C_arm from cov_scale: inverse CDF at u",
                         ["allocation", "parameter", "abs_eta", "window", "u"],
                         per(lambda T: par(lambda p: eta_bins(lambda e: ints("window", 2, lambda w: inv_cdf_node(T["kernel"][p][e, w])))))))
    cs.append(correction("lambda", "scale of eps so that the 68 % residual width matches the forced refit's",
                         ["allocation", "parameter", "window", "abs_eta", "pt"],
                         per(lambda T: par(lambda p: ints("window", 2, lambda w: eta_bins(lambda e: pt_bins(
                             lambda i: float(T["lambda"][p][w, e, i]))))))))
    nm = len(mask_classes)

    def by_mask(fn):                       # today's layer mask: its own class if among the frequent ones, else "other"
        return category("layer_mask", [(int(m), fn(i)) for i, m in enumerate(mask_classes)], default=fn(nm))
    cs.append(correction("cov_scale", "precision-part covariance-diagonal ratio = exp(a + b ln sigma_today + sigma eta), the "
                                      "exponent clipped to ln 0.2 ... ln 10000: the coefficient a, b or sigma",
                         ["allocation", "parameter", "coefficient", "window", "abs_eta", "pt", "layer_mask"],
                         per(lambda T: par(lambda p: category("coefficient", [
                             (c, ints("window", 2, lambda w, k=k: eta_bins(lambda e_: pt_bins(lambda i: by_mask(
                                 lambda j: float(T["cov_scale"][p][w, e_, i, j, k]))))))
                             for k, c in enumerate(("a", "b", "sigma"))])))))
    cs.append(correction("cov_eta", "standardized residual of the covariance line: inverse CDF at the covariance draw u (one draw per "
                                    "track for all five parameters)", ["allocation", "parameter", "u"],
                         per(lambda T: par(lambda p: inv_cdf_node(T["cov_eta"][p])))))
    cs.append(correction("rho", "Gaussian-copula correlation of a parameter's draw with its partner's (z0 with d0, tanL with z0, "
                                "rInv and phi0 with d0); d0 has none", ["allocation", "parameter", "abs_eta", "window"],
                         per(lambda T: category("parameter", [(p, eta_bins(lambda e: ints("window", 2, lambda w: float(T["rho"][p][e, w]))))
                                                              for p in PM.DRAW_ORDER[1:]]))))
    sc = lambda fn: category("score", [(s, fn(s)) for s in PM.SCORES])
    cs.append(correction("score_map", "precision-part map s' = a + b s + sigma eta of a head input in its natural variable",
                         ["allocation", "score", "coefficient", "window_arm", "n_hits", "abs_eta"],
                         per(lambda T: sc(lambda s: category("coefficient", [(c, ints("window_arm", 2, lambda w: binning(
                             "n_hits", [0.0, *PM.NACC_EDGES, INF], [eta_bins(lambda e: float(T["map"][s][w, k, e, i])) for k in range(4)])))
                                                                             for i, c in enumerate(("a", "b", "sigma"))])))))
    cs.append(correction("score_eta", "standardized residual of the same-cluster score map: inverse CDF at u, by |2 u_rank - 1|",
                         ["allocation", "score", "tail", "u"],
                         per(lambda T: sc(lambda s: binning("tail", [0.0, *PM.TAIL_EDGES, 1.0], [inv_cdf_node(T["eta"][s][k]) for k in range(3)])))))
    dq = []
    for grp, names in (("matched_ratio", PM.PARAMS), ("delta_ratio", PM.PARAMS), ("cov_ratio", PM.PARAMS), ("counts", PM.COUNTS),
                       ("values", PM.SCORES)):
        dq += [(f"{grp}_{n}" if grp not in ("counts", "values") else n, grp, i) for i, n in enumerate(names)]
    edges = [float(x) for x in np.linspace(0, 1, PM.N_DONORS + 1)]
    r6 = lambda x: float(f"{x:.6g}")
    cs.append(correction("donor", "changed state: the donor's value, picked by the rank draw u. matched_ratio_<p>: (arm - truth) / "
                                  "sigma_today; delta_ratio_<p>: (arm - today) / sigma_today; cov_ratio_<p>: arm / today covariance diagonal; "
                                  "the spix* and h_* names: the donor's own values of those head inputs",
                         ["allocation", "quantity", "matched", "window", "abs_eta", "pull", "u"],
                         per(lambda T: category("quantity", [(q, ints("matched", 2, lambda m: ints("window", 3, lambda w: eta_bins(
                             lambda e: binning("pull", [0.0, *PM.PULL_EDGES, INF], [binning("u", edges, [r6(v) for v in T["donors"][g][m, w, e, j, :, i]])
                                                                                    for j in range(3)])))))
                                                             for q, g, i in dq]))))
    return {"schema_version": 2, "description": json.dumps(meta), "corrections": cs}


def main():
    import ext_scan as XS
    import heads as HD
    import param_closure as PC
    import study as ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=("v1", "v2"), default="v2")
    ap.add_argument("--scen", nargs="+", default=list(ST.SCENARIOS))
    ap.add_argument("--allocs", nargs="+", default=None)
    ap.add_argument("--files", type=int, nargs="+", default=list(ST.SCAN_FILES))
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    import subprocess
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    XS.VERSION = a.version
    root = Path(a.root) if a.root else XS.out_root()
    allocs = a.allocs or [x for x in XS.SETS[a.version] if x != "implemented_now"]
    # the head models shipped beside the files: the retrained best per detector scenario, and the deployed ones
    for name, src in (("retrained", ST.LOCAL / "heads_retrained" / "models"), ("deployed", ST.CACHE / "heads")):
        for scen in a.scen:
            dst = out / "heads" / name / "SmartPixels" / scen
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src / "SmartPixels" / scen, dst)
    commit = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    payload = json.load(open(ST.CACHE / "checksums.json"))
    shutil.copy(Path(__file__).parent / "spixbits_apply.py", out / "spixbits_apply.py")
    summary = {}
    for scen in a.scen:
        T, today, _, truth, m = PC.load(root, scen, a.files)
        D = PM.draws(T["run"], T["lumi"], T["event"], T["track"])
        terc = np.quantile(today["margins"][today["refit"], 0], [1 / 3, 2 / 3])
        allsel = np.ones(len(T["run"]), bool)
        tables = {al: PM.fit(today, PC.load_arm(root, scen, a.files, al), truth, D, terc, allsel) for al in allocs}
        meta = {"what": "SmartPixels output bits -> track refit: per-track parametrization of quantization scenarios relative to "
                        "today's production (PLAN.md section 7b; the heads' inputs are emulated and the real head models evaluated)",
                "detector_scenario": scen, "quantization_scenarios": allocs, "extended_scan_version": a.version,
                "fit_sample": f"canon_v5 TTbar files {list(a.files)}, extended collection, every track ({len(T['run'])} tracks, "
                              f"{len(np.unique(T['event_key']))} events)", "scan_commit": commit,
                "payload_refit_calibration_sha256": payload["refit_calibration_v3.json"]["sha256"],
                "kpt_GeV_cm": m["kpt"], "rinv_sign_times_charge": m["rinv_sign"],
                "clean_margin_tercile_edges": [float(x) for x in terc], "knots": [float(k) for k in PM.KNOTS],
                "n_donors": PM.N_DONORS, "layer_mask_classes": [int(x) for x in today["mask_classes"]],
                "cov_scatter": PM.COV_SCATTER, "head_inputs_14_38": HD_NAMES}
        doc = build(tables, terc, meta, today["mask_classes"])
        path = out / f"spixbits_{scen}.json.gz"
        with gzip.open(path, "wt") as fh:
            json.dump(doc, fh)
        # consistency: the helper through correctionlib against the numpy evaluation, the same draws given to both
        import spixbits_apply as SA
        cset, M = SA.load(path)
        H = HD.Heads(ST.LOCAL / "heads_retrained" / "models", scen)
        Hs = SA.load_heads(out / "heads" / "retrained", scen)
        rng = np.random.default_rng(1)
        pick = np.sort(rng.choice(len(T["run"]), min(20000, len(T["run"])), replace=False))
        sub = lambda d: {k: (v[pick] if isinstance(v, np.ndarray) and len(v) == len(T["run"]) else v) for k, v in d.items()}
        worst = {}
        for al in (allocs[:3] + allocs[-1:]):
            ref = PM.apply(tables[al], sub(today), truth[pick], sub(D), terc, H)
            got = SA.apply(cset, M, al, Hs, a=today["a"][pick], cov_diag=today["cov_diag"][pick], refit=today["refit"][pick],
                           clean_margin=today["margins"][pick, 0], inputs=today["inputs"][pick], ot_inputs=today["ot_inputs"][pick],
                           ot_a=today["ot_a"][pick], ot_cov_diag=today["ot_cov_diag"][pick], truth=truth[pick], draws=sub(D))
            sg_ = np.sqrt(today["cov_diag"][pick])                 # parameters: difference in units of today's sigma
            worst[al] = {"parameters_in_today_sigma": float(np.nanmax(np.abs(PM.residual(got["a"].copy(), ref["a"])) / sg_)),
                         "cov_diag_relative": float(np.nanmax(np.abs(got["cov_diag"] / ref["cov_diag"] - 1))),
                         "head_outputs_share_beyond_0.01": float((np.abs(got["margins"] - ref["margins"]) > 0.01).any(1).mean()),
                         "changed_differs_share": float(np.mean(got["changed"] != ref["changed"])),
                         "inputs_share_beyond_1e-3_relative": float((np.abs(got["inputs"] - ref["inputs"])
                                                                     > 1e-3 * (np.abs(ref["inputs"]) + 1e-3)).any(1).mean())}
        u = cset["uniform"].evaluate(T["run"][pick], T["lumi"][pick], T["event"][pick], T["track"][pick].astype(np.int64), 2)
        summary[scen] = {"file": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                         "helper_vs_fit_code": worst, "hashprng_uniform_mean": float(np.mean(u)),
                         "tracks": int(len(T["run"])), "meta": meta}
        print(scen, json.dumps({k: v for k, v in summary[scen].items() if k != "meta"}, indent=1), flush=True)
    json.dump(summary, open(out / "export_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
