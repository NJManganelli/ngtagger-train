"""Fit the shipped parametrization on ALL scan events and write it as correctionlib JSON, one file per scenario
(PLAN.md section 7b): doc-archive/spix-bits-refit-study/parametrization/v2/spixbits_<scenario>.json.gz (the first
version, fitted on extended scan version 1 with tabulated head outputs, stays in the folder above), with the
reference helper spixbits_apply.py and a consistency check (helper through correctionlib == the fit code's numpy
evaluation, the same draws given to both).

  python param_export.py [--scen AAAA AAAI IAAA] [--root DIR] [--out DIR] [--allocs ...]

Corrections in a file (inputs in SmartPixels-study units: cm, GeV, 1/cm; `allocation` is one of the shipped names,
anything else raises):
  uniform, normal   the per-track draws: hashprng of (run, lumi, event, track, draw)
  p_changed         share of tracks with a material hit-choice part against today
  eps, lambda, rho  precision part: inverse CDF of the parameter change in units of the sigma the scenario adds, its
                    scale, the copula correlations
  donor             hit-choice part: the donor's values, picked by the rank draw
Beside each file: spixbits_<scenario>_maps.npz, the tree maps of the covariance diagonal and of the heads' inputs (per
quantization scenario: 67 boosted tree models as padded arrays, the residual quantiles, ranges and copula factors),
and the head models (heads/retrained: the best of 20 per detector scenario; heads/deployed) that the helper evaluates
on the emulated inputs. The uniform / normal draws: 1 = state, 2 = rank (uniform); 11-14 = z0, tanL, 1/R, phi0,
21-25 = covariance columns, 41-53 = mapped inputs (normal); 61-73 = "input missing" (uniform).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
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
     "abs_eta": ("real", "|eta| of today's refit track (asinh of tanL)"), "pt": ("real", "today's refit pT [GeV]"),
     "mult": ("real", "today's spixMaxWindowMult"),
     "pull": ("real", "max(|d0 - truth d0| / sigma_d0, |z0 - truth z0| / sigma_z0) of today's track; 0 for a track without a particle"), "clean_margin": ("real", "today's clean head as a margin: logit of head_clean"),
     "u": ("real", "a uniform draw in [0, 1)"), "quantity": ("string", "donor quantity"),
     "run": ("int", "run"), "lumi": ("int", "luminosity block"), "event": ("int", "event"),
     "track": ("int", "index of the track along axis 1 of the event's L1TSmartPixelsExtTrack collection"),
     "draw": ("int", "which draw (see the file description)")}


def var(n):
    return {"name": n, "type": V[n][0], "description": V[n][1]}


def binning(name, edges, content, flow="clamp"):
    return {"nodetype": "binning", "input": name, "edges": [float(e) if e != INF else INF for e in edges], "content": content, "flow": flow}


def category(name, items):
    return {"nodetype": "category", "input": name, "content": [{"key": k, "value": v} for k, v in items]}


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


def build(tables, terc, meta):
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
                                "sqrt(|C_arm - C_today| + (0.05 sigma_today)^2) with C_arm from the covariance map: inverse CDF at u",
                         ["allocation", "parameter", "abs_eta", "window", "u"],
                         per(lambda T: par(lambda p: eta_bins(lambda e: ints("window", 2, lambda w: inv_cdf_node(T["kernel"][p][e, w])))))))
    cs.append(correction("lambda", "scale of eps so that the 68 % residual width matches the forced refit's",
                         ["allocation", "parameter", "window", "abs_eta", "pt"],
                         per(lambda T: par(lambda p: ints("window", 2, lambda w: eta_bins(lambda e: pt_bins(
                             lambda i: float(T["lambda"][p][w, e, i]))))))))
    cs.append(correction("rho", "Gaussian-copula correlation of a parameter's draw with its partner's (z0 with d0, tanL with z0, "
                                "rInv and phi0 with d0); d0 has none", ["allocation", "parameter", "abs_eta", "window"],
                         per(lambda T: category("parameter", [(p, eta_bins(lambda e: ints("window", 2, lambda w: float(T["rho"][p][e, w]))))
                                                              for p in PM.DRAW_ORDER[1:]]))))
    dq = []
    for grp, names in (("matched_ratio", PM.PARAMS), ("delta_ratio", PM.PARAMS), ("counts", PM.COUNTS)):
        dq += [(f"{grp}_{n}" if grp != "counts" else n, grp, i) for i, n in enumerate(names)]
    edges = [float(x) for x in np.linspace(0, 1, PM.N_DONORS + 1)]
    r6 = lambda x: float(f"{x:.6g}")
    cs.append(correction("donor", "changed state: the donor's value, picked by the rank draw u. matched_ratio_<p>: (arm - truth) / "
                                  "sigma_today; delta_ratio_<p>: (arm - today) / sigma_today; the spix* names: the donor's own refit flag, "
                                  "crossings, accepted hits, layer mask and largest window multiplicity",
                         ["allocation", "quantity", "matched", "window", "abs_eta", "pull", "u"],
                         per(lambda T: category("quantity", [(q, ints("matched", 2, lambda m: ints("window", 3, lambda w: eta_bins(
                             lambda e: binning("pull", [0.0, *PM.PULL_EDGES, INF], [binning("u", edges, [r6(v) for v in T["donors"][g][m, w, e, j, :, i]])
                                                                                    for j in range(3)])))))
                                                             for q, g, i in dq]))))
    return {"schema_version": 2, "description": json.dumps(meta), "corrections": cs}


def maps_arrays(T):
    """The tree maps of one quantization scenario as arrays: the models in the order covariance columns (mean, mean, ln
    scale each), mapped inputs (the same), "input missing" classifiers; their trees stacked, `start` = first tree of
    each model."""
    models = [m for blk in ("cov_map", "score_map") for c in T[blk]["cols"] for m in (*c["mean"], c["scale"])] + T["missing"]
    arr = [m.arrays() for m in models]
    n = max(a[1].shape[1] for a in arr)
    pad = lambda x, fill: np.concatenate([x, np.full((len(x), n - x.shape[1]), fill, x.dtype)], 1)
    out = {"base": np.array([a[0] for a in arr]), "start": np.cumsum([0] + [len(a[1]) for a in arr]),
           "feature": np.concatenate([pad(a[1], -1) for a in arr]), "threshold": np.concatenate([pad(a[2], 0) for a in arr]),
           "left": np.concatenate([pad(a[3], 0) for a in arr]), "right": np.concatenate([pad(a[4], 0) for a in arr]),
           "value": np.concatenate([pad(a[5], 0) for a in arr])}
    for name, blk in (("cov", "cov_map"), ("score", "score_map")):
        out[name + "_q"] = np.array([c["q"] for c in T[blk]["cols"]])
        out[name + "_lo"] = np.array([c["lo"] for c in T[blk]["cols"]])
        out[name + "_hi"] = np.array([c["hi"] for c in T[blk]["cols"]])
        out[name + "_chol"] = T[blk]["chol"]
    return out


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
    ap.add_argument("--out", default="", help="default: the parametrization folder for scan version 1, its v2/ for version 2")
    a = ap.parse_args()
    import subprocess
    out = Path(a.out) if a.out else (OUT if a.version == "v1" else OUT / a.version)
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
    if not commit:                                  # a lane job's frozen copy is not a repository: its manifest has the commit
        for d in Path(__file__).resolve().parents:
            if (d / "manifest.json").exists():
                commit = json.load(open(d / "manifest.json")).get("commit", "")
                break
    payload = json.load(open(ST.CACHE / "checksums.json"))
    tmp = out / f"spixbits_apply.py.{os.getpid()}.tmp"       # several exports (one per detector scenario) may run side by side
    shutil.copy(Path(__file__).parent / "spixbits_apply.py", tmp)
    os.replace(tmp, out / "spixbits_apply.py")
    summary = {}
    for scen in a.scen:
        T, today, _, truth, m = PC.load(root, scen, a.files)
        D = PM.draws(T["run"], T["lumi"], T["event"], T["track"])
        terc = np.quantile(today["margins"][today["refit"], 0], [1 / 3, 2 / 3])
        allsel = np.ones(len(T["run"]), bool)
        checked = allocs[:3] + allocs[-1:]              # the helper is compared with the fit code on these
        tables, arrays = {}, {}
        for al in allocs:
            tables[al] = PM.fit(today, PC.load_arm(root, scen, a.files, al), truth, D, terc, allsel)
            arrays.update({f"{al}/{k}": v for k, v in maps_arrays(tables[al]).items()})
            if al not in checked:                        # the fitted tree models are large: keep the arrays only
                for k in ("cov_map", "score_map", "missing"):
                    del tables[al][k]
            print(scen, "fitted", al, flush=True)
        meta = {"what": "SmartPixels output bits -> track refit: per-track parametrization of quantization scenarios relative to "
                        "today's production (PLAN.md section 7b; the heads' inputs are emulated and the real head models evaluated)",
                "detector_scenario": scen, "quantization_scenarios": allocs, "extended_scan_version": a.version,
                "fit_sample": f"canon_v5 TTbar files {list(a.files)}, extended collection, every track ({len(T['run'])} tracks, "
                              f"{len(np.unique(T['event_key']))} events)", "scan_commit": commit,
                "payload_refit_calibration_sha256": payload["refit_calibration_v3.json"]["sha256"],
                "kpt_GeV_cm": m["kpt"], "rinv_sign_times_charge": m["rinv_sign"],
                "clean_margin_tercile_edges": [float(x) for x in terc], "knots": [float(k) for k in PM.KNOTS],
                "n_donors": PM.N_DONORS, "head_inputs_14_38": HD_NAMES, "mapped_inputs": list(PM.SCORES),
                "maps_file": f"spixbits_{scen}_maps.npz"}
        doc = build(tables, terc, meta)
        path = out / f"spixbits_{scen}.json.gz"
        with gzip.open(path, "wt") as fh:
            json.dump(doc, fh)
        mpath = out / f"spixbits_{scen}_maps.npz"
        np.savez_compressed(mpath, **arrays)
        del arrays
        # consistency: the helper through correctionlib against the numpy evaluation, the same draws given to both
        import spixbits_apply as SA
        cset, M = SA.load(path)
        maps = SA.load_maps(mpath)
        H = HD.Heads(ST.LOCAL / "heads_retrained" / "models", scen)
        Hs = SA.load_heads(out / "heads" / "retrained", scen)
        rng = np.random.default_rng(1)
        pick = np.sort(rng.choice(len(T["run"]), min(20000, len(T["run"])), replace=False))
        sub = lambda d: {k: (v[pick] if isinstance(v, np.ndarray) and len(v) == len(T["run"]) else v) for k, v in d.items()}
        worst = {}
        for al in checked:
            ref = PM.apply(tables[al], sub(today), truth[pick], sub(D), terc, H)
            got = SA.apply(cset, M, maps, al, Hs, a=today["a"][pick], cov_diag=today["cov_diag"][pick], refit=today["refit"][pick],
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
                         "maps_file": str(mpath), "maps_bytes": mpath.stat().st_size,
                         "maps_sha256": hashlib.sha256(open(mpath, "rb").read()).hexdigest(),
                         "helper_vs_fit_code": worst, "hashprng_uniform_mean": float(np.mean(u)),
                         "tracks": int(len(T["run"])), "meta": meta}
        print(scen, json.dumps({k: v for k, v in summary[scen].items() if k != "meta"}, indent=1), flush=True)
        json.dump(summary[scen], open(out / f"export_summary_{scen}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
