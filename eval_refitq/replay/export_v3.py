"""Export the refit-v3 specification for the CMSSW port (the physicist's GO, 2026-10-20; wpc/scan/PLAN.md §4).

  python export_v3.py --out /Volumes/WDMac/smartpixels_scratch/v3cmssw/spec [--nev 10]

Writes, under --out:
  config_v3.json          every switch of the v3 arm (INN-D with the two-layer hit choice) and the producer settings it
                          keeps, with the payload each switch reads
  payload/                the payload files, copied, and PROVENANCE.json (source path, sha256, size, registry ids)
  code/                   the reference implementation as committed (the replay modules and the helpers they import)
  reference/<scen>_f04.jsonl  the replay's per-track output on the first --nev events of canon_v4 file 4, per scenario:
                          one OT track per line (axis 0 = the event's OT tracks, axis 1 = the event's clusters)
The descriptions (README.md, ALGORITHMS.md) are written by hand beside them.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

import arms as AR
import common as CM
import cuts as CU
import disc_angles as DA
import disc_material as DM
import seedmodel as SMOD
from common import OP

REPO = Path(__file__).resolve().parents[2]
C1B = CM.OUT / "c1b"
SCALE_FILE = C1B / "scale" / "scalefit_summary.json"
PAYLOAD = {
    "seed_model_v1.json": (SMOD.MODEL_FILE, "wpc.c1b.seed-model-v1", "the smooth two-surface seed correction (A16)"),
    "disc_material_v1.json": (Path(DM.TABLE), "wpc.c1b.j3-material-scale-smooth",
                              "the path-ordered, extent-aware material planes of disc-going tracks (C1b)"),
    "scalefit_summary.json": (SCALE_FILE, "wpc.c1b.j3-material-scale-smooth", "the disc material scale (2.671)"),
    "refit_calibration_v3.json": (Path(AR.CALIBRATION), "wpc.c1a (entry)",
                                  "the barrel material planes and their scale, the position-error (R) tables"),
    "angle_table_v1a1e_canon_v3.json": (Path(AR.AI.TABLE), "p03.table.v1a1e", "the barrel angle table (P0.3)"),
    "angle_table_v1a1e_disclocal_canon_v3.json": (Path(DA.DISC_TABLE), "p03.table.v1a1e-disclocal",
                                                  "the disc angle table in the module frame (P0.3)"),
    "wpc_cuts_c1b_v1.json": (C1B / "cuts" / "wpc_cuts_c1b_v1.json", "wpc.c1b.j4-cut-fit",
                             "the 99 % gate cuts: the two-layer choice's skip price P"),
    "d1_exempt_v2.json": (C1B / "d1_exempt_v2.json", "wpc.c1b.verdict",
                          "the beta edge-bin exemption: a judging rule (known limitation), not an algorithm input"),
    "spix_module_geometry_D121.json": (Path(CM.GEOMETRY), "", "the module geometry the replay projects onto"),
}
CODE = ["eval_refitq/replay/arms.py", "eval_refitq/replay/engine.py", "eval_refitq/replay/angle_import.py",
        "eval_refitq/replay/disc_angles.py", "eval_refitq/replay/disc_material.py", "eval_refitq/replay/seedmodel.py",
        "eval_refitq/replay/cuts.py", "eval_refitq/replay/common.py", "eval_refitq/replay/diagnostics.py",
        "eval_refitq/combinatorics/ot_projection.py", "eval_refitq/combinatorics/ot_refit_projection.py",
        "eval_refitq/combinatorics/refit_calibration_fit.py", "src/ngtagger/spdefs/angles.py",
        "src/ngtagger/data/spix_nano.py"]
PARAMS = ("rInv", "phi0", "tanL", "z0", "d0")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def export_payload(out):
    d = out / "payload"
    d.mkdir(parents=True, exist_ok=True)
    prov = {}
    for name, (src, reg, what) in PAYLOAD.items():
        shutil.copyfile(src, d / name)
        prov[name] = {"source": str(src), "sha256": _sha(src), "bytes": os.path.getsize(src), "registry": reg,
                      "what": what}
    with open(d / "PROVENANCE.json", "w") as fh:
        json.dump(prov, fh, indent=1)


def export_code(out):
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    d = out / "code"
    for rel in CODE:
        dst = d / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(["git", "-C", str(REPO), "show", f"{head}:{rel}"], capture_output=True, check=True).stdout
        dst.write_bytes(blob)
    (d / "COMMIT").write_text(head + "\n")
    return head


def export_config(out, cfg, scale, head):
    arm = AR.v3(cfg)
    a = dataclasses.asdict(arm)
    a["use"] = sorted([list(x) for x in arm.use])
    cuts = CU.Cuts.load(str(PAYLOAD["wpc_cuts_c1b_v1.json"][0]))
    key = CU.table_key("gate", arm.dims_key("rank"), "calibrated")
    doc = {"name": "SmartPixels seedless refit v3 (the replay's INN-D with the two-layer hit choice)",
           "decided": "2026-10-20, the physicist's GO (wpc/scan/PLAN.md §4; registry wpc.scan.go-table)",
           "reference_commit": head,
           "arm": a,
           "disc_material_scale": scale,
           "skip_price_table": {"file": "payload/wpc_cuts_c1b_v1.json", "key": key,
                                "bins": cuts.doc["cuts"][key]},
           "known_bads": {k: {"switches": kw, "disc_scale": sc} for k, (kw, sc) in AR.V3_KNOWN_BADS.items()},
           "kpt": "pT = kpt / |rInv|; the reference uses the chunk median of pT |rInv| (0.0114256970); WPC_KPT=cmssw "
                  "(the producer's float32 MagConstant x b_field) or a number overrides it",
           "producer_settings_kept": {k: cfg[k] for k in ("maxKFUpdates", "chi2UpdateGate", "predAngleMaxAbs",
                                                         "measAngleMaxAbs", "jacobianMaxAbs", "layerOrder", "minHits",
                                                         "otNPar")},
           "payload_per_switch": {
               "seed_cov=smooth": "payload/seed_model_v1.json",
               "scattering=highland, inner_material": "payload/refit_calibration_v3.json (planes, material_scale) for "
                                                      "barrel-only tracks; payload/disc_material_v1.json x "
                                                      "disc_material_scale for disc-going tracks",
               "pos_err=rtables": "payload/refit_calibration_v3.json (R_tables)",
               "ang_err=table": "payload/angle_table_v1a1e_canon_v3.json (barrel), "
                                "payload/angle_table_v1a1e_disclocal_canon_v3.json (discs)",
               "K=tree2": "payload/wpc_cuts_c1b_v1.json (skip price P)"}}
    with open(out / "config_v3.json", "w") as fh:
        json.dump(doc, fh, indent=1, default=float)


def export_reference(out, nev):
    geo = OP.load_geometry(CM.GEOMETRY)
    with open(SCALE_FILE) as fh:
        scale = float(json.load(fh)["scale"])
    cuts = CU.Cuts.load(str(PAYLOAD["wpc_cuts_c1b_v1.json"][0]))
    d = out / "reference"
    d.mkdir(parents=True, exist_ok=True)
    for scen in CM.SCENARIOS:
        path = CM.input_path(scen, CM.PRED_FILE)
        cfg = CM.producer_config(path)
        tools = AR.Tools(cfg, cuts=cuts, disc=AR.DiscTools(disc_scale=scale))
        arm = AR.v3(cfg)
        br = list(dict.fromkeys(AR.branches(path) + ["run", "luminosityBlock", "event"]))
        with open(d / f"{scen}_f{CM.PRED_FILE:02d}.jsonl", "w") as fh:
            for start, A in CM.iterate(path, br, chunk=nev, nev=nev):
                ctx = AR.prepare(A, geo)
                out_ = AR.run_arm(ctx, geo, cfg, scen, arm, tools)
                o = np.arange(ctx["ntrk"])
                seed_a, seed_C = AR.seed(ctx, cfg, arm, tools, o)
                tev = ctx["tev"]
                first = np.r_[0, np.cumsum(np.bincount(tev, minlength=len(A)))[:-1]]
                dg = AR.disc_going(AR.visit_plan(ctx, geo, cfg, scen, arm)[0])
                rec = out_["rec"]
                for t in o:
                    e = int(tev[t])
                    steps = []
                    for j in range(out_["seq"].shape[1]):
                        S = int(out_["seq"][t, j])
                        if S < 0:
                            continue
                        ci = int(rec["ci"][t, j]) if rec["acc"][t, j] else -1
                        steps.append({"step": j, "surface": S, "visited": bool(rec["visited"][t, j]),
                                      "accepted": bool(rec["acc"][t, j]),
                                      "detId": int(geo["detId"][rec["mi"][t, j]]) if rec["acc"][t, j] else None,
                                      "cluster": ci - int(ctx["coff"][e]) if ci >= 0 else -1,
                                      "ncand": int(rec["ncand"][t, j]), "chi2": [float(x) for x in rec["chi2"][t, j]],
                                      "kchoice": int(rec["kchoice"][t, j])})
                    row = {"run": int(A["run"][e]), "lumi": int(A["luminosityBlock"][e]), "event": int(A["event"][e]),
                           "track": int(t - first[e]), "refit": bool(out_["refit"][t]), "nacc": int(out_["nacc"][t]),
                           "disc_going": bool(dg[t]),
                           "ot": dict(zip(PARAMS, map(float, ctx["a"][t]))),
                           "seed_C": seed_C[t].tolist(), "steps": steps,
                           "final": dict(zip(PARAMS, map(float, out_["a"][t]))), "final_C": out_["C"][t].tolist()}
                    fh.write(json.dumps(row) + "\n")
        print(f"{scen}: {ctx['ntrk']} tracks", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--nev", type=int, default=10)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(SCALE_FILE) as fh:
        scale = float(json.load(fh)["scale"])
    export_payload(out)
    head = export_code(out)
    export_config(out, CM.producer_config(CM.input_path("AAAA", CM.PRED_FILE)), scale, head)
    export_reference(out, a.nev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
