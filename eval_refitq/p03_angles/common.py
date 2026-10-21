"""P0.3 shared helpers: load the per-file caches written by extract.py and derive the angle
quantities through the shared module (ngtagger.spdefs.angles), never by local copies."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from ngtagger.spdefs import angles as SA

# THE INPUT ERA (P03_ERA; default canon_v4, the campaign after canon_v3). canon_v3's stored results stay where they
# are (p03_angles: caches, tables named *_canon_v3.json, the ROC and summary); another era writes beside them in
# p03_angles_<era>, so a rerun never overwrites a stored result. The caches record their own input path.
ERA = os.environ.get("P03_ERA", "canon_v4")
SCRATCH = Path("/Volumes/WDMac/smartpixels_scratch")
IN_DIR = SCRATCH / ERA
OUT_DIR = Path(os.environ.get("P03_OUT", str(SCRATCH / ("p03_angles" if ERA == "canon_v3" else f"p03_angles_{ERA}"))))
CACHE_DIR = Path(os.environ.get("P03_CACHE", str(OUT_DIR / "cache")))
SCEN = "AAAA"
HERE = Path(__file__).resolve().parent
EVENTS_PER_FILE = 100
# THE ERA'S FILE ROLES (fit on spent files; protect the gate and the known-good). canon_v3: the sheet's (fit 1-2, test
# 3-10; demonstration fit 7-8, test 3-6, 9-10). canon_v4 (WDMac p03_disc_v4/PLAN.md, approved 2026-10-21): files 3 (the
# fast gate's input) and 4 (the census known-good) protected; demonstration fit 7-8, test 5, 9, 10; gated fit 1, 2, 5,
# 7-10, test 6.
if ERA == "canon_v3":
    FIT_FILES, TEST_FILES, DEMO_SPLIT = SA.FIT_FILES, SA.TEST_FILES, ((7, 8), (3, 4, 5, 6, 9, 10))
else:
    FIT_FILES, TEST_FILES, DEMO_SPLIT = (1, 2, 5, 7, 8, 9, 10), (6,), ((7, 8), (5, 9, 10))
# THE GLOBAL DISC ROWS' FACING AND EDGES. canon_v3: the clusters' orientation rule and the sheet's edges. canon_v4 (the
# CMSSW disc sense fix; the clusters' rule splits a module 66/34 there): the geometric facing (the module's position
# fit) and the 9.5 deg inner edge (SA.DISC_ABS_EDGES_V4).
GEOMETRIC_FACING = ERA != "canon_v3"
GLOBAL_DISC_EDGES = None if ERA == "canon_v3" else SA.DISC_ABS_EDGES_V4


def table_name(stem: str, suffix: str = "") -> str:
    """A table's file name in this era: <stem>_<era><suffix>.json."""
    return f"{stem}_{ERA}{suffix}.json"


def cache_path(f: int, scen: str = SCEN) -> Path:
    return CACHE_DIR / f"p03_{scen}_f{f:02d}.npz"


def cache_meta(f: int, scen: str = SCEN) -> dict:
    with np.load(cache_path(f, scen)) as Z:
        return json.loads(str(Z["meta"]))


def load(files, cols, mask_fn=None, scen: str = SCEN) -> dict:
    """Concatenated cluster columns of the cached files; `mask_fn(Z) -> bool array` selects
    clusters per file before concatenation (memory). Adds 'file'."""
    out = {c: [] for c in cols}
    out["file"] = []
    for f in files:
        with np.load(cache_path(f, scen)) as Z:
            m = mask_fn(Z) if mask_fn is not None else slice(None)
            for c in cols:
                out[c].append(Z[c][m])
            out["file"].append(np.full(len(Z["ev"][m]), f, np.int8))
    return {c: np.concatenate(v) for c, v in out.items()}


def load_events(files, scen: str = SCEN) -> dict:
    keys = ("ev_ev", "ev_genvtx_z", "ev_jet_n", "ev_counts")
    out = {k: [] for k in keys}
    sh = []
    jets = {c: [] for c in ("jet_pt", "jet_eta", "jet_phi", "jet_hadronFlavour")}
    for f in files:
        with np.load(cache_path(f, scen)) as Z:
            for k in keys:
                out[k].append(Z[k])
            sh.append(Z["shared_counts"])
            for c in jets:
                jets[c].append(Z[c])
    r = {k: np.concatenate(v) for k, v in out.items()}
    r["shared_counts"] = np.concatenate(sh)
    r.update({c: np.concatenate(v) for c, v in jets.items()})
    return r


ANGLE_COLS = ("surface", "detId", "ev", "w", "tpEmuClass", "tpIdx", "hasAlpha", "hasBeta", "globalR", "globalZ",
              "globalPhi", "globalClusterPhi", "globalClusterCotTheta", "sigGlobalClusterPhi",
              "sigGlobalClusterCotTheta", "tpGlobalClusterPhi", "tpGlobalClusterCotTheta", "localCotAlpha",
              "localCotBeta", "tp_pt")


def table_population(Z) -> np.ndarray:
    """angle.error_table population: class T, usable direction, TP pT >= 1 GeV (all kept, w = 1)."""
    d = {c: Z[c] for c in ("tpEmuClass", "hasAlpha", "hasBeta", "localCotAlpha", "localCotBeta",
                           "sigGlobalClusterPhi", "sigGlobalClusterCotTheta")}
    return SA.class_T(d["tpEmuClass"]) & SA.usable(d) & (Z["tp_pt"] >= SA.PT_MIN)


def orientation_map(files, scen: str = SCEN):
    """frame.module_orientation from the emitted columns of every cached cluster of `files`."""
    C = load(files, ("detId", "localCotBeta", "globalClusterCotTheta", "hasAlpha", "hasBeta", "localCotAlpha",
                     "sigGlobalClusterPhi", "sigGlobalClusterCotTheta"), scen=scen)
    return SA.orientation_from_clusters(C["detId"], C["localCotBeta"], C["globalClusterCotTheta"], SA.usable(C))


def derive(C: dict, orient) -> dict:
    """Add the shared module's quantities: orientation, facing, pulls (published sigma), keys."""
    C["orient"] = SA.orientation_lookup(C["detId"], *orient)
    C["facing"] = SA.facing(C["surface"], C["orient"])
    C["pull"] = {a: SA.residual(a, SA.emitted(a, C), SA.truth(a, C)) / SA.sigma_of(a, C) for a in SA.ANGLES}
    C["kv"] = {"emitted": SA.incidence_key(C["surface"], C["globalClusterCotTheta"]),
               "true": SA.incidence_key(C["surface"], C["tpGlobalClusterCotTheta"])}
    C["cotv"] = {"emitted": np.asarray(C["globalClusterCotTheta"], np.float64),
                 "true": np.asarray(C["tpGlobalClusterCotTheta"], np.float64)}
    return C


def git_state(repo: Path = HERE.parents[1]) -> dict:
    """Commit and tree state of the code that runs (the frozen copy when launched: its
    manifest is in the job record; here the copy has no .git, so the launcher's record holds)."""
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src/ngtagger/spdefs",
                                "eval_refitq/p03_angles"], capture_output=True, text=True).stdout.strip()
        return {"commit": commit, "p03_files_dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "note": "no git (frozen copy): see the launcher's job record",
                "job_id": os.environ.get("SP_JOB_ID"), "job_dir": os.environ.get("SP_JOB_DIR"),
                "frozen_root": os.environ.get("SP_FROZEN_ROOT")}


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def inputs_provenance(files, scen: str = SCEN) -> list:
    out = []
    for f in files:
        m = cache_meta(f, scen)
        out.append({"cache": str(cache_path(f, scen)), "input": m["input"], "input_bytes": m["input_bytes"],
                    "input_mtime": m["input_mtime"], "input_sha256_partial": m["input_sha256_partial"],
                    "events": m["events"]})
    return out


def jdump(obj, path: Path) -> None:
    tmp = str(path) + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        f.write("\n")
    os.replace(tmp, path)
