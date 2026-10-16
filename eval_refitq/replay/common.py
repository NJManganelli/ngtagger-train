"""WP-C entry (replay readiness) shared constants and input helpers.

Plan: /Volumes/WDMac/smartpixels_scratch/wpc/PLAN.md (round 1 + the pre-declared changes of §10).
File roles are fixed there (PD21): the TTbar fit files 1, 2, 3, 5, 7-10, file 4 predictions / known-goods, file 6
C1b's gate. The inputs are canon_v4 (c1b/PLAN.md §8 A14: canon_v3's events, bit-identical for the replay); the
stored results up to 2026-10-16 were made on canon_v3.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
COMB = HERE.parent / "combinatorics"
for p in (str(COMB), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import omnibus_support as OSU          # noqa: E402
import ot_projection as OP             # noqa: E402
import ot_refit_projection as ORP      # noqa: E402

CANON = Path(os.environ.get("WPC_CANON", "/Volumes/WDMac/smartpixels_scratch/canon_v4"))
OUT = Path(os.environ.get("WPC_OUT", "/Volumes/WDMac/smartpixels_scratch/wpc"))
CUTS = OUT / "cuts" / "wpc_cuts_v6.json"     # the consumers' 99 % cuts (rr per scenario, PD15; smoothed, PD17;
                                              # 4-sigma window and PD19's rr population, PD20; the fit files, PD21)
GEOMETRY = os.environ.get("WPC_GEOMETRY",
                          "/Users/nmangane/smartpixels/cmssw/work/otstub_arm/spix_module_geometry_D121.json")
SCENARIOS = ("AAAA", "AAAI", "IAAA")
FIT_FILES = (1, 2, 3, 5, 7, 8, 9, 10)        # PD21: every file but the known-good (PRED_FILE) and C1b's gate (6)
PRED_FILE, GATED_FILE = 4, 3                  # GATED_FILE: the entry gate's file (spent)
RR_CHECK_FILE = 5                             # PD15: the per-scenario rr cuts' gated file
REFIT_HIT = "L1TSmartPixelsRefitHit"         # the (unsuffixed) refit hit table
SP_TRACK = "L1TSmartPixelsTrack"
CHUNK = 10                                    # events per chunk (ORP's study-20 default)


def input_path(scen: str, f: int) -> Path:
    return CANON / f"spstudy_{scen}_f{f:02d}_100ev.root"


def event_rank(path) -> np.ndarray:
    """Per entry of an input file, the rank of its (run, lumi, event) within the file. The scenarios' files hold the
    same events in different orders, so the rank (not the entry) identifies an event across scenarios: the joint
    event bootstrap must resample by it (c1b/PLAN.md §8 A14)."""
    t = uproot.open(str(path))["Events"]
    r, l, e = (t[k].array(library="np") for k in ("run", "luminosityBlock", "event"))
    return np.argsort(np.lexsort((e, l, r)), kind="stable")


def producer_config(path) -> dict:
    """The refit producer's configuration (ORP.PRODUCER_DEFAULTS keys plus the guards the
    mirror needs) read from the file's ParameterSets; a missing producer is an error."""
    lab, params = OSU.producer_params(OSU.pset_blob(str(path)), SP_TRACK)
    if not params:
        raise SystemExit(f"{path}: no L1SmartPixelsTrackProducer behind {SP_TRACK}")
    cfg = dict(ORP.PRODUCER_DEFAULTS, **OSU.refit_producer_config(params))
    cfg.update({"label": lab,
                "measAngleMaxAbs": float(params["seedlessRefitMeasAngleMaxAbs"]),
                "predAngleMaxAbs": float(params["seedlessRefitPredAngleMaxAbs"]),
                "chi2UpdateGate": float(params["seedlessRefitChi2UpdateGate"]),
                "jacobianMaxAbs": float(params["seedlessRefitJacobianMaxAbs"]),
                "minHits": int(params["seedlessRefitMinHits"]),
                "otNPar": int(params["seedlessRefitOTNPar"]),
                "activeLayers": params.get("smartPixelsActiveLayers"),
                "discScope": params.get("seedlessRefitDiscScope")})
    return cfg


def mask_of(cfg: dict) -> str:
    """The activeSP mask string (A / I per IL1-IL4) of a producer configuration."""
    return "".join("A" if c == "1" else "I" for c in cfg["activeLayers"])


def iterate(path, branches, chunk=CHUNK, nev=None):
    """uproot.iterate over one file, at most `nev` events; yields (start entry, arrays)."""
    done = 0
    for A in uproot.iterate([f"{path}:Events"], branches, step_size=chunk):
        if nev is not None and done >= nev:
            return
        if nev is not None and done + len(A) > nev:
            A = A[:nev - done]
        yield done, A
        done += len(A)


def concat(parts):
    """Concatenate a list of {key: array} dicts key by key (empty list -> {})."""
    if not parts:
        return {}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
