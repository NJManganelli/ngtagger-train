"""WP-C entry (replay readiness) shared constants and input helpers.

Plan: /Volumes/WDMac/smartpixels_scratch/wpc/PLAN.md (round 1 + the pre-declared changes of §10).
File roles are fixed there: canon_v3 TTbar files 1-2 fit only, file 4 predictions / known-goods,
file 3 gated (judged once), file 5 gated for the per-scenario rr cuts only (PD15), files 6-10 untouched.
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

CANON_V3 = Path(os.environ.get("WPC_CANON", "/Volumes/WDMac/smartpixels_scratch/canon_v3"))
OUT = Path(os.environ.get("WPC_OUT", "/Volumes/WDMac/smartpixels_scratch/wpc"))
CUTS = OUT / "cuts" / "wpc_cuts_v2.json"     # the consumers' 99 % cuts (rr per scenario, PLAN PD15)
GEOMETRY = os.environ.get("WPC_GEOMETRY",
                          "/Users/nmangane/smartpixels/cmssw/work/otstub_arm/spix_module_geometry_D121.json")
SCENARIOS = ("AAAA", "AAAI", "IAAA")
FIT_FILES, PRED_FILE, GATED_FILE = (1, 2), 4, 3
RR_CHECK_FILE = 5                             # PD15: the per-scenario rr cuts' gated file
REFIT_HIT = "L1TSmartPixelsRefitHit"         # canon_v3's (unsuffixed) refit hit table
SP_TRACK = "L1TSmartPixelsTrack"
CHUNK = 10                                    # events per chunk (ORP's study-20 default)


def input_path(scen: str, f: int) -> Path:
    return CANON_V3 / f"spstudy_{scen}_f{f:02d}_100ev.root"


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
