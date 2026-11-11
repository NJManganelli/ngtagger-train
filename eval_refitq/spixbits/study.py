"""SmartPixels output bits -> track refit study: the pinned environment (PLAN.md sections 3, 9).

Importing this module points the replay at canon_v5 and at the LOCAL copies of the v3 calibration set canon_v5 was
produced with (the v3cmssw spec payload, payload set v1; cache_inputs.py copies it and records the checksums), with
the producer's kpt constant and the float32 module geometry, as closure_v3.py runs the "implemented now" refit.
Nothing but the canon_v5 nanos is read from the external disk.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

LOCAL = Path(os.environ.get("SPIXBITS_OUT", str(Path.home() / "smartpixels/scratch_local/spixbits")))
CACHE = LOCAL / "cache"
PAYLOAD = CACHE / "payload"
CANON_V5 = "/Volumes/WDMac/smartpixels_scratch/canon_v5"
os.environ.setdefault("WPC_CANON", CANON_V5)
os.environ["WPC_KPT"] = "cmssw"
os.environ["WPC_ANGLE_TABLE"] = str(PAYLOAD / "angle_table_v1a1e_canon_v3.json")
os.environ["WPC_GEOMETRY"] = str(PAYLOAD / "spix_module_geometry_D121.json")

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "replay", HERE.parent / "combinatorics", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                    # noqa: E402

import arms as AR                                     # noqa: E402
import common as CM                                   # noqa: E402
import cuts as CU                                     # noqa: E402
from common import OP                                 # noqa: E402

SCENARIOS = ("AAAA", "AAAI", "IAAA")
DEV_FILE = 4
SCAN_FILES = (5, 7, 8, 9, 10)
ALL_FILES = tuple(range(1, 11))


def geometry():
    """The module geometry with origins and frames rounded to float32, as CMSSW holds them (closure_v3)."""
    geo = OP.load_geometry(CM.GEOMETRY)
    g = dict(geo)
    for k in ("origin", "ex", "ey", "ez"):
        g[k] = np.asarray(geo[k]).astype(np.float32).astype(np.float64)
    return g


def tools(cfg):
    """The replay's tools on the pinned local payload copies."""
    import seedmodel as SMOD
    with open(PAYLOAD / "scalefit_summary.json") as fh:
        scale = float(json.load(fh)["scale"])
    disc = AR.DiscTools(table=str(PAYLOAD / "angle_table_v1a1e_disclocal_canon_v3.json"),
                        material=str(PAYLOAD / "disc_material_v1.json"), disc_scale=scale)
    t = AR.Tools(cfg, calibration=str(PAYLOAD / "refit_calibration_v3.json"),
                    angle_table=str(PAYLOAD / "angle_table_v1a1e_canon_v3.json"),
                    cuts=CU.Cuts.load(str(PAYLOAD / "wpc_cuts_c1b_v1.json")), disc=disc,
                    seed_model=SMOD.SeedModel(str(PAYLOAD / "seed_model_v1.json")))
    with open(HERE / "sizeblind_R_v3.json") as fh:          # the size-blind position errors (fit_sizeblind.py)
        t.sizeblind = json.load(fh)
    return t


def input_path(scen, f):
    return CM.input_path(scen, f)
