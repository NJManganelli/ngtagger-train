"""Real-data smoke for the fastHisto (dx, dy) estimator on an existing nano
file carrying the extended-track table (L1TExtTrack, 5-par d0).

Thin wrapper over ngtagger.train.vtxstudy.run_vertex_dxy_smoke (the reusable
compute lives in the package). Writes realdata_smoke.json + realdata_smoke.png
to this directory.

This is a sanity first-look, NOT a measurement: with RelVal-scale statistics
the per-event PV-window (dx, dy) should cluster near the beam-spot origin with
a spread set by the d0 resolution / sqrt(N_window).

Usage:
  pixi run python eval_refitq/vtxdxy/realdata_smoke.py [nano_file ...]
  ngtagger vtx-study --realdata nano_file [nano_file ...]
Default file: the PU100 TrkSmartPix withGen smoke nano (read-only).
"""
from __future__ import annotations

import os
import sys

from ngtagger.train.vtxstudy import run_vertex_dxy_smoke

OUT = os.path.dirname(os.path.abspath(__file__))
DEFAULT = ["/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/"
           "nano_pu100_TrkSmartPix_withGen.root"]
TRACK = "L1TExtTrack"


def main():
    files = sys.argv[1:] or DEFAULT
    run_vertex_dxy_smoke(files, OUT, track_table=TRACK, d0_gate=0.15)


if __name__ == "__main__":
    main()
