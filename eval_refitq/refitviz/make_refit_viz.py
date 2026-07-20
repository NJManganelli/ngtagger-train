#!/usr/bin/env python
"""Regenerate the standalone SmartPixels digiRefit refit-replay HTML.

Usage (from the repo root, in the pixi env):
    pixi run python eval_refitq/refitviz/make_refit_viz.py \
        [--nano PATH] [--out PATH] [--events N] [--tracks e:i e:i ...]

The output HTML is fully self-contained (Plotly inlined) and opens offline in a
browser; it is also embeddable in a Jupyter/RISE notebook via an <iframe> or by
calling ngtagger.viz.build_refit_viz(..., return_fig=True) and displaying the fig.
"""

from __future__ import annotations

import argparse
import os

from ngtagger.viz.refit_replay import build_refit_viz

_DEFAULT_NANO = "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/nano_pE_TrkSmartPix_withGen_file1.root"
_DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refit_replay.html")


def _parse_track(s):
    e, i = s.split(":")
    return (int(e), int(i))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nano", default=_DEFAULT_NANO, help="SmartPixels nano file (read-only)")
    ap.add_argument("--out", default=_DEFAULT_OUT, help="output standalone HTML path")
    ap.add_argument("--events", type=int, default=6, help="max events to scan for curation")
    ap.add_argument("--tracks", nargs="*", default=None, metavar="EVENT:IDX",
                    help="force-select tracks (e.g. 0:19 0:8); else auto-curate "
                         "clean/wrong/displaced/fake")
    ap.add_argument("--n-clean", type=int, default=2)
    ap.add_argument("--n-wrong", type=int, default=1)
    ap.add_argument("--n-displaced", type=int, default=1)
    ap.add_argument("--n-fake", type=int, default=1)
    args = ap.parse_args()

    tracks = [_parse_track(s) for s in args.tracks] if args.tracks else None
    res = build_refit_viz(args.nano, tracks=tracks, out_html=args.out,
                          n_each=(args.n_clean, args.n_wrong, args.n_displaced, args.n_fake),
                          max_events=args.events)
    print("curated tracks (event, idx, archetype):")
    for p in res["picks"]:
        print(f"   {p}")
    print(f"wrote {res['html_path']} ({os.path.getsize(res['html_path']) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
