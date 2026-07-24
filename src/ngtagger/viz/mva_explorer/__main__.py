"""CLI: python -m ngtagger.viz.mva_explorer <cmd> [...]

  export-regressions   Panel 1: evaluate + quantize the correctionlib presets
  export-tkquality     Panel 2: score the unified coherent nanos with the Stage-3 BDTs
  export-tagger        Panel 3: convert Stage-4 prediction dumps to site tables
  export-file          Panel 1 generic path for an arbitrary correctionlib JSON
  make-site            copy the static app (html/js/plotly) + write manifest.json
  export-all           regressions + tkquality + tagger + make-site

All commands default to writing into eval_mva_explorer/site/ (created on
demand, never committed).  Serve it with `python -m http.server` and open
explorer.html — see README.md next to this file.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from argparse import ArgumentParser

DEFAULT_OUT = "eval_mva_explorer/site"
HERE = os.path.dirname(os.path.abspath(__file__))


def _parse_linspace(items):
    out = {}
    for it in items or []:
        name, spec = it.split("=", 1)
        lo, hi, n = spec.split(":")
        out[name] = (float(lo), float(hi), int(n))
    return out


def find_plotly() -> str | None:
    """Local plotly.min.js: prefer the python package's bundled copy, fall back
    to the eval_spixel prototype's."""
    try:
        import plotly

        cand = os.path.join(os.path.dirname(plotly.__file__),
                            "package_data", "plotly.min.js")
        if os.path.exists(cand):
            return cand
    except ImportError:
        pass
    cand = os.path.join(HERE, "..", "..", "..", "..",
                        "eval_spixel", "site", "plotly.min.js")
    return os.path.abspath(cand) if os.path.exists(cand) else None


def make_site(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for fname in ("explorer.html", "explorer_core.js"):
        shutil.copy(os.path.join(HERE, "site_src", fname), out_dir)
    pl = find_plotly()
    if pl:
        shutil.copy(pl, os.path.join(out_dir, "plotly.min.js"))
    else:
        print("WARNING: plotly.min.js not found (install plotly or point at "
              "eval_spixel/site); the page will not render without it")
    write_manifest(out_dir)
    print(f"site assembled in {out_dir}; serve with:\n"
          f"  (cd {out_dir} && python3 -m http.server 8742)\n"
          f"then open http://127.0.0.1:8742/explorer.html")


def write_manifest(out_dir: str) -> None:
    """Scan out_dir for *_meta.json and write manifest.json for the page."""
    regs, tkq, tagger = [], None, None
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith("_meta.json"):
            continue
        with open(os.path.join(out_dir, f)) as fh:
            meta = json.load(fh)
        entry = {"id": meta["id"], "title": meta.get("title", meta["id"]),
                 "meta": f}
        if meta["id"] == "tkq":
            tkq = entry
        elif meta["id"] == "tagger":
            tagger = entry
        else:
            regs.append(entry)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({"regressions": regs, "tkquality": tkq, "tagger": tagger}, fh)


def main(argv=None) -> int:
    parser = ArgumentParser(prog="python -m ngtagger.viz.mva_explorer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_out(p):
        p.add_argument("--out", default=DEFAULT_OUT, help="site output dir")

    add_out(sub.add_parser("export-regressions", help="Panel 1 presets"))

    p_f = sub.add_parser("export-file", help="Panel 1: arbitrary correctionlib JSON")
    add_out(p_f)
    p_f.add_argument("json_path")
    p_f.add_argument("--id", default=None, help="dataset id (default: file stem)")
    p_f.add_argument("--title", default=None)
    p_f.add_argument("--linspace", nargs="*", default=None, metavar="INPUT=LO:HI:N",
                     help="grid override for formula-only real inputs")

    p_t = sub.add_parser("export-tkquality", help="Panel 2")
    add_out(p_t)
    p_t.add_argument("--nano-dir", default=None)
    p_t.add_argument("--models-dir", default=None)
    p_t.add_argument("--n-files", type=int, default=10)
    p_t.add_argument("--max-events", type=int, default=None)

    p_j = sub.add_parser("export-tagger", help="Panel 3")
    add_out(p_j)
    p_j.add_argument("--dumps-dir", default=None)

    add_out(sub.add_parser("make-site", help="copy static app + manifest"))
    add_out(sub.add_parser("export-all", help="everything + site"))

    args = parser.parse_args(argv)
    out = args.out

    if args.cmd in ("export-regressions", "export-all"):
        from ngtagger.viz.mva_explorer.presets import export_all_regressions

        for meta in export_all_regressions(out):
            n = (len(meta.get("corrections", []))
                 if meta["type"] == "generic" else
                 len(meta["configs"]) * len(meta["params"]) * len(meta["kinds"]))
            print(f"[reg] {meta['id']}: {n} corrections -> {meta['file']}")

    if args.cmd == "export-file":
        from ngtagger.viz.mva_explorer.correctionlib_ingest import export_generic_dataset

        did = args.id or os.path.basename(args.json_path).split(".")[0]
        meta = export_generic_dataset(
            args.json_path, f"reg_{did}", args.title or did, out,
            linspace_overrides=_parse_linspace(args.linspace))
        print(f"[reg] {meta['id']}: {len(meta['corrections'])} corrections, "
              f"{len(meta['skipped'])} skipped")

    if args.cmd in ("export-tkquality", "export-all"):
        from ngtagger.viz.mva_explorer import tkquality_export as tq

        kw = {}
        if getattr(args, "nano_dir", None):
            kw["nano_dir"] = args.nano_dir
        if getattr(args, "models_dir", None):
            kw["models_dir"] = args.models_dir
        if args.cmd == "export-tkquality":
            kw["n_files"] = args.n_files
            kw["max_events"] = args.max_events
        meta = tq.export_tkquality(out, **kw)
        print(f"[tkq] {sum(g['n_rows'] for g in meta['groups'])} tracks in "
              f"{len(meta['groups'])} views -> {meta['file']}")

    if args.cmd in ("export-tagger", "export-all"):
        from ngtagger.viz.mva_explorer import tagger_export as tg

        kw = {}
        if getattr(args, "dumps_dir", None):
            kw["dumps_dir"] = args.dumps_dir
        try:
            meta = tg.export_tagger(out, **kw)
            print(f"[tagger] {len(meta['groups'])} (cell, seed) groups -> "
                  f"{meta['file']}")
        except FileNotFoundError as e:
            if args.cmd != "export-all":
                raise
            print(f"[tagger] skipped: {e}")

    if args.cmd in ("make-site", "export-all"):
        make_site(out)
    elif os.path.isdir(out):  # any data export refreshes the manifest
        write_manifest(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
