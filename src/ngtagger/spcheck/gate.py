"""The fast gate (plan App. C.3-1, C.4, D.0, D.1).

    python -m ngtagger.spcheck.gate list
    python -m ngtagger.spcheck.gate run        [--items ID ...] [--json OUT] [--record-manifest HASH]
    python -m ngtagger.spcheck.gate known-bad  [--items ID ...] [--json OUT]
    python -m ngtagger.spcheck.gate derive-refs [--items ID ...] [--refs DIR]

`run` judges the code it is run from (under the launcher: the frozen copy) on the known-good
input; exit 0 only when every applicable item is green. Items whose script tree is not in
this code root are reported "n/a" (they do not test code this job runs). `known-bad` runs
each item's named known-bad and exits 0 only when every one is RED. `derive-refs` writes the
reference statistics (refs/<item>.json) from the known-good with the code as it is; they
are registered with the code that made them before any gated run uses them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback

from pathlib import Path

from . import core
from .core import KNOWN_GOOD_EVENTS, KNOWN_GOOD_INPUT, Ctx, NotApplicable, jsonable, now_utc
from . import items_census, items_defs, items_infra, items_nano, items_tagger

ALL_ITEMS = (items_infra.ITEMS + items_defs.ITEMS + items_nano.ITEMS + items_census.ITEMS + items_tagger.ITEMS)
BY_ID = {it.id: it for it in ALL_ITEMS}


def _select(ids):
    if not ids:
        return list(ALL_ITEMS)
    missing = [i for i in ids if i not in BY_ID]
    if missing:
        raise SystemExit(f"unknown items {missing}; known: {sorted(BY_ID)}")
    return [BY_ID[i] for i in ids]


def _ctx(a) -> Ctx:
    if getattr(a, "refs", None):
        core.REFS_DIR = Path(a.refs)
    scratch = a.scratch or tempfile.mkdtemp(prefix="spcheck_")
    return Ctx(input_path=a.input, events=a.events, scratch=scratch)


def cmd_list(a) -> int:
    for it in ALL_ITEMS:
        print(f"{it.id:28s} {it.title}\n{'':28s} C.4: {it.c4_row}\n{'':28s} known-bad: {it.known_bad_desc}\n")
    return 0


def _run_items(a, mode: str) -> dict:
    ctx = _ctx(a)
    res = {"mode": mode, "time": now_utc(), "input": ctx.input, "events": ctx.events,
           "code_root": str(ctx.code_root), "frozen_root": os.environ.get("SP_FROZEN_ROOT"),
           "manifest_hash": a.record_manifest if hasattr(a, "record_manifest") else None, "items": {}}
    for it in _select(a.items):
        t0 = time.time()
        ref = it.load_ref()
        rec = {"title": it.title, "c4_row": it.c4_row, "rule": it.rule_desc, "known_bad": it.known_bad_desc}
        try:
            if mode == "run":
                stat = it.compute(ctx)
                ok, why = it.judge(stat, ref)
                rec["status"] = "green" if ok else "red"
            else:
                stat = it.known_bad(ctx)
                red, why = it.known_bad_is_red(stat, ref)
                rec["status"] = "red-as-required" if red else "NOT-RED (the check cannot fail)"
            rec["why"] = why
            rec["stat"] = jsonable(stat)
        except NotApplicable as e:
            rec["status"], rec["why"] = "n/a", str(e)
        except Exception as e:
            rec["status"], rec["why"] = "error", f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-3000:]
        rec["wall_s"] = round(time.time() - t0, 1)
        res["items"][it.id] = rec
        print(f"{rec['status']:16s} {it.id:28s} {rec['wall_s']:6.1f}s  {str(rec['why'])[:220]}", flush=True)
    return res


def cmd_run(a) -> int:
    res = _run_items(a, "run")
    st = [r["status"] for r in res["items"].values()]
    res["verdict"] = "green" if all(s in ("green", "n/a") for s in st) and "green" in st else "red"
    _write(a, res)
    if a.record_manifest and a.record_manifest != "none":
        from ngtagger.spops import registry as R

        res["result_file"] = a.json
        R.store_gate(None, a.record_manifest, res)
    print(f"FAST GATE {res['verdict'].upper()}")
    return 0 if res["verdict"] == "green" else 1


def cmd_known_bad(a) -> int:
    res = _run_items(a, "known-bad")
    ok = all(r["status"] in ("red-as-required", "n/a") for r in res["items"].values())
    res["verdict"] = "every known-bad red" if ok else "SOME KNOWN-BAD NOT RED"
    _write(a, res)
    print(res["verdict"])
    return 0 if ok else 1


def cmd_derive_refs(a) -> int:
    ctx = _ctx(a)
    if ctx.input != KNOWN_GOOD_INPUT or ctx.events != KNOWN_GOOD_EVENTS:
        print("derive-refs: references come from the known-good input only", file=sys.stderr)
        return 2
    core.REFS_DIR.mkdir(parents=True, exist_ok=True)
    bad = 0
    for it in _select(a.items):
        try:
            stat = jsonable(it.compute(ctx))
        except NotApplicable as e:
            print(f"n/a   {it.id}: {e}")
            continue
        ref = {"item": it.id, "derived": now_utc(), "input": ctx.input, "events": ctx.events,
               "code_root": str(ctx.code_root), "job_id": os.environ.get("SP_JOB_ID"),
               "rule": it.rule_desc, "known_bad": it.known_bad_desc, "stat": stat}
        ok, why = it.judge(stat, ref)
        if not ok:
            bad += 1
            print(f"NOT WRITTEN {it.id}: the known-good fails its own rule: {why}")
            continue
        with open(it.ref_path(), "w") as f:
            json.dump(ref, f, indent=1, sort_keys=True, default=str)
            f.write("\n")
        print(f"wrote {it.ref_path()}  ({why})")
    return 1 if bad else 0


def _write(a, res):
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=str)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spcheck.gate")
    sub = ap.add_subparsers(dest="what", required=True)
    sub.add_parser("list")
    for name in ("run", "known-bad", "derive-refs"):
        s = sub.add_parser(name)
        s.add_argument("--items", nargs="*")
        s.add_argument("--input", default=KNOWN_GOOD_INPUT)
        s.add_argument("--events", type=int, default=KNOWN_GOOD_EVENTS)
        s.add_argument("--scratch")
        s.add_argument("--json")
        s.add_argument("--refs", help="reference directory (default: refs/ beside this module)")
        if name == "run":
            s.add_argument("--record-manifest", default=None)
    a = ap.parse_args(argv)
    return {"list": cmd_list, "run": cmd_run, "known-bad": cmd_known_bad, "derive-refs": cmd_derive_refs}[a.what](a)


if __name__ == "__main__":
    sys.exit(main())
