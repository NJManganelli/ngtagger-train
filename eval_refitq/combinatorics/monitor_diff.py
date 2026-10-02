#!/usr/bin/env python
"""Compare two SmartPixels omnibus monitor summaries (spix_monitor_summary.json).

The omnibus writes one summary per run: the key numbers per variant / surface /
|eta| bin (R1 track parameters, R2 hit selection, R3 angle monitor, R4 covariance
check, R5 track MVAs, studies 19 / 20) and the provenance (inputs, the refit
producers' configuration and the emulator tables read from the nanos' ParameterSets,
the sigma table's sha256, the code's git state, the CMSSW build when given). This
prints what changed between two of them:

  * provenance: every differing leaf (inputs, configuration, table hashes, ...);
  * numbers: every numeric leaf present in both whose change exceeds the
    thresholds, |B - A| > --abs AND |B - A| / |A| > --rel, largest relative
    change first; leaves present in only one summary are listed separately.

    pixi run python eval_refitq/combinatorics/monitor_diff.py A/spix_monitor_summary.json \\
        B/spix_monitor_summary.json [--rel 0.05] [--abs 0] [--only R1] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math

SKIP_PROVENANCE = ("created", "argv")


def flatten(d, prefix=""):
    """{dotted path: leaf} of a nested dict/list (list items by index)."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}." if prefix or k else k))
    elif isinstance(d, list) and d and all(isinstance(x, (dict, list)) for x in d):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and math.isfinite(x)


def _status(S):
    """{page:study: status} of a summary's run status (time and memory dropped)."""
    return {f"{e['page']}:{e['study']}": e["status"] for e in S.get("status", [])}


def compare(A, B, rel=0.05, abs_=0.0, only=None):
    """(provenance differences, numeric changes, only-in-A, only-in-B). The run
    status is compared per study (ran / skipped / FAILED), not by list position."""
    pa, pb = flatten(A.get("provenance", {})), flatten(B.get("provenance", {}))
    prov = [(k, pa.get(k), pb.get(k)) for k in sorted(set(pa) | set(pb))
            if pa.get(k) != pb.get(k) and not k.split(".")[0] in SKIP_PROVENANCE]
    skip = ("provenance", "status")
    na = flatten({k: v for k, v in A.items() if k not in skip})
    nb = flatten({k: v for k, v in B.items() if k not in skip})
    na.update({f"status.{k}": v for k, v in _status(A).items()})
    nb.update({f"status.{k}": v for k, v in _status(B).items()})
    if only:
        na = {k: v for k, v in na.items() if only in k}
        nb = {k: v for k, v in nb.items() if only in k}
    changes = []
    for k in sorted(set(na) & set(nb)):
        a, b = na[k], nb[k]
        if _num(a) and _num(b):
            d = b - a
            r = abs(d) / abs(a) if a != 0 else (math.inf if d != 0 else 0.0)
            if abs(d) > abs_ and r > rel:
                changes.append((k, a, b, d, r))
        elif a != b and not (_num(a) and _num(b)):
            # non-numeric, or a number that became None/NaN (a sparse or emptied bin)
            changes.append((k, a, b, None, None))
    changes.sort(key=lambda c: -(c[4] if c[4] is not None else math.inf))
    return prov, changes, sorted(set(na) - set(nb)), sorted(set(nb) - set(na))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", help="reference summary (the earlier run)")
    ap.add_argument("b", help="new summary")
    ap.add_argument("--rel", type=float, default=0.05, help="relative-change threshold (default 0.05)")
    ap.add_argument("--abs", type=float, default=0.0, help="absolute-change threshold (default 0)")
    ap.add_argument("--only", default=None, help="compare only leaves whose path contains this")
    ap.add_argument("--json", default=None, help="also write the differences to this JSON")
    ap.add_argument("--max-rows", type=int, default=200)
    args = ap.parse_args()
    with open(args.a) as fh:
        A = json.load(fh)
    with open(args.b) as fh:
        B = json.load(fh)
    prov, changes, only_a, only_b = compare(A, B, args.rel, args.abs, args.only)
    print(f"A = {args.a}\nB = {args.b}\n")
    print(f"PROVENANCE: {len(prov)} difference(s)")
    for k, a, b in prov:
        print(f"  {k}\n      A: {a}\n      B: {b}")
    print(f"\nNUMBERS changed by more than {args.rel:.0%} (and {args.abs:g} absolute): {len(changes)}")
    w = max([len(c[0]) for c in changes[:args.max_rows]] + [10])
    for k, a, b, d, r in changes[:args.max_rows]:
        if d is None:
            print(f"  {k:<{w}}  A {a!s:>12}  B {b!s:>12}")
        else:
            print(f"  {k:<{w}}  A {a:>12.5g}  B {b:>12.5g}  diff {d:>+12.5g}  ({100 * r:+.1f}%)"
                  if math.isfinite(r) else f"  {k:<{w}}  A {a:>12.5g}  B {b:>12.5g}  diff {d:>+12.5g}")
    if len(changes) > args.max_rows:
        print(f"  ... {len(changes) - args.max_rows} more (--max-rows)")
    print(f"\nonly in A: {len(only_a)}" + "".join(f"\n  {k}" for k in only_a[:args.max_rows]))
    print(f"only in B: {len(only_b)}" + "".join(f"\n  {k}" for k in only_b[:args.max_rows]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"a": args.a, "b": args.b, "rel": args.rel, "abs": args.abs,
                       "provenance": [{"key": k, "a": a, "b": b} for k, a, b in prov],
                       "changes": [{"key": k, "a": a, "b": b, "diff": d, "rel": r if r is None or math.isfinite(r) else None}
                                   for k, a, b, d, r in changes],
                       "only_in_a": only_a, "only_in_b": only_b}, fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
