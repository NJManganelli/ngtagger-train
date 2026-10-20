"""Bitwise comparison of two nano files on every common Events branch, event by (run, lumi, event) (v3cmssw/PLAN.md
rev 1 §3: the v2 regression, IIII, the canon_v5 bridge, run-twice).

    python eval_refitq/v3cmssw/bitwise_compare.py A.root B.root [--exclude GLOB ...] [--only GLOB ...] [--out o.json]

Default exclusion: L1puppiMLMET* (not deterministic run to run, WP-D's run-twice). Events present in only one file
are reported; the comparison uses the common events. Branches in only one file are listed (new / removed columns).
NaN equals NaN. Exit 0 when every compared branch is bitwise equal on every common event.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys

import awkward as ak
import numpy as np
import uproot

DEFAULT_EXCLUDE = ("L1puppiMLMET*",)


def keys_of(t):
    r, l, e = (t[k].array(library="np") for k in ("run", "luminosityBlock", "event"))
    return list(zip(r.tolist(), l.tolist(), e.tolist()))


def same(a, b):
    a, b = ak.to_numpy(ak.flatten(a, axis=None)), ak.to_numpy(ak.flatten(b, axis=None))
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f":
        return bool(np.array_equal(a.view(np.uint32 if a.dtype == np.float32 else np.uint64),
                                   b.view(np.uint32 if b.dtype == np.float32 else np.uint64))
                    or np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--exclude", action="append", default=list(DEFAULT_EXCLUDE))
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    ta, tb = uproot.open(a.a)["Events"], uproot.open(a.b)["Events"]
    ka, kb = keys_of(ta), keys_of(tb)
    common = sorted(set(ka) & set(kb))
    ia = np.array([ka.index(k) for k in common])
    ib = np.array([kb.index(k) for k in common])
    ba, bb = set(ta.keys()), set(tb.keys())
    names = sorted(n for n in ba & bb
                   if not any(fnmatch.fnmatch(n, x) for x in a.exclude)
                   and (not a.only or any(fnmatch.fnmatch(n, x) for x in a.only)))
    diff = []
    for n in names:
        if not same(ta[n].array()[ia], tb[n].array()[ib]):
            diff.append(n)
    res = {"a": a.a, "b": a.b, "events_a": len(ka), "events_b": len(kb), "common_events": len(common),
           "compared_branches": len(names), "excluded": a.exclude, "differing": diff,
           "only_in_a": sorted(ba - bb), "only_in_b": sorted(bb - ba),
           "verdict": "pass" if not diff and common else "FAIL"}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 40 else v[:40] + ["..."]) for k, v in res.items()},
                     indent=1))
    sys.exit(0 if res["verdict"] == "pass" else 1)


if __name__ == "__main__":
    main()
