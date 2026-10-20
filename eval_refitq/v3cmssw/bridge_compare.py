"""The closure-nano bridge (v3cmssw/PLAN.md rev 1 §3): the closure nano's SmartPixels cluster columns and OT track
columns, rounded to the mantissa bits canon_v4 stores them with, equal canon_v4 bitwise on the common events
(paired by run, lumi, event). It catches a defect upstream of the refit that the CMSSW job and the replay's closure
arm (which reads the closure nano) would share.

    python eval_refitq/v3cmssw/bridge_compare.py CANON_V4.root CLOSURE.root --cluster-src L1SmartPixelsClusterTableProducer.cc \
        [--prefix L1TSmartPixelsCluster_ --prefix L1TTrack_] [--out o.json]

The stored mantissa bits of each reduced-precision column are the ones the table producer declares
(addColumn<float>("name", ..., bits(N)), which closurePrecision turns into full float); they are cross-checked
against canon_v4's values (every value's mantissa must fit in the declared bits). Every other column of the
prefixes must be bitwise equal as written. Rounding: MiniFloatConverter::reduceMantissaToNbitsRounding.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import awkward as ak
import numpy as np
import uproot


def round_mantissa(x, bits):
    """MiniFloatConverter::reduceMantissaToNbitsRounding (DataFormats/Math/interface/libminifloat.h)."""
    shift = 23 - bits
    i = np.asarray(x, np.float32).view(np.uint32).copy()
    up = (i & np.uint32(1 << (shift - 1))) != 0
    mant = (i & np.uint32(0x007FFFFF)) >> np.uint32(shift)
    mant = np.where(up & (mant < np.uint32((1 << bits) - 2)), mant + np.uint32(1), mant)
    i = np.where(up, (i & np.uint32(0xFF800000)) | (mant << np.uint32(shift)),
                 i & np.uint32(((0xFFFFFFFF >> shift) << shift) & 0xFFFFFFFF))
    return i.astype(np.uint32).view(np.float32)


def declared_bits(src):
    text = open(src).read()
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r'addColumn<float>\(\s*"(\w+)"(?:(?!addColumn).)*?bits\((\d+)\)\s*\)\s*;', text, re.S)}


def fits(values, bits):
    """Every value's mantissa has only `bits` significant bits (the low 23 - bits are zero)."""
    i = np.asarray(values, np.float32).view(np.uint32)
    return bool(np.all((i & np.uint32((1 << (23 - bits)) - 1)) == 0))


def keys_of(t):
    r, l, e = (t[k].array(library="np") for k in ("run", "luminosityBlock", "event"))
    return list(zip(r.tolist(), l.tolist(), e.tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("canon")
    ap.add_argument("closure")
    ap.add_argument("--cluster-src", required=True)
    ap.add_argument("--prefix", action="append", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    prefixes = tuple(a.prefix or ("L1TSmartPixelsCluster_", "L1TTrack_"))
    cluster_bits = declared_bits(a.cluster_src)
    ta, tb = uproot.open(f"{a.canon}:Events"), uproot.open(f"{a.closure}:Events")
    ka, kb = keys_of(ta), keys_of(tb)
    pos_a = {k: i for i, k in enumerate(ka)}
    common = [k for k in kb if k in pos_a]
    ia = np.array([pos_a[k] for k in common], np.int64)
    ib = np.array([kb.index(k) for k in common], np.int64)
    names = sorted(k for k in set(ta.keys()) & set(tb.keys()) if k.startswith(prefixes))
    res = {"canon": a.canon, "closure": a.closure, "common_events": len(common), "prefixes": list(prefixes),
           "compared": len(names), "rounded": {}, "differing": [], "declared_bits_violated": [],
           "only_in_canon": sorted(k for k in set(ta.keys()) - set(tb.keys()) if k.startswith(prefixes)),
           "only_in_closure": sorted(k for k in set(tb.keys()) - set(ta.keys()) if k.startswith(prefixes))}
    for name in names:
        va = ak.flatten(ta[name].array(library="ak")[ia], axis=None)
        vb = ak.flatten(tb[name].array(library="ak")[ib], axis=None)
        va, vb = ak.to_numpy(va), ak.to_numpy(vb)
        col = name.split("_", 1)[1]
        bits = cluster_bits.get(col) if name.startswith("L1TSmartPixelsCluster_") else None
        if bits is not None and va.dtype == np.float32:
            res["rounded"][name] = bits
            if not fits(va, bits):
                res["declared_bits_violated"].append(name)
            vb = round_mantissa(vb, bits)
        same = len(va) == len(vb) and va.dtype == vb.dtype and np.array_equal(
            va.view(f"u{va.dtype.itemsize}") if va.dtype.kind == "f" else va,
            vb.view(f"u{vb.dtype.itemsize}") if vb.dtype.kind == "f" else vb)
        if not same:
            res["differing"].append(name)
    ok = not res["differing"] and not res["declared_bits_violated"] and len(common) > 0
    res["verdict"] = "pass" if ok else "FAIL"
    text = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(json.dumps({k: (v if k != "rounded" else len(v)) for k, v in res.items()}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
