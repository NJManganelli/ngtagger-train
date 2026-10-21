"""Parity of the CMSSW track heads (L1SmartPixelsTrackHeadsTableProducer, writeFeatures) against the tagger's
reference (heads_cmssw/parity/parity_*.json, SPEC.md §5): per track, matched by (run, lumi, event, collection, index
along axis 1), the 13 / 38 features bit for bit in float32 and the six outputs (the width heads' margins exactly,
P(clean) to float rounding).

    python eval_refitq/v3cmssw/heads_parity.py --nano heads.root --parity parity.json [--suffix ""] [--out o.json]
"""
from __future__ import annotations

import argparse
import json

import awkward as ak
import numpy as np
import uproot

TABLES = {"OT": "L1TExtTrack", "SmartPixels": "L1TSmartPixelsExtTrack"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nano", required=True)
    ap.add_argument("--parity", required=True)
    ap.add_argument("--suffix", default="", help="the SmartPixels table suffix (multi-mask job)")
    ap.add_argument("--prefix", default="head_")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ref = json.load(open(a.parity))
    t = uproot.open(f"{a.nano}:Events")
    ev = {(r, l, e): i for i, (r, l, e) in enumerate(zip(*(t[k].array(library="np").tolist()
                                                         for k in ("run", "luminosityBlock", "event"))))}
    cache = {}

    def col(table, name):
        k = f"{table}_{name}"
        if k not in cache:
            cache[k] = t[k].array()
        return cache[k]

    res = {}
    for coll in ("OT", "SmartPixels"):
        table = TABLES[coll] + (a.suffix if coll == "SmartPixels" else "")
        recs = [r for r in ref["tracks"] if r["collection"] == coll]
        names = list(recs[0]["features"]) if recs else []
        fdiff = {n: 0 for n in names}
        mdiff = {h: 0 for h in ref["heads"]}
        max_mdiff = {h: 0.0 for h in ref["heads"]}
        missing = 0
        for r in recs:
            i = ev.get((r["run"], r["lumi"], r["event"]))
            if i is None:
                missing += 1
                continue
            j = r["track_index"]
            for f, n in enumerate(names):
                got = np.float32(col(table, f"{a.prefix}feature{f}")[i][j])
                want = np.float32(r["features"][n])
                if got.view(np.uint32) != want.view(np.uint32):
                    fdiff[n] += 1
            for h in ref["heads"]:
                got = float(col(table, f"{a.prefix}{h}")[i][j])
                want = r["p_clean"] if h == "clean" else r["margin"][h]
                d = abs(got - want)
                max_mdiff[h] = max(max_mdiff[h], d)
                if (h == "clean" and d > 1e-6) or (h != "clean" and np.float32(got) != np.float32(want)):
                    mdiff[h] += 1
        res[coll] = {"tracks": len(recs) - missing, "unmatched": missing,
                     "features_differing": {n: c for n, c in fdiff.items() if c},
                     "outputs_differing": {h: c for h, c in mdiff.items() if c},
                     "max_abs_output_difference": max_mdiff}
    text = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
