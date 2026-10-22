"""Checks of the tagger tier (one job per mask) and the linked truth on the dev outputs (file 4, 10 events):

  truth   the otlink truth on the SmartPixels track tables (otTp*, otGenuine, ...) equals the OT track's columns at
          l1TrackIdx bit for bit; selTpIdx equals L1TSmartPixelsCluster_tpIdx at selClusterIdx; and the tagger's
          otlink truth (ngtagger.train.trackmva.tracks_from_batch: the base table's TP and t_n_it_wrong_ot, which
          today reads L1TSmartPixelsCluster) is reproduced exactly without the cluster table
  tier    every branch of the tier file is in the canonical file and bitwise equal by (run, lumi, event); the
          canonical branches it lacks are exactly the dropped tables'; its size per event

    python eval_refitq/v3cmssw/tagtier_check.py --dir DIR [--tagger-src SRC] [--out o.json]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys

import awkward as ak
import numpy as np
import uproot

DROPPED = ("L1TSmartPixelsCluster", "L1TTP", "L1TOTStub", "L1TOTStubDTC")
EXCLUDE = ("L1puppiMLMET*",)
OTLINK = {"tpIdx": "otTpIdx", "tpPdgId": "otTpPdgId", "genuine": "otGenuine", "looselyGenuine": "otLooselyGenuine",
          "combinatoric": "otCombinatoric", "unknown": "otUnknown", "tpFromHardInteraction": "otTpFromHardInteraction",
          "tp_pt": "otTp_pt", "tp_eta": "otTp_eta", "tp_phi0": "otTp_phi0", "tp_tanL": "otTp_tanL",
          "tp_charge": "otTp_charge", "tp_rInv": "otTp_rInv", "tp_d0": "otTp_d0", "tp_z0": "otTp_z0"}


def ids(t):
    return list(zip(*(t[k].array(library="np").tolist() for k in ("run", "luminosityBlock", "event"))))


def bits(a):
    a = ak.to_numpy(ak.flatten(a, axis=None)) if isinstance(a, ak.Array) else np.asarray(a)
    return a.view(f"u{a.dtype.itemsize}") if a.dtype.kind == "f" else a


def same(a, b):
    na, nb = ak.num(a, axis=1) if a.ndim > 1 else None, ak.num(b, axis=1) if b.ndim > 1 else None
    if (na is None) != (nb is None) or (na is not None and not np.array_equal(ak.to_numpy(na), ak.to_numpy(nb))):
        return False
    fa, fb = ak.to_numpy(ak.flatten(a, axis=None)), ak.to_numpy(ak.flatten(b, axis=None))
    return fa.dtype == fb.dtype and np.array_equal(bits(fa), bits(fb))


def paired(ta, tb):
    ka, kb = ids(ta), ids(tb)
    common = [k for k in ka if k in set(kb)]
    return np.array([ka.index(k) for k in common]), np.array([kb.index(k) for k in common])


def compare(ta, tb, names_a):
    ia, ib = paired(ta, tb)
    diff, missing = [], []
    kb = set(tb.keys())
    for n in names_a:
        if n not in kb:
            missing.append(n)
            continue
        if not same(ta[n].array()[ia], tb[n].array()[ib]):
            diff.append(n)
    return {"common_events": int(len(ia)), "compared": len(names_a) - len(missing), "differing": diff,
            "missing_in_b": missing}


def excluded(n):
    return any(fnmatch.fnmatch(n, x) for x in EXCLUDE)


def table_of(n):
    t = n.split("_", 1)[0]
    return t[1:] if "_" not in n and t.startswith("n") else t


def check_truth(fn, tagger_src):
    t = uproot.open(f"{fn}:Events")
    keys = set(t.keys())
    res = {}
    for ot, sp in (("L1TTrack", "L1TSmartPixelsTrack"), ("L1TExtTrack", "L1TSmartPixelsExtTrack")):
        link = t[f"{sp}_l1TrackIdx"].array()
        bad = []
        for c, oc in OTLINK.items():
            src, dst = t[f"{ot}_{c}"].array(), t[f"{sp}_{oc}"].array()
            ok = ak.all(link >= 0)
            want = src[ak.where(link >= 0, link, 0)]
            if not (ok and same(want, dst)):
                bad.append(oc)
        res[sp] = {"otlink_columns": len(OTLINK), "differing": bad}
    for hit in ("L1TSmartPixelsRefitHit", "L1TSmartPixelsExtRefitHit"):
        sel, stp = t[f"{hit}_selClusterIdx"].array(), t[f"{hit}_selTpIdx"].array()
        ctp = t["L1TSmartPixelsCluster_tpIdx"].array()
        want = ak.values_astype(ak.where(sel >= 0, ctp[ak.where(sel >= 0, sel, 0)], -1), np.int32)
        res[hit] = {"hits": int(ak.sum(ak.num(sel))), "selTpIdx_equal": bool(same(want, stp))}
    # the tagger's otlink truth: with the cluster table (today) and from the new columns alone
    sys.path.insert(0, tagger_src)
    from ngtagger.train import trackmva as tm
    br = [b for b in tm.nano_branches(keys, "canonical_ext", "", True) if b in keys]
    A = t.arrays(br + [b for b in ("L1TSmartPixelsExtRefitHit_selTpIdx",) if b in keys], library="ak")
    T, ntrk = tm.tracks_from_batch(A, "canonical_ext", "", True)
    off = np.concatenate([[0], np.cumsum(ntrk)[:-1]])
    tidx = A["L1TSmartPixelsExtRefitHit_trackIdx"]
    g = off[np.repeat(np.arange(len(ntrk)), ak.to_numpy(ak.num(tidx)))] + ak.to_numpy(ak.flatten(tidx))
    acc = ak.to_numpy(ak.flatten(A["L1TSmartPixelsExtRefitHit_hitAccepted"])) > 0
    htp = ak.to_numpy(ak.flatten(A["L1TSmartPixelsExtRefitHit_selTpIdx"])).astype(np.int64)
    sp = uproot.open(f"{fn}:Events").arrays(["L1TSmartPixelsExtTrack_otTpIdx"], library="ak")
    btp = ak.to_numpy(ak.flatten(sp["L1TSmartPixelsExtTrack_otTpIdx"])).astype(np.int64)
    wrong = ((htp != btp[g]) | (btp[g] == -1)) & acc
    n_wrong = np.bincount(g[wrong], minlength=len(btp)).astype(np.float64)
    res["tagger_otlink"] = {
        "tracks": int(len(btp)),
        "t_n_it_wrong_ot_equal": bool(np.array_equal(n_wrong, T["t_n_it_wrong_ot"])),
        "base_tpIdx_equal_otTpIdx": bool(np.array_equal(T["b_tpIdx"].astype(np.int64), btp)),
        "tp_pt_d0_z0_equal": all(np.array_equal(
            T[f"b_tp_{c}"].astype(np.float32).view(np.uint32),
            ak.to_numpy(ak.flatten(uproot.open(f"{fn}:Events")[f"L1TSmartPixelsExtTrack_otTp_{c}"].array())).view(np.uint32))
            for c in ("pt", "d0", "z0")),
    }
    return res


def check_tier(can, tier):
    tc, tt = uproot.open(f"{can}:Events"), uproot.open(f"{tier}:Events")
    kc, kt = set(tc.keys()), set(tt.keys())
    lacking = sorted(kc - kt)
    unexpected = sorted(n for n in lacking if table_of(n) not in DROPPED)
    res = compare(tt, tc, sorted(n for n in kt if not excluded(n)))
    res.update({"tier_branches": len(kt), "canonical_branches": len(kc), "extra_in_tier": sorted(kt - kc),
                "canonical_only_not_dropped": unexpected,
                "dropped_tables_absent": sorted({table_of(n) for n in lacking}),
                "mb_per_event": sum(b.compressed_bytes for b in tt.branches) / tt.num_entries / 1e6,
                "canonical_mb_per_event": sum(b.compressed_bytes for b in tc.branches) / tc.num_entries / 1e6})
    return res


def cpu(log):
    t = open(log).read()
    m = re.search(r"event loop CPU/event = ([0-9.]+)", t)
    r = re.search(r"event loop Real/event = ([0-9.]+)", t)
    return (float(m.group(1)) if m else None, float(r.group(1)) if r else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tagger-src", default="/Users/nmangane/smartpixels/ngtagger-train-tagger/src")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    masks = ("AAAA", "AAAI", "IAAA")
    res = {"truth": {m: check_truth(f"{a.dir}/can_{m}.root", a.tagger_src) for m in masks},
           "tier": check_tier(f"{a.dir}/can_AAAA.root", f"{a.dir}/tier_AAAA.root"),
           "cpu_per_event": {k: cpu(f"{a.dir}/{k}.log") for k in ("can_AAAA", "can_AAAI", "can_IAAA", "can_IIII",
                                                                  "tier_AAAA")}}
    text = json.dumps(res, indent=1, default=float)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
