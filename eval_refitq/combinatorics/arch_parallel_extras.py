"""Two readouts beside the parallel seed study's matching pass (arch_parallel_census.py), from the same censuses and
canon_v5 files; reads only, no fit:
  - hits per IT mini-track (the census's standard entries that survive duplicate removal), with the owned share at each
    hit count: how much of the IT side is 3-hit fakes;
  - particles with an OT track (distinct L1TTrack_tpIdx >= 0 per event), and how many of them also own an IT
    mini-track: the per-particle figure's denominator against all the OT side's particles.
Events are the file's entries from 0, as the census reads them (the matching pass checks them by (run, lumi, event)).

  python arch_parallel_extras.py -c CENSUS_DIR [CENSUS_DIR ...] -i FILE [FILE ...] -o OUT.json   (one census per file)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_findability as TF         # noqa: E402
from arch_parallel_census import key  # noqa: E402


def one_file(census, path):
    """(hits per IT mini-track, owner per IT mini-track, IT owner keys, OT particle keys, n_events) of one pair."""
    d = Path(TF.census_dir(census))
    meta = json.load(open(d / "manifest.json"))
    cols = {c: i for i, c in enumerate(meta["track_cols"])}
    nh, own, ev = [], [], []
    for p in sorted(d.glob("chunk_*.npz")):
        z = np.load(p)
        for i, soft in enumerate(meta["seed_soft"]):
            if soft or f"trk_{i}" not in z.files:
                continue
            dr = z[f"hit_dr{i}"].astype(bool)
            R = z[f"trk_{i}"][dr]
            nh.append((z[f"hit_{i}"][dr].astype(np.int64) >= 0).sum(1))
            own.append(R[:, cols["own_tpidx"]].astype(np.int64))
            ev.append(R[:, cols["event"]].astype(np.int64))
    nh, own, ev = np.concatenate(nh), np.concatenate(own), np.concatenate(ev)
    nev = int(meta["n_events"])
    with uproot.open(path) as f:
        A = f["Events"].arrays(["L1TTrack_tpIdx"], entry_stop=nev)
    n = np.asarray([len(x) for x in A["L1TTrack_tpIdx"]])
    tp = np.concatenate([np.asarray(x, np.int64) for x in A["L1TTrack_tpIdx"]])
    oev = np.repeat(np.arange(nev, dtype=np.int64), n)
    ot_keys = np.unique(key(oev[tp >= 0], tp[tp >= 0]))
    it_keys = np.unique(key(ev[own >= 0], own[own >= 0]))
    return nh, own, it_keys, ot_keys, nev


def run(a):
    if len(a.census) != len(a.input):
        raise SystemExit("one census per input file")
    nh, own, it_k, ot_k, nev = [], [], [], [], 0
    for c, path in zip(a.census, a.input):
        h, o, ik, ok, n = one_file(c, path)
        nh.append(h)
        own.append(o)
        it_k.append(ik + key(nev, 0))                   # the files' events in sequence, as the matching pass
        ot_k.append(ok + key(nev, 0))
        nev += n
    nh, own = np.concatenate(nh), np.concatenate(own)
    it_k, ot_k = np.concatenate(it_k), np.concatenate(ot_k)
    fake = own < 0
    res = {"census": a.census, "input": a.input, "n_events": nev, "it_per_event": len(nh) / nev,
           "it_owned": float(np.mean(~fake)), "hits": {}}
    for lab, b in (("3", nh == 3), ("4", nh == 4), ("5+", nh >= 5)):
        res["hits"][lab] = {"share": float(b.mean()), "owned": float(np.mean(~fake[b])) if b.any() else None,
                            "share_of_fakes": float((b & fake).sum() / max(fake.sum(), 1))}
    both = np.intersect1d(it_k, ot_k)
    res["particles"] = {"with_ot_track_per_event": len(ot_k) / nev, "with_both_per_event": len(both) / nev,
                        "with_both": int(len(both)), "both_share_of_ot": len(both) / max(len(ot_k), 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["hits"]), json.dumps(res["particles"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-c", "--census", nargs="+", required=True, help="the IT-only censuses, one per input file")
    ap.add_argument("-i", "--input", nargs="+", required=True, help="the canon_v5 files, in the censuses' order")
    ap.add_argument("-o", "--out", required=True)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
