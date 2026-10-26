"""The grid-produced slim tagger tier (SPTaggerTier) as the tagger study's input (TAGGER_SP_TIER=slim).

  TAGGER_SP_TIER=slim python eval_refitq/tagger_sp/slimtier.py freeze [--wait 180]   inventory + frozen file set
  TAGGER_SP_TIER=slim python eval_refitq/tagger_sp/slimtier.py status [--wait 60]    the same checks, nothing written

Files: <BASE>/slimtaggertier_slimtaggertier_SP<mask>_RelValTTbar_14TeV/<timestamp>/0000/taggertier_<N>.root, one CRAB
task per mask, 100 events per job (N = the job number; it does NOT identify the same events in another mask).
A file is USABLE only if
  - complete: its size and mtime are equal across two checks `wait` seconds apart, and its mtime is older than
    `wait` (a file still being copied is never opened);
  - it opens and Events holds EVENTS_PER_FILE entries;
  - its markers are right: smartPixelsTier starts with "SPTaggerTier" and the job configuration (ParameterSets)
    names the directory's mask (mask=<MASK>); SmartPixels masks: smartPixelsRefit starts with "v3 ", no accepted IT
    hit on an inactive layer (the OR of L1TSmartPixelsExtTrack_spixLayerHitMask has no bit of an I layer, IL_k /
    ID_k = bits k-1 / k+3) and every track has spixRefitVersion 3; IIII (the OT-only control, a passthrough): no
    smartPixelsRefit marker and no refit columns (its heads are the OT heads on L1TExtTrack).
`freeze` writes the usable files per mask (cache/slim_inventory.json; rejected files with the reason) and each
file's event identities; every run re-scans, so files that arrived since are picked up. Masks are paired by EVENT
IDENTITY: the study uses only the events present in every mask (cache/slim_common_events.npy); a file sharing events
with an earlier usable file of its mask (a resubmitted job) is rejected, so no event enters twice.
Splits are by event identity and stable as files arrive: a 64-bit hash of (run, lumi, event); test = 20 % of the
events, validation = 10 % of the rest.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

BASE = os.environ.get("SLIMTIER_BASE", "/Volumes/NJM256GBSD/smartpixels/slimtaggertier/RelValTTbar_14TeV")
PATTERN = "slimtaggertier_slimtaggertier_SP{mask}_RelValTTbar_14TeV/*/0000/taggertier_*.root"
EVENTS_PER_FILE = 100
TIER_MARK, REFIT_MARK = "SPTaggerTier", "v3 "
EVENT_ID = ("run", "luminosityBlock", "event")
HASH_SEEDS = {"test": 0x5EED_7E57, "val": 0x5EED_0A11}
# dev tests only: restrict the masks scanned and the files used per mask (the study uses every mask and file)
DEV_MASKS = os.environ.get("SLIMTIER_DEV_MASKS", "").split() or None
DEV_MAX_FILES = int(os.environ.get("SLIMTIER_DEV_MAX_FILES", "0")) or None


def masks():
    return DEV_MASKS or C.MASKS


def inventory_path():
    return os.path.join(C.OUT, "cache", "slim_inventory.json")


def common_path():
    return os.path.join(C.OUT, "cache", "slim_common_events.npy")


def evid_path(mask, job):
    d = os.path.join(C.OUT, "cache", "slim_evid")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{mask}_{job:04d}.npy")


def candidates(mask):
    out = []
    for p in sorted(glob.glob(os.path.join(BASE, PATTERN.format(mask=mask)))):
        m = re.search(r"taggertier_(\d+)\.root$", p)
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    return out[:DEV_MAX_FILES] if DEV_MAX_FILES else out


def stat(p):
    s = os.stat(p)
    return s.st_size, s.st_mtime


def allowed_bits(mask):
    """spixLayerHitMask bits of the ACTIVE layers: letter k (IL_k and ID_k) -> bits k-1 and k+3."""
    return sum((1 << i) | (1 << (i + 4)) for i, c in enumerate(mask) if c == "A")


def config_mask(f):
    """The mask named in the file's job configuration (ParameterSets blob, raw bytes)."""
    br = f["ParameterSets"]["IdToParameterSetsBlobs"]
    raw = b"".join(bytes(br.basket(i).data) for i in range(br.num_baskets))
    found = set(m.decode() for m in re.findall(rb"mask=\+Z\(([AI]{4})\x00\)", raw))
    return found


def check_file(mask, p):
    """(None, event ids) if usable, else (reason, None). Never called on a file still being copied."""
    import awkward as ak
    import uproot

    try:
        f = uproot.open(p)
        t = f["Events"]
        if t.num_entries != EVENTS_PER_FILE:
            return f"{t.num_entries} events, not {EVENTS_PER_FILE}", None
        tier = str(f["smartPixelsTier"])
        if not tier.startswith(TIER_MARK):
            return f"tier marker {tier!r}", None
        cm = config_mask(f)
        if cm != {mask}:
            return f"configured mask {sorted(cm)} != {mask}", None
        keys = set(t.keys())
        if mask == "IIII":                       # the OT-only control: no refit (no marker, no refit columns)
            if "smartPixelsRefit" in [k.split(";")[0] for k in f.keys()]:
                return "IIII with a smartPixelsRefit marker", None
            if any(k.startswith("L1TSmartPixelsExtTrack_spix") for k in keys):
                return "IIII with refit columns", None
            A = t.arrays([*EVENT_ID], library="ak")
        else:
            refit = str(f["smartPixelsRefit"]) if "smartPixelsRefit" in [k.split(";")[0] for k in f.keys()] else ""
            if not refit.startswith(REFIT_MARK):
                return f"refit marker {refit!r}", None
            A = t.arrays(["L1TSmartPixelsExtTrack_spixLayerHitMask", "L1TSmartPixelsExtTrack_spixRefitVersion",
                          *EVENT_ID], library="ak")
            bits = int(np.bitwise_or.reduce(ak.to_numpy(ak.flatten(A["L1TSmartPixelsExtTrack_spixLayerHitMask"]))
                                            .astype(np.int64), initial=0))
            if bits & ~allowed_bits(mask):
                return f"accepted hits on an inactive layer (layer-hit bits {bits:#010b})", None
            ver = ak.to_numpy(ak.flatten(A["L1TSmartPixelsExtTrack_spixRefitVersion"]))
            if len(ver) and not np.all(ver == 3):
                return f"spixRefitVersion {sorted(set(ver.tolist()))} (not all 3)", None
        ev = np.column_stack([ak.to_numpy(A[c]).astype(np.int64) for c in EVENT_ID])
        return None, ev
    except Exception as e:                                       # a damaged or unreadable file
        return f"unreadable: {type(e).__name__}: {e}"[:300], None


def freeze(wait=180.0, log=lambda *x: print(*x, flush=True), write=True):
    """Scan, wait, re-scan; check (and with `write`, record) the usable files of every mask and the events common
    to all."""
    first = {m: {p: stat(p) for _, p in candidates(m)} for m in masks()}
    if wait > 0:
        log(f"slim tier: {sum(len(v) for v in first.values())} files seen; waiting {wait:.0f} s for size stability")
        time.sleep(wait)
    now = time.time()
    inv = {"base": BASE, "frozen": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wait_s": wait, "masks": {}}
    sets = {}
    for mask in masks():
        use, rej, seen = [], [], {}
        for job, p in candidates(mask):
            st = stat(p)
            if p not in first[mask] or first[mask][p] != st or now - st[1] < wait:
                rej.append({"path": p, "reason": "not stable yet (size/mtime changed or too recent)"})
                continue
            why, ev = check_file(mask, p)
            if why:
                rej.append({"path": p, "reason": why})
                continue
            keys = [tuple(r) for r in ev.tolist()]
            if len(set(keys)) != len(keys):
                rej.append({"path": p, "reason": "duplicate event identities inside the file"})
                continue
            dup = sorted({seen[k] for k in keys if k in seen})
            if dup:                                              # a resubmitted job: the earlier file is kept
                rej.append({"path": p, "reason": f"shares events with the usable job(s) {dup} of this mask"})
                continue
            if write:
                np.save(evid_path(mask, job), ev)
            for k in keys:
                seen[k] = job
            use.append({"job": job, "path": p, "size": st[0], "mtime": st[1], "n_events": len(keys)})
        jobs = [u["job"] for u in use]
        if len(set(jobs)) != len(jobs):
            raise RuntimeError(f"{mask}: two usable files with one job number (two task directories?): resolve first")
        inv["masks"][mask] = {"usable": use, "rejected": rej, "n_events": len(seen)}
        sets[mask] = set(seen)
        log(f"slim tier {mask}: {len(use)} usable files, {len(rej)} rejected, {len(seen)} distinct events")
    common = set.intersection(*sets.values()) if all(sets.values()) else set()
    arr = np.array(sorted(common), np.int64).reshape(-1, 3)
    inv["common_events"] = int(len(arr))
    inv["runs"] = sorted({int(r) for r in arr[:, 0]}) if len(arr) else []
    if write:
        os.makedirs(os.path.dirname(common_path()), exist_ok=True)
        np.save(common_path(), arr)
        with open(inventory_path(), "w") as f:
            json.dump(inv, f, indent=1)
    log(f"slim tier: {len(arr)} events common to {masks()}")
    if DEV_MASKS or DEV_MAX_FILES:
        inv["DEV"] = {"masks": DEV_MASKS, "max_files": DEV_MAX_FILES}
    return inv


def frozen_files(mask):
    """[(job, path)] of the frozen usable files of a mask."""
    inv = json.load(open(inventory_path()))
    return [(u["job"], u["path"]) for u in inv["masks"][mask]["usable"]]


_COMMON = None


def common_keys():
    global _COMMON
    if _COMMON is None:
        _COMMON = np.sort(key64(np.load(common_path())))
    return _COMMON


def key64(ev):
    """One int64 per (run, lumi, event) (run < 2^20, lumi < 2^20, event < 2^23 checked)."""
    ev = np.asarray(ev, np.int64).reshape(-1, 3)
    if len(ev) and (ev[:, 0].max() >= 1 << 20 or ev[:, 1].max() >= 1 << 20 or ev[:, 2].max() >= 1 << 23):
        raise ValueError("run / lumi / event beyond the key range")
    return (ev[:, 0] << 43) | (ev[:, 1] << 23) | ev[:, 2]


def in_common(ev):
    k = key64(ev)
    c = common_keys()
    i = np.clip(np.searchsorted(c, k), 0, max(len(c) - 1, 0))
    return (c[i] == k) if len(c) else np.zeros(len(k), bool)


def _u(ev, which):
    """A uniform [0, 1) number per event from a splitmix64 hash of its identity (stable as files arrive)."""
    x = key64(ev).astype(np.uint64) ^ np.uint64(HASH_SEEDS[which])
    with np.errstate(over="ignore"):
        x = (x + np.uint64(0x9E3779B97F4A7C15))
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def is_test(ev):
    return _u(ev, "test") < C.TEST_FRACTION


def is_val(ev):
    return ~is_test(ev) & (_u(ev, "val") < C.VAL_FRACTION)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["freeze", "status"])
    ap.add_argument("--wait", type=float, default=180.0)
    a = ap.parse_args()
    if C.TIER != "slim":
        raise SystemExit("set TAGGER_SP_TIER=slim")
    inv = freeze(a.wait, write=(a.what == "freeze"))
    if a.what == "status":
        for m, x in inv["masks"].items():
            print(m, len(x["usable"]), "usable;", "rejected:", [(os.path.basename(r["path"]), r["reason"]) for r in x["rejected"]][:8])


if __name__ == "__main__":
    main()
