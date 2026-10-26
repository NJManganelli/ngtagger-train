"""Shared definitions of the SmartPixels jet-tagging study (eval_refitq/tagger_sp).

Scenarios = canonical (coopt) nanos: the whole downstream L1 chain (GTT vertex, PF, PUPPI, SC4 NG
jets) is re-emulated from the SmartPixels (SP) tracks of each IT active-layer mask. IIII =
passthrough (SP track == its OT track), i.e. the OT-only baseline through the identical pipeline.
  SP masks (AAAA, AAAI, IAAA): canon_v5 (refit v3: the seedless SmartPixels refit).
  IIII: canon_v4. canon_v5 has no IIII: it carries no refit, so it would equal canon_v4's; canon_v5
  equals canon_v4 bit for bit on every branch that does not depend on the refit (bridge of
  2026-10-21), so both hold the same events.

Samples (PU200): ttbar = spstudy_<mask>_f01..f10_100ev.root (1000 ev),
htt = htt/spstudy_<mask>_htt_f<NN>.root (QQ -> H -> tautau RelVal; every file
file number finished for ALL masks is used -- `study_files` -- so every scenario sees the
same events and adding tau files is a rerun of run_all.sh). Files younger than MIN_AGE_S
are skipped (may still be being copied). build.py freezes the set it used in a per-mask
manifest that train.py reads (files appearing mid-run are ignored until the next rerun).

Truth: the OT track's TP everywhere track truth is needed (user decision): the SP
heads use trackmva refit_truth="otlink"; jet labels come from GenJet / GenVisTau /
prompt GenPart only (framework labels.py), identical in every scenario.

Event split (identical for every mask; events are the same across masks):
  ttbar: 20% of ALL ttbar events (200) are test, drawn only from f03..f10 -- f01/f02
         trained the cluster-angle CNN (IT angles in-sample), so they are always train.
  htt:   20% of each file's events are test.
  validation: VAL_FRACTION of each sample's TRAINING events (val_table), the same events for every
         mask, cell and trial: early stopping, the trial screen and the trial choice use them only.
Chosen once with SPLIT_SEED on the EVENT IDENTITY (run, lumi, event): the entry order of a
file differs between masks (multithreaded output) and between canon versions, so entries are
never used as keys.
"""
from __future__ import annotations

import os

import numpy as np

# TAGGER_SP_TIER: "canonical" (the canon_v5 / canon_v4 nanos, below) or "slim" (the grid-produced slim tagger tier,
# slimtier.py: its own file inventory, event pairing across masks, hash splits and the stored CMSSW track heads)
TIER = os.environ.get("TAGGER_SP_TIER", "canonical")
if TIER not in ("canonical", "slim"):
    raise ValueError(f"TAGGER_SP_TIER {TIER!r}")
# TAGGER_SP_OUT: another output folder (dev tests on a few files)
OUT = os.environ.get("TAGGER_SP_OUT", "/Volumes/WDMac/smartpixels_scratch/tagger_slimtier" if TIER == "slim"
                     else "/Volumes/WDMac/smartpixels_scratch/tagger_v5")
MASKS = ["IIII", "AAAA", "AAAI", "IAAA"]       # IIII = OT baseline (passthrough)
BASELINE_MASK = "IIII"
# the SP masks read canon_v5 (refit v3); IIII reads canon_v4 (canon_v5 has none: see the docstring)
CANON = {"sp": "/Volumes/WDMac/smartpixels_scratch/canon_v5", BASELINE_MASK: "/Volumes/WDMac/smartpixels_scratch/canon_v4"}
SAMPLES = {   # sample: (sub-directory of the canon folder, file pattern, file numbers or None = every one present)
    "ttbar": ("", "spstudy_{mask}_f{n:02d}_100ev.root", range(1, 11)),
    "htt": ("htt", "spstudy_{mask}_htt_f{n:02d}.root", None),
}
MIN_AGE_S = 180
TTBAR_TRAIN_ONLY = (1, 2)       # cluster-CNN training files
TEST_FRACTION = 0.2
VAL_FRACTION = 0.1              # of each sample's training events (val_table)
SPLIT_SEED = 20260930

FAMILY = "canonical_ext"        # L1TExtTrack + L1TSmartPixelsExtTrack (+ ext refit hits)
TRACK_TABLE = "L1TSmartPixelsExtTrack"   # what L1ExtPuppiCand.l1TrackIdx points into (coopt)
REFIT_TRUTH = "otlink"          # SP heads: the OT track's TP for residuals and clean (user)
N_CONST = 16

# acceptance of the study (train and test): reco jet pT > 15 GeV (the framework's lowest
# weight bin) and |eta| < 2.4 (tracker acceptance; outside it the scenarios cannot differ
# except through PU)
PT_MIN, ETA_MAX = 15.0, 2.4
ETA_BINS = [0.0, 1.1, 1.8, 2.4]
PT_BINS = [15, 25, 40, 70, 1e9]

GROUPS_ALL = ["baseline", "track", "trkquality", "vertexdxy_pv", "trackres", "trackmva"]
# the slim tier carries the CMSSW-computed reference heads (*_head_*): the "plus track-quality heads" cell reads them
# ("trackheads", the deployed path) instead of fitting the Python heads
HEAD_GROUP = "trackheads" if TIER == "slim" else "trackmva"
if TIER == "slim":
    GROUPS_ALL = [g if g != "trackmva" else HEAD_GROUP for g in GROUPS_ALL]
# Feature variants (cumulative), all with the same tagger / training protocol. The FACTORISATION
# LADDER (user): L0 = "baseline" on IIII (the OT reference, NGT-like tagger); L1 = "baseline" on an
# SP mask (refit tracks only through PF/PUPPI/jets upstream); L2 = "track" (+ the constituent
# track's parameters, fit quality, resolution, IT hits); L3 = "trackmva" (+ track-MVA scores).
# Controls: L2' / L3' = "track" / "trackmva" on IIII (the same slots filled from the OT track,
# OT-only MVA heads) -> separates "more features" from "better tracks".
VARIANTS = {
    "baseline": ["baseline"],
    "track": ["baseline", "track", "trkquality", "vertexdxy_pv", "trackres"],
    "trackmva": ["baseline", "track", "trkquality", "vertexdxy_pv", "trackres", HEAD_GROUP],
}
# jointly trained track-quality heads (joint_heads.py): the "track" inputs + the six heads of one head version
# (reference = uncoupled, A = outputs apart, B = errors apart), each from its own jet caches (jets_dir)
JOINT_VERSIONS = ("ref", "A", "B")
VARIANTS.update({f"joint_{v}": VARIANTS["track"] + ["trackjoint"] for v in JOINT_VERSIONS})
LEVEL = {"baseline": 1, "track": 2, "trackmva": 3, **{f"joint_{v}": 3 for v in JOINT_VERSIONS}}

def sample_dir(sample: str, mask: str) -> str:
    return os.path.join(CANON[BASELINE_MASK if mask == BASELINE_MASK else "sp"], SAMPLES[sample][0])


def path(sample: str, mask: str, n: int) -> str:
    return os.path.join(sample_dir(sample, mask), SAMPLES[sample][1].format(mask=mask, n=n))


def files(sample: str, mask: str, require_all: bool = False) -> list[tuple[int, str]]:
    """(file number, path) of the finished files present (no *_rerun; older than MIN_AGE_S)."""
    import glob
    import re
    import time

    _, pat, nums = SAMPLES[sample]
    d = sample_dir(sample, mask)
    if nums is None:
        rx = re.compile(re.escape(pat.format(mask=mask, n=0)).replace("00", r"(\d+)", 1) + "$")
        nums = sorted(int(m.group(1)) for f in glob.glob(os.path.join(d, f"spstudy_{mask}_*.root"))
                      if (m := rx.match(os.path.basename(f))))
    out = [(n, path(sample, mask, n)) for n in nums if os.path.exists(path(sample, mask, n))
           and time.time() - os.path.getmtime(path(sample, mask, n)) > MIN_AGE_S]
    if require_all and len(out) != len(nums):
        raise FileNotFoundError(f"{sample} {mask}: {len(out)}/{len(nums)} files present")
    return out


def slim():
    import slimtier

    return slimtier


def fileset_path() -> str:
    return os.path.join(OUT, "cache", "fileset.json")


def freeze_fileset(samples) -> dict:
    """Freeze, for this run, the file numbers per sample finished for EVERY mask (run_all.sh
    calls it once at the start, so files finishing mid-run cannot split the masks)."""
    import json

    if TIER == "slim":                               # its own inventory (complete files, markers, common events)
        if list(samples) != ["ttbar"]:
            raise ValueError("the slim tier holds TTbar only")
        inv = slim().freeze()
        return {m: [u["job"] for u in inv["masks"][m]["usable"]] for m in MASKS}
    fs = {s: sorted(set.intersection(*(set(n for n, _ in files(s, m)) for m in MASKS))) for s in samples}
    os.makedirs(os.path.dirname(fileset_path()), exist_ok=True)
    with open(fileset_path(), "w") as f:
        json.dump(fs, f)
    return fs


def study_files(sample: str, mask: str) -> list[tuple[int, str]]:
    """The files of `mask` whose file NUMBER is in the frozen file set (freeze_fileset), or,
    without one, finished for every mask. TTbar must be complete (the split spans all ten)."""
    import json

    if TIER == "slim":
        return slim().frozen_files(mask)
    if os.path.exists(fileset_path()):
        common = set(json.load(open(fileset_path()))[sample])
    else:
        common = set.intersection(*(set(n for n, _ in files(sample, m)) for m in MASKS))
    out = [(n, p) for n, p in files(sample, mask, require_all=(sample == "ttbar")) if n in common]
    if sample == "ttbar" and len(out) != len(SAMPLES[sample][2]):
        raise FileNotFoundError(f"ttbar: files {sorted(common)} common to all masks")
    return out


def manifest_path(mask: str) -> str:
    return os.path.join(OUT, "cache", f"manifest_{mask}.json")


def built_samples(mask: str) -> list[str]:
    """The samples of the mask's current build (its manifest): prediction files of another sample set
    (an earlier run) are stale and ignored."""
    import json

    with open(manifest_path(mask)) as f:
        return list(json.load(f)["samples"])


def split_table(sample: str, evids: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """{file number: bool test mask in the file's entry order}, from each file's (n, 3)
    (run, lumi, event) array. The choice is made on the events sorted by identity, so it
    is the same set of events in every mask whatever the entry order."""
    if TIER == "slim":                               # a hash of the identity: stable as files arrive
        return {n: slim().is_test(e) for n, e in evids.items()}
    order = {n: np.lexsort(e.T[::-1]) for n, e in evids.items()}      # entry indices, sorted by id
    test = {n: np.zeros(len(e), bool) for n, e in evids.items()}
    if sample == "ttbar":
        rng = np.random.default_rng([SPLIT_SEED, 0])
        total = sum(len(e) for e in evids.values())
        pool = [(n, i) for n in sorted(evids) if n not in TTBAR_TRAIN_ONLY for i in order[n]]
        for j in rng.permutation(len(pool))[:int(round(TEST_FRACTION * total))]:
            n, i = pool[j]
            test[n][i] = True
    else:
        for n in sorted(evids):
            k = len(evids[n])
            pick = np.random.default_rng([SPLIT_SEED, 1, n]).permutation(k)[:int(round(TEST_FRACTION * k))]
            test[n][order[n][pick]] = True
    return test


def val_table(sample: str, evids: dict[int, np.ndarray], test: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """{file number: bool validation mask in the file's entry order}: VAL_FRACTION of the sample's
    TRAINING events (not in `test`), chosen on the events sorted by identity, so it is the same set
    of events in every mask, cell and trial."""
    if TIER == "slim":
        return {n: slim().is_val(e) & ~test[n] for n, e in evids.items()}
    order = {n: np.lexsort(e.T[::-1]) for n, e in evids.items()}
    val = {n: np.zeros(len(e), bool) for n, e in evids.items()}
    pool = [(n, i) for n in sorted(evids) for i in order[n] if not test[n][i]]
    for j in np.random.default_rng([SPLIT_SEED, 2, list(SAMPLES).index(sample)]).permutation(len(pool))[
            :int(round(VAL_FRACTION * len(pool)))]:
        n, i = pool[j]
        val[n][i] = True
    return val


def event_key(sample_idx, fileno, evid):
    """int64 key of (sample, file, lumi, event) -- unique within the study. The slim tier keys by identity only
    (its job numbers do not identify the same events in another mask)."""
    evid = np.asarray(evid, np.int64)
    if TIER == "slim":
        fileno = 0
    if (evid[:, 2] >= 10**7).any() or (evid[:, 1] >= 10**6).any():
        raise ValueError("event / lumi number beyond the key range")
    return ((np.asarray(sample_idx, np.int64) * 100 + fileno) * 10**6 + evid[:, 1]) * 10**7 + evid[:, 2]


def event_filter(evid) -> np.ndarray:
    """Per (run, lumi, event): used by the study? The slim tier uses only the events present in every mask."""
    evid = np.asarray(evid, np.int64).reshape(-1, 3)
    return slim().in_common(evid) if TIER == "slim" else np.ones(len(evid), bool)


def tag(sample: str, mask: str, n: int) -> str:
    return f"{sample}_{mask}_f{n:02d}"


def cache_dir(kind: str) -> str:
    d = os.path.join(OUT, "cache", kind)
    os.makedirs(d, exist_ok=True)
    return d


def tf_setup(threads: int = 2, gpu: bool = False):
    """Keras on TensorFlow with bounded CPU threads (shared machine); the Metal GPU is
    hidden unless `gpu` (another study may be using it)."""
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    if not gpu:
        tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)


def jets_dir(variant: str | None = None) -> str:
    """The jet caches a variant trains on: the joint-head variants have their own (joint_heads.py jets)."""
    return cache_dir(f"jets_{variant}") if variant and variant.startswith("joint_") else cache_dir("jets")


def masks_with_jets(sample: str) -> list[str]:
    """Masks whose jet caches exist for every file of `sample` present on disk."""
    out = []
    for m in MASKS:
        fs = files(sample, m)
        if fs and all(os.path.exists(os.path.join(cache_dir("jets"), tag(sample, m, n) + ".npz"))
                      for n, _ in fs):
            out.append(m)
    return out

