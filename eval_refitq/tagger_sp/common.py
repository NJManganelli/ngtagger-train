"""Shared definitions of the SmartPixels jet-tagging study (eval_refitq/tagger_sp).

Scenarios = canonical (coopt) nanos, build 92bad91b2ba (disc refit): the whole
downstream L1 chain (GTT vertex, PF, PUPPI, SC4 NG jets) is re-emulated from the
SmartPixels (SP) tracks of each IT active-layer mask. IIII = passthrough (SP track ==
its OT track), i.e. the OT-only baseline through the identical pipeline.

Samples (PU200): ttbar = canon_v2 spstudy_<mask>_f01..f10_100ev.root (1000 ev),
htt = canon_v2_htt spstudy_<mask>_htt_f<NN>.root (QQ -> H -> tautau RelVal; every file
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
Chosen once with SPLIT_SEED on the EVENT IDENTITY (run, lumi, event): the entry order of a
file differs between masks (multithreaded output), so entries are never used as keys.
"""
from __future__ import annotations

import os

import numpy as np

OUT = "/Volumes/WDMac/smartpixels_scratch/tagger_sp"
MASKS = ["IIII", "AAAA", "AAAI", "IAAA"]       # IIII = OT baseline (passthrough)
BASELINE_MASK = "IIII"
SAMPLES = {
    "ttbar": ("/Volumes/WDMac/smartpixels_scratch/canon_v2", "spstudy_{mask}_f{n:02d}_100ev.root",
              range(1, 11)),
    "htt": ("/Volumes/WDMac/smartpixels_scratch/canon_v2_htt", "spstudy_{mask}_htt_f{n:02d}.root",
            None),                              # None: every file number present
}
MIN_AGE_S = 180
TTBAR_TRAIN_ONLY = (1, 2)       # cluster-CNN training files
TEST_FRACTION = 0.2
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
# Feature variants (cumulative), all with the same tagger / training protocol. The FACTORISATION
# LADDER (user): L0 = "baseline" on IIII (the OT reference, NGT-like tagger); L1 = "baseline" on an
# SP mask (refit tracks only through PF/PUPPI/jets upstream); L2 = "track" (+ the constituent
# track's parameters, fit quality, resolution, IT hits); L3 = "trackmva" (+ track-MVA scores).
# Controls: L2' / L3' = "track" / "trackmva" on IIII (the same slots filled from the OT track,
# OT-only MVA heads) -> separates "more features" from "better tracks".
VARIANTS = {
    "baseline": ["baseline"],
    "track": ["baseline", "track", "trkquality", "vertexdxy_pv", "trackres"],
    "trackmva": ["baseline", "track", "trkquality", "vertexdxy_pv", "trackres", "trackmva"],
}
LEVEL = {"baseline": 1, "track": 2, "trackmva": 3}

def path(sample: str, mask: str, n: int) -> str:
    d, pat, _ = SAMPLES[sample]
    return os.path.join(d, pat.format(mask=mask, n=n))


def files(sample: str, mask: str, require_all: bool = False) -> list[tuple[int, str]]:
    """(file number, path) of the finished files present (no *_rerun; older than MIN_AGE_S)."""
    import glob
    import re
    import time

    d, pat, nums = SAMPLES[sample]
    if nums is None:
        rx = re.compile(re.escape(pat.format(mask=mask, n=0)).replace("00", r"(\d+)", 1) + "$")
        nums = sorted(int(m.group(1)) for f in glob.glob(os.path.join(d, f"spstudy_{mask}_*.root"))
                      if (m := rx.match(os.path.basename(f))))
    out = [(n, path(sample, mask, n)) for n in nums if os.path.exists(path(sample, mask, n))
           and time.time() - os.path.getmtime(path(sample, mask, n)) > MIN_AGE_S]
    if require_all and len(out) != len(nums):
        raise FileNotFoundError(f"{sample} {mask}: {len(out)}/{len(nums)} files present")
    return out


def study_files(sample: str, mask: str) -> list[tuple[int, str]]:
    """The files of `mask` whose file NUMBER is finished for every mask (the study's
    common event set). TTbar must be complete (the split spans all ten files)."""
    common = set.intersection(*(set(n for n, _ in files(sample, m)) for m in MASKS))
    out = [(n, p) for n, p in files(sample, mask, require_all=(sample == "ttbar")) if n in common]
    if sample == "ttbar" and len(out) != len(SAMPLES[sample][2]):
        raise FileNotFoundError(f"ttbar: files {sorted(common)} common to all masks")
    return out


def manifest_path(mask: str) -> str:
    return os.path.join(OUT, "cache", f"manifest_{mask}.json")


def split_table(sample: str, evids: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """{file number: bool test mask in the file's entry order}, from each file's (n, 3)
    (run, lumi, event) array. The choice is made on the events sorted by identity, so it
    is the same set of events in every mask whatever the entry order."""
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


def event_key(sample_idx, fileno, evid):
    """int64 key of (sample, file, lumi, event) -- unique within the study."""
    evid = np.asarray(evid, np.int64)
    if (evid[:, 2] >= 10**7).any() or (evid[:, 1] >= 10**6).any():
        raise ValueError("event / lumi number beyond the key range")
    return ((np.asarray(sample_idx, np.int64) * 100 + fileno) * 10**6 + evid[:, 1]) * 10**7 + evid[:, 2]


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


def masks_with_jets(sample: str) -> list[str]:
    """Masks whose jet caches exist for every file of `sample` present on disk."""
    out = []
    for m in MASKS:
        fs = files(sample, m)
        if fs and all(os.path.exists(os.path.join(cache_dir("jets"), tag(sample, m, n) + ".npz"))
                      for n, _ in fs):
            out.append(m)
    return out

