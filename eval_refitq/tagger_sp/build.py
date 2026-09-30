"""Per-mask inputs of the jet-tagging study: track-MVA heads + per-jet tensors.

  pixi run python eval_refitq/tagger_sp/build.py --mask AAAA [--samples ttbar htt]

1. tracks  per file: the framework's per-track table of the canonical_ext family
           (ngtagger.train.trackmva.tracks_from_batch: L1TExtTrack + L1TSmartPixelsExtTrack +
           ext refit-hit aggregates, TP truth), cached as npz. Checks that
           L1TSmartPixelsExtTrack_l1TrackIdx is the row index (1:1 with L1TExtTrack).
2. mva     the five heads (clean, d0acc, z0acc, log sigma_hat d0/z0; trackmva.fit_frozen)
           fitted on the TRAINING events of every sample given (the tagger's 80%; the
           correlation with the tagger training is ignored by design). SP masks: the
           "refit" heads (SP track + refit features); IIII: the "base" (OT-only) heads -- its
           SP tracks are passthrough copies with no refit columns (read base-only). Truth =
           the OT track's TP for every head (refit_truth "otlink", user decision).
           Saved as a framework bundle (trackmva.save_bundle) under OUT/mva/<mask>/; a
           held-out check on the TEST events' tracks is written next to it.
3. jets    per file: ngtagger.data.nano.load_jets with every feature group (baseline,
           track, trkquality, vertexdxy_pv, trackres, trackmva via the bundle), constituents from
           L1SC4NGJetCands (slot < 16) -> L1ExtPuppiCand -> l1TrackIdx -> L1TSmartPixelsExtTrack,
           labels.label_jets (8 classes), features.build_features. Kept: labelled jets
           (framework `keep`); the study acceptance is applied at training time.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from ngtagger.data.features import build_features, resolve_feature_groups  # noqa: E402
from ngtagger.data.labels import label_jets  # noqa: E402
from ngtagger.data.nano import load_jets  # noqa: E402
from ngtagger.train import trackmva as tm  # noqa: E402

STEP = 25
EVENT_ID = ["run", "luminosityBlock", "event"]


def log(*a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30   # bytes on Darwin
    print(time.strftime("%H:%M:%S"), f"[{rss:.2f} GB]", *a, flush=True)


def event_ids(fn):
    with uproot.open(fn) as f:
        return np.column_stack([f["Events"][c].array(library="np").astype(np.int64) for c in EVENT_ID])


# ---------------------------------------------------------------- 1. tracks
def base_only(mask):
    return mask == C.BASELINE_MASK


def build_tracks(sample, mask, n, fn):
    out = os.path.join(C.cache_dir("tracks"), C.tag(sample, mask, n) + ".npz")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(fn):
        return out
    t = uproot.open(fn)["Events"]
    br = list(dict.fromkeys(tm.nano_branches(t.keys(), C.FAMILY, "", True, base_only=base_only(mask))
                            + [f"{C.TRACK_TABLE}_l1TrackIdx"] + EVENT_ID))
    Ts, ntrks, ids = [], [], []
    for start in range(0, t.num_entries, STEP):
        A = t.arrays(br, entry_start=start, entry_stop=min(start + STEP, t.num_entries))
        l1 = A[f"{C.TRACK_TABLE}_l1TrackIdx"]
        if not ak.all(l1 == ak.local_index(l1)):
            raise RuntimeError(f"{fn}: {C.TRACK_TABLE}_l1TrackIdx is not the row index")
        T, ntrk = tm.tracks_from_batch(A, C.FAMILY, "", True, base_only(mask))
        if not np.array_equal(ak.to_numpy(ak.num(l1)), ntrk):
            raise RuntimeError(f"{fn}: {C.TRACK_TABLE} and the OT table differ in rows per event")
        Ts.append(T)
        ntrks.append(ntrk)
        ids.append(np.column_stack([ak.to_numpy(A[c]).astype(np.int64) for c in EVENT_ID]))
    T = {k: np.concatenate([x[k] for x in Ts]) for k in Ts[0]}
    ntrk = np.concatenate(ntrks)
    np.savez_compressed(out, ntrk=ntrk, evid=np.concatenate(ids), **{"T_" + k: v for k, v in T.items()})
    log(f"tracks {C.tag(sample, mask, n)}: {len(ntrk)} ev, {len(T['b_pt'])} tracks")
    return out


def load_tracks(p):
    d = np.load(p)
    return {k[2:]: d[k] for k in d.files if k.startswith("T_")}, d["ntrk"], d["evid"]


# ---------------------------------------------------------------- 2. mva
def train_mva(mask, fileset, splits):
    """fileset: [(sample, n, path, tracks npz)]; splits: {sample: {n: test mask}}."""
    coll = "base" if base_only(mask) else "refit"
    odir = os.path.join(C.OUT, "mva", mask)
    key = sorted(os.path.basename(p) for _, _, p, _ in fileset)
    mpath = os.path.join(odir, tm.bundle_stem(C.FAMILY, "") + "_meta.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            meta = json.load(f)
        if sorted(x["basename"] for x in meta["training_files"]) == key \
                and meta.get("refit_truth") == C.REFIT_TRUTH:
            log(f"mva {mask}: bundle up to date ({mpath})")
            return odir, meta["model_sha256"]
    parts_tr, parts_te = [], []
    for sample, n, p, tp in fileset:
        T, ntrk, _ = load_tracks(tp)
        te_trk = np.repeat(splits[sample][n], ntrk)
        parts_tr.append({k: v[~te_trk] for k, v in T.items()})
        parts_te.append({k: v[te_trk] for k, v in T.items()})
    Ttr = {k: np.concatenate([x[k] for x in parts_tr]) for k in parts_tr[0]}
    Tte = {k: np.concatenate([x[k] for x in parts_te]) for k in parts_te[0]}
    del parts_tr, parts_te
    log(f"mva {mask} ({coll}): {len(Ttr['b_pt'])} train / {len(Tte['b_pt'])} test tracks")
    models, info = tm.fit_frozen(Ttr, (coll,), tm.SEED, None, C.REFIT_TRUTH, log)
    meta = {
        "what": "tagger_sp study heads: trackmva.fit_frozen on the tagger's training events",
        "spec_version": tm.SPEC_VERSION, "family": C.FAMILY, "config": "",
        "tables": tm.table_names(C.FAMILY, ""), "colls": [coll],
        "features": {coll: list(tm.features(Ttr, coll))}, "outputs": tm.OUTPUTS,
        "head_definitions": tm.HEAD_DEFINITIONS, "withheld_features": tm.WITHHELD,
        "refit_truth": C.REFIT_TRUTH, "windows": {coll: tm.window_names(coll)},
        "sigma_floor_um": tm.SIGMA_FLOOR_UM, "quantiles": list(tm.QUANTILES),
        "hgb_params": tm.HGB_PARAMS, "random_state": tm.SEED, "frozen_fit": info,
        "training_files": [{"basename": os.path.basename(p), "path": p, "size": os.path.getsize(p)}
                           for _, _, p, _ in fileset],
        "event_split": "eval_refitq/tagger_sp/common.split_table (train events only)",
        "package_versions": tm._package_versions(), "git": tm._git_state(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    mp = tm.save_bundle(odir, models, meta, C.FAMILY, "")
    # held-out check on the test events' tracks (per-track, not the tagger's metric)
    mva = tm.load_bundle(odir)
    P = mva.predict(Tte, coll)
    matched, clean, rd0, rz0 = tm.truth(Tte, coll, C.REFIT_TRUTH)
    from sklearn.metrics import roc_auc_score

    chk = {"coll": coll, "n_test_tracks": int(len(clean)),
           "auc_clean": float(roc_auc_score(clean, P["clean"])) if 0 < clean.sum() < len(clean) else None}
    m = matched.astype(bool)
    if m.sum() > 100:
        rd0u = np.abs(rd0[m])                      # um
        chk["median_sigma_hat_d0_um"] = float(np.median(np.exp(P["log_sd0"][m])))
        chk["sigma68_d0_resid_um"] = float(np.percentile(rd0u, 68.3))
        chk["coverage_1sigma_d0"] = float(np.mean(rd0u <= np.exp(P["log_sd0"][m])))
    with open(os.path.join(odir, "heldout_check.json"), "w") as f:
        json.dump(chk, f, indent=1)
    log(f"mva {mask}: {mp}; held-out {chk}")
    return odir, json.load(open(mp))["model_sha256"]


# ---------------------------------------------------------------- 3. jets
def build_jets(sample, mask, n, fn, bundle_dir, sha, test_mask):
    out = os.path.join(C.cache_dir("jets"), C.tag(sample, mask, n) + ".npz")
    if os.path.exists(out):
        d = np.load(out)
        if str(d["bundle_sha"]) == sha and [str(x) for x in d["feature_names"]] == \
                resolve_feature_groups(C.GROUPS_ALL):
            return out
    jets, const, gen = load_jets([fn], n_const=C.N_CONST, feature_groups=C.GROUPS_ALL,
                                 track_table=C.TRACK_TABLE, trackmva_model=bundle_dir)
    label, target_pt, target_pt_phys, keep = label_jets(jets, gen, max_dr=0.4)
    X, names = build_features(jets, const, n_const=C.N_CONST, feature_groups=C.GROUPS_ALL)
    nj = ak.to_numpy(ak.num(jets.pt, axis=1))
    entry = np.repeat(np.arange(len(nj)), nj)
    k = ak.to_numpy(ak.flatten(keep))
    evid = event_ids(fn)
    np.savez_compressed(
        out, X=X[k], feature_names=np.array(names), label=ak.to_numpy(ak.flatten(label))[k],
        target_pt=target_pt[k], truth_pt=ak.to_numpy(ak.flatten(target_pt_phys))[k],
        pt=ak.to_numpy(ak.flatten(jets.pt))[k], eta=ak.to_numpy(ak.flatten(jets.eta))[k],
        nconst=ak.to_numpy(ak.flatten(ak.num(const.pt, axis=2)))[k],
        entry=entry[k], is_test=test_mask[entry[k]], evid=evid, test_entries=test_mask,
        n_jets_all=int(len(k)),
        bundle_sha=np.array(sha))
    log(f"jets {C.tag(sample, mask, n)}: {int(k.sum())} labelled of {len(k)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, choices=C.MASKS)
    ap.add_argument("--samples", nargs="+", default=["ttbar", "htt"], choices=sorted(C.SAMPLES))
    a = ap.parse_args()
    fileset, splits = [], {}
    for s in a.samples:
        fs = C.files(s, a.mask, require_all=(s == "ttbar"))
        if not fs:
            raise FileNotFoundError(f"no {s} files for {a.mask}")
        ev = {n: event_ids(p) for n, p in fs}
        splits[s] = C.split_table(s, ev)
        log(f"{s} {a.mask}: {len(fs)} files, {sum(len(e) for e in ev.values())} ev, "
            f"{sum(int(v.sum()) for v in splits[s].values())} test")
        for n, p in fs:
            fileset.append((s, n, p, build_tracks(s, a.mask, n, p)))
    bundle_dir, sha = train_mva(a.mask, fileset, splits)
    for s, n, p, _ in fileset:
        build_jets(s, a.mask, n, p, bundle_dir, sha, splits[s][n])
    log("done")


if __name__ == "__main__":
    main()
