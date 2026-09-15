"""Per-TrackingParticle findability census, built in event chunks and cached.

WHAT THIS PRODUCES. One entry for EVERY TrackingParticle that left at least one
SmartPixels cluster or OT stub -- 9.52 million of them in the 1000-event PU200
ttbar set, 9,515 per event -- carrying

    key            (event << 20) | tpIdx, globally unique across the input set
    pt eta phi     TP truth kinematics
    d0 z0 vr       transverse impact parameter, longitudinal vertex, production radius
    hit_it hit_ot  bitmasks of which IL1-IL4 / OL1-OL6 the TP actually hit
    found          BITMAP over every candidate seed: did that seed recover this TP

so that efficiency in any slice of (pT, eta, phi, d0, z0, build, seed menu) is a
masked popcount over columns that are already in memory, with no re-reading of
the nano and no re-running of the seeding.

WHY CHUNKING IS EXACT HERE, not an approximation. Every stage of the seeding is
INTRA-EVENT: cluster pairing folds the event into the sort key, and projection
searches an index built per event. A TrackingParticle belongs to exactly one
event, so partitioning events across chunks partitions TPs across chunks with no
overlap and no boundary term. Per-chunk censuses therefore CONCATENATE -- there
is no merge-by-key, no double-counting, and the result is bit-identical to a
single monolithic pass. What chunking buys is that the ~1e6 candidate triplets
per event per seed exist for one chunk at a time instead of all thousand events
at once, which is what made the 1000-event run impossible before.

TWO THINGS MUST BE FROZEN ACROSS CHUNKS or the chunks stop being comparable:
the per-layer OT stub z resolution (measured, used as the projection target
sigma) and the per-layer median radius (which picks each doublet's adjacent
target layers). Both are measured ONCE on a calibration pass and written into
the cache manifest, so every chunk and every later run sees the same numbers.

KINEMATICS ARE MISSING FOR OT-ONLY TPs -- 1.87M of the 9.52M. The nano stores
tpEta/tpPhi/tpVx/tpVy/tpVz on the SmartPixels cluster table only; the OT stub
table carries tpIdx and tpPt and nothing else. A TP with stubs but no IT cluster
therefore has pT and hit pattern but no direction or vertex, and its kinematic
columns are NaN with has_kin = False. That is a producer limitation, not a
choice made here: adding those five branches to the OT stub table would close it.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import awkward as ak
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402
import kf_emulation as KF           # noqa: E402
import seed_arity as SA             # noqa: E402

NAME = {1: "IL1", 2: "IL2", 3: "IL3", 4: "IL4",
        11: "OL1", 12: "OL2", 13: "OL3", 14: "OL4", 15: "OL5", 16: "OL6"}
CODE = {v: k for k, v in NAME.items()}
ALL_LAYERS = (1, 2, 3, 4, 11, 12, 13, 14, 15, 16)
OT_COLS = ["layer", "isBarrel", "r", "phi", "z", "tpIdx", "tpPt"]

# Bumped whenever a change alters the NUMBERS in a shard. The cache key folds it
# in, so a stale cache cannot be silently reused across a semantic change.
FORMAT_VERSION = 12
# Residuals kept per seed PER CHUNK for the robust spreads. Fixed per chunk, not
# derived from the requested event count, so the same shard serves a 100-event
# and a 1000-event request identically.
QUAL_PER_CHUNK = 400


# ==========================================================================
# cache identity
# ==========================================================================
def input_manifest(spec, hash_content=False):
    """Identify the inputs: path, size, mtime, and optionally a content digest.

    Size and mtime are the default because the ttbar set is 1.7 GB of ROOT per
    file and digesting ten of them costs ~30 s of pure I/O on every invocation,
    which defeats the point of a cache. --hash-content is there for when a file
    may have been rewritten in place with the same length.
    """
    out = []
    for p in M.expand_inputs(spec):
        st = os.stat(p)
        rec = {"path": os.path.abspath(p), "size": st.st_size,
               "mtime_ns": st.st_mtime_ns}
        if hash_content:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(blk)
            rec["sha256"] = h.hexdigest()
            rec.pop("mtime_ns")          # content digest supersedes mtime
        out.append(rec)
    return out


def cache_key(manifest, cfg):
    payload = json.dumps({"format": FORMAT_VERSION, "inputs": manifest,
                          "config": cfg}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# ==========================================================================
# frozen calibration
# ==========================================================================
def calibrate(spec, nev, ptmin):
    """Per-layer OT stub z resolution and per-layer median radius, measured once.

    The stub z resolution is the spread of (stub z) - (z0 + r*sinh(eta)) taken
    from the TP the stub belongs to, so it is the real pointing resolution the
    projection has to open its window to, not a nominal strip pitch.
    """
    I, nI, _ = M.load_flat(spec, M.IT_TABLE, list(M.IT_COLS), nev)
    O, _, _ = M.load_ot(spec, nev)
    bar = (O["isBarrel"] > 0) & (O["eta"] <= M.ETA_MATCHED)
    mi = I["tpIdx"] >= 0
    k = M.tp_key(I["event"][mi], I["tpIdx"][mi])
    o = np.argsort(k, kind="stable")
    KV, VZ, ET = k[o], I["tpVz"][mi][o], I["tpEta"][mi][o]
    f = np.r_[True, KV[1:] != KV[:-1]] if len(KV) else np.zeros(0, bool)
    KV, VZ, ET = KV[f], VZ[f], ET[f]

    def rs(x):
        q = np.percentile(x, [15.865, 84.135])
        return float(0.5 * (q[1] - q[0]))

    sigz = {}
    for L in range(1, 7):
        m = bar & (O["layer"] == L)
        mt = m & (O["tpIdx"] >= 0) & (O["tpPt"] >= ptmin)
        if mt.sum() < 200 or not len(KV):
            sigz[L] = 0.2
            continue
        kk = M.tp_key(O["event"][mt], O["tpIdx"][mt])
        p = np.clip(np.searchsorted(KV, kk), 0, len(KV) - 1)
        g = KV[p] == kk
        sigz[L] = rs(O["z"][mt][g] - (VZ[p[g]] + O["r"][mt][g] * np.sinh(ET[p[g]]))) \
            if g.sum() >= 200 else 0.2
    rmed = {}
    for L in ALL_LAYERS:
        s = (I["layer"] == L) if L <= 4 else (bar & (O["layer"] == L - 10))
        r = (I["globalR"] if L <= 4 else O["r"])[s]
        rmed[L] = float(np.median(r)) if len(r) else 0.0
    return {"sigz_ot": sigz, "r_median": rmed, "calib_events": nI}


BUILD_MASKS = ["AAAA", "AAAI", "AAIA", "AIAA", "IAAA", "AAII", "AIAI",
               "AIIA", "IAAI", "IAIA", "IIAA", "AIII", "IAII", "IIAI", "IIIA"]
IL_OF = {0: 1, 1: 2, 2: 3, 3: 4}
OT_BARREL = (11, 12, 13, 14, 15, 16)


def seed_universe_over_builds(rmed, masks):
    """Union over builds of each build's own doublet and triplet seeds.

    Which seeds exist depends on which IT layers are instrumented, but a seed's
    RESULT does not, so every distinct seed is run once and each build composes
    its menu from that. Enumeration and the locality rule live in seed_arity.
    """
    seen, out = {}, []
    for mask in masks:
        il = [IL_OF[i] for i, ch in enumerate(mask) if ch == "A"]
        for sd in SA.enumerate_seeds(il):
            if sd.layers not in seen:
                seen[sd.layers] = sd
                out.append(sd)
    return sorted(out, key=lambda sd: (sd.arity,) + tuple(rmed.get(L, 0.0)
                                                          for L in sd.layers))


def seed_tag(s):
    return s.tag if hasattr(s, "tag") else SA.Seed(s).tag


# ==========================================================================
# chunked unified IT+OT table
# ==========================================================================
def unified_chunks(spec, nev, step, sigz_ot):
    """Yield one unified IT+OT hit table per chunk of `step` events.

    Event numbering carries a running offset across chunks AND across files, so
    (event, tpIdx) stays globally unique and chunk censuses concatenate.
    """
    srcs = [f"{p}:Events" for p in M.expand_inputs(spec)]
    itk = [f"{M.IT_TABLE}_{c}" for c in M.IT_COLS]
    otk = [f"{M.OT_TABLE}_{c}" for c in OT_COLS]
    seen = 0
    for A in uproot.iterate(srcs, itk + otk, step_size=step):
        # uproot.iterate IGNORES entry_stop when handed a list of files, so the
        # event limit has to be enforced here or -n 8 reads all 1000.
        if nev is not None:
            if seen >= nev:
                return
            if len(A) > nev - seen:
                A = A[:nev - seen]
        nI = ak.to_numpy(ak.num(A[itk[0]]))
        nO = ak.to_numpy(ak.num(A[otk[0]]))
        ev = np.arange(seen, seen + len(nI))
        I = {c: ak.to_numpy(ak.flatten(A[f"{M.IT_TABLE}_{c}"])) for c in M.IT_COLS}
        I["event"] = np.repeat(ev, nI)
        O = {c: ak.to_numpy(ak.flatten(A[f"{M.OT_TABLE}_{c}"])) for c in OT_COLS}
        O["event"] = np.repeat(ev, nO)
        seen += len(nI)
        yield _unify(I, O, sigz_ot), ev
        del A, I, O


def _unify(I, O, sigz_ot):
    eta = np.abs(np.arcsinh(O["z"] / np.maximum(O["r"], 1e-6)))
    bar = (O["isBarrel"] > 0) & (eta <= M.ETA_MATCHED)
    n = int(bar.sum())
    sig = np.empty(n)
    ol = O["layer"][bar]
    for L in range(1, 7):
        sig[ol == L] = sigz_ot.get(L, sigz_ot.get(str(L), 0.2))
    U = {"layer": np.r_[I["layer"], ol + 10],
         "globalR": np.r_[I["globalR"], O["r"][bar]],
         "globalZ": np.r_[I["globalZ"], O["z"][bar]],
         "globalPhi": np.r_[I["globalPhi"], O["phi"][bar]],
         "sigY": np.r_[I["sigY"], sig],
         # an OT stub has no r-phi CPE sigma of its own; ot_variances builds its
         # variance from pitch and scattering instead, so this is never read there
         "sigX": np.r_[I["sigX"], sig],
         "globalClusterCotTheta": np.r_[I["globalClusterCotTheta"], np.zeros(n)],
         "sigGlobalClusterCotTheta": np.r_[I["sigGlobalClusterCotTheta"],
                                           np.full(n, 1e9)],
         "tpIdx": np.r_[I["tpIdx"], O["tpIdx"][bar]],
         "tpPt": np.r_[I["tpPt"], O["tpPt"][bar]],
         "event": np.r_[I["event"], O["event"][bar]]}
    for c in ("tpVx", "tpVy", "tpPhi", "tpEta", "tpVz"):
        U[c] = np.r_[I[c], np.full(n, np.nan)]
    QI = M.it_prepare({c: I[c] for c in M.IT_COLS}, None)
    # An OT stub carries no alpha/beta. Huge sigmas make every angle gate pass,
    # and the joint (z0,phi) pairing then treats the stub as a phi-only wildcard.
    Q = {"kap_a": np.r_[QI["kap_a"], np.zeros(n)],
         "s_kap": np.r_[QI["s_kap"], np.full(n, 1e9)],
         "z0": np.r_[QI["z0"], np.zeros(n)],
         "s_z0": np.r_[QI["s_z0"], np.full(n, 1e9)],
         "ovf_a": np.r_[QI["ovf_a"], np.zeros(n, bool)],
         "ovf_z": np.r_[QI["ovf_z"], np.zeros(n, bool)]}
    return U, Q


# ==========================================================================
# per-chunk census
# ==========================================================================
def chunk_census(U):
    """One entry per TP with >= 1 hit in this chunk, plus its hit pattern.

    The IT row wins when a TP has both, because tpEta/tpPhi/tpV* live only on the
    cluster table and an OT stub row carries NaN for all of them.
    """
    m = U["tpIdx"] >= 0
    if not m.any():
        z = np.zeros(0)
        return {"key": z.astype(np.int64), "event": z.astype(np.int32),
                "pt": z.astype(np.float32), "eta": z.astype(np.float32),
                "phi": z.astype(np.float32), "d0": z.astype(np.float32),
                "z0": z.astype(np.float32), "vr": z.astype(np.float32),
                "hit_it": z.astype(np.uint8), "hit_ot": z.astype(np.uint8)}
    gi = np.flatnonzero(m)
    key = M.tp_key(U["event"][gi], U["tpIdx"][gi])
    lay = U["layer"][gi]
    is_ot = (lay > 4).astype(np.int8)
    order = np.lexsort((is_ot, key))          # IT rows first within a key
    key_s, gi_s = key[order], gi[order]
    first = np.r_[True, key_s[1:] != key_s[:-1]]
    uk = key_s[first]
    rep = gi_s[first]
    pos = np.cumsum(first) - 1                # entry index of every hit
    hit_it = np.zeros(len(uk), np.uint8)
    hit_ot = np.zeros(len(uk), np.uint8)
    ls = lay[order]
    it = ls <= 4
    np.bitwise_or.at(hit_it, pos[it], (1 << (ls[it] - 1)).astype(np.uint8))
    np.bitwise_or.at(hit_ot, pos[~it], (1 << (ls[~it] - 11)).astype(np.uint8))
    vx, vy, ph = U["tpVx"][rep], U["tpVy"][rep], U["tpPhi"][rep]
    return {"key": uk.astype(np.int64),
            "event": (uk >> M.TP_KEY_SHIFT).astype(np.int32),
            "pt": U["tpPt"][rep].astype(np.float32),
            "eta": U["tpEta"][rep].astype(np.float32),
            "phi": ph.astype(np.float32),
            "d0": (-vx * np.sin(ph) + vy * np.cos(ph)).astype(np.float32),
            "z0": U["tpVz"][rep].astype(np.float32),
            "vr": np.hypot(vx, vy).astype(np.float32),
            "hit_it": hit_it, "hit_ot": hit_ot}


def process_chunk(U, Q, seeds, ptmin, rng, qual_per_chunk, kf_opts=None,
                  feat_per_chunk=0, do_dr=True):
    """Run every seed over one chunk and reduce to census + bitmap + counters.

    TWO PASSES, because duplicate removal is inherently cross-seed. The first
    pass runs each seed through pairing, projection, the 4-layer rule and the KF
    chi2 acceptance and holds its survivors. DR then runs ONCE over all of them
    together -- several seeds finding the same track is exactly what it exists
    to collapse -- and only then is each seed credited with what survived.

    Quality and MVA rows are taken after DR too, so the resolutions and the
    training sample describe the tracks a real system would actually emit.
    """
    kf_opts = kf_opts or {}
    d0_opt = {k: v for k, v in kf_opts.items() if k == "d0_prior_cm"}
    cen = chunk_census(U)
    ntp, nw = len(cen["key"]), (len(seeds) + 63) // 64
    # TWO BITMAPS, and the distinction is not bookkeeping.
    #   found    = what a seed reaches ON ITS OWN, before DR.
    #   found_dr = what it still owns after DR.
    # DR assigns each real track to exactly ONE seed, so found_dr destroys
    # precisely the overlap information the greedy menu composition needs:
    # the greedy has to know that seed B would have found a TP if seed A were
    # not in the menu, and post-DR that is unknowable. Composition therefore
    # uses `found`, while track rates, resolutions and MVA rows use the post-DR
    # survivors, which is what a real system emits.
    found = np.zeros((ntp, nw), np.uint64)
    found_dr = np.zeros((ntp, nw), np.uint64)
    targets = list(SA.IL) + list(SA.OT_BARREL)
    counters, qual, feats, labels = {}, {}, {}, {}

    # ---- pass 1: seed, follow, fit, chi2-accept --------------------------
    held = {}
    for s_i, sd in enumerate(seeds):
        try:
            o = SA.run_seed(U, Q, sd, ptmin, targets)
        except M.TooWide as e:
            counters[s_i] = {"infeasible": 1.0, "n": float(e.n)}
            continue
        if o is None or not o.get("tracks_to_fit"):
            counters[s_i] = {"pairs": 0.0, "cand": 0.0, "fit": 0.0}
            continue
        n_pre = int(o["tracks_to_fit"])
        gidx_all = KF.hits_from_seed(U, o)
        gc0 = o.get("_gC")
        trip0 = (o["_gA"], o["_gB"], gc0 if gc0 is not None else o["_gB"])
        fit0 = KF.fit_tracks(U, Q, trip0, gidx=gidx_all, use_angles=False,
                             **d0_opt)
        keep, chi2s = KF.good_state(fit0, ptmin)
        base = {"arity": float(sd.arity), "before_chi2": float(n_pre),
                "seed_objects": float(o.get("seed_objects", 0)),
                "pairs": float(o.get("tracklets", 0)),
                "cand": float(o.get("cand_all_targets", 0)),
                "before_minlayers": float(o.get("tracks_before_minlayers", 0)),
                "fit": 0.0}
        if not keep.any():
            counters[s_i] = base
            continue
        w_ang = 0.30 if sd.arity == 2 else 0.03
        # standalone reach, recorded before DR can reassign anything
        o_keep = SA.filter_tracks(o, keep)
        rk0 = SA.recovered_keys(U, o_keep)
        if len(rk0):
            p0 = np.searchsorted(cen["key"], rk0)
            found[p0, s_i >> 6] |= np.uint64(1) << np.uint64(s_i & 63)
        base["found_standalone"] = float(len(rk0))
        held[s_i] = {"o": o_keep, "sd": sd, "base": base,
                     "gidx": gidx_all[keep],
                     "score": KF.rank_score(fit0, chi2s, w_angle=w_ang)[keep],
                     "chi2s": chi2s[keep],
                     "c2a": fit0["chi2_angle"][keep],
                     "nang": fit0["n_angle"][keep]}
        del o, fit0

    # ---- duplicate removal, once, across every seed ---------------------
    keys = sorted(held)
    if keys and do_dr:
        G = np.concatenate([held[i]["gidx"] for i in keys])
        SC = np.concatenate([held[i]["score"] for i in keys])
        surv = SA.duplicate_removal(G, SC)
        off = 0
        for i in keys:
            n = len(held[i]["score"])
            held[i]["dr"] = surv[off:off + n]
            off += n
    else:
        for i in keys:
            held[i]["dr"] = np.ones(len(held[i]["score"]), bool)

    # ---- pass 2: credit each seed with what survived --------------------
    TP = KF.tp_truth_table(U)
    for s_i in keys:
        h = held[s_i]
        sd, dr = h["sd"], h["dr"]
        c = dict(h["base"])
        c["before_dr"] = float(len(dr))
        if not dr.any():
            counters[s_i] = c
            continue
        o = SA.filter_tracks(h["o"], dr)
        gidx = h["gidx"][dr]
        ga, gb, gc = o["_gA"], o["_gB"], o.get("_gC")
        ta = U["tpIdx"][ga]
        real = (ta >= 0) & (ta == U["tpIdx"][gb])
        if gc is not None:
            real &= ta == U["tpIdx"][gc]
        c.update({
            "fit": float(o["tracks_to_fit"]),
            "chi2_scaled_sum": float(h["chi2s"][dr].sum()),
            "chi2_angle_sum": float(h["c2a"][dr].sum()),
            "n_angle_sum": float(h["nang"][dr].sum()),
            "rank_sum": float(h["score"][dr].sum()),
            "nlayer_sum": float((sd.arity + o["_nconf"]).sum()),
            # NOT comparable across arity: a doublet is credited when its two
            # seed clusters share a TP, a triplet when all three do, and a
            # 2-cluster coincidence is far likelier.
            "trip": float(len(ta)), "trip_true": float(real.sum())})
        counters[s_i] = c
        rkeys = SA.recovered_keys(U, o)
        if len(rkeys):
            p = np.searchsorted(cen["key"], rkeys)
            if p.max() >= ntp or not np.array_equal(cen["key"][p], rkeys):
                raise SystemExit("recovered TP absent from the chunk census")
            found_dr[p, s_i >> 6] |= np.uint64(1) << np.uint64(s_i & 63)
        trip = (ga, gb, gc if gc is not None else gb)
        # ---- resolutions, on truth-matched survivors --------------------
        idx = np.flatnonzero(real)
        if len(idx):
            if len(idx) > qual_per_chunk:
                idx = idx[rng.choice(len(idx), qual_per_chunk, replace=False)]
            tq = tuple(x[idx] for x in trip)
            fq = KF.fit_tracks(U, Q, tq, gidx=gidx[idx], use_angles=False,
                               **d0_opt)
            dk, dd, dc, dz, rm = KF.truth_residuals(U, tq, fq, TP)
            qual[s_i] = (np.stack([dk, dd, dc, dz,
                                   fq["nhit"][rm].astype(np.float64)],
                                  axis=1).astype(np.float32)
                         if len(dk) else np.zeros((0, 5), np.float32))
        # ---- MVA rows, on every survivor whether real or not ------------
        if feat_per_chunk:
            ks = np.arange(o["tracks_to_fit"])
            if len(ks) > feat_per_chunk:
                ks = rng.choice(len(ks), feat_per_chunk, replace=False)
            tf = tuple(x[ks] for x in trip)
            Tf = KF.gather_hits(U, Q, gidx[ks], KF.LAYER_ORDER)
            ff = KF.fit_tracks(U, Q, tf, gidx=gidx[ks], use_angles=False,
                               **d0_opt)
            ff["_arity"] = sd.arity
            feats[s_i] = KF.track_features(U, Q, Tf, ff)
            labels[s_i] = KF.track_labels(U, Tf, ff, tf, TP)
    return cen, found, found_dr, counters, qual, feats, labels


def _shard_path(d, i):
    return os.path.join(d, f"chunk_{i:05d}.npz")


def build(spec, nev, ptmin, layers, n_adjacent, chunk, cache_dir, mode,
          hash_content=False, calib_events=100, verbose=True,
          budget_gb=0.4, rss_gb=8.0, masks=None, kf_opts=None,
          feat_per_chunk=0):
    """Build (or load) the census. Returns a dict of concatenated arrays.

    The cache is a DIRECTORY OF PER-CHUNK SHARDS, not one file, so a run that
    dies or is killed at chunk 40 of 63 resumes at 41 instead of starting over.
    A thousand events of ninety seeds is hours of work; it must be restartable.
    That also makes the RSS ceiling survivable: ru_maxrss is a high-water mark
    and never falls, so one runaway seed aborts the job permanently -- but every
    shard already written stays valid and the rerun resumes past it.
    """
    man = input_manifest(spec, hash_content)
    # nev is deliberately NOT part of the identity. Shards are per-chunk and
    # event order is fixed, so a 100-event cache is the first seven chunks of the
    # 1000-event one: asking for more RESUMES rather than rebuilding, and asking
    # for less reads a prefix.
    # kf_opts IS part of the identity. It changes the quality columns in every
    # shard, so a cache built with one angle weighting or d0 prior must not be
    # silently extended by a run using another.
    cfg = {"ptmin": ptmin, "layers": sorted(layers), "n_adjacent": n_adjacent,
           "chunk": chunk, "calib_events": calib_events, "masks": masks,
           "kf_opts": dict(sorted((kf_opts or {}).items())),
           "feat_per_chunk": feat_per_chunk}
    kh = cache_key(man, cfg)
    d = os.path.join(cache_dir, f"tpcensus_{kh}")
    mpath = os.path.join(d, "manifest.json")
    if mode == "rebuild" and os.path.isdir(d):
        for f in sorted(Path(d).glob("chunk_*.npz")):
            f.unlink()
    meta = None
    if mode != "off" and os.path.exists(mpath):
        meta = json.load(open(mpath))
        if meta.get("cache_key") != kh:
            meta = None
    if meta is None:
        if mode == "require":
            raise SystemExit(f"no cache for key {kh} under {cache_dir}")
        os.makedirs(d, exist_ok=True)
        cal = calibrate(spec, calib_events, ptmin)
        seeds = (seed_universe_over_builds(cal["r_median"], masks)
                 if masks else SA.enumerate_seeds(list(SA.IL)))
        meta = {"cache_key": kh, "format": FORMAT_VERSION, "inputs": man,
                "config": cfg, "calibration": cal,
                "seeds": [list(sd.layers) for sd in seeds],
                "seed_notes": [sd.note for sd in seeds],
                "seed_tags": [sd.tag for sd in seeds],
                "chunks_done": 0, "n_events": 0}
        json.dump(meta, open(mpath, "w"), indent=1, default=float)
    seeds = [SA.Seed(tuple(x)) for x in meta["seeds"]]
    sigz = {int(k): float(v) for k, v in meta["calibration"]["sigz_ot"].items()}
    if verbose:
        print(f"cache {d}\n  {len(seeds)} seeds, chunk = {chunk} events, "
              f"{meta['chunks_done']} chunk(s) already done", flush=True)

    rng = np.random.default_rng(12345)
    done = meta["chunks_done"]
    nev_seen = meta["n_events"]
    need = None if nev is None else max(0, nev - nev_seen)
    if mode != "off":
        t0 = time.time()
        M.set_limits(M.triplets_for_budget(budget_gb),
                     min(rss_gb, M.SAFE_RSS_FRAC * M.PHYS_RAM_GB))
        want = None if nev is None else nev
        for ci, ((U, Q), ev) in enumerate(unified_chunks(spec, want, chunk, sigz)):
            if ci < done:
                del U, Q                  # resume re-reads, but never re-seeds
                continue
            cen, found, found_dr, counters, qual, feats, labels = process_chunk(
                U, Q, seeds, ptmin, rng, QUAL_PER_CHUNK, kf_opts,
                feat_per_chunk)
            np.savez_compressed(
                _shard_path(d, ci), found=found, found_dr=found_dr,
                n_events=len(ev),
                counters=json.dumps({str(k): v for k, v in counters.items()}),
                **{f"cen_{k}": v for k, v in cen.items()},
                **{f"qual_{k}": v for k, v in qual.items()},
                **{f"feat_{k}": v for k, v in feats.items()},
                **{f"lab_{k}": v for k, v in labels.items()})
            done, nev_seen = ci + 1, nev_seen + len(ev)
            meta["chunks_done"], meta["n_events"] = done, nev_seen
            json.dump(meta, open(mpath, "w"), indent=1, default=float)
            if verbose:
                el = time.time() - t0
                print(f"  chunk {ci:>4}  {len(ev):>3} ev  {len(cen['key']):>8,} TPs"
                      f"  {el:7.1f}s  peak {M.peak_gb():.2f} GB", flush=True)
            del U, Q, cen, found
    return load(d, nev=nev, verbose=verbose)


def load(d, nev=None, verbose=True):
    meta = json.load(open(os.path.join(d, "manifest.json")))
    shards = sorted(Path(d).glob("chunk_*.npz"))
    if not shards:
        raise SystemExit(f"{d} has no shards")
    cols = ["key", "event", "pt", "eta", "phi", "d0", "z0", "vr", "hit_it", "hit_ot"]
    parts = {c: [] for c in cols}
    fnd, fdr, cnt, qs, fts, lbs, nread = [], [], {}, {}, {}, {}, 0
    for sh in shards:
        if nev is not None and nread >= nev:
            break                    # a prefix of the shards is a valid census
        z = np.load(sh, allow_pickle=False)
        for c in cols:
            parts[c].append(z[f"cen_{c}"])
        fnd.append(z["found"])
        fdr.append(z["found_dr"])
        nread += int(z["n_events"])
        for k, v in json.loads(str(z["counters"])).items():
            a = cnt.setdefault(int(k), {})
            for kk, vv in v.items():
                a[kk] = a.get(kk, 0.0) + vv
        for f in z.files:
            if f.startswith("qual_"):
                qs.setdefault(int(f[5:]), []).append(z[f])
            elif f.startswith("feat_"):
                fts.setdefault(int(f[5:]), []).append(z[f])
            elif f.startswith("lab_"):
                lbs.setdefault(int(f[4:]), []).append(z[f])
    C = {c: np.concatenate(parts[c]) for c in cols}
    C["found"] = np.concatenate(fnd, axis=0)
    C["found_dr"] = np.concatenate(fdr, axis=0)
    C["n_events"] = nread
    C["seeds"] = [SA.Seed(tuple(x)) for x in meta["seeds"]]
    C["seed_tags"] = meta["seed_tags"]
    C["seed_notes"] = meta.get("seed_notes", [""] * len(meta["seeds"]))
    C["calibration"] = meta["calibration"]
    C["counters"] = {i: cnt.get(i, {}) for i in range(len(C["seeds"]))}
    C["qual"] = {i: (np.concatenate(v) if v else np.zeros((0, 5), np.float32))
                 for i, v in qs.items()}
    C["feat"] = {i: np.concatenate(v) for i, v in fts.items()}
    C["labels"] = {i: np.concatenate(v) for i, v in lbs.items()}
    C["feature_names"] = list(KF.FEATURE_NAMES)
    C["label_names"] = list(KF.LABEL_NAMES)
    C["cache_dir"] = d
    if verbose:
        print(f"loaded {len(fnd)} shard(s): {nread} events, "
              f"{len(C['key']):,} TPs, {len(C['seeds'])} seeds, "
              f"{C['found'].nbytes / 1e6:.1f} MB bitmap", flush=True)
    return C


# ==========================================================================
# query helpers -- what the interactive tool is built on
# ==========================================================================
def seed_mask(C, i, after_dr=False):
    """Boolean over TPs: did seed i recover this TrackingParticle?

    after_dr=False is the seed's STANDALONE reach and is what menu composition
    must use; after_dr=True is what it still owns once duplicate removal has
    assigned each track to a single seed.
    """
    b = C["found_dr"] if after_dr else C["found"]
    return (b[:, i >> 6] >> np.uint64(i & 63)) & np.uint64(1) != 0


def union_mask(C, idxs):
    out = np.zeros(len(C["key"]), bool)
    for i in idxs:
        out |= seed_mask(C, i)
    return out


def robust_sigma(x):
    if len(x) < 50:
        return float("nan")
    q = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q[1] - q[0]))


def seed_table(C):
    """Per seed: TPs found, cost per event, fake fraction, seed-level resolutions."""
    nev = max(C["n_events"], 1)
    rows = []
    for i, tag in enumerate(C["seed_tags"]):
        sd = C["seeds"][i]
        c = C["counters"].get(i, {})
        if c.get("infeasible"):
            rows.append({"seed": tag, "arity": sd.arity, "infeasible": True})
            continue
        q = C["qual"].get(i, np.zeros((0, 5), np.float32))
        rows.append({
            "seed": tag, "arity": sd.arity, "note": C["seed_notes"][i],
            "n_found": int(seed_mask(C, i).sum()),
            "n_found_dr": int(seed_mask(C, i, after_dr=True).sum()),
            "min_proj": sd.min_proj(),
            "nlayer": c.get("nlayer_sum", 0.0) / max(c.get("fit", 1.0), 1.0),
            "pass_frac": c.get("fit", 0.0) / max(c.get("before_minlayers", 1.0), 1.0),
            "chi2_pass": c.get("before_dr", 0.0) / max(c.get("before_chi2", 1.0), 1.0),
            "dr_pass": c.get("fit", 0.0) / max(c.get("before_dr", 1.0), 1.0),
            "chi2_per_layer": c.get("chi2_scaled_sum", 0.0) / max(c.get("fit", 1.0), 1.0),
            "chi2_angle_per_cl": (c.get("chi2_angle_sum", 0.0)
                                  / max(c.get("n_angle_sum", 1.0), 1.0)),
            "pairs": c.get("pairs", 0.0) / nev, "cand": c.get("cand", 0.0) / nev,
            "fit": c.get("fit", 0.0) / nev,
            "fake": 1.0 - c.get("trip_true", 0.0) / max(c.get("trip", 0.0), 1.0),
            "sig_kappa": robust_sigma(q[:, 0]), "sig_d0_cm": robust_sigma(q[:, 1]),
            "sig_cot": robust_sigma(q[:, 2]), "sig_z0_cm": robust_sigma(q[:, 3]),
            "n_fit": int(len(q)),
            "mean_nhit": float(q[:, 4].mean()) if len(q) else float("nan")})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=None)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--layers", default=",".join(NAME[L] for L in ALL_LAYERS))
    ap.add_argument("--n-adjacent", type=int, default=2)
    ap.add_argument("--masks", default=",".join(BUILD_MASKS),
                    help="activeSP builds to enumerate seeds for; 'none' for a "
                         "flat enumeration over --layers")
    # SMALLER CHUNKS ARE FASTER HERE, which is backwards from the usual
    # amortisation argument and so is worth stating. MEASURED on the same 16
    # events, same 131 seeds, identical input (127,282 TPs both times, so this is
    # not process variance): chunk = 4 took 444 s, chunk = 16 took 1006 s -- 27.8
    # against 62.9 s/event, a 2.27x penalty for the larger chunk. Chunk size is
    # not the memory control either (see --menu-budget-gb): the blow-up is the
    # candidate triplets of a SINGLE event and no chunk size reaches below one.
    ap.add_argument("--chunk", type=int, default=4, help="events per chunk")
    ap.add_argument("--calib-events", type=int, default=100)
    ap.add_argument("--cache-dir", default="eval_refitq/combinatorics/cache")
    ap.add_argument("--cache", default="auto",
                    choices=["auto", "rebuild", "off", "require"])
    ap.add_argument("--features-per-chunk", type=int, default=0,
                    help="export this many per-track MVA training rows per seed "
                         "per chunk; 0 disables")
    ap.add_argument("--d0-prior-cm", type=float, default=KF.D0_PRIOR_CM)
    ap.add_argument("--kf-angles", default="on", choices=["on", "off"],
                    help="use the SmartPixels alpha/beta in the KF updates")
    ap.add_argument("--alpha-scale", type=float, default=1.0,
                    help="scale on the alpha sigma; <1 trusts it more")
    ap.add_argument("--beta-scale", type=float, default=1.0)
    ap.add_argument("--hash-content", action="store_true",
                    help="digest file contents instead of trusting size+mtime")
    ap.add_argument("-o", "--out", default=None, help="write the seed table as JSON")
    a = ap.parse_args()
    layers = [CODE[x.strip()] for x in a.layers.split(",") if x.strip()]
    masks = None if a.masks.strip().lower() == "none" else \
        [m.strip() for m in a.masks.split(",") if m.strip()]
    C = build(a.input, a.nev, a.ptmin, layers, a.n_adjacent, a.chunk,
              a.cache_dir, a.cache, a.hash_content, a.calib_events, masks=masks,
              kf_opts={"use_angles": a.kf_angles == "on",
                       "alpha_scale": a.alpha_scale, "beta_scale": a.beta_scale,
                       "d0_prior_cm": a.d0_prior_cm},
              feat_per_chunk=a.features_per_chunk)
    nev = C["n_events"]
    kin = np.isfinite(C["eta"])
    print(f"\n{nev} events, {len(C['key']):,} TPs with >= 1 hit "
          f"({len(C['key']) / nev:,.0f}/event); {int((~kin).sum()):,} are OT-only "
          f"and carry no truth kinematics")
    for c in (0.5, 1.0, 2.0, 5.0):
        m = C["pt"] >= c
        print(f"    pT >= {c:>3} GeV: {int(m.sum()):>9,}")
    rows = [r for r in seed_table(C) if not r.get("infeasible")]
    rows.sort(key=lambda r: -r["n_found"])
    print(f"\n{'seed':<20}{'ar':>3}{'found':>9}{'cand/ev':>11}{'fit/ev':>8}"
          f"{'ownDR':>8}{'X2%':>5}{'DR%':>5}{'X2ang':>8}{'nlay':>6}{'fake':>7}"
          f"{'s(kap)':>9}{'s(d0)um':>9}{'s(cot)':>9}{'s(z0)um':>9}")
    for r in rows[:30]:
        print(f"{r['seed']:<20}{r['arity']:>3}{r['n_found']:>9,}{r['cand']:>11,.0f}"
              f"{r['fit']:>8,.0f}{r['n_found_dr']:>8,}"
              f"{100 * r['chi2_pass']:>5.0f}{100 * r['dr_pass']:>5.0f}"
              f"{r['chi2_angle_per_cl']:>8.1f}{r['nlayer']:>6.1f}"
              f"{r['fake']:>7.3f}{r['sig_kappa']:>9.4f}{1e4 * r['sig_d0_cm']:>9.0f}"
              f"{r['sig_cot']:>9.4f}{1e4 * r['sig_z0_cm']:>9.0f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_events": nev, "n_tp": int(len(C["key"])),
                   "pt_min": a.ptmin, "cache_dir": C["cache_dir"],
                   "seeds": seed_table(C)}, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
