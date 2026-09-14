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

NAME = {1: "IL1", 2: "IL2", 3: "IL3", 4: "IL4",
        11: "OL1", 12: "OL2", 13: "OL3", 14: "OL4", 15: "OL5", 16: "OL6"}
CODE = {v: k for k, v in NAME.items()}
ALL_LAYERS = (1, 2, 3, 4, 11, 12, 13, 14, 15, 16)
OT_COLS = ["layer", "isBarrel", "r", "phi", "z", "tpIdx", "tpPt"]

# Bumped whenever a change alters the NUMBERS in a shard. The cache key folds it
# in, so a stale cache cannot be silently reused across a semantic change.
FORMAT_VERSION = 4
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


def seed_universe(rmed, layers, n_adjacent):
    """Every ordered (inner, outer) pair of instrumented layers, each projected to
    its n_adjacent nearest-in-radius OTHER layers.

    No acceptance heuristic picks the target. An earlier revision chose the
    single target with the largest TP overlap, which for IL1+IL2 selected OL2
    over OL1 on a 1.3% margin while OL1 wins on efficiency, cost and fake rate,
    and which buried IL4 -- the cheapest target of all -- because a TP reaching
    IL2 often stops before IL4.
    """
    L = sorted(layers, key=lambda x: rmed.get(x, 0.0))
    out = []
    for i, la in enumerate(L):
        for lb in L[i + 1:]:
            rest = [x for x in L if x not in (la, lb)]
            rmid = 0.5 * (rmed[la] + rmed[lb])
            for lc in sorted(rest, key=lambda x: abs(rmed[x] - rmid))[:n_adjacent]:
                out.append((la, lb, lc))
    return out


BUILD_MASKS = ["AAAA", "AAAI", "AAIA", "AIAA", "IAAA", "AAII", "AIAI",
               "AIIA", "IAAI", "IAIA", "IIAA", "AIII", "IAII", "IIAI", "IIIA"]
IL_OF = {0: 1, 1: 2, 2: 3, 3: 4}
OT_BARREL = (11, 12, 13, 14, 15, 16)


def seed_universe_over_builds(rmed, masks, n_adjacent):
    """Union over builds of each build's OWN seed enumeration.

    A target layer must itself be instrumented, and which layers are adjacent
    depends on which exist -- so the enumeration is not build-independent even
    though a seed's RESULT is. In AAIA, IL1+IL2's two nearest available targets
    are IL4 and OL1; over all ten layers they would be IL3 and IL4, and
    IL1+IL2>OL1 -- the seed that actually leads AAIA's cost-weighted menu -- would
    never be run. Enumerating per build and taking the union is what makes one
    pass serve every build.
    """
    seen, out = set(), []
    for mask in masks:
        layers = [IL_OF[i] for i, ch in enumerate(mask) if ch == "A"] + list(OT_BARREL)
        for s in seed_universe(rmed, layers, n_adjacent):
            if s not in seen:
                seen.add(s)
                out.append(s)
    return sorted(out, key=lambda s: (rmed[s[0]], rmed[s[1]], rmed[s[2]]))


def seed_tag(s):
    return f"{NAME[s[0]]}+{NAME[s[1]]}>{NAME[s[2]]}"


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


def _qual(U, trip, TPK, TPVZ, TPET):
    """Seed-level residuals for the truth-matched triples: kappa, cot(theta), z0.

    Truth is looked up PER TRACKINGPARTICLE, never off the inner hit row: OT stub
    rows carry NaN for tpVz/tpEta, so reading truth off the inner hit gives
    nonsense for every OT-only seed -- it read sigma(z0) as 42,000-64,000 um and
    sigma(cot) as nan before this was keyed on the TP.
    """
    ga, gb, gc = trip
    dr = U["globalR"][gb] - U["globalR"][ga]
    okp = np.abs(dr) > 0.5
    safe = np.where(okp, dr, 1e9)
    kap = M.wrap(U["globalPhi"][ga] - U["globalPhi"][gb]) / (M.C_BEND * safe)
    cot = (U["globalZ"][gb] - U["globalZ"][ga]) / safe
    z0 = U["globalZ"][ga] - U["globalR"][ga] * cot
    ta = U["tpIdx"][ga]
    real = okp & (ta >= 0) & (ta == U["tpIdx"][gb]) & (ta == U["tpIdx"][gc])
    if not len(TPK) or not real.any():
        return np.zeros((0, 3), np.float32)
    kk = M.tp_key(U["event"][ga], np.maximum(ta, 0))
    p = np.clip(np.searchsorted(TPK, kk), 0, len(TPK) - 1)
    real &= TPK[p] == kk
    if not real.any():
        return np.zeros((0, 3), np.float32)
    r = real
    return np.stack([np.abs(kap[r]) - 1.0 / np.maximum(U["tpPt"][ga][r], 1e-6),
                     cot[r] - np.sinh(TPET[p[r]]),
                     z0[r] - TPVZ[p[r]]], axis=1).astype(np.float32)


def process_chunk(U, Q, seeds, ptmin, rng, qual_per_chunk):
    """Run every seed over one chunk and reduce to census + bitmap + counters."""
    cen = chunk_census(U)
    ntp, nw = len(cen["key"]), (len(seeds) + 63) // 64
    found = np.zeros((ntp, nw), np.uint64)
    # per-TP truth table, for the quality residuals
    itrow = (U["layer"] <= 4) & (U["tpIdx"] >= 0)
    tk = M.tp_key(U["event"][itrow], U["tpIdx"][itrow])
    to = np.argsort(tk, kind="stable")
    tk, tvz, tet = tk[to], U["tpVz"][itrow][to], U["tpEta"][itrow][to]
    tf = np.r_[True, tk[1:] != tk[:-1]] if len(tk) else np.zeros(0, bool)
    TPK, TPVZ, TPET = tk[tf], tvz[tf], tet[tf]
    allidx = np.arange(len(U["layer"]))
    counters, qual = {}, {}
    for s_i, (la, lb, lc) in enumerate(seeds):
        try:
            o = M.it_pair_seed(U, Q, allidx, la, lb, lc, ptmin, True, False, 0.0)
        except M.TooWide as e:
            counters[s_i] = {"infeasible": 1.0, "n": float(e.n)}
            continue
        c = {"pairs": float(o.get("tracklets", 0)),
             "cand": float(o.get("match_cand", 0)),
             "cand_z": float(o.get("match_cand_z", 0)),
             "fit": float(o.get("tracks_to_fit", 0))}
        if "_trip" in o:
            trip = o["_trip"]
            nc, nt = M.cand_purity(U, *trip)
            c["trip"], c["trip_true"] = float(nc), float(nt)
            keys = M.recovered_keys(U, *trip)
            if len(keys):
                # Every recovered key IS in this chunk's census -- the census is
                # every TP with a hit and a recovered TP has three. So this is a
                # direct placement, and a miss means the two disagree about event
                # numbering, which is worth failing on rather than filtering away.
                p = np.searchsorted(cen["key"], keys)
                if p.max() >= ntp or not np.array_equal(cen["key"][p], keys):
                    raise SystemExit("recovered TP absent from the chunk census")
                found[p, s_i >> 6] |= np.uint64(1) << np.uint64(s_i & 63)
            q = _qual(U, trip, TPK, TPVZ, TPET)
            if len(q) > qual_per_chunk:
                q = q[rng.choice(len(q), qual_per_chunk, replace=False)]
            qual[s_i] = q
            del o["_trip"]
        counters[s_i] = c
    return cen, found, counters, qual


# ==========================================================================
# build + cache
# ==========================================================================
def _shard_path(d, i):
    return os.path.join(d, f"chunk_{i:05d}.npz")


def build(spec, nev, ptmin, layers, n_adjacent, chunk, cache_dir, mode,
          hash_content=False, calib_events=100, verbose=True,
          budget_gb=0.4, rss_gb=8.0, masks=None):
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
    cfg = {"ptmin": ptmin, "layers": sorted(layers), "n_adjacent": n_adjacent,
           "chunk": chunk, "calib_events": calib_events, "masks": masks}
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
        seeds = (seed_universe_over_builds(cal["r_median"], masks, n_adjacent)
                 if masks else seed_universe(cal["r_median"], layers, n_adjacent))
        meta = {"cache_key": kh, "format": FORMAT_VERSION, "inputs": man,
                "config": cfg, "calibration": cal,
                "seeds": [list(s) for s in seeds],
                "seed_tags": [seed_tag(s) for s in seeds],
                "chunks_done": 0, "n_events": 0}
        json.dump(meta, open(mpath, "w"), indent=1, default=float)
    seeds = [tuple(s) for s in meta["seeds"]]
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
            cen, found, counters, qual = process_chunk(U, Q, seeds, ptmin, rng,
                                                       QUAL_PER_CHUNK)
            np.savez_compressed(
                _shard_path(d, ci), found=found, n_events=len(ev),
                counters=json.dumps({str(k): v for k, v in counters.items()}),
                **{f"cen_{k}": v for k, v in cen.items()},
                **{f"qual_{k}": v for k, v in qual.items()})
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
    fnd, cnt, qs, nread = [], {}, {}, 0
    for sh in shards:
        if nev is not None and nread >= nev:
            break                    # a prefix of the shards is a valid census
        z = np.load(sh, allow_pickle=False)
        for c in cols:
            parts[c].append(z[f"cen_{c}"])
        fnd.append(z["found"])
        nread += int(z["n_events"])
        for k, v in json.loads(str(z["counters"])).items():
            a = cnt.setdefault(int(k), {})
            for kk, vv in v.items():
                a[kk] = a.get(kk, 0.0) + vv
        for f in z.files:
            if f.startswith("qual_"):
                qs.setdefault(int(f[5:]), []).append(z[f])
    C = {c: np.concatenate(parts[c]) for c in cols}
    C["found"] = np.concatenate(fnd, axis=0)
    C["n_events"] = nread
    C["seeds"] = [tuple(s) for s in meta["seeds"]]
    C["seed_tags"] = meta["seed_tags"]
    C["calibration"] = meta["calibration"]
    C["counters"] = {i: cnt.get(i, {}) for i in range(len(C["seeds"]))}
    C["qual"] = {i: (np.concatenate(v) if v else np.zeros((0, 3), np.float32))
                 for i, v in qs.items()}
    C["cache_dir"] = d
    if verbose:
        print(f"loaded {len(fnd)} shard(s): {nread} events, "
              f"{len(C['key']):,} TPs, {len(C['seeds'])} seeds, "
              f"{C['found'].nbytes / 1e6:.1f} MB bitmap", flush=True)
    return C


# ==========================================================================
# query helpers -- what the interactive tool is built on
# ==========================================================================
def seed_mask(C, i):
    """Boolean over TPs: did seed i recover this TrackingParticle?"""
    return (C["found"][:, i >> 6] >> np.uint64(i & 63)) & np.uint64(1) != 0


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
        c = C["counters"].get(i, {})
        if c.get("infeasible"):
            rows.append({"seed": tag, "infeasible": True})
            continue
        q = C["qual"].get(i, np.zeros((0, 3), np.float32))
        rows.append({
            "seed": tag, "n_found": int(seed_mask(C, i).sum()),
            "pairs": c.get("pairs", 0.0) / nev, "cand": c.get("cand", 0.0) / nev,
            "fit": c.get("fit", 0.0) / nev,
            "fake": 1.0 - c.get("trip_true", 0.0) / max(c.get("trip", 0.0), 1.0),
            "sig_kappa": robust_sigma(q[:, 0]), "sig_cot": robust_sigma(q[:, 1]),
            "sig_z0_cm": robust_sigma(q[:, 2])})
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
    ap.add_argument("--hash-content", action="store_true",
                    help="digest file contents instead of trusting size+mtime")
    ap.add_argument("-o", "--out", default=None, help="write the seed table as JSON")
    a = ap.parse_args()
    layers = [CODE[x.strip()] for x in a.layers.split(",") if x.strip()]
    masks = None if a.masks.strip().lower() == "none" else \
        [m.strip() for m in a.masks.split(",") if m.strip()]
    C = build(a.input, a.nev, a.ptmin, layers, a.n_adjacent, a.chunk,
              a.cache_dir, a.cache, a.hash_content, a.calib_events, masks=masks)
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
    print(f"\n{'seed':<18}{'found':>10}{'cand/ev':>11}{'fit/ev':>9}{'fake':>7}"
          f"{'s(kap)':>9}{'s(cot)':>9}{'s(z0)um':>9}")
    for r in rows[:25]:
        print(f"{r['seed']:<18}{r['n_found']:>10,}{r['cand']:>11,.0f}"
              f"{r['fit']:>9,.0f}{r['fake']:>7.3f}{r['sig_kappa']:>9.4f}"
              f"{r['sig_cot']:>9.4f}{1e4 * r['sig_z0_cm']:>9.0f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_events": nev, "n_tp": int(len(C["key"])),
                   "pt_min": a.ptmin, "cache_dir": C["cache_dir"],
                   "seeds": seed_table(C)}, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
