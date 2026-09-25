"""Per-TrackingParticle findability census, built in event chunks and cached.

WHAT THIS PRODUCES. One entry for EVERY TrackingParticle that left at least one
SmartPixels cluster or OT stub -- 9.52 million of them in the 1000-event PU200
ttbar set, 9,515 per event -- carrying

    key            (event << 20) | tpIdx, globally unique across the input set
    pt eta phi     TP truth kinematics; phi [rad] is the azimuth at PRODUCTION
    d0 z0 vr       L1TTP d0 and z0 [cm] at the POCA to the beamline, production radius [cm]
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

TRUTH KINEMATICS COME FROM THE L1TTP TABLE, joined on (event, tpIdx) for IT
clusters and OT stubs alike. L1TTP holds charged TPs with pT >= 1 GeV only, so a
TP below 1 GeV has pT (the hit's own tpPt) and hit pattern but NaN kinematics. So
does a NEUTRAL TP: the cluster producer credits a cluster to the PARENT of its
dominant contributor, which for a Geant4 secondary without its own TP (e.g. a
conversion electron) can be a photon; ~1.8% of the >= 1 GeV TPs with an IT
cluster in the 5-event PU200 test file are of that kind.
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
# tpGenuine/tpCombinatoric/tpUnknown are the per-stub CMS truth flags, needed
# because tpIdx = -1 cannot distinguish a merged cluster from noise. bend is the
# stub's own curvature estimate, which the seed and projection gates want and
# could not use while it was not loaded at all.
OT_COLS = ["layer", "isBarrel", "r", "phi", "z", "tpIdx", "tpPt",
           "tpGenuine", "tpCombinatoric", "tpUnknown", "bend"]

# Bumped whenever a change alters the NUMBERS in a shard. The cache key folds it
# in, so a stale cache cannot be silently reused across a semantic change.
FORMAT_VERSION = 19   # TP truth (census d0/z0/eta/phi/vr, qual_* and track-row
                      # residuals, OT sigma z) from L1TTP at the POCA

# ---- THE OUTPUT CEILING OF THE REAL OT TRACK FINDER ------------------------
# THE TARGET TO DESIGN AGAINST, stated once here so it does not get re-derived
# wrongly a third time:
#
#     9 phi nonants x 2 eta sectors x 104 tracks = 1872 tracks per event
#
# 104 per region is the tuned OUTPUT ceiling, sized on ttbar PU200 at roughly a
# 5-sigma upward fluctuation in track multiplicity.
#
# WHAT I GOT WRONG BEFORE, recorded so it is not repeated. I read
# maxstep_["DR"] = 108 (Settings.h:849-860) together with
#     if (inputtracklets_.size() >= settings_.maxStep("DR")) continue;
# (PurgeDuplicate.cc:156) as "108 fitted tracks per NONANT may enter duplicate
# removal", and concluded our menu overruns the cap by 8.4x. That reading
# cannot be right: a 108-per-nonant INPUT cap would make a 104-per-region
# OUTPUT unreachable, so the system could never exercise its own ceiling. The
# header says what 108 is in its own first line --
#     "Number of processing steps for one event (108=18TM*240MHz/40MHz)"
# -- a CLOCK BUDGET: 18 time-multiplexed regions x (240/40) cycles. And 18 is
# exactly 9 nonants x 2 eta sectors. So maxstep_ counts processing steps per
# module instance, not tracks per nonant, and the DR comment's "per bin" means
# one of those 18 regions (optionally subdivided by rinvBins x phiBins, which
# default to a single bin each).
#
# STILL TRUE, AND STILL THE POINT: wherever the ceiling bites, it is ARBITRARY
# WITH RESPECT TO QUALITY -- no ranking is applied where the cut happens -- so a
# good displaced track is as likely to be dropped as a fake. That is why a
# menu's per-region load belongs next to its efficiency.
#
# NOT MEASURABLE FROM THESE NTUPLES: L1TTrack_etaSector is 99 for every track in
# every file we have, so the eta boundary below is an ASSUMPTION, not a
# measurement. Splitting at eta = 0 (the +z / -z halves) pending confirmation;
# changing ETA_SECTOR_EDGES only redistributes load between two bins and cannot
# change the total.
N_NONANT = 9
N_ETA_SECTOR = 2
ETA_SECTOR_EDGES = (0.0,)          # ASSUMED; see the note above
TRACKS_PER_REGION = 104            # tuned output ceiling, ttbar PU200 + 5 sigma
N_REGION = N_NONANT * N_ETA_SECTOR
DR_MAX_TRACKS = TRACKS_PER_REGION
# Residuals kept per seed PER CHUNK for the robust spreads. Fixed per chunk, not
# derived from the requested event count, so the same shard serves a 100-event
# and a 1000-event request identically.
QUAL_PER_CHUNK = 400
# one emitted track in this many contributes per-cluster angle rows
ANGLE_SAMPLE = 12


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
    """Identity of a census: format, inputs, config, columns AND the numerics.

    TRACK_COLS BELONGS IN THE KEY. The shards store bare arrays whose names come
    from KF.TRACK_COLS at load time, so adding a column changes what every shard
    means -- but it changes neither the inputs nor the config, and a census
    re-run therefore RESUMED a complete cache and exited in seconds while
    appearing to succeed. Hashing the column list makes that impossible instead
    of relying on remembering to bump FORMAT_VERSION.

    KF.numerics_key() closes the same hole for VALUES rather than columns. The
    third-order helix terms changed every residual in the table while leaving
    the format, the inputs, the config and the column list identical, so the
    cache stayed "valid" and the exported site kept serving the uncorrected fit.
    """
    payload = json.dumps({"format": FORMAT_VERSION, "inputs": manifest,
                          "config": cfg, "track_cols": list(KF.TRACK_COLS),
                          "numerics": KF.numerics_key(),
                          "seed_menu": SA.menu_fingerprint()},
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# ==========================================================================
# frozen calibration
# ==========================================================================
def calibrate(spec, nev, ptmin):
    """Per-layer OT stub z resolution and per-layer median radius, measured once.

    The stub z resolution is the spread of (stub z) - (z0 + r*tanL) with the
    L1TTP helix of the TP the stub belongs to, so it is the real pointing
    resolution the projection has to open its window to, not a nominal strip
    pitch.
    """
    I, nI, _ = M.load_flat(spec, M.IT_TABLE, ["layer", "globalR"], nev)
    O, _, _ = M.load_ot(spec, nev, tp=("tp_z0", "tp_tanL"))
    bar = (O["isBarrel"] > 0) & (O["eta"] <= M.ETA_MATCHED)

    def rs(x):
        q = np.percentile(x, [15.865, 84.135])
        return float(0.5 * (q[1] - q[0]))

    sigz = {}
    for L in range(1, 7):
        g = (bar & (O["layer"] == L) & (O["tpIdx"] >= 0) & (O["tpPt"] >= ptmin)
             & np.isfinite(O["tp_z0"]))
        sigz[L] = rs(O["z"][g] - (O["tp_z0"][g] + O["r"][g] * O["tp_tanL"][g])) \
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


SOFT_BUILDS = ("AAAA", "AAAI", "AAIA", "AIAA", "IAAA")
# Soft tracks make OT stubs almost exclusively on OL1-OL3. MEASURED at
# 1.0-1.5 GeV: OL1 14.5%, OL2 25.3%, OL3 13.4%, then OL4 2.5%, OL5 2.0%,
# OL6 1.7%. Including the outer three would buy ~2% occupancy for their full
# combinatorial cost.
SOFT_TARGETS = (1, 2, 3, 4, 11, 12, 13)


def seed_universe_over_builds(rmed, masks, seed_classes=None):
    """Union over builds of each build's own doublet and triplet seeds.

    Which seeds exist depends on which IT layers are instrumented, but a seed's
    RESULT does not, so every distinct seed is run once and each build composes
    its menu from that. Enumeration and the locality rule live in seed_arity.
    """
    seen, out = {}, []
    for mask in masks:
        il = [IL_OF[i] for i, ch in enumerate(mask) if ch == "A"]
        for sd in SA.enumerate_seeds(il):
            if sd.layers in seen:
                continue
            # THE FILTER WAS ACCEPTED AND IGNORED. seed_classes arrived as a
            # parameter and was never read, so --seed-classes has been a silent
            # no-op: every run got all 30 seeds while claiming otherwise, and
            # the only visible difference was the cache key. Any conclusion
            # drawn from it describes the full menu, not the requested classes.
            if seed_classes is not None \
                    and KF.sysclass_of(sd.layers) not in seed_classes:
                continue
            seen[sd.layers] = sd
            out.append(sd)
    return sorted(out, key=lambda sd: (sd.arity,) + tuple(rmed.get(L, 0.0)
                                                          for L in sd.layers))


def seed_tag(s):
    return s.tag if hasattr(s, "tag") else SA.Seed(s).tag


# ==========================================================================
# chunked unified IT+OT table
# ==========================================================================
def unified_chunks(spec, nev, step, sigz_ot, bench=None):
    """Yield one unified IT+OT hit table per chunk of `step` events.

    `bench` is (alpha_bits, beta_bits) for M.it_prepare, or None for full float.
    It is applied HERE rather than at fit time because the quantised angle feeds
    the seeding and projection windows through kap_a/s_kap as well as the KF
    update, so a bit width changes which candidates exist, not merely how well
    they are fitted.

    Event numbering carries a running offset across chunks AND across files, so
    (event, tpIdx) stays globally unique and chunk censuses concatenate.
    """
    paths = M.expand_inputs(spec)
    M.require_tp_table(paths)
    srcs = [f"{p}:Events" for p in paths]
    itk = [f"{M.IT_TABLE}_{c}" for c in M.IT_COLS]
    otk = [f"{M.OT_TABLE}_{c}" for c in OT_COLS]
    seen = 0
    for A in uproot.iterate(srcs, itk + otk + M.tp_branches(), step_size=step):
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
        U, Q = _unify(I, O, sigz_ot, bench)
        M.attach_tp_truth(U, M.tp_table(A, ev))
        yield (U, Q), ev
        del A, I, O, U, Q


def _unify(I, O, sigz_ot, bench=None):
    eta = np.abs(np.arcsinh(O["z"] / np.maximum(O["r"], 1e-6)))
    bar = (O["isBarrel"] > 0) & (eta <= M.ETA_MATCHED)
    n = int(bar.sum())
    sig = np.empty(n)
    ol = O["layer"][bar]
    for L in range(1, 7):
        sig[ol == L] = sigz_ot.get(L, sigz_ot.get(str(L), 0.2))
    # PER-STUB TRUTH IS THREE-VALUED AND tpIdx ALONE CANNOT EXPRESS IT.
    # An OT stub carries tpIdx >= 0 only when the producer called it GENUINE.
    # A COMBINATORIC stub is a real merge of this particle's cluster with
    # another particle's, and an UNKNOWN stub is not linked to any particle;
    # both carry tpIdx = -1 and so were indistinguishable, which made every
    # "wrong OT hit" look like noise when a quarter of them still carry this
    # track's position. Required, not inferred: guessing the split from the
    # sign of tpIdx would reproduce exactly the miscount being removed.
    need = ("tpGenuine", "tpCombinatoric", "tpUnknown")
    miss = [c for c in need if c not in O]
    if miss:
        raise SystemExit(
            f"L1TOTStub is missing {miss}: the OT truth counters need the "
            f"per-stub CMS flags, and inferring them from tpIdx would silently "
            f"merge combinatoric stubs into noise. Re-make the ntuple with "
            f"these branches.")
    ot_flag = np.where(O["tpGenuine"][bar] > 0, 1,
                       np.where(O["tpCombinatoric"][bar] > 0, 2, 3)).astype(np.int8)
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
         # 0 on every IT cluster: the code is only defined for an OT stub, and
         # track_rows gates on is_ot before reading it
         "ot_flag": np.r_[np.zeros(len(I["layer"]), np.int8), ot_flag],
         # likewise bend: a pixel cluster has none, and the bend gate runs only
         # for OT doublet seeds, so the IT zeros are never compared
         "bend": np.r_[np.zeros(len(I["layer"])), O["bend"][bar]],
         "event": np.r_[I["event"], O["event"][bar]]}
    QI = M.it_prepare({c: I[c] for c in M.IT_COLS}, bench)
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

    Kinematics are the L1TTP truth every hit of the TP carries identically
    (M.attach_tp_truth); NaN for TPs not in L1TTP (neutral or below 1 GeV).
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
    order = np.argsort(key, kind="stable")
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
    return {"key": uk.astype(np.int64),
            "event": (uk >> M.TP_KEY_SHIFT).astype(np.int32),
            "pt": U["tpPt"][rep].astype(np.float32),
            "eta": U["tp_eta"][rep].astype(np.float32),
            "phi": U["tp_phi_prod"][rep].astype(np.float32),
            "d0": U["tp_d0"][rep].astype(np.float32),
            "z0": U["tp_z0"][rep].astype(np.float32),
            "vr": np.hypot(U["tp_vx"][rep], U["tp_vy"][rep]).astype(np.float32),
            "hit_it": hit_it, "hit_ot": hit_ot}


def process_chunk(U, Q, seeds, ptmin, rng, qual_per_chunk, kf_opts=None,
                  export_tracks=0, do_dr=True, targets=None,
                  min_layers=SA.MIN_LAYERS, d0_window_cm=0.0, it_ptmin=None,
                  min_it_layers=None, min_ot_conf=0):
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
    # THE pT FLOOR IS PER SEED, not global. The OT stub algorithm's own bend
    # windows are built for pT >~ 2, so below that a particle usually makes no
    # stub at all (0.87 OT layers below 2 GeV against 2.46 above) and an OT or
    # mixed seed cannot exist. An IT-only seed has no such limit, so it can run
    # at a lower floor and cover 1-2 GeV INCLUSIVELY, keeping the >= 2 GeV
    # tracks it already found. A lower floor widens that seed's windows
    # (kmax = 1/ptmin), which is why it must not be applied to seeds that
    # cannot benefit.
    def _ptmin_for(sd):
        # Only a dedicated recovery seed runs low. The standard entries keep
        # their floor, so nothing they used to find can be lost to the soft
        # configuration.
        return it_ptmin if (sd.soft and it_ptmin is not None) else ptmin

    def _rule_for(sd):
        """The layer rule for one seed: per system only where the floor moved.

        THE RULE HAS TO BE SCOPED THE WAY THE pT FLOOR IS. min_it_layers says
        "the IT must supply this many layers by itself", which is the right
        requirement for a soft IT seed and nonsense for an OT one: an OT seed
        has no IT layers, so min_it_layers = 3 demands it find THREE IT
        clusters. Measured on an 8-event probe with the rule applied globally,
        OL2+OL3 fell to 22 found and 3.1 fits per event -- the OT menu gutted by
        a rule written for the IT.

        So when --it-ptmin moves only the IT-only seeds, the rule follows only
        those seeds, and everything else keeps the standard layer count. When
        there is no --it-ptmin every seed is at one floor and the rule applies
        globally, which is what the IT-only soft arms use.
        """
        if min_it_layers is None:
            return None, 0
        if it_ptmin is None:
            return min_it_layers, min_ot_conf   # one floor: rule applies to all
        return (min_it_layers, min_ot_conf) if sd.soft else (None, 0)
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
    targets = list(targets) if targets else list(SA.IL) + list(SA.OT_BARREL)
    counters, qual, trk, hits, ang = {}, {}, {}, {}, {}

    # ---- pass 1: seed, follow, fit, chi2-accept --------------------------
    TP = KF.tp_truth_table(U)
    held = {}
    for s_i, sd in enumerate(seeds):
        try:
            pt_s = _ptmin_for(sd)
            mil, moc = _rule_for(sd)
            o = SA.run_seed(U, Q, sd, pt_s, targets, min_layers=min_layers,
                            d0_cm=d0_window_cm, min_it_layers=mil,
                            min_ot_conf=moc)
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
        keep, chi2s = KF.good_state(fit0, pt_s)
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
        # STANDALONE, before DR can reassign anything. Everything the menu uses
        # -- reach, fake rate, layer count, resolutions, MVA rows -- is recorded
        # here, because composition asks "if I add this seed, what does it
        # bring?" and the answer is what it finds ON ITS OWN. Post-DR numbers
        # answer a different question ("in a 30-seed system, what does it end up
        # owning?") and mixing the two put a seed in the menu for 66,803 TPs it
        # reaches while reporting fake 0.973 for the remnant it does not own.
        o_keep = SA.filter_tracks(o, keep)
        rk0 = SA.recovered_keys(U, o_keep)
        if len(rk0):
            p0 = np.searchsorted(cen["key"], rk0)
            found[p0, s_i >> 6] |= np.uint64(1) << np.uint64(s_i & 63)
        base["found_standalone"] = float(len(rk0))
        h_score = KF.rank_score(fit0, chi2s, w_angle=w_ang)[keep]
        gk = gidx_all[keep]
        gck = o_keep.get("_gC")
        tripk = (o_keep["_gA"], o_keep["_gB"],
                 gck if gck is not None else o_keep["_gB"])
        ta = U["tpIdx"][o_keep["_gA"]]
        real = (ta >= 0) & (ta == U["tpIdx"][o_keep["_gB"]])
        if gck is not None:
            real &= ta == U["tpIdx"][gck]
        base.update({
            "fit": float(o_keep["tracks_to_fit"]),
            "found_standalone": float(len(rk0)),
            "chi2_scaled_sum": float(chi2s[keep].sum()),
            "chi2_angle_sum": float(fit0["chi2_angle"][keep].sum()),
            "n_angle_sum": float(fit0["n_angle"][keep].sum()),
            "nlayer_sum": float((sd.arity + o_keep["_nconf"]).sum()),
            # NOT comparable across arity: a doublet is credited when its two
            # seed clusters share a TP, a triplet when all three do.
            "trip": float(len(ta)), "trip_true": float(real.sum())})
        counters[s_i] = base
        idx = np.flatnonzero(real)
        if len(idx):
            if len(idx) > qual_per_chunk:
                idx = idx[rng.choice(len(idx), qual_per_chunk, replace=False)]
            tq = tuple(x[idx] for x in tripk)
            fq = KF.fit_tracks(U, Q, tq, gidx=gk[idx], use_angles=False,
                               **d0_opt)
            dk, dd, dc, dz, rm = KF.truth_residuals(U, tq, fq, TP)
            # dd is a TTTrack-convention d0 residual. Shards written before
            # 2026-09-23 hold it with the OPPOSITE sign (the KF then computed in
            # dxy); every consumer uses only its width (robust_sigma), which a
            # global sign flip leaves unchanged -- verified: a rebuilt shard
            # matches the old one bit for bit except this column's sign.
            qual[s_i] = (np.stack([dk, dd, dc, dz,
                                   fq["nhit"][rm].astype(np.float64)],
                                  axis=1).astype(np.float32)
                         if len(dk) else np.zeros((0, 5), np.float32))
        if export_tracks:
            # EVERY track, not a sample: the page needs the full fake population
            # to bin fake rate, and fakes are ~98% of the emitted collection. A
            # cap would quietly soften exactly the distribution it exists to show.
            ks = np.arange(o_keep["tracks_to_fit"])
            if 0 < export_tracks < len(ks):
                ks = rng.choice(len(ks), export_tracks, replace=False)
            tf = tuple(x[ks] for x in tripk)
            Tf = KF.gather_hits(U, Q, gk[ks], KF.LAYER_ORDER)
            ff = KF.fit_tracks(U, Q, tf, gidx=gk[ks], use_angles=False,
                               **d0_opt)
            trk[s_i] = KF.track_rows(U, Q, Tf, ff, tf, TP, s_i, sd.arity,
                                     KF.sysclass_of(sd.layers), h_score[ks])
            # HIT LISTS, for duplicate removal the way a real system does it.
            # DR conflicts on >= 3 SHARED HITS, which cannot be reconstructed
            # from fitted parameters, so the cluster indices have to travel with
            # the track. 10 int32 per track, ~500 MB over the full sample --
            # affordable, and it keeps min_shared (and the criterion itself)
            # changeable without re-running the seeding.
            hits[s_i] = gk[ks].astype(np.int32)
            # PER-CLUSTER ANGLE RESIDUALS, on a sample. Four IT hits per track
            # over 12.5M tracks would be a 50M-row table; one track in
            # ANGLE_SAMPLE keeps it to ~100 MB, which is ample for a residual
            # distribution and is stated on the page.
            asel = np.arange(0, len(ks), ANGLE_SAMPLE)
            if len(asel):
                ta = tuple(x[ks][asel] for x in tripk)
                Ta = KF.gather_hits(U, Q, gk[ks][asel], KF.LAYER_ORDER)
                fa = KF.fit_tracks(U, Q, ta, gidx=gk[ks][asel],
                                   use_angles=False, **d0_opt)
                ang[s_i] = KF.angle_rows(U, Ta, fa, ta, asel.astype(np.int64))
        held[s_i] = {"o": o_keep, "sd": sd, "score": h_score}
        del o, fit0

    # ---- phi-nonant occupancy, on the collection that ENTERS DR ---------
    # Binned on the INNER SEED CLUSTER's azimuth, because a tracklet is
    # processed in the sector its seeding stub falls in -- not on the fitted
    # phi0, which is a property of the track rather than of where the hardware
    # handled it. Stored as counts per (event, nonant) rather than as a
    # pass/fail at 108, so the cap can be varied afterwards without re-running.
    keys = sorted(held)
    ev0 = int(U["event"].min()) if len(U["event"]) else 0
    nev_c = int(U["event"].max()) - ev0 + 1 if len(U["event"]) else 0
    nonant = np.zeros(max(nev_c, 0) * N_REGION, np.int32)
    if keys and nev_c > 0:
        gA = [held[i]["o"]["_gA"] for i in keys]
        ev_all = np.concatenate([U["event"][g] for g in gA])
        ph_all = np.concatenate([U["globalPhi"][g] for g in gA])
        # eta of the seed's inner cluster, the same object the phi bin uses
        et_all = np.concatenate([np.arcsinh(U["globalZ"][g]
                                            / np.maximum(U["globalR"][g], 1e-6))
                                 for g in gA])
        nz = np.clip(((ph_all + np.pi) % (2 * np.pi))
                     / (2 * np.pi / N_NONANT), 0, N_NONANT - 1e-9).astype(np.int64)
        es = np.searchsorted(np.asarray(ETA_SECTOR_EDGES), et_all)
        es = np.clip(es, 0, N_ETA_SECTOR - 1).astype(np.int64)
        loc = ev_all - ev0                   # chunk-local event index
        ok_ = (loc >= 0) & (loc < nev_c)
        nonant = np.bincount(((loc[ok_] * N_NONANT + nz[ok_]) * N_ETA_SECTOR
                              + es[ok_]),
                             minlength=nev_c * N_REGION).astype(np.int32)
    if keys and do_dr:
        G = np.concatenate([KF.hits_from_seed(U, held[i]["o"]) for i in keys])
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

    # ---- pass 2: post-DR OWNERSHIP only ---------------------------------
    for s_i in keys:
        h = held[s_i]
        dr = h["dr"]
        counters[s_i]["owned"] = float(dr.sum())
        counters[s_i]["before_dr"] = float(len(dr))
        if not dr.any():
            continue
        o = SA.filter_tracks(h["o"], dr)
        rkeys = SA.recovered_keys(U, o)
        if len(rkeys):
            p = np.searchsorted(cen["key"], rkeys)
            found_dr[p, s_i >> 6] |= np.uint64(1) << np.uint64(s_i & 63)
    return cen, found, found_dr, counters, qual, trk, hits, ang, nonant


def truncation_table(C, cap=TRACKS_PER_REGION):
    """What the per-region output ceiling would cost this menu.

    Returns None when the census predates the occupancy record rather than
    reporting zeros, which would read as "no truncation" instead of "not
    measured".
    """
    occ = C.get("nonant")
    if occ is None or not len(occ):
        return None
    occ = np.asarray(occ, np.int64)
    over = np.maximum(occ - cap, 0)
    n_tracks = int(occ.sum())
    nev = max(len(occ) // N_REGION, 1)
    return {"cap": int(cap), "n_bins": int(len(occ)),
            "n_tracks": n_tracks,
            "per_event": n_tracks / nev,
            "per_region": n_tracks / max(len(occ), 1),
            "budget_per_event": N_REGION * cap,
            "frac_of_budget": n_tracks / nev / (N_REGION * cap),
            "max_occ": int(occ.max()),
            "n_over_bins": int((occ > cap).sum()),
            "frac_bins_over": float((occ > cap).mean()),
            "n_dropped": int(over.sum()),
            "frac_dropped": float(over.sum() / max(n_tracks, 1)),
            "pct": {p: float(np.percentile(occ, p)) for p in (50, 90, 99, 100)},
            "scan": {c: float(np.maximum(occ - c, 0).sum() / max(n_tracks, 1))
                     for c in (52, 104, 208, 416)}}


def _shard_path(d, i):
    return os.path.join(d, f"chunk_{i:05d}.npz")


def build(spec, nev, ptmin, layers, n_adjacent, chunk, cache_dir, mode,
          hash_content=False, calib_events=100, verbose=True,
          budget_gb=0.4, rss_gb=8.0, masks=None, kf_opts=None,
          export_tracks=0, seed_classes=None, targets=None,
          min_layers=SA.MIN_LAYERS, bench=None, d0_window_cm=0.0,
          it_ptmin=None, min_it_layers=None, min_ot_conf=0):
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
           "export_tracks": export_tracks,
           "seed_classes": sorted(seed_classes) if seed_classes else None,
           "targets": sorted(targets) if targets else None,
           "min_layers": min_layers,
           # PART OF THE IDENTITY. The per-system rule changes which tracks
           # exist, so a cache built with one must never be extended by a run
           # using another.
           "min_it_layers": min_it_layers,
           "min_ot_conf": int(min_ot_conf),
           # part of the identity: a bit width changes the candidates themselves
           "bench": list(bench) if bench else None,
           # the impact-parameter allowance the projection windows carry. A
           # doublet assumes d0 = 0, so at r = 3 cm a real 1 mm displacement
           # throws the azimuth by 33 mrad and falls outside the window: this is
           # the knob that tests whether that is where displaced tracks are lost
           "d0_window_cm": float(d0_window_cm),
           "it_ptmin": None if it_ptmin is None else float(it_ptmin)}
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
        seeds = (seed_universe_over_builds(cal["r_median"], masks, seed_classes)
                 if masks else SA.enumerate_seeds(list(SA.IL)))
        # DEDICATED ENTRIES, APPENDED. The standard seeds above are left exactly
        # as they are; --it-ptmin adds a parallel set that runs at the lower
        # floor under the per-system rule. Switching the extension on and off is
        # then a seed selection, and the 2 GeV numbers cannot move.
        if it_ptmin is not None:
            seeds = list(seeds) + SA.soft_seeds(seeds)
        if not seeds:
            raise SystemExit("no seeds after filtering; check --seed-classes")
        # bend -> kappa per OT layer, measured from real on-track stubs. Stored
        # in meta so a resumed run gates pairs with the SAME slopes the earlier
        # chunks used; recomputing it per resume would make shards inconsistent.
        meta = {"cache_key": kh, "format": FORMAT_VERSION, "inputs": man,
                "config": cfg, "calibration": cal,
                "ot_bend_cal": {str(k): list(v) for k, v in
                                M.calibrate_ot_bend(spec, calib_events).items()},
                "seeds": [list(sd.layers) for sd in seeds],
                "seed_soft": [bool(sd.soft) for sd in seeds],
                "seed_notes": [sd.note for sd in seeds],
                "seed_tags": [sd.tag for sd in seeds],
                "chunks_done": 0, "n_events": 0}
        json.dump(meta, open(mpath, "w"), indent=1, default=float)
    _soft = meta.get("seed_soft") or [False] * len(meta["seeds"])
    seeds = [SA.Seed(tuple(x), soft=bool(s))
             for x, s in zip(meta["seeds"], _soft)]
    sigz = {int(k): float(v) for k, v in meta["calibration"]["sigz_ot"].items()}
    # Set on BOTH paths -- fresh build and cache resume -- because the OT bend
    # gate raises rather than silently falling back to the nominal slope, and a
    # resume that skipped this would fail at the first OT doublet.
    obc = meta.get("ot_bend_cal")
    if obc:
        M.set_ot_bend_cal({int(k): tuple(v) for k, v in obc.items()})
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
        for ci, ((U, Q), ev) in enumerate(
                unified_chunks(spec, want, chunk, sigz, bench)):
            if ci < done:
                del U, Q                  # resume re-reads, but never re-seeds
                continue
            cen, found, found_dr, counters, qual, trk, hits, ang, nonant = process_chunk(
                U, Q, seeds, ptmin, rng, QUAL_PER_CHUNK, kf_opts,
                export_tracks, targets=targets, min_layers=min_layers,
                d0_window_cm=d0_window_cm, it_ptmin=it_ptmin,
                min_it_layers=min_it_layers, min_ot_conf=min_ot_conf)
            np.savez_compressed(
                _shard_path(d, ci), found=found, found_dr=found_dr,
                n_events=len(ev),
                counters=json.dumps({str(k): v for k, v in counters.items()}),
                **{f"cen_{k}": v for k, v in cen.items()},
                **{f"qual_{k}": v for k, v in qual.items()},
                # n_clusters lets load() offset the chunk-local cluster indices
                # into a global space; chunks are written independently and
                # resumably, so the offset cannot be baked in at write time
                n_clusters=len(U["layer"]), nonant=nonant,
                **{f"trk_{k}": v for k, v in trk.items()},
                **{f"hit_{k}": v for k, v in hits.items()},
                **{f"ang_{k}": v for k, v in ang.items()})
            done, nev_seen = ci + 1, nev_seen + len(ev)
            meta["chunks_done"], meta["n_events"] = done, nev_seen
            json.dump(meta, open(mpath, "w"), indent=1, default=float)
            if verbose:
                el = time.time() - t0
                print(f"  chunk {ci:>4}  {len(ev):>3} ev  {len(cen['key']):>8,} TPs"
                      f"  {el:7.1f}s  peak {M.peak_gb():.2f} GB", flush=True)
            del U, Q, cen, found
    return load(d, nev=nev, verbose=verbose)


def load(d, nev=None, verbose=True, with_tracks=False):
    meta = json.load(open(os.path.join(d, "manifest.json")))
    shards = sorted(Path(d).glob("chunk_*.npz"))
    if not shards:
        raise SystemExit(f"{d} has no shards")
    cols = ["key", "event", "pt", "eta", "phi", "d0", "z0", "vr", "hit_it", "hit_ot"]
    parts = {c: [] for c in cols}
    fnd, fdr, cnt, qs, fts, hts, ags, nread = [], [], {}, {}, {}, {}, {}, 0
    non = []
    cl_tot = cl_off = 0
    for sh in shards:
        if nev is not None and nread >= nev:
            break                    # a prefix of the shards is a valid census
        z = np.load(sh, allow_pickle=False)
        for c in cols:
            parts[c].append(z[f"cen_{c}"])
        fnd.append(z["found"])
        fdr.append(z["found_dr"])
        cl_off = cl_tot
        cl_tot += int(z["n_clusters"]) if "n_clusters" in z.files else 0
        nread += int(z["n_events"])
        if "nonant" in z.files:
            non.append(z["nonant"])
        for k, v in json.loads(str(z["counters"])).items():
            a = cnt.setdefault(int(k), {})
            for kk, vv in v.items():
                a[kk] = a.get(kk, 0.0) + vv
        for f in z.files:
            if f.startswith("qual_"):
                qs.setdefault(int(f[5:]), []).append(z[f])
            elif with_tracks and f.startswith("trk_"):
                fts.setdefault(int(f[4:]), []).append(z[f])
            elif with_tracks and f.startswith("ang_"):
                ags.setdefault(int(f[4:]), []).append(z[f])
            elif with_tracks and f.startswith("hit_"):
                h = z[f]
                hts.setdefault(int(f[4:]), []).append(
                    np.where(h >= 0, h + cl_off, -1).astype(np.int64))
    C = {c: np.concatenate(parts[c]) for c in cols}
    C["found"] = np.concatenate(fnd, axis=0)
    C["found_dr"] = np.concatenate(fdr, axis=0)
    C["n_events"] = nread
    C["nonant"] = np.concatenate(non) if non else np.zeros(0, np.int32)
    # the soft flag has to survive the reload too, or the report describes the
    # recovery entries under the standard floor and the standard rule
    _sf = meta.get("seed_soft") or [False] * len(meta["seeds"])
    C["seeds"] = [SA.Seed(tuple(x), soft=bool(f))
                  for x, f in zip(meta["seeds"], _sf)]
    C["seed_tags"] = meta["seed_tags"]
    C["seed_notes"] = meta.get("seed_notes", [""] * len(meta["seeds"]))
    C["calibration"] = meta["calibration"]
    C["ptmin"] = float(meta["config"].get("ptmin", 2.0))
    C["min_layers"] = int(meta["config"].get("min_layers", SA.MIN_LAYERS))
    mil = meta["config"].get("min_it_layers")
    C["min_it_layers"] = None if mil is None else int(mil)
    C["min_ot_conf"] = int(meta["config"].get("min_ot_conf", 0) or 0)
    ipm = meta["config"].get("it_ptmin")
    C["it_ptmin"] = None if ipm is None else float(ipm)
    C["counters"] = {i: cnt.get(i, {}) for i in range(len(C["seeds"]))}
    C["qual"] = {i: (np.concatenate(v) if v else np.zeros((0, 5), np.float32))
                 for i, v in qs.items()}
    # tracks are NOT concatenated unless asked: the full table is ~6M rows and
    # the menu study has no use for it
    C["tracks"] = {i: np.concatenate(v) for i, v in fts.items()}
    # A CACHE BUILT UNDER A DIFFERENT COLUMN LIST MUST NOT BE READ. The shards
    # store bare arrays and the names come from KF.TRACK_COLS at load time, so a
    # width mismatch would map every name onto the wrong column -- plausible
    # numbers, silently wrong plots. FORMAT_VERSION is in the cache key to stop
    # this, and this check catches a cache reached by any other route.
    for i, v in C["tracks"].items():
        if v.shape[1] != len(KF.TRACK_COLS):
            raise SystemExit(
                f"{d}: seed {i} track rows are {v.shape[1]} columns but "
                f"KF.TRACK_COLS has {len(KF.TRACK_COLS)}. This cache was built "
                f"by a different version; rebuild it or check out the matching "
                f"code.")
    C["hits"] = {i: np.concatenate(v) for i, v in hts.items()}
    C["angles"] = {i: np.concatenate(v) for i, v in ags.items()}
    C["angle_cols"] = list(KF.ANGLE_COLS)
    C["track_cols"] = list(KF.TRACK_COLS)
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
    def _need_for(C, sd):
        """Confirmations this seed had to find, under whichever rule was set."""
        mil = C.get("min_it_layers")
        if mil is not None and C.get("it_ptmin") is not None and not sd.soft:
            mil = None          # the per-system rule applies to recovery seeds
        if mil is not None:
            return float(max(int(mil) - sum(1 for L in sd.layers if L <= 4), 0))
        ml = C.get("min_layers")
        return float(max(int(ml) - sd.arity, 0)) if ml is not None \
            else float(sd.min_proj())

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
            # THE RULE THE SEED ACTUALLY RAN UNDER, derived from the stored
            # config. This called sd.min_proj(), which recomputes the hardcoded
            # 4-layer rule and ignores --min-layers / --min-it-layers entirely,
            # so every arm of a layer-rule scan reported an identical number
            # and the scan looked like it had never been applied. It had been;
            # only the label was wrong, which is worse than either, because it
            # sent me looking for a bug in the physics.
            "soft": bool(sd.soft),
            "ptmin": (C.get("it_ptmin") if sd.soft and C.get("it_ptmin")
                      else C.get("ptmin")),
            "min_proj": _need_for(C, sd),
            "min_ot_conf": float(C.get("min_ot_conf") or 0),
            "conf_it": c.get("conf_it_sum", 0.0) / max(c.get("n_pairs_rule", 1.0), 1.0),
            "conf_ot": c.get("conf_ot_sum", 0.0) / max(c.get("n_pairs_rule", 1.0), 1.0),
            "nlayer": c.get("nlayer_sum", 0.0) / max(c.get("fit", 1.0), 1.0),
            "pass_frac": c.get("fit", 0.0) / max(c.get("before_minlayers", 1.0), 1.0),
            "chi2_pass": c.get("fit", 0.0) / max(c.get("before_chi2", 1.0), 1.0),
            "dr_pass": c.get("owned", 0.0) / max(c.get("before_dr", 1.0), 1.0),
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
    ap.add_argument("--seed-classes", default=None,
                    help="restrict the seed universe by class: comma list of "
                         "it,ot,mixed. Soft-track finding wants 'it' alone, "
                         "because the OT cannot SEED below its stub threshold "
                         "-- only 4.4%% of 1-2 GeV TPs reach >= 4 OT layers.")
    ap.add_argument("--targets", default=None,
                    help="layers to follow into, e.g. IL1,IL2,IL3,IL4,OL1,OL2,OL3. "
                         "Soft tracks make OT stubs almost only on OL1-OL3 "
                         "(measured 14.5/25.3/13.4%% at 1.0-1.5 GeV, then ~2%% "
                         "on OL4-OL6), so following the outer three buys "
                         "combinatorics and no acceptance.")
    ap.add_argument("--min-it-layers", type=int, default=None,
                    help="layers the IT must supply BY ITSELF. Overrides "
                         "--min-layers. Use with --min-ot-conf to say whether "
                         "an OT confirmation is optional (0) or required (1); "
                         "a flat --min-layers lets an IT doublet complete on a "
                         "single OT stub, which is not a track worth having.")
    ap.add_argument("--min-ot-conf", type=int, default=0,
                    help="OT confirmations required, counted separately from "
                         "the IT. Only meaningful with --min-it-layers.")
    ap.add_argument("--min-layers", type=int, default=SA.MIN_LAYERS,
                    help="layers required on a track. The OT's 4 is defined "
                         "against six barrel layers; a 3-layer IT-only build "
                         "cannot supply a fourth from the IT alone.")
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
    ap.add_argument("--export-tracks", type=int, default=0, nargs="?",
                    const=-1,
                    help="write per-track rows for the interactive page and the "
                         "quality MVA; bare flag exports EVERY track, a number "
                         "caps per seed per chunk, 0 disables")
    ap.add_argument("--it-ptmin", type=float, default=None,
                    help="add DEDICATED sub-2-GeV recovery seeds at this floor. "
                         "They duplicate the IT-only seeds as separate menu "
                         "entries running at the lower floor under "
                         "--min-it-layers; every existing seed keeps its own "
                         "floor and rule, so the higher-pT result is unchanged "
                         "and the extension can be switched on and off by "
                         "selecting seeds. Only IT seeds are duplicated: the OT "
                         "cannot seed below its stub threshold.")
    ap.add_argument("--d0-window-cm", type=float, default=0.0,
                    help="impact-parameter allowance in the projection windows "
                         "(cm). Widens the phi road by d0/r, which is what a "
                         "doublet seed cannot otherwise afford.")
    ap.add_argument("--alpha-bits", type=int, default=None,
                    help="quantise cot(alpha) to this many bits over "
                         "ALPHA_RANGE before seeding, projection and fitting; "
                         "omit for full float")
    ap.add_argument("--beta-bits", type=int, default=None,
                    help="same for the beta-derived per-cluster z0 over "
                         "Z0_RANGE. Both must be given together.")
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
    if (a.alpha_bits is None) != (a.beta_bits is None):
        raise SystemExit("--alpha-bits and --beta-bits must be given together: "
                         "the quantiser takes both or neither")
    layers = [CODE[x.strip()] for x in a.layers.split(",") if x.strip()]
    masks = None if a.masks.strip().lower() == "none" else \
        [m.strip() for m in a.masks.split(",") if m.strip()]
    CLS = {"it": KF.SYS_IT, "ot": KF.SYS_OT, "mixed": KF.SYS_MIX}
    scl = ({CLS[x.strip()] for x in a.seed_classes.split(",") if x.strip()}
           if a.seed_classes else None)
    tgt = ([CODE[x.strip()] for x in a.targets.split(",") if x.strip()]
           if a.targets else None)
    C = build(a.input, a.nev, a.ptmin, layers, a.n_adjacent, a.chunk,
              a.cache_dir, a.cache, a.hash_content, a.calib_events, masks=masks,
              kf_opts={"use_angles": a.kf_angles == "on",
                       "alpha_scale": a.alpha_scale, "beta_scale": a.beta_scale,
                       "d0_prior_cm": a.d0_prior_cm},
              export_tracks=a.export_tracks, seed_classes=scl, targets=tgt,
              min_layers=a.min_layers, min_it_layers=a.min_it_layers,
              min_ot_conf=a.min_ot_conf,
              bench=((a.alpha_bits, a.beta_bits)
                     if a.alpha_bits and a.beta_bits else None),
              d0_window_cm=a.d0_window_cm, it_ptmin=a.it_ptmin)
    nev = C["n_events"]
    kin = np.isfinite(C["eta"])
    print(f"\n{nev} events, {len(C['key']):,} TPs with >= 1 hit "
          f"({len(C['key']) / nev:,.0f}/event); {int((~kin).sum()):,} are not in "
          f"L1TTP (neutral or pT < 1 GeV) and carry no truth kinematics")
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
    trunc = truncation_table(C)
    if trunc:
        print(f"\n=== per-region load, against the tuned ceiling of "
              f"{TRACKS_PER_REGION} tracks per region per event "
              f"({N_NONANT} nonants x {N_ETA_SECTOR} eta sectors = "
              f"{N_REGION * TRACKS_PER_REGION}/event) ===")
        print(f"  fitted tracks           {trunc['n_tracks']:12,d}"
              f"   ({trunc['per_event']:.0f}/event, {trunc['per_region']:.1f}/region)")
        print(f"  fraction of the budget  {100 * trunc['frac_of_budget']:12.1f}%")
        print(f"  busiest region seen     {trunc['max_occ']:12,d}")
        print(f"  region-events over cap  {trunc['n_over_bins']:12,d}"
              f"   of {trunc['n_bins']:,} ({100 * trunc['frac_bins_over']:.2f}%)")
        print(f"  TRACKS DROPPED          {trunc['n_dropped']:12,d}"
              f"   ({100 * trunc['frac_dropped']:.2f}% of all fitted tracks)")
        print("  occupancy percentiles   "
              + "  ".join(f"p{p}={v:.0f}" for p, v in trunc["pct"].items()))
        print("  dropped at other caps:  "
              + "  ".join(f"{c}:{100 * f:.2f}%" for c, f in trunc["scan"].items()))
        print("  NOTE: the cut is arbitrary with respect to quality -- there is "
              "no ranking where it is applied, so a good displaced track is as "
              "likely to go as a fake. The eta-sector boundary is ASSUMED at "
              "eta = 0 (L1TTrack_etaSector is unfilled in these ntuples); that "
              "redistributes load between two bins but cannot change the total.")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_events": nev, "n_tp": int(len(C["key"])),
                   "pt_min": a.ptmin, "cache_dir": C["cache_dir"],
                   "truncation": trunc,
                   "seeds": seed_table(C)}, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
