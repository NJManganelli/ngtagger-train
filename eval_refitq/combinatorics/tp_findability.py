"""Per-TrackingParticle findability census, built in event chunks and cached.

WHAT THIS PRODUCES. One entry for EVERY TrackingParticle credited with at least
one SmartPixels cluster or OT stub (M.credited_tps: a cluster's dominant TP, every
TP behind a stub's two clusters) -- 9.52 million of them in the 1000-event PU200
ttbar set before the stub credit (format < 28), 9,515 per event -- carrying

    key            M.tp_key(event, tpIdx), globally unique across the input set
    pt eta phi     TP truth kinematics; phi [rad] is the azimuth at PRODUCTION
    d0 z0 vr       L1TTP d0 and z0 [cm] at the POCA to the beamline, production radius [cm]
    charge         L1TTP charge (findable() requires a charged TP)
    hit_it hit_ot  bitmasks of which IL1-IL4 / OL1-OL6 the TP actually hit
    hit_otd        bitmask of which OT discs OD1-OD5 it hit (format 36; either side)
    hit_itd        bitmask of which IT discs ID1-ID4 it hit (format 36; either side)
    n_mod_it/_ot   IT / OT modules carrying a credited hit (findable() at >= 4 in all)
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

TRUTH KINEMATICS COME FROM THE L1TTP TABLE, read from the TP's row for IT
clusters and OT stubs alike. L1TTP holds charged TPs with |p| > 1 GeV only (older
nanos pT >= 1 GeV), so a TP outside it (tpIdx <= -3, matched but not written) has
pT (the hit's own tpPt) and hit pattern but NaN kinematics. So
does a NEUTRAL TP: the cluster producer credits a cluster to the PARENT of its
dominant contributor, which for a Geant4 secondary without its own TP (e.g. a
conversion electron) can be a photon; ~1.8% of the >= 1 GeV TPs with an IT
cluster in the 5-event PU200 test file are of that kind.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sys, time, zlib
from pathlib import Path
import numpy as np
import awkward as ak
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402
from ngtagger.data import spix_nano as SPN  # noqa: E402
import kf_emulation as KF           # noqa: E402
import seed_arity as SA             # noqa: E402
import census_angles as CA          # noqa: E402
import ot_chain as OC               # noqa: E402

NAME = dict(SA.NAME)
CODE = {v: k for k, v in NAME.items()}
# THE OT DISCS (format 36, WP-Disc): an OT stub's census code is its L1TOTStub_layer + 10, so the barrel's 1-6
# are OL1-OL6 (11-16) and the discs' 11-15 (either side; the side is the stub's z) are OD1-OD5 (21-25). With the
# discs off (ot_chain.set_discs) the census holds no disc stub: format 35's population.
OT_DISC_CODES = M.OT_DISC_CODES
ALL_LAYERS = (1, 2, 3, 4, 11, 12, 13, 14, 15, 16)
# tpGenuine/tpCombinatoric/tpUnknown are the per-stub CMS truth flags, needed
# because tpIdx = -1 cannot distinguish a merged cluster from noise. bend is the
# stub's own curvature estimate, which the seed and projection gates want and
# could not use while it was not loaded at all. isPS / isTilted are the module
# flags the KF variance reads (OT_MODULE_FLAGS); tpIdxInner / tpIdxOuter the
# owners of the stub's two clusters, which the stub is credited to as well
# (M.STUB_TP_COLS).
# x / y: the OT stub azimuth is atan2(y, x) (format 35), in place of L1TOTStub_phi, which canon_v3 stored at
# 12 mantissa bits.
# innerStrip / isFlipped: a 2S disc stub's strip and orientation, the KF's alpha (format 36).
OT_COLS = ["layer", "isBarrel", "r", "z", "x", "y", "tpIdx", "tpPt",
           "tpGenuine", "tpCombinatoric", "tpUnknown", "bend", "isPS", "isTilted",
           "tpIdxInner", "tpIdxOuter", "detId", "innerStrip", "isFlipped"]
# THE DTC FRAMES (format 35, canon_v4): one row per stub copy in a track-finder processing region, with the
# 64-bit word the tracklet finder reads (ot_dtc). A stub with no copy was dropped by the DTC and never reaches
# the finder; the census seeds and follows with the DTC-accepted stubs only, while the findable count keeps
# every stub (census.findable is a property of the TP's hits, not of the trigger's input stage).
DTC_TABLE = "L1TOTStubDTC"
DTC_COLS = ("otStubIdx", "region", "channel", "index", "frameHi", "frameLo")
# the module (rawId) of every IT cluster: with L1TOTStub_detId, what the findable
# count counts (one per module; census.findable)
IT_MODULE_COL = "detId"

# THE INPUT CONTRACT (format 27). Every census input carries the module flags
# the L1 track finder attaches to a stub, L1TOTStub_isPS (PS vs 2S module) and
# L1TOTStub_isTilted (tilted barrel ring), and the census reads them as they
# are: never inferred from the detId, the layer or the stub's (r, z). canon_v2
# predates them and is refused (require_census_inputs) -- here, in the census,
# and not in the shared loader ngtagger.data.spix_nano, which still serves
# canon_v2 to every other tool. Since format 28 the contract also names the
# owners of the stub's clusters (the stub truth, M.STUB_TP_COLS), and since
# format 31 the module of every IT cluster and OT stub (detId), which the
# findable count counts (one hit per module).
OT_MODULE_FLAGS = ("isPS", "isTilted")
OT_REQUIRED = OT_MODULE_FLAGS + M.STUB_TP_COLS[1:] + ("detId", "x", "y", "innerStrip", "isFlipped")
IT_REQUIRED = (IT_MODULE_COL,)

# Bumped whenever a change alters the NUMBERS in a shard. The cache key folds it
# in, so a stale cache cannot be silently reused across a semantic change.
FORMAT_VERSION = 36   # 36: THE OT DISCS (WP-Disc; ot_chain.set_discs, on by default): the OT disc stubs
                      # enter the census table as OD1-OD5 (21-25) and the findable count (n_mod_ot
                      # counts disc modules; new column hit_otd); the closed arm runs the 8 prompt
                      # seeds (D1D2, D3D4, L1D1, L2D1 too) with every disc projection, its final stubs
                      # with OD1-OD5 columns (OT_STUB_COLS); the WP-Disc corrections (ot_chain.FIXES)
                      # on. With the discs off, format 35's numbers (gate G0).
                      # 35: THE OT INPUTS AS THE TRACK FINDER READS THEM (WP-D; canon_v4 only, canon_v3
                      # refused: no DTC frames): the OT stub azimuth is atan2(y, x) (was the 12-bit
                      # L1TOTStub_phi); the census seeds and follows with the DTC-accepted stubs only
                      # (L1TOTStubDTC) while the findable count keeps every stub (U["truth"]); the
                      # prompt OT doublets' pairs are CMSSW's (A-0: ot_chain's TrackletProcessor
                      # wiring, tracklet-engine LUTs on the exact bend codes, the integer tracklet
                      # calculator with its 1.91 GeV rinv, 15 / 22.5 cm z0 and inSector gates; one
                      # pair per stub pair over sectors); angle cut files bound to the format.
                      # 34: S5 OFF BY DEFAULT (physicist, 2026-10-15: the angles "add far
                      # too little" in the fit; M.DEFAULT_STAGES, a switch as before) and
                      # the angle compatibility exported per track for later MVAs:
                      # track columns n_angle_meas and chi2_angle_per_meas appended
                      # (S6's statistic over both angles).
                      # 33: THE ANGLE IMPORT (WP-D0b; sheet angle.error_table,
                      # angle.cut_level, angle.k_knob): every angle comes from the
                      # shared angle module and the locked table p03.table.v1a1e
                      # (census_angles): per cluster the knob (k per angle, fixed /
                      # scaled), the emitted-key centre and width; S1, S2a, S2z /
                      # S2j, S3, S4w and S6 cut at their cut file's per-bin 99 %
                      # thresholds (decision 2) instead of 3 census sigma; S4a adds
                      # the prediction error, S4a and S5 have a table-key switch;
                      # S3-S6 per-angle sets; the per-layer scales, the census knob
                      # and alpha_scale / beta_scale gone; records (rec_*) and the
                      # knob's moved clusters (n_moved) in every shard.
                      # 32: A TIE GIVES NO OWNER (decision 12, "as CMSSW"; CMSSW truth
                      # spec sec. 4): KF.majority_owner returns owner -1 / n_own 0 when
                      # two or more TPs share the largest credit count (<= 31: the
                      # larger index won), and M.seed_tp -1 when several TPs are
                      # credited by every seed hit; track column n_tp_top (TPs
                      # sharing the largest count) appended.
                      # 31: ONE HIT PER MODULE (census.findable; decision of 2026-10-14):
                      # the findable count is the number of distinct modules (IT
                      # L1TSmartPixelsCluster_detId, OT L1TOTStub_detId, the two
                      # systems kept apart) carrying a credited hit of the TP, columns
                      # n_mod_it / n_mod_ot in place of 30's n_hit_it / n_hit_ot.
                      # 30: THE FINDABLE DENOMINATOR (definition sheet census.findable;
                      # decision of 2026-10-13): census columns n_hit_it, n_hit_ot
                      # (credited IT clusters and OT stubs, each ONE hit per TP)
                      # and charge; findable() = charged with >= FINDABLE_MIN_HITS
                      # credited hits. Every credited hit counts the same: the
                      # position-based diagnostics of 29 are gone (found_dr_pos,
                      # n_ot_right_outer_only); n_tp_majority stays. Tracks,
                      # counts and every other column as in 29.
                      # 29: truth diagnostics, nothing else moves: track columns
                      # n_ot_right_outer_only and n_tp_majority (KF.TRACK_COLS,
                      # appended), the seed-truthful bitmap found_dr_pos counting
                      # position-carrying credits only (no outer-only credit), the
                      # track column list in the manifest; inputs with truth -2
                      # ("unavailable") in the stub or cluster tpIdx refused.
                      # 28: an OT stub is credited to every TP behind its clusters
                      # (M.credited_tps; definition sheet v2 truth.stub): census
                      # entries and hit patterns, the seed-truthful bitmaps and
                      # trip_true, the owners and every truth column; tracks,
                      # counts and costs bit-identical to 27.
                      # 27: the OT module flags from the input (OT_MODULE_FLAGS):
                      # the KF's PS/2S pitch and length and its tilted-module z
                      # variance (approxB^2 exactly on isTilted stubs, where
                      # format 26 took max(1, approxB) on every PS stub);
                      # inputs without the flags (canon_v2) refused.
                      # 26: the position arbitration of a projection's candidates
                      # (config "pos_arb", M.POS_ARB; default r-phi + z pulls).
                      # 25: the angle stages (M.ANGLE_STAGES, config "angles") and
                      # the resolution knob ("angle_res") are in the identity;
                      # angle sigmas carry the per-layer pull scales at every stage;
                      # counters pairs_phi, s3/s4w/s6 rejections.
                      # 24: KF.majority_owner fixed (track owner columns n_own,
                      # own_tpidx, n_wrong, is_clean, tp_*, d_* were wrong on
                      # tracks with many empty layers since format 21).
                      # 23: entries = seed x build follow set (a build's seeds follow
                      # only into its active IT layers + OT; manifest seed_follow,
                      # builds, dr_groups), DR and nonant occupancy per DR group,
                      # quality sample drawn per (chunk, entry).
                      # 22: per-seed seed multiplicity by |eta| (shard seed_eta).
                      # 21: TPs identified by tpIdx (L1TTP row; <= -3 matched but not in
                      # L1TTP), census key = M.tp_key(event, tpIdx) with the 22-bit low
                      # word; key-era shards are not comparable.
                      # 20: the IT seeder vetoes clusters whose global direction
                      # fails the angle cap (SPN.direction_usable) from angle-using
                      # seeds. 19: TP truth (census d0/z0/eta/phi/vr, qual_* and
                      # track-row residuals, OT sigma z) from L1TTP at the POCA
# The formats load() READS: this one, 35 (WP-D's, barrel OT), and 34, the last census built from canon_v3
# (rebuild one at the tag census-canon_v3-format34), so the canon_v3 results stay readable. Two formats are different
# numbers under the same names, so load() records the one it read (C["format"]) and every comparison or
# aggregation of censuses goes through one_format(), which refuses a mix unless the caller marks the
# comparison cross-format.
READ_FORMATS = (34, 35, FORMAT_VERSION)


def one_format(runs, cross_format=False):
    """The census formats of `runs` ({label: a loaded census or its manifest}, each with "format"), as a
    sorted tuple. SystemExit when they differ, unless `cross_format` (a comparison explicitly across
    formats, which then reports them side by side)."""
    fmts = {lab: r.get("format") for lab, r in runs.items()}
    got = tuple(sorted(set(fmts.values()), key=str))
    if len(got) > 1 and not cross_format:
        raise SystemExit("census formats differ (" + ", ".join(f"{k}: {v}" for k, v in fmts.items())
                         + "): two formats are different numbers under the same names. Use censuses of "
                         "one format, or mark the comparison cross-format.")
    return got


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


def require_census_inputs(paths):
    """SystemExit naming the first input that breaks the input contract: an OT
    stub table without the module flags, the cluster owners, the module id or the
    position (OT_REQUIRED; canon_v2 lacks the flags), an IT cluster table without
    the module id (IT_REQUIRED), no DTC frame table (DTC_TABLE; canon_v3 has none),
    or truth marked unavailable (-2) in a stub's or an IT cluster's tpIdx columns,
    which would silently credit nobody."""
    for p in paths:
        t = uproot.open(f"{p}:Events")
        keys = set(t.keys())
        miss = [f"{M.OT_TABLE}_{c}" for c in OT_REQUIRED
                if not SPN.has_column(keys, M.OT_TABLE, c)]
        miss += [f"{M.IT_TABLE}_{c}" for c in IT_REQUIRED
                 if not SPN.has_column(keys, M.IT_TABLE, c)]
        miss += [f"{DTC_TABLE}_{c}" for c in DTC_COLS if not SPN.has_column(keys, DTC_TABLE, c)]
        if miss:
            raise SystemExit(
                f"{p} lacks {', '.join(miss)}: the census (format {FORMAT_VERSION}) "
                f"reads the OT module flags, the owners of each stub's clusters, "
                f"the module (detId) of every cluster and stub and the DTC frames "
                f"the track finder reads from its input and never infers them (no "
                f"layer or (r, z) fallback). canon_v2 predates the flags and canon_v3 "
                f"the DTC frames; run the census on canon_v4.")
        truth = [b for c in M.STUB_TP_COLS for b in SPN.branch_names(keys, M.OT_TABLE, [c])]
        if SPN.has_column(keys, M.IT_TABLE, "tpIdx"):
            truth += SPN.branch_names(keys, M.IT_TABLE, ["tpIdx"])
        bad = [b for b in truth if np.any(ak.to_numpy(ak.flatten(t[b].array())) == -2)]
        if bad:
            raise SystemExit(
                f"{p}: truth unavailable (tpIdx -2) in {', '.join(bad)}; the census "
                f"would credit those objects to nobody. Re-make the input with its "
                f"TrackingParticle products.")


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
    I, nI, _ = M.load_flat(spec, M.IT_TABLE, ["layer", "globalR"], nev, discs=OC.DISCS)
    O, _, _ = M.load_ot(spec, nev, tp=("tp_z0", "tp_tanL"))
    bar = (O["isBarrel"] > 0) & (O["eta"] <= M.ETA_MATCHED)

    def rs(x):
        q = np.percentile(x, [15.865, 84.135])
        return float(0.5 * (q[1] - q[0]))

    sigz = {}
    for L in range(1, 7):
        g = (bar & (O["layer"] == L) & SPN.tp_matched(O["tpIdx"]) & (O["tpPt"] >= ptmin)
             & np.isfinite(O["tp_z0"]))
        sigz[L] = rs(O["z"][g] - (O["tp_z0"][g] + O["r"][g] * O["tp_tanL"][g])) \
            if g.sum() >= 200 else 0.2
    rmed = {}
    for L in ALL_LAYERS + ((M.IT_DISC_CODES + OT_DISC_CODES) if OC.DISCS else ()):
        if L <= 10:
            s = I["layer"] == L
        elif L <= 16:
            s = bar & (O["layer"] == L - 10)
        else:                                   # an OT disc: L1TOTStub disc layer 11-15
            s = (O["isBarrel"] == 0) & (O["layer"] == L - 10)
        r = (I["globalR"] if L <= 10 else O["r"])[s]
        rmed[L] = float(np.median(r)) if len(r) else 0.0
    return {"sigz_ot": sigz, "r_median": rmed, "calib_events": nI}


# the census's position arbitration (M.POS_ARB): r-phi and z pulls, each in its
# own window sigma, so no angle configuration is compared against a
# z-blind baseline
POS_ARB_DEFAULT = "rphiz"

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


def seed_universe_over_builds(rmed, masks, seed_classes=None, discs=None):
    """Union over builds of each build's own doublet and triplet seeds, each
    once, in the census order. Enumeration and the locality rule live in
    seed_arity; what a seed follows into is per build (census_entries).
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
    for sd in (discs or {}).get("seeds", ()):           # the disc seeds (format 36), after the barrel's
        if sd.layers not in seen and (seed_classes is None or KF.sysclass_of(sd.layers) in seed_classes):
            seen[sd.layers] = sd
            out.append(sd)
    return sorted(out, key=lambda sd: (bool(M.is_disc(sd.layers).any()), sd.arity)
                  + tuple(rmed.get(L, 0.0) for L in sd.layers))


def seed_tag(s):
    return s.tag if hasattr(s, "tag") else SA.Seed(s).tag


def census_entries(rmed, masks, targets=None, seed_classes=None, it_ptmin=None, discs=None):
    """The census's units of work and each build's menu over them.

    A BUILD'S SURFACES ARE ITS OWN. A mask's letters say which IT layers exist
    (A = active, I = inactive; character k = ILk), and an inactive layer emits
    nothing at L1, so a seed run for a build follows only into that build's
    active IT layers and the OT: the layer rule and the fit then see only those.
    Following every seed into IL1-IL4 whatever the build credited AAAI with IL4
    hits (37.6% of its fitted tracks; 26.1% passed the layer rule only through
    one) and IAAA with IL1 hits (45.7% / 31.1%).

    So an ENTRY is a seed with its follow set: build surfaces within `targets`,
    minus the seed's own layers. A seed in builds with different surfaces is one
    entry per distinct follow set, and builds whose follow sets coincide (e.g.
    OT-only targets) share the entry. Entries come in the seed universe's order,
    standard before the --it-ptmin recovery entries, each seed's variants in mask
    order. `masks` None is the full IT (AAAA).

    Returns (entries, builds): entries are SA.Seed with .follow set; builds maps
    each mask to the indices of its menu's entries.

    THE DISCS (census format 36; `discs` = {"it": the build's IT discs, "ot": the OT discs, "seeds": the disc
    seeds, disc_menu}): every build's surfaces gain its IT discs and the OT discs, its menu the disc seeds (the
    same for every mask: the SmartPixels discs are not varied by the build masks), after the barrel entries.
    """
    masks = list(masks) if masks else ["AAAA"]
    dk = discs or {"it": (), "ot": (), "seeds": ()}
    tg = list(targets) if targets else list(SA.IL) + list(dk["it"]) + list(SA.OT_BARREL) + list(dk["ot"])
    order = {sd.layers: k for k, sd in
             enumerate(seed_universe_over_builds(rmed, masks, seed_classes, discs))}
    rows = []
    for bi, mask in enumerate(masks):
        il = [IL_OF[i] for i, ch in enumerate(mask) if ch == "A"]
        menu = [sd for sd in SA.enumerate_seeds(il) if sd.layers in order]
        if it_ptmin is not None:
            menu += SA.soft_seeds(menu)
        menu += [sd for sd in dk["seeds"] if sd.layers in order]
        for sd in menu:
            follow = tuple(sorted(L for L in tg
                                  if (L > 10 or L in il or L in dk["it"]) and L not in sd.layers))
            rows.append(((sd.soft, order[sd.layers], bi), sd, follow, mask))
    rows.sort(key=lambda r: r[0])
    index, entries, builds = {}, [], {m: [] for m in masks}
    for _, sd, follow, mask in rows:
        k = (sd.layers, sd.soft, follow)
        if k not in index:
            index[k] = len(entries)
            entries.append(SA.Seed(sd.layers, note=sd.note, soft=sd.soft,
                                   category=sd.category, follow=follow))
        builds[mask].append(index[k])
    return entries, builds


def dr_groups(builds, n_entries):
    """Entries partitioned into the sets duplicate removal runs over.

    DR IS PER BUILD: it collapses one system's duplicates, and the same seed in
    two builds is two systems' tracks, not a duplicate. A build whose entries are
    all its own (every build with its IT layers among the targets) is its own
    group. Builds that SHARE an entry must share its DR, so a group is a
    connected component of builds linked by shared entries; with OT-only targets
    every build shares every entry and the group is the whole census.
    Returns the groups as sorted entry lists, ordered by their first entry.
    """
    parent = list(range(n_entries))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for idx in builds.values():
        for i in idx[1:]:
            parent[root(i)] = root(idx[0])
    groups = {}
    for i in range(n_entries):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def entry_rng(ci, sd):
    """The random stream for one entry in one chunk (the quality sample).

    PER ENTRY AND CHUNK, not one stream for the run: with a shared stream an
    entry's sample depended on every entry drawn before it -- i.e. one build's
    numbers on which other builds the census ran -- and a resumed run, which
    skips done chunks without drawing, wrote different shards from a straight
    one."""
    key = repr((sd.layers, bool(sd.soft), sd.follow)).encode()
    return np.random.default_rng([12345, int(ci), zlib.crc32(key)])


def entry_results(U, Q, seeds, ptmin_of, rule_of, min_layers, d0_cm=0.0,
                  ang=None, pos_arb=None):
    """Yield (index, result) per entry: SA.run_seed's result for the entry's
    seed followed into its own follow set, None if the seed formed nothing, or
    the M.TooWide its formation raised.

    A SEED IS FORMED AND PROJECTED ONCE for all of its entries, over the union of
    their follow sets (SA.follow_seed); each entry then applies the layer rule to
    its own (SA.apply_layer_rule). Entries of one seed are contiguous
    (census_entries), so only the current one is held. ptmin_of(sd) and
    rule_of(sd) -> (min_it_layers, min_ot_conf) give an entry's floor and rule.
    `ang` is the census's M.AngleUse (None: every stage), `pos_arb` its position
    arbitration (M.POS_ARB; None: POS_ARB_DEFAULT).
    """
    ang = M.AngleUse(M.DEFAULT_STAGES) if ang is None else ang
    pos_arb = POS_ARB_DEFAULT if pos_arb is None else pos_arb
    formed, F, err = None, None, None
    for s_i, sd in enumerate(seeds):
        if formed != (sd.layers, sd.soft):
            formed, F, err = (sd.layers, sd.soft), None, None
            union = sorted({L for e in seeds if (e.layers, e.soft) == formed
                            for L in e.follow})
            try:
                F = SA.follow_seed(U, Q, sd, ptmin_of(sd), union, ang, d0_cm,
                                   pos_arb)
            except M.TooWide as e:
                err = e
        if err is not None:
            yield s_i, err
            continue
        mil, moc = rule_of(sd)
        yield s_i, (None if F is None else SA.apply_layer_rule(
            F, sd, sd.follow, min_layers=min_layers, min_it_layers=mil,
            min_ot_conf=moc))


# ==========================================================================
# chunked unified IT+OT table
# ==========================================================================
def unified_chunks(spec, nev, step, sigz_ot, bench=None, model=None, it_discs=False, angle_null=None):
    """Yield one unified IT+OT hit table per chunk of `step` events.

    `model` is the census angle model (census_angles.Model: the locked table, the
    knob k per angle and its sigma mode), `bench` (alpha_bits, beta_bits) or None
    for full float. Both are applied HERE, once, rather than at fit time,
    because the angles feed the seeding and projection windows as well as the KF
    update: a bit width or a resolution changes which candidates exist, not
    merely how well they are fitted, and every stage must see the same angles.
    Without a model the angles are M.it_prepare's legacy ones (only tools that
    use no census angle stage may ask for that).

    Event numbering carries a running offset across chunks AND across files, so
    (event, tpIdx) stays globally unique and chunk censuses concatenate.

    With the discs on (ot_chain.DISCS, format 36) the SmartPixels disc clusters ID1-ID4 are read too, as census
    codes 5-8: always into the findable count (U["truth"]), and into the build's hit table when `it_discs`.
    """
    paths = M.expand_inputs(spec)
    M.require_tp_table(paths)
    require_census_inputs(paths)
    srcs = [f"{p}:Events" for p in paths]
    # class T and its truth direction: the knob's reference and the records'
    # population (truth only: never a decision input)
    itc = M.IT_COLS + [IT_MODULE_COL] + list(IT_TRUTH_COLS) + list(IT_DISC_COLS)
    itk = M.table_branches(paths[0], M.IT_TABLE, itc, discs=OC.DISCS)
    otk = M.table_branches(paths[0], M.OT_TABLE, OT_COLS)
    keys0 = uproot.open(f"{paths[0]}:Events").keys()
    tpk = M.tp_branches(paths[0]) + SPN.branch_names(keys0, M.TP_TABLE, TP_GENEALOGY_COLS)
    dtk = SPN.branch_names(keys0, DTC_TABLE, DTC_COLS)
    seen = 0
    for A in uproot.iterate(srcs, itk + otk + tpk + dtk, step_size=step):
        # uproot.iterate IGNORES entry_stop when handed a list of files, so the
        # event limit has to be enforced here or -n 8 reads all 1000.
        if nev is not None:
            if seen >= nev:
                return
            if len(A) > nev - seen:
                A = A[:nev - seen]
        JI = M.table_columns(A, M.IT_TABLE, itc, discs=OC.DISCS)  # TBPX, and ID1-ID4 with the discs
        JO = M.table_columns(A, M.OT_TABLE, OT_COLS)
        nI = ak.to_numpy(ak.num(JI[M.IT_COLS[0]]))
        nO = ak.to_numpy(ak.num(JO[OT_COLS[0]]))
        ev = np.arange(seen, seen + len(nI))
        I = {c: ak.to_numpy(ak.flatten(JI[c])) for c in itc}
        I["event"] = np.repeat(ev, nI)
        if angle_null == "permute":
            permute_angles(I, ev[0])
        O = {c: ak.to_numpy(ak.flatten(JO[c])) for c in OT_COLS}
        O["event"] = np.repeat(ev, nO)
        offO = np.r_[0, np.cumsum(nO)[:-1]].astype(np.int64)
        O["row"] = np.arange(len(O["event"]), dtype=np.int64) - np.repeat(offO, nO)
        JD = SPN.table_columns(A, DTC_TABLE, DTC_COLS)
        nD = ak.to_numpy(ak.num(JD[DTC_COLS[0]]))
        Dt = {c: ak.to_numpy(ak.flatten(JD[c])) for c in DTC_COLS}
        Dt["event"] = np.repeat(ev, nD)
        # the flat L1TOTStub index of every copy
        Dt["stub"] = np.repeat(offO, nD) + Dt["otStubIdx"].astype(np.int64)
        seen += len(nI)
        U, Q = _unify(I, O, Dt, sigz_ot, bench, model, it_discs)
        # the chunk's L1TTP rows travel with U: a TP credited only through a
        # stub's cluster (U["tps"]) has no hit carrying its truth columns; the
        # genealogy columns (truth.genealogy) serve the angle records
        U["l1ttp"] = M.tp_table(A, ev)
        JG = SPN.table_columns(A, M.TP_TABLE, TP_GENEALOGY_COLS)
        for c in TP_GENEALOGY_COLS:
            U["l1ttp"][c] = ak.to_numpy(ak.flatten(JG[c]))
        M.attach_tp_truth(U, U["l1ttp"])
        U["truth"]["l1ttp"] = U["l1ttp"]
        M.attach_tp_truth(U["truth"], U["l1ttp"])
        yield (U, Q), ev
        del A, I, O, U, Q


# TRUTH COLUMNS read for the angle model and its records, never for a decision:
# class T (tpEmuClass == 1) and its truth direction, the knob's reference
# (census_angles); the L1TTP genealogy (truth.genealogy) of the cut calibration
# population
IT_TRUTH_COLS = ("tpEmuClass", "tpGlobalClusterPhi", "tpGlobalClusterCotTheta",
                 # the module-frame truth of a disc cluster (the knob's reference on the discs, format 36)
                 "tpLocalCotAlpha", "tpLocalCotBeta")
# the disc stages' published module-frame sigmas (census_angles.disc_components; format 36)
IT_DISC_COLS = ("sigAlpha", "sigBeta")
TP_GENEALOGY_COLS = ("genFromB", "genFromC", "evt")


# THE NULL CONTROL (census_scan): a fixed seed, the chunk's first event number added
NULL_SEED = 20261020


def permute_angles(I, ev0):
    """The information-free control: per event and IT surface, the emitted angles of the clusters with a usable
    direction permuted among themselves (in place). The transverse angle alpha (direction azimuth less the cluster's
    azimuth) and cot theta move together, a disc cluster's module-frame cots with them; each cluster keeps its
    position, surface, published sigmas, usability flags and truth."""
    from ngtagger.spdefs import angles as SD
    ok = M.SPN.direction_usable(I)
    alpha = SD.alpha_of(I["globalPhi"], I["globalClusterPhi"])
    rng = np.random.default_rng([NULL_SEED, int(ev0)])
    grp = np.flatnonzero(ok)
    key = I["event"][grp] * 100 + np.asarray(I["layer"])[grp]
    order = np.argsort(key, kind="stable")
    grp, key = grp[order], key[order]
    bounds = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    perm = grp.copy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        perm[a:b] = grp[a:b][rng.permutation(b - a)]
    new_alpha = alpha.copy()
    new_alpha[grp] = alpha[perm]
    I["globalClusterPhi"] = SD.angle_from("alpha", new_alpha, I)
    for c in ("globalClusterCotTheta", "localCotAlpha", "localCotBeta"):
        if c in I:
            v = np.asarray(I[c]).copy()
            v[grp] = np.asarray(I[c])[perm]
            I[c] = v


# WHAT THE NULL CONTROL'S SHUFFLED ANGLES MAY FEED (the physicist, 2026-10-21): the cuts only. Hit arbitration (S4a) and the
# duplicate-removal ranking (S7) use what angles-off uses, no angle: with shuffled angles they steered the hit choice and the
# ranking at random and cost the null about 10 pt of efficiency at pT >= 2 GeV (census scan, recalibrated null).
NULL_OFF_STAGES = frozenset({"proj_arb", "dr"})


def null_angle_use(ang):
    """The null control's angle use: `ang` with the arbitration and ranking stages off (NULL_OFF_STAGES)."""
    ang.on = frozenset(ang.on) - NULL_OFF_STAGES
    return ang


def angle_regions(U, Q, regions):
    """AngleUse.regions: the off region's IT clusters lose their direction flags (no angle test or measurement
    reads them: a wildcard everywhere) and are marked Q["ang_off"] (S1 passes them, as with every stage off); every
    position role is kept."""
    Q = dict(Q)
    lay = np.asarray(U["layer"])
    off = ((lay >= 5) & (lay <= 8)) if regions == "barrel" else ((lay >= 1) & (lay <= 4))
    Q["ang_off"] = off
    Q["dir_ok"] = Q["dir_ok"] & ~((lay >= 1) & (lay <= 4) & off)
    if "ldir_ok" in Q:
        Q["ldir_ok"] = Q["ldir_ok"] & ~off
    return Q


def module_key(system, det_id):
    """int64 key of a module: its rawId (uint32) with the system (0 IT, 1 OT) above
    bit 32, so an IT and an OT module can never share a key."""
    return (np.int64(system) << np.int64(32)) | np.asarray(det_id).astype(np.int64)


def _unify(I, O, Dt, sigz_ot, bench=None, model=None, it_discs=False):
    """The chunk's unified hit table U (IT clusters, then the DTC-accepted OT stubs: with the discs on every one,
    barrel and disc (OD1-OD5); off, the barrel's at |eta| <= M.ETA_MATCHED, format 35) and its angle table Q.
    U["truth"] holds every hit the findable count counts (every IT cluster read, the IT discs' too, and those OT
    stubs, accepted by the DTC or not; chunk_census), U["ot_in"] the whole stub and DTC tables for the OT chain
    (ot_chain). The IT disc clusters (codes 5-8) enter U only with `it_discs` (the build's IT discs on)."""
    Ia = I
    if not it_discs:
        I = {c: v[np.asarray(I["layer"]) <= 4] for c, v in I.items()}
    eta = np.abs(np.arcsinh(O["z"] / np.maximum(O["r"], 1e-6)))
    disc = O["isBarrel"] == 0
    pop = np.ones(len(eta), bool) if OC.DISCS else ~disc & (eta <= M.ETA_MATCHED)
    dtc_ok = np.zeros(len(eta), bool)
    dtc_ok[Dt["stub"]] = True
    if OC.KNOWN_BAD == "dtc_acceptance_ignored":
        dtc_ok[:] = True
    bar = pop & dtc_ok
    n = int(bar.sum())
    phi = np.arctan2(O["y"], O["x"])
    sig = np.empty(n)
    ol = O["layer"][bar]
    for L in range(1, 7):
        sig[ol == L] = sigz_ot.get(L, sigz_ot.get(str(L), 0.2))
    # a disc stub's measured coordinate along the strip is r: its sigma the strip (PS macro-pixel) length / sqrt 12
    dsk = disc[bar]
    sig[dsk] = np.where(O["isPS"][bar][dsk] > 0, OC.KF.LENGTH[True], OC.KF.LENGTH[False]) / math.sqrt(12.0)
    # PER-STUB TRUTH IS THREE-VALUED AND tpIdx ALONE CANNOT EXPRESS IT.
    # An OT stub carries a matched tpIdx (SPN.tp_matched) only when the producer
    # called it GENUINE (a unique TP in both clusters). A COMBINATORIC stub has no
    # unique TP in both clusters: two particles' clusters, one cluster without a
    # TP, or two or more TPs shared by both (CMSSW truth spec sec. 2, 5). An
    # UNKNOWN stub has no TP in either cluster. Combinatoric and unknown both
    # carry tpIdx = -1 and so were indistinguishable by it. Required, not
    # inferred: guessing the split from the sign of tpIdx would reproduce exactly
    # the miscount being removed. (Which TPs a stub is credited to is the
    # separate question of M.credited_tps, column "tps" below.)
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
    U = {"layer": np.r_[M.it_codes(I["layer"], "tp_findability._unify"), ol + 10],
         "globalR": np.r_[I["globalR"], O["r"][bar]],
         "globalZ": np.r_[I["globalZ"], O["z"][bar]],
         "globalPhi": np.r_[I["globalPhi"], phi[bar]],
         "sigY": np.r_[I["sigY"], sig],
         # an OT stub has no r-phi CPE sigma of its own; ot_variances builds its
         # variance from pitch and scattering instead, so this is never read there
         "sigX": np.r_[I["sigX"], sig],
         "tpIdx": np.r_[I["tpIdx"], O["tpIdx"][bar]],
         # TRUTH: every TP each hit is credited to (M.credited_tps): an IT
         # cluster's dominant TP; an OT stub's own tpIdx and its clusters' owners
         "tps": np.r_[M.credited_tps(I["tpIdx"]),
                      M.credited_tps(*(O[c][bar] for c in M.STUB_TP_COLS))],
         "tpPt": np.r_[I["tpPt"], O["tpPt"][bar]],
         # 0 on every IT cluster: the code is only defined for an OT stub, and
         # track_rows gates on is_ot before reading it
         "ot_flag": np.r_[np.zeros(len(I["layer"]), np.int8), ot_flag],
         # likewise bend: a pixel cluster has none, and the bend gate runs only
         # for OT doublet seeds, so the IT zeros are never compared
         "bend": np.r_[np.zeros(len(I["layer"])), O["bend"][bar]],
         # the OT module flags (input contract), False on every IT cluster:
         # only the OT variance (KF.ot_variances) reads them
         "ot_ps": np.r_[np.zeros(len(I["layer"]), bool), O["isPS"][bar].astype(bool)],
         "ot_tilted": np.r_[np.zeros(len(I["layer"]), bool),
                            O["isTilted"][bar].astype(bool)],
         # TRUTH-SIDE KEY: the module of every hit, the IT and OT systems kept apart
         # (module_key), for the findable count's one hit per module
         "module": np.r_[module_key(0, I[IT_MODULE_COL]), module_key(1, O["detId"][bar])],
         "event": np.r_[I["event"], O["event"][bar]],
         # the OT stub's L1TOTStub row in its event (-1 on every IT cluster): the OT chain's tracks map onto U
         "ot_row": np.r_[np.full(len(I["layer"]), -1, np.int64), O["row"][bar]]}
    U["truth"] = {"layer": np.r_[Ia["layer"], O["layer"][pop] + 10],
                  "tpIdx": np.r_[Ia["tpIdx"], O["tpIdx"][pop]],
                  "tps": np.r_[M.credited_tps(Ia["tpIdx"]),
                               M.credited_tps(*(O[c][pop] for c in M.STUB_TP_COLS))],
                  "tpPt": np.r_[Ia["tpPt"], O["tpPt"][pop]],
                  "module": np.r_[module_key(0, Ia[IT_MODULE_COL]), module_key(1, O["detId"][pop])],
                  "event": np.r_[Ia["event"], O["event"][pop]]}
    U["ot_in"] = {"stub": {c: O[c] for c in ("layer", "isBarrel", "isPS", "isTilted", "x", "y", "z",
                                             "innerStrip", "isFlipped")},
                  "event": O["event"], "row": O["row"], "dtc": Dt}
    # EVERY angle any stage reads comes from here: the census angle model (the
    # knob, the table's centre and width, then the bench quantisation), or the
    # legacy pull-scaled angles without one. An OT stub carries no alpha/beta:
    # huge sigmas make every angle gate pass, and the joint (z0,phi) pairing then
    # treats the stub as a phi-only wildcard (M.OT_STUB_ANGLES).
    if model is not None:
        Q = CA.with_ot_stubs(CA.prepare({c: I[c] for c in M.IT_COLS + list(IT_TRUTH_COLS) + list(IT_DISC_COLS)
                                         + [IT_MODULE_COL]},
                                        model, bench), n)
    else:
        Q = M.q_with_ot_stubs(M.it_prepare({c: I[c] for c in M.IT_COLS}, bench), n)
    return U, Q


# ==========================================================================
# per-chunk census
# ==========================================================================
def chunk_census(U):
    """One entry per TP credited with >= 1 hit in this chunk (U["tps"]: an IT
    cluster's dominant TP, every TP behind an OT stub's clusters), plus its hit
    pattern. U is the hit table the findable count counts: the unified table's
    U["truth"] (every OT barrel stub, whether the DTC passed it or not), with its
    L1TTP rows (U["l1ttp"]).

    Kinematics are the TP's L1TTP truth (M.attach_tp_truth); NaN for TPs not in
    L1TTP (tpIdx <= -3: neutral or below the L1TTP momentum cut). They are read
    from a hit whose OWN tpIdx is the TP where there is one (pt is then that
    hit's tpPt, defined outside L1TTP too); a TP credited only through a stub's
    cluster takes its L1TTP row (pt NaN outside L1TTP). The two pT sources agree
    to 5e-4 relative (run C: no TP crosses the 1 or 2 GeV cuts between them).
    """
    T = U["tps"]
    hi, sl = np.nonzero(SPN.tp_matched(T))
    if not len(hi):
        z = np.zeros(0)
        return {"key": z.astype(np.int64), "event": z.astype(np.int32),
                "pt": z.astype(np.float32), "eta": z.astype(np.float32),
                "phi": z.astype(np.float32), "d0": z.astype(np.float32),
                "z0": z.astype(np.float32), "vr": z.astype(np.float32),
                "charge": z.astype(np.float32),
                "hit_it": z.astype(np.uint8), "hit_ot": z.astype(np.uint8), "hit_otd": z.astype(np.uint8),
                "hit_itd": z.astype(np.uint8),
                "n_mod_it": z.astype(np.int16), "n_mod_ot": z.astype(np.int16),
                "n_mod_itd": z.astype(np.int16), "n_mod_otd": z.astype(np.int16)}
    key = M.tp_key(U["event"][hi], T[hi, sl])
    # per TP: hits on which it is the hit's own tpIdx (slot 0) first, each group
    # in hit order, so a TP some hit owns is represented by the first such hit
    order = np.lexsort((hi, sl > 0, key))
    key_s, hi_s, sl_s = key[order], hi[order], sl[order]
    first = np.r_[True, key_s[1:] != key_s[:-1]]
    uk = key_s[first]
    rep = hi_s[first]
    own = sl_s[first] == 0                  # the representative hit's own TP
    pos = np.cumsum(first) - 1                # entry index of every credit
    hit_it = np.zeros(len(uk), np.uint8)
    hit_ot = np.zeros(len(uk), np.uint8)
    hit_otd = np.zeros(len(uk), np.uint8)
    hit_itd = np.zeros(len(uk), np.uint8)
    ls = U["layer"][hi_s]
    it = ls <= 4
    itd = np.isin(ls, M.IT_DISC_CODES)
    otd = np.isin(ls, OT_DISC_CODES)
    otb = ~it & ~itd & ~otd
    np.bitwise_or.at(hit_it, pos[it], (1 << (ls[it] - 1)).astype(np.uint8))
    np.bitwise_or.at(hit_itd, pos[itd], (1 << (ls[itd] - M.IT_DISC_CODES[0])).astype(np.uint8))
    np.bitwise_or.at(hit_ot, pos[otb], (1 << (ls[otb] - 11)).astype(np.uint8))
    np.bitwise_or.at(hit_otd, pos[otd], (1 << (ls[otd] - 21)).astype(np.uint8))
    # CREDITED MODULES per TP (census.findable): credited_tps lists a TP once per
    # hit, so an OT stub is one hit for each TP it is credited to however many of
    # its clusters credit it, and hits of one TP on one module count once
    pm, first_hit = np.unique(np.stack([pos, U["module"][hi_s]], axis=1), axis=0, return_index=True)
    it_m = (pm[:, 1] >> 32) == 0
    n_mod_it = np.bincount(pm[it_m, 0], minlength=len(uk)).astype(np.int16)
    n_mod_ot = np.bincount(pm[~it_m, 0], minlength=len(uk)).astype(np.int16)
    # of which disc modules (format 36: the findable count over barrel and disc modules, and its known-bad that
    # counts the barrel's only, findable_barrel_only)
    dm = M.is_disc(ls[first_hit])
    n_mod_itd = np.bincount(pm[it_m & dm, 0], minlength=len(uk)).astype(np.int16)
    n_mod_otd = np.bincount(pm[~it_m & dm, 0], minlength=len(uk)).astype(np.int16)
    L = U["l1ttp"]
    p, row = M.tp_rows(L, U["event"][rep], T[rep, sl_s[first]])
    nan = np.float32(np.nan)

    def kin(*pairs):
        """(hit column, L1TTP column) pairs -> one float32 column per pair."""
        out = []
        for hc, lc in pairs:
            lv = np.where(row, L[lc][p], nan) if len(L["key"]) else np.full(len(uk), nan)
            out.append(np.where(own, U[hc][rep], lv).astype(np.float32))
        return out
    pt, eta, phi, d0, z0, vx, vy, charge = kin(
        ("tpPt", "pt"), ("tp_eta", "eta"), ("tp_phi_prod", "phi"), ("tp_d0", "d0"),
        ("tp_z0", "z0"), ("tp_vx", "vx"), ("tp_vy", "vy"), ("tp_charge", "charge"))
    return {"key": uk.astype(np.int64),
            "event": M.tp_key_parts(uk)[0].astype(np.int32),
            "pt": pt, "eta": eta, "phi": phi, "d0": d0, "z0": z0,
            "vr": np.hypot(vx, vy).astype(np.float32), "charge": charge,
            "hit_it": hit_it, "hit_ot": hit_ot, "hit_otd": hit_otd, "hit_itd": hit_itd, "n_mod_it": n_mod_it,
            "n_mod_ot": n_mod_ot, "n_mod_itd": n_mod_itd, "n_mod_otd": n_mod_otd}


# THE CENSUS EFFICIENCY DENOMINATOR (definition sheet census.findable; decisions of
# 2026-10-13 and 2026-10-14): a TP is FINDABLE when it is charged and carries
# credited hits on at least FINDABLE_MIN_MODULES distinct modules, IT clusters plus OT
# stubs in any mix: hits, not layers (two overlapping modules of one layer both
# count), but two or more credited hits of the TP on ONE module count once, and an
# OT stub is one hit. Charged as in CMSSW's L1TrackNtupleMaker (charge == 0
# skipped): a neutral TP (photon, K0L, neutron) can be credited with its unstored
# daughters' hits. The pT and |eta| selections are the consumer's.
FINDABLE_MIN_MODULES = 4


# the census columns format 36 added (their dtype: zeros when reading 34 / 35)
FORMAT36_COLS = {"hit_otd": np.uint8, "hit_itd": np.uint8, "n_mod_itd": np.int16, "n_mod_otd": np.int16}


def findable(C, barrel_only=False):
    """Boolean over the census TP table C: the findable TPs (FINDABLE_MIN_MODULES), over barrel and disc modules
    (format 36); `barrel_only`, the WP-Disc known-bad that counts the barrel's modules only."""
    n = C["n_mod_it"].astype(np.int64) + C["n_mod_ot"]
    if barrel_only:
        n = n - C["n_mod_itd"] - C["n_mod_otd"]
    return (n >= FINDABLE_MIN_MODULES) & np.isfinite(C["charge"]) & (C["charge"] != 0)


# THE CLOSED OT ARM (WP-D, census format 35; WP-Disc, format 36): the prompt OT seeds run through the CMSSW chain
# itself (ot_chain: multi-match with the match processor's rule and order, the exact hybrid KF, merge duplicate
# removal before the fit) instead of the census's projection, fit and score DR. OT-only by construction: its
# entries are the prompt OT seeds the chain runs (the four barrel doublets; with the discs on also D1D2, D3D4,
# L1D1 and L2D1) following the OT only. OT_CHAIN_DEFAULT is the census chain; the closed arm's options are its
# caps (TB 108, DR 108, 32 comparators) and its match order. "discs" (both modes) is ot_chain.set_discs: on, the
# OT discs and the WP-Disc corrections (format 36); off, format 35's barrel chain and population (gate G0).
OT_CHAIN_DEFAULT = {"mode": "census", "discs": True}
OT_CHAIN_MODES = ("census", "closed")
# the column of each OT census code (OL1-OL6 11-16, OD1-OD5 21-25) in a track's hit row (KF.LAYER_ORDER)
_OT_COL = {L: KF.LAYER_ORDER.index(L) for L in tuple(range(11, 17)) + OT_DISC_CODES}


def ot_chain_config(mode="census", caps=True, order="dtc", known_bad=None, discs=True):
    """The census's OT chain configuration (cache identity): mode census or closed; for closed, the caps
    (on / off) and the match order ("dtc" or "random:<seed>"); a known-bad variant (ot_chain.set_known_bad,
    the closure's known-bad runs only) in either mode; the discs on or off (ot_chain.set_discs)."""
    if mode not in OT_CHAIN_MODES:
        raise SystemExit(f"--ot-chain {mode!r}: one of {', '.join(OT_CHAIN_MODES)}")
    kb = {} if known_bad is None else {"known_bad": known_bad}
    if mode == "census":
        return {**OT_CHAIN_DEFAULT, "discs": bool(discs), **kb}
    if order == "dtc":
        return {"mode": "closed", "caps": bool(caps), "order": "dtc", "order_seed": 0, "discs": bool(discs), **kb}
    kind, _, sd = order.partition(":")
    if kind != "random" or not sd.isdigit():
        raise SystemExit(f"--ot-order {order!r}: 'dtc' or 'random:<seed>'")
    return {"mode": "closed", "caps": bool(caps), "order": "random", "order_seed": int(sd), "discs": bool(discs),
            **kb}


def disc_menu(cfg):
    """census_entries' `discs` of a census configuration: None with the discs off; else the build's IT discs (with
    --it-discs), the OT discs and the disc seeds (SA.disc_seeds, restricted to cfg["disc_seeds"])."""
    if not cfg["ot_chain"]["discs"]:
        return None
    it_d = M.IT_DISC_CODES if cfg["it_discs"] else ()
    chosen = None if cfg["disc_seeds"] is None else set(cfg["disc_seeds"])
    return {"it": it_d, "ot": OT_DISC_CODES,
            "seeds": SA.disc_seeds(it_d, SA.OT_BARREL + OT_DISC_CODES, chosen)}


def closed_entries(seeds, builds):
    """The closed arm's entries: the census entries restricted to the prompt OT seeds the chain runs (the barrel
    doublets; with the discs on also the disc and overlap doublets, census_entries' disc seeds), following the OT
    only (SystemExit otherwise), with each build's menu re-indexed."""
    keep = [i for i, sd in enumerate(seeds) if sd.layers in SA.OT_CHAIN_SEED and not sd.soft]
    bad = [seeds[i].tag for i in keep if any(L <= 10 for L in seeds[i].follow)]
    if bad or not keep:
        raise SystemExit("the closed OT arm is OT-only: run it with --masks IIII --targets OL1,...,OL6[,OD1,...,OD5] "
                         f"(entries following IT layers: {bad})")
    new = {old: k for k, old in enumerate(keep)}
    return [seeds[i] for i in keep], {m: [new[i] for i in es if i in new] for m, es in builds.items()}


# THE FINAL STUBS OF EVERY EXPORTED OT TRACK (shard keys hit_o<i>, and hit_om<i> for the closed arm's
# merge-DR tracks; int32): the event, then the L1TOTStub row in that event of the track's fitted stub on each
# OT barrel layer OL1-OL6 and (format 36) each OT disc OD1-OD5 (-1 none). From these and the input's own
# L1TOTStub table the closure (closure_vs_cmssw) builds both sides' per-track records with one function, the
# CMSSW tracks' from their fitted stubs alike.
OT_STUB_COLS = ("event",) + tuple(f"row_OL{L}" for L in range(1, 7)) + tuple(f"row_OD{d}" for d in range(1, 6))


def _stub_col(nano_layer):
    """The OT_STUB_COLS column of an L1TOTStub_layer (barrel 1-6, discs 11-15)."""
    return nano_layer if nano_layer <= 6 else nano_layer - 4


def _ot_rows(U, G):
    """OT_STUB_COLS rows of census tracks with hit rows G (census table indices in KF.LAYER_ORDER)."""
    out = np.full((len(G), len(OT_STUB_COLS)), -1, np.int64)
    if not len(G):
        return out
    first = np.where(G >= 0, G, np.iinfo(np.int64).max).min(axis=1)
    out[:, 0] = np.where(first < len(U["event"]), U["event"][np.minimum(first, len(U["event"]) - 1)], -1)
    for code, col in _OT_COL.items():
        g = G[:, col]
        out[:, _stub_col(code - 10)] = np.where(g >= 0, U["ot_row"][np.maximum(g, 0)], -1)
    return out


def _closed_rows(chain, ev, tracks, n_layers):
    """(hit rows, seed stubs, final stubs, stubs outside the census table) of closed-arm tracks: one row of
    census table indices per track in KF.LAYER_ORDER (the fitted stubs), the tracklet's two seed stubs, and
    the fitted stubs as OT_STUB_COLS (L1TOTStub rows, whether or not the census table has them)."""
    G = np.full((len(tracks), n_layers), -1, np.int64)
    S = np.full((len(tracks), 2), -1, np.int64)
    O = np.full((len(tracks), len(OT_STUB_COLS)), -1, np.int64)
    O[:, 0] = ev
    lay = chain.stub["layer"][np.searchsorted(chain.ev, ev):]
    outside = 0
    for i, t in enumerate(tracks):
        u = chain.to_u(ev, t.rows)
        outside += int((u < 0).sum())
        for r, g in zip(t.rows, u):
            L = int(lay[r])
            if g >= 0 and G[i, _OT_COL[L + 10]] < 0:
                G[i, _OT_COL[L + 10]] = g
            if O[i, _stub_col(L)] < 0:
                O[i, _stub_col(L)] = r
        S[i] = chain.to_u(ev, t.input_rows[:2])
    return G, S, O, outside


def process_chunk_closed(U, seeds, groups, export_tracks, ot_chain):
    """The closed OT arm on one chunk; the same outputs as process_chunk. found: TPs a seed's no-DR tracks
    recover (seed-truthful on the tracklet's two stubs, as the census), found_dr: those its merge-DR tracks
    recover. Exported tracks (shard keys hit_<entry>, hit_m<entry>; hit_s<entry>, hit_sm<entry>; hit_o<entry>,
    hit_om<entry>): the no-DR and the merge-DR tracks' fitted stubs, their tracklets' seed stubs and their
    final stubs as OT_STUB_COLS. No track rows, resolution sample or angle rows (census chain only)."""
    cen = chunk_census(U["truth"])
    ntp, nw = len(cen["key"]), (len(seeds) + 63) // 64
    found = np.zeros((ntp, nw), np.uint64)
    found_dr = np.zeros((ntp, nw), np.uint64)
    entry = {SA.OT_CHAIN_SEED[sd.layers]: i for i, sd in enumerate(seeds)}
    if "_ot_chain" not in U:
        U["_ot_chain"] = OC.ChunkChain(U["ot_in"], U["event"], U["ot_row"])
    chain = U["_ot_chain"]
    nl = len(KF.LAYER_ORDER)
    acc = {i: {"G": [], "S": [], "Gm": [], "Sm": [], "O": [], "Om": []} for i in range(len(seeds))}
    # the census's counter names, with the closed arm's meaning: pairs = tracklets, cand = before_chi2 =
    # track-builder candidates (the KF input), fit = no-DR tracks the KF accepts, before_dr = the DR input
    # (capped), owned = merge-DR tracks; trip / trip_true = no-DR tracks with both seed stubs in the census
    # table / seed-truthful (the fake column); stubs_outside_census: no-DR tracks' stubs the table lacks
    counters = {i: {k: 0.0 for k in ("pairs", "cand", "before_chi2", "fit", "before_dr", "owned", "trip",
                                     "trip_true", "nlayer_sum", "stubs_outside_census")}
                for i in range(len(seeds))}
    seed_eta = np.zeros((len(seeds), len(SA.SEED_ETA_EDGES) - 1), np.int64)
    ev0 = int(chain.events.min()) if len(chain.events) else 0
    nev_c = int(chain.events.max()) - ev0 + 1 if len(chain.events) else 0
    nonant = np.zeros((len(groups), nev_c * N_REGION), np.int32)
    for ev in chain.events:
        cp = chain.copies(ev)
        col = {}
        nodr, dr = OC.run_copies(cp, ot_chain["caps"], ot_chain["order"],
                                 [ot_chain["order_seed"], int(ev)], collect=col)
        for sd, i in entry.items():
            trs = [t for t in col.get("tracklets", []) if t.seed == sd]
            counters[i]["pairs"] += len(trs)
            seed_eta[i] += SA.seed_eta_hist([t.t for t in trs])
            nc = sum(1 for c in col.get("candidates", []) if c.tracklet.seed == sd)
            counters[i]["cand"] += nc
            counters[i]["before_chi2"] += nc
            counters[i]["before_dr"] += sum(1 for c in col.get("dr_input", []) if c.tracklet.seed == sd)
        for c in col.get("candidates", []):
            tr = c.tracklet
            eta = np.arcsinh(cp.z[tr.inner] / max(math.hypot(cp.x[tr.inner], cp.y[tr.inner]), 1e-6))
            es = int(np.clip(np.searchsorted(np.asarray(ETA_SECTOR_EDGES), eta), 0, N_ETA_SECTOR - 1))
            nonant[0, ((int(ev) - ev0) * N_NONANT + tr.sector) * N_ETA_SECTOR + es] += 1
        for tracks, kg, ks, ko, kc in ((nodr, "G", "S", "O", "fit"), (dr, "Gm", "Sm", "Om", "owned")):
            for sd, i in entry.items():
                mine = [t for t in tracks if t.seed == sd]
                G, S, O, outside = _closed_rows(chain, ev, mine, nl)
                acc[i][kg].append(G)
                acc[i][ks].append(S)
                acc[i][ko].append(O)
                counters[i][kc] += len(mine)
                if kc == "fit":
                    counters[i]["stubs_outside_census"] += outside
                    ok = (S >= 0).all(axis=1)
                    counters[i]["trip"] += float(ok.sum())
                    counters[i]["trip_true"] += float(M.seed_tp(U, S[ok, 0], S[ok, 1])[1].sum())
                    counters[i]["nlayer_sum"] += float((G >= 0).sum())
    hits = {}
    for i in range(len(seeds)):
        a = acc[i]
        for kg, ks, bitmap, tag in (("G", "S", found, ""), ("Gm", "Sm", found_dr, "m")):
            G = np.concatenate(a[kg]) if a[kg] else np.zeros((0, nl), np.int64)
            S = np.concatenate(a[ks]) if a[ks] else np.zeros((0, 2), np.int64)
            ok = (S >= 0).all(axis=1)
            rk = M.recovered_keys(U, S[ok, 0], S[ok, 1])
            if len(rk):
                bitmap[np.searchsorted(cen["key"], rk), i >> 6] |= np.uint64(1) << np.uint64(i & 63)
            if export_tracks:
                hits[i if not tag else f"m{i}"] = G.astype(np.int32)
                hits[f"s{i}" if not tag else f"sm{i}"] = S.astype(np.int32)
                O = a["O" if not tag else "Om"]
                hits[f"o{i}" if not tag else f"om{i}"] = (np.concatenate(O) if O else np.zeros(
                    (0, len(OT_STUB_COLS)), np.int64)).astype(np.int32)
        counters[i]["found_standalone"] = float(((found[:, i >> 6] >> np.uint64(i & 63)) & np.uint64(1)).sum())
    return cen, found, found_dr, counters, {}, {}, hits, {}, nonant, seed_eta


def process_chunk(U, Q, seeds, groups, ptmin, ci, qual_per_chunk, kf_opts=None,
                  export_tracks=0, do_dr=True,
                  min_layers=SA.MIN_LAYERS, d0_window_cm=0.0, it_ptmin=None,
                  min_it_layers=None, min_ot_conf=0, ang=None, pos_arb=None,
                  entry_builds=None, ot_chain=None):
    """Run every entry over chunk `ci` and reduce to census + bitmap + counters.

    `seeds` are the census entries (census_entries: a seed and the layers it
    follows into), `groups` the entry sets duplicate removal runs over
    (dr_groups). Entries of one seed share its formation and projections
    (SA.follow_seed); each applies the layer rule to its own follow set.

    `entry_builds` names each entry's S6 build bin (entry_build_key).

    `ang` (M.AngleUse; None = every stage) says where the SmartPixels angles
    are used: the seeding stages S1-S4 inside the follow, the KF update S5, the
    acceptance S6 and the DR rank S7 here (S8 is the exporters' feature set).
    `pos_arb` is the position part of every one-hit-per-pair choice (M.POS_ARB).

    TWO PASSES, because duplicate removal is inherently cross-seed. The first
    pass runs each entry through pairing, projection, the 4-layer rule and the
    KF chi2 acceptance and holds its survivors. DR then runs once per group over
    all of its entries together -- several seeds finding the same track is
    exactly what it exists to collapse -- and only then is each entry credited
    with what survived.
    """
    if ot_chain is not None and ot_chain.get("mode") == "closed":
        return process_chunk_closed(U, seeds, groups, export_tracks, ot_chain)
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
    ang = M.AngleUse(M.DEFAULT_STAGES) if ang is None else ang
    # every fit of the census: angles in the update only under S5, with its
    # angle set and table key (the census angle model's sigmas)
    fopt = {"use_angles": ang("kf"), "angles": tuple(sorted(ang.angles["kf"])),
            "key": ang.keys["kf"], "model": ang.model,
            **{k: v for k, v in kf_opts.items() if k == "d0_prior_cm"}}
    s5_state = ang.subset("kf") if ang("kf") else "off"
    s5_key = ang.keys["kf"] if ang("kf") else "emitted"
    # S1 RECORDS, once per chunk and floor: the test is per cluster, so every
    # seed with that floor applies the same one
    if ang.rec is not None and ang("veto"):
        floors = {float(ptmin)} | ({float(it_ptmin)} if it_ptmin is not None else set())
        for fl in sorted(floors):
            ang.rec.s1(fl, None if ang.cuts_open else ang.cuts.s1_thr(fl, Q["row"]))
            if "ldir_ok" in Q:                          # the IT discs (format 36)
                ang.rec.s1d(fl, None if ang.cuts_open else ang.cuts)
    cen = chunk_census(U["truth"])
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
    counters, qual, trk, hits, angr = {}, {}, {}, {}, {}
    # SEED MULTIPLICITY BY |eta|: every seed object a seed forms (the funnel's
    # seed_objects), binned on SA.SEED_ETA_EDGES. Counts, so chunks add.
    seed_eta = np.zeros((len(seeds), len(SA.SEED_ETA_EDGES) - 1), np.int64)

    # ---- pass 1: seed, follow, fit, chi2-accept --------------------------
    TP = KF.tp_truth_table(U)
    held = {}
    for s_i, o in entry_results(U, Q, seeds, _ptmin_for, _rule_for, min_layers,
                                d0_window_cm, ang, pos_arb):
        sd = seeds[s_i]
        pt_s = _ptmin_for(sd)
        if isinstance(o, M.TooWide):
            counters[s_i] = {"infeasible": 1.0, "n": float(o.n)}
            continue
        if o is not None:
            seed_eta[s_i] = o["seed_eta_hist"]
        if o is None or not o.get("tracks_to_fit"):
            counters[s_i] = {"pairs": 0.0, "cand": 0.0, "fit": 0.0}
            continue
        n_pre = int(o["tracks_to_fit"])
        gidx_all = KF.hits_from_seed(U, o)
        gc0 = o.get("_gC")
        trip0 = (o["_gA"], o["_gB"], gc0 if gc0 is not None else o["_gB"])
        fit0 = KF.fit_tracks(U, Q, trip0, gidx=gidx_all, **fopt)
        keep, chi2s = KF.good_state(fit0, pt_s)
        # the NaN guard on S6's statistic, cut or feature (the exported chi2_angle_per_meas), among the fits
        # that are tracks: a diverged fit (KF.fit_tracks) is rejected as such and counted apart
        ang.count_nonfinite("S6", fit0["chi2_angle"][fit0["ok"] & (fit0["n_angle_meas"] > 0)])
        base = {"arity": float(sd.arity), "before_chi2": float(n_pre),
                "kf_diverged": float(fit0["diverged"].sum()),
                "seed_objects": float(o.get("seed_objects", 0)),
                "pairs": float(o.get("tracklets", 0)),
                # cluster pairs EXAMINED (after the S2j pre-gate, before the
                # pair gates): the pair stage's work
                "pairs_phi": float(o.get("pairs_phi", 0)),
                "s3_rejected": float(o.get("angle_window_rejected", 0)),
                "s4w_rejected": float(o.get("follow_angle_rejected", 0)),
                "cand": float(o.get("cand_all_targets", 0)),
                "before_minlayers": float(o.get("tracks_before_minlayers", 0)),
                "fit": 0.0}
        if ang("accept"):
            # S6, after the position chi2 acceptance: the final-state angle chi2
            # over the stage's angles, cut at the cut file's threshold for the
            # entry's build, its number of angle measurements and whether one is
            # a disc's (CA.s6_group; KF.angle_gate)
            sub6 = ang.subset("accept")
            c2, n6 = CA.s6_stat(fit0, sub6)
            disc6 = (fit0["hits"]["VALID"] & fit0["hits"]["DANG"]).any(axis=1)
            k6 = (np.ones(len(c2), bool) if ang.cuts_open else
                  KF.angle_gate(c2, n6, ang.cuts.s6_thr(entry_builds[s_i], s5_state,
                                                          s5_key, sub6, n6, disc6)))
            if ang.rec is not None:
                CA.record_s6(ang.rec, U, Q, fit0, keep, k6, gidx_all, trip0, pt_s,
                             entry_builds[s_i], s_i, c2, n6, fopt)
            base["s6_rejected"] = float((keep & ~k6).sum())
            keep = keep & k6
        if not keep.any():
            counters[s_i] = base
            continue
        w_ang = ang.dr_weight[0 if sd.arity == 2 else 1] if ang("dr") else 0.0
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
        # seed-truthful: one TP credited by every seed hit (M.seed_tp)
        real = M.seed_tp(U, *tripk)[1]
        base.update({
            "fit": float(o_keep["tracks_to_fit"]),
            "found_standalone": float(len(rk0)),
            "chi2_scaled_sum": float(chi2s[keep].sum()),
            "chi2_angle_sum": float(fit0["chi2_angle"][keep].sum()),
            "n_angle_sum": float(fit0["n_angle"][keep].sum()),
            "nlayer_sum": float((sd.arity + o_keep["_nconf"]).sum()),
            # NOT comparable across arity: a doublet is credited when its two
            # seed clusters share a TP, a triplet when all three do.
            "trip": float(len(real)), "trip_true": float(real.sum())})
        counters[s_i] = base
        rng = entry_rng(ci, sd)
        idx = np.flatnonzero(real)
        if len(idx):
            if len(idx) > qual_per_chunk:
                idx = idx[rng.choice(len(idx), qual_per_chunk, replace=False)]
            tq = tuple(x[idx] for x in tripk)
            fq = KF.fit_tracks(U, Q, tq, gidx=gk[idx], **fopt)
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
            ff = KF.fit_tracks(U, Q, tf, gidx=gk[ks], **fopt)
            trk[s_i] = KF.track_rows(U, Q, Tf, ff, tf, TP, s_i, sd.arity,
                                     KF.sysclass_of(sd.layers), h_score[ks])
            # HIT LISTS, for duplicate removal the way a real system does it.
            # DR conflicts on >= 3 SHARED HITS, which cannot be reconstructed
            # from fitted parameters, so the cluster indices have to travel with
            # the track. 10 int32 per track, ~500 MB over the full sample --
            # affordable, and it keeps min_shared (and the criterion itself)
            # changeable without re-running the seeding.
            hits[s_i] = gk[ks].astype(np.int32)
            # each track's angle-bearing class-C clusters (the CNN stand-in's; census_scan: fakes split by class C)
            Th = ff["hits"]
            gi = np.clip(Th["GIDX"], 0, None)
            ncc = (Th["VALID"] & Th["ANG"] & ~Q["is_T"][gi]).sum(axis=1)
            if "DANG" in Th and "lis_T" in Q:
                ncc = ncc + (Th["VALID"] & Th["DANG"] & ~Q["lis_T"][gi]).sum(axis=1)
            hits[f"nc{s_i}"] = ncc.astype(np.int16)
            # the final stubs as L1TOTStub rows (OT_STUB_COLS), for the closure
            hits[f"o{s_i}"] = _ot_rows(U, gk[ks]).astype(np.int32)
            # PER-CLUSTER ANGLE RESIDUALS, on a sample. Four IT hits per track
            # over 12.5M tracks would be a 50M-row table; one track in
            # ANGLE_SAMPLE keeps it to ~100 MB, which is ample for a residual
            # distribution and is stated on the page.
            asel = np.arange(0, len(ks), ANGLE_SAMPLE)
            if len(asel):
                ta = tuple(x[ks][asel] for x in tripk)
                Ta = KF.gather_hits(U, Q, gk[ks][asel], KF.LAYER_ORDER)
                fa = KF.fit_tracks(U, Q, ta, gidx=gk[ks][asel], **fopt)
                angr[s_i] = KF.angle_rows(U, Ta, fa, ta, asel.astype(np.int64))
        held[s_i] = {"o": o_keep, "sd": sd, "score": h_score,
                     "ks": ks if export_tracks else None}
        del o, fit0

    # ---- phi-nonant occupancy, on the collection that ENTERS DR ---------
    # Binned on the INNER SEED CLUSTER's azimuth, because a tracklet is
    # processed in the sector its seeding stub falls in -- not on the fitted
    # phi0, which is a property of the track rather than of where the hardware
    # handled it. Stored as counts per (DR group, event, nonant) rather than as
    # a pass/fail at 108, so the cap can be varied afterwards without
    # re-running. Per DR group, i.e. per build: one system's load.
    ev0 = int(U["event"].min()) if len(U["event"]) else 0
    nev_c = int(U["event"].max()) - ev0 + 1 if len(U["event"]) else 0
    nonant = np.zeros((len(groups), max(nev_c, 0) * N_REGION), np.int32)
    for g, members in enumerate(groups):
        keys = [i for i in members if i in held]
        if keys and nev_c > 0:
            gA = [held[i]["o"]["_gA"] for i in keys]
            ev_all = np.concatenate([U["event"][x] for x in gA])
            ph_all = np.concatenate([U["globalPhi"][x] for x in gA])
            # eta of the seed's inner cluster, the same object the phi bin uses
            et_all = np.concatenate([np.arcsinh(U["globalZ"][x]
                                                / np.maximum(U["globalR"][x], 1e-6))
                                     for x in gA])
            # the clip on the integer: in float32 (the azimuth's type) the bound N_NONANT - 1e-9 rounds to
            # N_NONANT, so an azimuth at +pi gave nonant 9
            nz = np.clip((((ph_all + np.pi) % (2 * np.pi)) / (2 * np.pi / N_NONANT)).astype(np.int64),
                         0, N_NONANT - 1)
            es = np.searchsorted(np.asarray(ETA_SECTOR_EDGES), et_all)
            es = np.clip(es, 0, N_ETA_SECTOR - 1).astype(np.int64)
            loc = ev_all - ev0                   # chunk-local event index
            ok_ = (loc >= 0) & (loc < nev_c)
            nonant[g] = np.bincount(((loc[ok_] * N_NONANT + nz[ok_]) * N_ETA_SECTOR
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
    for s_i in sorted(held):
        h = held[s_i]
        dr = h["dr"]
        counters[s_i]["owned"] = float(dr.sum())
        counters[s_i]["before_dr"] = float(len(dr))
        if h["ks"] is not None:
            # which exported tracks survive DR (shard key hit_dr<entry>)
            hits[f"dr{s_i}"] = dr[h["ks"]].astype(np.int8)
        if not dr.any():
            continue
        o = SA.filter_tracks(h["o"], dr)
        rkeys = SA.recovered_keys(U, o)
        if len(rkeys):
            p = np.searchsorted(cen["key"], rkeys)
            found_dr[p, s_i >> 6] |= np.uint64(1) << np.uint64(s_i & 63)
    return cen, found, found_dr, counters, qual, trk, hits, angr, nonant, seed_eta


def truncation_table(C, group, cap=TRACKS_PER_REGION):
    """What the per-region output ceiling would cost DR group `group`'s menu
    (C["dr_groups"]; one build, or the builds sharing its entries)."""
    occ = np.asarray(C["nonant"][group], np.int64)
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
          it_ptmin=None, min_it_layers=None, min_ot_conf=0, ang=None,
          model=None, cuts=None, cuts_open=False, pos_arb=None, ot_chain=None, it_discs=True, disc_seeds=None,
          angle_null=None):
    """Build (or load) the census. Returns a dict of concatenated arrays.

    `it_discs`: the build's SmartPixels discs ID1-ID4 on (their clusters in the hit table; with the discs on only,
    ot_chain["discs"]); the findable count holds them whenever the discs are on. `disc_seeds`: the disc seeds'
    tags to run (SA.disc_seeds; None: all).

    `ot_chain` is the OT chain configuration (ot_chain_config; None = the census
    chain): the closed arm runs the prompt OT doublets through the CMSSW chain
    (process_chunk_closed) and is OT-only (closed_entries).

    `pos_arb` is the position arbitration (M.POS_ARB; None = POS_ARB_DEFAULT),
    `ang` the M.AngleUse (None = every stage), `model` the angle model
    (census_angles.Model; None = the locked table at k = 1), `cuts` its cut file
    (census_angles.CutSet) or `cuts_open` (the calibration run: every cutting
    stage records and passes everything). All are in the identity; a cut file
    made for another configuration is refused (CutSet.check).

    The cache is a DIRECTORY OF PER-CHUNK SHARDS, not one file, so a run that
    dies or is killed at chunk 40 of 63 resumes at 41 instead of starting over.
    A thousand events of ninety seeds is hours of work; it must be restartable.
    That also makes the RSS ceiling survivable: ru_maxrss is a high-water mark
    and never falls, so one runaway seed aborts the job permanently -- but every
    shard already written stays valid and the rerun resumes past it.
    """
    # the input contract before anything is written: a refused input leaves no
    # cache directory behind
    require_census_inputs(M.expand_inputs(spec))
    man = input_manifest(spec, hash_content)
    # nev is deliberately NOT part of the identity. Shards are per-chunk and
    # event order is fixed, so a 100-event cache is the first seven chunks of the
    # 1000-event one: asking for more RESUMES rather than rebuilding, and asking
    # for less reads a prefix.
    # kf_opts IS part of the identity. It changes the quality columns in every
    # shard, so a cache built with one angle weighting or d0 prior must not be
    # silently extended by a run using another.
    ang = M.AngleUse(M.DEFAULT_STAGES) if ang is None else ang
    pos_arb = POS_ARB_DEFAULT if pos_arb is None else pos_arb
    model = CA.Model() if model is None else model
    ot_chain = dict(OT_CHAIN_DEFAULT) if ot_chain is None else ot_chain
    # a known-bad census (WP-D closure) runs with its variant switched on, and the OT discs on or off (format
    # 36 / 35's population and chain), before the cache is read or built
    try:
        OC.set_known_bad(ot_chain.get("known_bad"))
    except ValueError as e:
        raise SystemExit(str(e)) from None
    OC.set_discs(ot_chain["discs"], fixes=ot_chain["discs"])
    if bench is not None and "prediction" in ang.keys.values():
        raise SystemExit("the prediction table key reads unquantised angles: not with --alpha-bits")
    cut_stages = [s for s in ("veto", "pair_alpha", "pair_z0", "pair_joint", "attach",
                              "proj_window", "accept") if ang(s)]
    if cut_stages and cuts is None and not cuts_open:
        raise SystemExit(f"angle cutting stages {cut_stages} need --angle-cuts (or --calibrate)")
    if cuts is not None:
        cuts.check(model, cut_identity(ang, bench, kf_opts, ptmin, it_ptmin, masks,
                                       bool(it_discs and ot_chain["discs"])))
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
           # how a projection picks one hit among its candidates: position
           # policy, the same for every angle configuration
           "pos_arb": pos_arb,
           # WHERE THE ANGLES ARE USED, with which angles and table keys; the
           # angle model (the locked table, k per angle, the sigma mode) and the
           # cut file the cutting stages read ("open": the calibration run)
           "angles": ang.key(),
           "angle_model": model.identity(),
           "angle_cuts": ("open" if cuts_open else None if cuts is None else
                          {"path": cuts.path, "sha256": cuts.sha,
                           "registry": cuts.ident.get("registry"),
                           "identity": cuts.ident}),
           # the impact-parameter allowance the projection windows carry. A
           # doublet assumes d0 = 0, so at r = 3 cm a real 1 mm displacement
           # throws the azimuth by 33 mrad and falls outside the window: this is
           # the knob that tests whether that is where displaced tracks are lost
           "d0_window_cm": float(d0_window_cm),
           "it_ptmin": None if it_ptmin is None else float(it_ptmin),
           # the build's IT discs (format 36): their clusters in the hit table; the disc seeds run (None: all)
           "it_discs": bool(it_discs and ot_chain["discs"]),
           "disc_seeds": None if disc_seeds is None or not ot_chain["discs"] else sorted(disc_seeds),
           # the OT chain: the census's, or the closed arm (CMSSW's) with its caps and match order
           "ot_chain": dict(sorted(ot_chain.items())),
           # the information-free control (census_scan NULL): None, or "permute"
           "angle_null": angle_null}
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
        # DEDICATED ENTRIES, APPENDED. The standard entries are left exactly as
        # they are; --it-ptmin adds a parallel set (each build's IT-only seeds)
        # that runs at the lower floor under the per-system rule. Switching the
        # extension on and off is then a seed selection, and the 2 GeV numbers
        # cannot move.
        seeds, builds = census_entries(cal["r_median"], masks, targets,
                                       seed_classes, it_ptmin, disc_menu(cfg))
        if ot_chain["mode"] == "closed":
            seeds, builds = closed_entries(seeds, builds)
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
                # the layers each entry follows into, and each build's entries
                "seed_follow": [list(sd.follow) for sd in seeds],
                "builds": builds,
                "dr_groups": dr_groups(builds, len(seeds)),
                "seed_eta_edges": list(SA.SEED_ETA_EDGES),
                # the shards' track column names, for readers without this code
                "track_cols": list(KF.TRACK_COLS),
                # what the angle figures of this census can and cannot say
                "angle_notes": {"discs": CA.DISC_NOTE, "class_c": CA.CLASS_C_NOTE},
                "chunks_done": 0, "n_events": 0}
        json.dump(meta, open(mpath, "w"), indent=1, default=float)
    seeds = [SA.Seed(tuple(x), soft=bool(s), follow=f)
             for x, s, f in zip(meta["seeds"], meta["seed_soft"],
                                meta["seed_follow"])]
    sigz = {int(k): float(v) for k, v in meta["calibration"]["sigz_ot"].items()}
    # Set on BOTH paths -- fresh build and cache resume -- because the OT bend
    # gate raises rather than silently falling back to the nominal slope, and a
    # resume that skipped this would fail at the first OT doublet.
    obc = meta.get("ot_bend_cal")
    if obc:
        M.set_ot_bend_cal({int(k): tuple(v) for k, v in obc.items()})
    ang.model, ang.cuts, ang.cuts_open = model, cuts, cuts_open
    entry_builds = [entry_build_key([m for m, es in meta["builds"].items() if i in es])
                    for i in range(len(seeds))]
    if verbose:
        print(f"cache {d}\n  {len(seeds)} entries ({len({(s.layers, s.soft) for s in seeds})} "
              f"seeds) over {len(meta['builds'])} build(s), chunk = {chunk} events, "
              f"{meta['chunks_done']} chunk(s) already done", flush=True)

    done = meta["chunks_done"]
    nev_seen = meta["n_events"]
    need = None if nev is None else max(0, nev - nev_seen)
    if mode != "off":
        t0 = time.time()
        M.set_limits(M.triplets_for_budget(budget_gb),
                     min(rss_gb, M.SAFE_RSS_FRAC * M.PHYS_RAM_GB))
        want = None if nev is None else nev
        for ci, ((U, Q), ev) in enumerate(
                unified_chunks(spec, want, chunk, sigz, bench, model, cfg["it_discs"], angle_null)):
            if ci < done:
                del U, Q                  # resume re-reads, but never re-seeds
                continue
            if ang.regions != "both":
                Q = angle_regions(U, Q, ang.regions)
            ang.rec = CA.Recorder(U, Q, model)
            ang.nonfinite = {s: 0 for s in M.NONFINITE_STAGES}
            (cen, found, found_dr, counters, qual, trk, hits, angr, nonant,
             seed_eta) = process_chunk(
                U, Q, seeds, meta["dr_groups"], ptmin, ci, QUAL_PER_CHUNK, kf_opts,
                export_tracks, min_layers=min_layers,
                d0_window_cm=d0_window_cm, it_ptmin=it_ptmin,
                min_it_layers=min_it_layers, min_ot_conf=min_ot_conf, ang=ang,
                pos_arb=pos_arb, entry_builds=entry_builds, ot_chain=ot_chain)
            recs = ang.rec.arrays()
            nonfinite, ang.rec, ang.nonfinite = ang.nonfinite, None, None
            moved = np.array([[Q["n_moved"][a][f"IL{L}"] for L in (1, 2, 3, 4)]
                              for a in CA.ANGLES], np.int64)
            np.savez_compressed(
                _shard_path(d, ci), found=found, found_dr=found_dr,
                n_events=len(ev),
                counters=json.dumps({str(k): v for k, v in counters.items()}),
                **{f"cen_{k}": v for k, v in cen.items()},
                **{f"qual_{k}": v for k, v in qual.items()},
                # n_clusters lets load() offset the chunk-local cluster indices
                # into a global space; chunks are written independently and
                # resumably, so the offset cannot be baked in at write time
                n_clusters=len(U["layer"]), nonant=nonant, seed_eta=seed_eta,
                **{f"trk_{k}": v for k, v in trk.items()},
                **{f"hit_{k}": v for k, v in hits.items()},
                **{f"ang_{k}": v for k, v in angr.items()},
                # the angle model's records (census_angles.Recorder) and the
                # clusters the knob moved per angle x IL1-IL4
                **{f"rec_{c}__{k}": v for c, d_ in recs.items() for k, v in d_.items()},
                n_moved=moved, nonfinite=json.dumps(nonfinite))
            done, nev_seen = ci + 1, nev_seen + len(ev)
            meta["chunks_done"], meta["n_events"] = done, nev_seen
            json.dump(meta, open(mpath, "w"), indent=1, default=float)
            if verbose:
                el = time.time() - t0
                print(f"  chunk {ci:>4}  {len(ev):>3} ev  {len(cen['key']):>8,} TPs"
                      f"  {el:7.1f}s  peak {M.peak_gb():.2f} GB"
                      + (f"  clusters moved by the knob: alpha {int(moved[0].sum()):,} "
                         f"beta {int(moved[1].sum()):,} of {int((U['layer'] <= 4).sum()):,}"
                         if moved.any() else ""),
                      flush=True)
            del U, Q, cen, found
    return load(d, nev=nev, verbose=verbose)


def entry_build_key(masks):
    """The S6 build bin of a census entry: the builds whose menus hold it."""
    return "+".join(sorted(masks))


def cut_identity(ang, bench, kf_opts, ptmin, it_ptmin, masks, it_discs=False):
    """The run's identity an angle cut file is checked against (census_angles.CutSet.check); `it_discs`: the
    builds' IT discs are on, so the disc consumers of the stages the run cuts with need thresholds."""
    s5 = ang.subset("kf") if ang("kf") else "off"
    disc_stages = [c for c, st in (("s1d", "veto"), ("s2ad", "pair_alpha"), ("s2zd", "pair_z0"),
                                   ("wind", "proj_window")) if it_discs and ang(st)]
    return {"format": FORMAT_VERSION,
            "bench": list(bench) if bench else None,
            "d0_prior_cm": float((kf_opts or {}).get("d0_prior_cm", KF.D0_PRIOR_CM)),
            "level": CA.LEVEL,
            "floors": sorted({float(ptmin)} | ({float(it_ptmin)} if it_ptmin else set())),
            "s6": ((s5, ang.keys["kf"] if s5 != "off" else "emitted", ang.subset("accept"))
                   if ang("accept") else None),
            "win_subsets": sorted({ang.subset(s) for s in ("attach", "proj_window") if ang(s)}),
            "disc_stages": disc_stages, "it_discs": bool(it_discs),
            "builds": [entry_build_key([m]) for m in (masks or ["AAAA"])]}


def census_dir(path):
    """The census at `path`: `path` itself when it is one (holds manifest.json),
    else the ONLY tpcensus_* directory under it. More than one is refused: the
    names are hashes with no order, so taking the last of a sorted listing picks
    an arbitrary census -- e.g. the stale one left beside a rebuild."""
    if os.path.exists(os.path.join(path, "manifest.json")):
        return str(path)
    ds = sorted(Path(path).glob("tpcensus_*"))
    if not ds:
        raise SystemExit(f"no census under {path}")
    if len(ds) > 1:
        raise SystemExit(f"{len(ds)} censuses under {path} ({', '.join(d.name for d in ds)}); "
                         f"pass the one to use as the path itself")
    return str(ds[0])


def hit_key(name):
    """(kind, entry) of an exported-track shard key hit_<kind><entry>: kind "", "dr", "nc", "m", "s", "sm", "o" or
    "om"."""
    m = re.fullmatch(r"hit_([a-z]*)(\d+)", name)
    if m is None:
        raise SystemExit(f"unknown shard key {name}")
    return m.group(1), int(m.group(2))


def load(d, nev=None, verbose=True, with_tracks=False):
    meta = json.load(open(os.path.join(d, "manifest.json")))
    # ONLY READ_FORMATS ARE READ. A census of another format means other numbers
    # under the same names -- before 23, every build's seeds followed into all
    # of IL1-IL4 -- so it is refused, never reinterpreted.
    if meta.get("format") not in READ_FORMATS:
        raise SystemExit(f"{d} is census format {meta.get('format')}; this code "
                         f"reads {', '.join(map(str, READ_FORMATS))} only. Rebuild it.")
    shards = sorted(Path(d).glob("chunk_*.npz"))
    if not shards:
        raise SystemExit(f"{d} has no shards")
    cols = ["key", "event", "pt", "eta", "phi", "d0", "z0", "vr", "charge", "hit_it", "hit_ot", "hit_otd",
            "hit_itd", "n_mod_it", "n_mod_ot", "n_mod_itd", "n_mod_otd"]
    parts = {c: [] for c in cols}
    fnd, fdr, cnt, qs, fts, hts, ags, nread = [], [], {}, {}, {}, {}, {}, 0
    non = []
    moved = np.zeros((len(CA.ANGLES), 4), np.int64)
    nonfin = {}
    seta = None
    cl_tot = cl_off = 0
    for sh in shards:
        if nev is not None and nread >= nev:
            break                    # a prefix of the shards is a valid census
        z = np.load(sh, allow_pickle=False)
        for c in cols:
            # hit_otd / hit_itd / n_mod_itd / n_mod_otd from format 36; formats 34 / 35 hold no disc hit
            parts[c].append(z[f"cen_{c}"] if f"cen_{c}" in z.files or c not in FORMAT36_COLS
                            else np.zeros(len(z["cen_key"]), FORMAT36_COLS[c]))
        fnd.append(z["found"])
        fdr.append(z["found_dr"])
        cl_off = cl_tot
        cl_tot += int(z["n_clusters"])
        nread += int(z["n_events"])
        non.append(z["nonant"])
        moved += z["n_moved"]
        # the NaN guard's counts; None when a shard predates them (written before 2026-10-15)
        if "nonfinite" not in z.files:
            nonfin = None
        elif nonfin is not None:
            for k, v in json.loads(str(z["nonfinite"])).items():
                nonfin[k] = nonfin.get(k, 0) + v
        seta = z["seed_eta"] if seta is None else seta + z["seed_eta"]
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
                # hit_<i>: hit rows; hit_dr<i>: census-chain DR survivor flags; hit_nc<i>: each track's class-C
                # angle clusters (census_scan); hit_o<i>: the final stubs
                # (OT_STUB_COLS); closed arm: hit_m<i> the merge-DR tracks' rows, hit_s<i> / hit_sm<i> the seed
                # stubs (census table indices, offset globally), hit_om<i> the merge-DR final stubs
                kind, num = hit_key(f)
                h = z[f]
                hts.setdefault(kind, {}).setdefault(num, []).append(
                    h.astype(bool) if kind == "dr" else h.astype(np.int64) if kind in ("o", "om", "nc")
                    else np.where(h >= 0, h + cl_off, -1).astype(np.int64))
    C = {c: np.concatenate(parts[c]) for c in cols}
    C["found"] = np.concatenate(fnd, axis=0)
    C["found_dr"] = np.concatenate(fdr, axis=0)
    C["n_events"] = nread
    # per DR group (C["dr_groups"]): (group, event x region) occupancy
    C["nonant"] = np.concatenate(non, axis=1)
    # seeds formed per entry and |eta| bin, summed over the events read
    C["seed_eta"] = seta
    C["seed_eta_edges"] = meta["seed_eta_edges"]
    # the soft flag and the follow set have to survive the reload too, or the
    # report describes the recovery entries under the standard floor and rule,
    # or an entry without the build surfaces it ran on
    C["seeds"] = [SA.Seed(tuple(x), soft=bool(f), follow=fl)
                  for x, f, fl in zip(meta["seeds"], meta["seed_soft"],
                                      meta["seed_follow"])]
    C["seed_tags"] = meta["seed_tags"]
    C["seed_notes"] = meta["seed_notes"]
    # each build's entries, and the entry sets DR ran over
    C["builds"] = meta["builds"]
    C["dr_groups"] = meta["dr_groups"]
    C["calibration"] = meta["calibration"]
    # where the angles were used and at what resolution; the MVA feature set
    # follows S8 (KF.mva_features), so an exporter cannot train on angles the
    # census was told not to use
    C["angle_use"] = M.AngleUse.from_key(meta["config"]["angles"])
    C["pos_arb"] = meta["config"]["pos_arb"]
    # the angle model (table identity, k per angle, sigma mode), the cut file it
    # cut with ("open": the calibration run), what its angle figures cannot say,
    # and the clusters the knob moved per angle x IL1-IL4
    C["angle_model"] = meta["config"]["angle_model"]
    C["angle_cuts"] = meta["config"]["angle_cuts"]
    C["angle_notes"] = meta["angle_notes"]
    C["n_moved"] = {a: {f"IL{L}": int(moved[i, L - 1]) for L in (1, 2, 3, 4)}
                    for i, a in enumerate(CA.ANGLES)}
    C["angle_nonfinite"] = nonfin
    C["mva_features"] = KF.mva_features(C["angle_use"])
    C["ptmin"] = float(meta["config"]["ptmin"])
    C["min_layers"] = int(meta["config"]["min_layers"])
    mil = meta["config"]["min_it_layers"]
    C["min_it_layers"] = None if mil is None else int(mil)
    C["min_ot_conf"] = int(meta["config"]["min_ot_conf"])
    ipm = meta["config"]["it_ptmin"]
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
    hk = {"": "hits", "dr": "hits_dr", "m": "hits_merged", "s": "hits_seed", "sm": "hits_seed_merged",
          "o": "track_stubs", "om": "track_stubs_merged", "nc": "track_class_c"}
    for kind, name in hk.items():
        C[name] = {i: np.concatenate(v) for i, v in hts.get(kind, {}).items()}
    C["angles"] = {i: np.concatenate(v) for i, v in ags.items()}
    C["angle_cols"] = list(KF.ANGLE_COLS)
    C["track_cols"] = list(KF.TRACK_COLS)
    C["cache_dir"] = d
    C["format"] = int(meta["format"])
    if verbose:
        print(f"loaded census format {C['format']}, {len(fnd)} shard(s): {nread} events, "
              f"{len(C['key']):,} TPs, {len(C['seeds'])} entries, "
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


def entry_builds(C, i):
    """The builds whose menus hold entry i."""
    return [m for m, idx in C["builds"].items() if i in idx]


def _builds_label(builds):
    return ",".join(builds) if len(builds) <= 3 else f"{len(builds)} builds"


def seed_table(C):
    """Per entry: TPs found, cost per event, fake fraction, seed-level
    resolutions, and the builds and follow layers it ran for."""
    def _need_for(C, sd):
        """Confirmations this seed had to find, under whichever rule was set."""
        mil = C.get("min_it_layers")
        if mil is not None and C.get("it_ptmin") is not None and not sd.soft:
            mil = None          # the per-system rule applies to recovery seeds
        if mil is not None:
            return float(max(int(mil) - sum(1 for L in sd.layers if L <= 10), 0))
        ml = C.get("min_layers")
        return float(max(int(ml) - sd.arity, 0)) if ml is not None \
            else float(sd.min_proj())

    nev = max(C["n_events"], 1)
    rows = []
    for i, tag in enumerate(C["seed_tags"]):
        sd = C["seeds"][i]
        c = C["counters"].get(i, {})
        where = {"builds": entry_builds(C, i),
                 "follow": [SA.NAME[L] for L in sd.follow]}
        if c.get("infeasible"):
            rows.append({"seed": tag, "arity": sd.arity, **where,
                         "infeasible": True})
            continue
        q = C["qual"].get(i, np.zeros((0, 5), np.float32))
        rows.append({
            "seed": tag, "arity": sd.arity, **where, "note": C["seed_notes"][i],
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
                    help="layers to follow into, e.g. IL1,IL2,IL3,IL4,OL1,OL2,OL3, "
                         "within each build's own surfaces (an inactive IT "
                         "layer is never followed). "
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
                    help="activeSP builds to run: each build's own seeds, "
                         "followed into its own active IT layers + OT; 'none' "
                         "for the full IT (AAAA) alone")
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
                    help="quantise the alpha word (sin of the direction's "
                         "azimuth relative to the radial) to this many bits over "
                         "ALPHA_RANGE, once at input, for every stage; omit for "
                         "full float")
    ap.add_argument("--beta-bits", type=int, default=None,
                    help="same for the beta-derived per-cluster z0 word over "
                         "Z0_RANGE (every beta use derives from it). Both must "
                         "be given together.")
    stages = "; ".join(f"{v[0]}={k}: {v[1]}" for k, v in M.ANGLE_STAGES.items())
    ap.add_argument("--angle-stages", default="default",
                    help="where the SmartPixels angles are used: comma list of "
                         "stages (names or S-labels; S2/S4 = the family), "
                         "'default' (every stage but S5; S6's statistic is "
                         "exported per track either way), 'all' or 'none'. "
                         f"Stages: {stages}")
    ap.add_argument("--angle-stages-off", default="",
                    help="stages to remove from --angle-stages (leave-one-out)")
    ap.add_argument("--pos-arb", default=POS_ARB_DEFAULT, choices=M.POS_ARB,
                    help="how a projection picks one hit among its candidates: "
                         "rphiz = smallest (dphi/sigma)^2 + (dz/sigma)^2, dphi = "
                         "smallest |dphi| (before 2026-10-03; it ignores the z "
                         "residual, so beta in S4a gets credit for z information)")
    P = M.AngleUse.PARAMS
    ap.add_argument("--angle-sets", default="",
                    help="per-angle use of the stages that read both angles "
                         "(S3, S4w, S4a, S5, S6): e.g. 'S3=alpha,S5=beta,S6=' "
                         "(angles joined by '+'; empty = none); default both")
    ap.add_argument("--angle-keys", default="",
                    help="table key of the weighting stages: e.g. "
                         "'S4a=emitted,S5=prediction' (default "
                         + ", ".join(f"{k}={v}" for k, v in M.DEFAULT_KEYS.items()) + ")")
    ap.add_argument("--arb-weight", type=float, default=P["arb_weight"],
                    help="S4a weight of the angle pulls^2 against the "
                         "position pull^2 in the hit arbitration")
    ap.add_argument("--angle-table", default=CA.TABLE_DEFAULT,
                    help="the angle-uncertainty table (sheet angle.error_table; "
                         f"registry {CA.TABLE_REGISTRY})")
    ap.add_argument("--angle-table-disc", default=CA.DISC_TABLE_DEFAULT,
                    help="the disc module-frame angle table (registry "
                         f"{CA.DISC_TABLE_REGISTRY}; the disc stages, format 36)")
    ap.add_argument("--angle-cuts", default=None,
                    help="the angle cut file the cutting stages read "
                         "(census_angles.CutSet; refused if made for another "
                         "configuration)")
    ap.add_argument("--s1-at", default=None, choices=list(M.AngleUse.S1_AT),
                    help="where S1 acts: pairs (pair formation only, the physicist's ruling; default with --discs on) "
                         "or all (also the attach / follow targets; default with --discs off, format 35)")
    ap.add_argument("--angle-regions", default="both", choices=list(M.AngleUse.REGIONS),
                    help="whose angles the stages use: both, barrel (the IT disc clusters' angles off) or disc (the "
                         "barrel clusters' off); an off region's clusters keep every position role (census_scan)")
    ap.add_argument("--angle-null", default="none", choices=["none", "permute"],
                    help="the information-free control (census_scan NULL): every usable IT cluster's emitted angles "
                         "permuted among the event's usable clusters of its surface, its sigmas kept (fixed seed); the "
                         "shuffled angles feed the cuts only (hit arbitration and duplicate-removal ranking off)")
    ap.add_argument("--s1-d0max", type=float, default=None,
                    help="S1's displaced tolerance, cm: the allowed alpha band adds asin(D / r) (census_angles."
                         f"S1_D0MAX_CM); default {CA.S1_D0MAX_CM:g} with --discs on, 0 (format 35's prompt band) off")
    ap.add_argument("--calibrate", action="store_true",
                    help="the calibration run: every angle cutting stage records "
                         "its statistic and passes everything")
    ap.add_argument("--k-alpha", type=float, default=1.0,
                    help="angle improvement knob for alpha, 0 <= k <= 1 "
                         "(sheet angle.k_knob): alpha' = truth + k (alpha - truth) "
                         "on class T, class C as emitted")
    ap.add_argument("--k-beta", type=float, default=1.0, help="same for beta")
    ap.add_argument("--k-mode", default="fixed", choices=["fixed", "scaled"],
                    help="the uncertainty at k < 1: fixed, or scaled by "
                         "max(k, 0.05); both are reported (decision 10)")
    ap.add_argument("--d0-prior-cm", type=float, default=KF.D0_PRIOR_CM)
    ap.add_argument("--hash-content", action="store_true",
                    help="digest file contents instead of trusting size+mtime")
    ap.add_argument("--ot-chain", default="census", choices=OT_CHAIN_MODES,
                    help="census: the census's OT chain; closed: the prompt OT doublets "
                         "through the CMSSW chain (OT-only: --masks IIII --targets "
                         "OL1,...,OL6)")
    ap.add_argument("--ot-caps", default="on", choices=["on", "off"],
                    help="closed arm: the track builder, DR input and comparator caps")
    ap.add_argument("--ot-order", default="dtc",
                    help="closed arm: the match order, 'dtc' or 'random:<seed>'")
    ap.add_argument("--discs", default="on", choices=["on", "off"],
                    help="the OT discs (format 36): on, the disc stubs in the census table and the findable "
                         "count, the closed arm's disc and overlap seeds and disc projections, the WP-Disc "
                         "corrections; off, format 35's barrel population and chain (gate G0)")
    ap.add_argument("--disc-seeds", default="all",
                    help="the disc seeds to run (with --discs on), comma-separated tags, e.g. "
                         "ID1+ID2,ID2+ID3,ID3+ID4,ID1+OD1,OD1+OD2,OD3+OD4,OL1+OD1,OL2+OD1; all (default) or none")
    ap.add_argument("--it-discs", default="on", choices=["on", "off"],
                    help="the build's SmartPixels discs ID1-ID4 (with --discs on): their clusters in the hit "
                         "table for seeding and following")
    ap.add_argument("--ot-known-bad", default=None,
                    help="a known-bad variant of the OT chain or fit (ot_chain.CHAIN_KNOWN_BADS, "
                         "cmssw_kf.KF_KNOWN_BADS), for the WP-D closure's known-bad runs only")
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
    s1_at = a.s1_at if a.s1_at is not None else ("pairs" if a.discs == "on" else "all")
    ang = M.AngleUse.parse(a.angle_stages, a.angle_stages_off, a.angle_sets,
                           a.angle_keys, arb_weight=a.arb_weight, s1_at=s1_at, regions=a.angle_regions)
    if a.angle_null != "none":
        ang = null_angle_use(ang)
    if a.calibrate and a.angle_cuts:
        raise SystemExit("--calibrate runs with every cut open: no --angle-cuts")
    s1_d0max = a.s1_d0max if a.s1_d0max is not None else (CA.S1_D0MAX_CM if a.discs == "on" else 0.0)
    model = CA.Model(a.angle_table, a.k_alpha, a.k_beta, a.k_mode, disc_table_path=a.angle_table_disc,
                     s1_d0max=s1_d0max)
    cuts = CA.CutSet.load(a.angle_cuts) if a.angle_cuts else None
    print(f"angle stages: {ang.label()}; keys {ang.keys}; k alpha {a.k_alpha:g}, "
          f"beta {a.k_beta:g} ({a.k_mode}); cuts "
          f"{'open (calibration)' if a.calibrate else a.angle_cuts}; position "
          f"arbitration {a.pos_arb}", flush=True)
    C = build(a.input, a.nev, a.ptmin, layers, a.n_adjacent, a.chunk,
              a.cache_dir, a.cache, a.hash_content, a.calib_events, masks=masks,
              kf_opts={"d0_prior_cm": a.d0_prior_cm},
              ang=ang, model=model, cuts=cuts, cuts_open=a.calibrate,
              pos_arb=a.pos_arb,
              export_tracks=a.export_tracks, seed_classes=scl, targets=tgt,
              min_layers=a.min_layers, min_it_layers=a.min_it_layers,
              min_ot_conf=a.min_ot_conf,
              bench=((a.alpha_bits, a.beta_bits)
                     if a.alpha_bits and a.beta_bits else None),
              d0_window_cm=a.d0_window_cm, it_ptmin=a.it_ptmin,
              ot_chain=ot_chain_config(a.ot_chain, a.ot_caps == "on", a.ot_order, a.ot_known_bad,
                                       a.discs == "on"),
              it_discs=a.it_discs == "on", angle_null=None if a.angle_null == "none" else a.angle_null,
              disc_seeds=(None if a.disc_seeds == "all" else
                          [] if a.disc_seeds == "none" else [x.strip() for x in a.disc_seeds.split(",") if x.strip()]))
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
          f"{'s(kap)':>9}{'s(d0)um':>9}{'s(cot)':>9}{'s(z0)um':>9}  builds")
    for r in rows[:30]:
        print(f"{r['seed']:<20}{r['arity']:>3}{r['n_found']:>9,}{r['cand']:>11,.0f}"
              f"{r['fit']:>8,.0f}{r['n_found_dr']:>8,}"
              f"{100 * r['chi2_pass']:>5.0f}{100 * r['dr_pass']:>5.0f}"
              f"{r['chi2_angle_per_cl']:>8.1f}{r['nlayer']:>6.1f}"
              f"{r['fake']:>7.3f}{r['sig_kappa']:>9.4f}{1e4 * r['sig_d0_cm']:>9.0f}"
              f"{r['sig_cot']:>9.4f}{1e4 * r['sig_z0_cm']:>9.0f}  {_builds_label(r['builds'])}")
    # PER DR GROUP: one build's menu (or the builds sharing its entries) is one
    # system, and its load is what the ceiling applies to
    trunc = [{"builds": [m for m, idx in C["builds"].items()
                         if any(i in g for i in idx)],
              **truncation_table(C, k)}
             for k, g in enumerate(C["dr_groups"])]
    print(f"\n=== per-region load per build, against the tuned ceiling of "
          f"{TRACKS_PER_REGION} tracks per region per event "
          f"({N_NONANT} nonants x {N_ETA_SECTOR} eta sectors = "
          f"{N_REGION * TRACKS_PER_REGION}/event); fitted tracks entering DR ===")
    print(f"  {'build(s)':<20}{'fit/ev':>9}{'budget%':>9}{'max':>7}{'over cap':>10}"
          f"{'dropped%':>10}{'p50':>6}{'p99':>6}  dropped at 52/104/208/416")
    for t in trunc:
        print(f"  {_builds_label(t['builds']):<20}{t['per_event']:>9,.0f}"
              f"{100 * t['frac_of_budget']:>9.1f}{t['max_occ']:>7,d}"
              f"{t['n_over_bins']:>10,d}{100 * t['frac_dropped']:>10.2f}"
              f"{t['pct'][50]:>6.0f}{t['pct'][99]:>6.0f}  "
              + "/".join(f"{100 * f:.2f}%" for f in t["scan"].values()))
    print("  NOTE: the cut is arbitrary with respect to quality -- there is "
          "no ranking where it is applied, so a good displaced track is as "
          "likely to go as a fake. The eta-sector boundary is ASSUMED at "
          "eta = 0 (L1TTrack_etaSector is unfilled in these ntuples); that "
          "redistributes load between two bins but cannot change the total.")
    if C["angle_nonfinite"] and any(C["angle_nonfinite"].values()):
        print(f"\nWARNING: non-finite angle statistics (the NaN guard; a NaN vetoes silently): "
              f"{C['angle_nonfinite']}")
    diverged = {}
    for i, c in C["counters"].items():
        if c.get("kf_diverged"):
            diverged[C["seed_tags"][i]] = diverged.get(C["seed_tags"][i], 0) + int(c["kf_diverged"])
    if diverged:
        print(f"\nNOTE: KF fits that diverged (finite inputs, a non-finite state; rejected, not tracks), per seed "
              f"over its builds: {diverged}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_events": nev, "n_tp": int(len(C["key"])),
                   "pt_min": a.ptmin, "cache_dir": C["cache_dir"],
                   "pos_arb": C["pos_arb"],
                   "angles": C["angle_use"].key(), "angle_model": C["angle_model"],
                   "angle_cuts": C["angle_cuts"], "angle_notes": C["angle_notes"],
                   "n_moved": C["n_moved"], "angle_nonfinite": C["angle_nonfinite"], "kf_diverged": diverged,
                   "truncation": trunc,
                   "seeds": seed_table(C)}, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
