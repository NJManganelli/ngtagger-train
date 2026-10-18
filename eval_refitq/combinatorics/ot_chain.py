"""Emulation of the prompt hybrid OT track finder (barrel seeds, barrel targets) from the DTC frames.

THE CHAIN, as CMSSW_20_1_0_pre1 runs it for L1TTrack, stage by stage, and what is emulated:

  inputs     L1TOTStubDTC (ot_dtc): every stub copy per processing region (nonant = sector) in stream
             order, with its integer phi / z / r and bend code. A stub in no stream was dropped by the
             DTC. The phi correction (Sector.cc:60-78, VMPhiCorrL<n>) is applied exactly.
  seeding    per sector and barrel seed (L1L2, L2L3, L3L4, L5L6): the VMRouter cuts on the inner stub
             (VMRouterCM.cc, integer |z|), the tracklet engine's r-z window (TP_<seed> tables, VMRME) and
             stub-pT LUTs (TP_<seed> stubpt inner / outer tables, indexed by the fine-phi difference and the
             bend code), and the tracklet calculator IN INTEGERS (TrackletCalculatorBase calcPars from the
             DTC words, then barrelSeeding's gates: |irinv| below the 1.91 GeV cut, |iz0| < 15 cm (L1L2) or
             22.5 cm, and the integer inSector window at rcrit = 55 cm). The float exact two-stub helix
             (exacttracklet) is kept beside the integers: it is the KF prior, as in CMSSW.
  matching   the integer projections (ProjectionCalculator::projLayer from the tracklet's integer parameters,
             at the layer's rmean, with its validity and edge cuts) go to the match processors as wired
             (per layer and allstub region; L1L2's to the duplicated processors of L3-L6 regions B, C), which
             run clock by clock (match_processor: the input-memory reading, the projection buffer, four match
             engines with their pipelines and candidate buffers, one match calculation per clock, 108 clocks
             with caps on): the engines read the VM memory bins (fine phi as CMSSW computes it, fine z, the
             bend table METable_L<n> indexed by the projection's 5-bit rinv), the calculator takes the integer
             residual with its r corrections, cut by MP_L<n>PHIA phicut / zcut by seed in the IMPROVING CHAIN
             (MatchProcessor.cc:684-714): kept iff |dphi| <= best and -best <= dz < best, which becomes the
             best. The VM memory order is the VMRouter's reading order (the `dtc` order mode) or a seeded
             random permutation within each bin (`random`).
  builder    a tracklet with matches in >= 2 layers becomes a candidate; its stub list is the two seed
             stubs, then the matches by layer (FitTrack.cc:67-89). Two track builders per sector, seeds
             {L1L2, L2L3, L5L6, L2D1} and {L3L4, D1D2, D3D4, L1D1}, each processing tracklets in TCID
             order (seed, then tracklet order) up to 108 (caps on).
  fit        cmssw_kf: the exact port of the hybrid KF.
  no-DR      every candidate is fitted on its own list (RemovalType = "").
  merge DR   PurgeDuplicate (merge): per sector, the candidates of builder 0 then builder 1 up to 108,
             ordered by the static seed rank {1, 5, 2, 7, 4, 3, 8, 6}; CompareBest on the best stub per
             layer by |phi_stub - phi_proj(rmean)| (0 for the seed layers); duplicates when >= 3 layers
             share their best stub; only the first 32 unmerged candidates compare; the preferred
             candidate takes the rejected one's new stubs; each survivor is fitted once on its merged
             list with its own tracklet.

CAPS (input_caps, projection_pages, build_candidates, merge_dr), each in the finder's reading order (the
VMRouter's: vmr_order). The step caps, switch `caps`: IR 156 per DTC link, VMR 108, TB 108, DR-input 108 and
the 32 comparators. The memory sizes, always: 128 per VMRouter, 15 per VM bin of the tracklet-engine and
match-engine memories, 63 projections per page; and MP 108 (the clocks of match_processor; caps off:
until drained). The DTC's own truncation is in the table. Not emulated: TP 108 steps / 127 tracklets and
PC 108; the no-truncation sibling production measures them.
"""
from __future__ import annotations
import dataclasses
import hashlib
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

import cmssw_kf as KF
import ot_dtc as D

LUT_DIR = Path(__file__).resolve().parent / "luts"
# THE 8 PROMPT SEEDS (Settings.h:54, 679-686; Seed enum order), as (inner, outer) layer / disk codes: barrel
# layers 1-6, discs D1-D5 7-11 (ot_dtc). The disc (D1D2, D3D4) and overlap (L1D1, L2D1) seeds run with the
# discs on (set_discs); with them off the chain is census format 35's barrel chain, unchanged.
SEEDS = {0: (1, 2), 1: (2, 3), 2: (3, 4), 3: (5, 6), 4: (7, 8), 5: (9, 10), 6: (1, 7), 7: (2, 7)}
SEED_NAME = {0: "L1L2", 1: "L2L3", 2: "L3L4", 3: "L5L6", 4: "D1D2", 5: "D3D4", 6: "L1D1", 7: "L2D1"}
BARREL_SEEDS, DISK_SEEDS, OVERLAP_SEEDS = (0, 1, 2, 3), (4, 5), (6, 7)
# the projections as TrackletConfigBuilder wires them (TrackletConfigBuilder.h:298-305, matchport_; not
# projlayers_ / projdisks_, which list an L2L3 -> L6 projection that is not wired); the barrel ones alone with
# the discs off
TARGETS = {0: (3, 4, 5, 6, 7, 8, 9, 10), 1: (1, 4, 5, 7, 8, 9, 10), 2: (1, 2, 5, 6, 7, 8), 3: (1, 2, 3, 4),
           4: (1, 2, 9, 10, 11), 5: (1, 7, 8, 11), 6: (8, 9, 10, 11), 7: (1, 8, 9, 10)}
DISCS = True
# THE CORRECTIONS found in WP-Disc that apply to the barrel too (format 35 lacked them): projections go only to the
# wired targets (format 35 sent L2L3 to L6, which TrackletConfigBuilder does not wire) and only into the merged
# projection memories that exist (buildProjections' emptyProjCombined); the projection calculator's 108-tracklet
# cap; the KF's state and stub sorts as libstdc++'s std::sort (cmssw_kf.STD_SORT: its order of equal keys). Off,
# with the discs off, the chain is census format 35's, bit for bit (gate G0).
FIXES = True


def set_discs(on, fixes=True):
    """The disc and overlap seeds and every projection to a disc on (census format 36) or off, and the WP-Disc
    corrections (FIXES) on or off; with both off, format 35's barrel chain, bit for bit."""
    global DISCS, FIXES
    DISCS, FIXES = bool(on), bool(fixes)
    KF.STD_SORT = FIXES


def active_seeds():
    return tuple(sd for sd in SEEDS if DISCS or sd in BARREL_SEEDS)


def targets(seed):
    if not FIXES and seed == 1:
        return (1, 4, 5, 6) + ((7, 8, 9, 10) if DISCS else ())        # format 35's L2L3 targets, L6 included
    return tuple(L for L in TARGETS[seed] if DISCS or L <= 6)


def layer_name(code):
    return f"L{code}" if code <= 6 else f"D{code - 6}"
TB_OF_SEED = {0: 0, 1: 0, 3: 0, 7: 0, 2: 1, 4: 1, 5: 1, 6: 1}   # Settings tbseed_ {{0,1,3,7},{2,4,5,6}}
SEED_RANK = (1, 5, 2, 7, 4, 3, 8, 6)                           # PurgeDuplicate.cc ranks
TB_MAX, DR_MAX, DR_COMPARED, MIN_SHARE = 108, 108, 32, 3
NRINV = 5                                                       # NRINVBITS
NBITS_PHIDER = 10                                               # nbitsphiprojderL123 / L456
NBEND = {L: (3 if L <= 3 else 4) for L in range(1, 7)}       # barrel bend bits; a disc stub's: PS 3, 2S 4
RINVCUT_PT, Z0CUT = 1.91, 15.0                                  # Settings ptcut_, z0cut_
RCRIT, DPHICRIT_MC = 55.0, 0.005
# VMRouterCM's inner-stub cuts (VMRouterCM.cc:262-281, Settings.h:49-52): |z| in integer units on the barrel
# (L1 of the overlap memories M / R / L: |z| >= 70 cm; L1 of the barrel ones, L3, L5: |z| <= 95 cm; L2: |z| >=
# 50 cm), and on D1 / D3 PS stubs with r_value <= 55 cm / kr (the 11-bit r counts from 7.5 cm)
VMR_ZMAX = 95.0
VMR_ZMIN_L2 = 50.0
VMR_ZMIN_L1_OVERLAP = 70.0
VMR_RMAX_D1D3 = 55.0
ZCUT_L2L3_LUT = 52.0                                            # TrackletLUT zcutL2L3
R_PROT_DISK, R_PROT_OVERLAP = 2.0, 1.5   # TrackletCalculatorBase: r2 >= r1 + this (floats), disc / overlap seeds
# per layer / disk code 1-11 (Settings nbitsallstubs_, nbitsvmme_, vmrlutzbits_, vmrlutrbits_)
NBITS_ALLSTUBS = (3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
NBITS_VMME = (2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2)
NFINEPHI, NFINERZ, ME_BINS_BITS = 3, 3, 3
VMR_LUT_ZBITS = (7, 7, 7, 7, 7, 7, 3, 3, 3, 3, 3)
VMR_LUT_RBITS = (4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8)
NRZBITS_TE_DISK = 2                      # TrackletProcessor nbitsrzbin_ for D1D2 / D3D4 (N_RZBITS = 3 otherwise)
NRBITS_TE_PT = 3                         # TrackletEngineUnit: the disc / overlap seeds' pT tables add 3 r bits
DISK_OUTER_TE = {8: "VMRTE_D2", 10: "VMRTE_D4", 7: "VMRTE_D1"}  # the router's disc TE tables (outer layers)
# TrackletLUT::writeTable writes a -1 entry of a positive table as (1 << nbits) - 1: in the inner tables
# (TP_<seed>.tab, 10 bits: fine bin 3, next 1, start bin 3, spread 3) that is 1023, "no tracklet"
TP_INNER_NONE = (1 << (2 * NFINERZ + 1 + ME_BINS_BITS)) - 1


# the emulated logic's version: bump it when a stage changes (a hash of the tables cannot see that).
# 2: the match processor clock by clock (MP 108), the VMRouter's reading order and the input-memory caps.
# 3: the IR cap counts the link's stubs of every layer (disc stubs included).
# 4: the inner tables' "no tracklet" entries (written as 1023) reject the inner stub.
# 5: the TrackletProcessor wiring is CMSSW_20_1's own (TrackletConfigBuilder's table; L1L2 differs from the
#    cms-data wires file the earlier versions read).
CHAIN_VERSION = 5
# KNOWN-BAD VARIANTS of the chain (WP-D plan sec. 3), like cmssw_kf.KF_KNOWN_BADS: None in every real run;
# set_known_bad() switches one on (a chain variant here, a fit variant in cmssw_kf) for a known-bad run.
CHAIN_KNOWN_BADS = {
    "a0_off": "the tracklet calculator's rinv, z0 and inSector gates off",
    "gates_float": "the tracklet gates on the float exact helix instead of calcPars' integers",
    "insector_off": "the inSector gate off",
    "mp_first_in_window": "the match processor keeps the first candidate inside the cuts, no improving chain",
    "nshare_2": "duplicate removal at 2 shared layers instead of 3",
    "comparators_unlimited": "duplicate removal without the 32-comparator limit",
    "dr_cap_off": "duplicate removal without the 108-candidate input cap",
    "kf_preferred_only": "a merged track fitted on its preferred candidate's stubs only",
    "comparebest_stub_r": "the duplicate-removal residual at the stub's radius instead of the layer's",
    "score_dr": "the census's score DR on the fitted no-DR tracks instead of merge DR",
    "l2l3_no_z52": "the L2L3 seed without its |z| >= 50 / 52 cm inner-stub cuts",
    "dtc_acceptance_ignored": "the census table keeps the stubs the DTC dropped (census chain only)",
}
KNOWN_BAD = None


def set_known_bad(name):
    """Switch on one known-bad variant, of the chain (CHAIN_KNOWN_BADS) or of the fit (cmssw_kf); None: none."""
    global KNOWN_BAD
    if name is not None and name not in CHAIN_KNOWN_BADS and name not in KF.KF_KNOWN_BADS:
        raise ValueError(f"unknown known-bad {name!r}; known: "
                         f"{', '.join(list(CHAIN_KNOWN_BADS) + list(KF.KF_KNOWN_BADS))}")
    KNOWN_BAD = name if name in CHAIN_KNOWN_BADS else None
    KF.set_known_bad(name if name in KF.KF_KNOWN_BADS else None)
_TABLE_SHA = {}


def _read(name):
    """The text of luts/`name`, its sha256 recorded for numerics_key."""
    b = (LUT_DIR / name).read_bytes()
    _TABLE_SHA[name] = hashlib.sha256(b).hexdigest()
    return b.decode()


def _tab(name):
    t = _read(name).strip().strip("{};")
    return np.array([int(x) for x in t.replace("\n", "").split(",") if x.strip()], np.int64)


def numerics_key():
    """The chain's identity for the census cache key: its version, its constants and every table it reads."""
    luts()
    return {"chain_version": CHAIN_VERSION, "rinvcut_pt": RINVCUT_PT, "z0cut": Z0CUT, "rcrit": RCRIT,
            "dphicrit_mc": DPHICRIT_MC, "vmr_z": [VMR_ZMAX, VMR_ZMIN_L2, ZCUT_L2L3_LUT],
            "caps": [TB_MAX, DR_MAX, DR_COMPARED, MIN_SHARE], "b_field": KF.K.b,
            "tables": dict(sorted(_TABLE_SHA.items()))}


class Luts:
    """The firmware tables the chain reads (dumped from CMSSW, luts/PROVENANCE.md)."""

    def __init__(self):
        self.phicorr = {L: _tab(f"VMPhiCorrL{L}.tab") for L in range(1, 7)}
        self.me = {L: _tab(f"METable_{layer_name(L)}.tab") for L in range(1, 12)}
        self.vmrme = {L: _tab(f"VMRME_{layer_name(L)}.tab") for L in range(1, 12)}
        self.phicut = {L: _tab(f"MP_L{L}PHIA_phicut.tab") for L in range(1, 7)}
        self.zcut = {L: _tab(f"MP_L{L}PHIA_zcut.tab") for L in range(1, 7)}
        # the disc match processors' tables (identical in every phi region; the A ones read): phi and r cuts by
        # seed for PS and 2S stubs, the 2S rings' radii and alpha factors, the projection-bend table
        self.disc_mp = {L: {t: _tab(f"MP_{layer_name(L)}PHIA_{t}.tab")
                            for t in ("PSphicut", "PSrcut", "2Sphicut", "2Srcut", "alphainner", "alphaouter",
                                      "rDSSinner", "rDSSouter")} for L in range(7, 12)}
        self.projbend = {L: _tab(f"MP_ProjectionBend_{layer_name(L)}.tab") for L in range(7, 12)}
        self.vmrte_disk = {L: _tab(f"{nm}.tab") for L, nm in DISK_OUTER_TE.items()}
        self.te = {}
        for sd, nm in SEED_NAME.items():
            # the stub-pT tables of a seed's TPs are identical (checked); index (dphi [<< 3 + ir], bend)
            ti = np.loadtxt(io.StringIO(_read(f"TP_{nm}A_stubptinnercut.dat")), dtype=np.int64)
            to = np.loadtxt(io.StringIO(_read(f"TP_{nm}A_stubptoutercut.dat")), dtype=np.int64)
            la = SEEDS[sd][0]
            extra = NBEND[la] if sd in BARREL_SEEDS else 3 + NRBITS_TE_PT    # inner stubs PS: 3 bend bits
            nbd = int(round(math.log2(len(ti) >> extra)))
            self.te[sd] = (ti, to, nbd)
        self.tp_inner = {sd: _tab(f"TP_{nm}.tab") for sd, nm in SEED_NAME.items()}
        self.tp = _tp_wiring()
        name_seed = {nm: sd for sd, nm in SEED_NAME.items()}
        self.empty_proj = set()
        for line in _read("TP_empty_projections.txt").splitlines():
            if line.startswith("#"):
                continue
            m = re.fullmatch(r"TPROJ_([LD]\d[LD]\d)([A-L])_([LD])(\d)PHI([A-H])", line.strip())
            code = int(m.group(4)) + (6 if m.group(3) == "D" else 0)
            self.empty_proj.add((name_seed[m.group(1)], ord(m.group(2)) - ord("A"), code, ord(m.group(5)) - ord("A")))


@dataclass
class TP:
    """One TrackletProcessor: its letter (iTC; TCIndex = (iSeed << 4) + iTC), the allstub phi region of its
    outer VM memory, and its inner allstub memories as (phi region, memtype) in its reading order."""
    seed: int
    index: int
    outer_region: int
    inner: list = field(default_factory=list)


def _tp_wiring():
    """The TrackletProcessors' inputs of the 8 prompt seeds (luts/TP_wiring.txt), as TrackletProcessor::addInput
    stores them: a third inner memory goes to the front (the "innerallstubs_.size() == 2" insert)."""
    name_seed = {nm: sd for sd, nm in SEED_NAME.items()}
    tps = {}
    for line in _read("TP_wiring.txt").splitlines():
        if line.startswith("#"):
            continue
        mem, _, src, _, dst = line.split()
        tp = re.fullmatch(r"TP_([LD]\d[LD]\d)([A-L])\.(\w+)", dst)
        sd, itc, port = name_seed[tp.group(1)], ord(tp.group(2)) - ord("A"), tp.group(3)
        t = tps.setdefault((sd, itc), TP(sd, itc, -1))
        if port == "outervmstubin":
            t.outer_region = ord(re.fullmatch(r"VMR_[LD]\dPHI([A-H])\.\w+", src).group(1)) - ord("A")
        else:
            # the memory type: the name's 12th character (B for the barrel seeds' L1-L5, D for the disc seeds,
            # O for the overlap seeds, then the type)
            m = re.fullmatch(r"AS_[LD]\dPHI([A-H])_[BDO]([A-Z])(_\w+)?", mem)
            entry = (ord(m.group(1)) - ord("A"), m.group(2))
            if len(t.inner) == 2:
                t.inner.insert(0, entry)
            else:
                t.inner.append(entry)
    out = {}
    for (sd, itc), t in sorted(tps.items()):
        assert t.outer_region >= 0 and t.inner, (sd, itc)
        out.setdefault(sd, []).append(t)
    return out


_LUTS = None


def luts():
    global _LUTS
    if _LUTS is None:
        _LUTS = Luts()
    return _LUTS


# ---- inputs ------------------------------------------------------------------
@dataclass
class Copies:
    """The stub copies of one event, one per (stub, region) frame: barrel, and disc ones with the discs on."""
    row: np.ndarray          # L1TOTStub index
    region: np.ndarray
    order: np.ndarray        # position in the finder's loop (region, channel, stream index)
    layer: np.ndarray        # layer / disk code: 1..6 barrel, 7..11 D1..D5
    ps: np.ndarray
    tilted: np.ndarray
    iphi: np.ndarray
    iz: np.ndarray
    ir: np.ndarray           # the barrel's signed r; a disc stub's Stub::rvalue() (PS: full r / kr; 2S: ring index)
    bend: np.ndarray
    phicorr: np.ndarray
    r: np.ndarray            # the integer r, physical: rmean + ir krbarrel; disc PS ir kr + 7.5, 2S the ring radius
    x: np.ndarray            # float stub position (the KF's input)
    y: np.ndarray
    z: np.ndarray
    link: np.ndarray = None  # the DTC link (region * 48 + channel)
    link_rank: np.ndarray = None   # the stub's place among the link's stubs of every layer, in stream order
    # what survives the finder's input memories (input_caps): None = everything
    alive: np.ndarray = None
    te_ok: np.ndarray = None
    me_ok: np.ndarray = None
    neg: np.ndarray = None   # a disc stub on the -z side (the DTC word's negDisk bit)
    alpha: np.ndarray = None # the DTC word's alpha of a 2S disc stub (0 elsewhere)
    strip: np.ndarray = None # L1TOTStub innerStrip / isFlipped (the KF's 2S disc alpha)
    flipped: np.ndarray = None


def vmr_order(region, channel, index):
    """The VMRouter's reading order of the stubs, as a rank: per region, the InputLink memories in
    TrackletConfigBuilder's wiring order (the DTCs PS first, then 2S; +z before -z; by ATCA slot; side A then
    B; L1FPGATrackProducer's channel = slot + 12 (-z) + 24 (B)), then each link's stream order."""
    channel = np.asarray(channel, np.int64)
    slot, neg, side = channel % 12, (channel % 24) >= 12, channel >= 24
    key = np.lexsort((np.asarray(index), side, slot, neg, slot >= 6, np.asarray(region)))
    rank = np.empty(len(key), np.int64)
    rank[key] = np.arange(len(key))
    return rank


def event_copies(stub, dtc):
    """stub: dict of L1TOTStub numpy columns for one event; dtc: dict of L1TOTStubDTC columns. The barrel copies,
    and with the discs on (DISCS) the disc copies too (no phi correction on discs: Sector.cc corrects the
    barrel only)."""
    row = np.asarray(dtc["otStubIdx"], np.int64)
    link_all = np.asarray(dtc["region"], np.int64) * 48 + np.asarray(dtc["channel"], np.int64)
    srt = np.lexsort((np.asarray(dtc["index"], np.int64), link_all))
    start = np.r_[0, np.flatnonzero(link_all[srt][1:] != link_all[srt][:-1]) + 1]
    rank_all = np.empty(len(row), np.int64)
    rank_all[srt] = np.arange(len(row)) - np.repeat(start, np.diff(np.r_[start, len(row)]))
    barrel = np.asarray(stub["isBarrel"], bool)[row]
    keep = barrel | DISCS
    row, barrel = row[keep], barrel[keep]
    region = np.asarray(dtc["region"], np.int64)[keep]
    channel = np.asarray(dtc["channel"], np.int64)[keep]
    index = np.asarray(dtc["index"], np.int64)[keep]
    order = vmr_order(region, channel, index)
    lay_raw = np.asarray(stub["layer"], np.int64)[row]
    layer = np.where(barrel, lay_raw, lay_raw - 4)                 # L1TOTStub disc layer 11-15 -> 7-11
    ps = np.asarray(stub["isPS"], bool)[row]
    mt = D.module_type(barrel, ps)
    _, bend, alpha, iphi, iz, ir, nd = D.decode(np.asarray(dtc["frameHi"])[keep], np.asarray(dtc["frameLo"])[keep], mt)
    L = luts()
    phicorr = iphi.copy()
    r = np.zeros(len(row))
    b = np.flatnonzero(barrel)
    rbin = (ir[b] + (1 << 6)) >> 4                               # (r + 2^(nrbits-1)) >> (nrbits - 3)
    corr = np.array([L.phicorr[int(l)][int(q) * 8 + int(rb)] for l, q, rb in zip(layer[b], bend[b], rbin)], np.int64)
    nphib = np.array(D.NPHIBITS_L, np.int64)[layer[b] - 1]
    phicorr[b] = np.clip(iphi[b] - corr, 0, (np.int64(1) << nphib) - 1)
    rm = np.array([D.rmean(l) for l in range(1, 7)])
    r[b] = rm[layer[b] - 1] + ir[b] * D.kr_barrel()
    d = np.flatnonzero(~barrel)
    if len(d):
        ir[d] = D.disc_rvalue(ir[d], ps[d])                      # the disc stubs' r as Stub::rvalue() reads it
        r[d] = D.disc_physical(layer[d] - 6, nd[d] == 1, ps[d], iphi[d], iz[d], ir[d], region[d])[2]
    strip = np.asarray(stub["innerStrip"], np.float64)[row] if "innerStrip" in stub else np.zeros(len(row))
    flipped = np.asarray(stub["isFlipped"], bool)[row] if "isFlipped" in stub else np.zeros(len(row), bool)
    return Copies(row, region, order, layer, ps, np.asarray(stub["isTilted"], bool)[row], iphi, iz, ir, bend,
                  phicorr, r,
                  np.asarray(stub["x"], np.float64)[row], np.asarray(stub["y"], np.float64)[row],
                  np.asarray(stub["z"], np.float64)[row], region * 48 + channel, rank_all[keep],
                  neg=nd == 1, alpha=np.where(barrel | ps, 0, alpha), strip=strip, flipped=flipped)


# ---- the finder's input memories ---------------------------------------------------------------
IR_MAX = 156             # DTCLinkMemory: maxStep("IR")
VMR_MAX = 108            # VMRouterCM: maxStep("VMR")
MEM_MAX = 1 << 7         # VMRouterCM / VMStubMERouter: 1 << N_BITSMEMADDRESS
VMBIN_MAX = 15           # VMStubsTEMemory N_VMSTUBSMAX, VMStubsMEMemory maxStubsPerBin_
PROJ_PAGE_MAX = (1 << 6) - 1   # TrackletProjectionsMemory: (1 << (N_BITSMEMADDRESS - 1)) - 1 per page


def _first_n(group, order, n):
    """Mask of the elements that are among the first n of their group in `order`."""
    idx = np.lexsort((order, group))
    g = group[idx]
    start = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    pos = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
    keep = np.zeros(len(group), bool)
    keep[idx[pos < n]] = True
    return keep


def input_caps(cp, caps=True):
    """Set cp.alive / te_ok / me_ok: the copies the finder keeps through its input memories, in their reading
    order (cp.order). The step caps (caps on): the first 156 stubs of a DTC link, of every layer (IR), and
    108 per VMRouter.
    The memory sizes (always): 128 per VMRouter, 15 per VM bin of the tracklet-engine memories (outer stubs:
    phi region, ivmte, rz bin) and of the match-engine memories (phi region, VM bin, rz bin). A disc stub's
    match-engine bin is its VMRME bin, + 8 on the -z side (VMStubMERouter); an outer disc stub enters the
    tracklet-engine memory only on a PS module with a nonzero VMRTE entry, its bin + 4 on the -z side
    (VMRouterCM, VMStubsTEMemory::addVMStub)."""
    n = len(cp.row)
    alive = np.ones(n, bool)
    if caps:
        alive &= cp.link_rank < IR_MAX
    layers = np.unique(cp.layer)
    preg, sub3 = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for L in layers:
        m = cp.layer == L
        preg[m], sub3[m] = _phi_region(cp, np.flatnonzero(m), int(L))
    vmr = (cp.region * 16 + cp.layer) * 8 + preg
    a = np.flatnonzero(alive)
    keep = _first_n(vmr[a], cp.order[a], min(VMR_MAX, MEM_MAX) if caps else MEM_MAX)
    alive[a[~keep]] = False
    zbin = np.zeros(n, np.int64)
    vm = np.zeros(n, np.int64)
    tebin = np.full(n, -1, np.int64)
    for L in layers:
        m = np.flatnonzero(cp.layer == L)
        idx = _vmr_lut_index(cp, m, int(L))
        zbin[m] = luts().vmrme[int(L)][idx] >> NFINERZ
        nb, nall, nvm = D.NPHIBITS_L[L - 1], NBITS_ALLSTUBS[L - 1], NBITS_VMME[L - 1]
        vm[m] = (cp.phicorr[m] >> (nb - nall - nvm)) & ((1 << nvm) - 1)
        if L <= 6:
            tebin[m] = zbin[m]
        else:
            zbin[m] += np.where(cp.neg[m], 1 << NFINERZ, 0)
            if L in DISK_OUTER_TE:
                lut = luts().vmrte_disk[int(L)][idx]
                ok = cp.ps[m] & (lut != 0)
                tebin[m[ok]] = (lut[ok] >> NFINERZ) + np.where(cp.neg[m[ok]], 4, 0)
    a = np.flatnonzero(alive)
    me = np.zeros(n, bool)
    me[a] = _first_n(((vmr[a] * 8 + vm[a]) * 16) + zbin[a], cp.order[a], VMBIN_MAX)
    te = np.zeros(n, bool)
    outer = a[np.isin(cp.layer[a], [lb for sd in active_seeds() for lb in SEEDS[sd][1:]]) & (tebin[a] >= 0)]
    te[outer] = _first_n(((vmr[outer] * 8 + sub3[outer]) * 16) + tebin[outer], cp.order[outer], VMBIN_MAX)
    cp.alive, cp.te_ok, cp.me_ok = alive, te, me


# ---- the tracklet calculator and the projection calculator, in integers ---------------
# TrackletCalculatorBase::init (barrel seeds): the widths of the tracklet parameters (Settings nbitsrinv 14,
# nbitsphi0 18, nbitst 15, nbitsz0 11; nphibitsstub(L6) 17, nrbitsstub(D1) + 1 = 12, nzbitsstub(L1) - 1 = 11)
# and of the intermediate products.
N_PHI, N_R, N_Z, N_Z0, N_PHI0, N_RINV, N_T = 17, 12, 11, 12, 16, 13, 10
N_DELTA0, N_DELTAZ, N_DELTA1, N_DELTA2, N_DELTA12, N_A = 9, 9, 10, 12, 13, 15
N_R6, N_IFACT, N_DELTA02, N_X6, N_IT1, N_HG, N_DELTAR = 6, 12, 12, 16, 5, 15, 24
# ProjectionCalculator's own copy of the widths: its n_t_ is still 9 (nbitst 14), not the calculator's 10
PC_N_PHI, PC_N_R, PC_N_Z, PC_N_PHI0, PC_N_RINV, PC_N_T = 17, 12, 11, 16, 13, 9
PC_N_S, PC_N_S6, PC_N_IV = 12, 14, 4
KR = D.RMAXDISK / (1 << 12)                     # Settings::kr(): rmaxdisk / 2^(nrbitsstub(D1) + 1)


# per seed family (TrackletCalculatorBase::init): (n_delta0, n_deltaz, n_a, n_r6, n_ifact, LUT size); the overlap
# seeds' own widths, every other one as the barrel's
CALC_WIDTHS = {"barrel": (9, 9, 15, 6, 12, 512), "disk": (9, 9, 15, 6, 12, 512), "overlap": (9, 6, 14, 6, 10, 1024)}


def _family(seed):
    return "barrel" if seed in BARREL_SEEDS else "disk" if seed in DISK_SEEDS else "overlap"


def _idrinv(seed, cache={}):
    """LUT_idrinv_ (TrackletCalculatorBase::init): 0.5 + float(2^24) / idr, the division in float; the barrel
    seeds' idr is the signed 9-bit difference about the two layers' rmean (idrabs)."""
    if seed not in cache:
        fam = _family(seed)
        lut = np.zeros(CALC_WIDTHS[fam][5], np.int64)
        if fam == "barrel":
            la, lb = SEEDS[seed]
            for idr in range(-256, 256):
                idrabs = idr + D.IRMEAN[lb - 1] - D.IRMEAN[la - 1]
                lut[idr % 512] = int(0.5 + float(np.float32(1 << N_DELTAR) / np.float32(idrabs)))
        else:
            for idr in range(1, len(lut)):
                lut[idr] = int(0.5 + float(np.float32(1 << N_DELTAR) / np.float32(idr)))
        cache[seed] = lut
    return cache[seed]


def calc_inputs(cp, i, j, seed):
    """The integers barrelSeeding / diskSeeding / overlapSeeding hand calcPars for the pair (i inner, j outer):
    (idr, iphi1, ir1, iz1, iphi2, ir2, iz2), or None where overlapSeeding drops the pair (idr outside its LUT).
    Barrel: phi to the L6 width, r to 8 bits about rmean, z to the L1 width. Discs: the raw r_value (no 7.5 cm
    offset), phi << 3, z about the side's zmean. Overlap: the barrel r to 8 bits plus rmean, the disc's raw
    r_value, its z about zmean on the inner stub's side."""
    la, lb = SEEDS[seed]
    if seed in BARREL_SEEDS:
        ir1 = int(cp.ir[i]) << (8 - D.NRBITS_L[la - 1])
        ir2 = int(cp.ir[j]) << (8 - D.NRBITS_L[lb - 1])
        iphi1 = int(cp.iphi[i]) << (N_PHI - D.NPHIBITS_L[la - 1])
        iphi2 = int(cp.iphi[j]) << (N_PHI - D.NPHIBITS_L[lb - 1])
        iz1 = int(cp.iz[i]) << (D.NZBITS_L[0] - D.NZBITS_L[la - 1])
        iz2 = int(cp.iz[j]) << (D.NZBITS_L[0] - D.NZBITS_L[lb - 1])
        idr = ir2 - ir1
        if idr < 0:
            idr += 512
        return idr, iphi1, ir1 + D.IRMEAN[la - 1], iz1, iphi2, ir2 + D.IRMEAN[lb - 1], iz2
    shift = D.NPHIBITS_L[5] - D.NPHIBITS_L[0]
    iphi1, iphi2 = int(cp.iphi[i]) << shift, int(cp.iphi[j]) << shift
    ir2, iz2 = int(cp.ir[j]), int(cp.iz[j])
    if seed in DISK_SEEDS:
        sign = -1 if cp.neg[i] else 1
        ir1, iz1 = int(cp.ir[i]), int(cp.iz[i])
        idr = ir2 - ir1
        if not 0 < idr < 512:
            return None          # CMSSW asserts here; r2 >= r1 + 2 cm (floats) keeps idr positive
        return (idr, iphi1, ir1, iz1 + sign * D.IZMEAN[la - 7], iphi2, ir2, iz2 + sign * D.IZMEAN[lb - 7])
    ir1abs = (int(cp.ir[i]) << (8 - D.NRBITS_L[la - 1])) + D.IRMEAN[la - 1]
    iz1 = int(cp.iz[i])
    idr = ir2 - ir1abs
    if not 0 <= idr < 1024:
        return None
    iz2mean = -D.IZMEAN[lb - 7] if iz1 < 0 else D.IZMEAN[lb - 7]
    return idr, iphi1, ir1abs, iz1, iphi2, ir2, iz2 + iz2mean


def calc_pars(cp, i, j, seed, k=KF.K):
    """(irinv, iphi0, iz0, it) of the pair (i inner, j outer): calcPars (TrackletCalculatorBase.cc:116-177) on
    calc_inputs, line for line, with the seed family's widths; None where the inputs are refused."""
    inp = calc_inputs(cp, i, j, seed)
    if inp is None:
        return None
    idr, iphi1, r1, iz1, iphi2, r2, iz2 = inp
    n_delta0, n_deltaz, n_a, n_r6, n_ifact, _ = CALC_WIDTHS[_family(seed)]
    hg = k.dphisector_hg
    invdr = int(_idrinv(seed)[idr])
    idelta0 = ((iphi2 - iphi1) * invdr) >> n_delta0
    ideltaz = ((iz2 - iz1) * invdr) >> n_deltaz
    idelta1 = (r1 * idelta0) >> N_DELTA1
    idelta2 = (r2 * idelta0) >> N_DELTA2
    idelta12 = (idelta1 * idelta2) >> N_DELTA12
    ihg = int(hg * hg * (1 << N_HG))
    ia = (1 << n_a) - ((idelta12 * ihg) >> (2 * N_DELTAR + 2 * N_PHI + N_HG - 2 * n_delta0 - N_DELTA1 - N_DELTA2
                                            - N_DELTA12 + 1 - n_a))
    ifact = int((1 << n_ifact) * hg * hg / 6.0)
    ir6 = ((r1 + r2) * ifact) >> (n_ifact - n_r6)
    idelta02 = (idelta0 * idelta2) >> N_DELTA02
    ix6 = -(1 << N_X6) + ((ir6 * idelta02) >> (n_r6 + 2 * N_DELTAR + 2 * N_PHI - N_X6 - N_DELTA2 - N_DELTA02
                                                - 2 * n_delta0))
    it1 = (r1 * ideltaz) >> (N_DELTAR - n_deltaz - N_IT1)
    irinv = (((-idelta0 * ia) >> (N_PHI + n_a - N_RINV + N_DELTAR - n_delta0 - N_R - 1 - 1)) + 1) >> 1
    iphi0 = (((iphi1 + ((idelta1 * ix6) >> (N_DELTAR + N_X6 - n_delta0 - N_DELTA1))) >> (N_PHI - N_PHI0 - 1))
             + 1) >> 1
    it = (((ideltaz * ia) >> (N_DELTAR + n_a + N_Z - N_T - n_deltaz - N_R - 1)) + 1) >> 1
    iz0 = (iz1 << 1) + ((((it1 * ix6) >> (N_X6 + N_IT1 - 2)) + 1) >> 1)
    return irinv, iphi0, iz0, it


def tc_gates(irinv, iphi0, iz0, seed, k=KF.K, in_sector=True):
    """The calculator's rinv and z0 cuts and (in_sector) inSector with usephicritapprox_ false, in integers: z0 at
    15 cm for L1L2 and the disc and overlap seeds, 22.5 cm for the other barrel seeds."""
    hg = k.dphisector_hg
    rinvcut = 0.01 * KF.C_TRK * k.b / RINVCUT_PT
    if not abs(irinv) < rinvcut * (120.0 * (1 << N_RINV)) / hg:
        return False
    z0lim = (1.5 * Z0CUT * (1 << N_Z0) / 120.0 if seed in (1, 2, 3) else Z0CUT * (1 << N_Z0) / 120.0)
    if not abs(iz0) < z0lim:
        return False
    if not in_sector:
        return True
    ifactor = int(0.5 * RCRIT * k.krinvpars / k.kphi0pars * (1 << 8))
    iphicrit = iphi0 - (irinv >> 8) * ifactor
    phicritminmc = (0.5 * hg - math.pi / KF.N_SECTOR) - DPHICRIT_MC
    phicritmaxmc = (hg - 0.5 * hg + math.pi / KF.N_SECTOR) + DPHICRIT_MC
    return iphi0 >= 0 and int(phicritminmc / k.kphi0pars) < iphicrit < int(phicritmaxmc / k.kphi0pars)


def _gates(ipars, floats, seed, k=KF.K):
    """The tracklet calculator's gates: tc_gates, or a known-bad variant of them."""
    if KNOWN_BAD == "a0_off":
        return True
    if KNOWN_BAD == "gates_float":
        rinv, phi0, z0 = floats
        hg = k.dphisector_hg
        phicrit = phi0 - math.asin(0.5 * RCRIT * rinv)
        return (abs(rinv) < 0.01 * KF.C_TRK * k.b / RINVCUT_PT and abs(z0) < Z0CUT * (1.5 if seed in (1, 2, 3) else 1.0)
                and phi0 >= 0 and (0.5 * hg - math.pi / KF.N_SECTOR) - DPHICRIT_MC < phicrit
                < (hg - 0.5 * hg + math.pi / KF.N_SECTOR) + DPHICRIT_MC)
    return tc_gates(*ipars[:3], seed, k, in_sector=KNOWN_BAD != "insector_off")


def proj_layer(ipars, layer, k=KF.K):
    """ProjectionCalculator for one barrel layer: (iphi, iz, iphider, izder) in the layer's stub units,
    or None where the calculator drops the projection (outside the z / phi range, or at an extreme z)."""
    irinv, iphi0, iz0, it = ipars
    ir = D.IRMEAN[layer - 1]
    irtilde = int(ir * k.dphisector_hg / math.sqrt(6.0) + 0.5)
    is_ = (irtilde * irinv) >> PC_N_S
    is6 = (1 << PC_N_S6) + ((is_ * is_) >> (2 + 2 * PC_N_R + 2 * PC_N_RINV - 2 * PC_N_S - PC_N_S6))
    iu = (ir * irinv) >> (PC_N_R + PC_N_RINV + 1 - PC_N_PHI)
    iphi = (iphi0 << (PC_N_PHI - PC_N_PHI0)) - ((iu * is6) >> PC_N_S6)
    iv = (it * ir) >> (1 + PC_N_R + PC_N_T - PC_N_Z - PC_N_IV)
    iz = (iz0 + ((iv * is6) >> (PC_N_IV + PC_N_S6 - 1)) + 1) >> 1
    if not (-(1 << (D.NZBITS_L[0] - 1)) <= iz < (1 << (D.NZBITS_L[0] - 1)) and 0 < iphi < (1 << 17) - 1):
        return None
    if layer <= 3:
        iphi >>= 3
    else:
        iz >>= 4
    nz = D.NZBITS_L[layer - 1]
    if iz == -(1 << (nz - 1)) or iz == (1 << (nz - 1)) - 1:
        return None
    iphider, izder = -(irinv >> 4), it >> 4
    if not -(1 << (NBITS_PHIDER - 1)) <= iphider < (1 << (NBITS_PHIDER - 1)):
        return None        # beyond the 10-bit derivative word: only a tracklet past the rinv gate (a0_off)
    return iphi, iz, iphider, izder


# ProjectionCalculator's disc widths (n_tinv 14, n_y 14, n_x 14, n_xx6 15; n_phidisk = n_phi - 3, n_rdisk = n_r)
PC_N_TINV, PC_N_Y, PC_N_X, PC_N_XX6, PC_N_PHIDISK, PC_N_RDISK = 14, 14, 14, 15, 14, 12
# Settings projlayers_ / projdisks_ by seed (what the calculator projects to; only the wired ones are written,
# TARGETS), as layer / disk codes
PROJ_LAYERS = {0: (3, 4, 5, 6), 1: (1, 4, 5, 6), 2: (1, 2, 5, 6), 3: (1, 2, 3, 4), 4: (1, 2), 5: (1,), 6: (),
               7: (1,)}
PROJ_DISKS = {0: (7, 8, 9, 10), 1: (7, 8, 9, 10), 2: (7, 8), 3: (), 4: (9, 10, 11), 5: (7, 8, 11),
              6: (8, 9, 10, 11), 7: (8, 9, 10)}
RMINDISK, RMINDISKVM = 20.0, 22.5          # Settings rmindisk_, rmindiskvm_
NBITS_RPROJDER_DISK = 10                    # 1 + nrbitsprojderdisk_


def _itinv(it, cache=[]):
    """ProjectionCalculator's LUT_itinv_: 0 below 100, else 0.5 + float(2^(n_t + n_tinv)) / |it| (float)."""
    if not cache:
        lut = np.zeros(8192, np.int64)
        for i in range(100, 8192):
            lut[i] = int(0.5 + float(np.float32(1 << (PC_N_T + PC_N_TINV)) / np.float32(i)))
        cache.append(lut)
    return int(cache[0][abs(it)])


def proj_disk(ipars, disk, k=KF.K):
    """ProjectionCalculator::projDisk at disc `disk` (1-5) and the calculator's disc validity (r in
    [rmindisk, rmaxdisk) / kr and |it| > 1 / ktpars + 1; phi strictly inside the 14-bit word): (iphi, ir,
    iphider, irder) in the disc's units (phi 14 bits, r in kr), or None."""
    irinv, iphi0, iz0, it = ipars
    if abs(it) >= 8192:
        return None
    iz = D.IZMEAN[disk - 1]
    iz0_sign = iz0 if it > 0 else -iz0
    itinv = _itinv(it)
    iderphi = (-irinv * itinv) >> (PC_N_TINV + 4)
    iderr = ((itinv >> (PC_N_TINV - 9 - 2)) + 1) >> 1
    if it < 0:
        iderphi, iderr = -iderphi, -iderr
    nw = 2
    iw = (((iz << (PC_N_R - PC_N_Z)) - (iz0_sign << (PC_N_R - PC_N_Z - 1))) * itinv) >> (PC_N_TINV - nw - 1)
    iphi = (iphi0 >> (PC_N_PHI0 - PC_N_PHIDISK)) - ((iw * irinv) >> (1 + PC_N_R + PC_N_RINV - PC_N_PHIDISK + nw))
    ifact = int((1 << PC_N_Y) * k.dphisector_hg / math.sqrt(6.0))
    iy = (ifact * irinv) >> PC_N_Y
    ix = (iw * iy) >> (PC_N_X + nw)
    ix6 = (1 << PC_N_XX6) - ((ix * ix) >> (2 + 2 * PC_N_R + 2 * PC_N_RINV - 2 * PC_N_X - PC_N_XX6))
    ir = (((iw * ix6) >> (PC_N_R - PC_N_RDISK + PC_N_XX6 + nw - 1)) + 1) >> 1
    tcut = int(1.0 / k.ktpars + 1)
    if not (RMINDISK / D.kr() <= ir < D.RMAXDISK / D.kr() and (it > tcut or it < -tcut)):
        return None
    if iphi <= 0 or iphi >= (1 << D.NPHIBITS_L[6]) - 1:
        return None
    return iphi, ir, iderphi, iderr


def tracklet_projections(ipars, seed, k=KF.K):
    """The projections ProjectionCalculator writes for a tracklet: to the layers and discs it projects to
    (PROJ_LAYERS / PROJ_DISKS), a barrel or D1D2 seed's disc projection dropped when the corresponding layer
    already took one (D1 / L6, D2 / L5, D3 / L4, D4 / L3; the "iSeed <= 4" rule, counting the layers it projects
    to whether wired or not), a disc projection kept only for r * kr in [rmindiskvm, rmaxdisk]; then only the
    wired targets (targets()) reach a match processor. {code: projection}."""
    out = {}
    added = set()
    for L in PROJ_LAYERS[seed]:
        pj = proj_layer(ipars, L, k)
        if pj is not None:
            out[L] = pj
            added.add(L)
    if DISCS:
        for L in PROJ_DISKS[seed]:
            pj = proj_disk(ipars, L - 6, k)
            if pj is None:
                continue
            if seed <= 4 and L <= 10 and 13 - L in added:   # D1 <-> L6, D2 <-> L5, D3 <-> L4, D4 <-> L3
                continue
            if not RMINDISKVM <= pj[1] * D.kr() <= D.RMAXDISK:
                continue
            out[L] = pj
    wired = set(targets(seed))
    return {L: pj for L, pj in out.items() if L in wired}


def _ilog2(x):
    """trklet::ilog2: the rounded log2 of an exact power of two."""
    p = math.log(x) / math.log(2)
    assert abs(p - round(p)) < 0.1
    return int(round(p))


def mp_shifts(layer, k=KF.K):
    """MatchProcessor's shifts: (phishift, dzshift, icorrshift, icorzshift); on a disc icorrshift from
    kphi(disc) / (kz() kphiderdisk) and icorzshift = 2 + log2(kr / (kz() krder)) (MatchProcessor.cc:52-58)."""
    kphi0 = k.dphisector_hg / (1 << D.NPHIBITS_L[0])                # Settings::kphi()
    if layer >= 7:
        kz0 = D.kz(1)                                               # Settings::kz()
        kphiderdisk, krder = kphi0 / D.kr() / 128, 1.0 / 128
        return (KF.NPHIBITS_L6 - D.NPHIBITS_L[layer - 1], D.NZBITS_L[0] - D.NZBITS_L[layer - 1],
                _ilog2(D.kphi(layer, k) / (kz0 * kphiderdisk)), 2 + _ilog2(D.kr() / (kz0 * krder)))
    kphider = kphi0 / KR / 256
    kzder = 1.0 / 64
    return (KF.NPHIBITS_L6 - D.NPHIBITS_L[layer - 1], D.NZBITS_L[0] - D.NZBITS_L[layer - 1],
            _ilog2(D.kphi(layer, k) / (D.kr_barrel() * kphider)), _ilog2(D.kz(layer) / (D.kr_barrel() * kzder)))


# ---- tracklets -----------------------------------------------------------------
@dataclass
class Tracklet:
    sector: int
    seed: int
    inner: int               # index into Copies
    outer: int
    rinv: float              # exacttracklet on the float stubs: the KF prior
    phi0: float              # relative to the sector's phimin
    t: float
    z0: float
    ipars: tuple = ()        # (irinv, iphi0, iz0, it): calcPars, which the gates and projections use
    tcid: tuple = ()
    matches: dict = field(default_factory=dict)   # layer -> list of Copies indices (improving-chain order)
    projs: dict = field(default_factory=dict)     # layer -> proj_layer(), the valid projections


def _inner_ok(cp, idx, seed):
    """VMRouterCM's cuts on the seed's inner allstub memories (VMRouterCM.cc:262-281): D1 / D3 stubs with
    2 N_DSS_MOD <= r_value <= 55 cm / kr (the lower cut drops the 2S stubs, whose r_value is a ring index); L1 of the barrel seeds, L3, L5 |z| <= 95 cm; L1 of the overlap seeds |z| >= 70 cm;
    L2 |z| >= 50 cm, and L2L3's |z| >= 52 cm (TrackletLUT zcutL2L3)."""
    L = SEEDS[seed][0]
    if L >= 7:
        return (cp.ir[idx] <= VMR_RMAX_D1D3 / D.kr()) & (cp.ir[idx] >= 2 * D.N_DSS_MOD)
    absz = np.abs(cp.iz[idx])
    kzl = D.kz(L)
    if L == 1 and seed in OVERLAP_SEEDS:
        return absz >= VMR_ZMIN_L1_OVERLAP / kzl
    if L == 2:
        ok = absz >= VMR_ZMIN_L2 / kzl
        if seed == 1:
            ok &= np.abs(cp.z[idx]) >= ZCUT_L2L3_LUT
            if KNOWN_BAD == "l2l3_no_z52":
                ok = np.ones(len(idx), bool)
        return ok
    return absz <= VMR_ZMAX / kzl


def _float_geom(cp, i):
    return math.hypot(cp.x[i], cp.y[i]), float(cp.z[i]), math.atan2(cp.y[i], cp.x[i])


def _vmr_lut_index(cp, idx, layer):
    """The (z, r) index of the VMRouter / TrackletProcessor tables: on the barrel the top 7 bits of z and 4 of r;
    on a disc the top 3 bits of z (mirrored on the -z side) and the r bits (PS: r >> 4; 2S: the ring index)."""
    nz, nr = D.NZBITS_L[layer - 1], D.NRBITS_L[layer - 1]
    zb, rb = VMR_LUT_ZBITS[layer - 1], VMR_LUT_RBITS[layer - 1]
    iz = (cp.iz[idx] >> (nz - zb)) & ((1 << zb) - 1)
    if layer <= 6:
        ir = (cp.ir[idx] >> (nr - rb)) & ((1 << rb) - 1)
    else:
        iz = np.where(cp.neg[idx], (1 << zb) - 1 - iz, iz)
        ir = np.where(cp.ps[idx], cp.ir[idx] >> (nr + 1 - rb), cp.ir[idx])
    return (iz << rb) + ir


def te_rz_window(cp, inn, out, seed):
    """The tracklet engine's r-z window (TrackletProcessor.cc:362-395, TrackletEngineUnit::step): the inner
    stub's TP_<seed> table entry gives the outer layer's first rz bin (2 bits for the disc seeds, + 4 for an
    inner stub on the -z side; 3 otherwise), whether the next one is used, the first fine bin and the allowed
    spread; the outer stub's bin and fine position are its VMRME entry (barrel) or VMRTE entry (+ 4 on the -z
    side; discs). Returns (window, ir): ir, per outer stub, the 3 r bits the disc and overlap seeds' pT tables
    add ((bin & 3) << 1 | the fine position's top bit)."""
    la, lb = SEEDS[seed]
    lut = luts().tp_inner[seed][_vmr_lut_index(cp, inn, la)]
    none = lut == TP_INNER_NONE              # the table's -1: no tracklet from this inner stub (lookup -1)
    nrz = NRZBITS_TE_DISK if seed in DISK_SEEDS else ME_BINS_BITS
    first = lut & ((1 << NFINERZ) - 1)
    nxt = (lut >> NFINERZ) & 1
    start = (lut >> (NFINERZ + 1)) & ((1 << nrz) - 1)
    diffmax = (lut >> (NFINERZ + 1 + nrz)) & ((1 << NFINERZ) - 1)
    if seed in DISK_SEEDS:
        start = start + np.where(cp.neg[inn], 1 << nrz, 0)
    idx = _vmr_lut_index(cp, out, lb)
    if lb <= 6:
        olut = luts().vmrme[lb][idx]
        obin = olut >> NFINERZ
    else:
        olut = luts().vmrte_disk[lb][idx]
        obin = (olut >> NFINERZ) + np.where(cp.neg[out], 4, 0)
    ofine = olut & ((1 << NFINERZ) - 1)
    in_first = obin[None, :] == start[:, None]
    in_next = (nxt[:, None] == 1) & (obin[None, :] == start[:, None] + 1)
    rz = np.where(in_next, ofine[None, :] + (1 << NFINERZ), ofine[None, :])
    win = (in_first | in_next) & (rz >= first[:, None]) & (rz - first[:, None] <= diffmax[:, None])
    ir = ((obin & 3) << 1) + (ofine >> (NFINERZ - 1))
    if KNOWN_BAD == "l2l3_no_z52" and seed == 1:
        return win | none[:, None], ir      # the table's cut entries: no r-z restriction
    return win & ~none[:, None], ir


def _phi_region(cp, idx, layer):
    """(allstub phi region, the 3 bits below it) of copies `idx` from their corrected phi (VMRouterCM)."""
    nb, nall = D.NPHIBITS_L[layer - 1], NBITS_ALLSTUBS[layer - 1]
    pc = cp.phicorr[idx]
    return pc >> (nb - nall), (pc >> (nb - nall - 3)) & 7


def _in_memory(layer, iphipos, memtype):
    """VMRouterCM's allinnerstubs memtype selection on the 3 phi bits below the allstub region."""
    hi, lo = (4, 4) if layer == 1 else (6, 2)
    ok = np.ones(len(iphipos), bool)
    if memtype == "R":
        ok &= iphipos >= hi
    if memtype == "L":
        ok &= iphipos < lo
    if memtype in "ACE":
        ok &= iphipos >= 4
    if memtype in "BDF":
        ok &= iphipos < 4
    return ok


def seed_sector(cp, sector, seed, k=KF.K, work=None):
    """Tracklets of one seed in one sector: per TrackletProcessor (its inner memories against its outer phi
    region), the tracklet-engine LUTs and the integer tracklet calculator (module docstring); the disc and
    overlap seeds' float radius protections (r2 >= r1 + 2.0 / 1.5 cm) before it. The tcid
    orders tracklets as TCID does: seed, TP letter, then the TP's reading order (inner memory, inner stub,
    outer stub). `work` (a dict) accumulates the pairs in the engines' r-z windows ("te_pairs") and those
    the stub-pT LUTs pass ("lut_pairs")."""
    la, lb = SEEDS[seed]
    inn_m = (cp.region == sector) & (cp.layer == la)
    out_m = (cp.region == sector) & (cp.layer == lb)
    if cp.alive is not None:
        inn_m &= cp.alive
        out_m &= cp.te_ok
    inn_all, out_all = np.nonzero(inn_m)[0], np.nonzero(out_m)[0]
    inn_all = inn_all[_inner_ok(cp, inn_all, seed)]
    if len(inn_all) == 0 or len(out_all) == 0:
        return []
    reg_in, pos_in = _phi_region(cp, inn_all, la)
    reg_out, _ = _phi_region(cp, out_all, lb)
    ti, to, nbd = luts().te[seed]
    res = []
    for tp in luts().tp[seed]:
        out = out_all[reg_out == tp.outer_region]
        if len(out) == 0:
            continue
        for mpos, (region, memtype) in enumerate(tp.inner):
            inn = inn_all[(reg_in == region) & _in_memory(la, pos_in, memtype)]
            if len(inn) == 0:
                continue
            fa = cp.phicorr[inn] >> (D.NPHIBITS_L[la - 1] - 8)
            fb = cp.phicorr[out] >> (D.NPHIBITS_L[lb - 1] - 8)
            d = fb[None, :] - fa[:, None]
            inr = (d < (1 << (nbd - 1))) & (d >= -(1 << (nbd - 1)))
            dm = d & ((1 << nbd) - 1)
            rzok, ir_out = te_rz_window(cp, inn, out, seed)
            if seed in BARREL_SEEDS:
                ii = (dm << NBEND[la]) + cp.bend[inn][:, None]
                io = (dm << NBEND[lb]) + cp.bend[out][None, :]
            else:                                   # inner and outer stubs are PS: 3 bend bits
                idphi = (dm << NRBITS_TE_PT) + ir_out[None, :]
                ii = (idphi << 3) + cp.bend[inn][:, None]
                io = (idphi << 3) + cp.bend[out][None, :]
            ok = inr & (ii < len(ti)) & (io < len(to))
            acc = np.zeros(d.shape, bool)
            acc[ok] = (ti[ii[ok]] > 0) & (to[io[ok]] > 0)
            acc &= rzok
            if work is not None:
                work["te_pairs"] = work.get("te_pairs", 0) + int(rzok.sum())
                work["lut_pairs"] = work.get("lut_pairs", 0) + int(acc.sum())
            for a, b in zip(*np.nonzero(acc)):
                i, j = int(inn[a]), int(out[b])
                g1, g2 = _float_geom(cp, i), _float_geom(cp, j)
                if seed in DISK_SEEDS and g2[0] < g1[0] + R_PROT_DISK:
                    continue
                if seed in OVERLAP_SEEDS and g2[0] < g1[0] + R_PROT_OVERLAP:
                    continue
                ipars = calc_pars(cp, i, j, seed, k)
                if ipars is None:
                    continue
                rinv, phi0, t, z0 = KF.exact_tracklet(*g1, *g2, sector, k)
                if not _gates(ipars, (rinv, phi0, z0), seed, k):
                    continue
                tr = Tracklet(sector, seed, i, j, rinv, phi0, t, z0, ipars,
                              tcid=(seed, tp.index, mpos, int(cp.order[i]), int(cp.order[j])))
                tr.projs = tracklet_projections(ipars, seed, k)
                res.append(tr)
    return res


# ---- projections and matching ------------------------------------------------


# ---- the match processor, clock by clock (MatchProcessor::execute, MatchEngineUnit) ----------------------
MP_MAX = 108                   # maxStep("MP")
N_MEU = 4                      # MatchProcessor nMatchEngines_
N_RZBINS = 1 << ME_BINS_BITS   # VM z bins per phi bin of the match-engine memory
TCID_IDLE = (math.inf,)        # an idle engine's TCID: above every tracklet's
# tracklet processors per merged projection calculator (TrackletConfigBuilder::iMergedTCStr: L1L2 AB, CD, ...,
# L2L3 ABCD, L3L4 AB, CD, L5L6 ABCD): a TP's projections go to page (letter mod this) of memory (letter div)
PC_PAGES = {0: 2, 1: 4, 2: 2, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4}
DUP_MP_LAYERS = (3, 4, 5, 6, 9, 10)     # layersDisksDuplicated*ProjBalance_: L3-L6, D3, D4 (regions B, C; L1L2)
ALPHA_SHIFT = 12                       # Settings alphashift_


def rz_bins(layer):
    """VM rz bins per phi bin of a match-engine memory: 8 on the barrel, 16 on a disc (8 per side)."""
    return N_RZBINS * (1 if layer <= 6 else 2)


def me_fields(cp):
    """Per copy, the match-engine memory's view (VMStubMERouter): (VM bin within the allstub region, z bin,
    fine z, fine phi)."""
    n = len(cp.row)
    vm, zbin, finez, finephi = (np.zeros(n, np.int64) for _ in range(4))
    for L in np.unique(cp.layer):
        m = np.flatnonzero(cp.layer == L)
        nb, nall, nvm = D.NPHIBITS_L[L - 1], NBITS_ALLSTUBS[L - 1], NBITS_VMME[L - 1]
        vm[m] = (cp.phicorr[m] >> (nb - nall - nvm)) & ((1 << nvm) - 1)
        finephi[m] = (cp.phicorr[m] >> (nb - nall - nvm - NFINEPHI)) & ((1 << NFINEPHI) - 1)
        melut = luts().vmrme[int(L)][_vmr_lut_index(cp, m, int(L))]
        zbin[m], finez[m] = melut >> NFINERZ, melut & ((1 << NFINERZ) - 1)
        if L >= 7:
            zbin[m] += np.where(cp.neg[m], 1 << NFINERZ, 0)        # the -z side's bins (VMStubMERouter)
    return vm, zbin, finez, finephi


def meu_finephi(projfinephi, second, shift):
    """MatchEngineUnit::step's projection fine phi for a candidate in the projection's own bin (second 1 when
    shift is -1, else 0) or its neighbour: moved by one VM bin with the OPPOSITE sign to the geometry, so a
    neighbour-bin stub (dphi then 8 +- a few) never passes |dphi| < 3 in CMSSW. Reproduced as CMSSW runs it."""
    if second == 0 and shift == -1:
        return projfinephi - (1 << NFINEPHI)
    if second == 1 and shift == 1:
        return projfinephi + (1 << NFINEPHI)
    return projfinephi


@dataclass
class ProjTemp:
    """MatchProcessor's ProjectionTemp: what a match engine is initialised with."""
    tr: object
    slot: int
    projrinv: int
    projfinerz: int
    projfinephi: int
    ivm_minus: int
    shift: int
    use: list
    ps_seed: bool


class Buffer:
    """trklet::CircularBuffer: 2^nbits slots, one always free."""

    def __init__(self, nbits):
        self.size = 1 << nbits
        self.buf = [None] * self.size
        self.r = self.w = 0

    def empty(self):
        return self.r == self.w

    def full(self):
        return (self.w + 1) % self.size == self.r

    def near(self, k):
        """nearfull (k = 3) / nearfull4 (k = 4): the write pointer within k of the read pointer."""
        return any((self.w + i) % self.size == self.r for i in range(1, k + 1))

    def store(self, x):
        assert not self.full()
        self.buf[self.w] = x
        self.w = (self.w + 1) % self.size

    def read(self):
        x = self.buf[self.r]
        self.r = (self.r + 1) % self.size
        return x

    def peek(self):
        return self.buf[self.r]


class MEU:
    """MatchEngineUnit: reads one VM-memory stub per clock, a one-register pipeline, then the fine-phi,
    fine-rz and bend-table test; passing pairs go to its candidate buffer (8 slots, nearfull at 3). On a disc the
    fine-r cut is the stub module's (PS 1, 2S 3) and the bend table is indexed (diskPS, projrinv, bend) with the
    stub's own bend width (MatchEngineUnit::processPipeline)."""

    def __init__(self, layer, fields, ps=None):
        self.layer, self.fields, self.ps = layer, fields, ps
        self.nbins = rz_bins(layer)
        self.lut = luts().me[layer]
        self.cand = Buffer(3)
        self.idle, self.afs = True, False
        self.good2 = self.good4 = False
        self.s2 = self.s4 = None
        self.p, self.mem = None, None

    def init(self, mem, p):
        self.mem, self.p = mem, p
        self.idle, self.istub, self.iuse, self.good2 = False, 0, 0, False

    def step(self):
        self.good2 = not self.idle and not self.afs
        if not self.good2:
            return
        p = self.p
        first, second = p.use[self.iuse]
        stubs = self.mem.get((p.ivm_minus + second) * self.nbins + p.slot + first, [])
        self.s2 = (stubs[self.istub], p.projfinerz - (1 << NFINERZ) * first,
                   meu_finephi(p.projfinephi, second, p.shift), p)
        self.istub += 1
        if self.istub >= len(stubs):
            self.iuse += 1
            if self.iuse < len(p.use):
                self.istub = 0
            else:
                self.idle = True

    def process(self, bend):
        if self.good4:
            c, fz, fp, p = self.s4
            _, _, finez, finephi = self.fields
            if self.layer <= 6:
                ok = (abs(int(finephi[c]) - fp) < 3 and abs(int(finez[c]) - fz) <= (1 if p.ps_seed else 5)
                      and self.lut[(p.projrinv << NBEND[self.layer]) + int(bend[c])])
            else:
                ps = bool(self.ps[c])
                index = (int(ps) << (4 + NRINV)) + (p.projrinv << (3 if ps else 4)) + int(bend[c])
                ok = (abs(int(finephi[c]) - fp) < 3 and abs(int(finez[c]) - fz) <= (1 if ps else 3)
                      and self.lut[index])
            if ok:
                self.cand.store((p.tr, int(c)))
        self.s4, self.good4 = self.s2, self.good2

    def tcid(self):
        if not self.cand.empty():
            return self.cand.peek()[0].tcid
        if self.good4:
            return self.s4[3].tr.tcid
        if self.good2:
            return self.s2[3].tr.tcid
        return TCID_IDLE if self.idle else self.p.tr.tcid

    def active(self):
        return not self.idle or self.good2 or self.good4 or not self.cand.empty()


def _proj_temp(tr, layer, mem):
    """MatchProcessor::execute's step 1: a projection's VM bins, z slot, fine positions and the bins to read
    (None when every bin it would read is empty)."""
    iphi, iz, iphider, _ = tr.projs[layer]
    nb, nz = D.NPHIBITS_L[layer - 1], D.NZBITS_L[layer - 1]
    nvmbins = 1 << NBITS_VMME[layer - 1]
    nvmbits = NBITS_VMME[layer - 1] + NBITS_ALLSTUBS[layer - 1]
    ivm = (iphi >> (nb - nvmbits)) & (nvmbins - 1)
    extra = (iphi >> (nb - nvmbits - 4)) & 3                   # FPGAWord::bits(nbits - overlapbits - 2, 2)
    ivm_plus = ivm_minus = ivm
    shift = 0
    if extra == 3 and ivm != nvmbins - 1:
        shift, ivm_plus = 1, ivm + 1
    if extra == 0 and ivm != 0:
        shift, ivm_minus = -1, ivm - 1
    projfinephi = (iphi >> (nb - nvmbits - NFINEPHI)) & ((1 << NFINEPHI) - 1)
    ps_seed = tr.seed not in (2, 3)                            # Tracklet::PSseed: inner L1, L2 or a disc
    if layer >= 7:
        return _proj_temp_disc(tr, layer, mem, ivm_minus, ivm_plus, shift, ps_seed)
    offset = 1 if ps_seed else 4                               # Projection::init
    ztemp = iz >> (nz - ME_BINS_BITS - NFINERZ)
    zbin1 = (1 << (ME_BINS_BITS - 1)) + ((ztemp - offset) >> NFINERZ)
    zbin2 = (1 << (ME_BINS_BITS - 1)) + ((ztemp + offset) >> NFINERZ)
    zbin1 = 0 if not 0 <= zbin1 < N_RZBINS else zbin1
    zbin2 = min(zbin2, N_RZBINS - 1)
    second = zbin1 != zbin2
    projfinerz = ((1 << (ME_BINS_BITS + NFINERZ - 1)) + ztemp) - (zbin1 << NFINERZ)
    n = lambda v, s: len(mem.get(v * N_RZBINS + zbin1 + s, []))
    use = [u for u, ok in (((0, 0), n(ivm_minus, 0) != 0), ((1, 0), second and n(ivm_minus, 1) != 0),
                           ((0, 1), ivm_plus != ivm_minus and n(ivm_plus, 0) != 0),
                           ((1, 1), ivm_plus != ivm_minus and second and n(ivm_plus, 1) != 0)) if ok]
    if not use:
        return None
    projrinv = (1 << (NRINV - 1)) - 1 - (iphider >> (NBITS_PHIDER - NRINV))
    return ProjTemp(tr, zbin1, projrinv, projfinerz, projfinephi, ivm_minus, shift, use, ps_seed)


def disk_radius_table(nrbits=8, cache=[]):
    """TrackletLUT::initProjectionDiskRadius: per projected r bin, (finer << 4) | (rbin1 << 1) | (rbin1 != rbin2),
    with the 3 cm offset either side and rbin1 / rbin2 of 8 over [rmindiskvm, rmaxdisk)."""
    if not cache:
        span = D.RMAXDISK - RMINDISKVM
        tab = []
        for ir in range(1 << nrbits):
            r = ir * D.RMAXDISK / (1 << nrbits)
            rbin1 = max(int(N_RZBINS * (r - 3.0 - RMINDISKVM) / span), 0)
            rbin2 = min(max(int(N_RZBINS * (r + 3.0 - RMINDISKVM) / span), 0), N_RZBINS - 1)
            finer = int((1 << (ME_BINS_BITS + NFINERZ)) * ((r - RMINDISKVM) - rbin1 * span / N_RZBINS) / span)
            finer = min(max(finer, 0), (1 << (NFINERZ + 1)) - 1)
            tab.append((finer << (ME_BINS_BITS + 1)) + (rbin1 << 1) + int(rbin1 != rbin2))
        cache.append(np.array(tab, np.int64))
    return cache[0]


def _proj_temp_disc(tr, layer, mem, ivm_minus, ivm_plus, shift, ps_seed):
    """MatchProcessor::execute's step 1 for a disc projection: the predicted bend from MP_ProjectionBend_D<n>
    (r index 5 bits, phi-derivative index 6 bits, the side), the r slot from the disc-radius table (+ 8 on the
    -z side), the second bin flag and the fine r (MatchProcessor.cc:470-511)."""
    iphi, ir, iphider, irder = tr.projs[layer]
    nb = D.NPHIBITS_L[layer - 1]
    nvmbits = NBITS_VMME[layer - 1] + NBITS_ALLSTUBS[layer - 1]
    nrz = D.NRBITS_L[layer - 1] + 1                            # the projected r word: 12 bits
    rindex = (ir >> (nrz - 5 + 1)) & 31
    phiderindex = (iphider >> (NBITS_PHIDER - 6)) & 63
    projrinv = int(luts().projbend[layer][(int(irder < 0) << 11) + (rindex << 6) + phiderindex])
    projfinephi = (iphi >> (nb - nvmbits - NFINEPHI)) & ((1 << NFINEPHI) - 1)
    word = int(disk_radius_table()[(ir >> (nrz - 8 - 1)) >> 1])
    slot = ((word >> 1) & (N_RZBINS - 1)) + (N_RZBINS if irder < 0 else 0)
    second = bool(word & 1)
    projfinerz = word >> 4
    nb2 = rz_bins(layer)
    n = lambda v, s: len(mem.get(v * nb2 + slot + s, []))
    use = [u for u, ok in (((0, 0), n(ivm_minus, 0) != 0), ((1, 0), second and n(ivm_minus, 1) != 0),
                           ((0, 1), ivm_plus != ivm_minus and n(ivm_plus, 0) != 0),
                           ((1, 1), ivm_plus != ivm_minus and second and n(ivm_plus, 1) != 0)) if ok]
    if not use:
        return None
    return ProjTemp(tr, slot, projrinv, projfinerz, projfinephi, ivm_minus, shift, use, ps_seed)


def disc_match(cp, tr, c, layer, icorrshift, icorzshift):
    """MatchProcessor::matchCalculator on a disc (MatchProcessor.cc:776-912), in integers: the projection moved
    to the stub's z, the 2S stub's r from its ring table and its phi corrected by dr x alpha x ialphafact; the
    stub matches iff |dphi * r_stub| < the seed's rphi cut and |dr| < its r cut (PS or 2S tables). Every stub
    inside the cuts is kept: no improving chain on discs (the best values are updated but never read)."""
    iphiproj, irproj, iphider, irder = tr.projs[layer]
    iz = int(cp.iz[c])
    iphi = iphiproj + ((iz * iphider) >> icorrshift)
    ir = irproj + ((((iz * irder) >> (icorzshift - 1)) + 1) >> 1)
    ideltaphi = int(cp.iphi[c]) - iphi
    irstub = int(cp.ir[c])
    tabs = luts().disc_mp[layer]
    ps = bool(cp.ps[c])
    if not ps:
        inner = layer - 7 <= 1
        ialphafact = int(tabs["alphainner" if inner else "alphaouter"][irstub])
        irstub = int(tabs["rDSSinner" if inner else "rDSSouter"][irstub])
    ideltar = irstub - ir
    if not ps:
        ideltaphi += (ideltar * int(cp.alpha[c]) * ialphafact) >> ALPHA_SHIFT
    t = "PS" if ps else "2S"
    return (abs(ideltaphi * irstub) < int(tabs[t + "phicut"][tr.seed])
            and abs(ideltar) < int(tabs[t + "rcut"][tr.seed]))


def match_processor(cp, layer, pages, mem, fields, matches, k=KF.K, max_steps=MP_MAX):
    """One MatchProcessor (MatchProcessor::execute), clock by clock for max_steps clocks: `pages` its input
    projection memories and pages in reading order (each a list of Tracklets in memory order), `mem` its VM
    memory (bin -> copies in memory order). Matches are appended to matches[(id(tracklet), layer)] in the
    order the match calculator accepts them (barrel: the improving chain, best values reset on a new tracklet;
    disc: disc_match)."""
    entries = [list(pg) for pg in pages]
    hasdata = [len(e) > 0 for e in entries]
    meus = [MEU(layer, fields, cp.ps) for _ in range(N_MEU)]
    inbuf = Buffer(3)
    phishift, dzshift, icorrshift, icorzshift = mp_shifts(layer, k)
    L = luts()
    imem = read_address = mem_read_addr = 0
    projdata = projdata_ = None
    validin = validin_ = validmem = False
    cand = None
    prev = None
    best_phi = best_z = 0
    for _ in range(max_steps):
        near4 = inbuf.near(4)
        for m in meus:
            m.afs = m.cand.near(3)
        best = min(range(N_MEU), key=lambda i: meus[i].tcid())
        if cand is not None and layer >= 7:                    # matchCalculator, disc
            tr, c = cand
            if disc_match(cp, tr, c, layer, icorrshift, icorzshift):
                matches.setdefault((id(tr), layer), []).append(c)
        elif cand is not None:                                 # matchCalculator, barrel
            tr, c = cand
            if tr is not prev:
                best_phi, best_z = int(L.phicut[layer][tr.seed]), int(L.zcut[layer][tr.seed])
                prev = tr
            iphiproj, izproj, iphider, izder = tr.projs[layer]
            ir = int(cp.ir[c])
            dphi = (int(cp.iphi[c]) - (iphiproj + ((ir * iphider) >> icorrshift))) << phishift
            dz = (int(cp.iz[c]) - (izproj + ((ir * izder + (1 << (icorzshift - 1))) >> icorzshift))) << dzshift
            if KNOWN_BAD == "mp_first_in_window":
                ok = improving(dphi, dz, best_phi, best_z)[0] and not matches.get((id(tr), layer))
            else:
                ok, best_phi, best_z = improving(dphi, dz, best_phi, best_z)
            if ok:
                matches.setdefault((id(tr), layer), []).append(c)
        cand = meus[best].cand.read() if not meus[best].cand.empty() else None
        added = False
        for m in meus:
            if not added and m.idle and not inbuf.empty():
                m.init(mem, inbuf.read())
                added = True
            else:
                m.step()
            m.process(cp.bend)
        if validin_:
            pt = _proj_temp(projdata_, layer, mem)
            if pt is not None:
                inbuf.store(pt)
        projdata_, validin_ = projdata, validin
        validin = validmem
        if validin:
            projdata = entries[imem][read_address]
        validmem = not near4
        read_address = mem_read_addr
        if validmem:
            if any(hasdata):
                imem = hasdata.index(True)
                if mem_read_addr + 1 >= len(entries[imem]):
                    mem_read_addr = 0
                    hasdata[imem] = False
                else:
                    mem_read_addr += 1
            else:
                validmem = False
        if (not validin and not validin_ and not validmem and inbuf.empty() and cand is None
                and not any(m.active() for m in meus)):
            break                                              # drained: the remaining clocks do nothing


def run_matching(cp, trs, caps=True, rng=None, k=KF.K):
    """Every match processor of one sector: the projections of the sector's tracklets (trs) per (layer,
    allstub region), split as TrackletConfigBuilder wires them (the duplicated MPs of L3-L6, D3, D4 regions B, C
    take the L1L2 projections), their input memories ordered by seed and merged projection calculator and
    their pages by tracklet processor, against the region's VM memory (alive, me_ok copies, in reading order;
    `rng` permutes the order within each bin: the random order mode). Fills tr.matches."""
    if not trs:
        return
    sector = trs[0].sector
    fields = me_fields(cp)
    vm, zbin = fields[0], fields[1]
    inst = {}
    for tr in sorted(trs, key=lambda q: q.tcid):
        for L, pj in tr.projs.items():
            reg = pj[0] >> (D.NPHIBITS_L[L - 1] - NBITS_ALLSTUBS[L - 1])
            dup = L in DUP_MP_LAYERS and reg in (1, 2) and tr.seed == 0   # MP_<L>PHI{B,C}_E
            pc, page = divmod(tr.tcid[1], PC_PAGES[tr.seed])
            inst.setdefault((L, reg, dup), {}).setdefault((tr.seed, pc, page), []).append(tr)
    matches = {}
    for (L, reg, _), mems in inst.items():
        sel = np.flatnonzero((cp.region == sector) & (cp.layer == L) & cp.alive & cp.me_ok)
        nb, nall = D.NPHIBITS_L[L - 1], NBITS_ALLSTUBS[L - 1]
        sel = sel[(cp.phicorr[sel] >> (nb - nall)) == reg]
        key = cp.order[sel] if rng is None else rng.permutation(len(sel))
        mem = {}
        for c in sel[np.argsort(key, kind="stable")]:
            mem.setdefault(int(vm[c]) * rz_bins(L) + int(zbin[c]), []).append(int(c))
        pages = [mems[q] for q in sorted(mems)]
        match_processor(cp, L, pages, mem, fields, matches, k, MP_MAX if caps else 10 ** 9)
    for tr in trs:
        for L in targets(tr.seed):
            got = matches.get((id(tr), L))
            if got:
                tr.matches[L] = got


def improving(dphi, dz, best_phi, best_z):
    """MatchProcessor::matchCalculator's rule for one candidate (MatchProcessor.cc:684-714): kept iff
    |dphi| <= best and -best <= dz < best, and its |dphi|, |dz| then become the best. Returns (kept, best_phi,
    best_z); the best values start at the seed's cuts for each new tracklet."""
    if abs(dphi) <= best_phi and -best_z <= dz < best_z:
        return True, abs(dphi), abs(dz)
    return False, best_phi, best_z


# ---- candidates, fit, duplicate removal -------------------------------------------
@dataclass
class Candidate:
    tracklet: Tracklet
    stubs: list              # Copies indices: seed stubs, then matches by layer
    tb: int


@dataclass
class Track:
    sector: int
    seed: int
    rows: tuple              # L1TOTStub rows of the fitted stubs
    out: object              # cmssw_kf.FitOutput
    input_rows: tuple        # the candidate's input stub rows
    tracklet: object = None  # the (preferred) Tracklet


def kf_region(rinv, phi0, t, z0, k=KF.K):
    """The KF eta region (KFbase::kalmanLayer's, from HybridFit's seed) of a tracklet's float parameters:
    0-3 is the barrel the closure judges (z0 + 50 t inside |eta| 0.9)."""
    return KF.kf_eta_region(KF.make_seed(rinv, phi0, t, z0, 0, 0, k).eta_reg)


def kf_stub(cp, c):
    """The KF's stub (cmssw_kf.make_stub): barrel by layer; a disc stub by its disc number, with the 2S alpha from
    innerStrip (truncated, as L1TStub's unsigned int) and isFlipped."""
    L = int(cp.layer[c])
    if L <= 6:
        return KF.make_stub(int(cp.row[c]), cp.x[c], cp.y[c], cp.z[c], L, bool(cp.ps[c]), bool(cp.tilted[c]))
    return KF.make_stub(int(cp.row[c]), cp.x[c], cp.y[c], cp.z[c], L - 6, bool(cp.ps[c]), bool(cp.tilted[c]),
                        barrel=False, strip=cp.strip[c], flipped=bool(cp.flipped[c]))


def fit(cp, tr, stub_idx, k=KF.K):
    stubs = [kf_stub(cp, c) for c in stub_idx]
    seed = KF.make_seed(tr.rinv, tr.phi0, tr.t, tr.z0, tr.sector, len(stubs), k)
    state = KF.do_kf(seed, stubs)
    out = KF.fitted_output(state, seed, k)
    return out


def build_candidates(cp, sector, tracklets, caps=True):
    """FitTrack: tracklets with matches in >= 2 layers, per track builder in TCID order (cap 108)."""
    cands = []
    count = {0: 0, 1: 0}
    for tr in sorted(tracklets, key=lambda q: q.tcid):
        if not tr.matches:
            continue
        tb = TB_OF_SEED[tr.seed]
        if caps and count[tb] >= TB_MAX:
            continue
        count[tb] += 1
        if len([L for L, v in tr.matches.items() if v]) < 2:
            continue
        stubs = [tr.inner, tr.outer] + [c for L in sorted(tr.matches) for c in tr.matches[L]]
        cands.append(Candidate(tr, stubs, tb))
    return cands


def phi_res(cp, tr, c, k=KF.K):
    """PurgeDuplicate::getPhiRes: 0 for a seed-layer stub, else |deltaPhi(stub phi, phiproj + phimin)|, with
    phiproj the integer projection times kphi (ProjectionCalculator)."""
    L = int(cp.layer[c])
    if L in SEEDS[tr.seed]:
        return 0.0
    phiproj = tr.projs[L][0] * D.kphi(L, k) + (tr.sector * k.dphisector - 0.5 * k.dphisector_hg)
    if KNOWN_BAD == "comparebest_stub_r":
        phiproj -= (math.hypot(cp.x[c], cp.y[c]) - D.rmean(L)) * 0.5 * tr.rinv
    return abs(KF.delta_phi(math.atan2(cp.y[c], cp.x[c]), phiproj))


def dr_input(cands, caps=True):
    """PurgeDuplicate's input in one sector: the candidates of builder 0, then builder 1, up to 108 (caps)."""
    inp = [c for tb in (0, 1) for c in cands if c.tb == tb]
    return inp[:DR_MAX] if caps and KNOWN_BAD != "dr_cap_off" else inp


def merge_dr(cp, cands, caps=True, k=KF.K):
    """PurgeDuplicate (merge) for one sector; returns the surviving (tracklet, merged stub list)."""
    inp = dr_input(cands, caps)
    n = len(inp)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (SEED_RANK[inp[i].tracklet.seed], i))
    srt = [inp[i] for i in order]
    ids = [[(int(cp.layer[c]), int(cp.row[c])) for c in s.stubs] for s in srt]
    lists = [list(s.stubs) for s in srt]
    best = []
    for s in srt:
        b = {}
        for pos, c in enumerate(s.stubs):
            L = int(cp.layer[c])
            if L not in b:
                b[L] = pos
            elif phi_res(cp, s.tracklet, c, k) < phi_res(cp, s.tracklet, s.stubs[b[L]], k):
                b[L] = pos
        best.append(b)
    dup = np.zeros((n, n), bool)
    merged = np.zeros(n, bool)
    cm = 0
    for i in range(n - 1):
        if merged[i]:
            continue
        cm += 1
        if caps and cm > DR_COMPARED and KNOWN_BAD != "comparators_unlimited":
            continue
        for j in range(i + 1, n):
            share = sum(1 for L, pi in best[i].items() if L in best[j] and ids[i][pi] == ids[j][best[j][L]])
            if share >= (2 if KNOWN_BAD == "nshare_2" else MIN_SHARE):
                dup[i, j] = dup[j, i] = True
                merged[j] = True
    rejected = np.zeros(n, bool)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if dup[i, j]:
                have = {(int(cp.layer[c]), int(cp.row[c])) for c in lists[i]}
                if KNOWN_BAD != "kf_preferred_only":
                    lists[i] = lists[i] + [c for c in lists[j] if (int(cp.layer[c]), int(cp.row[c])) not in have]
                rejected[j] = True
    return [(srt[i].tracklet, lists[i]) for i in range(n) if not rejected[i]]


def proj_memory_exists(seed, pc, layer, region):
    """Whether the merged projection calculator `pc` of `seed` has a memory for (layer, region): one of its TPs
    has the projection outside TrackletConfigBuilder's emptyProjCombined (luts/TP_empty_projections.txt)."""
    L = luts()
    tps = [t.index for t in L.tp[seed] if t.index // PC_PAGES[seed] == pc]
    return any((seed, i, layer, region) not in L.empty_proj for i in tps)


def wired_projections(tracklets):
    """ProjectionCalculator writes a projection only into a memory that exists (proj_memory_exists); the others
    are lost. Off with FIXES off (format 35)."""
    if not FIXES:
        return
    for tr in tracklets:
        pc = tr.tcid[1] // PC_PAGES[tr.seed]
        for L in list(tr.projs):
            reg = tr.projs[L][0] >> (D.NPHIBITS_L[L - 1] - NBITS_ALLSTUBS[L - 1])
            if not proj_memory_exists(tr.seed, pc, L, reg):
                del tr.projs[L]


PC_MAX = 108                  # maxStep("PC")


def pc_cap(tracklets, caps=True):
    """ProjectionCalculator's step cap (caps on; ProjectionCalculator::execute nPar1 / nPar2): a merged
    projection calculator forwards the first 108 tracklets of its TPs (TP order, then each TP's tracklet order) to
    the track builder and projects them; later ones are lost. Returns the kept tracklets. Off with FIXES off
    (format 35)."""
    if not caps or not FIXES:
        return list(tracklets)
    count = {}
    kept = []
    for tr in sorted(tracklets, key=lambda q: q.tcid):
        key = (tr.seed, tr.tcid[1] // PC_PAGES[tr.seed])
        count[key] = count.get(key, 0) + 1
        if count[key] <= PC_MAX:
            kept.append(tr)
    return kept


def projection_pages(tracklets):
    """TrackletProjectionsMemory's page size: a TrackletProcessor's tracklets (its page of each merged memory)
    project into one (layer, phi region) at most 63 times, in its tracklet order; later projections are lost."""
    count = {}
    for tr in sorted(tracklets, key=lambda q: q.tcid):
        for L in list(tr.projs):
            reg = tr.projs[L][0] >> (D.NPHIBITS_L[L - 1] - NBITS_ALLSTUBS[L - 1])
            key = (tr.seed, tr.tcid[1], L, reg)
            count[key] = count.get(key, 0) + 1
            if count[key] > PROJ_PAGE_MAX:
                del tr.projs[L]


def run_event(stub, dtc, caps=True, order="dtc", seed=0, k=KF.K, do_dr=True, collect=None):
    """The chain on one event from its L1TOTStub and L1TOTStubDTC columns (run_copies)."""
    return run_copies(event_copies(stub, dtc), caps, order, seed, k, do_dr, collect)


def run_copies(cp, caps=True, order="dtc", seed=0, k=KF.K, do_dr=True, collect=None):
    """The chain on one event's Copies. Returns (no-DR tracks, DR tracks), each a list of Track.
    order: "dtc" (stream order) or "random" (match candidates permuted, numpy seed `seed`).
    collect: an optional dict that receives the Copies ("copies"), the tracklets ("tracklets"), the
    track-builder candidates ("candidates") and the duplicate-removal input ("dr_input")."""
    if collect is not None:
        collect["copies"] = cp
    input_caps(cp, caps)
    rng = None if order == "dtc" else np.random.default_rng(seed)
    nodr, dr = [], []
    for sector in range(KF.N_SECTOR):
        trs = []
        for sd in active_seeds():
            trs += seed_sector(cp, sector, sd, k)
        trs = pc_cap(trs, caps)
        wired_projections(trs)
        projection_pages(trs)
        run_matching(cp, trs, caps, rng, k)
        cands = build_candidates(cp, sector, trs, caps)
        if collect is not None:
            collect.setdefault("tracklets", []).extend(trs)
            collect.setdefault("candidates", []).extend(cands)
            collect.setdefault("dr_input", []).extend(dr_input(cands, caps))
        for c in cands:
            out = fit(cp, c.tracklet, c.stubs, k)
            if out.accepted and out.consistent_sector:
                nodr.append(Track(sector, c.tracklet.seed, tuple(s.idx for s in out.stubs), out,
                                  tuple(int(cp.row[x]) for x in c.stubs), c.tracklet))
        if do_dr and KNOWN_BAD == "score_dr":
            dr += score_dr([t for t in nodr if t.sector == sector])
        elif do_dr:
            for tr, lst in merge_dr(cp, cands, caps, k):
                out = fit(cp, tr, lst, k)
                if out.accepted and out.consistent_sector:
                    dr.append(Track(sector, tr.seed, tuple(s.idx for s in out.stubs), out,
                                    tuple(int(cp.row[x]) for x in lst), tr))
    return nodr, dr


def score_dr(tracks):
    """The census's score DR (seed_arity.duplicate_removal with kf_emulation.rank_score) on fitted tracks, for
    the score_dr known-bad: best score first (stubs, then chi2 per stub), a track sharing >= 3 stubs with a
    kept one is dropped."""
    def score(t):
        n = len(t.rows)
        return n - 0.02 * (float(t.out.chi2rphi) + float(t.out.chi2rz)) / max(n, 1)
    kept, owners = [], {}
    for t in sorted(tracks, key=score, reverse=True):
        shared = {}
        for r in t.rows:
            for o in owners.get(r, ()):
                shared[o] = shared.get(o, 0) + 1
        if shared and max(shared.values()) >= MIN_SHARE:
            continue
        for r in t.rows:
            owners.setdefault(r, []).append(len(kept))
        kept.append(t)
    return kept


# ---- the census's view: the chain on one census chunk ---------------------------------
class ChunkChain:
    """The chain's inputs for one census chunk (tp_findability U["ot_in"]: every L1TOTStub row and every
    L1TOTStubDTC copy, flat, with event numbers) and the map of the chunk's stubs onto the census table U
    (U["event"], U["ot_row"]; -1 where the census has no such stub)."""

    def __init__(self, ot_in, u_event, u_ot_row):
        self.stub, self.dtc = ot_in["stub"], ot_in["dtc"]
        self.ev, self.row = ot_in["event"], ot_in["row"]
        self.events = np.unique(self.ev)
        self._cp = {}
        sel = np.asarray(u_ot_row) >= 0
        key = (np.asarray(u_event)[sel].astype(np.int64) << 32) | np.asarray(u_ot_row)[sel]
        order = np.argsort(key, kind="stable")
        self._ukey, self._uidx = key[order], np.flatnonzero(sel)[order]

    def copies(self, ev):
        """The event's Copies (event_copies), built once."""
        if ev not in self._cp:
            a, b = np.searchsorted(self.ev, [ev, ev + 1])
            c, d = np.searchsorted(self.dtc["event"], [ev, ev + 1])
            self._cp[ev] = event_copies({k: v[a:b] for k, v in self.stub.items()},
                                        {k: v[c:d] for k, v in self.dtc.items()})
        return self._cp[ev]

    def to_u(self, ev, rows):
        """U indices of L1TOTStub rows `rows` of event `ev` (-1: not in the census table)."""
        key = (np.int64(ev) << 32) | np.asarray(rows, np.int64)
        if not len(self._ukey):
            return np.full(len(key), -1, np.int64)
        p = np.minimum(np.searchsorted(self._ukey, key), len(self._ukey) - 1)
        return np.where(self._ukey[p] == key, self._uidx[p], -1)


def census_pairs(chain, seed, k=KF.K):
    """The stub pairs CMSSW seeds with `seed` (0-3) in the chunk, as census table indices: every pair that
    forms a tracklet in at least one sector (a pair formed in two overlapping sectors is one census pair).
    Returns (gA inner, gB outer, work), pairs in event order; pairs with a stub outside the census table are
    dropped and counted in work["pairs_outside_census"]."""
    work = {}
    ga, gb = [], []
    for ev in chain.events:
        cp = dataclasses.replace(chain.copies(ev), alive=None, te_ok=None, me_ok=None)   # no caps: A-0 only
        pairs = set()
        for sector in range(KF.N_SECTOR):
            for tr in seed_sector(cp, sector, seed, k, work):
                pairs.add((int(cp.row[tr.inner]), int(cp.row[tr.outer])))
        if not pairs:
            continue
        rows = np.array(sorted(pairs), np.int64)
        a, b = chain.to_u(ev, rows[:, 0]), chain.to_u(ev, rows[:, 1])
        ok = (a >= 0) & (b >= 0)
        work["pairs_outside_census"] = work.get("pairs_outside_census", 0) + int((~ok).sum())
        ga.append(a[ok])
        gb.append(b[ok])
    cat = (lambda x: np.concatenate(x) if x else np.zeros(0, np.int64))
    return cat(ga), cat(gb), work
