#!/usr/bin/env python
"""IT-vs-OT tracklet cost, with the REAL OT topology and windows.

Supersedes tracklet_cost_model.py, which modelled one seed pair projecting to one
layer on both sides. That was comparable but it was not what the OT does, so it
understated the OT baseline by ~7x in projection paths.

WHAT IS DIFFERENT HERE
  * OT seeds, projections, Tracklet-Calculator counts and match windows are read
    from L1Trigger/TrackFindingTracklet/interface/Settings.h, not invented.
  * Two OT study points. "full" is the OT's real phase space. "matched" is
    |eta| < ETA_MATCHED with EVERY disc-inclusive seed and EVERY disk projection
    removed, because a track inside that range stays in the barrel -- so
    including D1D2/D3D4/L1D1/L2D1 would charge the OT for phase space the
    comparison has excluded.
  * IT topology per 3-layer configuration: all THREE pair seeds (so a missing
    layer does not kill the seed) each projecting to the remaining layer, plus
    ONE 3-layer displaced seed that needs no projection. Mirrors the OT's
    prompt-pairs-plus-displaced-triplets structure.
  * The sensor angle words are QUANTIZED as a stage, so the bit-budget
    benchmarks are measured through the pipeline instead of estimated
    analytically.

WHAT THIS STILL CANNOT SAY
  * The OT model runs the same gate logic on stubs as the IT model runs on
    clusters. Real OT firmware differs in detail (its TE uses precomputed
    lookup tables on binned r/phi/z/bend, not arithmetic gates), so treat the
    OT numbers as "the same algorithm on OT data", not as an OT emulation.
  * sigma(z0) on the IT side is the section-4a OPTIMISTIC BOUND (the PixelAV
    payload's own smear). If the true resolution is worse, every quantization
    penalty here shrinks and the case for more bits weakens.
"""
from __future__ import annotations
import argparse, json, os, resource
from pathlib import Path
import sys as _sys
import awkward as ak
import numpy as np
import uproot

from ngtagger.data import spix_nano as SPN
import ot_chain as OC

IT_TABLE, OT_TABLE = SPN.CLUSTER_TABLE, SPN.OT_TABLE
C_BEND = 0.29979246 * 3.8 / 2.0 / 100.0
Z_LUMI, NSIG = 15.0, 3.0
# LEGACY per-layer pull widths of the SmartPixels angles at scale 1, which
# scale the reported per-cluster sigmas (sigGlobalClusterPhi for alpha,
# sigGlobalClusterCotTheta for beta) in it_prepare, for the paused / pre-D10
# studies only. Source: canon_v2 f03-f10 (site_v2/emu_pulls_table.{txt,json}):
# barrel clusters linked to a charged TP with a consistent helix, core RMS about
# zero of a mode-seeded 3-sigma clip, pT >= 2 (1-2 GeV within 0.03). Alpha
# 0.958/0.988/0.915/0.899 and beta 0.836/0.831/0.837/0.845 on IL1-IL4. The
# CENSUS no longer uses them (census format 33, WP-D0b): its angles come from
# the shared angle module and the locked per-surface x signed-incidence table
# (census_angles), the inclusive limit of which these scales are.
ALPHA_PULL_SCALE = {1: 0.96, 2: 0.99, 3: 0.92, 4: 0.90}
BETA_PULL_SCALE = {1: 0.84, 2: 0.83, 3: 0.84, 4: 0.85}
THETA_MS_MRAD = 1.36             # per ~1% X0 layer at 1 GeV  [ASSUMED]
ETA_MATCHED = 1.4
PT_MINS = (2.0, 1.5, 1.0)

# THE OT-TRUNCATED REGION (definition sheet census.ot_truncated; decision 12 =
# D13 b). The census reads OT barrel stubs only, up to |stub eta| ETA_MATCHED,
# and the OT barrel layers end between |eta| 0.94 (the outermost) and 1.31 (the
# innermost): above 0.94 a particle leaves part of its OT hits on the disks the
# census does not read. Every census figure whose |eta| range reaches into this
# region carries the label, until the disc work.
OT_TRUNCATED_ETA = (0.94, ETA_MATCHED)
OT_TRUNCATED_LABEL = "OT truncated"


def eta_range_label(lo, hi):
    """OT_TRUNCATED_LABEL when the |eta| range [lo, hi) overlaps OT_TRUNCATED_ETA,
    else ''."""
    return OT_TRUNCATED_LABEL if lo < OT_TRUNCATED_ETA[1] and hi > OT_TRUNCATED_ETA[0] else ""


# ---- prompt d0 allowance -----------------------------------------------------
# The pair phi window must admit a track whose production point is NOT on the
# beamline: a d0 displaces the azimuth by ~d0/r, worst at the innermost layer.
# An earlier revision omitted this entirely ("d0-blind"), which cost 42-61% of
# findable tracks because a d0 of only 100 um is 3.3 mrad at r = 2.9 cm against
# a ~11 mrad total window.
#
# MEASURED on PU200 ttbar, TPs above 2 GeV: the |d0| distribution is a 9.2 um
# robust-sigma CORE (the beamspot) plus a heavy tail that is REAL PHYSICS -- B
# and D decays, material secondaries -- not resolution.
#
# AN EARLIER REVISION CONCLUDED "SIZE THE WINDOW ON THE CORE". THAT WAS WRONG,
# and wrong in the direction that guts the target physics. The tail being real
# is the reason to COVER it. B hadrons have c*tau ~ 450-500 um and D hadrons
# 100-300 um, and a decay product's impact parameter is O(c*tau) largely
# independent of the parent boost, so heavy-flavour tracks sit at d0 ~ 100 um to
# a few mm -- exactly the band a core-sized window throws away. A seeder that
# only reaches 37 um finds only the tracks nobody needed help with.
#
# THREE REGIMES, not two. The prompt path must cover the first TWO:
#   prompt core      ~10 um   (beamspot)
#   heavy flavour    100 um - few mm   <-- THE TARGET
#   long-lived       cm       (separate displaced treatment)
# Sizing "prompt" at the core and "displaced" at 5 mm leaves heavy flavour
# served by neither, which is the gap this scan exists to close.
D0_CORE_SIGMA_CM = 9.2e-4
# TWO TARGETS, costed separately because they need different mechanisms.
#
# HEAVY FLAVOUR (100 um - 1 mm) rides the PROMPT pair seeds via curvature
# widening. This is the OT-equivalent path: VERIFIED that the Hnpar=5 prompt
# collection seeds from pairs only (nbitsseed_ = 3 -> 8 seeds, while the four
# displaced triplets sit behind nbitsseedextended_ = 4), and recovers d0 in the
# 5-parameter fit afterwards rather than at seeding.
#
# BSM DISPLACED (cm) cannot ride that path: matching a 1 cm d0 by curvature
# widening needs kappa to +-10, i.e. no pT floor at all. It requires the triplet,
# where three azimuths determine (kappa, phi0, d0) exactly.
# The triplet's d0 acceptance, cut on the SOLVED d0 rather than assumed. B
# hadrons have c*tau ~ 450-500 um and D hadrons 100-300 um, and a decay
# product's impact parameter is O(c*tau) largely independent of the parent
# boost, so 1 mm covers the heavy-flavour target.
TRIPLET_D0_MAX_CM = 0.1
HF_D0_SETTINGS = {"prompt_only": 0.0, "100um": 100e-4, "500um": 500e-4, "1mm": 1000e-4}
BSM_D0_SETTINGS = {"1mm": 0.1, "5mm": 0.5, "10mm": 1.0}
D0_DISPLACED_CM = 0.5

# ---- IT topology: 3 pair seeds + 1 displaced triplet, per configuration ----
# The SPANNING pair (L1L3 in the first config, L2L4 in the second) is dropped: it
# was the most expensive of the three (3594 tracklets/event against 1945 and 727)
# and it projects to a layer the two adjacent pairs already cover, so it bought
# messier seeds and no acceptance.
IT_CONFIGS = {
    "L1L2L3": {"pairs": [((1, 2), 3), ((2, 3), 1)], "triplet": (1, 2, 3)},
    "L2L3L4": {"pairs": [((2, 3), 4), ((3, 4), 2)], "triplet": (2, 3, 4)},
}

# ---- OT topology, from Settings.h --------------------------------------------
# seedlayers_ / projlayers_ / projdisks_ / ntc_ .  Barrel layers 1-6, disks 1-5.
OT_SEEDS = {
    "L1L2":   {"pair": (1, 2), "proj_l": (3, 4, 5, 6), "proj_d": (1, 2, 3, 4), "ntc": 12, "disc": False},
    "L2L3":   {"pair": (2, 3), "proj_l": (1, 4, 5, 6), "proj_d": (1, 2, 3, 4), "ntc": 4,  "disc": False},
    "L3L4":   {"pair": (3, 4), "proj_l": (1, 2, 5, 6), "proj_d": (1, 2),       "ntc": 4,  "disc": False},
    "L5L6":   {"pair": (5, 6), "proj_l": (1, 2, 3, 4), "proj_d": (),           "ntc": 4,  "disc": False},
    "D1D2":   {"pair": None,   "proj_l": (1, 2),       "proj_d": (3, 4, 5),    "ntc": 4,  "disc": True},
    "D3D4":   {"pair": None,   "proj_l": (1,),         "proj_d": (1, 2, 5),    "ntc": 4,  "disc": True},
    "L1D1":   {"pair": None,   "proj_l": (),           "proj_d": (2, 3, 4, 5), "ntc": 8,  "disc": True},
    "L2D1":   {"pair": None,   "proj_l": (1,),         "proj_d": (2, 3, 4),    "ntc": 4,  "disc": True},
    # displaced triplets; the pair is the FIRST TWO entries of seedlayers_
    "L2L3L4": {"pair": (3, 4), "third": 2, "proj_l": (1, 5, 6), "proj_d": (1, 2, 3), "ntc": 10, "disc": False},
    "L4L5L6": {"pair": (5, 6), "third": 4, "proj_l": (1, 2, 3), "proj_d": (),        "ntc": 10, "disc": False},
    "L2L3D1": {"pair": None,   "proj_l": (1,),         "proj_d": (2, 3, 4),    "ntc": 10, "disc": True},
    "D1D2L2": {"pair": None,   "proj_l": (1,),         "proj_d": (3, 4),       "ntc": 10, "disc": True},
}
# rphimatchcut_ / zmatchcut_ [cm], indexed [layer][seed]; 0.0 means "not used".
OT_SEED_ORDER = ["L1L2", "L2L3", "L3L4", "L5L6", "D1D2", "D3D4", "L1D1", "L2D1",
                 "L2L3L4", "L4L5L6", "L2L3D1", "D1D2L2"]
OT_RPHI_CUT = np.array([
    [0.0, 0.1, 0.07, 0.08, 0.07, 0.05, 0.0, 0.05, 0.08, 0.15, 0.125, 0.15],
    [0.0, 0.0, 0.06, 0.08, 0.05, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0],
    [0.19, 0.19, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.4, 0.4, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0]])
OT_Z_CUT = np.array([
    [0.0, 0.7, 5.5, 15.0, 1.5, 2.0, 0.0, 1.5, 1.0, 8.0, 1.0, 1.5],
    [0.0, 0.0, 3.5, 15.0, 1.25, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0],
    [0.7, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
    [3.0, 3.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0, 3.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 0.0, 0.0, 0.0],
    [4.0, 0.0, 9.5, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 0.0, 0.0, 0.0]])

def ot_numerics_key():
    """OT-side numbers that move a hit assignment without moving a column.

    Hashed into the census cache key by KF.numerics_key; see the note there for
    why a table of match windows has to participate in cache identity.
    """
    return {"policy_version": int(OT_POLICY_VERSION),
            # the OT barrel stubs the census keeps (|stub eta| <= this)
            "eta_matched": float(ETA_MATCHED),
            "nsig": float(NSIG), "z_lumi": float(Z_LUMI),
            "theta_ms_mrad": float(THETA_MS_MRAD), "c_bend": float(C_BEND),
            "triplet_d0_max_cm": float(TRIPLET_D0_MAX_CM),
            "d0_displaced_cm": float(D0_DISPLACED_CM),
            "rphi_cut": OT_RPHI_CUT.tolist(), "z_cut": OT_Z_CUT.tolist(),
            "seed_order": list(OT_SEED_ORDER),
            "it_widen": float(OT_IT_SEED_WIDEN),
            "it_proj_z_widen": float(OT_IT_PROJ_Z_WIDEN),
            "bend_te": {k: list(v) for k, v in sorted(OT_BEND_CUT_TE.items())},
            "fe_bend_cut": float(FE_BEND_CUT),
            "bend_lut": ot_luts() is not None,
            "vmr_z": {k: [float(x) for x in v]
                      for k, v in sorted(OT_VMR_Z.items())},
            # the prompt OT doublets' pairs come from the CMSSW seeding emulation (census format 35)
            "ot_chain": OC.numerics_key()}


# ---- window policy for stubs attached to a track AFTER seeding ---------------
# The real finder matches a projected stub inside rphimatchcut_/zmatchcut_ above,
# tabulated per (projection layer, seed) for a 2 GeV seed. Two departures are
# deliberate and are the whole reason this is a named policy rather than a bare
# table lookup:
#
# 1. AN IT SEED MAY CARRY A 1 GeV FLOOR. Its projection into the OT therefore
#    has to cover the extra sagitta and the extra multiple scattering that a
#    2 GeV-tuned window was never sized for. The widening is DERIVED, not
#    guessed: the sagitta term scales as 1/pT, and the scattering term is the
#    difference of the MS angle at 1 and 2 GeV, which together come to 2.5x the
#    tabulated cut. It is NOT the 8-10x a naive "open it until nothing is lost"
#    scan would pick, and the difference is a factor ~16 in candidate pairs.
#
# 2. AN OT SEED KEEPS THE REAL WINDOW. An OT seed has a 2 GeV floor by
#    construction, so widening it would hand the OT baseline combinatorics it
#    does not actually suffer -- overstating the value of the IT angles by
#    handicapping the system they are meant to improve on.
OT_IT_SEED_WIDEN = 2.5

# ---- the IT -> OT z road, DERIVED because no table can supply it ------------
# rphimatchcut_/zmatchcut_ are indexed by OT seed, and an IT seed's lever arm
# into the OT matches no OT seed's: it extrapolates from r ~ 3-15 cm to
# r ~ 23-108 cm. So the road was measured instead -- z_stub - z_projected for
# true IT-pair-to-genuine-OT-stub associations, on seeds that would actually be
# formed (unique cluster per layer, |z0| <= Z_LUMI), 150 events.
#
# WHAT THE MEASUREMENT SAID. Above 2 GeV the existing derived road already
# contains 95-99% on every layer, so nothing needed widening there -- the
# assumption that it was "overly generous" was wrong for z. The deficit is all
# at the 1 GeV floor with the SHORT-LEVER IL1+IL2 seed, and most of it was not
# the road at all but the straight-line projection, now fixed (see the arc
# length note in it_project). After that fix, containment at 1-1.5 GeV is
#     OL1 90.1%   OL3 90.9%   OL4 67.7%   OL6 61.1%
# against 87.8 / 76.7 / 66.8 / 54.2% before.
#
# SET TO 1.0 BECAUSE WIDENING WAS MEASURED TO BE NET NEGATIVE, which is not
# what the containment study on its own predicted. Trying 1.5x at the 1 GeV
# floor: candidate counts IDENTICAL (the z road here gates the final match, not
# the candidate generation, which has its own road), TPs found DOWN 0.5-1%
# (IL2+IL3 1451 -> 1438, IL1+IL4 1242 -> 1229) and fake fractions UP 1-3%
# relative on every seed.
#
# WHY CONTAINMENT DID NOT TRANSLATE. Hit attachment keeps the BEST match per
# layer, not every match in the road. A correct stub just outside the old road
# was usually already losing the arbitration to a wrong stub inside it, so
# admitting it changes little; what the wider road really adds is a wrong stub
# on layers that previously got NO hit -- and an extra wrong hit costs more at
# the KF chi2 acceptance than the occasional recovered correct one gains.
#
# So the road is NOT the limitation above 2 GeV (95-99% contained already), and
# at the 1 GeV floor the limitation is the projection bias, now fixed by the arc
# length, plus scattering tails on OL4-OL6 that no affordable road reaches
# (sigma 4.5 cm over a 100 cm extrapolation; 99% would need ~13 cm). That is a
# physics limit on projecting a 1 GeV IT doublet to the outer OT, and it argues
# for pointing soft projections at OL1-OL3 -- which is what the soft arms do.
#
# Left as a named knob rather than deleted: the measurement above is the reason
# it is 1.0, and someone who changes the attachment rule to keep more than the
# best match per layer would have to re-run it.
OT_IT_PROJ_Z_WIDEN = 1.0

# BUMP THIS WHEN THE WINDOW *LOGIC* CHANGES, not just the numbers in the tables.
# numerics_key hashes constants, so it catches a retuned cut but NOT a rewritten
# ot_match_window -- and enforcing the tabulated window for the first time
# changed every OT track while leaving every constant identical. A hash cannot
# see that; a hand-maintained integer can.
OT_POLICY_VERSION = 8

# VMRouter cuts on the INNER stub of a PROMPT DOUBLET seed. Read off the code
# (VMRouterCM.cc:238-283), not the comments, because two of the comments are
# wrong about their own direction: VMROUTERCUTZL1 says "Max z" and is used as a
# minimum, and VMROUTERCUTZL2 says "Min L2 z" and is one.
#
#   VMROUTERCUTZL2     = 50  MINIMUM on |z| of the L2 stub of L2L3
#   VMROUTERCUTZL1L3L5 = 95  MAXIMUM on |z| of the inner barrel stub otherwise
#   VMROUTERCUTZL1     = 70  MINIMUM, but only for the L1D1 overlap memories
#   VMROUTERCUTRD1D3   = 55  MAXIMUM on r of an inner disk stub
#
# L2L3 IS A FORWARD SEED BY DESIGN, and that is the surprise here: its L2 stub
# is required to be at |z| >= 50, tightened to 52 by a separate hardcoded
# zcutL2L3 in the LUT generator (TrackletLUT.cc:1195-1201) which is the binding
# one. Reading VMROUTERCUTZL2 as a ceiling would have inverted the seed's whole
# purpose -- it exists to cover the barrel-endcap transition, not the core.
#
# SCOPE. These live in the allinnerstubs_ loop, whose memory is read only by
# TrackletProcessor -- so prompt doublets only, inner stub only. The triplet
# path (TrackletProcessorDisplaced) reads plain AllStubs and sees NONE of them.
# The L1 >= 70 and the disk r cuts are for L1D1/L2D1/D1D2/D3D4, which need disk
# surfaces this barrel-only study does not have, so they are recorded and unused.
#
# Values are the INTEGER-QUANTISED thresholds, not the nominal cm: absz is an
# int in units of kz(layer) and the threshold is a double, so L2 rejects 853 and
# accepts 854. L5 carries only 8 z bits (kz = 0.9375) so its "95" is really
# 94.69.
OT_VMR_Z = {
    "L1L2": (0.0, 94.99),    # inner L1, memtype A/B/C/D -> the 95 ceiling
    "L2L3": (52.0, 1e9),     # inner L2, the 52 floor binds over the 50
    "L3L4": (0.0, 94.99),    # inner L3
    "L5L6": (0.0, 94.69),    # inner L5, 8 z bits
}


# ---- bend consistency, PROMPT DOUBLETS ONLY ---------------------------------
# The stub bend measures the local track angle across the two sensors of a
# module, so it is an independent curvature estimate that the pair's own
# two-point kappa can be checked against. TrackletEngineUnit::step does exactly
# this for both stubs of a prompt seed pair (via pttableinner/outer LUTs).
#
# THE TRIPLET SEEDS GET NOTHING, and that is not an omission. In 20_1 there is
# no bend cut anywhere on the displaced path: VMStubsTEMemory short-circuits to
# pass=true when extended() and its bend table is empty, the table setter is
# never called, TripletEngineUnit tests only r/z bins, and bendcutTE_ has rows
# only for the prompt seeds. Adding one would make our purity look BETTER than
# the emulator and our efficiency worse -- inventing a rejection the system
# does not have, on the very seeds the comparison is about.
FE_BEND_CUT = (1.0 / 6.0) ** 0.5     # FEbendcut, Settings.h:812
# bendcutTE_ (inner, outer) in units of FEbendcut, Settings.h:814-821
OT_BEND_CUT_TE = {"L1L2": (2.2, 2.5), "L2L3": (2.0, 2.0),
                  "L3L4": (2.0, 2.6), "L5L6": (2.4, 2.4)}
SENSOR_SPACING_2S = 0.18             # sensorSpacing_2S_, Settings.h:1035
STRIP_PITCH_PS, STRIP_PITCH_2S = 0.01, 0.009      # Settings.h:1017-1018


# bend -> kappa slope per OT layer, filled once per run by calibrate_ot_bend
# from REAL on-track stubs against L1TTrack_rInv. Deliberately not a constant:
# see bend_expected.
OT_BEND_CAL = None


def set_ot_bend_cal(cal):
    global OT_BEND_CAL
    OT_BEND_CAL = dict(cal) if cal else None


def bend_expected(layer, kap):
    """Bend in full strips a track of curvature kap leaves in OT barrel `layer`.

    USES THE MEASURED PER-LAYER SLOPE, not the nominal geometric formula. The
    analytic version, bendstrip with the nominal 1.8 mm spacing and the
    r-thresholded pitch, agrees with the measured slope to 1.3-3.5% on L2-L6 but
    is 29% off on L1, where module tilt (the CF = |sinTilt|*z/r + cosTilt factor
    that only enters the emulator inside getBendCut) matters most. Calibrating
    against L1TTrack_rInv absorbs tilt, the true pitch and the true sensor
    separation without assuming any of them.

    THE SIGN IS MEASURED TWICE, INDEPENDENTLY, and it is opposite to the naive
    geometric one. (a) Against a signed kappa built from two same-particle stub
    azimuths, the L1TOTStub bend branch agrees with -bendstrip on 93-99% of
    genuine barrel stubs per layer. (b) The slope fitted here against real track
    rInv is negative on all six layers (-0.302 to -0.078 per strip). Without the
    minus the gate would reject essentially every CORRECT pair, and it would
    have presented as "the bend cut is too tight" rather than as a sign error.
    Magnitudes confirm the branch is in FULL strips (half-strip steps).
    """
    if OT_BEND_CAL is None:
        raise RuntimeError(
            "bend gate requested with no calibration loaded: call "
            "set_ot_bend_cal(calibrate_ot_bend(...)) first. Refusing to fall "
            "back to the nominal geometric slope, which is 29% wrong on L1.")
    lay = np.asarray(layer, dtype=np.int64)
    s = np.array([OT_BEND_CAL.get(int(L), (np.nan,))[0] for L in range(0, 8)])
    return np.asarray(kap) / s[lay]


def ot_bend_cut(seed_layers):
    """(inner, outer) bend tolerance in full strips, or None if no cut applies."""
    if len(seed_layers) != 2:
        return None
    t = OT_BEND_CUT_TE.get(ot_seed_name(seed_layers))
    return None if t is None else (t[0] * FE_BEND_CUT, t[1] * FE_BEND_CUT)


# ---- the REAL seed-pair acceptance, read from the firmware LUTs -------------
# TP_<seed>_stubpt{inner,outer}cut is a one-bit accept map over (dphi, bend) per
# stub, dumped from CMSSW (see luts/PROVENANCE.md). It supersedes the analytic
# bend gate above and also SUBSUMES the pT cut: a band of dphi bins rejects
# every bend, which is passptcut, so this is one lookup instead of two cuts.
#
# MEASURED, 60 events, genuine same-particle barrel pairs with tpPt >= 2 against
# random same-event pairs of different particles:
#     L1L2 98.4% signal / 200x background rejection
#     L2L3 97.5% / 192x      L3L4 97.2% / 426x      L5L6 95.6% / 565x
# The analytic gate it replaces kept 77-93% of the same signal and needed a
# separate pT bound.
#
# THREE CONVENTIONS HAD TO BE PINNED, and guessing any of them wrong silently
# rejects almost everything (the naive choice gave 2-25% signal). A scan over
# all eight sign combinations has exactly two maxima, mirror images of each
# other; the one adopted here is the physically coherent one:
#   bend  : L1TOTStub_bend is MINUS the firmware's decoded bend. Independently
#           established earlier from the bend-vs-curvature correlation, so this
#           is a second, agreeing measurement rather than a fitted fudge.
#   corr  : phicorr = phi - correction, as Stub::setPhiCorr does.
#   dphi  : idphi = finephi(outer) - finephi(inner), as TrackletEngineUnit does.
#
# THE BEND ENCODING IS RECOVERED, NOT ASSUMED. The tables are indexed by the
# ENCODED bend word, but the ntuple carries the pre-degradation FE bend (9/13/17
# distinct values on PS layers, where the code is only 3 bits). getphiCorrValue
# is linear in each code's decoded bend, so inverting VMPhiCorrL<n> recovers the
# code -> bend map to within 0.8 counts. That map is per LAYER, because
# getBendCut averages over a layer's sensor modules, whereas degradeBend is per
# MODULE -- so a stub whose raw bend falls between two code midpoints may be
# assigned differently than the firmware would. That is the residual
# approximation here, and it is why the signal efficiency is 96-98% and not 100%.
OT_LUT_DIR = Path(__file__).resolve().parent / "luts"
_OT_LUT = None

_LUT_RMAXDISK, _LUT_DELTARZ = 120.0, 32.0
_LUT_IRMEAN = (851, 1269, 1784, 2347, 2936, 3697)
_LUT_NPHIBITS = (14, 14, 14, 17, 17, 17)
_LUT_NBEND = (3, 3, 3, 4, 4, 4)          # N_BENDBITS_PS / _2S


def _lut_tab(p):
    t = open(p).read().strip().strip("{};")
    return np.array([int(x) for x in t.replace("\n", "").split(",") if x.strip()])


def ot_luts():
    """Load and invert the firmware tables once. None if they are not present."""
    global _OT_LUT
    if _OT_LUT is not None:
        return _OT_LUT or None
    d = Path(OT_LUT_DIR)
    if not d.is_dir() or not list(d.glob("TP_*_stubptinnercut.dat")):
        _OT_LUT = {}
        return None
    drmax = _LUT_RMAXDISK / _LUT_DELTARZ
    rinvmax = 0.01 * 0.299792458 * 3.8112 / 2.0
    dphi_hg = 2 * np.pi / 9 + rinvmax * max(55.0 - 21.8, 112.7 - 55.0)
    mid, corr = {}, {}
    for L in range(1, 7):
        nb = _LUT_NBEND[L - 1]
        g = _lut_tab(d / f"VMPhiCorrL{L}.tab").reshape(1 << nb, 8)
        rmean = _LUT_IRMEAN[L - 1] * _LUT_RMAXDISK / 4096
        pitch = STRIP_PITCH_PS if L <= 3 else STRIP_PITCH_2S
        kphi = dphi_hg / (1 << _LUT_NPHIBITS[L - 1])
        delta = ((np.arange(8) + 0.5) * (2.0 * drmax / 8)) - drmax
        coef = -(delta / 0.18) * pitch / rmean / kphi
        mid[L] = np.array([np.dot(g[b], coef) / np.dot(coef, coef)
                           for b in range(1 << nb)])
        corr[L] = g
    tab = {}
    for nm in ("L1L2", "L2L3", "L3L4", "L5L6"):
        fi = sorted(d.glob(f"TP_{nm}?_stubptinnercut.dat"))
        fo = sorted(d.glob(f"TP_{nm}?_stubptoutercut.dat"))
        if not fi or not fo:
            continue
        # every TP instance of a seed is byte-identical (verified), so one each
        ti = np.loadtxt(fi[0], dtype=np.int64)
        to = np.loadtxt(fo[0], dtype=np.int64)
        la = int(nm[1]); lb = int(nm[3])
        nbd = int(round(np.log2(len(ti) >> _LUT_NBEND[la - 1])))
        tab[nm] = (ti, to, nbd)
    _OT_LUT = {"mid": mid, "corr": corr, "tab": tab, "dphi_hg": dphi_hg,
               "kfine": dphi_hg / 256.0, "drmax": drmax}
    return _OT_LUT


def _lut_fine(Lk, phi_rel, bend, r, K):
    """(fine phi bin, bend code) for one stub, in the firmware's encoding."""
    nb = _LUT_NBEND[Lk - 1]
    kphi = K["dphi_hg"] / (1 << _LUT_NPHIBITS[Lk - 1])
    rmean = _LUT_IRMEAN[Lk - 1] * _LUT_RMAXDISK / 4096
    rbin = np.clip(((r - rmean + K["drmax"])
                    / (2 * K["drmax"] / 8)).astype(np.int64), 0, 7)
    m = K["mid"][Lk]
    cand = np.arange(1 << nb)
    cand = cand[cand != (1 << (nb - 1))]      # duplicates code 0; never emitted
    code = cand[np.argmin(np.abs(-np.asarray(bend)[:, None] - m[None, cand]),
                          axis=1)]
    c = K["corr"][Lk][code, rbin]
    return np.floor((phi_rel - c * kphi) / K["kfine"]).astype(np.int64), code


def ot_lut_accept(la, lb, phiA, rA, bA, phiB, rB, bB):
    """Firmware accept/reject for a prompt OT seed pair. None if no LUTs."""
    K = ot_luts()
    if K is None:
        return None
    nm = ot_seed_name((la, lb))
    if nm not in K["tab"]:
        return None
    ti, to, nbd = K["tab"][nm]
    ka, kb = la - 10, lb - 10
    # A COMMON ORIGIN IS ENOUGH: only the DIFFERENCE of the two fine-phi words
    # is indexed, so the nonant's absolute lower edge cancels. Both stubs of a
    # pair are in one sector by construction, so both use the inner stub's.
    n = np.floor((phiA + np.pi) / (2 * np.pi / 9))
    o = n * (2 * np.pi / 9) - np.pi - (K["dphi_hg"] - 2 * np.pi / 9) / 2
    fa, ca = _lut_fine(ka, phiA - o, bA, rA, K)
    fb, cb = _lut_fine(kb, phiB - o, bB, rB, K)
    d = fb - fa
    # inrange is PART OF THE CUT, not a guard: the mask below would alias an
    # out-of-range dphi onto a valid-looking index (TrackletEngineUnit.cc:82).
    inr = (d < (1 << (nbd - 1))) & (d >= -(1 << (nbd - 1)))
    d = d & ((1 << nbd) - 1)
    ii = (d << _LUT_NBEND[ka - 1]) + ca
    io = (d << _LUT_NBEND[kb - 1]) + cb
    ok = inr & (ii < len(ti)) & (io < len(to))
    out = np.zeros(len(d), bool)
    out[ok] = (ti[ii[ok]] > 0) & (to[io[ok]] > 0)
    return out


def ot_seed_name(seed_layers):
    """'L1L2' / 'L2L3L4' for an all-OT seed, else None.

    Accepts layers in CONSTRUCTION order (pair first) and sorts for the name,
    which is how CMSSW labels them too: the enum is Seed::L2L3L4 while the
    module is TPD_L3L4L2.
    """
    ls = [int(L) - 10 for L in seed_layers]
    if not ls or any(L < 1 or L > 6 for L in ls):
        return None
    nm = "".join(f"L{L}" for L in sorted(ls))
    return nm if nm in OT_SEED_ORDER else None


def ot_vmr_inner_z(seed_layers):
    """(|z|min, |z|max) for the inner stub of a prompt OT doublet, else None.

    None for triplets on purpose: TrackletProcessorDisplaced never reads the
    AllInnerStubs memory these cuts gate, so applying them to a triplet would
    delete acceptance the emulator keeps.
    """
    if len(seed_layers) != 2:
        return None
    return OT_VMR_Z.get(ot_seed_name(seed_layers))


# Sentinels, because "no window configured" and "no cap wanted" are OPPOSITE
# instructions and returning None for both silently turned a forbidden
# projection into an unrestricted one.
OT_PROJ_FORBIDDEN = "forbidden"


def ot_match_window(proj_layer, seed_layers):
    """Match window for attaching a stub in OT barrel layer 1-6 to a track.

    Returns
      (rphi_cm, z_cm)      cap the match at this window
      OT_PROJ_FORBIDDEN    an OT seed that the real menu does not let project
                           to this layer at all -- attach nothing
      None                 no cap (layer outside the tabulated range)

    AN OT SEED USES ITS OWN COLUMN, which is the point: rphimatchcut_ is indexed
    [layer][seed] precisely because the window depends on the extrapolation
    distance, and a zero entry means projlayers_ does not contain that layer for
    that seed. Honouring the zero is what restricts an OT seed to the standard
    OT projection menu instead of letting it follow a track into layers the
    firmware never looks in.

    AN IT OR JOINT SEED HAS NO COLUMN. Its lever arm into the OT differs from
    every OT seed's, so there is no honest table entry to borrow; it gets the
    most generous configured window for the layer, widened by the 1 GeV policy
    factor, purely as a CEILING against runaway following. In z that ceiling
    lands beyond the luminous region and is therefore inert -- stated here
    rather than hidden, because it means the IT->OT z window is still the
    derived one and is an open question, not a solved one.
    """
    i = int(proj_layer) - 1
    if not 0 <= i < OT_RPHI_CUT.shape[0]:
        return None
    name = ot_seed_name(seed_layers)
    if name is not None:
        # A TRIPLET HAS ITS OWN COLUMN and it is not its pair's. L2L3L4 projects
        # to L1, L5, L6 where the L3L4 doublet it is built from projects to
        # L1, L2, L5, L6 -- borrowing the pair's column would let the triplet
        # follow tracks into L2, which its own row forbids.
        si = OT_SEED_ORDER.index(name)
        wr, wz = float(OT_RPHI_CUT[i][si]), float(OT_Z_CUT[i][si])
        return (wr, wz) if (wr > 0 and wz > 0) else OT_PROJ_FORBIDDEN
    r_vals = OT_RPHI_CUT[i][OT_RPHI_CUT[i] > 0]
    z_vals = OT_Z_CUT[i][OT_Z_CUT[i] > 0]
    if not len(r_vals) or not len(z_vals):
        return None
    return (float(r_vals.max()) * OT_IT_SEED_WIDEN,
            float(z_vals.max()) * OT_IT_SEED_WIDEN)

# ---- sensor word quantization benchmarks -------------------------------------
# Reserve 3 codes: two SIGNED overflow (sign carries charge for alpha, and
# forward/backward for z0) and one invalid. The two overflows are complementary
# and jointly form the displaced tag: alpha overflow means transversely
# displaced (|d0| beyond ~5 mm at L1) or below the pT floor, z0 overflow means
# longitudinally displaced. Neither alone would cover it.
QUANT_RESERVE = 3
ALPHA_RANGE = 0.34               # +-0.17 in cot(alpha): prompt to 0.5 GeV at L4
Z0_RANGE = 30.0                  # +-15 cm: the luminous region, by physics
BENCHMARKS = {"native": None, "A_a3b5": (3, 5), "B_a4b6": (4, 6), "C_a3b7": (3, 7)}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def seed_code(layers, soft=False):
    """An integer naming a seed's layer set (and the soft recovery flag): the
    census angle cuts' seed bin."""
    c = 0
    for i, L in enumerate(layers):
        c += int(L) * 100 ** i
    return c + (10 ** 7 if soft else 0)


def quantize(v, bits, rng):
    """Mid-tread uniform quantizer over +-rng/2 with signed overflow.

    Returns (value, overflow_mask). Overflow keeps the SIGN, so a saturated word
    still says which way -- the thing a naive clamp destroys.
    """
    n = 2 ** bits - QUANT_RESERVE
    lsb = rng / n
    ovf = np.abs(v) > rng / 2.0
    qv = np.clip(np.round(v / lsb) * lsb, -rng / 2.0, rng / 2.0)
    return np.where(ovf, np.sign(v) * rng / 2.0, qv), ovf, lsb


def quant_sigma(bits, rng):
    return (rng / (2 ** bits - QUANT_RESERVE)) / np.sqrt(12.0)


# Event stride for the composite sort key below. Must exceed the 6*pi band that
# the wrap-around duplication occupies, plus twice the widest search window, so
# one event's band can never reach its neighbour's.
EVENT_STRIDE = 64.0

# ---- candidate-triplet budget ------------------------------------------------
# A projection stage expands to CANDIDATE TRIPLETS: one surviving cluster pair
# tupled with one layer-C cluster inside the projection window. That count -- not
# the event count -- sets peak memory, at ~128 bytes per candidate triplet.
#
# THE 40 GB INCIDENT (2026-09-09), and why --batch-events could not prevent it.
# The projection window carries the d0 allowance, NSIG * (... + d0/r_inner). At a
# displaced point (d0 = 0.5-1 cm against r_inner ~ 2.9 cm) that window is
# ~0.5-1 rad, so each surviving cluster pair matches ~1e3 layer-C clusters
# instead of the ~54 measured on the prompt path. Candidate triplets per event go
# from ~2.2e5 to ~1e8, i.e. tens of GB INSIDE ONE EVENT -- below batch
# granularity, so no batch size can bound it. It exhausted a 24 GB machine.
#
# Two limits, because the failure has two independent halves:
#   PAIR_BUDGET  bounds ONE slice, so memory is flat in the candidate-triplet
#                count however large that count becomes.
#   ROW_CEILING  refuses a seed whose total is absurd, because bounding memory
#                alone converts a crash into a run that never finishes.
# Both are read from the per-cluster-pair candidate counts, which searchsorted
# yields with NO expansion. The previous guard polled RSS between file chunks:
# that can only autopsy an allocation that has already completed.
TRIPLET_BYTES = 8 * 16
PHYS_RAM_GB = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1024.0 ** 3
SAFE_RSS_FRAC = 0.40
DEFAULT_PAIR_BUDGET_GB = 1.5
DEFAULT_TRIPLET_CEILING = 300_000_000


def triplets_for_budget(gb):
    return max(1 << 16, int(gb * (1024.0 ** 3) / TRIPLET_BYTES))


_MAX_SLICE = triplets_for_budget(DEFAULT_PAIR_BUDGET_GB)
_RSS_LIMIT_GB = None
_RSS_DIV = 1024.0 ** 3 if _sys.platform == "darwin" else 1024.0 ** 2


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_DIV


def set_limits(slice_triplets, rss_gb):
    global _MAX_SLICE, _RSS_LIMIT_GB
    _MAX_SLICE, _RSS_LIMIT_GB = slice_triplets, rss_gb


def rss_check():
    """Called per slice, so it can interrupt work IN PROGRESS."""
    if _RSS_LIMIT_GB is not None and peak_gb() > _RSS_LIMIT_GB:
        raise SystemExit(f"ABORT: peak RSS {peak_gb():.1f} GB exceeded ceiling "
                         f"{_RSS_LIMIT_GB:.1f} GB. Lower --pair-budget-gb.")


class TooWide(Exception):
    """A seed's candidate-triplet count exceeds the ceiling.

    Not a crash and not something to raise the ceiling for: it means the
    configuration is combinatorially infeasible as formulated, which is a finding
    to record and move past.
    """

    def __init__(self, n, ceiling):
        super().__init__(f"{n:.3g} candidate triplets exceeds ceiling {ceiling:.3g}")
        self.n, self.ceiling = int(n), int(ceiling)


def _window_index(evA, phiA, evB, phiB, half):
    """Per-A candidate ranges, with NO expansion. O((NA + NB) log NB).

    The event is folded into a composite key
        k = event * EVENT_STRIDE + (phi + pi)
    so a phi-window search is a plain interval search in key space, and because
    EVENT_STRIDE dwarfs the 6*pi band the duplication below occupies, a window
    can never leak into a neighbouring event.

    Wrap-around is handled by duplicating B at k-2pi, k, k+2pi. Both copies of a
    pair can only be inside one window if that window were >= 2pi wide, which it
    never is, so nothing is double counted.

    An earlier revision looped over events and called searchsorted per event.
    That is O(N) per event to build the mask alone, i.e. O(N*E) -- tolerable at
    40 events and hopeless at 1e6. This is O(N log N) once.
    """
    half = np.asarray(half)
    if half.size and float(np.max(half)) > (EVENT_STRIDE - 6.0 * np.pi) / 2.0:
        raise SystemExit("search window too wide for EVENT_STRIDE; raise the stride")
    kb = evB.astype(np.float64) * EVENT_STRIDE + (phiB + np.pi)
    kb3 = np.concatenate([kb - 2.0 * np.pi, kb, kb + 2.0 * np.pi])
    src3 = np.tile(np.arange(len(kb), dtype=np.int64), 3)
    o = np.argsort(kb3, kind="stable")
    kb3, src3 = kb3[o], src3[o]
    ka = evA.astype(np.float64) * EVENT_STRIDE + (phiA + np.pi)
    lo = np.searchsorted(kb3, ka - half, "left")
    hi = np.searchsorted(kb3, ka + half, "right")
    return src3, lo, (hi - lo).astype(np.int64)


def _expand(src3, lo, cnt, a0):
    """Expand one contiguous A-range into explicit (ia, ib); ia is absolute."""
    tot = int(cnt.sum())
    ia = np.repeat(np.arange(a0, a0 + len(cnt), dtype=np.int64), cnt)
    ramp = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return ia, src3[np.repeat(lo, cnt) + ramp]


def pairs_in_window_slices(evA, phiA, evB, phiB, half, max_slice=None,
                           ceiling=DEFAULT_TRIPLET_CEILING):
    """Yield (ia, ib) in CONTIGUOUS SLICES OF A, each at most max_slice wide.

    Slicing on A is what keeps the downstream arbitration exact: every candidate
    sharing an A index lands in the same slice, so a per-A argmin over a slice is
    the same answer as over the whole set. A single A whose own count exceeds
    max_slice is emitted whole rather than split, because splitting it WOULD
    break that property; its count is bounded by 3 * N_B, which is small.

    Raises TooWide, before yielding anything, if the total exceeds `ceiling`.
    """
    if max_slice is None:
        max_slice = _MAX_SLICE
    src3, lo, cnt = _window_index(evA, phiA, evB, phiB, half)
    tot = int(cnt.sum())
    if tot > ceiling:
        raise TooWide(tot, ceiling)
    if not tot:
        return
    edges = np.r_[0, np.cumsum(cnt)]
    a0 = 0
    while a0 < len(cnt):
        a1 = int(np.searchsorted(edges, edges[a0] + max_slice, "right")) - 1
        a1 = max(a1, a0 + 1)                       # always make progress
        yield _expand(src3, lo[a0:a1], cnt[a0:a1], a0)
        a0 = a1


def pairs_in_window(evA, phiA, evB, phiB, half, ceiling=DEFAULT_TRIPLET_CEILING):
    """All (A,B) pairs within +-half in phi, for all events at once.

    Materialises the whole expansion, so it is only for stages whose count is
    known small (the OT stub seeds). Anything driven by a d0-inflated window must
    use pairs_in_window_slices instead.
    """
    src3, lo, cnt = _window_index(evA, phiA, evB, phiB, half)
    tot = int(cnt.sum())
    if tot > ceiling:
        raise TooWide(tot, ceiling)
    if not tot:
        return (np.empty(0, dtype=np.int64),) * 2
    return _expand(src3, lo, cnt, 0)


def expand_inputs(spec):
    """One or many files: comma-separated paths and/or globs, in given order.

    The studies were single-file, which made the 1000-event ttbar set unusable
    without concatenating by hand. Event numbering stays globally unique because
    the chunk loader carries a running offset across every yield, files included.
    """
    import glob as _glob
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        hits = sorted(_glob.glob(part)) if any(c in part for c in "*?[") else [part]
        if not hits:
            raise SystemExit(f"no input matched: {part}")
        out.extend(hits)
    if not out:
        raise SystemExit("no inputs given")
    return out


# ---- TrackingParticle truth: the L1TTP table ---------------------------------
# One row per charged TP with 3-momentum |p| > 1 GeV (L1TrackingParticleTableProducer;
# older nanos cut pT >= 1 GeV, a subset of the same TPs). Every object links its TP
# by tpIdx: >= 0 its ROW in the event's L1TTP, -1 unmatched, -2 truth unavailable,
# <= -3 matched to a TP not written to L1TTP (SPN.tp_matched counts it as matched;
# it has no L1TTP truth). Equal tpIdx <=> same TP.
# phi0/d0/z0 are at the POCA to the beamline, curvature-aware, in the TTTrack
# convention (d0 = x0 sin(phi0) - y0 cos(phi0), as L1TTrack_d0); phi is the
# momentum azimuth at PRODUCTION and vx/vy the production vertex. The per-cluster
# tpPhi/tpVx/tpVy/tpVz columns are production quantities too, and a helix built
# from them is wrong for any TP produced off its POCA, so TP truth is read here
# and nowhere else.
TP_TABLE = "L1TTP"
TP_COLS = ["pt", "eta", "phi", "phi0", "charge", "vx", "vy", "d0", "z0"]
TP_REGEN = "/Users/nmangane/smartpixels/cmssw/work/otstub_arm/itot_tp_f*_100ev.root"
# per-object truth column -> column of the per-TP table built by tp_table
TP_TRUTH = {"tp_pt": "pt", "tp_eta": "eta", "tp_tanL": "tanL",
            "tp_phi0": "phi0", "tp_phi_prod": "phi", "tp_charge": "charge",
            "tp_vx": "vx", "tp_vy": "vy", "tp_d0": "d0", "tp_z0": "z0"}


def require_tp_table(paths):
    """SystemExit naming the first input without the complete L1TTP table."""
    for p in paths:
        keys = set(uproot.open(f"{p}:Events").keys())
        miss = [f"{TP_TABLE}_{c}" for c in TP_COLS if not SPN.has_column(keys, TP_TABLE, c)]
        if miss:
            raise SystemExit(
                f"{p} has no {TP_TABLE} TrackingParticle table (missing "
                f"{', '.join(miss)}). TP truth comes only from {TP_TABLE}; use "
                f"the regenerated nanos {TP_REGEN}.")


def tp_branches(path):
    """The L1TTP branches of files shaped like `path`."""
    return SPN.branch_names(uproot.open(f"{path}:Events").keys(), TP_TABLE, TP_COLS)


def tp_table(A, ev):
    """Per-TP truth of one uproot batch `A`, in L1TTP row order (so sorted by
    tp_key: key = tp_key(event, row)).

    `ev` is the event number of each entry of `A`, increasing, and must be the
    numbering the object tables of the same batch get, or the join pairs TPs of
    different events.
    """
    ev = np.asarray(ev)
    if len(ev) > 1 and not (np.diff(ev) > 0).all():
        raise ValueError("tp_table needs increasing event numbers")
    J = SPN.table_columns(A, TP_TABLE, TP_COLS)
    n = ak.to_numpy(ak.num(J[TP_COLS[0]]))
    T = {c: ak.to_numpy(ak.flatten(J[c])) for c in TP_COLS}
    ev_rep = np.repeat(ev, n)
    T["key"] = tp_key(ev_rep, np.arange(len(ev_rep)) - np.repeat(np.cumsum(n) - n, n))
    T["tanL"] = np.sinh(T["eta"])
    return T


def tp_rows(TP, event, tpidx):
    """(position in the tp_table `TP`, has_row) of each object's L1TTP row:
    the event's first row plus tpIdx, for tpIdx >= 0. Objects without a row
    (unmatched, truth unavailable, or a TP not written to L1TTP) get has_row
    False and position 0. A tpIdx past the event's L1TTP rows is an error."""
    tpidx = np.asarray(tpidx)
    has = tpidx >= 0
    p = np.zeros(len(tpidx), np.int64)
    if has.any():
        ev, row = np.asarray(event)[has], tpidx[has]
        ph = np.searchsorted(TP["key"], tp_key(ev, np.zeros(len(ev), np.int64))) + row
        ok = ph < len(TP["key"])
        ok[ok] = TP["key"][ph[ok]] == tp_key(ev[ok], row[ok])
        if not ok.all():
            raise SystemExit(f"{int((~ok).sum())} tpIdx values point past their event's "
                             f"{TP_TABLE} rows")
        p[has] = ph
    return p, has


def attach_tp_truth(D, TP, names=None):
    """Add to object table D (clusters or stubs) the L1TTP truth of each
    object's TP, read from its L1TTP row (tp_rows). NaN where the object has
    no TP (tpIdx -1/-2) or its TP is not in L1TTP (tpIdx <= -3: neutral, or
    below the L1TTP momentum cut). `names` selects from TP_TRUTH; default all."""
    names = list(TP_TRUTH) if names is None else list(names)
    nan = np.float32(np.nan)
    p, hit = tp_rows(TP, D["event"], D["tpIdx"])
    for nm in names:
        D[nm] = (np.where(hit, TP[TP_TRUTH[nm]][p], nan) if len(TP["key"])
                 else np.full(len(D["tpIdx"]), nan))
    return D


# ---- whom a hit is credited to (definition sheet truth.stub; decision 12 = D13 a)
# A credit means the hit was made by the TP or by one of its unstored Geant4
# daughters: SIM gives a daughter not stored (born outside the tracker volume, or
# with kinetic energy <= 0.2 GeV) its parent's track id
# (TkAccumulatingSensitiveDetector.cc:147-156, 236-249; WDMac
# wpd0a/oot_truth_investigation.md), so a credited hit need not lie on the TP's own
# path. An IT cluster is credited to its dominant TP (tpIdx). An OT stub is credited
# to its principal TP and to the dominant TP of each of its two clusters, the nano's
# emulation of CMSSW's TP -> stubs map, which credits every contributor of either
# cluster (TTStubAssociator.cc:72-88; the CMSSW truth spec, WDMac
# wpd0a/cmssw_truth_spec.md sec. 2):
#   tpIdx       the principal TP: set only on a genuine stub, present in BOTH
#               clusters (occasionally neither cluster's dominant TP); built by
#               the input's CMSSW_20_0_0_pre1 associator (spec sec. 0, 2);
#   tpIdxInner  the dominant TP of the inner-sensor cluster (clusterRef(0)): the
#               largest SUMMED TP pT over its strip simlinks (not charge, so one
#               strip of a high-pT TP can outrank the particle that made most of
#               the cluster); set for combinatoric clusters too;
#   tpIdxOuter  the same for the outer-sensor cluster (clusterRef(1)).
# The nano carries one TP per cluster where CMSSW credits every contributor, so a
# minor contributor goes uncredited here (eval_refitq/wpd0a/owner_vs_cmssw.py
# measures what that costs). An unknown stub (no TP in either cluster) is credited
# to nobody. No bunch-crossing or event cut: out-of-time and pile-up TPs are
# credited like any other. EVERY CREDITED HIT COUNTS THE SAME in every census
# figure, whatever cluster credits it (decision of 2026-10-13); a stub is ONE hit
# for each TP it is credited to, however many of its clusters credit that TP.
# TRUTH ONLY: none of this may reach the seeding, the follow or the fit.
STUB_TP_COLS = ("tpIdx", "tpIdxInner", "tpIdxOuter")
N_CREDIT = len(STUB_TP_COLS)


def credited_tps(*cols):
    """(n, N_CREDIT) int32: the TPs each object is credited to, from its truth
    columns (an IT cluster's tpIdx; an OT stub's STUB_TP_COLS, in that order).
    Each matched TP appears ONCE per object, so a stub whose clusters both credit
    a TP counts once for it; a repeat and an unused slot are -1. Credited: a row
    of L1TTP (>= 0) and a real TP outside L1TTP (<= -3). Not credited: -1 (no TP)
    and -2 (truth missing; refused at the census input,
    tp_findability.require_census_inputs)."""
    T = np.full((len(cols[0]), N_CREDIT), -1, np.int32)
    for j, c in enumerate(cols):
        c = np.asarray(c, np.int32)
        keep = SPN.tp_matched(c)
        if j:
            keep &= ~(T[:, :j] == c[:, None]).any(axis=1)
        T[:, j] = np.where(keep, c, -1)
    return T


def hit_tps(U, G):
    """G.shape + (N_CREDIT,): the TPs each hit index of G is credited to (U["tps"],
    built by credited_tps); -1 throughout where G < 0 (no hit)."""
    G = np.asarray(G)
    return np.where((G >= 0)[..., None], U["tps"][np.clip(G, 0, None)], -1)


def credits_of(D, g):
    """(len(g), N_CREDIT) credited TPs of the objects g of table D: D["tps"] when
    the table carries it (the census's unified table), else credited_tps of the
    truth columns it has (an IT cluster table: tpIdx; an OT stub table:
    STUB_TP_COLS)."""
    return D["tps"][g] if "tps" in D else credited_tps(*(D[c][g] for c in STUB_TP_COLS if c in D))


def seed_credit(D, ga, gb, gc=None):
    """(ok, A): A = the credited TPs of each seed's first hit (credits_of), ok its
    (n, N_CREDIT) mask of the TPs credited by every hit of the seed (A, B and C
    when given), the seed-truthful rule (definition sheet truth.seed_truthful)."""
    A = credits_of(D, ga)
    ok = SPN.tp_matched(A)
    for g in (gb, gc):
        if g is not None:
            ok &= (A[:, :, None] == credits_of(D, g)[:, None, :]).any(axis=2)
    return ok, A


def seed_tp(D, ga, gb, gc=None):
    """(tp, truthful) per seed: the TP every seed hit is credited to, -1 if none
    or if several are (a tie gives no TP, as CMSSW and the majority owner);
    truthful whenever at least one is (definition sheet truth.seed_truthful)."""
    ok, A = seed_credit(D, ga, gb, gc)
    v = np.where(ok, A.astype(np.int64), np.iinfo(np.int64).min).max(axis=1)
    real = ok.any(axis=1)
    # credited_tps lists a TP once per hit, so ok counts the distinct TPs
    return np.where(ok.sum(axis=1) == 1, v, -1), real


def recovered_keys(D, ga, gb, gc=None):
    """Sorted keys of the TPs a seed recovers: every TP credited by all of its
    hits (seed_credit), each such TP once."""
    ok, A = seed_credit(D, ga, gb, gc)
    if not ok.any():
        return np.empty(0, dtype=np.int64)
    r, s_ = np.nonzero(ok)
    return np.unique(tp_key(D["event"][ga][r], A[r, s_]))


def cand_purity(D, ga, gb, gc):
    """(n_candidates, n_seed_truthful) for one batch of candidate triples."""
    real = seed_tp(D, ga, gb, gc)[1]
    return len(real), int(real.sum())


# THE COMBINED IT+OT HIT CODE SPACE of these studies: 1-4 = IL1-IL4 and
# 11-16 = OL1-OL6 (OT barrel stub layer + 10). The nano's cluster `surface`
# ALSO uses 11-14, for the discs ID1-ID4, so an IT disc cluster that reached a
# combined array would silently be read as an OT barrel stub. Disc clusters are
# kept out of the census and the combinatorics altogether until a disc-seeding
# extension gives them their own codes; it_barrel_codes is the guard every
# combined array's IT half goes through.
def it_barrel_codes(layer, where):
    """`layer` unchanged, after asserting every IT code is a barrel layer 1-4."""
    layer = np.asarray(layer)
    if layer.size and not ((layer >= 1) & (layer <= 4)).all():
        bad = np.unique(layer[(layer < 1) | (layer > 4)])
        raise AssertionError(
            f"{where}: IT cluster codes {bad.tolist()} outside 1-4 would enter the "
            f"combined IT+OT code space, where 11-16 = OL1-OL6. Disc clusters "
            f"(surface 11-14) are not supported here; read the table with "
            f"table_columns, which keeps TBPX only.")
    return layer


# The IT studies here are TBPX-only (IT_CONFIGS are layer triples), so the
# cluster table is read barrel-only: on a unified nano the disc rows that follow
# the barrel would otherwise enter every per-layer count with layer = 0.
def table_branches(path, table, cols):
    keys = uproot.open(f"{path}:Events").keys()
    if table == IT_TABLE:
        return SPN.cluster_branches(keys, list(cols) + ["surface"], surfaces=SPN.BARREL)
    return SPN.branch_names(keys, table, cols)


def table_columns(A, table, cols):
    if table == IT_TABLE:
        J = SPN.cluster_columns(A, list(cols) + ["surface"], surfaces=SPN.BARREL)
        it_barrel_codes(ak.to_numpy(ak.flatten(J["surface"])), "table_columns")
        return {c: J[c] for c in cols}
    return SPN.table_columns(A, table, cols)


def load_flat(spec, table, cols, nev=None, tp=()):
    """Flatten `cols` of `table` across every input file into one dict.

    Stops once nev events have been read. Streams rather than concatenating
    whole files, so the footprint stays bounded on the full ttbar set.
    `tp` names TP_TRUTH columns to join from L1TTP, read from the same entries
    as the objects so both sides share one event numbering.
    """
    paths = expand_inputs(spec)
    tp = list(tp)
    if tp:
        if "tpIdx" not in cols:
            raise ValueError("joining TP truth needs tpIdx among cols")
        require_tp_table(paths)
    keys = table_branches(paths[0], table, cols)
    parts, counts, seen = [], [], 0
    for A in uproot.iterate([f"{p}:Events" for p in paths],
                            keys + (tp_branches(paths[0]) if tp else []), step_size=200):
        if nev is not None and seen + len(A) > nev:
            keep = nev - seen
            if keep <= 0:
                break
            A = A[:keep]
        J = table_columns(A, table, cols)
        n = ak.to_numpy(ak.num(J[cols[0]]))
        ev = np.arange(seen, seen + len(n))
        P = {c: ak.to_numpy(ak.flatten(J[c])) for c in cols}
        P["event"] = np.repeat(ev, n)
        if tp:
            attach_tp_truth(P, tp_table(A, ev), tp)
        parts.append(P)
        counts.append(n)
        seen += len(n)
        if nev is not None and seen >= nev:
            break
    if not parts:
        raise SystemExit("inputs contained no events")
    D = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
    nall = np.concatenate(counts)
    if table == IT_TABLE and all(c in D for c in SPN.ANGLE_FLAGS.values()):
        log_angle_cap(angle_cap_census(D), "load_flat")
    return D, len(nall), nall


IT_COLS = ["layer", "globalR", "globalZ", "globalPhi", "globalClusterPhi",
           "globalClusterCotTheta", "sigGlobalClusterPhi", "sigGlobalClusterCotTheta",
           # sigX is the r-phi CPE sigma; the KF emulation needs it for the
           # position variance and sigY alone (the z sigma) is not a substitute.
           "sigX", "sigY", "tpIdx", "tpPt",
           # the angle cap: globalClusterPhi/CotTheta are built from BOTH module-frame
           # angles whatever the flags say, so it_prepare gates them with
           # SPN.direction_usable (flags are capped by the loader)
           "hasAlpha", "hasBeta", "localCotAlpha", "localCotBeta"]


def it_chunks(path, nev, step=16):
    """Stream the cluster table in chunks, yielding one flat dict per chunk,
    with every TP_TRUTH column joined from L1TTP.

    THE BATCH SIZE IS THE MEMORY CONTROL. The stages are vectorised across every
    THE CHUNK SIZE IS NOT THE MEMORY CONTROL -- see --pair-budget-gb. An earlier
    revision claimed here that peak RSS was "~0.2 GB/event, so 16 ~ 3 GB". That
    was wrong by ~30x: measured, the displaced configuration reached 12.4 GB on
    TWO events, because the blow-up is in the candidate triplets of a SINGLE
    event and no chunk size can reach below one event. The chunk size only trades
    I/O calls against the per-cluster arrays, which are small.
    """
    seen = 0
    paths = expand_inputs(path)
    require_tp_table(paths)
    srcs = [f"{p}:Events" for p in paths]
    for A in uproot.iterate(srcs, table_branches(paths[0], IT_TABLE, IT_COLS) + tp_branches(paths[0]),
                            step_size=step):
        # ENFORCE THE EVENT LIMIT HERE. uproot.iterate ignores entry_stop when it
        # is handed a LIST of files, so -n 8 across ten ttbar files silently
        # processed all 1000 -- the kind of miss that turns a quick check into a
        # long run and makes two results incomparable.
        if nev is not None:
            if seen >= nev:
                return
            if seen + len(A) > nev:
                A = A[:nev - seen]
        J = table_columns(A, IT_TABLE, IT_COLS)
        n = ak.to_numpy(ak.num(J[IT_COLS[0]]))
        ev = np.arange(seen, seen + len(n))
        D = {c: ak.to_numpy(ak.flatten(J[c])) for c in IT_COLS}
        D["event"] = np.repeat(ev, n)
        D["_events"] = ev
        attach_tp_truth(D, tp_table(A, ev))
        seen += len(n)
        yield D
        del A, D


def load_ot(path, nev, tp=()):
    """The OT stub table; `tp` as for load_flat. It carries the stub's principal
    tpIdx but not its clusters' owners, so truth computed on it through
    credits_of credits the principal only (the stub credit of census formats
    <= 27): the cost and omnibus studies built on it keep that, while the census
    reads the owners itself (tp_findability.OT_COLS)."""
    paths = expand_inputs(path)
    t = uproot.open(f"{paths[0]}:Events")
    cols = ["layer", "isBarrel", "r", "phi", "z", "bend", "tpIdx", "tpPt",
            "tpGenuine", "tpCombinatoric", "tpUnknown"]
    have = [c for c in cols if SPN.has_column(t.keys(), OT_TABLE, c)]
    D, nev_read, n = load_flat(path, OT_TABLE, have, nev, tp)
    D["eta"] = np.abs(np.arcsinh(D["z"] / np.maximum(D["r"], 1e-6)))
    return D, nev_read, have


# ==========================================================================
# IT stages
# ==========================================================================
# WHY THE DISPLACED TRIPLET IS A DIFFERENT PROBLEM, NOT A HARDER ONE.
# A 3-hit seed with d0 FREE has 5 parameters (kappa, phi0, d0, z0, cotTheta).
# The three r-phi measurements determine (kappa, phi0, d0) EXACTLY -- zero
# degrees of freedom -- so there is NO r-phi chi2 to reject on. The three r-z
# measurements give 2 parameters and 1 DOF. So a displaced triplet's only
# rejection comes from that single r-z constraint plus the per-cluster sensor
# angles, which supply 6 further constraints on an otherwise exactly-determined
# system. For prompt seeding the angles are a convenience worth ~3x; for
# displaced seeding they are the ONLY discrimination there is.
# ---- the exact three-point d0 solve ---------------------------------------
# A cluster PAIR has two azimuths for three unknowns, so it can only get
# curvature by ASSUMING d0 = 0, and a real d0 biases it by
#     kappa_bias = d0 * (1/rA - 1/rB) / (c * dr)
# = 10.34 per cm for IT L1L2. Covering d0 = 500 um therefore needs the curvature
# gate opened by 0.517 against kappa_max = 0.500 at 2 GeV, i.e. a seed labelled
# 2 GeV really admits 0.98 GeV. That coupling is what this removes.
#
# Three azimuths give three unknowns and zero remaining r-phi freedom, so d0 is
# SOLVED. The model is linear in its unknowns to first order in d0/r and kappa*r:
#     phi(r) = phi0 + A/r + B*r ,    A = +d0 ,  B = -c*kappa
# The sign of A is MEASURED (robust slope of A on the TP's own d0 is +1.11), not
# asserted: writing A = -d0 makes the residual -2*d0, which reads as sigma(d0)
# degrading with d0 rather than as a sign flip.
#
# MEASURED, 100-event PU200 ttbar, correct cluster triples, pT > 2 GeV: sigma(d0)
# is 39 um (L1L2L3) / 78 um (L2L3L4) and FLAT in d0, and sigma(kappa) is flat at
# 0.016-0.033 where the pair degrades to 0.52 -- 15.7x/24.9x better at d0 = 1-5 mm.
#
# FIRMWARE: the coefficients depend only on the three radii, and quantising each
# radius to the 48 bins already built for the projection search reproduces the
# per-cluster result exactly (37 um vs 37 um). Layer-median radii do NOT: 2.9x
# worse in d0, 3.4x in kappa, because the +-5 mm ladder stagger matters.
def solve3(r1, p1, r2, p2, r3, p3):
    """phi0, d0, kappa from three clusters. Closed form, vectorised.

    Azimuths are differenced against the innermost cluster first, so the 2x2 that
    remains cannot carry a 2*pi wrap.
    """
    u2, u3 = 1.0 / r2 - 1.0 / r1, 1.0 / r3 - 1.0 / r1
    v2, v3 = r2 - r1, r3 - r1
    d2, d3 = wrap(p2 - p1), wrap(p3 - p1)
    det = u2 * v3 - u3 * v2
    ok = np.abs(det) > 1e-9
    det = np.where(ok, det, 1.0)
    A = (d2 * v3 - d3 * v2) / det
    B = (u2 * d3 - u3 * d2) / det
    # phi(r) = phi0 + A/r + B r fitted; in TTTrack's d0 convention phi(r) =
    # phi0 - d0/r - c kappa r, so d0 = -A and kappa = -B/c
    return wrap(p1 - A / r1 - B * r1), -A, -B / C_BEND, ok


def angle_cap_census(D):
    """SPN.angle_cap_census of an IT_COLS dict, per layer and TP link (the flags
    in D are the loader's CAPPED ones, so n_dropped_by_cap is 0 by construction;
    n_fail is the count the cap removes)."""
    g = SPN.angle_cap_groups(D["layer"], D["tpIdx"])
    return SPN.angle_cap_census(D, g)


def log_angle_cap(census, where):
    """One line per angle: valid, passing, failing |cot| <= cap, all TBPX clusters."""
    for ang, cells in census["angles"].items():
        c = cells["all"]
        f = c["frac_fail"]
        print(f"  angle cap [{where}] {ang:9s} of {c['n_total']:,} clusters: "
              f"valid {c['n_valid']:,}  pass {c['n_pass']:,}  fail {c['n_fail']:,} "
              f"({'-' if f is None else f'{100 * f:.3f}%'} of valid)"
              + (f"; {c['n_total'] - c['n_pass']:,} (fail or absent) vetoed from "
                 "angle-using seeds" if ang == "direction" else ""))


# ==========================================================================
# WHERE THE SMARTPIXELS ANGLES ARE USED: one switch per stage
# ==========================================================================
# The census exists to measure what the angles add, so every use of them is a
# named stage that can be switched on its own, and the switches travel in the
# census cache key and manifest. With every stage off the census is the
# POSITION-ONLY system: the same position windows and policy, no angle anywhere
# (the "do not handicap the baseline" rule, docs/ot-emulation-fidelity.md).
#
# What each stage compares, in one place (pull = residual / per-cluster sigma; in
# the census the sigma is the angle table's width x the published sigma, the
# angle centre-corrected: census_angles; legacy callers use it_prepare's
# pull-scaled sigmas):
#   alpha: the cluster's kappa_alpha = sin(phi_pos - phi_dir) / (c r) against
#          the predicted kappa - d0 / (c r^2) (TTTrack d0; d0 = 0 for a pair)
#   beta:  the cluster's cot(theta) against the predicted cot(theta); at the
#          pair stage, equivalently, the two clusters' z0 = z - r cot(theta)
# A cluster without a usable direction (dir_ok False: an angle absent or past the
# |cot| <= 12 cap) or an OT stub is a WILDCARD for every comparison: it passes
# every angle gate and contributes no angle pull.
ANGLE_STAGES = {
    "veto": ("S1", "cluster preselection: alpha against the pT floor, and the "
                   "angle cap (dir_ok)"),
    "pair_alpha": ("S2a", "pair formation: each cluster's alpha against the "
                          "pair curvature"),
    "pair_z0": ("S2z", "pair formation: the two clusters' beta-derived z0 agree"),
    "pair_joint": ("S2j", "pair formation: the joint (z0, phi) pairing pre-gate "
                          "on the beta-derived z0"),
    "attach": ("S3", "triplet third-hit attach: its alpha/beta against the "
                     "pair's curvature and cot"),
    "proj_window": ("S4w", "projection to a follow layer: an alpha/beta pull "
                           "window on each candidate"),
    "proj_arb": ("S4a", "hit arbitration (third hit and follow layers): "
                        "position + angle pulls instead of smallest dphi"),
    "kf": ("S5", "KF update with alpha and beta"),
    "accept": ("S6", "track acceptance: angle chi2 at the final state"),
    "dr": ("S7", "duplicate-removal ranking: angle chi2 weight"),
    "mva": ("S8", "quality-MVA angle features"),
}
# THE DEFAULT: every stage but S5 (physicist, 2026-10-15). In the fit the angles "add far too
# little" (measured on CAL: switching S5 moves the final state by ~1e-6 and the angle chi2 by a
# median relative 4e-5), "while they have good value in MVAs afterwards as signs of
# contamination/compatibility". S6, the final angle check against the position-only track, CUTS:
# its thresholds remove 36-54 % of fakes for ~1 % of clean tracks (on OFF34), "worth the 1% loss of
# good tracks" because it "reduces the total tracks that must be sent, which is a serious limitation
# in the hardware" (the physicist, reversing for S6 the same day's feature-only default). Its
# statistic is exported per track either way (n_angle_meas, chi2_angle_per_meas). Both stay
# switches ('all' or naming kf turns S5 on; --angle-stages-off accept makes S6 a feature only).
DEFAULT_STAGES = frozenset(ANGLE_STAGES) - {"kf"}
# what use_angles=True meant before the stages existed: the seeding gates only
LEGACY_ANGLE_STAGES = ("veto", "pair_alpha", "pair_z0", "pair_joint")
# THE STAGES THAT READ BOTH ANGLES, each switchable per angle (WP-D0b): what
# alpha and beta each add has to be measurable at every stage that uses both.
# S1 and S2a read alpha only, S2z and S2j beta only; S7 and S8 read the S6
# statistic's pulls of both angles.
PER_ANGLE_STAGES = ("attach", "proj_window", "proj_arb", "kf", "accept")
ANGLES = ("alpha", "beta")
# THE WEIGHTING STAGES' TABLE KEY (WP-D0b): "emitted", the cluster's prepared
# value (the table row of its own emitted cot theta), or "prediction", the
# true-key row at the predicted cot theta of the hypothesis being tested
KEYED_STAGES = ("proj_arb", "kf")
TABLE_KEYS = ("emitted", "prediction")
DEFAULT_KEYS = {"proj_arb": "prediction", "kf": "emitted"}


# THE NaN GUARD (census.angle_nonfinite): the stages whose angle statistic is counted when it is not
# finite among the IT inputs with a usable direction. A NaN statistic fails every threshold, so it
# vetoes silently (S1 vetoed every OT stub that way until 2026-10-15).
NONFINITE_STAGES = ("S1", "S2a", "S2z", "S3", "S4w", "S4a", "S6")


class AngleUse:
    """Which stages use the SmartPixels angles, with which angles, and how.

    on           the stages that use the angles (ANGLE_STAGES)
    angles       per PER_ANGLE_STAGES stage, the angles it reads (default both);
                 a stage left with no angle behaves as off
    keys         per KEYED_STAGES stage, the table key (TABLE_KEYS)
    arb_weight   S4a: weight of the angle pulls^2 against the position pull^2
    dr_weight    S7: (doublet, triplet) weight of the angle chi2 per cluster in
                 the duplicate-removal rank
    model        the census angle model (census_angles.Model), cuts its cut file
                 (census_angles.CutSet), cuts_open the calibration run (every
                 cutting stage records and passes everything), rec the chunk's
                 recorder: set by the census. Legacy callers have no model and
                 may use only the seeding stages, with the 3-sigma cuts on
                 it_prepare's pull-scaled sigmas.
    """
    PARAMS = {"arb_weight": 1.0, "dr_weight": (0.30, 0.03)}

    def __init__(self, on=(), angles=None, keys=None, **params):
        on = frozenset(on)
        bad = on - set(ANGLE_STAGES)
        if bad:
            raise ValueError(f"unknown angle stages {sorted(bad)}")
        badp = set(params) - set(self.PARAMS)
        if badp:
            raise ValueError(f"unknown angle parameters {sorted(badp)}")
        self.on = on
        self.angles = {s: frozenset(ANGLES) for s in PER_ANGLE_STAGES}
        for s, a in (angles or {}).items():
            if s not in PER_ANGLE_STAGES or not set(a) <= set(ANGLES):
                raise ValueError(f"angle set {s}={sorted(a)}: stage must be one of "
                                 f"{PER_ANGLE_STAGES}, angles of {ANGLES}")
            self.angles[s] = frozenset(a)
        self.keys = dict(DEFAULT_KEYS)
        for s, k in (keys or {}).items():
            if s not in KEYED_STAGES or k not in TABLE_KEYS:
                raise ValueError(f"table key {s}={k}: stage one of {KEYED_STAGES}, "
                                 f"key one of {TABLE_KEYS}")
            self.keys[s] = k
        p = dict(self.PARAMS, **params)
        self.arb_weight = float(p["arb_weight"])
        self.dr_weight = tuple(float(x) for x in p["dr_weight"])
        self.model = self.cuts = self.rec = None
        self.cuts_open = False
        self.nonfinite = None       # per chunk {stage: count}, set by the census (NaN guard)

    def __call__(self, stage):
        """Whether `stage` uses any angle (a per-angle stage with no angle left is off)."""
        if stage not in ANGLE_STAGES:
            raise KeyError(stage)
        return stage in self.on and (stage not in self.angles or bool(self.angles[stage]))

    def uses(self, stage, angle):
        """Whether `stage` uses `angle`."""
        if angle not in ANGLES:
            raise KeyError(angle)
        if stage in ("veto", "pair_alpha"):
            return angle == "alpha" and stage in self.on
        if stage in ("pair_z0", "pair_joint"):
            return angle == "beta" and stage in self.on
        return self(stage) and (stage not in self.angles or angle in self.angles[stage])

    def subset(self, stage):
        """The census_angles.SUBSETS name of a per-angle stage's angle set ("ab", "a", "b")."""
        a = self.angles[stage]
        return "ab" if len(a) == 2 else ("a" if "alpha" in a else "b")

    @classmethod
    def coerce(cls, x):
        """An AngleUse as is; a bool is the pre-stage use_angles shorthand: True
        = the seeding gates (LEGACY_ANGLE_STAGES), False = no angle at all."""
        if isinstance(x, cls):
            return x
        return cls(LEGACY_ANGLE_STAGES if x else ())

    @classmethod
    def parse(cls, on="default", off="", sets="", keys="", **params):
        """From CLI lists: stage names or their S-labels, 'default' (DEFAULT_STAGES),
        'all' or 'none'; per-angle
        sets as 'S3=alpha,S5=beta,S6=' (stage=angles separated by '+', empty =
        none); keys as 'S4a=emitted,S5=prediction'."""
        alias = {v[0].lower(): k for k, v in ANGLE_STAGES.items()}

        def stage(t):
            t = t.strip().lower()
            if t in ANGLE_STAGES:
                return t
            if t in alias:
                return alias[t]
            raise SystemExit(f"unknown angle stage {t!r}; known: "
                             + ", ".join(f"{v[0]}={k}" for k, v in ANGLE_STAGES.items()))

        def names(spec):
            out = set()
            for t in (x.strip().lower() for x in spec.split(",")):
                if not t or t == "none":
                    continue
                if t == "all":
                    out |= set(ANGLE_STAGES)
                elif t == "default":
                    out |= DEFAULT_STAGES
                elif t in ("s2", "s4"):        # a whole family
                    out |= {k for k, v in ANGLE_STAGES.items()
                            if v[0].lower().startswith(t)}
                else:
                    out.add(stage(t))
            return out

        def pairs(spec):
            out = {}
            for t in (x for x in spec.split(",") if x.strip()):
                if "=" not in t:
                    raise SystemExit(f"{t!r}: give stage=value")
                k, v = t.split("=", 1)
                out[stage(k)] = v.strip()
            return out
        angle_sets = {s: [a for a in v.split("+") if a] for s, v in pairs(sets).items()}
        return cls(names(on) - names(off), angles=angle_sets, keys=pairs(keys), **params)

    def count_nonfinite(self, stage, values):
        """The NaN guard: add the non-finite entries of a stage's statistic (IT inputs with a
        usable direction only) to this chunk's count; a no-op outside the census."""
        if self.nonfinite is not None:
            v = np.asarray(values, np.float64)
            self.nonfinite[stage] = self.nonfinite.get(stage, 0) + int(v.size - np.isfinite(v).sum())

    def key(self):
        """The identity of this configuration, for cache keys and manifests."""
        return {"stages": sorted(self.on),
                "angles": {s: sorted(a) for s, a in self.angles.items()},
                "keys": dict(self.keys), "arb_weight": self.arb_weight,
                "dr_weight": list(self.dr_weight)}

    @classmethod
    def from_key(cls, k):
        return cls(k["stages"], angles=k.get("angles"), keys=k.get("keys"),
                   **{p: k[p] for p in cls.PARAMS})

    def label(self):
        parts = []
        for s in ANGLE_STAGES:
            if not self(s):
                continue
            lab = ANGLE_STAGES[s][0]
            if s in self.angles and len(self.angles[s]) == 1:
                lab += "(" + next(iter(self.angles[s]))[0] + ")"
            parts.append(lab)
        return ",".join(parts) or "none"


def it_prepare(D, bench):
    """Per-cluster kappa_alpha, z0, cot(theta) and their sigmas for the LEGACY
    callers (paused or pre-D10 studies): the published sigmas times the per-layer
    pull scales (ALPHA_PULL_SCALE / BETA_PULL_SCALE), no table, no centre, no
    knob, then quantized (bench). The census prepares its angles with
    census_angles.prepare (the shared angle module and the locked table) instead.

    bench = (alpha_bits, beta_bits): alpha is the sin(half) word over
    ALPHA_RANGE, beta the per-cluster z0 word over Z0_RANGE; cot_b is derived
    FROM the quantized z0, so the bit width reaches every beta use, not only the
    pair z0 gate.

    dir_ok: SPN.direction_usable. The angle-derived quantities of a cluster
    without it are garbage (built from a capped or absent angle); alpha_veto
    removes such a cluster from every angle-using seed and projection target,
    and every other stage treats it as a wildcard."""
    r = D["globalR"]
    lay = np.clip(np.asarray(D["layer"]).astype(np.int64), 0, 4)
    # in the sigma columns' own precision, so that scales of 1 reproduce the
    # unscaled arithmetic bit for bit
    dt = np.asarray(D["sigGlobalClusterPhi"]).dtype
    sa = np.array([ALPHA_PULL_SCALE.get(L, 1.0) for L in range(5)], dt)[lay]
    sb = np.array([BETA_PULL_SCALE.get(L, 1.0) for L in range(5)], dt)[lay]
    half = wrap(D["globalPhi"] - D["globalClusterPhi"])
    cot = D["globalClusterCotTheta"]
    sig_half = D["sigGlobalClusterPhi"] * sa
    s_cot = D["sigGlobalClusterCotTheta"] * sb
    cot_a = np.sin(half) / (C_BEND * r)          # kappa implied by alpha
    s_kap = sig_half * np.abs(np.cos(half)) / (C_BEND * r)
    z0 = D["globalZ"] - r * cot
    s_z0 = r * s_cot
    ovf_a = np.zeros(len(z0), bool)
    ovf_z = np.zeros(len(z0), bool)
    if bench is not None:
        ab, bb = bench
        # alpha is encoded as sin(half); convert the range through c*r
        q, ovf_a, lsb_a = quantize(np.sin(half), ab, ALPHA_RANGE)
        cot_a = q / (C_BEND * r)
        s_kap = np.hypot(s_kap, quant_sigma(ab, ALPHA_RANGE) / (C_BEND * r))
        z0, ovf_z, lsb_z = quantize(z0, bb, Z0_RANGE)
        s_z0 = np.hypot(s_z0, quant_sigma(bb, Z0_RANGE))
        cot = (D["globalZ"] - z0) / r
        s_cot = s_z0 / r
    return {"kap_a": cot_a, "s_kap": s_kap, "z0": z0, "s_z0": s_z0,
            "cot_b": cot, "s_cot": s_cot,
            "ovf_a": ovf_a, "ovf_z": ovf_z, "dir_ok": SPN.direction_usable(D)}


# keys of an it_prepare result that are per-cluster arrays, with the value an OT
# stub (no angle at all) carries: huge sigmas make every angle gate pass and give
# a zero pull, and the joint (z0, phi) pairing treats the stub as a phi-only
# wildcard
OT_STUB_ANGLES = {"kap_a": 0.0, "s_kap": 1e9, "z0": 0.0, "s_z0": 1e9,
                  "cot_b": 0.0, "s_cot": 1e9, "ovf_a": False, "ovf_z": False,
                  "dir_ok": True}


def q_with_ot_stubs(QI, n):
    """An it_prepare result extended by n trailing OT stubs (OT_STUB_ANGLES), for
    the combined IT + OT hit arrays."""
    return {k: np.r_[QI[k], np.full(n, v)] for k, v in OT_STUB_ANGLES.items()}


# ---- projecting to the third layer: SEARCH ON z, CONFIRM ON phi -----------
# THE SEARCH ORDER IS CONDITIONAL, and the condition changed under it. z-first
# was adopted because the phi window had to carry the d0 allowance: at d0 =
# 500 um the phi half-window is 68.9 mrad against a z road of 391 um, so z-first
# won by 11x on IT L1L2 -> L3. That remains true for any d0-TOLERANT seed, which
# is every displaced and triplet seed here.
#
# But beamline-constrained pair seeding then set d0 = 0 for the pair seeds, which
# collapses their phi window 9x to 16.3 mrad and takes most of z-first's margin
# with it -- the advantage drops from 11x to 2.6x. That 2.6x is INSIDE this
# file's own systematic tilt: sigma(cot) is taken from cluster positions alone
# (5.7e-4) where the MEASURED value including scattering is 1.7e-3, optimistic in
# z by 2.0x; and sigma(kappa) assumes 1 mrad per cluster (0.080) where the
# MEASURED value is 0.031, pessimistic in phi by 2.2x. Both biases favour
# z-first, for a combined 4.3x tilt.
#
# So for the d0 = 0 pair seeds the order is UNDETERMINED until both windows are
# built from measured resolutions. With measured values phi-first wins 1.6x on
# this stage. The z-first implementation is not wrong -- --selftest shows it
# accepts an identical set -- it may simply no longer be the cheaper order for
# those seeds. Do not read the 28x headline as unconditional.

# MEASURED (projection_residuals.py, PU200 ttbar, 100 events, pT > 2 GeV, correct
# cluster triples only): between the prompt core and d0 = 1-5 mm, sigma(dphi) at
# layer C widens 40.4x (0.671 -> 27.1 mrad) while sigma(dz) is FLAT within 10%
# (77.8 -> 68.8 um). L2L3->L4 gives 24.2x on phi and flat on z.
# An earlier 40-event pass read sigma(dz) as slightly NARROWING at large d0
# (0.7x); that was 17 entries of noise. It is flat. A
# pair-derived curvature assumes d0 = 0, so phi carries the full d0 bias; the z
# prediction z0 + r*cot(theta) is untouched by a TRANSVERSE impact parameter.
#
# Searching phi therefore pays for heavy-flavour coverage twice: the window must
# open to ~58 mrad to keep d0 ~ 1 mm tracks, and every prompt seed pays it too.
# Searching z reaches the same tracks in a 464 um window that never widens.
#
# THE INTRA-LAYER RADIAL SPREAD IS WHAT MAKES THIS NON-TRIVIAL, and an earlier
# estimate here ignored it. A layer is not a thin shell: MEASURED ptp is ~1.1 cm
# (staggered/tilted ladders, and the outer shell alone is ~5.7 mm thick and
# continuous, so splitting into two shells does not help -- equal-count binning
# saturates near 4.4 mm even at 16 bins). Searching z at one reference radius
# needs padding (spread/2)*|cot|, which at |cot| = 1.16 is ~5000 um against a
# 232 um resolution window: that is WORSE than the phi search it replaces.
#
# So bin the layer in radius finely enough that geometry stops dominating:
# N_RBINS = 48 over ~1.1 cm gives 229 um bins, whose padding (bin/2)*|cot| is
# ~133 um at the median |cot| -- under the resolution window, so the search is
# resolution-limited as intended.
Z_STRIDE = 200.0        # z spans ~40 cm; windows are mm-scale
N_RBINS = 48


def build_layer_z_index(cidx, event, r, z, sig, nrbins=N_RBINS):
    """Per radial bin of one layer: cluster indices sorted by (event, z).

    Returned per bin: the sorted composite key, the GLOBAL cluster indices in
    that order, the bin's mid radius and half thickness, and the worst sigma in
    the bin (so a search window can bound the per-candidate sigma without
    gathering). The indices must be global: an earlier revision stored positions
    within the layer-C subset, which the caller then used to index D directly --
    the selftest caught it as "missed 947 of 947".
    """
    if not len(r):
        return []
    edges = np.linspace(r.min(), r.max() + 1e-9, nrbins + 1)
    b = np.clip(np.searchsorted(edges, r, "right") - 1, 0, nrbins - 1)
    out = []
    for ib in range(nrbins):
        sel = np.flatnonzero(b == ib)
        if not len(sel):
            continue
        k = event[sel].astype(np.float64) * Z_STRIDE + z[sel]
        o = np.argsort(k, kind="stable")
        out.append((k[o], cidx[sel[o]], 0.5 * (edges[ib] + edges[ib + 1]),
                    0.5 * (edges[ib + 1] - edges[ib]), float(sig[sel].max())))
    return out


def _slices(cnt, max_slice):
    """Contiguous A-ranges whose candidate counts sum to at most max_slice."""
    edges = np.r_[0, np.cumsum(cnt)]
    a0 = 0
    while a0 < len(cnt):
        a1 = int(np.searchsorted(edges, edges[a0] + max_slice, "right")) - 1
        yield a0, max(a1, a0 + 1)
        a0 = max(a1, a0 + 1)


def candidates_by_z(ev_p, z0p, cot, sct, ms, zidx, ceiling=DEFAULT_TRIPLET_CEILING):
    """Yield (pair index, layer-C cluster index) for z-consistent candidates.

    One interval search per radial bin. A given cluster pair therefore appears in
    several yields -- once per bin that has a candidate -- so the caller must
    arbitrate AFTER collecting every bin, not per yield.
    """
    total = 0
    for kz, cidx, rmid, rhalf, sgmax in zidx:
        zpred = z0p + rmid * cot
        half = (NSIG * (np.hypot(sgmax, rmid * sct) + rmid * ms)
                + rhalf * np.abs(cot))
        ka = ev_p.astype(np.float64) * Z_STRIDE + zpred
        lo = np.searchsorted(kz, ka - half, "left")
        hi = np.searchsorted(kz, ka + half, "right")
        cnt = (hi - lo).astype(np.int64)
        total += int(cnt.sum())
        if total > ceiling:
            raise TooWide(total, ceiling)
        if not cnt.sum():
            continue
        for a0, a1 in _slices(cnt, _MAX_SLICE):
            ja, jb = _expand(cidx, lo[a0:a1], cnt[a0:a1], a0)
            yield ja, jb


_EXHAUSTIVE = False


def candidates_exhaustive(ev_p, ev_c, cidx, ceiling=DEFAULT_TRIPLET_CEILING):
    """EVERY layer-C cluster in the pair's own event. The --selftest oracle.

    Deliberately does no geometric reasoning, so comparing against it isolates
    exactly one thing: whether the z-binned search drops a candidate the exact
    test would have kept. Only tractable on a couple of events.
    """
    o = np.argsort(ev_c, kind="stable")
    evs, cs = ev_c[o], cidx[o]
    lo = np.searchsorted(evs, ev_p, "left")
    cnt = (np.searchsorted(evs, ev_p, "right") - lo).astype(np.int64)
    if int(cnt.sum()) > ceiling:
        raise TooWide(int(cnt.sum()), ceiling)
    for a0, a1 in _slices(cnt, _MAX_SLICE):
        yield _expand(cs, lo[a0:a1], cnt[a0:a1], a0)


# ---- stage 1: pair on (z0, phi) JOINTLY, not phi-then-gate -----------------
# Measured selectivities against the 61.6M possible L1xL2 cluster pairs in one
# PU200 event: phi window alone 0.886%, z0 consistency alone 3.97%, the two
# together 0.035%. The old order formed 116,829 cluster pairs on phi and then
# discarded 89% at the z0 gate, so pairing on both at once forms far fewer
# cluster pairs for an identical accepted set. Same mistake the projection stage
# had, one stage earlier.
#
# THE sigma(z0) TAIL IS WHAT MAKES THIS NON-TRIVIAL. Per-cluster sigma(z0)
# measured: median 0.140 cm, q99 0.580, q99.9 2.56, MAX 21.77 cm. Sizing buckets
# on the median (0.596 cm) but the offset loop on the maximum needs K = 156, i.e.
# 313 buckets scanned per cluster -- slower than the phi search it replaces. So
# clusters with sigma above Z0_SPLIT_Q are WILDCARDS searched on phi alone, on
# BOTH sides of the pair: they carry no usable z0, and pretending otherwise is
# what blows up the loop. Capping sigma instead would be cheaper and would
# silently drop real clusters, so it is not done.
PHI_STRIDE = 24.0       # > 6*pi, the span the +-2pi duplication occupies
Z0_SPLIT_Q = 99.0       # percentile of sigma(z0) above which a cluster is wild


def build_phi_z0_index(z0, phi, s_z0, event, usable=None, nsig=NSIG):
    """Sharp clusters keyed by (event, z0 bucket, phi); wild ones by phi alone.

    Every key duplicates phi at +-2pi so a wrap-around window is a plain
    interval search. PHI_STRIDE exceeds the 6*pi span those copies occupy, so one
    bucket's band can never reach its neighbour's. `usable` False (no usable
    direction, so no z0) makes a cluster wild whatever its sigma says. `nsig` is
    the largest z0 reach multiplier the pairing will use (the census's per-bin S2z
    thresholds; NSIG for legacy callers): it sets the bucket-offset span.
    """
    if not len(z0):
        return None
    # PER LAYER, because this function is handed one layer's clusters. sigma(z0)
    # is r * sigma(cot theta) * inflate and sigma(cot theta) is FLAT across
    # layers (measured median 0.0199/0.0197/0.0216/0.0218 for L1..L4), so the
    # whole per-layer difference in sigma(z0) is LEVER ARM: L4's median is 5.2x
    # L1's purely because z0 extrapolates to r = 0 and L4 sits 5.1x further out.
    # A single global percentile would therefore be a disguised cut on radius --
    # measured 65.8% of its wildcards at L4 against 2.3% at L1 -- and would gut
    # the L2L3L4 configuration's z0 search power for a purely geometric reason.
    us = np.ones(len(z0), bool) if usable is None else np.asarray(usable, bool)
    s_split = float(np.percentile(s_z0[us], Z0_SPLIT_Q)) if us.any() else 0.0
    wild = (s_z0 > s_split) | ~us
    sharp = ~wild
    med = float(np.median(s_z0[sharp])) if sharp.any() else 1.0
    w = max(2.0 * NSIG * np.sqrt(2.0) * med, 0.05)
    z_lo, z_hi = float(z0.min()), float(z0.max())
    nb = max(int(np.ceil((z_hi - z_lo) / w)) + 1, 1)
    kmax_off = int(np.ceil(nsig * np.sqrt(2.0) * s_split / w)) + 1

    def _key(pos, bucket):
        if not len(pos):
            return np.empty(0), np.empty(0, np.int64)
        base = (event[pos].astype(np.float64) * nb + bucket) * PHI_STRIDE
        p = phi[pos] + np.pi
        k = np.concatenate([base + p - 2 * np.pi, base + p, base + p + 2 * np.pi])
        s = np.tile(pos, 3)
        o = np.argsort(k, kind="stable")
        return k[o], s[o]

    ps = np.flatnonzero(sharp)
    bs = np.clip(((z0[ps] - z_lo) / w).astype(np.int64), 0, nb - 1)
    ksharp, ssharp = _key(ps, bs)
    bmax = np.zeros(nb)
    if len(ps):
        np.maximum.at(bmax, bs, s_z0[ps])
    zeros = lambda n: np.zeros(n, np.int64)
    pw = np.flatnonzero(wild)
    kwild, swild = _key(pw, zeros(len(pw)))
    # A-side wildcards need every B cluster, sharp or not, on phi alone
    pall = np.arange(len(z0))
    kall, sall = _key(pall, zeros(len(pall)))
    return dict(ksharp=ksharp, ssharp=ssharp, kwild=kwild, swild=swild,
                kall=kall, sall=sall, w=w, z_lo=z_lo, nb=nb, bmax=bmax,
                kmax_off=kmax_off, s_split=s_split)


def pairs_joint_z0_phi(evA, phiA, z0A, sA, B, half_phi,
                       ceiling=DEFAULT_TRIPLET_CEILING, usableA=None, nsigA=None):
    """Yield (ia, ib) for cluster pairs consistent in BOTH z0 and phi.

    ib indexes the array handed to build_phi_z0_index, matching
    pairs_in_window_slices so the caller is unchanged. `usableA` False makes an
    A cluster a wildcard (searched on phi alone), as for build_phi_z0_index.
    `nsigA`, per A cluster, is the z0 reach in combined sigmas: the census's S2z
    threshold of the cluster's bin (so S2j covers S2z exactly); NSIG when None.
    """
    nsA = NSIG if nsigA is None else np.asarray(nsigA, np.float64)
    nb, w, z_lo = B["nb"], B["w"], B["z_lo"]
    total = 0
    bA = np.clip(((z0A - z_lo) / w).astype(np.int64), 0, nb - 1)
    pa = phiA + np.pi
    # A-SIDE WILDCARDS. A cluster whose own sigma(z0) exceeds the split has a z0
    # reach wider than the offset loop spans, so the bucketed search would
    # under-reach and silently DROP real cluster pairs. Measured on a first
    # revision that only split the B side: 0.2-1.4 lost cluster pairs/event, and
    # only on seeds with layer 1 as the inner layer, where the small radius
    # amplifies z0. They get phi alone, which is what their z0 is worth.
    # EXACT CONDITION, not a percentile. An A cluster is safe only if its own z0
    # reach fits inside the span the bucket loop covers; otherwise the loop
    # under-reaches and silently drops real cluster pairs. An earlier revision
    # tested sA against layer B's q99 sigma, which is a cross-layer comparison
    # between two different lever arms and answers a question nobody asked.
    reach_max = nsA * np.hypot(sA, B["bmax"].max() if B["nb"] else 0.0)
    wildA = reach_max > B["kmax_off"] * B["w"]
    if usableA is not None:
        wildA = wildA | ~np.asarray(usableA, bool)
    sharpA = ~wildA
    for off in range(-B["kmax_off"], B["kmax_off"] + 1):
        bt = bA + off
        ok = sharpA & (bt >= 0) & (bt < nb)
        if not ok.any():
            continue
        bt_c = np.clip(bt, 0, nb - 1)
        # does the pair's z0 window actually reach this bucket's z0 span?
        reach = nsA * np.hypot(sA, B["bmax"][bt_c])
        blo = z_lo + bt_c * w
        ok &= (z0A + reach >= blo) & (z0A - reach <= blo + w)
        if not ok.any():
            continue
        key = (evA.astype(np.float64) * nb + bt_c) * PHI_STRIDE + pa
        lo = np.searchsorted(B["ksharp"], key - half_phi, "left")
        hi = np.searchsorted(B["ksharp"], key + half_phi, "right")
        cnt = np.where(ok, hi - lo, 0).astype(np.int64)
        total += int(cnt.sum())
        if total > ceiling:
            raise TooWide(total, ceiling)
        if not cnt.sum():
            continue
        for a0, a1 in _slices(cnt, _MAX_SLICE):
            yield _expand(B["ssharp"], lo[a0:a1], cnt[a0:a1], a0)
    # two remaining families, disjoint from the above and from each other, so no
    # cluster pair is counted twice: sharp A x wild B, and wild A x every B.
    key = evA.astype(np.float64) * nb * PHI_STRIDE + pa
    for kk, ss, rows in ((B["kwild"], B["swild"], sharpA),
                         (B["kall"], B["sall"], wildA)):
        if not len(kk) or not rows.any():
            continue
        lo = np.searchsorted(kk, key - half_phi, "left")
        hi = np.searchsorted(kk, key + half_phi, "right")
        cnt = np.where(rows, hi - lo, 0).astype(np.int64)
        total += int(cnt.sum())
        if total > ceiling:
            raise TooWide(total, ceiling)
        if not cnt.sum():
            continue
        for a0, a1 in _slices(cnt, _MAX_SLICE):
            yield _expand(ss, lo[a0:a1], cnt[a0:a1], a0)


def alpha_veto(D, Q, idxL, kmax, displaced, ang=None):
    """Single-cluster alpha veto (S1). Independent per layer, so doublet and
    triplet seeds apply exactly the same test to whichever layers they use.

    With the census angle model (ang.model) the statistic is the alpha test
    against the pT floor in table units (census_angles.s1_stat), cut at its
    per-row threshold from the cut file (open in the calibration run); without
    it (legacy callers), |kappa_alpha| <= kmax + NSIG sigma on it_prepare's
    pull-scaled sigmas."""
    if ang is not None and ang.model is not None:
        floor = 1.0 / kmax
        st = ang.model.s1(Q, idxL, floor)
        ang.count_nonfinite("S1", st[(D["layer"][idxL] <= 4) & Q["dir_ok"][idxL]])
        if ang.cuts_open:
            k = np.ones(len(idxL), bool)
        else:
            # S1 TESTS SMARTPIXELS CLUSTERS ONLY. An OT stub has no SmartPixels angle (its
            # components are NaN, so its statistic is NaN, which no threshold passes): it is a
            # wildcard here as at every other stage. Until 2026-10-15 the comparison vetoed every
            # OT stub of a cut run -- every OT-seeded entry and every OT follow hit (runs KG, G).
            k = (D["layer"][idxL] > 4) | (st <= ang.cuts.s1_thr(floor, Q["row"][idxL]))
    else:
        k = np.abs(Q["kap_a"][idxL]) <= kmax + NSIG * Q["s_kap"][idxL]
    k |= Q["ovf_a"][idxL] & displaced      # displaced KEEPS alpha overflow
    # the angle cap: no direction, no angle-using seed (overflow cannot rescue it).
    # Before 2026-09-27 such clusters entered with a direction built from the
    # capped or absent angle.
    k &= Q["dir_ok"][idxL]
    return idxL[k]


def pair_kinematics(D, gA, gB):
    """(kap, cot, z0p, ok) of hit pairs (gA inner, gB outer): the d0 = 0 curvature, cot(theta) and z0 of the
    straight r-z line; ok is False where the radii are too close to define them."""
    drp = D["globalR"][gB] - D["globalR"][gA]
    ok = np.abs(drp) > 0.5
    kap = wrap(D["globalPhi"][gA] - D["globalPhi"][gB]) / np.where(ok, C_BEND * drp, 1e9)
    cot = (D["globalZ"][gB] - D["globalZ"][gA]) / np.where(ok, drp, 1e9)
    z0p = D["globalZ"][gA] - D["globalR"][gA] * cot
    return kap, cot, z0p, ok


def _pair_geometry(D, gA, gB, kmax, kap_slack, kap_min, displaced, la, lb, bend_cut):
    """The position gates of cluster pairs (gA, gB): (kap, cot, z0p, ok, steps), with
    `steps` the pass mask after each gate in order ("kappa", "z0lumi", "bend") for
    the stage counters. Shared by the pair loop and the census's truth enumeration
    of correct pairs (the S2a / S2z records)."""
    kap, cot, z0p, ok = pair_kinematics(D, gA, gB)
    ok &= np.abs(kap) <= kmax + kap_slack
    if kap_min > 0.0:
        # LOWER curvature gate, for a doublet aimed at the band where the
        # TRIPLET fails. Measured: the triplet's misses concentrate at
        # |kappa| in 0.40-0.50, i.e. pT just above threshold, where its own
        # sigma(kappa) ~ 0.022 scatters tracks across its |kappa| <= kappa_max
        # cut. The doublets recover 74.8% of the misses there against
        # 0.34-0.43 elsewhere, so 76% of their entire non-redundant value
        # sits in that one band. This gate is evaluated from the doublet's
        # OWN pair, so it runs in parallel with the triplet and adds no
        # latency -- unlike removing the clusters the triplet consumed,
        # which would serialise the two.
        ok &= np.abs(kap) >= kap_min
    steps = {"kappa": ok.copy()}
    if not displaced:                      # beamspot / luminous-region gate
        ok &= np.abs(z0p) <= Z_LUMI
        steps["z0lumi"] = ok.copy()
    if bend_cut is not None and ot_luts() is not None:
        lut = ot_lut_accept(la, lb,
                            D["globalPhi"][gA], D["globalR"][gA], D["bend"][gA],
                            D["globalPhi"][gB], D["globalR"][gB], D["bend"][gB])
        if lut is not None:
            ok &= lut
            steps["bend"] = ok.copy()
    elif bend_cut is not None:
        # APPROXIMATION, AND A KNOWN ONE. The emulator's live test is a
        # window OVERLAP -- mid +- cut against the whole range of bends the
        # (dphi, r) cell can produce, with (mid, cut) derived at build time
        # from each sensor module's stub window and tilt -- not the
        # point residual below, and useCalcBendCuts = true makes the
        # tabulated benddecode_/bendcut_ in Settings.h dead code. Matching
        # it properly means dumping TP_*_stubpt*cut.tab from a real run and
        # loading the LUTs. This gets the right magnitude and the right
        # per-seed tolerance; it will disagree on borderline stubs.
        # FIRMWARE TOLERANCE, MEASURED CONVERSION. The window is
        # bendcutTE_ in full strips, which is the emulator's own number; the
        # strips-per-kappa that turns the pair's curvature into a predicted
        # bend is calibrated from data. Mixing them this way keeps the cut
        # as tight as the real one without inheriting a geometric
        # approximation the real one does not make.
        for g_, tol in ((gA, bend_cut[0]), (gB, bend_cut[1])):
            ok &= np.abs(D["bend"][g_]
                         - bend_expected(D["layer"][g_] - 10, kap)) <= tol
        steps["bend"] = ok.copy()
    return kap, cot, z0p, ok, steps


def _pair_angle_gates_legacy(Q, ang, gA, gB, kap, displaced):
    """(S2a pass, S2z pass) for legacy callers: NSIG on it_prepare's pull-scaled
    sigmas.

    OVERFLOW SEMANTICS, and they are opposite for the two targets. The code
    ranges are set AT the physics boundary (+-0.17 in cot alpha is the 0.5 GeV
    floor; +-15 cm in z0 is the luminous region), so for a PROMPT seed an
    overflowed word says "inconsistent with the prompt hypothesis" and must
    REJECT. Treating overflow as a gate bypass -- which an earlier revision did --
    lets 40% of clusters through ungated and makes coarser quantization look
    BETTER than none, which is how the bug announced itself. For a DISPLACED seed
    the same overflow carries no information against the hypothesis, so it is
    permissive. A cluster without a usable direction is a wildcard."""
    pass_a = np.ones(len(gA), bool)
    pass_z = np.ones(len(gA), bool)
    if ang("pair_alpha"):
        for g_ in (gA, gB):
            cons = np.abs(kap - Q["kap_a"][g_]) <= NSIG * np.maximum(Q["s_kap"][g_], 1e-9)
            pass_a &= ~Q["dir_ok"][g_] | ((cons | Q["ovf_a"][g_]) if displaced
                                          else (cons & ~Q["ovf_a"][g_]))
    if ang("pair_z0"):
        sz = np.hypot(Q["s_z0"][gA], Q["s_z0"][gB])
        consz = np.abs(Q["z0"][gA] - Q["z0"][gB]) <= NSIG * sz
        anyovf = Q["ovf_z"][gA] | Q["ovf_z"][gB]
        pass_z = ~(Q["dir_ok"][gA] & Q["dir_ok"][gB]) | ((consz | anyovf) if displaced
                                                         else (consz & ~anyovf))
    return pass_a, pass_z


def _pair_angle_gates(D, Q, ang, gA, gB, kap, ok, ptmin, la, lb, displaced, both_it,
                      record=False):
    """(S2a pass, S2z pass) in the census (the angle model): each pair's own statistic
    (census_angles.s2a_stat: the larger of the two IT clusters' alpha pulls
    against the pair curvature; s2z_stat: the z0 difference over the combined
    sigma, IT pairs only), cut at the cut file's threshold of its bin (open in the
    calibration run); with `record`, the pairs the position gates kept (`ok`) are
    recorded before the angle cuts (the census's correct pairs, enumerated by
    truth: _record_correct_pairs). Overflow keeps
    the legacy semantics (prompt: reject; displaced: pass); a cluster without a
    usable direction, or an OT stub, is a wildcard."""
    m = ang.model
    useA = Q["dir_ok"][gA] & (D["layer"][gA] <= 4)
    useB = Q["dir_ok"][gB] & (D["layer"][gB] <= 4)
    n = len(gA)
    st_a = np.full(n, np.nan)
    st_z = np.full(n, np.nan)
    pass_a = np.ones(n, bool)
    pass_z = np.ones(n, bool)
    if ang("pair_alpha") and (la <= 4 or lb <= 4):
        st_a = m.s2a(kap, Q, gA, gB, useA, useB)
        if not record:
            ang.count_nonfinite("S2a", st_a[useA | useB])
        ovf = (useA & Q["ovf_a"][gA]) | (useB & Q["ovf_a"][gB])
        cons = (np.ones(n, bool) if ang.cuts_open
                else st_a <= ang.cuts.s2a_thr(ptmin, la, lb))
        pass_a = (cons | ovf) if displaced else (cons & ~ovf)
    if ang("pair_z0") and both_it:
        both = useA & useB
        st_z = np.where(both, m.s2z(Q, gA, gB), 0.0)
        if not record:
            ang.count_nonfinite("S2z", st_z[both])
        ovf = both & (Q["ovf_z"][gA] | Q["ovf_z"][gB])
        cons = (np.ones(n, bool) if ang.cuts_open
                else st_z <= ang.cuts.s2z_thr(ptmin, la, lb, Q["row"][gA]))
        pass_z = ~both | ((cons | ovf) if displaced else (cons & ~ovf))
    if record and ok.any():
        sel = np.flatnonzero(ok)
        ang.rec.pairs(gA[sel], gB[sel], kap[sel], ptmin, la, lb, st_a[sel], st_z[sel],
                      pass_a[sel], pass_z[sel])
    return pass_a, pass_z


def it_pairs(D, Q, ev_idx, la, lb, ptmin, use_angles, displaced=False,
             d0_cm=0.0, kap_min=0.0, z_inner=None, bend_cut=None, fixed=None):
    """STAGE 1 ONLY: the surviving cluster pairs, over an ENTIRE CHUNK.

    Split out of it_pair_seed so that a DOUBLET seed (which projects to every
    remaining layer) and a TRIPLET seed (which requires a confirmed hit in one
    named layer before projecting to the rest) share one implementation of the
    pair formation, gating and arbitration rather than two that can drift.

    The event is folded into the pair-matching sort key, so cross-event pairs are
    impossible by construction rather than by iteration.

    use_angles: an AngleUse (or the bool shorthand, AngleUse.coerce); this
    stage reads S1 (veto), S2a (pair_alpha), S2z (pair_z0), S2j (pair_joint).

    `fixed` (gA, gB, work) replaces the pair stage by given pairs: the prompt OT
    doublets as the CMSSW track finder forms them (ot_chain.census_pairs; census
    format 35), with the finder's work counters. The layer sets and their radii
    (idx) are built as for any seed; no pair gate runs.

    Returns (out, gA, gB, kap, cot, z0p, idx) where idx carries the
    alpha-vetoed cluster index sets for la and lb.
    """
    ang = AngleUse.coerce(use_angles)
    idx = {L: ev_idx[D["layer"][ev_idx] == L] for L in (la, lb)}
    if z_inner is not None:
        # The VMRouter gate on the seed's inner stub. la is the inner layer
        # because seed layers are built inner-first for every doublet.
        az = np.abs(D["globalZ"][idx[la]])
        idx[la] = idx[la][(az >= z_inner[0]) & (az <= z_inner[1])]
    out = {"in_A": len(idx[la]), "in_B": len(idx[lb])}
    none = (out, None, None, None, None, None, idx)
    if min(len(idx[L]) for L in (la, lb)) < 2:
        return none
    kmax = 1.0 / ptmin
    # THE POSITION WINDOWS ARE BUILT ON THE LAYERS' FULL CLUSTER SETS, before
    # any angle preselection, so every angle configuration -- the no-angle arm
    # included -- runs exactly the same windows. (Until 2026-10-03 the median
    # radii were taken after the alpha veto.)
    rA = np.median(D["globalR"][idx[la]]); rB = np.median(D["globalR"][idx[lb]])
    if ang("veto"):
        for L in (la, lb):
            idx[L] = alpha_veto(D, Q, idx[L], kmax, displaced, ang)
        out["after_alpha_veto"] = sum(len(idx[L]) for L in (la, lb))
    if min(len(idx[L]) for L in (la, lb)) < 2:
        return none
    # the same radii for every projection of these pairs (it_project)
    idx["r_med"] = {la: rA, lb: rB}
    if fixed is not None:
        gA, gB, work = fixed
        kap, cot, z0p, _ = pair_kinematics(D, gA, gB)
        out.update({"pairs_phi": int(work.get("te_pairs", 0)),
                    "pairs_bend_ok": int(work.get("lut_pairs", 0)),
                    "pairs_outside_census": int(work.get("pairs_outside_census", 0)),
                    "tracklets": int(len(gA))})
        return (out, gA, gB, kap, cot, z0p, idx) if len(gA) else none
    dr = abs(rB - rA)
    ms = (THETA_MS_MRAD * 1e-3 / ptmin) * np.sqrt(2.0)
    # d0 allowance: prompt seeds assume d0=0; displaced must open the window by
    # d0/r at the inner layer, which is where it hurts most.
    d0_eff = D0_DISPLACED_CM if displaced else d0_cm
    d0_allow = d0_eff / min(rA, rB)
    # THE CURVATURE ACCEPTANCE, not the phi window, is what gates d0 tolerance.
    # A pair-derived kappa assumes d0 = 0; a real d0 biases it by
    #     kappa_bias = d0 * (1/rA - 1/rB) / (c * dr)
    # = 10.1 per cm for IT L1L2 against 0.21 per cm for OT L1L2, so the IT pair
    # seed is ~48x more d0-sensitive purely from radius. An earlier revision
    # widened only the phi window: the extra pairs entered and were then killed
    # by |kappa| <= kmax, so efficiency stayed flat at 0.707 while pair work grew
    # 4.4x. Widening BOTH is the fix, and it costs the same as lowering the pT
    # floor by the same factor.
    kap_bias_per_cm = abs(1.0 / rA - 1.0 / rB) / (C_BEND * max(dr, 0.1))
    kap_slack = kap_bias_per_cm * d0_eff
    half_w = C_BEND * dr * kmax + ms + 5e-4 + d0_allow
    # ---- the S2a / S2z RECORDS: every correct pair (both hits credited to one
    # TP) that the position gates keep, enumerated by truth, so that what the S2j
    # pre-search never generates is still counted against S2z (S2j's reach covers
    # S2z's exactly, so S2j then S2z is S2z on these pairs)
    model = ang.model
    both_it = la <= 4 and lb <= 4
    if (model is not None and ang.rec is not None and (ang("pair_alpha") or ang("pair_z0"))
            and (la <= 4 or lb <= 4)):
        tA, tB = ang.rec.correct_pairs(idx[la], idx[lb])
        if len(tA):
            inw = np.abs(wrap(D["globalPhi"][tB] - D["globalPhi"][tA])) <= half_w
            tA, tB = tA[inw], tB[inw]
            tk, _, _, tok, _ = _pair_geometry(D, tA, tB, kmax, kap_slack, kap_min,
                                              displaced, la, lb, bend_cut)
            _pair_angle_gates(D, Q, ang, tA, tB, tk, tok, ptmin, la, lb, displaced,
                              both_it, record=True)
    # ---- stage 1: CLUSTER PAIRS, streamed in slices ------------------------
    # Only the SURVIVING cluster pairs are retained; the full expansion of
    # layer-A x layer-B never exists at once.
    keepA, keepB, keepK, keepC, keepZ = [], [], [], [], []
    n_phi = n_kap = n_z0l = n_aok = n_zok = n_bok = 0
    # S2j, THE JOINT (z0, phi) PRE-GATE: pairs are searched only within the
    # beta-derived z0 reach of each other. Its reach covers the S2z test exactly
    # (bucket sigma >= each cluster's; in the census the reach multiplier is the
    # S2z threshold of the A cluster's own bin), so with S2z on it only saves
    # work; with S2z off it is a looser beta gate of its own. Off: phi alone. In
    # the calibration run (cuts open) the search is phi alone: nothing is cut.
    nsigA = None
    if model is not None and ang("pair_joint") and not ang.cuts_open and both_it:
        nsigA = ang.cuts.s2z_thr(ptmin, la, lb, Q["row"][idx[la]])
    use_joint = ang("pair_joint") and not (model is not None and ang.cuts_open)
    Bx = (build_phi_z0_index(Q["z0"][idx[lb]], D["globalPhi"][idx[lb]],
                             Q["s_z0"][idx[lb]], D["event"][idx[lb]],
                             usable=Q["dir_ok"][idx[lb]],
                             nsig=NSIG if nsigA is None else max(float(nsigA.max()), 0.0))
          if use_joint else None)
    if Bx is not None:
        pair_gen = pairs_joint_z0_phi(
            D["event"][idx[la]], D["globalPhi"][idx[la]], Q["z0"][idx[la]],
            Q["s_z0"][idx[la]], Bx, half_w, usableA=Q["dir_ok"][idx[la]],
            nsigA=nsigA)
    else:
        pair_gen = pairs_in_window_slices(
            D["event"][idx[la]], D["globalPhi"][idx[la]],
            D["event"][idx[lb]], D["globalPhi"][idx[lb]],
            np.full(len(idx[la]), half_w))
    for ia, ib in pair_gen:
        rss_check()
        n_phi += len(ia)
        gA, gB = idx[la][ia], idx[lb][ib]
        kap, cot, z0p, ok, steps = _pair_geometry(D, gA, gB, kmax, kap_slack, kap_min,
                                                  displaced, la, lb, bend_cut)
        n_kap += int(steps["kappa"].sum())
        if "z0lumi" in steps:
            n_z0l += int(steps["z0lumi"].sum())
        if "bend" in steps:
            n_bok += int(steps["bend"].sum())
        if ang("pair_alpha") or ang("pair_z0"):
            pass_a, pass_z = (
                _pair_angle_gates(D, Q, ang, gA, gB, kap, ok, ptmin, la, lb,
                                  displaced, both_it) if model is not None
                else _pair_angle_gates_legacy(Q, ang, gA, gB, kap, displaced))
            if ang("pair_alpha"):
                ok &= pass_a
                n_aok += int(ok.sum())
            if ang("pair_z0"):
                ok &= pass_z
                n_zok += int(ok.sum())
        if ok.any():
            keepA.append(gA[ok]); keepB.append(gB[ok])
            keepK.append(kap[ok]); keepC.append(cot[ok]); keepZ.append(z0p[ok])
    out["pairs_phi"] = n_phi
    out["pairs_kappa"] = n_kap
    out["kappa_slack"] = float(kap_slack)
    if not displaced:
        out["pairs_z0lumi"] = n_z0l
    out["pairs_bend_ok"] = n_bok
    if ang("pair_alpha"):
        out["pairs_alpha_ok"] = n_aok
    if ang("pair_z0"):
        out["pairs_z0_ok"] = n_zok
    out["tracklets"] = int(sum(len(x) for x in keepA))
    if not keepA:
        return none
    gA, gB = np.concatenate(keepA), np.concatenate(keepB)
    kap, cot, z0p = (np.concatenate(keepK), np.concatenate(keepC),
                     np.concatenate(keepZ))
    del keepA, keepB, keepK, keepC, keepZ
    return out, gA, gB, kap, cot, z0p, idx



def it_project(D, Q, out, gA, gB, kap, cot, z0p, idx, la, lb, lc, ptmin,
               displaced=False, d0_cm=0.0, d0_max=TRIPLET_D0_MAX_CM,
               use_angles=True, d0_meas=None, seed_layers=None, role="follow",
               pos_arb="dphi", seed_id=None, seed_c=None):
    """STAGE 2: project a set of cluster pairs to ONE target layer.

    Called once by a triplet seed (for its required third layer, role
    "attach") and once per remaining layer by a doublet seed (role "follow"),
    so both pay the same per-target cost and the counters are commensurable.
    `idx` is it_pairs' (it carries the pairs' position radii, "r_med").

    use_angles: an AngleUse (AngleUse.coerce). This stage reads S1 (veto of the
    target layer's clusters), S3 (attach: the alpha/beta pull window on the third
    hit), S4w (follow: the same window on a follow-layer candidate) and S4a (the
    one-hit-per-pair arbitration on position + angle pulls, either role).

    pos_arb, THE POSITION PART OF THE ONE-HIT-PER-PAIR CHOICE (POS_ARB):
    "dphi" the smallest |dphi|; "rphiz" the smallest (dphi/sigma_phi)^2 +
    (dz/sigma_z)^2, each with the sigma its window is built from. S4a adds its
    angle pulls^2 to that same position term.

    S3 / S4w / S4a need the census angle model (ang.model). seed_id is the
    seed's code (seed_code; its layers when None), seed_c the triplet's third
    seed hit per pair (a follow after a triplet): the cut bins and the records'
    definition of a correct candidate (credited to a TP every seed hit is
    credited to) use them.
    """
    if role not in ("attach", "follow"):
        raise ValueError(f"role {role!r}")
    if pos_arb not in POS_ARB:
        raise ValueError(f"pos_arb {pos_arb!r}")
    ang = AngleUse.coerce(use_angles)
    kmax = 1.0 / ptmin
    idx = dict(idx)
    idx[lc] = np.arange(len(D["layer"]))[D["layer"] == lc]
    # the target layer's radius from its FULL cluster set, like the pair radii
    # (it_pairs), so the position windows do not depend on the angle stages
    rCmed = float(np.median(D["globalR"][idx[lc]])) if len(idx[lc]) else 0.0
    if ang("veto"):
        idx[lc] = alpha_veto(D, Q, idx[lc], kmax, displaced, ang)
    out["in_C"] = len(idx[lc])
    if len(idx[lc]) < 1:
        return out
    rA, rB = idx["r_med"][la], idx["r_med"][lb]
    dr = abs(rB - rA)
    ms = (THETA_MS_MRAD * 1e-3 / ptmin) * np.sqrt(2.0)
    d0_eff = D0_DISPLACED_CM if displaced else d0_cm
    d0_allow = d0_eff / min(rA, rB)
    # ---- per-cluster layer-C arrays, gathered ONCE --------------------------
    # NOT D["globalR"][idx[lc]][jb] inside the loop, which the previous revision
    # did: that copies the whole layer-C column and THEN indexes it, rebuilding
    # the copy on every slice, and it replicates each layer-C cluster's radius
    # once per cluster pair that might match it.
    cC = idx[lc]
    rC, phC, zC = D["globalR"][cC], D["globalPhi"][cC], D["globalZ"][cC]
    sgC = np.maximum(D["sigY"][cC], 1e-6)

    # d0_meas is the SOLVED impact parameter, available only once three
    # azimuths exist. A doublet has none and passes None, which is the d0 = 0
    # assumption its window must then pay for; a triplet passes the solve3 value
    # and projects along the real trajectory instead of a prompt approximation.
    dm = np.zeros(len(gA)) if d0_meas is None else d0_meas
    phi0 = wrap(D["globalPhi"][gA] + dm / D["globalR"][gA]
                + C_BEND * D["globalR"][gA] * kap)
    # PROJECT TO EACH CANDIDATE'S OWN RADIUS, not the layer median. Comparing a
    # median-radius projection against a candidate's actual z carries an error
    #     dz = (r_actual - r_median) * cot(theta)
    # which for |cot| ~ 2 and the mm-scale radius spread within a layer is ~1 cm
    # against an ~800 um window. That one approximation rejected 52.7% of
    # findable prompt-core tracks at the z step, while an independent residual
    # measurement showed the window contains 93.4% of correct triples -- i.e. the
    # entire "core efficiency loss" was this bug, not physics.
    # AN OT TARGET GETS THE REAL TABULATED WINDOW, AS A CEILING.
    # Everything below derives a window from resolution + scattering + a d0
    # allowance, which is the right thing for an IT layer read by a pixel CPE
    # but is far wider than the fixed rphimatchcut_/zmatchcut_ the OT actually
    # applies. Left alone it hands us OT track-following the real system does
    # not have, which would overstate what the SmartPixels angles add by
    # comparing them against a straw-man OT. Imposed as min(derived, tabulated)
    # so it can only ever TIGHTEN: if physics already says the window is
    # narrower than the firmware's, physics wins and we do not widen to match.
    ot_cap = (ot_match_window(lc - 10, seed_layers or (la, lb))
              if lc > 10 else None)
    # widen the z road only for an IT/joint seed reaching into the OT
    z_widen = (OT_IT_PROJ_Z_WIDEN
               if (lc > 10 and ot_seed_name(seed_layers or (la, lb)) is None)
               else 1.0)
    if ot_cap is OT_PROJ_FORBIDDEN:
        # Not in this seed's projlayers_. Report the layer as looked-at-and-empty
        # rather than returning early, so the cost counters still show that the
        # real menu declines to pay for it.
        out["match_cand"] = 0
        out["match_cand_z"] = 0
        out["tracks_to_fit"] = 0
        return out
    cap_phi = None if ot_cap is None else ot_cap[0] / max(rCmed, 1e-6)
    cap_z = None if ot_cap is None else ot_cap[1]
    skap = np.sqrt(2.0) * 1e-3 / (C_BEND * max(dr, 0.1))
    sph = np.hypot(5e-4, C_BEND * rCmed * skap) + ms + d0_allow
    # sigma(cot theta) from the pair's two z measurements, using the MEASURED
    # per-cluster CPE sigma (sigY ~ 12.5 um) rather than an assumed 30 um.
    sct = np.hypot(np.maximum(D["sigY"][gA], 1e-6),
                   np.maximum(D["sigY"][gB], 1e-6)) / max(dr, 0.1)
    out["projections"] = int(len(gA))
    # ---- the ANGLE window and arbitration (S3 / S4w / S4a) ------------------
    # Each candidate's alpha and beta against the seed's prediction at its own
    # radius: kappa_alpha = kappa - d0/(c r^2) (d0 = the triplet's solved d0, 0
    # for a pair) and cot(theta) = the pair's. A cluster without a usable
    # direction, or an OT stub, is a wildcard (no window, no pull), so the angle
    # stages act on IT targets only.
    # S3 / S4w CUT, on ONE statistic per consumer (WP-D0b; decision 2): the
    # candidate passes when every angle the stage uses lies inside its threshold
    # for the bin (stage x seed layer set x target x seed-measured pT x floor),
    # the per-angle thresholds being the equivalent form of a 99 % cut on
    # max(F_alpha(|p_alpha|), F_beta(|p_beta|)). The pulls have no prediction
    # term: the quantile contains it (sheet angle.cut_level).
    # S4a WEIGHTS: its pulls add the seed's prediction error once (skap, sct),
    # in the table key of ang.keys["proj_arb"].
    win_stage = "attach" if role == "attach" else "proj_window"
    win_on = ang(win_stage) and lc <= 4
    arb_on = ang("proj_arb") and lc <= 4
    model, rec = ang.model, ang.rec
    if (win_on or arb_on) and model is None:
        raise ValueError("S3 / S4w / S4a need the census angle model (census_angles)")
    role_i = 0 if role == "attach" else 1
    sid = seed_code(seed_layers or (la, lb)) if seed_id is None else seed_id
    sub = ang.subset(win_stage) if win_on else None
    seed_hits = (gA, gB) if seed_c is None else (gA, gB, seed_c)
    rec_arb = rec is not None and arb_on
    n_ang_rej = 0

    # ---- stage 2: CANDIDATE TRIPLETS, found by z, confirmed by phi ---------
    zidx = build_layer_z_index(cC, D["event"][cC], rC, zC, sgC)
    n_cand = 0
    sv_p, sv_c, sv_res = [], [], []
    sv_pos, sv_kp, sv_cp, sv_sct = [], [], [], []
    gen = (candidates_exhaustive(D["event"][gA], D["event"][cC], cC) if _EXHAUSTIVE
           else candidates_by_z(D["event"][gA], z0p, cot, sct, ms, zidx))
    for ja, gC in gen:
        rss_check()
        n_cand += len(ja)
        rc_i = D["globalR"][gC]
        # PROJECT ALONG THE ARC, NOT THE RADIUS. z advances with PATH LENGTH,
        # and for a curved track the path to radius r exceeds r:
        #     s = 2R asin(r/2R) ~ r (1 + (r/2R)^2/6)
        # This is the same deltaS the KF already applies (KFbase.cc:632, and
        # ho_rz in kf_emulation); the projection was still using the straight
        # line, which is a pure bias that grows as (r*kappa)^2 and therefore
        # bites exactly where the soft-track study lives.
        #
        # MEASURED on true IT-pair-to-OT-stub projections, IL1+IL2 at 1-1.5 GeV:
        #     OL3  sigma 0.552 -> 0.254 cm (2.2x), containment 76.7% -> 90.9%
        #     OL6  sigma 7.90  -> 4.50  cm (1.8x), containment 54.2% -> 61.1%
        # and nothing at all above 3 GeV, where (r*kappa)^2 is negligible. Those
        # figures use the TRUE curvature; a doublet seed only has its own pair
        # kappa, so it realises less of the gain than a triplet does.
        s_arc = rc_i * (1.0 + np.square(rc_i * C_BEND * kap[ja]) / 6.0)
        zC_i = z0p[ja] + s_arc * cot[ja]
        szp_i = np.hypot(np.maximum(D["sigY"][gC], 1e-6), rc_i * sct[ja]) + rc_i * ms
        phiC_i = wrap(phi0[ja] - dm[ja] / rc_i - C_BEND * rc_i * kap[ja])   # TTTrack d0
        dphi = np.abs(wrap(D["globalPhi"][gC] - phiC_i))
        zwin = NSIG * szp_i * z_widen
        zwin = zwin if cap_z is None else np.minimum(zwin, cap_z)
        dz = np.abs(D["globalZ"][gC] - zC_i)
        good = dz <= zwin
        if displaced:
            # THE EXACT THREE-POINT SOLVE IS THE DISCRIMINANT HERE, not a phi
            # window. With three azimuths in hand, (phi0, d0, kappa) are
            # determined, so the pT cut can be the REAL |kappa| <= kappa_max with
            # no slack, and d0 is CUT ON as a measurement instead of being
            # absorbed into a widened window. The pair-stage kappa bound above
            # stays, but it is now only a SEARCH bound that keeps the
            # combinatorics finite -- it is no longer the physics cut, which is
            # why the pT threshold survives.
            _, d0_3, kap_3, ok3 = solve3(
                D["globalR"][gA[ja]], D["globalPhi"][gA[ja]],
                D["globalR"][gB[ja]], D["globalPhi"][gB[ja]],
                rc_i, D["globalPhi"][gC])
            good &= ok3 & (np.abs(kap_3) <= kmax) & (np.abs(d0_3) <= d0_max)
        else:
            pwin = NSIG * sph if cap_phi is None else min(NSIG * sph, cap_phi)
            good &= dphi <= pwin
        # THE POSITION TERM of the arbitration, the same for every angle
        # configuration; S4a adds its angle pulls to it below
        pos = (np.square(dphi / sph) + np.square(dz / szp_i)) if pos_arb == "rphiz" \
            else None
        score = dphi if pos is None else pos
        # under "dphi" the term S4a adds to is (dphi / sigma_phi)^2, which ranks
        # exactly as dphi does, so with no usable angle the choice is the
        # position-only one either way
        base = np.square(dphi / sph) if pos is None else pos
        if (win_on or arb_on) and good.any():
            kp = kap[ja] - dm[ja] / (C_BEND * rc_i * rc_i)
            cp = cot[ja]
            u_ = Q["dir_ok"][gC]
            if win_on:
                pa_, pb_ = cand_angle_pulls(Q, gC, kp, cp)
                ang.count_nonfinite("S3" if role_i == 0 else "S4w", np.r_[pa_[good & u_], pb_[good & u_]])
                ptb = model.ptbin(kap[ja])
                if ang.cuts_open:
                    inw = np.ones(len(gC), bool)
                else:
                    qa, qb = ang.cuts.win_thr(sub, role_i, sid, lc, ptmin, ptb)
                    inw = (np.abs(pa_) <= qa) & (np.abs(pb_) <= qb)
                ovf = Q["ovf_a"][gC] | Q["ovf_z"][gC]
                inw = (inw | ovf) if displaced else (inw & ~ovf)
                passed = ~u_ | inw
                if rec is not None:
                    rec.window(role_i, sid, lc, ptmin, ja, gC, seed_hits, kp, cp, pa_, pb_,
                               passed, ptb, skap, sct[ja], good & u_)
                g0 = int(good.sum())
                good &= passed
                n_ang_rej += g0 - int(good.sum())
            if arb_on:
                pa2, pb2 = model.arb_pulls(Q, gC, kp, cp, skap, sct[ja], ang.keys["proj_arb"])
                t = ((pa2 * pa2 if ang.uses("proj_arb", "alpha") else 0.0)
                     + (pb2 * pb2 if ang.uses("proj_arb", "beta") else 0.0))
                ang.count_nonfinite("S4a", np.broadcast_to(t, u_.shape)[good & u_])
                score = base + ang.arb_weight * np.where(u_, t, 0.0)
        if not good.any():
            continue
        sv_p.append(ja[good]); sv_c.append(gC[good]); sv_res.append(score[good])
        if rec_arb:
            sv_pos.append(base[good]); sv_kp.append(kp[good]); sv_cp.append(cp[good])
            sv_sct.append(sct[ja][good])
    if win_on:
        out["angle_window_rejected"] = n_ang_rej
    out["match_cand"] = n_cand
    if not sv_p:
        out["match_cand_z"] = 0
        out["tracks_to_fit"] = 0
        return out
    pa, pc, pr = np.concatenate(sv_p), np.concatenate(sv_c), np.concatenate(sv_res)
    out["match_cand_z"] = int(len(pa))
    if rec_arb:
        # S4a's decisions holding a correct candidate, every candidate of each, for
        # the paired comparison of the two table keys (census_angles.Recorder)
        rec.decisions(lc, pa, pc, seed_hits, np.concatenate(sv_kp), np.concatenate(sv_cp),
                      skap, np.concatenate(sv_sct), np.concatenate(sv_pos))
    # ARBITRATE ONCE, over every radial bin together: a cluster pair's candidates
    # are spread across bins, so a per-yield argmin would pick a different winner.
    # The score is the position term (pos_arb), plus the angle pulls^2 under S4a.
    o = np.lexsort((pr, pa))
    first = np.r_[True, pa[o][1:] != pa[o][:-1]]
    pk = o[first]
    out["tracks_to_fit"] = int(len(pk))
    out["_trip"] = (gA[pa[pk]], gB[pa[pk]], pc[pk])
    # WHICH cluster pair each match belongs to. _trip carries global cluster
    # indices, which cannot be attributed back to a seed once several seeds
    # share a cluster -- a doublet following four target layers has to know
    # which of its pairs each confirmation came from to count layers per track.
    out["_pairidx"] = pa[pk]
    return out


# THE POSITION ARBITRATION when a projection window holds several candidates
# for one cluster pair. "dphi" (smallest |dphi|) ignores the z residual the
# window already measured, so an angle arbitration that uses beta would get
# credit for z information the positions carry: the census default is "rphiz"
# (user, 2026-10-03, after an independent review). Other callers keep "dphi".
POS_ARB = ("dphi", "rphiz")


def cand_angle_pulls(Q, g, kap_pred, cot_pred):
    """(alpha pull, beta pull) of clusters g against a predicted kappa_alpha and
    cot(theta), with the stage sigmas of Q (the census: the table's, centre-
    corrected, census_angles.prepare; legacy: it_prepare's pull-scaled ones),
    without a prediction term. The wildcard mask is Q["dir_ok"][g]; an OT stub's
    huge sigmas make its pulls ~0."""
    pa = (Q["kap_a"][g] - kap_pred) / np.maximum(Q["s_kap"][g], 1e-9)
    pb = (Q["cot_b"][g] - cot_pred) / np.maximum(Q["s_cot"][g], 1e-9)
    return pa, pb


def it_pair_seed(D, Q, ev_idx, la, lb, lc, ptmin, use_angles, displaced=False,
                 d0_cm=0.0, d0_max=TRIPLET_D0_MAX_CM, kap_min=0.0):
    """A TRIPLET seed: pair la+lb, then require a confirmed hit in lc.

    Kept as the composition of the two stages it always was, so --selftest still
    exercises exactly this path. A DOUBLET seed is the same stage 1 followed by
    it_project over every remaining layer instead of one named one.
    """
    out, gA, gB, kap, cot, z0p, idx = it_pairs(
        D, Q, ev_idx, la, lb, ptmin, use_angles, displaced, d0_cm, kap_min)
    if gA is None:
        return out
    return it_project(D, Q, out, gA, gB, kap, cot, z0p, idx, la, lb, lc, ptmin,
                      displaced, d0_cm, d0_max, use_angles, role="attach")


def ot_seed_cost(D, ev_idx, name, ptmin, use_bend, cal, eta_max):
    """One OT seed: pair formation + projections, with the REAL match windows."""
    S = OT_SEEDS[name]
    if S["pair"] is None:
        return None
    si = OT_SEED_ORDER.index(name)
    keep = D["isBarrel"][ev_idx] > 0
    if eta_max is not None:
        keep &= D["eta"][ev_idx] < eta_max
    base = ev_idx[keep]
    la, lb = S["pair"]
    idx = {L: base[D["layer"][base] == L] for L in (la, lb)}
    out = {"in_A": len(idx[la]), "in_B": len(idx[lb]),
           "ntc": S["ntc"], "n_proj_paths": len(S["proj_l"]) + (0 if eta_max else len(S["proj_d"]))}
    if min(len(v) for v in idx.values()) < 2:
        return out
    kmax = 1.0 / ptmin
    if use_bend:
        for L in (la, lb):
            if L in cal:
                s, sg, _ = cal[L]
                idx[L] = idx[L][np.abs(s * D["bend"][idx[L]]) <= kmax + NSIG * sg]
        out["after_bend"] = sum(len(idx[L]) for L in (la, lb))
    if min(len(v) for v in idx.values()) < 2:
        return out
    rA = np.median(D["r"][idx[la]]); rB = np.median(D["r"][idx[lb]])
    dr = abs(rB - rA)
    ms = (THETA_MS_MRAD * 1e-3 / ptmin) * np.sqrt(2.0)
    ia, ib = pairs_in_window(D["event"][idx[la]], D["phi"][idx[la]],
                             D["event"][idx[lb]], D["phi"][idx[lb]],
                             np.full(len(idx[la]), C_BEND * dr * kmax + ms + 5e-4))
    out["pairs_phi"] = int(len(ia))
    if not len(ia):
        return out
    gA, gB = idx[la][ia], idx[lb][ib]
    drp = D["r"][gB] - D["r"][gA]
    ok = np.abs(drp) > 0.5
    kap = wrap(D["phi"][gA] - D["phi"][gB]) / np.where(ok, C_BEND * drp, 1e9)
    cot = (D["z"][gB] - D["z"][gA]) / np.where(ok, drp, 1e9)
    z0 = D["z"][gA] - D["r"][gA] * cot
    ok &= (np.abs(kap) <= kmax) & (np.abs(z0) <= Z_LUMI)
    out["pairs_kappa_z0"] = int(ok.sum())
    if use_bend:
        for g_, L in ((gA, la), (gB, lb)):
            if L in cal:
                s, sg, _ = cal[L]
                ok &= np.abs(kap - s * D["bend"][g_]) <= NSIG * sg
        out["pairs_bend_ok"] = int(ok.sum())
    out["tracklets"] = int(ok.sum())
    if not ok.sum():
        return out
    kap, cot, z0 = kap[ok], cot[ok], z0[ok]
    phi0 = wrap(D["phi"][gA[ok]] + C_BEND * D["r"][gA[ok]] * kap)
    tot_c = tot_cz = tot_fit = 0
    trips = []
    for lp in S["proj_l"]:                       # layer projections only in matched
        tgt = base[D["layer"][base] == lp]
        if len(tgt) < 2:
            continue
        rC = np.median(D["r"][tgt])
        wr = OT_RPHI_CUT[lp - 1][si]; wz = OT_Z_CUT[lp - 1][si]
        if wr <= 0:                              # not a configured projection
            continue
        phiC = wrap(phi0 - C_BEND * rC * kap); zC = z0 + rC * cot
        ja, jb = pairs_in_window(D["event"][gA[ok]], phiC,
                                 D["event"][tgt], D["phi"][tgt],
                                 np.full(len(phiC), wr / rC))
        tot_c += len(ja)
        if len(ja):
            good = np.abs(D["z"][tgt][jb] - zC[ja]) <= wz
            tot_cz += int(good.sum())
            mm = np.zeros(len(phiC), bool); mm[ja[good]] = True
            tot_fit += int(mm.sum())
            # COLLECT THE TRIPLES. `trips` was initialised and then never
            # appended to, so the truth block below could not run and the OT
            # reported cost only while the IT reported efficiency and fake rate.
            # The comment there warns that exactly this asymmetry biases any
            # conclusion drawn from the comparison, and it did: the OT side has
            # been quoted on combinatorics alone.
            if good.any():
                trips.append((gA[ok][ja[good]], gB[ok][ja[good]], tgt[jb[good]]))
    # FINDABLE, the OT analogue of the IT's "cluster on all three layers": a
    # TrackingParticle above ptmin with a stub on BOTH seed layers and on at
    # least one projection layer, inside the same eta restriction as the seed.
    # Without a denominator, unique_true_found below is a bare count and cannot
    # be compared with the IT efficiency at all.
    if "tpIdx" in D and "tpPt" in D:
        b0 = base[SPN.tp_matched(D["tpIdx"][base]) & (D["tpPt"][base] >= ptmin)]
        kk = tp_key(D["event"][b0], D["tpIdx"][b0])
        ll = D["layer"][b0]
        ka = np.unique(kk[ll == la]); kb = np.unique(kk[ll == lb])
        kp = np.unique(kk[np.isin(ll, list(S["proj_l"]))]) if S["proj_l"] else kk
        fkeys = np.intersect1d(np.intersect1d(ka, kb), kp)
        out["n_findable"] = int(len(fkeys))
        # Export the SET, not just the count: an efficiency has to be banded by
        # d0 and pT alongside the found set, and a bare count cannot be.
        out["_findable_keys"] = fkeys
    out["projections"] = int(len(phi0)) * max(len(S["proj_l"]), 1)
    out["match_cand"] = tot_c
    out["match_cand_z"] = tot_cz
    out["tracks_to_fit"] = tot_fit
    # ---- truth, so the OT reports the SAME quality metrics as the IT --------
    # Without this the OT could only be compared on combinatorics while the IT
    # had efficiency and fake rate, and that asymmetry silently biases any
    # conclusion drawn from the pair.
    if "tpIdx" in D and trips:
        na = nt_ = 0
        found_keys = []
        for ga_, gb_, gc_ in trips:
            n_, t_ = cand_purity(D, ga_, gb_, gc_)
            na += n_; nt_ += t_
            found_keys.append(recovered_keys(D, ga_, gb_, gc_))
        # UNIQUE ACROSS ALL PROJECTION LAYERS AT ONCE. Taking the unique count
        # per layer and summing double counts any TP the seed finds on more than
        # one projection layer, which made unique_true_found exceed n_findable.
        uk = np.unique(np.concatenate(found_keys)) if found_keys else np.empty(0, np.int64)
        nu = int(len(uk))
        # EXPORT the recovered keys, not just their count. Which TrackingParticles
        # a seed finds is the only way to ask what one seed loses RELATIVE to
        # another; a per-seed efficiency cannot distinguish two seeds that find
        # the same TPs from two that are complementary. Underscore-prefixed, so
        # acc_add skips it rather than trying to sum arrays.
        out["_found_keys"] = uk
        out["cand_truth_matched"] = nt_
        out["unique_true_found"] = nu
        out["cand_total"] = na
        out["fake_fraction"] = 1.0 - nt_ / max(na, 1)
        if out.get("n_findable"):
            out["efficiency"] = nu / out["n_findable"]
    return out


# ==========================================================================
# Truth bookkeeping -- ARRAY BASED
# ==========================================================================
# An earlier revision did this with Python dicts and sets of (event, TP)
# tuples, rebuilt per benchmark x configuration x pT floor. On 40 events that
# reached 9.7 GB resident and was growing ~1 GB/min: ~1.07 M boxed tuple keys
# plus per-combination set rebuilds. It was killed before finishing.
#
# Everything below is int64 arrays and sorted-array set algebra, so cost is
# O(N log N) in clusters with no Python-level per-track work. The only remaining
# per-event Python loop is pair formation, which is inherently per-event and is
# itself vectorised inside; input is streamed in chunks so nothing scales with
# total event count.
# A matched tpIdx is a row (>= 0) or a TP not written to L1TTP (<= -3, -3 - tpIdx
# its index in the unpruned TP collection); both are packed, into disjoint halves
# of the low word, so every matched TP gets its own key.
TP_KEY_SHIFT = 22
_TP_KEY_PRUNED = 1 << (TP_KEY_SHIFT - 1)


def tp_key(event, tpidx):
    """Pack (event, matched tpIdx) into one int64: the low word is the L1TTP
    row, or _TP_KEY_PRUNED + (-3 - tpIdx) for a TP not in L1TTP. Event is the
    HIGH word so that sorting by key groups by event (and, within an event,
    rows by row number), which the pair stages and tp_rows rely on. Unmatched
    tpIdx (-1, -2) cannot be packed."""
    tpidx = np.asarray(tpidx).astype(np.int64)
    if tpidx.size and ((tpidx >= _TP_KEY_PRUNED) | ((tpidx < 0) & (tpidx > -3))
                       | (tpidx <= -3 - _TP_KEY_PRUNED)).any():
        raise ValueError(f"tp_key packs matched tpIdx only (row < 2^{TP_KEY_SHIFT - 1}, "
                         f"or <= -3 with -3 - tpIdx < 2^{TP_KEY_SHIFT - 1})")
    low = np.where(tpidx >= 0, tpidx, _TP_KEY_PRUNED + (-3 - tpidx))
    return np.asarray(event).astype(np.int64) * (1 << TP_KEY_SHIFT) + low


def tp_key_parts(key):
    """(event, tpIdx) of tp_key values."""
    key = np.asarray(key, dtype=np.int64)
    low = key & ((1 << TP_KEY_SHIFT) - 1)
    return key >> TP_KEY_SHIFT, np.where(low >= _TP_KEY_PRUNED, -3 - (low - _TP_KEY_PRUNED), low)


def build_tp_index(D):
    """One row per (event, TP): sorted key array plus aligned |d0| and pT.

    Sorted, so every later lookup is a searchsorted rather than a hash. |d0| is
    L1TTP's POCA d0 [cm] (D needs attach_tp_truth), NaN for TPs not in L1TTP.
    """
    if "tp_d0" not in D:
        raise SystemExit("build_tp_index needs the L1TTP truth: build D with "
                         "it_chunks or join it with attach_tp_truth")
    m = SPN.tp_matched(D["tpIdx"])
    k = tp_key(D["event"][m], D["tpIdx"][m])
    order = np.argsort(k, kind="stable")
    k = k[order]
    first = np.r_[True, k[1:] != k[:-1]]
    uk = k[first]
    sel = np.flatnonzero(m)[order][first]
    return {"key": uk, "d0": np.abs(D["tp_d0"][sel]), "pt": D["tpPt"][sel]}


def findable_keys(D, la, lb, lc, ptmin):
    """Sorted keys of TPs above ptmin with a cluster on ALL THREE layers.

    Fully vectorised: the event is inside the key, so there is no event loop.
    """
    base = SPN.tp_matched(D["tpIdx"]) & (D["tpPt"] >= ptmin)
    sets = []
    for L in (la, lb, lc):
        m = base & (D["layer"] == L)
        sets.append(np.unique(tp_key(D["event"][m], D["tpIdx"][m])))
    out = sets[0]
    for nxt in sets[1:]:
        out = np.intersect1d(out, nxt, assume_unique=True)
    return out


def eff_by_d0(rec, find, tpi, edges=(0.0, 100e-4, 500e-4, 1e9)):
    """Efficiency split by |d0| band, array based.

    The INCLUSIVE number cannot judge heavy flavour: only ~5% of pT>2 TPs sit
    above 100 um, so it can move by at most ~3.5 points however well the HF path
    works. The band split is the quantity that decides whether b-tagging-relevant
    tracks are seeded.
    """
    pos = np.searchsorted(tpi["key"], find)
    ok = (pos < len(tpi["key"])) & (tpi["key"][np.minimum(pos, len(tpi["key"]) - 1)] == find)
    find, d0 = find[ok], tpi["d0"][pos[ok]]
    got = np.isin(find, rec, assume_unique=True)
    out = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        b = (d0 >= lo) & (d0 < hi)
        if not b.any():
            continue
        lab = (f"d0_{lo*1e4:.0f}to{hi*1e4:.0f}um" if hi < 1e8 else f"d0_gt{lo*1e4:.0f}um")
        out[lab] = {"n_findable": int(b.sum()), "efficiency": float(got[b].mean())}
    return out


def render(R):
    """Print every table this study reports. Kept in the script so the tables
    are reproducible from the JSON instead of being retyped by hand."""
    d0 = R.get("d0_measured", {})
    if d0:
        print("\n=== TP transverse impact parameter, pT>2 (MEASURED) ===")
        print(f"  n={d0['n_tp']}  core sigma={d0['core_sigma_cm']*1e4:.1f} um")
        print(f"  |d0| q50={d0['q50_cm']*1e4:.1f}  q68={d0['q68_cm']*1e4:.1f}  "
              f"q95={d0['q95_cm']*1e4:.1f}  q99.73={d0['q99p73_cm']*1e4:.0f}  "
              f"q99.994={d0['q99p994_cm']*1e4:.0f} um")
        print(f"  within 100um: {d0['frac_within_100um']:.4f}   200um: {d0['frac_within_200um']:.4f}")
    if R.get("hf_scan"):
        print("\n=== HEAVY FLAVOUR on prompt pair seeds (4a+6b, pT>2) ===")
        print(f"{'setting':20s}{'d0[um]':>8}{'pairs_phi':>11}{'tracklets':>11}"
              f"{'hits tested':>13}{'per proj':>10}{'efficiency':>12}{'fake':>8}")
        for k, v in R["hf_scan"].items():
            print(f"{k:20s}{v['d0_um']:8.0f}{v['pairs_phi']:11.0f}{v['tracklets']:11.0f}"
                  f"{v['hits_tested']:13.0f}{v['per_projection']:10.1f}"
                  f"{v['efficiency']:12.3f}{v['fake_fraction']:8.3f}")
    if R.get("proj_cost"):
        print("\n=== projection cost (per event) ===")
        print(f"{'side / seed':26s}{'tracklets':>10}{'projections':>12}{'hits tested':>12}"
              f"{'per proj':>10}{'pass z':>8}")
        for k, v in R["proj_cost"].items():
            print(f"{k:26s}{v['tracklets']:10.0f}{v['projections']:12.0f}"
                  f"{v['hits_tested']:12.0f}{v['per_projection']:10.1f}{v['pass_z']:8.0f}")


def calibrate_ot_bend(path, nev):
    """bend -> kappa per OT layer, MEASURED from on-track stubs vs track rInv,
    so no strip pitch or sensor spacing has to be assumed."""
    srcs = [f"{p}:Events" for p in expand_inputs(path)]
    stub_cols = ["bend", "layer", "trackIdx"]
    keys = uproot.open(srcs[0]).keys()
    S = uproot.concatenate(srcs, SPN.branch_names(keys, "L1TTrackStub", stub_cols))
    K = uproot.concatenate(srcs, ["L1TTrack_rInv"])
    if nev is not None:
        S, K = S[:nev], K[:nev]
    S = SPN.table_columns(S, "L1TTrackStub", stub_cols)   # link-only nanos: via L1TOTStub
    ns = ak.to_numpy(ak.num(S["bend"]))
    nt = ak.to_numpy(ak.num(K["L1TTrack_rInv"]))
    evs = np.repeat(np.arange(len(ns)), ns)
    bend = ak.to_numpy(ak.flatten(S["bend"]))
    lay = ak.to_numpy(ak.flatten(S["layer"])).astype(np.int64)
    ti = ak.to_numpy(ak.flatten(S["trackIdx"])).astype(np.int64)
    rinv = ak.to_numpy(ak.flatten(K["L1TTrack_rInv"]))
    off = np.concatenate([[0], np.cumsum(nt)])
    kap = np.full(len(bend), np.nan)
    ok = ti >= 0
    kap[ok] = rinv[off[evs[ok]] + ti[ok]] / (2.0 * C_BEND)
    cal = {}
    for L in range(1, 7):
        m = (lay == L) & np.isfinite(kap) & (np.abs(bend) > 1e-6)
        if m.sum() < 200:
            continue
        s = float(np.sum(bend[m] * kap[m]) / np.sum(bend[m] ** 2))
        r = kap[m] - s * bend[m]
        cal[L] = (s, float(1.4826 * np.median(np.abs(r - np.median(r)))), int(m.sum()))
    return cal


def acc_add(acc, o):
    for k, v in o.items():
        if k.startswith("_") or isinstance(v, str):
            continue
        acc[k] = acc.get(k, 0) + v


def selftest(a):
    """Does the z-binned search accept exactly what an exhaustive search does?"""
    global _EXHAUSTIVE
    set_limits(triplets_for_budget(a.pair_budget_gb),
               min(a.rss_ceiling_gb, SAFE_RSS_FRAC * PHYS_RAM_GB))
    nbad = ncmp = 0
    for D in it_chunks(a.input, a.nev or 2, a.batch_events):
        allidx = np.arange(len(D["layer"]))
        for bname, bench in BENCHMARKS.items():
            Q = it_prepare(D, bench)
            for cname, cfg in IT_CONFIGS.items():
                # BEAMLINE-CONSTRAINED PAIR SEEDING. Pair seeds take d0 = 0, so
                # kappa_slack = 0 and |kappa| <= kappa_max is a REAL pT cut: a 2 GeV
                # seed admits 2 GeV, not 0.98 GeV. Heavy flavour is not abandoned, it
                # is DELEGATED to the triplet below, where the three-point solve
                # MEASURES d0 instead of the pair absorbing it into a widened
                # curvature gate. This mirrors the OT's own split -- prompt pair seeds,
                # displaced triplets behind nbitsseedextended_ -- and it is the whole
                # point of having a triplet at all.
                jobs = [((la, lb), lc, False, 0.0) for (la, lb), lc in cfg["pairs"]]
                jobs.append((cfg["triplet"][:2], cfg["triplet"][2], True, 0.0))
                for (la, lb), lc, disp, d0c in jobs:
                    got = {}
                    for mode in (False, True):
                        _EXHAUSTIVE = mode
                        # the reference pairs on phi alone (no S2j pre-gate)
                        ang = AngleUse(set(LEGACY_ANGLE_STAGES)
                                       - ({"pair_joint"} if mode else set()))
                        try:
                            o = it_pair_seed(D, Q, allidx, la, lb, lc, 2.0,
                                             ang, disp, d0c)
                        except TooWide:
                            got = None
                            break
                        t = o.get("_trip")
                        # NOT just the triples. A first revision compared only those, and a
                        # joint-pairing bug that dropped 0.2-1.4 cluster pairs/event passed
                        # clean, because the lost pairs never reached a triple. pairs_phi and
                        # match_cand are WORK counters and MUST differ, so they stay out.
                        got[mode] = (
                            (set() if t is None else
                             set(zip(t[0].tolist(), t[1].tolist(), t[2].tolist()))),
                            tuple(round(o.get(c, -1), 6) for c in
                                  ("tracklets", "match_cand_z", "tracks_to_fit")))
                    _EXHAUSTIVE = False
                    if got is None:
                        continue
                    ncmp += 1
                    tag = (f"{bname}|{cname}|L{la}L{lb}->L{lc}"
                           + ("|displaced" if disp else ""))
                    if got[False] != got[True]:
                        nbad += 1
                        (sF, cF), (sT, cT) = got[False], got[True]
                        print(f"  MISMATCH {tag}: missed {len(sT - sF)}, invented "
                              f"{len(sF - sT)}, of {len(sT)} reference triples; "
                              f"counters {cF} vs {cT}")
        break
    print(f"selftest: {ncmp} seed configurations compared, {nbad} mismatched")
    raise SystemExit(1 if nbad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=None,
                    help="stop after this many events; default all")
    ap.add_argument("--batch-events", type=int, default=16,
                    help="events per streamed chunk. NOT the memory knob -- see "
                         "--pair-budget-gb. Peak memory is set by candidate triplets "
                         "held at once, which for a d0-inflated projection window "
                         "explodes WITHIN a single event. Does not change results.")
    ap.add_argument("--pair-budget-gb", type=float, default=DEFAULT_PAIR_BUDGET_GB,
                    help="THE MEMORY KNOB: bytes per candidate-triplet slice. Peak "
                         "memory is flat in the total candidate-triplet count.")
    ap.add_argument("--rss-ceiling-gb", type=float,
                    default=round(min(6.0, SAFE_RSS_FRAC * PHYS_RAM_GB), 1),
                    help=f"abort above this. HARD-CLAMPED to {SAFE_RSS_FRAC:.0%} of "
                         f"physical RAM ({PHYS_RAM_GB:.1f} GB here); a ceiling above "
                         "RAM cannot protect anything.")
    ap.add_argument("--benchmark", default=None, choices=sorted(BENCHMARKS),
                    help="run ONE angle-bit benchmark; default all")
    ap.add_argument("--d0", type=float, default=500e-4,
                    help="prompt d0 allowance [cm] for the benchmark pass")
    ap.add_argument("--skip-hf-scan", action="store_true")
    ap.add_argument("--skip-ot", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="compare the z-binned projection search against an "
                         "exhaustive one on a few events; exits nonzero on any "
                         "difference in the accepted cluster triples")
    ap.add_argument("-o", "--out", default="tracklet_topology_cost.json")
    a = ap.parse_args()
    benches = ({a.benchmark: BENCHMARKS[a.benchmark]} if a.benchmark else dict(BENCHMARKS))
    if a.selftest:
        return selftest(a)

    acc, purity, reck, findk = {}, {}, {}, {}
    tp_k, tp_d0 = [], []
    nev = ncl = 0

    # A CEILING ABOVE PHYSICAL RAM IS NOT A CEILING. Passing
    # --rss-ceiling-gb 40 on a 24 GB machine is what turned a diagnosable bug
    # into a hard lockup, so the option is clamped here rather than trusted.
    ceiling = min(a.rss_ceiling_gb, SAFE_RSS_FRAC * PHYS_RAM_GB)
    if ceiling < a.rss_ceiling_gb:
        print(f"  --rss-ceiling-gb {a.rss_ceiling_gb:.1f} clamped to {ceiling:.1f} "
              f"({SAFE_RSS_FRAC:.0%} of {PHYS_RAM_GB:.1f} GB physical)")
    set_limits(triplets_for_budget(a.pair_budget_gb), ceiling)

    cap_census = None
    for D in it_chunks(a.input, a.nev, a.batch_events):
        rss_check()
        nev += len(D["_events"])
        ncl += len(D["layer"])
        cap_census = SPN.angle_cap_census_add(cap_census, angle_cap_census(D))
        ti = build_tp_index(D)
        tp_k.append(ti["key"]); tp_d0.append(ti["d0"])
        allidx = np.arange(len(D["layer"]))

        for cname, cfg in IT_CONFIGS.items():
            for ptmin in PT_MINS:
                findk.setdefault((cname, ptmin), []).append(
                    findable_keys(D, *cfg["triplet"], ptmin))

        for bname, bench in benches.items():
            Q = it_prepare(D, bench)
            for cname, cfg in IT_CONFIGS.items():
                for ptmin in PT_MINS:
                    # BEAMLINE-CONSTRAINED PAIR SEEDING. Pair seeds take d0 = 0, so
                    # kappa_slack = 0 and |kappa| <= kappa_max is a REAL pT cut: a 2 GeV
                    # seed admits 2 GeV, not 0.98 GeV. Heavy flavour is not abandoned, it
                    # is DELEGATED to the triplet below, where the three-point solve
                    # MEASURES d0 instead of the pair absorbing it into a widened
                    # curvature gate. This mirrors the OT's own split -- prompt pair seeds,
                    # displaced triplets behind nbitsseedextended_ -- and it is the whole
                    # point of having a triplet at all.
                    jobs = [((la, lb), lc, False, 0.0) for (la, lb), lc in cfg["pairs"]]
                    jobs.append((cfg["triplet"][:2], cfg["triplet"][2], True, 0.0))
                    for (la, lb), lc, disp, d0c in jobs:
                        tag = f"{bname}|{cname}|pt{ptmin:g}|" + (
                            "displaced" if disp else f"L{la}L{lb}->L{lc}")
                        A = acc.setdefault(tag, {})
                        P = purity.setdefault(tag, [0, 0])
                        # ONE call for the whole chunk -- no event loop
                        try:
                            o = it_pair_seed(D, Q, allidx, la, lb, lc, ptmin,
                                             True, disp, d0c)
                        except TooWide as e:
                            # A RESULT, not a failure: this design point needs
                            # more candidate triplets than any sane machine can
                            # hold, so it is infeasible as formulated.
                            A["infeasible_candidate_triplets"] = e.n
                            print(f"  INFEASIBLE {tag}: {e.n:.3g} candidate "
                                  f"triplets (ceiling {e.ceiling:.3g})")
                            continue
                        acc_add(A, o)
                        keys = []
                        if "_trip" in o:
                            ga, gb, gc = o["_trip"]
                            keys.append(recovered_keys(D, ga, gb, gc))
                            nc, nt = cand_purity(D, ga, gb, gc)
                            P[0] += nc; P[1] += nt
                        if keys:
                            # collapse per chunk so the key list cannot grow with
                            # the sample -- this is what blew up at 9.7 GB before
                            prev = reck.get(tag)
                            merged = np.concatenate(keys + ([prev] if prev is not None else []))
                            reck[tag] = np.unique(merged)
        del D

    tpi = {"key": np.concatenate(tp_k), "d0": np.concatenate(tp_d0)}
    o = np.argsort(tpi["key"], kind="stable")
    tpi = {"key": tpi["key"][o], "d0": tpi["d0"][o]}
    find = {k: np.unique(np.concatenate(v)) for k, v in findk.items()}

    R = {"n_events": nev, "n_clusters": ncl, "clusters_per_event": ncl / max(nev, 1),
         "nsig": NSIG, "z_lumi_cm": Z_LUMI, "eta_matched": ETA_MATCHED,
         "d0_allowance_cm": a.d0, "benchmarks_run": sorted(benches), "it": {},
         "angle_cap_sentinel": cap_census}
    print(f"IT: {nev} events, {ncl/max(nev,1):.0f} clusters/event")
    if cap_census is not None:
        log_angle_cap(cap_census, "it_chunks")
    for tag in sorted(acc):
        bname, cname, ptl, seed = tag.split("|")
        ptmin = float(ptl[2:])
        per = {k: v / max(nev, 1) for k, v in acc[tag].items() if not isinstance(v, dict)}
        fk = find[(cname, ptmin)]
        rk = reck.get(tag, np.empty(0, dtype=np.int64))
        hit = np.isin(fk, rk, assume_unique=True)
        per["n_findable"] = int(len(fk))
        per["efficiency"] = float(hit.mean()) if len(fk) else 0.0
        per["fake_fraction"] = 1.0 - purity[tag][1] / max(purity[tag][0], 1)
        per["by_d0"] = eff_by_d0(rk, fk, tpi)
        pj = per.get("projections", 0)
        per["hits_per_projection"] = (per.get("match_cand", 0) / pj) if pj else 0.0
        R["it"][tag] = per
        print(f"  {tag:44s} trk={per.get('tracklets',0):9.0f} "
              f"hits/proj={per['hits_per_projection']:6.1f} "
              f"eff={per['efficiency']:.3f} fake={per['fake_fraction']:.3f}")
        for bk, bv in per["by_d0"].items():
            print(f"      {bk:20s} eff={bv['efficiency']:.3f} (n={bv['n_findable']})")

    # ---- UNION over seeds, which is the only system-level efficiency --------
    # A per-seed efficiency cannot answer "is one seed type enough", because two
    # seeds may find the same TrackingParticles or complementary ones and the
    # per-seed numbers look identical either way. This unions the recovered keys
    # so the triplet alone can be compared against the pairs alone and against
    # everything together, on the same findable denominator.
    R["it_union"] = {}
    for (bname, cname, ptmin) in sorted({(t.split("|")[0], t.split("|")[1],
                                          float(t.split("|")[2][2:])) for t in acc}):
        fk = find[(cname, ptmin)]
        if not len(fk):
            continue
        groups = {"pairs_only": [], "triplet_only": [], "all_seeds": []}
        for t in acc:
            b2, c2, p2, seed = t.split("|")
            if (b2, c2, float(p2[2:])) != (bname, cname, ptmin):
                continue
            rk = reck.get(t)
            if rk is None:
                continue
            groups["all_seeds"].append(rk)
            groups["triplet_only" if seed == "displaced" else "pairs_only"].append(rk)
        ent = {"n_findable": int(len(fk))}
        for g, arrs in groups.items():
            u = np.unique(np.concatenate(arrs)) if arrs else np.empty(0, np.int64)
            hit = np.isin(fk, u, assume_unique=True)
            ent[g] = {"efficiency": float(hit.mean()),
                      "by_d0": eff_by_d0(u, fk, tpi)}
        R["it_union"][f"{bname}|{cname}|pt{ptmin:g}"] = ent
        if ptmin == 2.0:
            print(f"  UNION {bname}|{cname}|pt2  findable={len(fk):,}  "
                  + "  ".join(f"{g}={ent[g]['efficiency']:.3f}" for g in
                              ("pairs_only", "triplet_only", "all_seeds")))

    if not a.skip_ot:
        try:
            O, onev, have = load_ot(a.input, a.nev or 10 ** 9)
        except Exception as ex:
            print(f"OT SKIPPED: {type(ex).__name__}: {ex}")
            O = None
        if O is not None:
            eta = O["eta"]
            R["ot_input"] = {"stubs_per_event_full": len(O["layer"]) / onev,
                             "stubs_per_event_matched": float((eta < ETA_MATCHED).sum() / onev),
                             "has_truth": "tpIdx" in have}
            print(f"OT: {R['ot_input']['stubs_per_event_full']:.0f} stubs/event full, "
                  f"{R['ot_input']['stubs_per_event_matched']:.0f} matched")
            cal = calibrate_ot_bend(a.input, a.nev or 10 ** 9)
            R["ot"] = {}
            oall = np.arange(len(O["layer"]))
            for ename, emax in (("barrel_alleta", None), ("barrel_matched", ETA_MATCHED)):
                seeds = [k for k, v in OT_SEEDS.items()
                         if v["pair"] is not None and (emax is None or not v["disc"])]
                R["ot"][f"{ename}_topology"] = {
                    "seeds": seeds, "n_seeds": len(seeds),
                    "n_projection_paths": sum(len(OT_SEEDS[k]["proj_l"])
                                              + (0 if emax else len(OT_SEEDS[k]["proj_d"]))
                                              for k in seeds),
                    "n_tc": sum(OT_SEEDS[k]["ntc"] for k in seeds)}
                for ptmin in PT_MINS:
                    for scheme in ("position", "bend"):
                        tot = {}
                        for sname in seeds:
                            A = {}
                            r = ot_seed_cost(O, oall, sname, ptmin,
                                             scheme == "bend", cal, emax)
                            if r:
                                acc_add(A, r)
                            # RATIOS MUST NOT BE DIVIDED BY THE EVENT COUNT.
                            # efficiency and fake_fraction are already fractions;
                            # dividing them by onev turned an efficiency of ~0.9
                            # into 0.034 at 40 events.
                            _rat = ("efficiency", "fake_fraction")
                            tot[sname] = {k: (v if k in _rat else v / onev)
                                          for k, v in A.items()}
                        tot["_sum_tracklets"] = sum(v.get("tracklets", 0)
                                                    for v in tot.values() if isinstance(v, dict))
                        R["ot"][f"{ename}_pt{ptmin:g}_{scheme}"] = tot
                        print(f"  OT {ename:14s} pT>{ptmin:<4g} {scheme:8s} "
                              f"trk={tot['_sum_tracklets']:9.0f}")
    R["peak_rss_gb"] = round(peak_gb(), 2)
    R["batch_events"] = a.batch_events
    print(f"peak RSS {R['peak_rss_gb']:.2f} GB at {a.batch_events} events/batch")
    json.dump(R, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
