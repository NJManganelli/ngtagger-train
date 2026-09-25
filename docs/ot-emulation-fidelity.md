# Our OT emulation vs the real Phase-2 track finder: what we match, what we
# deliberately do not, and why

Date: 2026-09-17. Reference code:
`work/CMSSW_20_1_0_pre1/src/L1Trigger/{TrackFindingTracklet,TrackFindingTMTT}`
(branch `smartpixels-phase3-tier2refit`; in these two packages it is
byte-identical to `cms-l1tk/L1TK-PR-20_1_0_pre1`, verified by `diff -rq`).
Reference collection: nano `L1TTrack` <- `l1tTTTracksFromTrackletEmulation:
Level1TTTracks`, produced with `Extended=False` and `Hnpar=5`.
Diagram: `eval_refitq/combinatorics/results/seed_coverage_rz.png`.

## The measurement that started this

Restricted to OT-only seeds and OT barrel layers, our emulation gave
409 tracks/event after duplicate removal against the real 75.7 (5.4x), and
sigma(d0) = 690 um on clean tracks against the real 435 um on genuine ones
(1.6x). Neither gap was understood; both now are.

## THE OUTPUT CEILING TO DESIGN AGAINST

**9 phi nonants x 2 eta sectors x 104 tracks = 1872 tracks per event.**

104 per region is the tuned OUTPUT ceiling of the OT track finder, sized on
ttbar PU200 at roughly a 5-sigma upward fluctuation in track multiplicity. This
is the number a proposed seed menu has to be costed against.

Measured for scale, `L1TTrack` in our own files (100-200 events each of
`itot_truth`, `itot_ttbar_f02`, `itot_pu200`): **181-185 tracks/event**, about
20 per nonant, busiest nonant 47-56. That is ~10% of the 1872 budget, which is
what a ceiling tuned for 5-sigma fluctuations plus margin should look like.

### CORRECTION: the "108 per nonant" reading below was wrong

An earlier version of this document (and of `tp_findability.py`) read
`maxstep_["DR"] = 108` as "108 candidates per phi nonant", and concluded the
real chain is limited to <= 972 tracks per event. **That cannot be right**: a
108-per-nonant input cap would put the 104-per-region output ceiling out of
reach, so the system could never exercise its own tuned maximum. `Settings.h`
says what 108 is in the first line of the block --

    //Number of processing steps for one event (108=18TM*240MHz/40MHz)

-- a CLOCK BUDGET: 18 time-multiplexed regions x (240/40) cycles. And 18 is
exactly 9 nonants x 2 eta sectors. So `maxstep_` counts processing steps per
module instance; it is not a per-nonant track count, and "per bin" in the DR
comment means one of those 18 regions. The per-region occupancy counter in
`tp_findability.truncation_table` now bins on (nonant x eta sector) against 104.

`L1TTrack_etaSector` is 99 for every track in every file we have, so the eta
boundary is currently an assumption (split at eta = 0) rather than a
measurement; it redistributes load between two bins and cannot change the total.

## DELIBERATE DEVIATION: we do not emulate the per-region cap

The budget is consumed in memory order
(`TB_AAAA` before `TB_BBBB`, `src/TrackletConfigBuilder.cc:1257-1261` with
`interface/TrackletConfigBuilder.h:315`), so in a busy nonant L3L4 and the disk
seeds are starved outright, and the seed-rank sort happens AFTER the cap.

**Decision: not emulated, on purpose.** It is a firmware/bandwidth truncation of
a phi nonant, not physics. We are trying to learn what is POSSIBLE and to dial
in windows and seeds; a cap applied now would clip the distribution we are
trying to measure and would hide which stage is actually generous. It becomes
relevant later, when a chosen configuration has to be shown feasible - at that
point it is the right way to express the cost ceiling, and it belongs in the
census as an option, not as a default.

Consequence to keep in mind when quoting numbers: our candidate counts are
NOT comparable to the real ones while this is off. Comparisons of RESOLUTION,
purity and per-seed reach remain valid.

What the load actually looks like, full 30-seed exploratory menu, measured:
**8192 fitted tracks/event = 437% of the 1872 budget**, 77% of them beyond a
104-per-region cap. But that is a property of the exploratory menu, not of the
detector: three degenerate long-lever seeds carry 62% of it (`IL1+OL1` alone is
343 per nonant at 96% fake), while **all the OT-only seeds together come to
~21 per nonant, under 20% of the budget**. So truncation does not threaten the
good-d0 tracks in a sensible menu; it only bites if the degenerate seeds are
kept. Those seeds are already labelled `DEGENERATE: long lever arm` in
`seed_arity.enumerate_seeds`, kept deliberately so the cost columns could
condemn them -- which they now do quantitatively.

## WINDOW POLICY: match the real tuning, then widen only by what the physics requires

The real per-(seed, layer) match cuts are `rphimatchcut_` and `zmatchcut_`
(`Settings.h:724-741`, applied at `src/MatchProcessor.cc:684` via
`src/TrackletLUT.cc:192-215`). They are already transcribed in our tree as
`OT_RPHI_CUT` / `OT_Z_CUT` (`tracklet_topology_cost.py:124-137`) - and until
today they were used only by `ot_seed_cost` (`:1104`), never by the
`it_pairs`/`it_project` path the census actually calls.

Those cuts are tuned for **prompt, pT >= 2**. We want prompt AND displaced down
to **1 GeV**, so the window must be widened - but only by the two terms that
actually change:

    window = real cut
           + (scattering at 1 GeV - scattering at 2 GeV)     # our THETA_MS_MRAD/pT
           + d0_allowance / r                                 # displaced coverage

Measured, in cm of r*dphi:

| seed | target | real | +1 GeV MS | +1 mm d0 | needed | our old window |
|---|---|---|---|---|---|---|
| L1L2 | OL3 | 0.100 | 0.050 | 0.100 | 0.250 | 1.081 |
| L1L2 | OL6 | 0.500 | 0.104 | 0.100 | 0.704 | 4.237 |
| L3L4 | OL5 | 0.080 | 0.083 | 0.100 | 0.263 | 2.377 |
| L5L6 | OL4 | 0.050 | 0.066 | 0.100 | 0.216 | 1.212 |

**Median: real 0.100 cm, needed 0.250 cm = 2.5x the real cut.** Our old windows
were 8-10x the real cut, so roughly 3-4x of our width was pure slop, not
coverage. On the z side the real cuts (0.7-15 cm) already dwarf the 1 GeV
scattering addition (0.07-0.31 cm), so z needs essentially no widening and our
2x z excess was entirely slop.

Why this matters beyond combinatorics: a Poisson accidental model using our old
windows and the measured per-layer occupancies predicts ~3900 four-layer
candidates/event against the 3856 we measured - i.e. **essentially every
candidate we were producing was an accidental assembly**, where real windows
give ~80. The slop was not buying efficiency; it was manufacturing fakes.

### DO NOT HANDICAP THE BASELINE THE ANGLES ARE MEANT TO IMPROVE

SmartPixels cluster angles genuinely reduce combinatorics. But if the no-angle
OT baseline runs with 8x windows, its accidental rate is inflated 100x
(computed per seed: 105x for L1L2, 144x for L2L3, 77x for L3L4, 11x for L5L6),
and ANY angle-based pruning would then show a spectacular improvement that is
mostly an artefact of the strawman. Every angle-vs-no-angle comparison must use
the policy windows above on both arms. This is the single easiest way to
accidentally overstate the SmartPixels case, and it is why the window fix comes
before any further angle study.

## FIDELITY FIXES APPLIED TO THE OT SIDE

1. Adopt `OT_RPHI_CUT`/`OT_Z_CUT` in the `it_project` path, widened by the
   policy above. Replaces a generic 3-sigma envelope built on a hard-coded
   **1 mrad per-stub r-phi resolution** - the true OT stub resolution is
   0.116 mrad at OL1 (PS) and 0.024 mrad at OL6 (2S), so the old model was
   pessimistic by 10-40x, and its `5e-4` floor plus `THETA_MS_MRAD = 1.36`
   scattering term alone exceeded the real window for 9 of 15 wired targets.
2. Inner-stub |z| acceptance at the seed stage, absent until now
   (`src/VMRouterCM.cc:261-281`, `src/TrackletLUT.cc:1199-1201`):
   `VMROUTERCUTZL2 = 50` (L2 inner stubs need |z| >= 50; >= 52 for L2L3),
   `VMROUTERCUTZL1L3L5 = 95` (L1/L3/L5 inner stubs need |z| <= 95).
   Measured acceptance in our barrel selection: L2 **0.167**, L5 0.798,
   L3 0.970, L1 1.000. So **L2L3 is by design a transition-region seed
   (|eta| 1.14-1.89) and L5L6 a central one (|eta| <= 0.95)**; we had been
   running both across the whole barrel, over-counting OL2+OL3 pairs 6x.
3. Third-order helix corrections in the KF residual
   (`TrackFindingTMTT/src/KFbase.cc:603,610`, active because
   `kalmanHOhelixExp_=true, kalmanHOfw_=false` in the hybrid DEFAULT
   constructor, `src/Settings.cc:60,63`):
   `correction[0] += (1/6)(r*inv2R)^3`, `+= (1/6)(d0/r)^3` for 5-par, and
   r-z `deltaS = (1/6) r (r*inv2R)^2`, `correction[1] -= deltaS*tanL`.
   Omitting these is the measured cause of the sigma(d0) gap: the term is odd
   in charge, so it broadens symmetrically with no mean offset, predicted
   1151 um at pT=2 falling as 1/pT^3. Confirmed in our own data - clean
   OT-only sigma(d0) runs 1127 um at pT 2-2.5 and plateaus at ~180 um above
   10 GeV, exactly the predicted shape.
4. Bend / rinv consistency for OT stubs, at both the seed stage
   (`src/TrackletEngineUnit.cc:105-109`, `Settings.h:813-820`) and the match
   stage (`src/MatchEngineUnit.cc:160-190`, `Settings.h:822-837`). The hooks
   existed in our code but were inert: `tp_findability.py` set
   `s_kap = s_z0 = 1e9` for every OT row, making every gate tautologically
   true. Worth ~5x per stub at OL5/OL6, nearly nothing at OL1/OL2.
5. Drop our two OT triplet seeds from the baseline. `data/seedWiring.json`
   contains only the four EXTENDED triplets and is read only by `*Ext`
   functions behind `if (extended_)` (`src/TrackletConfigBuilder.cc:1381`);
   the nano `L1TTrack` comes from the `Extended=False` producer. Our triplets
   were 23% of our track count and have no counterpart in the reference. Their
   pair ordering was also wrong against the extended design: `Settings.h:687`
   pairs L3L4 with third L2, while our `Seed` class took `layers[0], layers[1]`
   and so seeded on OL2+OL3.
6. d0 SIGN. `KFParamsComb::matrixH` uses `matH(PHI,D0) = -1/r`
   (`KFParamsComb.cc:171`) and TMTT truth is `d0 = vx sin(phi0) - vy cos(phi0)`
   (`src/TP.cc:31`), whereas our model is `phi(r) = phi0 + d0/r` with
   `d0 = -vx sin + vy cos`. Ours is internally consistent, so our sigma is
   unaffected - but **our fitted d0 is the negative of `L1TTrack_d0`**, so any
   overlay against the nano, or any refit seeded from a nano d0, is flipped.

## VERIFIED ALREADY CORRECT (do not "fix" these)

- r-phi measurement variance is bit-for-bit the real model: `sigmaPerp =
  pitch/sqrt(12)`, `vphi = sigmaPerp^2/r^2 + (0.00075/pt)^2`, with the pT of the
  SEED tracklet held fixed across layers (`KFParamsComb.cc:98-102`).
- chi2/z0/d0/pT acceptance tables, their indexing by layer count, and
  `chi2scaled = chi2rphi/8 + chi2rz` (`Settings.cc:50-58`).
- `MIN_SHARED_LAYERS = 3` matches `minIndStubs_ = 3`; the static seed ranks
  `{1,5,2,7}` match `PurgeDuplicate.cc:171`.
- The 4-layer rule IS arity-aware in the real chain: a prompt doublet needs
  >= 2 unique matched layers, a triplet >= 1 (`src/FitTrack.cc:935-937`),
  exactly what `Seed.min_proj` does. **Our `seed_arity.py` docstring claiming
  CMSSW demands 5 layers of a triplet-seeded track is wrong** - that
  `kfMinProj = 2` belongs to the newer HPH/TrackerTFP KF, not this chain.
- Our 7-feature description of the real track-quality MVA is right:
  `tanl, z0_scaled, bendchi2_bin, nstub, nlaymiss_interior, chi2rphi_bin,
  chi2rz_bin` (`python/l1tTTTracksFromTrackletEmulation_cfi.py:34-41`,
  `interface/FeatureTransform.h:51-60`), digitised bin indices rather than
  floats. Its score only populates `trkMVA1`; **no track is cut on any MVA
  anywhere in the chain** (`plugins/L1FPGATrackProducer.cc:722-727`).
  `data/HYBRID_NEW_KF_TQ_XGBOOST_v0.json` is a different, 6-feature model for
  the TrackerTFP "NewKF" chain and is not scheduled in production.
- The d0 prior cannot explain any resolution gap: with six stub azimuths, the
  real informative prior, our prior and a flat prior agree within 2%.


## THE IT IS BARREL-ONLY HERE, AND SEED DESIGN MUST NOT BAKE THAT IN

The SmartPixels collection in these files is purely TBPX barrel: layer values
1-4 only, max |z| = 20.1 cm, max r = 15.5 cm. No disk surfaces exist in it at
all. This is a pixelAV limitation, not a detector design -- the code says so
itself: `L1Trigger/Phase3SmartPixels/interface/SmartPixelsHelixProjector.h:6,23`
is explicitly "Inner-Tracker BARREL (TBPX) module lookup" and notes that
endcap TEPX/TFPX handling "land[s] HERE once"; the refit sidecar's field is
`uint8_t layer  // TBPX layer 1..4`.

### What this invalidates

Measured cluster occupancy by |eta| band shows IL1 reaching |eta| = 2.65 while
IL4 stops at 1.13, from which it is tempting to conclude that IL1+IL2 is the
only usable seed above |eta| ~ 1.2 and that a 4-layer IT track is impossible
there. **Both are artefacts of the missing disks.** In the real Phase-2 Inner
Tracker the TFPX disks cover precisely that region, so a high-eta track has
four or more surfaces available - they are simply not barrel layers. Designing
eta-specialised IT seeds on the barrel-only geometry would hard-code a
placeholder into the seed menu.

### The real geometry, read from CVMFS (so it need not be guessed again)

Source: `Geometry/TrackerCommonData/data/PhaseII/Tracker_DD4hep_compatible_2021_02/
pixel.xml` in `/cvmfs/cms.cern.ch/*/cms/cmssw/CMSSW_20_1_0_pre1`, which is what
`cmsExtendedGeometryRun4D121XML_cfi.py` loads for the D121 geometry used to make
these files. Disc z = `ZPixelForward` (291 mm, from `pixfwd.xml`) plus the placed
offset; radial extents from each `ITDisc` Tubs. The 291 mm anchor is inferred
from the constant and cross-checked against the resulting layout, not read from
a PosPart directly.

| surface | z [cm] | r [cm] | \|eta\| covered |
|---|---|---|---|
| TBPX IL1-IL4 | \|z\| < 20 | 2.85, 5.95, 10.33, 14.50 | 0 - 1.13 (four layers) |
| TFPX ITDisc1-8 | 25.3, 32.3, 41.2, 52.6, 67.2, 84.2, 110.9, 139.7 | 3.05 - 16.15 | 1.23 - 4.52 |
| TEPX ITDisc9-12 | 175.0, 201.0, 230.8, 265.0 | 6.24 - 25.51 | 2.62 - 4.44 |

**Surfaces a track from the origin crosses, real geometry against ours:**

| \|eta\| | barrel | discs | REAL total | our barrel-only census |
|---|---|---|---|---|
| 1.1 | 4 | 0 | 4 | 4 |
| 1.3 | 3 | 1 | **4** | 3 |
| 1.5 | 2 | 2 | **4** | 2 |
| 1.8 | 2 | 3 | 5 | 2 |
| 2.5 | 1 | 6 | 7 | 1 |
| 3.0 | 0 | 10 | 10 | 0 |

The real Inner Tracker keeps **at least four surfaces at every \|eta\| out to 4.0**;
the barrel-only placeholder falls to two by 1.5 and to zero by 3.0. So the
four-surface requirement is satisfiable everywhere in the real detector, met by
barrel, then barrel+disc, then disc-only as \|eta\| grows -- structurally the same
progression the OT uses (L1L2 -> L2D1 -> D1D2).

### What to do instead

- **Identify a seed by a pair of SURFACES, not by barrel layer index.** A
  surface is a barrel layer or a disk, and adjacency is along the trajectory in
  r-z. This is exactly how the OT expresses it (L1D1, L2D1, D1D2 are
  surface pairs sitting alongside L1L2 and L3L4), so the abstraction is already
  proven. Adding TFPX/TEPX then becomes data, not a redesign.
- **State the layer rule as "surfaces crossed", not "barrel layers hit."**
  Otherwise the 4-layer requirement silently becomes unsatisfiable above
  |eta| = 1.13 rather than being met by disks.
- **Scope every eta-dependent conclusion to |eta| < ~1.1**, where the
  barrel-only geometry is complete, and label anything beyond it as not yet
  measurable rather than as a result.

### The asymmetry to carry forward

Disk-based IT seeds will have POSITIONS ONLY for the foreseeable future: there
is no pixelAV angle regression for the orientations the disks impose. So every
capability that leans on cluster angles - combinatoric rejection at the seed
stage, the angle-pull features that dominate the cleanliness MVAs, the
consistency checks that give a 3-hit track its only degrees of freedom - is a
BARREL capability today. Quoting an angle-derived gain as if it applied to the
whole SmartPixels acceptance would overstate it in exactly the forward region
where the disks do the work.

## OPEN, NOT YET ANSWERED

- Whether the 108/sector cap is actually saturated at PU200 - only measurable
  by instrumenting `PurgeDuplicate.cc:156` and reading `FitTrack.cc:1040-1042`.
- `matchport_` (`TrackletConfigBuilder.h:298`) says L2L3 does NOT project to
  OL6, while `Settings.h:697` `projlayers_` and `python/Setup_cfi.py:51` say it
  does. `matchport_` drives the wiring and the zero cut corroborates it, but
  the conflict is unresolved.
- Per-stub truth on OT stubs is only populated for GENUINE stubs: of 294k OT
  stubs, 49.2% genuine (tpIdx set), 26.3% combinatoric and 24.6% unknown (both
  tpIdx = -1). So "wrong hit" on an OT stub means "not provably from the
  owner". New counters `n_ot_right/wrong/comb/unknown` are being added; the IT
  side cannot be split the same way because clusters carry only `tpIdx`.
