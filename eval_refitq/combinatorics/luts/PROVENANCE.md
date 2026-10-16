# Tracklet LUTs, dumped from CMSSW

Source: CMSSW_20_1_0_pre1, `L1Trigger/TrackFindingTracklet`, geometry
**D121** (era Phase2C22I13M9, GT `auto:phase2_realistic_T35`), 1 event of
`RelValTTbar_D121_noPU_regen.root`.

The tables are built at `beginRun` from geometry + `Settings` only -- no event
data enters them -- so one event is enough; the event only forces `beginRun`.

## How they were produced

`writeTable_` is a hardcoded `false` private member with **no cfi parameter and
no setter** (`Settings.h:914`; checked against all 25 setters and every
`settings_.set*` call site). Producing these required a temporary local edit:

    -    bool writeTable_{false};
    +    bool writeTable_{true};

then `scram b L1Trigger/TrackFindingTracklet`, one `cmsRun`, then the edit was
**reverted and the package rebuilt**. The working area is left as it was found.

Config kept at `cmssw/work/_lutdump/dumpLUTs_cfg.py`.

## GEOMETRY IS NOT INTERCHANGEABLE

The bend cuts come from each `trackerDTC::SensorModule`'s stub window, tilt and
sensor separation, so a D110 dump would NOT substitute for this one. If the
ntuples move to another geometry, re-dump.

## What is here

- `TP_<seed><TP-instance>_stubpt{inner,outer}cut.{tab,dat}` -- the seed-pair
  bend cut, one pass/fail bit per (dphi, bend) cell. 176 files: 8 prompt seeds
  x up to 12 TP instances x inner/outer.
- `METable_<L>.{tab,dat}` -- the projection/match bend cut.
- `VMPhiCorrL<n>.{tab,dat}` -- the bend-dependent phi correction that turns a
  raw digitised stub phi into `phicorr`, which is what the fine-phi index is
  built from. Required to index the tables above; not optional.
- plus the VMR / projection / match-cut tables the same flag writes.

`.dat` is zero-padded uppercase hex, one value per line, and is the easier one
to load (`np.loadtxt(..., dtype=np.uint8)` for the 1-bit tables). `.tab` is a
C-initialiser block: `{`, then decimal values comma-separated one per line,
then `};`.

## TrackletProcessor wiring (`TP_wiring_barrel.txt`)

The inner allstub memories (phi region and memory type) and the outer region of every barrel prompt
TrackletProcessor (TP_L1L2A-L, TP_L2L3A-D, TP_L3L4A-D, TP_L5L6A-D), as CMSSW_20_1_0_pre1 builds the prompt chain:
the hardcoded `TPs` table of `TrackletConfigBuilder::writeASMemories` (src/TrackletConfigBuilder.cc:1025-1051),
written out in the wires-file layout. The inner memory type is the name's 12th character, as VMRouterCM reads it.

An earlier version took this from the cms-data `L1Trigger-TrackFindingTracklet` V00-06-00 wires file
(`wires_hourglassExtendedAllCombined.dat`), which the extended / reduced configurations read. Its L2L3, L3L4 and
L5L6 rows are the same, but its **L1L2 rows are not**: the prompt chain's L1 inner memories are wired
differently, and the emulation then formed L1L2 tracklets CMSSW does not (about 3 % of L1L2 no-DR tracks,
concentrated in two processors). The file fixes each tracklet's TP letter, i.e. its TCID order, and which inner
stubs pair with which outer region.
