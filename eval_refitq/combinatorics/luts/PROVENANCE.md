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

The inner allstub memories (phi region and memtype) and the outer VM memory of every barrel prompt
TrackletProcessor (TP_L1L2A-L, TP_L2L3A-D, TP_L3L4A-D, TP_L5L6A-D), extracted in file order from the cms-data
package `L1Trigger-TrackFindingTracklet` V00-06-00, `data/wires_hourglassExtendedAllCombined.dat`
(sha256 18734bc9f63c5eac8e976de8aa48eab06d87a0619cfa55aee5fa82596f5353d6). In CMSSW_20_1_0_pre1 the prompt chain's
wiring is generated in code by TrackletConfigBuilder; that its TrackletProcessor inputs equal this file's is
ASSUMED (the tracklet-engine LUTs already confine the pairs to the wired combinations: adding the wiring changed no
tracklet on the canon_v4 10-event test). It fixes the TP letter of each tracklet, i.e. its TCID order.
