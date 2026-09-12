# ngtagger-train

Training pipelines for the CMS Phase-2 **GTT / Correlator ML algorithms**,
reading the extended L1 nano tiers (**L1TrkNano / L1PFNano / L1PFTrkNano**)
directly — no intermediate data formats. Covers:

| pipeline | command | what it retrains | stock reference in nano |
|---|---|---|---|
| NG jet tagger | `train` / `multishot` | DeepSet HGQ2 multiclass+regression (8 classes) | `L1puppiJetSC4NG_*TagScore*` |
| Contrastive tagger | `train` (contrastive config) | SimCLR embedding pre-train → HGQ2 fine-tune | same |
| Track quality | `train-trkquality` | TrackerTFP GBDT (genuine-vs-fake) | `L1TTrack_trkMVA1` |
| E2E vertexing | `train-nnvtx` | NNVtx weight/pattern + association networks | `L1Vertex_z0/sumPt` |
| Displaced vertex | `train-dispvtx` | GTT DV conifer GBDT | `L1DispVertex_score` |

Inspired by and tracking [CMS-L1T-Jet-Tagging/TrainTagger](https://github.com/CMS-L1T-Jet-Tagging/TrainTagger)
(`Keras_v3` / `embedding_kv3` lines): HGQ2 quantization-aware training,
da4ml distributed arithmetic, SimCLR contrastive pre-training.

---

## 1. Environment (pixi — fully reproducible)

Install [pixi](https://pixi.sh), then:

```bash
git clone <this repo> && cd ngtagger-train
pixi install                 # macOS arm64 (M-series; tensorflow + tensorflow-metal)
pixi run test                # verify: full synthetic test suite
```

On **macOS x86_64 (Intel)** use the `jax` environment instead:

```bash
pixi install -e jax          # osx-64 (also available on arm64)
pixi run -e jax test
```

Keras 3 is multi-backend and each environment exports `KERAS_BACKEND` on
activation, so the same code trains on either. The split is forced by upstream
wheels rather than preference: TensorFlow ships no x86-macOS wheel past 2.16.2,
and that wheel requires `numpy<2.0` while `da4ml` requires `numpy>=2`, so the
TensorFlow backend cannot be installed on Intel at any version. JAX is the only
backend with a usable x86-macOS wheel (`jaxlib` caps at 0.4.38 there). The
`jax` environment is available on arm64 too, for backend cross-checks.

Note that the arm64-only environments (`default`, `tune`) can only be
*re-locked* from an arm64 machine, because pixi builds the git-sourced QAT
stack's metadata with an interpreter for the current platform.

On a **linux + NVIDIA GPU** node (Elastic Analysis Facility):

```bash
cd eaf && pixi install       # linux-64, CUDA 12 (see eaf/pixi.toml)
pixi run train ...           # same CLI, GPU backend
```

Optional: `pixi install -e tune` adds `ray[tune]` for hyperparameter scans
(`jax-tune` for the JAX equivalent).
All runs log to **mlflow** (`pixi run mlflow-ui` to browse `./mlruns`).

## 2. Input data: producing L1*Nano

The nano tiers are defined on the `l1nano-smartpixels-master` branch of
[NJManganelli/cmssw](https://github.com/NJManganelli/cmssw) (packages
`DPGAnalysis/Phase2L1TNanoAOD` + `PhysicsTools/NanoAOD/autoNANO.py`):

```bash
cmsrel CMSSW_20_1_0_pre1 && cd CMSSW_20_1_0_pre1/src && cmsenv
git cms-init --upstream-only -q -y
git cms-addpkg DPGAnalysis/Phase2L1TNanoAOD PhysicsTools/NanoAOD
git remote add njmanganelli-fork https://github.com/NJManganelli/cmssw.git
git fetch njmanganelli-fork l1nano-smartpixels-master
git checkout l1nano-smartpixels-master
scram b -j8
# then, in a workflow that runs the Phase-2 L1 emulation (GEN-SIM-DIGI-RAW input):
cmsDriver.py ... -s ...,NANO:@L1PFTrkNanowithGen --datatier NANOAOD ...
```

Flavors (`autoNANO.py`): `@L1TrkNano` (track tables), `@L1PFNano`
(PF/Puppi candidates + jet-constituent links), `@L1PFTrkNano` (both);
each with a `withGen` variant that adds gen objects **and the MC truth**
(track `genuine/fake` labels, TrackingParticle info, displaced vertices) —
**use `withGen` for anything you want to train on**. The truth associators
and the `DisplacedVertexProducer` are scheduled automatically if the
upstream workflow didn't run them.

Key tables: `L1puppiJetSC4NG` (tagged jets), `L1SC4NGJetCands`
(jet↔constituent links: `jetIdx`/`candIdx`/`slot`/`inTagger`),
`L1ExtPuppiCand`/`L1PuppiCand` (candidates with `l1TrackIdx` /
`hgcClusterIdx` / `jetIdx` crossrefs), `L1TTrack`/`L1TExtTrack` (tracks:
floats + raw hardware track-word bits + truth), `L1HGCCluster`,
`L1Vertex`, `L1DispVertex`, `GenJet`/`GenVisTau`/`GenPart`/`GenVtx`.

Vertexing algorithm in production is chosen in the **L1 step** (the nano
just tables the result): `l1tVertexFinderEmulator.VertexReconstruction.
Algorithm = "fastHistoEmulation"` (default) or `"NNEmulation"` (E2E; models
via `TrackWeightGraph`/`PatternRecGraph`), association via
`l1tTrackVertexAssociationProducer.useAssociationNetwork`/`associationGraph`.

## 3. Reading nano: the coffea schema

```python
from coffea.nanoevents import NanoEventsFactory
from ngtagger.coffea_schema import L1NanoSchema, jet_constituents

events = NanoEventsFactory.from_root({f: "Events"}, schemaclass=L1NanoSchema).events()

links = events.L1SC4NGJetCands       # jet-constituent association table
links.jet.pt, links.cand.pt          # crossrefs
links.pt_rel, links.deta, links.dphi # derived tagger variables, on demand
nested = jet_constituents(events)    # (event, jet, constituent) grouping

cands = events.L1ExtPuppiCand
cands.matched_track, cands.matched_cluster, cands.matched_jet
cands.trk_genuine                    # truth: direct column OR track indirection

dv = events.L1DispVertex
dv.first_track.pt, dv.second_track.pt
```

One schema serves every tier: a **partially** present collection with
missing crossref targets triggers a `RuntimeWarning`; a fully absent
collection is silent.

**Hardware fields stay raw; floats are decoded on demand.** The `hw*`
columns are exactly the bits the firmware sees; the schema adds lazy
properties implementing the `TTTrack_TrackWord` conversions (nothing is
read or computed until you access them, with the lazy/virtual backend
loading only the needed branch):

```python
trk = events.L1TTrack
trk.hwZ0                # raw bits, as stored
trk.z0FromHw            # (two's complement + 0.5) * LSB  -> cm, float64
trk.rinvFromHw, trk.phiFromHw, trk.tanlFromHw, trk.d0FromHw
trk.bendChi2FromHw, trk.chi2RPhiFromHw, trk.chi2RZFromHw   # bin-value lookups
trk.mvaQualityFromHw    # track-quality MVA bin value
```

## 4. Studies using nano variants

Seeding and combinatorics studies that consume the SmartPixels cluster nano
(`L1TSmartPixelsCluster`) and the all-stub table (`L1TOTStub`). They live in
`eval_refitq/combinatorics/` and are independent of training.

**Every tool takes one or many inputs**: `-i` accepts comma-separated paths
and/or globs, order preserved. `-n` limits events and is enforced across files
(uproot's own `entry_stop` is silently ignored when it is handed a file list, so
relying on it would read every event).

### Samples

Produced into `cmssw/work/otstub_arm/`, all with OT stub truth links and the
reco-angle sigma lookup:

| set | events | note |
|---|---|---|
| `itot_truth_100ev.root` + `itot_ttbar_f02..f10_100ev.root` | 1000 | ttbar PU200, 1000 **distinct** events |
| `itot_htt_100ev.root` | 100 | VBF H->tautau PU200 |
| `itot_nopu_100ev.root` | 100 | ttbar noPU, ~45x fewer clusters |

`itot_truth_40ev.root` and `itot_pu200_60ev.root` are **superseded**: they
predate the sigma fix and are nested (40 is a subset of 60 is a subset of the
100-event file1), so summing them double counts.

Keep the three processes separate. PU200 ttbar and PU200 H->tautau have nearly
identical occupancy because pileup dominates the cluster count, but different
heavy-flavour content; noPU is the only independent lever on how cost scales
with occupancy.

### `validate_samples.py` — are these files what they claim?

```bash
python eval_refitq/combinatorics/validate_samples.py \
  -i "../cmssw/work/otstub_arm/itot_truth_100ev.root,../cmssw/work/otstub_arm/itot_ttbar_f*_100ev.root"
```

Checks column completeness (a study that silently loses a column reads as a
physics result), event distinctness by `(run, lumi, event)`, and occupancy
consistency. Run it first on any new production. It reports summed versus
distinct events and names any subset relationship outright.

### `seeder_design_basis.py` — the measurements the design constants rest on

```bash
python eval_refitq/combinatorics/seeder_design_basis.py -i "<inputs>" \
  --payload ../cmssw/work/spxsmoke/spix_angle_response_Conv1D_Full-2bit_v4fixed.json \
  [--study geometry --study firmware-rom] -o design_basis.json
```

Six studies, each naming the constant it justifies. `geometry` (sets
`N_RBINS=48`), `sigma-z0` (why the wildcard split is per layer), `selectivity`
(why cluster pairs are formed on z0 and phi jointly), `pt-threshold` (how a d0
allowance destroys the pT floor), `firmware-rom` (whether the three-point solve
can use layer-constant coefficients), `sigma-provenance` (whether the published
angle sigma is an affordable lookup and whether it is keyed on reco or true
angles). Default runs all.

### `projection_residuals.py` — why the third layer is searched on z, not phi

```bash
python eval_refitq/combinatorics/projection_residuals.py -i "<inputs>" [--ptmin 2.0]
```

Takes the one correct cluster triple per TrackingParticle and measures the
residual of the pair's projection against the true third cluster, binned in the
TP's own d0. A resolution measurement, no combinatorics.

### `triplet_d0_solve.py` — can three clusters determine (phi0, d0, kappa)?

```bash
python eval_refitq/combinatorics/triplet_d0_solve.py -i "<inputs>" [--ptmin 2.0]
```

Validates the exact three-point solve against truth and compares its curvature
resolution with the pair's. Prints **median bias per d0 band**, which is the
statistic that decides whether a calibration is needed -- the regression slope
of fit against true is not, because 95% of TrackingParticles sit below
d0 = 100 um where there is no lever on the scale.

### `tracklet_topology_cost.py` — the seeding cost model

```bash
python eval_refitq/combinatorics/tracklet_topology_cost.py -i "<inputs>" \
  --batch-events 8 -o cost.json
python eval_refitq/combinatorics/tracklet_topology_cost.py -i "<one file>" \
  -n 2 --selftest        # must print "0 mismatched" before trusting any output
```

IT seeding (cluster pairs, projection, arbitration) against the real OT tracklet
topology and match windows. `--selftest` compares the z-binned projection search
and the joint (z0, phi) pairing against an exhaustive reference; run it after any
change to the search.

**Memory**: `--pair-budget-gb` (default 1.5) is the knob, not `--batch-events`.
Peak memory is set by candidate triplets held at once, and a d0-inflated window
can explode inside a *single* event -- which no batch size can bound.
`--rss-ceiling-gb` is hard-clamped to 40% of physical RAM.

### `run_all_studies.sh` — everything, per process

```bash
bash eval_refitq/combinatorics/run_all_studies.sh        # -> eval_refitq/combinatorics/results/
OUT=/tmp/myrun bash eval_refitq/combinatorics/run_all_studies.sh
```

Runs all five tools over ttbar PU200, H->tautau PU200 and ttbar noPU separately,
writing one text and json output per tool per process.

## 5. Training

### NG jet tagger

```bash
# single training
pixi run train -c configs/deepset_hgq2.yaml -i nano*.root -o output/baseline

# multi-shot: N seeds in parallel, best-of (QAT local-minima mitigation)
pixi run multishot -c configs/deepset_hgq2.yaml -i nano*.root -o output/ms -n 5 -p 2

# contrastive: SimCLR embedding pre-train -> quantized fine-tune
pixi run train -c configs/deepset_contrastive.yaml -i nano*.root -o output/contrastive
```

Constituent features are **composable groups** in `data_config.feature_groups`:
`baseline` (upstream hardware inputs) + `track` (track-word floats via
`l1TrackIdx`) + `trkquality` (the BDT score, separate option) + `cluster`
(HGCal shapes). Labels (b/c/uds/g/τ+/τ−/μ/e) and pt-regression targets are
built in-pipeline from gen matching. Parallel shots default to 2 on an
M4 Pro (24 GB) and 1 on a GPU partition; override with `-p`.

### Track quality / vertexing / displaced vertices

```bash
pixi run python -m ngtagger.cli train-trkquality -i nano*.root -o output/trkq --conifer
pixi run python -m ngtagger.cli train-nnvtx      -i nano*.root -o output/nnvtx \
      --extra-features log_pt nlaymiss_interior     # optional new inputs
pixi run python -m ngtagger.cli train-dispvtx    -i nano*.root -o output/dv --conifer
```

Every pipeline computes **stock-vs-retrained comparisons** against the
scores already in the file (`compare_vertex_scores`,
`compare_dispvtx_scores`, AUCs vs truth) and logs them to mlflow.

## 6. Firmware / CMSSW deployment

```bash
pixi run export -m output/ms/best -o firmware/            # hls4ml Vitis + da4ml
pixi run export -m output/ms/best -o firmware/ --build    # + HLS synthesis
pixi run export -m output/ms/best --emulator-repo ../L1TSC4NGJetModel --version-tag v2
```

GBDTs (`--conifer`) export the conifer json formats CMSSW loads directly
(`TrackQuality_params.Model`, `DisplacedVertexProducer.model`).

**Deployment-parity caveats** (read before replacing a shipped model):
the deployed models operate in *digitized* feature space — track-quality
uses signed track-word integers with base shifts (this pipeline decodes
two's complement; the base shifts are LSB drops), the DV tagger is the
"Shifted13p8" `ap_fixed<13,8>` convention, and NNVtx graphs take GTT-word
inputs (frozen-graph export is stubbed pending that mode). Working points
downstream (e.g. `tqMVABins`) need revalidation after any retrain.

## 7. Tests

```bash
pixi run test                                        # full synthetic suite
NGTAGGER_TEST_NANO=/path/nano.root pixi run test     # + real-file pipeline test
```

Tests are synthetic-data based (no grid access needed): they generate
NanoAOD-shaped files with uproot and verify the schema crossrefs, the
warning semantics, hw-decode conventions, fastHisto reproduction of the
stock vertex, and that each training converges on separable toys.
