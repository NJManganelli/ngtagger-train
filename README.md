# ngtagger-train

NG jet tagger training pipeline reading **L1PFTrkNano** (the extended Phase-2
L1Nano tier from `DPGAnalysis/Phase2L1TNanoAOD`) directly — no intermediate
data formats. Inspired by, and tracking the improvements of,
[CMS-L1T-Jet-Tagging/TrainTagger](https://github.com/CMS-L1T-Jet-Tagging/TrainTagger)
(`Keras_v3` / `embedding_kv3` lines): HGQ2 quantization-aware training,
da4ml distributed arithmetic, SimCLR contrastive pre-training.

## Environments (pixi)

```bash
pixi install                 # default: osx-arm64 (M-series, tensorflow-metal)
pixi install -e gpu          # linux-64 + CUDA 12 (Elastic Analysis Facility)
pixi install -e tune         # + ray[tune] for hyperparameter scans
```

## Data

Input files are L1PFTrkNano (`cmsDriver ... -s NANO:@L1PFTrkNanowithGen`).
The reader uses the jet-constituent association table
(`L1SC4NGJetCands`: jetIdx/candIdx/slot/inTagger) to rebuild the exact
16-constituent tagger tensor, and the candidate crossrefs
(`l1TrackIdx`, `hgcClusterIdx`) for the `extended` feature set.
Truth labels (b/c/uds/g/tau+/tau-/mu/e) are built in-pipeline from
`GenJet` flavour + `GenVisTau`/`GenPart` deltaR matching.

## Training

```bash
# single shot
pixi run train -c configs/deepset_hgq2.yaml -i /path/to/l1pftrknano*.root -o output/baseline

# multi-shot (N seeds, best-of; QAT local-minima mitigation) with mlflow tracking
pixi run multishot -c configs/deepset_hgq2.yaml -i nano.root -o output/ms -n 5 -p 2

# contrastive (SimCLR embedding pre-train -> quantized fine-tune)
pixi run train -c configs/deepset_contrastive.yaml -i nano.root -o output/contrastive

# watch runs
pixi run mlflow-ui
```

Parallel shots default to 2 on an M4 Pro (24 GB) and 1 on a GPU partition
(A100 20GB MIG); override with `-p`.

## Firmware / emulator export

```bash
pixi run export -m output/ms/best -o firmware/             # hls4ml (Vitis, da4ml strategy)
pixi run export -m output/ms/best -o firmware/ --build     # + HLS synthesis
pixi run export -m output/ms/best --emulator-repo ../L1TSC4NGJetModel --version-tag L1TSC4NGJetModel_v2
```

The emulator packaging drops the HLS project into a
[cms-hls4ml/L1TSC4NGJetModel](https://github.com/cms-hls4ml/L1TSC4NGJetModel)
checkout; build it with hls4mlEmulatorExtras inside a CMSSW area and point
`process.l1tSC4NGJetProducer.l1tSC4NGJetModelPath` at the new version to plug
the trained model back into the L1Nano production.

## Tests

```bash
pixi run test                              # synthetic smoke tests
NGTAGGER_TEST_NANO=/path/nano.root pixi run test   # + real nano pipeline test
```
