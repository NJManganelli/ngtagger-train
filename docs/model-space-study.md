# Model-space exploration: refit track-quality MVA + NG jet tagger

Status: v1 (2026-07-19). Companion artifacts: `eval_refitq/modelspace/`
(study scripts + JSON results referenced throughout). Contract references:
`L1Trigger/Phase3SmartPixels/doc/RefitSidecarSpec.md` v0.2 (spec §numbers below).

Hard framing constraint applied throughout: **L1 trigger hardware budgets**
(latency O(100 ns)–O(1 µs), FPGA LUT/DSP/BRAM, HGQ2/da4ml quantization,
conifer tree-depth/count limits). Where a direction exceeds plausible budgets
it is explicitly labeled an offline / upper-bound study.

## 0. Data, statistics, and method

Single study sample: `nano_pu100_TrkSmartPix_withGen.root` — 100 PU events,
17 324 reference tracks, **16 845 genuine / 479 fake**, four digiRefit
layer-config variants (AIII/AAII/AAAI/AAAA) with per-track extension columns
and per-hit link tables. Verified structural facts:

- exactly **one crossing per (track, layer)** and at most one accepted hit
  per layer → per-layer features are well-defined without aggregation;
- the chi2 columns are **pre-guard** (spec §6b): 34 tracks with
  `chi2IncRPhiTot > 2e6` (max 3.3e9), 58 on RZ. Values above ~2e6 are
  numerical-Jacobian pathology, not physics; treated as such below.

Statistical discipline: with 479 fakes the test-split AUC sigma is ~0.011
(`eval_refitq/models/auc_seed_std.json`), which dwarfs every effect of
interest. All comparisons here therefore use **8 shared split seeds and
paired per-seed AUC deltas** (identical splits across feature sets/models);
the honest uncertainty is the paired `delta_std`, typically 0.003–0.006 —
3–4× smaller than the naive split spread. "improved N/8" counts seeds where
the delta is positive.

---

## Part A — refit track-quality MVA

### A.1 Feature critique of REFIT_BDT_FEATURES v0 (spec §6a)

#### A.1.1 The chi2/pull weighting question

The spec's `sumPull*2` and `chi2Inc*Tot` features ARE heteroscedastically
weighted per hit — each scalar-update pull is r/sqrt(S) with S carrying the
hit sigma and the propagated track covariance. What the sums do NOT encode is
the **layer identity** of the evidence, and the layers are measurably not
exchangeable:

- extrapolation q68 grows 0.041 → 0.899 cm from L1 → L4 (2–5 GeV, from the
  Tier-2 producer studies), so wrong-hit contamination grows outward while the
  innermost layer stays golden;
- hit-level single-variable fake-AUC of |pull| (accepted hits,
  `probe_hits.py`): **pullBeta L1 = 0.735**, falling to 0.628/0.638/0.614 at
  L2/L3/L4; pullAlpha 0.641 at L1 falling outward; pullX is *anti*-separating
  (0.43–0.53: best-chi2 selection in dense windows cherry-picks consistent
  x positions for fakes — the known inverted-kick phenomenology);
- mutual information of |pull| with the label reproduces the same hierarchy
  (pullBeta_L1 = 0.0083, the largest of all 16 per-layer pulls, monotone
  falling outward).

So the per-layer structure is REAL at hit level. However, at track level
(`perlayer_study.py`, AAAA and AIII):

| feature set (vs spec17, paired) | AAAA delta | AIII delta | improved |
|---|---|---|---|
| + per-layer pulls (16 cols) | −0.0022 ± 0.0039 | −0.0015 ± 0.0025 | 2/8, 3/8 |
| + per-layer occupancy | −0.0001 ± 0.0019 | +0.0003 ± 0.0007 | 3/8, 5/8 |
| + per-layer chi2 | −0.0025 ± 0.0034 | +0.0004 ± 0.0006 | 2/8, 5/8 |
| + everything per-layer | −0.0023 ± 0.0042 | −0.0020 ± 0.0027 | 2/8, 1/8 |
| per-layer REPLACING pooled sums | −0.0016 ± 0.0038 | −0.0019 ± 0.0023 | 3/8, 1/8 |
| **+ classic 7 TRKQ hw features** | **+0.0144 ± 0.0033** | **+0.0166 ± 0.0031** | **8/8, 8/8** |

A logistic regression allowed to learn optimal per-layer weights on the pull²
blocks produces noisy, sign-flipping weights — the 479-fake sample cannot
constrain 16 weights. **Verdict: the pooled sums are NOT the binding
limitation at current statistics.** The per-layer decomposition is a
physics-motivated v1+ candidate that must wait for a larger fake sample
(A.3); adopting it now would be fitting noise.

#### A.1.2 The one evidenced spec change: add the classic TRKQ features (v1)

The single robust feature finding: `seedTrkMVA1` (feature 16) is a lossy
scalar compression of the classic 7 track-word features (tanl, z0_scaled,
bendchi2_bin, nstub, nlaymiss_interior, chi2rphi_bin, chi2rz_bin). Adding the
raw 7 to the refit BDT gains **+0.014–0.017 AUC, 8/8 seeds in both configs**
— by far the largest effect observed in this study, larger than every
refit-feature ablation and every architecture change. This also subsumes the
"hitPattern × layerHitMask cross terms" idea: `nlaymiss_interior` +
`layerHitMask` in one tree model provides exactly that cross information.

Hardware cost: **zero new information crosses any boundary** — all 7 are
fields of the input track word the producer already holds; conifer input
width grows 17 → 24. Requires: spec §6a version bump (REFIT_BDT_FEATURES v1,
24 ordered features), the producer feature-vector block extension, and a
retrain/re-export. Effort: **S**. This is the top Part-A recommendation.

#### A.1.3 Missing-feature candidates — evaluated or dispositioned

| candidate | status | disposition |
|---|---|---|
| per-layer residual/pull vectors | **tested** | null at track level now (A.1.1); revisit at 10–50× fakes |
| per-layer window occupancy | **tested** | null (window mult is weakly *anti*-separating per layer: fake mean 3.4–5.2 vs genuine 3.8–5.7) |
| selChi2 selection margin (best vs runner-up) | **not derivable offline** — only the selected hit is persisted | needs producer change: add `selChi2Margin` per crossing to the sidecar + one feature. Physically well-motivated (margin ≈ selection confidence in dense windows; directly attacks the inverted-kick cherry-picking). Flagged for spec v1+; effort S (producer) once a bigger sample can measure it |
| angle-consistency between layers (Δ cotAlphaMeas L_i−L_j vs helix expectation) | derivable offline from `cotAlphaMeas` per layer | untested here (needs care with the per-layer validity pattern); queue for the larger-sample per-layer pass |
| seed-quality interactions (seedTrkMVA1 × refit) | trees model interactions natively | superseded by A.1.2 (give the trees the raw classic features instead) |
| hitPattern × layerHitMask cross terms | **covered** | the +tierA block contains it; adopted via A.1.2 |

#### A.1.4 chi2 numerical tail and the §6b guards

`spec17_guard` (chi2 features log1p-clipped at 2e6) is AUC-identical to raw
spec17 (delta +0.000 ± 0.0004): monotone tree splits are insensitive to a
monotone tail. The §6b guards therefore need no ML justification — they
remain motivated by hardware fidelity (finite-precision arithmetic bounds
these quantities structurally) and by keeping the compact-word quantizer
`q(c)` from saturating on garbage. No offline transform is needed for BDTs;
NN consumers (A.2) DO need the clip (raw 1e9 values destroy input scaling).

#### A.1.5 Transmitted-subset reality check (spec §3)

`transmitted_subset_study.py` measures, at track level, what a downstream
consumer BDT recovers at each transmission tier (scorer = spec17 BDT trained
on the train split only; quantizers reproduce spec §3 bit-exactly):

| tier | content | extra bits | AUC (mean ± split std) | paired delta vs TS0 |
|---|---|---|---|---|
| score alone | refit BDT score as discriminant | 0 | 0.9333 ± 0.0118 | (+0.0175)* |
| TS0 | consumer BDT on {score, seedTrkMVA1} | 0 | 0.9158 ± 0.0169 | — |
| TS1 | + compact 16-bit word (mask/qchi2/occ) | 16 | 0.9175 ± 0.0178 | +0.0017 ± 0.0057 |
| TS1b | + per-layer 2-bit occupancy (proposed) | +8 | 0.9173 ± 0.0176 | +0.0015 ± 0.0056 |
| TS2 | + full spec17 floats (upper bound) | ~544 | 0.9176 ± 0.0167 | +0.0018 ± 0.0035 |

(*the "score alone" line beats the TS0 consumer because retraining a coarse
80×3 BDT on 2 inputs re-bins an already-optimal discriminant — an artifact of
consumer capacity, not extra information. It is the right baseline for "use
the score directly".)

**Conclusions:** (1) For the genuine-vs-fake axis, the score is already an
(effectively) sufficient statistic — TS1 and even TS2 add nothing resolvable.
Bit-for-bit, the score field is worth more than the entire compact word. (2)
The proposed TS1b per-layer-occupancy variant buys nothing; do not spend the
reserved bits on it. (3) Ordering of feature value per bit for the compact
word, from this + the feature study: layerHitMask (4b) ≥ q(chi2RZ) ≥
q(chi2RPhi) > occ ≥ anything per-layer. (4) IMPORTANT SCOPE LIMIT: this
answers the *fake-rejection* information question only. The jet-tagger
question (Part B.1) involves *flavor* information (e.g. signed d0 kicks on
genuine displaced tracks) that a genuine-vs-fake score is not trained to
encode — the TS ranking can differ there, which is exactly why the Part-B
study matrix is still needed.

### A.2 Model class (`model_class_study.py`, AAAA, paired vs xgb spec17 60–80×3)

| model | features | AUC mean ± split std | paired delta | improved |
|---|---|---|---|---|
| xgb 80×3 (baseline) | spec17 | 0.9400 ± 0.0090 | — | — |
| xgb 80×3 | spec24 (=17+classic7) | 0.9544 ± 0.0081 | +0.0144 ± 0.0033 | 8/8 |
| xgb 200×4 | spec24 | 0.9557 ± 0.0060 | +0.0156 ± 0.0049 | 8/8 |
| xgb 400×6 | spec24 | 0.9541 ± 0.0076 | +0.0141 ± 0.0051 | 8/8 |
| xgb 80×3 bagged ×5 | spec24 | 0.9581 ± 0.0069 | +0.0181 ± 0.0041 | 8/8 |
| MLP (32,32), scaled | spec24 | 0.9286 ± 0.0197 | −0.0114 ± 0.0202 | 3/8 |
| DeepSet over hit set (φ 16-16, pool, ρ 32) | per-hit + globals | 0.9584 ± 0.0077 | +0.0183 ± 0.0082 | 8/8 |

Reading the paired deltas *between* the spec24 rows: deeper/more trees add
+0.001 ± 0.004 over the small tree — nothing. Bagging and the DeepSet each
add ~+0.004 over xgb spec24 with paired spreads of similar size — suggestive,
not resolvable. The MLP underperforms (14k rows, 2.8% minority class — a
known regime where trees win; not an architecture verdict at scale).

**Verdict: features ≫ architecture at these statistics.** The information
added by the classic-7 block is 4× anything any architecture buys.

Hardware mapping:

- **conifer GBDT 80×3, 24 features** (recommendation): same class as the
  deployed GTT TrackQuality GBDT (conifer JSON, FileInPath). Depth-3 trees
  are comparator ladders; latency a few clocks at GTT frequencies, resources
  O(10k) LUT scale — comfortably within the budget the current TQ BDT
  already occupies. Effort S (retrain + re-export via the existing
  `--export-conifer` path with n_features bumped).
- xgb 400×6 → conifer: ~5× trees × 8× leaves ≈ O(40×) resource scale for
  null gain. Rejected.
- bag5 at the bit level: 5× the BDT block for +0.004 unresolved. Rejected
  for hardware; harmless offline as a training-time variance reducer.
- MLP 2×32 HGQ2: input 24 → 32 → 32 → 1 ≈ 2k MACs, quantized well under
  the tagger's dense heads; feasible but currently *worse* — revisit only
  with more data. Effort M (new HGQ2 pipeline + export parity for a
  producer-side NN, which has no current CMSSW evaluation path — the
  producer speaks conifer, so an NN also needs an emulator wrapper).
- **DeepSet over ≤4 hits** (the structurally-right model for the variable
  hit set): φ(16,16) on ~11 per-hit inputs + masked pool + ρ(32) with track
  globals ≈ 3–4k MACs quantized — small; and it runs BEFORE the transmission
  boundary (in-producer, like the BDT), so per-hit inputs are free on-chip.
  This is the growth path IF per-layer information becomes resolvable at
  larger statistics (A.1.1 says: not yet). Effort M–L (HGQ2 training +
  hls4ml export + producer-side evaluation infrastructure).

### A.3 The binding constraint: statistics

Every Part-A conclusion above saturates at 479 fakes. Before the next round
of feature/architecture refinement, produce a **10–50× fake sample**
(PU 200 and/or O(1k–5k) events; posture A `fromFile` on the PU RelVal with
`seedCovMode=parametrized`, as established). The per-layer features, the
selection-margin feature, the DeepSet, and the NN-vs-BDT question all become
measurable at ~5k–25k fakes. Effort M (production + one rerun of these
scripts — they are sample-path-parametrized).

---

## Part B — NG jet tagger

### B.0 Data reality check

The study nano is Trk-flavored: it has **no Puppi candidate tables, no
L1SC4NGJetCands links, no NG jet table** — so no per-candidate refit feature
can be derived from it, and no tagger training can run on it. First
dependency for everything in this part: a combined production
(`@L1PFTrkNano`-withGen + SmartPixels digiRefit tables in ONE nano), which
the WF1 coexist machinery already supports. Effort S–M (production config,
no new code).

### B.1 SmartPixels features through L1PFCandidates

Mechanics (per spec §4): candidates carry `l1TrackIdx`; the glue producer
`l1tSmartPixCandExtraProducer` resolves cand → trackRef → index → sidecar
row (valid row-wise in both coopt and coexist via the 1:1 invariant). In
ngtagger-train, the same resolution is `crossref_gather` on `l1TrackIdx` —
the per-candidate refit features are one feature-group away once the
combined nano exists (the `track`/`trkquality` groups are the template;
add a `refit` group reading the `L1TSmartPixels*Track*` extension columns
and kicks).

Candidate-feature menu (per constituent, charged only; neutrals get
sentinels): refit-quality score; layerHitMask/nAcceptedHits; d0/z0/pt kicks
(signed! the sign×charge correlation carries displacement physics);
chi2IncTot (guard-clipped); maxWindowMult.

**The TS0-vs-TS2 question** ("how much can one score bit-budget proxy for
many feature bits"): the track-level answer (A.1.5) is that for FAKE
REJECTION the score is sufficient. But the tagger consumes tracks for
FLAVOR, and the refit's flavor-relevant content — genuine b/c tracks
acquiring *coherent, signed* d0 kicks from real displaced hits vs light/gluon
tracks acquiring incoherent ones — is a projection the genuine-vs-fake score
never learned. The jet-level study must therefore separate two hypotheses:
(H1) refit info improves the tagger only by cleaning fakes → score
suffices, TS0 wins per bit; (H2) refit info adds displacement information →
raw kicks matter, TS2 > TS0 specifically on b/c vs light and tau ROC.

Proposed study matrix (one combined-nano production, four trainings × N
seeds, paired deltas per class-pair ROC):

1. baseline feature groups (control);
2. + `refitscore` (TS0: one score per charged constituent);
3. + `refitcompact` (TS1: unpacked 16-bit word, quantizers from the shared
   header, reproduced bit-exactly like the trkquality decode);
4. + `refitfull` (TS2: float kicks/pulls/chi2 — upper bound, studies-only).

Predicted ranking: on the all-class average, (2)≈(3)≈(4) > (1) with small
gaps (mostly fake-cleaning, PU 100); on b-vs-light specifically, (4) > (2)
if H2 is real — and if it is, the hardware follow-up is NOT "transmit
floats" but "add a second, displacement-trained score or 2–4 signed-kick
bits to the compact word". Decision metric: paired delta on b-tag ROC AUC
and on tau-vs-QCD, at fixed jet pt. Effort M once the nano exists.

### B.2 Architecture exploration

#### B.2.1 Current and extended I/O layout

Current (DeepSetHGQ2): input (16 const × 20 feat) → φ Conv1D[10,10] →
avg-pool → heads: `jet_id_output` (8 logits, CCE-from-logits) +
`pT_output` (1, LogCosh on ratio clipped [0.3, 2]).

Extended head layout (all sharing the pooled latent; each head is 1–2 small
QDense layers, cost is dominated by the trunk):

```
pooled latent (10–16 dims)
 ├─ jet_id_output      8 logits              (existing)
 ├─ pT_output          1 regression           (existing)
 ├─ charge_output      3 logits {q−, neutral/g, q+} → 2-bit field   (new, B.2.3)
 └─ embed_output       E=8–16 linear nodes, L2-normalized offline    (new, B.2.4)
```

Output bit budget at the boundary: today ≈ 8 scores × ~8b + pt ~10b. The
charge head adds 2 bits. The embedding head is the expensive one (B.2.4).

#### B.2.2 MDMM for multi-objective training

What it is: Platt–Barr Modified Differential Multiplier Method — constrained
optimization `max accuracy s.t. g_i(θ) ≤ ε_i` via Lagrange multipliers with
gradient *ascent* on λ and a damping term; replaces hand-tuned fixed
`loss_weights: [1.0, 1.0]` with constraints that have physical units.

Packaging: the pip `mdmm` package (added to pixi.toml, verified importable)
is **torch-only**. Two integration routes into this Keras-3 trainer:

- Route A (cheap prototype): run Keras 3 with `KERAS_BACKEND=torch` (HGQ2 is
  Keras-3 native, so QAT layers should follow the backend; needs a one-time
  verification that hgq/da4ml paths don't assume TF ops) and use `mdmm.MDMM`
  wrapping per-constraint losses in a custom `train_step`.
- Route B (clean): port MDMM to `keras.ops` (~60 lines: one extra trainable
  λ per constraint, `loss = main + Σ λ_i g_i − damping/2 Σ g_i²`, with λ
  updated by ascent — implementable as a Keras Model subclass with a second
  optimizer, backend-agnostic, mirrors `_SimCLRModel`'s custom train_step
  pattern already in `models/deepset_contrastive.py`).

Physically sensible constraints (each replaces a today-implicit trade-off):

- pt-regression bias: `|mean(pt_pred/pt_true − 1)|` ≤ 1% (optionally per pt
  bin) while maximizing tagger AUC — the exact example use case;
- rate-proxy floors: background efficiency at the working point ≤ target
  while maximizing signal efficiency (differentiable proxy: loss on a fixed
  threshold quantile);
- **EBOPs budget as a constraint**: HGQ2's `beta` is a soft resource-accuracy
  weight scanned by hand today; MDMM can hold `EBOPs ≤ budget` exactly while
  maximizing accuracy — turning the firmware budget into a first-class
  training constraint. This is the most novel payoff and worth a paper-grade
  study;
- charge/aux-head accuracy floors so new heads cannot degrade the main
  tagger (constraint: aux CE ≤ ε instead of a weighted sum).

Effort: M (Route B port + one constrained training reproducing the baseline;
EBOPs-constraint study +M). Dependency: none (mdmm/torch installed; toy data
possible; real data needs the B.0 nano for tagger-level results).

#### B.2.3 Charge classifier head (2–3 bits)

Ground truth: from gen — for jets labeled light by hadronFlavour==0, the
parton charge of the matched `GenJet.partonFlavour` (u +2/3, d −1/3, signed
by particle/antiparticle); gluons and neutral cases central. The nano
already carries `GenJet_partonFlavour` (this file included), so labels are
in-pipeline: 3 classes {negative-parton, neutral/gluon, positive-parton};
taus/leptons excluded or handled by the existing lepton classes (their
charge is already in the 8-class scheme via taup/taum, muon/electron charge
signs in the inputs).

Physics: jet charge `Q_κ = Σ q_i pt_i^κ / (Σ pt_i)^κ`, κ ≈ 0.5, is the
classical engineered observable; at LHC granularities u-vs-d separation is
modest (distributions overlap heavily; single-jet purity gains of order
10–20% over a coin flip are the realistic scale, improving with pt). L1
specifics: 16 constituents, charge known only for tracked constituents
(charged hadrons/leptons carry `charge` in the candidate table — already a
baseline feature as isChargedHadronPlus/Minus etc.), puppi weights help.
Expectation: a learned head should meet or slightly beat engineered `Q_κ`;
the decision output is coarse anyway — quantize to 2 bits
{−, 0, +} (+1 spare state) or 3 bits for a signed-confidence ordinal.
Include `Q_κ` (κ = 0.3/0.5/1.0) as engineered input features AND as the
benchmark the head must beat to justify its bits.

Trigger use: W′/H± charge asymmetries, VBF same-sign topologies, ttbar
charge tagging at GT — cheap bits with real menu value if separation
materializes. Effort: S–M (labels + head + MDMM floor constraint; no new
producer). Dependency: B.0 nano.

#### B.2.4 GloParT-style analyst embeddings

Concept: expose E near-output hidden nodes ("embedding head") trained so a
FROZEN trunk supports analyst-trained downstream classifiers (di-Higgs
bbbb / bbtautau / bbgammagamma discriminants at GT, trained offline on
embedding outputs, deployed as tiny GT-side MLPs or cut tables).

Relation to the existing branches: `DeepSetContrastive` (SimCLR NT-Xent,
mirrors upstream FloatingDeepSetEmbeddingModel/embedding_kv3) already
produces a class-agnostic embedding — but purely-augmentation-contrastive
embeddings optimize invariance, not analyst utility. The GloParT recipe is
better served by **supervised multi-task pretraining**: train trunk +
8-class head + pt head (+ charge head), then designate the last shared layer
(width E) as the embedding, optionally with an auxiliary supervised
-contrastive (SupCon) loss to structure it. The two-stage machinery
(pretrain → freeze → head-tune) already exists in `deepset_contrastive.py`
and transfers directly.

Where the L1 boundary cuts — three designs, increasing cost:

1. **Offline-only embeddings** (effort S): embeddings are a training-time
   artifact; analysts fine-tune heads offline and the WINNING head is then
   distilled/merged into the on-chip model as an extra output at the next
   retrain. No boundary bits; latency unchanged; analyst turnaround =
   retrain cadence, not firmware change. Recommended starting point.
2. **On-chip embedding, GT-side analyst heads** (effort L): transmit E
   quantized dims instead of / alongside the 8 class scores. Bit budget:
   E=8 × 4–6 b = 32–48 bits ≈ the 8-score budget (~64 b) — feasible IF the
   embedding replaces most scores (keep 2–3 primary scores + embedding).
   GT heads are tiny MLPs/LUTs on combined per-jet embeddings — this is the
   real "analysts program the trigger" design and the interesting long-term
   target. Requires: link-format negotiation (out of our control) — frame as
   a design study with the bbtautau case worked end to end.
3. Full GloParT-style wide embedding (E ≥ 32, offline teacher): exceeds any
   plausible boundary budget — offline/upper-bound study only, useful as the
   distillation teacher (B.2.5).

Export-path changes for (1): none beyond an extra Keras output (hls4ml
multi-output conversion already handled by the existing exporter); for (2):
quantizer config on the embedding output (HGQ2 handles it), plus emulator
packaging of the changed output word.

Quality metric for the embedding: frozen-trunk linear-probe AUC on held-out
tasks (bb vs light-light jet pairs; tau-pair mass regression) vs the full
fine-tuned model — the gap measures how much analysts lose to freezing.

#### B.2.5 Other promising directions (surveyed)

- **Knowledge distillation from a big offline teacher** (high promise,
  effort M): train an unquantized, wide teacher (large DeepSet/JEDI-class or
  MLP-Mixer at generous width — no hardware constraint) on the same nano
  labels, then train the HGQ2 student on soft teacher logits + hard labels.
  Consistently worth O(small-but-real) accuracy in the QAT regime upstream;
  fully in-pipeline (one extra config + a `distill` loss option in the
  trainer). No hardware cost by construction. Also directly reusable for
  A.2's producer BDT (teacher NN → student BDT via soft targets).
- **Linformer / MLP-Mixer at larger widths** (upper-bound study, effort M):
  upstream lines exist (LinformerHGQ2, MLPmixerHGQ2; arXiv 2503.03103 shows
  HGQ+da4ml mixers synthesizable at L1-compatible latency). Worth one scan
  to establish the accuracy ceiling vs constituent-count/width on our nano;
  any adoption decision is then a resource negotiation, stated as such.
- **pquant pruning schedules**: WARNING — the PyPI name `pquant` is
  **squatted by an unrelated Chinese quant-trading tool**; do NOT
  `pip install pquant`. The ML pruning+quantization package must come in as
  a git dependency (upstream Keras_v3 sources it that way; candidate repo:
  cern-nextgen/PQuantML). Not added to pixi in this pass — flagged as a
  follow-up once the correct source is pinned by upstream. Structured
  pruning on the φ network is the natural target (constituent-feature
  fan-in dominates EBOPs).
- **Multi-seed ensembling at the bit level**: 2× quantized models + averaged
  logits = 2× trunk resources for the usual +0.002–0.005; only viable if the
  da4ml-optimized trunk lands far under budget. The training-side variant
  (multishot best-of, already implemented) captures most of the benefit for
  free. Low priority.
- **Quantile/binned pt-regression head** (effort S): replace the single
  LogCosh output with a small softmax over pt-ratio bins or a 3-quantile
  head — gives GT a per-jet pt uncertainty at +2–4 output bits; pairs
  naturally with an MDMM bias constraint.

### B.3 SC8 pipeline (implemented)

Implemented in this pass (quick win):

- `src/ngtagger/train/trainer.py`: `prepare_dataset`/`run_training` now take
  the nano table names from `data_config` (`jet_table`, `link_table`,
  `cand_table`, `track_table`, `cluster_table`) plus `gen_match_dr` — the
  pipeline was SC4-hardcoded only through these defaults; no code fork.
- `configs/deepset_hgq2_sc8.yaml`: SC8 config (n_constituents 32,
  gen_match_dr 0.8, tables `L1puppiJetSC8NG`/`L1SC8NGJetCands`, distinct
  firmware project name).
- `tests/test_sc8_pipeline.py`: synthetic SC8-named nano end-to-end test
  (read → group → features → labels) + a config-plumbing test pinning the
  yaml knobs to `prepare_dataset`.

Production-side gap (do-not-edit-cmssw noted): current nanos carry only the
kinematic `L1puppiJetSC8` table (pt/eta/phi/et/mass — verified in the study
nano). An SC8 NG tagger jet table + SC8 jet↔constituent link table
(`L1SC8NGJetCands` naming assumed) require cmssw-side producers (SC8 NG
tagger instance + an SC8 `L1JetCandLinkTableProducer` clone in the L1PFNano
flavors). Until then the SC8 config fails loudly at read time (missing
branches), which is the intended behavior. Effort: S–M on the cmssw side.

Two real bugs found (and fixed) while implementing the test — see Part C.

---

## Part C — framework / install audit

### C.1 Dependency audit (probe: `eval_refitq/modelspace/probe_imports.py`)

| package | status | action |
|---|---|---|
| conifer | **already present** (1.9, imports fine) | none — the "likely absent" flag was stale. Benign startup notice: "Could not import conifer ydf converter" (optional `ydf` not installed; only the upstream Vector_Trees line uses it) |
| mdmm | was missing | **added** to `pixi.toml` + `eaf/pixi.toml` (pypi), verified importable. Note: torch-based — pulls CPU/MPS torch on osx-arm64, CUDA torch on the EAF env (heavy but that env is GPU anyway) |
| pquant | missing | **deliberately NOT added**: PyPI `pquant` is a name-squatted trading tool, not the ML pruning package. Needs a pinned git source (see B.2.5) |
| onnx / xgboost→onnx | missing | not needed: no code path imports onnx; exports go hls4ml (NN) and conifer JSON (BDT) |
| hls4ml / hgq2 / da4ml / xgboost / mlflow / coffea / uproot / awkward | present | all lazy imports in `src/ngtagger` covered |

osx-arm64 solve: clean (mdmm + torch solved without conflict against the
tensorflow/keras pins). Nothing identified that requires falling back to the
eaf/linux env for this study's scope.

### C.2 Framework gaps found while working

1. **`data/nano.py` `gather`/`crossref_gather` were broken for ALL inputs**
   (fixed): they indexed a depth-2 (event, cand) array with a depth-3
   (event, jet, constituent) jagged index, which awkward rejects
   unconditionally. Every `load_jets` call would have crashed on real data.
   Fixed with a flatten/gather/unflatten helper (`_gather2`).
2. **`data/labels.py` `_match` crashed on events with zero match candidates**
   (fixed): (a) the manual `unflatten` pair-grouping placed zero-length
   groups ambiguously (all in the first event) when `other` was empty; (b)
   the `safe_idx = where(idx>=0, idx, 0)` pattern eagerly read index 0 of
   empty per-event collections (any event without a prompt muon — i.e. most
   events). Rewritten with `nested=True` cartesian + option-masked gathers
   (`_take`).
3. Root cause both survived: the only end-to-end data test
   (`test_nano_pipeline`) is gated on `NGTAGGER_TEST_NANO` and evidently
   never ran. The new synthetic `tests/test_sc8_pipeline.py` now covers the
   full read→features→labels path in CI unconditionally. Recommendation:
   add an SC4-named twin (trivial parametrization of the same fixture) so
   the default path is covered too.
4. uproot `how="zip"` groups jagged branches by shared offsets, not by name
   prefix — synthetic fixtures must use pairwise-distinct per-collection
   multiplicities (documented in the test fixture; real NanoAOD is grouped
   correctly). Worth knowing for any future synthetic-nano test.
5. mlflow hygiene (not fixed, report only): `mlflow.db` and stray
   `*.parquet` artifacts sit in the repo root untracked-but-committed-risk;
   suggest `.gitignore` entries and `mlruns/` as the single store.
6. Dataset caching (report only): every training re-reads and re-builds
   tensors from ROOT; multishot re-does it per shot in each subprocess. A
   keyed parquet/npz cache of `prepare_dataset` output (hash of files +
   feature_groups + n_const + table knobs) would cut multishot wall time
   roughly by the read fraction. Effort S; worth doing before the first
   real multishot campaign on large nanos.

---

## Ranked recommendations

| # | recommendation | evidence / payoff | effort | dependencies |
|---|---|---|---|---|
| 1 | **REFIT_BDT_FEATURES v1 = v0 + classic 7 TRKQ features** (spec bump + producer block + retrain) | +0.014–0.017 AUC, 8/8 seeds, both configs; zero boundary bits; largest effect found anywhere in this study | S | spec §6a version bump; producer change (trivial, all inputs on-chip) |
| 2 | **Combined L1PFTrkNano(withGen) + SmartPixels nano production**, then the B.1 four-training TS matrix (baseline / +score / +compact / +full) | unblocks ALL of Part B; decides H1-vs-H2 (fake-cleaning vs displacement info) and the boundary bit spend | M | nano production (WF1 coexist already supports it); no cmssw code |
| 3 | **Bigger-fakes production (PU200 / O(1k) events, 10–50× fakes)** and rerun the modelspace scripts | every Part-A refinement (per-layer features, selChi2 margin, DeepSet-vs-BDT) is stats-blocked at 479 fakes | M | production only; scripts are path-parametrized |
| 4 | **Knowledge distillation option in the trainer** (offline teacher → HGQ2 student soft labels; also teacher-NN → producer-BDT) | reliable QAT gains at zero hardware cost; reusable across taggers | M | none (works on synthetic now, real payoff needs #2) |
| 5 | **MDMM constrained training (Route-B keras.ops port), first target: pt-bias ≤ 1% constraint, second: EBOPs-as-constraint** | replaces hand-tuned loss weights with physical constraints; EBOPs constraint makes the firmware budget a training-time invariant | M | mdmm installed (done); #2 for tagger-level results |
| 6 | Charge head (3-class, 2-bit) + engineered Q_κ benchmark | cheap menu value (charge asymmetries at GT); go/no-go on beating Q_κ | S–M | #2; labels already derivable (GenJet partonFlavour present) |
| 7 | Embedding head, design 1 (offline-only, supervised multi-task + SupCon; frozen-trunk linear-probe metric; bbtautau worked example) | analyst-extensibility path with no boundary cost; groundwork for the GT-side design study | M | #2 |
| 8 | selChi2 selection-margin sidecar field + feature | attacks the inverted-kick cherry-picking directly; well-motivated | S (producer) | needs producer change + #3 to measure |
| 9 | SC8 NG cmssw production side (tagger instance + link-table clone) | training side is DONE and tested (this pass); config fails loudly until tables exist | S–M | cmssw work (out of scope here) |
| 10 | Dataset cache for `prepare_dataset` + mlflow/.gitignore hygiene; SC4 twin of the synthetic end-to-end test | multishot wall-time; CI coverage of the default path | S | none |

Explicitly rejected (with evidence): deeper/wider BDTs (null at 40× resource
scale), per-layer features NOW (null at current stats — revisit under #3),
TS1b per-layer-occupancy bits (null), bit-level model ensembling (cost ≫
unresolved gain), pip `pquant` (wrong package).
