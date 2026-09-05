# Stage-4 calibration: training statistics needed to beat the embedded SC4 NG tagger

Two linked studies calibrating what training statistics the SmartPixels jet-tagger
program needs to demonstrate ABSOLUTE gains over the embedded (centrally-trained
hls4ml) SC4 NG tagger. Host-side pixi env only; inputs are the unified coherent nanos
(`nano_fat_{1111_coopt,1100_coopt,0000_baseline}_file{1..10}.root`, ~100 events/file)
and the Stage-4 artifacts.

- Scripts: `scripts/00..05_*.py` (Study 1: 01+03; Study 2: 02+04; plots: 05)
- Machine-readable results: `embedded_eval.json`, `paired_gaps.json`,
  `learning_curve.json`, `calibration_summary.json`
- Plots: `plots/learning_curve.png`, `plots/gain_vs_N.png`,
  `plots/per_class_embedded_vs_scratch.png`
- Per-jet aligned dumps: `dumps/embedded_<view>.npz`, `dumps/lc_*.npz`

---

## Study 1 — embedded NG tagger AUC on OUR labeled jets (the plateau anchor)

### Alignment: exact, index-level (no matching needed)

The embedded scores are branches of the **same jet table the training pipeline
reads** (`L1puppiJetSC4NG_{b,c,uds,g,tau_p,tau_n,mu,e}TagScore`; `load_jets`
already pulls them via its `*TagScore*` glob). The gen-match `keep` mask and the
train/test permutation (`prepare_dataset`, `default_rng(12345)`, test_fraction
0.2 — from `run_matrix.run_cell`) were replicated row-for-row and **verified
against the Stage-4 prediction dumps: `kin_pt` and `y_true` match exactly for
all three views (1314 / 1320 / 1328 test jets)**. Match rate is therefore 100%
by construction; no deltaR fallback was needed.

Class mapping (verified empirically): `bTagScore→b, cTagScore→charm,
udsTagScore→light, gTagScore→gluon, tau_pTagScore→taup, tau_nTagScore→taum,
muTagScore→muon, eTagScore→electron`. Score sums: 0.946 ± 0.247 (quantized
softmax, max 1.06–1.13). **6.4–6.8% of labeled jets carry all-zero embedded
scores** (median pT ~11–15 GeV — below the central tagger's scoring threshold;
mostly light/gluon/b). They are kept: this *is* the central operating point.
Excluding them would raise the embedded macro AUC by ~0.02 (0.7376 → 0.7577 for
0000 all-jets).

### Embedded macro OvR AUC (8 classes)

| view | all labeled jets (~6.6k) | Stage-4 test subset |
|---|---|---|
| 1111 | 0.7311 ± 0.0074 | 0.7429 ± 0.0181 (n=1314) |
| 1100 | 0.7356 ± 0.0069 | 0.7575 ± 0.0141 (n=1320) |
| 0000 | **0.7376 ± 0.0075** | **0.7469 ± 0.0143** (n=1328) |

(± = bootstrap over jets, 1000 resamples. Per-class AUCs with Hanley–McNeil SEs
in `embedded_eval.json` / `paired_gaps.json`.)

### Three-way physics readout

1. **The absolute bar (embedded@0000)** = 0.7469 ± 0.0143 on the 0000 test
   split; 0.7376 ± 0.0075 on all 6643 labeled 0000 jets (best precision).
2. **Plateau gap, gap0 = embedded@0000 − scratch@0000** (same 1328 test jets,
   paired bootstrap): **+0.0610 ± 0.0166** (best-of-3-seeds scratch 0.6860).
   For 1100: +0.0447 ± 0.0147; for 1111: +0.0133 ± 0.0197 (each vs the embedded
   scores in its own view).
3. **embedded@1111 vs embedded@0000, event-matched jets** (6329 matched of
   6574/6643, match rate 96.3%/95.3%, deltaR<0.2 within same event; 3 ambiguous
   multi-matches; label agreement 99.9%; paired bootstrap on the 6322
   label-agreeing pairs): **Δmacro = −0.0083 ± 0.0034**. The refit-driven
   coherent reco does **NOT** lift the frozen central model — it *hurts* it
   slightly (2.4σ). Per class: b **+0.0136** and gluon **+0.0168** improve;
   taum −0.042, electron −0.030, taup −0.023 degrade (rare-class HM errors are
   large). Zeroth-order verdict: **the absolute gain of the coherent view is
   not free — it must be *learned*; the frozen central weights are mis-calibrated
   for the refit inputs (while the b/gluon separation genuinely improves).**
4. **Today's distance to the bar: scratch@1111 (0.7296) vs embedded@0000
   (0.7469)** = −0.0173 ± 0.0212 (unpaired, different test sets, quadrature) —
   already **statistically consistent with the bar** (0.8σ), but not a
   demonstrated crossing. Vs the all-jets bar (0.7376) the distance is only
   −0.0080.

Per-class picture (`plots/per_class_embedded_vs_scratch.png`): our scratch
models already **beat the embedded tagger on gluon** (0000: 0.756 vs 0.636)
and match it on light/charm; the macro deficit is carried almost entirely by
the **rare classes** (taup/taum/muon, 10–45 test jets each) where the central
tagger's much larger training sample dominates.

---

## Study 2 — learning curve from training-set subsampling (the exponent)

### Protocol

Views 1111 and 0000, baseline feature group, stage-4 model + fit protocol
(imported from `run_matrix`: same Conv1D-DeepSet, Adam 1e-3, batch 256,
val_split 0.15, EarlyStopping patience 8). Fractions {1/16, 1/8, 1/4, 1/2, 1}
× seeds {1,2,3}, **stratified by class** (≥1 jet/class), epoch budget
min(320, 40/frac) so small-N runs get comparable optimizer steps; at frac=1 the
protocol is *exactly* stage-4 and reproduces the anchor (seed 1: 0.7296).
**Fixed full Stage-4 test split for every point.** Dumps under
`calibration/dumps/` (stage4/pred_dumps untouched).

Macro AUC (mean ± std over 3 seeds):

| N_train (1111) | 1111 | N_train (0000) | 0000 |
|---|---|---|---|
| 329 | 0.575 ± 0.012 | 334 | 0.544 ± 0.054 |
| 657 | 0.660 ± 0.035 | 664 | 0.625 ± 0.070 |
| 1316 | 0.702 ± 0.001 | 1328 | 0.669 ± 0.021 |
| 2630 | 0.717 ± 0.008 | 2658 | 0.692 ± 0.006 |
| 5260 | 0.721 ± 0.007 | 5315 | 0.680 ± 0.006 |

Per-class train counts at 1/16 (1111): b 90, charm 39, light 97, gluon 76,
taup 4, taum 3, muon 10, electron 10 — the taus are nearly empty at the lowest
point, so its macro AUC is dominated by rare-class noise.

### Fitted exponent

err(N) = err_inf + a·N^(−b), err = 1 − macro AUC, fit on all 15 seed-points
per view; uncertainty = bootstrap over seeds (2000 refits):

| view | b (with floor) | AUC_inf | b (floorless) | joint shared-b |
|---|---|---|---|---|
| 1111 | 1.21 ± 0.29 (q16–84: 0.84–1.52) | 0.728 (q16–84: 0.723–0.744) | 0.164 ± 0.010 | 1.23 |
| 0000 | 1.24 ± 0.43 (q16–84: 0.62–1.53) | 0.694 (q16–84: 0.691–0.729) | 0.145 ± 0.030 | — |

b-class-only fits: b = 0.23 (1111) / 0.85 (0000) — poorly constrained, quoted
for completeness.

**Honesty about degeneracy:** the two models bracket the truth and disagree
wildly. With five descending points over a 16× range, the floor `err_inf` and
the exponent trade off almost freely: the with-floor fit (RMS 0.017) reads the
1316→5260 flattening as an asymptote *just below the bar*; the floorless fit
(RMS 0.026, visibly poor at high N) says progress continues forever with b≈0.16.
The steep b≈1.2 is *not* a believable universal learning exponent (typical
NN curves show b ≈ 0.1–0.5); it is what a floor+power-law does to a curve that
is saturating in-range. Treat the floor fit as the **pessimistic** and the
floorless fit as the **optimistic** envelope.

### gain(N) = AUC_1111(N) − AUC_0000(N), paired by seed

| N | gain ± sem |
|---|---|
| ~330 | +0.031 ± 0.037 |
| ~660 | +0.035 ± 0.025 |
| ~1320 | +0.033 ± 0.015 |
| ~2630 | +0.025 ± 0.002 |
| ~5260 | +0.041 ± 0.003 |

**The refit gain is remarkably stable (~+0.03) across a 16× range in N, and if
anything grows at full statistics (+0.041 ± 0.003).** It does not shrink as the
models get better — the coherent-view information is complementary, not a
small-N crutch. This is the strongest evidence that the gain will survive at
higher statistics, and it is what makes any extrapolation worth quoting at all.

### N_cross: where does scratch@1111 exceed the embedded@0000 bar?

Conversions use measured rates: 6.57 labeled jets/event (1111), 80% to train →
5.26 train jets/event; class fractions b/charm/light/gluon/taup/taum/mu/e =
27.2/11.4/30.2/23.1/1.2/1.0/3.2/2.9%.

| fit | bar | N_cross (train jets) | events | notes |
|---|---|---|---|---|
| with floor | 0.7469 (test) | central: **never** (AUC_inf 0.728 < bar); boot median 25.7k, q16–84 [13.1k, 74.5k] | ~4,900 [2,500–14,200] | only **12%** of bootstrap fits reach the bar at all |
| with floor | 0.7376 (all-jets) | median 12.4k [7.7k, 33.8k] | ~2,360 [1,460–6,420] | 24% reach |
| with floor | bar − 1σ (0.7326) | median 8.9k [5.9k, 23.7k] | ~1,700 | 35% reach |
| floorless | 0.7469 (test) | 5.55k [5.1k, 6.0k] | ~1,060 | i.e. "already there" — too optimistic |

Per-class train jets at the floor-fit median (25.7k): b ~6,980, charm ~2,910,
light ~7,740, gluon ~5,910, taup ~300, taum ~250, muon ~830, electron ~740.

**Plain-language readout:** the extrapolation is honest only as a range. The
optimistic envelope says we cross at ≈1× current statistics (consistent with
scratch@1111 already being within 1σ of the bar); the pessimistic envelope says
the baseline-feature stage-4 architecture *saturates ~0.02 below the bar* and no
amount of data crosses it, with the reachable tail of that fit at 2.5–14× current
statistics. **A crossover claim from extrapolation alone is beyond the trustworthy
range of a descending 5-point curve; 25.7k train jets (~5k events/view, ≈5×
current) is the defensible planning number** — it is where the pessimistic fit's
reachable half crosses, and it simultaneously shrinks the test-side error bar
(±0.014 → ±0.006 with a 5× test split) enough to resolve the expected +0.03–0.04
gain at >3σ paired.

---

## Recommendation

- **Plan ~5× current statistics (≈5,000 events/view, ~26k labeled train jets)**
  as the milestone dataset; ~15× (≈15k events/view) buys the q84 of the
  pessimistic fit and ~50 τ⁺/τ⁻ test jets per class.
- The decisive comparison should be **paired on identical jets** (embedded vs
  scratch scores on the same test rows, as in `paired_gaps.json`) — paired
  bootstrap errors are ~half the unpaired ones.
- Rare classes gate the macro metric: at current stats τ⁺/τ⁻ have 10–13 test
  jets. Either report macro-over-6-classes alongside, or enrich taus.

## Caveats (honest list)

1. **Descending-curve extrapolation:** floor/exponent degeneracy is severe; b is
   only constrained to [0.6, 1.5] (floor) vs 0.15 (floorless). N_cross spans
   "already there" to "never". We quote the envelope, not a point.
2. **Bar uncertainty:** the embedded@0000 bar itself carries ±0.014 (test) /
   ±0.0075 (all jets); at current test-set size a real +0.02 absolute gain is
   ~1σ. Growing the *test* set matters as much as the train set.
3. **Tau statistics:** taup/taum have ~10 test jets and 40–60 train jets at
   full statistics; their OvR AUCs (and hence macro) fluctuate by ±0.05–0.1.
4. **Zero-score jets:** 6.5% of labeled jets get no embedded score (low pT).
   Kept as-is (it is the central operating point); excluding them raises the
   bar by ~0.02 — the crossover statement is robust in direction either way,
   but quote which convention is used.
5. **Single-coherent-view nanos, ~1000 events/view**, one gen sample; the
   embedded@1111 degradation (−0.0083 ± 0.0034) is a 2.4σ effect on one sample.
6. **Protocol freeze:** learning-curve points reuse the stage-4 architecture
   and epochs=40-at-full-N budget (mean seed AUC at N=5260 is 0.721, i.e. the
   anchor's best-of-3 0.7296 is a favorable draw); a better-tuned model would
   shift the whole curve up and shrink N_cross.
