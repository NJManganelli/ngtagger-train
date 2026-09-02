# How many bits does a refit chi2 delta need? (v2.6 BDT input design)

Date: 2026-09-01. Scripts: `eval_refitq/quantstudy/{chi2_bitwidth_study,
chi2_code_efficiency}.py`. Results: `eval_refitq/quantstudy/*.json`.
Sample: `clamp_on_f{1,2}.root` — the only POST-guard v2.5 sidecar production
(200 events, PU200 TT, refit config AAAA), 32 799 refit tracks.

## Why v2.5 nano can answer a v2.6 question (validated, not assumed)

In the producer's scalar KF update, `pull[k] = r/sqrt(S)` and
`chi2inc[k] = r*r/S` come from the same `r`, `S`, behind the same
`chi2UpdateGate`. Therefore, exactly:

    chi2Inc<D>Tot  ==  sum over accepted crossings of pull<D>^2

Measured on 32 799 tracks against the stored v2.5 joint columns: median
relative deviation **3.7e-8**, p99 1.3e-7, max 2.4e-7 — float32 storage
precision. Two consequences:

- v2.5 nano yields the v2.6 split totals **exactly**, so no new production was
  needed for this study.
- `chi2IncXTot` and friends are **mathematically identical** to
  `REFIT_BDT_FEATURES` features 5-8 (`sumPullX2`...`sumPullBeta2`), which the
  BDT already consumes at full float. The 4-way split therefore adds no new
  per-track information to the BDT; the open question is purely quantization.

## Dynamic range: the deployed quantizer is mis-sized

Post-guard, per-dimension delta spans (p1 -> p99.9, and median):

| dim | median | p99 | max | octaves |
|---|---|---|---|---|
| X | 15.7 | 2.9e5 | 8.4e5 | **28.2** |
| Y | 7.7 | 5.5e3 | 8.6e3 | 22.3 |
| Alpha | 14.4 | 2.1e4 | 7.7e5 | 23.4 |
| Beta | 10.9 | 7.6e4 | 9.2e5 | 22.7 |

The compact word's `q(c) = clamp(round(2*log2(1+c)), 0, 15)` covers ~7.5
octaves in 4 bits, against 22-28 octaves of actual range. Result for X: **35.5%
of tracks land in the top code**, the 4-bit field carries only 3.33 bits of
entropy, and Spearman vs float is 0.953. At the same 4 bits with one code per
octave (`k=1`): 8.4% saturation, 3.79 bits of entropy, Spearman 0.992.

Note the range is not reducible by normalizing per hit: the reduced
(per-accepted-hit) delta still spans 26.3 octaves.

## The label matters more than the bit width

Two labels, same tracks:

- `genuine` — the deployed objective, a truth property of the **OT seed's**
  match. 31 775 positive / 1 024 negative.
- `clean` — every accepted IT hit came from the track's own TP, i.e. the
  failure mode **the refit itself controls**. 14 871 / 17 928.

**54.7% of refits used at least one wrong hit** (0.82 wrong of 2.72 accepted
per track). The `genuine` label is nearly blind to this, which is why chi2
features have always looked weak:

| feature set added | dAUC on `genuine` | dAUC on `clean` |
|---|---|---|
| float position deltas, over base | +0.0023 [-0.0009, +0.0055] | **+0.2876** [+0.2815, +0.2928] |
| float angle deltas, over position | +0.0011 [-0.0018, +0.0040] | **+0.0203** [+0.0192, +0.0215] |
| original seed chi2, over deltas | **+0.0434** [+0.0371, +0.0509] | +0.0001 [-0.0000, +0.0002] |
| refit deltas, over original seed chi2 | +0.0033 [+0.0009, +0.0060] | **+0.3139** [+0.3080, +0.3186] |

The last two rows are the important ones: the seed chi2 and the refit deltas are
**complementary, not redundant**. Seed chi2 carries the fake-rejection power;
the refit deltas carry the wrong-hit power. The BDT needs both, in all
scenarios. Single-feature AUC for `clean`: Beta 0.879, Y 0.768, X 0.753,
Alpha 0.706, all-four-summed 0.941.

Because `genuine` has only ~1e3 negatives, every quantization effect there sits
inside a +-0.002..0.003 CI. The bit-width conclusions below therefore come from
`clean` (+-0.0005), which is also the label that actually responds to these
features.

## Position deltas: 4 bits is enough, at <= ~1 code per octave

Paired dAUC vs float position deltas, `clean` label:

| bits | k=0.75 | k=1.0 | k=1.5 | k=2.0 (deployed) |
|---|---|---|---|---|
| 3 | -0.0005 | -0.0024 | -0.0182 | **-0.0588** |
| 4 | -0.0001 | **+0.0002** | -0.0003 | -0.0014 |
| 5 | -0.0001 | +0.0001 | +0.0000 | +0.0000 |
| 6 | -0.0001 | +0.0001 | +0.0000 | -0.0000 |

Equal-frequency (best possible n-bit code) for reference: 3 bits -0.0016,
4 bits -0.0005, 5 bits -0.0002.

- **4 bits with k=1** is the recommendation: statistically indistinguishable
  from float, 8.4% saturation, 94.7% code efficiency.
- k=0.75 removes saturation entirely (0.02%) for -0.0001 — pick it if a
  populated top code is objectionable.
- **The deployed k=2 is the defect, not the width.** It costs -0.0014 at 4 bits
  and collapses (-0.059) at 3.
- 5+ bits buys nothing measurable. 3 bits at k=0.75 is a viable frugal option.

## Angle deltas: give them the same 4 bits, not "a few"

Paired dAUC over 4-bit position-only, `clean` label — monotonic and every step
significant:

| angle bits | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| dAUC | +0.0032 | +0.0071 | +0.0156 | **+0.0209** |

Beta is the single strongest wrong-hit discriminant of any chi2 dimension
(AUC 0.879 alone). Truth-classified pull widths (robust sigma, correct hits vs
wrong-track hits) show why:

| dim | correct hits | wrong hits | ratio |
|---|---|---|---|
| pullX | 2.04 | 2.27 | **1.1** |
| pullY | 1.61 | 6.29 | 3.9 |
| pullAlpha | 2.20 | 5.81 | 2.6 |
| pullBeta | 1.30 | 46.3 | **35.6** |

Position-x barely separates correct from wrong hits in the core (ratio 1.1);
the angles separate strongly. Do not starve the angle fields.

## Two corrections this study forces

1. **The documented v2.6 rationale is wrong.** The spec says summing the angle
   term into the position chi2 "dilutes any total-chi2 discriminant". Measured:
   the plain 4-way sum is the *best* single scalar (0.941 for `clean`), beating
   position-only (0.768) and angle-only (0.883). Summing does not dilute. The
   real reasons to split are (a) the angle terms are the strongest wrong-hit
   discriminants and the old (rphi, rz) pairing hid them by mixing them with
   position, and (b) separability lets the consumer weight dimensions. The
   conclusion (split) stands; the stated reason does not.

2. **The KF error model is mis-sized.** For hits truthfully matched to the
   track's own TP the pulls should be N(0,1); measured robust sigmas are 2.04
   (X), 1.61 (Y), 2.20 (Alpha), 1.30 (Beta), and 29.8% of *correct* x-hits sit
   beyond 3 sigma. So the bulk of every chi2 delta is inflated by a factor
   ~1.7-4.8 from mismodeling rather than from hit correctness. Fixing the
   measurement/prediction covariance would compress the 22-28 octave range,
   reduce the bits required, and improve discrimination. This is the highest
   value follow-up here, ahead of any further bit tuning.

## Per-field constants (the pipeline tool)

`ngtagger calibrate-chi2-quant -i '<glob>'` runs this calibration over a glob of
nano files and emits pasteable C++/Python constants plus the evidence. It reads
v2.6 split columns directly, or reconstructs the totals exactly from per-hit
pulls with `--allow-pull-derived` for pre-v2.6 files. Each field gets its own
`(bits, k)`; a per-width equal-frequency LUT is measured alongside the log form,
so a log shortfall can be attributed to the code shape rather than the budget.

Selection: fewest bits accepted by `--tol` and `--sat-max`; among those, prefer
codes statistically indistinguishable from unquantized; then least saturation.
Candidates are filtered only on label-free criteria and then re-measured with
real fits — the single-feature proxy is used solely to shortlist, because its
*ordering* disagrees with fits (it prefers k=0.5 for X, which a fit shows losing
0.0022, over k=1.0-1.25, which fits put at parity).

Result on the reference sample (`clamp_on_f{1,2}`, label `clean`, isolated
per-field paired dAUC vs unquantized):

| field | chosen | dAUC | statistical parity would need |
|---|---|---|---|
| X | 4 bits, k=0.75 | -0.00095 | 5 bits quantile (-0.00024) |
| Y | 4 bits, k=1.0 | -0.00053 | 5 bits, k=2.0 (-0.00011) |
| Alpha | 6 bits, quantile LUT | -0.00442 | not reached by 6 bits |
| Beta | 6 bits, quantile LUT | -0.00350 | not reached by 6 bits |

Jointly, all four coded vs all four float: **dAUC -0.00048** [-0.00064,
-0.00032]. The joint loss is an order of magnitude below the per-field isolated
losses because the four fields are mutually redundant — with the other three at
full precision, every individual field looks nearly free.

Two conclusions worth separating:

- **Position needs 4 bits and the code shape does not matter.** At 4 bits the
  quantile LUT (-0.0011) is no better than log (-0.00095 / -0.00053), so the log
  form is the right choice: one multiply and a clamp, no threshold table.
- **The angle fields are not quantizer-limited, they are information-rich.**
  The quantile LUT roughly matches log at 6 bits (Alpha -0.0044 vs -0.0063,
  Beta -0.0035 vs -0.0037) and the loss only halves per added bit
  (4/5/6 bits -> -0.012 / -0.008 / -0.0044 for Alpha), so parity would need
  ~8 bits. Since the refit BDT sits INSIDE the producer, ahead of any
  transmission boundary, full-precision angle inputs are available on chip and
  quantizing them buys only comparator width. Quantize the angles only for a
  study of what could cross a hardware boundary, and then quote the loss.

## Methodology notes

- Scores are 5-fold out-of-fold, so all negatives enter the AUC while every
  prediction stays out-of-sample. Comparisons are paired (identical tracks and
  folds, features differing only in quantization) with a paired bootstrap over
  tracks, which cancels the shared noise; independent AUCs on `genuine` would
  not resolve any of these effects.
- AUC uses tie-averaged ranks (verified against `sklearn.roc_auc_score`);
  quantized features are heavily tied, so this is load-bearing.
- The equal-frequency quantizer is fit on the full-sample feature distribution
  (no labels), so it is a mild transductive upper bound, quoted only as such.
- `clean`'s absolute AUC (~0.996 with float 4-way inputs) should not be read as
  physics performance: a wrong hit mechanically produces a large pull, so the
  label is nearly separable given the pulls. Its value here is sensitivity to
  coarsening, which is exactly what a bit-width study needs.
