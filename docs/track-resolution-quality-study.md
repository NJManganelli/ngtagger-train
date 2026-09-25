# Per-track resolution as the quality target (IT+OT combined finding)

Date: 2026-09-16. Scripts: `eval_refitq/combinatorics/hit_composition.py`
(measurement + per-track table) and `eval_refitq/combinatorics/resolution_mva.py`
(the regression on top). Outputs: `eval_refitq/combinatorics/results/
hit_composition.{log,json,npz}`, `resolution_mva.json`.
Sample: 200 events, PU200 ttbar, `itot_*_100ev.root`, seeding pT >= 2 GeV,
all 30 seeds, KF chi2 acceptance, hit-sharing duplicate removal.

Measured on 2 427 710 accepted tracks, 2 103 594 of which have a majority owner
carrying an IT-based truth helix and therefore a defined residual.

## Three quantities that all reference d0, and must not be conflated

| name | meaning | needs truth | available online |
|---|---|---|---|
| `sigma_res(d0)` | width of (d0_fit - d0_true) over an ensemble, or over the tracks sharing a feature vector | yes | no |
| `sigma_KF(d0)` = `sqrt(var_d0)` | the fit's CLAIMED parameter error | no | yes |
| `pull` | (d0_fit - d0_true) / `sigma_KF(d0)` | yes | no |

`sigma_KF` is a function of the hit pattern and the assumed per-hit errors: the
covariance update never touches the measured residuals, so it cannot know that a
hit came from another particle. Measured on clean tracks (every hit from one TP,
angles disabled) it is **optimistic by 1.6-3.4x, and the factor grows with hit
count**, which points at accumulated scattering or correlated noise missing from
the covariance rather than a flat mis-scaling:

| correct hits | pull(d0) | observed sigma(d0) | claimed sigma_KF |
|---|---|---|---|
| 4 | 1.62 | 242 um | 62 um |
| 6 | 2.37 | 72 um | 21 um |
| 8 | 3.18 | 69 um | 17 um |
| 10 | 3.32 | 40 um | 12 um |

Contamination then takes the pull to 9.6 / 14.8 / 17.7 at 1 / 2 / >= 3 wrong
hits, and pull(z0) to 1766. So `sigma_KF` is neither the resolution nor a
correctly scaled version of it, for two independent reasons. It belongs in a
model as an input feature and nowhere else. **The clean-track pull of 1.6-3.4 is
a KF calibration defect in its own right and is worth chasing separately from
the quality question.**

Consequence for anything downstream: `d0_over_sigma` (= |d0| / `sigma_KF`) is a
model-claimed significance, not an impact-parameter significance, and overstates
it by that factor.

## Why the existing quality labels do not answer the question

The deployed objectives are binary truth flags: `genuine`, `looselyGenuine`
(`train/trkquality.py`), `clean` (the refit-quality trainer). They answer "is
there a particle" and "did every accepted hit come from the track's own TP".
Neither orders tracks by how much a tagger should trust their d0, and the
measurement shows why a contamination count cannot be rescued into that role:

| configuration | N | sigma(d0) | sigma(1/pT) | sigma(z0) |
|---|---|---|---|---|
| 6 hits, all one TP | 17 249 | 72 um | 0.00635 | 175 um |
| 10 hits, 7 + 3 | 1 163 | 130 um | 0.01286 | 63 um |
| 4 hits, 3 + 1 | 19 970 | 525 um | 0.01370 | 18 224 um |
| 10 hits, all one TP | 148 913 | 40 um | 0.00486 | 50 um |
| 4 hits, all one TP | 5 407 | 242 um | 0.00920 | 1 432 um |

The ordering is **not the same for every parameter**: 7 + 3 is 1.8x worse than
the clean 6-hit track on d0 and 2x worse on 1/pT, but 2.8x BETTER on z0. A
single quality scalar cannot express that.

Weighted fit of log sigma over 134 composition cells (2 097 949 tracks):

| composition scalar | R^2 for sigma(d0) | R^2 for sigma(1/pT) |
|---|---|---|
| `n_wrong` (the deployed axis) | 0.435 | 0.873 |
| `n_own` | 0.525 | 0.881 |
| `own_frac` | 0.429 | 0.966 |
| `n_own` + `n_wrong` | 0.526 | 0.909 |
| four counts split by system | 0.801 | 0.944 |

A contamination count explains **44%** of the variance of log `sigma_res(d0)`;
for curvature, `own_frac` alone reaches 0.966. Different axes for different
parameters.

Where the hits sit, read off the grid (the fit's marginal coefficients are
confounded by the zero-correct-IT rows and should not be quoted):

```
sigma(d0) [um], rows = correct IT hits, cols = correct OT hits
        0     1     2     3     4     5     6
 0      -   523   377   507   617   560   585
 3    452    75    64    47    53    80    94
 4    180   110    95    58    48    45    40
```

With no correct IT hit, sigma(d0) is 0.4-0.6 mm however many OT hits are
attached, and gets slightly worse with more of them. One wrong hit also costs
differently by system: at 3 correct IT + 3 correct OT a wrong OT hit gives
88 um, while at 3 IT + 6 OT a wrong IT hit gives 138 um.

## The target that does answer it

Estimate the conditional width of the truth residual per track, by a quantile
pair: `sigma_hat = (q84.1 - q15.9)/2` from two
`HistGradientBoostingRegressor(loss="quantile")` fits. No Gaussian assumption,
and the contaminated tail does not drag it. One head per parameter (d0, z0,
1/pT), because of the ranking conflict above.

Two hard requirements on the label:

1. **Owner-referenced residuals.** The residual must be taken against the TP
   that owns the most hits on the track, not against the seed's TP. In this
   sample they disagree for **45.8%** of accepted tracks (10.5% of those are
   owned by >= 3 hits of the other TP), and the seed-referenced convention
   reports 1440 um where the owner-referenced value is 372 um in the
   2-wrong-hit population -- a 3.9x overstatement that the interactive page
   currently inherits.
2. **Conditional on the track being real.** A fake has no true d0, so these
   tracks carry no label and the width is a quantity conditional on existence.
   A tagger weight is therefore roughly P(real) x 1/`sigma_hat`^2, which keeps
   the real-vs-fake classifier as a separate, still-necessary head.

Evaluation is calibration and sharpness, never R^2:

- pull width per decile of predicted sigma, which must be 1.00 across the range;
- mean Gaussian NLL against one global sigma, sigma per true wrong-hit class
  (an oracle), and `sigma_KF` — the NLL gap is the per-track log-likelihood a
  tagger gains;
- the 10-90 percentile spread of the prediction, so calibration cannot be
  achieved by predicting the mean.

### The d0 feature is a trap -- MEASURED, and it fired

Predicted quantiles (15.9 / 50 / 84.1) of the signed residual came out
*apparently excellent*: coverage 67.9% against a nominal 68.3%, centred widths
of 9-10 um where the raw spread is 259 um. It is an artefact, and the test that
exposes it is one correlation:

| clean tracks (0 wrong hits) | value |
|---|---|
| true \|d0\| of the owner, median / p90 / p99 | 6.0 / 18.6 / 569 um |
| residual spread | 50.6 um |
| corr(residual, FITTED d0) | **0.853** |

For a clean track the true d0 (~6 um) is far below the measurement error
(~50 um), so the fitted d0 is mostly error rather than displacement. A median
head therefore learns `q50 ~ d0_fit`, i.e. "the true d0 is zero", and the
centred residual becomes `-d0_true`, whose spread is the width of the true d0
distribution -- 6-24 um. That is exactly the "9-10 um, calibrated" result. The
model is reporting the sample's PROMPT PRIOR, not a measurement.

Consequences, which apply to any future training on these targets:

- **Never ship a bias/centring head** trained on a prompt-dominated sample. It
  shrinks d0 toward zero, which tells a tagger that every track is consistent
  with the primary vertex.
- **Exclude d0-derived features** (`d0`, `d0_over_sigma`) from a width head, or
  it narrows the band via the same prior. Measured: dropping them holds coverage
  at 67.6% against 67.9% and improves the NLL by three orders of magnitude, so
  they were buying pathologically narrow bands rather than information.
- **The penalty is now measured on the deployed-style classifiers too.** All
  three discriminants shipped with the interactive page target
  `n_wrong <= 1` and are scored on genuinely clean tracks at a 0.9 cut:

  | discriminant | AUC | AUC, displaced | clean pass rate, prompt | clean pass rate, displaced |
  |---|---|---|---|---|
  | `mva_tq` (OT feature set) | 0.9927 | 0.9708 | 78.4% | 59.1% |
  | `mva_refit` | 0.9931 | 0.9775 | 79.5% | 60.1% |
  | `mva_clean` | 0.9951 | 0.9846 | 80.1% | 60.0% |

  A ~20 point penalty for being displaced, on tracks whose every hit came from
  the owning particle. The classifiers have partly learned "displaced means
  suspicious", so a tagger cutting on any of them discards its own signal. The
  exporter now reports these two rates for every discriminant so the effect
  cannot hide behind a high aggregate AUC.
- **Validate on the clean-AND-displaced subsample** and require the predicted
  width there to match clean prompt tracks. In 200 events there are 7 320 such
  tracks (2.3% of clean, \|true d0\| > 200 um) -- too few to affect any loss, so
  they must be an explicit constraint, not left to the optimiser.
- Any bias correction needs a sample with realistic displaced content (b/c
  decays, K/Lambda) before it can be trusted at all.

What survives and is usable: the RANKING. Selecting on predicted width gives
actual sigma(d0) of 67.9 um for the best 10% of tracks, 84.6 um for the best
20%, 131 um at 50%, against 259 um for the whole collection, with a sharp cliff
after the second decile in every parameter. And the three predicted widths are
nearly orthogonal (rho 0.32 for d0 vs z0, -0.03 for d0 vs 1/pT, 0.34 for z0 vs
1/pT), so one trustworthiness scalar is unavailable in principle.

## Owner referencing also fixes the LABEL, not just the measurement

Re-referencing truth to the majority owner raised the held-out AUC of the
refit-population discriminant from 0.9672 to 0.9931 with no change of features.
The old target asked it to learn `n_wrong <= 1` counted against the seed's
particle, which mislabels the 45.8% of tracks owned by someone else: the
discriminant was being trained partly on noise. A cleaner label is worth more
than any feature added so far.

## Two efficiencies, hosted side by side

| | rule | judges | responds to browser cuts |
|---|---|---|---|
| seed-truthful | the seed's own clusters all belong to the TP | a SEED: can this layer pair start a search that reaches this particle | no, fixed at census time |
| delivered | some surviving track is majority-owned by the TP | the SYSTEM: does a usable track come out | yes: seeds, DR, cuts and score thresholds |

Seed-truthful is the more generous number. Delivered is computed in the browser
from the surviving track set precisely so that it moves when a score threshold
moves, which is the whole point of the MVA-threshold workflow. It needs an exact
track-to-TP link, which is why `own_tpidx` exists: the packed
`(event << 20) | tpIdx` key needs 30 bits, the track rows are float32, and at
event 999 keys 128 apart collapse onto one value -- so `tp_key` must never be
used for equality matching.

## Six taggers x two variants, measured

Scripts: `eval_refitq/combinatorics/quality_taggers.py`. Results:
`results/quality_taggers.json`, `taggers_no_d0.json`, `taggers_with_d0.json`.
Variants split on the **fitted** |d0| at 200 um, so the variant a track is
scored under is decidable online, as Phase-2 routes tracks through prompt
versus extended tracking. Each parameter target asks "is this parameter as
accurate as a clean track's", i.e. |residual| below the clean-population 68.3
percentile in the same window.

### The variant split is mandatory, and d0 is why

Prompt-trained model scoring displaced tracks, against the displaced-trained
model on the same tracks, matched 20% keep rate:

| target | own AUC | prompt-model AUC |
|---|---|---|
| clean | 0.9859 | 0.9789 |
| invpt | 0.9432 | 0.9270 |
| cot | 0.9688 | 0.9597 |
| z0 | 0.9439 | 0.8962 |
| phi0 | 0.8571 | 0.8004 |
| **d0** | **0.9720** | **0.4534** |

Five of six transfer with a few points lost. d0 goes **worse than random**: the
prompt model learned "small |d0| means an accurate d0", true where the fitted d0
is mostly error, and that rule inverts once the displacement is real.

### Six taggers are really four

Rank correlation, prompt window: `invpt`-`phi0` 0.95, `cot`-`z0` 0.89, `d0`
~0.45 against everything, `clean` 0.09-0.19 against everything. Displaced
window: `cot`-`z0` 0.93, `invpt`-`phi0` 0.62, `d0` independent (0.12, 0.38,
-0.01, 0.03). So the natural set is {cleanliness, r-phi (curvature+phi0), r-z
(cot+z0), d0} x {prompt, displaced}, and d0 is the member that must be
per-variant.

### Angles find the contamination

`second_angle_pull` is the top cleanliness feature in both windows and dominant
in the displaced one (+0.1065, four times the next). `chi2_angle_per_cl` leads
the r-z taggers in the displaced window (+0.0722 for z0). The per-cluster
direction is doing the work on exactly the tracks that are hardest to judge.

### The d0 features: what they buy and what they cost

|  | with `d0`, `d0_over_sigma`, `sig_kf_d0` | without |
|---|---|---|
| d0 / prompt AUC | 0.9788 | 0.7263 |
| d0 / displaced AUC | 0.9720 | 0.8136 |
| prompt model on displaced | 0.4534 | 0.7072 |
| sigma(d0) at 20% keep, prompt | 19.9 um | 33.6 um |
| clean / prompt AUC | 0.9965 | 0.9965 |
| clean / displaced AUC | 0.9859 | 0.9844 |

Retention of GOOD tracks at a 20% cut, by the owner's TRUE |d0|:

| window / target | 0-50 um | 50-200 um | 200-1000 um | >1000 um |
|---|---|---|---|---|
| clean / displaced, with | 98.6% | 98.3% | 99.3% | 95.5% |
| d0 / displaced, with | 40.3% | 42.8% | 45.3% | **11.3%** |
| d0 / displaced, without | 36.0% | 39.5% | 57.0% | **36.7%** |
| d0 / prompt, with | 53.0% | 18.5% | 12.8% | -- |
| d0 / prompt, without | 42.5% | 61.7% | 57.2% | -- |

Conclusions:

- **The earlier "20 point displaced penalty" was largely a window artefact.**
  It compared prompt against displaced tracks without splitting on the fitted
  d0. Inside the displaced window the cleanliness tagger keeps 95-99% of good
  tracks at every true displacement -- no penalty at all. The variant split is
  what fixes it.
- In the PROMPT window a truly displaced track is one whose displacement was
  mismeasured as prompt, so a low keep rate there (4.5% above 1 mm) is correct
  behaviour, not unfairness.
- **The d0-accuracy tagger must not see the d0 features.** They triple the
  rejection of well-measured strongly displaced tracks (36.7% -> 11.3%) and
  they are what inverts the cross-variant transfer. Their apparent AUC gain is
  substantially tautological: on a prompt-dominated sample "small fitted |d0|"
  nearly restates "small residual".
- Cleanliness can drop them for free (0.9965 / 0.9844 against 0.9965 / 0.9859).
- `clean` and `d0` scores are independent (rho 0.05 prompt, -0.17 displaced):
  both are needed.

### The signed-versus-vector question, settled by construction

Decomposing the impact parameter into signed (dx, dy) and targeting
sqrt(dx^2+dy^2) changes nothing, and cannot: the perigee is DEFINED
perpendicular to the track direction, so the transverse perigee position has one
degree of freedom, not two. Measured on clean tracks, the perigee error across
the track is 50.0 um -- identical to the signed d0 residual -- and the error
along the track is 0.0 um. `dx` and `dy` are the same residual rotated into the
lab frame (each ~50/sqrt(2) = 30 um), and the correlation with the fitted |d0|
is +0.721 for both forms to three decimals.

What the exercise did reveal: **the d0 sign is a coin flip for prompt tracks**
(45.6% wrong below 10 um of true displacement, 34.9% at 10-50 um) and becomes
reliable exactly where a flavour tagger needs it (7.1% at 50-200 um, 5.7% at
200-1000 um). Since taggers consume the SIGNED impact-parameter significance, a
predicted sign reliability may be worth more than a predicted resolution.

### The real limitation is in the finding, not the tagger

Clean tracks -- every hit from the owning particle -- get dramatically worse as
the displacement grows, and it is not the d0 prior: fitted/true magnitude is
1.00-1.06 in every band, so nothing is shrunk toward zero.

| true \|d0\| | N | sigma(d0 residual) | median n_it | median nhit |
|---|---|---|---|---|
| 0-50 um | 1 573 422 | 49 um | 4.0 | 9 |
| 200-1000 um | 29 458 | 66-252 um | 3.0 | 9 |
| 1000-3000 um | 6 200 | 1100 um | 2.0 | 7 |
| >3000 um | 1 796 | 2711 um | 2.0 | 6 |

The IT hits vanish. Above 1 mm the tracks are delivered overwhelmingly by OUTER
seeds -- OL2+OL3 (15.7%), OL2+OL3+OL4 (15.7%), OL1+OL2 (15.6%) -- where the
prompt population is seeded much more evenly and with IT involvement. The
mechanism is the d0/r term: at r = 3 cm a 1 mm impact parameter throws the
azimuth by 33 mrad, far outside the inner-layer pairing and projection windows,
while at OT radii the same displacement is under 0.5 mrad and invisible. Triplet
seeds, which solve d0 rather than assuming it, retain twice as many IT hits in
that band (median 2 against 1).

So the tracks whose d0 matters most arrive with half the IT hits that set d0
resolution.

### Widening the projection windows does NOT fix it -- measured, hypothesis dead

`run_scan.sh` phase 1, 100 events, scanning the impact-parameter allowance the
projection roads carry (`--d0-window-cm`):

| allowance | pairs/ev | projections/ev | fits/ev | owned/ev | sigma(d0) | eff <10um | 50-200um | >1mm |
|---|---|---|---|---|---|---|---|---|
| 0 um | 294 105 | 45.3M | 12 462 | 6 211 | 55.5 um | 92.3% | 92.6% | 80.2% |
| 500 um | 366 522 | 57.3M | 10 579 | 4 950 | 53.7 um | 87.5% | 84.2% | 77.4% |
| 1000 um | 435 925 | 66.7M | 6 879 | 3 103 | 52.5 um | 85.3% | 81.1% | 74.9% |
| 2000 um | 562 899 | 83.7M | 4 089 | 1 559 | 51.0 um | 81.8% | 78.5% | 69.6% |

Every band gets WORSE, including the displaced ones the change was meant to
rescue, while the projection cost nearly doubles. The follow stage attaches one
hit per layer: a wider road admits more wrong candidates, the wrong one is
picked more often, and the track then fails the chi2 gate -- fits per event fall
3x and survivors 4x. The apparent sigma(d0) improvement is survivorship, not
resolution.

This also corrects the framing. Seed-truthful REACH for >1 mm particles is
already 80.2%: the seeding finds them. What fails is HIT COMPLETENESS -- two
correct IT hits instead of four -- so the problem is hit SELECTION, not road
width, and making the road wider is direct evidence for that.

The open question is therefore narrower: for a displaced particle reached by an
outer seed, is the correct inner-layer cluster inside the window but out-ranked,
or outside it entirely? The scan points at out-ranked, which would make this a
hit-ranking fix rather than a re-roading one.

## To fold into ngtagger-train (not a one-off)

This study is deliberately standalone because the truth it needs is not in the
nano tables. To train on these targets directly in the main pipeline:

- **Truth columns to produce.** Per track: the majority-owner TP index and its
  hit count, and the correct/incorrect hit counts split by system
  (`n_it_right`, `n_it_wrong`, `n_ot_right`, `n_ot_wrong`). The existing
  associator flags (`genuine`, `looselyGenuine`) cannot be post-processed into
  these — the owner identity is required, and a plurality owner is not a flag.
  Residuals against the owner (`own_d_d0`, `own_d_z0`, `own_d_kappa`) follow.
- **Label layer** (`src/ngtagger/data/labels.py`): add a resolution-target
  family returning the owner-referenced residual per track, rather than a class.
- **Trainer** (`src/ngtagger/train/trkquality.py`, `refitquality.py`): add a
  quantile-pair regression mode and the `--label resolution-{d0,z0,invpt}`
  choices alongside `genuine` / `looselyGenuine` / `clean`; the CLI options live
  in `src/ngtagger/cli.py`.
- **Evaluation**: pull-calibration deciles, NLL against the three baselines and
  prediction sharpness, as above. R^2 and AUC are not adequate for a width.
- **Keep `sigma_KF` as a feature** and never as a normaliser or as a shipped
  uncertainty until its pull is measured at 1.00 on clean tracks.
