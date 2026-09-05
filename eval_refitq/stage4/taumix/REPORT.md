# Tau-mixing study (taumix): QQToHToTauTau unified nanos in the stage-4 jet-tagger training

Two 50-event QQToHToTauTau RelVal files produced as unified coherent nanos (1111_coopt + 0000_baseline per file) and mixed into the stage-4 training (10 ttbar files) per view. All metrics: one-vs-rest AUC, 3 seeds (mean±seed std), Hanley–McNeil SE (HM) for per-class AUCs.

## Provenance / production notes

- htt nanos produced with the NEW HashPRNG-compound angle-synthesis producer (A/B-validated statistically identical to old code); the 10 ttbar unified nanos were made with the OLD code. Mixing is defensible (per-layer residual KS p=0.12-0.74) but the training set is cross-provenance. Also: the deployed l1tPh3SmartPixelsNano_cff lost the unified-nano helpers in the migration; production used the Jul-21 superset installed as l1tPh3SmartPixelsNanoFat_cff (new module, deployment untouched).

- Split: ttbar block keeps the ORIGINAL stage-4 test membership (no leakage into the continuity re-score); htt block stratified-by-class 20%; train/test orders globally shuffled (ttbar+htt fully interleaved).

## Jet inventory (htt files, per view)

| view | htt labeled jets | b | charm | light | gluon | taup | taum | muon | electron | taus/event |
|---|---|---|---|---|---|---|---|---|---|---|
| 1111 | 385 (from 100 events) | 2 | 26 | 151 | 110 | 48 | 48 | 0 | 0 | 0.96 |
| 0000 | 393 (from 100 events) | 5 | 27 | 152 | 114 | 47 | 48 | 0 | 0 | 0.95 |

## Split (per-class train/test counts, mixed dataset)

**view 1111** — original-split verification vs stage-4 dump: `{'n_dump': 1314, 'n_ours': 1314, 'pt_match': True, 'y_match': True}`

| | b | charm | light | gluon | taup | taum | muon | electron |
|---|---|---|---|---|---|---|---|---|
| train | 1446 | 643 | 1668 | 1296 | 103 | 92 | 167 | 153 |
| test | 345 | 129 | 466 | 329 | 21 | 20 | 45 | 36 |
| test (htt part) | 0 | 5 | 30 | 22 | 10 | 10 | 0 | 0 |

**view 0000** — original-split verification vs stage-4 dump: `{'n_dump': 1328, 'n_ours': 1328, 'pt_match': True, 'y_match': True}`

| | b | charm | light | gluon | taup | taum | muon | electron |
|---|---|---|---|---|---|---|---|---|
| train | 1438 | 626 | 1716 | 1336 | 101 | 95 | 171 | 147 |
| test | 374 | 147 | 431 | 322 | 22 | 22 | 41 | 47 |
| test (htt part) | 1 | 5 | 30 | 23 | 9 | 10 | 0 | 0 |

## Tau AUCs on the mixed stratified test set

**view 1111**

| model | taup AUC | taum AUC | macro AUC |
|---|---|---|---|
| ttbar-only (ctrl, same test set) | 0.675±0.018 (HM ±0.065, n=21) | 0.700±0.043 (HM ±0.064, n=20) | 0.717±0.008 |
| tau-mixed | 0.667±0.011 (HM ±0.065, n=21) | 0.704±0.017 (HM ±0.066, n=20) | 0.719±0.006 |
| tau-mixed +refitbdt | 0.673±0.013 (HM ±0.065, n=21) | 0.693±0.028 (HM ±0.066, n=20) | 0.716±0.013 |
| embedded NG tagger | 0.709 (HM ±0.064, n=21) | 0.755 (HM ±0.063, n=20) | 0.735 |
| stage-4 anchor (ttbar-only train+test) | 0.706 | 0.727 | 0.730 |

**view 0000**

| model | taup AUC | taum AUC | macro AUC |
|---|---|---|---|
| ttbar-only (ctrl, same test set) | 0.591±0.022 (HM ±0.064, n=22) | 0.676±0.026 (HM ±0.064, n=22) | 0.709±0.009 |
| tau-mixed | 0.583±0.009 (HM ±0.064, n=22) | 0.752±0.018 (HM ±0.062, n=22) | 0.717±0.006 |
| embedded NG tagger | 0.790 (HM ±0.058, n=22) | 0.728 (HM ±0.062, n=22) | 0.740 |
| stage-4 anchor (ttbar-only train+test) | 0.534 | 0.564 | 0.686 |

## Continuity: original stage-4 ttbar-only test set (no leakage — frozen out of mixed training)

**view 1111**

| model | taup AUC | taum AUC | macro AUC |
|---|---|---|---|
| ttbar-only (ctrl) | 0.674±0.010 (HM ±0.089, n=11) | 0.689±0.059 (HM ±0.091, n=10) | 0.717±0.008 |
| tau-mixed | 0.674±0.007 (HM ±0.090, n=11) | 0.693±0.030 (HM ±0.093, n=10) | 0.720±0.006 |
| embedded NG tagger | 0.704 (HM ±0.089, n=11) | 0.813 (HM ±0.083, n=10) | 0.743 |

**view 0000**

| model | taup AUC | taum AUC | macro AUC |
|---|---|---|---|
| ttbar-only (ctrl) | 0.538±0.023 (HM ±0.082, n=13) | 0.588±0.037 (HM ±0.086, n=12) | 0.691±0.011 |
| tau-mixed | 0.530±0.018 (HM ±0.081, n=13) | 0.680±0.020 (HM ±0.087, n=12) | 0.702±0.007 |
| embedded NG tagger | 0.730 (HM ±0.080, n=13) | 0.845 (HM ±0.071, n=12) | 0.747 |

## Class/pt weighting check (trainer `class_pt_weights`, onlyclass)

**view 1111** (weight = b-count / class-count on the train set)

| train set | b | charm | light | gluon | taup | taum | muon | electron |
|---|---|---|---|---|---|---|---|---|
| ttbar_only_train | 1.00 | 2.32 | 0.93 | 1.20 | 22.21 | 26.74 | 8.65 | 9.44 |
| mixed_train | 1.00 | 2.25 | 0.87 | 1.12 | 14.04 | 15.72 | 8.66 | 9.45 |

**view 0000** (weight = b-count / class-count on the train set)

| train set | b | charm | light | gluon | taup | taum | muon | electron |
|---|---|---|---|---|---|---|---|---|
| ttbar_only_train | 1.00 | 2.37 | 0.90 | 1.15 | 22.76 | 25.16 | 8.39 | 9.76 |
| mixed_train | 1.00 | 2.30 | 0.84 | 1.08 | 14.24 | 15.14 | 8.41 | 9.78 |

## Verdicts

1. **Does tau mixing improve tau performance?** View-dependent. At **0000** yes for taum: 0.676±0.026 → 0.752±0.018 on the same mixed test set (Δ=+0.076, ~2.4σ by seed spread; the gain transfers to the original ttbar test set: 0.588 → 0.680); 0000 macro +0.009. At **1111** no resolvable change (taup 0.675 → 0.667, taum 0.700 → 0.704; all Δ within seed spread). The 1111 view already learned taus adequately from ttbar; 0000 was tau-training-starved.

2. **Does the refit tau gain (1111 vs 0000) sharpen with real tau stats?** No — it SHRINKS. Stage-4 anchors suggested tau refit gains of +0.17/+0.16 (taup/taum). With tau-mixed training and ~2x tau test stats: taup +0.084 (persists), taum -0.048 (gone). Mean tau AUC difference +0.018, not resolvable at these SEs (~0.065/class). A large part of the stage-4 'refit tau gain' was 0000's trainability deficit at ~50 training taus, not refit information per se.

3. **Does the refit-BDT tau hint survive?** Not at this scale: 1111 +refitbdt taup 0.673±0.013, taum 0.693±0.028 vs baseline taumix 0.667/0.704 — no lift beyond seed spread.

4. **Does scratch-with-tau-mixing close the tau gap to the embedded model?** Partially. 0000 taum closes fully (0.752 vs embedded 0.728); 0000 taup does not (0.583 vs 0.790, gap ~0.2). 1111 taus remain ~0.04–0.05 below embedded (0.667/0.704 vs 0.709/0.755), within ~1 HM SE. Macro gap to embedded is ~0.02 (0.719/0.717 vs 0.735/0.740).

## Honest accounting / caveats

- Tau yield vs expectation: 96 labeled tau jets / 100 events (0.96/evt) vs ~1.3/evt naive expectation; GenVisTau rate is 1.19/evt, so ~81% of visible taus yield a labeled jet — gen-matching is healthy; the shortfall is jet-pt/acceptance, not labeling.
- Tau statistics roughly DOUBLE (test taus 21→41 at 1111, 25→44 at 0000; per-class HM SE ±0.08–0.09 → ±0.065). Resolvable at this level: tau effects of Δ≳0.1 (direction + machinery validation); paired same-test-set deltas down to ~0.05 via seed spread. NOT resolvable: 0.01-level tau claims — those need the ≥5k-tau campaign.
- Class weighting (`class_pt_weights`, onlyclass): tau weights fall ~22–27 → ~14–16 with mixing; the VBF-like light influx moves the light weight 0.93→0.87 and gluon 1.20→1.12 (<8% shift for non-tau classes) — behavior sensible, no pathologies. Note the stage-4 matrix protocol (replicated here) trains UNWEIGHTED; the weights are what `run_training` would apply.
- htt-only test subset is much harder for every model incl. embedded (macros 0.56–0.68): different topology (VBF light jets, tau-rich, b-poor) and tiny (77 jets) — treat per-subset numbers as indicative only.
- Cross-provenance training set (see provenance note above): old-code ttbar nanos + new-HashPRNG htt nanos, defensible per A/B KS tests but recorded.
- ttbar block of the split keeps the original stage-4 test membership rather than a fully re-stratified split — this is what makes the continuity re-score leakage-free; htt block is stratified-by-class (20%).

## Plots

- `plots/per_class_auc.png` — per-class AUC comparison
- `plots/tau_roc_overlay.png` — tau ROC overlay (mixed vs ttbar-only vs embedded)
