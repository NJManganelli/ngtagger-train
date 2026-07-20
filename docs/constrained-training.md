# MDMM constrained training + charge head — implementation notes

Implements two items of `docs/model-space-study.md` (B.2.2 Route B and
B.2.3); this file records the design decisions taken where the study left
freedom. Tests: `tests/test_mdmm.py`, `tests/test_charge.py`.

## 1. MDMM (Route-B keras.ops port) — `src/ngtagger/train/mdmm.py`

Backend-agnostic port of the Platt–Barr Modified Differential Multiplier
Method, semantics-matched to the pip `mdmm` torch package (kept installed as
the reference). Each constraint adds

```
penalty_i = scale_i * (lambda_i * g_i + damping_i/2 * g_i^2)
```

with pip-mdmm infeasibility conventions (`eq`: `g = value - fn`; `max`:
`g = max - fn - slack^2`; `min`: `g = fn - min - slack^2`; plus the
slack-free `max_hard`/`min_hard`). Gradient ascent on `lambda` is realized
with a **stop-gradient split** inside `Model.compute_loss` — a value-neutral
term pair whose only gradient is `d/dlambda = -scale*g` — so one ordinary
optimizer drives theta (descent), slack (descent) and lambda (ascent) at
once, on any Keras-3 backend. `tests/test_mdmm.py` verifies **bit-level
lockstep** with pip mdmm under SGD (same state, same batches, 400 steps) and
exact value parity of the augmented penalty, plus KKT-point convergence on
toys (min `(x-2)^2` s.t. `x<=1` → `x*=1`, `|lambda*|=2`).

Deviations from pip mdmm (documented in the module docstring):

- slack variables are constant-initialized (`slack_init`, default 0) instead
  of lazily from the first `fn()` — lazy nan-init does not survive graph
  tracing. Same fixed points; the cross-check test passes an explicit init.
- the single shared optimizer (Adam by default) applies its adaptive scaling
  to lambda too, exactly as pip mdmm's negative-lr scheme does with Adamax.

### Trainer integration (opt-in)

A top-level `constraints:` section in the training yaml (see
`configs/deepset_hgq2_mdmm.yaml`) activates it; without it nothing changes.
`DeepSetHGQ2.compile` then wraps the functional model in a training-only
`MDMMModel` (weights shared; `save()`/export still see the plain model) and
`fit` attaches `MDMMLogger`, which injects `lambda_<name>` / `inf_<name>`
into the history logs; `run_training` logs the constraint list as an mlflow
param and the final multiplier state as mlflow metrics.

Design decision: constraints **add** Lagrangian terms on top of the compiled
losses; to fully replace a fixed loss weight by a constraint, set that head's
`loss_weights` entry to 0 and bound it with `head_loss` (the example config
does exactly this for the charge head — the study's "floor constraint
instead of a weighted sum").

### First-class targets

- `pt_bias` — per-batch differentiable proxy for the pt-regression response
  bias: the head regresses `r = pt_gen/pt_reco` (clipped ≥ 0.3, safe
  denominator), so `fn = |mean_batch(pred/true - 1)|`; a `max` constraint at
  0.01 is the study's "bias ≤ 1%". At batch 2048 the estimator noise is
  ~`sigma_response/45`, well below the 1% scale. Per-pt-bin variants are a
  follow-up (one constraint per bin with masked means).
- `bkg_eff` / `sig_eff` — rate/efficiency proxies as formulated in the study
  ("loss on a fixed threshold quantile"): soft counting efficiency
  `mean(sigmoid((s - threshold)/temperature))` over jets whose true class is
  in the selected set, `s` = summed softmax probability of `score_classes`.
  `bkg_eff` + `max` bounds the rate at a fixed working point; `sig_eff` +
  `min` is an efficiency floor. `temperature → 0` recovers hard counting.
- `head_loss` — mean per-batch loss of one named head (CE/mse/mae/logcosh):
  the aux-head accuracy floor ("aux CE ≤ eps").
- `ebops` — see below.
- programmatic escape hatch: an item may carry a callable `fn` directly.

### EBOPs budget constraint — status: WIRED (callback route)

HGQ2 layers compute a differentiable `ebops` tensor in-call and add
`beta * ebops` to the model losses, where `beta` is a **non-trainable
variable** per layer (mirrored into a uint32 `_ebops` variable for readout).
`EBOPsConstraint` (a callback, auto-attached for `target: ebops`) exploits
exactly that:

- primal term: assigning `beta_l := scale * lambda` makes the already-present
  `beta * ebops(theta)` losses the MDMM term `lambda * EBOPs(theta)` — no
  extra gradient wiring needed (same actuator as HGQ2's own
  `hgq.utils.sugar.BaseBetaPID`, with multiplier dynamics instead of a PID);
- dual update (per batch, no gradients needed):
  `g = EBOPs/budget - 1` (relative, default) and
  `lambda ← max(0, lambda + lambda_lr * g)`, reading the `_ebops` mirrors
  (one step stale — standard for MDMM-on-metrics); damping enters as
  `beta = scale*(lambda + damping*max(g,0))`.

Notes: this **replaces** the static `model_config.beta` — don't hand-scan
beta and constrain EBOPs simultaneously. `lambda_lr` needs tuning per budget
scale (start ~1e-6 with relative g). The **adapter point** for external
resource estimators is `EBOPsConstraint(ebops_fn=callable(model) -> float)`
(e.g. a da4ml/hls4ml post-synthesis cost model); the default sums the layer
`_ebops` variables. A fully in-graph differentiable variant (summing the
live per-layer ebops tensors inside `compute_loss`) is possible but needs an
upstream HGQ2 hook to expose the live tensors separately from the other
regularization losses — left as the documented follow-up for the
"paper-grade" study.

### Config schema

```yaml
constraints:
  damping: 1.0              # default for all items
  items:
    - target: pt_bias       # pt_bias | head_loss | bkg_eff | sig_eff | ebops
      type: max             # eq | max | min | max_hard | min_hard
      value: 0.01           # bound / target / budget (ebops)
      scale: 1.0            # optional
      damping: 10.0         # optional per-item override
      # head_loss:  head, loss, from_logits
      # bkg/sig_eff: head, background_classes/signal_classes, score_classes,
      #              threshold, temperature
      # ebops:      lambda_lr, relative, update_every
```

Backend note: developed and tested on the repo's default TensorFlow backend;
the implementation uses only `keras.ops` + variable assignment (JAX's purely
functional training loop would need the `_last_inf` bookkeeping removed).

## 2. Charge-classifier head — scaffold

- **Model** (`src/ngtagger/models/deepset_hgq2.py`): opt-in via
  `model_config.charge_layers: [..]` — a QDense stack + 3-logit output
  (`charge_output`, CCE-from-logits, `categorical_accuracy` metric) sharing
  the pooled trunk, following the existing head pattern. `loss_weights`
  grows to 3 entries (or `charge_loss_weight`). Enabled head + missing
  labels fails loudly; `DeepSetContrastive` rejects the knob explicitly.
  2-bit output quantization at the boundary is an export-time concern
  (states {-, 0, +} + spare), not a training-time one.
- **Labels** (`src/ngtagger/data/labels.py`): `label_jet_charge` matches
  jets to GenJet (same deltaR machinery) and maps `partonFlavour` through
  `parton_charge_class` — exact map in its docstring (d/s/b → q−, u/c/t →
  q+, antiquarks flipped, gluon/unmatched → neutral). Additive and
  backward-compatible: `label_jets`/8-class scheme untouched;
  `prepare_dataset` now also returns `charge_train/test` one-hots
  (`CHARGE_CLASS_LABELS = [qminus, neutral, qplus]`) filtered by the same
  keep mask. Leptonic jets also receive a charge class (their charge already
  lives in the 8-class labels); mask downstream for quark-only studies.
- **Benchmark** (`src/ngtagger/eval/charge_baseline.py`): `jet_charge_kappa`
  implements the study's `Q_κ = Σ q_i pt_i^κ / (Σ pt_i)^κ` (default) with the
  classical Field–Feynman normalization `Σ q_i pt_i^κ / Σ pt_i^κ` as
  `norm="pow_sum"`; `jet_charge_from_features` reconstructs constituent
  charge from the baseline one-hot charge flags (so the benchmark consumes
  exactly what the model sees); `evaluate_charge_baseline` reports the
  must-beat q+ vs q− ROC AUC and per-class means. Adding Q_κ (κ=0.3/0.5/1.0)
  as engineered *input* features is a follow-up (needs a jet-level feature
  path; the current feature tensor is per-constituent).

## Real-data smoke

Needs an L1PFTrkNano file with `GenJet_partonFlavour` (any withGen flavor);
none is available on this machine outside the container areas, so the
real-data tests keep the existing `NGTAGGER_TEST_NANO` opt-in gate
(`tests/test_charge.py::test_charge_labels_from_nano`,
`tests/test_smoke.py::test_nano_pipeline`). Everything else is synthetic by
design.
