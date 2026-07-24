# MVA explorer

One static, offline, interactive site that browses all three SmartPixels MVA
families with the same overlay/range/aggregation machinery:

1. **Regressions** — correctionlib payloads (tkLayout smearing, CalV1 compound
   smearing, the digiRefit v4fixed payloads, or any schema-v2 file you point
   it at), grid-evaluated at export time and sliced in the browser.
2. **tkQuality** — the Stage-3 refit-quality BDT scored per track on the unified
   coherent nanos.
3. **Jet tagger** — the Stage-4 per-jet test-set predictions across the full
   (view × feature-variant × seed) matrix.

Everything is evaluated **once, in python, at export time** and stored as
compact typed binaries (int16 log10 quantization where values are strictly
positive, raw float32 otherwise); the browser does all slicing, binning,
efficiency/AUC math in pure JS (`site_src/explorer_core.js`). No server-side
compute, no CDN — the site works fully offline. This generalizes the
`eval_spixel` ratio-builder architecture.

## Quick start (export everything + open)

```bash
cd ~/smartpixels/ngtagger-train

# 1. export all panels + assemble the static site
pixi run python -m ngtagger.viz.mva_explorer export-all

# 2. serve + open (binary fetches are blocked from file://)
(cd eval_mva_explorer/site && python3 -m http.server 8742)
# then open  http://127.0.0.1:8742/explorer.html
```

`export-all` skips the jet-tagger panel with a note if the Stage-4 prediction
dumps have not been produced yet (see Panel 3 below); the tab is greyed out
until they exist.

## Per-panel export

### Panel 1 — Regressions

```bash
pixi run python -m ngtagger.viz.mva_explorer export-regressions
```

Exports the three presets:

* `reg_tklayout` — `tkLayoutRedux/tkLayout/spixel_smear_tklayout_trigger_MS.json.gz`,
  a structured cube (2 kinds × 16 configs × 5 params × pt × |eta| × z0 × |d0|).
  *kind* = `sigma` (absolute smear width) or `relative`
  (σ(cfg)/σ(0000) as stored in the payload).
* `reg_calv1` — `cmssw/spixel_smear_all_configs_barrel_CalV1_v2p1_compound.json`,
  same structure on (pt × |eta|) only; stored raw float32 because uncovered
  bins carry exact zeros.
* `reg_smarthit_true` / `reg_smarthit_fake` / `reg_spx_angle` — the digiRefit
  v4fixed payloads through the fully generic ingester (category axes such as
  the TBPX layer become dropdowns; binning/multibinning edges become the grid).

Any other correctionlib schema-v2 file:

```bash
pixi run python -m ngtagger.viz.mva_explorer export-file /path/to/payload.json \
    --id mypayload --linspace someFormulaInput=0:5:40
```

The ingester handles every node type (binning, multibinning, category,
formula, formularef, transform) by evaluating with the `correctionlib`
package on a grid harvested from the payload's own bin edges; real inputs that
only appear in formulas get a linspace (override with `--linspace`).
Corrections containing the non-deterministic `hashprng` node — and compound
stacks that include it — cannot be rendered as a static grid; they are skipped
and listed in the meta/status line, while their physical sub-corrections are
exported normally. **Exception — synthesis envelopes**: a compound of exactly
the shape `[sigma-like correction, hashprng stdnormal]` with `output_op "*"`
(the SmartPixels angle-smear factorization, e.g. `spx_angle_alpha_smear`)
appears in the correction dropdown as `<name> [envelope]` and renders the
**deterministic envelope**: the matching `*_bias` grid (0 when absent) as the
central curve with bias ± 1σ and ± 2σ bands, labelled *"synthesis envelope,
throw ~ N(bias, sigma) via HashPRNG"*. The raw hash noise is never rendered —
the envelope is the complete deterministic content of the throw. With a 2nd
axis selected the envelope entry falls back to the central bias surface
(bands are a 1D concept). Non-matching hashprng compounds (e.g. the fused
3-stack `spx_angle_*_shift`) keep the skip path.

**UI**: dataset → (structured) kind/param/num/den config — configs always in
the canonical combinatoric order `0000, 1000, 0100, 0010, 0001, 1100, 1010,
1001, 0110, 0101, 0011, 1110, 1101, 1011, 0111, 1111` — or (generic)
correction + category dropdowns; choose the x axis (and optionally a 2nd axis
for a rotatable surface); min/max cuts on the non-plotted axes restrict the
aggregation range; the 16–84% band shows the spread over the integrated axes.
`den = (none)` shows the absolute value; picking a denominator config gives
the num/den ratio (the eval_spixel ratio-builder mode).

### Panel 2 — tkQuality

```bash
pixi run python -m ngtagger.viz.mva_explorer export-tkquality
```

Scores every refit-performed L1TTrack of the unified coherent nanos with the
deployed Stage-3 conifer models using the repo's bit-faithful walker
(`ngtagger.train.refitquality.conifer_json_walk`) on the exact 24-feature
REFIT_BDT_FEATURES v1 vector (5-par prompt-track framing, identical to
`eval_refitq/stage3/train_stage3.py`):

| view | files | model |
|------|-------|-------|
| 1111 | `nano_fat_1111_coopt_file{1..10}.root` | `refitq_AAAA_conifer.json` |
| 1100 | `nano_fat_1100_coopt_file{1..10}.root` | `refitq_AAII_conifer.json` |

0000 has no refit tables → nothing to score (noted in the UI). The export is
dataset-agnostic: a future `nano_pG` export can append all 15 configs as
additional row groups without touching the JS.

Stored per track: `score` (sigmoid of the raw logit margin), `label`
(genuine), and conditioning variables `pt, |eta|, phi, z0, |d0|, nstub,
chi2rphi_bin, chi2rz_bin`.

### Panel 3 — Jet tagger

Two steps. First produce the per-jet prediction dumps by re-running the
Stage-4 matrix (~25 min; writes a *separate* summary so the original
`stage4_summary.json` is never touched):

```bash
STAGE4_SUMMARY=$PWD/eval_refitq/stage4/stage4_summary_rerun.json \
  pixi run python eval_refitq/stage4/scripts/run_matrix.py all --force
```

This drops one npz per (cell, seed) into `eval_refitq/stage4/pred_dumps/`
with all 8 class probabilities, charge-head outputs where present, true
labels, and jet pt/|eta|/phi/n-constituents. (Prediction dumping is also
default-on in the regular trainer path — `ngtagger.train.trainer.run_training`
writes `pred_test.npz` next to every trained model.)

Then convert to the site table:

```bash
pixi run python -m ngtagger.viz.mva_explorer export-tagger
```

**UI**: view (order 1111, 1100, 0000 — refit views first), feature variant
(`baseline`, `+refitBDT`, `+vtxDxy`, `+both`, `both+chargeHead`), output class
(8 flavors + 3 charge classes on the charge-head cell), seed (1/2/3 or
*seed band* = mean across seeds with a min–max band).

### Shared y-quantities (panels 2 + 3)

* **score vs x** — mean/median score with a 16–84% quantile band;
* **efficiency @ WP vs x** — drag the score-cut slider; solid = signal
  (one-vs-rest positive) efficiency, dashed = mistag of everything else;
* **per-bin AUC vs x** — one-vs-rest AUC computed per bin in the browser,
  with the per-bin counts printed on the plot (statistics are thin —
  ~1300 test jets/view for the tagger — so the counts are shown, not hidden);
* **score distributions** — overlaid normalized histograms; solid = selected
  class, dashed = the rest.

Every curve in the overlay list is its own full selector tuple, so all three
comparison modes come for free: same class across configs/views, different
classes at a fixed config, and different feature variants at a fixed
config+class. Legends spell out the full tuple
(e.g. `1111 · +refitBDT · b vs all · seed-band`).

Min/max cut boxes on the other columns restrict every computation to that
region; the `bins` control changes the number of x bins (0 = per-variable
default, log-spaced for pt).

## Testing

```bash
pixi run pytest tests/test_mva_explorer.py -q
```

covers the correctionlib ingester on synthetic files with every node type,
the quantization round-trip, the table exporters on synthetic inputs, and —
on macOS — runs the pure-JS core against python-generated reference fixtures
(`osascript -l JavaScript`) plus a smoke pass over the real rendered site
data. The big-nano integration test is skipped automatically when the nanos
are not present.

## Layout

```
src/ngtagger/viz/mva_explorer/
  __main__.py             CLI (export-regressions/-tkquality/-tagger/-file,
                          make-site, export-all)
  correctionlib_ingest.py generic schema-v2 ingester + structured smear export
  presets.py              the three first-class regression presets
  tkquality_export.py     Stage-3 per-track score table export
  tagger_export.py        Stage-4 prediction-dump -> site table export
  quantize.py             log10-int16 quantization (+ float32 fallback)
  site_src/               explorer.html, explorer_core.js, JXA tests/fixtures
eval_mva_explorer/site/   rendered site + data (regenerable, git-ignored)
```
