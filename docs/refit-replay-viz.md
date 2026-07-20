# SmartPixels digiRefit refit-replay visualizer (5-par prompt framing)

Interactive, self-contained (kernel-free) Plotly tool that shows a **PROMPT**
digiRefit L1Track Kalman refit **step by step**, across the 3 angle modes
(`none` / `alpha` / `alphaBeta`) and the 15 SmartPixels layer configs
(`1000` .. `1111`), for a curated set of example tracks. Built for the RISE talk;
opens offline in a browser and embeds in a Jupyter/RISE notebook.

The framing (`mem:smartpixels-5par-framing-directive`) is **5-par OT-only vs 5-par
OT+IT** — for b-tagging (the guiding di-Higgs measurement) we need 5-par prompt
tracks with a real impact parameter `d0`, and the question is: **does adding the
SmartPixels inner-tracker (IT) hits improve the 5-par prompt track** (impact
parameter, vertexing)? The 4-par pinned-`d0` path is dropped entirely.

- Package: `ngtagger.viz.refit_replay` (entry point `build_refit_viz`), plus the
  KF core `ngtagger.viz._kf.replay_track`, I/O `ngtagger.viz._dataio`, truth
  `ngtagger.viz._truth`, curation `ngtagger.viz._curate`.
- Regenerate the HTML: `pixi run python eval_refitq/refitviz/make_refit_viz.py`
- Output: `eval_refitq/refitviz/refit_replay.html` (self-contained, ~18 MB).
- Data (read-only, prime-target sample):
  `…/spxsmoke/nano/nano_pE_TrkSmartPix_withGen_file1.root`.

## The collections (re-anchored to prompt-5par)

| role | collection |
|---|---|
| **SEED** = 5-par OT-only track | `L1TTrack` (real `d0`/`z0`/`phi`/`rInv`/`tanL`, reduced fit `chi2XYRed`/`chi2ZRed`, `tpPt`/`tpEta`/`genuine`) |
| **REFIT** = 5-par OT+IT | `L1TSmartPixelsTrackDigiRefit{AIII,AAII,AAAI,AAAA}` (**prompt**, not the `Ext`/displaced twins) |
| refit per-hit sidecar | `L1TSmartPixelsRefitHitDigiRefit{…}` |
| **real OT stubs** | `L1TTrackStub` (`x`/`y`/`z`, `r`, `layer`, `isBarrel`) — linked per track by `trackIdx` |
| truth | `GenVtx_{x,y,z}` + `tpPt`/`tpEta` |

## The physics being shown (seed = 5-par OT-only, refit = 5-par OT+IT)

The **seed helix** is the 5-par OT-only `L1TTrack` from the L1TrackFinder (tracklet):
fit from **outer-tracker stubs + the beamline**, `promptHnpar=5` (real `d0`), anchored
way out at r ≈ 25–108 cm. The **refit helix** is the prompt `digiRefit`: the same track
after adding **SmartPixels inner-tracker hits** at r < 16 cm. So the refit is not
"fit 4 inner hits" — it is an OT-anchored, long-lever-arm track *gaining inner-pixel
constraints near the vertex*. The near-vertex `d0`/`z0` improvement is the payoff, and
the seed's inward OT→IT **extrapolation** is where the multiple-scattering **bulge**
lives (r-φ deviation grows outward across the IT layers).

## What the panels show

- **(a) OVERVIEW (full radius, r → 115 cm)**: the long-lever-arm picture. The 6 OT
  barrel layer circles (tan guide rings at 25.0 / 37.2 / 52.2 / 68.7 / 86.0 / 108.6 cm),
  the **REAL OT stubs** from `L1TTrackStub` (cyan squares at their persisted `x`/`y`),
  the 4 IT SmartPixels layers near the vertex (grey dotted), both helices, and the IT
  selected hits. Barrel stubs are drawn at their true coordinates now — not schematic.
- **(b) IT-zoom r-φ (x–y, r < 18 cm)**: the per-hit refit action at IT scale —
  seed (blue) vs refit (red) helices, and the selected-hit markers colored by
  `selHitClass` (green=same-TP/true, orange=other-TP/wrong, purple=noise).
- **(c) IT-zoom r–z**: IT layers as horizontal lines; seed vs refit `(z, r)` + IT hits.
- **(d) Kalman step table** — the key readout, see next section.

## The Kalman step table: resolution-to-truth + reduced χ2 including OT

One row per fed IT layer, plus a `seed` row, a `REFIT` row, and a `truth` footer row.
Columns:

- `d0`, `z0`, `rInv`, `phi0` — the running state.
- **`d0−tru`, `z0−tru`** — the **resolution to truth** at each step. The footer row
  shows `|seed−tru| → |refit−tru|` and whether `d0` moved **toward truth** or away.
  This replaces the old bare seed→refit Δ: the viewer now SEES whether adding IT moves
  the track toward the true parameters, for the 5-par OT-only seed vs the OT+IT refit.
- **`χ2red(OT+IT)`** — the running **reduced** χ2 (χ2/ndof) starting from the OT-only
  L1Track fit and growing as IT layers add (see accounting below).
- `Σχ2(rz)`, `Σχ2(rφ)` — the raw cumulative IT increments. `Σχ2(rφ)` is dominated by
  the parametrized-seed r-φ covariance and is **unphysically inflated** (raw,
  illustrative — see fidelity caveat); it is shown but NOT folded into the reduced χ2.

For a PRODUCED config at `alphaBeta` the `REFIT` row's `d0`/`z0` (marked `REFIT*`) are
the **real** production values from the variant track table, so the resolution-to-truth
is the bit-exact production answer; for other config×mode combos it is the replay.

### Truth convention (nano_pE interim, prompt-only) and the sign check

`ngtagger.viz._truth.track_truth`:

- `z0_true = GenVtx_z` — the generated primary-vertex z (−11…8 cm span on this
  sample; a real, resolvable truth for prompt tracks from the hard vertex).
- `d0_true = GenVtx_x·sin(phi) − GenVtx_y·cos(phi)` — the transverse-vertex
  projection (the vertex-study convention). On nano_pE this is the **beamspot**
  projection (GenVtx_{x,y} ≈ few tens of µm) so `d0_true ≈ 0` (≈5 µm). That is the
  **correct** truth for genuinely prompt tracks (true `d0`≈0) but **WRONG** for
  displaced / b-decay tracks, whose real production vertex is offset.
- `pt_true = tpPt`, `eta_true = tpEta`.

**Sign check (verified):** the `d0` convention matches `L1TTrack_d0`. The KF POCA is
`x0 = d0·sin(phi0), y0 = −d0·cos(phi0)`, so `x0·sin(phi) − y0·cos(phi) = d0` identically
— i.e. `d0_true = GenVtx_x·sin(phi) − GenVtx_y·cos(phi)` is sign-consistent with
`L1TTrack_d0`. Confirmed numerically two ways: (1) against matched-`GenPart`
production vertices, `sign(L1TTrack_d0) == sign(+(vx·sinφ − vy·cosφ))` for tracks with
a resolvable displaced vertex; (2) nano's `GenPart_dXY` uses the **opposite** sign
convention (`GenPart_dXY = −(vx·sinφ − vy·cosφ)`, 99.9% sign agreement), so do **not**
copy the `GenPart_dXY` sign.

**CAVEAT (captioned in the figure):** `GenVtx = PV` is valid `d0`/`z0` truth **only for
prompt tracks** matched to the hard vertex. For displaced/b tracks the matched-TP
`d0`/`z0` is the right truth. The truth function is structured to **swap** transparently
to matched-TP params (`tpD0`/`tpZ0`, when the richer **nano_pF** provides them) — no viz
change is needed to upgrade the truth once pF lands (`test_truth_swaps_to_matched_tp`).

### Reduced-χ2-with-OT accounting (`_reduced_chi2_running`)

The cumulative χ2 must start from the OT-only L1Track fit, not from zero. The nano
`L1TTrack_chi2XYRed`/`chi2ZRed` are **already reduced** (χ2/ndof), so:

- **OT ndof** = `2·nStubs − 5` (each OT stub is 2 measurements, 5-par helix).
- **OT absolute χ2** = `chi2ZRed · ndof_OT` (recovered from the reduced value).
- Each accepted **IT hit** adds 2 position measurements (local x, local y) and, in
  `alpha`/`alphaBeta` modes, up to 2 angle measurements (α, β). ndof grows by the
  per-hit dof (1 in `none`, 2 in `alphaBeta` for the r-z channel driving the reduced
  total).
- The running reduced χ2 = (OT absolute χ2 + Σ IT r-z increments) / (growing ndof).

We drive the running reduced value with the **physically-scaled r-z channel**; the
r-φ increments are parametrized-seed-inflated (raw values in the thousands) and are
shown separately, not folded in. The readout works as intended: **clean prompt tracks
keep the reduced χ2 near ~1–2** (good hits), a **wrong outer-layer hit inflates** it
(e.g. the wrong pick jumps to ~59 at the bad layer), and **fakes blow up**.

## Interaction model (kernel-free)

Plotly `updatemenus` are independent and **stateless** — a button can only set a
full trace-visibility vector, and cannot read the other menus' current selection.
A single (track, config, angle) combo owns all 11 of its traces (across the four
panels), so no single "axis" menu can compose with the others without a live kernel.
The design is:

- **`full combo (authoritative)`** dropdown — reaches every one of the
  `tracks × 15 × 3` states exactly (label reads `archetype | config | angle`). Source
  of truth.
- Three **convenience** dropdowns (`track` / `config` / `angle`) that jump to that one
  selection with the other two axes reset to the produced/real defaults
  (`config=1111`, `angle=alphaBeta`).

All combos are precomputed into hidden trace groups at build time; the dropdowns only
toggle visibility, so the HTML needs no Python kernel.

## The replay formulation (faithful port of the producer KF)

Ported from `L1Trigger/Phase3SmartPixels/plugins/L1SmartPixelsTrackProducer.cc`
(the `digiRefit` branch: the `scalarUpdate` lambda, diagonal-R sequential updates,
numerical-Jacobian measurement model `h=(localx, localy, cotAlpha, cotBeta)`).

- **Seed state** `a=(rInv, phi0, tanL, z0, d0)` from the `L1TTrack` reference
  (`phi0 = momentum().phi()`), `seed_npar=5` (real `d0`).
- **Seed covariance**: parametrized `diag(paramSigmas²)` (fidelity caveat below).
- **Measurement Jacobian**: analytic helix → cylinder(mean-R) crossing (mirrors
  `SmartPixelsHelixProjector::crossLayer`, incl. the `|R|` law-of-cosines fix), then a
  numerically-differenced `H[4][5]` against a **frozen module-local basis** at the
  nominal crossing (the load-bearing detail — the producer's `det->toLocal` is a fixed
  per-module rotation). Handedness recovered offline from the recorded angles:
  `sy = sign(parCotBeta / analytic_cotBeta)`, `sx = sy`.
- **Innovations** (the replay never reads the refit answer): x → `resX`;
  y → `resY − H[1]·(a−aLin)`; α/β → `pull·√S`, `S = Hₖ·C·Hₖ + σₖ²`.
- **Angle-mode gating** = `useAngles`; **config** = which layers' crossings are fed.
- Position sigmas `σx = 0.0025/√12`, `σy = 0.010/√12`; angle sigmas from the sidecar.

## Two-posture faithfulness (what is REAL vs REPLAY)

- **χ2 evolution & pulls** come straight from the nano sidecar for the 4 PRODUCED
  configs (`AIII/AAII/AAAI/AAAA` at `alphaBeta`); replay otherwise.
- **Parameter-state evolution** is always the replay. `rInv`, `phi0`, `d0` are faithful
  in sign and order-of-magnitude; `tanL`/`z0` are **illustrative**. The
  **resolution-to-truth `REFIT` row uses the REAL produced `d0`/`z0`** for produced
  configs, so the "toward truth?" verdict there is the bit-exact production answer.

## Fidelity caveat — the seed covariance

Nano does not persist the per-track helix covariance, so the replay uses a
**parametrized** seed covariance `sqrt-diag = (3e-5, 3e-4, 1e-3, 0.02, 0.03)` — a
documented choice motivated by realistic Phase-2 L1 precision, **not** a per-track fit
to the answer. Because it sets the Kalman gains and cannot carry the trackCov
**correlations**, the replayed `tanL`/`z0` sit near chance sign agreement (illustrative)
and the r-φ χ2 increments are inflated. `rInv/phi0/d0` survive because their gains are
dominated by the (well-modelled) r-φ geometry.

## Validation gate (replay vs real, per config, alphaBeta — prime-target sample)

6-event sample (n = 822–942 refit tracks per config; sign = fraction with matching Δ
sign vs seed):

| config (variant) | n | rInv sign | phi0 sign | **d0 sign** | χ2(rφ) corr |
|---|---|---|---|---|---|
| 1000 (AIII) | 822 | 0.39 | 0.33 | **0.84** | 1.00 |
| 1100 (AAII) | 915 | 0.63 | 0.63 | **0.80** | 0.99 |
| 1110 (AAAI) | 937 | 0.69 | 0.68 | **0.81** | 0.94 |
| 1111 (AAAA) | 942 | 0.69 | 0.67 | **0.81** | 0.96 |

Reading: **d0** (the physics payoff) is faithful — 80–84% sign agreement.
**rInv/phi0** improve monotonically with more layers (0.39 → 0.69); for single-layer
`AIII` they are at chance **by physics**. **χ2(rφ)** tracks the real totals with corr
0.94–1.00. `tanL/z0` are illustrative and intentionally not sign-gated.

Enforced by `tests/test_refit_replay.py::test_validation_gate_produced_configs`
(d0 sign > 0.65, χ2(rφ) corr > 0.80 per config; rInv/phi0 > 0.55 for multi-layer).

## Curated example tracks (5-par prompt framing)

Auto-curated deterministically in (event, idx) order (clean / wrong / **displaced** /
fake). Current default picks on `nano_pE_…file1.root`:

| archetype | event:idx | why |
|---|---|---|
| clean | 0:64 | genuine prompt from the hard interaction, all 4 TBPX layers same-TP hits, `z0≈PV` (`d0`≈`z0`≈0 truth both valid) — the textbook refit; reduced χ2 stays ~1–2, z0 tracks truth. |
| clean | 1:59 | second genuine prompt 4-layer track; z0 residual shrinks 0.16 → 0.003 cm (toward truth). |
| wrong | 0:11 | genuine with an **other-TP** wrong hit on an outer layer — d0/z0 still move toward truth but the reduced χ2 **inflates** at the bad layer (≈59): the wrong-hit signature. |
| displaced | 0:4 | genuine **displaced** (`d0`≈+0.26 cm) — the b-tagging-relevant case; the OT+IT refit sharpens the real d0 (0.26 → 0.13 toward 0). CAVEAT: GenVtx(PV) d0-truth≈0 is itself the interim-truth limitation here; needs matched-TP d0 (nano_pF). |
| fake  | 0:1  | unmatched/fake track — the refit is pulled by inconsistent hits (d0 moves **away**, reduced χ2 blows up): the fake-separation story. |

Override with `--tracks EVENT:IDX …` on the make script; picks are curated with
`--n-clean/--n-wrong/--n-displaced/--n-fake`.

## Usage

Standalone HTML (self-contained, offline):

```bash
pixi run python eval_refitq/refitviz/make_refit_viz.py
# -> eval_refitq/refitviz/refit_replay.html   (open in any browser)
```

In a Jupyter/RISE notebook:

```python
from ngtagger.viz import build_refit_viz
res = build_refit_viz(NANO, return_fig=True)   # or out_html=... to also write HTML
res["figure"]                                   # display inline in the slide
```

Embed the standalone file in a slide via an `<iframe src="refit_replay.html">`.
