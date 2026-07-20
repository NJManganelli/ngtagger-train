# SmartPixels digiRefit refit-replay visualizer

Interactive, self-contained (kernel-free) Plotly tool that shows a digiRefit
L1Track Kalman refit **step by step**, across the 3 angle modes
(`none` / `alpha` / `alphaBeta`) and the 15 SmartPixels layer configs
(`1000` .. `1111`), for a curated set of example tracks. Built for the RISE talk;
opens offline in a browser and embeds in a Jupyter/RISE notebook.

- Package: `ngtagger.viz.refit_replay` (entry point `build_refit_viz`), plus the
  KF core `ngtagger.viz._kf.replay_track`, I/O `ngtagger.viz._dataio`, curation
  `ngtagger.viz._curate`.
- Regenerate the HTML: `pixi run python eval_refitq/refitviz/make_refit_viz.py`
- Output: `eval_refitq/refitviz/refit_replay.html` (self-contained, ~14 MB).
- Data (read-only): `…/spxsmoke/nano/nano_puD_TrkSmartPix_withGen.root`.

## The physics being shown (seed = OT-only, refit = OT + SmartPixels)

The **seed helix** is the OT-only L1Track from the L1TrackFinder (tracklet): it is
fit from **outer-tracker stubs + the beamline**, anchored way out at r ≈ 25–108 cm.
The **refit helix** is `digiRefit`: the same track after adding **SmartPixels
inner-tracker hits** at r < 16 cm. So the refit is not "fit 4 inner hits" — it is
an OT-anchored, long-lever-arm track *gaining inner-pixel constraints near the
vertex*. The near-vertex d0/z0 improvement is exactly the payoff of that: pinning
the innermost end of a long lever arm. And the seed's inward OT→IT **extrapolation**
is where the multiple-scattering **bulge** lives — the r-φ deviation grows outward
across the IT layers (visible as growing resX/pullX on L3/L4), because the
beamline+OT-constrained fit absorbs MS kinks into compensating phi0/rInv and the
deviation bulges between the two anchors.

## What the panels show

- **(a) OVERVIEW (full radius, r → 115 cm)**: the long-lever-arm picture. The 6 OT
  barrel layer circles (tan, radii 24.9 / 37.2 / 52.3 / 68.8 / 86.0 / 108.3 cm from
  the tracklet firmware), the **OT stubs** on the seed helix (cyan squares, decoded
  from the track `hitPattern` — see below), the 4 IT SmartPixels layers near the
  vertex (grey dotted), both helices, and the IT selected hits. This panel makes the
  seed read as OT-anchored and the refit as adding inner constraints.
- **(b) IT-zoom r-φ (x–y, r < 18 cm)**: the per-hit refit action at IT scale —
  seed (blue) vs refit (red) helices, and the selected-hit markers colored by
  `selHitClass` (green=same-TP/true, orange=other-TP/wrong, purple=noise).
- **(c) IT-zoom r–z**: IT layers as horizontal lines; seed vs refit `(z, r)` +
  IT hits.
- **(d) Kalman step table**: one row per fed IT layer — the 5 state params
  `(rInv, phi0, tanL, z0, d0)`, the running `Δd0`/`Δz0` (the physics payoff), and
  the cumulative `χ2(rφ)`/`χ2(rz)`; seed row blue, `REFIT` row red.

### OT stubs from `hitPattern` (schematic-on-helix, honestly labelled)

The reference `L1TExtTrack` carries `hitPattern` (+ `nStubs`). On this sample
`popcount(hitPattern) == nStubs` exactly, and bit *i* (LSB) maps to OT barrel layer
*i+1*: central tracks are dominated by `0111111` (all 6 barrel layers) and `0001111`
(inner 4); forward tracks set bit 6 (a forward-disk slot, placed schematically at
the outermost barrel radius). The OT stub **x-y positions are not persisted in nano**,
so the stubs are drawn **on the seed helix** at the decoded OT radii — schematic, not
the true stub coordinates. This is captioned in the figure. The OT barrel radii come
from the tracklet firmware constants (`L1Trigger/TrackFindingTracklet` `Settings.h`:
`irmean_ = {851,1269,1784,2347,2936,3697}` × `rmaxdisk_/4096`, `rmaxdisk_=120`).

## Interaction model (kernel-free)

Plotly `updatemenus` are independent and **stateless** — a button can only set a
full trace-visibility vector, and cannot read the other menus' current selection.
A single (track, config, angle) combo owns all 11 of its traces (across the four
panels), so no single "axis" menu can compose with the others without a live
kernel. The design is:

- **`full combo (authoritative)`** dropdown — reaches every one of the
  `tracks × 15 × 3` states exactly. This is the source of truth.
- Three **convenience** dropdowns (`track` / `config` / `angle`) that jump to that
  one selection with the other two axes reset to the produced/real defaults
  (`config=1111`, `angle=alphaBeta`). Use them for the common "show me config X"
  click; use the authoritative menu to set all three axes independently.

All combos are precomputed into hidden trace groups at build time; the dropdowns
only toggle visibility, so the HTML needs no Python kernel.

## The replay formulation (faithful port of the producer KF)

Ported from `L1Trigger/Phase3SmartPixels/plugins/L1SmartPixelsTrackProducer.cc`
(the `digiRefit` branch: the `scalarUpdate` lambda, diagonal-R sequential updates,
numerical-Jacobian measurement model `h=(localx, localy, cotAlpha, cotBeta)`).

- **Seed state** `a=(rInv, phi0, tanL, z0, d0)` from the REFERENCE track
  (`phi0 = momentum().phi()`); `d0` seeded for 5-par.
- **Seed covariance**: parametrized `diag(paramSigmas²)`. See the fidelity caveat
  below — this is the one documented deviation from production.
- **Measurement Jacobian**: analytic helix → cylinder(mean-R) crossing (mirrors
  `SmartPixelsHelixProjector::crossLayer`, including the `|R|` law-of-cosines fix),
  then a numerically-differenced `H[4][5]` against a **frozen module-local basis**
  taken at the nominal crossing. Freezing the basis is the load-bearing detail: the
  producer's `det->toLocal` is a fixed per-module rotation, so the measurement axes
  must not rotate with the perturbed crossing point (otherwise `H[localx]` comes
  out identically zero). The per-module local-frame **handedness** (which flips
  around the barrel) is recovered offline from the recorded angles:
  `sy = sign(parCotBeta / analytic_cotBeta)` (100% consistent per detId), and
  `sx = sy` for the right-handed GeomDet frame.
- **Innovations** (the replay never reads the refit answer):
  - x (first update): `r = resX` exactly (`a−aLin=0`).
  - y: `r = resY − H[1]·(a−aLin)` (relinearized after the x-update moved `a`).
  - alpha/beta: `r = pull·√S`, `S = Hₖ·C·Hₖ + σₖ²`, reconstructed from the recorded
    pull and sigma.
- **Angle-mode gating** = `useAngles`: `none` = {x,y}, `alpha` = {x,y,α},
  `alphaBeta` = {x,y,α,β}.
- **Config** = which layers' crossings are fed (a subset of the AAAA layers present
  for that track — the AAAA sidecar carries all four layers' selected hits).
- Position sigmas `σx = 0.0025/√12`, `σy = 0.010/√12` (TBPX pitch/√12);
  angle sigmas from the sidecar.

## Two-posture faithfulness (what is REAL vs REPLAY)

- **χ2 evolution & pulls** are **bit-exact** for the 4 PRODUCED configs
  (`AIII/AAII/AAAI/AAAA` at `useAngles=alphaBeta`): they come straight from the nano
  sidecar (`chi2IncRPhi ≡ pullX² + pullAlpha²`, `chi2IncRZ ≡ pullY² + pullBeta²`,
  verified exact; per-hit increments sum to `spxChi2Inc{RPhi,RZ}Tot` for all
  tracks). For the other 41 config×mode combinations, χ2 is the offline replay.
- **Parameter-state evolution** is always the replay. `rInv`, `phi0`, `d0` are
  reproduced faithfully in sign and order-of-magnitude scale; `tanL` and `z0` are
  labelled **illustrative** (see below).

## Fidelity caveat — the seed covariance

This nano file was produced with `seedCovMode="trackCov"` (real per-track helix
covariance from posture-B in-job tracks; confirmed: `spxParametrizedSeed`
fraction = 0, `spxSeedCovOK` = 1). Nano does **not** persist the per-track
covariance, so the replay uses a **parametrized** seed covariance
`sqrt-diag = (3e-5, 3e-4, 1e-3, 0.02, 0.03)`. This is a documented parametrized-seed
choice motivated by realistic Phase-2 L1 track precision (tighter on rInv/phi0 than
the producer's ablation `paramSigmas`), **not** a per-track fit to the answer — it
generalizes identically across all four configs. Because the seed covariance sets
the Kalman gains, and because `tanL`/`z0` hinge on the per-track trackCov
**correlations** the parametrized diagonal cannot carry, the replayed `tanL`/`z0`
sit at chance-level sign agreement and are labelled illustrative. `rInv/phi0/d0`
survive this because their gains are dominated by the (well-modelled) r-φ geometry.

## Validation gate (replay vs real, per config, alphaBeta)

Replaying the 4 produced configs at `alphaBeta` and comparing the replayed final
params and χ2 totals to the REAL variant-track-table values (6-event sample, n =
985–1376 refit tracks per config; sign = fraction with matching Δ sign vs seed;
`d0 ratio_med` = median replay-Δ / real-Δ):

| config (variant) | n | rInv sign | phi0 sign | **d0 sign** | tanL sign | z0 sign | χ2(rφ) corr | χ2(rz) corr | d0 ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1000 (AIII) | 985  | 0.43 | 0.37 | **0.81** | 0.63 | 0.40 | 1.00 | 1.00 | +0.87 |
| 1100 (AAII) | 1214 | 0.62 | 0.60 | **0.73** | 0.56 | 0.41 | 0.99 | 1.00 | +0.72 |
| 1110 (AAAI) | 1329 | 0.66 | 0.65 | **0.74** | 0.52 | 0.42 | 0.91 | 1.00 | +0.66 |
| 1111 (AAAA) | 1376 | 0.67 | 0.64 | **0.75** | 0.51 | 0.43 | 0.95 | 1.00 | +0.59 |

Reading:

- **d0** (the physics payoff) is faithful — 73–81% sign agreement, replayed
  magnitude within a factor ~0.6–0.9 of the real kick.
- **rInv/phi0** are faithful and improve monotonically with more layers
  (0.43 → 0.67); for the single-layer `AIII` they are at chance **by physics** —
  one layer cannot constrain curvature or azimuth.
- **χ2(rφ)/χ2(rz)** track the real totals with correlation 0.91–1.00 — a strong
  confirmation that the Jacobian, innovation sourcing, and per-hit S are correct.
- **tanL/z0** are illustrative (~0.4–0.6), for the seed-covariance reason above.

The gate is enforced by `tests/test_refit_replay.py::test_validation_gate_produced_configs`
(d0 sign > 0.65, χ2(rφ) corr > 0.80 per config; rInv/phi0 > 0.55 for multi-layer
configs). tanL/z0 are intentionally not sign-gated.

## Curated example tracks (reproducible indices, default noPU-D-adjacent file)

Auto-curated deterministically in (event, idx) order; the default picks are:

| archetype | event:idx | why |
|---|---|---|
| clean | 0:28 | genuine, all 4 TBPX layers same-TP hits — the textbook refit; shows the outer-layer r-φ extrapolation **bulge** (resX and pullX grow to L3/L4, the MS-bulge signature) driving a large d0 kick. |
| clean | 0:34 | second textbook genuine 4-layer track (pt≈3.1) for contrast. |
| wrong | 0:11 | genuine with an **other-TP** wrong hit picked up on L3 — the KF gets pulled; illustrates why outer layers are wrong-hit-rich. |
| fake  | 0:3  | unmatched/fake track (all-wrong hits) — no consistent parent; best-χ2 selection cherry-picks consistent-looking hits (smaller kicks; the fake-separation story). |

Override with `--tracks EVENT:IDX …` on the make script.

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
