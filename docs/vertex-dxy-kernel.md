# Vertex transverse position (dx, dy) and FastHisto kernel option

Prototype/study tooling added to `src/ngtagger/train/nnvtx.py` (numpy fastHisto
reference), plumbed into the tagger feature set via `data/features.py` +
`data/nano.py`, with study scripts under `eval_refitq/vtxdxy/`. This is
first-look tooling, not a final measurement — the deciding statistics arrive
with the future PU production.

## 1. Vertex (dx, dy) estimator

### Physics constraint
A single track does not measure (dx, dy). It measures the projection of the
transverse displacement onto its own normal direction, i.e. the impact
parameter `d0_i`. For a prompt track from a vertex at (x_v, y_v), ignoring
curvature at the vertex scale,

    d0_i = x_v * sin(phi_i) - y_v * cos(phi_i) + noise.

The global (x_v, y_v) is recovered by a pt-weighted least-squares solve over
the PV-window tracks.

### Sign convention (verified)
The repo/TTTrack convention is `d0 = x0*sin(phi0) - y0*cos(phi0)`, from
`DataFormats/L1TrackTrigger/interface/TTTrack.h`:

    thePOCA_(d0 * sin(phi0), -d0 * cos(phi0), z0)          // POCA from (d0, phi0)
    // beamspot correction comment: d0 - (XB*sin(phi) - YB*cos(phi))

The nano `L1TExtTrack_d0` / `_phi` are `Var("d0()")` / `Var("phi()")` on that
same TTTrack, so the estimator adopts

    d0_i = x_v * sin(phi_i) - y_v * cos(phi_i).

Note this is the **negative** of the offline `TrackBase::dxy()` convention;
document the sign whenever comparing to offline quantities. The synthetic
closure test `test_convention_exact_recovery` throws noiseless
`d0 = x_v s - y_v c` and proves the solve inverts it **exactly** (to float
roundoff) — that test is the guardrail against a future sign regression.

### Normal equations
Minimise `Sum_i w_i (d0_i - x_v s_i + y_v c_i)^2` with `s=sin(phi)`,
`c=cos(phi)`, `w = min(pt, max_track_pt)` (the fastHisto pt weight). With
`S_ab = Sum w a b`, `S_wds = Sum w d0 s`, `S_wdc = Sum w d0 c`:

    [  S_ss  -S_sc ] [x_v]   [ +S_wds ]
    [ -S_sc   S_cc ] [y_v] = [ -S_wdc ]

    x_v = (S_cc S_wds - S_sc S_wdc) / det
    y_v = (S_sc S_wds - S_ss S_wdc) / det          det = S_ss S_cc - S_sc^2

Implemented in `fast_histo_vtx` (`estimator="lsq"`, default) from **parallel
per-z-bin accumulator histograms** `w, w*d0*s, w*d0*c, w*s^2, w*c^2, w*s*c,
w*d0^2, n` filled alongside the pt-weighted z histogram and aggregated over the
found peak window (the user's "parallel histograms + counts", generalised to
the sums the solve needs).

### Cheap isotropic variant
`estimator="isotropic"`: `x_v = +2 S_wds/S_w`, `y_v = -2 S_wdc/S_w`. Exact only
for phi-isotropic weighted coverage (`S_ss = S_cc = S_w/2`, `S_sc = 0`).
`test_biased_phi_coverage_lsq_vs_isotropic` shows that on a quarter phi arc
(0..pi/2, `S_sc != 0`) the LSQ solve still closes while the isotropic shortcut
picks up an O(few x100 um) bias — and `phi_condition = det/(S_ss S_cc)` (≈0.6
there) flags the imbalance.

### Uncertainty / significance
The residual scatter of d0 about the fit is computed **from the sums** (no
second track pass):

    S_wrr = S_wdd - 2x S_wds + 2y S_wdc + x^2 S_ss + y^2 S_cc - 2xy S_sc
    sigma^2_d0 = S_wrr / (n - 2),   Cov(x, y) = sigma^2_d0 * A^-1
    sigma_dx = sqrt(sigma^2_d0 * S_cc/det),  dxsig = dx / sigma_dx  (and y)

Pragmatic model: assumes `sigma_{d0,i} ~ 1/sqrt(w_i)` up to a common scale.
Needs `n>=3` for the scatter, `n>=2` and `phi_condition >= min_condition` for
the solve; otherwise the result is NaN — degraded inputs stay **visible, not
silently zero** (`test_degenerate_phi_guard`).

### Track collection (explicit choice)
d0 must come from a **5-parameter (Extended) track collection**: nano
`L1TExtTrack`. The prompt 4-parameter `L1TTrack` has `d0` pinned to 0 and
carries no transverse information. `vertex_dxy_features` raises loudly (a
`ValueError` mentioning the 4-parameter collection) if handed an
identically-zero d0 column, and `KeyError` if `pt/phi/z0/d0` are missing —
truth-required-style loud failure rather than a silent dx=dy=0 degradation.

### Prompt-track gate (real-data finding)
`d0_gate` (cm, optional) excludes `|d0| > d0_gate` tracks from the transverse
accumulators only (the z histogram / window selection stays emulator-faithful).
This is motivated by the real-data smoke below: L1 extended-track collections
in a jet environment carry displaced/loose tracks with `|d0|` up to O(cm) that
violate the prompt approximation and dominate the ungated fit.

## 2. FastHisto peak-finder kernel option

### Weakness
The flat boxcar window can prefer the midpoint *between* two similarly-hard
vertices over either true peak (the sum of two adjacent half-peaks beats a
single peak). NNVtx effectively learns a better kernel via its convolution.

### Implementation
`make_kernel` + a `kernel` argument on `fast_histo_z0` / `fast_histo_vtx`:
`flat` (default), `triangular`, `gaussian(sigma_bins)`, `epanechnikov`, or an
arbitrary `kernel_array`. The kernel is applied as a weighted correlation of
the z histogram before the arg-max window selection.

**Decisions (documented):**
- The kernel enters **only the window selection**. The z estimate stays the
  plain pt-weighted centroid of the selected window (emulator convention). A
  kernel-weighted centroid would pull z toward the kernel centre and break
  stock-vs-recomputed comparisons.
- **`kernel="flat"` is bit-identical to the pre-existing emulator boxcar.**
  Proven by `test_default_bit_identical_to_frozen` (exact `array_equal` against
  a frozen verbatim copy of the original implementation, for window_bins
  3/4/5) and `test_flat_kernel_paths_equivalent`. Backward compatibility is a
  hard requirement met by construction.

## Study results (first look)

### Kernel two-close-vertices scan (`eval_refitq/vtxdxy/kernel_scan.py`)
Grid over z-separation (1–5 bins) × relative hardness (0.6/0.8/1.0), 400
events/cell, plus a single-vertex resolution measurement. `kernel_scan.json` +
`kernel_scan.png`.

Headline (seed 0): **a tapered kernel measurably reduces midpoint picks.**

| kernel        | mean midpoint-pick rate | single-vertex \|res\| q68 |
|---------------|------------------------:|--------------------------:|
| flat          | 0.206                   | baseline                  |
| triangular    | 0.106                   | +0.0007 cm                |

The triangular kernel roughly **halves** the midpoint-pick rate at negligible
single-vertex resolution cost. (gaussian/epanechnikov also in the JSON;
triangular is the headline best on this grid.)

### Real-data (dx, dy) smoke (`eval_refitq/vtxdxy/realdata_smoke.py`)
On `nano_pu100_TrkSmartPix_withGen.root` (100 evt, L1TExtTrack, read-only),
`realdata_smoke.json` + `realdata_smoke.png`:

- 99% of events solve; PV-window multiplicity median ~26.
- **Ungated** LSQ (dx, dy) spread ~1.2 cm, d0 scatter median ~9.5 mm — the
  extended-track d0 population is *not* beam-spot-prompt (all-track median
  \|d0\| 0.077 cm but q95 ~7 cm; pt>5 GeV tracks have median \|d0\| ~1.7 cm:
  displaced/loose tracks dominate). GenVtx is at ~1–12 um.
- **Gated** (`|d0| < 0.15 cm`): window multiplicity ~21, dx/dy spread ~180–190
  um, d0 scatter ~360 um — a beam-spot-plausible transverse look, medians near
  0. This is the expected `d0_resolution / sqrt(N)`-scale clustering, once the
  prompt approximation is enforced.

Interpretation: the estimator is physically correct; the raw L1 extended-track
collection needs a prompt selection before its d0 supports a transverse-vertex
fit. This is a genuine first-look finding, to be revisited with the PU
production and a validated prompt-track definition.

## 3. Tagger feedthrough (tooling, not a measurement)

New composable feature group `vertexdxy` in `FEATURE_GROUPS`
(`data/features.py`): `vtx_dx, vtx_dy, vtx_dxsig, vtx_dysig`, one global value
per event broadcast to every constituent slot. Computed at dataset-build time
in `data/nano.py::load_jets` from the extended-track table via
`vertex_dxy_features` (which forwards `**fast_histo_kwargs`, e.g. `d0_gate`,
`kernel`). Requires `track_table="L1TExtTrack"`; on a 4-par table it fails
loudly (`test_load_jets_vertexdxy_missing_track_pt_raises`). Padded/unfilled
constituent slots are zeroed like every other feature.

Requesting it: `feature_groups: [baseline, vertexdxy]` (and set the extended
track table). End-to-end wiring and broadcast verified by
`test_load_jets_vertexdxy_group`.

**What a real study needs / would decide the hypothesis:** paired-seed AUC
deltas on b-vs-light with and without the `vertexdxy` group, using the same
modelspace methodology as `docs/model-space-study.md`. That needs the future
good-PU production (per the tier-2 plan) and a validated prompt-track d0
selection so the (dx, dy) input is physical, not dominated by displaced
tracks. The current d0 population in the smoke file is not yet suitable for a
verdict.

## Open items / TODO

- **CLI wiring (TODO, left deliberately un-wired):** a `vtx-study` / kernel-scan
  entry point in `cli.py` was NOT added — `cli.py` is under concurrent edit by
  another effort. Run the study scripts directly
  (`pixi run python eval_refitq/vtxdxy/kernel_scan.py`,
  `... realdata_smoke.py [file]`) for now, and add the CLI subcommands once the
  refitquality/CLI work lands.
- **NNVtx comparison hook:** the real kernel study should compare the flat/kernel
  arg-max against a trained convolution via `nnvtx.compare_vertex_scores`
  (NNVtx effectively learns the kernel). Left as a documented hook — needs the
  PU production and a trained `e2e_nnvtx` model.
- **Prompt-track definition:** validate `d0_gate` (or a proper prompt/quality
  selection) against truth-matched prompt tracks on the PU production before
  using `vertexdxy` as a tagger input.
- **Two-vertex / biased-coverage toys** are unit tests today
  (`test_biased_phi_coverage_lsq_vs_isotropic`, the kernel scan's two-vertex
  grid). A dedicated two-vertex transverse toy (two displaced PVs) is a natural
  extension once the PU study starts.
