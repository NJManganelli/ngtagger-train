# SmartPixels RISE talk

30-minute internal talk on the Tier-2 `digiRefit` campaign: refit approach, the
interactive step-by-step visualizer, the sign-bug story, refit-quality BDT, the
stage-4 coherent tagger matrix, and vertex-emulator R&D. Dark-sector physics is
backup-only.

**No running kernel is needed during the talk** — every cell is markdown; the
interactives are standalone HTML.

## Files

| file | what |
|---|---|
| `smartpixels_rise.ipynb` | the RISE notebook (slide metadata on every cell) |
| `smartpixels_rise.slides.html` | nbconvert reveal.js export — the no-dependency fallback (figures embedded; reveal.js itself loads from a CDN, so it wants network once) |
| `figures/` | committed PNGs used by the slides |
| `figures_src/` | scripts that regenerate the PNGs from the eval jsons (`pixi run python rise/figures_src/make_*.py`) |

## Presenting

### Option A — jupyterlab-rise (preferred)

```bash
cd ngtagger-train
pixi run jupyter lab rise/smartpixels_rise.ipynb
```

Open the notebook, then click the **Render with RISE** toolbar button
(or `Alt+R`). Navigate with Space / Shift+Space; subslides are down-arrow.

### Option B — the exported reveal.js HTML

Open `rise/smartpixels_rise.slides.html` in a browser (keep it inside `rise/`
so the relative iframe path to the visualizer resolves). Press `S` for the
reveal **speaker view** — that is where the per-slide speaker notes
(`slide_type: notes` cells) appear, with the timing map baked in.

### Before you start — checklist

1. **Serve the MVA explorer** (its binary model files cannot be fetched over
   `file://`):

   ```bash
   cd ngtagger-train
   python3 -m http.server 8000
   ```

   then `http://localhost:8000/eval_mva_explorer/site/explorer.html`.
2. **Open the refit-replay visualizer in its own browser tab**:
   `eval_refitq/refitviz/refit_replay.html` (self-contained ~17 MB, no server).
   The deck embeds both as iframes, but a dedicated tab is the reliable way to
   drive a demo (and the iframe may not resolve inside JupyterLab's preview).
3. Regenerate the visualizer if needed:
   `pixi run python eval_refitq/refitviz/make_refit_viz.py`.

## Speaker notes

Every slide has a companion cell with `slide_type: notes` containing the spoken
narrative, a `[mm:ss]` timing hint, and audience-calibration cues (what to
expand for undergrads / Muon Collider students vs what the CMS experts already
know). Read them in the reveal speaker view (`S`) or directly in JupyterLab.

## 30-minute timing map

| t | slide(s) |
|---|---|
| 0:00 | title |
| 0:30 | thesis + R&D-payoff framing (cheat-sheet subslide) |
| 1:30 | tier model (where Tier-2 came from) |
| 3:30 | Nano capability now (2 slides: tables/truth; coherent nanos) |
| 6:30 | digiRefit: the idea |
| 8:30 | **interactive: refit-replay visualizer** (centerpiece, ~3.5 min) |
| 12:00 | the sign-bug story (3 fragments) |
| 14:30 | post-fix resolution (d0 −46%, z0 −56%) |
| 16:00 | limitations (own slide, unhurried) |
| 17:30 | refit-quality BDT (15-config AUC band) |
| 19:30 | stage-4 tagger matrix (headline + caveat together) |
| 21:30 | coherent feature-level story |
| 22:30 | **interactive: MVA explorer** (first thing to cut if late) |
| 24:30 | vertex emulators |
| 26:00 | close: payoff + what's next (TS0 lever) |
| 27:00 | questions / if-time backup (dark sector) |

## Rebuilding

```bash
# figures (from eval_refitq jsons; resolution uses a committed number cache,
# add --recompute when the pF nanos are mounted)
pixi run python rise/figures_src/make_resolution_fig.py
pixi run python rise/figures_src/make_stage3_fig.py
pixi run python rise/figures_src/make_stage4_fig.py
pixi run python rise/figures_src/make_geometry_fig.py

# reveal export
pixi run jupyter nbconvert --to slides --embed-images \
    rise/smartpixels_rise.ipynb --output smartpixels_rise.slides
# nbconvert appends ".slides.html"; keep the committed name:
mv rise/smartpixels_rise.slides.slides.html rise/smartpixels_rise.slides.html
```
