"""SmartPixels digiRefit refit-replay VISUALIZER (5-par prompt framing).

Interactive, self-contained (kernel-free) Plotly figure showing a PROMPT digiRefit
L1Track Kalman refit step-by-step across the 3 angle modes (none / alpha /
alphaBeta) and the 15 SmartPixels layer configs (1000..1111), for a curated set
of example tracks.

The framing (mem:smartpixels-5par-framing-directive) is 5-par OT-only vs 5-par
OT+IT -- does adding the SmartPixels inner-tracker (IT) hits IMPROVE the 5-par
prompt track (impact parameter for b-tagging, vertexing). The 4-par pinned-d0 path
is DROPPED:
  * SEED  = 5-par OT-only PROMPT track = reference L1TTrack (real d0 + reduced fit
            chi2 chi2XYRed/chi2ZRed).
  * REFIT = 5-par OT+IT = prompt digiRefit variant L1TSmartPixelsTrackDigiRefit<CFG>.
  * OT stubs = the REAL embedded L1TTrackStub table (x/y/z, layer) -- not schematic.

Two-posture faithfulness (see docs/refit-replay-viz.md):
  * chi2 evolution & pulls: for the 4 PRODUCED configs (AIII/AAII/AAAI/AAAA at
    useAngles=alphaBeta) these come straight from the nano sidecar; for the other
    41 config x mode combinations they are the offline REPLAY.
  * parameter-state evolution: always the offline replay
    (:func:`ngtagger.viz.refit_replay.replay_track`). rInv / phi0 / d0 are
    reproduced faithfully in sign and order-of-magnitude scale; tanL / z0 hinge
    on the per-track trackCov correlations that nano does not persist and are
    labelled ILLUSTRATIVE.

The key readout is RESOLUTION vs TRUTH: the Kalman step table shows (param - truth)
for the 5-par OT-only seed and for the final OT+IT refit, so the viewer SEES whether
adding IT moves the track TOWARD truth. Truth is GenVtx(PV) on nano_pE (prompt-only;
swaps to matched-TP on nano_pF) -- see :mod:`ngtagger.viz._truth`.

Public entry point: :func:`build_refit_viz`.
"""

from __future__ import annotations

import numpy as np

from ngtagger.viz._curate import curate_tracks
from ngtagger.viz._dataio import (
    ALL_CONFIGS,
    ANGLE_MODES,
    OT_BARREL_RADII,
    PRODUCED_CONFIGS,
    barrel_stub_xy,
    config_active_layers,
    hit_class_name,
    load_nano,
)
from ngtagger.viz._kf import (
    _REPLAY_SEED_SIGMAS,
    hitset_chi2,
    reconstruct_hit_globals,
    replay_track,
)
from ngtagger.viz._truth import track_truth, truth_is_prompt_only

PARAM_NAMES = ["rInv", "phi0", "tanL", "z0", "d0"]
PARAM_UNITS = ["cm$^{-1}$", "rad", "", "cm", "cm"]
# Which replayed parameters are faithful vs illustrative (expert validation gate).
FAITHFUL_PARAMS = {"rInv", "phi0", "d0"}
ILLUSTRATIVE_PARAMS = {"tanL", "z0"}

# Colors
C_SEED = "#1f77b4"     # 5-par OT-only L1Track seed helix / seed state
C_REFIT = "#d62728"    # OT+IT digiRefit helix / final state
C_LAYER = "#888888"    # IT (SmartPixels TBPX) layer guides
C_OTLAYER = "#c2b280"  # OT barrel layer guides
C_OTSTUB = "#17becf"   # REAL OT stubs (L1TTrackStub x/y/z)
C_HIT_TRUE = "#2ca02c"    # selHitClass 0
C_HIT_WRONG = "#ff7f0e"   # selHitClass 1
C_HIT_NOISE = "#9467bd"   # selHitClass 2
_HIT_COLOR = {0: C_HIT_TRUE, 1: C_HIT_WRONG, 2: C_HIT_NOISE, -1: C_LAYER}

# Helix labels (round 4: NO user-visible "seed" -- an L1Track is itself built from
# a seed triplet, so calling it "seed" was confusing): the initial track is the
# "OT L1Track" = 5-par OT-only L1Track (L1TrackFinder fit on outer-tracker stubs +
# beamline, promptHnpar=5); refit = OT+IT (prompt digiRefit adds SmartPixels).
LABEL_SEED = "OT L1Track (5-par OT-only)"
LABEL_REFIT = "5-par OT+IT digiRefit (adds SmartPixels)"
R_OVERVIEW = 115.0     # overview panel outer radius [cm] (past OT-L6 = 108)
R_ZOOM = 18.0          # IT-zoom panel outer radius [cm]


# --------------------------------------------------------------------------
# helix drawing
# --------------------------------------------------------------------------
def _poca(rInv, phi0, d0):
    return d0 * np.sin(phi0), -d0 * np.cos(phi0)


def helix_points(rInv, phi0, tanL, z0, d0, rmax=17.5, npts=250):
    """Sample the helix from POCA outward to radius rmax. Returns x, y, z, r."""
    # CMSSW TTTrack sign convention: with the stored (signed) rInv, positive
    # curvature puts the centre of curvature to the RIGHT of the momentum
    # (phi0_hat rotated by -90deg), i.e. centre = POCA + R*(sin phi0, -cos phi0).
    # The bare (sin(phi0+psi))/rInv parametrisation below curls the OTHER way
    # (centre to the LEFT), which mirror-flips the transverse helix and makes it
    # peel away from its own OT stubs (verified: as-is the drawn helix misses the
    # nano stubs by up to ~25 cm on soft tracks; flipping the curl lands it on
    # them to <1 mm). Negate the signed curvature once so R, the centre, psimax
    # and the arc-length s all pick up the correct handedness. This is a pure
    # bending-plane sign; s = psi/rInv and z are unchanged (r-z projection is
    # unaffected), and _poca / the straight-line branch do not depend on it.
    rInv = -rInv
    x0, y0 = _poca(rInv, phi0, d0)
    if abs(rInv) < 1e-9:
        b = x0 * np.cos(phi0) + y0 * np.sin(phi0)
        c = x0 * x0 + y0 * y0 - rmax * rmax
        smax = -b + np.sqrt(max(b * b - c, 0.0))
        s = np.linspace(0.0, max(smax, 0.0), npts)
        x = x0 + s * np.cos(phi0)
        y = y0 + s * np.sin(phi0)
    else:
        R = 1.0 / rInv
        cx = x0 - R * np.sin(phi0)
        cy = y0 + R * np.cos(phi0)
        dcen = np.hypot(cx, cy)
        absR = abs(R)
        if rmax > dcen + absR:      # curler that never reaches rmax
            psimax = np.sign(rInv) * np.pi
        else:
            cosArg = (absR * absR + dcen * dcen - rmax * rmax) / (2.0 * absR * dcen)
            psimax = np.sign(rInv) * np.arccos(max(-1.0, min(1.0, cosArg)))
        psi = np.linspace(0.0, psimax, npts)
        x = x0 + (np.sin(phi0 + psi) - np.sin(phi0)) / rInv
        y = y0 - (np.cos(phi0 + psi) - np.cos(phi0)) / rInv
        s = psi / rInv
    z = z0 + tanL * s
    return x, y, z, np.hypot(x, y)


def _crossing_xy(rInv, phi0, tanL, z0, d0, R):
    """(x, y, z) where the helix crosses cylinder radius R, or None."""
    x, y, z, r = helix_points(rInv, phi0, tanL, z0, d0, rmax=R + 0.5, npts=1500)
    hit = np.where(r >= R)[0]
    if len(hit) == 0:
        return None
    i = hit[0]
    return float(x[i]), float(y[i]), float(z[i])


# --------------------------------------------------------------------------
# reduced-chi2 accounting (OT anchor + IT increments), per mem directive
# --------------------------------------------------------------------------
def _reduced_chi2_running(track, chi2_rz_steps, angle_mode):
    """Running REDUCED chi2 (chi2/ndof) starting from the OT-only L1Track fit.

    The nano L1TTrack_chi2XYRed / chi2ZRed are ALREADY reduced (chi2/ndof). We
    recover the OT absolute chi2 = reduced * ndof_OT with ndof_OT = 2*nStubs - 5
    (5-par helix), then add the IT r-z increments per accepted hit and re-divide by
    the growing ndof.

    ndof accounting (documented in docs/refit-replay-viz.md):
      * OT stubs contribute 2 measurements each (r-phi + r-z); 5-par fit -> subtract
        5 -> ndof_OT = 2*nStubs - 5.
      * each accepted IT hit adds 2 position measurements (local x, local y) and, in
        alpha/alphaBeta modes, up to 2 angle measurements (alpha, beta). We add the
        position dof always and the angle dof by mode.

    We drive the running REDUCED chi2 with the physically-scaled r-z channel (the
    r-phi increments are dominated by the parametrized-seed r-phi cov and are
    unphysically inflated; they are shown raw but NOT folded into the reduced total).
    Returns (labels-aligned running_reduced list of length len(steps)+1, ndof list).
    """
    n_stub = max(int(track.truth["nStubs"]), 3)
    ndof_ot = max(2 * n_stub - 5, 1)
    chi2_ot_abs = track.seed["chi2ZRed"] * ndof_ot            # OT z-fit absolute chi2

    running_abs = chi2_ot_abs
    ndof = ndof_ot
    per_hit_dof = 1 + (1 if angle_mode == "alphaBeta" else 0)  # r-z channels per IT hit
    red = [running_abs / ndof]
    ndofs = [ndof]
    for inc in chi2_rz_steps:
        running_abs += inc
        ndof += per_hit_dof
        red.append(running_abs / max(ndof, 1))
        ndofs.append(ndof)
    return red, ndofs


# --------------------------------------------------------------------------
# per-combo replay payload
# --------------------------------------------------------------------------
def _combo_payload(track, config, angle_mode, radii, seed_sigmas):
    """Compute everything a (track, config, angle) combo needs to draw.

    Returns a dict with seed/final params, per-step state table rows, resolution-to-
    truth deltas, chi2 evolution (raw increments + running REDUCED chi2 including the
    OT anchor), hit markers, and a real-vs-replay label. For the 4 produced configs at
    alphaBeta the chi2 & pulls are taken from the sidecar; the state trajectory is
    always the replay.
    """
    active = config_active_layers(config)
    seed = [track.seed[k] for k in PARAM_NAMES]
    # feed only the active layers that were actually crossed+accepted in AAAA
    hits = [track.hits[L] for L in sorted(track.hits) if active[L - 1]]

    rep = replay_track(seed, hits, layer_radii=radii, use_angles=angle_mode,
                       active_layers=active, param_sigmas=seed_sigmas, seed_npar=5)

    variant = PRODUCED_CONFIGS.get(config)
    is_produced = (variant is not None and angle_mode == "alphaBeta"
                   and track.real[variant]["refitPerformed"])

    # per-fed-layer step rows: OT L1Track (initial), after L_i, ... , final
    fed_layers = [L for L in sorted(track.hits) if active[L - 1]]
    step_states = [np.asarray(seed, float)] + list(rep["states"])
    step_labels = ["OT L1Track"] + [f"after L{L}" for L in fed_layers[: len(rep["states"])]]

    # chi2 evolution: prefer sidecar for produced-at-alphaBeta, else replay
    if is_produced:
        chi2_rphi_steps, chi2_rz_steps = [], []
        for L in fed_layers[: len(rep["states"])]:
            chi2_rphi_steps.append(track.hits[L]["chi2IncRPhi"])
            chi2_rz_steps.append(track.hits[L]["chi2IncRZ"])
    else:
        chi2_rphi_steps = list(rep["chi2_rphi"])
        chi2_rz_steps = list(rep["chi2_rz"])
    cum_rphi = np.cumsum([0.0] + chi2_rphi_steps)
    cum_rz = np.cumsum([0.0] + chi2_rz_steps)

    # running REDUCED chi2 including the OT-only anchor (mem directive #3)
    red_running, ndofs = _reduced_chi2_running(track, chi2_rz_steps, angle_mode)

    # resolution-to-truth: (param - truth) for the OT-only seed and the OT+IT refit.
    # For a PRODUCED config at alphaBeta use the REAL refit d0/z0 (the bit-exact
    # production answer) so the resolution-to-truth is faithful; else the replay
    # (whose d0 is faithful in sign, z0 illustrative).
    tr_truth = track_truth(track.seed, track.truth, track.genvtx)
    final = rep["final"]
    if is_produced:
        refit_d0 = track.real[variant]["d0"]
        refit_z0 = track.real[variant]["z0"]
    else:
        refit_d0 = final[4]
        refit_z0 = final[3]
    resid = dict(
        seed_d0=seed[4] - tr_truth["d0"], refit_d0=refit_d0 - tr_truth["d0"],
        seed_z0=seed[3] - tr_truth["z0"], refit_z0=refit_z0 - tr_truth["z0"],
        refit_d0_val=refit_d0, refit_z0_val=refit_z0, refit_real=is_produced,
        truth=tr_truth, prompt_only=truth_is_prompt_only(tr_truth),
    )

    # ---- config-wide FULL-HIT-SET reduced chi2 (round 4) ----
    # ONE fixed hit set per (track, config): the linked OT BARREL stubs + the
    # reconstructed SmartPixels hit positions on the config's ACTIVE layers.
    # EVERY displayed variant (the OT-only L1Track, each intermediate KF state,
    # the final refit) is charged against the SAME set with the SAME sigmas and
    # ONE common ndof, so the rows are directly comparable -- the OT L1Track is
    # now "charged" for the smartpix hits its fit never saw. See the block
    # comment in _kf.py for the exact channel content (OT r-phi + OT z + IT
    # local-x; the IT z channel is excluded as replay-illustrative).
    hit_globals = getattr(track, "_hitset_globals", None)
    if hit_globals is None:
        hit_globals = reconstruct_hit_globals(
            [track.seed[k] for k in PARAM_NAMES], list(track.hits.values()),
            layer_radii=radii, param_sigmas=seed_sigmas)
        track._hitset_globals = hit_globals
    hs_kw = dict(stubs=track.stubs, hit_globals=hit_globals,
                 active_layers=active, layer_radii=radii, pt=track.seed["pt"])
    hs_states = [hitset_chi2(st, **hs_kw) for st in step_states]
    full_ndof = max((hs_states[0]["n_meas"] if hs_states else 0) - 5, 1)
    full_red = [h["chi2"] / full_ndof for h in hs_states]
    if is_produced:
        refit_state = [track.real[variant][k] for k in PARAM_NAMES]
    else:
        refit_state = final
    full_refit_red = hitset_chi2(refit_state, **hs_kw)["chi2"] / full_ndof

    # covariance-evolution trajectory (per KF stage: seed, after-L1, ..., final).
    # traj_sigmas aligns 1:1 with step_states (both start at the seed). This is the
    # per-param 1sigma from the running 5x5 covariance -> it SHRINKS as IT hits fold in.
    step_sigmas = [np.asarray(sg, float) for sg in rep["traj_sigmas"]]

    # resolution-to-truth trajectory + %Δ toward/away from truth per KF step. For a
    # parameter p with truth p_t, res_i = |p_i - p_t|; the percent change vs the
    # previous step is 100*(res_i - res_{i-1})/res_{i-1} (NEGATIVE = moved TOWARD
    # truth, the good direction). Computed for d0 (index 4), z0 (3) and rInv (0).
    _RES_IDX = {"d0": 4, "z0": 3, "rInv": 0}
    _tru = {"d0": tr_truth["d0"], "z0": tr_truth["z0"], "rInv": tr_truth.get("rInv")}
    res_abs = {}      # name -> list over steps of |param - truth|
    res_pct = {}      # name -> list over steps of %Δ vs previous ("" for seed / n/a)
    for nm, j in _RES_IDX.items():
        t = _tru.get(nm)
        vals = [abs(st[j] - t) if t is not None else None for st in step_states]
        res_abs[nm] = vals
        pct = [None]
        for i in range(1, len(vals)):
            prev, cur = vals[i - 1], vals[i]
            if prev is None or cur is None or prev < 1e-12:
                pct.append(None)
            else:
                pct.append(100.0 * (cur - prev) / prev)
        res_pct[nm] = pct

    return dict(
        seed=np.asarray(seed, float),
        final=final,
        delta=rep["delta"],
        step_states=step_states,
        step_labels=step_labels,
        step_sigmas=step_sigmas,
        res_abs=res_abs,
        res_pct=res_pct,
        chi2_rphi_steps=chi2_rphi_steps,
        chi2_rz_steps=chi2_rz_steps,
        cum_rphi=cum_rphi,
        cum_rz=cum_rz,
        red_running=red_running,
        ndofs=ndofs,
        full_red=full_red,
        full_refit_red=full_refit_red,
        full_ndof=full_ndof,
        resid=resid,
        fed_layers=fed_layers[: len(rep["states"])],
        is_produced=is_produced,
        variant=variant,
        pulls=rep["pulls"],
    )


# --------------------------------------------------------------------------
# figure builder
# --------------------------------------------------------------------------
def _fmt(v, k):
    if k == "rInv":
        return f"{v:.2e}"
    if k in ("phi0", "tanL"):
        return f"{v:.4f}"
    return f"{v:.4f}"


# ---- Kalman step table (shared by the static default + the animation frames) ----
# The COVARIANCE is the protagonist (coordinator refinement #2): show the FULL per-step
# covariance DIAGONAL -- σrInv, σφ0, σtanL, σz0, σd0 (all 5 params, from traj_sigmas) --
# each visibly SHRINKING per KF stage. Alongside: the state (d0/z0/rInv), resolution-to-
# truth d0−tru, running reduced χ2, and (coordinator refinement #6) a compact per-row
# %Δd0→tru column = per-step %-change in |d0−truth| (−=toward truth), so the improvement-
# per-iteration reads across ALL rows, not only the current animated step. To keep width,
# z0−tru is dropped from the table (z0 value + σz0 stay; z0 resolution is in the banner).
# Round 4: +1 column "χ²/ndof all hits" = the config-wide FULL-HIT-SET reduced
# chi2 (same fixed hit set + same ndof for every row -> rows directly comparable;
# the OT L1Track row is charged for the smartpix hits its fit never saw).
# Readability (round 4): taller rows, right-aligned numerics, σz0/σd0 in µm
# (integers), σtanL in sci notation, z0 to 3 decimals.
_TABLE_HEADER = ["step", "d0", "d0−tru", "%Δd0", "z0", "rInv", "χ²red",
                 "χ²/ndof<br>all hits",
                 "σrInv", "σφ0", "σtanL", "σz0<br>µm", "σd0<br>µm"]
_TABLE_COLW = [1.2, 0.95, 0.95, 0.72, 0.85, 0.95, 0.66, 0.95,
               0.82, 0.78, 0.78, 0.62, 0.62]
# first column left-aligned, all numeric columns right-aligned
_TABLE_ALIGN = ["left"] + ["right"] * (len(_TABLE_HEADER) - 1)
# a faint blue tint on the 5 covariance columns so the "covariance block" is obvious
_SIG_COL_FILL = "#eaf1fb"
# a faint amber tint on the full-hit-set chi2 column (the round-4 comparator)
_FULL_COL_FILL = "#fdf6e3"
_FULL_COL_IDX = _TABLE_HEADER.index("χ²/ndof<br>all hits")


def _pct(v):
    return "" if v is None else f"{v:+.0f}%"


def _sig_fmt(v, k):
    """Compact per-param sigma. rInv/phi0/tanL tiny -> sci; z0/d0 -> µm ints."""
    if v is None or not np.isfinite(v):
        return "—"
    if k in ("rInv", "phi0", "tanL"):
        return f"{v:.1e}"
    return f"{v * 1e4:.0f}"           # z0/d0 [cm] -> µm


def _chi2_fmt(v):
    """Adaptive-precision chi2: 2dp small, 1dp medium, integer large, sci huge."""
    if v is None or not np.isfinite(v):
        return "—"
    if v < 10:
        return f"{v:.2f}"
    if v < 1000:
        return f"{v:.1f}"
    if v < 1e6:
        return f"{v:.0f}"
    return f"{v:.1e}"


def _table_values(pl, highlight_step=None):
    """Build (columns, rowcolors) for the Kalman step table from a combo payload.

    ``highlight_step`` (0-based over the OT-L1Track/after-Li stages) highlights the
    row for the KF stage currently reached by the animation; None -> the plain static
    table. Rows: OT L1Track, after-L1..Ln, REFIT (final), truth-reference footer. The
    "χ²/ndof all hits" column charges EVERY row against the SAME fixed full hit set
    (all OT barrel stubs + the active-layer SmartPixels hits) with one common ndof
    (shown in the footer), so the OT-only vs refit chi2 are directly comparable. The
    last 5 columns are the covariance-diagonal 1σ for (rInv, φ0, tanL, z0, d0),
    shrinking per stage.
    """
    res = pl["resid"]; tt = res["truth"]
    d0_true = tt["d0"]; z0_true = tt["z0"]
    step_states = pl["step_states"]; step_labels = pl["step_labels"]
    step_sigmas = pl["step_sigmas"]
    res_pct = pl["res_pct"]
    full_red = pl["full_red"]
    rows = []

    def _sig_cells(sg):
        # sg order matches the state: (rInv, phi0, tanL, z0, d0)
        return [_sig_fmt(sg[0], "rInv"), _sig_fmt(sg[1], "phi0"),
                _sig_fmt(sg[2], "tanL"), _sig_fmt(sg[3], "z0"), _sig_fmt(sg[4], "d0")]

    for i, (lab, st) in enumerate(zip(step_labels, step_states)):
        sg = step_sigmas[i] if i < len(step_sigmas) else np.full(5, np.nan)
        rows.append([lab, f"{st[4]:+.4f}", f"{st[4] - d0_true:+.4f}",
                     _pct(res_pct["d0"][i]), f"{st[3]:+.3f}",
                     _fmt(st[0], "rInv"), f"{pl['red_running'][i]:.2f}",
                     _chi2_fmt(full_red[i] if i < len(full_red) else None),
                     *_sig_cells(sg)])
    refit_tag = "REFIT*" if res["refit_real"] else "REFIT"
    rows.append([refit_tag, f"{res['refit_d0_val']:+.4f}", f"{res['refit_d0']:+.4f}",
                 "", f"{res['refit_z0_val']:+.3f}",
                 _fmt(pl['final'][0], "rInv"), f"{pl['red_running'][-1]:.2f}",
                 _chi2_fmt(pl["full_refit_red"]),
                 *_sig_cells(step_sigmas[-1])])
    srcs = f"{tt['d0_source']}"
    improved = abs(res["refit_d0"]) < abs(res["seed_d0"])
    arrow = "→toward truth" if improved else "→away"
    rows.append([f"truth[{srcs}]", f"{d0_true:+.4f}",
                 f"|Δ|{abs(res['seed_d0']):.3f}→{abs(res['refit_d0']):.3f}",
                 "d0 " + arrow, f"{z0_true:+.3f}",
                 "", "", f"ndof={pl['full_ndof']}", "", "", "", "", ""])
    n = len(rows)
    # seed row (light blue), interior white, refit row (light red), truth row (light green)
    rowcolors = (["#eef4fb"] + ["white"] * (n - 3) + ["#fbeeee", "#eefbf0"]
                 if n >= 3 else ["white"] * n)
    # animation: recolor the currently-reached KF stage row yellow so the eye tracks it
    if highlight_step is not None and 0 <= highlight_step < n:
        rowcolors = list(rowcolors)
        rowcolors[highlight_step] = "#fff3b0"
    # per-column fill: the 5 covariance columns get a faint blue tint (except the yellow-
    # highlighted / footer rows keep their row color). fill_color is per-column, each a
    # list over rows.
    col_fills = []
    for c in range(len(_TABLE_HEADER)):
        if c >= len(_TABLE_HEADER) - 5:      # the 5 sigma columns
            col_fills.append([_SIG_COL_FILL if rc == "white" else rc for rc in rowcolors])
        elif c == _FULL_COL_IDX:             # the full-hit-set chi2 comparator
            col_fills.append([_FULL_COL_FILL if rc == "white" else rc for rc in rowcolors])
        else:
            col_fills.append(list(rowcolors))
    cols = list(map(list, zip(*rows))) if rows else [[] for _ in _TABLE_HEADER]
    return cols, col_fills


def _make_table(pl, visible, highlight_step=None):
    import plotly.graph_objects as go
    cols, col_fills = _table_values(pl, highlight_step)
    return go.Table(
        columnwidth=_TABLE_COLW,
        header=dict(values=_TABLE_HEADER, fill_color="#34495e",
                    font=dict(color="white", size=9), align="center", height=30),
        cells=dict(values=cols, fill_color=col_fills,
                   font=dict(size=9), align=_TABLE_ALIGN, height=22),
        visible=visible)


def build_figure(data, picks, radii, seed_sigmas):
    """Assemble the self-contained interactive Plotly figure.

    Three plot panels + the Kalman step table (round-4 layout: the table sits in
    the upper-right quadrant; the two IT zooms share the bottom row side by side):
      (0,0) OVERVIEW (full radius to ~115 cm): the long-lever-arm picture — the
            OT barrel layers, the REAL OT stubs (L1TTrackStub x/y) that anchor the
            5-par OT-only L1Track way out at r~25-108 cm, the IT SmartPixels layers
            near the vertex, both helices, and the IT selected hits. Shows the OT
            L1Track is OT-anchored and the OT+IT refit adds inner-pixel constraints.
      (0,1) Kalman step table (state · resolution-to-truth · χ² columns).
      (1,0) IT-ZOOM r-z: OT L1Track vs refit (z, r) + IT hits.
      (1,1) IT-ZOOM r-φ (r<18 cm): the per-hit refit action (residuals, hit
            selection) at IT scale — next to the r-z zoom.
    Precomputes every (track x config x angle) combo into hidden trace groups;
    dropdowns just toggle visibility (no kernel). Returns a plotly Figure.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    tracks = [p[0] for p in picks]
    track_labels = [f"{arch}: ev{tr.event} trk{tr.idx} (pt={tr.seed['pt']:.1f})"
                    for tr, arch, _ in picks]

    combos = [(ti, cfg, am) for ti in range(len(tracks))
              for cfg in ALL_CONFIGS for am in ANGLE_MODES]
    # Default view = first (clean) track at the PRODUCED/real state config 1111 @
    # alphaBeta. This MUST match the animation's featured combo (same track + config)
    # so the always-visible animation overlay coincides with the static default panels
    # (coherent star/helix/stubs); the animation's final frame == this static refit.
    default_combo = combos.index((0, "1111", "alphaBeta"))

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "table"}],
               [{"type": "xy"}, {"type": "xy"}]],
        row_heights=[0.55, 0.45],
        column_widths=[0.5, 0.5],
        subplot_titles=(
            "OVERVIEW — OT-anchored L1Track + IT refit (long lever arm, r→115 cm)",
            "Kalman step table (state · truth resolution · χ² own-fit vs ALL-hits)",
            "IT zoom: r–z projection",
            "IT zoom: r-φ projection (x–y, r<18 cm)"),
        horizontal_spacing=0.09, vertical_spacing=0.11,
    )

    # ---- static geometry (drawn once, always visible) ----
    theta = np.linspace(0, 2 * np.pi, 220)
    # overview (row1,col1): OT barrel circles + IT circles
    for R in OT_BARREL_RADII:
        fig.add_trace(go.Scatter(x=R * np.cos(theta), y=R * np.sin(theta), mode="lines",
                                 line=dict(color=C_OTLAYER, width=1), hoverinfo="skip",
                                 showlegend=False), row=1, col=1)
    for R in radii:
        fig.add_trace(go.Scatter(x=R * np.cos(theta), y=R * np.sin(theta), mode="lines",
                                 line=dict(color=C_LAYER, width=1, dash="dot"), hoverinfo="skip",
                                 showlegend=False), row=1, col=1)
    # IT-zoom r-phi (row2,col2): IT circles
    for R in radii:
        fig.add_trace(go.Scatter(x=R * np.cos(theta), y=R * np.sin(theta), mode="lines",
                                 line=dict(color=C_LAYER, width=1, dash="dot"), hoverinfo="skip",
                                 showlegend=False), row=2, col=2)
    # IT-zoom r-z (row2,col1): IT layer lines
    for R in radii:
        fig.add_trace(go.Scatter(x=[-40, 40], y=[R, R], mode="lines",
                                 line=dict(color=C_LAYER, width=1, dash="dot"), hoverinfo="skip",
                                 showlegend=False), row=2, col=1)
    n_static = len(OT_BARREL_RADII) + 3 * len(radii)

    # per-combo trace groups (fixed count N_TRACES): overview(seed,refit,otstub,ithit),
    # zoom-rphi(seed,refit,ithit), zoom-rz(seed,refit,ithit), table.
    N_TRACES = 11
    trace_combo = []

    def _ithit_markers(tr, s, fed_layers, rz=False):
        xs, ys, cols, txt = [], [], [], []
        for L in fed_layers:
            cr = _crossing_xy(*s, R=radii[L - 1])
            if cr is None:
                continue
            cls = tr.hits[L]["selHitClass"]
            if rz:
                xs.append(cr[2]); ys.append(radii[L - 1])
            else:
                xs.append(cr[0]); ys.append(cr[1])
            cols.append(_HIT_COLOR.get(cls, C_LAYER))
            txt.append(f"IT L{L} {hit_class_name(cls)}<br>resX={tr.hits[L]['resX']:.3f} cm "
                       f"resY={tr.hits[L]['resY']:.3f} cm<br>winMult={tr.hits[L]['windowMult']}")
        return xs, ys, cols, txt

    def add_combo(ci, visible):
        ti, cfg, am = combos[ci]
        tr = tracks[ti]
        pl = _combo_payload(tr, cfg, am, radii, seed_sigmas)
        s = pl["seed"]; f = pl["final"]

        # REAL OT barrel stubs from the embedded L1TTrackStub table (x/y persisted).
        otx, oty, ottxt = [], [], []
        for (sxx, syy, srr, lab) in barrel_stub_xy(tr.stubs):
            otx.append(sxx); oty.append(syy)
            ottxt.append(f"{lab} real stub (r={srr:.1f} cm)")

        # ---- overview (row1,col1): full-radius, both helices to R_OVERVIEW ----
        sx, sy, _, _ = helix_points(*s, rmax=R_OVERVIEW)
        fx, fy, _, _ = helix_points(*f, rmax=R_OVERVIEW)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=C_SEED, width=2),
                                 name=LABEL_SEED, legendgroup="otl1", showlegend=True,
                                 visible=visible, hovertemplate=LABEL_SEED + "<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", line=dict(color=C_REFIT, width=2),
                                 name=LABEL_REFIT, legendgroup="refit", showlegend=True,
                                 visible=visible, hovertemplate=LABEL_REFIT + "<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=otx, y=oty, mode="markers",
                                 marker=dict(size=9, color=C_OTSTUB, symbol="square",
                                             line=dict(width=1, color="#0b6b74")),
                                 name="OT stub (real)", legendgroup="otstub", showlegend=True,
                                 visible=visible, text=ottxt,
                                 hovertemplate="%{text}<extra></extra>"), row=1, col=1)
        ohx, ohy, ohc, oht = _ithit_markers(tr, s, pl["fed_layers"])
        fig.add_trace(go.Scatter(x=ohx, y=ohy, mode="markers",
                                 marker=dict(size=9, color=ohc, symbol="x-thin",
                                             line=dict(width=3, color=ohc)),
                                 name="SmartPixels hit", legendgroup="hit", showlegend=True,
                                 visible=visible, text=oht,
                                 hovertemplate="%{text}<extra></extra>"), row=1, col=1)

        # ---- IT-zoom r-phi (row2,col2) ----
        zsx, zsy, _, _ = helix_points(*s, rmax=R_ZOOM)
        zfx, zfy, _, _ = helix_points(*f, rmax=R_ZOOM)
        fig.add_trace(go.Scatter(x=zsx, y=zsy, mode="lines", line=dict(color=C_SEED, width=2),
                                 showlegend=False, visible=visible,
                                 hovertemplate=LABEL_SEED + "<extra></extra>"), row=2, col=2)
        fig.add_trace(go.Scatter(x=zfx, y=zfy, mode="lines", line=dict(color=C_REFIT, width=2),
                                 showlegend=False, visible=visible,
                                 hovertemplate=LABEL_REFIT + "<extra></extra>"), row=2, col=2)
        hx, hy, hc, ht = _ithit_markers(tr, s, pl["fed_layers"])
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers",
                                 marker=dict(size=12, color=hc, symbol="x-thin",
                                             line=dict(width=3, color=hc)),
                                 showlegend=False, visible=visible, text=ht,
                                 hovertemplate="%{text}<extra></extra>"), row=2, col=2)

        # ---- IT-zoom r-z (row2,col1) ----
        _, _, szz, szr = helix_points(*s, rmax=R_ZOOM)
        _, _, fzz, fzr = helix_points(*f, rmax=R_ZOOM)
        fig.add_trace(go.Scatter(x=szz, y=szr, mode="lines", line=dict(color=C_SEED, width=2),
                                 showlegend=False, visible=visible,
                                 hovertemplate=LABEL_SEED + "<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=fzz, y=fzr, mode="lines", line=dict(color=C_REFIT, width=2),
                                 showlegend=False, visible=visible,
                                 hovertemplate=LABEL_REFIT + "<extra></extra>"), row=2, col=1)
        rzx, rzy, rzc, rzt = _ithit_markers(tr, s, pl["fed_layers"], rz=True)
        fig.add_trace(go.Scatter(x=rzx, y=rzy, mode="markers",
                                 marker=dict(size=12, color=rzc, symbol="x-thin",
                                             line=dict(width=3, color=rzc)),
                                 showlegend=False, visible=visible, hoverinfo="skip"),
                      row=2, col=1)

        # ---- step table (row1,col2): state + RESOLUTION-TO-TRUTH (+%Δ, +σ) + the
        # per-fit reduced chi2 + the config-wide full-hit-set χ²/ndof comparator ----
        # Built by the shared _make_table (same columns as the animation frames).
        fig.add_trace(_make_table(pl, visible), row=1, col=2)

        for _ in range(N_TRACES):
            trace_combo.append(ci)

    for ci in range(len(combos)):
        add_combo(ci, visible=(ci == default_combo))

    # ----------------------------------------------------------------------
    # ANIMATION (mem directive #4 + coordinator refinements): a Plotly SLIDER +
    # PLAY/PAUSE over the KF steps (seed -> after-L1 -> ... -> final). Each frame =
    # a full KF UPDATE (fold in one hit -> RE-ESTIMATE state + covariance), NOT merely
    # a hit appearing. The COVARIANCE is the visible protagonist: a shaded ±σd0 band
    # (r-φ) and ±σz0 band (r-z) around the refit helix visibly NARROW each step, and a
    # per-frame annotation states the Δ and σ-shrink.
    #
    # FEATURED TRACK == the DEFAULT-VIEW track (the first curated CLEAN track, ti of
    # default_combo). This is critical: the overlay is drawn always-visible on TOP of
    # the static default combo, so the animated track MUST equal the static default
    # track or the two mismatch (previously the biggest-d0 "mover" was overlaid on the
    # clean default -> star on the wrong branch, spurious V through the origin). The
    # clean track barely moves in position -- that is fine; the covariance bands + the
    # σ-diagonal table (σd0 ~300->12 µm) are the visible KF story. The mover and all
    # other tracks remain SECONDARY selectables via the track dropdown.
    #
    # LIMITATION (kernel-free): the overlay is fixed to the default track; it does NOT
    # re-target when the track dropdown changes (Plotly frames bind to fixed trace
    # indices and cannot be rebuilt without a kernel). Selecting another track shows its
    # STATIC panels correctly, but the animation keeps demonstrating the clean default.
    anim_am = "alphaBeta"
    anim_cfg = "1111"
    feat_ti = combos[default_combo][0]        # == the static default-view track (clean)
    feat_tr = tracks[feat_ti]
    feat_arch = f"{picks[feat_ti][1]} ev{feat_tr.event} trk{feat_tr.idx}"
    feat_pl = _combo_payload(feat_tr, anim_cfg, anim_am, radii, seed_sigmas)
    feat_fed = feat_pl["fed_layers"]
    n_steps = len(feat_pl["step_states"])              # seed + one per fed layer

    # VISUAL magnification for the ±σ bands (coordinator refinement #3: ONE honest
    # factor for BOTH bands, not two different fudge factors). The true σd0 (~300→12 µm)
    # and σz0 (~200→30 µm) are sub-mm and invisible on an 18 cm panel, so BOTH bands use
    # a SINGLE magnification chosen so the seed σd0 band half-width is a readable ~1.5 cm.
    # Every frame uses the same factor, so each band SHRINKS proportionally to its real σ
    # (faithful in RATIO, exaggerated only in absolute size). The one factor is stated
    # once in the legend + caption so it is not mistaken for real scale.
    _sig_d0_seed = float(feat_pl["step_sigmas"][0][4]) or 1e-3
    BAND_MAG = round(1.5 / _sig_d0_seed / 5.0) * 5.0 or 1.0   # single factor, rounded to 5

    def _anim_helix(state, rmax):
        x, y, z, r = helix_points(*state, rmax=rmax)
        return x, y, z, r

    def _sigma_band_xy(state, sig, rmax):
        """±sig band around the helix in the x-y (r-φ) plane: offset the helix curve
        perpendicular to its local direction by ±sig, returned as ONE closed polygon
        (out along +sig, back along -sig) for a filled Scatter. sig ~ σd0 (a transverse
        offset) so this is the ±σd0 uncertainty tube; it NARROWS as σd0 shrinks."""
        x, y, z, r = helix_points(*state, rmax=rmax)
        if len(x) < 2:
            return [], []
        dx = np.gradient(x); dy = np.gradient(y)
        nrm = np.hypot(dx, dy); nrm[nrm < 1e-12] = 1e-12
        # left normal (perpendicular)
        nx = -dy / nrm; ny = dx / nrm
        xp = x + sig * nx; yp = y + sig * ny
        xm = x - sig * nx; ym = y - sig * ny
        px = np.concatenate([xp, xm[::-1]])
        py = np.concatenate([yp, ym[::-1]])
        return px.tolist(), py.tolist()

    def _sigma_band_rz(state, sig, rmax):
        """±sig band in the r-z plane (sig ~ σz0, a z offset): shift z by ±sig at fixed
        r, as a filled polygon. Narrows as σz0 shrinks."""
        x, y, z, r = helix_points(*state, rmax=rmax)
        if len(z) < 2:
            return [], []
        zp = z + sig; zm = z - sig
        px = np.concatenate([zp, zm[::-1]])
        py = np.concatenate([r, r[::-1]])
        return px.tolist(), py.tolist()

    def _anim_hit_marker(state, k, rz=False):
        """The single IT hit being ADDED at KF step k (k==0 seed -> none)."""
        if k <= 0 or k - 1 >= len(feat_fed):
            return [], [], []
        L = feat_fed[k - 1]
        cr = _crossing_xy(*state, R=radii[L - 1])
        if cr is None:
            return [], [], []
        cls = feat_tr.hits[L]["selHitClass"]
        c = _HIT_COLOR.get(cls, C_LAYER)
        if rz:
            return [cr[2]], [radii[L - 1]], [c]
        return [cr[0]], [cr[1]], [c]

    def _step_annotation(k):
        """Per-frame headline: state re-estimated + Δ + covariance shrink (fix #3)."""
        st = feat_pl["step_states"][k]
        sg = feat_pl["step_sigmas"][k]
        tt = feat_pl["resid"]["truth"]
        d0_tru = tt["d0"]
        if k == 0:
            return (f"<b>OT L1Track (OT-only, before IT)</b>  ·  d0={st[4]:+.4f} cm  "
                    f"(|d0−tru|={abs(st[4]-d0_tru):.4f})  ·  "
                    f"<b>σd0={sg[4]*1e4:.0f} µm</b>  σz0={sg[3]*1e4:.0f} µm  "
                    f"— starting covariance")
        st_prev = feat_pl["step_states"][k - 1]
        sg_prev = feat_pl["step_sigmas"][k - 1]
        L = feat_fed[k - 1] if k - 1 < len(feat_fed) else "?"
        dd0 = st[4] - st_prev[4]
        pct = feat_pl["res_pct"]["d0"][k]
        toward = ("moved TOWARD truth" if (pct is not None and pct < 0)
                  else "moved away" if pct is not None else "")
        return (f"<b>KF UPDATE after L{L}</b>: fold hit → <b>re-estimate state + "
                f"covariance</b>  ·  Δd0={dd0*1e4:+.0f} µm ({toward})  ·  "
                f"<b>σd0 {sg_prev[4]*1e4:.0f}→{sg[4]*1e4:.0f} µm</b>  "
                f"σz0 {sg_prev[3]*1e4:.0f}→{sg[3]*1e4:.0f} µm (covariance tightens)")

    # index bookkeeping: remember where the animation overlay traces start so the
    # frames can target them by absolute trace index.
    anim_start = len(fig.data)
    s0 = feat_pl["step_states"][0]
    sg0 = feat_pl["step_sigmas"][0]
    # The animated "current KF state" helix + its ±σ bands use a DISTINCT color
    # (magenta C_ANIM) so it is clearly the moving element and does NOT mask the static
    # blue OT-only SEED or the static red OT+IT REFIT (coordinator refinement #4 --
    # keep seed vs refit distinguishable). The bands share the magenta family.
    C_ANIM = "#c71585"          # mediumvioletred -- the animated KF-state overlay
    BAND_FILL = "rgba(199,21,133,0.15)"
    # overlay build order (9 traces): rphi σd0-BAND, overview helix, overview hit,
    # rphi helix, rphi hit, rz σz0-BAND, rz helix, rz hit, table.
    # (the σ bands are drawn FIRST so the helix/hit render on top of the shaded tube.)
    bx, by = _sigma_band_xy(s0, sg0[4] * BAND_MAG, R_ZOOM)
    fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", fill="toself",
                             fillcolor=BAND_FILL, line=dict(width=0),
                             name=f"±σd0 band (×{BAND_MAG:.0f})", legendgroup="anim",
                             showlegend=True, hoverinfo="skip"), row=2, col=2)
    ox, oy, _, _ = _anim_helix(s0, R_OVERVIEW)
    fig.add_trace(go.Scatter(x=ox, y=oy, mode="lines",
                             line=dict(color=C_ANIM, width=2.5, dash="dash"),
                             name="KF state (animated)", legendgroup="anim", showlegend=True,
                             hovertemplate="animated KF state @ step<extra></extra>"),
                  row=1, col=1)
    hmx, hmy, hmc = _anim_hit_marker(s0, 0)
    fig.add_trace(go.Scatter(x=hmx, y=hmy, mode="markers",
                             marker=dict(size=16, color="#ffd000", symbol="star",
                                         line=dict(width=2, color=C_ANIM)),
                             name="IT hit added", legendgroup="anim", showlegend=True,
                             hoverinfo="skip"), row=1, col=1)
    zx, zy, _, _ = _anim_helix(s0, R_ZOOM)
    fig.add_trace(go.Scatter(x=zx, y=zy, mode="lines",
                             line=dict(color=C_ANIM, width=2.5, dash="dash"), showlegend=False,
                             hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=hmx, y=hmy, mode="markers",
                             marker=dict(size=18, color="#ffd000", symbol="star",
                                         line=dict(width=2, color=C_ANIM)),
                             showlegend=False, hoverinfo="skip"), row=2, col=2)
    rbx, rby = _sigma_band_rz(s0, sg0[3] * BAND_MAG, R_ZOOM)
    fig.add_trace(go.Scatter(x=rbx, y=rby, mode="lines", fill="toself",
                             fillcolor=BAND_FILL, line=dict(width=0),
                             name=f"±σz0 band (×{BAND_MAG:.0f})", legendgroup="anim",
                             showlegend=True, hoverinfo="skip"), row=2, col=1)
    _, _, rzz, rzr = _anim_helix(s0, R_ZOOM)
    fig.add_trace(go.Scatter(x=rzz, y=rzr, mode="lines",
                             line=dict(color=C_ANIM, width=2.5, dash="dash"), showlegend=False,
                             hoverinfo="skip"), row=2, col=1)
    hzx, hzy, hzc = _anim_hit_marker(s0, 0, rz=True)
    fig.add_trace(go.Scatter(x=hzx, y=hzy, mode="markers",
                             marker=dict(size=18, color="#ffd000", symbol="star",
                                         line=dict(width=2, color=C_ANIM)),
                             showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(_make_table(feat_pl, True, highlight_step=0), row=1, col=2)
    # absolute indices of the 9 animation traces
    A_RPHI_BAND = anim_start
    A_OV_HELIX, A_OV_HIT = anim_start + 1, anim_start + 2
    A_RPHI_HELIX, A_RPHI_HIT = anim_start + 3, anim_start + 4
    A_RZ_BAND = anim_start + 5
    A_RZ_HELIX, A_RZ_HIT = anim_start + 6, anim_start + 7
    A_TABLE = anim_start + 8
    n_anim = 9

    # the per-frame KF-update annotation lives just above the plot panels; each frame
    # rewrites its text. The full annotation list (bottom captions + this step
    # annotation) is patched into every frame AFTER the final layout is assembled (see
    # the _patch_frame_annotations call below) so frames never drop the captions.
    step_ann_base = dict(x=0.5, y=1.25, xref="paper", yref="paper",
                         showarrow=False, xanchor="center",
                         font=dict(size=12, color="#8B0000"),
                         bgcolor="rgba(255,243,176,0.9)", bordercolor="#d62728",
                         borderwidth=1, borderpad=4)
    step_ann = dict(step_ann_base, text=_step_annotation(0))

    # Build one Plotly FRAME per KF step. Each frame rewrites ONLY the 9 overlay traces
    # (by index) -> never clobbers the combo traces. (Frame layout.annotations patched later.)
    frames = []
    step_titles = []
    step_ann_texts = []
    for k in range(n_steps):
        st = feat_pl["step_states"][k]
        sg = feat_pl["step_sigmas"][k]
        bxk, byk = _sigma_band_xy(st, sg[4] * BAND_MAG, R_ZOOM)
        rbxk, rbyk = _sigma_band_rz(st, sg[3] * BAND_MAG, R_ZOOM)
        ovx, ovy, _, _ = _anim_helix(st, R_OVERVIEW)
        zpx, zpy, _, _ = _anim_helix(st, R_ZOOM)
        _, _, rzpz, rzpr = _anim_helix(st, R_ZOOM)
        mhx, mhy, mhc = _anim_hit_marker(st, k)
        mzx, mzy, mzc = _anim_hit_marker(st, k, rz=True)
        cols, col_fills = _table_values(feat_pl, highlight_step=k)
        lab = feat_pl["step_labels"][k]
        step_titles.append(lab)
        step_ann_texts.append(_step_annotation(k))
        frames.append(go.Frame(
            name=str(k),
            data=[
                go.Scatter(x=bxk, y=byk),
                go.Scatter(x=ovx, y=ovy),
                go.Scatter(x=mhx, y=mhy),
                go.Scatter(x=zpx, y=zpy),
                go.Scatter(x=mhx, y=mhy),
                go.Scatter(x=rbxk, y=rbyk),
                go.Scatter(x=rzpz, y=rzpr),
                go.Scatter(x=mzx, y=mzy),
                go.Table(cells=dict(values=cols, fill_color=col_fills)),
            ],
            traces=[A_RPHI_BAND, A_OV_HELIX, A_OV_HIT, A_RPHI_HELIX, A_RPHI_HIT,
                    A_RZ_BAND, A_RZ_HELIX, A_RZ_HIT, A_TABLE],
        ))
    fig.frames = frames

    def visibility(ci):
        vis = [True] * n_static
        vis += [trace_combo[k] == ci for k in range(len(trace_combo))]
        vis += [True] * n_anim          # animation overlay: always visible
        return vis

    def combo_id(ti, cfg, am):
        return combos.index((ti, cfg, am))

    # Kernel-free interaction. Plotly updatemenus are independent and stateless: a
    # button can only set a FULL visibility vector, it cannot read the other menus'
    # current selection. A combo owns all 7 of its traces, so a single "axis" menu
    # cannot compose with the others. The robust kernel-free design is therefore:
    #   * one AUTHORITATIVE (track|config|angle) menu that reaches every one of the
    #     len(combos) states exactly (grouped label so it reads track-first);
    #   * three CONVENIENCE menus (track / config / angle) whose buttons jump to that
    #     selection with the other two axes reset to a sensible default
    #     (config=1111, angle=alphaBeta = the produced/real state) so the common
    #     "show me config X" / "show me track Y" clicks are one action.
    # The authoritative menu is the source of truth documented in the viz doc.
    DEF_CFG, DEF_AM = "1111", "alphaBeta"
    tracks_menu = dict(
        buttons=[dict(label=track_labels[ti], method="update",
                      args=[{"visible": visibility(combo_id(ti, DEF_CFG, DEF_AM))}])
                 for ti in range(len(tracks))],
        direction="down", showactive=True, x=0.0, xanchor="left", y=1.13, yanchor="top",
        pad={"r": 4, "t": 4}, bgcolor="#eef4fb",
    )
    config_menu = dict(
        buttons=[dict(label=f"config {cfg}" + (" *" if cfg in PRODUCED_CONFIGS else ""),
                      method="update",
                      args=[{"visible": visibility(combo_id(0, cfg, DEF_AM))}])
                 for cfg in ALL_CONFIGS],
        direction="down", showactive=True, x=0.22, xanchor="left", y=1.13, yanchor="top",
        pad={"r": 4, "t": 4}, bgcolor="#f0f0f0",
    )
    angle_menu = dict(
        buttons=[dict(label=f"angles: {am}", method="update",
                      args=[{"visible": visibility(combo_id(0, DEF_CFG, am))}])
                 for am in ANGLE_MODES],
        direction="down", showactive=True, x=0.44, xanchor="left", y=1.13, yanchor="top",
        pad={"r": 4, "t": 4}, bgcolor="#eefbf0",
    )
    arch_by_ti = [p[1] for p in picks]
    full_menu = dict(
        buttons=[dict(label=f"{arch_by_ti[ti]} | {cfg} | {am}",
                      method="update",
                      args=[{"visible": visibility(combo_id(ti, cfg, am))}])
                 for (ti, cfg, am) in combos],
        direction="down", showactive=True, x=0.66, xanchor="left", y=1.13, yanchor="top",
        pad={"r": 4, "t": 4}, bgcolor="#f7f0fb",
    )
    # ---- animation controls: Play (loop) / Pause / single-step buttons + a
    # KF-step slider ----
    # Frame timing: ~2.5 s per KF update so each covariance shrink is readable
    # (mem directive #4). The native Play runs the frames ONCE; a small injected JS
    # snippet (see build_refit_viz) turns Play into a CONTINUOUS LOOP until Pause.
    # Round 4: "◀ Step" / "Step ▶" move exactly one KF step back/forward (clamped
    # at the ends; stepping pauses a running loop first). They use method="skip"
    # -- entirely driven by the same injected JS, which also keeps the slider's
    # active position in sync.
    FRAME_MS = 2500
    TRANS_MS = 600
    _play_args = [None, {"frame": {"duration": FRAME_MS, "redraw": True},
                         "fromcurrent": True, "mode": "immediate",
                         "transition": {"duration": TRANS_MS}}]
    _pause_args = [[None], {"frame": {"duration": 0, "redraw": False},
                            "mode": "immediate", "transition": {"duration": 0}}]
    anim_buttons = dict(
        type="buttons", direction="right", showactive=False,
        x=0.0, xanchor="left", y=-0.055, yanchor="top", pad={"r": 6, "t": 4},
        buttons=[
            dict(label="▶ Play (loop)", method="animate", args=_play_args),
            dict(label="⏸ Pause", method="animate", args=_pause_args),
            dict(label="◀ Step", method="skip"),
            dict(label="Step ▶", method="skip"),
        ],
    )
    fig.update_layout(updatemenus=[tracks_menu, config_menu, angle_menu, full_menu,
                                   anim_buttons])

    # KF-step slider: each slider step jumps to that frame (seed / after-Lk / final).
    slider_steps = []
    for k, lab in enumerate(step_titles):
        slider_steps.append(dict(
            label=lab, method="animate",
            args=[[str(k)], {"frame": {"duration": 0, "redraw": True},
                             "mode": "immediate", "transition": {"duration": 0}}],
        ))
    fig.update_layout(sliders=[dict(
        active=0, x=0.34, xanchor="left", y=-0.055, yanchor="top", len=0.64,
        pad={"t": 4, "b": 4},
        currentvalue=dict(prefix="KF step: ", visible=True,
                          font=dict(size=11, color="#d62728")),
        steps=slider_steps,
    )])

    # axis cosmetics (round-4 layout: table upper-right; the two IT zooms share the
    # bottom row -> r-z = axes 2 (row2,col1), r-phi = axes 3 (row2,col2))
    # overview (row1,col1): full radius, equal aspect
    fig.update_xaxes(title_text="x [cm]", row=1, col=1, range=[-R_OVERVIEW, R_OVERVIEW],
                     scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="y [cm]", row=1, col=1, range=[-R_OVERVIEW, R_OVERVIEW])
    # IT-zoom r-z (row2,col1) -- keeps the round-3 r range (0, 18)
    fig.update_xaxes(title_text="z [cm]", row=2, col=1, range=[-30, 30])
    fig.update_yaxes(title_text="r [cm]", row=2, col=1, range=[0, R_ZOOM])
    # IT-zoom r-phi (row2,col2)
    fig.update_xaxes(title_text="x [cm]", row=2, col=2, range=[-R_ZOOM, R_ZOOM],
                     scaleanchor="y3", scaleratio=1)
    fig.update_yaxes(title_text="y [cm]", row=2, col=2, range=[-R_ZOOM, R_ZOOM])

    # Vertical layout zones (paper coords), each in its OWN band, nothing overlapping:
    #   title 1.36 · KF-step status 1.25 · menu labels 1.175 · menus 1.13
    #   · PLOTS [0..1] · anim controls (buttons+slider) -0.05 · captions -0.15/-0.20/-0.24
    # LEGEND (coordinator refinement #2): moved OUT of the dropdown row into the RIGHT
    # margin as a VERTICAL legend (x≈1.01), so it can never overlap the track/config/
    # angle/full-combo selectors along the top. The right margin is widened to hold it.
    # The big bottom margin (b=340) reserves a clear band for the controls AND the
    # trimmed captions with generous separation (coordinator refinement #1).
    fig.update_layout(
        height=1290, width=1330,
        margin=dict(t=250, b=340, l=60, r=210),
        legend=dict(orientation="v", x=1.012, y=1.0, yanchor="top", xanchor="left",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc",
                    borderwidth=1, font=dict(size=10)),
        annotations=list(fig.layout.annotations) + [
            dict(text=("<b>SmartPixels digiRefit replay</b>  —  5-par OT-only vs 5-par OT+IT: "
                       "does adding SmartPixels improve the prompt track?"),
                 x=0.5, y=1.36, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=15), xanchor="center"),
            dict(text="track", x=0.0, y=1.175, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11), xanchor="left"),
            dict(text="config (* = produced/real)", x=0.22, y=1.175, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11), xanchor="left"),
            dict(text="angle mode", x=0.44, y=1.175, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11), xanchor="left"),
            dict(text="full combo (authoritative)", x=0.66, y=1.175, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11), xanchor="left"),
            # KF-step STATUS line (updated per animation frame; see frame patch below).
            step_ann,
            # --- captions below the controls, trimmed hard; detail is in hovertext ---
            # Animation caption (the ▶/⏸/◀ Step/Step ▶ + slider band sits at y≈-0.05).
            dict(text=(f"<b>▶ Play / ⏸ Pause / ◀ Step / Step ▶</b> + slider — steps the KF on the "
                       f"DEFAULT-view track <b>{feat_arch}</b> "
                       "(<span style='color:#c71585'>magenta</span> = animated "
                       "KF state, gold ★ = hit added). Each frame = one full KF UPDATE (fold hit → "
                       f"re-estimate state + covariance); the ±σ bands (×{BAND_MAG:.0f} visual) "
                       "NARROW. Play loops until Pause; Step moves one KF step (clamped)."),
                 x=0.5, y=-0.15, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=10), xanchor="center"),
            # Framing caption (OT L1Track vs refit + truth + the full-hit-set χ2 column).
            dict(text=("<span style='color:#1f77b4'>OT L1Track</span> = 5-par OT-only (OT stubs "
                       "r≈25–108 cm) · <span style='color:#d62728'>refit</span> = +SmartPixels IT "
                       "(r&lt;16 cm). Truth = matched-TP <b>tp_d0/tp_z0</b> (nano_pF). "
                       "<b>χ²/ndof all hits</b> = every row charged against the SAME fixed hit set "
                       "(all OT barrel stubs + this config's active-layer SPX hits; OT rφ+z and IT "
                       "local-x channels, common ndof in the footer) — so the OT L1Track pays for "
                       "the SPX hits its fit never saw, and a refit that truly helps LOWERS it."),
                 x=0.5, y=-0.20, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=10), xanchor="center"),
            # Fidelity key + single-factor + kernel-free coupling note.
            dict(text=("<span style='color:#d62728'>REAL*</span> = produced (AIII/AAII/AAAI/AAAA "
                       f"@ alphaBeta) · else <span style='color:#7f7f7f'>REPLAY</span>. Both ±σ "
                       f"bands use ONE ×{BAND_MAG:.0f} visual factor. NOTE: the animation always "
                       "demonstrates the default track (kernel-free); selecting another track "
                       "updates only the static panels."),
                 x=0.5, y=-0.24, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=9.5), xanchor="center"),
        ],
    )

    # Patch every frame's layout.annotations to the FULL final set (bottom captions +
    # the KF-step status annotation with that frame's text) so the animation never drops
    # the captions and the status line updates per step. The status annotation is the
    # LAST entry appended above; we rebuild the list per frame swapping only its text.
    final_anns = list(fig.layout.annotations)
    # find the status annotation (the one we tagged with the red border + yellow bg)
    status_i = None
    for i, a in enumerate(final_anns):
        if getattr(a, "bordercolor", None) == "#d62728" and getattr(a, "bgcolor", None):
            status_i = i
            break
    if status_i is not None:
        import copy
        for fr, txt in zip(fig.frames, step_ann_texts):
            anns = copy.deepcopy(final_anns)
            anns[status_i]["text"] = txt
            fr.layout = go.Layout(annotations=anns)

    # surface the animation timing/frame-count so build_refit_viz can size the
    # continuous-loop JS (setInterval cadence == the per-step Play duration).
    fig._refit_anim = {"frame_ms": FRAME_MS, "n_frames": n_steps}
    return fig


def _title(pick):
    tr, arch, reason = pick
    truth = tr.truth
    badge_true = "GENUINE" if truth["genuine"] else ("looseGen" if truth["looselyGenuine"] else "FAKE/unmatched")
    hard = "hardInt" if truth["fromHard"] else "PU/other"
    # Identity + truth only (the static REAL/REPLAY fidelity key is a caption
    # under the table, so the title stays one clean line clear of the menus).
    return (f"<b>SmartPixels digiRefit replay</b>  ·  [{arch}] ev{tr.event} trk{tr.idx}"
            f"<span style='font-size:12px'>  ·  truth: {badge_true} ({hard}, "
            f"d0={tr.seed['d0']:+.3f}, tpPt={truth['tpPt']:.1f}, nStubs={truth['nStubs']})</span>")


# --------------------------------------------------------------------------
# continuous-loop-until-pause + single-step JS (standalone / kernel-free)
# --------------------------------------------------------------------------
# Plotly's native ▶ Play runs the frames only ONCE. For a self-contained HTML we
# inject a tiny controller that (1) finds the Plotly graph div, (2) runs a
# setInterval loop stepping Plotly.animate frame-by-frame and WRAPPING AROUND so it
# loops forever, (3) hooks the rendered Plotly updatemenu buttons (matched by their
# label text in the SVG) with capture-phase click listeners: ▶ starts the JS loop
# (and stops the native single-pass animate via stopPropagation), ⏸ clears the
# interval, and (round 4) ◀ Step / Step ▶ pause the loop then move EXACTLY one KF
# step back/forward, CLAMPED at the ends (no wrap). Every JS-driven frame change
# also relayouts sliders[0].active, so the slider position, the loop and the step
# buttons all share one source of truth. The slider keeps working natively (a drag
# sets the frame + active index, which loop/steps resume from). No kernel, no
# external requests.
_LOOP_JS = """
<script>
(function(){
  var FRAME_MS = __FRAME_MS__, NFR = __NFRAMES__;
  function gd(){ return document.querySelector('div.js-plotly-plot'); }
  function frameNames(g){
    var f = (g && g._transitionData && g._transitionData._frames) || [];
    return f.filter(function(x){return x && x.name!=null;}).map(function(x){return x.name;});
  }
  var timer = null, idx = 0;
  function curIdx(g){
    // resume from wherever the slider / last frame left off
    var s = g && g.layout && g.layout.sliders && g.layout.sliders[0];
    if (s && typeof s.active === 'number') return s.active;
    return idx;
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } }
  function gotoFrame(g, i, wrap){
    var names = frameNames(g); if(!names.length) return;
    if(wrap){ i = ((i % names.length) + names.length) % names.length; }
    else { i = Math.max(0, Math.min(names.length - 1, i)); }   // clamp at the ends
    idx = i;
    Plotly.animate(g, [names[i]],
      {mode:'immediate', frame:{duration:0, redraw:true}, transition:{duration:0}});
    // keep the slider position in sync (shared state for loop + step buttons)
    Plotly.relayout(g, {'sliders[0].active': i});
  }
  function start(g){
    stop();
    var names = frameNames(g); if(!names.length) return;
    idx = curIdx(g);
    // show the current frame immediately, then advance (wrapping) on the interval
    gotoFrame(g, idx, true);
    timer = setInterval(function(){ gotoFrame(gd(), idx + 1, true); }, FRAME_MS);
  }
  function step(delta){
    var g = gd(); if(!g) return;
    stop();                                  // stepping while playing pauses first
    gotoFrame(g, curIdx(g) + delta, false);  // clamped, no wrap
  }
  function hook(){
    var g = gd(); if(!g){ return setTimeout(hook, 200); }
    // find the ▶ Play / ⏸ Pause / ◀ Step / Step ▶ updatemenu buttons by label text
    var btns = g.querySelectorAll('g.updatemenu-button');
    var wired = 0;
    btns.forEach(function(b){
      var t = (b.textContent||'').trim();
      var fn = null;
      if(t.indexOf('Play') !== -1){ fn = function(){ start(gd()); }; }
      else if(t.indexOf('Pause') !== -1){ fn = stop; }
      else if(t.indexOf('Step') !== -1 && t.indexOf('\\u25c0') !== -1){  // ◀ Step
        fn = function(){ step(-1); };
      }
      else if(t.indexOf('Step') !== -1){ fn = function(){ step(1); }; }  // Step ▶
      if(fn){
        b.addEventListener('click', function(ev){
          ev.stopPropagation(); ev.preventDefault(); fn();
        }, true);
        wired++;
      }
    });
    if(wired < 4){ return setTimeout(hook, 200); }   // retry until all four wired
  }
  if(document.readyState === 'complete' || document.readyState === 'interactive'){
    setTimeout(hook, 300);
  } else {
    window.addEventListener('load', function(){ setTimeout(hook, 300); });
  }
})();
</script>
"""


def _inject_loop_js(html, frame_ms, n_frames):
    js = (_LOOP_JS.replace("__FRAME_MS__", str(int(frame_ms)))
                  .replace("__NFRAMES__", str(int(n_frames))))
    if "</body>" in html:
        return html.replace("</body>", js + "\n</body>", 1)
    return html + js


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def build_refit_viz(nano_file, tracks=None, out_html=None,
                    layer_radii=(3.0, 6.8, 10.9, 16.0),
                    seed_sigmas=_REPLAY_SEED_SIGMAS,
                    n_each=(2, 1, 1, 1), max_events=6, return_fig=False):
    """Build the standalone interactive refit-replay HTML.

    Parameters
    ----------
    nano_file : path to the SmartPixels nano file (read-only).
    tracks : optional list of (event, idx) tuples to force-select; otherwise the
        curated archetypes (clean/wrong/displaced/fake) are chosen automatically.
    out_html : path to write the self-contained HTML (include_plotlyjs=True). If
        None, the HTML is not written.
    layer_radii : TBPX mean layer radii [cm] (projector convention).
    seed_sigmas : parametrized seed-covariance sqrt-diagonal (documented caveat:
        production used trackCov, not persisted in nano).
    n_each : (n_clean, n_wrong, n_displaced, n_fake) archetypes to curate when
        tracks is None (a 3-tuple maps to clean/wrong/fake with 0 displaced).

    Returns
    -------
    dict with keys 'html_path', 'picks' (list of (event, idx, archetype)), and
    'figure' (only if return_fig).
    """
    data = load_nano(nano_file, layer_radii=layer_radii, max_events=max_events)

    if tracks is not None:
        by_key = {(t.event, t.idx): t for t in data.tracks}
        picks = []
        for (e, i) in tracks:
            tr = by_key[(e, i)]
            if not tr.truth["genuine"]:
                arch = "fake"
            elif abs(tr.seed["d0"]) > 0.1:
                arch = "displaced"
            elif len(tr.hits) == 4:
                arch = "clean"
            else:
                arch = "wrong"
            picks.append((tr, arch, "user-selected"))
    else:
        picks = curate_tracks(data, n_each=n_each)

    fig = build_figure(data, picks, np.asarray(layer_radii, float), np.asarray(seed_sigmas, float))

    result = {"picks": [(p[0].event, p[0].idx, p[1]) for p in picks], "html_path": None}
    if out_html is not None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
        html = fig.to_html(include_plotlyjs=True, full_html=True,
                           config={"displaylogo": False})
        anim = getattr(fig, "_refit_anim", {"frame_ms": 2500, "n_frames": 1})
        html = _inject_loop_js(html, anim["frame_ms"], anim["n_frames"])
        with open(out_html, "w") as fh:
            fh.write(html)
        result["html_path"] = out_html
    if return_fig:
        result["figure"] = fig
    return result
