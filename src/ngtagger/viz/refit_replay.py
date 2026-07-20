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
from ngtagger.viz._kf import _REPLAY_SEED_SIGMAS, replay_track
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

# Helix labels: seed = 5-par OT-only L1Track (L1TrackFinder fit on outer-tracker
# stubs + beamline, promptHnpar=5); refit = OT+IT (prompt digiRefit adds SmartPixels).
LABEL_SEED = "5-par OT-only L1Track (seed)"
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

    # per-fed-layer step rows: seed, after L_i, ... , final
    fed_layers = [L for L in sorted(track.hits) if active[L - 1]]
    step_states = [np.asarray(seed, float)] + list(rep["states"])
    step_labels = ["seed"] + [f"after L{L}" for L in fed_layers[: len(rep["states"])]]

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
    return dict(
        seed=np.asarray(seed, float),
        final=final,
        delta=rep["delta"],
        step_states=step_states,
        step_labels=step_labels,
        chi2_rphi_steps=chi2_rphi_steps,
        chi2_rz_steps=chi2_rz_steps,
        cum_rphi=cum_rphi,
        cum_rz=cum_rz,
        red_running=red_running,
        ndofs=ndofs,
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


def build_figure(data, picks, radii, seed_sigmas):
    """Assemble the self-contained interactive Plotly figure.

    Three plot panels + the Kalman step table:
      (0,0) OVERVIEW (full radius to ~115 cm): the long-lever-arm picture — the
            OT barrel layers, the REAL OT stubs (L1TTrackStub x/y) that anchor the
            5-par OT-only seed way out at r~25-108 cm, the IT SmartPixels layers near
            the vertex, both helices, and the IT selected hits. Shows the seed is an
            OT-anchored track and the OT+IT refit adds inner-pixel constraints.
      (0,1) IT-ZOOM r-φ (r<18 cm): the per-hit refit action (residuals, hit
            selection) at IT scale.
      (1,0) IT-ZOOM r-z: seed vs refit (z, r) + IT hits.
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
    default_combo = 0  # first track, first config (1000), first mode (none)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "table"}]],
        row_heights=[0.58, 0.42],
        column_widths=[0.5, 0.5],
        subplot_titles=(
            "OVERVIEW — OT-anchored seed + IT refit (long lever arm, r→115 cm)",
            "IT zoom: r-φ projection (x–y, r<18 cm)",
            "IT zoom: r–z projection",
            "Kalman step table (state · resolution-to-truth · running reduced χ2)"),
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
    # IT-zoom r-phi (row1,col2): IT circles
    for R in radii:
        fig.add_trace(go.Scatter(x=R * np.cos(theta), y=R * np.sin(theta), mode="lines",
                                 line=dict(color=C_LAYER, width=1, dash="dot"), hoverinfo="skip",
                                 showlegend=False), row=1, col=2)
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
        first = (ci == default_combo)

        # REAL OT barrel stubs from the embedded L1TTrackStub table (x/y persisted).
        otx, oty, ottxt = [], [], []
        for (sxx, syy, srr, lab) in barrel_stub_xy(tr.stubs):
            otx.append(sxx); oty.append(syy)
            ottxt.append(f"{lab} real stub (r={srr:.1f} cm)")

        # ---- overview (row1,col1): full-radius, both helices to R_OVERVIEW ----
        sx, sy, _, _ = helix_points(*s, rmax=R_OVERVIEW)
        fx, fy, _, _ = helix_points(*f, rmax=R_OVERVIEW)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=C_SEED, width=2),
                                 name=LABEL_SEED, legendgroup="seed", showlegend=first,
                                 visible=visible, hovertemplate=LABEL_SEED + "<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", line=dict(color=C_REFIT, width=2),
                                 name=LABEL_REFIT, legendgroup="refit", showlegend=first,
                                 visible=visible, hovertemplate=LABEL_REFIT + "<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=otx, y=oty, mode="markers",
                                 marker=dict(size=9, color=C_OTSTUB, symbol="square",
                                             line=dict(width=1, color="#0b6b74")),
                                 name="OT stub (real)", legendgroup="otstub", showlegend=first,
                                 visible=visible, text=ottxt,
                                 hovertemplate="%{text}<extra></extra>"), row=1, col=1)
        ohx, ohy, ohc, oht = _ithit_markers(tr, s, pl["fed_layers"])
        fig.add_trace(go.Scatter(x=ohx, y=ohy, mode="markers",
                                 marker=dict(size=9, color=ohc, symbol="x-thin",
                                             line=dict(width=3, color=ohc)),
                                 name="SmartPixels hit", legendgroup="hit", showlegend=first,
                                 visible=visible, text=oht,
                                 hovertemplate="%{text}<extra></extra>"), row=1, col=1)

        # ---- IT-zoom r-phi (row1,col2) ----
        zsx, zsy, _, _ = helix_points(*s, rmax=R_ZOOM)
        zfx, zfy, _, _ = helix_points(*f, rmax=R_ZOOM)
        fig.add_trace(go.Scatter(x=zsx, y=zsy, mode="lines", line=dict(color=C_SEED, width=2),
                                 showlegend=False, visible=visible,
                                 hovertemplate=LABEL_SEED + "<extra></extra>"), row=1, col=2)
        fig.add_trace(go.Scatter(x=zfx, y=zfy, mode="lines", line=dict(color=C_REFIT, width=2),
                                 showlegend=False, visible=visible,
                                 hovertemplate=LABEL_REFIT + "<extra></extra>"), row=1, col=2)
        hx, hy, hc, ht = _ithit_markers(tr, s, pl["fed_layers"])
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers",
                                 marker=dict(size=12, color=hc, symbol="x-thin",
                                             line=dict(width=3, color=hc)),
                                 showlegend=False, visible=visible, text=ht,
                                 hovertemplate="%{text}<extra></extra>"), row=1, col=2)

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

        # ---- step table (row2,col2): state + RESOLUTION-TO-TRUTH + running reduced chi2 ----
        # d0-truth / z0-truth (constant across steps); (param - truth) per row shows
        # whether the refit moves TOWARD truth. Running REDUCED chi2 starts from the
        # OT-only anchor (seed row) and grows as IT layers add.
        res = pl["resid"]; tt = res["truth"]
        d0_true = tt["d0"]; z0_true = tt["z0"]
        header_vals = ["step", "d0", "d0−tru", "z0", "z0−tru",
                       "rInv", "phi0", "χ2red(OT+IT)", "Σχ2(rz)", "Σχ2(rφ)"]
        rows = []
        for i, (lab, st) in enumerate(zip(pl["step_labels"], pl["step_states"])):
            rows.append([lab, f"{st[4]:+.4f}", f"{st[4] - d0_true:+.4f}",
                         f"{st[3]:+.4f}", f"{st[3] - z0_true:+.4f}",
                         _fmt(st[0], "rInv"), f"{st[1]:.4f}",
                         f"{pl['red_running'][i]:.2f}",
                         f"{pl['cum_rz'][i]:.2f}", f"{pl['cum_rphi'][i]:.1f}"])
        refit_tag = "REFIT*" if res["refit_real"] else "REFIT"
        rows.append([refit_tag, f"{res['refit_d0_val']:+.4f}", f"{res['refit_d0']:+.4f}",
                     f"{res['refit_z0_val']:+.4f}", f"{res['refit_z0']:+.4f}",
                     _fmt(f[0], "rInv"), f"{f[1]:.4f}",
                     f"{pl['red_running'][-1]:.2f}",
                     f"{pl['cum_rz'][-1]:.2f}", f"{pl['cum_rphi'][-1]:.1f}"])
        # truth-reference footer row so |seed−tru| vs |refit−tru| is readable at a glance
        srcs = f"{tt['d0_source']}"
        improved = abs(res["refit_d0"]) < abs(res["seed_d0"])
        arrow = "→ toward truth" if improved else "→ away"
        rows.append([f"truth[{srcs}]", f"{d0_true:+.4f}", f"|Δ|:{abs(res['seed_d0']):.3f}→{abs(res['refit_d0']):.3f}",
                     f"{z0_true:+.4f}", f"|Δ|:{abs(res['seed_z0']):.3f}→{abs(res['refit_z0']):.3f}",
                     "d0 " + arrow, "", "", "", ""])
        cols = list(map(list, zip(*rows))) if rows else [[] for _ in header_vals]
        n = len(rows)
        # seed row (light blue), interior white, refit row (light red), truth row (light green)
        rowcolors = ["#eef4fb"] + ["white"] * (n - 3) + ["#fbeeee", "#eefbf0"] if n >= 3 else ["white"] * n
        fig.add_trace(go.Table(
            columnwidth=[1.3, 1, 1, 1, 1, 1, 1, 1.2, 1, 1],
            header=dict(values=header_vals, fill_color="#34495e",
                        font=dict(color="white", size=9), align="center"),
            cells=dict(values=cols, fill_color=[rowcolors],
                       font=dict(size=9), align="center", height=19),
            visible=visible), row=2, col=2)

        for _ in range(N_TRACES):
            trace_combo.append(ci)

    for ci in range(len(combos)):
        add_combo(ci, visible=(ci == default_combo))

    def visibility(ci):
        vis = [True] * n_static
        vis += [trace_combo[k] == ci for k in range(len(trace_combo))]
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
        direction="down", showactive=True, x=0.0, xanchor="left", y=1.15, yanchor="top",
        pad={"r": 4, "t": 4}, bgcolor="#eef4fb",
    )
    config_menu = dict(
        buttons=[dict(label=f"config {cfg}" + (" *" if cfg in PRODUCED_CONFIGS else ""),
                      method="update",
                      args=[{"visible": visibility(combo_id(0, cfg, DEF_AM))}])
                 for cfg in ALL_CONFIGS],
        direction="down", showactive=True, x=0.22, xanchor="left", y=1.15, yanchor="top",
        pad={"r": 4, "t": 4}, bgcolor="#f0f0f0",
    )
    angle_menu = dict(
        buttons=[dict(label=f"angles: {am}", method="update",
                      args=[{"visible": visibility(combo_id(0, DEF_CFG, am))}])
                 for am in ANGLE_MODES],
        direction="down", showactive=True, x=0.44, xanchor="left", y=1.15, yanchor="top",
        pad={"r": 4, "t": 4}, bgcolor="#eefbf0",
    )
    arch_by_ti = [p[1] for p in picks]
    full_menu = dict(
        buttons=[dict(label=f"{arch_by_ti[ti]} | {cfg} | {am}",
                      method="update",
                      args=[{"visible": visibility(combo_id(ti, cfg, am))}])
                 for (ti, cfg, am) in combos],
        direction="down", showactive=True, x=0.66, xanchor="left", y=1.15, yanchor="top",
        pad={"r": 4, "t": 4}, bgcolor="#f7f0fb",
    )
    fig.update_layout(updatemenus=[tracks_menu, config_menu, angle_menu, full_menu])

    # axis cosmetics
    # overview (row1,col1): full radius, equal aspect
    fig.update_xaxes(title_text="x [cm]", row=1, col=1, range=[-R_OVERVIEW, R_OVERVIEW],
                     scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="y [cm]", row=1, col=1, range=[-R_OVERVIEW, R_OVERVIEW])
    # IT-zoom r-phi (row1,col2)
    fig.update_xaxes(title_text="x [cm]", row=1, col=2, range=[-R_ZOOM, R_ZOOM],
                     scaleanchor="y2", scaleratio=1)
    fig.update_yaxes(title_text="y [cm]", row=1, col=2, range=[-R_ZOOM, R_ZOOM])
    # IT-zoom r-z (row2,col1)
    fig.update_xaxes(title_text="z [cm]", row=2, col=1, range=[-30, 30])
    fig.update_yaxes(title_text="r [cm]", row=2, col=1, range=[0, R_ZOOM])

    fig.update_layout(
        # Vertical zones (top->bottom): title 1.30 · labels 1.205 · menus 1.15 ·
        # legend 1.02 · plots 1.0 · two captions below. Clear separation.
        height=1120, width=1240,
        margin=dict(t=235, b=170, l=60, r=20),
        legend=dict(orientation="h", x=0.0, y=1.02, yanchor="bottom", xanchor="left"),
        annotations=list(fig.layout.annotations) + [
            dict(text=("<b>SmartPixels digiRefit replay</b>  —  5-par OT-only vs 5-par OT+IT: "
                       "does adding SmartPixels improve the prompt track?"),
                 x=0.5, y=1.30, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=15), xanchor="center"),
            dict(text="track", x=0.0, y=1.205, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11), xanchor="left"),
            dict(text="config (* = produced/real)", x=0.22, y=1.205, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11), xanchor="left"),
            dict(text="angle mode", x=0.44, y=1.205, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11), xanchor="left"),
            dict(text="full combo (authoritative)", x=0.66, y=1.205, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11), xanchor="left"),
            # OT-context caption (5-par framing + real-stub note).
            dict(text=("<span style='color:#1f77b4'>seed</span> = 5-par OT-only L1Track "
                       "(L1TrackFinder fit on OUTER-tracker stubs + beamline, promptHnpar=5; "
                       "anchored at r≈25–108 cm) &nbsp;·&nbsp; "
                       "<span style='color:#d62728'>refit</span> = 5-par OT+IT digiRefit adds "
                       "SmartPixels IT hits (r&lt;16 cm). The question: does adding IT IMPROVE "
                       "the 5-par prompt track (d0 for b-tagging, vertexing)? "
                       "<span style='color:#17becf'>OT stubs</span> are the REAL embedded "
                       "L1TTrackStub positions (x/y/z)."),
                 x=0.5, y=-0.12, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=10.5), xanchor="center"),
            # Resolution-to-truth + reduced-chi2 accounting caption.
            dict(text=("Table shows RESOLUTION-TO-TRUTH: <b>d0−tru</b>/<b>z0−tru</b> per step + "
                       "|seed−tru|→|refit−tru| (does IT move TOWARD truth?). "
                       "Truth (nano_pE, PROMPT-only): z0=GenVtx_z, "
                       "d0=GenVtx_x·sinφ−GenVtx_y·cosφ (beamspot≈0; matches L1Track_d0 sign; "
                       "for DISPLACED/b tracks use matched-TP d0 from nano_pF). "
                       "<b>χ2red(OT+IT)</b> = running reduced χ2 from the OT-only anchor "
                       "(chi2ZRed·ndof_OT, ndof_OT=2·nStubs−5) + IT r-z increments / growing ndof; "
                       "Σχ2(rφ) is parametrized-seed-inflated (raw, illustrative)."),
                 x=0.5, y=-0.175, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=9.5), xanchor="center"),
            # Static fidelity key.
            dict(text=("<span style='color:#d62728'>REAL</span> = produced "
                       "(AIII/AAII/AAAI/AAAA @ alphaBeta) &nbsp;·&nbsp; "
                       "else <span style='color:#7f7f7f'>REPLAY</span> "
                       "(rInv/phi0/d0 faithful, tanL/z0 illustrative — parametrized seed cov)"),
                 x=0.5, y=-0.225, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=10.5), xanchor="center"),
        ],
    )
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
        with open(out_html, "w") as fh:
            fh.write(html)
        result["html_path"] = out_html
    if return_fig:
        result["figure"] = fig
    return result
