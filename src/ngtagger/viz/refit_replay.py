"""SmartPixels digiRefit refit-replay VISUALIZER.

Interactive, self-contained (kernel-free) Plotly figure showing a digiRefit
L1Track Kalman refit step-by-step across the 3 angle modes (none / alpha /
alphaBeta) and the 15 SmartPixels layer configs (1000..1111), for a curated set
of example tracks.

Two-posture faithfulness (see docs/refit-replay-viz.md):
  * chi2 evolution & pulls: for the 4 PRODUCED configs (AIII/AAII/AAAI/AAAA at
    useAngles=alphaBeta) these come straight from the nano sidecar and are
    BIT-EXACT to production; for the other 41 config x mode combinations they are
    the offline REPLAY.
  * parameter-state evolution: always the offline replay
    (:func:`ngtagger.viz.refit_replay.replay_track`). rInv / phi0 / d0 are
    reproduced faithfully in sign and order-of-magnitude scale; tanL / z0 hinge
    on the per-track trackCov correlations that nano does not persist and are
    labelled ILLUSTRATIVE.

Public entry point: :func:`build_refit_viz`.
"""

from __future__ import annotations

import numpy as np

from ngtagger.viz._curate import curate_tracks
from ngtagger.viz._dataio import (
    ALL_CONFIGS,
    ANGLE_MODES,
    PRODUCED_CONFIGS,
    config_active_layers,
    hit_class_name,
    load_nano,
)
from ngtagger.viz._kf import _REPLAY_SEED_SIGMAS, replay_track

PARAM_NAMES = ["rInv", "phi0", "tanL", "z0", "d0"]
PARAM_UNITS = ["cm$^{-1}$", "rad", "", "cm", "cm"]
# Which replayed parameters are faithful vs illustrative (expert validation gate).
FAITHFUL_PARAMS = {"rInv", "phi0", "d0"}
ILLUSTRATIVE_PARAMS = {"tanL", "z0"}

# Colors
C_SEED = "#1f77b4"     # seed helix / seed state
C_REFIT = "#d62728"    # refit helix / final state
C_LAYER = "#888888"
C_HIT_TRUE = "#2ca02c"    # selHitClass 0
C_HIT_WRONG = "#ff7f0e"   # selHitClass 1
C_HIT_NOISE = "#9467bd"   # selHitClass 2
_HIT_COLOR = {0: C_HIT_TRUE, 1: C_HIT_WRONG, 2: C_HIT_NOISE, -1: C_LAYER}


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
# per-combo replay payload
# --------------------------------------------------------------------------
def _combo_payload(track, config, angle_mode, radii, seed_sigmas):
    """Compute everything a (track, config, angle) combo needs to draw.

    Returns a dict with seed/final params, per-step state table rows, chi2 evolution,
    hit markers, and a real-vs-replay label. For the 4 produced configs at alphaBeta
    the chi2 & pulls are taken from the sidecar (bit-exact); the state trajectory is
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

    # chi2 evolution: prefer sidecar for produced-at-alphaBeta (bit-exact), else replay
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

    final = rep["final"]
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
               [{"type": "table", "colspan": 2}, None]],
        row_heights=[0.60, 0.40],
        column_widths=[0.5, 0.5],
        subplot_titles=("r-φ projection (x–y)", "r–z projection",
                        "Kalman step table (state · Δ vs seed · cumulative χ2)"),
        horizontal_spacing=0.08, vertical_spacing=0.10,
    )

    # static layer geometry (drawn once, always visible)
    theta = np.linspace(0, 2 * np.pi, 200)
    for R in radii:
        fig.add_trace(go.Scatter(x=R * np.cos(theta), y=R * np.sin(theta),
                                 mode="lines", line=dict(color=C_LAYER, width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[-40, 40], y=[R, R], mode="lines",
                                 line=dict(color=C_LAYER, width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False), row=1, col=2)

    # per-combo trace groups: [seed_xy, refit_xy, hits_xy, seed_rz, refit_rz, hits_rz, table]
    N_TRACES = 7
    trace_combo = []   # parallel index -> combo id

    def add_combo(ci, visible):
        ti, cfg, am = combos[ci]
        tr = tracks[ti]
        pl = _combo_payload(tr, cfg, am, radii, seed_sigmas)
        s = pl["seed"]; f = pl["final"]

        # r-phi (x-y): seed & refit helices + selected-hit markers
        sx, sy, _, _ = helix_points(*s)
        fx, fy, _, _ = helix_points(*f)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
                                 line=dict(color=C_SEED, width=2),
                                 name="seed helix", legendgroup="seed",
                                 showlegend=(ci == default_combo), visible=visible,
                                 hovertemplate="seed<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines",
                                 line=dict(color=C_REFIT, width=2),
                                 name="refit helix", legendgroup="refit",
                                 showlegend=(ci == default_combo), visible=visible,
                                 hovertemplate="refit<extra></extra>"), row=1, col=1)
        hx, hy, hcol, htxt = [], [], [], []
        for L in pl["fed_layers"]:
            cr = _crossing_xy(*s, R=radii[L - 1])
            if cr is None:
                continue
            cls = tr.hits[L]["selHitClass"]
            hx.append(cr[0]); hy.append(cr[1]); hcol.append(_HIT_COLOR.get(cls, C_LAYER))
            htxt.append(f"L{L} {hit_class_name(cls)}<br>resX={tr.hits[L]['resX']:.3f} cm "
                        f"resY={tr.hits[L]['resY']:.3f} cm<br>winMult={tr.hits[L]['windowMult']}")
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers",
                                 marker=dict(size=11, color=hcol, symbol="x-thin",
                                             line=dict(width=3, color=hcol)),
                                 name="selected hit", legendgroup="hit",
                                 showlegend=False, visible=visible,
                                 text=htxt, hovertemplate="%{text}<extra></extra>"), row=1, col=1)

        # r-z: (z, r) trajectories + hits
        szx, szy, szz, szr = helix_points(*s)
        fzx, fzy, fzz, fzr = helix_points(*f)
        fig.add_trace(go.Scatter(x=szz, y=szr, mode="lines",
                                 line=dict(color=C_SEED, width=2), showlegend=False,
                                 visible=visible, hovertemplate="seed<extra></extra>"), row=1, col=2)
        fig.add_trace(go.Scatter(x=fzz, y=fzr, mode="lines",
                                 line=dict(color=C_REFIT, width=2), showlegend=False,
                                 visible=visible, hovertemplate="refit<extra></extra>"), row=1, col=2)
        rz_z, rz_r, rz_c = [], [], []
        for L in pl["fed_layers"]:
            cr = _crossing_xy(*s, R=radii[L - 1])
            if cr is None:
                continue
            rz_z.append(cr[2]); rz_r.append(radii[L - 1])
            rz_c.append(_HIT_COLOR.get(tr.hits[L]["selHitClass"], C_LAYER))
        fig.add_trace(go.Scatter(x=rz_z, y=rz_r, mode="markers",
                                 marker=dict(size=11, color=rz_c, symbol="x-thin",
                                             line=dict(width=3, color=rz_c)),
                                 showlegend=False, visible=visible,
                                 hoverinfo="skip"), row=1, col=2)

        # step table
        header_vals = ["step"] + PARAM_NAMES + ["Δd0", "Δz0", "Σχ2(rφ)", "Σχ2(rz)"]
        rows = []
        for i, (lab, st) in enumerate(zip(pl["step_labels"], pl["step_states"])):
            dd0 = st[4] - s[4]
            dz0 = st[3] - s[3]
            rows.append([lab] + [_fmt(st[j], PARAM_NAMES[j]) for j in range(5)]
                        + [f"{dd0:+.4f}", f"{dz0:+.4f}",
                           f"{pl['cum_rphi'][i]:.2f}", f"{pl['cum_rz'][i]:.2f}"])
        # final row
        dd0 = f[4] - s[4]; dz0 = f[3] - s[3]
        rows.append(["REFIT"] + [_fmt(f[j], PARAM_NAMES[j]) for j in range(5)]
                    + [f"{dd0:+.4f}", f"{dz0:+.4f}",
                       f"{pl['cum_rphi'][-1]:.2f}", f"{pl['cum_rz'][-1]:.2f}"])
        cols = list(map(list, zip(*rows))) if rows else [[] for _ in header_vals]
        # highlight the seed and REFIT rows
        n = len(rows)
        rowcolors = ["#eef4fb"] + ["white"] * (n - 2) + ["#fbeeee"] if n >= 2 else ["white"] * n
        fig.add_trace(go.Table(
            header=dict(values=header_vals, fill_color="#34495e",
                        font=dict(color="white", size=11), align="center"),
            cells=dict(values=cols, fill_color=[rowcolors],
                       font=dict(size=11), align="center", height=22),
            visible=visible), row=2, col=1)

        for _ in range(N_TRACES):
            trace_combo.append(ci)

    for ci in range(len(combos)):
        add_combo(ci, visible=(ci == default_combo))

    n_static = 2 * len(radii)

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
    full_menu = dict(
        buttons=[dict(label=f"{['clean','wrong','fake','tk3'][ti] if ti < 4 else 'tk'+str(ti)}"
                            f" | {cfg} | {am}",
                      method="update",
                      args=[{"visible": visibility(combo_id(ti, cfg, am))}])
                 for (ti, cfg, am) in combos],
        direction="down", showactive=True, x=0.66, xanchor="left", y=1.15, yanchor="top",
        pad={"r": 4, "t": 4}, bgcolor="#f7f0fb",
    )
    fig.update_layout(updatemenus=[tracks_menu, config_menu, angle_menu, full_menu])

    # axis cosmetics
    fig.update_xaxes(title_text="x [cm]", row=1, col=1, range=[-18, 18], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="y [cm]", row=1, col=1, range=[-18, 18])
    fig.update_xaxes(title_text="z [cm]", row=1, col=2, range=[-30, 30])
    fig.update_yaxes(title_text="r [cm]", row=1, col=2, range=[0, 18])

    fig.update_layout(
        # Vertical zones (top->bottom): title 1.30 · labels 1.205 · menus 1.15 ·
        # legend 1.02 · plots 1.0. Clear separation so nothing collides.
        height=960, width=1180,
        margin=dict(t=235, b=70, l=60, r=20),
        legend=dict(orientation="h", x=0.0, y=1.02, yanchor="bottom", xanchor="left"),
        annotations=list(fig.layout.annotations) + [
            dict(text=("<b>SmartPixels digiRefit replay</b>  —  step-by-step Kalman "
                       "refit of an L1 track against SmartPixels hits"),
                 x=0.5, y=1.30, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=16), xanchor="center"),
            dict(text="track", x=0.0, y=1.205, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11), xanchor="left"),
            dict(text="config (* = produced/real)", x=0.22, y=1.205, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11), xanchor="left"),
            dict(text="angle mode", x=0.44, y=1.205, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11), xanchor="left"),
            dict(text="full combo (authoritative)", x=0.66, y=1.205, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11), xanchor="left"),
            # Static fidelity key, as a caption BELOW the table (not crammed on top).
            dict(text=("<span style='color:#d62728'>REAL</span> = produced "
                       "(AIII/AAII/AAAI/AAAA @ alphaBeta, chi2/pulls bit-exact) &nbsp;·&nbsp; "
                       "else <span style='color:#7f7f7f'>REPLAY</span> "
                       "(rInv/phi0/d0 faithful, tanL/z0 illustrative — parametrized seed cov)"),
                 x=0.5, y=-0.11, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=11), xanchor="center"),
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
            f"tpPt={truth['tpPt']:.1f}, nStubs={truth['nStubs']})</span>")


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def build_refit_viz(nano_file, tracks=None, out_html=None,
                    layer_radii=(3.0, 6.8, 10.9, 16.0),
                    seed_sigmas=_REPLAY_SEED_SIGMAS,
                    n_each=(2, 1, 1), max_events=6, return_fig=False):
    """Build the standalone interactive refit-replay HTML.

    Parameters
    ----------
    nano_file : path to the SmartPixels nano file (read-only).
    tracks : optional list of (event, idx) tuples to force-select; otherwise the
        curated archetypes (clean/wrong/fake) are chosen automatically.
    out_html : path to write the self-contained HTML (include_plotlyjs=True). If
        None, the HTML is not written.
    layer_radii : TBPX mean layer radii [cm] (projector convention).
    seed_sigmas : parametrized seed-covariance sqrt-diagonal (documented caveat:
        production used trackCov, not persisted in nano).
    n_each : (n_clean, n_wrong, n_fake) archetypes to curate when tracks is None.

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
            arch = ("clean" if (tr.truth["genuine"] and len(tr.hits) == 4)
                    else "fake" if not tr.truth["genuine"] else "wrong")
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
