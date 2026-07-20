"""Per-track TRUTH for the 5-par prompt resolution-vs-truth columns.

The framing (mem:smartpixels-5par-framing-directive): the comparison is
5-par OT-only (reference L1TTrack) vs 5-par OT+IT (prompt digiRefit). The metric
is RESOLUTION TO TRUTH: does adding the inner-tracker hits move the track TOWARD
the true parameters, param by param.

Interim truth on nano_pE (this file has ONLY GenVtx + tpPt/tpEta on the track):
  * z0_true = GenVtx_z                        (the primary-vertex z; -11..8 cm span,
                                               a real resolvable truth for PROMPT tracks).
  * d0_true = GenVtx_x*sin(phi) - GenVtx_y*cos(phi)   (transverse-vertex projection,
                                               the vertex-study convention; VERIFIED
                                               sign-consistent with L1TTrack_d0 =
                                               x0*sin(phi) - y0*cos(phi) via matched
                                               GenPart production vertices, and with
                                               nano's GenPart_dXY = -(x0 sinφ - y0 cosφ)).
                                               On nano_pE this is the BEAMSPOT projection
                                               (~5 um) -> d0_true ≈ 0. That is correct
                                               truth for genuinely PROMPT tracks (true
                                               d0 ~ 0) but WRONG for displaced/b tracks,
                                               whose true production vertex is offset.
  * pt_true  = tpPt   ;   eta_true = tpEta    (matched TrackingParticle kinematics).

CAVEAT (surfaced in the viz caption): GenVtx = PV is only valid d0 truth for PROMPT
tracks. For DISPLACED / b-decay tracks the matched-TP d0/z0 is the right truth. The
richer nano_pF adds tp_d0/tp_z0/tp_phi/tp production vertex; :func:`track_truth`
transparently SWAPS to those columns when they are present, so no viz change is
needed to upgrade the truth once pF lands.
"""

from __future__ import annotations

import numpy as np

# Column names the richer nano_pF is expected to carry on the track table (matched-TP
# full helix params). When present these override the GenVtx interim truth below.
_TP_D0_KEYS = ("tpD0", "tp_d0", "matchTpD0")
_TP_Z0_KEYS = ("tpZ0", "tp_z0", "matchTpZ0")


def _first_present(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            v = float(d[k])
            if v > -900.0:      # nano sentinel for "no match"
                return v
    return None


def track_truth(seed, truth, genvtx):
    """Return the truth-parameter dict for one track.

    Parameters
    ----------
    seed : dict with at least ``phi0`` (the track azimuth at POCA).
    truth : dict with ``tpPt``, ``tpEta`` and (once pF lands) optional matched-TP
        ``tpD0``/``tpZ0`` full-helix params.
    genvtx : dict with ``x``, ``y``, ``z`` of the generated primary vertex.

    Returns
    -------
    dict with keys ``d0``, ``z0``, ``pt``, ``eta`` (float or None if unavailable)
    plus ``d0_source`` / ``z0_source`` tags ('matched-TP' or 'GenVtx(PV)') so the
    viz can label the provenance and flag the prompt-only caveat.
    """
    phi = float(seed["phi0"])

    tp_d0 = _first_present(truth, _TP_D0_KEYS)
    tp_z0 = _first_present(truth, _TP_Z0_KEYS)

    if tp_d0 is not None:
        d0_true, d0_src = tp_d0, "matched-TP"
    else:
        d0_true = genvtx["x"] * np.sin(phi) - genvtx["y"] * np.cos(phi)
        d0_src = "GenVtx(PV)"

    if tp_z0 is not None:
        z0_true, z0_src = tp_z0, "matched-TP"
    else:
        z0_true, z0_src = float(genvtx["z"]), "GenVtx(PV)"

    tp_pt = truth.get("tpPt", -1.0)
    tp_eta = truth.get("tpEta", None)
    return dict(
        d0=float(d0_true), z0=float(z0_true),
        pt=(float(tp_pt) if tp_pt is not None and tp_pt > 0 else None),
        eta=(float(tp_eta) if tp_eta is not None else None),
        d0_source=d0_src, z0_source=z0_src,
    )


def truth_is_prompt_only(truth_dict):
    """True if the d0/z0 truth came from GenVtx (PV) -> valid ONLY for prompt tracks.
    False once matched-TP params are used (valid for displaced tracks too)."""
    return truth_dict["d0_source"] == "GenVtx(PV)"
