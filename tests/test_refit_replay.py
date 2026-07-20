"""Tests for the SmartPixels digiRefit refit-replay visualizer.

Two layers:
  * always-on synthetic tests: the offline KF replay obeys its structural
    invariants (chi2 = sum of squared pulls, monotone update ordering, angle-mode
    gating) and the Plotly figure builds self-contained from a synthetic sample.
  * NGTAGGER_TEST_NANO gated: the VALIDATION GATE -- replay the 4 produced configs
    at alphaBeta on the real nano and check the documented sign/scale agreement vs
    the real variant-track-table values, plus that the standalone HTML builds.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ngtagger.viz._dataio import (
    PRODUCED_CONFIGS,
    NanoData,
    TrackRecord,
    config_active_layers,
)
from ngtagger.viz._kf import _REPLAY_SEED_SIGMAS, replay_track

_NANO = os.environ.get("NGTAGGER_TEST_NANO")
_RADII = (3.0, 6.8, 10.9, 16.0)


# --------------------------------------------------------------------------
# synthetic sample (no nano needed)
# --------------------------------------------------------------------------
def _synth_hits(rng, n_layers=4):
    hits = {}
    for L in range(1, n_layers + 1):
        # small physical residuals; angle info present
        hits[L] = dict(
            layer=L, hitAccepted=True,
            resX=float(rng.normal(0, 0.02)), resY=float(rng.normal(0, 0.03)),
            cotAlphaMeas=float(rng.normal(0, 0.05)), cotBetaMeas=float(rng.normal(0.3, 0.02)),
            parCotAlpha=float(rng.normal(0, 0.05)), parCotBeta=0.3,
            sigAlpha=0.02, sigBeta=0.02,
            pullX=0.0, pullY=0.0,
            pullAlpha=float(rng.normal(0, 1)), pullBeta=float(rng.normal(0, 1)),
            hasAlpha=True, hasBeta=True,
            chi2IncRPhi=0.0, chi2IncRZ=0.0, selChi2Margin=1.0, selHitClass=0,
            windowMult=3, windowTruncated=False,
        )
    return hits


def _synth_stubs(rng, n=6):
    """A few real-style OT barrel stubs (x/y/z, r, layer, isBarrel)."""
    radii = [25.0, 37.2, 52.2, 68.7, 86.0, 108.6]
    out = []
    for L in range(1, n + 1):
        r = radii[L - 1]
        ph = float(rng.uniform(-0.5, 0.5))
        out.append(dict(layer=L, isBarrel=True, r=r,
                        x=r * float(np.cos(ph)), y=r * float(np.sin(ph)),
                        z=float(rng.uniform(-20, 20))))
    return out


def _synth_track(rng, event=0, idx=0, genuine=True, n_layers=4):
    seed = dict(rInv=float(rng.uniform(-0.004, 0.004)), phi0=float(rng.uniform(-0.5, 0.5)),
                tanL=float(rng.uniform(-1.5, 1.5)), z0=float(rng.uniform(-3, 3)),
                d0=float(rng.uniform(-0.05, 0.05)), pt=float(rng.uniform(3, 10)),
                eta=0.5, chi2XYRed=float(rng.uniform(0.1, 2.0)),
                chi2ZRed=float(rng.uniform(0.1, 2.0)))
    truth = dict(genuine=genuine, looselyGenuine=genuine, tpPt=seed["pt"] if genuine else -1.0,
                 tpEta=0.5 if genuine else 0.0, fromHard=genuine, pdgId=211, nStubs=6,
                 hitPattern=0b0111111)
    hits = _synth_hits(rng, n_layers)
    real = {v: dict(rInv=seed["rInv"], phi0=seed["phi0"], tanL=seed["tanL"], z0=seed["z0"],
                    d0=seed["d0"], pt=seed["pt"], chi2XYRed=seed["chi2XYRed"],
                    chi2ZRed=seed["chi2ZRed"], nAcc=n_layers, nUpd=n_layers,
                    layerMask=(1 << n_layers) - 1, chi2RPhiTot=0.0, chi2RZTot=0.0,
                    refitPerformed=True, maxWinMult=3)
            for v in PRODUCED_CONFIGS.values()}
    genvtx = dict(x=float(rng.normal(0, 0.002)), y=float(rng.normal(0, 0.002)),
                  z=float(rng.uniform(-8, 8)))
    return TrackRecord(event=event, idx=idx, seed=seed, truth=truth, hits=hits,
                       stubs=_synth_stubs(rng), real=real, genvtx=genvtx)


def _synth_data(n=12, seed=1):
    rng = np.random.default_rng(seed)
    d = NanoData(radii=_RADII)
    for i in range(n):
        d.tracks.append(_synth_track(rng, idx=i, genuine=(i % 4 != 0),
                                     n_layers=4 if i % 3 else 3))
    return d


# --------------------------------------------------------------------------
# KF structural invariants
# --------------------------------------------------------------------------
def test_replay_chi2_equals_sum_of_squared_pulls():
    """Per-step chi2(rphi)=pullX^2+pullAlpha^2, chi2(rz)=pullY^2+pullBeta^2."""
    rng = np.random.default_rng(0)
    tr = _synth_track(rng)
    seed = [tr.seed[k] for k in ["rInv", "phi0", "tanL", "z0", "d0"]]
    r = replay_track(seed, list(tr.hits.values()), layer_radii=_RADII,
                     use_angles="alphaBeta", active_layers=(1, 1, 1, 1))
    for pk, crphi, crz in zip(r["pulls"], r["chi2_rphi"], r["chi2_rz"]):
        px = pk["pullX"] or 0.0
        pa = pk["pullAlpha"] or 0.0
        py = pk["pullY"] or 0.0
        pb = pk["pullBeta"] or 0.0
        assert crphi == pytest.approx(px * px + pa * pa, abs=1e-9)
        assert crz == pytest.approx(py * py + pb * pb, abs=1e-9)


def test_angle_mode_gating():
    """none feeds {x,y}; alpha adds alpha; alphaBeta adds beta. So alpha/beta pulls
    appear only in the corresponding modes, and the rphi/rz chi2 grows monotonically
    as more scalar updates are enabled."""
    rng = np.random.default_rng(3)
    tr = _synth_track(rng)
    seed = [tr.seed[k] for k in ["rInv", "phi0", "tanL", "z0", "d0"]]
    hits = list(tr.hits.values())
    r_none = replay_track(seed, hits, layer_radii=_RADII, use_angles="none")
    r_a = replay_track(seed, hits, layer_radii=_RADII, use_angles="alpha")
    r_ab = replay_track(seed, hits, layer_radii=_RADII, use_angles="alphaBeta")
    assert all(pk["pullAlpha"] is None for pk in r_none["pulls"])
    assert all(pk["pullBeta"] is None for pk in r_none["pulls"])
    assert any(pk["pullAlpha"] is not None for pk in r_a["pulls"])
    assert all(pk["pullBeta"] is None for pk in r_a["pulls"])
    assert any(pk["pullBeta"] is not None for pk in r_ab["pulls"])
    # enabling more scalar updates only adds non-negative chi2 contributions
    assert r_ab["chi2_rphi_tot"] >= r_none["chi2_rphi_tot"] - 1e-9
    assert r_ab["chi2_rz_tot"] >= r_none["chi2_rz_tot"] - 1e-9


def test_config_subset_feeds_fewer_layers():
    """config selects which layers are fed: a 1-layer config yields <= the states of
    a 4-layer config."""
    rng = np.random.default_rng(5)
    tr = _synth_track(rng)
    seed = [tr.seed[k] for k in ["rInv", "phi0", "tanL", "z0", "d0"]]
    hits = list(tr.hits.values())
    r1 = replay_track(seed, hits, layer_radii=_RADII, active_layers=config_active_layers("1000"))
    r4 = replay_track(seed, hits, layer_radii=_RADII, active_layers=config_active_layers("1111"))
    assert len(r1["states"]) <= len(r4["states"])
    assert len(r1["states"]) >= 1


def test_truth_d0_sign_and_source():
    """d0_true = GenVtx_x*sin(phi) - GenVtx_y*cos(phi) (matches L1TTrack_d0 sign
    convention); z0_true = GenVtx_z. Source is GenVtx(PV) when no matched-TP params,
    and prompt-only in that case."""
    from ngtagger.viz._truth import track_truth, truth_is_prompt_only

    seed = dict(phi0=0.0)
    truth = dict(tpPt=5.0, tpEta=0.7)
    gv = dict(x=0.03, y=0.05, z=1.234)
    tt = track_truth(seed, truth, gv)
    # phi=0 -> d0 = x*sin(0) - y*cos(0) = -y
    assert tt["d0"] == pytest.approx(-gv["y"], abs=1e-9)
    assert tt["z0"] == pytest.approx(gv["z"], abs=1e-9)
    assert tt["pt"] == pytest.approx(5.0) and tt["eta"] == pytest.approx(0.7)
    assert tt["d0_source"] == "GenVtx(PV)" and truth_is_prompt_only(tt)
    # phi=pi/2 -> d0 = x
    tt2 = track_truth(dict(phi0=np.pi / 2), truth, gv)
    assert tt2["d0"] == pytest.approx(gv["x"], abs=1e-6)


def test_truth_swaps_to_matched_tp():
    """When the richer nano provides matched-TP d0/z0, track_truth uses them (valid
    for displaced tracks) instead of the GenVtx(PV) interim truth."""
    from ngtagger.viz._truth import track_truth, truth_is_prompt_only

    seed = dict(phi0=0.3)
    truth = dict(tpPt=5.0, tpEta=0.7, tpD0=0.42, tpZ0=-3.1)
    gv = dict(x=0.03, y=0.05, z=1.234)
    tt = track_truth(seed, truth, gv)
    assert tt["d0"] == pytest.approx(0.42) and tt["z0"] == pytest.approx(-3.1)
    assert tt["d0_source"] == "matched-TP" and not truth_is_prompt_only(tt)


def test_reduced_chi2_running_starts_from_ot_anchor():
    """The running reduced chi2 seed value equals the OT-only reduced z-fit chi2
    (chi2ZRed), and each IT r-z increment grows the numerator while ndof grows by the
    per-hit dof; alphaBeta mode adds one more dof per hit than 'none'."""
    from ngtagger.viz.refit_replay import _reduced_chi2_running

    rng = np.random.default_rng(11)
    tr = _synth_track(rng)
    inc = [1.0, 2.0, 0.5]
    red_ab, ndof_ab = _reduced_chi2_running(tr, inc, "alphaBeta")
    red_none, ndof_none = _reduced_chi2_running(tr, inc, "none")
    # seed value is the OT-only reduced z chi2
    assert red_ab[0] == pytest.approx(tr.seed["chi2ZRed"], rel=1e-9)
    assert red_none[0] == pytest.approx(tr.seed["chi2ZRed"], rel=1e-9)
    # ndof grows faster in alphaBeta (position + angle dof per hit)
    assert ndof_ab[-1] > ndof_none[-1]
    assert len(red_ab) == len(inc) + 1


def test_replay_never_reads_answer():
    """Passing garbage in the real-refit fields must not change the replay output
    (the replay consumes only seed + sidecar hit records)."""
    rng = np.random.default_rng(7)
    tr = _synth_track(rng)
    seed = [tr.seed[k] for k in ["rInv", "phi0", "tanL", "z0", "d0"]]
    a = replay_track(seed, list(tr.hits.values()), layer_radii=_RADII)
    tr.real["AAAA"]["d0"] = 999.0  # corrupt the answer
    b = replay_track(seed, list(tr.hits.values()), layer_radii=_RADII)
    assert np.allclose(a["final"], b["final"])


# --------------------------------------------------------------------------
# figure builds self-contained (synthetic)
# --------------------------------------------------------------------------
def test_figure_builds_self_contained(tmp_path):
    from ngtagger.viz.refit_replay import build_figure

    data = _synth_data()
    picks = [(data.tracks[1], "clean", "syn"),
             (data.tracks[0], "fake", "syn")]
    fig = build_figure(data, picks, np.asarray(_RADII), np.asarray(_REPLAY_SEED_SIGMAS))
    # static geometry (6 OT circles + 3*4 IT guides) + n_combos * 11 combo traces
    from ngtagger.viz._dataio import OT_BARREL_RADII
    n_static = len(OT_BARREL_RADII) + 3 * len(_RADII)
    n_combos = len(picks) * 15 * 3
    assert len(fig.data) == n_static + n_combos * 11
    assert len(fig.layout.updatemenus) == 4
    out = tmp_path / "syn.html"
    html = fig.to_html(include_plotlyjs=True, full_html=True)
    out.write_text(html)
    assert "Plotly.newPlot" in html
    assert '<script src="' not in html  # fully inlined, no external script
    assert out.stat().st_size > 1_000_000


# --------------------------------------------------------------------------
# VALIDATION GATE (real nano)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _NANO, reason="set NGTAGGER_TEST_NANO to the SmartPixels nano file")
def test_validation_gate_produced_configs():
    """Replay the 4 produced configs at alphaBeta and check the DOCUMENTED agreement
    vs the real variant-track-table values:
      * faithful params (rInv, phi0, d0): sign-agreement well above chance;
      * chi2 totals: strong rank correlation replay-vs-real.
    tanL/z0 are illustrative (trackCov-seed dependent) and are NOT gated on sign.
    """
    from ngtagger.viz._dataio import load_nano

    data = load_nano(_NANO, layer_radii=_RADII, max_events=6)
    for cfg, V in PRODUCED_CONFIGS.items():
        active = config_active_layers(cfg)
        drInv, dphi, dd0 = [], [], []
        rphi_rep, rphi_real = [], []
        for tr in data.tracks:
            real = tr.real[V]
            if not real["refitPerformed"]:
                continue
            seed = [tr.seed[k] for k in ["rInv", "phi0", "tanL", "z0", "d0"]]
            r = replay_track(seed, list(tr.hits.values()), layer_radii=_RADII,
                             use_angles="alphaBeta", active_layers=active, seed_npar=5)
            drInv.append((real["rInv"] - tr.seed["rInv"], r["delta"][0]))
            dphi.append((real["phi0"] - tr.seed["phi0"], r["delta"][1]))
            dd0.append((real["d0"] - tr.seed["d0"], r["delta"][4]))
            rphi_rep.append(r["chi2_rphi_tot"]); rphi_real.append(real["chi2RPhiTot"])

        dd0 = np.array(dd0)
        # d0 (the physics payoff) must be faithful in sign for a clear majority
        d0_sign = np.mean(np.sign(dd0[:, 0]) == np.sign(dd0[:, 1]))
        assert d0_sign > 0.65, f"{cfg}: d0 sign agreement {d0_sign:.2f} too low"
        # chi2(rphi) total: strong correlation replay-vs-real
        corr = np.corrcoef(rphi_rep, rphi_real)[0, 1]
        assert corr > 0.80, f"{cfg}: chi2RPhi corr {corr:.2f} too low"
        # for multi-layer configs, rInv/phi0 curvature/azimuth are also faithful
        if cfg != "1000":
            for arr, nm in ((np.array(drInv), "rInv"), (np.array(dphi), "phi0")):
                s = np.mean(np.sign(arr[:, 0]) == np.sign(arr[:, 1]))
                assert s > 0.55, f"{cfg}: {nm} sign agreement {s:.2f} too low"


@pytest.mark.skipif(not _NANO, reason="set NGTAGGER_TEST_NANO to the SmartPixels nano file")
def test_html_builds_from_real_nano(tmp_path):
    from ngtagger.viz.refit_replay import build_refit_viz

    out = tmp_path / "refit_replay.html"
    res = build_refit_viz(_NANO, out_html=str(out), n_each=(2, 1, 1, 1), max_events=6)
    assert res["html_path"] and out.exists()
    # clean/wrong/displaced/fake curation on the prime-target sample
    assert 3 <= len(res["picks"]) <= 5
    archs = {a for _, _, a in res["picks"]}
    assert archs <= {"clean", "wrong", "displaced", "fake"}
    html = out.read_text()
    assert "Plotly.newPlot" in html and '<script src="' not in html
