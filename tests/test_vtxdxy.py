"""Vertex (dx, dy) estimator + fastHisto kernel option tests.

Covers: bit-identity of the default fastHisto path against a frozen copy of
the pre-kernel implementation, kernel construction and midpoint-pick
behaviour, the d0 sign-convention closure (d0 = x_v sin(phi) - y_v cos(phi),
the TTTrack POCA convention), LSQ-vs-isotropic under biased phi coverage,
degenerate-coverage guards, and the "vertexdxy" feature-group plumbing
through load_jets/build_features.
"""

import awkward as ak
import numpy as np
import pytest

from ngtagger.train.nnvtx import (fast_histo_vtx, fast_histo_z0, make_kernel,
                                  vertex_dxy_features)

# --------------------------------------------------------------------------
# toy helpers
# --------------------------------------------------------------------------


def _toy_events(n_events=100, xv=0.0, yv=0.0, sigma_d0=0.02, n_pv=25, n_pu=40,
                phi_lo=-np.pi, phi_hi=np.pi, seed=0):
    """Padded (z0, pt, mask, d0, phi) toys: PV tracks at a random z_v from a
    vertex at (xv, yv); PU tracks spread in z with d0 from the origin."""
    rng = np.random.default_rng(seed)
    n_trk = n_pv + n_pu
    z0 = np.empty((n_events, n_trk))
    pt = np.empty_like(z0)
    d0 = np.empty_like(z0)
    phi = np.empty_like(z0)
    z_true = rng.uniform(-8, 8, n_events)
    for i in range(n_events):
        z0[i, :n_pv] = rng.normal(z_true[i], 0.04, n_pv)
        z0[i, n_pv:] = rng.uniform(-14, 14, n_pu)
        pt[i, :n_pv] = rng.uniform(3, 40, n_pv)
        pt[i, n_pv:] = rng.uniform(2, 6, n_pu)
        phi[i] = rng.uniform(phi_lo, phi_hi, n_trk)
        d0[i] = xv * np.sin(phi[i]) - yv * np.cos(phi[i]) + rng.normal(0, sigma_d0, n_trk)
        d0[i, n_pv:] = rng.normal(0, sigma_d0, n_pu)  # PU from the origin
    return z0, pt, np.ones_like(z0), d0, phi, z_true


# --------------------------------------------------------------------------
# regression: default behaviour bit-identical to the pre-kernel implementation
# --------------------------------------------------------------------------


def _fast_histo_z0_frozen(z0, pt, mask, window_bins=3, max_track_pt=50.0):
    """Verbatim copy of fast_histo_z0 BEFORE the kernel/vtx extension."""
    from ngtagger.train.nnvtx import HISTO_MAX, HISTO_MIN, N_BINS

    n_events = z0.shape[0]
    edges = np.linspace(HISTO_MIN, HISTO_MAX, N_BINS + 1)
    out = np.zeros(n_events)
    w = np.minimum(pt, max_track_pt) * mask
    for i in range(n_events):
        hist, _ = np.histogram(z0[i], bins=edges, weights=w[i])
        windows = np.convolve(hist, np.ones(window_bins), mode="valid")
        b = int(np.argmax(windows))
        sel = (z0[i] >= edges[b]) & (z0[i] < edges[b + window_bins]) & (w[i] > 0)
        out[i] = np.average(z0[i][sel], weights=w[i][sel]) if sel.any() else 0.5 * (edges[b] + edges[b + window_bins])
    return out


def test_default_bit_identical_to_frozen():
    z0, pt, mask, _, _, _ = _toy_events(n_events=150, seed=3)
    for wb in (3, 4, 5):
        new = fast_histo_z0(z0, pt, mask, window_bins=wb)
        old = _fast_histo_z0_frozen(z0, pt, mask, window_bins=wb)
        assert np.array_equal(new, old)  # bitwise, not allclose


def test_flat_kernel_paths_equivalent():
    z0, pt, mask, _, _, _ = _toy_events(n_events=50, seed=4)
    default = fast_histo_z0(z0, pt, mask)
    explicit = fast_histo_z0(z0, pt, mask, kernel="flat")
    via_array = fast_histo_z0(z0, pt, mask, kernel_array=np.ones(3))
    assert np.array_equal(default, explicit)
    assert np.array_equal(default, via_array)


def test_vtx_z0_matches_fast_histo_z0():
    z0, pt, mask, d0, phi, _ = _toy_events(n_events=50, seed=5)
    res = fast_histo_vtx(z0, pt, mask, d0, phi)
    assert np.array_equal(res["z0_pv"], fast_histo_z0(z0, pt, mask))


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


def test_make_kernel_shapes():
    assert np.array_equal(make_kernel("flat", 3), np.ones(3))
    np.testing.assert_allclose(make_kernel("triangular", 3), [0.5, 1.0, 0.5])
    np.testing.assert_allclose(make_kernel("epanechnikov", 3), [0.75, 1.0, 0.75])
    g = make_kernel("gaussian", 3, sigma_bins=1.0)
    assert len(g) == 7 and g.max() == 1.0 and g[0] < 0.02  # 3-sigma edge
    with pytest.raises(ValueError):
        make_kernel("boxcar", 3)
    with pytest.raises(ValueError):
        make_kernel(kernel_array=np.zeros(3))


def test_kernel_avoids_midpoint_window():
    """Two similarly-hard vertices two bins apart: the flat boxcar prefers the
    window containing BOTH (centroid lands between them); a tapered kernel
    picks the harder vertex's own window."""
    from ngtagger.train.nnvtx import HISTO_MIN, HISTO_WIDTH

    zA = HISTO_MIN + 100.5 * HISTO_WIDTH  # bin 100
    zB = HISTO_MIN + 102.5 * HISTO_WIDTH  # bin 102
    z0 = np.array([[zA] * 10 + [zB] * 9])
    pt = np.ones_like(z0) * 10.0
    mask = np.ones_like(z0)

    flat = fast_histo_z0(z0, pt, mask)  # window [100,103): centroid between A and B
    tri = fast_histo_z0(z0, pt, mask, kernel="triangular")
    epa = fast_histo_z0(z0, pt, mask, kernel="epanechnikov")
    gau = fast_histo_z0(z0, pt, mask, kernel="gaussian", sigma_bins=0.8)
    assert abs(flat[0] - zA) > 0.5 * HISTO_WIDTH  # midpoint-pulled
    for found in (tri, gau):
        assert abs(found[0] - zA) < 0.5 * HISTO_WIDTH, found
    # epanechnikov's 0.75 shoulder is too flat for a 2-bin doublet: midpoint
    # wins iff shoulder_weight * (h_A + h_B) > h_A, i.e. shoulder > 100/190
    assert abs(epa[0] - zA) > 0.5 * HISTO_WIDTH


# --------------------------------------------------------------------------
# (dx, dy) estimator: convention + closure
# --------------------------------------------------------------------------


def test_convention_exact_recovery():
    """Noiseless d0 = x_v sin(phi) - y_v cos(phi) must invert exactly: proves
    the sign convention and the normal-equation algebra together."""
    rng = np.random.default_rng(7)
    xv, yv = 0.08, -0.05
    phi = rng.uniform(-np.pi, np.pi, (4, 30))
    z0 = rng.normal(1.0, 0.04, (4, 30))
    pt = rng.uniform(2, 30, (4, 30))
    d0 = xv * np.sin(phi) - yv * np.cos(phi)
    res = fast_histo_vtx(z0, pt, np.ones_like(z0), d0, phi)
    np.testing.assert_allclose(res["dx"], xv, atol=1e-9)
    np.testing.assert_allclose(res["dy"], yv, atol=1e-9)
    # perfect fit -> zero scatter (float roundoff floor)
    np.testing.assert_allclose(res["d0_scatter"], 0.0, atol=1e-8)


def test_closure_with_noise_lsq():
    """LSQ toys from a known vertex with d0 noise: unbiased recovery and a
    calibrated pull (~unit width) under full phi coverage."""
    xv, yv = 0.05, 0.03
    z0, pt, mask, d0, phi, _ = _toy_events(n_events=300, xv=xv, yv=yv, seed=11)
    res = fast_histo_vtx(z0, pt, mask, d0, phi, estimator="lsq")
    ok = np.isfinite(res["dx"]) & np.isfinite(res["sigma_dx"])
    assert ok.mean() > 0.9
    for par, truth, sig in (("dx", xv, "sigma_dx"), ("dy", yv, "sigma_dy")):
        pull = (res[par][ok] - truth) / res[sig][ok]
        assert abs(np.mean(pull)) < 0.25, (par, np.mean(pull))
        assert 0.6 < np.std(pull) < 1.6, (par, np.std(pull))


def test_closure_with_noise_isotropic_unbiased():
    """The isotropic shortcut is UNBIASED under full phi coverage (its cheap
    uncertainty model is only approximate, so we check centering not pull
    width)."""
    xv, yv = 0.05, 0.03
    z0, pt, mask, d0, phi, _ = _toy_events(n_events=400, xv=xv, yv=yv, seed=11)
    res = fast_histo_vtx(z0, pt, mask, d0, phi, estimator="isotropic")
    for par, truth in (("dx", xv), ("dy", yv)):
        assert abs(np.nanmean(res[par]) - truth) < 0.01, (par, np.nanmean(res[par]))


def test_biased_phi_coverage_lsq_vs_isotropic():
    """Quarter phi arc (0..pi/2): S_sc != 0 so the isotropic shortcut (which
    assumes S_sc = 0) acquires a bias, while the full LSQ solve still closes
    and phi_condition (det/(Sss*Scc) ~ 0.6 here) flags the imbalance."""
    xv, yv = 0.10, 0.06
    common = dict(n_events=400, xv=xv, yv=yv, sigma_d0=0.008, n_pu=0,
                  phi_lo=0.0, phi_hi=np.pi / 2, seed=13)
    z0, pt, mask, d0, phi, _ = _toy_events(**common)
    lsq = fast_histo_vtx(z0, pt, mask, d0, phi, estimator="lsq")
    iso = fast_histo_vtx(z0, pt, mask, d0, phi, estimator="isotropic")
    lsq_dist = np.hypot(np.nanmean(lsq["dx"]) - xv, np.nanmean(lsq["dy"]) - yv)
    iso_dist = np.hypot(np.nanmean(iso["dx"]) - xv, np.nanmean(iso["dy"]) - yv)
    assert lsq_dist < 0.01
    assert iso_dist > 5 * lsq_dist  # isotropic shortcut biased under the arc
    # phi_condition flags the imbalance (det/(Sss*Scc) well below 1)
    assert np.nanmedian(lsq["phi_condition"]) < 0.9


def test_degenerate_phi_guard():
    """All tracks at one phi: rank-1 normal matrix -> NaN, never a crash or a
    silently wrong number."""
    z0 = np.full((2, 20), 1.0) + np.linspace(-0.03, 0.03, 20)
    pt = np.full_like(z0, 5.0)
    phi = np.full_like(z0, 0.7)
    d0 = 0.05 * np.sin(phi)
    res = fast_histo_vtx(z0, pt, np.ones_like(z0), d0, phi)
    assert np.all(np.isnan(res["dx"])) and np.all(np.isnan(res["dy"]))
    assert np.all(res["phi_condition"] < 1e-6)


def test_d0_gate_excludes_displaced_tracks():
    """A prompt vertex plus displaced-track contamination: |d0| gate on the
    transverse solve restores the truth and shrinks the scatter, while the z
    histogram (window selection) is untouched by the gate."""
    rng = np.random.default_rng(21)
    xv, yv = 0.02, -0.01
    n_prompt, n_disp = 30, 20
    n = n_prompt + n_disp
    z = np.concatenate([rng.normal(1.0, 0.04, n_prompt), rng.normal(1.0, 0.04, n_disp)])
    phi = rng.uniform(-np.pi, np.pi, n)
    d0 = np.empty(n)
    d0[:n_prompt] = xv * np.sin(phi[:n_prompt]) - yv * np.cos(phi[:n_prompt]) + rng.normal(0, 0.005, n_prompt)
    d0[n_prompt:] = rng.uniform(-2, 2, n_disp)  # displaced/loose contamination
    Z, P, PHI, D0 = z[None], np.full((1, n), 5.0), phi[None], d0[None]
    ungated = fast_histo_vtx(Z, P, np.ones_like(Z), D0, PHI)
    gated = fast_histo_vtx(Z, P, np.ones_like(Z), D0, PHI, d0_gate=0.1)
    # z estimate identical (gate never touches the histogram)
    assert np.array_equal(ungated["z0_pv"], gated["z0_pv"])
    # gate recovers the vertex and collapses the scatter
    assert abs(gated["dx"][0] - xv) < 0.02 and abs(gated["dy"][0] - yv) < 0.02
    assert gated["d0_scatter"][0] < 0.5 * ungated["d0_scatter"][0]
    assert gated["n_window"][0] < ungated["n_window"][0]


def test_empty_window_and_no_tracks():
    z0 = np.zeros((1, 4))
    pt = np.zeros_like(z0)
    res = fast_histo_vtx(z0, pt, np.zeros_like(z0), np.zeros_like(z0), np.zeros_like(z0))
    assert res["n_window"][0] == 0 and np.isnan(res["dx"][0])


# --------------------------------------------------------------------------
# jagged wrapper + feature-group plumbing
# --------------------------------------------------------------------------


def _jagged_tracks(xv=0.06, yv=-0.04, n_events=40, seed=17, zero_d0=False):
    rng = np.random.default_rng(seed)
    builder = {"pt": [], "phi": [], "z0": [], "d0": []}
    for _ in range(n_events):
        n = rng.integers(8, 30)
        phi = rng.uniform(-np.pi, np.pi, n)
        builder["pt"].append(rng.uniform(2, 30, n))
        builder["phi"].append(phi)
        builder["z0"].append(rng.normal(rng.uniform(-5, 5), 0.04, n))
        d0 = np.zeros(n) if zero_d0 else xv * np.sin(phi) - yv * np.cos(phi) + rng.normal(0, 0.005, n)
        builder["d0"].append(d0)
    return ak.zip({k: ak.Array(v) for k, v in builder.items()})


def test_vertex_dxy_features_jagged_closure():
    xv, yv = 0.06, -0.04
    tracks = _jagged_tracks(xv=xv, yv=yv)
    out = vertex_dxy_features(tracks)
    assert set(out) == {"vtx_dx", "vtx_dy", "vtx_dxsig", "vtx_dysig"}
    assert abs(np.nanmedian(out["vtx_dx"]) - xv) < 0.01
    assert abs(np.nanmedian(out["vtx_dy"]) - yv) < 0.01
    # significances are large for a strongly displaced vertex
    assert np.nanmedian(np.abs(out["vtx_dxsig"])) > 3


def test_vertex_dxy_features_zero_d0_raises():
    tracks = _jagged_tracks(zero_d0=True)
    with pytest.raises(ValueError, match="4-parameter"):
        vertex_dxy_features(tracks)


def test_vertex_dxy_features_missing_branch_raises():
    tracks = _jagged_tracks()
    with pytest.raises(KeyError, match="phi"):
        vertex_dxy_features(tracks[["pt", "z0", "d0"]])


def test_feature_group_resolution():
    from ngtagger.data.features import resolve_feature_groups

    names = resolve_feature_groups(["baseline", "vertexdxy"])
    for n in ("vtx_dx", "vtx_dy", "vtx_dxsig", "vtx_dysig"):
        assert n in names
    with pytest.raises(KeyError):
        resolve_feature_groups(["vertexdy"])


# ---- end-to-end through the nano reader ----------------------------------

uproot = pytest.importorskip("uproot")

XV, YV = 0.09, -0.06


def _write_vtx_nano(tmp_path, n_events=6):
    """Minimal L1PFTrkNano-shaped file whose extended-track table encodes a
    known (XV, YV) via the TTTrack d0 convention. Per-event counts are
    pairwise distinct (uproot how="zip" groups jagged branches by offsets)."""
    phi_trk = np.array([0.3, 1.2, 2.1, -0.8, -2.4, 2.9])  # 6 tracks
    d0_trk = XV * np.sin(phi_trk) - YV * np.cos(phi_trk)
    tree = {
        "run": np.ones(n_events, dtype=np.uint32),
        "luminosityBlock": np.ones(n_events, dtype=np.uint32),
        "event": np.arange(n_events, dtype=np.uint64),
        "L1PuppiCand": ak.zip({
            "pt": [[30.0, 5.0, 18.0, 3.0, 1.0]] * n_events,
            "eta": [[0.1, 0.2, -0.3, -0.4, 4.0]] * n_events,
            "phi": [[0.0, 0.4, -1.0, -0.6, 3.0]] * n_events,
            "mass": [[0.14, 0.0, 0.14, 0.0, 0.0]] * n_events,
            "charge": [[1, 0, -1, 0, 0]] * n_events,
            "id": [[0, 3, 0, 2, 3]] * n_events,
            "z0": [[0.01, 0.0, -0.02, 0.0, 0.0]] * n_events,
            "dxy": [[0.001, 0.0, -0.002, 0.0, 0.0]] * n_events,
            "puppiWeight": [[1.0, 0.7, 1.0, 0.4, 0.1]] * n_events,
            "hwEmID": [[0, 1, 0, 0, 1]] * n_events,
            "hwTkQuality": [[3, 0, 2, 0, 0]] * n_events,
            "l1TrackIdx": [[0, -1, 1, -1, -1]] * n_events,
            "hgcClusterIdx": [[-1, 0, -1, -1, -1]] * n_events,
        }),
        "L1puppiJetSC4NG": ak.zip({
            "pt": [[35.0, 20.0]] * n_events, "eta": [[0.15, -0.35]] * n_events,
            "phi": [[0.05, -0.8]] * n_events, "et": [[35.0, 20.0]] * n_events,
        }),
        "L1SC4NGJetCands": ak.zip({
            "jetIdx": [[0, 0, 1, 1]] * n_events,
            "candIdx": [[0, 1, 2, 3]] * n_events,
            "slot": [[0, 1, 0, 1]] * n_events,
            "inTagger": [[True, True, True, True]] * n_events,
        }),
        "L1TExtTrack": ak.zip({
            "pt": [[20.0, 15.0, 10.0, 8.0, 5.0, 3.0]] * n_events,
            "phi": [phi_trk] * n_events,
            "rInv": [[1e-3] * 6] * n_events, "tanL": [[0.2] * 6] * n_events,
            "z0": [[0.05, 0.02, -0.03, 0.04, -0.05, 0.01]] * n_events,
            "d0": [d0_trk] * n_events,
            "chi2XYRed": [[1.1] * 6] * n_events, "chi2ZRed": [[0.9] * 6] * n_events,
            "chi2BendRed": [[0.5] * 6] * n_events, "trkMVA1": [[0.9] * 6] * n_events,
        }),
        "L1HGCCluster": ak.zip({
            "hOverE": [[0.1] * 7] * n_events, "sigmaRRTot": [[0.02] * 7] * n_events,
            "zBarycenter": [[350.0] * 7] * n_events, "eMax": [[5.0] * 7] * n_events,
            "sigmaZZ": [[1.0] * 7] * n_events,
        }),
        "GenJet": ak.zip({
            "pt": [[36.0, 21.0] + [10.0] * 6] * n_events,
            "eta": [[0.16, -0.36] + [3.0 + i for i in range(6)]] * n_events,
            "phi": [[0.06, -0.82] + [3.0] * 6] * n_events,
            "partonFlavour": [[5, 21] + [1] * 6] * n_events,
            "hadronFlavour": [[5, 0] + [0] * 6] * n_events,
        }),
        "GenVisTau": ak.zip({
            "pt": [[12.0] * 9] * n_events, "eta": [[3.0 + i for i in range(9)]] * n_events,
            "phi": [[2.5] * 9] * n_events, "charge": [[1] * 9] * n_events,
        }),
        "GenPart": ak.zip({
            "pt": [[40.0] * 10] * n_events, "eta": [[3.0 + i for i in range(10)]] * n_events,
            "phi": [[-2.0] * 10] * n_events,
            "pdgId": [[11] * 10] * n_events, "statusFlags": [[1] * 10] * n_events,
        }),
    }
    f = uproot.recreate(tmp_path / "vtxdxy.root")
    types = {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in tree.items()}
    f.mktree("Events", types, counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(tmp_path / "vtxdxy.root")


def test_load_jets_vertexdxy_group(tmp_path):
    from ngtagger.data.features import build_features
    from ngtagger.data.nano import load_jets

    path = _write_vtx_nano(tmp_path)
    jets, constituents, gen = load_jets(
        [path], n_const=8, feature_groups=["baseline", "vertexdxy"],
        cand_table="L1PuppiCand", track_table="L1TExtTrack",
    )
    # broadcast: every constituent of every jet carries the per-event scalar
    dx = ak.flatten(constituents.vtx_dx, axis=None)
    np.testing.assert_allclose(ak.to_numpy(dx), XV, atol=1e-4)
    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(constituents.vtx_dy, axis=None)), YV, atol=1e-4)

    X, names = build_features(jets, constituents, n_const=8,
                              feature_groups=["baseline", "vertexdxy"])
    i_dx = names.index("vtx_dx")
    filled = X[..., names.index("isfilled")] > 0
    np.testing.assert_allclose(X[..., i_dx][filled], XV, atol=1e-4)
    assert np.all(X[..., i_dx][~filled] == 0.0)  # padded slots zeroed


def test_load_jets_vertexdxy_missing_track_pt_raises(tmp_path):
    """A track table without pt/phi (e.g. the slim fixture) must fail loudly
    when vertexdxy is requested, not silently degrade."""
    from ngtagger.data.nano import load_jets

    path = _write_vtx_nano(tmp_path)
    with pytest.raises((KeyError, ak.errors.FieldNotFoundError)):
        load_jets([path], n_const=8, feature_groups=["baseline", "vertexdxy"],
                  cand_table="L1PuppiCand", track_table="L1TTrack")


# --------------------------------------------------------------------------
# vtx-study drivers (importable package functions + CLI wiring)
# --------------------------------------------------------------------------


def test_run_kernel_scan_keys_and_flat_vs_triangular(tmp_path):
    """run_kernel_scan writes the expected JSON schema, and the committed
    finding holds: the flat kernel's mean midpoint-pick rate is >= the
    triangular kernel's (a taper reduces midpoint picks). Small grid to keep
    the unit test cheap; the seed makes it deterministic."""
    import json

    from ngtagger.train.vtxstudy import run_kernel_scan

    payload = run_kernel_scan(str(tmp_path), seps=(2, 3), ratios=(0.8, 1.0),
                              n_events=120, seed=0, make_plot=False)
    out = tmp_path / "kernel_scan.json"
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == payload
    for key in ("config", "grid", "results", "single_vertex", "headline"):
        assert key in payload
    for k in ("flat", "triangular", "epanechnikov", "gaussian"):
        assert set(payload["results"][k]) == {"midpoint_rate", "wrong_rate"}
        assert k in payload["single_vertex"]
    h = payload["headline"]
    for key in ("flat_mean_midpoint_rate", "taper_mean_midpoint_rate",
                "best_taper", "midpoint_reduction_vs_flat",
                "single_vertex_res_penalty_cm"):
        assert key in h
    # committed finding: flat picks the midpoint at least as often as triangular
    assert h["flat_mean_midpoint_rate"] >= h["taper_mean_midpoint_rate"]["triangular"]


def test_vtx_study_cli_dispatches_kernel_scan(tmp_path):
    """The `vtx-study --kernel-scan` subcommand parses and dispatches to the
    package driver (real, but tiny, so it stays argparse-level cheap)."""
    from ngtagger.cli import main

    rc = main(["vtx-study", "--kernel-scan", "--outdir", str(tmp_path),
               "--no-plot"])
    assert rc == 0
    assert (tmp_path / "kernel_scan.json").exists()


def test_vtx_study_cli_requires_a_mode():
    """--kernel-scan / --realdata are a required mutually-exclusive group."""
    from ngtagger.cli import main

    with pytest.raises(SystemExit):
        main(["vtx-study", "--outdir", "unused"])
