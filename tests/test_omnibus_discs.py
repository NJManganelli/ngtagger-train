"""The disc extension of the SmartPixels omnibus, on synthetic inputs.

Pins the pieces that decide WHICH surfaces a refit visits and how they are named:
the mask semantics (character k = ILk AND IDk), the disc-plane crossing, the
producer's path order with its traversed-surface count, the visit sequence, the
combination count over surface codes 11-14 (an 8-way key packing once merged
disc crossings into the next track), the ParameterSets parser, the sigma-table
lookup and the monitor-summary diff.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from ngtagger.data import spix_nano as SPN

_DIR = pathlib.Path(__file__).resolve().parents[1] / "eval_refitq" / "combinatorics"
sys.path.insert(0, str(_DIR))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def OP():
    return _load("ot_projection")


@pytest.fixture(scope="module")
def RP():
    return _load("ot_refit_projection")


@pytest.fixture(scope="module")
def OSU():
    return _load("omnibus_support")


@pytest.fixture(scope="module")
def om():
    return _load("spix_combinatorics_omnibus")


def test_mask_surfaces():
    assert SPN.mask_surfaces("AAAA") == (1, 2, 3, 4, 11, 12, 13, 14)
    assert SPN.mask_surfaces("AIAI") == (1, 3, 11, 13)
    assert SPN.mask_surfaces("IIII") == ()
    with pytest.raises(ValueError):
        SPN.mask_surfaces("AAA")
    assert SPN.is_disc_surface(np.array([1, 4, 11, 14])).tolist() == [False, False, True, True]
    assert [SPN.surface_name(s) for s in SPN.SP_SURFACES] == \
        ["IL1", "IL2", "IL3", "IL4", "ID1", "ID2", "ID3", "ID4"]


def _geo():
    """A toy geometry: layers at r = 3, 6 (|z| <= 20) and discs at |z| = 25, 32 (r 3-16)."""
    return {"surfaces": (1, 2, 11, 12), "R": {1: 3.0, 2: 6.0}, "halfZ": {1: 20.0, 2: 20.0},
            "Z": {11: 25.0, 12: 32.0}, "rext": {11: (3.0, 16.0), 12: (3.0, 16.0)}}


def _helix(rinv, phi, tanl, z0, d0):
    return np.array([[rinv, phi, tanl, z0, d0]], float)


def test_disc_plane_crossing(OP):
    a = _helix(1e-4, 0.3, 2.0, 1.0, 0.0)
    s = OP.disc_s(a, 25.0)[0]
    p, _ = OP.helix_at(a, np.array([s]))
    assert p[0, 2] == pytest.approx(25.0)            # lands on the plane on its own side
    a_back = _helix(1e-4, 0.3, -2.0, 1.0, 0.0)
    p, _ = OP.helix_at(a_back, OP.disc_s(a_back, 25.0))
    assert p[0, 2] == pytest.approx(-25.0)
    assert np.isnan(OP.disc_s(_helix(1e-4, 0.3, 0.0, 1.0, 0.0), 25.0)[0])   # parallel to the disc
    # a soft track that turns back before the plane: no crossing
    assert np.isnan(OP.disc_s(_helix(0.05, 0.3, 0.05, 0.0, 0.0), 25.0)[0])


def test_path_order_and_traversed(OP, RP):
    geo = _geo()
    # forward track (eta ~ 1.9): IL1 at z ~ 9.5 (traversed), IL2 at z ~ 19 (traversed),
    # the discs at r ~ 7.9 and 10.1 (traversed)
    fwd = _helix(1e-4, 0.3, 3.2, 0.0, 0.0)
    P = RP.surface_path(fwd, geo)
    assert P["codes"][0].tolist() == [1, 2, 11, 12]
    assert P["nT"][0].tolist() == [1, 2, 3, 4]                      # insertion order 1, 2, 11, 12
    # very forward (eta ~ 2.6): IL2 crossed beyond its |z| extent -> on the path, not traversed
    vf = _helix(1e-4, 0.3, 6.5, 0.0, 0.0)
    P = RP.surface_path(vf, geo)
    order = P["codes"][0].tolist()
    assert order == [1, 11, 12, 2]
    nT = dict(zip(P["surfaces"], P["nT"][0].tolist()))
    assert nT[2] == nT[12]                                          # not traversed: no new material
    seq = RP.visit_sequence(P, RP.active_surfaces("AAAA"))
    assert seq[0].tolist() == [2, 12, 11, 1]                        # outside-in along the path
    seq = RP.visit_sequence(P, RP.active_surfaces("AIII"))
    assert seq[0].tolist() == [11, 1]                               # mask char 1 = IL1 + ID1
    # central track: the discs are not on its path
    cen = _helix(1e-4, 0.3, 0.2, 0.0, 0.0)
    P = RP.surface_path(cen, geo)
    assert P["codes"][0].tolist() == [1, 2, -1, -1]


def test_process_noise_counts_traversed(RP):
    a = _helix(1e-3, 0.3, 1.0, 0.0, 0.0)
    C1 = np.zeros((1, 5, 5)); C2 = np.zeros((1, 5, 5))
    t = np.array([0])
    RP.add_process_noise(a, C1, t, np.array([3]), np.array([2]), np.array([6.0]), 0.0057)
    RP.add_process_noise(a, C2, t, np.array([4]), np.array([2]), np.array([6.0]), 0.0057)
    assert np.allclose(C2, 2 * C1)                                  # two traversed surfaces = 2 theta0^2
    C3 = np.zeros((1, 5, 5))
    RP.add_process_noise(a, C3, t, np.array([2]), np.array([2]), np.array([6.0]), 0.0057)
    assert np.allclose(C3, C1)                                      # the max(1, ...) floor
    # the kink Jacobian: d(d0) = +r_s d(phi0)
    assert C1[0, 4, 1] / C1[0, 1, 1] == pytest.approx(6.0)


def test_combinations_over_disc_codes(om):
    """Two tracks, the first crossing IL1 and ID1-ID3: its disc crossings must stay
    with it (a code >= 8 once spilled into the next track's key)."""
    X = {c: [] for c in ("event", "trackIdx", "layer", "detId", "projLocalX", "projLocalY",
                         "projSigX", "projSigY")}
    K = {c: [] for c in ("event", "detId", "localX", "localY", "tpIdx")}
    det = 100
    for trk, surfs in ((0, (1, 11, 12, 13)), (1, (1,))):
        for s in surfs:
            det += 1
            for c, v in (("event", 0), ("trackIdx", trk), ("layer", s), ("detId", det), ("projLocalX", 0.0),
                         ("projLocalY", 0.0), ("projSigX", 1.0), ("projSigY", 1.0)):
                X[c].append(v)
            for i in range(2):                                      # two candidates everywhere
                for c, v in (("event", 0), ("detId", det), ("localX", 0.01 * i), ("localY", 0.0), ("tpIdx", -1)):
                    K[c].append(v)
    X = {k: np.asarray(v) for k, v in X.items()}
    X["trk_tpIdx"] = np.full(len(X["layer"]), -1)
    K = {k: np.asarray(v) for k, v in K.items()}
    combos, ntrk, _, _ = om._combinations_per_track(X, K, [2.0])
    assert ntrk == 2
    assert sorted(combos[2.0].tolist()) == [2.0, 16.0]


def test_pset_parser(OSU):
    blob = (b"<@module_edm_type=+Z(EDProducer\x00);@module_label=+Z(prodA\x00);"
            b"@module_type=+Z(L1SmartPixelsTrackProducer\x00);seedlessRefitMaxKFUpdates=+I(+8);"
            b"seedlessRefitUseAngles=+Z(alphaBeta\x00);seedlessRefitWindowDiscR=+d({0.3,0.3,0.3,0.3});"
            b"smartPixelsActiveLayers=+Z(1010\x00)>@\x00\x01"
            b"<@module_edm_type=+Z(EDProducer\x00);@module_label=+Z(tab\x00);"
            b"@module_type=+Z(L1TTTrackFlatTableProducer\x00);name=+Z(L1TSmartPixelsTrack\x00);"
            b"src=+t(prodA:Level1TTTracks)>")
    lab, p = OSU.producer_params(blob, "L1TSmartPixelsTrack")
    assert lab == "prodA"
    cfg = OSU.refit_producer_config(p)
    assert cfg == {"useAngles": "alphaBeta", "maxKFUpdates": 8, "windowDiscR": (0.3, 0.3, 0.3, 0.3)}
    assert p["smartPixelsActiveLayers"] == "1010"


def test_sigma_table(OSU, tmp_path):
    J = {"corrections": [
        {"name": n, "data": {"nodetype": "category", "input": "surface", "content": [
            {"key": 1, "value": {"nodetype": "binning", "edges": [0.0, 1.0, 12.0], "content": [0.1, 0.5]}},
            {"key": 11, "value": {"nodetype": "binning", "edges": [0.0, 2.0, 12.0], "content": [0.2, 0.9]}}]}}
        for n in ("spix_emu_alpha_sigma", "spix_emu_beta_sigma")]}
    p = tmp_path / "sigma.json"
    p.write_text(json.dumps(J))
    T = OSU.SigmaTable(str(p))
    v = T.evaluate("beta", np.array([1, 1, 11, 11, 11]), np.array([0.5, 1.5, 1.0, 3.0, 40.0]))
    assert v.tolist() == [0.1, 0.5, 0.2, 0.9, 0.9]                  # clamped above the top edge
    assert OSU.width68(np.random.default_rng(0).normal(0, 2.0, 200000)) == pytest.approx(2.0, rel=0.02)


def test_monitor_diff():
    md = _load("monitor_diff")
    A = {"provenance": {"build": "x", "created": "t0"}, "R1": {"OT": {"d0": 100.0, "z0": 50.0}}, "flags": ["a"]}
    B = {"provenance": {"build": "y", "created": "t1"}, "R1": {"OT": {"d0": 110.0, "z0": 50.5}, "SP": {"d0": 1.0}},
         "flags": ["a"]}
    prov, ch, only_a, only_b = md.compare(A, B, rel=0.05)
    assert prov == [("build", "x", "y")]                             # 'created' is not a difference
    assert [c[0] for c in ch] == ["R1.OT.d0"]
    assert only_b == ["R1.SP.d0"] and only_a == []


def test_selection_closure(RP):
    """The producer's window + metric: box per surface, the first maxHitsPerWindow
    in readout order, lowest chi2 (first on a tie), truncation when another
    cluster of the module follows the last one kept."""
    cfg = dict(RP.PRODUCER_DEFAULTS, maxHitsPerWindow=3, useAngles="alpha")
    # crossing 0: barrel IL1 (window 0.05 x 0.45), crossing 1: disc ID2 (0.2 x 0.3)
    H = {"event": np.array([0, 0]), "detId": np.array([10, 20]), "layer": np.array([1, 12]),
         "projLocalX": np.array([0.0, 0.0]), "projLocalY": np.array([0.0, 0.0]),
         "projCotAlpha": np.array([0.0, 0.0]), "projCotBeta": np.array([0.0, 0.0])}
    #            in-window ............... out   in    in    in (4th -> truncated)
    xs = [0.04, 0.01, 0.06, 0.03, 0.0] + [0.1, 0.25, 0.15, 0.0]
    ys = [0.0, 0.0, 0.0, 0.40, 0.0] + [0.0, 0.0, 0.2, 0.31]
    K = {"event": np.zeros(9, int), "detId": np.array([10] * 5 + [20] * 4),
         "localX": np.array(xs), "localY": np.array(ys),
         "sigX": np.full(9, 0.01), "sigY": np.full(9, 0.1),
         "localCotAlpha": np.zeros(9), "localCotBeta": np.zeros(9),
         "sigAlpha": np.ones(9), "sigBeta": np.ones(9),
         "hasAlpha": np.ones(9, np.uint8), "hasBeta": np.ones(9, np.uint8)}
    sc = RP.selection_closure(H, K, np.array([0]), cfg)
    # IL1: clusters 0, 1, 3 in window (2 is beyond 0.05), the 3rd kept is row 3 and row 4
    # follows -> truncated; the best is row 1 (dx 0.01)
    assert sc["mult"].tolist() == [3, 2]
    assert sc["truncated"].tolist() == [True, False]
    assert sc["best"].tolist() == [1, 5]


def test_track_parameter_residuals(om):
    """R1's residual conventions: pT relative, phi0 wrapped, cotTheta = sinh(eta),
    the q/pT pull through rInv."""
    TP = {"pt": np.array([4.0]), "eta": np.array([0.5]), "phi0": np.array([3.1]), "d0": np.array([0.001]),
          "z0": np.array([1.0]), "charge": np.array([1.0])}
    kpt = 0.0057
    trk = {"pt": np.array([4.4]), "phi": np.array([-3.1]), "tanL": np.array([np.sinh(0.5) + 0.01]),
           "z0": np.array([1.002]), "d0": np.array([0.0015]), "rInv": np.array([kpt / 4.4]),
           "cov_rInv_rInv": np.array([1e-8]), "cov_phi_phi": np.array([1e-6]), "cov_tanL_tanL": np.array([1e-4]),
           "cov_z0_z0": np.array([1e-6]), "cov_d0_d0": np.array([1e-8])}
    R = om._param_residuals(trk, TP, np.array([0]), kpt)
    assert R["pt"][0][0] == pytest.approx(0.1)
    assert R["phi"][0][0] == pytest.approx(2 * np.pi - 6.2)          # wrapped across +-pi
    assert R["tanL"][0][0] == pytest.approx(0.01)
    assert R["d0"][1][0] == pytest.approx(0.0005 / 1e-4)
    assert R["pt"][1][0] == pytest.approx((kpt / 4.4 - kpt / 4.0) / 1e-4)
