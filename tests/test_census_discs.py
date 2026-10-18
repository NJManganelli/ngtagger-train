"""Census format 36, the discs in the joint census (WP-Disc half B): the code space, the disc seeds and menus, the
disc projection's geometry (r on a z plane), the OT disc windows, and the census KF on disc hits."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_refitq" / "combinatorics"))
import kf_emulation as KF          # noqa: E402
import seed_arity as SA            # noqa: E402
import tp_findability as TF        # noqa: E402
import tracklet_topology_cost as M  # noqa: E402


def test_code_space():
    assert M.IT_DISC_CODES == (5, 6, 7, 8) and M.OT_DISC_CODES == (21, 22, 23, 24, 25)
    assert M.is_disc([1, 4, 5, 8, 11, 16, 21, 25]).tolist() == [False, False, True, True, False, False, True, True]
    assert M.is_it([4, 8, 11]).tolist() == [True, True, False]
    assert [SA.NAME[c] for c in (5, 8, 21, 25)] == ["ID1", "ID4", "OD1", "OD5"]
    assert TF.NAME == SA.NAME
    # the KF's hit columns: the IT discs after IL4, the OT discs after OL6
    assert KF.LAYER_ORDER == (1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25)


def test_ot_seed_names_and_disc_windows():
    assert M.ot_seed_name((21, 22)) == "D1D2" and M.ot_seed_name((11, 21)) == "L1D1"
    assert M.ot_seed_name((12, 21)) == "L2D1" and M.ot_seed_name((11, 12)) == "L1L2"
    assert M.ot_seed_name((5, 21)) is None and M.ot_seed_name((21, 23)) is None
    # D1D2 projects to D3 (PS 0.15 / 0.5, 2S 0.15 / 3.6) and not to D1 (its own disc)
    assert M.ot_disc_match_window(3, (21, 22)) == ((0.15, 0.5), (0.15, 3.6))
    assert M.ot_disc_match_window(1, (21, 22)) is M.OT_PROJ_FORBIDDEN
    # an IT seed gets the disc's most generous window, widened (a ceiling only)
    (rp, rr), _ = M.ot_disc_match_window(1, (5, 6))
    assert rp == M.OT_RPHI_CUT_PS_D[0].max() * M.OT_IT_SEED_WIDEN and rr == M.OT_R_CUT_PS_D[0].max() * M.OT_IT_SEED_WIDEN


def test_disc_seed_menu():
    all_ot = SA.OT_BARREL + SA.OT_DISCS
    tags = [sd.tag for sd in SA.disc_seeds(SA.ID, all_ot)]
    assert tags == ["OD1+OD2", "OD3+OD4", "OL1+OD1", "OL2+OD1", "ID1+ID2", "ID2+ID3", "ID3+ID4", "ID1+OD1"]
    assert [sd.tag for sd in SA.disc_seeds((), all_ot)] == ["OD1+OD2", "OD3+OD4", "OL1+OD1", "OL2+OD1"]
    assert [sd.tag for sd in SA.disc_seeds(SA.ID, all_ot, {"ID1+ID2"})] == ["ID1+ID2"]
    cfg = {"ot_chain": {"discs": True}, "it_discs": True, "disc_seeds": ["ID2+ID3", "OD1+OD2"]}
    dk = TF.disc_menu(cfg)
    assert [sd.tag for sd in dk["seeds"]] == ["OD1+OD2", "ID2+ID3"] and dk["it"] == SA.ID
    assert TF.disc_menu({"ot_chain": {"discs": False}, "it_discs": False, "disc_seeds": None}) is None
    # every build follows into its IT discs and the OT; the disc entries come after the barrel's
    rmed = {L: float(L) for L in SA.IL + SA.ID + SA.OT_BARREL + SA.OT_DISCS}
    ents, builds = TF.census_entries(rmed, ["AAAA", "AAAI"], None, None, None, dk)
    tags = [e.tag for e in ents]
    assert tags.index("OD1+OD2") > tags.index("OL1+OL2") and tags.index("ID2+ID3") > tags.index("OL1+OL2")
    e = ents[tags.index("ID2+ID3")]
    assert 5 in e.follow and 8 in e.follow and 21 in e.follow and 6 not in e.follow and 7 not in e.follow
    assert all(any(ents[i].tag == "ID2+ID3" for i in idx) for idx in builds.values())


def test_disc_projection_geometry():
    """disc_r_pred inverts the barrel's arc length, and candidates_by_r finds the clusters of a helix."""
    kap, cot, z0 = 0.4, 3.0, 1.5
    r = np.array([5.0, 10.0, 15.0])
    s = r * (1.0 + np.square(r * M.C_BEND * kap) / 6.0)
    z = z0 + s * cot
    rp, sp = M.disc_r_pred(z, z0, cot, kap)
    assert np.allclose(rp, r, rtol=1e-6) and (sp > 0).all()
    # a pair on the other side reaches nothing
    assert M.disc_r_pred(-30.0, z0, cot, kap)[1] < 0
    # the index search: clusters on the helix at ID1's z, plus decoys on the other side and far in r
    zc = np.array([25.0, 25.9, 25.0, -25.0])
    rc = np.array([M.disc_r_pred(25.0, z0, cot, kap)[0], M.disc_r_pred(25.9, z0, cot, kap)[0], 13.5, 7.5])
    ev = np.zeros(4, np.int64)
    ridx = M.build_disc_r_index(np.arange(4), ev, zc, rc, np.full(4, 0.003))
    got = set()
    for ja, jb in M.candidates_by_r(np.zeros(1, np.int64), np.array([z0]), np.array([cot]), np.array([kap]),
                                    np.array([1e-3]), 1e-3, ridx):
        got |= set(jb.tolist())
    assert got >= {0, 1} and 3 not in got and 2 not in got


def test_kf_fits_a_forward_track_on_disc_hits():
    """A straight forward track with hits on IL1, ID1-ID4 and OD1-OD2: the census KF returns its cot and z0, the
    disc hits weighted by |cot| sigma_r in z."""
    cot, z0, phi0 = 4.0, 2.0, 0.3
    layers = KF.LAYER_ORDER
    zs = {5: 25.0, 6: 32.0, 7: 41.0, 8: 52.0, 21: 131.0, 22: 156.0}
    hits = {1: 3.0}
    hits.update({L: (z - z0) / cot for L, z in zs.items()})
    n = len(hits)
    U = {"layer": np.array(list(hits)), "globalR": np.array(list(hits.values())),
         "globalZ": np.array([z0 + cot * r for r in hits.values()]), "globalPhi": np.full(n, phi0),
         "sigY": np.r_[0.002, np.full(4, 0.003), 0.04, 0.04], "sigX": np.full(n, 0.001),
         "ot_ps": np.r_[np.zeros(5, bool), True, True], "ot_tilted": np.zeros(n, bool), "event": np.zeros(n, np.int64)}
    Q = {"dir_ok": np.zeros(n, bool), "ovf_a": np.zeros(n, bool), "ovf_z": np.zeros(n, bool),
         "kap_a": np.zeros(n), "s_kap": np.ones(n), "cot_b": np.zeros(n), "s_cot": np.ones(n)}
    G = np.full((1, len(layers)), -1, np.int64)
    for i, L in enumerate(hits):
        G[0, layers.index(L)] = i
    fit = KF.fit_tracks(U, Q, (np.array([1]), np.array([2]), None), use_angles=False, gidx=G)
    assert abs(fit["cot"][0] - cot) < 1e-6 and abs(fit["z0"][0] - z0) < 1e-4 and fit["nhit"][0] == n
    # an r error of one sigma on an IT disc hit (its sigY) is a cot x sigma z residual with a cot x sigma error:
    # chi2 rz ~ 1, the other hits pinning the line; an OT disc stub's r sigma is its module's (PS 0.1467 / sqrt 12)
    for i, sig, want in ((3, 0.01, 1.0), (5, 0.1467 / math.sqrt(12.0), 1.0), (5, 0.1, (0.1 * math.sqrt(12.0) / 0.1467) ** 2)):
        V = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in U.items()}
        V["globalR"][i] += sig
        if i == 3:
            V["sigY"][i] = sig
        fit2 = KF.fit_tracks(V, Q, (np.array([1]), np.array([2]), None), use_angles=False, gidx=G)
        assert 0.8 * want < fit2["chi2_rz"][0] < 1.25 * want
