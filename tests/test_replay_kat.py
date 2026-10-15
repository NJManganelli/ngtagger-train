"""WP-C entry gate E6: known-answer tests for every replay switch (PLAN §2, §4 E6, §10 PD2-PD3, PD4 (c)).

Each switch: an identity at its neutral (producer) setting, a non-null effect, its own-quantity test, and a
mis-wired variant that the test must catch. Real data: canon_v3 AAAA file 4 (the known-good file), the
first 5 events; toys: synthetic arrays. Run: pytest tests/test_replay_kat.py (about a minute).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "eval_refitq" / "replay"
sys.path.insert(0, str(HERE))

import angle_import as AI          # noqa: E402
import arms as AR
import cuts as CU                  # noqa: E402
import common as CM                # noqa: E402
import engine as E                 # noqa: E402
import identity as ID              # noqa: E402
from common import OP, ORP         # noqa: E402
from ngtagger.spdefs import angles as SA   # noqa: E402

pytestmark = pytest.mark.skipif(not CM.input_path("AAAA", 4).exists(), reason="canon_v3 file 4 not mounted")


class ConstCuts:
    """A cut-file stand-in: every chi2 cut and window cut equal to `v`."""

    rr_statistic = "smoothed"

    def __init__(self, v):
        self.v = v

    def chi2(self, consumer, dims, cov, surface, first, scen=None):
        return np.full(len(np.atleast_1d(surface)), self.v, float)

    def window(self, cov, surface, first, kb_a, kb_b):
        return np.full((len(surface), 2), self.v, float)


@pytest.fixture(scope="module")
def data():
    f = CM.input_path("AAAA", 4)
    cfg = CM.producer_config(f)
    geo = OP.load_geometry(CM.GEOMETRY)
    A = next(CM.iterate(f, AR.branches(f), chunk=5, nev=5))[1]
    ctx = AR.prepare(A, geo)
    return {"A": A, "cfg": cfg, "geo": geo, "ctx": ctx, "tools": AR.Tools(cfg, cuts=ConstCuts(np.inf))}


def run(d, arm, **kw):
    return AR.run_arm(d["ctx"], d["geo"], d["cfg"], "AAAA", arm, d["tools"], **kw)


# ---------------------------------------------------------------- identities with ORP
def test_forced_identity(data):
    r = ID.forced_identity(data["A"], data["geo"], data["cfg"], "AAAA")
    assert r["identical"] and r["n_calls_a"] > 0


def test_orp_cal_identity(data):
    r = ID.orp_cal_identity(data["A"], data["geo"], data["cfg"], "AAAA", data["tools"])
    assert r["identical"] and r["n_calls_a"] > 0


def test_orp_cal_identity_known_bads(data):
    """E2 / E3 known-bads (PD10 (c)): the swapped R-table lookup and the half-strength material break it."""
    for ex in ({"rtable_swap": True}, {"material_scale": 0.5}):
        la, lb = [], []
        with ID.recording(ORP, "kalman_update", la):
            ORP.run_chunk(data["A"], data["geo"], "AAAA", data["cfg"]["useAngles"], ("innovation",), 4.0, False,
                          data["tools"].cal, CM.REFIT_HIT, data["cfg"])
        use = data["ctx"]["cls"] >= 0
        with ID.recording(E, "kf_update", lb):
            AR.run_arm(data["ctx"], data["geo"], data["cfg"], "AAAA", AR.orp_cal(data["cfg"], **ex), data["tools"], tracks=use)
        assert not ID.compare_logs(la, lb, np.flatnonzero(use))["identical"]


def test_box_identity(data):
    r = ID.box_identity(data["A"], data["geo"], data["cfg"])
    assert r["same_pick"] == 1.0 and r["same_truncated"] == 1.0 and r["same_mult"] == 1.0


# ---------------------------------------------------------------- angle use: selection vs update
def _first_choice(out):
    return np.where(out["rec"]["acc"][:, 0], out["rec"]["ci"][:, 0], -1)


def test_angle_use_split(data):
    cfg = data["cfg"]
    none = run(data, AR.inn(cfg, name="none", use=frozenset()))
    upd = run(data, AR.inn(cfg, name="upd", use=AR.use_set(("update",), regions=("barrel",))))
    both = run(data, AR.inn(cfg))
    sel = run(data, AR.inn(cfg, name="sel", use=AR.use_set(("rank",), regions=("barrel",))))
    # one shared state at the first-visited surface: identical choices there
    assert np.array_equal(_first_choice(upd), _first_choice(none))
    assert np.array_equal(_first_choice(sel), _first_choice(both))
    # non-null: the update-only arm moves the final parameters of tracks with an angle-bearing update
    refit = upd["refit"] & none["refit"] & (upd["rec"]["acc"] == none["rec"]["acc"]).all(1)
    moved = (upd["a"][refit] != none["a"][refit]).any(1)
    assert moved.mean() >= 0.99


def test_selection_only_equals_positions_only_refit(data):
    """Selection-only's final parameters = a positions-only forced update on its own hits."""
    cfg, geo, ctx = data["cfg"], data["geo"], data["ctx"]
    sel_arm = AR.inn(cfg, name="sel", use=AR.use_set(("rank",), regions=("barrel",)))
    sel = run(data, sel_arm)
    t = np.flatnonzero(sel["refit"])
    path, seq = AR.visit_plan(ctx, geo, cfg, "AAAA", sel_arm)
    ci = np.where(sel["rec"]["acc"][t], sel["rec"]["ci"][t], -1)
    st = AR.refit_forced(AR.inn(cfg, name="pos", use=frozenset()), ctx, geo, cfg, data["tools"], t, seq[t],
                         sel["rec"]["mi"][t], ci, path, ORP.pt_constant(ctx["T"]))
    assert np.array_equal(st.a, sel["a"][t])


def test_single_mask_known_bad(data):
    """Mis-wired: one mask for ranking and update (ORP's use_dim) makes 'selection-only' update with angles,
    which the positions-only refit identity catches."""
    cfg, geo, ctx = data["cfg"], data["geo"], data["ctx"]
    bad = AR.inn(cfg, name="single-mask", use=AR.use_set(("rank", "update"), regions=("barrel",)))
    out = run(data, bad)
    t = np.flatnonzero(out["refit"])
    path, seq = AR.visit_plan(ctx, geo, cfg, "AAAA", bad)
    ci = np.where(out["rec"]["acc"][t], out["rec"]["ci"][t], -1)
    st = AR.refit_forced(AR.inn(cfg, name="pos", use=frozenset()), ctx, geo, cfg, data["tools"], t, seq[t],
                         out["rec"]["mi"][t], ci, path, ORP.pt_constant(ctx["T"]))
    assert not np.array_equal(st.a, out["a"][t])


# ---------------------------------------------------------------- own quantity
def test_covariance_switch_keeps_first_choice(data):
    cfg = data["cfg"]
    a = run(data, AR.pm(cfg))
    b = run(data, AR.pm(cfg, name="pm-calseed", seed_cov="calibrated"))
    assert np.array_equal(_first_choice(a), _first_choice(b))


def test_rtables_carried_by_covariance_known_bad(data):
    """Mis-wired: the R tables switched together with the covariance change first-surface choices under the
    producer metric (the own-quantity test catches it); non-null for the R tables themselves."""
    cfg = data["cfg"]
    a = run(data, AR.pm(cfg))
    b = run(data, AR.pm(cfg, name="pm-calseed-R", seed_cov="calibrated", pos_err="rtables"))
    assert not np.array_equal(_first_choice(a), _first_choice(b))


def test_window_switch_keeps_metric_values():
    """The window switch changes the candidate set, not the metric of a common candidate (the metric is a
    function of (r, S, sig, dims) only)."""
    rng = np.random.default_rng(1)
    r = rng.normal(size=(50, 4))
    Sm = np.einsum("nij,nkj->nik", *(2 * [rng.normal(size=(50, 4, 4))])) + np.eye(4)
    sig = np.abs(rng.normal(size=(50, 4))) + 0.1
    dims = np.ones((50, 4), bool)
    for metric in ("producer", "innovation"):
        v_all = AR.rank_values(metric, r, Sm, sig, dims)
        v_sub = AR.rank_values(metric, r[::2], Sm[::2], sig[::2], dims[::2])
        assert np.array_equal(v_all[::2], v_sub)


def test_hit_gate_identities(data):
    cfg = data["cfg"]
    off = run(data, AR.inn(cfg))
    tools_inf = AR.Tools(cfg, cuts=ConstCuts(np.inf))
    on_inf = AR.run_arm(data["ctx"], data["geo"], cfg, "AAAA", AR.inn(cfg, name="g", gate=True), tools_inf)
    assert np.array_equal(off["a"], on_inf["a"])
    tools0 = AR.Tools(cfg, cuts=ConstCuts(0.0))
    on0 = AR.run_arm(data["ctx"], data["geo"], cfg, "AAAA", AR.inn(cfg, name="g0", gate=True), tools0)
    assert on0["nacc"].sum() == 0 and np.array_equal(on0["a"], data["ctx"]["a"])


def test_reject_and_refit_identities(data):
    cfg, geo, ctx = data["cfg"], data["geo"], data["ctx"]
    off = run(data, AR.inn(cfg))
    on_inf = AR.run_arm(ctx, geo, cfg, "AAAA", AR.inn(cfg, name="rr", rr=True), AR.Tools(cfg, cuts=ConstCuts(np.inf)))
    assert np.array_equal(off["a"], on_inf["a"])
    # a removed hit gives exactly the forced refit (from the OT seed) without it (the 4-sigma window, so that a
    # zero cut acts on reject-and-refit only)
    arm = AR.inn(cfg, name="rr0", rr=True, rr_max_iter=1, window_cut="nsig")
    on0 = AR.run_arm(ctx, geo, cfg, "AAAA", arm, AR.Tools(cfg, cuts=ConstCuts(0.0)))
    t = np.flatnonzero(on0["removed"].any(1) & on0["refit"])
    assert len(t)
    path, seq = AR.visit_plan(ctx, geo, cfg, "AAAA", arm)
    ci = np.where(on0["rec"]["acc"][t], on0["rec"]["ci"][t], -1)
    st = AR.refit_forced(arm, ctx, geo, cfg, data["tools"], t, seq[t], on0["rec"]["mi"][t], ci, path,
                         ORP.pt_constant(ctx["T"]))
    assert np.array_equal(st.a, on0["a"][t])          # material changes C only, so a compares directly


def test_loo_smoothed_known_answers(data):
    """PD17: without material the smoothed leave-one-out equals the final-state one (no process noise:
    every step carries the same helix); with material they differ, and the smoother leaves a hit with no
    inner hits at its forward prediction."""
    cfg, geo, ctx = data["cfg"], data["geo"], data["ctx"]
    st = {}
    for mode in ("smoothed", "final"):
        for scale in (0.0, 1.0):
            out = run(data, AR.inn(cfg, name=f"loo-{mode}-{scale}", loo_pred=mode, material_scale=scale), loo=True)
            st[mode, scale] = out["loo"]
    for k in AR.DIMS:
        a, b = st["smoothed", 0.0]["dims"][k], st["final", 0.0]["dims"][k]
        ok = np.isfinite(a) & np.isfinite(b)
        assert ok.mean() > 0.99 and np.allclose(a[ok], b[ok], rtol=1e-6, atol=1e-9)
    a, b = st["smoothed", 1.0]["dims"]["xy"], st["final", 1.0]["dims"]["xy"]
    outer = st["smoothed", 1.0]["n_inner"] >= 2
    assert np.median(a[outer] / b[outer]) < 0.8          # the final state misses the scattering to the inner hits
    # the smoother's algebra: a hit with no inner hits keeps the forward prediction (smoothed = filtered there)
    hist = {"fa": np.zeros((1, 3, 5)), "pa": np.zeros((1, 3, 5)), "fC": np.tile(np.eye(5), (1, 3, 1, 1)),
            "pC": np.tile(2 * np.eye(5), (1, 3, 1, 1))}
    hist["fa"][0, 2] = 1.0
    aS, CS = AR.smooth_back(hist, np.array([2]))
    assert np.array_equal(aS, hist["fa"][:, 2]) and np.array_equal(CS, hist["fC"][:, 2])


def test_rr_keeps_one_smartpixels_cluster(data):
    """PD19: with a zero rr cut every hit is a removal candidate, yet every refitted track keeps exactly one
    SmartPixels cluster, and the tracks whose last cluster the rule kept are flagged."""
    cfg, geo, ctx = data["cfg"], data["geo"], data["ctx"]
    arm = AR.inn(cfg, name="rr-all", rr=True, rr_max_iter=8, window_cut="nsig")
    out = AR.run_arm(ctx, geo, cfg, "AAAA", arm, AR.Tools(cfg, cuts=ConstCuts(0.0)))
    had = out["removed"].any(1) | out["rr_protected"]
    assert had.any()
    assert np.all(out["rec"]["acc"][had].sum(1) == 1) and np.all(out["rr_protected"][had])


def test_wide_window_search_bound(data, monkeypatch):
    """PD18: the 'gate99' window (the 99 % hit cut) is searched within WINDOW_SEARCH_NSIG of the prediction;
    a wider search changes no selection, and the window accepts no candidate beyond its bin's cut."""
    if not CM.CUTS.exists():
        pytest.skip("no cut file")
    cfg, geo, ctx = data["cfg"], data["geo"], data["ctx"]
    tools = AR.Tools(cfg, cuts=CU.Cuts.load(str(CM.CUTS)))
    base = AR.run_arm(ctx, geo, cfg, "AAAA", AR.inn(cfg), tools)
    monkeypatch.setattr(AR, "WINDOW_SEARCH_NSIG", 40.0)
    wide = AR.run_arm(ctx, geo, cfg, "AAAA", AR.inn(cfg), tools)
    assert np.array_equal(base["rec"]["ci"], wide["rec"]["ci"]) and np.array_equal(base["rec"]["acc"], wide["rec"]["acc"])
    g = base["rec"]["gstat"][base["rec"]["acc"]]
    S = base["rec"]["surface"][base["rec"]["acc"]]
    first = np.nonzero(base["rec"]["acc"])[1] == 0
    assert np.all(g <= tools.cuts.chi2("gate", "xy", "calibrated", S, first))
    n4 = AR.run_arm(ctx, geo, cfg, "AAAA", AR.inn(cfg, window_cut="nsig"), tools)
    assert base["nacc"].sum() > n4["nacc"].sum()        # the 4-sigma ellipse accepts fewer hits


def test_inner_material_changes_covariance_only(data):
    cfg = data["cfg"]
    a = run(data, AR.inn(cfg, name="nomat", inner_material=False))
    b = run(data, AR.inn(cfg))
    assert np.array_equal(a["a"], b["a"])
    r = a["refit"]
    assert (b["C"][r, 4, 4] > a["C"][r, 4, 4]).mean() > 0.99


def test_material_on_parameters_known_bad():
    """Mis-wired: material applied to the parameters (a helix kink shifting d0) would move them; the real
    add_material leaves a untouched (checked on a toy state)."""
    a = np.array([[0.002, 0.3, 0.5, 1.0, 0.01]])
    C = np.eye(5)[None] * 1e-6
    geo = {"R": {0: 0.0, 1: 3.0, 2: 6.3, 3: 10.6, 4: 14.8}}
    cal = {"planes": [[2.3, 0.0022761, "pipe"]], "ot_inner_r_cm": 22.0, "material_scale": 1.0}
    a0, C0 = a.copy(), C.copy()
    ORP.add_material(a, C, np.array([0]), 0, np.array([1]), geo, cal, 0.0114)
    assert np.array_equal(a, a0) and not np.array_equal(C, C0)


# ---------------------------------------------------------------- the k knob (decision 10, PD3)
def _cluster_cols(ctx, ci):
    return {c: ctx["K"][c][ci] for c in AI.GLOBAL_COLS}


def test_k_knob(data):
    ctx, geo = data["ctx"], data["geo"]
    K = ctx["K"]
    ci = np.flatnonzero((K["mi"] >= 0) & (K["hasAlpha"] > 0) & (K["hasBeta"] > 0))
    mi = K["mi"][ci]
    surf = geo["surface"][mi]
    cols = _cluster_cols(ctx, ci)
    la, lb = K["localCotAlpha"][ci].astype(float), K["localCotBeta"][ci].astype(float)
    ex, ey, ez = geo["ex"][mi], geo["ey"][mi], geo["ez"][mi]
    # k = 1: untouched, bit for bit
    a1, b1 = AI.knob(cols, ex, ey, ez, surf, 1.0, 1.0, la, lb)
    assert np.array_equal(a1, la) and np.array_equal(b1, lb)
    # k = 0 on barrel class T: the truth local angles, to the 12-bit storage (non-null and discriminating:
    # a frame or sign error in the rotation back to local would fail it)
    a0, b0 = AI.knob(cols, ex, ey, ez, surf, 0.0, 0.0, la, lb)
    T = SA.class_T(K["tpEmuClass"][ci]) & (surf <= 4) & SA.usable(cols)
    tpa, tpb = K["tpLocalCotAlpha"][ci].astype(float), K["tpLocalCotBeta"][ci].astype(float)
    T &= (np.abs(tpa) < 12) & (np.abs(tpb) < 12)
    tol = lambda x: 5e-4 * np.abs(x) + 1e-4
    assert (np.abs(a0[T] - tpa[T]) <= tol(tpa[T])).mean() > 0.99
    assert (np.abs(b0[T] - tpb[T]) <= tol(tpb[T])).mean() > 0.99
    # class C and the discs untouched at any k (PD3)
    keep = ~SA.class_T(K["tpEmuClass"][ci]) | (surf > 10)
    a5, b5 = AI.knob(cols, ex, ey, ez, surf, 0.5, 0.5, la, lb)
    assert np.array_equal(a5[keep], la[keep]) and np.array_equal(b5[keep], lb[keep])
    assert (a5[T] != la[T]).mean() > 0.9
    # the per-angle knob acts on the table's (global) angles: k_alpha = 0.5, k_beta = 1 leaves the global
    # cot theta (recovered from the local result with the spdefs frame rule) at its emitted value, while
    # alpha moves. (Local cotBeta itself moves: cotBeta = s cot theta sqrt(1 + cotAlpha^2).)
    aa, bb = AI.knob(cols, ex, ey, ez, surf, 0.5, 1.0, la, lb)
    orient = data["tools"].angles.table.orientation(geo["detId"][mi])
    T2 = T & (orient != 0)
    cot_back = SA.key_from_local(surf[T2], aa[T2], bb[T2], orient[T2])
    cot_emit = cols["globalClusterCotTheta"][T2].astype(float)
    assert (np.abs(cot_back - cot_emit) <= tol(cot_emit)).mean() > 0.99
    assert (np.abs(aa[T] - la[T]) > tol(la[T])).mean() > 0.5


def test_k_sigma_modes():
    s = np.array([0.1, 0.2])
    assert np.array_equal(SA.sigma_k(s, 1.0, "scaled"), s) and np.array_equal(SA.sigma_k(s, 1.0, "fixed"), s)
    assert np.allclose(SA.sigma_k(s, 0.9999, "scaled"), s, rtol=1e-3)
    assert np.allclose(SA.sigma_k(s, 0.0, "scaled"), 0.05 * s)
    assert np.array_equal(SA.sigma_k(s, 0.3, "fixed"), s)


def test_centre_at_k0_known_bad(data):
    """The subtracted centre is k x c: at k = 0 the table centre vanishes (the known-bad, the k = 1 centre
    subtracted at k = 0, is what this test detects); at k = 1 it is applied where a row exists."""
    imp = data["tools"].angles
    ctx, geo = data["ctx"], data["geo"]
    K = ctx["K"]
    ci = np.flatnonzero((K["mi"] >= 0) & (K["hasAlpha"] > 0) & (K["hasBeta"] > 0))
    mi = K["mi"][ci]
    bar = geo["surface"][mi] <= 4
    ci, mi = ci[bar][:200], mi[bar][:200]
    args = (imp, geo["surface"][mi], geo["detId"][mi], K["localCotAlpha"][ci].astype(float),
            K["localCotBeta"][ci].astype(float), K["localCotAlpha"][ci].astype(float),
            K["localCotBeta"][ci].astype(float), K["sigAlpha"][ci].astype(float), K["sigBeta"][ci].astype(float))
    out0 = AI.calibrate(*args, 0.0, 0.0, "fixed")
    assert np.array_equal(out0[0], args[5]) and np.array_equal(out0[1], args[6])
    out1 = AI.calibrate(*args, 1.0, 1.0, "fixed")
    assert out1[4].mean() > 0.9 and (out1[1][out1[4]] != args[6][out1[4]]).mean() > 0.9


# ---------------------------------------------------------------- discriminating toys
def test_ranking_toy():
    """True cluster vs a wrong one 1 unit lower in position and 25 higher in beta (5 sigma): both and
    selection-only pick the true one; none and update-only (no angle in ranking) pick the wrong one."""
    r = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0]])
    Sm = np.zeros((2, 4, 4))
    sig = np.ones((2, 4))
    for metric in ("producer", "innovation"):
        with_b = AR.rank_values(metric, r, Sm, sig, np.array([[1, 1, 0, 1]] * 2, bool))
        no_ang = AR.rank_values(metric, r, Sm, sig, np.array([[1, 1, 0, 0]] * 2, bool))
        assert np.argmin(with_b) == 0 and np.argmin(no_ang) == 1


def test_window_cut_toy():
    z = np.array([[0.5, 0.5], [0.5, 5.0]])
    dims = np.ones((2, 2), bool)
    keep = AR.window_keep(z, dims, np.full((2, 2), 3.5))
    assert keep.tolist() == [True, False]
    assert AR.window_keep(z, np.zeros((2, 2), bool), np.full((2, 2), 3.5)).all()


def _k_state(gstat, acc, visited, surface, branch, o):
    st = AR.State.__new__(AR.State)
    st.o = np.asarray(o)
    st.branch = np.asarray(branch, np.int8)
    st.rec = {"acc": np.asarray(acc, bool), "gstat": np.asarray(gstat, float), "visited": np.asarray(visited, bool),
              "surface": np.asarray(surface, np.int64)}
    return st


def test_k_toys():
    """PD2: score = sum of the gate statistic over updates + the bin's 99 % cut per empty visited surface.
    Toy 3 (wrong hit ranked first, own second): the own branch wins. Toy 4 (own absent): skip beats a
    wrong hit beyond the cut. Toy 5 (correct hit, no wrong): the hit branch wins in >= 99 % of draws; the
    tail-probability rule (known-bad) loses ~40 %."""
    cut = 9.21                          # chi2_2 at 99 %
    arm = AR.Arm("k", K=2)
    tools = AR.Tools(None, calibration=None, angle_table=None, cuts=ConstCuts(cut))
    # toy 3: branches best (wrong, gstat 30), runner-up (own, 2), skip; one later surface each, own (2)
    st = _k_state([[30, 2], [2, 2], [np.nan, 2]], [[1, 1], [1, 1], [0, 1]], [[1, 1]] * 3, [[4, 3]] * 3, [0, 1, 2], [0, 0, 0])
    assert AR.resolve_k(arm, st, None, tools).tolist() == [1]
    # toy 4: best wrong (30) vs skip
    st = _k_state([[30, 2], [np.nan, 2]], [[1, 1], [0, 1]], [[1, 1]] * 2, [[4, 3]] * 2, [0, 2], [0, 0])
    assert AR.resolve_k(arm, st, None, tools).tolist() == [1]
    # toy 5: correct hit only, gstat ~ chi2_2, vs skip
    rng = np.random.default_rng(5)
    g = rng.chisquare(2, 20000)
    n = len(g)
    later = rng.chisquare(2, n)                         # the same continuation for both branches
    gst = np.concatenate([np.stack([g, later], 1), np.stack([np.full(n, np.nan), later], 1)])
    acc = np.concatenate([np.ones((n, 2)), np.stack([np.zeros(n), np.ones(n)], 1)])
    o = np.concatenate([np.arange(n), np.arange(n)])
    br = np.concatenate([np.zeros(n), np.full(n, 2)])
    st = _k_state(gst, acc, np.ones((2 * n, 2)), np.tile([4, 3], (2 * n, 1)), br, o)
    win = st.branch[AR.resolve_k(arm, st, None, tools)]
    assert (win == 0).mean() >= 0.985
    bad = AR.Arm("k", K=2, k_score="tailprob")
    win_bad = st.branch[AR.resolve_k(bad, st, None, tools)]
    assert (win_bad != 0).mean() > 0.2
