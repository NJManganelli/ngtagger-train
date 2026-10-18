"""WP-C entry gate E6: known-answer tests for every replay switch (PLAN §2, §4 E6, §10 PD2-PD3, PD4 (c)).

Each switch: an identity at its neutral (producer) setting, a non-null effect, its own-quantity test, and a
mis-wired variant that the test must catch. Real data: AAAA file 4 of common.CANON (the known-good file), the
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

pytestmark = pytest.mark.skipif(not CM.input_path("AAAA", 4).exists(), reason="file 4 not mounted")


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
    # a removed hit gives exactly the forced refit (from the OT seed) without it
    arm = AR.inn(cfg, name="rr0", rr=True, rr_max_iter=1)
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
    arm = AR.inn(cfg, name="rr-all", rr=True, rr_max_iter=8)
    out = AR.run_arm(ctx, geo, cfg, "AAAA", arm, AR.Tools(cfg, cuts=ConstCuts(0.0)))
    had = out["removed"].any(1) | out["rr_protected"]
    assert had.any()
    assert np.all(out["rec"]["acc"][had].sum(1) == 1) and np.all(out["rr_protected"][had])


def test_wide_window_search_bound(data, monkeypatch):
    """PD18: the 'gate99' window (the 99 % hit cut) is searched within WINDOW_SEARCH_NSIG of the prediction;
    a wider search changes no selection, and the window accepts no candidate beyond its bin's cut."""
    cut_file = CM.OUT / "cuts" / "wpc_cuts_v4.json"          # PD18's cut file (the gate99 window's)
    if not cut_file.exists():
        pytest.skip("no cut file")
    cfg, geo, ctx = data["cfg"], data["geo"], data["ctx"]
    tools = AR.Tools(cfg, cuts=CU.Cuts.load(str(cut_file)))
    base = AR.run_arm(ctx, geo, cfg, "AAAA", AR.inn(cfg, window_cut="gate99"), tools)
    monkeypatch.setattr(AR, "WINDOW_SEARCH_NSIG", 40.0)
    wide = AR.run_arm(ctx, geo, cfg, "AAAA", AR.inn(cfg, window_cut="gate99"), tools)
    assert np.array_equal(base["rec"]["ci"], wide["rec"]["ci"]) and np.array_equal(base["rec"]["acc"], wide["rec"]["acc"])
    g = base["rec"]["gstat"][base["rec"]["acc"]]
    S = base["rec"]["surface"][base["rec"]["acc"]]
    first = np.nonzero(base["rec"]["acc"])[1] == 0
    assert np.all(g <= tools.cuts.chi2("gate", "xy", "calibrated", S, first))
    n4 = AR.run_arm(ctx, geo, cfg, "AAAA", AR.inn(cfg), tools)                  # INN's 4-sigma ellipse (PD20)
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


# ---------------------------------------------------------------- C1b: the disc angle import
@pytest.mark.skipif(not Path("/Volumes/WDMac/smartpixels_scratch/p03_angles/angle_table_v1a1e_disclocal_canon_v3.json").exists(),
                    reason="no disc table")
def test_disc_import_exact_form():
    """C1b rev 4 §1 item 1: the disc import is exactly m - c sigma_local, w sigma_local per angle and facing, with
    no missing centre; the two facing known-bads differ from it; the knob is linear in cot."""
    import disc_angles as DA
    imp = DA.DiscImport()
    rng = np.random.default_rng(5)
    n = 4000
    surface = rng.choice([11, 12, 13, 14], n)
    cotA, cotB = rng.uniform(-0.5, 0.5, n), rng.uniform(-0.6, 0.6, n)
    facing = rng.choice([-1, 1], n).astype(np.int8)
    rows = imp.rows(surface, cotA, cotB)
    assert (rows >= 0).mean() > 0.95
    m, sig = rng.normal(0, 0.05, n), rng.uniform(0.01, 0.05, n)
    for angle in SA.ANGLES:
        c = imp.table.get(DA.KEY, angle, "centre", rows)
        w = imp.table.get(DA.KEY, angle, "width", rows)
        ok = rows >= 0
        assert np.all(np.isfinite(c[ok])) and np.all(np.isfinite(w[ok]))
        mm, ss = imp.calibrate(angle, m, sig, rows)
        assert np.array_equal(mm[ok], m[ok] - c[ok] * sig[ok]) and np.array_equal(ss[ok], w[ok] * sig[ok])
        for f in (-1, 1):                      # the same per facing: no orientation sign anywhere
            k = ok & (facing == f)
            assert np.array_equal(mm[k], m[k] - c[k] * sig[k])
        kb, _ = imp.calibrate(angle, m, sig, rows, variant="to_local", facing=facing)
        assert not np.allclose(kb[ok], mm[ok])
    kr = imp.rows_known_bad_key(surface, cotA, cotB, facing)
    assert np.array_equal(kr[facing == 1], rows[facing == 1]) and not np.array_equal(kr[facing == -1], rows[facing == -1])
    # the cot-linear knob
    tp, loc = rng.normal(0, 0.3, n), rng.normal(0, 0.3, n)
    T = rng.random(n) < 0.9
    assert np.array_equal(DA.knob(loc, tp, 1.0, T), loc)
    k0 = DA.knob(loc, tp, 0.0, T)
    assert np.array_equal(k0[T], tp[T]) and np.array_equal(k0[~T], loc[~T])
    assert np.allclose(DA.knob(loc, tp, 0.5, T)[T] - tp[T], 0.5 * (loc[T] - tp[T]))


def test_overlap_shape_fit():
    """C1b A6 / A8: on synthetic disc-like pairs (rotated module axes, both facings, a skewed size occupancy,
    p >= 10 GeV, module-local centres) with planted two-Gaussian single shapes, the joint unbinned fit recovers each
    populated bin's MAD / 0.6745 width and reports a sparse size bin as NaN; the width read-off is exact for a
    Gaussian; the beam pipe's Highland offset at IL1 is the expected 3-4 um at 1 GeV."""
    import disc_r as DR
    rng = np.random.default_rng(11)
    n = 40000
    tables = {"u": {"size_edges": [1, 2, 3, 4, 5, 99]}, "v": {"size_edges": [1, 2, 3, 5, 7, 9, 12, 99]}}
    Ru = np.array([6.0, 4.5, 4.0, 7.0, 90.0]) * 1e-4
    Rv = np.array([22.0, 13.0, 13.0, 14.0, 16.0, 18.0, 29.0]) * 1e-4
    mu_u = np.array([0.5, -0.3, 0, 0, 0]) * 1e-4
    phi = rng.uniform(0, 2 * np.pi, n)
    e_phi = np.stack([-np.sin(phi), np.cos(phi), 0 * phi], 1)
    e_r = np.stack([np.cos(phi), np.sin(phi), 0 * phi], 1)
    P = {"surface": np.full(n, 11), "ev": rng.integers(0, 200, n), "p": rng.uniform(10, 80, n), "eta": rng.uniform(1.3, 2.4, n),
         "e_phi": e_phi, "e_2": e_r, "g_phi": np.full(n, 0.5), "g_2": np.full(n, 0.5)}
    for j in ("1", "2"):
        psi = np.radians(rng.normal(0, 6, n))
        fac = rng.choice([-1.0, 1.0], n)
        u = fac[:, None] * (np.cos(psi)[:, None] * e_phi + np.sin(psi)[:, None] * e_r)
        v = -np.sin(psi)[:, None] * e_phi + np.cos(psi)[:, None] * e_r
        sx = rng.choice([1, 2, 3], n, p=[0.807, 0.19, 0.003]).astype(float)      # size bin 2 is sparse
        sy = rng.choice([1, 2], n, p=[0.75, 0.25]).astype(float)
        bu, bv = DR.size_bin(sx, tables["u"]["size_edges"]), DR.size_bin(sy, tables["v"]["size_edges"])
        P.update({f"ux{j}": u, f"vx{j}": v, f"sx{j}": sx, f"sy{j}": sy, f"cotA{j}": np.zeros(n), f"cotB{j}": np.zeros(n),
                  f"facing{j}": fac.astype(np.int8),
                  f"du{j}": Ru[bu] * DR.draw_unit("mix2", rng, n) + mu_u[bu],
                  f"dv{j}": Rv[bv] * DR.draw_unit("mix2", rng, n)})
    f = DR.ShapeFit(P, tables, np.ones(n, bool)).fit()
    assert f["converged"]
    # within ~3 sd of the per-sample spread (8 seeds: 1.1 / 3.2 % in u, 0.7 / 3.2 % in v; mean within its error)
    assert np.allclose(f["R_u"][:2], Ru[:2], rtol=[0.04, 0.10]) and np.allclose(f["R_v"][:2], Rv[:2], rtol=[0.03, 0.10])
    assert np.isnan(f["R_u"][2]) and f["n_u"][2] < DR.MIN_PARAM
    assert abs(f["shape_u"]["mu"][0] - mu_u[0]) < 0.3e-4
    from scipy import stats
    g = DR.single_width("mix2", np.array([[np.log(5.0), -30.0, 0.0, 0.0]]))           # eps -> 0: a Gaussian of 5 um
    assert abs(g[0] * DR.MAD_K - 5.0 * stats.norm.ppf(0.75)) < 1e-6                    # its MAD, exactly
    bu, _ = DR.beam_pipe_sigma(1.0, 0.0, 2.9, 0.0, 0.0)
    assert 3e-4 < bu < 4e-4


def test_empty_layer_material_is_charged():
    """The calibrated arm charges a layer's module material when the track steps past it, per crossing and not per
    hit (arms._q_step -> ot_refit_projection.add_material: every plane in (R_next, R_last visited]): a visited layer with
    no cluster and a layer the visit sequence skips both scatter the prediction of the next one."""
    import types
    import ot_refit_projection as ORP_
    geo = {"R": {1: 3.04, 2: 6.17, 3: 10.47, 4: 14.67}}
    planes = [["IL2", 0.0187, "IL2 module"], [7.29, 0.001, "HV L2"], ["IL3", 0.0148, "IL3 module"],
              [11.68, 0.0008, "HV L3"], ["IL4", 0.0148, "IL4 module"], [15.84, 0.0009, "HV L4"]]
    cal = {"planes": planes, "material_scale": 1.0, "ot_inner_r_cm": 22.0}
    no_il4 = {**cal, "planes": [p if p[0] != "IL4" else ["IL4", 0.0, "IL4 removed"] for p in planes]}
    arm = AR.inn({})
    kpt = 0.0057
    a = np.array([[kpt / 3.0, 0.1, 0.4, 1.0, 0.0]])                  # pT 3 GeV, tanL 0.4

    def predict(c, visits):
        st = AR.State(a.copy(), np.zeros((1, 5, 5)), np.array([0]), np.zeros((1, 4), np.int64))
        for S in visits:                                                # no update anywhere: the clusters do not matter
            AR._q_step(arm, st, np.array([0]), S, None, geo, types.SimpleNamespace(cal=c, disc=None), kpt, {})
        return st.C[0]

    def kink(r_s, xx0):                                                 # the Highland kink of one plane, as charged
        tanl = a[0, 2]
        p = kpt / a[0, 0] * np.sqrt(1 + tanl ** 2)
        th2 = ORP_._highland_theta2(np.array([xx0]), np.array([p]), np.array([tanl]))[0]
        sec, sec2 = np.sqrt(1 + tanl ** 2), 1 + tanl ** 2
        jT = np.array([0, sec, 0, 0, r_s * sec]); jL = np.array([0, 0, sec2, -r_s * sec2, 0])
        return th2 * (np.outer(jT, jT) + np.outer(jL, jL))

    il4 = kink(geo["R"][4], 0.0148)
    # IL4 visited without a cluster, then IL3: IL4's module is charged when predicting IL3
    assert np.allclose(predict(cal, [4, 3]) - predict(no_il4, [4, 3]), il4)
    # IL4 skipped by the visit sequence (OT -> IL3 directly): charged all the same
    assert np.allclose(predict(cal, [3]) - predict(no_il4, [3]), il4)
    # and nothing of IL4's is charged before it is crossed
    assert np.allclose(predict(cal, [4]), predict(no_il4, [4]))


def test_tree2_identity(data, monkeypatch):
    """Scan plan A3: with the skip priced at infinity (the constant cut) and no runner-up, tree2 commits the best hit
    at every surface: bit-identical to K = 1."""
    cfg = data["cfg"]
    k1 = run(data, AR.inn(cfg))
    ranked = AR._ranked
    monkeypatch.setattr(AR, "_ranked", lambda cands, nx: (ranked(cands, nx)[0], np.full(nx, -1, np.int64)))
    t2 = run(data, AR.inn(cfg, name="tree2", K="tree2"))
    for k in ("a", "C", "nacc", "refit"):
        assert np.array_equal(k1[k], t2[k], equal_nan=True), k
    for k in ("ci", "mi", "acc", "chi2"):
        assert np.array_equal(k1["rec"][k], t2["rec"][k]), k
    assert set(np.unique(t2["rec"]["kchoice"])) <= {-1, 0}


def test_angle_steps(data):
    """Scan plan B1: angles at the first visited surface only share the all-angle arm's first-step choices and the
    no-angle arm's choices after a common first step; angles at later visits only share the no-angle arm's first-step
    choices; "all" is the arm itself."""
    cfg = data["cfg"]
    both, none = run(data, AR.inn(cfg)), run(data, AR.inn(cfg, name="none", use=frozenset()))
    first = run(data, AR.inn(cfg, name="first", angle_steps="first"))
    later = run(data, AR.inn(cfg, name="later", angle_steps="later"))
    assert np.array_equal(_first_choice(first), _first_choice(both))
    assert np.array_equal(_first_choice(later), _first_choice(none))
    assert not np.array_equal(first["rec"]["ci"], both["rec"]["ci"]) or not np.array_equal(first["a"], both["a"])
    same = run(data, AR.inn(cfg, name="all", angle_steps="all"))
    assert np.array_equal(same["a"], both["a"]) and np.array_equal(same["rec"]["ci"], both["rec"]["ci"])


def test_tail_robust_term(data):
    """Scan plan B1 "t9": sigma x sqrt(1 + z^2 / 9): the core is untouched to first order, the chi2 of one angle
    saturates at 9; positions are untouched; the arm runs and moves angle-bearing results."""
    z = np.array([0.0, 1.0, 3.0, 30.0, np.nan])
    s = AR.tail_robust_scale(z)
    assert np.allclose(s[:4], np.sqrt(1 + z[:4] ** 2 / 9)) and s[4] == 1.0
    chi = (z[:4] / s[:4]) ** 2
    assert chi[0] == 0 and abs(chi[2] - 4.5) < 1e-12 and chi[3] < 9 and np.all(np.diff(chi) > 0)
    cfg = data["cfg"]
    g, t = run(data, AR.inn(cfg)), run(data, AR.inn(cfg, name="t9", angle_term="t9"))
    assert np.isfinite(t["a"][t["refit"]]).all() and not np.array_equal(g["a"], t["a"])


def _toy6(d, r, n, seed, score):
    """Scan plan A3 toy 6: the committed first-layer (correct, wrong, skip) rates of a score rule and K = 1's wrong rate
    on two coupled surfaces (a wrong first choice shifts the second layer's prediction by 0.5 r u1)."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    unit = lambda: (lambda u: u / np.linalg.norm(u, axis=1, keepdims=True))(rng.standard_normal((n, d)))
    u1, u2 = unit(), unit()
    c1, w1 = rng.standard_normal((n, d)), rng.standard_normal((n, d)) + r * u1
    c2, w2 = rng.standard_normal((n, d)), rng.standard_normal((n, d)) + r * u2
    q = lambda v: (v ** 2).sum(1)
    P = np.full(n, stats.chi2.ppf(0.99, d))
    wrong_best = q(w1) < q(c1)
    x1 = np.stack([np.minimum(q(c1), q(w1)), np.maximum(q(c1), q(w1))], 1)
    shift = 0.5 * r * u1
    after = {"correct": np.stack([q(c2), q(w2)], 1), "wrong": np.stack([q(c2 - shift), q(w2 - shift)], 1)}
    lay2 = lambda s: np.sort(after[s], 1)
    first_is_wrong = np.stack([wrong_best, ~wrong_best], 1)          # (best, runner-up) is the wrong hit
    x2 = np.stack([np.where(first_is_wrong[:, [c]], lay2("wrong"), lay2("correct")) for c in (0, 1)]
                  + [lay2("correct")], 1)
    ch = np.argmin(score(x1, x2, P), 1)
    wrong = np.where(ch == 2, False, first_is_wrong[np.arange(n), np.minimum(ch, 1)])
    return {"wrong": wrong.mean(), "skip": (ch == 2).mean(), "k1_wrong": wrong_best.mean(), "n": n}


def _wilson_upper(p, n, z=1.645):
    return (p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def _toy6_passes(res):
    ok = []
    for (d, r), x in res.items():
        wu, su = _wilson_upper(x["wrong"], x["n"]), _wilson_upper(x["skip"], x["n"])
        ok.append((wu <= 0.5 * x["k1_wrong"] if r < 6 else wu <= x["k1_wrong"] + 5e-4) and su <= 0.015)
    return all(ok)


@pytest.mark.xfail(strict=True, reason="scan plan A3 criterion 1 fails as pre-declared (2026-10-18): tree2 cuts K = 1's "
                   "wrong commits by 30-39 % at r = 4 (needed 50 %); skip <= 1.04 %; known-bads fail. Decision pending")
def test_toy6_tree2():
    """Scan plan A3 (amended before it ran): tree2's scores halve K = 1's committed wrong rate at r = 4, 5 (no worse at
    r = 6) with a skip rate <= 1.5 %; PD2's asymmetric penalty and the tail-probability rule fail."""
    n = 200_000
    grid = [(d, r) for d in (2, 4) for r in (4, 5, 6)]
    run6 = lambda score: {(d, r): _toy6(d, r, n, 100 * d + r, score) for d, r in grid}
    good = run6(AR.tree2_scores)
    assert _toy6_passes(good), good
    assert not _toy6_passes(run6(lambda x1, x2, P: AR.tree2_scores(x1, x2, P, p_first=0.5 * P)))
    assert not _toy6_passes(run6(lambda x1, x2, P: AR.tree2_scores(x1, x2, P, "tailprob")))


def test_inner_material_disc_going():
    """Scan plan B-pre: the inner material of a disc-going track whose innermost update is a disc (ID2) charges only the
    planes it crosses between the POCA and ID2 (beam pipe, IL1 module, ID1, ID2's own plane), never the barrel planes
    by radius; a barrel-only track is bit-identical to the barrel rule (ORP.add_material with the IL1 module and the
    beam pipe)."""
    import types
    kpt = 0.0057
    geo = {"R": {1: 3.04, 2: 6.17, 3: 10.47, 4: 14.67}, "Z": {11: 25.18, 12: 32.18}}
    cal_planes = [["IL2", 0.0187, "IL2 module"], [7.29, 0.001, "HV L2"], ["IL3", 0.0148, "IL3 module"]]
    cal = {"planes": cal_planes, "material_scale": 1.0, "ot_inner_r_cm": 22.0}
    cyl = lambda r, hz, x: {"kind": "cyl", "r": r, "half_z": hz, "xx0": x}
    disc = lambda z, x: {"kind": "disc", "z": z, "r_ext": [3.1, 16.0], "xx0": x}
    crossed = [cyl(3.04, 19.94, 0.0185), disc(25.18, 0.02), disc(32.18, 0.02)]
    uncrossed = [cyl(6.17, 20.07, 0.0187), cyl(7.29, 20.03, 0.001), cyl(10.47, 20.07, 0.0148)]
    a = np.array([[kpt / 3.0, 0.1, 0.5, 0.0, 0.0],                     # barrel-only, innermost update IL2
                  [kpt / 3.0, 0.1, np.sinh(2.2), 0.0, 0.0]])          # disc-going (|eta| 2.2), innermost update ID2
    arm = AR.inn_disc({})

    def charged(planes, disc_scale=1.0):
        st = AR.State(a.copy(), np.zeros((2, 5, 5)), np.array([0, 1]), np.zeros((2, 3), np.int64), dg=[False, True])
        st.rec["acc"][:, 0] = True
        st.rec["surface"][:, 0] = [2, 12]
        tools = types.SimpleNamespace(cal=cal, disc=types.SimpleNamespace(planes=planes, disc_scale=disc_scale))
        AR.inner_material(arm, st, geo, tools, kpt)
        return st.C

    full = charged(crossed + uncrossed)
    # the uncrossed barrel modules and supports are not charged; the crossed planes are, the disc scale on the discs
    assert np.array_equal(full[1], charged(crossed)[1])
    assert not np.allclose(full[1], charged(crossed[:1] + crossed[2:] + uncrossed)[1])
    assert (charged(crossed, 2.0)[1, 3, 3] > full[1, 3, 3]) and full[1, 4, 4] > 0
    # the barrel-only track: exactly the barrel rule
    ref = np.zeros((2, 5, 5))
    geo0 = {**geo, "R": {**geo["R"], 0: 0.0}}
    ORP.add_material(a.copy(), ref, np.array([0]), 0, np.array([2, 5]), geo0,
                     {**cal, "planes": cal_planes + [AR.IL1_MODULE, AR.BEAM_PIPE]}, kpt)
    assert np.array_equal(full[0], ref[0])


def test_clip_scale():
    """C1b A18: the clipped-likelihood scale is 1 for a unit Gaussian, the scale for a scaled one, and matches a direct
    1-dim minimisation of sum [2 log s + min(z^2 / s^2, 25)] on a heavy-tailed sample."""
    from scipy import optimize
    import gates as G
    import seedmodel as SM
    rng = np.random.default_rng(3)
    reps = G.replicas(nev=400, n=10)
    for x in (rng.standard_normal(100000), 1.3 * rng.standard_normal(100000), rng.standard_t(3, 100000)):
        s, sd = SM.clip_scale(x, rng.integers(0, 400, len(x)), reps)
        f = lambda ls: np.sum(2 * ls + np.minimum(x * x * np.exp(-2 * ls), 25.0))
        assert abs(s - np.exp(optimize.minimize_scalar(f, bounds=(-1, 1.5), method="bounded").x)) < 5e-4
        assert 0 < sd < 0.02
