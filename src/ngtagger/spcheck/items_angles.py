"""Fast-gate items on the census angle model (WP-D0b, census format 33; plan C.4 rows "an error
table from the wrong era or key", "k knob" and "dead angle flags"), on the first events of the gate
input (canon_v3 file 3): the census's corrected angles are the locked table's, the knob behaves as
the sheet says through the census path, and each per-angle switch removes exactly its angle. They
import the census from THIS code root's eval_refitq/combinatorics."""
from __future__ import annotations

import json
import os

import numpy as np

from .core import Item, NotApplicable, identity, same_input, t_family

CENSUS_DIR = "eval_refitq/combinatorics"
SIGZ = {L: 0.2 for L in range(1, 7)}          # OT stub sigma z: irrelevant to the angle checks
LOCKED_REGISTRY = "p03.table.v1a1e"
ATOL = 1e-12


def _chunk(ctx, model):
    """(U, Q) of the gate input's first ctx.events events, prepared with `model`."""
    TF = ctx.script_module(CENSUS_DIR, "tp_findability")
    key = ("angle_chunk", ctx.input, ctx.events, id(model))
    if key not in ctx.cache:
        (U, Q), _ = next(iter(TF.unified_chunks(ctx.input, ctx.events, ctx.events, SIGZ, None, model)))
        ctx.cache[key] = (U, Q)
    return ctx.cache[key]


def _boot_sd(x, ev, fn, n_rep=200, seed=7):
    """Event-bootstrap sd of fn(x) (x a 1-d sample, ev its events)."""
    evs, inv = np.unique(ev, return_inverse=True)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_rep):
        w = rng.multinomial(len(evs), np.full(len(evs), 1.0 / len(evs)))
        rep = np.repeat(np.arange(len(x)), w[inv])
        vals.append(fn(x[rep]))
    return float(np.std(vals, ddof=1))


# ---------------------------------------------------------------- the table import
class AngleImport(Item):
    id = "census.angle_import"
    title = ("the census's angles are the locked table's: per cluster the corrected angle and its "
             "uncertainty equal spdefs' (1e-12), the table is the locked one, class-T after-table "
             "pulls in the D.3 band; a cut file made for another configuration is refused")
    c4_row = "an error table from the wrong era or key (census side; WP-D0b)"
    known_bad_desc = ("four variants, each of which must be red: (scales) the legacy per-layer pull "
                      "scales and no centre in place of the table; (key) the cluster correction read "
                      "in the true key; (sign) the centre added instead of subtracted; (cut file) a "
                      "cut file whose S6 entries were made with S5 on, applied to an S5-off run")
    rule_desc = ("on the first events of the gate input, every IT barrel cluster's census-path "
                 "corrected alpha / beta and uncertainty (census_angles.prepare, k = 1) equal "
                 "spdefs' (the emitted-key row's centre subtracted, the width applied) to 1e-12; the "
                 "table's sha256 equals the reference's (the locked p03.table.v1a1e); per surface and "
                 "angle the class-T (usable, pT >= 1 GeV) after-table centre (25 %-trimmed mean) "
                 "within 0 +- 0.25 and width (95 %-trimmed) within 1 +- 0.1, each widened by "
                 "t(N = 8) event-bootstrap sd; a cut file is accepted for its own configuration "
                 "(CutSet.check) and refused for an S5-off run when made with S5 on; identity of the "
                 "cluster counts with the reference")
    trees = (CENSUS_DIR,)

    def _stat(self, ctx, variant=None):
        from ngtagger.spdefs import angles as SD

        CA = ctx.script_module(CENSUS_DIR, "census_angles")
        M = ctx.script_module(CENSUS_DIR, "tracklet_topology_cost")
        model = CA.Model()
        U, Q = _chunk(ctx, model)
        it = np.flatnonzero(U["layer"] <= 4)
        C = {k: Q[k][it] for k in CA.COMP_KEYS}
        lay = C["layer"]
        # the reference, from spdefs alone: the emitted-key row at the emitted cot theta
        rows = model.table.rows_of("emitted", lay, C["cot_e"], np.zeros(len(it), np.int8))
        ref = {}
        for a, ang, sig in (("alpha", "half", "sa_pub"), ("beta", "lam", "sb_pub")):
            c = model.table.get("emitted", a, "centre", rows)
            w = model.table.get("emitted", a, "width", rows)
            good = np.isfinite(c) & np.isfinite(w)
            c, w = np.where(good, c, 0.0), np.where(good, w, 1.0)
            ref[a] = (SD.WRAP[a](C[f"{ang}_e"] - c * C[sig]), w * C[sig])
        got = {"alpha": (Q["half_c"][it], Q["s_half_c"][it]), "beta": (Q["lam_c"][it], Q["s_lam_c"][it])}
        if variant == "scales":
            sa = np.array([M.ALPHA_PULL_SCALE.get(L, 1.0) for L in range(5)])[lay]
            sb = np.array([M.BETA_PULL_SCALE.get(L, 1.0) for L in range(5)])[lay]
            got = {"alpha": (C["half_e"], C["sa_pub"] * sa), "beta": (C["lam_e"], C["sb_pub"] * sb)}
        elif variant == "key":
            rt = model.rows("true", lay, C["cot_e"])
            got = {}
            for a, ang, sig in (("alpha", "half", "sa_pub"), ("beta", "lam", "sb_pub")):
                c, w = model.cw("true", a, rt)
                got[a] = (SD.WRAP[a](C[f"{ang}_e"] - c * C[sig]), w * C[sig])
        elif variant == "sign":
            got = {}
            for a, ang, sig, ck in (("alpha", "half", "sa_pub", "ca"), ("beta", "lam", "sb_pub", "cb")):
                got[a] = (SD.WRAP[a](C[f"{ang}_e"] + C[ck] * C[sig]), Q["s_half_c" if a == "alpha" else "s_lam_c"][it])
        use = Q["dir_ok"][it]
        diff = {}
        for a in ("alpha", "beta"):
            da = np.abs(SD.residual(a, got[a][0], ref[a][0]))[use]
            ds = np.abs(got[a][1] - ref[a][1])[use] / np.maximum(ref[a][1][use], 1e-300)
            diff[a] = float(max(da.max(initial=0.0), ds.max(initial=0.0)))
        # class-T after-table pulls per surface and angle, P0.3's population
        pt = U["tpPt"][it]
        pop = use & C["is_T"] & (pt >= 1.0)
        band = {}
        t = t_family(8)
        for a, ang in (("alpha", "half"), ("beta", "lam")):
            with np.errstate(all="ignore"):
                p = SD.residual(a, got[a][0], C[f"{ang}_t"]) / got[a][1]
            for L in (1, 2, 3, 4):
                m = pop & (lay == L)
                x, ev = p[m], U["event"][it][m]
                cen = float(SD.centre_of(x))
                wid = float(SD.weighting_width(np.abs(x - cen)))
                sdc = _boot_sd(x, ev, SD.centre_of)
                sdw = _boot_sd(x, ev, lambda y: SD.weighting_width(np.abs(y - SD.centre_of(y))))
                band[f"{a}/IL{L}"] = {"n": int(m.sum()), "centre": cen, "width": wid,
                                      "ok": bool(abs(cen) <= 0.25 + t * sdc and abs(wid - 1.0) <= 0.1 + t * sdw)}
        # the cut file's refusal: a synthetic file whose S6 entries were made with S5 on
        doc = {"schema": CA.CUT_SCHEMA,
               "identity": {"table_sha256": model.table_sha, "k_alpha": 1.0, "k_beta": 1.0,
                            "mode": "fixed", "bench": None, "d0_prior_cm": 1.0, "level": CA.LEVEL,
                            "floors": [2.0]},
               "s1": {"bins": [], "thr": []}, "s2a": {"bins": [], "thr": []},
               "s2z": {"bins": [], "thr": []}, "win": {"bins": [[0, 10203, 3, 1, 2.0]], "thr": {"ab": [[3.0, 4.0]]}},
               "s6": {"bins": [["AAAA", "ab", "emitted", "ab", 4]], "thr": [20.0]}}
        TF = ctx.script_module(CENSUS_DIR, "tp_findability")
        # S5 stated explicitly: the synthetic file's S6 entry was made with S5 on
        ang = M.AngleUse.parse("all", off="kf" if variant == "cutfile" else "")
        run = TF.cut_identity(ang, None, None, 2.0, None, ["AAAA", "AAAI", "IAAA"])   # the census's own identity
        try:
            CA.CutSet(doc).check(model, run)
            refused = None
        except SystemExit as e:
            refused = str(e)[:200]
        return {"input": ctx.input, "events": ctx.events, "variant": variant,
                "n_clusters": int(len(it)), "n_usable": int(use.sum()),
                "table_sha256": model.table_sha, "table_registry": model.registry,
                "max_diff": diff, "band": band, "cutfile_refused": refused}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return {"input": ctx.input, "events": ctx.events,
                "variants": {v: self._stat(ctx, v) for v in ("scales", "key", "sign", "cutfile")}}

    def judge(self, stat, ref):
        if "variants" in stat:
            res = {k: self.judge(v, ref) for k, v in stat["variants"].items()}
            return any(r[0] for r in res.values()), "; ".join(
                f"{k}: {'GREEN' if r[0] else 'red'} ({r[1][:160]})" for k, r in res.items())
        why = [f"{a} differs from spdefs by {d:.3g}" for a, d in stat["max_diff"].items() if not d <= ATOL]
        why += [f"{k} out of band (centre {v['centre']:.3f}, width {v['width']:.3f})"
                for k, v in stat["band"].items() if not v["ok"]]
        if stat["cutfile_refused"]:
            why.append(f"cut file refused: {stat['cutfile_refused']}")
        if stat["table_registry"] != LOCKED_REGISTRY:
            why.append(f"table registry {stat['table_registry']}")
        if ref is not None and stat["table_sha256"] != ref["stat"]["table_sha256"]:
            why.append("not the locked table (sha256)")
        if same_input(stat, ref) and stat.get("variant") == ref["stat"].get("variant"):
            why += identity(stat, ref, ["n_clusters", "n_usable"])[1]
        return not why, (f"{stat['n_usable']} usable clusters: corrected angles = spdefs, locked "
                         f"table, class-T pulls in band") if not why else "; ".join(why)


# ---------------------------------------------------------------- the knob
class KKnob(Item):
    id = "census.k_knob"
    title = ("the improvement knob through the census path: k = 1 is no knob, scaled is continuous "
             "at 1 and floored at 0, class C never moves, the centre scales with k, each angle by "
             "its own k")
    c4_row = "k knob (census side; WP-D0b)"
    known_bad_desc = "the k = 1 centre subtracted at k = 0 (the class-T after-table centre is then not 0)"
    rule_desc = ("census_angles.prepare on the first events of the gate input: (i) k = 1 in fixed and "
                 "scaled mode bit-identical, and equal to the knob-free correction; (ii) scaled at "
                 "k = 0.999 within 1.1e-3 of k = 1 in every usable cluster's sigma; (iii) scaled at "
                 "k = 0 every usable sigma finite and 0.05 x the k = 1 one; (iv) class C (not "
                 "tpEmuClass 1) never moved: its corrected angle plus k c sigma is the emitted angle "
                 "(1e-12) at k = 0 and 0.5; (v) at k = 0, in both modes, every class-T cluster the knob "
                 "moves has after-table residual 0 (centre k c); (vi) k_alpha = 0, k_beta = 1 moves "
                 "alpha to truth and leaves beta at k = 1")
    trees = (CENSUS_DIR,)

    def _stat(self, ctx, bad_centre=False):
        from ngtagger.spdefs import angles as SD

        CA = ctx.script_module(CENSUS_DIR, "census_angles")
        out = {"input": ctx.input, "events": ctx.events, "bad_centre": bad_centre}
        U, Q1 = _chunk(ctx, CA.Model())
        it = np.flatnonzero(U["layer"] <= 4)
        C = {k: Q1[k][it] for k in CA.COMP_KEYS}
        isT, isC = C["is_T"], ~C["is_T"]
        use = Q1["dir_ok"][it] & np.isfinite(C["sa_pub"]) & (C["sa_pub"] > 0) \
            & np.isfinite(C["sb_pub"]) & (C["sb_pub"] > 0)
        # the class-T clusters the knob moves (an alpha more than 90 deg from truth keeps its value)
        mvA = SD.improve("alpha", C["half_e"], C["half_t"], isT, 0.0)[1] & use
        mvB = SD.improve("beta", C["lam_e"], C["lam_t"], isT, 0.0)[1] & use

        def corr(ka, kb, mode):
            with np.errstate(all="ignore"):
                return CA.corrected_of(C, ka, kb, mode)
        f1, s1 = corr(1.0, 1.0, "fixed"), corr(1.0, 1.0, "scaled")
        knobfree = (SD.WRAP["alpha"](C["half_e"] - C["ca"] * C["sa_pub"]), C["wa"] * C["sa_pub"],
                    SD.WRAP["beta"](C["lam_e"] - C["cb"] * C["sb_pub"]), C["wb"] * C["sb_pub"])
        out["k1_identical"] = bool(all(np.array_equal(a, b) for a, b in zip(f1, s1))
                                   and all(np.array_equal(a, b) for a, b in zip(f1, knobfree))
                                   and np.array_equal(Q1["half_c"][it], f1[0]))
        near = corr(0.999, 0.999, "scaled")
        out["scaled_continuity"] = float(max(np.max(np.abs(near[1][use] / f1[1][use] - 1.0)),
                                             np.max(np.abs(near[3][use] / f1[3][use] - 1.0))))
        z = corr(0.0, 0.0, "scaled")
        out["scaled_floor"] = bool(np.all(np.isfinite(z[1][use])) and np.all(np.isfinite(z[3][use]))
                                   and np.allclose(z[1][use], 0.05 * f1[1][use], rtol=1e-12)
                                   and np.allclose(z[3][use], 0.05 * f1[3][use], rtol=1e-12))
        cdev = 0.0
        for k in (0.0, 0.5):
            h, _, lam, _ = corr(k, k, "fixed")
            cdev = max(cdev, float(np.max(np.abs(SD.residual("alpha", h + k * C["ca"] * C["sa_pub"], C["half_e"])[isC]), initial=0)),
                       float(np.max(np.abs(SD.residual("beta", lam + k * C["cb"] * C["sb_pub"], C["lam_e"])[isC]), initial=0)))
        out["class_c_moved"] = cdev
        r0 = 0.0
        for mode in ("fixed", "scaled"):
            h, _, lam, _ = corr(0.0, 0.0, mode)
            if bad_centre:                    # the k = 1 centre subtracted at k = 0
                h = SD.WRAP["alpha"](h - C["ca"] * C["sa_pub"])
                lam = SD.WRAP["beta"](lam - C["cb"] * C["sb_pub"])
            r0 = max(r0, float(np.max(np.abs(SD.residual("alpha", h, C["half_t"])[mvA]), initial=0)),
                     float(np.max(np.abs(SD.residual("beta", lam, C["lam_t"])[mvB]), initial=0)))
        out["k0_class_t_residual"] = r0
        h, _, lam, _ = corr(0.0, 1.0, "fixed")
        out["own_k"] = bool(np.allclose(h[mvA], C["half_t"][mvA], atol=ATOL)
                            and np.array_equal(lam, f1[2], equal_nan=True))
        out["n_class_t"], out["n_class_c"] = int(isT.sum()), int(isC.sum())
        out["n_moved"] = {"alpha": int(mvA.sum()), "beta": int(mvB.sum())}
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, bad_centre=True)

    def judge(self, stat, ref):
        why = []
        if not stat["k1_identical"]:
            why.append("k = 1 not bit-identical to the knob-free correction")
        if not stat["scaled_continuity"] <= 1.1e-3:
            why.append(f"scaled k -> 1 jump {stat['scaled_continuity']:.3g}")
        if not stat["scaled_floor"]:
            why.append("scaled k = 0 not floored at 0.05")
        if not stat["class_c_moved"] <= ATOL:
            why.append(f"class C moved by {stat['class_c_moved']:.3g}")
        if not stat["k0_class_t_residual"] <= 1e-9:
            why.append(f"class-T residual at k = 0: {stat['k0_class_t_residual']:.3g} (centre not k c)")
        if not stat["own_k"]:
            why.append("k_alpha != k_beta does not move each angle by its own k")
        if same_input(stat, ref) and stat["bad_centre"] == ref["stat"]["bad_centre"]:
            why += identity(stat, ref, ["n_class_t", "n_class_c"])[1]
        return not why, (f"{stat['n_class_t']} class-T, {stat['n_class_c']} class-C clusters: knob "
                         f"as the sheet says") if not why else "; ".join(why)


# ---------------------------------------------------------------- per-angle switches
class AngleSwitches(Item):
    id = "census.angle_switches"
    title = ("each per-angle switch removes exactly its angle: an empty set is the stage off, one "
             "angle is the stage with the other's term removed, and with every stage off the "
             "census reads no angle")
    c4_row = "dead angle flags; beta in the no-angle arm (per-angle switches; WP-D0b)"
    known_bad_desc = "S4a told to use beta only, scored with the alpha term still added"
    rule_desc = ("one IL1+IL2 doublet seed followed into IL3, IL4, OL1, OL2 on the first events of "
                 "the gate input, weighting stages only (no cut file needed): S4a with no angle == "
                 "S4a off; S4a with beta only == S4a with both angles and alpha's sigma infinite (the "
                 "same hits); S5 with no angle == S5 off (bit-identical fit); S5 with alpha only "
                 "differs from S5 with both; every stage off: hits and fit bit-identical under two "
                 "different knobs (k = 1 fixed; k = 0.3 scaled), so nothing reads an angle; identity "
                 "of the track count with the reference")
    trees = (CENSUS_DIR,)

    def _follow(self, ctx, model, ang, Qmod=None):
        SA = ctx.script_module(CENSUS_DIR, "seed_arity")
        U, Q = _chunk(ctx, model)
        if Qmod is not None:
            Q = Qmod(dict(Q))
        ang.model = model
        seed = SA.Seed((1, 2), follow=(3, 4, 11, 12))
        F = SA.follow_seed(U, Q, seed, 2.0, (3, 4, 11, 12), ang, 0.0, "rphiz")
        hits = {L: (np.asarray(v[0]), np.asarray(v[1])) for L, v in sorted(F["hits"].items())}
        return U, Q, F, hits

    @staticmethod
    def _same(h1, h2):
        return h1.keys() == h2.keys() and all(np.array_equal(h1[L][0], h2[L][0]) and
                                              np.array_equal(h1[L][1], h2[L][1]) for L in h1)

    def _stat(self, ctx, bad=False):
        M = ctx.script_module(CENSUS_DIR, "tracklet_topology_cost")
        KF = ctx.script_module(CENSUS_DIR, "kf_emulation")
        CA = ctx.script_module(CENSUS_DIR, "census_angles")
        model = CA.Model()
        keys = {"proj_arb": "emitted"}
        _, _, _, h_off = self._follow(ctx, model, M.AngleUse(()))
        _, _, _, h_empty = self._follow(ctx, model, M.AngleUse({"proj_arb"}, angles={"proj_arb": []}))
        b_set = ["alpha", "beta"] if bad else ["beta"]
        _, _, _, h_b = self._follow(ctx, model, M.AngleUse({"proj_arb"}, angles={"proj_arb": b_set}, keys=keys))

        def no_alpha(Q):
            Q["s_kap"] = np.where(Q["layer"] <= 4, np.inf, Q["s_kap"])
            return Q
        _, _, _, h_b_ref = self._follow(ctx, model, M.AngleUse({"proj_arb"}, keys=keys), no_alpha)
        U, Q, F, _ = self._follow(ctx, model, M.AngleUse(()))
        G = KF.hits_from_seed(U, {"_gA": F["gA"], "_gB": F["gB"], "_gC": F["gC"],
                                  "_hits": F["hits"]})
        trip = (F["gA"], F["gB"], F["gB"])
        fit_off = KF.fit_tracks(U, Q, trip, gidx=G, use_angles=False)
        fit_none = KF.fit_tracks(U, Q, trip, gidx=G, use_angles=True, angles=())
        fit_a = KF.fit_tracks(U, Q, trip, gidx=G, use_angles=True, angles=("alpha",))
        fit_ab = KF.fit_tracks(U, Q, trip, gidx=G, use_angles=True, angles=("alpha", "beta"))
        same_fit = all(np.array_equal(fit_off[k], fit_none[k], equal_nan=True)
                       for k in ("kappa", "d0", "cot", "z0", "chi2_rphi", "chi2_rz"))
        a_differs = not np.allclose(fit_a["d0"], fit_ab["d0"], equal_nan=True) or \
            not np.allclose(fit_a["cot"], fit_ab["cot"], equal_nan=True)
        m2 = CA.Model(k_alpha=0.3, k_beta=0.3, mode="scaled")
        U2, Q2, F2, h_off2 = self._follow(ctx, m2, M.AngleUse(()))
        G2 = KF.hits_from_seed(U2, {"_gA": F2["gA"], "_gB": F2["gB"], "_gC": F2["gC"],
                                    "_hits": F2["hits"]})
        fit_off2 = KF.fit_tracks(U2, Q2, (F2["gA"], F2["gB"], F2["gB"]), gidx=G2, use_angles=False)
        off_blind = self._same(h_off, h_off2) and np.array_equal(G, G2) and all(
            np.array_equal(fit_off[k], fit_off2[k], equal_nan=True) for k in ("kappa", "d0", "cot", "z0"))
        return {"input": ctx.input, "events": ctx.events, "bad": bad,
                "n_tracks": int(len(G)),
                "s4a_empty_is_off": self._same(h_empty, h_off),
                "s4a_one_angle_is_other_removed": self._same(h_b, h_b_ref),
                "s5_empty_is_off": bool(same_fit), "s5_alpha_only_differs": bool(a_differs),
                "all_off_reads_no_angle": bool(off_blind)}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, bad=True)

    def judge(self, stat, ref):
        why = [k for k in ("s4a_empty_is_off", "s4a_one_angle_is_other_removed", "s5_empty_is_off",
                           "s5_alpha_only_differs", "all_off_reads_no_angle") if not stat[k]]
        why = [f"{k} fails" for k in why]
        if same_input(stat, ref) and stat["bad"] == ref["stat"]["bad"]:
            why += identity(stat, ref, ["n_tracks"])[1]
        return not why, (f"{stat['n_tracks']} tracks: per-angle switches remove exactly their angle"
                         ) if not why else "; ".join(why)


# ---------------------------------------------------------------- the NaN guard
class AngleNonFinite(Item):
    id = "census.angle_nonfinite"
    title = ("the NaN guard: no stage of a cut run computes a non-finite angle statistic for an IT cluster "
             "with a usable direction (a NaN fails every threshold and vetoes silently)")
    c4_row = "a NaN angle statistic vetoes silently (WP-D0b, 2026-10-15: S1 and the OT stubs)"
    known_bad_desc = ("a missing table width on one surface: IL2's alpha uncertainty set to NaN on the gate input's "
                      "first events, through the census's own S1 (alpha_veto) with the cut file; its non-finite "
                      "statistics must be counted")
    rule_desc = ("the 4-event AAAI+IAAA census with the cut file (census.cut_entries' cut run): the census's NaN-guard "
                 "counts (non-finite S1, S2a, S2z, S3, S4w, S4a and S6 statistics among IT inputs with a usable "
                 "direction, M.NONFINITE_STAGES) are present for every stage and all 0; n/a without the cut file")
    trees = (CENSUS_DIR,)

    def compute(self, ctx):
        from .items_census import CUT_FILE, CutEntries

        if not os.path.exists(CUT_FILE):
            raise NotApplicable(f"no angle cut file at {CUT_FILE}")
        out = CutEntries()._run_cut(ctx)
        return {"input": ctx.input, "events": ctx.events, "variant": None,
                "nonfinite": json.load(open(os.path.join(out, "census.json")))["angle_nonfinite"]}

    def known_bad(self, ctx):
        from .items_census import CUT_FILE

        if not os.path.exists(CUT_FILE):
            raise NotApplicable(f"no angle cut file at {CUT_FILE}")
        CA = ctx.script_module(CENSUS_DIR, "census_angles")
        M = ctx.script_module(CENSUS_DIR, "tracklet_topology_cost")
        model = CA.Model()
        U, Q = _chunk(ctx, model)
        Q = dict(Q)
        Q["s_half_c"] = np.where(U["layer"] == 2, np.nan, Q["s_half_c"])     # IL2's width lost
        ang = M.AngleUse.parse("all")
        ang.model, ang.cuts = model, CA.CutSet.load(CUT_FILE)
        ang.nonfinite = {s: 0 for s in M.NONFINITE_STAGES}
        for L in (1, 2, 3, 4):
            M.alpha_veto(U, Q, np.flatnonzero(U["layer"] == L), 0.5, False, ang)
        return {"input": ctx.input, "events": ctx.events, "variant": "IL2 alpha width NaN",
                "nonfinite": ang.nonfinite}

    def judge(self, stat, ref):
        nf = stat["nonfinite"]
        why = []
        if nf is None:
            why.append("the census wrote no NaN-guard counts")
        else:
            bad = {k: v for k, v in nf.items() if v}
            if bad:
                why.append(f"non-finite angle statistics: {bad}")
            if ref is not None and set(nf) != set(ref["stat"]["nonfinite"] or {}):
                why.append(f"stages counted {sorted(nf)} != reference {sorted(ref['stat']['nonfinite'] or {})}")
        return not why, ("every stage's statistic finite for every IT input: " + ", ".join(
            f"{k} 0" for k in nf)) if not why else "; ".join(why)


ITEMS = [AngleImport(), KKnob(), AngleSwitches(), AngleNonFinite()]
