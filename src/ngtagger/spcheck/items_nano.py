"""Fast-gate items on the nano and the framework readers (plan App. C.4 rows: drop lists,
ambiguous joins, tp_phi at production, PUPPI dxy as cm, payload units, angle leak,
closure reference from another build)."""
from __future__ import annotations

import numpy as np

from .core import Item, event_replicas, identity, same_input, t_family

CL = "L1TSmartPixelsCluster"


def _np(a):
    import awkward as ak

    return ak.to_numpy(a)


# ---------------------------------------------------------------- branch list
class BranchList(Item):
    id = "nano.branch_list"
    title = "tier branch list (name and type) equals the registered reference of the production"
    c4_row = "drop lists; unscheduled default producer"
    known_bad_desc = "the same file with one branch removed (a drop-list change)"
    rule_desc = "identical set of (branch, type) to the reference; every difference listed"

    def _stat(self, ctx, drop=None):
        k = dict(ctx.keys())
        if drop:
            k.pop(drop, None)
        return {"input": ctx.input, "events": ctx.events, "n_branches": len(k),
                "branches": sorted(f"{n}:{t}" for n, t in k.items())}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, drop="L1TSmartPixelsCluster_tpEmuClass")

    @staticmethod
    def diff(stat, ref):
        a, b = set(stat["branches"]), set(ref["stat"]["branches"])
        return {"missing": sorted(b - a), "extra": sorted(a - b)}

    def judge(self, stat, ref):
        if ref is None:
            return False, "no registered reference"
        d = self.diff(stat, ref)
        ok = not d["missing"] and not d["extra"]
        return ok, "identical" if ok else f"missing {d['missing'][:10]}, extra {d['extra'][:10]}"


# ---------------------------------------------------------------- join keys
GATED_KEYS = {
    # name: (table, key columns, range checks {col: count branch})
    "L1TTrackStub(trackIdx,otStubIdx)": ("L1TTrackStub", ("trackIdx", "otStubIdx"),
                                         {"trackIdx": "nL1TTrack", "otStubIdx": "nL1TOTStub"}),
    "L1TExtTrackStub(trackIdx,otStubIdx)": ("L1TExtTrackStub", ("trackIdx", "otStubIdx"),
                                            {"trackIdx": "nL1TExtTrack", "otStubIdx": "nL1TOTStub"}),
    "L1TSmartPixelsTrackCluster(trackIdx,clusterIdx)": (
        "L1TSmartPixelsTrackCluster", ("trackIdx", "clusterIdx"),
        {"trackIdx": "nL1TSmartPixelsTrack", "clusterIdx": "nL1TSmartPixelsCluster"}),
    "L1TSmartPixelsExtTrackCluster(trackIdx,clusterIdx)": (
        "L1TSmartPixelsExtTrackCluster", ("trackIdx", "clusterIdx"),
        {"trackIdx": "nL1TSmartPixelsExtTrack", "clusterIdx": "nL1TSmartPixelsCluster"}),
    "L1TSmartPixelsTrack(l1TrackIdx)": ("L1TSmartPixelsTrack", ("l1TrackIdx",), {"l1TrackIdx": "nL1TTrack"}),
    "L1TSmartPixelsExtTrack(l1TrackIdx)": ("L1TSmartPixelsExtTrack", ("l1TrackIdx",),
                                           {"l1TrackIdx": "nL1TExtTrack"}),
    "L1TSmartPixelsCluster(detId,localX,localY)": (CL, ("detId", "localX", "localY"), {}),
    "L1TTP(genPartIdx>=0)": ("L1TTP", ("genPartIdx",), {"genPartIdx": "nGenPart"}),
    "L1SC4NGJetCands(jetIdx,candIdx)": ("L1SC4NGJetCands", ("jetIdx", "candIdx"),
                                        {"jetIdx": "nL1puppiJetSC4NG", "candIdx": "nL1ExtPuppiCand"}),
}
TRUTH_RANGES = {  # index columns that point into L1TTP (>= 0 entries must be valid rows)
    f"{CL}.tpIdx": (CL, "tpIdx"), "L1TOTStub.tpIdx": ("L1TOTStub", "tpIdx"),
    "L1TOTStub.tpIdxInner": ("L1TOTStub", "tpIdxInner"), "L1TOTStub.tpIdxOuter": ("L1TOTStub", "tpIdxOuter"),
    "L1TTrack.tpIdx": ("L1TTrack", "tpIdx"),
}
REPORTED_KEYS = {  # not unique by design: reported (identity), never used as join keys
    "L1TOTStub(detId,x,y,z)": ("L1TOTStub", ("detId", "x", "y", "z")),
    "L1TOTStub(detId,x,y,z,bend)": ("L1TOTStub", ("detId", "x", "y", "z", "bend")),
    "L1ExtPuppiCand(l1TrackIdx>=0)": ("L1ExtPuppiCand", ("l1TrackIdx",)),
}


def _dups(rows: np.ndarray) -> int:
    if len(rows) == 0:
        return 0
    return int(len(rows) - len(np.unique(rows, axis=0)))


class JoinUniqueness(Item):
    id = "nano.join_uniqueness"
    title = "every join key of the nano is unique and every link index is in range"
    c4_row = "ambiguous stub-truth join"
    known_bad_desc = "one (trackIdx, otStubIdx) link row of L1TTrackStub duplicated in event 0"
    rule_desc = ("zero duplicates and zero out-of-range indices for every gated key; event identities unique; "
                 "the reported (non-key) counts identical to the reference")

    def _branches(self, ctx):
        keys = ctx.keys()
        br = set()
        for tab, cols, rng in GATED_KEYS.values():
            br |= {f"{tab}_{c}" for c in cols} | set(rng.values())
        for tab, col in TRUTH_RANGES.values():
            br.add(f"{tab}_{col}")
        for tab, cols in REPORTED_KEYS.values():
            br |= {f"{tab}_{c}" for c in cols}
        br |= {"nL1TTP", "run", "luminosityBlock", "event"}
        return sorted(b for b in br if b in keys)

    def _stat(self, ctx, inject=False):
        import awkward as ak

        A = ctx.arrays(self._branches(ctx))
        f = set(A.fields)
        n_ev = len(A)
        out = {"input": ctx.input, "events": ctx.events, "gated": {}, "truth_ranges": {}, "reported": {}}
        for name, (tab, cols, rng) in GATED_KEYS.items():
            if not all(f"{tab}_{c}" in f for c in cols):
                out["gated"][name] = "absent"
                continue
            dup = bad = rows = 0
            for ev in range(n_ev):
                cs = [_np(A[f"{tab}_{c}"][ev]).astype(np.float64) for c in cols]
                M = np.stack(cs, 1) if cs[0].size else np.zeros((0, len(cols)))
                if inject and ev == 0 and name.startswith("L1TTrackStub(") and len(M):
                    M = np.vstack([M, M[:1]])
                if name == "L1TTP(genPartIdx>=0)":
                    M = M[M[:, 0] >= 0]
                if name == "L1TSmartPixelsExtTrack(l1TrackIdx)":
                    bad += int(not np.array_equal(M[:, 0], np.arange(len(M))))   # must be the row index
                rows += len(M)
                dup += _dups(M)
                for c, nb in rng.items():
                    if nb in f:
                        v = M[:, cols.index(c)]
                        lim = int(A[nb][ev])
                        bad += int(((v >= lim) | (v < (0 if c != "genPartIdx" else -1))).sum())
            out["gated"][name] = {"rows": rows, "duplicates": dup, "out_of_range": bad}
        for name, (tab, col) in TRUTH_RANGES.items():
            if f"{tab}_{col}" not in f:
                continue
            v = A[f"{tab}_{col}"]
            n_tp = A["nL1TTP"]
            out["truth_ranges"][name] = int(ak.sum(v >= n_tp))
        ids = np.stack([_np(A[c]).astype(np.int64) for c in ("run", "luminosityBlock", "event")], 1)
        out["event_id_duplicates"] = _dups(ids)
        for name, (tab, cols) in REPORTED_KEYS.items():
            dup = rows = 0
            for ev in range(n_ev):
                M = np.stack([_np(A[f"{tab}_{c}"][ev]).astype(np.float64) for c in cols], 1)
                if name.startswith("L1ExtPuppiCand"):
                    M = M[M[:, 0] >= 0]
                rows += len(M)
                dup += _dups(M)
            out["reported"][name] = {"rows": rows, "duplicates": dup}
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, inject=True)

    def judge(self, stat, ref):
        bad = [f"{k}: {v}" for k, v in stat["gated"].items()
               if isinstance(v, dict) and (v["duplicates"] or v["out_of_range"])]
        bad += [f"{k}: {v} indices beyond L1TTP" for k, v in stat["truth_ranges"].items() if v]
        if stat["event_id_duplicates"]:
            bad.append(f"event ids: {stat['event_id_duplicates']} duplicates")
        if same_input(stat, ref):
            ok, diffs = identity(stat, ref, ["reported"])
            bad += diffs
        return not bad, "all keys unique and in range" if not bad else "; ".join(bad[:8])


# ---------------------------------------------------------------- truth helix
# stored phi0 vs truth_helix.poca_phi0 on TPs with pT > 1 GeV: measured on the known-good (5 events,
# 72.7k TPs) max 6.7e-6 rad from the float32 inputs of displaced TPs (prompt: 2.6e-7); the tolerance
# is 15x that, and 25x below the median turn between production and POCA of the displaced TPs
# the item uses (~2.5e-3 rad, the 74 um median of the known-bad at r ~ 3 cm).
PHI0_TOL_RAD = 1e-4


class TpPhi0Helix(Item):
    id = "truth.tp_phi0_helix"
    title = "the truth helix at the POCA (phi0, d0 = -dxy) passes each TP's innermost cluster"
    c4_row = "tp_phi at production, not at the POCA"
    known_bad_desc = "the production azimuth (L1TTP_phi) substituted for phi0"
    rule_desc = ("D = median |r dphi| of displaced TPs (r_prod 0.05-2.5 cm) minus that of prompt TPs "
                 "(r_prod < 0.01 cm), pT > 1 GeV, innermost class-T barrel cluster, agrees with the registered "
                 "known-good D: |D - D_ref| <= t(1) = 2.58 combined event-bootstrap sigma (300 replicas); stored phi0 = truth_helix.poca_phi0 "
                 "to 1e-4 rad (pT > 1 GeV; measured max 6.7e-6); identity with the reference on the reference input")
    N_REP = 300

    def _rows(self, ctx):
        tc = ["pt", "eta", "phi", "phi0", "d0", "z0", "charge", "vx", "vy"]
        cc = ["tpIdx", "tpEmuClass", "globalR", "globalPhi", "subdet"]
        A = ctx.arrays([f"{CL}_{c}" for c in cc] + [f"L1TTP_{c}" for c in tc])
        rows = []
        for ev in range(len(A)):
            C = {c: _np(A[f"{CL}_{c}"][ev]) for c in cc}
            T = {c: _np(A[f"L1TTP_{c}"][ev]).astype(np.float64) for c in tc}
            sel = (C["tpIdx"] >= 0) & (C["tpEmuClass"] == 1) & (C["subdet"] == 1)
            idx, R, P = C["tpIdx"][sel], C["globalR"][sel].astype(np.float64), C["globalPhi"][sel].astype(np.float64)
            o = np.lexsort((R, idx))
            idx, R, P = idx[o], R[o], P[o]
            first = np.r_[True, idx[1:] != idx[:-1]] if len(idx) else np.zeros(0, bool)
            idx, R, P = idx[first], R[first], P[first]
            blk = np.column_stack([np.full(len(idx), ev), R, P] + [T[c][idx] for c in tc])
            rows.append(blk)
        a = np.vstack(rows)
        names = ["ev", "R", "P"] + tc
        return {n: a[:, i] for i, n in enumerate(names)}

    def _stat(self, ctx, use_production_phi=False):
        from ngtagger import truth_helix as TH
        from ngtagger.definitions import helix as H

        r = self._rows(ctx)
        recomputed = TH.poca_phi0(r["phi"], r["pt"], r["charge"], r["vx"], r["vy"])
        ok_q = (r["pt"] > 1.0) & (r["charge"] != 0)
        dphi0_max = float(np.max(np.abs(H.wrap(recomputed[ok_q] - r["phi0"][ok_q]))))
        phi_used = r["phi"] if use_production_phi else TH.tp_phi0({"phi0": r["phi0"]}, prefix="")
        x, y, _, _ = H.at_radius(r["R"], r["pt"], r["charge"], phi_used, r["d0"])
        res = np.abs(r["R"] * H.wrap(r["P"] - np.arctan2(y, x))) * 1e4          # um
        rprod = np.hypot(r["vx"], r["vy"])
        base = (r["pt"] > 1.0) & np.isfinite(res)
        disp = base & (rprod > 0.05) & (rprod < 2.5)
        prompt = base & (rprod < 0.01)
        ev = r["ev"].astype(int)
        n_ev = int(ev.max()) + 1

        def D(w):
            wd, wp = w[ev[disp]], w[ev[prompt]]
            return _wmedian(res[disp], wd) - _wmedian(res[prompt], wp)

        d0 = D(np.ones(n_ev))
        reps = np.array([D(c.astype(float)) for c in event_replicas(n_ev, self.N_REP, 20261009)])
        sd = float(np.std(reps, ddof=1))
        return {"input": ctx.input, "events": ctx.events, "n_prompt": int(prompt.sum()), "n_displaced": int(disp.sum()),
                "median_um_prompt": float(np.median(res[prompt])), "median_um_displaced": float(np.median(res[disp])),
                "p90_um_displaced": float(np.percentile(res[disp], 90)), "D_um": float(d0), "sigma_D_um": sd,
                "max_abs_phi0_minus_poca_phi0_rad": dphi0_max,
                "phi_used": "production phi (KNOWN-BAD)" if use_production_phi else "phi0 at the POCA"}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, use_production_phi=True)

    def judge(self, stat, ref):
        if ref is None:
            return False, "no registered reference"
        why = []
        r = ref["stat"]
        z = (stat["D_um"] - r["D_um"]) / max(np.hypot(stat["sigma_D_um"], r["sigma_D_um"]), 1e-12)
        if not abs(z) <= t_family(1):
            why.append(f"displaced-minus-prompt median {stat['D_um']:.1f} um vs known-good {r['D_um']:.1f} um: "
                       f"{z:.1f} sigma (t = {t_family(1):.2f})")
        if not stat["max_abs_phi0_minus_poca_phi0_rad"] < PHI0_TOL_RAD:
            why.append(f"stored phi0 differs from poca_phi0 by {stat['max_abs_phi0_minus_poca_phi0_rad']:.2e} rad")
        if same_input(stat, ref) and stat["phi_used"] == ref["stat"]["phi_used"]:
            ok, diffs = identity(stat, ref, ["n_prompt", "n_displaced", "median_um_prompt", "median_um_displaced",
                                             "D_um"])
            why += diffs
        return not why, (f"z vs known-good = {z:.2f}; displaced median {stat['median_um_displaced']:.1f} um, "
                         f"prompt {stat['median_um_prompt']:.1f} um") if not why else "; ".join(why)


def _wmedian(x, w):
    if len(x) == 0 or w.sum() <= 0:
        return np.nan
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


# ---------------------------------------------------------------- PUPPI dxy
class PuppiDxy(Item):
    id = "puppi.dxy_lut"
    title = "the PF/PUPPI candidate dxy is read through the code table, never as cm"
    c4_row = "PUPPI dxy read as cm"
    known_bad_desc = "the float `dxy` branch (hwDxy / 64, sqrt(cm) units) read as |d0| in cm"
    rule_desc = ("every linked charged candidate's |d0| lies in the |d0| interval its code decodes to "
                 "(fraction = 1 exactly); code 3 <-> d0 in [-39.0625, 78.125) um exactly; identity of counts")

    def _stat(self, ctx, read_float_as_cm=False):
        import awkward as ak

        from ngtagger.definitions import puppi_dxy as PD

        cand, trk = "L1ExtPuppiCand", "L1TSmartPixelsExtTrack"
        A = ctx.arrays([f"{cand}_hwDxy", f"{cand}_dxy", f"{cand}_l1TrackIdx", f"{trk}_d0"])
        li = A[f"{cand}_l1TrackIdx"]
        ok = li >= 0
        d0 = _np(ak.flatten(A[f"{trk}_d0"][ak.where(ok, li, 0)][ok])).astype(np.float64)
        hw = _np(ak.flatten(A[f"{cand}_hwDxy"][ok])).astype(np.int64)
        fl = _np(ak.flatten(A[f"{cand}_dxy"][ok])).astype(np.float64)
        d0_um = d0 * 1e4
        if read_float_as_cm:
            lo = (fl - 0.5 * PD.DXY_FLOAT_PER_CODE) * 1e4
            hi = (fl + 0.5 * PD.DXY_FLOAT_PER_CODE) * 1e4
            contains = (np.abs(d0_um) >= lo) & (np.abs(d0_um) < hi)
            code3_says_small = fl * 1e4 < 78.125
        else:
            contains = PD.code_contains_d0(hw, d0)
            code3_says_small = hw == 3
        truly_small = (d0_um >= -39.0625) & (d0_um < 78.125)
        return {"input": ctx.input, "events": ctx.events, "reader": "float as cm (KNOWN-BAD)" if read_float_as_cm
                else "code table", "n_linked": int(len(hw)), "frac_contained": float(np.mean(contains)),
                "frac_code3_consistent": float(np.mean(code3_says_small == truly_small)),
                "n_code3": int((hw == 3).sum()),
                "n_negative_d0_39_to_78um_code7": int(((d0_um >= -78.125) & (d0_um < -39.0625) & (hw == 7)).sum())}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, read_float_as_cm=True)

    def judge(self, stat, ref):
        why = []
        if stat["frac_contained"] != 1.0:
            why.append(f"only {stat['frac_contained']:.4f} of candidates decode to their track's d0")
        if stat["frac_code3_consistent"] != 1.0:
            why.append(f"code 3 <-> [-39, 78) um holds for {stat['frac_code3_consistent']:.4f}")
        if same_input(stat, ref) and stat["reader"] == ref["stat"]["reader"]:
            why += identity(stat, ref, ["n_linked", "n_code3", "n_negative_d0_39_to_78um_code7"])[1]
        return not why, f"{stat['n_linked']} candidates decode exactly" if not why else "; ".join(why)


# ---------------------------------------------------------------- payload units
ABS_COT_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 12.0)
ANGLES = (("alpha", "localCotAlpha", "tpLocalCotAlpha", "sigAlpha", "hasAlpha"),
          ("beta", "localCotBeta", "tpLocalCotBeta", "sigBeta", "hasBeta"))


class PayloadUnits(Item):
    id = "angles.payload_units"
    title = "class-T emitted-minus-true width in units of the published sigma, per surface and |cot| bin"
    c4_row = "payload units (angles 1.9-6.5x too precise)"
    known_bad_desc = "the published sigmas divided by 2 (input perturbation)"
    rule_desc = ("per surface x angle x |true local cot| bin with >= 200 class-T clusters: B.1 core of "
                 "(emitted - true) / sigma agrees with the reference within t(N) combined event-bootstrap "
                 "sigma (250 replicas) and the chi2 over bins has p > 0.01; identity on the reference input")
    N_REP = 250

    def _stat(self, ctx, sigma_scale=1.0):
        from ngtagger.data import spix_nano as SPN
        from ngtagger.definitions.estimators import core

        cols = ["surface", "tpEmuClass"] + [c for a in ANGLES for c in a[1:]]
        key = ("payload_clusters", ctx.input, ctx.events)
        if key not in ctx.cache:
            J = SPN.read_clusters(ctx.input, cols, entry_stop=ctx.events)
            import awkward as ak

            n = _np(ak.num(J["surface"]))
            D = {c: _np(ak.flatten(J[c])) for c in cols}
            D["event"] = np.repeat(np.arange(len(n)), n)
            ctx.cache[key] = D
        D = ctx.cache[key]
        T = D["tpEmuClass"] == 1
        reps = event_replicas(int(D["event"].max()) + 1, self.N_REP, 20261009)
        bins = {}
        for ang, e, tr, sg, has in ANGLES:
            has_ok = D[has].astype(bool) & (np.abs(D[e]) <= 12)
            for s in np.unique(D["surface"]):
                for lo, hi in zip(ABS_COT_EDGES[:-1], ABS_COT_EDGES[1:]):
                    m = T & has_ok & (D["surface"] == s) & (np.abs(D[tr]) >= lo) & (np.abs(D[tr]) < hi)
                    if m.sum() < 200:
                        continue
                    p = (D[e][m].astype(np.float64) - D[tr][m]) / (D[sg][m] * sigma_scale)
                    c = core(p)
                    if c is None:
                        continue
                    ev = D["event"][m]
                    order = np.argsort(ev, kind="stable")
                    p_ev = np.split(p[order], np.searchsorted(ev[order], np.arange(1, reps.shape[1])))
                    vals = []
                    for cnt in reps:
                        x = np.concatenate([np.tile(p_ev[i], k) for i, k in enumerate(cnt) if k])
                        cc = core(x)
                        if cc is not None:
                            vals.append(cc["rms0"])
                    bins[f"{ang}/S{int(s)}/{lo:g}-{hi:g}"] = {"n": int(m.sum()), "width": c["rms0"],
                                                               "sigma": float(np.std(vals, ddof=1))}
        return {"input": ctx.input, "events": ctx.events, "sigma_scale": sigma_scale, "bins": bins}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, sigma_scale=0.5)

    def judge(self, stat, ref):
        from scipy.stats import chi2

        if ref is None:
            return False, "no registered reference"
        rb = ref["stat"]["bins"]
        common = [k for k in stat["bins"] if k in rb]
        z = np.array([(stat["bins"][k]["width"] - rb[k]["width"])
                      / max(np.hypot(stat["bins"][k]["sigma"], rb[k]["sigma"]), 1e-12) for k in common])
        t = t_family(len(common))
        p = float(chi2.sf(float(np.sum(z ** 2)), len(common))) if len(common) else 0.0
        why = []
        if len(common) == 0:
            why.append("no common bins")
        if np.any(np.abs(z) > t):
            k = common[int(np.argmax(np.abs(z)))]
            why.append(f"{int(np.sum(np.abs(z) > t))} of {len(common)} bins beyond t(N) = {t:.2f}; worst {k}: "
                       f"{stat['bins'][k]['width']:.3f} vs {rb[k]['width']:.3f} ({z[np.argmax(np.abs(z))]:.1f} sigma)")
        if p <= 0.01:
            why.append(f"chi2 p = {p:.2g}")
        if same_input(stat, ref) and stat["sigma_scale"] == ref["stat"]["sigma_scale"]:
            ok, diffs = identity(stat, ref, ["bins"])
            why += diffs
        return not why, (f"{len(common)} bins, max |z| {np.max(np.abs(z)):.2f} (t = {t:.2f}), chi2 p {p:.2f}"
                         if not why else "; ".join(why[:6]))


# ---------------------------------------------------------------- angle leak
LEAK_COLUMNS = ("localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta", "sizeX", "sizeY", "charge",
                "hasAlpha", "hasBeta", "satAlpha", "satBeta", "localX", "localY")
TRUTH_PAIRS = (("localCotAlpha", "tpLocalCotAlpha"), ("localCotBeta", "tpLocalCotBeta"))


class AngleLeak(Item):
    id = "angles.leak"
    title = "no truth in emitted cluster columns: linked-vs-unlinked distributions per column (k = 1)"
    c4_row = "angle leak / class tells"
    known_bad_desc = "the true local cotBeta written into localCotBeta of the linked clusters"
    rule_desc = ("per surface x emitted column: the KS distance between TP-linked and unlinked clusters "
                 "identical to the reference; the fraction of linked clusters whose emitted angle equals the "
                 "truth exactly <= 1% (reference 0.02-0.09%)")

    def _stat(self, ctx, leak=False):
        import awkward as ak
        from scipy.stats import ks_2samp

        cols = ["surface", "tpIdx"] + list(LEAK_COLUMNS) + [t for _, t in TRUTH_PAIRS]
        A = ctx.arrays([f"{CL}_{c}" for c in cols])
        D = {c: _np(ak.flatten(A[f"{CL}_{c}"])).astype(np.float64) for c in cols}
        linked = D["tpIdx"] >= 0
        if leak:
            D["localCotBeta"] = np.where(linked, D["tpLocalCotBeta"], D["localCotBeta"])
        ks = {}
        for s in np.unique(D["surface"]):
            ms = D["surface"] == s
            for c in LEAK_COLUMNS:
                a, b = D[c][ms & linked], D[c][ms & ~linked]
                ks[f"S{int(s)}/{c}"] = float(ks_2samp(a, b).statistic) if len(a) and len(b) else None
        exact = {e: float(np.mean(D[e][linked] == D[t][linked])) for e, t in TRUTH_PAIRS}
        return {"input": ctx.input, "events": ctx.events, "leak_injected": leak, "ks": ks,
                "frac_linked_emitted_equals_truth": exact}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, leak=True)

    def judge(self, stat, ref):
        why = [f"{k}: {v:.3f} of linked clusters emit the truth" for k, v in
               stat["frac_linked_emitted_equals_truth"].items() if v > 0.01]
        if ref is None:
            why.append("no registered reference")
        elif same_input(stat, ref):
            why += identity(stat, ref, ["ks"])[1]
        return not why, "no leak signature" if not why else "; ".join(why[:6])


# ---------------------------------------------------------------- same OT tracks
OT_COLS = ("pt", "eta", "phi", "z0", "d0", "chi2XYRed", "chi2ZRed", "nStubs", "hitPattern")


class SameOtTracks(Item):
    id = "ot.same_tracks"
    title = "two files claimed to come from the same OT build hold the same L1TTrack tracks, event by event"
    c4_row = "closure reference from another build"
    known_bad_desc = "the partner file with one track's z0 moved by 1e-4 cm in event 0"
    rule_desc = ("for every event (matched by run, lumi, event) the sorted L1TTrack rows (pt, eta, phi, z0, d0, "
                 "chi2s, nStubs, hitPattern) are bit-identical in AAAA and IIII file 3")

    def partner(self, ctx):
        return ctx.input.replace("spstudy_AAAA_", "spstudy_IIII_")

    def _stat(self, ctx, perturb=False):
        ia, A = self._read(ctx, ctx.input)
        ib, B = self._read(ctx, self.partner(ctx), n_all=True)
        pos = {tuple(x): i for i, x in enumerate(ib)}
        n_same = n_missing = n_tracks = 0
        for i, x in enumerate(ia):
            j = pos.get(tuple(x))
            if j is None:
                n_missing += 1
                continue
            ra = np.stack([_np(A[f"L1TTrack_{c}"][i]).astype(np.float64) for c in OT_COLS], 1)
            rb = np.stack([_np(B[f"L1TTrack_{c}"][j]).astype(np.float64) for c in OT_COLS], 1)
            if perturb and i == 0 and len(rb):
                rb = rb.copy()
                rb[0, 3] += 1e-4
            ra, rb = ra[np.lexsort(ra.T[::-1])], rb[np.lexsort(rb.T[::-1])]
            n_tracks += len(ra)
            n_same += int(ra.shape == rb.shape and np.array_equal(ra, rb))
        return {"input": ctx.input, "events": ctx.events, "partner": self.partner(ctx), "perturbed": perturb,
                "n_events": len(ia), "n_events_identical": n_same, "n_events_missing_in_partner": n_missing,
                "n_tracks": n_tracks}

    def _read(self, ctx, path, n_all=False):
        import uproot

        t = uproot.open(path)["Events"]
        stop = None if n_all else ctx.events
        ids = np.stack([t[c].array(library="np", entry_stop=stop).astype(np.int64)
                        for c in ("run", "luminosityBlock", "event")], 1)
        A = ctx.arrays([f"L1TTrack_{c}" for c in OT_COLS], path=path, entry_stop=stop or t.num_entries)
        return ids, A

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, perturb=True)

    def judge(self, stat, ref):
        ok = stat["n_events"] > 0 and stat["n_events_identical"] == stat["n_events"]
        return ok, (f"{stat['n_events']} events, {stat['n_tracks']} tracks identical" if ok else
                    f"{stat['n_events'] - stat['n_events_identical']} of {stat['n_events']} events differ "
                    f"({stat['n_events_missing_in_partner']} missing)")


ITEMS = [BranchList(), JoinUniqueness(), TpPhi0Helix(), PuppiDxy(), PayloadUnits(), AngleLeak(), SameOtTracks()]
