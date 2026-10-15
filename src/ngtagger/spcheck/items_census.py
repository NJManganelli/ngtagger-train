"""Fast-gate items on the census (plan App. C.4 rows: int32 owner sentinel; per-build
follow and DR over all builds; the census input contract). They import the census from THIS
code root's eval_refitq/combinatorics (the frozen copy under the launcher) and run a 4-event,
two-build census (AAAI, IAAA: the builds with an inactive layer) into a scratch cache: ~70 s,
~1.2 GB."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np

from .core import Item, identity, same_input

CENSUS_DIR = "eval_refitq/combinatorics"
CENSUS_EVENTS = 4
CENSUS_TIMEOUT_S = 1800
CENSUS_BUILDS = "AAAI,IAAA"


def _np(a):
    import awkward as ak

    return ak.to_numpy(a)


# ---------------------------------------------------------------- majority owner
class MajorityOwner(Item):
    id = "census.majority_owner"
    title = ("the census majority owner on real track hit tables, every hit voting for every TP it is credited to: "
             "n_own <= real hits, owner on the track, dtypes")
    c4_row = "int32 owner sentinel (+ the format-28 stub credit, definition sheet v2 truth.stub)"
    known_bad_desc = ("two variants, each of which must be red: (sentinel) an int32 sentinel injected: empty layer "
                      "slots handed to the census as valid hits carrying np.iinfo(int32).min (the missing-mask/sentinel "
                      "class of the 813be3e bug); (index tie-break) the census owner with every tie given to the larger "
                      "index again, the rule of census formats <= 31")
    rule_desc = ("KF.credited_owner on the OT tracks of the gate input (stubs from L1TTrackStub -> L1TOTStub, each "
                 "credited to its tpIdx, tpIdxInner and tpIdxOuter through M.credited_tps; SP-refit IT clusters from "
                 "L1TSmartPixelsTrackCluster, credited to their tpIdx; empty slots invalid) equals a plain counting "
                 "reference (each real hit counts once for every distinct matched TP among its raw truth columns; "
                 "the most counts wins, a tie gives no owner, as CMSSW) on every track; n_own <= real hits; owner -1 or a TP credited by a hit on the track; input tpIdx "
                 "int32, output int64; identity of the counts with the reference, including the number of tracks "
                 "whose owner the cluster credit changes (against the stub's own tpIdx alone)")
    trees = (CENSUS_DIR,)

    def _tables(self, ctx):
        """(KF, credited TPs [track x slot x credit], raw truth columns [same shape], valid, dtypes read) for the OT
        tracks of the gate input. `credited` goes through the census's M.credited_tps; `raw` is the nano's columns
        as read (IT clusters: tpIdx, -1, -1), for the independent reference."""
        M = ctx.script_module(CENSUS_DIR, "tracklet_topology_cost")
        KF = ctx.script_module(CENSUS_DIR, "kf_emulation")
        scols = [f"L1TOTStub_{c}" for c in M.STUB_TP_COLS]
        A = ctx.arrays(["L1TTrackStub_trackIdx", "L1TTrackStub_otStubIdx", *scols, "L1TOTStub_layer",
                        "L1TOTStub_isBarrel", "nL1TTrack", "L1TSmartPixelsTrack_l1TrackIdx",
                        "L1TSmartPixelsTrackCluster_trackIdx", "L1TSmartPixelsTrackCluster_clusterIdx",
                        "L1TSmartPixelsCluster_tpIdx", "L1TSmartPixelsCluster_surface"])
        slot = {L: i for i, L in enumerate(KF.LAYER_ORDER)}
        K = M.N_CREDIT
        creds, raws, valids = [], [], []
        import awkward as ak

        dtype_in = sorted({str(ak.to_numpy(ak.flatten(A[c])).dtype)
                           for c in scols + ["L1TSmartPixelsCluster_tpIdx"]})
        rng = np.random.default_rng(7)
        for ev in range(len(A)):
            nt = int(A["nL1TTrack"][ev])
            # junk in empty slots
            raw = rng.integers(-5, 50, size=(nt, len(KF.LAYER_ORDER), K)).astype(np.int32)
            cred = raw.copy()
            va = np.zeros((nt, len(KF.LAYER_ORDER)), bool)
            ti = _np(A["L1TTrackStub_trackIdx"][ev])
            si = _np(A["L1TTrackStub_otStubIdx"][ev])
            lay = _np(A["L1TOTStub_layer"][ev])[si]
            brl = _np(A["L1TOTStub_isBarrel"][ev])[si].astype(bool)
            st = np.stack([_np(A[c][ev])[si] for c in scols], axis=1)
            sc = M.credited_tps(*st.T)
            for t, L, b, v, c in zip(ti, lay, brl, st, sc):
                if b and (10 + int(L)) in slot:
                    raw[t, slot[10 + int(L)]] = v
                    cred[t, slot[10 + int(L)]] = c
                    va[t, slot[10 + int(L)]] = True
            l1 = _np(A["L1TSmartPixelsTrack_l1TrackIdx"][ev])
            cti = _np(A["L1TSmartPixelsTrackCluster_trackIdx"][ev])
            cci = _np(A["L1TSmartPixelsTrackCluster_clusterIdx"][ev])
            ctp = _np(A["L1TSmartPixelsCluster_tpIdx"][ev])
            csf = _np(A["L1TSmartPixelsCluster_surface"][ev])
            for t, c in zip(cti, cci):
                s = int(csf[c])
                if s in (1, 2, 3, 4):
                    raw[l1[t], slot[s]] = (ctp[c], -1, -1)
                    cred[l1[t], slot[s]] = M.credited_tps(ctp[c:c + 1])[0]
                    va[l1[t], slot[s]] = True
            creds.append(cred)
            raws.append(raw)
            valids.append(va)
        return KF, np.vstack(creds), np.vstack(raws), np.vstack(valids), dtype_in

    @staticmethod
    def reference(raw, va, matched, ties="none"):
        """Plain counting: every valid hit counts once for each distinct matched TP among its raw columns; the most
        counts wins. A tie gives no owner (-1, 0) as CMSSW, or with ties="largest" (only to build the known-bad)
        the largest index, the rule of census formats <= 31."""
        own, n = np.full(len(raw), -1, np.int64), np.zeros(len(raw), np.int64)
        for i in range(len(raw)):
            c = Counter(int(x) for h in range(raw.shape[1]) if va[i, h]
                        for x in {int(y) for y, m in zip(raw[i, h], matched[i, h]) if m})
            if c:
                best = max(c.values())
                top = [k for k, v in c.items() if v == best]
                if len(top) == 1 or ties == "largest":
                    own[i], n[i] = max(top), best
        return own, n

    def _stat(self, ctx, inject=False, index_tiebreak=False):
        from ngtagger.data import spix_nano as SPN

        KF, cred, raw, va, dtype_in = self._tables(ctx)
        real = va & SPN.tp_matched(raw).any(axis=2)
        ref_own, ref_n = self.reference(raw, va, SPN.tp_matched(raw))
        # context: the owner with each stub credited to its own tpIdx alone (format <= 27)
        own27, _, _ = KF.majority_owner(raw[:, :, 0], va)
        if inject:
            cred = np.where(va[:, :, None], cred, np.iinfo(np.int32).min).astype(np.int32)
            va_in = np.ones_like(va)
        else:
            va_in = va
        own, n, _, _, n_top = KF.credited_owner(cred, va_in)
        if index_tiebreak:                       # every tie given to the larger index again (formats <= 31)
            lo, ln = self.reference(raw, va, SPN.tp_matched(raw), ties="largest")
            own, n = np.where(n_top >= 2, lo, own), np.where(n_top >= 2, ln, n)
        on_track = np.array([o == -1 or bool(np.any(real[i] & (raw[i] == o).any(axis=1))) for i, o in enumerate(own)])
        return {"input": ctx.input, "events": ctx.events, "sentinel_injected": inject,
                "index_tiebreak": index_tiebreak, "n_tracks": int(len(own)),
                "n_with_owner": int((ref_own != -1).sum()), "n_tied": int((n_top >= 2).sum()), "n_agree_with_reference": int(np.sum((own == ref_own) & (n == ref_n))),
                "n_owner_changed_by_cluster_credit": int(np.sum(ref_own != own27)),
                "n_own_gt_real_hits": int(np.sum(n > real.sum(1))), "n_owner_not_on_track": int(np.sum(~on_track)),
                "tpIdx_dtype_read": dtype_in, "tpIdx_dtype_census_input": str(cred.dtype),
                "owner_dtype": str(np.asarray(own).dtype), "count_dtype": str(np.asarray(n).dtype)}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return {"input": ctx.input, "events": ctx.events,
                "variants": {"sentinel": self._stat(ctx, inject=True),
                             "index_tiebreak": self._stat(ctx, index_tiebreak=True)}}

    def judge(self, stat, ref):
        if "variants" in stat:
            # a known-bad of several variants is red only when EVERY variant is red
            res = {k: self.judge(v, ref) for k, v in stat["variants"].items()}
            return any(r[0] for r in res.values()), "; ".join(
                f"{k}: {'GREEN' if r[0] else 'red'} ({r[1]})" for k, r in res.items())
        why = []
        if stat["n_agree_with_reference"] != stat["n_tracks"]:
            why.append(f"{stat['n_tracks'] - stat['n_agree_with_reference']} of {stat['n_tracks']} tracks disagree "
                       "with the counting reference")
        if stat["n_own_gt_real_hits"]:
            why.append(f"n_own > real hits on {stat['n_own_gt_real_hits']} tracks")
        if stat["n_owner_not_on_track"]:
            why.append(f"owner not among the track's hits on {stat['n_owner_not_on_track']} tracks")
        if stat["tpIdx_dtype_read"] != ["int32"] or stat["owner_dtype"] != "int64":
            why.append(f"dtypes read/out {stat['tpIdx_dtype_read']}/{stat['owner_dtype']}")
        if same_input(stat, ref) and stat["sentinel_injected"] == ref["stat"]["sentinel_injected"]:
            why += identity(stat, ref, ["n_tracks", "n_with_owner", "n_tied", "n_owner_changed_by_cluster_credit"])[1]
        return not why, (f"{stat['n_tracks']} tracks, owners identical to the reference "
                         f"({stat['n_tied']} tied, no owner; {stat['n_owner_changed_by_cluster_credit']} changed by "
                         f"the cluster credit)") \
            if not why else "; ".join(why)


# ---------------------------------------------------------------- census output
class CensusBuilds(Item):
    id = "census.builds_and_dr"
    title = "census output: no hit outside its build's surfaces, DR groups within one build, owner columns sane"
    c4_row = "per-build follow; DR over all builds (+ int32 owner sentinel on the output)"
    known_bad_desc = ("two variants of the same census output, each of which must be red: (structure) one AAAI "
                      "track's hit moved onto IL4 (another build's surface), one DR group spanning both builds, and "
                      "one n_own set beyond nhit; (findable count) one TP with credited hits on 3 modules given a "
                      "fourth, as a second hit on one module counted again would, judged by the identities; (tie "
                      "break) one tied track (n_tp_top >= 2) given an owner, as index tie-breaking would")
    rule_desc = ("on a fresh 4-event AAAI+IAAA census of the gate input run from this code root (angle cuts "
                 "open, --calibrate): every exported "
                 "hit lies on its build's active IT surfaces or the OT; every seed and follow surface is active in "
                 "its build; every DR group lies inside one build; n_own <= nhit; own_tpidx = -1 iff n_own = 0 iff "
                 "n_tp_top != 1 (no credited hit, or a tie: no owner, as CMSSW); "
                 "identity of the counts with the reference: tracks, and the truth the census wires in (TPs in the "
                 "census table, findable TPs (TF.findable: charged, credited hits on >= 4 modules, the efficiency "
                 "denominator), tied tracks, "
                 "and hits crediting their track's owner, which move if the stub credit of M.credited_tps stops "
                 "reaching the census)")
    trees = (CENSUS_DIR,)

    def _run(self, ctx) -> str:
        key = ("census_cache", ctx.input)
        if key in ctx.cache:
            return ctx.cache[key]
        root = ctx.tree(CENSUS_DIR)
        out = tempfile.mkdtemp(prefix="spcheck_census_", dir=ctx.scratch)
        # --calibrate: every angle cut open (census format >= 33 cuts with a cut file); the
        # structure and the truth this item checks do not depend on the cut levels, and the item
        # must run before any cut file exists
        cmd = [sys.executable, "-u", str(root / "tp_findability.py"), "-i", ctx.input, "-n", str(CENSUS_EVENTS),
               "--chunk", "2", "--calib-events", str(CENSUS_EVENTS), "--export-tracks", "--it-ptmin", "1.0",
               "--min-layers", "4", "--min-it-layers", "3", "--masks", CENSUS_BUILDS, "--cache-dir", out,
               "--calibrate", "-o", os.path.join(out, "census.json")]
        # A STARVED RUN FAILS LOUDLY: the 4-event census takes ~2 min (cuts open); under memory
        # pressure it once crawled for 2 h while holding the heavy lane, so it is bounded
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ctx.code_root),
                               timeout=CENSUS_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"the 4-event census exceeded {CENSUS_TIMEOUT_S} s (machine starved? "
                               f"check memory pressure and swap)") from None
        if r.returncode != 0:
            raise RuntimeError(f"census failed rc={r.returncode}: {r.stderr[-2000:]}")
        d = glob.glob(os.path.join(out, "tpcensus_*"))[0]
        ctx.cache[key] = d
        return d

    def _stat(self, ctx, perturb=False, perturb_findable=False, perturb_tie=False):
        from ngtagger.data import spix_nano as SPN

        KF = ctx.script_module(CENSUS_DIR, "kf_emulation")
        TF = ctx.script_module(CENSUS_DIR, "tp_findability")
        d = self._run(ctx)
        man = json.load(open(os.path.join(d, "manifest.json")))
        builds = {b: [int(e) for e in es] for b, es in man["builds"].items()}
        dr = [[int(e) for e in g] for g in man["dr_groups"]]
        seeds = man["seeds"] if isinstance(man["seeds"], list) else json.loads(man["seeds"])
        follow = man["seed_follow"] if isinstance(man["seed_follow"], list) else json.loads(man["seed_follow"])
        TC = list(KF.TRACK_COLS)
        it_slots = [i for i, L in enumerate(KF.LAYER_ORDER) if L <= 4]
        out = {"input": ctx.input, "events": CENSUS_EVENTS, "builds": sorted(builds), "perturbed": perturb,
               "findable_count_perturbed": perturb_findable, "tie_perturbed": perturb_tie,
               "n_tracks": 0, "hits_off_build": 0, "seed_or_follow_off_build": 0, "dr_groups_spanning_builds": 0,
               "n_own_gt_nhit": 0, "owner_sentinel_inconsistent": 0, "owner_tie_inconsistent": 0, "n_tied": 0,
               "n_owned_hits": 0,
               "n_census_tps": 0, "n_findable_tps": 0}
        for j, c in enumerate(sorted(glob.glob(os.path.join(d, "chunk_*.npz")))):
            z = np.load(c)
            out["n_census_tps"] += len(z["cen_key"])
            tpt = {k: z[f"cen_{k}"].copy() for k in ("n_mod_it", "n_mod_ot", "charge")}
            if perturb_findable and j == 0:      # one charged TP on 3 modules gets a 4th
                three = np.flatnonzero((tpt["n_mod_it"] + tpt["n_mod_ot"] == 3) & (tpt["charge"] != 0))
                tpt["n_mod_ot"][three[0]] += 1
            out["n_findable_tps"] += int(TF.findable(tpt).sum())
        if perturb:
            dr = [dr[0] + dr[1][:1], dr[1][1:]] + dr[2:]
        for g in dr:
            if sum(1 for es in builds.values() if set(g) <= set(es)) != 1:
                out["dr_groups_spanning_builds"] += 1
        chunks = sorted(glob.glob(os.path.join(d, "chunk_*.npz")))
        first_bad_done = tie_done = False
        for b, es in builds.items():
            active = set(SPN.mask_surfaces(b)) & {1, 2, 3, 4}
            for e in es:
                surf = set(seeds[e]) | set(follow[e])
                out["seed_or_follow_off_build"] += len({s for s in surf if s <= 4} - active)
                for ch in chunks:
                    z = np.load(ch)
                    if f"hit_{e}" not in z.files:
                        continue
                    h = z[f"hit_{e}"].copy()
                    t = z[f"trk_{e}"].copy()
                    if perturb and not first_bad_done and b == "AAAI" and len(h):
                        h[0, KF.LAYER_ORDER.index(4)] = int(h[0].max())      # a hit moved onto IL4
                        t[0, TC.index("n_own")] = t[0, TC.index("nhit")] + 1
                        first_bad_done = True
                    tied = t[:, TC.index("n_tp_top")] >= 2
                    if perturb_tie and not tie_done and tied.any():        # index tie-breaking: an owner on a tie
                        i = int(np.flatnonzero(tied)[0])
                        t[i, TC.index("own_tpidx")] = 0
                        t[i, TC.index("n_own")] = 1
                        tie_done = True
                    out["n_tracks"] += len(h)
                    for i in it_slots:
                        if KF.LAYER_ORDER[i] not in active:
                            out["hits_off_build"] += int((h[:, i] >= 0).sum())
                    n_own, nhit, own = t[:, TC.index("n_own")], t[:, TC.index("nhit")], t[:, TC.index("own_tpidx")]
                    out["n_own_gt_nhit"] += int(np.sum(n_own > nhit))
                    out["n_owned_hits"] += int(n_own.sum())
                    out["owner_sentinel_inconsistent"] += int(np.sum((own == -1) != (n_own == 0)))
                    n_top = t[:, TC.index("n_tp_top")]
                    out["owner_tie_inconsistent"] += int(np.sum((own == -1) != (n_top != 1)))
                    out["n_tied"] += int(np.sum(n_top >= 2))
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return {"input": ctx.input, "events": CENSUS_EVENTS,
                "variants": {"structure": self._stat(ctx, perturb=True),
                             "findable_count": self._stat(ctx, perturb_findable=True),
                             "tie_break": self._stat(ctx, perturb_tie=True)}}

    def judge(self, stat, ref):
        if "variants" in stat:
            # a known-bad of several variants is red only when EVERY variant is red
            res = {k: self.judge(v, ref) for k, v in stat["variants"].items()}
            ok = any(r[0] for r in res.values())
            return ok, "; ".join(f"{k}: {'GREEN' if r[0] else 'red'} ({r[1]})" for k, r in res.items())
        why = [f"{k} = {stat[k]}" for k in ("hits_off_build", "seed_or_follow_off_build", "dr_groups_spanning_builds",
                                            "n_own_gt_nhit", "owner_sentinel_inconsistent",
                                            "owner_tie_inconsistent") if stat[k]]
        if stat["n_tracks"] == 0:
            why.append("no exported tracks")
        if same_input(stat, ref) and stat["perturbed"] == ref["stat"]["perturbed"]:
            why += identity(stat, ref, ["n_tracks", "n_census_tps", "n_findable_tps", "n_tied", "n_owned_hits"])[1]
        return not why, (f"{stat['n_tracks']} census tracks over {stat['builds']} clean "
                         f"({stat['n_census_tps']} TPs, {stat['n_findable_tps']} findable, {stat['n_tied']} tied tracks, "
                         f"{stat['n_owned_hits']} owned hits)") if not why else "; ".join(why)


# ---------------------------------------------------------------- input contract
# CMSSW constants of the variance, written out here independently of kf_emulation: the tracklet
# Settings' stripPitch / stripLength per module type (PS, 2S) and TMTT's approxB defaults
# (Settings.cc:22-23), KalmanMultiScattTerm.
PITCH = {True: 0.01, False: 0.009}
LENGTH = {True: 0.1467, False: 5.025}
B_APPROX = (0.886454, 0.504148)
MULT_SCATT = 0.00075
VAR_RTOL = 1e-12


class CensusInputContract(Item):
    id = "census.input_contract"
    title = ("the census reads the OT module flags (isPS, isTilted), the stub's cluster owners and the cluster and "
             "stub module ids (detId) from its input and refuses an input without them, or with truth marked "
             "unavailable")
    c4_row = "(new, WP-D0a) the census input contract: module flags read, never inferred (canon_v2 refused)"
    known_bad_desc = ("five copies of the gate input's OT stub and IT cluster tables written to scratch under NanoAOD "
                      "names, each of which the census must refuse: (a) without L1TOTStub_isPS / isTilted (as "
                      "canon_v2); (b) without L1TOTStub_tpIdxInner / tpIdxOuter; (c) with one stub's tpIdxInner set to "
                      "-2 (truth unavailable); (d) without L1TOTStub_detId; (e) without L1TSmartPixelsCluster_detId. "
                      "The same copy with every column and no -2 is accepted, so each change alone causes its refusal")
    rule_desc = ("tp_findability.require_census_inputs accepts the input; on every OT barrel stub of the first 5 events "
                 "isPS == (layer <= 3) and isTilted only on PS modules (D121); KF.ot_variances with the stub's flags "
                 "equals KFParamsComb::matrixV written out independently (pitch and length by isPS, z variance x "
                 "approxB^2 exactly on tilted modules) to 1e-12 relative; identity of the stub counts with the reference")
    trees = (CENSUS_DIR,)

    def _stat(self, ctx):
        import awkward as ak

        TF = ctx.script_module(CENSUS_DIR, "tp_findability")
        KF = ctx.script_module(CENSUS_DIR, "kf_emulation")
        out = {"input": ctx.input, "events": ctx.events}
        try:
            TF.require_census_inputs([ctx.input])
            out["accepted"] = True
        except SystemExit as e:
            out.update(accepted=False, refusal=str(e).replace(ctx.input, "<input>")[:300])
            return out
        A = ctx.arrays([f"L1TOTStub_{c}" for c in ("layer", "isBarrel", "isPS", "isTilted", "r", "z")])
        f = {c: _np(ak.flatten(A[f"L1TOTStub_{c}"])) for c in ("layer", "isBarrel", "isPS", "isTilted", "r", "z")}
        b = f["isBarrel"].astype(bool)
        lay, ps, ti = f["layer"][b], f["isPS"][b].astype(bool), f["isTilted"][b].astype(bool)
        r, z = f["r"][b].astype(np.float64), f["z"][b].astype(np.float64)
        pt = np.full(len(r), 2.0)
        vphi, vz = KF.ot_variances(ps, ti, r, pt, z)
        sp = np.where(ps, PITCH[True], PITCH[False]) / np.sqrt(12.0)
        sl = np.where(ps, LENGTH[True], LENGTH[False]) / np.sqrt(12.0)
        ab = B_APPROX[0] * np.abs(z) / r + B_APPROX[1]
        vz_ref = sl * sl * np.where(ti, ab * ab, 1.0)
        vphi_ref = (sp / r) ** 2 + (MULT_SCATT / pt) ** 2
        out.update(n_barrel_stubs=int(b.sum()), n_ps=int(ps.sum()), n_tilted=int(ti.sum()),
                   n_ps_not_layer_1_3=int(np.sum(ps != (lay <= 3))), n_tilted_not_ps=int(np.sum(ti & ~ps)),
                   n_variance_mismatch=int(np.sum((np.abs(vz - vz_ref) > VAR_RTOL * vz_ref)
                                                  | (np.abs(vphi - vphi_ref) > VAR_RTOL * vphi_ref))),
                   # context, not judged: stubs whose z variance the format-26 (r, z) rule got wrong
                   n_flat_scaled_by_format26_rule=int(np.sum(ps & ~ti & (ab > 1.0))))
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    @staticmethod
    def _input_copy(ctx, drop, tag, unavailable=None) -> str:
        """The gate input's OT stub, IT cluster and DTC frame tables written to scratch under
        NanoAOD branch names (nL1TOTStub, L1TOTStub_<column>, nL1TSmartPixelsCluster, ...),
        without the branches in `drop` (a table left without columns is not written), and with
        the first stub's column `unavailable` set to -2 (truth unavailable) when given."""
        import awkward as ak
        import uproot

        data = {}
        for table in ("L1TOTStub", "L1TSmartPixelsCluster", "L1TOTStubDTC"):
            cols = [k for k in ctx.keys() if k.startswith(f"{table}_") and k not in drop]
            if not cols:
                continue
            A = ctx.arrays(cols)
            fields = {c[len(table) + 1:]: A[c] for c in cols}
            if unavailable and table == "L1TOTStub":
                x = A[f"L1TOTStub_{unavailable}"]
                flat = _np(ak.flatten(x)).copy()
                flat[0] = -2
                fields[unavailable] = ak.unflatten(flat, ak.num(x))
            data[table] = ak.zip(fields)
        path = os.path.join(tempfile.mkdtemp(prefix=f"spcheck_{tag}_", dir=ctx.scratch), f"{tag}.root")
        with uproot.recreate(path) as f:
            f.mktree("Events", {k: v.type for k, v in data.items()}, counter_name=lambda c: "n" + c,
                     field_name=lambda outer, inner: f"{outer}_{inner}")
            f["Events"].extend(data)
        return path

    def known_bad(self, ctx):
        from .core import Ctx

        TF = ctx.script_module(CENSUS_DIR, "tp_findability")
        try:                                     # the copy with every column must pass: only the change may fail
            TF.require_census_inputs([self._input_copy(ctx, (), "control")])
        except SystemExit as e:
            raise RuntimeError(f"the known-bad's writer is broken: the control copy is refused: {e}") from None
        dtc = tuple(k for k in ctx.keys() if k.startswith("L1TOTStubDTC_") or k == "nL1TOTStubDTC")
        cases = {"noflags": dict(drop=("L1TOTStub_isPS", "L1TOTStub_isTilted")),
                 "noowners": dict(drop=("L1TOTStub_tpIdxInner", "L1TOTStub_tpIdxOuter")),
                 "unavailable": dict(drop=(), unavailable="tpIdxInner"),
                 "nostubdetid": dict(drop=("L1TOTStub_detId",)),
                 "noclusterdetid": dict(drop=("L1TSmartPixelsCluster_detId",)),
                 # census format 35: the DTC frames (canon_v3 has none) and the stub positions
                 "nodtc": dict(drop=dtc),
                 "nostubxy": dict(drop=("L1TOTStub_x", "L1TOTStub_y"))}
        refusals = []
        for tag, kw in cases.items():
            bad = Ctx(input_path=self._input_copy(ctx, tag=tag, **kw), events=ctx.events,
                      code_root=ctx.code_root, scratch=ctx.scratch)
            st = self._stat(bad)
            if st["accepted"]:                   # one accepted copy: the known-bad is not red
                return {**st, "known_bad_case": tag}
            refusals.append(f"{tag}: {st['refusal'][:120]}")
        return {"input": f"{len(cases)} known-bad copies", "events": ctx.events, "accepted": False,
                "refusal": " | ".join(refusals)}

    def judge(self, stat, ref):
        if not stat["accepted"]:
            return False, f"input refused: {stat['refusal']}"
        why = [f"{k} = {stat[k]}" for k in ("n_ps_not_layer_1_3", "n_tilted_not_ps", "n_variance_mismatch") if stat[k]]
        if not stat["n_tilted"]:
            why.append("no tilted stub read")
        if same_input(stat, ref):
            why += identity(stat, ref, ["n_barrel_stubs", "n_ps", "n_tilted"])[1]
        return not why, (f"{stat['n_barrel_stubs']} barrel stubs ({stat['n_tilted']} tilted): flags consistent, "
                         f"variances = matrixV") if not why else "; ".join(why)


# ---------------------------------------------------------------- the OT arms (WP-D, census format 35)
OT_TARGETS = "OL1,OL2,OL3,OL4,OL5,OL6"


def _census_run(ctx, tag, args):
    """A CENSUS_EVENTS-event census of the gate input run from this code root with `args` (cached per tag)."""
    key = ("census_run", ctx.input, tag)
    if key in ctx.cache:
        return ctx.cache[key]
    root = ctx.tree(CENSUS_DIR)
    out = tempfile.mkdtemp(prefix=f"spcheck_{tag}_", dir=ctx.scratch)
    cmd = [sys.executable, "-u", str(root / "tp_findability.py"), "-i", ctx.input, "-n", str(CENSUS_EVENTS),
           "--chunk", "2", "--calib-events", str(CENSUS_EVENTS), "--export-tracks", "--cache-dir", out, *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ctx.code_root), timeout=CENSUS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"the {tag} census exceeded {CENSUS_TIMEOUT_S} s") from None
    if r.returncode != 0:
        raise RuntimeError(f"{tag} census failed rc={r.returncode}: {r.stderr[-2000:]}")
    d = glob.glob(os.path.join(out, "tpcensus_*"))[0]
    ctx.cache[key] = d
    return d


def _fingerprint(d, entries=None):
    """(sha256 of the census's outputs, tracks, TPs found) for `entries` (None: all): the found and found_dr
    bitmap columns, the counters and every exported track record, chunk by chunk."""
    import hashlib

    man = json.load(open(os.path.join(d, "manifest.json")))
    es = list(range(len(man["seeds"]))) if entries is None else list(entries)
    h = hashlib.sha256()
    n_tracks = n_found = 0
    for c in sorted(glob.glob(os.path.join(d, "chunk_*.npz"))):
        z = np.load(c)
        cnt = json.loads(str(z["counters"]))
        for e in es:
            col = lambda b: ((b[:, e >> 6] >> np.uint64(e & 63)) & np.uint64(1)).astype(np.uint8)
            f, fd = col(z["found"]), col(z["found_dr"])
            n_found += int(f.sum())
            h.update(f.tobytes()); h.update(fd.tobytes())
            h.update(json.dumps(cnt.get(str(e), {}), sort_keys=True).encode())
            for k in (f"hit_r{e}", f"hit_rm{e}"):
                if k in z.files:
                    h.update(z[k].tobytes())
                    n_tracks += len(z[k]) if k == f"hit_r{e}" else 0
    return h.hexdigest(), n_tracks, n_found


class OtAngleInert(Item):
    id = "census.ot_angle_inert"
    title = "the closed OT arm is bit-identical with every angle stage off and every stage on at k = 0.5"
    c4_row = "WP-D plan sec. 6.1 (the angle stages inert in the closure arm)"
    known_bad_desc = ("the same comparison with the second run's match order randomised (--ot-order random:1): a "
                      "change of the arm's output, which the identity must see")
    rule_desc = ("two fresh 4-event closed-arm censuses of the gate input (--ot-chain closed, --masks IIII, OT "
                 "targets), one with --angle-stages none and one with every stage on, cuts open (--calibrate) and "
                 "k = 0.5 for both angles: their found and found_dr bitmaps, counters and track records are "
                 "identical; identity of the track and found counts with the reference")
    trees = (CENSUS_DIR,)
    BASE = ["--ot-chain", "closed", "--masks", "IIII", "--targets", OT_TARGETS]

    def _stat(self, ctx, bad=False):
        a = _census_run(ctx, "ot_inert_off", self.BASE + ["--angle-stages", "none"])
        extra = ["--ot-order", "random:1"] if bad else []
        b = _census_run(ctx, "ot_inert_on" + ("_bad" if bad else ""),
                        self.BASE + ["--angle-stages", "all", "--calibrate", "--k-alpha", "0.5", "--k-beta", "0.5"]
                        + extra)
        fa, nt, nf = _fingerprint(a)
        fb, _, _ = _fingerprint(b)
        return {"input": ctx.input, "events": CENSUS_EVENTS, "identical": fa == fb, "n_tracks": nt, "n_found": nf}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, bad=True)

    def judge(self, stat, ref):
        why = [] if stat["identical"] else ["the angle-on run differs from the angle-off run"]
        if not stat["n_tracks"]:
            why.append("no closed-arm tracks")
        if same_input(stat, ref):
            why += identity(stat, ref, ["n_tracks", "n_found"])[1]
        return not why, (f"{stat['n_tracks']} tracks, {stat['n_found']} TP-seed founds, identical with the angles on"
                         if not why else "; ".join(why))


class SpItOffIsOt(Item):
    id = "census.sp_it_off_is_ot"
    title = "an SP build with every IT surface off equals the OT-only arm (census chain) bit for bit"
    c4_row = "WP-D plan sec. 6.2 (the T5 reference until WP-D2: the OT-only arm with D3-D5 off)"
    known_bad_desc = "the same comparison with IL1 left on (--masks AIII): the OT seeds then follow into IL1"
    rule_desc = ("two fresh 4-event censuses of the gate input with the census chain, cuts open (--calibrate): "
                 "--masks IIII with the default targets (every surface) and --masks IIII with the OT targets only: "
                 "the prompt OT doublets' found and found_dr bitmaps, counters and track records are identical; "
                 "identity of the track and found counts with the reference")
    trees = (CENSUS_DIR,)

    def _ot_entries(self, d):
        man = json.load(open(os.path.join(d, "manifest.json")))
        ot = [[11, 12], [12, 13], [13, 14], [15, 16]]
        return [i for i, sd in enumerate(man["seeds"]) if list(sd) in ot]

    def _stat(self, ctx, bad=False):
        a = _census_run(ctx, "sp_it_off" + ("_bad" if bad else ""),
                        ["--masks", "AIII" if bad else "IIII", "--calibrate"])
        b = _census_run(ctx, "ot_only", ["--masks", "IIII", "--targets", OT_TARGETS, "--calibrate"])
        fa, nt, nf = _fingerprint(a, self._ot_entries(a))
        fb, _, _ = _fingerprint(b, self._ot_entries(b))
        return {"input": ctx.input, "events": CENSUS_EVENTS, "identical": fa == fb, "n_tracks": nt, "n_found": nf}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, bad=True)

    def judge(self, stat, ref):
        why = [] if stat["identical"] else ["the IT-off SP build differs from the OT-only arm"]
        if not stat["n_tracks"]:
            why.append("no OT doublet tracks")
        if same_input(stat, ref):
            why += identity(stat, ref, ["n_tracks", "n_found"])[1]
        return not why, (f"{stat['n_tracks']} OT doublet tracks, {stat['n_found']} TP-seed founds, identical"
                         if not why else "; ".join(why))


ITEMS = [MajorityOwner(), CensusBuilds(), CensusInputContract(), OtAngleInert(), SpItOffIsOt()]
