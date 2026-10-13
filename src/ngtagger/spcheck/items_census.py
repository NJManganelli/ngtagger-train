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
    known_bad_desc = ("an int32 sentinel injected: empty layer slots handed to the census as valid hits carrying "
                      "np.iinfo(int32).min (the missing-mask/sentinel class of the 813be3e bug)")
    rule_desc = ("KF.credited_owner on the OT tracks of the gate input (stubs from L1TTrackStub -> L1TOTStub, each "
                 "credited to its tpIdx, tpIdxInner and tpIdxOuter through M.credited_tps; SP-refit IT clusters from "
                 "L1TSmartPixelsTrackCluster, credited to their tpIdx; empty slots invalid) equals a plain counting "
                 "reference (each real hit counts once for every distinct matched TP among its raw truth columns) on "
                 "every track; n_own <= real hits; owner -1 or a TP credited by a hit on the track; input tpIdx "
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
    def reference(raw, va, matched):
        """Plain counting: every valid hit counts once for each distinct matched TP among its raw columns; the most
        counts wins, ties to the largest index."""
        own, n = np.full(len(raw), -1, np.int64), np.zeros(len(raw), np.int64)
        for i in range(len(raw)):
            c = Counter(int(x) for h in range(raw.shape[1]) if va[i, h]
                        for x in {int(y) for y, m in zip(raw[i, h], matched[i, h]) if m})
            if c:
                best = max(c.values())
                own[i], n[i] = max(k for k, v in c.items() if v == best), best   # ties: largest index
        return own, n

    def _stat(self, ctx, inject=False):
        from ngtagger.data import spix_nano as SPN

        KF, cred, raw, va, dtype_in = self._tables(ctx)
        real = va & SPN.tp_matched(raw).any(axis=2)
        ref_own, ref_n = self.reference(raw, va, SPN.tp_matched(raw))
        # context: the owner with each stub credited to its own tpIdx alone (format <= 27)
        own27, _ = KF.majority_owner(raw[:, :, 0], va)
        if inject:
            cred = np.where(va[:, :, None], cred, np.iinfo(np.int32).min).astype(np.int32)
            va = np.ones_like(va)
        own, n, _, _ = KF.credited_owner(cred, va)
        on_track = np.array([o == -1 or bool(np.any(real[i] & (raw[i] == o).any(axis=1))) for i, o in enumerate(own)])
        return {"input": ctx.input, "events": ctx.events, "sentinel_injected": inject, "n_tracks": int(len(own)),
                "n_with_owner": int((ref_own != -1).sum()), "n_agree_with_reference": int(np.sum((own == ref_own) & (n == ref_n))),
                "n_owner_changed_by_cluster_credit": int(np.sum(ref_own != own27)),
                "n_own_gt_real_hits": int(np.sum(n > real.sum(1))), "n_owner_not_on_track": int(np.sum(~on_track)),
                "tpIdx_dtype_read": dtype_in, "tpIdx_dtype_census_input": str(cred.dtype),
                "owner_dtype": str(np.asarray(own).dtype), "count_dtype": str(np.asarray(n).dtype)}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, inject=True)

    def judge(self, stat, ref):
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
            why += identity(stat, ref, ["n_tracks", "n_with_owner", "n_owner_changed_by_cluster_credit"])[1]
        return not why, (f"{stat['n_tracks']} tracks, owners identical to the reference "
                         f"({stat['n_owner_changed_by_cluster_credit']} changed by the cluster credit)") \
            if not why else "; ".join(why)


# ---------------------------------------------------------------- census output
class CensusBuilds(Item):
    id = "census.builds_and_dr"
    title = "census output: no hit outside its build's surfaces, DR groups within one build, owner columns sane"
    c4_row = "per-build follow; DR over all builds (+ int32 owner sentinel on the output)"
    known_bad_desc = ("the same census output with one AAAI track's hit moved onto IL4 (another build's surface), "
                      "one DR group spanning both builds, and one n_own set beyond nhit")
    rule_desc = ("on a fresh 4-event AAAI+IAAA census of the gate input run from this code root: every exported "
                 "hit lies on its build's active IT surfaces or the OT; every seed and follow surface is active in "
                 "its build; every DR group lies inside one build; n_own <= nhit and own_tpidx = -1 iff n_own = 0; "
                 "identity of the counts with the reference")
    trees = (CENSUS_DIR,)

    def _run(self, ctx) -> str:
        key = ("census_cache", ctx.input)
        if key in ctx.cache:
            return ctx.cache[key]
        root = ctx.tree(CENSUS_DIR)
        out = tempfile.mkdtemp(prefix="spcheck_census_", dir=ctx.scratch)
        cmd = [sys.executable, "-u", str(root / "tp_findability.py"), "-i", ctx.input, "-n", str(CENSUS_EVENTS),
               "--chunk", "2", "--calib-events", str(CENSUS_EVENTS), "--export-tracks", "--it-ptmin", "1.0",
               "--min-layers", "4", "--min-it-layers", "3", "--masks", CENSUS_BUILDS, "--cache-dir", out,
               "-o", os.path.join(out, "census.json")]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ctx.code_root))
        if r.returncode != 0:
            raise RuntimeError(f"census failed rc={r.returncode}: {r.stderr[-2000:]}")
        d = glob.glob(os.path.join(out, "tpcensus_*"))[0]
        ctx.cache[key] = d
        return d

    def _stat(self, ctx, perturb=False):
        from ngtagger.data import spix_nano as SPN

        KF = ctx.script_module(CENSUS_DIR, "kf_emulation")
        d = self._run(ctx)
        man = json.load(open(os.path.join(d, "manifest.json")))
        builds = {b: [int(e) for e in es] for b, es in man["builds"].items()}
        dr = [[int(e) for e in g] for g in man["dr_groups"]]
        seeds = man["seeds"] if isinstance(man["seeds"], list) else json.loads(man["seeds"])
        follow = man["seed_follow"] if isinstance(man["seed_follow"], list) else json.loads(man["seed_follow"])
        TC = list(KF.TRACK_COLS)
        it_slots = [i for i, L in enumerate(KF.LAYER_ORDER) if L <= 4]
        out = {"input": ctx.input, "events": CENSUS_EVENTS, "builds": sorted(builds), "perturbed": perturb,
               "n_tracks": 0, "hits_off_build": 0, "seed_or_follow_off_build": 0, "dr_groups_spanning_builds": 0,
               "n_own_gt_nhit": 0, "owner_sentinel_inconsistent": 0}
        if perturb:
            dr = [dr[0] + dr[1][:1], dr[1][1:]] + dr[2:]
        for g in dr:
            if sum(1 for es in builds.values() if set(g) <= set(es)) != 1:
                out["dr_groups_spanning_builds"] += 1
        chunks = sorted(glob.glob(os.path.join(d, "chunk_*.npz")))
        first_bad_done = False
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
                    out["n_tracks"] += len(h)
                    for i in it_slots:
                        if KF.LAYER_ORDER[i] not in active:
                            out["hits_off_build"] += int((h[:, i] >= 0).sum())
                    n_own, nhit, own = t[:, TC.index("n_own")], t[:, TC.index("nhit")], t[:, TC.index("own_tpidx")]
                    out["n_own_gt_nhit"] += int(np.sum(n_own > nhit))
                    out["owner_sentinel_inconsistent"] += int(np.sum((own == -1) != (n_own == 0)))
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, perturb=True)

    def judge(self, stat, ref):
        why = [f"{k} = {stat[k]}" for k in ("hits_off_build", "seed_or_follow_off_build", "dr_groups_spanning_builds",
                                            "n_own_gt_nhit", "owner_sentinel_inconsistent") if stat[k]]
        if stat["n_tracks"] == 0:
            why.append("no exported tracks")
        if same_input(stat, ref) and stat["perturbed"] == ref["stat"]["perturbed"]:
            why += identity(stat, ref, ["n_tracks"])[1]
        return not why, f"{stat['n_tracks']} census tracks over {stat['builds']} clean" if not why else "; ".join(why)


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
    title = "the census reads the OT module flags (isPS, isTilted) from its input and refuses an input without them"
    c4_row = "(new, WP-D0a) the census input contract: module flags read, never inferred (canon_v2 refused)"
    known_bad_desc = ("the gate input's OT stub table without L1TOTStub_isPS / isTilted (as canon_v2), written to "
                      "scratch: the census must refuse it")
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
            out.update(accepted=False, refusal=str(e)[:300])
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

    def known_bad(self, ctx):
        import awkward as ak
        import uproot

        from .core import Ctx

        cols = [k for k in ctx.keys() if k.startswith("L1TOTStub_") and k not in
                ("L1TOTStub_isPS", "L1TOTStub_isTilted")]
        A = ctx.arrays(cols)
        path = os.path.join(tempfile.mkdtemp(prefix="spcheck_noflags_", dir=ctx.scratch), "noflags.root")
        with uproot.recreate(path) as f:
            f["Events"] = {"L1TOTStub": ak.zip({c[len("L1TOTStub_"):]: A[c] for c in cols})}
        bad = Ctx(input_path=path, events=ctx.events, code_root=ctx.code_root, scratch=ctx.scratch)
        return self._stat(bad)

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


ITEMS = [MajorityOwner(), CensusBuilds(), CensusInputContract()]
