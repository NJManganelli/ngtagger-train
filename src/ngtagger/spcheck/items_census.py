"""Fast-gate items on the census (plan App. C.4 rows: int32 owner sentinel; per-build
follow and DR over all builds). They import the census from THIS code root's
eval_refitq/combinatorics (the frozen copy under the launcher) and run a 4-event, two-build
census (AAAI, IAAA: the builds with an inactive layer) of the gate input into a scratch
cache: ~70 s, ~1.2 GB."""
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
    title = "the census majority owner on real track hit tables: n_own <= real hits, owner on the track, dtypes"
    c4_row = "int32 owner sentinel"
    known_bad_desc = ("an int32 sentinel injected: empty layer slots handed to the census as valid hits carrying "
                      "np.iinfo(int32).min (the missing-mask/sentinel class of the 813be3e bug)")
    rule_desc = ("KF.majority_owner on the OT tracks of the gate input (stubs from L1TTrackStub -> L1TOTStub_tpIdx, "
                 "SP-refit IT clusters from L1TSmartPixelsTrackCluster, empty slots invalid) equals a plain counting "
                 "reference on the real hits for every track; n_own <= real hits; owner -1 or a TP on the track; "
                 "input tpIdx int32, output int64; identity of the counts with the reference")
    trees = (CENSUS_DIR,)

    def _tables(self, ctx):
        """(KF module, tp, valid, dtype of tpIdx as read) for the OT tracks of the gate input."""
        KF = ctx.script_module(CENSUS_DIR, "kf_emulation")
        A = ctx.arrays(["L1TTrackStub_trackIdx", "L1TTrackStub_otStubIdx", "L1TOTStub_tpIdx", "L1TOTStub_layer",
                        "L1TOTStub_isBarrel", "nL1TTrack", "L1TSmartPixelsTrack_l1TrackIdx",
                        "L1TSmartPixelsTrackCluster_trackIdx", "L1TSmartPixelsTrackCluster_clusterIdx",
                        "L1TSmartPixelsCluster_tpIdx", "L1TSmartPixelsCluster_surface"])
        slot = {L: i for i, L in enumerate(KF.LAYER_ORDER)}
        tps, valids = [], []
        import awkward as ak

        dtype_in = sorted({str(ak.to_numpy(ak.flatten(A[c])).dtype)
                           for c in ("L1TOTStub_tpIdx", "L1TSmartPixelsCluster_tpIdx")})
        rng = np.random.default_rng(7)
        for ev in range(len(A)):
            nt = int(A["nL1TTrack"][ev])
            tp = rng.integers(-5, 50, size=(nt, len(KF.LAYER_ORDER))).astype(np.int32)  # junk in empty slots
            va = np.zeros((nt, len(KF.LAYER_ORDER)), bool)
            ti = _np(A["L1TTrackStub_trackIdx"][ev])
            si = _np(A["L1TTrackStub_otStubIdx"][ev])
            lay = _np(A["L1TOTStub_layer"][ev])[si]
            brl = _np(A["L1TOTStub_isBarrel"][ev])[si].astype(bool)
            stp = _np(A["L1TOTStub_tpIdx"][ev])[si]
            for t, L, b, v in zip(ti, lay, brl, stp):
                if b and (10 + int(L)) in slot:
                    tp[t, slot[10 + int(L)]] = v
                    va[t, slot[10 + int(L)]] = True
            l1 = _np(A["L1TSmartPixelsTrack_l1TrackIdx"][ev])
            cti = _np(A["L1TSmartPixelsTrackCluster_trackIdx"][ev])
            cci = _np(A["L1TSmartPixelsTrackCluster_clusterIdx"][ev])
            ctp = _np(A["L1TSmartPixelsCluster_tpIdx"][ev])
            csf = _np(A["L1TSmartPixelsCluster_surface"][ev])
            for t, c in zip(cti, cci):
                s = int(csf[c])
                if s in (1, 2, 3, 4):
                    tp[l1[t], slot[s]] = ctp[c]
                    va[l1[t], slot[s]] = True
            tps.append(tp)
            valids.append(va)
        tp, va = np.vstack(tps), np.vstack(valids)
        return KF, tp, va, dtype_in

    @staticmethod
    def reference(tp, va, matched):
        own, n = np.full(len(tp), -1, np.int64), np.zeros(len(tp), np.int64)
        for i in range(len(tp)):
            c = Counter(int(x) for x, ok in zip(tp[i], va[i] & matched[i]) if ok)
            if c:
                best = max(c.values())
                own[i], n[i] = max(k for k, v in c.items() if v == best), best   # ties: largest index
        return own, n

    def _stat(self, ctx, inject=False):
        from ngtagger.data import spix_nano as SPN

        KF, tp, va, dtype_in = self._tables(ctx)
        real = va & SPN.tp_matched(tp)
        ref_own, ref_n = self.reference(tp, va, SPN.tp_matched(tp))
        if inject:
            tp = np.where(va, tp, np.iinfo(np.int32).min).astype(np.int32)
            va = np.ones_like(va)
        own, n = KF.majority_owner(tp, va)
        on_track = np.array([o == -1 or bool(np.any(real[i] & (tp[i] == o))) for i, o in enumerate(own)])
        return {"input": ctx.input, "events": ctx.events, "sentinel_injected": inject, "n_tracks": int(len(own)),
                "n_with_owner": int((ref_own != -1).sum()), "n_agree_with_reference": int(np.sum((own == ref_own) & (n == ref_n))),
                "n_own_gt_real_hits": int(np.sum(n > real.sum(1))), "n_owner_not_on_track": int(np.sum(~on_track)),
                "tpIdx_dtype_read": dtype_in, "tpIdx_dtype_census_input": str(tp.dtype),
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
            why += identity(stat, ref, ["n_tracks", "n_with_owner"])[1]
        return not why, f"{stat['n_tracks']} tracks, owners identical to the reference" if not why else "; ".join(why)


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


ITEMS = [MajorityOwner(), CensusBuilds()]
