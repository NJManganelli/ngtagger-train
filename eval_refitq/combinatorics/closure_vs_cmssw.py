"""closure_vs_cmssw: the census OT chain closed against CMSSW, stage by stage (WP-D; wpd/PLAN.md sec. 2-4).

  bridge  C0  canon_v4 against canon_v3, branch by branch: equal, or in the declared lists (BRIDGE_*)
  twice   C0  run-twice determinism: two productions of one configuration are bit identical
  l2l3    C0  L2L3 tracks in KF regions 0-3: none in CMSSW and none in the closed arm
  c1          the KF unit: cmssw_kf forced onto every CMSSW no-DR track's own stubs (barrel seeds,
              barrel-only tracks, KF regions 0-3), quantised as HybridFit / Track
  c1b         the match lists against CMSSW's track-builder candidates, from a no-DR production with
              Fakefit (every candidate with its full stub list and its tracklet's integer parameters):
              (i) per tracklet, the emulated candidate against CMSSW's, and its integer parameters;
              (ii) the KF on CMSSW's own candidate list against CMSSW's no-DR track
  c23         C2 per candidate (no DR, per seed) and C3 delivered (merge DR, pooled): a closed-arm census
              against CMSSW, with the paired event bootstrap and equivalence at the plan's margins
  floor       the order floor: D_order and Delta_dtc from one dtc and ten random closed-arm censuses

Every stage prints its summary, writes its JSON report (-o) and exits 2 when one of its gates fails.

THE TRUTH IS THE CENSUS'S on both sides (plan sec. 2), from the same per-track records (tp_findability
RECORD_COLS): a track's TP is the single TP credited by both of its seed stubs (tracklet_topology_cost.seed_tp;
an OT stub credits its tpIdx and its clusters' owners; none on a tie) and it is a fake when it has none.
Tracks are judged in KF eta regions 0-3 (the seed's z0 + 50 t inside |eta| 0.9, ot_chain.kf_region), barrel
seeds only (L1L2, L2L3, L3L4, L5L6), and a TP is found by a side when one of its judged tracks has it. The
efficiency denominator is the census's: findable (>= 4 credited modules, charged), |eta| < 0.9, pT >= 2 GeV.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmssw_kf as KF                 # noqa: E402
import ot_chain as OC                 # noqa: E402
import ot_dtc as D                    # noqa: E402
import seed_arity as SA               # noqa: E402
import tp_findability as TF           # noqa: E402
import tracklet_topology_cost as M    # noqa: E402

FAIL_EXIT = 2
N_BOOT, BOOT_SEED = 400, 20261015
JUDGED_SEEDS = (0, 2, 3)                       # L1L2, L3L4, L5L6 (L2L3 is a C0 identity)
SEED_LAYERS = {0: (1, 2), 1: (2, 3), 2: (3, 4), 3: (5, 6)}
MAX_KF_REGION = 3
ETA_MAX, PT_MIN = 0.9, 2.0
# the margins (plan sec. 3, "decision arithmetic")
MARGIN = {"d_eff": 0.01, "disc": 0.02, "ln_cand": math.log(1.05), "ln_fake": math.log(1.25), "d_dup": 0.01}
SIGNED = ("d_eff", "ln_cand", "ln_fake", "d_dup")   # equivalence on both sides; disc is one-sided
C1_EDGE_LSB = 1e-6          # a census value this close to a bin edge is a predicted edge, not a failure
C1_CHI2_REL = 1e-5
C1_MAX_EXCLUDED = 0.02
C1B_MATCH_MIN = 0.99
# a PS stub whose nano azimuth is this far (in its layer's DTC phi LSB) from its DTC word is a known
# position-convention exception (the 74-stub class of the canon_v4 precision check), where no stubPosTT is
# written; the DTC word's own interpolation stays within one LSB
FRAME_PHI_TOL_LSB = 1.5

# THE BRIDGE (plan sec. 5): canon_v4 against canon_v3 on the same events. Only these may differ: the disc
# global-direction columns on disc rows (15c345f, the disc sense fix) and the full-float columns, which must
# equal canon_v3 after rounding to canon_v3's stored mantissa; only these may be new.
BRIDGE_DISC = tuple(f"L1TSmartPixelsCluster_{c}" for c in (
    "globalClusterPhi", "globalClusterCotTheta", "sigGlobalClusterPhi", "sigGlobalClusterCotTheta",
    "tpGlobalClusterPhi", "tpGlobalClusterCotTheta"))         # p03_angles/disc_sense_fix.md sec. 3
BRIDGE_ROUNDED = {"L1TOTStub_phi": 12, "L1TSmartPixelsCluster_globalR": 16,      # canon_v3's mantissa bits
                  "L1TSmartPixelsCluster_globalPhi": 16}
BRIDGE_NEW_PREFIX = ("L1TOTStubDTC_", "nL1TOTStubDTC")
BRIDGE_NEW = ("L1TTrack_seedType", "L1TExtTrack_seedType", "L1TOTStub_innerStrip", "L1TOTStub_isFlipped")


def _fail(report, out, ok):
    if out:
        Path(out).write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (list, dict)) or k == "gates"},
                     indent=1, default=float))
    if not ok:
        sys.exit(FAIL_EXIT)


def _events(path, prefixes, nev=None):
    t = uproot.open(f"{path}:Events")
    keys = [k for k in t.keys() if k.startswith(prefixes) or k == "event"]
    return t.arrays(keys, library="ak", entry_stop=nev)


def _table(A, e, prefix):
    n = len(prefix)
    return {k[n:]: ak.to_numpy(A[k][e]) for k in A.fields if k.startswith(prefix)}


# ---- C0 -------------------------------------------------------------------------------------------
def _round_mantissa(x, bits):
    """MiniFloatConverter::reduceMantissaToNbitsRounding (DataFormats/Math/interface/libminifloat.h), the
    NanoAOD storage of a float column at `bits` mantissa bits."""
    shift = 23 - bits
    i = np.asarray(x, np.float32).view(np.uint32).copy()
    up = (i & np.uint32(1 << (shift - 1))) != 0
    mant = (i & np.uint32(0x007FFFFF)) >> np.uint32(shift)
    mant = np.where(up & (mant < np.uint32((1 << bits) - 2)), mant + np.uint32(1), mant)
    i = np.where(up, (i & np.uint32(0xFF800000)) | (mant << np.uint32(shift)),
                 i & np.uint32(((0xFFFFFFFF >> shift) << shift) & 0xFFFFFFFF))
    return i.astype(np.uint32).view(np.float32)


def bridge(v4, v3, nev=None, stub_shift=False):
    """Every branch of canon_v3 equal in canon_v4 except the declared ones; new branches only the declared.
    stub_shift (the bridge's known-bad): canon_v4's L1TOTStub columns moved by one event."""
    t4, t3 = uproot.open(f"{v4}:Events"), uproot.open(f"{v3}:Events")
    k4, k3 = set(t4.keys()), set(t3.keys())
    missing = sorted(k3 - k4)
    new = sorted(k for k in k4 - k3 if not (k.startswith(BRIDGE_NEW_PREFIX) or k in BRIDGE_NEW))
    diff, declared, rounded_bad = [], [], []
    for k in sorted(k3 & k4):
        a = t4[k].array(library="ak", entry_stop=nev)
        b = t3[k].array(library="ak", entry_stop=nev)
        if stub_shift and k.startswith("L1TOTStub_"):
            a = ak.concatenate([a[1:], a[:1]])
        if _same(a, b):
            continue
        if k in BRIDGE_DISC:
            declared.append(k)
        elif k in BRIDGE_ROUNDED:
            fa = ak.to_numpy(ak.flatten(a, axis=None))
            fb = ak.to_numpy(ak.flatten(b, axis=None))
            ok = len(fa) == len(fb) and np.array_equal(_round_mantissa(fa, BRIDGE_ROUNDED[k]),
                                                       fb.astype(np.float32), equal_nan=True)
            (declared if ok else rounded_bad).append(k)
        else:
            diff.append(k)
    ok = not (missing or new or diff or rounded_bad)
    tables = sorted({k.split("_")[0].lstrip("n") for k in diff + rounded_bad})
    return {"stage": "C0 bridge", "v4": v4, "v3": v3, "stub_shift": stub_shift, "branches_common": len(k3 & k4),
            "missing_in_v4": missing, "undeclared_new": new, "undeclared_differences": diff,
            "tables_differing": tables, "declared_differences": declared,
            "rounded_columns_not_equal": rounded_bad, "gates": {"bridge": ok}}, ok


def _same(a, b):
    """Two branches bit identical (NaN equal to NaN): the same events, counts and values."""
    if len(a) != len(b):
        return False
    if a.ndim > 1:
        if b.ndim != a.ndim or not np.array_equal(ak.to_numpy(ak.num(a, axis=1)), ak.to_numpy(ak.num(b, axis=1))):
            return False
        a, b = ak.flatten(a), ak.flatten(b)
    fa, fb = ak.to_numpy(a), ak.to_numpy(b)
    return fa.dtype == fb.dtype and np.array_equal(fa, fb, equal_nan=fa.dtype.kind == "f")


def twice(f1, f2, nev=None):
    """Run-twice determinism: every branch bit identical."""
    t1, t2 = uproot.open(f"{f1}:Events"), uproot.open(f"{f2}:Events")
    k1, k2 = set(t1.keys()), set(t2.keys())
    diff = [k for k in sorted(k1 & k2)
            if not _same(t1[k].array(library="ak", entry_stop=nev), t2[k].array(library="ak", entry_stop=nev))]
    ok = k1 == k2 and not diff
    return {"stage": "C0 run twice", "files": [f1, f2], "branch_sets_equal": k1 == k2, "differences": diff,
            "gates": {"determinism": ok}}, ok


def _cmssw_tracks(path, nev=None, collection="L1TTrack"):
    """The records (tp_findability RECORD_COLS, the entry column holding the seed type) of a CMSSW track
    collection's barrel-seeded tracks, all events. Event index = entry in the file (the census's numbering of
    a one-file input)."""
    A = _events(path, (f"{collection}_", f"{collection}Stub_", "L1TOTStub_"), nev)
    recs = []
    for e in range(len(A)):
        T, S, L = _table(A, e, f"{collection}_"), _table(A, e, f"{collection}Stub_"), _table(A, e, "L1TOTStub_")
        L["event"] = np.full(len(L["layer"]), e, np.int64)
        for j in range(len(T["rInv"])):
            sd = int(T["seedType"][j])
            r = S["otStubIdx"][S["trackIdx"] == j].astype(np.int64)
            if sd not in SEED_LAYERS:
                continue
            by = {int(L["layer"][x]): x for x in r if L["isBarrel"][x]}
            la, lb = SEED_LAYERS[sd]
            if la not in by or lb not in by:
                recs.append((e, sd, -1, 0, -1, len(r), int(T["phiSector"][j])))
                continue
            a, b = by[la], by[lb]
            geo = [float(v) for v in (math.hypot(L["x"][a], L["y"][a]), L["z"][a], math.atan2(L["y"][a], L["x"][a]),
                                      math.hypot(L["x"][b], L["y"][b]), L["z"][b], math.atan2(L["y"][b], L["x"][b]))]
            kreg = OC.kf_region(*KF.exact_tracklet(*geo, 0))
            tp, real = M.seed_tp(L, np.array([a]), np.array([b]))
            key = int(M.tp_key(np.array([e]), np.array([max(int(tp[0]), 0)]))[0]) if tp[0] >= 0 else -1
            recs.append((e, sd, key, int(real[0]), kreg, len(r), int(T["phiSector"][j])))
    return np.array(recs, np.int64).reshape(-1, len(TF.RECORD_COLS))


def _judged(R):
    return R[(R[:, 4] >= 0) & (R[:, 4] <= MAX_KF_REGION)]


def _found(R, key):
    """Boolean over the census TP keys: found by a judged track of the records R."""
    J = _judged(R)
    return np.isin(key, J[J[:, 2] >= 0, 2])


def l2l3(census_dir, cmssw, nev=None):
    """L2L3 tracks in KF regions 0-3 (plan sec. 2): none in CMSSW, none in the closed arm."""
    R = _judged(_cmssw_tracks(cmssw, nev))
    n_cms = int((R[:, 1] == 1).sum())
    C = TF.load(TF.census_dir(census_dir), nev=nev, with_tracks=True, verbose=False)
    n_cen = 0
    for name in ("track_records", "track_records_merged"):
        for i, R in C[name].items():
            if C["seeds"][i].layers == (12, 13):
                n_cen += int(((R[:, 4] >= 0) & (R[:, 4] <= MAX_KF_REGION)).sum())
    ok = n_cms == 0 and n_cen == 0
    return {"stage": "C0 L2L3", "cmssw_tracks_regions_0_3": n_cms, "census_tracks_regions_0_3": n_cen,
            "gates": {"l2l3_zero": ok}}, ok


# ---- C1 -------------------------------------------------------------------------------------------
def _positions(L, Dt):
    """The KF's stub positions: the DTC table's stubPosTT where it was written (withPositions), else the
    nano's x / y / z."""
    x, y, z = (np.asarray(L[c], np.float64).copy() for c in ("x", "y", "z"))
    if "xTT" in Dt and len(Dt["xTT"]):
        r = Dt["otStubIdx"].astype(np.int64)
        x[r], y[r], z[r] = Dt["xTT"], Dt["yTT"], Dt["zTT"]
        return x, y, z, True
    return x, y, z, False


def _frame_exceptions(L, Dt):
    """Rows of PS barrel stubs whose nano azimuth is more than FRAME_PHI_TOL_LSB from their DTC word's."""
    r = Dt["otStubIdx"].astype(np.int64)
    bar, ps = L["isBarrel"][r].astype(bool), L["isPS"][r].astype(bool)
    q = bar & ps
    if not q.any():
        return set()
    r, reg = r[q], Dt["region"][q].astype(np.int64)
    lay = L["layer"][r].astype(np.int64)
    mt = D.module_type(np.ones(len(r), bool), np.ones(len(r), bool))
    _, _, _, iphi, iz, ir, _ = D.decode(Dt["frameHi"][q], Dt["frameLo"][q], mt)
    phi_f, _, _ = D.barrel_physical(lay, iphi, iz, ir, reg)
    kp = np.array([D.kphi(int(v)) for v in lay])
    dphi = np.abs(KF_reduce(np.arctan2(L["y"][r], L["x"][r]) - phi_f)) / kp
    return set(r[dphi > FRAME_PHI_TOL_LSB].tolist())


def KF_reduce(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def _edge(value, lsb):
    v = value / lsb
    return abs(v - round(v)) < C1_EDGE_LSB


def c1(nodr, nev=None):
    """The KF unit (plan sec. 3, C1) on a CMSSW no-DR production."""
    def excl(why):
        res["excluded"][why] = res["excluded"].get(why, 0) + 1

    A = _events(nodr, ("L1TTrack_", "L1TTrackStub_", "L1TOTStub_", "L1TOTStubDTC_"), nev)
    lsb = (KF.K.krinvpars, KF.K.kphi0pars, KF.K.ktpars, KF.K.kz0pars, KF.K.kd0pars)
    res = {"judged": 0, "identical": 0, "predicted_edge": 0, "known_exception": 0, "mismatch": 0,
           "chi2_outside": 0, "state_gate_fails": 0, "not_consistent_sector": 0, "excluded": {}}
    examples = []
    used_tt = False
    for e in range(len(A)):
        T, S = _table(A, e, "L1TTrack_"), _table(A, e, "L1TTrackStub_")
        L, Dt = _table(A, e, "L1TOTStub_"), _table(A, e, "L1TOTStubDTC_")
        x, y, z, tt = _positions(L, Dt)
        used_tt |= tt
        exc = _frame_exceptions(L, Dt) if not tt and "otStubIdx" in Dt else set()
        for j in range(len(T["rInv"])):
            sd = int(T["seedType"][j])
            r = S["otStubIdx"][S["trackIdx"] == j].astype(np.int64)
            if sd not in SEED_LAYERS or not L["isBarrel"][r].all():
                continue                      # disk / overlap seeds, disk stubs: outside C1 (WP-F)
            stubs = [KF.make_stub(int(i), x[i], y[i], z[i], int(L["layer"][i]), bool(L["isPS"][i]),
                                  bool(L["isTilted"][i])) for i in r]
            by = {s.layer: s for s in stubs}
            la, lb = SEED_LAYERS[sd]
            if la not in by or lb not in by:
                excl("seed stub skipped")
                continue
            if len(by) != len(stubs):
                excl("two stubs in a layer")
                continue
            isec = int(T["phiSector"][j])
            ga, gb = by[la], by[lb]
            geo = (math.hypot(x[ga.idx], y[ga.idx]), float(z[ga.idx]), math.atan2(y[ga.idx], x[ga.idx]),
                   math.hypot(x[gb.idx], y[gb.idx]), float(z[gb.idx]), math.atan2(y[gb.idx], x[gb.idx]))
            seed = KF.make_seed(*KF.exact_tracklet(*geo, isec), isec, len(stubs))
            if KF.kf_eta_region(seed.eta_reg) > MAX_KF_REGION:
                continue
            res["judged"] += 1
            state, gates, why = KF.forced_fit(seed, stubs)
            if state is None:
                excl(f"no forced fit: {why}")
                continue
            if not all(gates):
                res["state_gate_fails"] += 1
            out = KF.fitted_output(state, seed)
            if not out.consistent_sector:
                res["not_consistent_sector"] += 1
                continue
            mine = np.array(out.track, np.float32)
            theirs = np.array([T["rInv"][j], T["phi"][j], T["tanL"][j], T["z0"][j], T["d0"][j]], np.float32)
            if np.array_equal(mine, theirs):
                res["identical"] += 1
            else:
                pre = (0.01 * KF.C_TRK * KF.K.b * float(out.q_over_pt),
                       KF.reduce_phi(float(out.phi0) - isec * 2 * math.pi / KF.N_SECTOR + 0.5 * KF.K.dphisector_hg),
                       float(out.tanl), float(out.z0), float(out.d0))
                if any(_edge(p, l) for p, l in zip(pre, lsb)):
                    res["predicted_edge"] += 1
                elif any(int(i) in exc for i in r):
                    res["known_exception"] += 1
                else:
                    res["mismatch"] += 1
                    if len(examples) < 20:
                        examples.append({"event": e, "track": j, "seed": sd,
                                         "lsb_diff": ((theirs.astype(float) - mine.astype(float)) / lsb).round(3).tolist()})
            n = len(stubs)
            c1v, c2v = float(T["chi2XYRed"][j]) * (n - 3), float(T["chi2ZRed"][j]) * (n - 2)
            if (abs(c1v - float(out.chi2rphi)) > C1_CHI2_REL * max(1.0, abs(c1v))
                    or abs(c2v - float(out.chi2rz)) > C1_CHI2_REL * max(1.0, abs(c2v))):
                res["chi2_outside"] += 1
    n_ex = sum(res["excluded"].values())
    frac_ex = n_ex / max(res["judged"] + n_ex, 1)
    gates = {"identity": res["mismatch"] == 0, "chi2": res["chi2_outside"] == 0,
             "state_gates": res["state_gate_fails"] == 0, "exclusions": bool(frac_ex <= C1_MAX_EXCLUDED)}
    return {"stage": "C1", "input": nodr, "positions": "stubPosTT" if used_tt else "nano x/y/z",
            **res, "excluded_fraction": frac_ex, "mismatch_examples": examples, "gates": gates}, all(gates.values())


# ---- C1b ------------------------------------------------------------------------------------------
def _candidate_key(sd, sec, rows, L):
    la, lb = SEED_LAYERS[sd]
    by = {}
    for x in rows:
        by.setdefault(int(L["layer"][x]), x)
    if la not in by or lb not in by:
        return None
    return (sd, int(sec), int(by[la]), int(by[lb]))


def c1b(fakefit, nodr=None, nev=None):
    """(i) the emulated candidates against CMSSW's (Fakefit no-DR: every track-builder candidate, its full
    stub list in order and its tracklet's integer parameters); (ii) when the no-DR production is given,
    the KF on CMSSW's own candidate list against CMSSW's no-DR track."""
    A = _events(fakefit, ("L1TTrack_", "L1TTrackStub_", "L1TOTStub_", "L1TOTStubDTC_"), nev)
    B = _events(nodr, ("L1TTrack_", "L1TTrackStub_"), nev) if nodr else None
    evB = {int(v): i for i, v in enumerate(ak.to_numpy(B["event"]))} if B is not None else {}
    res = {"cmssw_candidates": 0, "emulated_candidates": 0, "same_tracklet": 0, "identical_list": 0,
           "identical_order": 0, "emulation_superset": 0, "emulation_subset": 0, "other": 0,
           "no_emulated_tracklet": 0, "tracklet_without_emulated_candidate": 0,
           "emulated_without_cmssw": 0, "tracklet_pars_identical": 0,
           "kf_judged": 0, "kf_identical": 0}
    examples = []
    for e in range(len(A)):
        T, S = _table(A, e, "L1TTrack_"), _table(A, e, "L1TTrackStub_")
        L, Dt = _table(A, e, "L1TOTStub_"), _table(A, e, "L1TOTStubDTC_")
        x, y, z, _ = _positions(L, Dt)
        col = {}
        OC.run_event(L, Dt, collect=col, do_dr=False)
        cp = col["copies"]
        trk = {(t.seed, t.sector, int(cp.row[t.inner]), int(cp.row[t.outer])): t for t in col["tracklets"]}
        cand = {}
        for c in col["candidates"]:
            t = c.tracklet
            cand[(t.seed, t.sector, int(cp.row[t.inner]), int(cp.row[t.outer]))] = [int(cp.row[q]) for q in c.stubs]
        seen = set()
        nd = None
        if B is not None and int(A["event"][e]) in evB:
            eb = evB[int(A["event"][e])]
            nd = (_table(B, eb, "L1TTrack_"), _table(B, eb, "L1TTrackStub_"))
        for j in range(len(T["rInv"])):
            sd = int(T["seedType"][j])
            r = [int(v) for v in S["otStubIdx"][S["trackIdx"] == j]]
            if sd not in SEED_LAYERS or not L["isBarrel"][r].all():
                continue
            key = _candidate_key(sd, T["phiSector"][j], r, L)
            if key is None:
                continue
            res["cmssw_candidates"] += 1
            seen.add(key)
            if key not in trk:
                res["no_emulated_tracklet"] += 1
                continue
            res["same_tracklet"] += 1
            tr = trk[key]
            _, tpars = KF.track_from_ipars(tr.ipars[0], tr.ipars[1], 0, tr.ipars[3], tr.ipars[2], tr.sector)
            theirs = np.array([T["rInv"][j], T["phi"][j], T["tanL"][j], T["z0"][j], T["d0"][j]], np.float32)
            res["tracklet_pars_identical"] += int(np.array_equal(np.array(tpars, np.float32), theirs))
            if key not in cand:
                res["tracklet_without_emulated_candidate"] += 1
                continue
            em = cand[key]
            if em == r:
                res["identical_order"] += 1
            if set(em) == set(r):
                res["identical_list"] += 1
            elif set(em) > set(r):
                res["emulation_superset"] += 1
            elif set(em) < set(r):
                res["emulation_subset"] += 1
            else:
                res["other"] += 1
                if len(examples) < 20:
                    examples.append({"event": e, "key": key, "cmssw": r, "emulated": em})
            if nd is not None:
                ok = _kf_on_list(nd, key, r, L, x, y, z, tr)
                if ok is not None:
                    res["kf_judged"] += 1
                    res["kf_identical"] += int(ok)
        res["emulated_candidates"] += len(cand)
        res["emulated_without_cmssw"] += len(set(cand) - seen)
    n = max(res["same_tracklet"], 1)
    frac = res["identical_list"] / n
    lo = _wilson_lower(res["identical_list"], n)
    gates = {"match_lists": bool(lo >= C1B_MATCH_MIN)}
    if res["kf_judged"]:
        gates["kf_on_cmssw_list"] = bool(res["kf_identical"] == res["kf_judged"])
    return {"stage": "C1b", "fakefit": fakefit, "nodr": nodr, **res, "identical_list_fraction": frac,
            "identical_list_lower95": lo, "examples": examples, "gates": gates}, all(gates.values())


def _wilson_lower(k, n, z=1.6449):
    """One-sided 95 % lower bound of a binomial fraction (Wilson)."""
    if n == 0:
        return 0.0
    p = k / n
    c = (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    return max(c, 0.0)


def _kf_on_list(nd, key, rows, L, x, y, z, tr):
    """The KF on CMSSW's candidate list `rows` (in its order) against the no-DR track of the same tracklet:
    None when CMSSW fitted no track for it, else whether the final stub set and the parameters agree."""
    T, S = nd
    for j in range(len(T["rInv"])):
        r = [int(v) for v in S["otStubIdx"][S["trackIdx"] == j]]
        if int(T["seedType"][j]) != key[0] or int(T["phiSector"][j]) != key[1]:
            continue
        if _candidate_key(key[0], key[1], r, L) != key:
            continue
        stubs = [KF.make_stub(i, x[i], y[i], z[i], int(L["layer"][i]), bool(L["isPS"][i]), bool(L["isTilted"][i]))
                 for i in rows]
        seed = KF.make_seed(tr.rinv, tr.phi0, tr.t, tr.z0, tr.sector, len(stubs))
        out = KF.fitted_output(KF.do_kf(seed, stubs), seed)
        if not (out.accepted and out.consistent_sector):
            return False
        mine = np.array(out.track, np.float32)
        theirs = np.array([T["rInv"][j], T["phi"][j], T["tanL"][j], T["z0"][j], T["d0"][j]], np.float32)
        return set(s.idx for s in out.stubs) == set(r) and np.array_equal(mine, theirs)
    return None


# ---- C2 / C3 --------------------------------------------------------------------------------------
def _denominator(C):
    """The census's efficiency denominator: findable, |eta| < 0.9, pT >= 2 GeV."""
    return TF.findable(C) & np.isfinite(C["eta"]) & (np.abs(C["eta"]) < ETA_MAX) & (C["pt"] >= PT_MIN)


def _per_event(nev, ev, w=None):
    return np.bincount(ev, weights=w, minlength=nev).astype(np.float64)


def _track_counts(R, nev):
    """Per event: judged tracks, fakes, duplicate excess (sum over TPs of n - 1) and distinct TPs."""
    R = _judged(R)
    ev = R[:, 0]
    n = _per_event(nev, ev)
    fake = _per_event(nev, ev, (R[:, 3] == 0).astype(float))
    t = R[R[:, 2] >= 0]
    keys, cnt = np.unique(t[:, 2], return_counts=True)
    kev = M.tp_key_parts(keys)[0].astype(np.int64)
    dup = _per_event(nev, kev, (cnt - 1).astype(float))
    ntp = _per_event(nev, kev)
    return n, fake, dup, ntp


def c23(census_dir, cmssw_nodr, cmssw_merge, nev=None):
    """C2 (per seed, no DR) and C3 (delivered, merge DR) of a closed-arm census against CMSSW."""
    C = TF.load(TF.census_dir(census_dir), nev=nev, with_tracks=True, verbose=False)
    nevc = int(C["n_events"])
    entry = {SA.OT_CHAIN_SEED[sd.layers]: i for i, sd in enumerate(C["seeds"]) if sd.layers in SA.OT_CHAIN_SEED}
    den = _denominator(C)
    key, tev = C["key"], M.tp_key_parts(C["key"])[0].astype(np.int64)
    parts = {}
    # C2: per seed
    if cmssw_nodr:
        R_cms = _cmssw_tracks(cmssw_nodr, nevc)
        for sd in JUDGED_SEEDS:
            i = entry[sd]
            f_cen = _found(C["track_records"][i], key) & den
            f_cms = _found(R_cms[R_cms[:, 1] == sd], key) & den
            nc, _, _, _ = _track_counts(C["track_records"][i], nevc)
            nm, _, _, _ = _track_counts(R_cms[R_cms[:, 1] == sd], nevc)
            parts[f"C2 {OC.SEED_NAME[sd]}"] = {
                "den": _per_event(nevc, tev[den]), "cen": _per_event(nevc, tev[f_cen]),
                "cms": _per_event(nevc, tev[f_cms]), "disc": _per_event(nevc, tev[f_cen ^ f_cms]),
                "ncen": nc, "ncms": nm}
    # C3: delivered
    if cmssw_merge:
        R_cen = np.concatenate([C["track_records_merged"][i] for i in entry.values()])
        R_cms = _cmssw_tracks(cmssw_merge, nevc)
        f_cen = _found(R_cen, key) & den
        f_cms = _found(R_cms, key) & den
        nc, fc, dc, tc = _track_counts(R_cen, nevc)
        nm, fm, dm, tm = _track_counts(R_cms, nevc)
        parts["C3 delivered"] = {"den": _per_event(nevc, tev[den]), "cen": _per_event(nevc, tev[f_cen]),
                                 "cms": _per_event(nevc, tev[f_cms]), "disc": _per_event(nevc, tev[f_cen ^ f_cms]),
                                 "fcen": fc, "fcms": fm, "dcen": dc, "tcen": tc, "dcms": dm, "tcms": tm,
                                 "ncen": nc, "ncms": nm}
    stats = {name: _stats(p, name.startswith("C3")) for name, p in parts.items()}
    boot = _bootstrap(parts, nevc)
    gates, table = {}, {}
    for name, st in stats.items():
        for s, v in st.items():
            q5, q95 = np.percentile(boot[name][s], [5, 95])
            m = MARGIN[s]
            ok = bool((q5 > -m and q95 < m) if s in SIGNED else q95 < m)
            table[f"{name}: {s}"] = {"value": v, "q5": float(q5), "q95": float(q95), "margin": m, "pass": ok}
            gates[f"{name}: {s}"] = ok
    power = _joint_power(stats, boot)
    return {"stage": "C2 / C3", "census": census_dir, "cmssw_nodr": cmssw_nodr, "cmssw_merge": cmssw_merge,
            "n_events": nevc, "n_boot": N_BOOT, "statistics": table, "joint_power": power,
            "gates": gates}, all(gates.values())


def _stats(p, delivered, w=None):
    """The gated statistics of one comparison from its per-event arrays (weights w: a bootstrap replica)."""
    S = (lambda a: float(np.sum(a if w is None else a * w)))
    den = max(S(p["den"]), 1.0)
    out = {"d_eff": (S(p["cen"]) - S(p["cms"])) / den, "disc": S(p["disc"]) / den,
           "ln_cand": math.log(max(S(p["ncen"]), 1.0) / max(S(p["ncms"]), 1.0))}
    if delivered:
        out.pop("ln_cand")
        out["ln_fake"] = math.log(max(S(p["fcen"]), 1.0) / max(S(p["fcms"]), 1.0))
        out["d_dup"] = S(p["dcen"]) / max(S(p["tcen"]), 1.0) - S(p["dcms"]) / max(S(p["tcms"]), 1.0)
    return out


def _bootstrap(parts, nev, n=N_BOOT, seed=BOOT_SEED):
    """The paired event bootstrap: the same resampled events for the census and CMSSW, every comparison."""
    rng = np.random.default_rng(seed)
    W = [np.bincount(rng.integers(0, nev, nev), minlength=nev).astype(np.float64) for _ in range(n)]
    out = {}
    for name, p in parts.items():
        reps = [_stats(p, name.startswith("C3"), w) for w in W]
        out[name] = {s: np.array([r[s] for r in reps]) for s in reps[0]}
    return out


def _joint_power(stats, boot):
    """The joint pass probability (plan sec. 3, stats L3): the fraction of replicas, each taken as the
    observed sample with the bootstrap's own interval around it, in which every gate passes."""
    names = [(c, s) for c in stats for s in stats[c]]
    if not names:
        return None
    n = len(next(iter(next(iter(boot.values())).values())))
    ok = np.ones(n, bool)
    for c, s in names:
        b = boot[c][s]
        q5, q95 = np.percentile(b, [5, 95])
        shift = b - stats[c][s]
        m = MARGIN[s]
        ok &= ((q5 + shift > -m) & (q95 + shift < m)) if s in SIGNED else (q95 + shift < m)
    return float(ok.mean())


# ---- the order floor --------------------------------------------------------------------------------
def floor(dtc_dir, random_dirs, cmssw_nodr=None, cmssw_merge=None, nev=None):
    """D_order (the mean delivered discordance over the random pairs, jackknifed over runs, with its 80 %
    upper bound) and Delta_dtc (each gated statistic of the dtc run minus the random runs' mean, against
    CMSSW, with the error of that mean) (plan sec. 3, the order floor)."""
    Cs = [TF.load(TF.census_dir(d), nev=nev, verbose=False) for d in random_dirs]
    den = _denominator(Cs[0])
    fd = []
    for C in Cs:
        entry = [i for i, sd in enumerate(C["seeds"]) if sd.layers in SA.OT_CHAIN_SEED]
        f = np.zeros(len(C["key"]), bool)
        for i in entry:
            f |= TF.seed_mask(C, i, after_dr=True)
        fd.append(f & den)
    nr = len(fd)
    pairs = list(itertools.combinations(range(nr), 2))
    disc = {p: float((fd[p[0]] ^ fd[p[1]]).sum()) / max(den.sum(), 1) for p in pairs}
    mean = float(np.mean(list(disc.values()))) if pairs else float("nan")
    jk = [np.mean([v for p, v in disc.items() if k not in p]) for k in range(nr)] if nr > 2 else []
    se = float(math.sqrt((nr - 1) / nr * np.sum((np.array(jk) - np.mean(jk)) ** 2))) if jk else float("nan")
    out = {"stage": "order floor", "dtc": dtc_dir, "random": random_dirs, "D_order": mean, "D_order_se_jk": se,
           "D_order_upper80": mean + 0.8416 * se, "half_margin_disc": MARGIN["disc"] / 2}
    if cmssw_nodr or cmssw_merge:
        rep = {d: c23(d, cmssw_nodr, cmssw_merge, nev)[0]["statistics"] for d in [dtc_dir] + list(random_dirs)}
        dd = {}
        for s in rep[dtc_dir]:
            r = np.array([rep[d][s]["value"] for d in random_dirs])
            dd[s] = {"delta_dtc": rep[dtc_dir][s]["value"] - float(r.mean()), "sigma": float(r.std(ddof=1) / math.sqrt(len(r))),
                     "half_margin": rep[dtc_dir][s]["margin"] / 2}
        out["delta_dtc"] = dd
    exceeds = out["D_order_upper80"] > out["half_margin_disc"] or any(
        abs(v["delta_dtc"]) + 2 * v["sigma"] > v["half_margin"] for v in out.get("delta_dtc", {}).values())
    out["floor_exceeds_half_margin"] = bool(exceeds)
    out["gates"] = {"floor_within_half_margin": not exceeds}
    return out, not exceeds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("bridge"); p.add_argument("v4"); p.add_argument("v3")
    p.add_argument("--stub-shift", action="store_true", help="the known-bad: canon_v4's stubs moved by an event")
    p = sub.add_parser("twice"); p.add_argument("first"); p.add_argument("second")
    p = sub.add_parser("l2l3"); p.add_argument("census"); p.add_argument("cmssw")
    p = sub.add_parser("c1"); p.add_argument("nodr")
    p = sub.add_parser("c1b"); p.add_argument("fakefit"); p.add_argument("--nodr", default=None)
    p = sub.add_parser("c23"); p.add_argument("census")
    p.add_argument("--nodr", default=None, help="CMSSW no-DR production (C2)")
    p.add_argument("--merge", default=None, help="CMSSW merge-DR production (C3)")
    p = sub.add_parser("floor"); p.add_argument("dtc"); p.add_argument("random", nargs="+")
    p.add_argument("--nodr", default=None); p.add_argument("--merge", default=None)
    for p in sub.choices.values():
        p.add_argument("-n", "--nev", type=int, default=None)
        p.add_argument("-o", "--out", default=None)
        p.add_argument("--known-bad", default=None,
                       help="run the stage with a known-bad variant of the chain or the fit switched on "
                            "(c1, c1b; the censuses of c23 / floor carry their own)")
    a = ap.parse_args()
    OC.set_known_bad(a.known_bad)
    if a.stage == "bridge":
        rep, ok = bridge(a.v4, a.v3, a.nev, a.stub_shift)
    elif a.stage == "twice":
        rep, ok = twice(a.first, a.second, a.nev)
    elif a.stage == "l2l3":
        rep, ok = l2l3(a.census, a.cmssw, a.nev)
    elif a.stage == "c1":
        rep, ok = c1(a.nodr, a.nev)
    elif a.stage == "c1b":
        rep, ok = c1b(a.fakefit, a.nodr, a.nev)
    elif a.stage == "c23":
        rep, ok = c23(a.census, a.nodr, a.merge, a.nev)
    else:
        rep, ok = floor(a.dtc, a.random, a.nodr, a.merge, a.nev)
    _fail(rep, a.out, ok)


if __name__ == "__main__":
    main()
