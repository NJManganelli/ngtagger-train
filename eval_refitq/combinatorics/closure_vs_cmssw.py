"""closure_vs_cmssw: the census OT chain closed against CMSSW, stage by stage (WP-D, wpd/PLAN.md sec. 2-4; WP-Disc,
wpdisc/PLAN.md sec. 3: the discs, --discs on|off).

  bridge  C0  canon_v4 against canon_v3, branch by branch: equal, or in the declared lists (BRIDGE_*)
  twice   C0  run-twice determinism: two productions of one configuration are bit identical
  l2l3    C0  L2L3 tracks in KF regions 0-3: none in CMSSW and none in the closed arm
  c1          the KF unit: cmssw_kf forced onto every CMSSW no-DR track's own stubs at the KF's own positions
              (with the discs on: every seed, stub and KF region; off: barrel seeds, barrel-only tracks, KF
              regions 0-3), quantised as HybridFit / Track
  c1b         the match lists against CMSSW's track-builder candidates, from a no-DR production with
              Fakefit (every candidate with its full stub list and its tracklet's integer parameters):
              (i) per tracklet, the emulated candidate against CMSSW's, and its integer parameters;
              (ii) the KF on every CMSSW candidate list against CMSSW's no-DR output: accepted or rejected
              alike, and when both accept, the same stubs and parameters
  g0          a format-36 census with the discs off against format 35's: shard by shard identical, format
              36's additions empty
  acceptance  each seed's acceptance (its KF eta indices) from a CMSSW no-DR production: C2's denominators
              with the discs on, frozen on the known-good file
  c23         C2 per candidate (no DR, per seed) and C3 delivered (merge DR; with the discs on per band and
              pooled): a closed-arm census against CMSSW, with the paired event bootstrap and equivalence at
              the plan's margins
  floor       the order floor: D_order and Delta_dtc from one dtc and ten random closed-arm censuses
  truncation  U, the unemulated truncation: the caps-on arm's statistics minus the caps-off arm's (paired);
              reported, not gated (the caps-on arm is not certified, plan sec. 11)
  register    a known-bad candidate against the known-good on the same events: its shift per gated
              statistic in margin units (paired), its gate outcome and the plan's classification

Every stage prints its summary, writes its JSON report (-o) and exits 2 when one of its gates fails.

ONE RECORD RULE FOR BOTH SIDES (plan sec. 2; code review 01, blocking 1). Every track, CMSSW's or the census's,
becomes a record from its FITTED stubs and the input's own L1TOTStub table (track_record): the seed TP is the
single TP credited by both of its final stubs in the seed type's two layers (tracklet_topology_cost.seed_tp;
an OT stub credits its tpIdx and its clusters' owners; none on a tie); the KF eta region is HybridFit's on
exacttracklet of those two stubs; a track without a stub in a seed layer is not judged. A track is a fake when
it has no seed TP; a TP is found by a side when one of its judged tracks (the scope's KF regions) has it. The census
exports its tracks' final stubs as L1TOTStub rows (tp_findability OT_STUB_COLS), so the rule runs on the same
code and the same stub table for both. The census's own physics (its found bitmaps) keeps the tracklet-seed
rule; the closure validates the chain under the final-stub rule.
The sheet's definitions (the majority owner of all final stubs, a fake = 2 or more stubs not crediting the
owner) are reported beside, ungated, from the same records.
The efficiency denominator is the census's: findable (>= 4 credited modules, charged), pT >= 2 GeV, |eta| < 0.9
(discs off, WP-D) or < 2.4 (discs on, WP-Disc: C2 per seed inside its acceptance, C3 per band and pooled).

THE GATED SAMPLE (plan sec. 3 with the statistics review's R1, pre-declared 2026-10-15): C2 / C3 are judged
on f01 events 10-99 (plus f03 if the power rule calls for it): f01 events 0-9 were used in development, so
c23 / floor take --exclude-events 0-9 on that run, applied to both sides together.
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
import kf_emulation as KFE            # noqa: E402
import ot_chain as OC                 # noqa: E402
import seed_arity as SA               # noqa: E402
import tp_findability as TF           # noqa: E402
import tracklet_topology_cost as M    # noqa: E402

FAIL_EXIT = 2
N_BOOT, BOOT_SEED = 400, 20261015
# THE SCOPE (WP-Disc, census format 36): with the discs on (ot_chain.DISCS) the 8 prompt seeds, disc stubs, all
# 8 KF eta regions and |eta| < 2.4; with them off WP-D's (format 35): the barrel seeds, barrel-only tracks in C1 /
# C1b, KF regions 0-3, |eta| < 0.9. L2L3 is judged only with the discs on (WP-D: a C0 identity, no tracks in KF
# regions 0-3). Each seed's two stubs as L1TOTStub_layer codes (barrel 1-6, discs 11-15 on either side).
SEED_LAYERS_ALL = {0: (1, 2), 1: (2, 3), 2: (3, 4), 3: (5, 6), 4: (11, 12), 5: (13, 14), 6: (1, 11), 7: (2, 11)}
MAX_KF_REGION_BARREL, MAX_KF_REGION_DISCS = 3, 7
ETA_MAX_BARREL, ETA_MAX_DISCS, PT_MIN = 0.9, 2.4, 2.0


def seed_layers():
    """{seed: its two stubs' L1TOTStub_layer codes} of the scope's seeds: the 8 with the discs on, the barrel's
    off (whatever a known-bad variant of the chain runs: the closure judges it against CMSSW's seeds)."""
    return {sd: v for sd, v in SEED_LAYERS_ALL.items() if OC.DISCS or sd in OC.BARREL_SEEDS}


def judged_seeds():
    """The seeds C2 judges: the 8 with the discs on, L1L2 / L3L4 / L5L6 with them off."""
    return tuple(seed_layers()) if OC.DISCS else (0, 2, 3)


def max_kf_region():
    return MAX_KF_REGION_DISCS if OC.DISCS else MAX_KF_REGION_BARREL


def eta_max():
    return ETA_MAX_DISCS if OC.DISCS else ETA_MAX_BARREL


# THE BANDS (WP-Disc plan sec. 3, C3 gated per band and pooled; pre-declared 2026-10-18): KF eta indices 0-3
# (|eta| < 0.9), 4-5 (0.9-1.68) and 6-7 (1.68-2.4), the KF's own region edges (the plan's 1.6 is the KF's
# 1.68). A TP's index is its truth |eta| on those edges; a track's is its KF eta region (record column kfreg).
KF_ETA_EDGES = (0.0, 0.2, 0.41, 0.62, 0.9, 1.26, 1.68, 2.08, 2.4)
BANDS = {"barrel": (0, 3), "overlap": (4, 5), "forward": (6, 7)}
# C2's denominators (plan sec. 3 with the statistics review's B1): per seed, the findable TPs inside its own
# acceptance, the KF eta indices holding >= ACCEPTANCE_MIN of that seed's CMSSW no-DR tracks on the known-good
# file (closure acceptance, frozen on f04 before f03 is read)
ACCEPTANCE_MIN = 0.01


def tp_kf_index(eta):
    """A TP's KF eta index: its truth |eta| on KF_ETA_EDGES (0-7; 8 beyond 2.4, NaN eta too)."""
    a = np.abs(np.asarray(eta, np.float64))
    return np.where(np.isfinite(a), np.searchsorted(np.asarray(KF_ETA_EDGES[1:]), a, side="right"), 8)


def _in_scope(L, rows):
    """A CMSSW track's stubs inside the scope: any with the discs on, barrel-only with them off."""
    return OC.DISCS or bool(L["isBarrel"][rows].all())


def _kf_stub(L, i, x, y, z):
    """The KF's stub (cmssw_kf.make_stub) for L1TOTStub index i at the KF's positions x, y, z: a barrel stub by its
    layer; a disc stub by its disc number, with innerStrip and isFlipped for the 2S alpha."""
    lay, ps, tilt = int(L["layer"][i]), bool(L["isPS"][i]), bool(L["isTilted"][i])
    if L["isBarrel"][i]:
        return KF.make_stub(int(i), x[i], y[i], z[i], lay, ps, tilt)
    return KF.make_stub(int(i), x[i], y[i], z[i], lay - 10, ps, tilt, barrel=False,
                        strip=float(L["innerStrip"][i]), flipped=bool(L["isFlipped"][i]))
# the margins (plan sec. 3, "decision arithmetic")
MARGIN = {"d_eff": 0.01, "disc": 0.02, "ln_cand": math.log(1.05), "ln_fake": math.log(1.25), "d_dup": 0.01}
SIGNED = ("d_eff", "ln_cand", "ln_fake", "d_dup")   # equivalence on both sides; disc is one-sided
C1_EDGE_LSB = 1e-6          # a census value this close to a bin edge is a predicted edge, not a failure
C1_F32_EDGE_ULP = 2         # a single +-1 LSB flip this many float32 ulps from the edge: MINOR (float32 edge)
C1_CHI2_REL = 1e-5
C1_MAX_SKIPPED = 0.02       # plan sec. 3: "exclusions (seed stub skipped) <= 2 %"
C1B_MATCH_MIN = 0.99
# THE PHYSICIST'S ACCEPTANCE OF THE MATCH-LIST GATE (2026-10-16, plan sec. 11): "match lists of 98.6% are
# GOOD-ENOUGH", at the agreement measured on the Fakefit copy (f01 events 0-9): 2,446 of 2,482 tracklets' lists
# identical. A C1b(i) at or above that fraction is accepted, and the order floor then takes the passed branch.
C1B_ACCEPTED = (2446, 2482)
C1B_SHORTFALL_MATERIAL = 0.01   # WP-Disc plan sec. 3: a forward C1b(i) shortfall of 1 % or more is MATERIAL
NODR_SHARE_MIN = 0.5        # a no-DR production: most tracks share >= 3 stubs with another of their sector
                            # (canon_v4 test, merge DR: 0.10; modecmp no-DR: 0.96)
# record columns (track_record): event, CMSSW seed type, seed TP key (-1 none), seed-truthful, KF eta region
# (-1 no stub in a seed layer), fitted stubs, owner key (sheet: majority owner of all fitted stubs; -1 none),
# stubs not crediting the owner
RECORD_COLS = ("event", "seed", "tp_key", "real", "kfreg", "nstub", "own_key", "n_wrong")

# THE BRIDGE (plan sec. 5): canon_v4 against canon_v3 on the same events. Only these may differ: the disc
# global-direction columns on disc rows (15c345f, the disc sense fix) and the full-float columns, which must
# equal canon_v3 after rounding to canon_v3's stored mantissa; only these may be new.
BRIDGE_DISC = tuple(f"L1TSmartPixelsCluster_{c}" for c in (
    "globalClusterPhi", "globalClusterCotTheta", "sigGlobalClusterPhi", "sigGlobalClusterCotTheta",
    "tpGlobalClusterPhi", "tpGlobalClusterCotTheta"))         # p03_angles/disc_sense_fix.md sec. 3
BRIDGE_ROW_TYPE = "L1TSmartPixelsCluster_surface"              # 1-4 barrel, 11-14 disc
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


def _bits(x):
    """The values as their bits: float arrays viewed as unsigned integers of the same width."""
    x = np.ascontiguousarray(x)
    return x.view(f"u{x.dtype.itemsize}") if x.dtype.kind == "f" else x


def _same(a, b):
    """Two branches bit identical: the same events, counts, dtype and bits (-0.0 and NaN payloads included)."""
    if len(a) != len(b):
        return False
    if a.ndim > 1:
        if b.ndim != a.ndim or not np.array_equal(ak.to_numpy(ak.num(a, axis=1)), ak.to_numpy(ak.num(b, axis=1))):
            return False
        a, b = ak.flatten(a), ak.flatten(b)
    fa, fb = ak.to_numpy(a), ak.to_numpy(b)
    return fa.dtype == fb.dtype and np.array_equal(_bits(fa), _bits(fb))


def _event_order(t_a, t_b):
    """The entries of tree t_a in the order of t_b's events (run, luminosity block, event): a multi-threaded
    job writes its events in a scheduling-dependent order. None when the two hold different events."""
    key = lambda t: list(zip(*(t[b].array(library="np").tolist() for b in ("run", "luminosityBlock", "event"))))
    ka, kb = key(t_a), key(t_b)
    if len(ka) != len(kb) or set(ka) != set(kb):
        return None
    pos = {k: i for i, k in enumerate(ka)}
    return np.array([pos[k] for k in kb], np.int64)


def bridge(v4, v3, nev=None, stub_shift=False):
    """Every branch of canon_v3 equal in canon_v4 except the declared ones; new branches only the declared.
    The events are compared by (run, lumi, event), whatever order each job wrote them in. The disc columns
    may differ on disc rows only. stub_shift (the bridge's known-bad): canon_v4's L1TOTStub columns moved by
    one event."""
    t4, t3 = uproot.open(f"{v4}:Events"), uproot.open(f"{v3}:Events")
    order = _event_order(t4, t3)
    if order is None:
        return {"stage": "C0 bridge", "v4": v4, "v3": v3, "same_events": False,
                "gates": {"bridge": False}}, False
    order = order[:nev]
    k4, k3 = set(t4.keys()), set(t3.keys())
    missing = sorted(k3 - k4)
    new = sorted(k for k in k4 - k3 if not (k.startswith(BRIDGE_NEW_PREFIX) or k in BRIDGE_NEW))
    barrel = ak.to_numpy(ak.flatten(t3[BRIDGE_ROW_TYPE].array(library="ak", entry_stop=nev))) <= 4
    diff, declared, rounded_bad, disc_on_barrel = [], [], [], []
    for k in sorted(k3 & k4):
        a = t4[k].array(library="ak")[order]
        b = t3[k].array(library="ak", entry_stop=nev)
        if stub_shift and k.startswith("L1TOTStub_"):
            a = ak.concatenate([a[1:], a[:1]])
        if _same(a, b):
            continue
        if k in BRIDGE_DISC:
            fa, fb = ak.to_numpy(ak.flatten(a)), ak.to_numpy(ak.flatten(b))
            ok = len(fa) == len(fb) == len(barrel) and np.array_equal(_bits(fa[barrel]), _bits(fb[barrel]))
            (declared if ok else disc_on_barrel).append(k)
        elif k in BRIDGE_ROUNDED:
            fa = ak.to_numpy(ak.flatten(a, axis=None))
            fb = ak.to_numpy(ak.flatten(b, axis=None))
            ok = len(fa) == len(fb) and np.array_equal(_bits(_round_mantissa(fa, BRIDGE_ROUNDED[k])),
                                                       _bits(fb.astype(np.float32)))
            (declared if ok else rounded_bad).append(k)
        else:
            diff.append(k)
    ok = not (missing or new or diff or rounded_bad or disc_on_barrel)
    tables = sorted({k.split("_")[0].lstrip("n") for k in diff + rounded_bad + disc_on_barrel})
    return {"stage": "C0 bridge", "v4": v4, "v3": v3, "same_events": True,
            "same_event_order": bool(np.array_equal(order, np.arange(len(order)))),
            "stub_shift": stub_shift, "branches_common": len(k3 & k4),
            "missing_in_v4": missing, "undeclared_new": new, "undeclared_differences": diff,
            "disc_columns_differing_on_barrel_rows": disc_on_barrel, "tables_differing": tables,
            "declared_differences": declared, "rounded_columns_not_equal": rounded_bad,
            "gates": {"bridge": ok}}, ok


def twice(f1, f2, nev=None):
    """Run-twice determinism: every branch bit identical, event by event (the order the jobs wrote their
    events in may differ; it is reported)."""
    t1, t2 = uproot.open(f"{f1}:Events"), uproot.open(f"{f2}:Events")
    order = _event_order(t1, t2)
    if order is None:
        return {"stage": "C0 run twice", "files": [f1, f2], "same_events": False,
                "gates": {"determinism": False}}, False
    order = order[:nev]
    k1, k2 = set(t1.keys()), set(t2.keys())
    diff = [k for k in sorted(k1 & k2)
            if not _same(t1[k].array(library="ak")[order], t2[k].array(library="ak", entry_stop=nev))]
    ok = k1 == k2 and not diff
    return {"stage": "C0 run twice", "files": [f1, f2], "same_events": True,
            "same_event_order": bool(np.array_equal(order, np.arange(len(order)))), "branch_sets_equal": k1 == k2,
            "differences": diff, "gates": {"determinism": ok}}, ok


# ---- G0: format 36 with the discs off against format 35 ------------------------------------------------
# format 36's new columns: a track's hit row gains ID1-ID4 and OD1-OD5 (KF.LAYER_ORDER), the final stubs OD1-OD5
# (TF.OT_STUB_COLS, at the end)
_G0_HIT_DISC_COLS = [i for i, L in enumerate(KFE.LAYER_ORDER) if M.is_disc(L)]


_G0_NEW_COUNTERS = ("kf_diverged",)


def _g0_array(name, a35, a36):
    """'' when shard array `name` of a format-36 census with the discs off equals format 35's, else why not.
    Format 36's additions must be empty: its census columns (cen_ + TF.FORMAT36_COLS) all 0; the disc columns of a
    track's hit row (hit_<i>, hit_m<i>: _G0_HIT_DISC_COLS) and of its final stubs (hit_o / hit_om: the trailing
    OD1-OD5) all -1; the per-seed counters' kf_diverged 0; S6's records without a disc track or group."""
    if a35 is None:
        if name.startswith("hit_nc"):              # the class-C count per exported track (census_scan): descriptive
            return ""
        if name == "rec_trk__disc":                # S6's disc group (census_angles.s6_group): empty
            return "" if not np.any(a36) else "S6 disc tracks with the discs off"
        if name == "rec_trk__s6g":
            return "" if not any(str(g).endswith(TF.CA.S6_DISC) for g in a36) else "S6 disc group with the discs off"
        return "" if name[4:] in TF.FORMAT36_COLS and name.startswith("cen_") and not np.any(a36) else "new array"
    if name == "counters":                        # per seed: format 36 adds kf_diverged, which must be 0
        c35, c36 = json.loads(str(a35)), json.loads(str(a36))
        if any(c.get("kf_diverged", 0.0) for c in c36.values()):
            return "kf_diverged nonzero"
        trim = {k: {kk: vv for kk, vv in c.items() if kk not in _G0_NEW_COUNTERS} for k, c in c36.items()}
        return "" if trim == c35 else "values differ"
    if a36.shape == a35.shape:
        return "" if np.array_equal(a35, a36, equal_nan=a35.dtype.kind == "f") else "values differ"
    if not (a36.ndim == 2 and a35.ndim == 2 and a36.shape[0] == a35.shape[0] and a36.shape[1] > a35.shape[1]):
        return f"shape {a35.shape} -> {a36.shape}"
    kind = TF.hit_key(name)[0] if name.startswith("hit_") else None
    new = _G0_HIT_DISC_COLS if kind in ("", "m") else list(range(a35.shape[1], a36.shape[1]))
    old = [i for i in range(a36.shape[1]) if i not in new]
    if len(old) != a35.shape[1]:
        return f"shape {a35.shape} -> {a36.shape}"
    if not np.array_equal(a36[:, old], a35):
        return "values differ"
    return "" if np.all(a36[:, new] == -1) else "disc columns filled"


def g0(c35, c36, nev=None):
    """G0 (WP-Disc plan sec. 3): a format-36 census with the discs off (--discs off) against format 35's on the
    same inputs and configuration, shard by shard: every array identical apart from format 36's additions, which
    must be empty (_g0_array); the counters, the entries and the configuration identical apart from the format
    and format 36's configuration keys: the discs key, it_discs (False), disc_seeds (none), the angle model's disc
    table identity (disc_table_*, inert with the discs off), S1's displaced tolerance (0, the prompt band), S1's
    placement and the angle regions (all, both) and the null control (none)."""
    d35, d36 = TF.census_dir(c35), TF.census_dir(c36)
    m35, m36 = (json.load(open(Path(d) / "manifest.json")) for d in (d35, d36))
    cfg35, cfg36 = dict(m35["config"]), dict(m36["config"])
    oc36 = dict(cfg36.pop("ot_chain"))
    disc_on = oc36.pop("discs", None)
    it_discs, disc_seeds = cfg36.pop("it_discs", False), cfg36.pop("disc_seeds", None)
    null = cfg36.pop("angle_null", None)
    ang36 = dict(cfg36.get("angles", {}))
    placement = (ang36.pop("s1_at", "all"), ang36.pop("regions", "both"))
    if "angles" in cfg36:
        cfg36["angles"] = ang36
    if isinstance(cfg36.get("angle_model"), dict):
        am = cfg36["angle_model"]
        if am.get("s1_d0max", 0.0) != 0.0:
            return {"stage": "G0", "gates": {"g0_identical": False},
                    "differences": [f"S1 displaced tolerance {am['s1_d0max']} with the discs off"]}, False
        cfg36["angle_model"] = {k: v for k, v in am.items() if not k.startswith("disc_table") and k != "s1_d0max"}
    same_cfg = (cfg35.pop("ot_chain") == oc36 and cfg35 == cfg36 and it_discs is False and not disc_seeds
                and null is None and placement == ("all", "both"))
    head = {"formats": [m35.get("format"), m36.get("format")], "discs": disc_on,
            "inputs_equal": m35["inputs"] == m36["inputs"], "config_equal": same_cfg,
            "entries_equal": (m35["seeds"], m35["seed_follow"]) == (m36["seeds"], m36["seed_follow"])}
    sh35, sh36 = sorted(Path(d35).glob("chunk_*.npz")), sorted(Path(d36).glob("chunk_*.npz"))
    diffs, nread = [], 0
    for a, b in zip(sh35, sh36):
        if nev is not None and nread >= nev:
            break
        z35, z36 = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
        nread += int(z35["n_events"])
        for name in sorted(set(z35.files) | set(z36.files)):
            if name not in z36.files:
                diffs.append(f"{b.name}: {name} missing")
                continue
            why = _g0_array(name, z35[name] if name in z35.files else None, z36[name])
            if why:
                diffs.append(f"{b.name}: {name} {why}")
    ok = (head["formats"] == [35, 36] and disc_on is False and head["inputs_equal"] and same_cfg
          and head["entries_equal"] and len(sh35) == len(sh36) and not diffs)
    return {"stage": "G0", "format35": d35, "format36_discs_off": d36, **head, "n_shards": [len(sh35), len(sh36)],
            "n_events_compared": nread, "differences": diffs[:50], "n_differences": len(diffs),
            "gates": {"g0_identical": ok}}, ok


# ---- the per-track records, one rule for both sides ---------------------------------------------------
def _stub_tables(spec, nev=None):
    """The L1TOTStub table of every event of `spec` (one file or several, comma-separated, in order), indexed
    as the census numbers events (a running count over the files)."""
    out = []
    for path in M.expand_inputs(spec):
        left = None if nev is None else nev - len(out)
        if left is not None and left <= 0:
            break
        A = _events(path, ("L1TOTStub_",), left)
        for k in range(len(A)):
            L = _table(A, k, "L1TOTStub_")
            L["event"] = np.full(len(L["layer"]), len(out), np.int64)
            out.append(L)
    return out


def track_record(L, e, sd, rows):
    """RECORD_COLS of one track of seed type sd (0-7) with fitted stubs `rows` (L1TOTStub indices of event e,
    L its stub table): the record rule of the module docstring."""
    rows = np.asarray(rows, np.int64)
    by = {int(L["layer"][x]): x for x in rows}
    tps = M.credits_of(L, rows)[None]
    own, _, owned, right, _ = KFE.credited_owner(tps, np.ones((1, len(rows)), bool))
    own_key = int(M.tp_key(np.array([e]), own)[0]) if owned[0] else -1
    n_wrong = int((~right[0]).sum())
    la, lb = SEED_LAYERS_ALL[sd]
    if la not in by or lb not in by:
        return (e, sd, -1, 0, -1, len(rows), own_key, n_wrong)
    a, b = by[la], by[lb]
    geo = [float(v) for v in (math.hypot(L["x"][a], L["y"][a]), L["z"][a], math.atan2(L["y"][a], L["x"][a]),
                              math.hypot(L["x"][b], L["y"][b]), L["z"][b], math.atan2(L["y"][b], L["x"][b]))]
    kreg = OC.kf_region(*KF.exact_tracklet(*geo, 0))
    tp, real = M.seed_tp(L, np.array([a]), np.array([b]))
    key = int(M.tp_key(np.array([e]), tp)[0]) if M.SPN.tp_matched(tp)[0] else -1
    return (e, sd, key, int(real[0]), kreg, len(rows), own_key, n_wrong)


def _records(recs):
    return np.array(recs, np.int64).reshape(-1, len(RECORD_COLS))


def _event_ids(spec, nev=None):
    """(run, lumi, event) of the events of `spec` (files comma-separated, in order), in the census's numbering
    (a running count over the files), the first `nev` of them."""
    out = []
    for path in M.expand_inputs(spec):
        left = None if nev is None else nev - len(out)
        if left is not None and left <= 0:
            break
        t = uproot.open(f"{path}:Events")
        out += list(zip(*(t[b].array(library="np", entry_stop=left).tolist()
                          for b in ("run", "luminosityBlock", "event"))))
    return out


def _cmssw_tracks(spec, ids, collection="L1TTrack"):
    """The records of a CMSSW track collection's barrel-seeded tracks, each event numbered as the census numbers
    the SAME event: `ids` the census's event identities (_event_ids of its input). A production holds its events
    in a scheduling-dependent axis-0 order (a sibling of the census input can hold the same event at another
    axis-0 index), so events are matched by (run, lumi, event), never by axis-0 index; events the census did not
    read are skipped, and a census event missing from the production is refused."""
    pos = {k: i for i, k in enumerate(ids)}
    recs, seen = [], set()
    for path in M.expand_inputs(spec):
        A = _events(path, (f"{collection}_", f"{collection}Stub_", "L1TOTStub_", "run", "luminosityBlock"))
        idk = zip(ak.to_numpy(A["run"]).tolist(), ak.to_numpy(A["luminosityBlock"]).tolist(),
                  ak.to_numpy(A["event"]).tolist())
        for k, ident in enumerate(idk):
            e = pos.get(ident)
            if e is None:
                continue
            seen.add(e)
            T, S, L = _table(A, k, f"{collection}_"), _table(A, k, f"{collection}Stub_"), _table(A, k, "L1TOTStub_")
            L["event"] = np.full(len(L["layer"]), e, np.int64)
            for j in range(len(T["rInv"])):
                sd = int(T["seedType"][j])
                if sd in seed_layers():
                    recs.append(track_record(L, e, sd, S["otStubIdx"][S["trackIdx"] == j]))
    if len(seen) != len(ids):
        raise SystemExit(f"{spec} lacks {len(ids) - len(seen)} of the census's events")
    return _records(recs)


def _census_tracks(C, entries, merged, stubs):
    """The records of a census's exported OT tracks (its final stubs, TF.OT_STUB_COLS), per entry; `stubs`
    the census input's L1TOTStub tables (_stub_tables)."""
    out = {}
    for i in entries:
        sd = SA.OT_CHAIN_SEED[C["seeds"][i].layers]
        O = C["track_stubs_merged" if merged else "track_stubs"].get(i, np.zeros((0, len(TF.OT_STUB_COLS)), np.int64))
        out[i] = _records([track_record(stubs[int(o[0])], int(o[0]), sd, o[1:][o[1:] >= 0]) for o in O])
    return out


def _census_inputs(C):
    meta = json.load(open(Path(C["cache_dir"]) / "manifest.json"))
    return ",".join(r["path"] for r in meta["inputs"])


def _judged(R):
    return R[(R[:, 4] >= 0) & (R[:, 4] <= max_kf_region())]


def _found(R, key, col=2):
    """Boolean over the census TP keys: the TP of a judged track of the records R (col 2 the seed TP, col 6
    the owner)."""
    J = _judged(R)
    return np.isin(key, J[J[:, col] >= 0, col])


def _ot_entries(C):
    return [i for i, sd in enumerate(C["seeds"]) if sd.layers in SA.OT_CHAIN_SEED and not sd.soft]


def l2l3(census_dir, cmssw, nev=None):
    """L2L3 tracks in KF regions 0-3 (plan sec. 2): none in CMSSW, none in the closed arm."""
    C = TF.load(TF.census_dir(census_dir), nev=nev, with_tracks=True, verbose=False)
    n_cms = int((_judged(_cmssw_tracks(cmssw, _event_ids(_census_inputs(C), C["n_events"])))[:, 1] == 1).sum())
    stubs = _stub_tables(_census_inputs(C), C["n_events"])
    ent = [i for i in _ot_entries(C) if C["seeds"][i].layers == (12, 13)]
    n_cen = sum(len(_judged(R)) for merged in (False, True)
                for R in _census_tracks(C, ent, merged, stubs).values())
    ok = n_cms == 0 and n_cen == 0
    return {"stage": "C0 L2L3", "cmssw_tracks_regions_0_3": n_cms, "census_tracks_regions_0_3": n_cen,
            "gates": {"l2l3_zero": ok}}, ok


# ---- C1 -------------------------------------------------------------------------------------------
def _positions(L, Dt, what):
    """The KF's stub positions, the DTC table's stubPosTT (written with withPositions): x, y, z per L1TOTStub
    row (NaN where the stub has no DTC copy). SystemExit without them: the nano's positions differ from the
    KF's on a few PS stubs, which an identity check cannot tell apart from a fit difference."""
    if "xTT" not in Dt:
        raise SystemExit(f"{what} needs the KF's own stub positions (L1TOTStubDTC xTT / yTT / zTT, the DTC "
                         f"table's withPositions); this input has none")
    x, y, z = (np.full(len(L["layer"]), np.nan) for _ in range(3))
    r = Dt["otStubIdx"].astype(np.int64)
    x[r], y[r], z[r] = Dt["xTT"], Dt["yTT"], Dt["zTT"]
    return x, y, z


def _nodr_share(T, S):
    """The fraction of tracks sharing >= 3 stubs with another track of their sector (NODR_SHARE_MIN)."""
    stubs = [set(S["otStubIdx"][S["trackIdx"] == j].tolist()) for j in range(len(T["rInv"]))]
    owners = {}
    for j, ss in enumerate(stubs):
        for s in ss:
            owners.setdefault((int(T["phiSector"][j]), s), []).append(j)
    n = 0
    for j, ss in enumerate(stubs):
        c = {}
        for s in ss:
            for o in owners[(int(T["phiSector"][j]), s)]:
                if o != j:
                    c[o] = c.get(o, 0) + 1
        n += bool(c) and max(c.values()) >= 3
    return n, len(stubs)


def _cmssw_ints(T, j):
    """The integer parameters (irinv, iphi0, id0, it, iz0) of CMSSW track j from its TTTrack floats."""
    sec = int(T["phiSector"][j])
    k = KF.K
    return (round(float(T["rInv"][j]) / k.krinvpars - 0.5),
            round(KF.reduce_phi(float(T["phi"][j]) - KF.sector_phimin(sec, k)) / k.kphi0pars),
            round(float(T["d0"][j]) / k.kd0pars - 0.5),
            round(float(T["tanL"][j]) / k.ktpars - 0.5),
            round(float(T["z0"][j]) / k.kz0pars - 0.5))


def _prefloor(out, isec):
    """The census's values before HybridFit's floor, in (irinv, iphi0, id0, it, iz0) order and units of LSB."""
    k = KF.K
    return (0.01 * KF.C_TRK * k.b * float(out.q_over_pt) / k.krinvpars,
            KF.reduce_phi(float(out.phi0) - isec * 2 * math.pi / KF.N_SECTOR + 0.5 * k.dphisector_hg) / k.kphi0pars,
            float(out.d0) / k.kd0pars, float(out.tanl) / k.ktpars, float(out.z0) / k.kz0pars)


def _f32_ulps_to_edge(v_lsb, lsb, native):
    """Distance of a census value (in LSB units) to its nearest bin edge, in float32 ulps of the value as the
    fit stores it (native: the physical value)."""
    edge = round(v_lsb)
    return abs(v_lsb - edge) * lsb / max(float(np.spacing(np.float32(abs(native)))), 1e-30)


def c1(nodr, nev=None):
    """The KF unit (plan sec. 3, C1) on a CMSSW no-DR production with the KF's stub positions.

    Every judged CMSSW track (in scope: every seed, stub and KF region with the discs on; barrel seeds and stubs,
    KF regions 0-3 with them off) is one of: identical; a predicted
    edge (every differing parameter differs by exactly 1 LSB and its census value lies within 1e-6 LSB of
    that bin edge); a float32 edge (one parameter, 1 LSB, census value within C1_F32_EDGE_ULP float32 ulps of
    the edge: MINOR, reported, not gated); or a disagreement: different parameters, no forced fit, two stubs
    in a layer, or the census KF's consistentSector rejecting a track CMSSW kept (CMSSW drops those). Only a
    skipped seed stub is an allowed exclusion (<= 2 %)."""
    A = _events(nodr, ("L1TTrack_", "L1TTrackStub_", "L1TOTStub_", "L1TOTStubDTC_"), nev)
    res = {"judged": 0, "identical": 0, "predicted_edge": 0, "float32_edge": 0, "mismatch": 0,
           "no_forced_fit": 0, "two_stubs_in_a_layer": 0, "census_not_consistent_sector": 0,
           "seed_stub_skipped": 0, "chi2_outside": 0, "state_gate_fails": 0}
    examples, shared = [], [0, 0]
    brk = {"seed": {}, "kf_region": {}}

    def tally(sd, kreg, cls):
        for part, k in (("seed", OC.SEED_NAME[sd]), ("kf_region", str(kreg))):
            c = brk[part].setdefault(k, {})
            c[cls] = c.get(cls, 0) + 1

    for e in range(len(A)):
        T, S = _table(A, e, "L1TTrack_"), _table(A, e, "L1TTrackStub_")
        L, Dt = _table(A, e, "L1TOTStub_"), _table(A, e, "L1TOTStubDTC_")
        x, y, z = _positions(L, Dt, "c1")
        n_sh, n_tr = _nodr_share(T, S)
        shared[0] += n_sh
        shared[1] += n_tr
        for j in range(len(T["rInv"])):
            sd = int(T["seedType"][j])
            r = S["otStubIdx"][S["trackIdx"] == j].astype(np.int64)
            if sd not in seed_layers() or not _in_scope(L, r):
                continue
            stubs = [_kf_stub(L, i, x, y, z) for i in r]
            by = {}
            for s in stubs:
                by.setdefault(int(L["layer"][s.idx]), s)
            la, lb = SEED_LAYERS_ALL[sd]
            if la not in by or lb not in by:
                res["seed_stub_skipped"] += 1     # before the region cut: conservative (no seed, no region)
                continue
            isec = int(T["phiSector"][j])
            ga, gb = by[la], by[lb]
            geo = (math.hypot(x[ga.idx], y[ga.idx]), float(z[ga.idx]), math.atan2(y[ga.idx], x[ga.idx]),
                   math.hypot(x[gb.idx], y[gb.idx]), float(z[gb.idx]), math.atan2(y[gb.idx], x[gb.idx]))
            seed = KF.make_seed(*KF.exact_tracklet(*geo, isec), isec, len(stubs))
            kreg = KF.kf_eta_region(seed.eta_reg)
            if kreg > max_kf_region():
                continue
            res["judged"] += 1
            tally(sd, kreg, "judged")
            if len({s.layer for s in stubs}) != len(stubs):
                res["two_stubs_in_a_layer"] += 1
                tally(sd, kreg, "two_stubs_in_a_layer")
                continue
            state, gates, why = KF.forced_fit(seed, stubs)
            if state is None:
                res["no_forced_fit"] += 1
                tally(sd, kreg, "no_forced_fit")
                if len(examples) < 20:
                    examples.append({"event": e, "track": j, "seed": sd, "class": f"no forced fit: {why}"})
                continue
            if not all(gates):
                res["state_gate_fails"] += 1
            out = KF.fitted_output(state, seed)
            if not out.consistent_sector:
                res["census_not_consistent_sector"] += 1
                tally(sd, kreg, "census_not_consistent_sector")
                continue
            mine = np.array(out.track, np.float32)
            theirs = np.array([T["rInv"][j], T["phi"][j], T["tanL"][j], T["z0"][j], T["d0"][j]], np.float32)
            if np.array_equal(mine, theirs):
                res["identical"] += 1
                tally(sd, kreg, "identical")
            else:
                cls, dint = _c1_class(out, T, j, isec)
                res[cls] += 1
                tally(sd, kreg, cls)
                if cls == "mismatch" and len(examples) < 20:
                    examples.append({"event": e, "track": j, "seed": sd, "class": cls,
                                     "int_lsb_diff_irinv_iphi0_id0_it_iz0": dint})
            n = len(stubs)
            c1v, c2v = float(T["chi2XYRed"][j]) * (n - 3), float(T["chi2ZRed"][j]) * (n - 2)
            if (abs(c1v - float(out.chi2rphi)) > C1_CHI2_REL * max(1.0, abs(c1v))
                    or abs(c2v - float(out.chi2rz)) > C1_CHI2_REL * max(1.0, abs(c2v))):
                res["chi2_outside"] += 1
    share = shared[0] / max(shared[1], 1)
    if share < NODR_SHARE_MIN:
        raise SystemExit(f"c1 needs a no-duplicate-removal production: only {share:.2f} of the tracks share 3 "
                         f"or more stubs with another track of their sector (a no-DR production: ~0.96)")
    frac_skip = res["seed_stub_skipped"] / max(res["judged"] + res["seed_stub_skipped"], 1)
    gates = {"identity": res["mismatch"] == 0, "no_forced_fit": res["no_forced_fit"] == 0,
             "two_stubs_in_a_layer": res["two_stubs_in_a_layer"] == 0,
             "consistent_sector": res["census_not_consistent_sector"] == 0, "chi2": res["chi2_outside"] == 0,
             "state_gates": res["state_gate_fails"] == 0, "seed_stub_skipped": bool(frac_skip <= C1_MAX_SKIPPED)}
    return {"stage": "C1", "input": nodr, "positions": "stubPosTT", "discs": OC.DISCS, "nodr_share": share, **res,
            "seed_stub_skipped_fraction": frac_skip, "by_seed": brk["seed"], "by_kf_region": brk["kf_region"],
            "examples": examples, "gates": gates}, all(gates.values())


def _c1_class(out, T, j, isec):
    """(class, integer differences) of a C1 track whose parameters differ (c1's docstring)."""
    mine, theirs = out.ipars, _cmssw_ints(T, j)
    dint = [int(t - m) for t, m in zip(theirs, mine)]
    diff = [i for i, d in enumerate(dint) if d]
    pre = _prefloor(out, isec)
    lsbs = (KF.K.krinvpars, KF.K.kphi0pars, KF.K.kd0pars, KF.K.ktpars, KF.K.kz0pars)
    native = (0.01 * KF.C_TRK * KF.K.b * float(out.q_over_pt), float(out.phi0), float(out.d0),
              float(out.tanl), float(out.z0))
    if diff and all(abs(dint[i]) == 1 and abs(pre[i] - round(pre[i])) < C1_EDGE_LSB for i in diff):
        return "predicted_edge", dint
    if len(diff) == 1 and abs(dint[diff[0]]) == 1 and \
            _f32_ulps_to_edge(pre[diff[0]], lsbs[diff[0]], native[diff[0]]) <= C1_F32_EDGE_ULP:
        return "float32_edge", dint
    return "mismatch", dint


# ---- C1b ------------------------------------------------------------------------------------------
def _candidate_key(sd, sec, rows, L):
    la, lb = SEED_LAYERS_ALL[sd]
    by = {}
    for x in rows:
        by.setdefault(int(L["layer"][x]), x)
    if la not in by or lb not in by:
        return None
    return (sd, int(sec), int(by[la]), int(by[lb]))


def _wilson_lower(k, n, z=1.6449):
    """One-sided 95 % lower bound of a binomial fraction (Wilson)."""
    if n == 0:
        return 0.0
    p = k / n
    c = (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    return max(c, 0.0)


def _min_sample(threshold=C1B_MATCH_MIN):
    """The smallest sample whose Wilson bound passes `threshold` with no mismatch."""
    n = 1
    while _wilson_lower(n, n) < threshold:
        n += 1
    return n


def _align_nodr(cands, nd):
    """CMSSW's no-DR tracks matched to the Fakefit candidates of one event, in order: no-DR fits the track
    builder's candidates in the same order and keeps the accepted ones, so its tracks are a subsequence of
    the candidates, each with the candidate's seed type and sector and stubs among the candidate's.
    Returns (candidate index -> no-DR track index, no-DR tracks left unmatched)."""
    T, S = nd
    nd_rows = [set(S["otStubIdx"][S["trackIdx"] == j].tolist()) for j in range(len(T["rInv"]))]
    match, c = {}, 0
    for j, rows in enumerate(nd_rows):
        while c < len(cands) and not (cands[c][0] == int(T["seedType"][j]) and cands[c][1] == int(T["phiSector"][j])
                                      and rows <= set(cands[c][2])):
            c += 1
        if c == len(cands):
            return match, len(nd_rows) - len(match)
        match[c] = j
        c += 1
    return match, 0


def c1b(fakefit, nodr=None, nev=None, order="dtc"):
    """(i) the emulated candidates against CMSSW's (Fakefit no-DR: every track-builder candidate, its full
    stub list in order and its tracklet's integer parameters); (ii) with the no-DR production of the same
    events, the KF on every CMSSW candidate list in c1's scope against CMSSW's
    no-DR output: both reject, or both accept with the same stubs and parameters; the census accepting what
    CMSSW rejected is the fake diagnostic (those tracks' seed truth is reported).
    Ungated, reported: candidates without an emulated tracklet, emulated candidates without a CMSSW one, the
    tracklet integer parameters. order: the emulation's match order, "dtc" or "random:<seed>" (the known-bad
    that must lower (i))."""
    kind, _, seed = order.partition(":")
    A = _events(fakefit, ("L1TTrack_", "L1TTrackStub_", "L1TOTStub_", "L1TOTStubDTC_"), nev)
    B = _events(nodr, ("L1TTrack_", "L1TTrackStub_"), nev) if nodr else None
    evB = {int(v): i for i, v in enumerate(ak.to_numpy(B["event"]))} if B is not None else {}
    res = {"cmssw_candidates": 0, "emulated_candidates": 0, "same_tracklet": 0, "identical_list": 0,
           "identical_order": 0, "emulation_superset": 0, "emulation_subset": 0, "other": 0,
           "no_emulated_tracklet": 0, "tracklet_without_emulated_candidate": 0,
           "emulated_without_cmssw": 0, "tracklet_pars_identical": 0,
           "kf_judged": 0, "kf_both_reject": 0, "kf_both_accept_identical": 0, "kf_both_accept_different": 0,
           "kf_census_accepts_cmssw_rejects": 0, "kf_census_rejects_cmssw_accepts": 0,
           "kf_census_only_fakes": 0, "nodr_tracks_unaligned": 0}
    examples = []
    group = {g: {"n": 0, "identical": 0} for g in ("barrel", "forward")}
    for e in range(len(A)):
        T, S = _table(A, e, "L1TTrack_"), _table(A, e, "L1TTrackStub_")
        L, Dt = _table(A, e, "L1TOTStub_"), _table(A, e, "L1TOTStubDTC_")
        L["event"] = np.full(len(L["layer"]), e, np.int64)
        x, y, z = _positions(L, Dt, "c1b")
        col = {}
        OC.run_event(L, Dt, collect=col, do_dr=False, order=kind, seed=int(seed or 0) + e)
        cp = col["copies"]
        trk = {(t.seed, t.sector, int(cp.row[t.inner]), int(cp.row[t.outer])): t for t in col["tracklets"]}
        cand = {}
        for c in col["candidates"]:
            t = c.tracklet
            cand[(t.seed, t.sector, int(cp.row[t.inner]), int(cp.row[t.outer]))] = [int(cp.row[q]) for q in c.stubs]
        cms = [(int(T["seedType"][j]), int(T["phiSector"][j]), [int(v) for v in S["otStubIdx"][S["trackIdx"] == j]])
               for j in range(len(T["rInv"]))]
        nd, match = None, {}
        if B is not None and int(A["event"][e]) in evB:
            eb = evB[int(A["event"][e])]
            nd = (_table(B, eb, "L1TTrack_"), _table(B, eb, "L1TTrackStub_"))
            match, left = _align_nodr(cms, nd)
            res["nodr_tracks_unaligned"] += left
        # every CMSSW candidate's key, before the scope's stub cut: with the discs off, an emulated candidate
        # (barrel targets only) of a CMSSW candidate that also holds disc stubs is not "without CMSSW"
        seen = {_candidate_key(sd, sec, r, L) for sd, sec, r in cms if sd in seed_layers()}
        for j, (sd, sec, r) in enumerate(cms):
            if sd not in seed_layers() or not _in_scope(L, r):
                continue
            key = _candidate_key(sd, sec, r, L)
            if key is None:
                continue
            if nd is not None:
                _kf_on_list(res, nd, match.get(j), key, r, L, x, y, z, e)
            res["cmssw_candidates"] += 1
            if key not in trk:
                res["no_emulated_tracklet"] += 1
                continue
            res["same_tracklet"] += 1
            # the forward group (WP-Disc C1b(i), pooled): the 4 new seeds' lists and the barrel seeds' lists with
            # a disc stub on either side; the barrel group: the barrel seeds' other lists (WP-D's C1b(i))
            em = cand.get(key, [])
            fwd = sd not in OC.BARREL_SEEDS or not L["isBarrel"][np.r_[r, em].astype(np.int64)].all()
            grp = group["forward" if fwd else "barrel"]
            grp["n"] += 1
            grp["identical"] += int(key in cand and set(em) == set(r))
            tr = trk[key]
            _, tpars = KF.track_from_ipars(tr.ipars[0], tr.ipars[1], 0, tr.ipars[3], tr.ipars[2], tr.sector)
            theirs = np.array([T["rInv"][j], T["phi"][j], T["tanL"][j], T["z0"][j], T["d0"][j]], np.float32)
            res["tracklet_pars_identical"] += int(np.array_equal(np.array(tpars, np.float32), theirs))
            if key in cand:
                em = cand[key]
                res["identical_order"] += int(em == r)
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
            else:
                res["tracklet_without_emulated_candidate"] += 1
        res["emulated_candidates"] += len(cand)
        res["emulated_without_cmssw"] += len(set(cand) - seen)
    n = res["same_tracklet"]
    nmin = _min_sample()
    if OC.DISCS:
        # WP-Disc C1b(i): the forward group pooled; its shortfall (99 % minus the one-sided bound) below 1 % is
        # MINOR (classified, reported), 1 % or more MATERIAL (the order floor re-run for the new seeds); the
        # barrel group by WP-D's rule
        for v in group.values():
            v["fraction"] = v["identical"] / max(v["n"], 1)
            v["lower95"], v["accepted_by_physicist"], v["passes"] = match_list_gate(v["identical"], v["n"])
        fw = group["forward"]
        fw["shortfall"] = max(C1B_MATCH_MIN - fw["lower95"], 0.0)
        fw["class"] = ("pass" if fw["shortfall"] == 0 else
                       "MINOR" if fw["shortfall"] < C1B_SHORTFALL_MATERIAL else "MATERIAL")
        lo, accepted = fw["lower95"], fw["accepted_by_physicist"]
        gates = {"match_lists": group["barrel"]["passes"] and fw["class"] != "MATERIAL",
                 "match_list_sample": fw["n"] >= nmin and group["barrel"]["n"] >= nmin}
    else:
        lo, accepted, ok_lists = match_list_gate(res["identical_list"], n)
        gates = {"match_lists": ok_lists, "match_list_sample": n >= nmin}
    if nodr:
        bad = (res["kf_both_accept_different"] + res["kf_census_accepts_cmssw_rejects"]
               + res["kf_census_rejects_cmssw_accepts"])
        gates["kf_on_cmssw_list"] = bad == 0 and res["kf_judged"] > 0
        gates["nodr_aligned"] = res["nodr_tracks_unaligned"] == 0
    return {"stage": "C1b", "fakefit": fakefit, "nodr": nodr, "discs": OC.DISCS, **res,
            "groups": group if OC.DISCS else None,
            "identical_list_fraction": res["identical_list"] / max(n, 1), "identical_list_lower95": lo,
            "match_lists_accepted_by_physicist": accepted,
            "match_list_min_sample": nmin, "examples": examples, "gates": gates}, all(gates.values())


def match_list_gate(identical, n):
    """C1b(i): (the one-sided 95 % lower bound of the identical-list fraction, accepted, gate). The gate passes
    when the bound reaches C1B_MATCH_MIN, or by the physicist's acceptance (C1B_ACCEPTED) when the fraction is
    at least the accepted one."""
    lo = _wilson_lower(identical, n)
    passed = bool(lo >= C1B_MATCH_MIN)
    accepted = not passed and n > 0 and identical * C1B_ACCEPTED[1] >= C1B_ACCEPTED[0] * n
    return lo, accepted, passed or accepted


def _kf_on_list(res, nd, jn, key, rows, L, x, y, z, e):
    """The census KF on CMSSW's candidate list `rows` (in its order; the prior exacttracklet on its two seed
    stubs at the KF's positions, as HybridFit's) against CMSSW's no-DR track jn of that candidate (None:
    CMSSW's KF rejected it), for candidates in the scope's KF regions; counts into res."""
    T, S = nd
    sd, sec, a, b = key
    geo = (math.hypot(x[a], y[a]), float(z[a]), math.atan2(y[a], x[a]),
           math.hypot(x[b], y[b]), float(z[b]), math.atan2(y[b], x[b]))
    stubs = [_kf_stub(L, i, x, y, z) for i in rows]
    seed = KF.make_seed(*KF.exact_tracklet(*geo, sec), sec, len(stubs))
    if KF.kf_eta_region(seed.eta_reg) > max_kf_region():
        return
    out = KF.fitted_output(KF.do_kf(seed, stubs), seed)
    acc = bool(out.accepted and out.consistent_sector)
    res["kf_judged"] += 1
    if jn is None:
        if acc:
            res["kf_census_accepts_cmssw_rejects"] += 1
            rec = track_record(L, e, sd, [s.idx for s in out.stubs])
            res["kf_census_only_fakes"] += int(rec[3] == 0)
        else:
            res["kf_both_reject"] += 1
        return
    if not acc:
        res["kf_census_rejects_cmssw_accepts"] += 1
        return
    r = [int(v) for v in S["otStubIdx"][S["trackIdx"] == jn]]
    theirs = np.array([T["rInv"][jn], T["phi"][jn], T["tanL"][jn], T["z0"][jn], T["d0"][jn]], np.float32)
    same = set(s.idx for s in out.stubs) == set(r) and np.array_equal(np.array(out.track, np.float32), theirs)
    res["kf_both_accept_identical" if same else "kf_both_accept_different"] += 1


# ---- C2 / C3 --------------------------------------------------------------------------------------
def _denominator(C):
    """The census's efficiency denominator: findable, pT >= 2 GeV, |eta| < 2.4 with the discs on (0.9 off)."""
    return TF.findable(C) & np.isfinite(C["eta"]) & (np.abs(C["eta"]) < eta_max()) & (C["pt"] >= PT_MIN)


def _per_event(nev, ev, w=None):
    return np.bincount(ev, weights=w, minlength=nev).astype(np.float64)


def _track_counts(R, nev):
    """Per event: judged tracks, fakes (no seed TP), duplicate excess (sum over seed TPs of n - 1), distinct
    seed TPs, and the sheet's fakes (2 or more stubs not crediting the owner)."""
    R = _judged(R)
    ev = R[:, 0]
    n = _per_event(nev, ev)
    fake = _per_event(nev, ev, (R[:, 3] == 0).astype(float))
    t = R[R[:, 2] >= 0]
    keys, cnt = np.unique(t[:, 2], return_counts=True)
    kev = M.tp_key_parts(keys)[0].astype(np.int64)
    dup = _per_event(nev, kev, (cnt - 1).astype(float))
    ntp = _per_event(nev, kev)
    fake_sheet = _per_event(nev, ev, (R[:, 7] >= 2).astype(float))
    return n, fake, dup, ntp, fake_sheet


def _event_mask(nev, exclude):
    """Events kept: all but the ranges of `exclude` ("a-b,c-d", the census's event numbering)."""
    keep = np.ones(nev, bool)
    for part in (exclude or "").split(","):
        if part.strip():
            a, _, b = part.partition("-")
            keep[int(a):int(b or a) + 1] = False
    return keep


# INJECTED KNOWN-BADS (physicist, 2026-10-18; plan sec. 13.4): a statistic no code-level variant moves gets a known
# shift of the census side, in margin units, to show its gate fails a decision-scale change. Drawn with a fixed seed.
INJECT_SEED = 20261018


def _inject_records(R, stat, m, margin, rng):
    """Census records with extra copies: ln_cand, of judged tracks (exp(m margin) - 1 of them); ln_fake, of judged
    fakes (no seed TP); d_dup, of judged seed-truthful tracks (m margin per seed TP)."""
    J = np.flatnonzero((R[:, 4] >= 0) & (R[:, 4] <= max_kf_region()))
    if stat == "ln_cand":
        pool, k = J, len(J) * (math.exp(m * margin) - 1)
    elif stat == "ln_fake":
        pool = J[R[J, 3] == 0]
        k = len(pool) * (math.exp(m * margin) - 1)
    elif stat == "d_dup":
        pool = J[R[J, 2] >= 0]
        k = len(np.unique(R[pool, 2])) * m * margin
    else:
        return R
    # more copies than the pool holds (a gross ln_fake: exp(10 ln 1.25) - 1 = 8.3 per fake): every record of the pool
    # copied the whole number of times, the remainder drawn without replacement (the same draw as before when k
    # does not exceed the pool)
    k = int(round(k))
    whole, rest = divmod(k, len(pool)) if len(pool) else (0, 0)
    extra = np.r_[np.tile(pool, whole), rng.choice(pool, rest, replace=False)].astype(np.int64)
    return np.concatenate([R, R[extra]])


def _inject_found(f_cen, f_cms, den, stat, m, margin, rng):
    """The census's found flags shifted: d_eff, m margin of the denominator's TPs it found are lost; disc, as many
    TPs found by both are lost as TPs found by neither are gained (efficiency unchanged), m margin in all."""
    f = f_cen.copy()
    n = int(den.sum())
    if stat == "d_eff":
        f[rng.choice(np.flatnonzero(f_cen), int(round(m * margin * n)), replace=False)] = False
    elif stat == "disc":
        k = int(round(m * margin * n / 2))
        f[rng.choice(np.flatnonzero(f_cen & f_cms), k, replace=False)] = False
        f[rng.choice(np.flatnonzero(den & ~f_cen & ~f_cms), k, replace=False)] = True
    return f


def _injection(inject, name):
    """{statistic: margins} injected into comparison `name` ("C2 L1L2", "C3 delivered") from "NAME: STAT=M" items."""
    out = {}
    for item in inject or ():
        target, _, m = item.rpartition("=")
        comp, _, stat = target.partition(":")
        if comp.strip() == name:
            out[stat.strip()] = float(m)
    return out


def _census_discs(C):
    """SystemExit unless the census ran with the discs as this closure does (ot_chain.DISCS): its scope, its
    population and its chain must be the closure's (a format-35 census has the discs off)."""
    on = bool(json.load(open(Path(C["cache_dir"]) / "manifest.json"))["config"]["ot_chain"].get("discs", False))
    if on != OC.DISCS:
        raise SystemExit(f"the census ran with the discs {'on' if on else 'off'}, the closure with them "
                         f"{'on' if OC.DISCS else 'off'}: run the closure with --discs {'on' if on else 'off'}")


def acceptance(cmssw_nodr, nev=None):
    """Each seed's acceptance (C2's denominator, ACCEPTANCE_MIN): the KF eta indices holding at least that
    fraction of the seed's CMSSW no-DR tracks of `cmssw_nodr` (run on the known-good file and frozen)."""
    recs = []
    for path in M.expand_inputs(cmssw_nodr):
        A = _events(path, ("L1TTrack_", "L1TTrackStub_", "L1TOTStub_"), nev)
        for e in range(len(A)):
            T, S, L = _table(A, e, "L1TTrack_"), _table(A, e, "L1TTrackStub_"), _table(A, e, "L1TOTStub_")
            for j in range(len(T["rInv"])):
                sd = int(T["seedType"][j])
                if sd in seed_layers():
                    recs.append(track_record(L, e, sd, S["otStubIdx"][S["trackIdx"] == j]))
    R = _records(recs)
    out = {}
    for sd in judged_seeds():
        r = R[(R[:, 1] == sd) & (R[:, 4] >= 0)][:, 4]
        frac = np.bincount(r, minlength=8)[:8] / max(len(r), 1)
        out[OC.SEED_NAME[sd]] = {"kf_indices": [int(k) for k in np.flatnonzero(frac >= ACCEPTANCE_MIN)],
                                 "fractions": [round(float(f), 5) for f in frac], "n_tracks": int(len(r))}
    return {"stage": "C2 acceptance", "input": cmssw_nodr, "min_fraction": ACCEPTANCE_MIN, "seeds": out,
            "gates": {}}, True


def _band_records(R, band):
    lo, hi = BANDS[band]
    return R[(R[:, 4] >= lo) & (R[:, 4] <= hi)]


def c23(census_dir, cmssw_nodr, cmssw_merge, nev=None, exclude=None, power_events=(), inject=(), acc=None):
    """C2 (per seed, no DR) and C3 (delivered, merge DR) of a closed-arm census against CMSSW; `exclude` the
    events left out on both sides; `power_events` the gated sizes the joint power is also computed at; `inject` the
    injected known-bads ("C2 L1L2: d_eff=1.5": that statistic shifted by 1.5 margins on the census side).
    With the discs on (format 36): C2 per seed over its own acceptance (`acc`, an acceptance report: its KF eta
    indices), C3 per band (BANDS: the TPs by truth index, the tracks by KF eta region) and pooled; off: WP-D's
    (C2 over the whole denominator, C3 delivered pooled)."""
    rng = np.random.default_rng(INJECT_SEED)
    C = TF.load(TF.census_dir(census_dir), nev=nev, with_tracks=True, verbose=False)
    _census_discs(C)
    if OC.DISCS and cmssw_nodr and acc is None:
        raise SystemExit("C2 with the discs on needs --acceptance (the seeds' acceptance, frozen on the "
                         "known-good file: closure_vs_cmssw.py acceptance)")
    seed_acc = None if acc is None else {sd: set(acc["seeds"][OC.SEED_NAME[sd]]["kf_indices"])
                                          for sd in judged_seeds()}
    nevc = int(C["n_events"])
    keep = _event_mask(nevc, exclude)
    entries = _ot_entries(C)
    entry = {SA.OT_CHAIN_SEED[C["seeds"][i].layers]: i for i in entries}
    stubs = _stub_tables(_census_inputs(C), nevc)
    ids = _event_ids(_census_inputs(C), nevc)
    key, tev = C["key"], M.tp_key_parts(C["key"])[0].astype(np.int64)
    den = _denominator(C) & keep[tev]
    tpk = tp_kf_index(C["eta"])
    parts = {}
    if cmssw_nodr:
        R_cms = _cmssw_tracks(cmssw_nodr, ids)
        R_cen = _census_tracks(C, entries, False, stubs)
        for sd in judged_seeds():
            i = entry[sd]
            dsd = den & np.isin(tpk, sorted(seed_acc[sd])) if OC.DISCS else den
            inj = _injection(inject, f"C2 {OC.SEED_NAME[sd]}")
            for st, m in inj.items():
                R_cen[i] = _inject_records(R_cen[i], st, m, MARGIN[st], rng)
            f_cen = _found(R_cen[i], key) & dsd
            f_cms = _found(R_cms[R_cms[:, 1] == sd], key) & dsd
            for st, m in inj.items():
                f_cen = _inject_found(f_cen, f_cms, dsd, st, m, MARGIN[st], rng)
            nc = _track_counts(R_cen[i], nevc)[0]
            nm = _track_counts(R_cms[R_cms[:, 1] == sd], nevc)[0]
            parts[f"C2 {OC.SEED_NAME[sd]}"] = {
                "den": _per_event(nevc, tev[dsd]), "cen": _per_event(nevc, tev[f_cen]),
                "cms": _per_event(nevc, tev[f_cms]), "disc": _per_event(nevc, tev[f_cen ^ f_cms]),
                "ncen": nc, "ncms": nm}
    sheet = {}
    if cmssw_merge:
        R_cen0 = np.concatenate(list(_census_tracks(C, entries, True, stubs).values()))
        R_cms0 = _cmssw_tracks(cmssw_merge, ids)
        bands = {"delivered": None} if not OC.DISCS else {"pooled": None, **{b: b for b in BANDS}}
        for name, band in bands.items():
            R_cen = R_cen0 if band is None else _band_records(R_cen0, band)
            R_cms = R_cms0 if band is None else _band_records(R_cms0, band)
            dband = den if band is None else den & (tpk >= BANDS[band][0]) & (tpk <= BANDS[band][1])
            inj = _injection(inject, f"C3 {name}")
            for st, m in inj.items():
                R_cen = _inject_records(R_cen, st, m, MARGIN[st], rng)
            # found: a TP with a judged track anywhere (a band's efficiency counts its TPs, wherever the track's
            # KF region lies; the injected copies of records add no found TP); fakes, duplicates and candidates:
            # the band's tracks
            f_cen = _found(R_cen0, key) & dband
            f_cms = _found(R_cms0, key) & dband
            for st, m in inj.items():
                f_cen = _inject_found(f_cen, f_cms, dband, st, m, MARGIN[st], rng)
            nc, fc, dc, tc, sc = _track_counts(R_cen, nevc)
            nm, fm, dm, tm, sm = _track_counts(R_cms, nevc)
            parts[f"C3 {name}"] = {"den": _per_event(nevc, tev[dband]), "cen": _per_event(nevc, tev[f_cen]),
                                   "cms": _per_event(nevc, tev[f_cms]),
                                   "disc": _per_event(nevc, tev[f_cen ^ f_cms]),
                                   "fcen": fc, "fcms": fm, "dcen": dc, "tcen": tc, "dcms": dm, "tcms": tm,
                                   "ncen": nc, "ncms": nm}
            if band is None:
                o_cen, o_cms = _found(R_cen, key, 6) & dband, _found(R_cms, key, 6) & dband
                sheet = {"owner_found_census": int(o_cen.sum()), "owner_found_cmssw": int(o_cms.sum()),
                         "owner_d_eff": float((o_cen.sum() - o_cms.sum()) / max(dband.sum(), 1)),
                         "sheet_fakes_census": float(sc[keep].sum()), "sheet_fakes_cmssw": float(sm[keep].sum()),
                         "seed_fakes_census": float(fc[keep].sum()), "seed_fakes_cmssw": float(fm[keep].sum()),
                         "judged_census": float(nc[keep].sum()), "judged_cmssw": float(nm[keep].sum())}
    parts = {name: {k: v[keep] for k, v in p.items()} for name, p in parts.items()}
    nkeep = int(keep.sum())
    stats = {name: _stats(p, name.startswith("C3")) for name, p in parts.items()}
    boot = _bootstrap(parts, nkeep)
    gates, table = {}, {}
    for name, st in stats.items():
        for s, v in st.items():
            q5, q95 = np.percentile(boot[name][s], [5, 95])
            m = MARGIN[s]
            ok = bool((q5 > -m and q95 < m) if s in SIGNED else q95 < m)
            table[f"{name}: {s}"] = {"value": v, "q5": float(q5), "q95": float(q95), "margin": m, "pass": ok,
                                     "replicas": boot[name][s].tolist()}
            gates[f"{name}: {s}"] = ok
    power = {str(nkeep): _joint_power(stats, boot)}
    for n_g in power_events:
        power[str(n_g)] = _joint_power(stats, boot, n_g / nkeep)
    return {"stage": "C2 / C3", "census": census_dir, "census_inputs": _census_inputs(C), "discs": OC.DISCS,
            "acceptance": None if acc is None else {k: v["kf_indices"] for k, v in acc["seeds"].items()},
            "cmssw_nodr": cmssw_nodr, "cmssw_merge": cmssw_merge, "n_events": nevc, "excluded_events": exclude,
            "n_events_judged": nkeep, "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "statistics": table,
            "joint_power_by_events": power, "sheet_definitions_ungated": sheet, "injected": list(inject or ()),
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
    """The paired event bootstrap: the same resampled events for the census and CMSSW, every comparison;
    the same replicas for every run of the same size (the fixed seed), so known-bad shifts are paired."""
    rng = np.random.default_rng(seed)
    W = [np.bincount(rng.integers(0, nev, nev), minlength=nev).astype(np.float64) for _ in range(n)]
    out = {}
    for name, p in parts.items():
        reps = [_stats(p, name.startswith("C3"), w) for w in W]
        out[name] = {s: np.array([r[s] for r in reps]) for s in reps[0]}
    return out


def _joint_power(stats, boot, k=1.0):
    """The joint pass probability (plan sec. 3, stats L3) at k times the run's events: each replica, shrunk
    towards the measured value by 1 / sqrt(k), is taken as an observed sample with the bootstrap interval,
    its half-widths scaled by 1 / sqrt(k), around it; the fraction in which every gate passes."""
    names = [(c, s) for c in stats for s in stats[c]]
    if not names:
        return None
    f = 1.0 / math.sqrt(k)
    n = len(next(iter(next(iter(boot.values())).values())))
    ok = np.ones(n, bool)
    for c, s in names:
        b, x = boot[c][s], stats[c][s]
        q5, q95 = np.percentile(b, [5, 95])
        obs = x + (b - x) * f
        lo, hi = obs - (x - q5) * f, obs + (q95 - x) * f
        m = MARGIN[s]
        ok &= ((lo > -m) & (hi < m)) if s in SIGNED else (hi < m)
    return float(ok.mean())


# ---- the order floor --------------------------------------------------------------------------------
def floor(dtc_dir, random_dirs, cmssw_nodr=None, cmssw_merge=None, nev=None, exclude=None, c1b_report=None,
          acc=None):
    """D_order (the mean delivered discordance over the random pairs, jackknifed over runs, with its 80 %
    upper bound) and Delta_dtc (each gated statistic of the dtc run minus the random runs' mean, against
    CMSSW, with the error of that mean) (plan sec. 3, the order floor). Found as the gate counts it (judged
    records). The floor is a gate only when the match lists are not shown to follow CMSSW's order: no
    c1b_report, or one whose match-list gate failed."""
    Cs = [TF.load(TF.census_dir(d), nev=nev, with_tracks=True, verbose=False) for d in random_dirs]
    # the dtc run against the random-order runs: one format
    TF.one_format({**dict(zip(random_dirs, Cs)),
                   dtc_dir: json.loads((Path(TF.census_dir(dtc_dir)) / "manifest.json").read_text())})
    nevc = int(Cs[0]["n_events"])
    keep = _event_mask(nevc, exclude)
    stubs = _stub_tables(_census_inputs(Cs[0]), nevc)
    tev = M.tp_key_parts(Cs[0]["key"])[0].astype(np.int64)
    den = _denominator(Cs[0]) & keep[tev]
    fd = []
    for C in Cs:
        R = np.concatenate(list(_census_tracks(C, _ot_entries(C), True, stubs).values()))
        fd.append(_found(R, C["key"]) & den)
    nr = len(fd)
    pairs = list(itertools.combinations(range(nr), 2))
    disc = {p: float((fd[p[0]] ^ fd[p[1]]).sum()) / max(den.sum(), 1) for p in pairs}
    mean = float(np.mean(list(disc.values()))) if pairs else float("nan")
    jk = [np.mean([v for p, v in disc.items() if k not in p]) for k in range(nr)] if nr > 2 else []
    se = float(math.sqrt((nr - 1) / nr * np.sum((np.array(jk) - np.mean(jk)) ** 2))) if jk else float("nan")
    out = {"stage": "order floor", "dtc": dtc_dir, "random": random_dirs, "excluded_events": exclude,
           "D_order": mean, "D_order_se_jk": se, "D_order_upper80": mean + 0.8416 * se,
           "half_margin_disc": MARGIN["disc"] / 2}
    if cmssw_nodr or cmssw_merge:
        rep = {d: c23(d, cmssw_nodr, cmssw_merge, nev, exclude, acc=acc)[0]["statistics"]
               for d in [dtc_dir] + list(random_dirs)}
        dd = {}
        for s in rep[dtc_dir]:
            r = np.array([rep[d][s]["value"] for d in random_dirs])
            dd[s] = {"delta_dtc": rep[dtc_dir][s]["value"] - float(r.mean()),
                     "sigma": float(r.std(ddof=1) / math.sqrt(len(r))), "half_margin": rep[dtc_dir][s]["margin"] / 2}
        out["delta_dtc"] = dd
    exceeds = out["D_order_upper80"] > out["half_margin_disc"] or any(
        abs(v["delta_dtc"]) + 2 * v["sigma"] > v["half_margin"] for v in out.get("delta_dtc", {}).values())
    c1b_ok = bool(c1b_report) and json.load(open(c1b_report))["gates"].get("match_lists", False)
    out["floor_exceeds_half_margin"] = bool(exceeds)
    out["order_shown_by_c1b"] = c1b_ok
    out["gates"] = {} if c1b_ok else {"floor_within_half_margin": not exceeds}
    return out, c1b_ok or not exceeds


# ---- the truncation decomposition ---------------------------------------------------------------------------
def truncation(caps_on, caps_off):
    """U, the unemulated truncation (plan sec. 2-3): for each statistic, the caps-on arm's value (census caps on
    against production / no-DR) minus the caps-off arm's (census caps off against the no-truncation siblings),
    i.e. CMSSW's truncation cost minus the census's emulated one; its paired sd from the saved replicas (the
    same events and bootstrap) and its one-sided 95 % bound against half the margin. REPORTED, NOT GATED: the
    tracklet processor's clock-level limits are not ported (physicist, 2026-10-16, plan sec. 11), so the caps-on
    arm is not certified and U is the measured cost of the unemulated limits."""
    A, B = json.load(open(caps_on)), json.load(open(caps_off))
    if A["census_inputs"] != B["census_inputs"] or A["n_events"] != B["n_events"] \
            or A["excluded_events"] != B["excluded_events"] or A["boot_seed"] != B["boot_seed"]:
        raise SystemExit("the two arms are not on the same events")
    rows = {}
    for name, a in A["statistics"].items():
        b = B["statistics"][name]
        d = np.array(a["replicas"]) - np.array(b["replicas"])
        u = a["value"] - b["value"]
        bound = float(abs(u) + 1.6449 * np.std(d, ddof=1))
        rows[name] = {"U": u, "paired_sd": float(np.std(d, ddof=1)), "bound95": bound,
                      "half_margin": a["margin"] / 2, "exceeds_half_margin": bound > a["margin"] / 2}
    return {"stage": "truncation decomposition", "caps_on": caps_on, "caps_off": caps_off, "statistics": rows,
            "caps_on_arm": "not certified (physicist, 2026-10-16)", "gates": {}}, True


# ---- known-bad registration -----------------------------------------------------------------------------
def register(good, bad, gross=False):
    """A known-bad candidate (its report `bad`) against the known-good (`good`) on the same events (plan
    sec. 3, stats M2): per gated statistic its shift in margin units with the paired sd (the same bootstrap
    replicas), its gate outcome and the classification: above its margin and passing = a gate defect
    (redesign); below and passing = below resolution; failing = registered (decision-scale when 1-2
    margins). For c1 / c1b the unit is the identity count / the 1 % allowance."""
    G, B = json.load(open(good)), json.load(open(bad))
    if G["stage"] != B["stage"]:
        raise SystemExit(f"{good} is {G['stage']}, {bad} is {B['stage']}")
    rows = {}
    if G["stage"] == "C2 / C3":
        same = all(G.get(k) == B.get(k) for k in ("census_inputs", "n_events", "excluded_events", "boot_seed"))
        if not same:
            raise SystemExit("the two runs are not on the same events (inputs, events, exclusions, bootstrap)")
        for name, g in G["statistics"].items():
            b = B["statistics"][name]
            shift = b["value"] - g["value"]
            sd = float(np.std(np.array(b["replicas"]) - np.array(g["replicas"]), ddof=1))
            units = abs(shift) / g["margin"]
            rows[name] = {"shift": shift, "paired_sd": sd, "shift_margins": units, "known_bad_passes": b["pass"],
                          "class": _kb_class(units, b["pass"])}
    elif G["stage"] == "C1":
        n = B["mismatch"] - G["mismatch"]
        rows["identity"] = {"shift": n, "shift_margins": float(n), "known_bad_passes": B["gates"]["identity"],
                            "class": _kb_class(float(n) if n else 0.0, B["gates"]["identity"])}
    elif G["stage"] == "C1b":
        # with the discs on, (i) is the forward group's (its 1 % shortfall the unit), and (ii) is registered too
        g, b = (G["groups"]["forward"]["fraction"], B["groups"]["forward"]["fraction"]) if G.get("groups") \
            else (G["identical_list_fraction"], B["identical_list_fraction"])
        u = (g - b) / (1 - C1B_MATCH_MIN)
        rows["match_lists"] = {"shift": g - b, "shift_margins": u, "known_bad_passes": B["gates"]["match_lists"],
                               "class": _kb_class(abs(u), B["gates"]["match_lists"])}
        if "kf_on_cmssw_list" in B["gates"]:
            n = sum(B[k] - G[k] for k in ("kf_both_accept_different", "kf_census_accepts_cmssw_rejects",
                                          "kf_census_rejects_cmssw_accepts"))
            rows["kf_on_cmssw_list"] = {"shift": n, "shift_margins": float(n),
                                        "known_bad_passes": B["gates"]["kf_on_cmssw_list"],
                                        "class": _kb_class(float(n), B["gates"]["kf_on_cmssw_list"])}
    else:
        raise SystemExit(f"no registration rule for stage {G['stage']}")
    defects = [k for k, r in rows.items() if r["class"] == "gate defect"]
    out = {"stage": "known-bad registration", "good": good, "bad": bad, "gross": gross, "statistics": rows,
           "fails_somewhere": any(not r["known_bad_passes"] for r in rows.values()),
           "decision_scale_failing": [k for k, r in rows.items() if r["class"] == "fails, decision-scale"],
           "gates": {"no_gate_defect": not defects}}
    return out, not defects


def _kb_class(units, passes):
    if passes:
        return "gate defect" if units > 1.0 else "below resolution"
    return "fails, decision-scale" if 1.0 <= units <= 2.0 else "fails"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("bridge"); p.add_argument("v4"); p.add_argument("v3")
    p.add_argument("--stub-shift", action="store_true", help="the known-bad: canon_v4's stubs moved by an event")
    p = sub.add_parser("twice"); p.add_argument("first"); p.add_argument("second")
    p = sub.add_parser("l2l3"); p.add_argument("census"); p.add_argument("cmssw")
    p = sub.add_parser("c1"); p.add_argument("nodr")
    p = sub.add_parser("acceptance"); p.add_argument("nodr")
    p = sub.add_parser("g0"); p.add_argument("format35"); p.add_argument("format36_discs_off")
    p = sub.add_parser("c1b"); p.add_argument("fakefit"); p.add_argument("--nodr", default=None)
    p.add_argument("--ot-order", default="dtc", help="the emulation's match order: dtc, or random:<seed>")
    for name in ("c23", "floor"):
        p = sub.add_parser(name)
        if name == "c23":
            p.add_argument("census")
            p.add_argument("--power-events", default="",
                           help="gated sizes (events, comma-separated) the joint power is also computed at")
            p.add_argument("--inject", action="append", default=[],
                           help="an injected known-bad, 'C2 L1L2: d_eff=1.5' (that statistic shifted by 1.5 margins "
                                "on the census side); repeatable")
        else:
            p.add_argument("dtc"); p.add_argument("random", nargs="+")
            p.add_argument("--c1b-report", default=None, help="the match-list check's report (c1b -o)")
        p.add_argument("--nodr", default=None, help="CMSSW no-DR production(s), comma-separated as the census (C2)")
        p.add_argument("--merge", default=None, help="CMSSW merge-DR production(s), comma-separated as the census (C3)")
        p.add_argument("--exclude-events", default=None,
                       help="events left out on both sides, 'a-b[,c-d]' in the census's numbering (the gated "
                            "f01 run: 0-9)")
        p.add_argument("--acceptance", default=None,
                       help="the seeds' acceptance report (acceptance -o, frozen on the known-good file): C2's "
                            "per-seed denominators with the discs on")
    p = sub.add_parser("truncation"); p.add_argument("caps_on"); p.add_argument("caps_off")
    p = sub.add_parser("register"); p.add_argument("good"); p.add_argument("bad")
    p.add_argument("--gross", action="store_true", help="the gate's gross known-bad")
    for p in sub.choices.values():
        p.add_argument("-n", "--nev", type=int, default=None)
        p.add_argument("-o", "--out", default=None)
        p.add_argument("--known-bad", default=None,
                       help="run the stage with a known-bad variant of the chain or the fit switched on "
                            "(c1, c1b; the censuses of c23 / floor carry their own)")
        p.add_argument("--discs", choices=("on", "off"), default="on",
                       help="on: the 8 seeds, disc stubs, every KF region (census format 36); off: format 35's "
                            "barrel scope and chain (the discs and the WP-Disc corrections off; gate G0)")
    a = ap.parse_args()
    OC.set_discs(a.discs == "on", fixes=a.discs == "on")
    OC.set_known_bad(a.known_bad)
    if a.stage == "bridge":
        rep, ok = bridge(a.v4, a.v3, a.nev, a.stub_shift)
    elif a.stage == "twice":
        rep, ok = twice(a.first, a.second, a.nev)
    elif a.stage == "l2l3":
        rep, ok = l2l3(a.census, a.cmssw, a.nev)
    elif a.stage == "c1":
        rep, ok = c1(a.nodr, a.nev)
    elif a.stage == "acceptance":
        rep, ok = acceptance(a.nodr, a.nev)
    elif a.stage == "g0":
        rep, ok = g0(a.format35, a.format36_discs_off, a.nev)
    elif a.stage == "c1b":
        rep, ok = c1b(a.fakefit, a.nodr, a.nev, a.ot_order)
    elif a.stage == "c23":
        pe = [int(x) for x in a.power_events.split(",") if x.strip()]
        acc = json.load(open(a.acceptance)) if a.acceptance else None
        rep, ok = c23(a.census, a.nodr, a.merge, a.nev, a.exclude_events, pe, a.inject, acc)
    elif a.stage == "floor":
        acc = json.load(open(a.acceptance)) if a.acceptance else None
        rep, ok = floor(a.dtc, a.random, a.nodr, a.merge, a.nev, a.exclude_events, a.c1b_report, acc)
    elif a.stage == "truncation":
        rep, ok = truncation(a.caps_on, a.caps_off)
    else:
        rep, ok = register(a.good, a.bad, a.gross)
    _fail(rep, a.out, ok)


if __name__ == "__main__":
    main()
