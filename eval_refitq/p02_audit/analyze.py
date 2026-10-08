#!/usr/bin/env python
"""P0.2 refit-v2 audit, step 2: every number from the extract.py cache, with event-level bootstrap errors.

Outputs (OUT_DIR):
  p02_results.json          all tables (value, bootstrap se and variance), definitions, provenance, checks
  p02_reference_per_event.npz  per-event numerators / denominators of H, P and attachment (for paired replay checks)
  p02_harmed_tracks.csv.gz  one row per harmed track: residuals, harm flags, the class / origin of each IT hit in
                            visit order, the surface of the first wrong hit
Definitions: see DEFINITIONS below and preregistration.txt (written before this ran).

    python analyze.py [--reps 400] [--seed 20261009]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import (CORE_MIN_N, SortedSample, per_event, ratio,  # noqa: E402
                        replica_weights, summ)

OUT_DIR = Path("/Volumes/WDMac/smartpixels_scratch/p02_refit_audit")
CACHE = OUT_DIR / "cache"
SCEN = ("AAAA", "AAAI", "IAAA")
FILES = tuple(range(3, 11))
FIT_FILES, TEST_FILES = (3, 4, 5, 6), (7, 8, 9, 10)
EV_PER_FILE = 100

ETA_BINS = ((0.0, 1.1), (1.1, 1.8), (1.8, 2.4))
PT_BINS = ((2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, np.inf))
HF_PT_BINS = ((2.0, 5.0), (5.0, np.inf))
R_PROMPT_CM = 0.01            # 100 um
R_IL1_CM = 2.75               # smallest IL1 sensor radius in the cluster table (canon_v2)
R_OLD_CM = 2.5                # the earlier check's secondary radius, for comparison only
SHARED_FRAC = 0.9             # dominant-TP charge fraction below which a cluster is "shared"
HARM_D0_UM, HARM_Z0_UM = 500.0, 1000.0
D0_THR_UM = (100.0, 200.0, 500.0, 1000.0, 5000.0)
Z0_THR_UM = (500.0, 1000.0, 5000.0)
FB_UNHARMED_REVERT = 0.012
INNERMOST = {"AAAA": 1, "AAAI": 1, "IAAA": 2}
FIRST_VISITED = {"AAAA": 4, "AAAI": 3, "IAAA": 4}
SURFACES = (4, 3, 2, 1, 14, 13, 12, 11)
SURF_NAME = {1: "IL1", 2: "IL2", 3: "IL3", 4: "IL4", 11: "ID1", 12: "ID2", 13: "ID3", 14: "ID4"}

ORIGINS = ("prompt", "hf", "displaced_inner", "beyond_il1")
ORIGIN_LABEL = {"prompt": "prompt (not b/c daughter, r_prod < 100 um)",
                "hf": "b/c-hadron daughter (genFromB or genFromC)",
                "displaced_inner": "other displaced (not b/c, 100 um <= r_prod <= 2.75 cm)",
                "beyond_il1": "born beyond IL1 (not b/c, r_prod > 2.75 cm)",
                "all": "all origins"}
# hit categories (accepted hits on truth-matched tracks)
HC = ("correct", "shared_own", "shared_other", "wrong_T", "wrong_C", "noise")
# wrong-hit origin
WO = ("unlinked", "shared", "same_interaction", "other_interaction")

DEFINITIONS = {
    "population": "truth-matched OT tracks (L1TTrack_tpIdx >= 0, the loosely-genuine TTTrackAssociator TP, which is "
                  "selHitClass's reference) with TP pT >= 2 GeV and |TP eta| < 2.4; passthrough (not refitted) "
                  "tracks included with SP = OT",
    "binning": "TRUE |eta| {0-1.1, 1.1-1.8, 1.8-2.4} x TRUE pT {2-3, 3-5, 5-10, >10} GeV; b/c daughters "
               "|eta|-inclusive in TRUE pT {2-5, >=5} GeV",
    "origin": ORIGIN_LABEL,
    "residual": "reconstructed minus TrackingParticle helix truth at the POCA to the beam line (L1TTP d0, z0; d0 sign "
                "as L1TTrack_d0), in um",
    "core": "B.1: RMS about zero of a mode-seeded iterative 3-sigma clip (census emu_pulls_table core()); "
            f"outside-core = 1 - kept fraction; not quoted below {CORE_MIN_N} tracks",
    "full_rms": "RMS about zero of every residual",
    "harm": "SP worse than OT: |res_SP| - |res_OT| > 0.5 mm (d0) or > 1 mm (z0); 'either' = d0 or z0 (the brief's "
            "harm); 'helped' = the mirror (OT worse than SP by the same margin)",
    "H": "harmed (either) fraction of prompt tracks, per |eta| x pT bin; d0-only and z0-only parts beside it",
    "pull": "residual / sqrt(cov) with each collection's own cov_d0_d0, cov_z0_z0; core by the B.1 estimator; "
            "coverage = fraction with |pull| < 1 (nominal 0.683) and < 2 (0.954)",
    "hit_class": f"accepted IT hit: correct (dominant TP = track TP, charge fraction >= {SHARED_FRAC}); shared_own / "
                 f"shared_other (dominant TP charge fraction < {SHARED_FRAC}; dominant TP = / != track TP); wrong_T "
                 "/ wrong_C (dominant TP another particle, tpEmuClass 1 = class T / 0 = class C); noise (no TP, class "
                 "C). Shared threshold: >= 10% of the charge not from the dominant TP, the complement of the refit's "
                 "clusterMergeFrac = 0.1 (no definition sheet yet)",
    "wrong_hit": "accepted hit whose dominant TP is not the track TP (selHitClass 1 or 2; shared_other included)",
    "first_wrong_hit": "the first wrong accepted hit in the refit's visit order (outside-in)",
    "wrong_hit_origin": "unlinked (noise) / shared (shared_other) / same interaction (dominant TP's L1TTP evt and bx "
                        "equal the track TP's) / other interaction; plus the dominant TP's pT < 0.5 GeV share",
    "P": "wrong accepted hits / accepted hits per surface (population tracks; all origins and prompt-only); "
         "per-crossing variant = wrong accepted hits / crossing records",
    "attachment": "fraction of tracks with a crossing record at the surface whose accepted hit is correct "
                  "(selHitClass 0); at the scenario's innermost active barrel surface (IL1 AAAA / AAAI, IL2 IAAA) "
                  "and at each track's innermost crossing (the last in visit order)",
    "fallback": "chi2 = sum over d0, z0 of (SP - OT)^2 / (sigma_SP^2 + sigma_OT^2), sigma = B.1 core of each arm's "
                "residual on files 3-6 (population tracks, all origins) per RECONSTRUCTED OT |eta| x pT bin; above "
                "the threshold the track takes the OT parameters; threshold = the files 3-6 unharmed tracks' "
                f"{100 * (1 - FB_UNHARMED_REVERT):.1f}% quantile (unharmed-revert fraction {FB_UNHARMED_REVERT}); "
                "quoted on files 7-10",
    "bootstrap": "event-level, multinomial over the 800 events; one replica matrix for all arms and scenarios "
                 "(paired); every core, the fallback sigma table and threshold re-fitted per replica",
}


# ----------------------------------------------------------------------------- loading
def load_scenario(scen):
    """Concatenate files 3-10 of a scenario, tracks ordered by (canonical event, row), hits re-pointed.
    Canonical event index = 100 * (file - 3) + rank of (run, lumi, event) within the file."""
    Ts, Hs, metas, evs, rows = [], [], [], [], []
    toff = 0
    for i, f in enumerate(FILES):
        d = np.load(CACHE / f"p02_{scen}_f{f:02d}.npz")
        metas.append(json.loads(str(d["meta"])))
        E = {k[2:]: d[k] for k in d.files if k.startswith("E_")}
        T = {k[2:]: d[k] for k in d.files if k.startswith("T_")}
        H = {k[2:]: d[k] for k in d.files if k.startswith("H_")}
        n = len(E["event"])
        assert n == EV_PER_FILE
        rank = np.empty(n, np.int64)
        rank[np.lexsort((E["event"], E["lumi"], E["run"]))] = np.arange(n)
        ev = EV_PER_FILE * i + rank[T["entry"]]
        eoff = np.concatenate([[0], np.cumsum(E["ntrk"])])
        row = np.arange(len(T["entry"])) - eoff[T["entry"]]
        H["trk"] = H["trk"] + toff
        toff += len(T["entry"])
        Ts.append(T); Hs.append(H); evs.append(ev); rows.append(row)
    T = {k: np.concatenate([t[k] for t in Ts]) for k in Ts[0]}
    H = {k: np.concatenate([h[k] for h in Hs]) for k in Hs[0]}
    ev, row = np.concatenate(evs), np.concatenate(rows)
    perm = np.lexsort((row, ev))
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    T = {k: v[perm] for k, v in T.items()}
    T["ev"] = ev[perm]
    trk_new = inv[H["trk"]]
    hord = np.argsort(trk_new, kind="stable")      # keeps the visit order within each track
    H = {k: v[hord] for k, v in H.items()}
    H["trk"] = trk_new[hord]
    return T, H, metas


# ----------------------------------------------------------------------------- helpers
class Ctx:
    def __init__(self, W, n_events):
        self.W, self.nev = W, n_events

    def frac(self, ev, ind):
        return ratio(per_event(ev, ind, self.nev), per_event(ev, np.ones(len(ev)), self.nev), self.W)

    def core(self, x, ev):
        S = SortedSample(x, ev)
        return np.array(S.core()), S.core_reps(self.W), S.n


def s_of(v, reps):
    return summ(v, reps)


def core_block(ctx, x, ev):
    nom, reps, n = ctx.core(x, ev)
    names = ("core", "centre", "sigma", "outside_core")
    return {nm: s_of(nom[i], reps[:, i]) for i, nm in enumerate(names)}, nom, reps


def resolution_block(ctx, res, pull, ev, thr_list):
    """T2 FoMs of one arm and parameter on one cell; returns (dict, replica store)."""
    out, store = {}, {}
    blk, nom, reps = core_block(ctx, res, ev)
    out.update(blk)
    store["core"] = (nom[0], reps[:, 0])
    store["outside_core"] = (nom[3], reps[:, 3])
    v, r = ratio(per_event(ev, res * res, ctx.nev), per_event(ev, np.ones(len(ev)), ctx.nev), ctx.W)
    out["full_rms"] = s_of(np.sqrt(v), np.sqrt(r))
    store["full_rms"] = (np.sqrt(v), np.sqrt(r))
    for t in thr_list:
        v, r = ctx.frac(ev, np.abs(res) > t)
        key = f"frac_gt_{t / 1000:g}mm"
        out[key] = s_of(v, r)
        store[key] = (v, r)
    ok = np.isfinite(pull)
    pblk, _, _ = core_block(ctx, pull[ok], ev[ok])
    out["pull_core"] = pblk["core"]
    out["pull_outside_core"] = pblk["outside_core"]
    for k in (1, 2):
        v, r = ctx.frac(ev[ok], np.abs(pull[ok]) < k)
        out[f"pull_cov{k}"] = s_of(v, r)
    out["n_pull"] = int(ok.sum())
    return out, store


def diff_block(a, b):
    """paired a - b over the shared replica stores."""
    return {k: s_of(a[k][0] - b[k][0], a[k][1] - b[k][1]) for k in a if k in b}


def bin_cells(T, pop):
    """{cell key: (mask, meta)} for the T2 tables."""
    pt, aeta, org = T["tp_pt"], np.abs(T["tp_eta"]), T["origin"]
    cells = {}
    for o in ORIGINS + ("all",):
        om = pop if o == "all" else pop & (org == ORIGINS.index(o))
        cells[f"{o}|incl"] = (om, {"origin": o, "eta": None, "pt": None})
        if o == "hf":
            for lo, hi in HF_PT_BINS:
                cells[f"hf|eta0-2.4|pt{lo:g}-{hi:g}"] = (om & (pt >= lo) & (pt < hi),
                                                        {"origin": o, "eta": (0.0, 2.4), "pt": (lo, hi)})
            continue
        if o == "all":
            continue
        for elo, ehi in ETA_BINS:
            em = om & (aeta >= elo) & (aeta < ehi)
            cells[f"{o}|eta{elo:g}-{ehi:g}|incl"] = (em, {"origin": o, "eta": (elo, ehi), "pt": None})
            for lo, hi in PT_BINS:
                cells[f"{o}|eta{elo:g}-{ehi:g}|pt{lo:g}-{hi:g}"] = (em & (pt >= lo) & (pt < hi),
                                                                    {"origin": o, "eta": (elo, ehi), "pt": (lo, hi)})
    return cells


# ----------------------------------------------------------------------------- hits
def hit_tables(T, H):
    """per-hit category / wrong / origin and per-track summaries."""
    ntrk = len(T["ev"])
    trk = H["trk"]
    acc = H["hitAccepted"].astype(bool)
    cls = H["selHitClass"]
    frac = H["cl_frac"]
    emu = H["cl_emu"]
    shared = acc & (cls != 2) & (frac < SHARED_FRAC)
    cat = np.full(len(trk), -1, np.int8)
    cat[acc & (cls == 0) & ~shared] = 0
    cat[acc & (cls == 0) & shared] = 1
    cat[acc & (cls == 1) & shared] = 2
    cat[acc & (cls == 1) & ~shared & (emu == 1)] = 3
    cat[acc & (cls == 1) & ~shared & (emu != 1)] = 4
    cat[acc & (cls == 2)] = 5
    assert not np.any(acc & (cat < 0))
    wrong = acc & (cls != 0)
    same = (H["cltp_evt"] == T["tp_evt"][trk]) & (H["cltp_bx"] == T["tp_bx"][trk])
    worig = np.full(len(trk), -1, np.int8)
    worig[wrong & (cls == 2)] = 0
    worig[wrong & (cls == 1) & shared] = 1
    worig[wrong & (cls == 1) & ~shared & same] = 2
    worig[wrong & (cls == 1) & ~shared & ~same] = 3
    lowpt = wrong & (cls == 1) & (H["cltp_pt"] < 0.5)

    tr = {}
    for k, nm in enumerate(HC):
        tr[f"n_{nm}"] = np.bincount(trk, weights=(cat == k), minlength=ntrk)
    tr["n_wrong"] = np.bincount(trk, weights=wrong, minlength=ntrk)
    tr["n_acc"] = np.bincount(trk, weights=acc, minlength=ntrk)
    tr["n_cross"] = np.bincount(trk, minlength=ntrk).astype(float)
    wi = np.flatnonzero(wrong)
    u, first = np.unique(trk[wi], return_index=True)
    fw = wi[first]
    for nm, src, fill in (("fw_surface", H["layer"], 0), ("fw_cat", cat, -1), ("fw_orig", worig, -1)):
        a = np.full(ntrk, fill, np.int16)
        a[u] = src[fw]
        tr[nm] = a
    a = np.zeros(ntrk, bool)
    a[u] = lowpt[fw]
    tr["fw_lowpt"] = a
    # innermost crossing = last record per track; scenario-surface crossing classes
    last = np.flatnonzero(np.r_[trk[1:] != trk[:-1], True])
    inn = np.full(ntrk, -9, np.int16)
    inn_ok = np.zeros(ntrk, bool)
    inn[trk[last]] = H["layer"][last]
    inn_ok[trk[last]] = acc[last] & (cls[last] == 0)
    tr["inn_surface"], tr["inn_correct"] = inn, inn_ok
    for s in SURFACES:
        hs = H["layer"] == s
        has = np.zeros(ntrk, bool); has[trk[hs]] = True
        okc = np.zeros(ntrk, bool); okc[trk[hs & acc & (cls == 0)]] = True
        wr = np.zeros(ntrk, bool); wr[trk[hs & wrong]] = True
        tr[f"cross_{s}"], tr[f"correct_{s}"], tr[f"wrong_{s}"] = has, okc, wr
    hit = {"cat": cat, "wrong": wrong, "worig": worig, "lowpt": lowpt, "acc": acc, "shared": shared}
    return hit, tr


# ----------------------------------------------------------------------------- fallback
def reco_bin(T):
    aeta = np.abs(T["ot_eta"])
    ie = np.clip(np.searchsorted([b[1] for b in ETA_BINS], aeta, side="right"), 0, len(ETA_BINS) - 1)
    ip = np.clip(np.searchsorted([b[1] for b in PT_BINS], T["ot_pt"], side="right"), 0, len(PT_BINS) - 1)
    return ie * len(PT_BINS) + ip


def wquantile(x, w, q):
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    i = int(np.searchsorted(cw, q * cw[-1], side="left"))
    return x[o][min(i, len(x) - 1)]


class SigmaTable:
    """B.1 core per reco bin and parameter for one arm (res: dict param -> um) on the tracks `sel`,
    pre-sorted once; ``at(w_ev)`` re-fits every bin with event weights w_ev."""

    def __init__(self, res, rb, ev, sel):
        nb = len(ETA_BINS) * len(PT_BINS)
        self.samples = {p: [SortedSample(res[p][sel & (rb == b)], ev[sel & (rb == b)]) for b in range(nb)]
                        for p in ("d0", "z0")}

    def at(self, w_ev):
        from estimators import core_sorted
        # every bin is quotable on the sample (asserted in fallback()); replicas are always re-fitted
        return {p: np.array([core_sorted(S.xs, w_ev[S.es], min_n=0 if S.n >= CORE_MIN_N else CORE_MIN_N)[0]
                             for S in L]) for p, L in self.samples.items()}


def fb_chi2(res_sp, res_ot, rb, tab_sp, tab_ot):
    c = np.zeros(len(rb))
    for p in ("d0", "z0"):
        d = res_sp[p] - res_ot[p]          # = SP - OT parameter difference (truth cancels)
        c += d * d / (tab_sp[p][rb] ** 2 + tab_ot[p][rb] ** 2)
    return c


def clean(o):
    """JSON-safe copy: numpy scalars to Python, non-finite floats (open bin edges, empty cells) to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ----------------------------------------------------------------------------- provenance
def git(*a, cwd):
    return subprocess.run(["git", "--no-optional-locks", *a], cwd=cwd, capture_output=True, text=True).stdout


def provenance(metas_all, t0):
    here = Path(__file__).resolve().parent
    repo = Path(git("rev-parse", "--show-toplevel", cwd=here).strip())
    commit = git("rev-parse", "HEAD", cwd=repo).strip()
    status = git("status", "--porcelain", "--untracked-files=no", cwd=repo)
    patch = git("diff", "HEAD", cwd=repo)
    mine = sorted(str(p.relative_to(repo)) for p in here.glob("*.py"))
    mine_status = git("status", "--porcelain", "--", *mine, cwd=repo)
    (OUT_DIR / "provenance_tree.patch").write_text(patch)
    files = {}
    for m in metas_all:
        files[Path(m["input"]).name] = {"sha256": m["sha256"], "size": m["size"], "mtime": m["mtime"],
                                        "n_events": m["n_events"]}
    code_sha = {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in mine}
    import awkward, uproot
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "runtime_s": round(time.time() - t0, 1),
        "inputs": files, "input_build": "CMSSW 92bad91b2ba (canon_v2)", "cache_dir": str(CACHE),
        "code_repo": str(repo), "code_commit": commit, "code_files_sha256": code_sha,
        "code_files_uncommitted": mine_status.strip() or "none (the audit's files are committed as run)",
        "tree_status_tracked": status.strip().splitlines(),
        "tree_patch_file": str(OUT_DIR / "provenance_tree.patch"),
        "tree_patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "tree_patch_note": "git diff HEAD of the shared tree at run time; none of the modified files is imported "
                           "by the audit (it imports numpy, and ngtagger.data.spix_nano in extract.py only)",
        "python": sys.version.split()[0], "numpy": np.__version__, "awkward": awkward.__version__,
        "uproot": uproot.__version__, "host": platform.node(),
        "preregistration": (OUT_DIR / "preregistration.txt").read_text(),
        "status": "refit-v2 (canon_v2, build 92bad91b2ba); measurement, no gate",
        "bias": "refit v2 harm: correct for v2, pessimistic as a bound on SmartPixels (plan B.3); truth = the OT "
                "track's loosely-genuine TP (a track whose OT stubs are mostly another particle's is not in the "
                "population); emulated angles, ideal readout, float refit",
    }


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()
    t0 = time.time()
    nev = EV_PER_FILE * len(FILES)
    W = replica_weights(nev, args.reps, args.seed)
    ctx = Ctx(W, nev)
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)

    data, metas_all = {}, []
    for s in SCEN:
        T, H, metas = load_scenario(s)
        data[s] = (T, H)
        metas_all += metas
        log(f"loaded {s}: {len(T['ev'])} tracks, {len(H['trk'])} crossings")
    T0 = data["AAAA"][0]
    checks = {}
    same_ot = all(np.array_equal(T0[k], data[s][0][k]) for s in SCEN[1:]
                  for k in ("ev", "ot_d0", "ot_z0", "ot_pt", "ot_eta", "ot_tpIdx", "tp_d0", "tp_z0", "ot_cov_d0_d0"))
    checks["ot_and_truth_identical_across_scenarios"] = bool(same_ot)
    assert same_ot
    checks["iiii_f03_ot_equals_aaaa_f03_ot"] = "checked 2026-10-09 (L1TTrack d0, z0, event order; IIII SP == OT)"

    # ---- population and origin (identical for every scenario)
    r = np.hypot(T0["tp_vx"], T0["tp_vy"])
    hf = (T0["tp_genFromB"] | T0["tp_genFromC"]).astype(bool)
    matched = T0["ot_tpIdx"] >= 0
    pop = matched & (T0["tp_pt"] >= 2.0) & (np.abs(T0["tp_eta"]) < 2.4)
    org = np.full(len(r), -1, np.int8)
    org[~hf & (r < R_PROMPT_CM)] = 0
    org[hf] = 1
    org[~hf & (r >= R_PROMPT_CM) & (r <= R_IL1_CM)] = 2
    org[~hf & (r > R_IL1_CM)] = 3
    for s in SCEN:
        data[s][0]["origin"] = org
    T0["origin"] = org
    ev = T0["ev"]
    fit_ev, test_ev = ev < EV_PER_FILE * len(FIT_FILES), ev >= EV_PER_FILE * len(FIT_FILES)
    popinfo = {"n_tracks": int(len(r)), "n_truth_matched": int(matched.sum()), "n_population": int(pop.sum()),
               "excluded_tp_pt_lt_2": int((matched & (T0["tp_pt"] < 2)).sum()),
               "excluded_abs_eta_ge_2.4": int((matched & (T0["tp_pt"] >= 2) & (np.abs(T0["tp_eta"]) >= 2.4)).sum()),
               "per_origin": {o: int((pop & (org == i)).sum()) for i, o in enumerate(ORIGINS)},
               "hard_interaction_share": float(((T0["tp_evt"] == 0) & (T0["tp_bx"] == 0))[pop].mean()),
               "hf_beyond_il1": int((pop & hf & (r > R_IL1_CM)).sum()),
               "n_events": nev}

    # ---- residuals (um) and pulls
    def arm(Tt, pre):
        out = {}
        for p in ("d0", "z0"):
            res = (Tt[f"{pre}_{p}"] - T0[f"tp_{p}"]).astype(np.float64) * 1e4
            cov = Tt[f"{pre}_cov_{p}_{p}"].astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                pull = np.where(cov > 0, res / (np.sqrt(cov) * 1e4), np.nan)
            out[p] = (res, pull)
        return out
    arms = {"OT": arm(T0, "ot")}
    for s in SCEN:
        arms[s] = arm(data[s][0], "sp")
    harm = {}
    for s in SCEN:
        hd0 = np.abs(arms[s]["d0"][0]) - np.abs(arms["OT"]["d0"][0]) > HARM_D0_UM
        hz0 = np.abs(arms[s]["z0"][0]) - np.abs(arms["OT"]["z0"][0]) > HARM_Z0_UM
        gd0 = np.abs(arms["OT"]["d0"][0]) - np.abs(arms[s]["d0"][0]) > HARM_D0_UM
        gz0 = np.abs(arms["OT"]["z0"][0]) - np.abs(arms[s]["z0"][0]) > HARM_Z0_UM
        harm[s] = {"d0": hd0, "z0": hz0, "either": hd0 | hz0, "helped_d0": gd0, "helped_z0": gz0}

    # ---- T2 per cell
    cells = bin_cells(T0, pop)
    t2 = {}
    for key, (m, meta) in cells.items():
        e = ev[m]
        cell = {"meta": meta, "n": int(m.sum()), "arms": {}, "harm": {}, "sp_minus_ot": {}}
        stores = {}
        for a in ("OT",) + SCEN:
            cell["arms"][a] = {}
            for p, thr in (("d0", D0_THR_UM), ("z0", Z0_THR_UM)):
                res, pull = arms[a][p]
                blk, st = resolution_block(ctx, res[m], pull[m], e, thr)
                cell["arms"][a][p] = blk
                stores[(a, p)] = st
        for s in SCEN:
            cell["harm"][s] = {k: s_of(*ctx.frac(e, v[m])) for k, v in harm[s].items()}
            cell["sp_minus_ot"][s] = {p: diff_block(stores[(s, p)], stores[("OT", p)]) for p in ("d0", "z0")}
        cell["arms"]["OT"]["note"] = "OT = the IIII baseline (identical OT tracks in every scenario)"
        t2[key] = cell
    log(f"T2: {len(t2)} cells")

    # ---- per scenario: hits, attribution, reference numbers, benefit, fallback
    rb = reco_bin(T0)
    per_ev_ref = {}
    harmed_rows = []
    scen_out = {}
    for s in SCEN:
        T, H = data[s]
        hit, tr = hit_tables(T, H)
        hs = harm[s]
        out = {"refit_fraction": s_of(*ctx.frac(ev[pop], T["sp_spixRefitPerformed"][pop].astype(bool))),
               "tracks_with_crossing": s_of(*ctx.frac(ev[pop], tr["n_cross"][pop] > 0))}
        # --- attribution of harmed tracks
        attr = {}
        groups = {"prompt": pop & (org == 0), "hf": pop & (org == 1), "displaced_inner": pop & (org == 2),
                  "beyond_il1": pop & (org == 3), "all": pop}
        for elo, ehi in ETA_BINS:
            groups[f"prompt|eta{elo:g}-{ehi:g}"] = groups["prompt"] & (np.abs(T0["tp_eta"]) >= elo) & (np.abs(T0["tp_eta"]) < ehi)
        for g, gm in groups.items():
            ga = {}
            for hk in ("either", "d0", "z0"):
                hm = gm & hs[hk]
                e = ev[hm]
                blk = {"n_harmed": int(hm.sum())}
                if hm.sum() == 0:
                    ga[hk] = blk
                    continue
                blk["with_wrong_hit"] = s_of(*ctx.frac(e, tr["n_wrong"][hm] > 0))
                blk["with_wrong_T"] = s_of(*ctx.frac(e, tr["n_wrong_T"][hm] > 0))
                blk["with_wrong_C_or_noise"] = s_of(*ctx.frac(e, (tr["n_wrong_C"][hm] + tr["n_noise"][hm]) > 0))
                blk["with_shared_other"] = s_of(*ctx.frac(e, tr["n_shared_other"][hm] > 0))
                fs = tr["fw_surface"][hm]
                blk["first_wrong_surface_share"] = {SURF_NAME[x]: s_of(*ctx.frac(e, fs == x)) for x in SURFACES}
                blk["first_wrong_surface_share"]["none"] = s_of(*ctx.frac(e, fs == 0))
                blk["any_wrong_at_surface_share"] = {SURF_NAME[x]: s_of(*ctx.frac(e, tr[f"wrong_{x}"][hm]))
                                                     for x in SURFACES}
                fc = tr["fw_cat"][hm]
                hw = fs > 0
                blk["first_wrong_class_share"] = {nm: s_of(*ctx.frac(e[hw], fc[hw] == k))
                                                  for k, nm in enumerate(HC) if nm not in ("correct", "shared_own")}
                fo = tr["fw_orig"][hm]
                blk["first_wrong_origin_share"] = {nm: s_of(*ctx.frac(e[hw], fo[hw] == k)) for k, nm in enumerate(WO)}
                blk["first_wrong_tp_pt_lt_0p5"] = s_of(*ctx.frac(e[hw], tr["fw_lowpt"][hm][hw]))
                # hit-level composition on harmed vs unharmed tracks of the group
                for lab, tm in (("harmed", hm), ("unharmed", gm & ~hs[hk])):
                    et = ev[tm]
                    den = per_event(et, tr["n_acc"][tm], nev)
                    blk[f"hit_class_share_{lab}"] = {nm: s_of(*ratio(per_event(et, tr[f"n_{nm}"][tm], nev), den, W))
                                                     for nm in HC}
                # all wrong hits on harmed tracks: origin shares
                hsel = np.isin(H["trk"], np.flatnonzero(hm)) & hit["wrong"]
                eh = ev[H["trk"][hsel]]
                blk["all_wrong_hits_origin_share"] = {nm: s_of(*ctx.frac(eh, hit["worig"][hsel] == k))
                                                      for k, nm in enumerate(WO)}
                blk["all_wrong_hits_tp_pt_lt_0p5"] = s_of(*ctx.frac(eh, hit["lowpt"][hsel]))
                blk["n_wrong_hits"] = int(hsel.sum())
                if g == "prompt" and hk == "either":
                    wsel = hsel & (H["selHitClass"] == 1)
                    blk["shared_threshold_sensitivity"] = {
                        f"dominant_charge_frac_lt_{t:g}": s_of(*ctx.frac(ev[H["trk"][wsel]], H["cl_frac"][wsel] < t))
                        for t in (0.5, 0.7, 0.9)}
                ga[hk] = blk
            # harm rate with vs without a wrong hit
            for lab, wm in (("with_wrong_hit", tr["n_wrong"] > 0), ("without_wrong_hit", tr["n_wrong"] == 0)):
                mm = gm & wm & (tr["n_acc"] > 0)
                ga[f"harm_rate_{lab}"] = s_of(*ctx.frac(ev[mm], hs["either"][mm]))
            ga["n_tracks"] = int(gm.sum())
            attr[g] = ga
        out["attribution"] = attr
        m25 = pop & ~hf & (r >= R_OLD_CM)
        out["comparison_nonHF_r_ge_2.5cm"] = {"n": int(m25.sum()),
                                             **{k: s_of(*ctx.frac(ev[m25], hs[k][m25])) for k in ("d0", "z0", "either")}}

        # --- reference numbers for the replay mirror check (E5)
        ref = {"H": {}, "P": {}, "P_prompt": {}, "P_per_crossing": {}, "empty_window_fraction": {}}
        pr = pop & (org == 0)
        for elo, ehi in ETA_BINS:
            for lo, hi in PT_BINS:
                m = pr & (np.abs(T0["tp_eta"]) >= elo) & (np.abs(T0["tp_eta"]) < ehi) & (T0["tp_pt"] >= lo) & (T0["tp_pt"] < hi)
                k = f"eta{elo:g}-{ehi:g}|pt{lo:g}-{hi:g}"
                ref["H"][k] = {hk: s_of(*ctx.frac(ev[m], hs[hk][m])) for hk in ("either", "d0", "z0")}
                ref["H"][k]["n"] = int(m.sum())
                for hk in ("either", "d0", "z0"):
                    per_ev_ref[f"{s}/H_{hk}/{k}/num"] = per_event(ev[m], hs[hk][m], nev)
                per_ev_ref[f"{s}/H/{k}/den"] = per_event(ev[m], np.ones(m.sum()), nev)
        hpop = pop[H["trk"]]
        hprompt = (pop & (org == 0))[H["trk"]]
        hev = ev[H["trk"]]
        for x in SURFACES:
            at = H["layer"] == x
            if not at.any():
                continue
            nm = SURF_NAME[x]
            for lab, hm in (("P", hpop), ("P_prompt", hprompt)):
                sel = hm & at & hit["acc"]
                num, den = per_event(hev[sel], hit["wrong"][sel], nev), per_event(hev[sel], np.ones(sel.sum()), nev)
                ref[lab][nm] = s_of(*ratio(num, den, W))
                ref[lab][nm]["n_accepted"] = int(sel.sum())
                per_ev_ref[f"{s}/{lab}/{nm}/num"], per_ev_ref[f"{s}/{lab}/{nm}/den"] = num, den
            sel = hpop & at
            ref["P_per_crossing"][nm] = s_of(*ctx.frac(hev[sel], hit["wrong"][sel]))
            ref["empty_window_fraction"][nm] = s_of(*ctx.frac(hev[sel], ~hit["acc"][sel]))
        ref["first_visited_barrel_surface"] = SURF_NAME[FIRST_VISITED[s]]
        out["reference_E5"] = ref

        # --- benefit baselines
        ben = {"attach_innermost_active": {}, "attach_track_innermost": {}, "attach_per_surface": {},
               "within_100um_d0": {}}
        si = INNERMOST[s]
        ben["innermost_active_surface"] = SURF_NAME[si]
        bcells = {f"{o}|incl": pop & (org == i) for i, o in enumerate(ORIGINS)}
        bcells["all|incl"] = pop
        for elo, ehi in ETA_BINS:
            for lo, hi in PT_BINS:
                bcells[f"prompt|eta{elo:g}-{ehi:g}|pt{lo:g}-{hi:g}"] = (
                    pr & (np.abs(T0["tp_eta"]) >= elo) & (np.abs(T0["tp_eta"]) < ehi) & (T0["tp_pt"] >= lo) & (T0["tp_pt"] < hi))
            bcells[f"prompt|eta{elo:g}-{ehi:g}|incl"] = pr & (np.abs(T0["tp_eta"]) >= elo) & (np.abs(T0["tp_eta"]) < ehi)
        for k, m in bcells.items():
            mc = m & tr[f"cross_{si}"]
            ben["attach_innermost_active"][k] = s_of(*ctx.frac(ev[mc], tr[f"correct_{si}"][mc]))
            ben["attach_innermost_active"][k]["n"] = int(mc.sum())
            ben["attach_innermost_active"][k]["crossing_fraction"] = s_of(*ctx.frac(ev[m], tr[f"cross_{si}"][m]))
            mi = m & (tr["n_cross"] > 0)
            ben["attach_track_innermost"][k] = s_of(*ctx.frac(ev[mi], tr["inn_correct"][mi]))
            ben["attach_track_innermost"][k]["n"] = int(mi.sum())
            ben["within_100um_d0"][k] = {"SP": s_of(*ctx.frac(ev[m], np.abs(arms[s]["d0"][0][m]) < 100.0)),
                                         "OT": s_of(*ctx.frac(ev[m], np.abs(arms["OT"]["d0"][0][m]) < 100.0))}
            if k.startswith("prompt|eta") and "|pt" in k:
                per_ev_ref[f"{s}/attach_innermost/{k}/num"] = per_event(ev[mc], tr[f"correct_{si}"][mc], nev)
                per_ev_ref[f"{s}/attach_innermost/{k}/den"] = per_event(ev[mc], np.ones(mc.sum()), nev)
                per_ev_ref[f"{s}/within100/{k}/num"] = per_event(ev[m], np.abs(arms[s]["d0"][0][m]) < 100.0, nev)
                per_ev_ref[f"{s}/within100/{k}/den"] = per_event(ev[m], np.ones(m.sum()), nev)
        for x in SURFACES:
            mc = pr & tr[f"cross_{x}"]
            if mc.sum():
                ben["attach_per_surface"][SURF_NAME[x]] = s_of(*ctx.frac(ev[mc], tr[f"correct_{x}"][mc]))
                ben["attach_per_surface"][SURF_NAME[x]]["n"] = int(mc.sum())
        ben["P_disc"] = {k: v for k, v in ref["P"].items() if k.startswith("ID")}
        out["benefit"] = ben

        # --- fallback ROC
        out["fallback"] = fallback(ctx, s, T0, arms, hs, pop, org, rb, ev, fit_ev, test_ev, data[s][0])
        log(f"{s}: attribution, reference, benefit, fallback done")

        # --- harmed-track list
        hm_any = pop & hs["either"]
        idx = np.flatnonzero(hm_any)
        hpos = np.searchsorted(H["trk"], idx, side="left")
        hend = np.searchsorted(H["trk"], idx, side="right")
        cat_short = {0: "ok", 1: "shOwn", 2: "shOther", 3: "wT", 4: "wC", 5: "noise", -1: "none"}
        orig_short = {0: "unlinked", 1: "shared", 2: "sameInt", 3: "otherInt", -1: ""}
        for t, a0, a1 in zip(idx, hpos, hend):
            hits = []
            for j in range(a0, a1):
                c = cat_short[int(hit["cat"][j])] if hit["acc"][j] else "empty"
                o = orig_short[int(hit["worig"][j])] if hit["wrong"][j] else ""
                hits.append(f"{SURF_NAME[int(H['layer'][j])]}:{c}" + (f"/{o}" if o else ""))
            harmed_rows.append({
                "scenario": s, "file": int(3 + ev[t] // EV_PER_FILE), "event_index": int(ev[t]),
                "origin": ORIGINS[org[t]], "tp_pt": round(float(T0["tp_pt"][t]), 3),
                "tp_eta": round(float(T0["tp_eta"][t]), 3), "r_prod_cm": round(float(r[t]), 4),
                "res_d0_SP_um": round(float(arms[s]["d0"][0][t]), 1), "res_d0_OT_um": round(float(arms["OT"]["d0"][0][t]), 1),
                "res_z0_SP_um": round(float(arms[s]["z0"][0][t]), 1), "res_z0_OT_um": round(float(arms["OT"]["z0"][0][t]), 1),
                "harm_d0": int(hs["d0"][t]), "harm_z0": int(hs["z0"][t]),
                "first_wrong_surface": SURF_NAME.get(int(tr["fw_surface"][t]), "none"),
                "first_wrong_class": cat_short[int(tr["fw_cat"][t])] if tr["fw_surface"][t] else "none",
                "first_wrong_origin": orig_short[int(tr["fw_orig"][t])] if tr["fw_surface"][t] else "",
                "hits_visit_order": " ".join(hits)})
        scen_out[s] = out

    result = {"definitions": DEFINITIONS, "population": popinfo, "checks": checks,
              "bootstrap": {"replicas": args.reps, "seed": args.seed, "unit": "event", "n_events": nev},
              "t2": t2, "scenarios": scen_out}
    result["provenance"] = provenance(metas_all, t0)
    with open(OUT_DIR / "p02_results.json", "w") as fh:
        json.dump(clean(result), fh, indent=1, allow_nan=False)
    np.savez_compressed(OUT_DIR / "p02_reference_per_event.npz",
                        event_index_note=np.array("index = 100 * (file - 3) + rank of (run, lumi, event) in the file"),
                        **{k.replace("/", "__"): v for k, v in per_ev_ref.items()})
    with gzip.open(OUT_DIR / "p02_harmed_tracks.csv.gz", "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(harmed_rows[0]))
        w.writeheader()
        w.writerows(harmed_rows)
    log(f"wrote {OUT_DIR / 'p02_results.json'} ({len(harmed_rows)} harmed-track rows)")


def fallback(ctx, s, T0, arms, hs, pop, org, rb, ev, fit_ev, test_ev, Ts):
    W, nev = ctx.W, ctx.nev
    res_sp = {p: arms[s][p][0] for p in ("d0", "z0")}
    res_ot = {p: arms["OT"][p][0] for p in ("d0", "z0")}
    harmed = hs["either"]
    fit_sel, test_sel = pop & fit_ev, pop & test_ev
    ones = np.ones(nev)

    st_sp, st_ot = SigmaTable(res_sp, rb, ev, fit_sel), SigmaTable(res_ot, rb, ev, fit_sel)

    def run(w_ev):
        tab_sp, tab_ot = st_sp.at(w_ev), st_ot.at(w_ev)
        chi2 = fb_chi2(res_sp, res_ot, rb, tab_sp, tab_ot)
        um = fit_sel & ~harmed
        thr = wquantile(chi2[um], w_ev[ev[um]], 1.0 - FB_UNHARMED_REVERT)
        return chi2, thr, tab_sp, tab_ot

    chi2, thr, tab_sp, tab_ot = run(ones)
    assert all(np.isfinite(t[p]).all() for t in (tab_sp, tab_ot) for p in t), "a fallback sigma bin is below CORE_MIN_N"
    reps = [run(W[i]) for i in range(W.shape[0])]
    rev_nom = chi2 > thr
    rev_reps = [c > t for c, t, _, _ in reps]

    def ev_frac(sel, ind_nom, ind_reps):
        """nominal fraction and replicas, where the indicator itself changes per replica."""
        e = ev[sel]
        den = per_event(e, np.ones(sel.sum()), nev)
        v = per_event(e, ind_nom[sel], nev).sum() / den.sum()
        r = np.array([W[i] @ per_event(e, ind_reps[i][sel], nev) / (W[i] @ den) for i in range(W.shape[0])])
        return s_of(v, r)

    out = {"threshold_chi2": s_of(thr, np.array([t for _, t, _, _ in reps])),
           "sigma_table_um": {"bins": {"eta": [list(b) for b in ETA_BINS], "pt": [[b[0], None if np.isinf(b[1]) else b[1]] for b in PT_BINS]},
                              "SP": {p: tab_sp[p].tolist() for p in tab_sp}, "OT": {p: tab_ot[p].tolist() for p in tab_ot}},
           "wp": {}, "cost": {}, "alternatives": {}}
    groups = {"all": pop, **{o: pop & (org == i) for i, o in enumerate(ORIGINS)}}
    for g, gm in groups.items():
        for split, sm in (("fit_files_3-6", fit_ev), ("test_files_7-10", test_ev)):
            m = gm & sm
            out["wp"].setdefault(g, {})[split] = {
                "unharmed_revert_fraction": ev_frac(m & ~harmed, rev_nom, rev_reps),
                "harmed_revert_fraction": ev_frac(m & harmed, rev_nom, rev_reps),
                "revert_fraction": ev_frac(m, rev_nom, rev_reps),
                "H_before": s_of(*ctx.frac(ev[m], harmed[m])),
                "H_after": ev_frac(m, harmed & ~rev_nom, [harmed & ~rr for rr in rev_reps]),
                "n": int(m.sum()), "n_harmed": int((m & harmed).sum())}
    # cost on files 7-10 per origin: SP vs fallback (FB) vs OT
    for g, gm in groups.items():
        m = gm & test_ev
        e = ev[m]
        cg = {}
        for p, thr_list in (("d0", (100.0, 200.0, 500.0, 1000.0)), ("z0", (500.0, 1000.0))):
            sp, ot = res_sp[p][m], res_ot[p][m]
            fbn = np.where(rev_nom[m], ot, sp)
            cp = {}
            for lab, x in (("SP", sp), ("OT", ot)):
                nom, rr, n = ctx.core(x, e)
                cp[lab] = {"core": s_of(nom[0], rr[:, 0]), "outside_core": s_of(nom[3], rr[:, 3])}
                for t in thr_list:
                    cp[lab][f"frac_gt_{t / 1000:g}mm"] = s_of(*ctx.frac(e, np.abs(x) > t))
            # FB: the collection itself changes per replica
            from estimators import core_sorted
            def fbcore(x, w):
                o = np.argsort(x, kind="stable")
                return core_sorted(x[o], w[o], min_n=0 if len(x) >= CORE_MIN_N else CORE_MIN_N)
            nomfb = fbcore(fbn, np.ones(len(fbn)))
            rfb = (np.array([fbcore(np.where(rev_reps[i][m], ot, sp), W[i][e]) for i in range(W.shape[0])])
                   if len(fbn) >= CORE_MIN_N else np.full((W.shape[0], 4), np.nan))
            cp["FB"] = {"core": s_of(nomfb[0], rfb[:, 0]), "outside_core": s_of(nomfb[3], rfb[:, 3])}
            nsp = {}
            for t in thr_list:
                k = f"frac_gt_{t / 1000:g}mm"
                cp["FB"][k] = ev_frac(gm & test_ev, np.abs(np.where(rev_nom, res_ot[p], res_sp[p])) > t,
                                      [np.abs(np.where(rr, res_ot[p], res_sp[p])) > t for rr in rev_reps])
                # paired FB - SP
                den = per_event(e, np.ones(len(e)), nev)
                dn = per_event(e, (np.abs(fbn) > t).astype(float) - (np.abs(sp) > t), nev)
                drep = np.array([W[i] @ per_event(e, (np.abs(np.where(rev_reps[i][m], ot, sp)) > t).astype(float)
                                                  - (np.abs(sp) > t), nev) / (W[i] @ den) for i in range(W.shape[0])])
                nsp[k] = s_of(dn.sum() / den.sum(), drep)
            spn, spr, _ = ctx.core(sp, e)
            nsp["core"] = s_of(nomfb[0] - spn[0], rfb[:, 0] - spr[:, 0])
            nsp["outside_core"] = s_of(nomfb[3] - spn[3], rfb[:, 3] - spr[:, 3])
            cp["FB_minus_SP"] = nsp
            cg[p] = cp
        out["cost"][g] = cg
    # ROC curves (nominal): unharmed vs harmed revert fractions over thresholds
    qs = np.concatenate([np.linspace(0.0, 0.9, 19), np.linspace(0.9, 0.999, 34)])
    um_fit = pop & fit_ev & ~harmed
    thrs = np.unique(np.quantile(chi2[um_fit], qs))
    roc = {}
    for split, sm in (("fit_files_3-6", fit_ev), ("test_files_7-10", test_ev)):
        u, h = pop & sm & ~harmed, pop & sm & harmed
        roc[split] = {"threshold": thrs.tolist(),
                      "unharmed_revert": [float((chi2[u] > t).mean()) for t in thrs],
                      "harmed_revert": [float((chi2[h] > t).mean()) for t in thrs]}
        for o_i, o in enumerate(ORIGINS):
            hh = h & (org == o_i)
            if hh.sum():
                roc[split][f"harmed_revert_{o}"] = [float((chi2[hh] > t).mean()) for t in thrs]
    out["roc"] = roc
    # alternative discriminants at the same 1.2% unharmed revert on files 3-6 (context)
    alts = {"refit_shift_chi2 (spixShiftChi2, reported covariances)": Ts["sp_spixShiftChi2"].astype(np.float64),
            "reported-covariance chi2": sum((res_sp[p] - res_ot[p]) ** 2
                                           / ((Ts[f"sp_cov_{p}_{p}"] + T0[f"ot_cov_{p}_{p}"]) * 1e8) for p in ("d0", "z0"))}
    alts["measured-resolution chi2, d0 only"] = (res_sp["d0"] - res_ot["d0"]) ** 2 / (tab_sp["d0"][rb] ** 2 + tab_ot["d0"][rb] ** 2)
    alts["measured-resolution chi2, z0 only"] = (res_sp["z0"] - res_ot["z0"]) ** 2 / (tab_sp["z0"][rb] ** 2 + tab_ot["z0"][rb] ** 2)
    for nm, x in alts.items():
        x = np.where(np.isfinite(x), x, np.inf)
        um = pop & fit_ev & ~harmed
        t_nom = wquantile(x[um], np.ones(um.sum()), 1 - FB_UNHARMED_REVERT)
        t_rep = [wquantile(x[um], W[i][ev[um]], 1 - FB_UNHARMED_REVERT) for i in range(W.shape[0])]
        rn, rr = x > t_nom, [x > t for t in t_rep]
        out["alternatives"][nm] = {
            "test_harmed_revert": ev_frac(pop & test_ev & harmed, rn, rr),
            "test_unharmed_revert": ev_frac(pop & test_ev & ~harmed, rn, rr),
            "note": "threshold at the files 3-6 unharmed 98.8% quantile, refit per replica (no sigma table)"
                    if "measured" not in nm else "sigma table nominal (files 3-6); threshold refit per replica"}
    return out


if __name__ == "__main__":
    main()
