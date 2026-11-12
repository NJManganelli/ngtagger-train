"""The checkpointed, resumable scan runner of the SmartPixels output-bits refit study (PLAN.md sections 5, 6, 9).

  python run_scan.py supervise --workers 3     start the workers and wait for them (the command the lane runs)
  python run_scan.py worker --shard 0/3        one worker
  python run_scan.py status                    finished / total units, free disk
  python run_scan.py invalidate --arms NAME ... [--dry]   set finished units with these arms aside (redone at the next start)
  python run_scan.py unit --file 4 --scen AAAA --arms implemented_now alloc_4444 [--nev 10] --out DIR   (development)

A UNIT is one file x one scenario x a group of at most 8 arms (5-10 minutes). Its output is one compressed npz written
to a temporary name and renamed, so a file that exists is complete; a restart skips finished units; re-running the
same command continues. Units are ordered so that the priority arms of every file and scenario come first.

Robustness: before each unit the external disk and the input file are checked; if missing, the worker waits (polls
every 60 s, logs every 10 minutes). A unit that dies of an I/O error is retried after the disk is back. The worker
stops cleanly when the free space of the output disk falls below 12 GB. Everything but the canon_v5 nanos is read
from the internal disk (study.py). Outputs: SCHEMA.md beside them.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MOUNT = "/Volumes/WDMac"
MIN_FREE_GB = 12.0
ARMS_PER_UNIT = 8
NSLOT = 8
BASE = "implemented_now"
MAX_RESTARTS = 200


def log(*a):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)


def free_gb(path):
    return shutil.disk_usage(path).free / 2 ** 30


def wait_for_input(path):
    """Block until the external disk is mounted and the input file is readable."""
    n = 0
    while not (os.path.ismount(MOUNT) and os.path.exists(path)):
        if n % 10 == 0:
            log(f"waiting for {path} ({MOUNT} mounted: {os.path.ismount(MOUNT)})")
        n += 1
        time.sleep(60)


# ---------------------------------------------------------------- units ----
def load_registry(out):
    """The registry frozen with the outputs (copied there at the first start, so the units never change)."""
    frozen = Path(out) / "arms.json"
    if not frozen.exists():
        Path(out).mkdir(parents=True, exist_ok=True)
        tmp = frozen.with_suffix(".tmp")
        shutil.copy(HERE / "arms.json", tmp)
        tmp.rename(frozen)
    with open(frozen) as fh:
        return json.load(fh)


def units_of(reg, files, scens):
    """The ordered unit list: per group (priority first), per chunk of the group's arms, per file, the scenarios in
    an order rotated with the file and the block of units, so that workers taking every n-th unit each get a mix of scenarios
    (AAAA units take about 1.7 x the time of IAAA units)."""
    units, block = [], 0
    for g in reg["group_order"]:
        names = [a["name"] for a in reg["arms"] if a["group"] == g]
        chunks = [names[i:i + ARMS_PER_UNIT] for i in range(0, len(names), ARMS_PER_UNIT)]
        for k, ch in enumerate(chunks):
            block += 1
            for fi, f in enumerate(files):
                for j in range(len(scens)):
                    s = scens[(j + fi + block) % len(scens)]
                    units.append({"group": g, "file": f, "scen": s, "arms": ch,
                                  "name": f"{g}/{s}_f{f:02d}_{g}_{k:02d}.npz"})
    return units


# ---------------------------------------------------------------- arms ----
def build_arm(AR, cfg, name, spec):
    """The replay arm of a registry entry. SmartPixels alpha is the code's "beta", SmartPixels beta the code's "alpha"."""
    kw = {}
    if spec.get("pos_err") == "sizeblind":
        kw["pos_err"] = "rtables_sizeblind"
    if spec.get("ang_err") == "linktable":
        kw["ang_err"] = "linktable"
    quantized = False
    if "send" in spec:
        code = tuple(c for c, sp in (("alpha", "beta"), ("beta", "alpha")) if sp in spec["send"])
        code = tuple(spec.get("code_angles_override", code))                # planted defect only
        kw["use"] = AR.use_set(("rank", "update"), code)
        quantized = any(spec.get(f"bits_{sp}") for sp in spec["send"])
    sx, sy = spec.get("step_x_um"), spec.get("step_y_um")
    if not spec.get("errors_not_following"):                                # planted defect only
        kw["pos_var_add"] = ((sy * 1e-4) ** 2 / 12 if sy else 0.0, (sx * 1e-4) ** 2 / 12 if sx else 0.0)
    kw["window_var_add"] = bool(spec.get("window_add", True))
    kw["flag_mode"] = spec.get("flag", "onesided") if quantized else "off"
    if "pos_err_scale_x" in spec:
        kw["pos_err_scale"] = (spec["pos_err_scale_y"], spec["pos_err_scale_x"])
    return AR.v3(cfg, name=name, **kw)


class Study:
    """What a worker keeps between units."""

    def __init__(self, link_tables=True):
        import emulate as EM
        import study as ST
        self.ST, self.EM, self.AR, self.CM = ST, EM, ST.AR, ST.CM
        self.geo = ST.geometry()
        self.net = EM.Network(ST.CACHE / "network_residuals.npz")
        self.links = EM.Links(self.geo, ST.CM.GEOMETRY)
        self.LT = EM.LinkTables(ST.CACHE / "link_tables.json") if link_tables else None
        self.lookups = {}

    def tools_for(self, tools, spec):
        if self.LT is None:
            return tools
        src = "today" if spec.get("angles", "today") == "today" else "net"
        key = (src, spec.get("bits_alpha"), spec.get("bits_beta"), spec.get("range", "link"),
               spec.get("angle_err_scale", 1.0), spec.get("errors_not_following", False))
        if key not in self.lookups:
            self.lookups[key] = self.EM.LinkLookup(self.LT, src, key[1], key[2], key[3], scale=key[4],
                                                   qvar_on=not key[5])
        t = copy.copy(tools)
        t.linktable = self.lookups[key]
        if spec.get("cuts_file"):                               # post-scan: the skip price re-derived for an arm
            import cuts as CU
            t.cuts = CU.Cuts.load(spec["cuts_file"])
        return t


# ---------------------------------------------------------------- truth ----
def crossing_clusters(S, ctx, pop, seq, kpt):
    """Per population track and visit slot: the particle's crossing cluster. A particle's cluster on a module is its
    crossing cluster if it is the one nearest the particle's helix crossing of that module in pitch units (25 um in
    localX, 100 um in localY): one per particle and module (refit_calibration_fit.primary_mask's rule). Returns
    prim_ci (n, NSLOT): the nearest crossing cluster of the slot's surface (global cluster index, -1 none), prim_n: the
    number of crossing clusters there, the helix point (u, v) [cm] for it, and the sorted keys track * 2^32 + cluster of
    every crossing cluster and of every cluster of the track's particle."""
    OP, geo, K = S.ST.OP, S.geo, ctx["K"]
    tp = ctx["trk_tp"][pop]
    n = len(pop)
    cl = np.flatnonzero((K["tp_global"] >= 0) & (K["mi"] >= 0))
    o = np.argsort(K["tp_global"][cl], kind="stable")
    keys = K["tp_global"][cl][o]
    lo, hi = np.searchsorted(keys, tp, "left"), np.searchsorted(keys, tp, "right")
    cnt = hi - lo
    ti = np.repeat(np.arange(n), cnt)
    ci = cl[o][np.repeat(lo, cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
    mi = K["mi"][ci]
    surf = geo["surface"][mi]
    TP = ctx["TP"]
    g = tp[ti]
    at = np.stack([TP["charge"][g] * kpt / np.maximum(TP["pt"][g], 1e-9), TP["phi0"][g], np.sinh(TP["eta"][g]),
                   TP["z0"][g], TP["d0"][g]], 1)
    u = np.full(len(ci), np.nan)
    v = np.full(len(ci), np.nan)
    for s_ in np.unique(surf):
        m = np.flatnonzero(surf == s_)
        pj = OP.project_to_modules(at[m], OP.surface_s(at[m], geo, int(s_)), geo, mi[m])
        u[m], v[m] = np.where(pj["ok"], pj["u"], np.nan), np.where(pj["ok"], pj["v"], np.nan)
    d = np.hypot((K["localX"][ci] - u) / 0.0025, (K["localY"][ci] - v) / 0.0100)
    d = np.where(np.isfinite(d), d, 1e30)
    grp = ti.astype(np.int64) * 8192 + mi
    oo = np.lexsort((d, grp))
    first = np.r_[True, grp[oo][1:] != grp[oo][:-1]] if len(oo) else np.zeros(0, bool)
    is_prim = np.zeros(len(ci), bool)
    is_prim[oo[first]] = True
    prim_ci = np.full((n, NSLOT), -1, np.int64)
    prim_n = np.zeros((n, NSLOT), np.int8)
    prim_u = np.full((n, NSLOT), np.nan, np.float32)
    prim_v = np.full((n, NSLOT), np.nan, np.float32)
    sq = seq[pop]
    for j in range(sq.shape[1]):
        m = np.flatnonzero(is_prim & (surf == sq[ti, j]))
        if not len(m):
            continue
        np.add.at(prim_n[:, j], ti[m], 1)
        o2 = m[np.lexsort((d[m], ti[m]))]
        f2 = np.r_[True, ti[o2][1:] != ti[o2][:-1]]
        w = o2[f2]
        prim_ci[ti[w], j], prim_u[ti[w], j], prim_v[ti[w], j] = ci[w], u[w], v[w]
    key = ti.astype(np.int64) * (1 << 32) + ci
    every = {"track": ti[is_prim], "ci": ci[is_prim], "u": u[is_prim], "v": v[is_prim]}   # every crossing cluster
    return prim_ci, prim_n, prim_u, prim_v, np.sort(key[is_prim]), np.sort(key), every


def truth_rows(S, A, ctx, start, tr, pop, seq, path_, cx, B, has_vz):
    AR = S.AR
    ev = ctx["tev"][pop]
    toff = np.concatenate([[0], np.cumsum(ctx["T"]["_n"])])
    tp = ctx["trk_tp"][pop]
    TP = ctx["TP"]
    f32 = np.float32
    prim_ci = cx[0]
    sent = np.where(prim_ci >= 0, B["sent_pt"][np.maximum(prim_ci, 0)], True)
    sent_random = np.where(prim_ci >= 0, B["sent"][np.maximum(prim_ci, 0)], True)
    pt_bend = np.where(prim_ci >= 0, B["pt_bend"][np.maximum(prim_ci, 0)], np.nan)
    nsub = np.where(prim_ci >= 0, B["n_in_subarea"][np.maximum(prim_ci, 0)], 0)
    d = {"run": np.asarray(A["run"])[ev].astype(np.int64), "lumi": np.asarray(A["luminosityBlock"])[ev].astype(np.int64),
         "event": np.asarray(A["event"])[ev].astype(np.int64), "track": (pop - toff[ev]).astype(np.int32),
         "entry": (start + ev).astype(np.int32),
         "ot_a": ctx["a"][pop].astype(f32), "ot_cov_diag": np.einsum("nii->ni", ctx["C"][pop]).astype(f32),
         "ot_pt": ctx["T"]["pt"][pop].astype(f32), "ot_eta": ctx["T"]["eta"][pop].astype(f32),
         "ot_nstubs": ctx["T"]["nStubs"][pop].astype(np.int8),
         "tp_pt": tr["pt"][pop].astype(f32), "tp_eta": tr["eta"][pop].astype(f32), "tp_phi0": TP["phi0"][tp].astype(f32),
         "tp_d0": tr["d0"][pop].astype(f32), "tp_z0": tr["z0"][pop].astype(f32), "tp_charge": TP["charge"][tp].astype(np.int8),
         "tp_vx": TP["vx"][tp].astype(f32), "tp_vy": TP["vy"][tp].astype(f32),
         "tp_from_b": (TP["genFromB"][tp] > 0), "tp_from_c": (TP["genFromC"][tp] > 0),
         "origin": tr["origin"][pop].astype(np.int8), "disc_going": AR.disc_going(path_)[pop],
         "slot_surface": _pad(seq[pop], -1).astype(np.int8),
         "crossing_cluster": np.where(prim_ci >= 0, prim_ci - ctx["coff"][ev][:, None], -1).astype(np.int32),
         "n_crossing_clusters": cx[1], "crossing_cluster_sent_under_cap": sent,
         "crossing_cluster_sent_under_random_cap": sent_random, "crossing_cluster_pt_from_bending_GeV": pt_bend.astype(f32),
         "crossing_cluster_n_in_subarea": np.minimum(nsub, 127).astype(np.int8),
         "crossing_helix_localX_cm": cx[2], "crossing_helix_localY_cm": cx[3]}
    if has_vz:
        import awkward as ak
        vz = ak.to_numpy(ak.flatten(A[f"{AR.SPN.TP_TABLE}_vz"]))
        d["tp_vz"] = vz[tp].astype(f32)
    return d


def _pad(x, fill):
    if x.shape[1] == NSLOT:
        return x
    assert x.shape[1] < NSLOT, f"{x.shape[1]} visit slots > {NSLOT}"
    out = np.full((x.shape[0], NSLOT) + x.shape[2:], fill, x.dtype)
    out[:, :x.shape[1]] = x
    return out


IU = np.triu_indices(5)


def _f16(x):
    """float16 for diagnostics (3 significant digits), values beyond its range clipped to +-6e4."""
    return np.clip(x, -6e4, 6e4).astype(np.float16)


def arm_rows(S, ctx_a, out, F, pop, cx, B, ctx):
    """Per population track, what one arm delivered (SCHEMA.md)."""
    f32 = np.float32
    rec = out["rec"]
    acc = rec["acc"][pop]
    ci = np.where(acc, rec["ci"][pop], -1)
    K = ctx_a["K"]
    c0 = np.maximum(ci, 0)
    surf = np.where(acc, rec["surface"][pop], -1)
    bit = np.where(surf > 10, surf - 11 + 4, surf - 1)
    mask = np.where(acc, 1 << np.maximum(bit, 0), 0).sum(1).astype(np.uint8)
    chi2 = np.where(acc[:, :, None], rec["chi2"][pop], 0.0)
    key = np.arange(len(pop))[:, None].astype(np.int64) * (1 << 32) + c0
    tpc = ctx["K"]["tp_global"][c0]
    code = np.where(~acc, 0, np.where(np.isin(key, cx[4]), 1, np.where(np.isin(key, cx[5]), 2,
                                                                      np.where(tpc >= 0, 3, 4)))).astype(np.int8)
    bits = (K["hasAlpha"][c0] > 0) * 1 + (K["hasBeta"][c0] > 0) * 2 + (~B["class_T"][c0]) * 16
    if "sb_flagAlpha" in K:
        bits = bits + (K["sb_flagAlpha"][c0] != 0) * 4 + (K["sb_flagBeta"][c0] != 0) * 8
    ev = ctx["tev"][pop]
    # the crossing cluster's emulated position minus the particle's helix, where the slot is barrel layer 1
    pc = cx[0][:, :rec["surface"].shape[1]]               # the visit slots of this scenario (the truth arrays are padded)
    il1 = (rec["surface"][pop] == 1) & (pc >= 0)
    has1 = il1.any(1)
    j1 = np.argmax(il1, 1)
    r = np.arange(len(pop))
    p1 = np.maximum(pc[r, j1], 0)
    dx = np.where(has1, (K["localX"][p1] - cx[2][r, j1]) * 1e4, np.nan)
    dy = np.where(has1, (K["localY"][p1] - cx[3][r, j1]) * 1e4, np.nan)
    Fa, FC = ctx["a"][pop].copy(), ctx["C"][pop][:, IU[0], IU[1]].copy()
    frefit = np.zeros(len(pop), bool)
    if F is not None:
        pos = np.full(ctx["ntrk"], -1, np.int64)
        pos[F.o] = np.arange(len(F.o))
        k = pos[pop]
        ok = (k >= 0) & (F.rec["acc"][np.maximum(k, 0)].sum(1) >= S.cfg.get("minHits", 1))
        Fa[ok], FC[ok] = F.a[k[ok]], F.C[k[ok]][:, IU[0], IU[1]]
        frefit = ok
    return {"a": out["a"][pop].astype(f32), "cov": out["C"][pop][:, IU[0], IU[1]].astype(f32),
            "refit": out["refit"][pop], "n_crossings": rec["visited"][pop].sum(1).astype(np.int8),
            "n_accepted": acc.sum(1).astype(np.int8), "layer_mask": mask,
            "any_window_truncated": rec["trunc"][pop].any(1), "max_candidates": rec["ncand"][pop].max(1).astype(np.int16),
            "chi2_sum": chi2.sum(1).astype(f32),
            "hit_accepted": _pad(acc, False), "hit_abs_pull": _pad(np.sqrt(chi2), 0.0).astype(np.float16),
            "hit_candidates": _pad(rec["ncand"][pop], 0).astype(np.int16),
            "hit_bits": _pad(np.where(acc, bits, 0), 0).astype(np.uint8),
            "hit_cluster": _pad(np.where(acc, ci - ctx["coff"][ev][:, None], -1), -1).astype(np.int32),
            "hit_code": _pad(code, 0), "hit_choice": _pad(rec["kchoice"][pop], -1).astype(np.int8),
            "il1_crossing_dx_um": dx.astype(f32), "il1_crossing_dy_um": dy.astype(f32),
            "hit_rank_chi2": _pad(_f16(rec["rankchi2"][pop]), np.nan), "window_scores": _pad(_f16(rec["kscores"][pop]), np.nan),
            "hit_candidates_classC": _pad(rec["ncand_c"][pop], 0).astype(np.int16),
            "crossing_in_window": _pad(rec["watch_in"][pop], False),
            "crossing_chi2": _pad(_f16(rec["watch_chi2"][pop]), np.nan),
            "forced_a": Fa.astype(f32), "forced_cov": FC.astype(f32), "forced_refit": frefit}


# ---------------------------------------------------------------- one unit ----
def run_unit(S, reg, f, scen, arm_names, nev=None, defect="", blank=False, emulation_off=False):
    """All arms of a unit in one pass over the file. Returns (truth dict, {arm: dict})."""
    ST, EM, AR, CM = S.ST, S.EM, S.AR, S.CM
    import fom as FM
    path = ST.input_path(scen, f)
    cfg = CM.producer_config(path)
    assert CM.mask_of(cfg) == scen
    S.cfg = cfg
    tools = ST.tools(cfg)
    specs = {a["name"]: a["spec"] for a in reg["arms"]}
    arms = {n: build_arm(AR, cfg, n, specs[n]) for n in dict.fromkeys([BASE] + list(arm_names))}
    keys = AR.OP._keys(str(path))
    has_vz = f"{AR.SPN.TP_TABLE}_vz" in keys
    br = list(dict.fromkeys(AR.branches(path) + ["run", "luminosityBlock", "event"]
                            + AR.SPN.cluster_branches(keys, list(EM.EXTRA_CLUSTER_COLS), disc_scope="all")
                            + ([f"{AR.SPN.TP_TABLE}_vz"] if has_vz else [])))
    truth, rows = [], {n: [] for n in arm_names}
    for start, A in CM.iterate(path, br, chunk=10, nev=nev):
        ctx = AR.prepare(A, S.geo)
        kpt = CM.pt_constant(ctx)
        tr = FM.track_truth(ctx)
        pop = np.flatnonzero(tr["pop"])
        B = EM.base(ctx, S.geo, A, S.net, S.links, S.LT, defect=defect)
        path_, seq = AR.visit_plan(ctx, S.geo, cfg, scen, arms[BASE])
        cx = crossing_clusters(S, ctx, pop, seq, kpt)
        truth.append(truth_rows(S, A, ctx, start, tr, pop, seq, path_, cx, B, has_vz))
        watch = np.full(seq.shape, -1, np.int64)                # the crossing cluster per OT track and visit step
        watch[pop] = cx[0][:, :seq.shape[1]]
        diag = {"sb_watch": watch, "sb_classC": (~B["class_T"]).astype(np.float64)}
        if blank:
            diag = {}
        base_out = AR.run_arm({**ctx, **diag}, S.geo, cfg, scen, arms[BASE], tools)
        o = np.flatnonzero(base_out["rec"]["acc"].any(1))
        sel_mi = base_out["rec"]["mi"][o]
        sel_ci = np.where(base_out["rec"]["acc"][o], base_out["rec"]["ci"][o], -1)
        for n in arm_names:
            spec = specs[n]
            t_a = S.tools_for(tools, spec)
            if n == BASE or emulation_off:
                ctx_a = ctx if n == BASE else EM.emulate(ctx, B, {}, S.LT)
            else:
                ctx_a = EM.emulate(ctx, B, spec, S.LT)
            if blank and n != BASE:
                ctx_a = EM.blank_unsent_information(ctx_a)
            out = base_out if (n == BASE and not blank) else AR.run_arm({**ctx_a, **diag}, S.geo, cfg, scen, arms[n], t_a)
            F = None
            if len(o):
                F = AR.refit_forced(arms[n], ctx_a, S.geo, cfg, t_a, o, seq[o], sel_mi, sel_ci, path_, kpt)
                AR.inner_material(arms[n], F, S.geo, t_a, kpt)
            rows[n].append(arm_rows(S, ctx_a, out, F, pop, cx, B, ctx))
    return CM.concat(truth), {n: CM.concat(v) for n, v in rows.items()}


def save_npz(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def do_unit(S, reg, u, out):
    t0 = time.time()
    truth, arms = run_unit(S, reg, u["file"], u["scen"], u["arms"])
    tfile = Path(out) / "units" / "truth" / f"{u['scen']}_f{u['file']:02d}_truth.npz"
    if not tfile.exists():
        save_npz(tfile, truth)
    save_npz(Path(out) / "units" / u["name"], {f"{n}__{k}": v for n, d in arms.items() for k, v in d.items()})
    log(f"done {u['name']}: {len(u['arms'])} arms, {len(truth['run'])} tracks, {time.time() - t0:.0f} s, "
        f"free {free_gb(out):.1f} GB")


def worker(a):
    out = Path(a.out)
    reg = load_registry(out)
    units = units_of(reg, a.files, a.scen)
    i, n = (int(x) for x in a.shard.split("/"))
    mine = units[i::n]
    if a.reverse:                                               # an extra worker helping a slow shard from its end
        mine = mine[::-1]
    log(f"worker {i}/{n}: {len(mine)} of {len(units)} units, pid {os.getpid()}")
    S = None
    for u in mine:
        target = out / "units" / u["name"]
        if target.exists():
            continue
        if free_gb(out) < MIN_FREE_GB:
            log(f"STOP: free space {free_gb(out):.1f} GB < {MIN_FREE_GB} GB")
            return 3
        for attempt in range(1, 6):
            path = f"{MOUNT}/smartpixels_scratch/canon_v5/spstudy_{u['scen']}_f{u['file']:02d}_100ev.root"
            wait_for_input(path)
            try:
                if S is None:
                    S = Study()
                do_unit(S, reg, u, out)
                break
            except Exception:                                   # noqa: BLE001 (logged; the next unit goes on)
                tb = traceback.format_exc()
                gone = not (os.path.ismount(MOUNT) and os.path.exists(path))
                log(f"unit {u['name']} failed (attempt {attempt}; input {'gone' if gone else 'present'}):\n{tb}")
                if not gone and attempt >= 2:                   # not an I/O loss: record it, go on
                    fail = out / "units" / (u["name"] + ".failed.txt")
                    fail.parent.mkdir(parents=True, exist_ok=True)
                    fail.write_text(tb)
                    break
                time.sleep(30)
    log(f"worker {i}/{n}: finished its list")
    return 0


def supervise(a):
    """Start the workers (own session, own logs) and wait for them; re-running continues."""
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    load_registry(out)
    pidfile = out / "logs" / "scan.pids"
    if pidfile.exists():                                        # never two supervisors on the same outputs
        for line in pidfile.read_text().splitlines():
            if line.startswith("supervisor "):
                try:
                    os.kill(int(line.split()[1]), 0)
                    log(f"a supervisor is already running (pid {line.split()[1]}); not starting another")
                    return 4
                except (OSError, ValueError):
                    pass
    def start(i):
        lf = open(out / "logs" / f"scan_worker{i}.log", "a")
        cmd = [sys.executable, str(HERE / "run_scan.py"), "worker", "--shard", f"{i}/{a.workers}", "--out", str(out),
               "--files", *map(str, a.files), "--scen", *a.scen]
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def record(procs):
        pidfile.write_text(" ".join(str(p.pid) for p in procs.values()) + f"\nsupervisor {os.getpid()}\n")

    procs = {}
    for i in range(a.workers):
        procs[i] = start(i)
        record(procs)
        time.sleep(20)                                          # stagger the start-up memory
    log(f"supervisor {os.getpid()}: workers {[p.pid for p in procs.values()]}")
    restarts, final = 0, {}
    while procs:                                                # a worker killed by a vanished disk (a bus error on a
        time.sleep(30)                                          # mapped file) or anything else is started again
        for i, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            if rc in (0, 3) or restarts >= MAX_RESTARTS:        # finished, or stopped for disk space
                final[i] = rc
                del procs[i]
                continue
            restarts += 1
            log(f"supervisor: worker {i} exited {rc}; restart {restarts} in 60 s")
            time.sleep(60)
            procs[i] = start(i)
            record(procs)
    log(f"supervisor: workers exited {final}")
    return max(final.values(), default=0)


def status(a):
    out = Path(a.out)
    reg = load_registry(out)
    units = units_of(reg, a.files, a.scen)
    done = [u for u in units if (out / "units" / u["name"]).exists()]
    by = {}
    for u in units:
        e = by.setdefault(u["group"], [0, 0])
        e[1] += 1
        e[0] += (out / "units" / u["name"]).exists()
    size = sum(f.stat().st_size for f in (out / "units").rglob("*.npz")) / 2 ** 30 if (out / "units").exists() else 0
    failed = [str(f) for f in (out / "units").rglob("*.failed.txt")] if (out / "units").exists() else []
    print(json.dumps({"done": len(done), "total": len(units), "by_group": by, "units_GB": round(size, 3),
                      "free_GB": round(free_gb(out), 1), "failed": failed}, indent=1))
    return 0


def invalidate(a):
    """Set aside every finished unit that holds one of --arms (renamed to <name>.invalid-<time>, never deleted), so
    that the next (re)start of the scan redoes exactly those units. A worker that has already passed a unit in its
    list does not come back to it: relaunch after the running job. --refresh-registry also replaces the registry
    frozen beside the outputs by the code's arms.json (only if the unit composition is unchanged)."""
    out = Path(a.out)
    reg = load_registry(out)
    units = units_of(reg, a.files, a.scen)
    if a.refresh_registry:
        new = json.load(open(HERE / "arms.json"))
        if [(u["name"], u["arms"]) for u in units_of(new, a.files, a.scen)] != [(u["name"], u["arms"]) for u in units]:
            log("REFUSED: the new registry changes the unit composition")
            return 5
        tmp = out / "arms.json.tmp"
        shutil.copy(HERE / "arms.json", tmp)
        os.replace(out / "arms.json", out / f"arms.replaced-{time.strftime('%Y%m%dT%H%M%S')}.json")
        tmp.rename(out / "arms.json")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    hit = []
    for u in units:
        f = out / "units" / u["name"]
        if set(u["arms"]) & set(a.arms) and f.exists():
            if not a.dry:
                f.rename(f.with_name(f.name + f".invalid-{stamp}"))
            hit.append(u["name"])
    print(json.dumps({"invalidated" if not a.dry else "would_invalidate": hit, "n": len(hit)}, indent=1))
    return 0


def unit_cmd(a):
    S = Study(link_tables=not a.no_link_tables)
    reg = json.load(open(HERE / "arms.json"))
    t0 = time.time()
    truth, arms = run_unit(S, reg, a.file, a.scen[0], a.arms, nev=a.nev, defect=a.defect, blank=a.blank)
    save_npz(Path(a.out) / f"{a.scen[0]}_f{a.file:02d}_truth.npz", truth)
    save_npz(Path(a.out) / f"{a.scen[0]}_f{a.file:02d}_arms.npz", {f"{n}__{k}": v for n, d in arms.items() for k, v in d.items()})
    log(f"{len(a.arms)} arms, {len(truth['run'])} tracks, {time.time() - t0:.1f} s")
    return 0


def main(argv=None):
    import study as ST
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("supervise", "worker", "status", "unit", "invalidate"))
    ap.add_argument("--out", default=str(ST.LOCAL))
    ap.add_argument("--files", type=int, nargs="+", default=list(ST.SCAN_FILES))
    ap.add_argument("--scen", nargs="+", default=list(ST.SCENARIOS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--file", type=int, default=ST.DEV_FILE)
    ap.add_argument("--arms", nargs="+", default=[BASE])
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--defect", default="")
    ap.add_argument("--blank", action="store_true")
    ap.add_argument("--no-link-tables", action="store_true")
    ap.add_argument("--refresh-registry", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="worker: take the shard's units from the end")
    ap.add_argument("--dry", action="store_true", help="invalidate: only list the units")
    a = ap.parse_args(argv)
    return {"supervise": supervise, "worker": worker, "status": status, "unit": unit_cmd, "invalidate": invalidate}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
