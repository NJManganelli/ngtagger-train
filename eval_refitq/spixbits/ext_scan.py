"""The shipped allocations on the EXTENDED (displaced-capable) track collection, with the deployed heads scored on
every arm: the source of the parametrization (PLAN.md sections 7, 7b). Checkpointed like the scan.

  python ext_scan.py supervise --workers 3      start the workers and wait (the command the lane runs)
  python ext_scan.py worker | status
  python ext_scan.py unit --file 4 --scen AAAA --arms implemented_now alloc_4444 --nev 10 --out DIR    (development)

Shipped arms (only allocations run directly are shipped: the hit-choice part is not additive): implemented now,
network float, the eleven 16-bit allocations with per-link ranges (4/4/4/4 ... 8/8/0/0; 6/6/0/4 is the
bending-angle-only cluster), 4/4/4/4 over the whole sensor, and the position scans in y (2-10 sub-area bits) and x
(2-8 bits) at the network-float angles. Files: the five scan files; every scenario.

A unit = one file x one scenario x up to 8 arms; one compressed npz, written atomically; a worker claims a unit with
an exclusive claim file (a dead claimant's claim is taken over), so the workers balance themselves; finished units
are skipped; the workers wait while the external disk is missing and stop cleanly below 12 GB free.

Outputs: ~/smartpixels/scratch_local/spixbits/ext/ (SCHEMA_ext.md beside them): population = EVERY track of the
extended collection, matched to a particle or not.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import run_scan as RS
from run_scan import log

HERE = Path(__file__).resolve().parent
ARMS_PER_UNIT = 8
BASE = RS.BASE
SHIPPED = (["implemented_now", "network_float", "alloc_4444", "alloc_6604", "alloc_8800", "alloc_4444_wholesensor",
            "alloc_5533", "alloc_6622", "alloc_5623", "alloc_4633", "alloc_4723", "alloc_5722", "alloc_6703", "alloc_7702"]
           + [f"down_y_{b}b" for b in range(2, 11)] + [f"down_x_{b}b" for b in range(2, 9)])
IU = np.triu_indices(5)


def out_root():
    import study as ST
    return ST.LOCAL / "ext"


def units_of(files, scens):
    chunks = [SHIPPED[i:i + ARMS_PER_UNIT] for i in range(0, len(SHIPPED), ARMS_PER_UNIT)]
    return [{"file": f, "scen": s, "arms": ch, "name": f"{s}_f{f:02d}_{k:02d}"}
            for k, ch in enumerate(chunks) for f in files for s in scens]


def run_unit(S, reg, f, scen, arm_names, nev=None):
    """All arms of a unit in one pass over the file. Returns (per-file truth / today dict, {arm: dict})."""
    import fom as FM
    import heads as HD
    ST, EM, AR, CM = S.ST, S.EM, S.AR, S.CM
    assert CM.COLLECTION == "extended"
    path = ST.input_path(scen, f)
    HD.refuse_truth(path)
    cfg = CM.producer_config(path)
    assert CM.mask_of(cfg) == scen
    S.cfg = cfg
    tools = ST.tools(cfg)
    H = HD.Heads(ST.CACHE / "heads", scen)
    specs = {a["name"]: a["spec"] for a in reg["arms"]}
    arms = {n: RS.build_arm(AR, cfg, n, specs[n]) for n in dict.fromkeys([BASE] + list(arm_names))}
    keys = AR.OP._keys(str(path))
    br = list(dict.fromkeys(AR.branches(path) + HD.nano_branches(keys) + ["run", "luminosityBlock", "event",
                                                                         "L1TTrack_genuine", "L1TTrack_phi"]
                            + AR.SPN.cluster_branches(keys, list(EM.EXTRA_CLUSTER_COLS), disc_scope="all")))
    f32 = np.float32
    truth, rows = [], {n: [] for n in arm_names}
    for start, A in CM.iterate(path, br, chunk=10, nev=nev):
        ctx = AR.prepare(A, S.geo)
        n = ctx["ntrk"]
        kpt = CM.pt_constant(ctx)
        tr = FM.track_truth(ctx)
        B = EM.base(ctx, S.geo, A, S.net, S.links, S.LT)
        Ts = HD.stored_columns(A)
        path_, seq = AR.visit_plan(ctx, S.geo, cfg, scen, arms[BASE])
        base_out = AR.run_arm(ctx, S.geo, cfg, scen, arms[BASE], tools)
        o = np.flatnonzero(base_out["rec"]["acc"].any(1))
        sel_mi = base_out["rec"]["mi"][o]
        sel_ci = np.where(base_out["rec"]["acc"][o], base_out["rec"]["ci"][o], -1)
        b_acc = base_out["rec"]["acc"] & base_out["refit"][:, None]
        b_ci = np.where(b_acc, base_out["rec"]["ci"], -1)
        ev = ctx["tev"]
        toff = np.concatenate([[0], np.cumsum(ctx["T"]["_n"])])
        tp = ctx["trk_tp"]
        TP = ctx["TP"]
        gt = lambda c: np.where(tp >= 0, TP[c][np.maximum(tp, 0)], np.nan)
        import awkward as ak
        xs, _ = HD.matrix(Ts)
        truth.append({
            "run": np.asarray(A["run"])[ev].astype(np.int64), "lumi": np.asarray(A["luminosityBlock"])[ev].astype(np.int64),
            "event": np.asarray(A["event"])[ev].astype(np.int64), "track": (np.arange(n) - toff[ev]).astype(np.int32),
            "entry": (start + ev).astype(np.int32),
            "ot_inputs": xs[:, :13], "ot_a": ctx["a"].astype(f32), "ot_cov_diag": np.einsum("nii->ni", ctx["C"]).astype(f32),
            "ot_pt": ctx["T"]["pt"].astype(f32), "ot_eta": ctx["T"]["eta"].astype(f32), "ot_phi": ctx["a"][:, 1].astype(f32),
            "ot_genuine": ak.to_numpy(ak.flatten(A["L1TTrack_genuine"])) > 0,
            "matched": tp >= 0, "tp_pt": gt("pt").astype(f32), "tp_eta": gt("eta").astype(f32), "tp_phi0": gt("phi0").astype(f32),
            "tp_d0": gt("d0").astype(f32), "tp_z0": gt("z0").astype(f32), "tp_charge": np.nan_to_num(gt("charge")).astype(np.int8),
            "origin": tr["origin"].astype(np.int8), "disc_going": AR.disc_going(path_),
            "stored_cmssw_inputs": xs[:, 13:], "stored_cmssw_d0": Ts["r_d0"].astype(f32)})
        for nme in arm_names:
            spec = specs[nme]
            t_a = S.tools_for(tools, spec)
            ctx_a = ctx if nme == BASE else EM.emulate(ctx, B, spec, S.LT)
            out = base_out if nme == BASE else AR.run_arm(ctx_a, S.geo, cfg, scen, arms[nme], t_a)
            Fa, FC, frefit = ctx["a"].copy(), np.einsum("nii->ni", ctx["C"]).copy(), np.zeros(n, bool)
            if len(o):
                F = AR.refit_forced(arms[nme], ctx_a, S.geo, cfg, t_a, o, seq[o], sel_mi, sel_ci, path_, kpt)
                AR.inner_material(arms[nme], F, S.geo, t_a, kpt)
                ok = F.rec["acc"].sum(1) >= cfg.get("minHits", 1)
                Fa[o[ok]], FC[o[ok]], frefit[o[ok]] = F.a[ok], np.einsum("nii->ni", F.C[ok]), True
            sent = spec.get("send", ["alpha", "beta"])
            Tr = HD.replay_columns(S, ctx_a, out, Ts, kpt, cfg, alpha_sent="beta" in sent)
            X, _ = HD.matrix(Tr)
            acc = out["rec"]["acc"] & out["refit"][:, None]
            ci = np.where(acc, out["rec"]["ci"], -1)
            ctp = ctx["K"]["tp_global"][np.maximum(ci, 0)]
            wrong = acc & ((ctp != tp[:, None]) | (tp[:, None] < 0))
            d = {"a": out["a"].astype(f32), "cov_diag": np.einsum("nii->ni", out["C"]).astype(f32), "refit": out["refit"],
                 "inputs": X[:, 13:], "margins": H.margins(X, Tr["r_d0"]).astype(f32),
                 "same_clusters": (ci == b_ci).all(1) & (out["refit"] == base_out["refit"]),
                 "n_wrong_hits": wrong.sum(1).astype(np.int8),
                 "forced_a": Fa.astype(f32), "forced_cov_diag": FC.astype(f32), "forced_refit": frefit}
            if nme == BASE:
                d["cov"] = out["C"][:, IU[0], IU[1]].astype(f32)
            rows[nme].append(d)
    return CM.concat(truth), {k: CM.concat(v) for k, v in rows.items()}


# ---------------------------------------------------------------- workers ----
def done(u):
    return (out_root() / "units" / f"{u['name']}.npz").exists()


def claim(u):
    c = out_root() / "units" / f"{u['name']}.claim"
    c.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            os.kill(int(c.read_text() or "0"), 0)
        except (OSError, ValueError):
            c.unlink(missing_ok=True)
        return False


def worker(a):
    import study as ST
    ST.CM.set_collection("extended")
    S = RS.Study()
    reg = json.load(open(HERE / "arms.json"))
    us = units_of(a.files, a.scen)
    log(f"extended-scan worker pid {os.getpid()}: {len(us)} units")
    fails = {}
    while True:
        todo = [u for u in us if not done(u) and fails.get(u["name"], 0) < 2]
        if not todo:
            log("worker: nothing left to do")
            return 0 if all(done(u) for u in us) else 1
        if RS.free_gb(ST.LOCAL) < RS.MIN_FREE_GB:
            log(f"STOP: free space {RS.free_gb(ST.LOCAL):.1f} GB < {RS.MIN_FREE_GB} GB")
            return 3
        u = next((x for x in todo if claim(x)), None)
        if u is None:
            time.sleep(20)
            continue
        t0 = time.time()
        path = str(ST.input_path(u["scen"], u["file"]))
        try:
            RS.wait_for_input(path)
            truth, arms = run_unit(S, reg, u["file"], u["scen"], u["arms"])
            tf = out_root() / "truth" / f"{u['scen']}_f{u['file']:02d}_truth.npz"
            if not tf.exists():
                RS.save_npz(tf, truth)
            RS.save_npz(out_root() / "units" / f"{u['name']}.npz", {f"{n}__{k}": v for n, d in arms.items() for k, v in d.items()})
            log(f"done {u['name']}: {len(u['arms'])} arms, {len(truth['run'])} tracks, {time.time() - t0:.0f} s, "
                f"free {RS.free_gb(ST.LOCAL):.1f} GB")
        except Exception:                                       # noqa: BLE001
            tb = traceback.format_exc()
            gone = not (os.path.ismount(RS.MOUNT) and os.path.exists(path))
            log(f"unit {u['name']} failed (input {'gone' if gone else 'present'}):\n{tb}")
            if not gone:
                fails[u["name"]] = fails.get(u["name"], 0) + 1
                if fails[u["name"]] >= 2:
                    (out_root() / "units" / f"{u['name']}.failed.txt").write_text(tb)
            time.sleep(30)
        finally:
            (out_root() / "units" / f"{u['name']}.claim").unlink(missing_ok=True)


def supervise(a):
    import study as ST
    logs = ST.LOCAL / "logs"

    def start(i):
        lf = open(logs / f"ext_worker{i}.log", "a")
        cmd = [sys.executable, str(HERE / "ext_scan.py"), "worker", "--files", *map(str, a.files), "--scen", *a.scen]
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    procs = {}
    for i in range(a.workers):
        procs[i] = start(i)
        time.sleep(20)
    (logs / "ext.pids").write_text(" ".join(str(p.pid) for p in procs.values()) + f"\nsupervisor {os.getpid()}\n")
    restarts, final = 0, {}
    while procs:
        time.sleep(30)
        for i, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            if rc in (0, 1, 3) or restarts >= 100:
                final[i] = rc
                del procs[i]
            else:
                restarts += 1
                log(f"supervisor: worker {i} exited {rc}; restart {restarts}")
                time.sleep(60)
                procs[i] = start(i)
    log(f"supervisor: workers exited {final}")
    return max(final.values(), default=0)


def status(a):
    us = units_of(a.files, a.scen)
    d = [u for u in us if done(u)]
    size = sum(f.stat().st_size for f in out_root().rglob("*.npz")) / 2 ** 30 if out_root().exists() else 0
    failed = sorted(p.name for p in (out_root() / "units").glob("*.failed.txt")) if out_root().exists() else []
    print(json.dumps({"done": len(d), "total": len(us), "GB": round(size, 3), "free_GB": round(RS.free_gb(Path.home()), 1),
                      "failed": failed}, indent=1))
    return 0


def unit_cmd(a):
    import study as ST
    ST.CM.set_collection("extended")
    S = RS.Study()
    t0 = time.time()
    truth, arms = run_unit(S, json.load(open(HERE / "arms.json")), a.file, a.scen[0], a.arms, nev=a.nev)
    RS.save_npz(Path(a.out) / f"{a.scen[0]}_f{a.file:02d}_truth.npz", truth)
    RS.save_npz(Path(a.out) / f"{a.scen[0]}_f{a.file:02d}_arms.npz", {f"{n}__{k}": v for n, d in arms.items() for k, v in d.items()})
    log(f"{len(a.arms)} arms, {len(truth['run'])} tracks, {time.time() - t0:.1f} s")
    return 0


def main(argv=None):
    import study as ST
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("supervise", "worker", "status", "unit"))
    ap.add_argument("--files", type=int, nargs="+", default=list(ST.SCAN_FILES))
    ap.add_argument("--scen", nargs="+", default=list(ST.SCENARIOS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--file", type=int, default=ST.DEV_FILE)
    ap.add_argument("--arms", nargs="+", default=[BASE])
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    return {"supervise": supervise, "worker": worker, "status": status, "unit": unit_cmd}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
