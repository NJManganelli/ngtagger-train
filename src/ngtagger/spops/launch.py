"""The launcher every heavy job goes through (plan App. C.3-1, -3, -4, -5).

    ops/spheavy run [options] -- <command ...>
    ops/spheavy status
    ops/spheavy log [-n 20]

`run`, in order:
  1. (--frozen) builds the frozen copy of whole trees (src/ngtagger + every --tree), stores
     the manifest and the diff of the tracked files against HEAD (tree.patch) in the job
     directory and in the registry's provenance store -- captured now, at launch;
  2. checks the declared data files (--data) where the job will read them (inside the copy
     for repository paths) and records their checksums; a missing one stops the launch;
  3. waits for the heavy lane: no legacy job (bracketed pgrep form), no cmsRun/scram inside
     the Docker container, enough memory and disk, then takes the lock and re-checks all
     of it with the lock held (ngtagger.spops.lane); --on-busy skip logs and exits 75;
  4. (--gate) runs the fast gate from the copy, unless a green result exists for the same
     manifest hash; red stops the launch (65);
  5. runs the command: from the copy (cwd = copy root, PYTHONPATH = the copy's guard and
     src, repository paths in the arguments rewritten into the copy, `python` = the pixi
     environment's interpreter), with the lock descriptor inherited, so the lane stays
     taken while any process of the job lives;
  6. records the exit, the guard's findings and the peak memory in exit.json and the lane log.

Exit codes: the job's own on failure; 97 import-guard violation; 98 the job's Python never
ran the guard (PYTHONPATH did not reach it); 75 lane busy (skip / --max-wait); 65 fast
gate red; 66 launch refused (copy, data, pre-registration); 2 usage.

Stdlib only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import shlex
import signal
import subprocess
import sys
import time

from . import frozen as F
from . import lane as L

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.environ.get("SP_REPO") or os.path.abspath(os.path.join(HERE, "..", "..", ".."))
JOBS_DIR = os.environ.get("SP_JOBS_DIR", os.path.join(L.HOME, "smartpixels", "sp_jobs"))
KINDS = ("census", "tagger", "replay", "cmssw", "build", "gate", "selftest", "other")
PREREG_KINDS = ("census", "tagger", "replay")
GUARD_START_TIMEOUT_S = 60

EXIT_GUARD_VIOLATION, EXIT_GUARD_MISSING, EXIT_BUSY, EXIT_GATE_RED, EXIT_REFUSED = 97, 98, 75, 65, 66


def env_python(repo: str) -> str:
    return os.path.join(repo, ".pixi", "envs", "default", "bin", "python")


def _registry_dir(a) -> str:
    from . import registry as R

    return a.registry_dir or R.REGISTRY_DIR


def rewrite_command(cmd: list[str], repo: str, copy_root: str | None, python: str, trees) -> tuple[list, list]:
    """(command, notes). `python`/`python3` -> the environment's interpreter; `pixi run`
    dropped (it would re-resolve the environment and run from the manifest's directory);
    absolute repository paths inside a frozen tree -> the copy."""
    notes, out = [], list(cmd)
    if out[:2] == ["pixi", "run"]:
        out = out[2:]
        notes.append("dropped `pixi run`: the job runs the environment's interpreter directly")
    if out and out[0] in ("python", "python3"):
        out[0] = python
        notes.append(f"python -> {python}")
    if copy_root is None:
        return out, notes
    roots = [os.path.join(repo, t) for t in trees]

    def mapped(p: str) -> str | None:
        ap = os.path.abspath(p)
        if any(ap == r or ap.startswith(r + os.sep) for r in roots):
            return os.path.join(copy_root, os.path.relpath(ap, repo))
        return None

    for i, arg in enumerate(out):
        if i == 0 and arg == python:
            continue
        if os.path.isabs(arg) and mapped(arg):
            out[i] = mapped(arg)
            notes.append(f"argument {arg} -> {out[i]}")
        elif arg.startswith("--") and "=" in arg:
            k, v = arg.split("=", 1)
            if os.path.isabs(v) and mapped(v):
                out[i] = f"{k}={mapped(v)}"
                notes.append(f"argument {arg} -> {out[i]}")
    return out, notes


def guard_records(job_dir: str) -> dict:
    g = os.path.join(job_dir, "guard")
    starts = sorted(glob.glob(os.path.join(g, "*.start.json")))
    exits = sorted(glob.glob(os.path.join(g, "*.exit.json")))
    viol = []
    vp = os.path.join(g, "violations.jsonl")
    if os.path.exists(vp):
        with open(vp) as f:
            viol = [json.loads(x) for x in f if x.strip()]
    outside = []
    for p in exits:
        try:
            with open(p) as f:
                for line in f:
                    outside += json.loads(line).get("modules_outside", [])
        except (OSError, ValueError):
            pass
    return {"python_processes_guarded": len(starts), "python_processes_exited": len(exits),
            "violations": viol, "modules_outside_at_exit": outside}


def guard_verdict(rc: int, records: dict, expect_guard: bool, guard_missing: bool = False) -> tuple[str, int]:
    """(status, exit code) of a frozen-copy job from its exit code and its guard records:
    a violation (97) beats a job whose Python never ran the guard (98) beats a failure."""
    if records.get("violations") or records.get("modules_outside_at_exit"):
        return "guard-violation", EXIT_GUARD_VIOLATION
    if guard_missing or (expect_guard and records.get("python_processes_guarded", 0) == 0):
        return "guard-missing", EXIT_GUARD_MISSING
    return ("ok", 0) if rc == 0 else ("failed", rc)


def _is_python(cmd: list[str], python: str) -> bool:
    return bool(cmd) and (cmd[0] == python or os.path.basename(cmd[0]).startswith("python"))


def _job_id(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{safe}-{os.getpid()}"


def _reusable_gate(reg_dir: str, manifest_hash: str) -> dict | None:
    p = os.path.join(reg_dir, "gates", f"{manifest_hash}.json")
    try:
        with open(p) as f:
            g = json.load(f)
    except (OSError, ValueError):
        return None
    return g if g.get("verdict") == "green" else None


def cmd_run(a) -> int:
    repo = F.repo_root(a.repo)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        print("spheavy run: no command given after --", file=sys.stderr)
        return 2
    lane_dir = a.lane_dir or L.LANE_DIR
    job_id = _job_id(a.name)
    job_dir = os.path.join(a.jobs_dir or JOBS_DIR, job_id)
    os.makedirs(os.path.join(job_dir, "guard"))
    python = a.python or env_python(repo)
    launch = {"schema": "sp-launch/1", "job_id": job_id, "name": a.name, "kind": a.kind,
              "requested_command": cmd, "launched": L.now_utc(), "user": os.environ.get("USER"),
              "launcher_pid": os.getpid(), "cwd_at_launch": os.getcwd(), "repo": repo,
              "prereg": None, "frozen": bool(a.frozen)}

    def refuse(msg: str) -> int:
        launch["refused"] = msg
        F.write_json(os.path.join(job_dir, "launch.json"), launch)
        L.log_event("refused", lane_dir, job=job_id, reason=msg)
        print(f"spheavy: launch REFUSED: {msg}", file=sys.stderr)
        return EXIT_REFUSED

    # pre-registration (C.3-1)
    if a.prereg:
        text = open(a.prereg[1:]).read() if a.prereg.startswith("@") else a.prereg
        launch["prereg"] = text.strip()
    elif a.kind in PREREG_KINDS and not a.prereg_waived:
        return refuse(f"a {a.kind} job needs --prereg TEXT|@file (definitions and versions, expected "
                      "outcome, bias) or --prereg-waived REASON")
    if a.prereg_waived:
        launch["prereg_waived"] = a.prereg_waived

    # 1. frozen copy + launch-time tree state
    copy_root = None
    if a.frozen:
        copy_root = os.path.join(job_dir, "copy")
        try:
            man = F.build_copy(repo, copy_root, a.tree, a.include, a.exclude,
                               untracked_code=not a.no_untracked_code,
                               remove_for_test=a.test_remove_from_copy, read_only=not a.writable_copy)
            ts = F.tree_state(repo, man["trees"], os.path.join(job_dir, "tree.patch"))
        except F.FrozenCopyError as e:
            return refuse(f"frozen copy: {e}")
        unt = [f for f in man["files"] if f["origin"] in ("untracked-code", "included")]
        ts["untracked_included"] = [f["path"] for f in unt]
        if unt and ts["state"] == "clean":
            ts["state"] = "dirty (untracked files included)"
        tar = os.path.join(job_dir, "untracked.tar.gz")
        F.tar_files(copy_root, [f["path"] for f in unt if f["bytes"] <= 10 << 20], tar)
        F.write_json(os.path.join(job_dir, "manifest.json"), man)
        launch["manifest_hash"] = man["manifest_hash"]
        launch["trees"] = man["trees"]
        launch["copy"] = copy_root
        launch["copy_files"] = man["n_files"]
        launch["untracked_left_out"] = man["untracked_left_out"]
        launch["removed_for_test"] = man["removed_for_test"]
        launch["tree_state"] = ts
        try:
            from . import registry as R

            launch["provenance_store"] = R.store_provenance(
                _registry_dir(a), man["manifest_hash"], os.path.join(job_dir, "manifest.json"),
                os.path.join(job_dir, "tree.patch"), ts, untracked_tar=tar)
        except Exception as e:                               # never lose the job's own copy over it
            launch["provenance_store_error"] = repr(e)
    else:
        try:
            launch["tree_state"] = F.tree_state(repo, [], os.path.join(job_dir, "tree.patch"))
        except F.FrozenCopyError as e:
            launch["tree_state"] = {"error": str(e)}

    # 2. data files
    recs, problems = F.check_data(a.data, repo, copy_root)
    launch["data"] = recs
    if problems:
        return refuse("; ".join(problems))

    trees = launch.get("trees", [])
    run_cmd, notes = rewrite_command(cmd, repo, copy_root, python, trees)
    launch["command"] = run_cmd
    launch["command_notes"] = notes
    env = dict(os.environ)
    env.update({"SP_JOB_ID": job_id, "SP_JOB_DIR": job_dir})
    if copy_root:
        launch["pythonpath_before"] = env.get("PYTHONPATH")
        env.update({"SP_FROZEN_ROOT": copy_root,
                    "PYTHONPATH": os.pathsep.join([os.path.join(copy_root, F.GUARD_DIR),
                                                   os.path.join(copy_root, "src")]),
                    "SP_GUARD_MODE": "report" if a.guard_report_only else "strict",
                    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
        if a.guard_allow:
            env["SP_GUARD_ALLOW"] = os.pathsep.join(os.path.abspath(p) for p in a.guard_allow)
        if a.test_unset_pythonpath:                    # selftest: the known-bad of C.4
            env.pop("PYTHONPATH", None)
            launch["test_unset_pythonpath"] = True
    cwd = a.cwd or (copy_root if copy_root else os.getcwd())
    launch["job_cwd"] = cwd
    launch["env"] = {k: env.get(k) for k in ("PYTHONPATH", "SP_FROZEN_ROOT", "SP_JOB_DIR", "SP_JOB_ID",
                                              "SP_GUARD_MODE", "SP_GUARD_ALLOW", "OMP_NUM_THREADS")}
    if a.dry_run:
        F.write_json(os.path.join(job_dir, "launch.json"), launch)
        print(json.dumps({"job_dir": job_dir, "command": run_cmd, "notes": notes}, indent=1))
        return 0

    # 3. the lane
    policy = L.Policy(patterns=() if a.no_legacy_check else (a.legacy_pattern or L.LEGACY_PATTERNS),
                      docker=not a.no_docker_check, container=a.container,
                      mem_min_free_pct=a.mem_min_free, disk_min_gb=a.disk_min_gb,
                      disk_paths=tuple(dict.fromkeys(["/", *a.disk_path])), swap_max_mb=a.swap_max_mb,
                      on_busy=a.on_busy, poll_s=a.poll, max_wait_s=a.max_wait)
    launch["policy"] = policy.as_dict()
    lock = L.HeavyLock(lane_dir)
    info = {"job_id": job_id, "name": a.name, "kind": a.kind, "command": shlex.join(run_cmd),
            "job_dir": job_dir}
    L.log_event("request", lane_dir, job=job_id, name=a.name, kind=a.kind, command=shlex.join(run_cmd),
                manifest_hash=launch.get("manifest_hash"))
    try:
        launch["lane"] = L.enter_lane(lock, info, policy, lane_dir)
    except L.LaneBusy as e:
        launch["lane"] = {"busy": e.blockers}
        F.write_json(os.path.join(job_dir, "launch.json"), launch)
        print(f"spheavy: heavy lane busy, job not started: {json.dumps(e.blockers, default=str)}",
              file=sys.stderr)
        return EXIT_BUSY
    F.write_json(os.path.join(job_dir, "launch.json"), launch)

    try:
        # 4. fast gate
        if a.gate:
            reg = _registry_dir(a)
            prior = _reusable_gate(reg, launch.get("manifest_hash", "")) if copy_root else None
            if prior:
                launch["gate"] = {"reused": prior.get("result_file"), "verdict": "green"}
                L.log_event("gate-reused", lane_dir, job=job_id, manifest_hash=launch.get("manifest_hash"))
            else:
                gcmd = [python, "-m", "ngtagger.spcheck.gate", "run", "--json",
                        os.path.join(job_dir, "gate.json"), "--record-manifest",
                        launch.get("manifest_hash") or "none"]
                L.log_event("gate-start", lane_dir, job=job_id)
                genv = dict(env, SP_JOB_DIR=os.path.join(job_dir, "gate"))   # its own guard records
                rc = subprocess.call(gcmd, env=genv, cwd=cwd)
                launch["gate"] = {"rc": rc, "result": os.path.join(job_dir, "gate.json"),
                                  "verdict": "green" if rc == 0 else "red"}
                L.log_event("gate-end", lane_dir, job=job_id, rc=rc)
                if rc != 0:
                    F.write_json(os.path.join(job_dir, "launch.json"), launch)
                    print("spheavy: fast gate RED, job not started", file=sys.stderr)
                    return EXIT_GATE_RED
            F.write_json(os.path.join(job_dir, "launch.json"), launch)

        # 5. the job
        t0 = time.time()
        L.log_event("start", lane_dir, job=job_id, command=shlex.join(run_cmd), cwd=cwd)
        os.set_inheritable(lock.fd, True)
        child = subprocess.Popen(run_cmd, env=env, cwd=cwd, pass_fds=(lock.fd,))

        def forward(sig, _frame):
            L.log_event("signal", lane_dir, job=job_id, signal=sig)
            try:
                child.send_signal(sig)
            except ProcessLookupError:
                pass

        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, forward)
        guard_missing = False
        expect_guard = bool(copy_root) and _is_python(run_cmd, python)
        while child.poll() is None:
            if (expect_guard and time.time() - t0 > a.guard_start_timeout
                    and not glob.glob(os.path.join(job_dir, "guard", "*.start.json"))):
                guard_missing = True
                L.log_event("guard-missing", lane_dir, job=job_id,
                            detail="no guarded Python process after the start timeout; terminating")
                child.terminate()
                try:
                    child.wait(30)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
            time.sleep(0.5)
        rc = child.wait()
        wall = round(time.time() - t0, 1)

        # 6. the record
        g = guard_records(job_dir) if copy_root else {}
        if copy_root:
            status, out_rc = guard_verdict(rc, g, expect_guard or a.expect_python, guard_missing)
        else:
            status, out_rc = ("ok", 0) if rc == 0 else ("failed", rc)
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak_gb = peak / 2**30 if sys.platform == "darwin" else peak / 2**20
        rec = {"job_id": job_id, "rc": rc, "status": status, "exit_code": out_rc, "wall_s": wall,
               "ended": L.now_utc(), "peak_child_rss_gb": round(peak_gb, 3), "guard": g}
        F.write_json(os.path.join(job_dir, "exit.json"), rec)
        L.log_event("finished", lane_dir, job=job_id, rc=rc, status=status, wall_s=wall,
                    peak_child_rss_gb=round(peak_gb, 3))
        if status == "guard-missing":
            print("spheavy: JOB FAILED: its Python never ran the import guard -- PYTHONPATH did not point at "
                  "the frozen copy, so it may have imported the shared working tree", file=sys.stderr)
        elif status == "guard-violation":
            print(f"spheavy: JOB FAILED: import-guard violation(s): "
                  f"{json.dumps(g['violations'] or g['modules_outside_at_exit'], default=str)[:2000]}",
                  file=sys.stderr)
        elif status == "failed":
            print(f"spheavy: JOB FAILED with exit code {rc}", file=sys.stderr)
        return out_rc
    finally:
        lock.close_own_descriptor()
        if lock.held_by_anyone():
            L.log_event("release-pending", lane_dir, job=job_id,
                        detail="a leftover process of the job still holds the lock descriptor",
                        users=L.lock_users(lane_dir))
        else:
            try:
                with open(lock.holder_path) as f:
                    if json.load(f).get("token") == lock.token:
                        os.remove(lock.holder_path)
            except (OSError, ValueError):
                pass
            L.log_event("released", lane_dir, job=job_id)
        registry_snapshot(job_id)


def registry_snapshot(job_id: str) -> None:
    """Copy the results registry to WDMac after a job (ops/registry_snapshot.sh: only when it changed;
    skips without WDMac). Runs after the lock is released and never affects the job's exit code."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ops", "registry_snapshot.sh")
    if os.path.exists(script):
        try:
            subprocess.run(["bash", script, job_id], timeout=120, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            pass


def cmd_status(a) -> int:
    lane_dir = a.lane_dir or L.LANE_DIR
    lock = L.HeavyLock(lane_dir)
    blockers, snap = L.scan(L.Policy(container=a.container))
    out = {"lane_dir": lane_dir, "lock_held": lock.held_by_anyone(), "holder": lock.holder(),
           "lock_users": L.lock_users(lane_dir), "blockers": blockers, "snapshot": snap}
    print(json.dumps(out, indent=1, default=str))
    return 0


def cmd_log(a) -> int:
    p = os.path.join(a.lane_dir or L.LANE_DIR, "heavy.log")
    if os.path.exists(p):
        with open(p) as f:
            for line in f.readlines()[-a.n:]:
                print(line.rstrip())
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="spheavy", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="what", required=True)
    r = sub.add_parser("run", help="run a heavy job through the lane")
    r.add_argument("--name", required=True)
    r.add_argument("--kind", choices=KINDS, required=True)
    r.add_argument("--frozen", action="store_true", help="run from a frozen copy of whole trees")
    r.add_argument("--tree", action="append", default=[], help="repo-relative tree to copy (src/ngtagger always)")
    r.add_argument("--include", action="append", default=[], help="extra file or directory for the copy")
    r.add_argument("--exclude", action="append", default=[], help="glob of repo-relative paths to leave out")
    r.add_argument("--no-untracked-code", action="store_true", help="copy tracked files and --include only")
    r.add_argument("--writable-copy", action="store_true", help="do not make the copy read-only")
    r.add_argument("--data", action="append", default=[], help="a data file the job reads (checked at launch)")
    r.add_argument("--prereg", help="pre-registration text, or @file")
    r.add_argument("--prereg-waived", help="reason a census/tagger/replay job has no pre-registration")
    r.add_argument("--gate", action="store_true", help="run the fast gate first (reused per manifest hash)")
    r.add_argument("--python", help="interpreter (default: the repo's pixi env)")
    r.add_argument("--cwd", help="job working directory (default: the copy root, or here)")
    r.add_argument("--on-busy", choices=("wait", "skip"), default="wait")
    r.add_argument("--poll", type=float, default=60.0)
    r.add_argument("--max-wait", type=float, default=None)
    r.add_argument("--mem-min-free", type=int, default=L.MEM_MIN_FREE_PCT)
    r.add_argument("--disk-min-gb", type=float, default=L.DISK_MIN_GB)
    r.add_argument("--disk-path", action="append", default=[], help="another volume that needs --disk-min-gb")
    r.add_argument("--swap-max-mb", type=float, default=None)
    r.add_argument("--container", default=L.DOCKER_CONTAINER)
    r.add_argument("--guard-allow", action="append", default=[], help="extra root the guard accepts")
    r.add_argument("--guard-report-only", action="store_true", help="record violations, fail at exit")
    r.add_argument("--guard-start-timeout", type=float, default=GUARD_START_TIMEOUT_S)
    r.add_argument("--expect-python", action="store_true",
                   help="a wrapped command (shell script) runs Python: fail if none ran the guard")
    r.add_argument("--repo", default=DEFAULT_REPO)
    r.add_argument("--jobs-dir")
    r.add_argument("--lane-dir")
    r.add_argument("--registry-dir")
    r.add_argument("--dry-run", action="store_true", help="build the copy and the record, do not run")
    # tests only (ngtagger.spops.selftest): the C.4 known-bads and private lanes
    r.add_argument("--legacy-pattern", action="append", help=argparse.SUPPRESS)
    r.add_argument("--no-legacy-check", action="store_true", help=argparse.SUPPRESS)
    r.add_argument("--no-docker-check", action="store_true", help=argparse.SUPPRESS)
    r.add_argument("--test-unset-pythonpath", action="store_true", help=argparse.SUPPRESS)
    r.add_argument("--test-remove-from-copy", action="append", default=[], help=argparse.SUPPRESS)
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    s = sub.add_parser("status", help="lock holder, legacy jobs, Docker, resources (read-only)")
    s.add_argument("--lane-dir")
    s.add_argument("--container", default=L.DOCKER_CONTAINER)
    g = sub.add_parser("log", help="the last lane events")
    g.add_argument("-n", type=int, default=20)
    g.add_argument("--lane-dir")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    return {"run": cmd_run, "status": cmd_status, "log": cmd_log}[a.what](a)


if __name__ == "__main__":
    sys.exit(main())
