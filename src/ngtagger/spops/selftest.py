"""End-to-end self-test of the heavy lane and the frozen-copy launcher, each check shown
passing on a good input and failing on its known-bad (plan App. C.4, D.1).

    python -m ngtagger.spops.selftest [--work DIR] [--only NAME ...] [--json OUT]

Everything runs through the real CLI (ops/spheavy) on a PRIVATE lane, jobs and registry
directory under --work, so the machine-wide lock is never taken. The Docker scenario only
reads the container's process list. Jobs are seconds long and import numpy/awkward at most.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from . import lane as L

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SPHEAVY = os.path.join(REPO, "ops", "spheavy")
PROBE_TREE = "src/ngtagger/spops/probe"
PY = os.path.join(REPO, ".pixi", "envs", "default", "bin", "python")
DUMMY = "sp_selftest_dummy_legacy.py"


class Ctx:
    def __init__(self, work: str):
        self.work = work
        self.lane = os.path.join(work, "lane")
        self.jobs = os.path.join(work, "jobs")
        self.reg = os.path.join(work, "registry")
        for d in (self.lane, self.jobs, self.reg):
            os.makedirs(d, exist_ok=True)

    def base(self, name: str, kind: str = "selftest") -> list[str]:
        return [SPHEAVY, "run", "--name", name, "--kind", kind, "--lane-dir", self.lane, "--jobs-dir", self.jobs,
                "--registry-dir", self.reg, "--mem-min-free", "0", "--poll", "1"]

    def run(self, args: list[str], env_extra: dict | None = None, timeout: float = 300) -> dict:
        env = dict(os.environ, SP_LANE_QUIET="1", **(env_extra or {}))
        t0 = time.time()
        r = subprocess.run(args, capture_output=True, text=True, env=env, timeout=timeout)
        return {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "wall_s": round(time.time() - t0, 2)}

    def last_job(self, name: str) -> str:
        js = sorted(d for d in os.listdir(self.jobs) if f"-{name}-" in d)
        return os.path.join(self.jobs, js[-1])

    def lane_events(self) -> list[dict]:
        p = os.path.join(self.lane, "heavy.log")
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return [json.loads(x) for x in f if x.strip()]


def _probe_report(stdout: str) -> dict | None:
    for line in stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[6:])
    return None


def _frozen(ctx: Ctx, name: str, extra=(), env_extra=None) -> tuple[dict, str]:
    args = ctx.base(name) + ["--frozen", "--tree", PROBE_TREE, "--include", f"{PROBE_TREE}/probe_data.txt",
                             "--no-legacy-check", "--no-docker-check", *extra, "--", "python", f"{PROBE_TREE}/probe_job.py"]
    r = ctx.run(args, env_extra)
    return r, ctx.last_job(name)


def _exit(job: str) -> dict:
    p = os.path.join(job, "exit.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _inside(path: str | None, root: str) -> bool:
    return bool(path) and os.path.realpath(path).startswith(os.path.realpath(root) + os.sep)


# ---------------------------------------------------------------- scenarios
def s_frozen_green(ctx):
    r, job = _frozen(ctx, "frozen-green")
    rep, ex = _probe_report(r["stdout"]) or {}, _exit(job)
    copy = os.path.join(job, "copy")
    launch = json.load(open(os.path.join(job, "launch.json")))
    mods = {k: rep.get(k) for k in ("ngtagger", "labels", "lazy_helper", "lazy_pkg")}
    ok = (r["rc"] == 0 and ex.get("status") == "ok" and all(_inside(v, copy) for v in mods.values())
          and rep.get("data") == "probe data v1" and ex["guard"]["python_processes_guarded"] >= 1
          and launch.get("manifest_hash") and os.path.exists(os.path.join(job, "tree.patch")))
    return ok, {"rc": r["rc"], "status": ex.get("status"), "modules": mods, "manifest_hash": launch.get("manifest_hash"),
                "tree_state": launch["tree_state"]["state"], "copy_files": launch.get("copy_files"),
                "guarded_processes": ex.get("guard", {}).get("python_processes_guarded")}


def s_kb_pythonpath_unset(ctx):
    r, job = _frozen(ctx, "kb-pythonpath-unset", ["--test-unset-pythonpath"])
    rep, ex = _probe_report(r["stdout"]) or {}, _exit(job)
    leaked = rep.get("ngtagger")
    ok = r["rc"] == 98 and ex.get("status") == "guard-missing" and leaked and not _inside(leaked, job)
    return ok, {"rc": r["rc"], "status": ex.get("status"), "ngtagger_imported_from": leaked,
                "launcher_said": [x for x in r["stderr"].splitlines() if "spheavy" in x][-1:]}


def s_kb_lazy_module_shared_fallback(ctx):
    r, job = _frozen(ctx, "kb-lazy-shared", ["--test-remove-from-copy", f"{PROBE_TREE}/probe_lazy_helper.py"],
                     {"SP_PROBE_SHARED_DIR": os.path.join(REPO, PROBE_TREE)})
    ex = _exit(job)
    v = (ex.get("guard") or {}).get("violations") or []
    ok = r["rc"] == 97 and ex.get("status") == "guard-violation" and any(x["name"] == "probe_lazy_helper" for x in v)
    return ok, {"rc": r["rc"], "status": ex.get("status"), "violations": [(x["kind"], x["name"], x["path"]) for x in v]}


def s_kb_lazy_package_module_missing(ctx):
    r, job = _frozen(ctx, "kb-lazy-missing", ["--test-remove-from-copy", "src/ngtagger/train/prediction_dump.py"])
    ex = _exit(job)
    err = [x for x in r["stderr"].splitlines() if "ImportError" in x or "ModuleNotFoundError" in x][-1:]
    ok = r["rc"] not in (0, 97, 98) and ex.get("status") == "failed" and bool(err)
    return ok, {"rc": r["rc"], "status": ex.get("status"), "error": err}


def s_limitation_swallowed_lazy_import(ctx):
    """NOT a known-bad: documents that an import failing inside try/except (the
    trainer.py:164-169 pattern) is invisible to any import hook. Expected: rc 0."""
    r, job = _frozen(ctx, "limit-swallowed", ["--test-remove-from-copy", f"{PROBE_TREE}/probe_lazy_helper.py"],
                     {"SP_PROBE_SWALLOW": "1"})
    rep = _probe_report(r["stdout"]) or {}
    ok = r["rc"] == 0 and str(rep.get("lazy_helper", "")).startswith("skipped")
    return ok, {"rc": r["rc"], "lazy_helper": rep.get("lazy_helper"),
                "note": "silent by construction: nothing is loaded, so nothing can be checked"}


def s_kb_data_missing(ctx):
    r1, job1 = _frozen(ctx, "data-undeclared", ["--test-remove-from-copy", f"{PROBE_TREE}/probe_data.txt"])
    rep1 = _probe_report(r1["stdout"]) or {}
    r2, job2 = _frozen(ctx, "kb-data-missing", ["--test-remove-from-copy", f"{PROBE_TREE}/probe_data.txt",
                                                "--data", f"{PROBE_TREE}/probe_data.txt"])
    ran = os.path.exists(os.path.join(job2, "exit.json"))
    ok = rep1.get("data") is None and r1["rc"] == 0 and r2["rc"] == 66 and not ran
    return ok, {"undeclared": {"rc": r1["rc"], "data_read": rep1.get("data"), "note": "runs, silently without its data"},
                "declared": {"rc": r2["rc"], "job_ran": ran,
                             "refusal": [x for x in r2["stderr"].splitlines() if "REFUSED" in x][-1:]}}


def _sleep_cmd(s: float) -> list[str]:
    return [PY, "-c", f"import time; time.sleep({s})"]


def s_lock_serialises(ctx):
    a = subprocess.Popen(ctx.base("lock-a") + ["--no-legacy-check", "--no-docker-check", "--", *_sleep_cmd(5)],
                         env=dict(os.environ, SP_LANE_QUIET="1"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2.0)
    rb = ctx.run(ctx.base("lock-b") + ["--no-legacy-check", "--no-docker-check", "--", *_sleep_cmd(0.1)])
    a.wait()
    ev = ctx.lane_events()
    ja, jb = ctx.last_job("lock-a"), ctx.last_job("lock-b")
    ida, idb = os.path.basename(ja), os.path.basename(jb)
    fin_a = [e for e in ev if e.get("job") == ida and e["ev"] == "finished"]
    acq_b = [e for e in ev if e.get("job") == idb and e["ev"] == "acquired"]
    waits_b = [e for e in ev if e.get("job") == idb and e["ev"] == "wait"]
    ok = (a.returncode == 0 and rb["rc"] == 0 and fin_a and acq_b and fin_a[0]["t"] <= acq_b[0]["t"]
          and any(b["kind"] == "lock" for w in waits_b for b in w["blockers"]))
    return ok, {"a_rc": a.returncode, "b_rc": rb["rc"], "b_waited_s": acq_b[0]["waited_s"] if acq_b else None,
                "b_wait_blockers": [b["kind"] for w in waits_b for b in w["blockers"]]}


def s_lock_skip_when_busy(ctx):
    a = subprocess.Popen(ctx.base("skip-holder") + ["--no-legacy-check", "--no-docker-check", "--", *_sleep_cmd(4)],
                         env=dict(os.environ, SP_LANE_QUIET="1"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2.0)
    rc = ctx.run(ctx.base("skip-c") + ["--no-legacy-check", "--no-docker-check", "--on-busy", "skip", "--",
                                      *_sleep_cmd(0.1)])
    a.wait()
    idc = os.path.basename(ctx.last_job("skip-c"))
    skips = [e for e in ctx.lane_events() if e.get("job") == idc and e["ev"] == "skip"]
    ok = rc["rc"] == 75 and skips and not os.path.exists(os.path.join(ctx.last_job("skip-c"), "exit.json"))
    return ok, {"rc": rc["rc"], "skip_logged": bool(skips), "blockers": skips[0]["blockers"] if skips else None}


def s_stale_holder(ctx):
    d = subprocess.Popen(ctx.base("stale-d") + ["--no-legacy-check", "--no-docker-check", "--", *_sleep_cmd(60)],
                         env=dict(os.environ, SP_LANE_QUIET="1"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         start_new_session=True)
    time.sleep(2.5)
    os.killpg(d.pid, signal.SIGKILL)                  # launcher and job die without releasing
    d.wait()
    time.sleep(0.5)
    re_ = ctx.run(ctx.base("stale-e") + ["--no-legacy-check", "--no-docker-check", "--max-wait", "20", "--",
                                        *_sleep_cmd(0.1)])
    ev = ctx.lane_events()
    ide = os.path.basename(ctx.last_job("stale-e"))
    stale = [e for e in ev if e["ev"] == "stale-holder-record" and e["pid"] != d.pid]
    acq = [e for e in ev if e.get("job") == ide and e["ev"] == "acquired"]
    ok = re_["rc"] == 0 and stale and acq and acq[0]["waited_s"] < 5
    return ok, {"rc": re_["rc"], "stale_record_logged": bool(stale),
                "stale_previous_pid_alive": stale[-1].get("previous_pid_alive") if stale else None,
                "waited_s": acq[0]["waited_s"] if acq else None}


def s_legacy_wait(ctx):
    dummy = subprocess.Popen(["bash", "-c", f"exec -a {DUMMY} sleep 5"])
    time.sleep(0.5)
    # the job's own command line carries the pattern: it must not block itself
    rf = ctx.run(ctx.base("legacy-f") + ["--no-docker-check", "--legacy-pattern", DUMMY, "--",
                                        PY, "-c", f"print('{DUMMY}')"])
    dummy.wait()
    idf = os.path.basename(ctx.last_job("legacy-f"))
    ev = ctx.lane_events()
    waits = [b for e in ev if e.get("job") == idf and e["ev"] == "wait" for b in e["blockers"]]
    acq = [e for e in ev if e.get("job") == idf and e["ev"] == "acquired"]
    ok = (rf["rc"] == 0 and any(b.get("pid") == dummy.pid for b in waits) and acq and acq[0]["waited_s"] >= 3)
    return ok, {"rc": rf["rc"], "waited_s": acq[0]["waited_s"] if acq else None,
                "blocked_by": [(b.get("pattern"), b.get("pid"), b.get("cmd")) for b in waits][:2]}


def s_plain_pattern_self_match(ctx):
    """Why the bracketed form. macOS pgrep skips its own ancestors, but not OTHER waiters:
    two shell loops `until ! pgrep -f <plain pattern>` each see the other's command line
    and wait forever although no job runs (Linux pgrep, e.g. in the container, even sees
    its own parent shell). The same two loops with the bracketed pattern run at once."""
    def waiters(pat: str) -> list[int]:
        loop = f"until ! pgrep -f '{pat}' >/dev/null; do sleep 0.3; done; exit 0"
        ps = [subprocess.Popen(["bash", "-c", loop]) for _ in range(2)]
        t0, rcs = time.time(), [None, None]
        while time.time() - t0 < 5 and None in rcs:
            rcs = [p.poll() for p in ps]
            time.sleep(0.2)
        for p in ps:
            if p.poll() is None:
                p.kill()
                p.wait()
        return rcs
    plain = waiters("sp_selftest_plain_marker")
    brk = waiters(L.bracketed("sp_selftest_plain_marker"))
    ok = plain == [None, None] and brk == [0, 0]
    return ok, {"plain_pattern_waiters_finished_in_5s": [x is not None for x in plain],
                "bracketed_waiters_finished_in_5s": [x is not None for x in brk],
                "bracketed": L.bracketed("sp_selftest_plain_marker")}


def s_resources_block(ctx):
    rm = ctx.run(ctx.base("res-mem")[:-4] + ["--mem-min-free", "101", "--poll", "1", "--no-legacy-check",
                                              "--no-docker-check", "--on-busy", "skip", "--", *_sleep_cmd(0)])
    rd = ctx.run(ctx.base("res-disk") + ["--disk-min-gb", "1e9", "--no-legacy-check", "--no-docker-check",
                                         "--on-busy", "skip", "--", *_sleep_cmd(0)])
    ev = ctx.lane_events()
    kinds = {n: [b["kind"] for e in ev if e.get("job") == os.path.basename(ctx.last_job(n)) and e["ev"] == "skip"
                 for b in e["blockers"]] for n in ("res-mem", "res-disk")}
    ok = rm["rc"] == 75 and rd["rc"] == 75 and "memory" in kinds["res-mem"] and "disk" in kinds["res-disk"]
    return ok, {"rc_mem": rm["rc"], "rc_disk": rd["rc"], "blockers": kinds}


def s_docker_live(ctx):
    """Reads the real container's process list (no lock taken): while the CMSSW production
    runs, a launch must refuse (skip) with the cmsRun as the blocker."""
    state = L.docker_running()
    r = ctx.run(ctx.base("docker-live") + ["--no-legacy-check", "--on-busy", "skip", "--", *_sleep_cmd(0)])
    idd = os.path.basename(ctx.last_job("docker-live"))
    blk = [b for e in ctx.lane_events() if e.get("job") == idd and e["ev"] == "skip" for b in e["blockers"]]
    if state["busy"]:
        ok = r["rc"] == 75 and any(b["kind"] == "docker" for b in blk)
    else:
        ok = r["rc"] == 0
    return ok, {"container_state": state, "rc": r["rc"], "blockers": blk}


SCENARIOS = {
    "frozen-green": s_frozen_green,
    "kb-pythonpath-unset": s_kb_pythonpath_unset,
    "kb-lazy-module-shared-fallback": s_kb_lazy_module_shared_fallback,
    "kb-lazy-package-module-missing": s_kb_lazy_package_module_missing,
    "limitation-swallowed-lazy-import": s_limitation_swallowed_lazy_import,
    "kb-data-missing": s_kb_data_missing,
    "lock-serialises": s_lock_serialises,
    "lock-skip-when-busy": s_lock_skip_when_busy,
    "stale-holder": s_stale_holder,
    "legacy-wait": s_legacy_wait,
    "plain-pattern-self-match": s_plain_pattern_self_match,
    "resources-block": s_resources_block,
    "docker-live": s_docker_live,
}


def run_all(work: str, only=None) -> dict:
    ctx = Ctx(work)
    out = {}
    for name, fn in SCENARIOS.items():
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            ok, details = fn(ctx)
        except Exception as e:                               # a crash is a failure, reported
            ok, details = False, {"exception": repr(e)}
        out[name] = {"pass": bool(ok), "wall_s": round(time.time() - t0, 1), "details": details}
        print(f"{'PASS' if ok else 'FAIL'}  {name:36s} {json.dumps(details, default=str)[:300]}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(L.HOME, "smartpixels", "sp_selftest",
                                                   time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--json")
    ap.add_argument("--clean", action="store_true", help="delete the work directory afterwards")
    a = ap.parse_args(argv)
    res = run_all(a.work, a.only)
    summary = {"work": a.work, "when": L.now_utc(), "results": res,
               "all_pass": all(v["pass"] for v in res.values())}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=1, default=str)
    if a.clean:
        from .frozen import make_writable

        make_writable(a.work)
        shutil.rmtree(a.work)
    print("ALL PASS" if summary["all_pass"] else "SOME FAILED")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
