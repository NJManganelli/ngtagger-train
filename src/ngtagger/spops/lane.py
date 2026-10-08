"""The heavy-job lane: one machine-wide lock plus the checks a launcher makes before a
heavy job (census, tagger training, replay scan, CMSSW cmsRun or build in Docker).

Rules implemented here (plan App. C.3-3):
  * one lock for the whole machine: an flock on <lane>/heavy.lock. The kernel drops it
    when every process holding the descriptor has exited, so a crashed holder never
    leaves a stale lock; holder.json records who took it (pid, job, command) and is
    reported as stale when its pid is gone;
  * legacy launchers that cannot take the lock are waited on by their command lines,
    always with the bracketed pgrep form (``pgrep -lf "[t]p_findability[.]py"``), which
    cannot match the waiting process's own command line; the launcher also skips its own
    pid and its ancestors;
  * Docker is checked INSIDE the container (``docker exec <c> ps -C cmsRun,scram``):
    host command lines miss ``docker exec -d``;
  * resources (free memory, disk, swap) are checked before every launch;
  * every wait, skip, acquisition, re-check and release is one JSON line in <lane>/heavy.log.

Stdlib only.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
LANE_DIR = os.environ.get("SP_HEAVY_DIR", os.path.join(HOME, "smartpixels", "heavy_lane"))
DOCKER_CONTAINER = os.environ.get("SP_DOCKER_CONTAINER", "cmssw-el9-native-1")
DOCKER_PROCS = ("cmsRun", "scram")

# Launchers that do not take the lock (plan C.3-3), plus the CMSSW production driver,
# which leaves Docker idle for seconds between files and checks only for cmsRun itself.
LEGACY_PATTERNS = (
    "tp_findability.py",
    "tagger_sp/train.py",
    "run_d0.sh",
    "export_site.py",
    "spix_combinatorics_omnibus.py",
    "ot_refit_projection.py",
    "train-trackmva",
    "drive_v2.sh",
)

# RESOURCE RULE defaults: free memory >= 25 %, >= 10 GB free on the internal disk.
MEM_MIN_FREE_PCT = 25
DISK_MIN_GB = 10.0


# ---------------------------------------------------------------- logging
def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log_event(event: str, lane_dir: str = LANE_DIR, **fields) -> dict:
    """Append one JSON line to <lane>/heavy.log (and echo it to stderr)."""
    rec = {"t": now_utc(), "ev": event, "pid": os.getpid(), "host": socket.gethostname()}
    rec.update(fields)
    line = json.dumps(rec, sort_keys=False, default=str)
    os.makedirs(lane_dir, exist_ok=True)
    fd = os.open(os.path.join(lane_dir, "heavy.log"), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, (line + "\n").encode())
    finally:
        os.close(fd)
    if os.environ.get("SP_LANE_QUIET") != "1":
        print(f"[spheavy] {event}: {line}", file=sys.stderr, flush=True)
    return rec


# ---------------------------------------------------------------- processes
def bracketed(pattern: str) -> str:
    """The pgrep pattern that matches `pattern` in other command lines but not its own
    text: the first letter in brackets, every '.' as '[.]' ('tp_findability.py' ->
    '[t]p_findability[.]py')."""
    out, done = [], False
    for ch in pattern:
        if ch == ".":
            out.append("[.]")
        elif not done and ch.isalnum():
            out.append(f"[{ch}]")
            done = True
        elif ch in "^$*+?()[]{}|\\":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def process_parents() -> dict[int, int]:
    r = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, check=False)
    parent = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            parent[int(parts[0])] = int(parts[1])
    return parent


def ancestors(pid: int, parent: dict[int, int] | None = None) -> set[int]:
    parent = process_parents() if parent is None else parent
    out, p = set(), pid
    while p and p not in out and p > 1:
        out.add(p)
        p = parent.get(p, 0)
    return out


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def legacy_running(patterns=LEGACY_PATTERNS, exclude: set[int] | None = None) -> list[dict]:
    """Processes whose command line matches a legacy pattern (bracketed pgrep form),
    excluding `exclude` (default: this process and its ancestors)."""
    if exclude is None:
        exclude = ancestors(os.getpid())
    found = []
    for pat in patterns:
        r = subprocess.run(["pgrep", "-lf", bracketed(pat)], capture_output=True, text=True, check=False)
        for line in r.stdout.splitlines():
            pid_s, _, cmd = line.strip().partition(" ")
            if not pid_s.isdigit() or int(pid_s) in exclude:
                continue
            found.append({"kind": "legacy", "pattern": pat, "pid": int(pid_s), "cmd": cmd[:240]})
    return found


def docker_running(container: str = DOCKER_CONTAINER, procs=DOCKER_PROCS, timeout: float = 30.0) -> dict:
    """{'busy': bool, 'state': ..., 'procs': [...]} from ps INSIDE the container.
    A missing CLI, a stopped daemon or an absent container cannot run cmsRun (not busy,
    state recorded); any other failure is treated as busy (unknown is not idle)."""
    if not shutil.which("docker"):
        return {"busy": False, "state": "no-docker-cli", "procs": []}
    try:
        r = subprocess.run(["docker", "exec", container, "ps", "-C", ",".join(procs), "-o", "pid=,etime=,args="],
                           capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"busy": True, "state": "unknown: docker exec timed out", "procs": []}
    out, err = r.stdout.strip(), r.stderr.strip()
    if out:
        return {"busy": True, "state": "running", "procs": [ln.strip()[:200] for ln in out.splitlines()]}
    if r.returncode in (0, 1) and not err:
        return {"busy": False, "state": "idle", "procs": []}
    low = err.lower()
    for key, state in (("no such container", "container-absent"), ("is not running", "container-stopped"),
                       ("cannot connect to the docker daemon", "daemon-down"),
                       ("docker daemon is not running", "daemon-down")):
        if key in low:
            return {"busy": False, "state": state, "procs": []}
    return {"busy": True, "state": f"unknown: rc={r.returncode} {err[:200]}", "procs": []}


# ---------------------------------------------------------------- resources
def mem_free_pct() -> int | None:
    try:
        r = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"free percentage:\s*(\d+)%", r.stdout)
    return int(m.group(1)) if m else None


def swap_used_mb() -> float | None:
    r = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False)
    m = re.search(r"used\s*=\s*([\d.]+)M", r.stdout)
    return float(m.group(1)) if m else None


def disk_free_gb(path: str) -> float:
    p = path
    while p and not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p or "/").free / 2**30


def resources(disk_paths=("/",)) -> dict:
    return {"mem_free_pct": mem_free_pct(), "swap_used_mb": swap_used_mb(),
            "disk_free_gb": {p: round(disk_free_gb(p), 1) for p in disk_paths}}


# ---------------------------------------------------------------- policy + scan
class Policy:
    """What a launch waits for. Every check can be switched off only explicitly."""

    def __init__(self, patterns=LEGACY_PATTERNS, docker=True, container=DOCKER_CONTAINER,
                 mem_min_free_pct=MEM_MIN_FREE_PCT, disk_min_gb=DISK_MIN_GB, disk_paths=("/",),
                 swap_max_mb=None, on_busy="wait", poll_s=60.0, max_wait_s=None):
        self.patterns = tuple(patterns)
        self.docker = docker
        self.container = container
        self.mem_min_free_pct = mem_min_free_pct
        self.disk_min_gb = disk_min_gb
        self.disk_paths = tuple(disk_paths)
        self.swap_max_mb = swap_max_mb
        self.on_busy = on_busy
        self.poll_s = poll_s
        self.max_wait_s = max_wait_s

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def scan(policy: Policy, exclude: set[int] | None = None) -> tuple[list[dict], dict]:
    """(blockers, snapshot). An empty blocker list means the lane may be entered."""
    blockers = legacy_running(policy.patterns, exclude) if policy.patterns else []
    snap: dict = {}
    if policy.docker:
        d = docker_running(policy.container)
        snap["docker"] = d
        if d["busy"]:
            blockers.append({"kind": "docker", "container": policy.container, "state": d["state"],
                             "procs": d["procs"]})
    res = resources(policy.disk_paths)
    snap["resources"] = res
    if policy.mem_min_free_pct is not None:
        mf = res["mem_free_pct"]
        if mf is None or mf < policy.mem_min_free_pct:
            blockers.append({"kind": "memory", "free_pct": mf, "min_pct": policy.mem_min_free_pct})
    if policy.disk_min_gb is not None:
        for p, gb in res["disk_free_gb"].items():
            if gb < policy.disk_min_gb:
                blockers.append({"kind": "disk", "path": p, "free_gb": gb, "min_gb": policy.disk_min_gb})
    if policy.swap_max_mb is not None and (res["swap_used_mb"] or 0) > policy.swap_max_mb:
        blockers.append({"kind": "swap", "used_mb": res["swap_used_mb"], "max_mb": policy.swap_max_mb})
    return blockers, snap


def _signature(blockers: list[dict]) -> str:
    return json.dumps(sorted((b["kind"], b.get("pattern", ""), b.get("pid", 0), b.get("state", ""))
                             for b in blockers))


# ---------------------------------------------------------------- the lock
class LaneBusy(RuntimeError):
    def __init__(self, blockers):
        super().__init__(f"heavy lane busy: {blockers}")
        self.blockers = blockers


class HeavyLock:
    """flock on <lane>/heavy.lock; holder.json says who holds it."""

    def __init__(self, lane_dir: str = LANE_DIR):
        self.lane_dir = lane_dir
        os.makedirs(lane_dir, exist_ok=True)
        self.path = os.path.join(lane_dir, "heavy.lock")
        self.holder_path = os.path.join(lane_dir, "holder.json")
        self.fd: int | None = None
        self.token: str | None = None

    def held_by_anyone(self) -> bool:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            os.close(fd)        # closing drops the probe's lock
        return False

    def holder(self) -> dict | None:
        try:
            with open(self.holder_path) as f:
                h = json.load(f)
        except (OSError, ValueError):
            h = None
        if h is not None:
            h["pid_alive"] = pid_alive(h.get("pid", 0))
            h["lock_held"] = self.held_by_anyone()
        return h

    def try_acquire(self, info: dict) -> bool:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        self.fd = fd
        prev = None
        try:
            with open(self.holder_path) as f:
                prev = json.load(f)
        except (OSError, ValueError):
            pass
        if prev:
            log_event("stale-holder-record", self.lane_dir, previous=prev,
                      previous_pid_alive=pid_alive(prev.get("pid", 0)))
        self.token = f"{os.getpid()}-{time.time_ns()}"
        rec = dict(info, pid=os.getpid(), token=self.token, acquired=now_utc(), host=socket.gethostname())
        tmp = self.holder_path + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        os.replace(tmp, self.holder_path)
        os.ftruncate(fd, 0)
        os.pwrite(fd, (json.dumps(rec, default=str) + "\n").encode(), 0)
        return True

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            with open(self.holder_path) as f:
                h = json.load(f)
            if h.get("token") == self.token:
                os.remove(self.holder_path)
        except (OSError, ValueError):
            pass
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None

    def close_own_descriptor(self) -> None:
        """Close this process's descriptor without unlocking: a child that inherited it
        keeps the lane until it (and every process it handed the descriptor to) exits."""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def lock_users(lane_dir: str = LANE_DIR) -> list[str]:
    """Processes with heavy.lock open (lsof), for diagnosing a lock kept by a leftover."""
    path = os.path.join(lane_dir, "heavy.lock")
    if not shutil.which("lsof") or not os.path.exists(path):
        return []
    r = subprocess.run(["lsof", "-F", "pc", path], capture_output=True, text=True, check=False)
    out, cur = [], None
    for line in r.stdout.splitlines():
        if line.startswith("p"):
            cur = line[1:]
        elif line.startswith("c") and cur:
            out.append(f"{cur} {line[1:]}")
    return out


def enter_lane(lock: HeavyLock, info: dict, policy: Policy, lane_dir: str = LANE_DIR,
               exclude: set[int] | None = None) -> dict:
    """Wait until no blocker is present, take the lock, re-check every blocker with the lock
    held, and return wait statistics. Raises LaneBusy under on_busy='skip' (or when
    max_wait_s runs out). Every wait, re-check, skip and acquisition is logged."""
    t0 = time.time()
    last_sig, last_log, n_wait, seen = None, 0.0, 0, []
    job = info.get("job_id")
    while True:
        blockers, snap = scan(policy, exclude)
        if not blockers:
            if lock.try_acquire(info):
                blockers, snap = scan(policy, exclude)        # re-check with the lock held
                if not blockers:
                    waited = round(time.time() - t0, 1)
                    log_event("acquired", lane_dir, job=job, waited_s=waited, polls_waited=n_wait,
                              snapshot=snap)
                    return {"waited_s": waited, "polls_waited": n_wait,
                            "blockers_seen": [b for _, b in seen], "snapshot_at_entry": snap}
                lock.release()
                log_event("recheck-blocked", lane_dir, job=job, blockers=blockers)
            else:
                blockers = [{"kind": "lock", "holder": lock.holder()}]
        if policy.on_busy == "skip":
            log_event("skip", lane_dir, job=job, blockers=blockers, snapshot=snap)
            raise LaneBusy(blockers)
        sig = _signature(blockers)
        if sig not in [s for s, _ in seen]:
            seen.append((sig, blockers))
        if sig != last_sig or time.time() - last_log > 600:
            log_event("wait", lane_dir, job=job, waited_s=round(time.time() - t0, 1), blockers=blockers)
            last_sig, last_log = sig, time.time()
        if policy.max_wait_s is not None and time.time() - t0 > policy.max_wait_s:
            log_event("give-up", lane_dir, job=job, waited_s=round(time.time() - t0, 1), blockers=blockers)
            raise LaneBusy(blockers)
        n_wait += 1
        time.sleep(policy.poll_s)
