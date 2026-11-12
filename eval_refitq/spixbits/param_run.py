"""The parametrization stage as one checkpointed job (run through ops/spheavy): per detector scenario, side by side, the
cross-fitted closure (one finished quantization scenario = one file, skipped on a restart) and then the export of the
shipped files. Each step writes a marker under checks/param_<version>/ when it has finished and is skipped on a
restart; a step that fails (the external disk gone) is tried again after the disk is back, up to MAX_TRIES times.

  python param_run.py [--version v2] [--scen AAAA AAAI IAAA] [--steps closure export]
  python param_run.py --steps slim            the slim-tier identity check of the heads reader (not version dependent)
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_TRIES = 6
THREADS = "4"                                   # per process: three run side by side on 12 cores


def main():
    import ext_scan as XS
    import study as ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=["closure", "export"])
    ap.add_argument("--version", choices=("v1", "v2"), default="v2")
    ap.add_argument("--scen", nargs="+", default=list(ST.SCENARIOS))
    ap.add_argument("--one", default="", help="internal: the chain of one detector scenario")
    a = ap.parse_args()
    mark = ST.LOCAL / "checks" / f"param_{a.version}"
    mark.mkdir(parents=True, exist_ok=True)
    logs = ST.LOCAL / "logs"
    stamp = lambda: time.strftime("[%H:%M:%S]")
    if a.one:                                   # closure, then export, of one detector scenario
        XS.VERSION = a.version
        env = dict(os.environ, OMP_NUM_THREADS=THREADS)
        for step in [s for s in ("closure", "export") if s in a.steps]:
            done = mark / f"{step}_{a.one}.done"
            for attempt in range(1, MAX_TRIES + 1):
                if done.exists():
                    break
                while not XS.out_root().exists():
                    print(stamp(), "waiting for", XS.out_root(), flush=True)
                    time.sleep(60)
                print(stamp(), "start", step, a.one, "attempt", attempt, flush=True)
                code = subprocess.run([sys.executable, str(HERE / f"param_{step}.py"), "--version", a.version, "--scen", a.one],
                                      env=env).returncode
                print(stamp(), "end", step, a.one, "exit", code, flush=True)
                if code == 0:
                    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
                else:
                    time.sleep(60)
            if not done.exists():
                return 1
        return 0
    rc = 0
    if set(a.steps) & {"closure", "export"}:
        jobs = [(s, subprocess.Popen([sys.executable, str(HERE / "param_run.py"), "--one", s, "--version", a.version, "--steps", *a.steps],
                                     stdout=open(logs / f"param_{a.version}_{s}.log", "a"), stderr=subprocess.STDOUT)) for s in a.scen]
        for s, p in jobs:
            code = p.wait()
            print(stamp(), "detector scenario", s, "exit", code, flush=True)
            rc = max(rc, code)
    if "slim" in a.steps:
        rc = max(rc, subprocess.run([sys.executable, str(HERE / "slim_identity.py")]).returncode)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
