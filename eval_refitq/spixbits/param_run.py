"""The parametrization stage as one checkpointed job (run through ops/spheavy): cross-fitted closure per scenario
(three processes side by side), then the export of the shipped correctionlib files, then the slim-tier identity check.
Each step writes a marker under checks/param/ when it has finished and is skipped on a restart.

  python param_run.py [--steps closure export slim]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    import study as ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=["closure", "export", "slim"])
    a = ap.parse_args()
    mark = ST.LOCAL / "checks" / "param"
    mark.mkdir(parents=True, exist_ok=True)
    logs = ST.LOCAL / "logs"
    rc = 0

    def run(name, cmd):
        if (mark / f"{name}.done").exists():
            print(f"skip {name} (done)", flush=True)
            return None
        print(time.strftime("[%H:%M:%S]"), "start", name, flush=True)
        return name, subprocess.Popen([sys.executable, *cmd], stdout=open(logs / f"param_{name}.log", "w"), stderr=subprocess.STDOUT)

    def wait(jobs):
        nonlocal rc
        for j in [j for j in jobs if j]:
            code = j[1].wait()
            print(time.strftime("[%H:%M:%S]"), "end", j[0], "exit", code, flush=True)
            if code == 0:
                (mark / f"{j[0]}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
            rc = max(rc, code)
    if "closure" in a.steps:
        wait([run(f"closure_{s}", [str(HERE / "param_closure.py"), "--scen", s]) for s in ST.SCENARIOS])
    if "export" in a.steps and rc == 0:
        wait([run("export", [str(HERE / "param_export.py")])])
    if "slim" in a.steps:
        wait([run("slim_identity", [str(HERE / "slim_identity.py")])])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
