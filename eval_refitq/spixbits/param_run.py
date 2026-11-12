"""The parametrization stage as one checkpointed job (run through ops/spheavy): cross-fitted closure per scenario
(three processes side by side; one finished quantization scenario = one file, skipped on a restart), then the export of
the shipped correctionlib files. Each step writes a marker under checks/param_<version>/ when it has finished and is
skipped on a restart. (--steps slim: the slim-tier identity check of the heads reader.)

  python param_run.py [--version v2] [--steps closure export]
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
    ap.add_argument("--steps", nargs="+", default=["closure", "export"])
    ap.add_argument("--version", choices=("v1", "v2"), default="v2")
    a = ap.parse_args()
    mark = ST.LOCAL / "checks" / f"param_{a.version}"
    mark.mkdir(parents=True, exist_ok=True)
    logs = ST.LOCAL / "logs"
    rc = 0

    def run(name, cmd):
        if (mark / f"{name}.done").exists():
            print(f"skip {name} (done)", flush=True)
            return None
        print(time.strftime("[%H:%M:%S]"), "start", name, flush=True)
        return name, subprocess.Popen([sys.executable, *cmd], stdout=open(logs / f"param_{a.version}_{name}.log", "w"), stderr=subprocess.STDOUT)

    def wait(jobs):
        nonlocal rc
        for j in [j for j in jobs if j]:
            code = j[1].wait()
            print(time.strftime("[%H:%M:%S]"), "end", j[0], "exit", code, flush=True)
            if code == 0:
                (mark / f"{j[0]}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
            rc = max(rc, code)
    if "closure" in a.steps:
        wait([run(f"closure_{s}", [str(HERE / "param_closure.py"), "--version", a.version, "--scen", s]) for s in ST.SCENARIOS])
    if "export" in a.steps and rc == 0:
        wait([run("export", [str(HERE / "param_export.py"), "--version", a.version])])
    if "slim" in a.steps:
        wait([run("slim_identity", [str(HERE / "slim_identity.py")])])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
