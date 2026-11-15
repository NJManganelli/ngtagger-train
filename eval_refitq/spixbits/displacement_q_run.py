"""Displacement heads per quantization scenario, as one checkpointed job for the lane: the per-track scans with the
per-surface record (ext_scan.py --version p3, the prompt collection; x3, the extended one), then the heads and their
report (displacement_heads_q.py), the two collections side by side.

  python displacement_q_run.py [--steps scan_prompt scan_extended train report] [--workers 3]
A finished scan unit and a finished model are skipped on a restart; the scans wait while the external disk is missing.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = Path.home() / "smartpixels/scratch_local/spixbits/logs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=["scan_prompt", "scan_extended", "train", "report"])
    ap.add_argument("--collections", nargs="+", default=["prompt", "extended"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    for step in a.steps:
        print(time.strftime("[%H:%M:%S]"), "start", step, flush=True)
        if step.startswith("scan_"):
            ver = {"scan_prompt": "p3", "scan_extended": "x3"}[step]
            rc = subprocess.run([sys.executable, str(HERE / "ext_scan.py"), "supervise", "--version", ver, "--workers", str(a.workers)]).returncode
        else:
            jobs = [subprocess.Popen([sys.executable, str(HERE / "displacement_heads_q.py"), step, "--collections", c],
                                     stdout=open(LOGS / f"displacement_q_{step}_{c}.log", "a"), stderr=subprocess.STDOUT) for c in a.collections]
            rc = max(j.wait() for j in jobs)
        print(time.strftime("[%H:%M:%S]"), "end", step, "exit", rc, flush=True)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
