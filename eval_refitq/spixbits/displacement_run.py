"""The displacement heads (second version) on both track collections, side by side, as one job for the lane (displacement_heads.py:
train, then report). Checkpointed per model; a restart skips finished models.

  python displacement_run.py [--steps train report]
  python displacement_run.py --script pair_finder.py --steps heads pairs analyze     (the pair finder, the same way)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = {"prompt": {"TAGGER_SP_COLLECTION": "prompt", "TAGGER_SP_OUT": "/Volumes/WDMac/smartpixels_scratch/tagger_v5_prompt/heads_input"},
       "extended": {}}              # extended: the tagger study's own caches (tagger_v5), its default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=["train", "report"])
    ap.add_argument("--script", default="displacement_heads.py")
    a = ap.parse_args()
    logs = Path.home() / "smartpixels/scratch_local/spixbits/logs"
    rc = 0
    for step in a.steps:
        jobs = [subprocess.Popen([sys.executable, str(HERE / a.script), step], env=dict(os.environ, **e),
                                 stdout=open(logs / f"{a.script[:-3]}_{step}_{c}.log", "a"), stderr=subprocess.STDOUT) for c, e in ENV.items()]
        rc = max([rc] + [j.wait() for j in jobs])
        if rc:
            break
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
