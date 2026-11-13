"""The enlarged heads sample for the first-pass quantization scenarios (prompt collection), as one checkpointed job
for the lane: the per-track scan on the further files (ext_scan.py --version p2x), the heads on the enlarged sample
(retrain_heads_q.py with SPIXBITS_Q_VERSION=p2x), their evaluation, and the models shipped beside the prompt
parametrization as heads/per_quantization_enlarged/.

  python prompt_enlarged.py [--steps scan heads eval ship]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ("scan", "heads", "eval", "ship")


def main():
    import study as ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=STEPS)
    a = ap.parse_args()
    env = dict(os.environ, SPIXBITS_Q_VERSION="p2x")
    mark = ST.LOCAL / "checks" / "prompt_enlarged"
    mark.mkdir(parents=True, exist_ok=True)
    root_q = Path.home() / "smartpixels/scratch_local/spixbits/heads_retrained_per_quantization_prompt_enlarged"
    deliver = Path.home() / "smartpixels/doc-archive/spix-bits-refit-study/parametrization/prompt_v2/heads/per_quantization_enlarged"
    cmds = {"scan": ["ext_scan.py", "supervise", "--version", "p2x", "--workers", "3"], "heads": ["retrain_heads_q.py", "run"],
            "eval": ["heads_q_eval.py", "run"]}
    for step in a.steps:
        if (mark / f"{step}.done").exists():
            print(time.strftime("[%H:%M:%S]"), "skip", step, flush=True)
            continue
        print(time.strftime("[%H:%M:%S]"), "start", step, flush=True)
        if step == "ship":
            deliver.mkdir(parents=True, exist_ok=True)
            for p in sorted((root_q / "models").glob("*.npz")):
                shutil.copy(p, deliver / p.name)
            code = 0
        else:
            if step == "heads" and not (root_q / "event_split.npz").exists():
                print("event_split.npz is missing: run `SPIXBITS_Q_VERSION=p2x python retrain_heads_q.py split` first", flush=True)
                return 2
            code = subprocess.run([sys.executable, str(HERE / cmds[step][0]), *cmds[step][1:]], env=env).returncode
        print(time.strftime("[%H:%M:%S]"), "end", step, "exit", code, flush=True)
        if code != 0:
            return code
        (mark / f"{step}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
