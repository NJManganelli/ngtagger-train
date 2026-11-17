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
STEPS_5533 = ("scan5533", "heads5533", "eval5533", "closure5533", "ship5533")
STEPS_ONE = ("scan_one", "heads_one", "eval_one", "closure_one", "ship_one")       # one further scenario, named by --scenario (scan p2z)


def main():
    import study as ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=STEPS + STEPS_5533 + STEPS_ONE)
    ap.add_argument("--scenario", default="", help="the quantization scenario of the *_one steps (an upward-scan point, ext_scan.UP_SCAN)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    env = dict(os.environ, SPIXBITS_Q_VERSION="p2x")
    mark = ST.LOCAL / "checks" / "prompt_enlarged"
    mark.mkdir(parents=True, exist_ok=True)
    root_q = Path.home() / "smartpixels/scratch_local/spixbits/heads_retrained_per_quantization_prompt_enlarged"
    deliver = Path.home() / "smartpixels/doc-archive/spix-bits-refit-study/parametrization/prompt_v2/heads/per_quantization_enlarged"
    cmds = {"scan": ["ext_scan.py", "supervise", "--version", "p2x", "--workers", "3"], "heads": ["retrain_heads_q.py", "run"],
            "eval": ["heads_q_eval.py", "run"],
            # 5/5/3/3, added to the first pass later: its scan on the further files; "heads5533" / "eval5533" then rerun the
            # (checkpointed) heads and evaluation, which pick the new quantization scenario up; then its classifier closure
            "scan5533": ["ext_scan.py", "supervise", "--version", "p2y", "--workers", "3"], "heads5533": ["retrain_heads_q.py", "run"],
            "eval5533": ["heads_q_eval.py", "run"],
            "closure5533": ["param_closure_q.py", "--parallel", "--heads", "enlarged", "--allocs", "alloc_5533"],
            "scan_one": ["ext_scan.py", "supervise", "--version", "p2z", "--workers", str(a.workers)],
            "heads_one": ["retrain_heads_q.py", "run", "--allocs", a.scenario], "eval_one": ["heads_q_eval.py", "run", "--allocs", a.scenario],
            "closure_one": ["param_closure_q.py", "--parallel", "--heads", "enlarged", "--allocs", a.scenario]}
    if any(s in STEPS_ONE for s in a.steps) and not a.scenario:
        ap.error("the *_one steps need --scenario")
    for step in a.steps:
        done = mark / (f"{step}_{a.scenario}.done" if step in STEPS_ONE else f"{step}.done")
        if done.exists():
            print(time.strftime("[%H:%M:%S]"), "skip", step, flush=True)
            continue
        print(time.strftime("[%H:%M:%S]"), "start", step, flush=True)
        if step.startswith("ship"):
            deliver.mkdir(parents=True, exist_ok=True)
            for p in sorted((root_q / "models").glob("*.npz")):
                shutil.copy(p, deliver / p.name)
            code = 0
        else:
            if step == "heads" and not (root_q / "event_split.npz").exists():
                print("event_split.npz is missing: run `SPIXBITS_Q_VERSION=p2x python retrain_heads_q.py split` first", flush=True)
                return 2
            code = subprocess.run([sys.executable, str(HERE / cmds[step][0]), *cmds[step][1:]],
                                  env=dict(env, SPIXBITS_Q_VERSION="p2") if step.startswith("closure") else
                                  dict(env, SPIXBITS_EXT_ONLY=a.scenario) if step == "scan_one" else env).returncode
        print(time.strftime("[%H:%M:%S]"), "end", step, "exit", code, flush=True)
        if code != 0:
            return code
        done.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
