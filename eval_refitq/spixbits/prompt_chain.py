"""Everything that follows the prompt-collection per-track scan (ext_scan.py --version p2), as one checkpointed job
for the lane: the heads retrained per quantization scenario on prompt tracks, their evaluation, the parametrization on
the prompt collection (closure and export), and the joint closure with the per-quantization-scenario heads.

  python prompt_chain.py [--steps heads eval tables param closure_q ship]

Each step is itself checkpointed (per quantization scenario or per detector scenario) and writes a marker under
checks/prompt_chain/ when it has finished; a restart skips finished steps. Three processes side by side at most.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ("heads", "eval", "tables", "param", "closure_q", "ship")


def main():
    import study as ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=STEPS)
    a = ap.parse_args()
    env = dict(os.environ, SPIXBITS_Q_VERSION="p2")
    mark = ST.LOCAL / "checks" / "prompt_chain"
    mark.mkdir(parents=True, exist_ok=True)
    root_q = Path.home() / "smartpixels/scratch_local/spixbits/heads_retrained_per_quantization_prompt"
    deliver = Path.home() / "smartpixels/doc-archive/spix-bits-refit-study/parametrization/prompt_v2"
    cmds = {"heads": ["retrain_heads_q.py", "run"], "eval": ["heads_q_eval.py", "run"], "tables": ["heads_q_tables.py"],
            "param": ["param_run.py", "--version", "p2"], "closure_q": ["param_closure_q.py", "--parallel"]}
    for step in a.steps:
        if (mark / f"{step}.done").exists():
            print(time.strftime("[%H:%M:%S]"), "skip", step, flush=True)
            continue
        print(time.strftime("[%H:%M:%S]"), "start", step, flush=True)
        if step == "ship":                                   # the per-quantization-scenario heads beside the parametrization
            dst = deliver / "heads" / "per_quantization"
            dst.mkdir(parents=True, exist_ok=True)
            for p in sorted((root_q / "models").glob("*.npz")):
                shutil.copy(p, dst / p.name)
            code = 0
        else:
            if step == "heads" and not (root_q / "event_split.npz").exists():      # the same events as the extended scan's
                root_q.mkdir(parents=True, exist_ok=True)
                shutil.copy(root_q.parent / "heads_retrained_per_quantization" / "event_split.npz", root_q / "event_split.npz")
            code = subprocess.run([sys.executable, str(HERE / cmds[step][0]), *cmds[step][1:]], env=env).returncode
        print(time.strftime("[%H:%M:%S]"), "end", step, "exit", code, flush=True)
        if code != 0:
            return code
        (mark / f"{step}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
