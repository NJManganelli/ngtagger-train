"""Multi-shot training: run the same configuration with N seeds and keep the
best model, mitigating the local-minima vulnerability of quantized (QAT)
training. Shots run in parallel subprocesses sized to the machine:

  - M4 Pro / 24 GB   : default 2-3 concurrent shots (CPU/Metal)
  - A100 20GB (MIG)  : default 1 concurrent shot (GPU memory bound)

Each shot is a separate process (fresh TF/Keras state) and logs to mlflow as
a nested run under one parent run.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil


def _detect_parallel_default() -> int:
    import platform

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        return 1  # GPU-bound: serialize shots on the partition
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return 2
    return max(1, (os.cpu_count() or 4) // 8)


def _run_shot(args):
    config_path, files, output_dir, seed, max_events = args
    # isolate each shot in its own process; late imports keep the fork clean
    from ngtagger.train.trainer import run_training

    model = run_training(
        config_path, files, os.path.join(output_dir, f"shot_seed{seed}"),
        seed=seed, max_events=max_events, mlflow_run_name=f"shot-seed{seed}",
    )
    return seed, model.best_val_loss()


def run_multishot(config_path: str, files: list[str], output_dir: str,
                  n_shots: int = 5, parallel: int | None = None,
                  max_events: int | None = None, base_seed: int = 1000):
    parallel = parallel or _detect_parallel_default()
    seeds = [base_seed + i for i in range(n_shots)]
    jobs = [(config_path, files, output_dir, s, max_events) for s in seeds]

    try:
        import mlflow

        mlflow.set_experiment("ngtagger")
        parent = mlflow.start_run(run_name=f"multishot-{os.path.basename(config_path)}")
    except Exception:
        mlflow, parent = None, None

    ctx = mp.get_context("spawn")  # never fork a process holding a TF runtime
    if parallel > 1:
        with ctx.Pool(parallel) as pool:
            results = pool.map(_run_shot, jobs)
    else:
        results = [_run_shot(j) for j in jobs]

    results.sort(key=lambda r: r[1])
    best_seed, best_loss = results[0]
    best_dir = os.path.join(output_dir, f"shot_seed{best_seed}")
    final_dir = os.path.join(output_dir, "best")
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    shutil.copytree(best_dir, final_dir)

    summary = {
        "shots": [{"seed": s, "val_loss": l} for s, l in results],
        "best_seed": best_seed,
        "best_val_loss": best_loss,
    }
    with open(os.path.join(output_dir, "multishot_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if mlflow and parent:
        mlflow.log_metric("best_val_loss", best_loss)
        mlflow.log_param("best_seed", best_seed)
        mlflow.log_dict(summary, "multishot_summary.json")
        mlflow.end_run()

    print(f"multishot: best seed {best_seed} (val_loss={best_loss:.5f}) -> {final_dir}")
    return summary
