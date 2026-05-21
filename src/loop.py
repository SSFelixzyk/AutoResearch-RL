# src/loop.py
import csv
import os
import time
from pathlib import Path
from typing import Optional, List, Callable
import torch.multiprocessing as mp

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from src.action_space import ArchSpec, sample_random_spec
from src.history import HistoryBuffer
from src.trainer import evaluate_spec


def _eval_worker(args):
    """Worker function for parallel evaluation — runs in a subprocess."""
    spec, max_steps, data_root, seed = args
    return evaluate_spec(spec, max_steps=max_steps, data_root=data_root, seed=seed)


def evaluate_parallel(
    candidates: List[ArchSpec],
    max_steps: int,
    data_root: str,
    seed: int,
    n_workers: int,
) -> List[float]:
    """
    Evaluate all candidates simultaneously using multiprocessing.
    Uses 'spawn' context (required for CUDA in subprocesses).
    """
    ctx = mp.get_context("spawn")
    args = [(spec, max_steps, data_root, seed) for spec in candidates]
    with ctx.Pool(processes=n_workers) as pool:
        accs = pool.map(_eval_worker, args)
    return accs


def run_research_loop(
    condition_name: str,
    n_steps: int,
    G: int,
    use_history: bool,
    program_md: str,
    results_dir: str,
    data_root: str = "./data",
    max_train_steps: int = 500,
    history_k: int = 10,
    parallel_eval: bool = True,
    generate_fn: Optional[Callable] = None,
    grpo_update_fn: Optional[Callable] = None,
    wandb_run=None,
    seed: int = 0,
) -> List[float]:
    """
    Returns list of length n_steps: best_acc_so_far at each step.
    One 'step' = one G-rollout (evaluate G candidates, commit best).
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{condition_name}_steps.csv")

    buf = HistoryBuffer(max_k=history_k)
    best_acc_history: List[float] = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "candidate", "acc", "is_best",
                         "best_so_far", "spec_summary", "wall_time"])

        for step in range(n_steps):
            t0 = time.time()
            prompt = buf.build_prompt(program_md, use_history=use_history)

            # --- Generate G candidates ---
            if generate_fn is not None:
                candidates = generate_fn(prompt, n=G)
                # Replace None (invalid JSON parse) with random fallback
                candidates = [
                    c if c is not None else sample_random_spec(seed=seed + step * G + i)
                    for i, c in enumerate(candidates)
                ]
            else:
                candidates = [
                    sample_random_spec(seed=seed + step * G + i) for i in range(G)
                ]

            # --- Evaluate all G candidates (parallel or sequential) ---
            if parallel_eval and G > 1:
                accs = evaluate_parallel(candidates, max_train_steps, data_root, seed, n_workers=G)
            else:
                accs = [
                    evaluate_spec(spec, max_steps=max_train_steps,
                                  data_root=data_root, seed=seed)
                    for spec in candidates
                ]

            wall_time = time.time() - t0

            # --- Update history with all G results ---
            prev_best = buf.best_acc
            for i, (spec, acc) in enumerate(zip(candidates, accs)):
                is_new_best = acc > buf.best_acc
                buf.add(spec, acc)
                writer.writerow([step, i, f"{acc:.4f}", is_new_best,
                                 f"{buf.best_acc:.4f}", spec.to_summary(), f"{wall_time:.1f}"])

            # --- Optional GRPO weight update ---
            grpo_loss = None
            if grpo_update_fn is not None:
                grpo_loss = grpo_update_fn(
                    prompt=prompt,
                    candidates=candidates,
                    accs=accs,
                )

            best_acc_history.append(buf.best_acc)

            # --- Console ---
            print(f"[{condition_name}] step={step:3d}  "
                  f"best_of_G={max(accs)*100:.2f}%  "
                  f"best_so_far={buf.best_acc*100:.2f}%  "
                  f"wall={wall_time:.0f}s")

            # --- wandb ---
            if wandb_run is not None:
                log = {
                    f"{condition_name}/best_acc": buf.best_acc,
                    f"{condition_name}/step_best_acc": max(accs),
                    f"{condition_name}/step_mean_acc": sum(accs) / len(accs),
                    f"{condition_name}/wall_time_s": wall_time,
                    f"{condition_name}/improved": int(buf.best_acc > prev_best),
                }
                if grpo_loss is not None:
                    log[f"{condition_name}/grpo_loss"] = grpo_loss
                wandb_run.log(log, step=step)

    return best_acc_history
