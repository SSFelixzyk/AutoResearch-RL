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
from src.reward import compute_rewards
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
    history_top_k: int = 3,
    parallel_eval: bool = True,
    generate_fn: Optional[Callable] = None,
    grpo_update_fn: Optional[Callable] = None,
    invalid_penalty: float = 0.1,
    use_relative_reward: bool = False,
    novelty_coef: float = 0.0,
    reward_floor: Optional[float] = None,
    wandb_run=None,
    seed: int = 0,
) -> List[float]:
    """
    Returns list of length n_steps: best_acc_so_far at each step.
    One 'step' = one G-rollout (evaluate G candidates, commit best).
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{condition_name}_steps.csv")

    buf = HistoryBuffer(max_k=history_k, top_k=history_top_k)
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
                raw_candidates = generate_fn(prompt, n=G)
                if len(raw_candidates) != G:
                    raise ValueError(f"generate_fn returned {len(raw_candidates)} items, expected {G}")
                # Replace None (invalid JSON) with random fallback for evaluation
                candidates = [
                    c if c is not None else sample_random_spec(seed=seed + step * G + i)
                    for i, c in enumerate(raw_candidates)
                ]
            else:
                raw_candidates = [None] * G   # C1 random: no LLM output to penalise
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

            # --- Shaped rewards (computed before history update so best_so_far is pre-step) ---
            prev_best = buf.best_acc
            recent_specs = [e.spec for e in buf.recent]
            shaped_rewards, reward_stats = compute_rewards(
                raw_candidates, accs,
                best_so_far=prev_best,
                recent_specs=recent_specs,
                invalid_penalty=invalid_penalty,
                use_relative=use_relative_reward,
                novelty_coef=novelty_coef,
                reward_floor=reward_floor,
            )

            # --- Update history with all G results ---
            for i, (spec, acc) in enumerate(zip(candidates, accs)):
                is_new_best = acc > buf.best_acc
                buf.add(spec, acc)
                writer.writerow([step, i, f"{acc:.4f}", is_new_best,
                                 f"{buf.best_acc:.4f}", spec.to_summary(), f"{wall_time:.1f}"])
            f.flush()

            # --- Optional GRPO weight update ---
            grpo_loss = None
            if grpo_update_fn is not None:
                grpo_loss = grpo_update_fn(
                    prompt=prompt,
                    candidates=candidates,
                    accs=shaped_rewards,
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
                    f"{condition_name}/step": step,
                    f"{condition_name}/best_acc": buf.best_acc,
                    f"{condition_name}/step_best_acc": max(accs),
                    f"{condition_name}/step_mean_acc": sum(accs) / len(accs),
                    f"{condition_name}/wall_time_s": wall_time,
                    f"{condition_name}/improved": int(buf.best_acc > prev_best),
                    f"{condition_name}/reward_mean":   reward_stats["reward_mean"],
                    f"{condition_name}/reward_novelty_mean": reward_stats["novelty_mean"],
                    f"{condition_name}/reward_invalid_count": reward_stats["invalid_count"],
                }
                if grpo_loss is not None:
                    log[f"{condition_name}/grpo_loss"] = grpo_loss
                wandb_run.log(log)

    return best_acc_history
