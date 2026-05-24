# experiments/validate_top_configs.py
"""
Post-hoc stability validation: re-evaluate the top-K unique configs
found by a C5 (or any condition) run across multiple seeds.

Usage (from cifar10_automl/ directory):
  python experiments/validate_top_configs.py \\
      --csv results/steps-60-.../C5_grpo_G4_steps.csv

  python experiments/validate_top_configs.py \\
      --csv results/steps-60-.../C5_grpo_G4_steps.csv \\
      --top-k 5 --seeds 0 1 2 3 4 --max-train-steps 300
"""
import argparse
import csv
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.action_space import coerce_spec
from src.trainer import evaluate_spec


# ---------------------------------------------------------------------------
# Parse spec_summary string back into a raw dict for coerce_spec
# Summary format (from ArchSpec.to_summary):
#   stages=[C64x2(batch,silu,stride), ...] residual=True drop=0.1
#   opt=adamw(lr=0.003,wd=0.01) sched=cosine aug=medium
# ---------------------------------------------------------------------------
def parse_summary(summary: str) -> dict:
    stages = []
    for m in re.finditer(r'C(\d+)x(\d+)\((\w+),(\w+),(\w+)\)', summary):
        stages.append({
            "out_channels": int(m.group(1)),
            "num_blocks":   int(m.group(2)),
            "norm":         m.group(3),
            "activation":   m.group(4),
            "downsample":   m.group(5),
        })

    res_m  = re.search(r'residual=(True|False)', summary)
    drop_m = re.search(r'drop=([0-9.]+)', summary)
    opt_m  = re.search(r'opt=(\w+)\(lr=([0-9e.+\-]+),wd=([0-9e.+\-]+)\)', summary)
    sched_m = re.search(r'sched=(\w+)', summary)
    aug_m   = re.search(r'aug=(\w+)', summary)

    return {
        "stages":       stages,
        "use_residual": (res_m.group(1) == "True") if res_m else True,
        "dropout":      float(drop_m.group(1)) if drop_m else 0.0,
        "optimizer": {
            "type":         opt_m.group(1) if opt_m else "adamw",
            "lr":           float(opt_m.group(2)) if opt_m else 0.001,
            "weight_decay": float(opt_m.group(3)) if opt_m else 0.01,
        },
        "scheduler": sched_m.group(1) if sched_m else "cosine",
        "augment":   aug_m.group(1)   if aug_m   else "medium",
    }


def load_top_configs(csv_path: str, top_k: int):
    """Return top_k rows sorted by acc desc, deduplicated by spec_summary."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    seen, top = set(), []
    for row in sorted(rows, key=lambda r: float(r["acc"]), reverse=True):
        summary = row["spec_summary"]
        if summary not in seen:
            seen.add(summary)
            top.append(row)
        if len(top) >= top_k:
            break
    return top


def stdev(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Path to C5_grpo_G4_steps.csv (or any condition CSV)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top unique configs to validate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Seeds to average over")
    parser.add_argument("--max-train-steps", type=int, default=None,
                        help="Training steps per eval (default: inferred from folder name)")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--results-dir", default=None,
                        help="Output directory (default: same folder as --csv)")
    args = parser.parse_args()

    # Infer max_train_steps from directory name if not specified
    if args.max_train_steps is None:
        m = re.search(r'maxsteps-(\d+)', args.csv)
        args.max_train_steps = int(m.group(1)) if m else 300
        print(f"[info] max_train_steps inferred: {args.max_train_steps}")

    if args.results_dir is None:
        args.results_dir = os.path.dirname(os.path.abspath(args.csv))

    print(f"Loading top-{args.top_k} configs from: {args.csv}")
    top_configs = load_top_configs(args.csv, args.top_k)
    print(f"Found {len(top_configs)} unique configs.")
    print(f"Seeds: {args.seeds}  max_steps: {args.max_train_steps}\n")

    out_csv = os.path.join(args.results_dir,
                           f"top{args.top_k}_validated_{len(args.seeds)}seeds.csv")
    results = []

    with open(out_csv, "w", newline="") as f:
        fieldnames = (["rank", "orig_step", "orig_cand", "orig_acc",
                       "mean_acc", "std_acc", "min_acc", "max_acc", "spec_summary"]
                      + [f"seed_{s}" for s in args.seeds])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, row in enumerate(top_configs, 1):
            summary   = row["spec_summary"]
            orig_acc  = float(row["acc"])
            orig_step = row.get("step", "?")
            orig_cand = row.get("candidate", "?")

            raw  = parse_summary(summary)
            spec = coerce_spec(raw)
            if spec is None:
                print(f"  [rank {rank}] FAILED to parse spec — skipping\n    {summary}\n")
                continue

            print(f"  Rank #{rank}  orig={orig_acc*100:.2f}%  "
                  f"(step={orig_step}, cand={orig_cand})")
            print(f"  {summary}")

            seed_accs = []
            for seed in args.seeds:
                t0  = time.time()
                acc = evaluate_spec(spec, max_steps=args.max_train_steps,
                                    data_root=args.data_root, seed=seed)
                elapsed = time.time() - t0
                seed_accs.append(acc)
                print(f"    seed={seed}  acc={acc*100:.2f}%  ({elapsed:.0f}s)")

            mean_acc = sum(seed_accs) / len(seed_accs)
            std_acc  = stdev(seed_accs)
            min_acc  = min(seed_accs)
            max_acc  = max(seed_accs)
            print(f"  → mean={mean_acc*100:.2f}%  std=±{std_acc*100:.2f}%  "
                  f"[{min_acc*100:.2f}%, {max_acc*100:.2f}%]\n")

            results.append((rank, orig_acc, mean_acc, std_acc, min_acc, max_acc, seed_accs))
            writer.writerow({
                "rank":       rank,
                "orig_step":  orig_step,
                "orig_cand":  orig_cand,
                "orig_acc":   f"{orig_acc:.4f}",
                "mean_acc":   f"{mean_acc:.4f}",
                "std_acc":    f"{std_acc:.4f}",
                "min_acc":    f"{min_acc:.4f}",
                "max_acc":    f"{max_acc:.4f}",
                "spec_summary": summary,
                **{f"seed_{s}": f"{a:.4f}" for s, a in zip(args.seeds, seed_accs)},
            })
            f.flush()

    # Summary table
    print("=" * 70)
    print(f"{'#':<4} {'Orig':>7} {'Mean':>7} {'Std':>6} {'Min':>7} {'Max':>7}")
    print("=" * 70)
    for rank, orig, mean, std, mn, mx, _ in results:
        print(f"  #{rank}  {orig*100:>6.2f}%  {mean*100:>6.2f}%  "
              f"±{std*100:.2f}%  {mn*100:>6.2f}%  {mx*100:>6.2f}%")
    print("=" * 70)
    if results:
        best = max(results, key=lambda r: r[2])  # by mean_acc
        print(f"Best by mean: Rank #{best[0]}  mean={best[2]*100:.2f}%  max={best[5]*100:.2f}%")
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
