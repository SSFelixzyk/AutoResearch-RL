# experiments/plot_results.py
"""
Reads all_histories.json and produces comparison_plot.png.

Supports both single-run and multi-run (n_runs > 1) results:
- Single run  → plain curve per condition
- Multi-run   → mean curve + shaded ±1 std band per condition

Usage:
  python experiments/plot_results.py
  python experiments/plot_results.py --results-dir results/conds-05--steps-60-...
  python experiments/plot_results.py --results-dir results/conds-05--steps-60-... --no-band
"""
import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Canonical condition order, labels, colours, line styles
CONDITIONS = [
    ("C1_random",        "C1: Random Search",            "#999999", "--"),
    ("C2_llm_nohist_G1", "C2: LLM, no history, G=1",    "#4878CF", "-."),
    ("C3_llm_hist_G1",   "C3: LLM + history, G=1",      "#6ACC65", "-"),
    ("C4_llm_hist_G4",   "C4: LLM + history, G=4",      "#D65F5F", "-"),
    ("C5_grpo_G4",       "C5: GRPO, G=4",               "#B47CC7", "-"),
]


def _mean_std(curves):
    """Return (means, stds) lists across a list of equal-length curves."""
    n_steps = len(curves[0])
    means, stds = [], []
    for t in range(n_steps):
        vals = [c[t] for c in curves]
        m = sum(vals) / len(vals)
        std = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
        means.append(m)
        stds.append(std)
    return means, stds


def aggregate(histories: dict) -> dict:
    """
    Group run variants (C1_random_run0, C1_random_run1, …) under their base key.
    Returns {base_key: {"curves": [...], "mean": [...], "std": [...]}}.
    """
    groups: dict = {}
    for key, curve in histories.items():
        # Strip _run{i} suffix if present
        base = key
        for sep in ("_run",):
            idx = key.rfind(sep)
            if idx != -1 and key[idx + len(sep):].isdigit():
                base = key[:idx]
                break
        groups.setdefault(base, []).append(curve)

    result = {}
    for base, curves in groups.items():
        # Trim all curves to the shortest (in case runs differ in length)
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        means, stds = _mean_std(curves)
        result[base] = {"curves": curves, "mean": means, "std": stds, "n": len(curves)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--no-band", action="store_true",
                        help="Draw individual run lines instead of mean ± std band")
    parser.add_argument("--alpha", type=float, default=0.18,
                        help="Opacity of the std band (default: 0.18)")
    args = parser.parse_args()

    # Auto-find latest subdirectory if results-dir has no all_histories.json
    hist_path = os.path.join(args.results_dir, "all_histories.json")
    if not os.path.isfile(hist_path):
        subdirs = sorted(
            [os.path.join(args.results_dir, d)
             for d in os.listdir(args.results_dir)
             if os.path.isdir(os.path.join(args.results_dir, d))],
            key=os.path.getmtime, reverse=True,
        )
        for sd in subdirs:
            candidate = os.path.join(sd, "all_histories.json")
            if os.path.isfile(candidate):
                hist_path = candidate
                args.results_dir = sd
                print(f"[plot] Auto-selected: {sd}")
                break
        else:
            raise FileNotFoundError(
                f"No all_histories.json found under {args.results_dir}"
            )

    with open(hist_path) as f:
        histories = json.load(f)

    agg = aggregate(histories)
    n_runs_max = max(v["n"] for v in agg.values()) if agg else 1

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for base_key, label, color, style in CONDITIONS:
        if base_key not in agg:
            continue
        data = agg[base_key]
        means = [v * 100 for v in data["mean"]]
        stds  = [v * 100 for v in data["std"]]
        steps = list(range(1, len(means) + 1))
        n     = data["n"]

        if args.no_band or n == 1:
            # Single run or explicit --no-band: draw each curve individually
            for i, curve in enumerate(data["curves"]):
                lbl = f"{label} (run {i})" if n > 1 else label
                ax.plot(steps, [v * 100 for v in curve],
                        label=lbl, color=color, linestyle=style,
                        linewidth=1.5 if n > 1 else 2, alpha=0.7 if n > 1 else 1.0)
        else:
            # Multi-run: mean curve + ±1 std shaded band
            run_label = f"{label}  (n={n}, mean±std)"
            ax.plot(steps, means, label=run_label,
                    color=color, linestyle=style, linewidth=2)
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            ax.fill_between(steps, lo, hi, color=color, alpha=args.alpha)

    title_suffix = f"  [{n_runs_max} runs]" if n_runs_max > 1 else ""
    ax.set_xlabel("Research Step", fontsize=12)
    ax.set_ylabel("Best Val Accuracy So Far (%)", fontsize=12)
    ax.set_title(f"CIFAR-10 AutoML: History-Aware LLM vs GRPO{title_suffix}", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(args.results_dir, "comparison_plot.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

    # Print summary table
    print(f"\n{'Condition':<35} {'Runs':>4}  {'Final mean':>10}  {'±std':>6}")
    print("-" * 60)
    for base_key, label, _, _ in CONDITIONS:
        if base_key not in agg:
            continue
        data = agg[base_key]
        final_mean = data["mean"][-1] * 100
        final_std  = data["std"][-1]  * 100
        print(f"  {label:<33} {data['n']:>4}  {final_mean:>9.2f}%  ±{final_std:.2f}%")


if __name__ == "__main__":
    main()
