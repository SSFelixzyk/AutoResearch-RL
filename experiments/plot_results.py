# experiments/plot_results.py
"""
Usage: python experiments/plot_results.py [--results-dir ./results]
Reads all_histories.json and produces comparison_plot.png.
"""
import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LABELS = {
    "C1_random":         "C1: Random Search",
    "C2_llm_nohist_G1":  "C2: LLM, no history, G=1",
    "C3_llm_hist_G1":    "C3: LLM + history, G=1",
    "C4_llm_hist_G4":    "C4: LLM + history, G=4 (best-of-G)",
    "C5_grpo_G4":        "C5: GRPO, G=4",
}
COLORS = ["#999999", "#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
STYLES = ["--", "-.", "-", "-", "-"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    args = parser.parse_args()

    path = os.path.join(args.results_dir, "all_histories.json")
    with open(path) as f:
        histories = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 5))

    for (key, label), color, style in zip(LABELS.items(), COLORS, STYLES):
        if key not in histories:
            continue
        h = histories[key]
        ax.plot(range(1, len(h) + 1), [v * 100 for v in h],
                label=label, color=color, linestyle=style, linewidth=2)

    ax.set_xlabel("Research Step", fontsize=12)
    ax.set_ylabel("Best Val Accuracy So Far (%)", fontsize=12)
    ax.set_title("CIFAR-10 AutoML: History-Aware LLM vs GRPO", fontsize=13)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(args.results_dir, "comparison_plot.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
