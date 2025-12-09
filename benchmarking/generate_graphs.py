from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

# baseline models from the paper, no real totals or times reported
BASELINE_MODELS = {
    "ReProver (active retrieval)": {
        "total": None,
        "successes": None,
        "success_rate": 0.512,   # 51.2% pass@1
        "avg_time_success": None,
    },
    "ReProver (no retrieval)": {
        "total": None,
        "successes": None,
        "success_rate": 0.476,   # 47.6% pass@1
        "avg_time_success": None,
    },
    "GPT-4 (zero-shot)": {
        "total": None,
        "successes": None,
        "success_rate": 0.290,   # 29.0% pass@1
        "avg_time_success": None,
    },
}

def load_metrics_for_dataset(results_root: Path, category: str, split: str):
    """
    Claude-Haiku-4.5_random_val containing results.jsonl
    Only success rate is computed, search times are ignored
    """
    metrics = dict()

    for run_dir in results_root.iterdir():
        if not run_dir.is_dir():
            continue

        name = run_dir.name
        parts = name.split("_")
        if len(parts) < 3:
            continue

        run_split = parts[-1]
        run_category = parts[-2]
        model_name = "_".join(parts[:-2])

        if run_category != category or run_split != split:
            continue

        results_file = run_dir / "results.jsonl"
        if not results_file.exists():
            continue

        total = 0
        successes = 0

        with results_file.open("r", encoding="utf-8") as f:  # read line by line
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                total += 1
                if rec.get("success"):
                    successes += 1

        if total == 0:
            continue

        success_rate = successes / total

        metrics[model_name] = {
            "total": total,
            "successes": successes,
            "success_rate": success_rate,
        }

    return metrics

def plot_metrics(metrics, category: str, split: str):
    # combine benchmark runs + baselines for plotting
    all_names = list(metrics.keys()) + list(BASELINE_MODELS.keys())

    success_rates = [
        (metrics.get(name) or BASELINE_MODELS[name])["success_rate"] * 100
        for name in all_names
    ]

    fig_height = max(len(all_names), 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_positions = range(len(all_names))
    bars = ax.barh(y_positions, success_rates)

    for bar, name in zip(bars, all_names):
        if name in BASELINE_MODELS:
            bar.set_color("blueviolet")
        else:
            bar.set_color("tab:cyan")

    ax.set_title(f"Model Success Rates on {category}_{split}")
    ax.set_xlabel("Success Rate (%)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(all_names)

    for bar, val in zip(bars, success_rates):
        ax.text(
            bar.get_width() + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center",
            fontsize=12,
        )

    fig.tight_layout()
    plt.show()

def main():
    category = "random"  # "random" or "novel_premises"
    split = "val"        # "train", "val", or "test"

    results_root = Path(__file__).parent / "results"
    metrics = load_metrics_for_dataset(results_root, category, split)

    plot_metrics(metrics, category, split)


if __name__ == "__main__":
    main()