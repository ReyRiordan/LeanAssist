import json
from pathlib import Path
import matplotlib.pyplot as plt

# model performances as reported from paper, no direct values of total/successes
BASELINE_MODELS = {
    "ReProver (active retrieval)_random_val": {"success_rate": .512},
    "ReProver (no active retrieval)_random_val": {"success_rate": .476},
    "GPT-4 (zero-shot)_random_val": {"success_rate": .29}
}

def load_metrics(path):
    metrics = dict()

    for run_dir in path.iterdir():
        if not run_dir.is_dir():
            continue
        
        summary_file = run_dir / "summary.json"
        if not summary_file.exists():
            continue

        try:
            with summary_file.open("r") as file:
                summary = json.load(file)
        except(json.JSONDecodeError, OSError):
            continue
        
        accuracy = summary.get("accuracy")
        metrics[run_dir.name] = {"success_rate": float(accuracy)}

    return metrics

def plot_graph(metrics):
    # combine all benchmarked models + baselines for graphing
    models = list(metrics.keys()) + list(BASELINE_MODELS.keys())
    n = len(models)

    rates = []
    for name in models:
        if name in metrics:
            rates.append(metrics[name]["success_rate"] * 100)
        else:
            rates.append(BASELINE_MODELS[name]["success_rate"] * 100)
    
    _, ax = plt.subplots(figsize=(12, max(n, 1)))
    y_pos = range(n)
    bars = ax.barh(y_pos, rates)

    for bar, name, in zip(bars, models):
        if name in BASELINE_MODELS:
            bar.set_color("blueviolet") # baseline models
        else:
            bar.set_color("tab:cyan") # imported models tested
    
    ax.set_title("Model Success Rates", pad=25)
    ax.set_xlabel("Success Rates")

    ax.text(
        0.16, 1.01,
        "■ baseline performances from the paper",
        color="blueviolet",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
    )
    ax.text(
        0.52, 1.01,
        "■ performances we computed",
        color="tab:cyan",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlim(0, 80)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_width() + .25,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                fontsize=12
                )

    plt.tight_layout()
    plt.show()

def main():
    path = Path(__file__).parent / "results"
    metrics = load_metrics(path)
    plot_graph(metrics)

if __name__ == "__main__":
    main()