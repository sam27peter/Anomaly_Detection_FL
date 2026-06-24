import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_comparison_plot():

    experiments = list(

        Path(
            "results/experiments"
        ).glob("*")

    )

    rows = []

    for exp in experiments:

        metrics_file = (
            exp
            / "metrics.json"
        )

        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:

            metrics = json.load(f)

        rows.append({

            "Experiment":
                exp.name,

            "Accuracy":
                metrics["accuracy"]

        })

    df = pd.DataFrame(rows)

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(

        df["Experiment"],

        df["Accuracy"]

    )

    plt.xticks(
        rotation=90
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Experiment Comparison"
    )

    plt.tight_layout()

    save_dir = (

        Path("results")
        / "comparison"

    )

    save_dir.mkdir(
        exist_ok=True
    )

    plt.savefig(

        save_dir
        / "accuracy_comparison.png"

    )

    plt.close()


if __name__ == "__main__":

    create_comparison_plot()