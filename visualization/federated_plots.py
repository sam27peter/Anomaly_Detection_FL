
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_federated_plots(
        experiment_path: str
):

    experiment_path = Path(
        experiment_path
    )

    history_file = (
        experiment_path
        / "history.json"
    )

    metrics_file = (
        experiment_path
        / "metrics.json"
    )

    with open(history_file) as f:
        history = json.load(f)

    with open(metrics_file) as f:
        metrics = json.load(f)

    plot_dir = (
        experiment_path
        / "plots"
    )

    plot_dir.mkdir(
        exist_ok=True
    )

    # Accuracy

    plt.figure()

    plt.plot(
        history["accuracy"],
        marker="o"
    )

    plt.title(
        "Global Accuracy"
    )

    plt.xlabel("Round")
    plt.ylabel("Accuracy")

    plt.grid()

    plt.savefig(
        plot_dir /
        "global_accuracy.png"
    )

    plt.close()

    # Loss

    plt.figure()

    plt.plot(
        history["loss"],
        marker="o"
    )

    plt.title(
        "Global Loss"
    )

    plt.xlabel("Round")
    plt.ylabel("Loss")

    plt.grid()

    plt.savefig(
        plot_dir /
        "global_loss.png"
    )

    plt.close()

    # Client Accuracy

    client_acc = (
        history[
            "client_accuracy"
        ]
    )

    plt.figure()

    plt.bar(

        [
            f"C{i}"

            for i in range(
                1,
                len(client_acc)+1
            )
        ],

        client_acc

    )

    plt.title(
        "Client Accuracy"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.savefig(
        plot_dir /
        "client_accuracy.png"
    )

    plt.close()

    # Fairness

    fairness = {

        "Mean":
            metrics[
                "mean_accuracy"
            ],

        "Best":
            metrics[
                "best_client_accuracy"
            ],

        "Worst":
            metrics[
                "worst_client_accuracy"
            ]

    }

    plt.figure()

    plt.bar(

        fairness.keys(),

        fairness.values()

    )

    plt.title(
        "Fairness Metrics"
    )

    plt.savefig(
        plot_dir /
        "fairness.png"
    )

    plt.close()


import sys

if __name__ == "__main__":

    if len(sys.argv) > 1:

        experiment = (

            Path("results")
            / "experiments"
            / sys.argv[1]

        )

        generate_federated_plots(
            experiment
        )

    else:

        experiments = sorted(

            Path(
                "results/experiments"
            ).glob("*"),

            key=lambda x: x.stat().st_mtime,

            reverse=True

        )

        latest = experiments[0]

        print(
            f"Generating plots for "
            f"latest experiment: "
            f"{latest.name}"
        )

        generate_federated_plots(
            latest
        )
