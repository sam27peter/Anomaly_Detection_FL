import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay
)

from utils.logger import logger


BASE_DIR = (
    Path("results")
    / "single_machine"
)

HISTORY_DIR = BASE_DIR / "history"
METRICS_DIR = BASE_DIR / "metrics"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def plot_loss(dataset_type: str):

    with open(
            HISTORY_DIR /
            f"cnn_{dataset_type}_history.json"
    ) as f:

        history = json.load(f)

    plt.figure()

    plt.plot(
        history["loss"],
        marker="o"
    )

    plt.title(
        f"CNN {dataset_type} Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid()

    plt.savefig(
        PLOTS_DIR /
        f"{dataset_type}_loss.png"
    )

    plt.close()


def plot_accuracy(dataset_type: str):

    with open(
            HISTORY_DIR /
            f"cnn_{dataset_type}_history.json"
    ) as f:

        history = json.load(f)

    plt.figure()

    plt.plot(
        history["accuracy"],
        marker="o"
    )

    plt.title(
        f"CNN {dataset_type} Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.grid()

    plt.savefig(
        PLOTS_DIR /
        f"{dataset_type}_accuracy.png"
    )

    plt.close()


def plot_confusion_matrix(
        dataset_type: str
):

    with open(
            METRICS_DIR /
            f"cnn_{dataset_type}.json"
    ) as f:

        metrics = json.load(f)

    cm = np.array(
        metrics["confusion_matrix"]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(
        f"{dataset_type} Confusion Matrix"
    )

    plt.savefig(
        PLOTS_DIR /
        f"{dataset_type}_cm.png"
    )

    plt.close()


def plot_metric_bar_chart():

    files = list(
        METRICS_DIR.glob("*.json")
    )

    datasets = []

    accuracies = []

    f1_scores = []

    for file in files:

        with open(file) as f:

            metrics = json.load(f)

        datasets.append(
            metrics["dataset"]
        )

        accuracies.append(
            metrics["accuracy"]
        )

        f1_scores.append(
            metrics["f1_score"]
        )

    plt.figure()

    x = np.arange(
        len(datasets)
    )

    width = 0.35

    plt.bar(
        x - width/2,
        accuracies,
        width,
        label="Accuracy"
    )

    plt.bar(
        x + width/2,
        f1_scores,
        width,
        label="F1"
    )

    plt.xticks(
        x,
        datasets
    )

    plt.legend()

    plt.title(
        "Centralized Performance"
    )

    plt.savefig(
        PLOTS_DIR /
        "centralized_summary.png"
    )

    plt.close()


def generate_all_plots(
        dataset_type: str
):

    logger.info(
        f"Generating plots "
        f"{dataset_type}"
    )

    plot_loss(dataset_type)

    plot_accuracy(dataset_type)

    plot_confusion_matrix(
        dataset_type
    )

    plot_metric_bar_chart()

    logger.info(
        "Plots generated"
    )


if __name__ == "__main__":

    generate_all_plots(
        "SMAP"
    )

    generate_all_plots(
        "MSL"
    )

