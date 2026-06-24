import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay
)

from utils.logger import logger


# ==================================================
# PATHS
# ==================================================

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


# ==================================================
# LOSS CURVE
# ==================================================

def plot_loss(
        dataset_type: str
) -> None:
    """
    Plot training loss curve.
    """

    history_file = (
        HISTORY_DIR
        / f"cnn_{dataset_type}_history.json"
    )

    with open(history_file, "r") as f:
        history = json.load(f)

    plt.figure()

    plt.plot(
        history["loss"],
        marker="o"
    )

    plt.title(
        f"CNN-{dataset_type} Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)

    plt.savefig(
        PLOTS_DIR
        / f"cnn_{dataset_type}_loss.png"
    )

    plt.close()

    logger.info(
        f"Loss plot saved for {dataset_type}"
    )


# ==================================================
# ACCURACY CURVE
# ==================================================

def plot_accuracy(
        dataset_type: str
) -> None:
    """
    Plot training accuracy curve.
    """

    history_file = (
        HISTORY_DIR
        / f"cnn_{dataset_type}_history.json"
    )

    with open(history_file, "r") as f:
        history = json.load(f)

    plt.figure()

    plt.plot(
        history["accuracy"],
        marker="o"
    )

    plt.title(
        f"CNN-{dataset_type} Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.savefig(
        PLOTS_DIR
        / f"cnn_{dataset_type}_accuracy.png"
    )

    plt.close()

    logger.info(
        f"Accuracy plot saved for {dataset_type}"
    )


# ==================================================
# CONFUSION MATRIX
# ==================================================

def plot_confusion_matrix(
        dataset_type: str
) -> None:
    """
    Plot confusion matrix.
    """

    metrics_file = (
        METRICS_DIR
        / f"cnn_{dataset_type}.json"
    )

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    cm = np.array(
        metrics["confusion_matrix"]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(
        f"CNN-{dataset_type} Confusion Matrix"
    )

    plt.savefig(
        PLOTS_DIR
        / f"cnn_{dataset_type}_confusion_matrix.png"
    )

    plt.close()

    logger.info(
        f"Confusion matrix saved for {dataset_type}"
    )


# ==================================================
# GENERATE ALL PLOTS
# ==================================================

def generate_all_plots(
        dataset_type: str
) -> None:
    """
    Generate all centralized plots.
    """

    logger.info(
        f"Generating plots for {dataset_type}"
    )

    plot_loss(dataset_type)

    plot_accuracy(dataset_type)

    plot_confusion_matrix(dataset_type)

    logger.info(
        f"Completed plotting for {dataset_type}"
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    generate_all_plots("SMAP")

    generate_all_plots("MSL")