import json
import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay
)


# ==================================================
# LOSS CURVE
# ==================================================

def plot_loss(dataset_type):

    history_file = (
        f"results/single_machine/history/"
        f"cnn_{dataset_type}_history.json"
    )

    with open(history_file, "r") as f:
        history = json.load(f)

    plt.figure()

    plt.plot(history["loss"])

    plt.title(
        f"CNN-{dataset_type} Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.savefig(
        f"results/single_machine/plots/"
        f"cnn_{dataset_type}_loss.png"
    )

    plt.close()


# ==================================================
# ACCURACY CURVE
# ==================================================

def plot_accuracy(dataset_type):

    history_file = (
        f"results/single_machine/history/"
        f"cnn_{dataset_type}_history.json"
    )

    with open(history_file, "r") as f:
        history = json.load(f)

    plt.figure()

    plt.plot(history["accuracy"])

    plt.title(
        f"CNN-{dataset_type} Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.savefig(
        f"results/single_machine/plots/"
        f"cnn_{dataset_type}_accuracy.png"
    )

    plt.close()


# ==================================================
# CONFUSION MATRIX
# ==================================================

def plot_confusion_matrix(dataset_type):

    metrics_file = (
        f"results/single_machine/metrics/"
        f"cnn_{dataset_type}.json"
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

    plt.savefig(
        f"results/single_machine/plots/"
        f"cnn_{dataset_type}_confusion_matrix.png"
    )

    plt.close()


# ==================================================
# GENERATE ALL PLOTS
# ==================================================

def generate_all_plots(dataset_type):

    os.makedirs(
        "results/single_machine/plots",
        exist_ok=True
    )

    print(f"Generating Loss Plot for {dataset_type}")
    plot_loss(dataset_type)

    print(f"Generating Accuracy Plot for {dataset_type}")
    plot_accuracy(dataset_type)

    print(f"Generating Confusion Matrix for {dataset_type}")
    plot_confusion_matrix(dataset_type)

    print(
        f"Plots generated for Dataset {dataset_type}"
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    generate_all_plots("25")

    generate_all_plots("55")