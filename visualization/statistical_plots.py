from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_statistics():

    csv_path = (

        Path("results")
        / "statistics"
        / "statistics.csv"

    )

    if not csv_path.exists():

        print(
            "Run statistical analysis first."
        )

        return

    df = pd.read_csv(
        csv_path
    )

    plt.figure(
        figsize=(10, 6)
    )

    labels = (

        df["algorithm"]

        + "_"

        + df["dataset"]

    )

    plt.bar(

        labels,

        df["mean_accuracy"]

    )

    plt.xticks(
        rotation=45
    )

    plt.ylabel(
        "Mean Accuracy"
    )

    plt.title(
        "Mean Accuracy Across Experiments"
    )

    plt.tight_layout()

    save_path = (

        Path("results")
        / "statistics"
        / "mean_accuracy.png"

    )

    plt.savefig(
        save_path
    )

    plt.close()

    print(
        f"Saved {save_path}"
    )


if __name__ == "__main__":

    plot_statistics()