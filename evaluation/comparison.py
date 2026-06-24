import json
from pathlib import Path

import pandas as pd


def collect_experiments():

    results = []

    metrics_files = list(

        Path("results")
        .rglob("metrics.json")

    )

    for file in metrics_files:

        try:

            with open(file) as f:

                metrics = json.load(f)

            metrics["experiment"] = (
                str(file.parent)
            )

            results.append(
                metrics
            )

        except Exception:

            continue

    return pd.DataFrame(
        results
    )


def save_summary():

    df = collect_experiments()

    save_path = (

        Path("results")
        / "experiment_summary.csv"

    )

    df.to_csv(
        save_path,
        index=False
    )

    print(
        f"Saved {save_path}"
    )


if __name__ == "__main__":

    save_summary()
    