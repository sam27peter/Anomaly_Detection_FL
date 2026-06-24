import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import logger


class StatisticalAnalyzer:

    def __init__(self):

        self.experiments_dir = (
            Path("results")
            / "experiments"
        )

        self.output_dir = (
            Path("results")
            / "statistics"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ==================================================
    # LOAD EXPERIMENTS
    # ==================================================

    def load_results(self):

        rows = []

        experiments = list(
            self.experiments_dir.glob("*")
        )

        for exp in experiments:

            metrics_file = (
                exp / "metrics.json"
            )

            config_file = (
                exp / "experiment_config.json"
            )

            if (
                    not metrics_file.exists()
                    or
                    not config_file.exists()
            ):
                continue

            with open(metrics_file) as f:
                metrics = json.load(f)

            with open(config_file) as f:
                config = json.load(f)

            rows.append({

                "algorithm":
                    config["algorithm"],

                "dataset":
                    config["dataset"],

                "partition":
                    config["partition"],

                "accuracy":
                    metrics["accuracy"],

                "f1_score":
                    metrics["f1_score"],

                "roc_auc":
                    metrics.get(
                        "roc_auc",
                        0
                    )

            })

        return pd.DataFrame(rows)

    # ==================================================
    # COMPUTE STATISTICS
    # ==================================================

    def compute_statistics(
            self,
            df
    ):

        statistics = []

        grouped = df.groupby(

            [

                "algorithm",
                "dataset",
                "partition"

            ]

        )

        for name, group in grouped:

            accuracy = (
                group["accuracy"]
                .values
            )

            n = len(
                accuracy
            )

            mean_acc = np.mean(
                accuracy
            )

            std_acc = np.std(
                accuracy
            )

            ci95 = (

                1.96

                *

                (

                    std_acc
                    /
                    np.sqrt(n)

                )

            )

            statistics.append({

                "algorithm":
                    name[0],

                "dataset":
                    name[1],

                "partition":
                    name[2],

                "num_experiments":
                    n,

                "mean_accuracy":
                    mean_acc,

                "std_accuracy":
                    std_acc,

                "min_accuracy":
                    np.min(
                        accuracy
                    ),

                "max_accuracy":
                    np.max(
                        accuracy
                    ),

                "confidence_interval_95":
                    ci95

            })

        return pd.DataFrame(
            statistics
        )

    # ==================================================
    # RUN
    # ==================================================

    def run(self):

        logger.info(
            "Running statistical analysis"
        )

        df = self.load_results()

        if df.empty:

            logger.warning(
                "No experiments found"
            )

            return

        stats = (
            self.compute_statistics(
                df
            )
        )

        save_path = (

            self.output_dir
            / "statistics.csv"

        )

        stats.to_csv(
            save_path,
            index=False
        )

        logger.info(
            f"Saved {save_path}"
        )


if __name__ == "__main__":

    StatisticalAnalyzer().run()