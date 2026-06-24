import json
from pathlib import Path

import pandas as pd

from utils.logger import logger


class CentralizedVsFederatedComparison:

    def __init__(self):

        self.centralized_dir = (
            Path("results")
            / "single_machine"
            / "metrics"
        )

        self.experiments_dir = (
            Path("results")
            / "experiments"
        )

        self.output_dir = (
            Path("results")
            / "centralized_vs_fl"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ==================================================
    # LOAD CENTRALIZED RESULTS
    # ==================================================

    def load_centralized(self):

        rows = []

        files = list(
            self.centralized_dir.glob("*.json")
        )

        for file in files:

            with open(file) as f:

                metrics = json.load(f)

            dataset = metrics["dataset"]

            if dataset == "25":
                dataset = "SMAP"

            elif dataset == "55":
                dataset = "MSL"

            rows.append({

                "method":
                    "Centralized",

                "algorithm":
                    "CNN",

                "dataset":
                    dataset,

                "partition":
                    "N/A",

                "accuracy":
                    metrics["accuracy"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

                "f1_score":
                    metrics["f1_score"],

                "roc_auc":
                    metrics.get(
                        "roc_auc",
                        0
                    ),

                "training_time_sec":
                    metrics.get(
                        "training_time_sec",
                        0
                    )

            })

        return rows

    # ==================================================
    # LOAD FL RESULTS
    # ==================================================

    def load_federated(self):

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

                "method":
                    "Federated",

                "algorithm":
                    config["algorithm"],

                "dataset":
                    config["dataset"],

                "partition":
                    config["partition"],

                "accuracy":
                    metrics["accuracy"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

                "f1_score":
                    metrics["f1_score"],

                "roc_auc":
                    metrics.get(
                        "roc_auc",
                        0
                    ),

                "training_time_sec":
                    metrics.get(
                        "training_time_sec",
                        0
                    )

            })

        return rows

    # ==================================================
    # COMPUTE PERFORMANCE GAP
    # ==================================================

    def compute_gap(
            self,
            df
    ):

        gap_results = []

        centralized = df[
            df["method"] == "Centralized"
        ]

        federated = df[
            df["method"] == "Federated"
        ]

        for _, fl in federated.iterrows():

            dataset = fl["dataset"]

            cent = centralized[
                centralized["dataset"] == dataset
            ]

            if cent.empty:
                continue

            cent = cent.iloc[0]

            gap_results.append({

                "algorithm":
                    fl["algorithm"],

                "dataset":
                    dataset,

                "partition":
                    fl["partition"],

                "accuracy_drop":

                    cent["accuracy"]
                    -
                    fl["accuracy"],

                "f1_drop":

                    cent["f1_score"]
                    -
                    fl["f1_score"],

                "roc_auc_drop":

                    cent["roc_auc"]
                    -
                    fl["roc_auc"]

            })

        return pd.DataFrame(
            gap_results
        )

    # ==================================================
    # RUN
    # ==================================================

    def run(self):

        logger.info(
            "Generating centralized vs FL comparison"
        )

        rows = []

        rows.extend(
            self.load_centralized()
        )

        rows.extend(
            self.load_federated()
        )

        df = pd.DataFrame(rows)

        summary_path = (

            self.output_dir
            / "centralized_vs_fl.csv"

        )

        df.to_csv(
            summary_path,
            index=False
        )

        logger.info(
            "Saved comparison CSV"
        )

        gap_df = self.compute_gap(
            df
        )

        gap_path = (

            self.output_dir
            / "performance_gap.csv"

        )

        gap_df.to_csv(
            gap_path,
            index=False
        )

        logger.info(
            "Saved performance gap"
        )


if __name__ == "__main__":

    CentralizedVsFederatedComparison().run()