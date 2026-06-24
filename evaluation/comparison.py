import json
from pathlib import Path

import pandas as pd

from utils.logger import logger


class ComparisonEngine:
    """
    Compare all completed experiments.
    """

    def __init__(self):

        self.experiments_dir = (
            Path("results")
            / "experiments"
        )

        self.output_dir = (
            Path("results")
            / "comparison"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ==================================================
    # LOAD EXPERIMENTS
    # ==================================================

    def load_experiments(self):

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

            try:

                with open(metrics_file) as f:
                    metrics = json.load(f)

                with open(config_file) as f:
                    config = json.load(f)

                row = {

                    "experiment":
                        exp.name,

                    "algorithm":
                        config.get(
                            "algorithm",
                            "unknown"
                        ),

                    "dataset":
                        config.get(
                            "dataset",
                            "unknown"
                        ),

                    "partition":
                        config.get(
                            "partition",
                            "unknown"
                        ),

                    "accuracy":
                        metrics.get(
                            "accuracy",
                            0
                        ),

                    "precision":
                        metrics.get(
                            "precision",
                            0
                        ),

                    "recall":
                        metrics.get(
                            "recall",
                            0
                        ),

                    "f1_score":
                        metrics.get(
                            "f1_score",
                            0
                        ),

                    "roc_auc":
                        metrics.get(
                            "roc_auc",
                            0
                        ),

                    "pr_auc":
                        metrics.get(
                            "pr_auc",
                            0
                        ),

                    "training_time_sec":
                        metrics.get(
                            "training_time_sec",
                            0
                        ),

                    "mean_accuracy":
                        metrics.get(
                            "mean_accuracy",
                            0
                        ),

                    "std_accuracy":
                        metrics.get(
                            "std_accuracy",
                            0
                        ),

                    "accuracy_gap":
                        metrics.get(
                            "accuracy_gap",
                            0
                        )

                }

                rows.append(row)

            except Exception as e:

                logger.error(
                    f"{exp.name}: {e}"
                )

        return pd.DataFrame(rows)

    # ==================================================
    # SAVE SUMMARY
    # ==================================================

    def save_summary(
            self,
            df: pd.DataFrame
    ):

        save_path = (

            self.output_dir
            / "experiment_summary.csv"

        )

        df.to_csv(
            save_path,
            index=False
        )

        logger.info(
            f"Saved {save_path}"
        )

    # ==================================================
    # SAVE BEST EXPERIMENT
    # ==================================================

    def save_best_experiment(
            self,
            df: pd.DataFrame
    ):

        if df.empty:
            return

        best = df.sort_values(

            by="f1_score",

            ascending=False

        ).iloc[0]

        best_path = (

            self.output_dir
            / "best_experiment.json"

        )

        with open(
                best_path,
                "w"
        ) as f:

            json.dump(

                best.to_dict(),

                f,

                indent=4

            )

        logger.info(
            "Best experiment saved"
        )

    # ==================================================
    # ALGORITHM RANKING
    # ==================================================

    def save_algorithm_ranking(
            self,
            df: pd.DataFrame
    ):

        if df.empty:
            return

        ranking = (

            df.groupby("algorithm")

            [

                [
                    "accuracy",
                    "f1_score",
                    "roc_auc"
                ]

            ]

            .mean()

            .sort_values(

                by="f1_score",

                ascending=False

            )

        )

        save_path = (

            self.output_dir
            / "algorithm_ranking.csv"

        )

        ranking.to_csv(
            save_path
        )

        logger.info(
            "Algorithm ranking saved"
        )

    # ==================================================
    # RUN
    # ==================================================

    def run(self):

        logger.info(
            "Generating comparison files"
        )

        df = self.load_experiments()

        if df.empty:

            logger.warning(
                "No experiments found."
            )

            return

        self.save_summary(df)

        self.save_best_experiment(df)

        self.save_algorithm_ranking(df)

        logger.info(
            "Comparison completed"
        )


if __name__ == "__main__":

    ComparisonEngine().run()