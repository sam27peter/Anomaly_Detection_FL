from datetime import datetime
from pathlib import Path
import json


class ExperimentTracker:
    """
    Handles experiment folder creation and
    experiment metadata storage.
    """

    def __init__(
            self,
            algorithm: str,
            dataset: str,
            partition: str
    ):

        self.algorithm = algorithm.lower()
        self.dataset = dataset
        self.partition = partition

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.experiment_name = (

            f"{self.algorithm}_"
            f"{self.partition}_"
            f"{self.dataset}_"
            f"{timestamp}"

        )

        self.experiment_dir = (

            Path("results")
            / "experiments"
            / self.experiment_name

        )

        self.experiment_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ==================================================
    # GET PATHS
    # ==================================================

    def get_experiment_dir(self) -> Path:

        return self.experiment_dir

    def get_models_dir(self) -> Path:

        path = (
            self.experiment_dir
            / "models"
        )

        path.mkdir(
            exist_ok=True
        )

        return path

    def get_plots_dir(self) -> Path:

        path = (
            self.experiment_dir
            / "plots"
        )

        path.mkdir(
            exist_ok=True
        )

        return path

    def get_metrics_dir(self) -> Path:

        path = (
            self.experiment_dir
            / "metrics"
        )

        path.mkdir(
            exist_ok=True
        )

        return path

    def get_history_dir(self) -> Path:

        path = (
            self.experiment_dir
            / "history"
        )

        path.mkdir(
            exist_ok=True
        )

        return path

    # ==================================================
    # SAVE CONFIG
    # ==================================================

    def save_config(
            self,
            config: dict
    ) -> None:

        config_path = (

            self.experiment_dir
            / "experiment_config.json"

        )

        with open(
                config_path,
                "w"
        ) as f:

            json.dump(
                config,
                f,
                indent=4
            )