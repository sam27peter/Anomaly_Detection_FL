from pathlib import Path
import logging


class ExperimentLogger:

    def __init__(
            self,
            experiment_name: str
    ):

        self.log_dir = (
            Path("logs")
            / "experiments"
        )

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.logger = logging.getLogger(
            experiment_name
        )

        self.logger.setLevel(
            logging.INFO
        )

        if not self.logger.handlers:

            formatter = logging.Formatter(

                "%(asctime)s | "
                "%(levelname)s | "
                "%(message)s"

            )

            handler = logging.FileHandler(

                self.log_dir
                /
                f"{experiment_name}.log"

            )

            handler.setFormatter(
                formatter
            )

            self.logger.addHandler(
                handler
            )

    def info(
            self,
            message: str
    ):

        self.logger.info(
            message
        )