import subprocess
import sys
import time

from utils.logger import (
    project_logger as logger
)

from config.federated_config import (
    DATASETS
)

ALGORITHMS = [

    "fedavg",

    "fedprox"

]

PARTITIONS = [

    "iid",

    "non_iid"

]


class ExperimentOrchestrator:

    def __init__(self):

        self.start_time = None

    # ==================================================
    # EXECUTE COMMAND
    # ==================================================

    def run_command(
            self,
            command: list
    ):

        logger.info(
            f"Running: "
            f"{' '.join(command)}"
        )

        subprocess.run(

            command,

            check=True

        )

    # ==================================================
    # CENTRALIZED
    # ==================================================

    def run_centralized(self):

        logger.info(
            "=" * 60
        )

        logger.info(
            "RUNNING CENTRALIZED TRAINING"
        )

        logger.info(
            "=" * 60
        )

        self.run_command(

            [

                sys.executable,

                "-m",

                "train.centralized_runner"

            ]

        )

    # ==================================================
    # FEDERATED
    # ==================================================

    def run_federated(self):

        logger.info(
            "=" * 60
        )

        logger.info(
            "RUNNING FEDERATED TRAINING"
        )

        logger.info(
            "=" * 60
        )

        for algorithm in ALGORITHMS:

            for dataset in DATASETS:

                for partition in PARTITIONS:

                    logger.info(

                        f"{algorithm} | "

                        f"{dataset} | "

                        f"{partition}"

                    )

                    self.run_command(

                        [

                            sys.executable,

                            "-m",

                            "server.fl_server",

                            algorithm,

                            dataset,

                            partition

                        ]

                    )

    # ==================================================
    # EVALUATION
    # ==================================================

    def run_evaluation(self):

        logger.info(
            "=" * 60
        )

        logger.info(
            "RUNNING EVALUATIONS"
        )

        logger.info(
            "=" * 60
        )

        modules = [

            "evaluation.comparison",

            "evaluation.centralized_vs_fl",

            "evaluation.statistical_analysis"

        ]

        for module in modules:

            self.run_command(

                [

                    sys.executable,

                    "-m",

                    module

                ]

            )

    # ==================================================
    # VISUALIZATION
    # ==================================================

    def run_visualizations(self):

        logger.info(
            "=" * 60
        )

        logger.info(
            "GENERATING VISUALIZATIONS"
        )

        logger.info(
            "=" * 60
        )

        modules = [

            "visualization.comparison_plots",

            "visualization.comparison_table",

            "visualization.statistical_plots"

        ]

        for module in modules:

            self.run_command(

                [

                    sys.executable,

                    "-m",

                    module

                ]

            )

    # ==================================================
    # RUN
    # ==================================================

    def run(self):

        self.start_time = time.time()

        logger.info(
            "STARTING COMPLETE PIPELINE"
        )

        # --------------------------------

        self.run_centralized()

        # --------------------------------

        self.run_federated()

        # --------------------------------

        self.run_evaluation()

        # --------------------------------

        self.run_visualizations()

        # --------------------------------

        elapsed = (

            time.time()

            -

            self.start_time

        )

        logger.info(
            "=" * 60
        )

        logger.info(
            "PIPELINE COMPLETED"
        )

        logger.info(
            f"Total Time: "

            f"{elapsed:.2f} sec"

        )

        logger.info(
            "=" * 60
        )


if __name__ == "__main__":

    ExperimentOrchestrator().run()