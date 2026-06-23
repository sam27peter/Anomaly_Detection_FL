import json
from pathlib import Path

import numpy as np

from config.federated_config import (
    NUM_CLIENTS,
    DATASETS
)

from utils.logger import logger


class DatasetPartitioner:

    def __init__(self):

        self.num_clients = NUM_CLIENTS

        np.random.seed(42)

    # ==================================================

    def load_dataset(
            self,
            dataset_type: str
    ):

        dataset_dir = (

                Path("data")
                / "processed"
                / dataset_type

        )

        X = np.load(
            dataset_dir
            / f"X_{dataset_type}.npy"
        )

        y = np.load(
            dataset_dir
            / f"y_{dataset_type}.npy"
        )

        logger.info(
            f"{dataset_type} Loaded "
            f"{X.shape}"
        )

        return X, y

    # ==================================================

    def create_iid_partitions(
            self,
            X,
            y
    ):

        indices = np.arange(len(X))

        np.random.shuffle(indices)

        split_indices = np.array_split(
            indices,
            self.num_clients
        )

        return [

            (X[idx], y[idx])

            for idx in split_indices

        ]

    # ==================================================

    def create_non_iid_partitions(
            self,
            X,
            y
    ):

        normal_idx = np.where(
            y == 0
        )[0]

        anomaly_idx = np.where(
            y == 1
        )[0]

        np.random.shuffle(normal_idx)
        np.random.shuffle(anomaly_idx)

        anomaly_ratios = [
            0.02,
            0.05,
            0.10,
            0.20,
            0.40
        ]

        client_size = (
                len(X)
                // self.num_clients
        )

        clients = []

        n_ptr = 0
        a_ptr = 0

        for ratio in anomaly_ratios:

            a_count = int(
                client_size * ratio
            )

            n_count = (
                    client_size - a_count
            )

            selected_normal = normal_idx[
                              n_ptr:n_ptr + n_count
                              ]

            selected_anomaly = anomaly_idx[
                               a_ptr:a_ptr + a_count
                               ]

            n_ptr += n_count
            a_ptr += a_count

            indices = np.concatenate(
                [
                    selected_normal,
                    selected_anomaly
                ]
            )

            np.random.shuffle(indices)

            clients.append(
                (
                    X[indices],
                    y[indices]
                )
            )

        return clients

    # ==================================================

    def save_clients(
            self,
            clients,
            partition_type,
            dataset_type
    ):

        save_dir = (

                Path("data")
                / "partitions"
                / f"{partition_type}_{dataset_type}"

        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for idx, (
                X_client,
                y_client
        ) in enumerate(
            clients,
            start=1
        ):

            np.savez(

                save_dir
                / f"client_{idx}.npz",

                X=X_client,
                y=y_client
            )

    # ==================================================

    def generate_report(

            self,

            clients,

            partition_type,

            dataset_type

    ):

        report = []

        for idx, (_, y_client) in enumerate(
                clients,
                start=1
        ):

            total = len(y_client)

            anomaly = int(
                np.sum(y_client == 1)
            )

            normal = int(
                np.sum(y_client == 0)
            )

            report.append({

                "client":
                    idx,

                "samples":
                    total,

                "normal":
                    normal,

                "anomaly":
                    anomaly,

                "anomaly_ratio":

                    round(
                        anomaly / total,
                        4
                    )

            })

        save_dir = (
                Path("results")
                / "partition_reports"
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(

                save_dir
                / f"{partition_type}_{dataset_type}.json",

                "w"

        ) as f:

            json.dump(
                report,
                f,
                indent=4
            )

    # ==================================================

    def process_dataset(
            self,
            dataset_type
    ):

        logger.info(
            f"Processing {dataset_type}"
        )

        X, y = self.load_dataset(
            dataset_type
        )

        iid = self.create_iid_partitions(
            X,
            y
        )

        self.save_clients(
            iid,
            "iid",
            dataset_type
        )

        self.generate_report(
            iid,
            "iid",
            dataset_type
        )

        non_iid = (
            self.create_non_iid_partitions(
                X,
                y
            )
        )

        self.save_clients(
            non_iid,
            "non_iid",
            dataset_type
        )

        self.generate_report(
            non_iid,
            "non_iid",
            dataset_type
        )

    # ==================================================

    def run(self):

        for dataset in DATASETS:

            self.process_dataset(
                dataset
            )

        logger.info(
            "All partitions created"
        )


if __name__ == "__main__":

    DatasetPartitioner().run()