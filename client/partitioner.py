import os
import json

import numpy as np


class DatasetPartitioner:

    def __init__(self):

        self.num_clients = 5

        self.random_seed = 42

        np.random.seed(
            self.random_seed
        )

    # ==================================================
    # LOAD DATASET
    # ==================================================

    def load_dataset(
        self,
        dataset_type
    ):

        dataset_dir = (
            f"data/processed/dataset_{dataset_type}"
        )

        X = np.load(
            os.path.join(
                dataset_dir,
                f"X_{dataset_type}.npy"
            )
        )

        y = np.load(
            os.path.join(
                dataset_dir,
                f"y_{dataset_type}.npy"
            )
        )

        print(
            f"\nLoaded Dataset {dataset_type}"
        )

        print(
            f"X Shape : {X.shape}"
        )

        print(
            f"y Shape : {y.shape}"
        )

        return X, y

    # ==================================================
    # IID PARTITION
    # ==================================================

    def create_iid_partitions(
        self,
        X,
        y
    ):

        indices = np.arange(
            len(X)
        )

        np.random.shuffle(
            indices
        )

        split_indices = np.array_split(
            indices,
            self.num_clients
        )

        clients = []

        for idx in split_indices:

            clients.append(
                (
                    X[idx],
                    y[idx]
                )
            )

        return clients

    # ==================================================
    # NON IID PARTITION
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

        np.random.shuffle(
            normal_idx
        )

        np.random.shuffle(
            anomaly_idx
        )

        anomaly_ratios = [
            0.02,
            0.05,
            0.10,
            0.20,
            0.40
        ]

        client_size = (
            len(X) //
            self.num_clients
        )

        clients = []

        normal_pointer = 0
        anomaly_pointer = 0

        for ratio in anomaly_ratios:

            anomaly_count = int(
                client_size * ratio
            )

            normal_count = (
                client_size -
                anomaly_count
            )

            selected_normal = (
                normal_idx[
                    normal_pointer:
                    normal_pointer + normal_count
                ]
            )

            selected_anomaly = (
                anomaly_idx[
                    anomaly_pointer:
                    anomaly_pointer + anomaly_count
                ]
            )

            normal_pointer += (
                normal_count
            )

            anomaly_pointer += (
                anomaly_count
            )

            client_indices = np.concatenate(
                [
                    selected_normal,
                    selected_anomaly
                ]
            )

            np.random.shuffle(
                client_indices
            )

            clients.append(
                (
                    X[client_indices],
                    y[client_indices]
                )
            )

        return clients

    # ==================================================
    # SAVE CLIENT DATA
    # ==================================================

    def save_clients(
        self,
        clients,
        partition_type,
        dataset_type
    ):

        save_dir = (
            f"data/partitions/"
            f"{partition_type}_{dataset_type}"
        )

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        for i, (X_client, y_client) in enumerate(
            clients,
            start=1
        ):

            save_path = os.path.join(
                save_dir,
                f"client_{i}.npz"
            )

            np.savez(
                save_path,
                X=X_client,
                y=y_client
            )

            print(
                f"Saved Client {i}"
            )

    # ==================================================
    # REPORT GENERATION
    # ==================================================

    def generate_report(
        self,
        clients,
        partition_type,
        dataset_type
    ):

        os.makedirs(
            "results/partition_reports",
            exist_ok=True
        )

        report = []

        for i, (_, y_client) in enumerate(
            clients,
            start=1
        ):

            total = len(
                y_client
            )

            anomaly = int(
                np.sum(
                    y_client == 1
                )
            )

            normal = int(
                np.sum(
                    y_client == 0
                )
            )

            anomaly_ratio = (
                anomaly /
                total
            )

            report.append(
                {

                    "client": i,

                    "samples": total,

                    "normal": normal,

                    "anomaly": anomaly,

                    "anomaly_ratio":
                        round(
                            anomaly_ratio,
                            4
                        )
                }
            )

        report_path = (
            f"results/partition_reports/"
            f"{partition_type}_{dataset_type}_report.json"
        )

        with open(
            report_path,
            "w"
        ) as f:

            json.dump(
                report,
                f,
                indent=4
            )

        print(
            f"\nReport Saved:"
        )

        print(
            report_path
        )

    # ==================================================
    # PROCESS DATASET
    # ==================================================

    def process_dataset(
        self,
        dataset_type
    ):

        print(
            "\n" +
            "=" * 60
        )

        print(
            f"DATASET {dataset_type}"
        )

        print(
            "=" * 60
        )

        X, y = self.load_dataset(
            dataset_type
        )

        # ------------------
        # IID
        # ------------------

        print(
            "\nCreating IID partitions..."
        )

        iid_clients = (
            self.create_iid_partitions(
                X,
                y
            )
        )

        self.save_clients(
            iid_clients,
            "iid",
            dataset_type
        )

        self.generate_report(
            iid_clients,
            "iid",
            dataset_type
        )

        # ------------------
        # NON IID
        # ------------------

        print(
            "\nCreating Non-IID partitions..."
        )

        non_iid_clients = (
            self.create_non_iid_partitions(
                X,
                y
            )
        )

        self.save_clients(
            non_iid_clients,
            "non_iid",
            dataset_type
        )

        self.generate_report(
            non_iid_clients,
            "non_iid",
            dataset_type
        )

    # ==================================================
    # RUN
    # ==================================================

    def run(self):

        self.process_dataset(
            "25"
        )

        self.process_dataset(
            "55"
        )

        print(
            "\n" +
            "=" * 60
        )

        print(
            "ALL PARTITIONS CREATED"
        )

        print(
            "=" * 60
        )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    partitioner = DatasetPartitioner()

    partitioner.run()