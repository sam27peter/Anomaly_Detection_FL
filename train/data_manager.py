import os
import ast
import json
import numpy as np
import pandas as pd


class NASADataManager:

    def __init__(self):

        # ==================================================
        # PATHS
        # ==================================================

        self.raw_dir = "data/raw"

        self.train_dir = os.path.join(
            self.raw_dir,
            "train"
        )

        self.test_dir = os.path.join(
            self.raw_dir,
            "test"
        )

        self.label_file = os.path.join(
            self.raw_dir,
            "labeled_anomalies.csv"
        )

        self.processed_dir = "data/processed"

        self.report_dir = os.path.join(
            self.processed_dir,
            "reports"
        )

        os.makedirs(
            self.report_dir,
            exist_ok=True
        )

        self.labels_df = None

    # ==================================================
    # LOAD LABELS
    # ==================================================

    def load_labels(self):

        print("\nLoading labels...")

        self.labels_df = pd.read_csv(
            self.label_file
        )

        print(
            f"Loaded {len(self.labels_df)} channels"
        )

        print("\nColumns:")

        for col in self.labels_df.columns:
            print(f"  - {col}")

    # ==================================================
    # INSPECT DATASET
    # ==================================================

    def inspect_dataset(self):

        print("\nInspecting dataset...")

        train_files = sorted(
            [
                f for f in os.listdir(self.train_dir)
                if f.endswith(".npy")
            ]
        )

        test_files = sorted(
            [
                f for f in os.listdir(self.test_dir)
                if f.endswith(".npy")
            ]
        )

        print(
            f"\nTrain Channels : {len(train_files)}"
        )

        print(
            f"Test Channels  : {len(test_files)}"
        )

        csv_channels = set(
            self.labels_df["chan_id"]
        )

        train_channels = set(
            f.replace(".npy", "")
            for f in train_files
        )

        test_channels = set(
            f.replace(".npy", "")
            for f in test_files
        )

        missing_train = csv_channels - train_channels
        missing_test = csv_channels - test_channels

        print(
            f"\nChannels in CSV : {len(csv_channels)}"
        )

        print(
            f"Channels in Train : {len(train_channels)}"
        )

        print(
            f"Channels in Test : {len(test_channels)}"
        )

        print(
            f"\nMissing Train Channels : {len(missing_train)}"
        )

        print(
            f"Missing Test Channels : {len(missing_test)}"
        )

    # ==================================================
    # GENERATE DATASET REPORT
    # ==================================================

    def generate_dataset_report(self):

        print("\nGenerating Dataset Report...\n")

        report_rows = []

        total_anomaly_points = 0
        signal_lengths = []

        channel_ids = sorted(
            [
                f.replace(".npy", "")
                for f in os.listdir(self.test_dir)
                if f.endswith(".npy")
            ]
        )

        for channel in channel_ids:

            try:

                train_path = os.path.join(
                    self.train_dir,
                    f"{channel}.npy"
                )

                test_path = os.path.join(
                    self.test_dir,
                    f"{channel}.npy"
                )

                train_signal = np.load(
                    train_path
                )

                test_signal = np.load(
                    test_path
                )

                train_length = len(
                    train_signal
                )

                test_length = len(
                    test_signal
                )

                signal_lengths.append(
                    test_length
                )

                row = self.labels_df[
                    self.labels_df["chan_id"] == channel
                ]

                anomaly_intervals = 0
                anomaly_points = 0
                subsystem_class = "Unknown"

                if len(row) > 0:

                    subsystem_class = row.iloc[0]["class"]

                    intervals = ast.literal_eval(
                        row.iloc[0]["anomaly_sequences"]
                    )

                    anomaly_intervals = len(
                        intervals
                    )

                    for start, end in intervals:

                        anomaly_points += (
                            end - start + 1
                        )

                total_anomaly_points += (
                    anomaly_points
                )

                report_rows.append(
                    {
                        "channel_id": channel,
                        "train_length": train_length,
                        "test_length": test_length,
                        "min": float(
                            np.min(test_signal)
                        ),
                        "max": float(
                            np.max(test_signal)
                        ),
                        "mean": float(
                            np.mean(test_signal)
                        ),
                        "std": float(
                            np.std(test_signal)
                        ),
                        "anomaly_intervals": anomaly_intervals,
                        "anomaly_points": anomaly_points,
                        "class": subsystem_class
                    }
                )

                print(
                    f"Processed {channel}"
                )

            except Exception as e:

                print(
                    f"Error in {channel}: {e}"
                )

        # ==================================================
        # SAVE CHANNEL REPORT
        # ==================================================

        report_df = pd.DataFrame(
            report_rows
        )

        csv_path = os.path.join(
            self.report_dir,
            "channel_statistics.csv"
        )

        report_df.to_csv(
            csv_path,
            index=False
        )

        # ==================================================
        # DATASET SUMMARY
        # ==================================================

        summary = {

            "num_channels":
                len(report_df),

            "largest_signal":
                int(max(signal_lengths)),

            "smallest_signal":
                int(min(signal_lengths)),

            "average_signal_length":
                float(
                    np.mean(signal_lengths)
                ),

            "total_anomaly_points":
                int(total_anomaly_points)
        }

        summary_path = os.path.join(
            self.report_dir,
            "dataset_summary.json"
        )

        with open(
            summary_path,
            "w"
        ) as f:

            json.dump(
                summary,
                f,
                indent=4
            )

        print("\n===================================")
        print("DATASET REPORT COMPLETE")
        print("===================================")

        print(
            f"Channels Analysed : {len(report_df)}"
        )

        print(
            f"Largest Signal    : {summary['largest_signal']}"
        )

        print(
            f"Smallest Signal   : {summary['smallest_signal']}"
        )

        print(
            f"Average Length    : {summary['average_signal_length']:.2f}"
        )

        print(
            f"Total Anomaly Points : {summary['total_anomaly_points']}"
        )

        print(
            f"\nSaved CSV  : {csv_path}"
        )

        print(
            f"Saved JSON : {summary_path}"
        )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    manager = NASADataManager()

    manager.load_labels()

    manager.inspect_dataset()

    manager.generate_dataset_report()