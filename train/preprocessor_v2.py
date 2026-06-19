import os
import ast
import json
import numpy as np
import pandas as pd

WINDOW_SIZE = 100
STRIDE = 10


class NASAPreprocessorV2:

    def __init__(self):

        self.test_dir = "data/raw/test"

        self.labels_df = pd.read_csv(
            "data/raw/labeled_anomalies.csv"
        )

    # ==================================================
    # BUILD BINARY LABELS
    # ==================================================

    def build_binary_labels(
        self,
        signal_length,
        anomaly_sequences
    ):

        labels = np.zeros(
            signal_length,
            dtype=np.int8
        )

        intervals = ast.literal_eval(
            anomaly_sequences
        )

        for start, end in intervals:

            start = max(
                0,
                start
            )

            end = min(
                signal_length - 1,
                end
            )

            labels[start:end + 1] = 1

        return labels

    # ==================================================
    # NORMALIZE SIGNAL
    # ==================================================

    def normalize_signal(
        self,
        signal
    ):

        mean = np.mean(
            signal,
            axis=0
        )

        std = np.std(
            signal,
            axis=0
        )

        return (
            signal - mean
        ) / (
            std + 1e-8
        )

    # ==================================================
    # CREATE WINDOWS
    # ==================================================

    def create_windows(
        self,
        signal,
        labels
    ):

        X = []
        y = []

        for start in range(
            0,
            len(signal) - WINDOW_SIZE,
            STRIDE
        ):

            end = start + WINDOW_SIZE

            window = signal[
                start:end
            ]

            label = int(
                np.max(
                    labels[start:end]
                )
            )

            X.append(
                window
            )

            y.append(
                label
            )

        return (
            np.array(X),
            np.array(y)
        )

    # ==================================================
    # PROCESS SINGLE CHANNEL
    # ==================================================

    def process_channel(
        self,
        channel
    ):

        path = os.path.join(
            self.test_dir,
            f"{channel}.npy"
        )

        signal = np.load(path)

        row = self.labels_df[
            self.labels_df["chan_id"]
            == channel
        ]

        if len(row) == 0:

            return None

        labels = self.build_binary_labels(
            len(signal),
            row.iloc[0]["anomaly_sequences"]
        )

        signal = self.normalize_signal(
            signal
        )

        X, y = self.create_windows(
            signal,
            labels
        )

        feature_count = signal.shape[1]

        return X, y, feature_count

    # ==================================================
    # SAVE DATASET
    # ==================================================

    def save_dataset(
        self,
        X,
        y,
        folder,
        prefix
    ):

        os.makedirs(
            folder,
            exist_ok=True
        )

        np.save(
            os.path.join(
                folder,
                f"X_{prefix}.npy"
            ),
            X
        )

        np.save(
            os.path.join(
                folder,
                f"y_{prefix}.npy"
            ),
            y
        )

        metadata = {

            "window_size": WINDOW_SIZE,

            "stride": STRIDE,

            "samples": int(
                len(X)
            ),

            "shape": list(
                X.shape
            ),

            "normal_windows": int(
                np.sum(y == 0)
            ),

            "anomaly_windows": int(
                np.sum(y == 1)
            )
        }

        with open(
            os.path.join(
                folder,
                f"metadata_{prefix}.json"
            ),
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

    # ==================================================
    # PROCESS ALL CHANNELS
    # ==================================================

    def process_all_channels(self):

        X_25 = []
        y_25 = []

        X_55 = []
        y_55 = []

        channels = sorted(
            self.labels_df[
                "chan_id"
            ].tolist()
        )

        print(
            f"\nProcessing {len(channels)} channels\n"
        )

        for channel in channels:

            try:

                result = self.process_channel(
                    channel
                )

                if result is None:

                    continue

                X, y, feature_count = result

                if feature_count == 25:

                    X_25.append(
                        X
                    )

                    y_25.append(
                        y
                    )

                elif feature_count == 55:

                    X_55.append(
                        X
                    )

                    y_55.append(
                        y
                    )

                print(
                    f"{channel:<5}"
                    f" -> {feature_count} Features"
                    f" -> {len(X)} Windows"
                )

            except Exception as e:

                print(
                    f"Error in {channel}: {e}"
                )

        X_25 = np.concatenate(
            X_25,
            axis=0
        )

        y_25 = np.concatenate(
            y_25,
            axis=0
        )

        X_55 = np.concatenate(
            X_55,
            axis=0
        )

        y_55 = np.concatenate(
            y_55,
            axis=0
        )

        self.save_dataset(
            X_25,
            y_25,
            "data/processed/dataset_25",
            "25"
        )

        self.save_dataset(
            X_55,
            y_55,
            "data/processed/dataset_55",
            "55"
        )

        print("\n")
        print("=" * 60)
        print("DATASET 25")
        print("=" * 60)

        print(
            f"X_25 Shape : {X_25.shape}"
        )

        print(
            f"y_25 Shape : {y_25.shape}"
        )

        print("\n")
        print("=" * 60)
        print("DATASET 55")
        print("=" * 60)

        print(
            f"X_55 Shape : {X_55.shape}"
        )

        print(
            f"y_55 Shape : {y_55.shape}"
        )

    # ==================================================
    # RUN
    # ==================================================

    def run(self):

        self.process_all_channels()


if __name__ == "__main__":

    processor = NASAPreprocessorV2()

    processor.run()
