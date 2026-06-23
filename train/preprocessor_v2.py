import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.federated_config import (
    WINDOW_SIZE,
    STRIDE
)

from utils.logger import logger


class NASAPreprocessorV2:
    """
    NASA dataset preprocessing pipeline.
    """

    def __init__(self) -> None:

        self.test_dir = (
            Path("data")
            / "raw"
            / "test"
        )

        self.labels_df = pd.read_csv(
            Path("data")
            / "raw"
            / "labeled_anomalies.csv"
        )

    # ==================================================
    # LABEL CREATION
    # ==================================================

    def build_binary_labels(
            self,
            signal_length: int,
            anomaly_sequences: str
    ) -> np.ndarray:

        labels = np.zeros(
            signal_length,
            dtype=np.int8
        )

        intervals = ast.literal_eval(
            anomaly_sequences
        )

        for start, end in intervals:

            start = max(0, start)

            end = min(
                signal_length - 1,
                end
            )

            labels[start:end + 1] = 1

        return labels

    # ==================================================

    def normalize_signal(
            self,
            signal: np.ndarray
    ) -> np.ndarray:

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

    def create_windows(
            self,
            signal: np.ndarray,
            labels: np.ndarray
    ) -> tuple:

        X = []
        y = []

        for start in range(
                0,
                len(signal) - WINDOW_SIZE,
                STRIDE
        ):

            end = start + WINDOW_SIZE

            X.append(
                signal[start:end]
            )

            y.append(
                int(
                    np.max(
                        labels[start:end]
                    )
                )
            )

        return (
            np.array(X),
            np.array(y)
        )

    # ==================================================

    def process_channel(
            self,
            channel: str
    ):

        path = (
            self.test_dir
            / f"{channel}.npy"
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
            row.iloc[0][
                "anomaly_sequences"
            ]
        )

        signal = self.normalize_signal(
            signal
        )

        X, y = self.create_windows(
            signal,
            labels
        )

        return (
            X,
            y,
            signal.shape[1]
        )

    # ==================================================

    def save_dataset(
            self,
            X: np.ndarray,
            y: np.ndarray,
            folder: Path,
            dataset_name: str
    ) -> None:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            folder / f"X_{dataset_name}.npy",
            X
        )

        np.save(
            folder / f"y_{dataset_name}.npy",
            y
        )

        metadata = {

            "dataset":
                dataset_name,

            "window_size":
                WINDOW_SIZE,

            "stride":
                STRIDE,

            "samples":
                int(len(X)),

            "shape":
                list(X.shape),

            "normal_windows":
                int(np.sum(y == 0)),

            "anomaly_windows":
                int(np.sum(y == 1))
        }

        with open(
                folder
                / f"metadata_{dataset_name}.json",
                "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

    # ==================================================

    def process_all_channels(
            self
    ) -> None:

        smap_X = []
        smap_y = []

        msl_X = []
        msl_y = []

        channels = sorted(
            self.labels_df[
                "chan_id"
            ].tolist()
        )

        logger.info(
            f"Processing {len(channels)} channels"
        )

        for channel in channels:

            try:

                result = self.process_channel(
                    channel
                )

                if result is None:
                    continue

                X, y, features = result

                if features == 25:

                    smap_X.append(X)
                    smap_y.append(y)

                elif features == 55:

                    msl_X.append(X)
                    msl_y.append(y)

                logger.info(
                    f"{channel} -> "
                    f"{features} features "
                    f"-> {len(X)} windows"
                )

            except Exception as e:

                logger.error(
                    f"{channel}: {e}"
                )

        smap_X = np.concatenate(
            smap_X,
            axis=0
        )

        smap_y = np.concatenate(
            smap_y,
            axis=0
        )

        msl_X = np.concatenate(
            msl_X,
            axis=0
        )

        msl_y = np.concatenate(
            msl_y,
            axis=0
        )

        self.save_dataset(

            smap_X,
            smap_y,

            Path("data")
            / "processed"
            / "SMAP",

            "SMAP"
        )

        self.save_dataset(

            msl_X,
            msl_y,

            Path("data")
            / "processed"
            / "MSL",

            "MSL"
        )

        logger.info(
            f"SMAP Shape : {smap_X.shape}"
        )

        logger.info(
            f"MSL Shape : {msl_X.shape}"
        )

    # ==================================================

    def run(self):

        self.process_all_channels()


if __name__ == "__main__":

    NASAPreprocessorV2().run()