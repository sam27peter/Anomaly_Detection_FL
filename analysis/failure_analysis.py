import os
import numpy as np
import pandas as pd


RAW_DIR = "data/raw"

TRAIN_DIR = os.path.join(
    RAW_DIR,
    "train"
)

TEST_DIR = os.path.join(
    RAW_DIR,
    "test"
)

LABEL_FILE = os.path.join(
    RAW_DIR,
    "labeled_anomalies.csv"
)


def inspect_all_channels():

    print("\nNASA DATASET STRUCTURE\n")

    shape_counter = {}

    results = []

    for file in sorted(os.listdir(TEST_DIR)):

        if not file.endswith(".npy"):
            continue

        channel = file.replace(
            ".npy",
            ""
        )

        train_path = os.path.join(
            TRAIN_DIR,
            file
        )

        test_path = os.path.join(
            TEST_DIR,
            file
        )

        train_signal = np.load(
            train_path
        )

        test_signal = np.load(
            test_path
        )

        train_shape = train_signal.shape
        test_shape = test_signal.shape

        shape_counter[
            str(test_shape)
        ] = (
            shape_counter.get(
                str(test_shape),
                0
            )
            + 1
        )

        results.append(
            {
                "channel": channel,
                "train_shape": str(train_shape),
                "test_shape": str(test_shape),
                "train_ndim": train_signal.ndim,
                "test_ndim": test_signal.ndim,
            }
        )

        print(
            f"{channel:<6} "
            f"Train={train_shape} "
            f"Test={test_shape}"
        )

    print("\n")
    print("=" * 50)
    print("UNIQUE SHAPES")
    print("=" * 50)

    for shape, count in shape_counter.items():

        print(
            f"{shape} -> {count} channels"
        )

    df = pd.DataFrame(results)

    df.to_csv(
        "data/processed/channel_shapes.csv",
        index=False
    )

    print(
        "\nSaved: data/processed/channel_shapes.csv"
    )


if __name__ == "__main__":

    inspect_all_channels()