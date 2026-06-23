import os
import json
import numpy as np


RAW_DIR = "data/raw"
TEST_DIR = os.path.join(RAW_DIR, "test")

OUTPUT_DIR = "data/processed/reports"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def analyze_features():

    feature_groups = {}

    print("\nFEATURE ANALYSIS\n")

    for file in sorted(os.listdir(TEST_DIR)):

        if not file.endswith(".npy"):
            continue

        channel = file.replace(".npy", "")

        path = os.path.join(
            TEST_DIR,
            file
        )

        signal = np.load(path)

        feature_count = signal.shape[1]

        if feature_count not in feature_groups:

            feature_groups[
                feature_count
            ] = []

        feature_groups[
            feature_count
        ].append(channel)

    print("=" * 50)

    for feature_count, channels in sorted(
        feature_groups.items()
    ):

        print(
            f"\nFeature Count: {feature_count}"
        )

        print(
            f"Channels: {len(channels)}"
        )

        print(
            ", ".join(channels)
        )

    summary = {}

    for feature_count, channels in feature_groups.items():

        summary[str(feature_count)] = {

            "count": len(channels),

            "channels": channels
        }

    save_path = os.path.join(
        OUTPUT_DIR,
        "feature_analysis.json"
    )

    with open(
        save_path,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print("\n")
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)

    for feature_count, channels in feature_groups.items():

        print(
            f"{feature_count} Features -> {len(channels)} Channels"
        )

    print(
        f"\nSaved: {save_path}"
    )


if __name__ == "__main__":

    analyze_features()