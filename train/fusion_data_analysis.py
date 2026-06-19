import json
import os

import numpy as np
import pandas as pd


class FusionDataAnalysis:

    def __init__(self):

        self.labels = pd.read_csv(
            "data/raw/labeled_anomalies.csv"
        )

        self.test_dir = "data/raw/test"

    # ==========================================
    # ANALYZE CHANNEL TYPES
    # ==========================================

    def analyze_feature_groups(self):

        group_25 = []

        group_55 = []

        print("\nAnalyzing channels...\n")

        for _, row in self.labels.iterrows():

            channel = row["chan_id"]

            file_path = os.path.join(
                self.test_dir,
                f"{channel}.npy"
            )

            if not os.path.exists(file_path):
                continue

            signal = np.load(file_path)

            feature_count = signal.shape[1]

            if feature_count == 25:

                group_25.append(channel)

            elif feature_count == 55:

                group_55.append(channel)

        print("=" * 60)
        print("25 FEATURE CHANNELS")
        print("=" * 60)

        print(
            f"Count : {len(group_25)}"
        )

        print(group_25)

        print("\n")

        print("=" * 60)
        print("55 FEATURE CHANNELS")
        print("=" * 60)

        print(
            f"Count : {len(group_55)}"
        )

        print(group_55)

        return group_25, group_55

    # ==========================================
    # SPACECRAFT ANALYSIS
    # ==========================================

    def analyze_spacecrafts(self):

        print("\n")

        print("=" * 60)
        print("SPACECRAFT DISTRIBUTION")
        print("=" * 60)

        spacecraft_counts = (
            self.labels["spacecraft"]
            .value_counts()
        )

        print(spacecraft_counts)

        return spacecraft_counts.to_dict()

    # ==========================================
    # LENGTH ANALYSIS
    # ==========================================

    def analyze_lengths(self):

        lengths_25 = []

        lengths_55 = []

        for _, row in self.labels.iterrows():

            channel = row["chan_id"]

            file_path = os.path.join(
                self.test_dir,
                f"{channel}.npy"
            )

            if not os.path.exists(file_path):
                continue

            signal = np.load(file_path)

            length = signal.shape[0]

            features = signal.shape[1]

            if features == 25:

                lengths_25.append(length)

            elif features == 55:

                lengths_55.append(length)

        print("\n")

        print("=" * 60)
        print("LENGTH ANALYSIS")
        print("=" * 60)

        print(
            f"25 Feature Channels:"
        )

        print(
            f"Min : {min(lengths_25)}"
        )

        print(
            f"Max : {max(lengths_25)}"
        )

        print(
            f"Mean: {np.mean(lengths_25):.2f}"
        )

        print("\n")

        print(
            f"55 Feature Channels:"
        )

        print(
            f"Min : {min(lengths_55)}"
        )

        print(
            f"Max : {max(lengths_55)}"
        )

        print(
            f"Mean: {np.mean(lengths_55):.2f}"
        )

    # ==========================================
    # SAVE REPORT
    # ==========================================

    def save_report(
        self,
        group_25,
        group_55,
        spacecrafts
    ):

        os.makedirs(
            "results/fusion_analysis",
            exist_ok=True
        )

        report = {

            "dataset_25_channels":
                group_25,

            "dataset_55_channels":
                group_55,

            "dataset_25_count":
                len(group_25),

            "dataset_55_count":
                len(group_55),

            "spacecraft_distribution":
                spacecrafts
        }

        with open(
            "results/fusion_analysis/fusion_report.json",
            "w"
        ) as f:

            json.dump(
                report,
                f,
                indent=4
            )

    # ==========================================
    # RUN
    # ==========================================

    def run(self):

        group_25, group_55 = (
            self.analyze_feature_groups()
        )

        spacecrafts = (
            self.analyze_spacecrafts()
        )

        self.analyze_lengths()

        self.save_report(
            group_25,
            group_55,
            spacecrafts
        )

        print("\n")

        print(
            "Fusion analysis saved."
        )


if __name__ == "__main__":

    analyzer = FusionDataAnalysis()

    analyzer.run()