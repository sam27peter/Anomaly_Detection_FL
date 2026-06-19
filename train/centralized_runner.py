import subprocess
import sys

from train.cent_plots import generate_all_plots

DATASETS = [
    "25",
    "55"
]

for dataset in DATASETS:

    print("\n" + "=" * 60)
    print(f"RUNNING DATASET {dataset}")
    print("=" * 60)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "train.centralized",
            dataset
        ],
        check=True
    )

    generate_all_plots(dataset)

print("\n" + "=" * 60)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 60)