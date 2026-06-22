import subprocess
import sys

experiments = [

    ("25", "iid"),
    ("25", "non_iid"),
    ("55", "iid"),
    ("55", "non_iid")

]

for dataset, partition in experiments:

    print("\n")
    print("=" * 70)

    print(
        f"RUNNING FEDAVG | "
        f"DATASET={dataset} | "
        f"PARTITION={partition}"
    )

    print("=" * 70)

    subprocess.run(

        [

            sys.executable,   # <-- IMPORTANT

            "-m",

            "server.fl_server_avg",

            dataset,

            partition

        ],

        check=True

    )

print("\n")
print("=" * 70)
print("ALL FEDAVG EXPERIMENTS COMPLETED")
print("=" * 70)