import os
import json

import pandas as pd

from client.fl_client import (
    run_client
)

# ==========================================
# CONFIG
# ==========================================

DATASETS = [
    "25",
    "55"
]

PARTITIONS = [
    "iid",
    "non_iid"
]

CLIENT_IDS = [
    1,
    2,
    3,
    4,
    5
]

# ==========================================
# SUMMARY STORAGE
# ==========================================

summary = []

# ==========================================
# RUN ALL CLIENTS
# ==========================================

for dataset in DATASETS:

    for partition in PARTITIONS:

        for client_id in CLIENT_IDS:

            print(
                "\n" +
                "=" * 60
            )

            print(
                f"RUNNING CLIENT {client_id}"
            )

            print(
                f"Dataset : {dataset}"
            )

            print(
                f"Partition : {partition}"
            )

            metrics = run_client(
                client_id=client_id,
                dataset_type=dataset,
                partition_type=partition
            )

            summary.append({

                "client_id":
                    client_id,

                "dataset_type":
                    dataset,

                "partition_type":
                    partition,

                "accuracy":
                    metrics["accuracy"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

                "f1_score":
                    metrics["f1_score"],

                "final_loss":
                    metrics["final_loss"]
            })

# ==========================================
# SAVE SUMMARY
# ==========================================

os.makedirs(
    "results/federated",
    exist_ok=True
)

summary_df = pd.DataFrame(
    summary
)

summary_df.to_csv(

    "results/federated/client_summary.csv",

    index=False
)

print(
    "\nSummary Saved"
)

print(
    "results/federated/client_summary.csv"
)

print(
    "\nAll Client Experiments Complete."
)