import os
import json
import copy
import sys
import numpy as np

import torch
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from client.fl_client import run_client
from models.model_selector import get_model


# ==================================================
# CONFIG
# ==================================================

from config.federated_config import (
    NUM_CLIENTS,
    NUM_ROUNDS
)

DATASET_TYPE = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "SMAP"
)

PARTITION_TYPE = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "iid"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# FEDERATED PROX AGGREGATION
# ==================================================

def federated_average(local_weights):

    avg_weights = copy.deepcopy(
        local_weights[0]
    )

    for key in avg_weights.keys():

        avg_weights[key] = torch.stack(
            [
                w[key].float()
                for w in local_weights
            ],
            dim=0
        ).mean(dim=0)

    return avg_weights


# ==================================================
# FEDPROX SERVER
# ==================================================

def run_fedprox():

    print("\nStarting FedProx\n")

    num_features = int(
        DATASET_TYPE
    )

    global_model = get_model(
        "cnn",
        num_features
    ).to(DEVICE)

    global_acc_history = []

    global_loss_history = []

    final_client_accuracies = []

    save_dir = (
        f"results/federated/fedprox/"
        f"{PARTITION_TYPE}_{DATASET_TYPE}"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # ==================================================
    # COMMUNICATION NUM_ROUNDS
    # ==================================================

    for rnd in range(NUM_ROUNDS):

        print("\n" + "=" * 60)
        print(
            f"ROUND {rnd+1}/{NUM_ROUNDS}"
        )
        print("=" * 60)

        local_weights = []

        round_acc = []

        round_loss = []

        global_weights = copy.deepcopy(
            global_model.state_dict()
        )

        # ------------------------------------------

        for client_id in range(
                1,
                NUM_CLIENTS + 1
        ):

            print(
                f"\nTraining Client {client_id}"
            )

            weights, metrics = run_client(

                client_id=client_id,

                dataset_type=DATASET_TYPE,

                partition_type=PARTITION_TYPE,

                global_weights=global_weights
            )

            local_weights.append(
                weights
            )

            round_acc.append(
                metrics["accuracy"]
            )

            round_loss.append(
                metrics["final_loss"]
            )

        # ------------------------------------------
        # FEDPROX AGGREGATION
        # ------------------------------------------

        averaged_weights = (
            federated_average(
                local_weights
            )
        )

        global_model.load_state_dict(
            averaged_weights
        )

        round_global_acc = np.mean(
            round_acc
        )

        round_global_loss = np.mean(
            round_loss
        )

        global_acc_history.append(
            float(round_global_acc)
        )

        global_loss_history.append(
            float(round_global_loss)
        )

        final_client_accuracies = (
            round_acc
        )

        print(
            f"\nRound Accuracy : "
            f"{round_global_acc:.4f}"
        )

    # ==================================================
    # GLOBAL EVALUATION
    # ==================================================

    print("\n" + "=" * 60)
    print("GLOBAL MODEL EVALUATION")
    print("=" * 60)

    all_X = []

    all_y = []

    for client_id in range(
            1,
            NUM_CLIENTS + 1
    ):

        client_file = (
            f"data/partitions/"
            f"{PARTITION_TYPE}_{DATASET_TYPE}/"
            f"client_{client_id}.npz"
        )

        data = np.load(
            client_file
        )

        all_X.append(
            data["X"]
        )

        all_y.append(
            data["y"]
        )

    X = np.concatenate(
        all_X,
        axis=0
    )

    y = np.concatenate(
        all_y,
        axis=0
    )

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    y = torch.tensor(
        y,
        dtype=torch.float32
    )

    test_loader = DataLoader(
        TensorDataset(
            X,
            y
        ),
        batch_size=64,
        shuffle=False
    )

    global_model.eval()

    predictions = []

    ground_truth = []

    with torch.no_grad():

        for batch_x, batch_y in test_loader:

            batch_x = batch_x.to(
                DEVICE
            )

            outputs = global_model(
                batch_x
            )

            preds = (
                outputs > 0.5
            ).int()

            predictions.extend(
                preds.cpu().numpy()
            )

            ground_truth.extend(
                batch_y.numpy()
            )

    predictions = np.array(
        predictions
    ).flatten()

    ground_truth = np.array(
        ground_truth
    ).flatten()

    global_accuracy = accuracy_score(
        ground_truth,
        predictions
    )

    global_precision = precision_score(
        ground_truth,
        predictions,
        zero_division=0
    )

    global_recall = recall_score(
        ground_truth,
        predictions,
        zero_division=0
    )

    global_f1 = f1_score(
        ground_truth,
        predictions,
        zero_division=0
    )

    global_cm = confusion_matrix(
        ground_truth,
        predictions
    )

    print(f"Global Accuracy : {global_accuracy:.4f}")
    print(f"Global Precision: {global_precision:.4f}")
    print(f"Global Recall   : {global_recall:.4f}")
    print(f"Global F1 Score : {global_f1:.4f}")

    print("\nGlobal Confusion Matrix")
    print(global_cm)

    # ==================================================
    # GLOBAL CONFUSION MATRIX IMAGE
    # ==================================================

    disp = ConfusionMatrixDisplay(
        confusion_matrix=global_cm
    )

    disp.plot()

    plt.title(
        "Global Confusion Matrix"
    )

    plt.savefig(
        os.path.join(
            save_dir,
            "global_confusion_matrix.png"
        )
    )

    plt.close()

    # ==================================================
    # SAVE GLOBAL MODEL
    # ==================================================

    torch.save(
        global_model.state_dict(),
        os.path.join(
            save_dir,
            "global_model.pth"
        )
    )

    # ==================================================
    # SAVE HISTORY
    # ==================================================

    history = {

        "accuracy":
            global_acc_history,

        "loss":
            global_loss_history,

        "client_accuracy":
            final_client_accuracies
    }

    with open(
            os.path.join(
                save_dir,
                "history.json"
            ),
            "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # ==================================================
    # SAVE METRICS
    # ==================================================

    metrics = {

        "dataset_type":
            DATASET_TYPE,

        "partition_type":
            PARTITION_TYPE,

        "num_clients":
            NUM_CLIENTS,

        "rounds":
            NUM_ROUNDS,

        "global_accuracy":
            float(global_accuracy),

        "global_precision":
            float(global_precision),

        "global_recall":
            float(global_recall),

        "global_f1_score":
            float(global_f1),

        "confusion_matrix":
            global_cm.tolist()
    }

    with open(
            os.path.join(
                save_dir,
                "metrics.json"
            ),
            "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # ==================================================
    # ACCURACY CURVE
    # ==================================================

    plt.figure()

    plt.plot(
        global_acc_history,
        marker="o"
    )

    plt.title(
        "Global Accuracy vs Round"
    )

    plt.xlabel("Round")
    plt.ylabel("Accuracy")

    plt.grid()

    plt.savefig(
        os.path.join(
            save_dir,
            "accuracy_curve.png"
        )
    )

    plt.close()

    # ==================================================
    # LOSS CURVE
    # ==================================================

    plt.figure()

    plt.plot(
        global_loss_history,
        marker="o"
    )

    plt.title(
        "Global Loss vs Round"
    )

    plt.xlabel("Round")
    plt.ylabel("Loss")

    plt.grid()

    plt.savefig(
        os.path.join(
            save_dir,
            "loss_curve.png"
        )
    )

    plt.close()

    # ==================================================
    # CLIENT ACCURACY PLOT
    # ==================================================

    plt.figure()

    plt.bar(
        [
            f"C{i}"
            for i in range(1, 6)
        ],
        final_client_accuracies
    )

    plt.title(
        "Per Client Accuracy"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.savefig(
        os.path.join(
            save_dir,
            "client_accuracy.png"
        )
    )

    plt.close()

    summary_file = (
        "results/federated/"
        "fedprox_summary.csv"
    )

    summary = pd.DataFrame([

        {
            "dataset":
                DATASET_TYPE,

            "partition":
                PARTITION_TYPE,

            "accuracy":
                global_accuracy,

            "precision":
                global_precision,

            "recall":
                global_recall,

            "f1":
                global_f1

        }
    ])

    if os.path.exists(summary_file):

        summary.to_csv(
            summary_file,
            mode="a",
            header=False,
            index=False
        )

    else:

        summary.to_csv(
            summary_file,
            index=False
        )

    print("\nFedProx Completed")


if __name__ == "__main__":

    run_fedprox()