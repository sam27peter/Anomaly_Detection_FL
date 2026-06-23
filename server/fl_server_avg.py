import copy
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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
from utils.logger import logger

from config.federated_config import (
    NUM_CLIENTS,
    NUM_ROUNDS
)

# ==================================================
# CONFIG
# ==================================================

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
# FEDERATED AVERAGING
# ==================================================


def federated_average(
        local_weights: list
) -> dict:
    """
    Aggregate local client weights.

    Parameters
    ----------
    local_weights : list
        List of client model weights.

    Returns
    -------
    dict
        Averaged global weights.
    """

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
# FEDAVG SERVER
# ==================================================

def run_fedavg() -> None:
    """
    Run FedAvg training.
    """

    logger.info("Starting FedAvg")

    start_time = time.time()

    num_features = (
        25
        if DATASET_TYPE == "SMAP"
        else 55
    )

    global_model = get_model(
        "cnn",
        num_features
    ).to(DEVICE)

    global_acc_history = []
    global_loss_history = []

    final_client_accuracies = []

    save_dir = (
        Path("results")
        / "federated"
        / "fedavg"
        / f"{PARTITION_TYPE}_{DATASET_TYPE}"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==================================================
    # ROUNDS
    # ==================================================

    for rnd in range(NUM_ROUNDS):

        logger.info(
            f"ROUND {rnd + 1}/{NUM_ROUNDS}"
        )

        local_weights = []

        round_acc = []
        round_loss = []

        global_weights = copy.deepcopy(
            global_model.state_dict()
        )

        # ---------------------------------------------

        for client_id in range(
                1,
                NUM_CLIENTS + 1
        ):

            logger.info(
                f"Training Client {client_id}"
            )

            weights, metrics = run_client(

                client_id=client_id,

                dataset_type=DATASET_TYPE,

                partition_type=PARTITION_TYPE,

                global_weights=global_weights
            )

            local_weights.append(weights)

            round_acc.append(
                metrics["accuracy"]
            )

            round_loss.append(
                metrics["final_loss"]
            )

        # ---------------------------------------------
        # AGGREGATION
        # ---------------------------------------------

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

        final_client_accuracies = round_acc

        logger.info(
            f"Round Accuracy : "
            f"{round_global_acc:.4f}"
        )

    # ==================================================
    # GLOBAL EVALUATION
    # ==================================================

    logger.info(
        "Evaluating Global Model"
    )

    all_X = []
    all_y = []

    for client_id in range(
            1,
            NUM_CLIENTS + 1
    ):

        client_file = (
            Path("data")
            / "partitions"
            / f"{PARTITION_TYPE}_{DATASET_TYPE}"
            / f"client_{client_id}.npz"
        )

        data = np.load(client_file)

        all_X.append(data["X"])
        all_y.append(data["y"])

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
        TensorDataset(X, y),
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

    # ==================================================
    # METRICS
    # ==================================================

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

    training_time = (
        time.time() - start_time
    )

    accuracy_std = np.std(
        final_client_accuracies
    )

    accuracy_gap = (
            max(final_client_accuracies)
            -
            min(final_client_accuracies)
    )

    logger.info(
        f"Global Accuracy : "
        f"{global_accuracy:.4f}"
    )

    # ==================================================
    # SAVE MODEL
    # ==================================================

    torch.save(
        global_model.state_dict(),
        save_dir / "global_model.pth"
    )

    # ==================================================
    # HISTORY
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
            save_dir / "history.json",
            "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # ==================================================
    # METRICS
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

        "training_time_sec":
            float(training_time),

        "client_accuracy_std":
            float(accuracy_std),

        "accuracy_gap":
            float(accuracy_gap),

        "confusion_matrix":
            global_cm.tolist()
    }

    with open(
            save_dir / "metrics.json",
            "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    disp = ConfusionMatrixDisplay(
        confusion_matrix=global_cm
    )

    disp.plot()

    plt.savefig(
        save_dir
        / "global_confusion_matrix.png"
    )

    plt.close()

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
        save_dir / "accuracy_curve.png"
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
        save_dir / "loss_curve.png"
    )

    plt.close()

    # ==================================================
    # CLIENT ACCURACY
    # ==================================================

    plt.figure()

    plt.bar(
        [
            f"C{i}"
            for i in range(
                1,
                NUM_CLIENTS + 1
            )
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
        save_dir
        / "client_accuracy.png"
    )

    plt.close()

    # ==================================================
    # SUMMARY CSV
    # ==================================================

    summary_file = (
        Path("results")
        / "federated"
        / "fedavg_summary.csv"
    )

    summary = pd.DataFrame([{

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
            global_f1,

        "training_time":
            training_time

    }])

    if summary_file.exists():

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

    logger.info(
        "FedAvg Completed"
    )


if __name__ == "__main__":

    run_fedavg()