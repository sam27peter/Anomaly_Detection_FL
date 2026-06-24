import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from client.fl_client import run_client

from models.model_selector import (
    get_model
)

from utils.logger import logger

from evaluation.metrics import (
    compute_metrics
)

from evaluation.fairness import (
    compute_fairness
)

from server.strategies import (
    federated_average,
    get_prox_mu
)

from config.federated_config import (
    NUM_CLIENTS,
    NUM_ROUNDS
)

# ==================================================
# ARGUMENTS
# ==================================================

ALGORITHM = (

    sys.argv[1]

    if len(sys.argv) > 1

    else "fedavg"

).lower()

DATASET_TYPE = (

    sys.argv[2]

    if len(sys.argv) > 2

    else "SMAP"

)

PARTITION_TYPE = (

    sys.argv[3]

    if len(sys.argv) > 3

    else "iid"

)

DEVICE = (

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

# ==================================================
# SERVER
# ==================================================


def run_server():

    logger.info(
        f"Starting {ALGORITHM.upper()}"
    )

    experiment_start = time.time()

    num_features = (

        25

        if DATASET_TYPE == "SMAP"

        else 55

    )

    global_model = get_model(

        "cnn",

        num_features

    ).to(DEVICE)

    prox_mu = get_prox_mu(
        ALGORITHM
    )

    global_acc_history = []
    global_loss_history = []

    final_client_accuracies = []

    save_dir = (

        Path("results")

        / "federated"

        / ALGORITHM

        / f"{PARTITION_TYPE}_{DATASET_TYPE}"

    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==================================================
    # COMMUNICATION ROUNDS
    # ==================================================

    for rnd in range(NUM_ROUNDS):

        logger.info(
            f"ROUND {rnd + 1}/{NUM_ROUNDS}"
        )

        local_weights = []

        round_acc = []
        round_loss = []

        global_weights = (
            global_model.state_dict()
        )

        # ---------------------------------------------

        for client_id in range(

                1,

                NUM_CLIENTS + 1

        ):

            weights, metrics = run_client(

                client_id=client_id,

                dataset_type=DATASET_TYPE,

                partition_type=PARTITION_TYPE,

                global_weights=global_weights,

                mu=prox_mu
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

        # ---------------------------------------------

        aggregated_weights = (

            federated_average(
                local_weights
            )

        )

        global_model.load_state_dict(
            aggregated_weights
        )

        global_acc_history.append(

            float(
                np.mean(
                    round_acc
                )
            )

        )

        global_loss_history.append(

            float(
                np.mean(
                    round_loss
                )
            )

        )

        final_client_accuracies = (
            round_acc
        )

    # ==================================================
    # GLOBAL EVALUATION
    # ==================================================

    all_X = []
    all_y = []

    for client_id in range(

            1,

            NUM_CLIENTS + 1

    ):

        file = (

            Path("data")

            / "partitions"

            / f"{PARTITION_TYPE}_{DATASET_TYPE}"

            / f"client_{client_id}.npz"

        )

        data = np.load(file)

        all_X.append(
            data["X"]
        )

        all_y.append(
            data["y"]
        )

    X = np.concatenate(
        all_X
    )

    y = np.concatenate(
        all_y
    )

    loader = DataLoader(

        TensorDataset(

            torch.tensor(
                X,
                dtype=torch.float32
            ),

            torch.tensor(
                y,
                dtype=torch.float32
            )

        ),

        batch_size=64,

        shuffle=False
    )

    global_model.eval()

    predictions = []
    probabilities = []
    ground_truth = []

    with torch.no_grad():

        for batch_x, batch_y in loader:

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

            probabilities.extend(
                outputs.cpu().numpy()
            )

            ground_truth.extend(
                batch_y.numpy()
            )

    predictions = np.array(
        predictions
    ).flatten()

    probabilities = np.array(
        probabilities
    ).flatten()

    ground_truth = np.array(
        ground_truth
    ).flatten()

    metrics = compute_metrics(

        ground_truth,

        predictions,

        probabilities

    )

    fairness_metrics = (

        compute_fairness(

            final_client_accuracies

        )

    )

    metrics.update(
        fairness_metrics
    )

    metrics.update({

        "algorithm":
            ALGORITHM,

        "dataset":
            DATASET_TYPE,

        "partition":
            PARTITION_TYPE,

        "num_clients":
            NUM_CLIENTS,

        "rounds":
            NUM_ROUNDS,

        "training_time_sec":

            float(
                time.time()
                -
                experiment_start
            )

    })

    # ==================================================
    # SAVE
    # ==================================================

    torch.save(

        global_model.state_dict(),

        save_dir
        / "global_model.pth"

    )

    with open(

            save_dir
            / "metrics.json",

            "w"

    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    history = {

        "accuracy":
            global_acc_history,

        "loss":
            global_loss_history,

        "client_accuracy":
            final_client_accuracies

    }

    with open(

            save_dir
            / "history.json",

            "w"

    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # ==================================================
    # PLOTS
    # ==================================================

    plt.figure()

    plt.plot(
        global_acc_history,
        marker="o"
    )

    plt.title(
        f"{ALGORITHM.upper()} Accuracy"
    )

    plt.xlabel("Round")
    plt.ylabel("Accuracy")

    plt.grid()

    plt.savefig(
        save_dir
        / "accuracy_curve.png"
    )

    plt.close()

    plt.figure()

    plt.plot(
        global_loss_history,
        marker="o"
    )

    plt.title(
        f"{ALGORITHM.upper()} Loss"
    )

    plt.xlabel("Round")
    plt.ylabel("Loss")

    plt.grid()

    plt.savefig(
        save_dir
        / "loss_curve.png"
    )

    plt.close()

    logger.info(
        f"{ALGORITHM.upper()} Completed"
    )


if __name__ == "__main__":

    run_server()