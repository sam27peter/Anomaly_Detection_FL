import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    precision_recall_curve,
    auc
)

from sklearn.model_selection import train_test_split

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from config.federated_config import (
    LOCAL_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE
)

from models.model_selector import get_model
from utils.logger import logger


# ==================================================
# DEVICE CONFIGURATION
# ==================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# CLIENT TRAINING FUNCTION
# ==================================================

def run_client(
        client_id: int,
        dataset_type: str,
        partition_type: str,
        global_weights: dict | None = None,
        mu: float = 0.01
) -> tuple:
    """
    Train a local federated client.

    Parameters
    ----------
    client_id : int
        Client identifier.

    dataset_type : str
        Dataset type (SMAP/MSL).

    partition_type : str
        IID or non_iid.

    global_weights : dict, optional
        Global model weights from server.

    mu : float, optional
        FedProx regularization coefficient.

    Returns
    -------
    tuple
        (model_weights, metrics)
    """

    logger.info("=" * 60)
    logger.info(f"CLIENT {client_id}")
    logger.info("=" * 60)

    logger.info(f"Dataset Type : {dataset_type}")
    logger.info(f"Partition    : {partition_type}")

    # ==================================================
    # LOAD DATA
    # ==================================================

    client_file = (
        Path("data")
        / "partitions"
        / f"{partition_type}_{dataset_type}"
        / f"client_{client_id}.npz"
    )

    if not client_file.exists():
        raise FileNotFoundError(
            f"Client dataset not found: {client_file}"
        )

    data = np.load(client_file)

    X = data["X"]
    y = data["y"]

    logger.info(f"X Shape : {X.shape}")
    logger.info(f"y Shape : {y.shape}")

    # ==================================================
    # TRAIN TEST SPLIT
    # ==================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ==================================================
    # TENSORS
    # ==================================================

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    )

    # ==================================================
    # DATALOADERS
    # ==================================================

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE
    )

    # ==================================================
    # MODEL
    # ==================================================

    num_features = X.shape[2]

    model = get_model(
        "cnn",
        num_features
    ).to(DEVICE)

    # ==================================================
    # LOAD GLOBAL MODEL
    # ==================================================

    global_model = None

    if global_weights is not None:

        logger.info("Loading Global Weights")

        model.load_state_dict(global_weights)

        global_model = copy.deepcopy(model)

        for param in global_model.parameters():
            param.requires_grad = False

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ==================================================
    # SAVE DIRECTORY
    # ==================================================

    save_dir = (
        Path("results")
        / "federated"
        / "local_clients"
        / f"{partition_type}_{dataset_type}"
        / f"client_{client_id}"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==================================================
    # TRAINING
    # ==================================================

    logger.info("Training Started")

    start_time = time.time()

    loss_history = []

    for epoch in range(LOCAL_EPOCHS):

        model.train()

        epoch_loss = 0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(DEVICE)

            batch_y = (
                batch_y
                .unsqueeze(1)
                .to(DEVICE)
            )

            optimizer.zero_grad()

            outputs = model(batch_x)

            classification_loss = criterion(
                outputs,
                batch_y
            )

            loss = classification_loss

            # ==================================================
            # FEDPROX REGULARIZATION
            # ==================================================

            if mu > 0 and global_model is not None:

                proximal_term = 0.0

                for local_param, global_param in zip(
                        model.parameters(),
                        global_model.parameters()
                ):

                    proximal_term += (
                        torch.norm(
                            local_param - global_param,
                            p=2
                        ) ** 2
                    )

                loss += (
                    mu / 2
                ) * proximal_term

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = (
            epoch_loss / len(train_loader)
        )

        loss_history.append(float(avg_loss))

        logger.info(
            f"Epoch [{epoch + 1}/{LOCAL_EPOCHS}] "
            f"Loss: {avg_loss:.4f}"
        )

    training_time = (
        time.time() - start_time
    )

    # ==================================================
    # LOSS CURVE
    # ==================================================

    plt.figure()

    plt.plot(
        loss_history,
        marker="o"
    )

    plt.title(
        f"Client {client_id} Loss Curve"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid()

    plt.savefig(
        save_dir / "loss_curve.png"
    )

    plt.close()

    # ==================================================
    # EVALUATION
    # ==================================================

    logger.info("Evaluating Client")

    model.eval()

    predictions = []
    probabilities = []
    ground_truth = []

    with torch.no_grad():

        for batch_x, batch_y in test_loader:

            batch_x = batch_x.to(DEVICE)

            outputs = model(batch_x)

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

    predictions = np.array(predictions).flatten()
    probabilities = np.array(probabilities).flatten()
    ground_truth = np.array(ground_truth).flatten()

    # ==================================================
    # METRICS
    # ==================================================

    accuracy = accuracy_score(
        ground_truth,
        predictions
    )

    precision = precision_score(
        ground_truth,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        ground_truth,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        ground_truth,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        ground_truth,
        probabilities
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            ground_truth,
            probabilities
        )
    )

    pr_auc = auc(
        recall_curve,
        precision_curve
    )

    cm = confusion_matrix(
        ground_truth,
        predictions
    )

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.savefig(
        save_dir / "confusion_matrix.png"
    )

    plt.close()

    logger.info(
        f"Accuracy : {accuracy:.4f}"
    )

    logger.info(
        f"Precision : {precision:.4f}"
    )

    logger.info(
        f"Recall : {recall:.4f}"
    )

    logger.info(
        f"F1 Score : {f1:.4f}"
    )

    # ==================================================
    # SAVE MODEL
    # ==================================================

    torch.save(
        model.state_dict(),
        save_dir / "local_model.pth"
    )

    # ==================================================
    # SAVE METRICS
    # ==================================================

    metrics = {

        "client_id": client_id,

        "dataset_type": dataset_type,

        "partition_type": partition_type,

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "roc_auc": float(roc_auc),

        "pr_auc": float(pr_auc),

        "final_loss": float(
            loss_history[-1]
        ),

        "training_time_sec": float(
            training_time
        ),

        "confusion_matrix":
            cm.tolist(),

        "loss_history":
            loss_history
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

    logger.info(
        "Client Training Complete"
    )

    return (
        model.state_dict(),
        metrics
    )


# ==================================================
# STANDALONE TEST
# ==================================================

if __name__ == "__main__":

    run_client(
        client_id=1,
        dataset_type="SMAP",
        partition_type="iid"
    )