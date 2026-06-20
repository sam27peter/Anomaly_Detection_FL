import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from models.model_selector import get_model


LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def run_client(
    client_id,
    dataset_type,
    partition_type
):
    """
    Entire client logic goes inside this function.
    """

    print("\n" + "=" * 60)
    print(f"CLIENT {client_id}")
    print("=" * 60)

    print(f"Dataset Type : {dataset_type}")
    print(f"Partition    : {partition_type}")

    client_file = (
        f"data/partitions/"
        f"{partition_type}_{dataset_type}/"
        f"client_{client_id}.npz"
    )

    data = np.load(client_file)

    X = data["X"]
    y = data["y"]

    print("\nLoaded Client Data")
    print(f"X Shape : {X.shape}")
    print(f"y Shape : {y.shape}")

    # ==================================================
    # TRAIN / TEST SPLIT
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
        TensorDataset(
            X_train,
            y_train
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(
            X_test,
            y_test
        ),
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

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ==================================================
    # SAVE DIRECTORY
    # ==================================================

    save_dir = (
        f"results/federated/local_clients/"
        f"{partition_type}_{dataset_type}/"
        f"client_{client_id}"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # ==================================================
    # TRAINING
    # ==================================================

    print("\nTraining...\n")

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

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = (
            epoch_loss /
            len(train_loader)
        )

        loss_history.append(
            float(avg_loss)
        )

        print(
            f"Epoch [{epoch+1}/{LOCAL_EPOCHS}] "
            f"Loss: {avg_loss:.4f}"
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
        os.path.join(
            save_dir,
            "loss_curve.png"
        )
    )

    plt.close()

    # ==================================================
    # EVALUATION
    # ==================================================

    print("\nEvaluating...\n")

    model.eval()

    predictions = []
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

            ground_truth.extend(
                batch_y.numpy()
            )

    predictions = np.array(
        predictions
    ).flatten()

    ground_truth = np.array(
        ground_truth
    ).flatten()

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

    cm = confusion_matrix(
        ground_truth,
        predictions
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.savefig(
        os.path.join(
            save_dir,
            "confusion_matrix.png"
        )
    )

    plt.close()

    print("=" * 60)
    print("LOCAL CLIENT RESULTS")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    # ==================================================
    # SAVE MODEL
    # ==================================================

    torch.save(
        model.state_dict(),
        os.path.join(
            save_dir,
            "local_model.pth"
        )
    )

    metrics = {

        "client_id": client_id,
        "dataset_type": dataset_type,
        "partition_type": partition_type,

        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),

        "final_loss": float(
            loss_history[-1]
        ),

        "confusion_matrix":
            cm.tolist(),

        "loss_history":
            loss_history
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

    print("\nClient Training Complete.")

    return metrics


if __name__ == "__main__":

    run_client(
        client_id=1,
        dataset_type="25",
        partition_type="iid"
    )