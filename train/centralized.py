import sys
import json
import os

import numpy as np

import torch
import torch.nn as nn

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
    confusion_matrix
)

from models.model_selector import (
    get_model
)

# ==================================================
# CONFIG
# ==================================================

DATASET_TYPE = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "25"
)

BATCH_SIZE = 64

EPOCHS = 20

LEARNING_RATE = 0.001

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==================================================
# CREATE OUTPUT FOLDERS
# ==================================================

os.makedirs(
    "results/single_machine/models",
    exist_ok=True
)

os.makedirs(
    "results/single_machine/metrics",
    exist_ok=True
)

os.makedirs(
    "results/single_machine/history",
    exist_ok=True
)

# ==================================================
# LOAD DATA
# ==================================================

dataset_dir = (
    f"data/processed/dataset_{DATASET_TYPE}"
)

X = np.load(
    os.path.join(
        dataset_dir,
        f"X_{DATASET_TYPE}.npy"
    )
)

y = np.load(
    os.path.join(
        dataset_dir,
        f"y_{DATASET_TYPE}.npy"
    )
)

print(
    f"\nDataset Shape: {X.shape}"
)

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
# CONVERT TO TENSORS
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
# TRAINING
# ==================================================

print("\nStarting Training...\n")

loss_history = []

accuracy_history = []

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    correct = 0

    total = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(
            DEVICE
        )

        batch_y = (
            batch_y
            .unsqueeze(1)
            .to(DEVICE)
        )

        optimizer.zero_grad()

        outputs = model(
            batch_x
        )

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        preds = (
            outputs > 0.5
        ).float()

        correct += (
            preds == batch_y
        ).sum().item()

        total += (
            batch_y.size(0)
        )

    avg_loss = (
        epoch_loss /
        len(train_loader)
    )

    epoch_accuracy = (
        correct / total
    )

    loss_history.append(
        float(avg_loss)
    )

    accuracy_history.append(
        float(epoch_accuracy)
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.4f}"
    )

# ==================================================
# EVALUATION
# ==================================================

print("\nEvaluating...\n")

model.eval()

predictions = []

ground_truth = []

with torch.no_grad():

    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(
            DEVICE
        )

        outputs = model(
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

# ==================================================
# PRINT RESULTS
# ==================================================

print("\n==========================")
print("CENTRALIZED CNN RESULTS")
print("==========================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print("\nConfusion Matrix")

print(cm)

# ==================================================
# SAVE MODEL
# ==================================================

model_path = (
    f"results/single_machine/models/"
    f"cnn_{DATASET_TYPE}.pth"
)

torch.save(
    model.state_dict(),
    model_path
)

# ==================================================
# SAVE HISTORY
# ==================================================

history = {

    "dataset": DATASET_TYPE,

    "loss": loss_history,

    "accuracy": accuracy_history
}

history_path = (
    f"results/single_machine/history/"
    f"cnn_{DATASET_TYPE}_history.json"
)

with open(
    history_path,
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

    "dataset": DATASET_TYPE,

    "accuracy": float(
        accuracy
    ),

    "precision": float(
        precision
    ),

    "recall": float(
        recall
    ),

    "f1_score": float(
        f1
    ),

    "confusion_matrix":
        cm.tolist()
}

metrics_path = (
    f"results/single_machine/metrics/"
    f"cnn_{DATASET_TYPE}.json"
)

with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# ==================================================
# SUMMARY
# ==================================================

print("\nFiles Saved")

print(
    f"Model   : {model_path}"
)

print(
    f"Metrics : {metrics_path}"
)

print(
    f"History : {history_path}"
)

print(
    "\nCentralized training completed successfully."
)