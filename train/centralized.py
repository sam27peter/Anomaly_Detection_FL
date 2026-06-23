import sys
import json
import time
from pathlib import Path

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
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

from models.model_selector import get_model
from utils.logger import logger

# ==================================================
# CONFIG
# ==================================================

DATASET_TYPE = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "SMAP"
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
# OUTPUT DIRECTORIES
# ==================================================

BASE_DIR = (
    Path("results")
    / "single_machine"
)

MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"
HISTORY_DIR = BASE_DIR / "history"

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==================================================
# LOAD DATA
# ==================================================

dataset_dir = (
    Path("data")
    / "processed"
    / DATASET_TYPE
)

X = np.load(
    dataset_dir / f"X_{DATASET_TYPE}.npy"
)

y = np.load(
    dataset_dir / f"y_{DATASET_TYPE}.npy"
)

logger.info(
    f"Dataset Shape: {X.shape}"
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
# TRAINING
# ==================================================

logger.info("Starting Centralized Training")

start_time = time.time()

loss_history = []
accuracy_history = []

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0
    correct = 0
    total = 0

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

        preds = (
            outputs > 0.5
        ).float()

        correct += (
            preds == batch_y
        ).sum().item()

        total += batch_y.size(0)

    avg_loss = (
        epoch_loss / len(train_loader)
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

    logger.info(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.4f}"
    )

training_time = (
    time.time() - start_time
)

# ==================================================
# EVALUATION
# ==================================================

logger.info("Evaluating Model")

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

predictions = np.array(
    predictions
).flatten()

probabilities = np.array(
    probabilities
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

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

# ==================================================
# LOG RESULTS
# ==================================================

logger.info("=" * 50)
logger.info("CENTRALIZED CNN RESULTS")
logger.info("=" * 50)

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

logger.info(
    f"ROC-AUC : {roc_auc:.4f}"
)

logger.info(
    f"PR-AUC : {pr_auc:.4f}"
)

logger.info(
    f"Training Time : {training_time:.2f} sec"
)

# ==================================================
# SAVE MODEL
# ==================================================

model_path = (
    MODELS_DIR
    / f"cnn_{DATASET_TYPE}.pth"
)

torch.save(
    model.state_dict(),
    model_path
)

# ==================================================
# SAVE HISTORY
# ==================================================

history = {

    "dataset":
        DATASET_TYPE,

    "loss":
        loss_history,

    "accuracy":
        accuracy_history
}

history_path = (
    HISTORY_DIR
    / f"cnn_{DATASET_TYPE}_history.json"
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

    "dataset":
        DATASET_TYPE,

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "training_time_sec":
        float(training_time),

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1_score":
        float(f1),

    "roc_auc":
        float(roc_auc),

    "pr_auc":
        float(pr_auc),

    "false_positive_rate":
        float(fpr),

    "false_negative_rate":
        float(fnr),

    "confusion_matrix":
        cm.tolist()
}

metrics_path = (
    METRICS_DIR
    / f"cnn_{DATASET_TYPE}.json"
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

logger.info("Files Saved")

logger.info(
    f"Model : {model_path}"
)

logger.info(
    f"Metrics : {metrics_path}"
)

logger.info(
    f"History : {history_path}"
)

logger.info(
    "Centralized Training Completed Successfully"
)