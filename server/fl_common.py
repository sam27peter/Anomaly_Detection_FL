import copy
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models.model_selector import get_model


@dataclass
class FLConfig:
    datasets: Tuple[str, ...] = ("25", "55")
    partitions: Tuple[str, ...] = ("iid", "non_iid")
    client_ids: Tuple[int, ...] = (1, 2, 3, 4, 5)
    global_rounds: int = 2
    local_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    fedprox_mu: float = 0.01
    random_seed: int = 42
    test_size: float = 0.2


def get_default_config() -> FLConfig:
    return FLConfig()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_client_npz(dataset_type: str, partition_type: str, client_id: int) -> Tuple[np.ndarray, np.ndarray, str]:
    client_path = (
        f"data/partitions/{partition_type}_{dataset_type}/"
        f"client_{client_id}.npz"
    )

    if not os.path.exists(client_path):
        raise FileNotFoundError(
            f"Expected client partition not found: {client_path}. "
            "Run client/partitioner.py first to generate partitions."
        )

    data = np.load(client_path)
    return data["X"], data["y"], client_path


def _prepare_client_bundle(
    dataset_type: str,
    partition_type: str,
    client_id: int,
    config: FLConfig,
) -> Dict:
    X, y, path = _load_client_npz(dataset_type, partition_type, client_id)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_seed,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_seed,
            shuffle=True,
        )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=config.batch_size,
        shuffle=False,
    )

    normal_count = int(np.sum(y == 0))
    anomaly_count = int(np.sum(y == 1))

    return {
        "client_id": client_id,
        "source_path": path,
        "num_features": int(X.shape[2]),
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "total_samples": int(len(y)),
        "normal_samples": normal_count,
        "anomaly_samples": anomaly_count,
        "anomaly_ratio": float(anomaly_count / len(y)) if len(y) else 0.0,
        "X_test": X_test_t,
        "y_test": y_test_t,
    }


def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    model.eval()
    predictions: List[int] = []
    truth: List[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = (outputs > 0.5).int().cpu().numpy().flatten().tolist()
            predictions.extend(preds)
            truth.extend(batch_y.numpy().astype(int).flatten().tolist())

    if not truth:
        empty_cm = np.array([[0, 0], [0, 0]])
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": empty_cm.tolist(),
        }

    cm = confusion_matrix(truth, predictions, labels=[0, 1])

    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1_score": float(f1_score(truth, predictions, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def _build_global_test_loader(client_bundles: List[Dict], batch_size: int) -> DataLoader:
    X_all = torch.cat([bundle["X_test"] for bundle in client_bundles], dim=0)
    y_all = torch.cat([bundle["y_test"] for bundle in client_bundles], dim=0)
    return DataLoader(TensorDataset(X_all, y_all), batch_size=batch_size, shuffle=False)


def _state_dict_clone(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _aggregate_fedavg(states: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    total_samples = sum(n for _, n in states)
    if total_samples == 0:
        raise ValueError("Cannot aggregate with zero total samples.")

    aggregated = {}
    for key in states[0][0].keys():
        weighted_sum = sum(state[key] * (count / total_samples) for state, count in states)
        aggregated[key] = weighted_sum
    return aggregated


def _train_local(
    model: nn.Module,
    train_loader: DataLoader,
    local_epochs: int,
    lr: float,
    device: str,
    prox_mu: float = 0.0,
    global_params: Dict[str, torch.Tensor] = None,
) -> Dict:
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses: List[float] = []
    epoch_accuracies: List[float] = []

    global_tensor_params = None
    if prox_mu > 0 and global_params is not None:
        global_tensor_params = {
            k: v.to(device) for k, v in global_params.items()
        }

    for _ in range(local_epochs):
        cumulative_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            if global_tensor_params is not None:
                prox_term = torch.tensor(0.0, device=device)
                for name, param in model.named_parameters():
                    prox_term = prox_term + torch.sum((param - global_tensor_params[name]) ** 2)
                loss = loss + (prox_mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = cumulative_loss / max(1, len(train_loader))
        avg_acc = correct / max(1, total)
        epoch_losses.append(float(avg_loss))
        epoch_accuracies.append(float(avg_acc))

    return {
        "loss_history": epoch_losses,
        "accuracy_history": epoch_accuracies,
        "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "final_accuracy": epoch_accuracies[-1] if epoch_accuracies else 0.0,
    }


def _save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def _save_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_global_confusion(cm: List[List[int]], path: str, title: str) -> None:
    arr = np.array(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=arr)
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_accuracy_vs_round(round_rows: List[Dict], path: str, title: str) -> None:
    rounds = [row["round"] for row in round_rows]
    accs = [row["global_accuracy"] for row in round_rows]

    plt.figure()
    plt.plot(rounds, accs, marker="o")
    plt.xticks(rounds)
    plt.xlabel("Round")
    plt.ylabel("Global Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_per_client_accuracy(per_client_rows: List[Dict], path: str, title: str) -> None:
    client_labels = [f"Client {row['client_id']}" for row in per_client_rows]
    accs = [row["accuracy"] for row in per_client_rows]

    plt.figure(figsize=(8, 4))
    plt.bar(client_labels, accs)
    plt.ylim(0, 1)
    plt.xlabel("Client")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_circular_distribution(client_bundles: List[Dict], path: str, title: str) -> None:
    fig, axes = plt.subplots(1, len(client_bundles), figsize=(3.5 * len(client_bundles), 3.5))
    if len(client_bundles) == 1:
        axes = [axes]

    for idx, bundle in enumerate(client_bundles):
        normal = bundle["normal_samples"]
        anomaly = bundle["anomaly_samples"]
        axes[idx].pie(
            [normal, anomaly],
            labels=["Normal", "Anomaly"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": 0.5},
        )
        axes[idx].set_title(f"Client {bundle['client_id']}")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_federated_experiment(
    algorithm: str,
    dataset_type: str,
    partition_type: str,
    config: FLConfig = None,
) -> Dict:
    cfg = config or get_default_config()
    set_global_seed(cfg.random_seed)

    if partition_type not in cfg.partitions:
        raise ValueError(f"Unknown partition type: {partition_type}")
    if dataset_type not in cfg.datasets:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    algo = algorithm.lower()
    if algo not in {"fedavg", "fedprox"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    root = f"results/federated/{algo}/{dataset_type}/{partition_type}"
    os.makedirs(root, exist_ok=True)

    client_bundles = [
        _prepare_client_bundle(dataset_type, partition_type, client_id, cfg)
        for client_id in cfg.client_ids
    ]

    # Validation check: all client feature dimensions should match.
    feature_set = {bundle["num_features"] for bundle in client_bundles}
    if len(feature_set) != 1:
        raise ValueError(f"Inconsistent feature dimensions across clients: {feature_set}")

    num_features = next(iter(feature_set))
    device = _device()

    global_model = get_model("cnn", num_features).to(device)

    global_test_loader = _build_global_test_loader(client_bundles, cfg.batch_size)

    distribution_rows = []
    for bundle in client_bundles:
        distribution_rows.append(
            {
                "client_id": bundle["client_id"],
                "total_samples": bundle["total_samples"],
                "normal_samples": bundle["normal_samples"],
                "anomaly_samples": bundle["anomaly_samples"],
                "anomaly_ratio": bundle["anomaly_ratio"],
            }
        )

    _save_csv(
        f"{root}/distribution/distribution.csv",
        distribution_rows,
        ["client_id", "total_samples", "normal_samples", "anomaly_samples", "anomaly_ratio"],
    )
    _save_json(f"{root}/distribution/distribution.json", {"rows": distribution_rows})
    _plot_circular_distribution(
        client_bundles,
        f"{root}/distribution/circular_data_distribution.png",
        f"{algo.upper()} Circular Distribution - Dataset {dataset_type} ({partition_type})",
    )

    round_rows: List[Dict] = []
    client_round_rows: List[Dict] = []

    for round_idx in range(1, cfg.global_rounds + 1):
        local_states: List[Tuple[Dict[str, torch.Tensor], int]] = []

        # Keep a CPU copy for FedProx proximal term.
        current_global_state = _state_dict_clone(global_model)

        for bundle in client_bundles:
            local_model = get_model("cnn", num_features).to(device)
            local_model.load_state_dict(copy.deepcopy(current_global_state))

            local_train_metrics = _train_local(
                model=local_model,
                train_loader=bundle["train_loader"],
                local_epochs=cfg.local_epochs,
                lr=cfg.learning_rate,
                device=device,
                prox_mu=cfg.fedprox_mu if algo == "fedprox" else 0.0,
                global_params=current_global_state,
            )

            local_states.append((_state_dict_clone(local_model), bundle["train_samples"]))

            eval_local = _evaluate(local_model, bundle["test_loader"], device)
            client_round_rows.append(
                {
                    "round": round_idx,
                    "client_id": bundle["client_id"],
                    "algorithm": algo,
                    "dataset": dataset_type,
                    "partition": partition_type,
                    "local_final_loss": local_train_metrics["final_loss"],
                    "local_final_accuracy": local_train_metrics["final_accuracy"],
                    "accuracy": eval_local["accuracy"],
                    "precision": eval_local["precision"],
                    "recall": eval_local["recall"],
                    "f1_score": eval_local["f1_score"],
                }
            )

        aggregated = _aggregate_fedavg(local_states)
        global_model.load_state_dict(aggregated)

        global_eval = _evaluate(global_model, global_test_loader, device)

        round_rows.append(
            {
                "round": round_idx,
                "algorithm": algo,
                "dataset": dataset_type,
                "partition": partition_type,
                "global_accuracy": global_eval["accuracy"],
                "global_precision": global_eval["precision"],
                "global_recall": global_eval["recall"],
                "global_f1_score": global_eval["f1_score"],
                "global_confusion_matrix": global_eval["confusion_matrix"],
            }
        )

    final_global = round_rows[-1]

    final_per_client_rows = [
        {
            "client_id": bundle["client_id"],
            **_evaluate(global_model, bundle["test_loader"], device),
        }
        for bundle in client_bundles
    ]

    _save_csv(
        f"{root}/history/round_history.csv",
        [
            {
                k: v
                for k, v in row.items()
                if k != "global_confusion_matrix"
            }
            for row in round_rows
        ],
        [
            "round",
            "algorithm",
            "dataset",
            "partition",
            "global_accuracy",
            "global_precision",
            "global_recall",
            "global_f1_score",
        ],
    )
    _save_json(f"{root}/history/round_history.json", {"rounds": round_rows})

    _save_csv(
        f"{root}/clients/per_client_round_metrics.csv",
        client_round_rows,
        [
            "round",
            "client_id",
            "algorithm",
            "dataset",
            "partition",
            "local_final_loss",
            "local_final_accuracy",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ],
    )

    _save_csv(
        f"{root}/clients/per_client_accuracy_final.csv",
        [
            {
                "client_id": row["client_id"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1_score"],
            }
            for row in final_per_client_rows
        ],
        ["client_id", "accuracy", "precision", "recall", "f1_score"],
    )

    _save_json(
        f"{root}/global/global_metrics.json",
        {
            "algorithm": algo,
            "dataset": dataset_type,
            "partition": partition_type,
            "config": asdict(cfg),
            "final_global_accuracy": final_global["global_accuracy"],
            "final_global_precision": final_global["global_precision"],
            "final_global_recall": final_global["global_recall"],
            "final_global_f1_score": final_global["global_f1_score"],
            "global_confusion_matrix": final_global["global_confusion_matrix"],
            "global_acc": final_global["global_accuracy"],
            "per_client_acc": {
                str(row["client_id"]): row["accuracy"] for row in final_per_client_rows
            },
            "history": round_rows,
        },
    )

    _plot_global_confusion(
        final_global["global_confusion_matrix"],
        f"{root}/plots/global_confusion_matrix.png",
        f"{algo.upper()} Global Confusion - Dataset {dataset_type} ({partition_type})",
    )
    _plot_accuracy_vs_round(
        round_rows,
        f"{root}/plots/global_accuracy_vs_round.png",
        f"{algo.upper()} Global Accuracy vs Round - Dataset {dataset_type} ({partition_type})",
    )
    _plot_per_client_accuracy(
        final_per_client_rows,
        f"{root}/plots/per_client_accuracy_final.png",
        f"{algo.upper()} Per-Client Accuracy (Final) - Dataset {dataset_type} ({partition_type})",
    )

    return {
        "algorithm": algo,
        "dataset": dataset_type,
        "partition": partition_type,
        "final_global_accuracy": final_global["global_accuracy"],
        "final_global_precision": final_global["global_precision"],
        "final_global_recall": final_global["global_recall"],
        "final_global_f1_score": final_global["global_f1_score"],
        "global_confusion_matrix": final_global["global_confusion_matrix"],
        "output_root": root,
    }
