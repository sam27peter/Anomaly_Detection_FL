import numpy as np

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


def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
) -> dict:
    """
    Compute anomaly detection metrics.
    """

    metrics = {}

    metrics["accuracy"] = float(
        accuracy_score(
            y_true,
            y_pred
        )
    )

    metrics["precision"] = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    metrics["recall"] = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    metrics["f1_score"] = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    metrics["confusion_matrix"] = (
        cm.tolist()
    )

    tn, fp, fn, tp = cm.ravel()

    metrics["false_positive_rate"] = float(
        fp / (fp + tn)
    )

    metrics["false_negative_rate"] = float(
        fn / (fn + tp)
    )

    if y_prob is not None:

        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_prob
            )
        )

        precision_curve, recall_curve, _ = (
            precision_recall_curve(
                y_true,
                y_prob
            )
        )

        metrics["pr_auc"] = float(
            auc(
                recall_curve,
                precision_curve
            )
        )

    return metrics