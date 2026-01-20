from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray, labels_by_id: list[str]) -> dict[str, Any]:
    cm = np.asarray(cm, dtype=np.int64)
    c = int(cm.shape[0])
    supports = cm.sum(axis=1)
    total = int(cm.sum())
    correct = int(np.trace(cm))

    per_class: dict[str, Any] = {}
    f1s: list[float] = []
    f1s_present: list[float] = []
    recalls_present: list[float] = []

    for i in range(c):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = int(supports[i])

        f1s.append(float(f1))
        if support > 0:
            f1s_present.append(float(f1))
            recalls_present.append(float(recall))

        per_class[labels_by_id[i]] = {
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    accuracy = correct / total if total > 0 else 0.0
    macro_f1_all = float(np.mean(f1s)) if f1s else 0.0
    macro_f1_present = float(np.mean(f1s_present)) if f1s_present else 0.0
    balanced_acc_present = float(np.mean(recalls_present)) if recalls_present else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": float(accuracy),
        "macro_f1_all": macro_f1_all,
        "macro_f1_present": macro_f1_present,
        "balanced_accuracy_present": balanced_acc_present,
        "per_class": per_class,
    }

