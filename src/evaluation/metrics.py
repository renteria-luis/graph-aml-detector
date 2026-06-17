# src/evaluation/metrics.py
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


@dataclass
class Metrics:
    pr_auc: float
    precision: float
    recall: float
    f1: float


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Metrics:
    pr_auc = average_precision_score(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    return Metrics(pr_auc=float(pr_auc), precision=precision, recall=recall, f1=f1)