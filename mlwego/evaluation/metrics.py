"""Metrics helpers."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def metric_registry() -> Dict[str, Tuple[MetricFn, str]]:
    return {
        "accuracy": (lambda y_true, y_pred: accuracy_score(y_true, y_pred), "max"),
        "roc_auc": (lambda y_true, y_pred: roc_auc_score(y_true, y_pred), "max"),
        "log_loss": (lambda y_true, y_pred: -log_loss(y_true, y_pred), "max"),
        "rmse": (lambda y_true, y_pred: -mean_squared_error(y_true, y_pred, squared=False), "max"),
        "mae": (lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred), "max"),
    }


def get_metric(name: str) -> Tuple[MetricFn, str]:
    registry = metric_registry()
    if name not in registry:
        raise KeyError(f"Unknown metric: {name}")
    return registry[name]
