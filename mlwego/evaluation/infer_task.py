"""Task inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class TaskInfo:
    target: str
    metric: str
    task_type: str
    id_column: Optional[str]
    target_columns: List[str]


def infer_task(data_dir: Path, description: str = "") -> TaskInfo:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_path = data_dir / "sample_submission.csv"
    target = _infer_target(train, test)
    task_type = "classification" if train[target].nunique() <= 20 else "regression"
    metric = _infer_metric(description, task_type)
    id_column, target_columns = _infer_submission(sample_path)
    return TaskInfo(
        target=target,
        metric=metric,
        task_type=task_type,
        id_column=id_column,
        target_columns=target_columns,
    )


def _infer_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    candidates = [c for c in train.columns if c not in test.columns]
    if candidates:
        return candidates[0]
    return train.columns[-1]


def _infer_metric(description: str, task_type: str) -> str:
    lowered = description.lower()
    if "auc" in lowered:
        return "roc_auc"
    if "logloss" in lowered or "log loss" in lowered:
        return "log_loss"
    if "rmse" in lowered:
        return "rmse"
    if "mae" in lowered:
        return "mae"
    return "accuracy" if task_type == "classification" else "rmse"


def _infer_submission(sample_path: Path) -> tuple[Optional[str], List[str]]:
    if not sample_path.exists():
        return None, []
    submission = pd.read_csv(sample_path)
    id_column = submission.columns[0] if submission.columns.size > 0 else None
    target_columns = [c for c in submission.columns if c != id_column]
    return id_column, target_columns
