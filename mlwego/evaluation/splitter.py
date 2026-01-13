"""Cross-validation splitters."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit


def build_splitter(strategy: str, n_splits: int, seed: int) -> Any:
    if strategy == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if strategy == "group":
        return GroupKFold(n_splits=n_splits)
    if strategy == "time":
        return TimeSeriesSplit(n_splits=n_splits)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
