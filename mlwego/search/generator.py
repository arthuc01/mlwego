"""Generate candidate edits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CandidateEdit:
    description: str
    updates: dict


def baseline_candidates() -> List[CandidateEdit]:
    return [
        CandidateEdit(
            description="Increase number of trees for stability",
            updates={"model_params": {"n_estimators": 400, "random_state": 42}},
        ),
        CandidateEdit(
            description="Reduce tree depth to prevent overfitting",
            updates={"model_params": {"n_estimators": 200, "max_depth": 8, "random_state": 42}},
        ),
    ]
