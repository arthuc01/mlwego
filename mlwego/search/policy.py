"""Search policy settings."""

from dataclasses import dataclass


@dataclass
class SearchPolicy:
    budget: int = 10
    branch_factor: int = 2
    early_stop_rounds: int = 3
