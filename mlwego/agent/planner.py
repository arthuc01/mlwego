"""Planner that derives baseline plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mlwego.evaluation.infer_task import TaskInfo, infer_task


@dataclass
class Plan:
    task_info: TaskInfo
    baseline: str


def build_plan(data_dir: str, task_description: str) -> Plan:
    info = infer_task(Path(data_dir), task_description)
    baseline = "RandomForest with numeric/categorical preprocessing"
    return Plan(task_info=info, baseline=baseline)
