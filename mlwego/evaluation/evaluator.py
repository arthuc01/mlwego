"""Evaluation loop for candidate solutions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mlwego.execution.sandbox import ExecutionResult, run_python
from mlwego.execution.timeouts import PREDICT_TIMEOUT, TRAIN_TIMEOUT


@dataclass
class EvalResult:
    score: float
    score_std: float
    metric: str
    train_result: ExecutionResult


def evaluate_solution(run_dir: Path, timeout: int = TRAIN_TIMEOUT) -> EvalResult:
    src_dir = run_dir / "src"
    train_script = src_dir / "train.py"
    result = run_python(train_script, cwd=src_dir, timeout=timeout)
    if result.exit_code != 0:
        raise RuntimeError(
            "Training failed with exit code "
            f"{result.exit_code}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    metrics_path = run_dir / "artifacts" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "metrics.json not found after training.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return EvalResult(
        score=float(metrics["score"]),
        score_std=float(metrics.get("score_std", 0.0)),
        metric=str(metrics["metric"]),
        train_result=result,
    )


def run_predict(run_dir: Path, timeout: int = PREDICT_TIMEOUT) -> ExecutionResult:
    src_dir = run_dir / "src"
    predict_script = src_dir / "predict.py"
    return run_python(predict_script, cwd=src_dir, timeout=timeout)


def validate_submission(run_dir: Path) -> Optional[str]:
    artifacts = run_dir / "artifacts"
    submission_path = artifacts / "submission.csv"
    data_dir = run_dir / "data"
    if not submission_path.exists():
        return "submission.csv not found"
    sample_path = data_dir / "sample_submission.csv"
    if not sample_path.exists():
        return None
    sample_cols = sample_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    submission_cols = submission_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    if sample_cols != submission_cols:
        return f"Submission columns {submission_cols} do not match sample {sample_cols}"
    return None
