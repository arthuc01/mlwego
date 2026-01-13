from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def _handle_remove_readonly(func, path, exc_info) -> None:
    del exc_info
    os.chmod(path, 0o700)
    func(path)


def _safe_rmtree(path: Path, retries: int = 3, delay: float = 0.2) -> None:
    for attempt in range(retries):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onerror=_handle_remove_readonly)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def main() -> None:
    workspace = Path("demo_run")
    if workspace.exists():
        _safe_rmtree(workspace)
    data_dir = workspace / "data"
    data_dir.mkdir(parents=True)
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.drop(columns=["target"]).to_csv(data_dir / "test.csv", index=False)
    sample = pd.DataFrame({"id": range(len(test_df)), "target": 0})
    sample.to_csv(data_dir / "sample_submission.csv", index=False)
    task_path = workspace / "task.txt"
    task_path.write_text("Predict target; metric is accuracy.", encoding="utf-8")
    subprocess.run([
        "python",
        "-m",
        "mlwego.ui.cli",
        "init",
        "--task",
        str(task_path),
        "--data",
        str(data_dir),
        "--out",
        str(workspace),
    ], check=True)
    subprocess.run([
        "python",
        "-m",
        "mlwego.ui.cli",
        "run",
        "--out",
        str(workspace),
        "--budget",
        "1",
    ], check=True)
    best = subprocess.check_output([
        "python",
        "-m",
        "mlwego.ui.cli",
        "best",
        "--out",
        str(workspace),
    ])
    print(best.decode("utf-8"))
    report = {"best": json.loads(best)}
    (workspace / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
