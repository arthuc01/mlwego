"""Workspace initialization."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from mlwego.workspace.file_ops import ensure_dir, write_text
from mlwego.workspace.snapshot import write_baseline_src


def init_workspace(task_path: str, data_path: str, out_dir: Optional[str] = None) -> Path:
    task_name = Path(task_path).stem
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root = Path(out_dir) if out_dir else Path("runs") / f"{timestamp}_{task_name}"
    ensure_dir(root)
    for name in ["data", "src", "artifacts", "logs"]:
        ensure_dir(root / name)
    task_text = Path(task_path).read_text(encoding="utf-8")
    task_meta = {"task": task_text, "task_path": str(Path(task_path).resolve())}
    write_text(root / "task.json", json.dumps(task_meta, indent=2))
    _link_or_copy_data(Path(data_path), root / "data")
    write_baseline_src(root / "src")
    write_text(root / "report.md", f"# mlwego run\n\nTask: {task_name}\n")
    return root


def _link_or_copy_data(source: Path, target: Path) -> None:
    if target.exists() and any(target.iterdir()):
        return
    try:
        os.symlink(source.resolve(), target, target_is_directory=True)
    except OSError:
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
