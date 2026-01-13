"""Safe tools exposed to the LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from mlwego.execution.sandbox import run_python
from mlwego.workspace.file_ops import read_text, safe_path, write_text


class ToolContext:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def read_file(self, rel_path: str) -> str:
        path = safe_path(self.root, self.root / rel_path)
        return read_text(path)

    def write_file(self, rel_path: str, content: str) -> None:
        path = safe_path(self.root, self.root / rel_path)
        write_text(path, content)

    def list_dir(self, rel_path: str = ".") -> List[str]:
        path = safe_path(self.root, self.root / rel_path)
        return [p.name for p in path.iterdir()]

    def run(self, rel_path: str, timeout: int = 300) -> Dict[str, Any]:
        script_path = safe_path(self.root, self.root / rel_path)
        result = run_python(script_path, cwd=script_path.parent, timeout=timeout)
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "runtime": result.runtime,
        }

    def parse_csv_schema(self, rel_path: str) -> Dict[str, str]:
        path = safe_path(self.root, self.root / rel_path)
        df = pd.read_csv(path, nrows=5)
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    def apply_patch(self, rel_path: str, new_content: str) -> str:
        path = safe_path(self.root, self.root / rel_path)
        old = read_text(path) if path.exists() else ""
        write_text(path, new_content)
        return json.dumps({"old_length": len(old), "new_length": len(new_content)})
