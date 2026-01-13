"""Execute python scripts in a controlled subprocess."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExecutionResult:
    exit_code: int
    runtime: float
    stdout: str
    stderr: str
    command: List[str]
    cwd: str


def run_python(
    script_path: Path,
    cwd: Path,
    timeout: int,
    env: Optional[Dict[str, str]] = None,
) -> ExecutionResult:
    start = time.time()
    cmd = ["python", str(script_path)]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    runtime = time.time() - start
    return ExecutionResult(
        exit_code=proc.returncode,
        runtime=runtime,
        stdout=proc.stdout,
        stderr=proc.stderr,
        command=cmd,
        cwd=str(cwd),
    )
