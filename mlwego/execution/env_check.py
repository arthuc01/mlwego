"""Environment checks."""

from __future__ import annotations

import importlib.util
import platform
import subprocess
from dataclasses import dataclass
from typing import Dict


@dataclass
class EnvInfo:
    python_version: str
    packages: Dict[str, bool]
    has_gpu: bool


def check_env(packages: list[str]) -> EnvInfo:
    installed = {name: importlib.util.find_spec(name) is not None for name in packages}
    has_gpu = _detect_gpu()
    return EnvInfo(python_version=platform.python_version(), packages=installed, has_gpu=has_gpu)


def _detect_gpu() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=2)
        return result.returncode == 0
    except Exception:
        return False
