"""Safe file operations."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Iterable, Tuple


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def apply_patch(path: Path, new_content: str) -> Tuple[str, str]:
    old_content = read_text(path) if path.exists() else ""
    diff = "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )
    write_text(path, new_content)
    return old_content, diff


def diff_text(old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
    )


def safe_path(root: Path, target: Path) -> Path:
    resolved = target.resolve()
    if not str(resolved).startswith(str(root.resolve())):
        raise ValueError("Path traversal detected")
    return resolved
