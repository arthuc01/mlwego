"""Token budget utilities."""

from __future__ import annotations

from typing import Iterable, List


def truncate_logs(logs: Iterable[str], max_chars: int = 8000) -> List[str]:
    collected: List[str] = []
    total = 0
    for entry in logs:
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining <= 0:
                break
            collected.append(entry[:remaining])
            break
        collected.append(entry)
        total += len(entry)
    return collected
