"""Select next node to expand."""

from __future__ import annotations

from typing import Optional

from mlwego.search.solution_tree import SolutionTree


def select_best(tree: SolutionTree) -> Optional[str]:
    best = tree.best_node()
    return best.node_id if best else None
