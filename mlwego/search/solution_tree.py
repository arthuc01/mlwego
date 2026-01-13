"""Solution tree structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SolutionNode:
    node_id: str
    parent_id: Optional[str]
    score: float
    score_std: float
    diff: str
    children: List[str] = field(default_factory=list)


@dataclass
class SolutionTree:
    nodes: dict[str, SolutionNode] = field(default_factory=dict)

    def add_node(self, node: SolutionNode) -> None:
        self.nodes[node.node_id] = node
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children.append(node.node_id)

    def best_node(self) -> Optional[SolutionNode]:
        if not self.nodes:
            return None
        return max(self.nodes.values(), key=lambda n: n.score)
