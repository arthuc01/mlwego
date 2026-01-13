"""Controller orchestrating search."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from mlwego.evaluation.evaluator import EvalResult, evaluate_solution, run_predict
from mlwego.search.generator import CandidateEdit, baseline_candidates
from mlwego.search.policy import SearchPolicy
from mlwego.search.solution_tree import SolutionNode, SolutionTree
from mlwego.workspace.file_ops import read_text, write_text
from mlwego.workspace.snapshot import hash_src, snapshot_src


@dataclass
class RunSummary:
    node_id: str
    score: float
    metric: str


def apply_candidate(config_path: Path, candidate: CandidateEdit) -> str:
    config = json.loads(read_text(config_path))
    config.update(candidate.updates)
    write_text(config_path, json.dumps(config, indent=2))
    return json.dumps(candidate.updates)


def run_search(run_dir: Path, policy: SearchPolicy, timeout: int) -> tuple[SolutionTree, List[RunSummary]]:
    tree = SolutionTree()
    summaries: List[RunSummary] = []
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / "metrics.jsonl"
    config_path = run_dir / "src" / "config.json"
    baseline_config = read_text(config_path)
    eval_result = evaluate_solution(run_dir, timeout=timeout)
    root_hash = hash_src(run_dir / "src")
    _append_metrics(metrics_path, root_hash, eval_result)
    tree.add_node(
        SolutionNode(
            node_id=root_hash,
            parent_id=None,
            score=eval_result.score,
            score_std=eval_result.score_std,
            diff="baseline",
        )
    )
    summaries.append(RunSummary(node_id=root_hash, score=eval_result.score, metric=eval_result.metric))
    candidates = baseline_candidates()
    max_candidates = min(policy.budget, policy.branch_factor, len(candidates))
    for candidate in candidates[:max_candidates]:
        write_text(config_path, baseline_config)
        diff = apply_candidate(config_path, candidate)
        eval_result = evaluate_solution(run_dir, timeout=timeout)
        node_hash = hash_src(run_dir / "src")
        _append_metrics(metrics_path, node_hash, eval_result)
        tree.add_node(
            SolutionNode(
                node_id=node_hash,
                parent_id=root_hash,
                score=eval_result.score,
                score_std=eval_result.score_std,
                diff=diff,
            )
        )
        summaries.append(RunSummary(node_id=node_hash, score=eval_result.score, metric=eval_result.metric))
        snapshot_src(run_dir / "src", run_dir / "artifacts" / node_hash)
    return tree, summaries


def finalize_best(run_dir: Path, tree: SolutionTree) -> None:
    best = tree.best_node()
    if not best:
        return
    run_predict(run_dir)


def _append_metrics(path: Path, node_id: str, result: EvalResult) -> None:
    payload = {
        "node_id": node_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score": result.score,
        "score_std": result.score_std,
        "metric_name": result.metric,
        "cv": "default",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
