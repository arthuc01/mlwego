"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlwego.agent.controller import finalize_best, run_search
from mlwego.evaluation.evaluator import run_predict, validate_submission
from mlwego.search.policy import SearchPolicy
from mlwego.workspace.project_init import init_workspace


def cmd_init(args: argparse.Namespace) -> None:
    run_dir = init_workspace(args.task, args.data, args.out)
    print(str(run_dir))


def cmd_run(args: argparse.Namespace) -> None:
    run_dir = Path(args.out)
    policy = SearchPolicy(budget=args.budget)
    tree, summaries = run_search(run_dir, policy, timeout=args.timeout)
    finalize_best(run_dir, tree)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "logs" / "summary.json"
    summary_path.write_text(json.dumps([s.__dict__ for s in summaries], indent=2), encoding="utf-8")


def cmd_best(args: argparse.Namespace) -> None:
    run_dir = Path(args.out)
    summary_path = run_dir / "logs" / "summary.json"
    if not summary_path.exists():
        raise SystemExit("summary.json not found")
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    best = max(summaries, key=lambda item: item["score"])
    print(json.dumps(best, indent=2))


def cmd_submit(args: argparse.Namespace) -> None:
    run_dir = Path(args.out)
    run_predict(run_dir)
    error = validate_submission(run_dir)
    if error:
        raise SystemExit(error)
    print(str(run_dir / "artifacts" / "submission.csv"))


def cmd_replay(args: argparse.Namespace) -> None:
    run_dir = Path(args.out)
    node_dir = run_dir / "artifacts" / args.node / "src"
    if not node_dir.exists():
        raise SystemExit("Snapshot not found")
    src_dir = run_dir / "src"
    backup_dir = run_dir / "src_backup"
    if backup_dir.exists():
        for item in backup_dir.iterdir():
            if item.is_dir():
                for sub in item.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                item.rmdir()
            else:
                item.unlink()
        backup_dir.rmdir()
    src_dir.replace(backup_dir)
    node_dir.replace(src_dir)
    run_predict(run_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlwego")
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--task", required=True)
    init_parser.add_argument("--data", required=True)
    init_parser.add_argument("--out")
    init_parser.set_defaults(func=cmd_init)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--budget", type=int, default=10)
    run_parser.add_argument("--timeout", type=int, default=1200)
    run_parser.set_defaults(func=cmd_run)

    best_parser = sub.add_parser("best")
    best_parser.add_argument("--out", required=True)
    best_parser.set_defaults(func=cmd_best)

    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--out", required=True)
    submit_parser.set_defaults(func=cmd_submit)

    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--out", required=True)
    replay_parser.add_argument("--node", required=True)
    replay_parser.set_defaults(func=cmd_replay)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
