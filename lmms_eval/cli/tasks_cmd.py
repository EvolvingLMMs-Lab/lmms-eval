"""lmms-eval tasks — browse available benchmarks without downloading data."""

from __future__ import annotations

import argparse
import sys


def add_tasks_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "tasks",
        help="List available evaluation tasks, groups, and tags",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "action",
        nargs="?",
        default="list",
        choices=["list", "groups", "subtasks", "tags"],
        help=(
            "What to show (default: list)\n"
            "  list      — flat list of every registered name\n"
            "  groups    — markdown table of task groups\n"
            "  subtasks  — markdown table of leaf tasks with config path and output type\n"
            "  tags      — list of tags"
        ),
    )
    p.add_argument(
        "--include-path",
        type=str,
        default=None,
        help="Additional path to include external task definitions",
    )
    p.add_argument(
        "--verbosity",
        type=str,
        default="WARNING",
        help="Logging verbosity (default: WARNING)",
    )
    p.set_defaults(func=run_tasks)


def run_tasks(args: argparse.Namespace) -> None:
    from lmms_eval.tasks import TaskManager

    tm = TaskManager(args.verbosity, include_path=args.include_path)

    action = args.action
    if action == "list":
        names = sorted(tm.all_tasks)
        print(f"Available tasks ({len(names)}):\n")
        for name in names:
            print(f"  {name}")
    elif action == "groups":
        print(tm.list_all_tasks(list_subtasks=False, list_tags=False))
    elif action == "subtasks":
        print(tm.list_all_tasks(list_groups=False, list_tags=False))
    elif action == "tags":
        print(tm.list_all_tasks(list_groups=False, list_subtasks=False))
