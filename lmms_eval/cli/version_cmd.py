"""lmms-eval version — print version and environment info."""

from __future__ import annotations

import argparse


def add_version_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "version",
        help="Print version and environment information",
    )
    p.set_defaults(func=run_version)


def run_version(args: argparse.Namespace) -> None:
    import importlib.metadata
    import subprocess
    import sys

    try:
        version = importlib.metadata.version("lmms-eval")
    except importlib.metadata.PackageNotFoundError:
        version = "dev (not installed)"

    git_hash = "unknown"
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        pass

    print(f"lmms-eval {version}")
    print(f"Python     {sys.version.split()[0]}")
    print(f"Git        {git_hash}")

    # Optional: show torch/accelerate if available
    for pkg in ("torch", "accelerate", "transformers", "vllm"):
        try:
            v = importlib.metadata.version(pkg)
            print(f"{pkg:11s}{v}")
        except importlib.metadata.PackageNotFoundError:
            pass
