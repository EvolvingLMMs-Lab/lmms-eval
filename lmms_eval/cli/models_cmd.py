"""lmms-eval models — list registered model backends and aliases."""

from __future__ import annotations

import argparse


def add_models_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "models",
        help="List available model backends",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--aliases",
        action="store_true",
        default=False,
        help="Include model aliases in the listing",
    )
    p.set_defaults(func=run_models)


def _col(text: str, width: int) -> str:
    """Left-align text in a fixed-width column."""
    return text[:width].ljust(width)


def _model_rows() -> list[tuple[str, str, tuple[str, ...]]]:
    from lmms_eval.models import MODEL_REGISTRY_V2

    rows = []
    for manifest in MODEL_REGISTRY_V2.list_manifests():
        kind = "chat+simple" if manifest.chat_class_path and manifest.simple_class_path else "chat" if manifest.chat_class_path else "simple"
        rows.append((manifest.model_id, kind, manifest.aliases))
    return rows


def run_models(args: argparse.Namespace) -> None:
    rows = _model_rows()
    chat_only = [row for row in rows if row[1] == "chat"]
    dual = [row for row in rows if row[1] == "chat+simple"]
    simple_only = [row for row in rows if row[1] == "simple"]

    show_aliases = getattr(args, "aliases", False)

    if show_aliases:
        header = f"{_col('Name', 28)}{_col('Type', 14)}{_col('Aliases', 40)}"
        sep = "-" * 82
    else:
        header = f"{_col('Name', 28)}{_col('Type', 14)}"
        sep = "-" * 42
    print(f"\nRegistered Models ({len(rows)} total)\n")
    print(header)
    print(sep)

    def _print_row(name: str, typ: str, aliases: tuple[str, ...]) -> None:
        if show_aliases:
            alias = ", ".join(aliases)
            print(f"{_col(name, 28)}{_col(typ, 14)}{alias}")
        else:
            print(f"{_col(name, 28)}{_col(typ, 14)}")

    # Chat-only models first (recommended)
    for row in chat_only:
        _print_row(*row)
    # Dual-mode models
    for row in dual:
        _print_row(*row)
    # Simple-only models
    for row in simple_only:
        _print_row(*row)

    print(sep)
    print(f"\n  chat-only: {len(chat_only)}  |  chat+simple: {len(dual)}  |  simple-only: {len(simple_only)}")
    print("  Tip: chat models are recommended for new evaluations.\n")
