"""Shared helpers for power-analysis task loading."""

from __future__ import annotations

import traceback

import lmms_eval.tasks
from lmms_eval.tasks import TaskManager


def _is_debug_verbosity(verbosity: object) -> bool:
    return str(verbosity or "").upper() == "DEBUG"


def _format_task_load_error(exc: Exception) -> str:
    def _flatten(text: object) -> str:
        return " | ".join(line.strip() for line in str(text).splitlines() if line.strip())

    details = [f"{type(exc).__name__}: {_flatten(exc)}"]

    root = exc
    visited = {id(root)}
    while True:
        nxt = getattr(root, "__cause__", None) or getattr(root, "__context__", None)
        if nxt is None or id(nxt) in visited:
            break
        root = nxt
        visited.add(id(root))
    if root is not exc:
        details.append(f"root={type(root).__name__}: {_flatten(root)}")

    tb = traceback.extract_tb(exc.__traceback__)
    if tb:
        last = tb[-1]
        details.append(f"at {last.filename}:{last.lineno} ({last.name})")

    return " | ".join(details)


def collect_task_sizes(tasks_arg: str | None, *, verbosity: str = "WARNING", include_path: str | None = None) -> dict[str, int]:
    task_sizes: dict[str, int] = {}
    if not tasks_arg:
        return task_sizes

    task_manager = TaskManager(verbosity, include_path=include_path)
    requested_tasks = [task.strip() for task in tasks_arg.split(",") if task.strip()]
    task_names = task_manager.match_tasks(requested_tasks)

    missing_tasks = [task for task in requested_tasks if task not in task_names and "*" not in task]
    if missing_tasks:
        print(f"[warning] Unresolved task names (skipped): {', '.join(sorted(set(missing_tasks)))}")
    if not task_names:
        print("[warning] No valid tasks resolved from --tasks; running global power analysis only.")

    for task_name in task_names:
        try:
            task_dict = lmms_eval.tasks.get_task_dict([task_name], task_manager)
        except Exception as exc:
            print(f"[warning] Failed to load task '{task_name}' (skipped): {_format_task_load_error(exc)}")
            if _is_debug_verbosity(verbosity):
                print("[warning] Full traceback:")
                print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
            continue

        for name, task_obj in task_dict.items():
            if hasattr(task_obj, "eval_docs"):
                task_sizes[name] = len(task_obj.eval_docs)

    return task_sizes
