"""Educational-only publication scope for registered GRT adapters, not raw tasks."""

from __future__ import annotations

import argparse

SUPPORTED_GRT_TASKS = frozenset({"dive_bench_educational_high_fps", "densevideo"})
_SCOPE_ERROR = "Released GRT profiles support only Educational DIVE-Bench task identities; High-Motion and other tasks are not validated GRT profiles. Raw tasks and stock models remain available."


def require_grt_task(task: object) -> None:
    """Check an exact task identity without interpreting prompts or labels."""
    if not isinstance(task, str) or task not in SUPPORTED_GRT_TASKS:
        raise ValueError(_SCOPE_ERROR)


def preflight_grt_requests(requests):
    """Materialize once and check the whole batch before any backend side effect."""
    try:
        checked = list(requests)
    except TypeError as exc:
        raise ValueError("GRT requires an iterable of task-bound requests") from exc
    for request in checked:
        args = getattr(request, "args", None)
        if not isinstance(args, (tuple, list)) or len(args) != 6:
            raise ValueError("GRT requires six-field task-bound requests")
        task = args[4]
        require_grt_task(task)
        metadata = getattr(request, "metadata", None)
        if metadata is not None and (not isinstance(metadata, dict) or metadata.get("task") != task):
            raise ValueError("GRT request metadata and inference task identities disagree")
        task_name = getattr(request, "task_name", None)
        if task_name is not None and task_name != task:
            raise ValueError("GRT request metadata and inference task identities disagree")
    return checked


def require_grt_worker_tasks(argv) -> None:
    """Reject unsupported or ambiguous task flags before importing CUDA runtime."""
    for arg in argv:
        name = arg.split("=", 1)[0]
        if name.startswith("--") and name != "--tasks" and "--tasks".startswith(name):
            raise ValueError("GRT worker requires the explicit --tasks flag, without abbreviations")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--tasks", action="append")
    args, _ = parser.parse_known_args(argv)
    if not args.tasks or len(args.tasks) != 1:
        raise ValueError("GRT worker requires exactly one explicit --tasks selection")
    for task in args.tasks[0].split(","):
        require_grt_task(task)


class EducationalGRTScope:
    """Guard native entry points without changing any frozen model function body."""

    def generate_until(self, requests):
        return super().generate_until(preflight_grt_requests(requests))

    def loglikelihood(self, requests):
        return super().loglikelihood(preflight_grt_requests(requests))

    def generate_until_multi_round(self, requests):
        return super().generate_until_multi_round(preflight_grt_requests(requests))
