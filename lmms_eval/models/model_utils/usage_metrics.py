"""Central usage tracking for API-backed model calls.

Follows the same module-level pattern as ``gen_metrics.py``:
a global history list, a ``log_usage`` helper called by individual models,
and a ``summarize_usage_metrics`` aggregator consumed by the evaluator.

Thread-safe: models using ``ThreadPoolExecutor`` can call ``log_usage``
from worker threads without external synchronisation.
"""

import threading
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_USAGE_HISTORY: List[Dict[str, Any]] = []
_USAGE_LOCK = threading.Lock()

_BUDGET: Optional[int] = None  # max total tokens; None → no limit
_BUDGET_EXCEEDED: bool = False

# Set by the evaluator before ``task.process_results()`` so that judge
# providers (which have no visibility into the current task) can inherit
# the task name automatically.
_CURRENT_TASK_CONTEXT: Optional[str] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reset_usage_metrics() -> None:
    """Clear all accumulated usage data and budget state.

    Called once at the start of ``simple_evaluate()``, alongside
    ``reset_logged_metrics()``.
    """
    global _BUDGET, _BUDGET_EXCEEDED, _CURRENT_TASK_CONTEXT

    with _USAGE_LOCK:
        _USAGE_HISTORY.clear()
    _BUDGET = None
    _BUDGET_EXCEEDED = False
    _CURRENT_TASK_CONTEXT = None


def set_budget(max_tokens: Optional[int] = None) -> None:
    """Configure the token budget for the current evaluation run.

    Parameters
    ----------
    max_tokens:
        Maximum *total* tokens (input + output + reasoning) across the
        entire run.  ``None`` disables enforcement.
    """
    global _BUDGET
    _BUDGET = max_tokens


def set_task_context(task_name: Optional[str]) -> None:
    """Set the current task name used as fallback by ``log_usage``.

    The evaluator calls this before each task's ``process_results`` phase
    so that LLM-judge providers — which call ``log_usage(task_name=None)``
    — automatically inherit the correct task attribution.

    Single-threaded (postprocessing loop); no lock required.
    """
    global _CURRENT_TASK_CONTEXT
    _CURRENT_TASK_CONTEXT = task_name


def log_usage(
    model_name: str,
    task_name: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,
    source: str = "model",
) -> None:
    """Record token usage from a single API call.

    Parameters
    ----------
    model_name:
        The model identifier (e.g. ``"gpt-4o"``, ``"claude-sonnet-4"``).
    task_name:
        Evaluation task that triggered this call.  Falls back to
        ``_CURRENT_TASK_CONTEXT`` when ``None``.
    input_tokens / output_tokens / reasoning_tokens:
        Token counts extracted from the API response.
    source:
        ``"model"`` for primary inference calls, ``"judge"`` for
        LLM-as-judge scoring calls.
    """
    global _BUDGET_EXCEEDED

    if task_name is None:
        task_name = _CURRENT_TASK_CONTEXT

    record: Dict[str, Any] = {
        "model_name": model_name,
        "task_name": task_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "source": source,
    }

    with _USAGE_LOCK:
        _USAGE_HISTORY.append(record)
        _check_budget()

    eval_logger.debug(
        "Usage: model={} task={} in={} out={} reason={} source={}",
        model_name,
        task_name,
        input_tokens,
        output_tokens,
        reasoning_tokens,
        source,
    )


def is_budget_exceeded() -> bool:
    """Return whether the configured token budget has been exceeded.

    Reading a ``bool`` is atomic in CPython, so no lock is needed.  The
    worst-case scenario is one additional API call before detection.
    """
    return _BUDGET_EXCEEDED


def get_running_totals() -> Dict[str, Any]:
    """Return aggregated totals suitable for tqdm postfix display."""
    with _USAGE_LOCK:
        history = list(_USAGE_HISTORY)

    input_tokens = sum(r["input_tokens"] for r in history)
    output_tokens = sum(r["output_tokens"] for r in history)
    reasoning_tokens = sum(r["reasoning_tokens"] for r in history)
    total_tokens = input_tokens + output_tokens + reasoning_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "n_api_calls": len(history),
    }


def summarize_usage_metrics() -> Dict[str, Any]:
    """Aggregate all recorded usage into a summary dict.

    Returns an empty dict when no usage has been recorded.  The returned
    structure is injected into ``results["usage"]`` by the evaluator.
    """
    with _USAGE_LOCK:
        history = list(_USAGE_HISTORY)

    if not history:
        return {}

    def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        inp = sum(r["input_tokens"] for r in records)
        out = sum(r["output_tokens"] for r in records)
        reason = sum(r["reasoning_tokens"] for r in records)
        return {
            "input_tokens": inp,
            "output_tokens": out,
            "reasoning_tokens": reason,
            "total_tokens": inp + out + reason,
            "n_api_calls": len(records),
        }

    # -- total --
    total = _aggregate(history)

    # -- by_task --
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for r in history:
        key = r["task_name"] if r["task_name"] is not None else "_unknown"
        by_task.setdefault(key, []).append(r)
    by_task_agg = {k: _aggregate(v) for k, v in by_task.items()}

    # -- by_source --
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for r in history:
        by_source.setdefault(r["source"], []).append(r)
    by_source_agg = {k: _aggregate(v) for k, v in by_source.items()}

    return {
        "total": total,
        "by_task": by_task_agg,
        "by_source": by_source_agg,
        "budget_exceeded": _BUDGET_EXCEEDED,
        "budget_total_tokens": _BUDGET,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_budget() -> None:
    """Check whether the token budget has been exceeded.

    **Must be called while ``_USAGE_LOCK`` is held.**
    """
    global _BUDGET_EXCEEDED

    if _BUDGET is None or _BUDGET_EXCEEDED:
        return

    total = sum(r["input_tokens"] + r["output_tokens"] + r["reasoning_tokens"] for r in _USAGE_HISTORY)

    if total >= _BUDGET:
        _BUDGET_EXCEEDED = True
        eval_logger.warning("Token budget exceeded: {} / {} tokens used", f"{total:,}", f"{_BUDGET:,}")
