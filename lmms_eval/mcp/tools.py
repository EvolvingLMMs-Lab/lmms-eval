from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger
from mcp.server.fastmcp import FastMCP

from lmms_eval.mcp.schemas import (
    EvalRunResult,
    EvalRunStatus,
    EvalRunSubmitted,
    ModelInfo,
    ModelListResponse,
    TaskInfo,
    TaskListResponse,
)

_scheduler = None


def register_tools(mcp: FastMCP, scheduler) -> None:
    """Register all MCP tools on the given FastMCP server instance."""
    global _scheduler
    _scheduler = scheduler

    # ---- Tier 1: Discovery Tools (no torch imports) ----

    @mcp.tool()
    def list_tasks(query: Optional[str] = None) -> dict:
        """List available evaluation tasks/benchmarks.

        Args:
            query: Optional filter string to match task names (glob-style with *).

        Returns a list of tasks with their type (task/group/tag) and output type.
        """
        import fnmatch

        from lmms_eval.tasks import TaskManager

        tm = TaskManager(verbosity="ERROR")

        tasks = []
        for name in tm.all_tasks:
            if query and not fnmatch.fnmatch(name, query):
                continue

            info = tm.task_index[name]
            task_type = info["type"]
            yaml_path = info.get("yaml_path")
            yaml_path_str = str(yaml_path) if yaml_path and yaml_path != -1 else None

            output_type = None
            if yaml_path and yaml_path != -1 and task_type == "task":
                try:
                    from lmms_eval import utils

                    config = utils.load_yaml_config(yaml_path, mode="simple")
                    output_type = config.get("output_type")
                except Exception:
                    output_type = None

            tasks.append(TaskInfo(name=name, type=task_type, yaml_path=yaml_path_str, output_type=output_type))

        resp = TaskListResponse(tasks=tasks, total=len(tasks), query=query)
        return resp.model_dump()

    @mcp.tool()
    def get_task_info(task_name: str) -> dict:
        """Get detailed information about a specific evaluation task.

        Args:
            task_name: Name of the task to look up.

        Returns detailed task configuration including dataset path, metrics, splits, etc.
        """
        from lmms_eval import utils
        from lmms_eval.tasks import TaskManager

        tm = TaskManager(verbosity="ERROR")
        if task_name not in tm.task_index:
            raise ValueError(f"Task '{task_name}' not found. Use list_tasks() to see available tasks.")

        info = tm.task_index[task_name]
        yaml_path = info.get("yaml_path")

        result = {
            "name": task_name,
            "type": info["type"],
        }

        if yaml_path and yaml_path != -1:
            try:
                config = utils.load_yaml_config(yaml_path, mode="full")
                safe_keys = [
                    "task",
                    "dataset_path",
                    "dataset_name",
                    "output_type",
                    "test_split",
                    "validation_split",
                    "training_split",
                    "num_fewshot",
                    "metric_list",
                    "group",
                    "tag",
                    "doc_to_text",
                    "doc_to_visual",
                    "doc_to_messages",
                    "generation_kwargs",
                    "repeats",
                    "lmms_eval_specific_kwargs",
                ]
                for key in safe_keys:
                    if key not in config:
                        continue
                    val = config[key]
                    result[key] = str(val) if callable(val) else val
                result["yaml_path"] = str(yaml_path)
            except Exception as e:
                result["config_error"] = str(e)

        return result

    @mcp.tool()
    def list_models(include_aliases: bool = False) -> dict:
        """List available model backends.

        Args:
            include_aliases: If True, include model aliases (alternative names) in the list.

        Returns a list of model backends with their capabilities (chat vs simple).
        """
        from lmms_eval.models import MODEL_REGISTRY_V2

        models = []
        for model_id in MODEL_REGISTRY_V2.list_canonical_model_ids():
            manifest = MODEL_REGISTRY_V2.get_manifest(model_id)
            aliases = list(manifest.aliases) if include_aliases else []
            models.append(ModelInfo(model_id=model_id, has_chat=manifest.chat_class_path is not None, has_simple=manifest.simple_class_path is not None, aliases=aliases))

        resp = ModelListResponse(models=models, total=len(models))
        return resp.model_dump()

    @mcp.tool()
    def get_model_info(model_name: str) -> dict:
        """Get detailed information about a specific model backend.

        Args:
            model_name: Model name or alias to look up.

        Returns model details including class paths and interface type.
        """
        from lmms_eval.models import MODEL_REGISTRY_V2

        try:
            manifest = MODEL_REGISTRY_V2.get_manifest(model_name)
        except ValueError as e:
            raise ValueError(str(e))

        return {
            "model_id": manifest.model_id,
            "has_chat": manifest.chat_class_path is not None,
            "has_simple": manifest.simple_class_path is not None,
            "chat_class_path": manifest.chat_class_path,
            "simple_class_path": manifest.simple_class_path,
            "aliases": list(manifest.aliases),
        }

    # ---- Tier 2: Evaluation Lifecycle Tools ----

    @mcp.tool()
    async def evaluate(
        model: str,
        tasks: list[str],
        model_args: Optional[dict] = None,
        batch_size: Optional[int] = None,
        limit: Optional[int] = None,
        num_fewshot: Optional[int] = None,
        gen_kwargs: Optional[str] = None,
        log_samples: bool = True,
        num_gpus: int = 1,
        mode: str = "auto",
        wait_timeout_s: int = 90,
    ) -> dict:
        """Run a model evaluation on specified tasks.

        Args:
            model: Model backend name (e.g. "qwen2_5_vl", "vllm", "openai").
            tasks: List of task names to evaluate (e.g. ["mme", "mmmu_val"]).
            model_args: Model-specific arguments as a dict (e.g. {"pretrained": "Qwen/Qwen2.5-VL-7B-Instruct"}).
            batch_size: Batch size for evaluation.
            limit: Limit the number of examples per task (useful for testing).
            num_fewshot: Number of few-shot examples.
            gen_kwargs: Generation kwargs as string.
            log_samples: Whether to log individual sample results.
            num_gpus: Number of GPUs to use.
            mode: Execution mode - "auto" (wait up to wait_timeout_s, then return run_id), "async" (return run_id immediately), "sync" (wait until done).
            wait_timeout_s: For mode="auto", how long to wait before returning a run_id (default: 90s).

        Returns either the evaluation results (if completed within timeout) or a run_id for polling.
        """
        if _scheduler is None:
            raise RuntimeError("JobScheduler not initialized. Server may not be fully started.")

        if mode not in {"auto", "async", "sync"}:
            raise ValueError("mode must be one of: auto, async, sync")

        from lmms_eval.entrypoints.protocol import EvaluateRequest

        request = EvaluateRequest(
            model=model,
            tasks=tasks,
            model_args=model_args,
            batch_size=batch_size,
            limit=limit,
            num_fewshot=num_fewshot,
            gen_kwargs=gen_kwargs,
            log_samples=log_samples,
            num_gpus=num_gpus,
        )

        job_id, position = await _scheduler.add_job(request)
        logger.info(f"[MCP] Submitted evaluation job {job_id}, position={position}")

        if mode == "async":
            return EvalRunSubmitted(
                run_id=job_id,
                status="queued",
                position_in_queue=position,
                message=f"Job queued at position {position}. Use get_run_status('{job_id}') to check progress.",
            ).model_dump()

        timeout = None if mode == "sync" else wait_timeout_s
        try:
            elapsed = 0
            poll_interval = 2
            while True:
                job = await _scheduler.get_job(job_id)
                if job is None:
                    raise RuntimeError(f"Job {job_id} disappeared unexpectedly")

                if job.status.value in {"completed", "failed", "cancelled"}:
                    break

                if timeout is not None and elapsed >= timeout:
                    return EvalRunSubmitted(
                        run_id=job_id,
                        status=job.status.value,
                        position_in_queue=job.position_in_queue,
                        message=f"Job still {job.status.value} after {elapsed}s. Use get_run_status('{job_id}') to check progress.",
                    ).model_dump()

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            if job.status.value == "completed":
                return EvalRunResult(run_id=job_id, status="completed", results=job.result).model_dump()
            if job.status.value == "failed":
                return EvalRunResult(run_id=job_id, status="failed", error=job.error).model_dump()
            return EvalRunResult(run_id=job_id, status=job.status.value).model_dump()

        except Exception as e:
            logger.error(f"[MCP] Error waiting for job {job_id}: {e}")
            return EvalRunSubmitted(run_id=job_id, status="unknown", message=f"Error tracking job: {e}. Use get_run_status('{job_id}') to check.").model_dump()

    @mcp.tool()
    async def get_run_status(run_id: str) -> dict:
        """Get the status of an evaluation run.

        Args:
            run_id: The run ID returned by the evaluate tool.

        Returns current status, timestamps, and queue position if still queued.
        """
        if _scheduler is None:
            raise RuntimeError("JobScheduler not initialized.")

        job = await _scheduler.get_job_with_position(run_id)
        if job is None:
            raise ValueError(f"Run '{run_id}' not found.")

        return EvalRunStatus(
            run_id=run_id,
            status=job.status.value,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            position_in_queue=job.position_in_queue,
            error=job.error,
        ).model_dump()

    @mcp.tool()
    async def get_run_result(
        run_id: str,
        include_samples: bool = False,
        sample_limit: int = 100,
        sample_offset: int = 0,
    ) -> dict:
        """Get the results of a completed evaluation run.

        Args:
            run_id: The run ID returned by the evaluate tool.
            include_samples: Whether to include per-sample details (can be large).
            sample_limit: Maximum number of samples to include (default: 100).
            sample_offset: Offset for sample pagination (default: 0).

        Returns evaluation results with optional sample-level data.
        """
        if _scheduler is None:
            raise RuntimeError("JobScheduler not initialized.")
        if sample_offset < 0:
            raise ValueError("sample_offset must be >= 0")
        if sample_limit < 0:
            raise ValueError("sample_limit must be >= 0")

        job = await _scheduler.get_job(run_id)
        if job is None:
            raise ValueError(f"Run '{run_id}' not found.")

        if job.status.value != "completed":
            return EvalRunResult(run_id=run_id, status=job.status.value, error=f"Run is not completed yet. Status: {job.status.value}").model_dump()

        result_data = job.result
        if result_data:
            filtered = {}
            for model_name, data in result_data.items():
                model_result = {"results": data.get("results")}
                if include_samples:
                    samples = data.get("samples") or []
                    model_result["samples"] = samples[sample_offset : sample_offset + sample_limit]
                filtered[model_name] = model_result
            result_data = filtered

        return EvalRunResult(run_id=run_id, status="completed", results=result_data).model_dump()

    @mcp.tool()
    async def cancel_run(run_id: str) -> dict:
        """Cancel a queued evaluation run.

        Args:
            run_id: The run ID to cancel. Only queued (not yet running) runs can be cancelled.

        Returns success status and message.
        """
        if _scheduler is None:
            raise RuntimeError("JobScheduler not initialized.")

        success, message = await _scheduler.cancel_job(run_id)
        return {"success": success, "run_id": run_id, "message": message}
