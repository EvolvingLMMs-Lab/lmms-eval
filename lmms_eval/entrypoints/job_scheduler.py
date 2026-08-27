"""
Job Scheduler for LMMS-Eval Server.

This module provides a thread-safe job scheduler that manages evaluation jobs
with queue-based execution. Jobs are processed sequentially to ensure proper
GPU resource management.
"""

import asyncio
import os
import signal
import sys
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from lmms_eval.entrypoints.protocol import (
    EvaluateRequest,
    JobInfo,
    JobStatus,
)

# Env vars that Slurm's srun (and torch.distributed launchers generally) set
# on a process and that leak into a subprocess spawned from within it. If the
# HTTP eval server itself runs under such a launcher, these vars would
# otherwise leak into the eval subprocess and trick Accelerate's
# PartialState into attempting env:// rendezvous. Scrubbed from the
# subprocess env by _run_subprocess_with_log.
_SLURM_DISTRIBUTED_ENV_SCRUB = (
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "SLURM_NPROCS",
    "SLURM_NODEID",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_STEP_NUM_TASKS",
    "SLURM_STEP_NUM_NODES",
    "SLURM_STEP_TASKS_PER_NODE",
    "SLURM_TASKS_PER_NODE",
    "PMIX_RANK",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_CPU_BIND",
    "SLURM_CPU_BIND_LIST",
    "SLURM_CPU_BIND_TYPE",
    "SLURM_CPU_BIND_VERBOSE",
    "SLURM_MEM_BIND",
    "SLURM_MEM_BIND_LIST",
    "SLURM_MEM_BIND_TYPE",
    "SLURM_MEM_BIND_VERBOSE",
)

# =============================================================================
# Job Scheduler
# =============================================================================


class JobScheduler:
    """
    Thread-safe job scheduler for managing evaluation jobs.

    The scheduler maintains a queue of jobs and processes them sequentially
    using a background worker. This ensures proper GPU resource management
    and prevents concurrent evaluation conflicts.

    Usage:
        scheduler = JobScheduler()
        await scheduler.start()

        job_id, position = await scheduler.add_job(request)
        job = await scheduler.get_job(job_id)

        await scheduler.stop()
    """

    DEFAULT_MAX_COMPLETED_JOBS = 100
    DEFAULT_TEMP_DIR_PREFIX = "lmms_eval_"

    def __init__(
        self,
        max_completed_jobs: int = DEFAULT_MAX_COMPLETED_JOBS,
        temp_dir_prefix: str = DEFAULT_TEMP_DIR_PREFIX,
    ):
        self._job_queue: asyncio.Queue = None
        self._jobs: Dict[str, JobInfo] = {}
        self._jobs_lock: asyncio.Lock = None
        self._worker_task: asyncio.Task = None
        self._current_job_id: Optional[str] = None
        self._max_completed_jobs = max_completed_jobs
        self._temp_dir_prefix = temp_dir_prefix

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    async def start(self):
        """Initialize and start the job scheduler."""
        self._job_queue = asyncio.Queue()
        self._jobs = {}
        self._jobs_lock = asyncio.Lock()
        self._worker_task = asyncio.create_task(self._job_worker())
        self._current_job_id = None
        logger.info("JobScheduler started, worker ready")

    async def stop(self):
        """Stop the job scheduler and cleanup resources."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("JobScheduler stopped")

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._job_queue.qsize() if self._job_queue else 0

    @property
    def current_job_id(self) -> Optional[str]:
        """Get the ID of the currently running job."""
        return self._current_job_id

    # -------------------------------------------------------------------------
    # Job Operations (Thread-safe)
    # -------------------------------------------------------------------------

    async def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get a job by ID (thread-safe)."""
        async with self._jobs_lock:
            return self._jobs.get(job_id)

    async def get_job_with_position(self, job_id: str) -> Optional[JobInfo]:
        """Get a job by ID, updating queue position if queued (thread-safe)."""
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None

            if job.status == JobStatus.QUEUED:
                position = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED and j.created_at < job.created_at)
                job.position_in_queue = position

            return job

    async def add_job(self, request: EvaluateRequest) -> tuple[str, int]:
        """
        Create and queue a new job.

        Args:
            request: The evaluation request configuration.

        Returns:
            Tuple of (job_id, position_in_queue).
        """
        job_id = str(uuid.uuid4())

        async with self._jobs_lock:
            position = self._job_queue.qsize()
            job = JobInfo(
                job_id=job_id,
                status=JobStatus.QUEUED,
                created_at=datetime.now().isoformat(),
                request=request,
                position_in_queue=position,
            )
            self._jobs[job_id] = job
            await self._job_queue.put(job_id)

        return job_id, position

    async def cancel_job(self, job_id: str) -> tuple[bool, str]:
        """
        Cancel a queued job.

        Args:
            job_id: The ID of the job to cancel.

        Returns:
            Tuple of (success, message).
        """
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False, f"Job {job_id} not found"

            if job.status == JobStatus.RUNNING:
                return False, "Cannot cancel a running job"

            if job.status in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                return False, "Job already finished or cancelled"

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now().isoformat()
            return True, f"Job {job_id} cancelled"

    async def get_queue_stats(self) -> dict:
        """Get queue statistics (thread-safe)."""
        async with self._jobs_lock:
            queued = [jid for jid, j in self._jobs.items() if j.status == JobStatus.QUEUED]
            completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
            failed = sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)

            return {
                "queued": queued,
                "completed": completed,
                "failed": failed,
                "running_job": self._current_job_id,
            }

    async def cleanup_old_jobs(self) -> int:
        """
        Remove old completed/failed/cancelled jobs to prevent memory leak.

        Keeps at most `max_completed_jobs` finished jobs, removing oldest first.

        Returns:
            Number of jobs removed.
        """
        async with self._jobs_lock:
            terminal_statuses = {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }
            finished_jobs = [(jid, job) for jid, job in self._jobs.items() if job.status in terminal_statuses]

            if len(finished_jobs) <= self._max_completed_jobs:
                return 0

            finished_jobs.sort(key=lambda x: x[1].completed_at or "")
            to_remove = len(finished_jobs) - self._max_completed_jobs

            removed = 0
            for jid, _ in finished_jobs[:to_remove]:
                del self._jobs[jid]
                removed += 1

            if removed > 0:
                logger.info(f"Cleaned up {removed} old jobs")

            return removed

    # -------------------------------------------------------------------------
    # Internal Job State Transitions
    # -------------------------------------------------------------------------

    async def _start_job(self, job_id: str) -> Optional[dict]:
        """
        Mark a job as running and return its config.

        Returns None if job doesn't exist or is cancelled.
        """
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None or job.status == JobStatus.CANCELLED:
                return None

            self._current_job_id = job_id
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now().isoformat()
            return job.request.model_dump()

    async def _complete_job(self, job_id: str, result: Dict[str, Any]):
        """Mark a job as completed with results."""
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now().isoformat()
                job.result = result

    async def _fail_job(self, job_id: str, error: str):
        """Mark a job as failed with error message."""
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now().isoformat()
                job.error = error

    # -------------------------------------------------------------------------
    # Background Worker
    # -------------------------------------------------------------------------

    async def _job_worker(self):
        """
        Background worker that processes jobs one-by-one from the queue.

        This ensures sequential execution of GPU-intensive evaluation jobs.
        """
        while True:
            try:
                job_id = await self._job_queue.get()

                # Start job and get config (returns None if cancelled/missing)
                config = await self._start_job(job_id)
                if config is None:
                    self._job_queue.task_done()
                    continue

                try:
                    # Run evaluation (outside lock to allow other operations)
                    result = await self._run_evaluation(config)
                    await self._complete_job(job_id, result)

                except Exception as e:
                    await self._fail_job(job_id, str(e))

                finally:
                    self._current_job_id = None
                    self._job_queue.task_done()
                    await self.cleanup_old_jobs()

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but keep worker running
                logger.error(f"Worker error: {e}")

    # -------------------------------------------------------------------------
    # Evaluation Execution
    # -------------------------------------------------------------------------

    async def _run_evaluation(self, config: dict) -> dict:
        """
        Run evaluation in a subprocess (``python -m lmms_eval`` for a single
        GPU, ``accelerate launch`` for multi-GPU).

        This allows GPU-based evaluation to run in a separate process
        while the server remains responsive.
        """
        output_path = config.get("output_dir") or tempfile.mkdtemp(prefix=self._temp_dir_prefix)

        cmd = self._build_eval_cmd(config, output_path)

        log_path = Path(output_path) / f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        logger.info(f"[EVAL] Launching: {' '.join(cmd)}")
        logger.info(f"[EVAL] Subprocess output -> {log_path}")

        returncode = await self._run_subprocess_with_log(cmd, log_path)

        if returncode != 0:
            tail = self._tail_log(log_path)
            raise RuntimeError(f"Evaluation failed with return code {returncode}. " f"Last lines of {log_path}:\n{tail}")

        result = self._parse_output_directory(output_path)
        if not result:
            tail = self._tail_log(log_path)
            raise RuntimeError("Evaluation produced no parsed results. " f"Last lines of {log_path}:\n{tail}")

        return result

    @staticmethod
    def _build_eval_cmd(config: dict, output_path: str) -> List[str]:
        """Build the ``python -m lmms_eval`` (or accelerate-launched) cmd.

        For ``num_gpus == 1`` we invoke ``python -m lmms_eval`` directly. Going
        through ``accelerate launch --num_processes 1`` falls into
        ``simple_launcher``, which spawns the child without setting
        ``WORLD_SIZE`` / ``RANK`` / ``MASTER_ADDR``. lmms_eval's ``cli_evaluate``
        unconditionally constructs an ``Accelerator()``, which then tries
        env:// rendezvous and raises ``ValueError: ... environment variable
        WORLD_SIZE expected, but not set``.
        """
        num_gpus = int(config.get("num_gpus") or 1)
        if num_gpus > 1:
            cmd: List[str] = [
                sys.executable,
                "-m",
                "accelerate.commands.launch",
                "--multi_gpu",
                "--num_processes",
                str(num_gpus),
                "--num_machines",
                "1",
                "-m",
                "lmms_eval",
            ]
        else:
            cmd = [sys.executable, "-m", "lmms_eval"]

        cmd.extend(
            [
                "--model",
                config["model"],
                "--tasks",
                ",".join(config["tasks"]),
                "--output_path",
                output_path,
            ]
        )

        if config.get("model_args"):
            if isinstance(config["model_args"], dict):
                model_args_str = ",".join(f"{k}={v}" for k, v in config["model_args"].items())
            else:
                model_args_str = str(config["model_args"])
            cmd.extend(["--model_args", model_args_str])

        if config.get("batch_size"):
            cmd.extend(["--batch_size", str(config["batch_size"])])

        if config.get("limit"):
            cmd.extend(["--limit", str(config["limit"])])

        if config.get("num_fewshot") is not None:
            cmd.extend(["--num_fewshot", str(config["num_fewshot"])])

        if config.get("gen_kwargs"):
            cmd.extend(["--gen_kwargs", str(config["gen_kwargs"])])

        if config.get("log_samples"):
            cmd.append("--log_samples")

        if config.get("predict_only"):
            cmd.append("--predict_only")

        return cmd

    @staticmethod
    async def _run_subprocess_with_log(
        cmd: list,
        log_path: Path,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> int:
        """Spawn ``cmd`` with stdout+stderr redirected to ``log_path``, await exit.

        The HTTP server's event loop never touches the subprocess pipe — the
        kernel writes directly to the log file — so the StreamReader buffer /
        backpressure / readline-separator failure class is gone by design.
        The try/finally still kills and reaps the whole subprocess process
        group on cancellation or unexpected exceptions so we never leak a
        GPU-resident process — the multi-GPU worker grandchildren an
        ``accelerate launch`` parent spawns are killed with it, not orphaned.

        ``extra_env`` is merged on top of ``os.environ`` + the forced
        ``PYTHONUNBUFFERED=1`` default so callers can layer their own env
        without touching the helper's contract.

        Distributed-launcher parents (e.g. an ``srun``-spawned process) export
        ``SLURM_PROCID`` / ``SLURM_NTASKS`` / ``RANK`` / ``WORLD_SIZE`` etc.
        into the eval child's env. Accelerate's ``PartialState`` then treats
        the child as a Slurm-launched distributed task and tries env://
        rendezvous, which raises ``ValueError: ... WORLD_SIZE expected, but
        not set`` because the relevant vars aren't fully wired up. Scrub
        those vars from the child env so the eval subprocess runs as a clean
        single-process job.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Force unbuffered child stdout so `tail -f` on the log file shows
        # progress in real time. Without this, Python's default block
        # buffering when stdout isn't a TTY hides output until the subprocess
        # exits, which makes the log useless for live debugging.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        for var in _SLURM_DISTRIBUTED_ENV_SCRUB:
            env.pop(var, None)
        if extra_env:
            env.update(extra_env)
        with log_path.open("wb") as log_fp:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_fp,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            try:
                await proc.wait()
                assert proc.returncode is not None
                return proc.returncode
            finally:
                if proc.returncode is None:
                    # start_new_session=True put the child in its own process
                    # group; SIGKILL the whole group so a multi-GPU launcher's
                    # GPU worker grandchildren are killed too instead of being
                    # orphaned holding GPU memory.
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except Exception:
                        # getpgid/killpg unavailable or failed for any other
                        # reason — fall back to killing the direct child.
                        proc.kill()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=10)
                    except (asyncio.TimeoutError, ProcessLookupError):
                        pass

    @staticmethod
    def _tail_log(log_path: Path, max_lines: int = 50, max_bytes: int = 256 * 1024) -> str:
        """Best-effort read of the last ``max_lines`` from ``log_path``."""
        try:
            with log_path.open("rb") as fp:
                fp.seek(0, 2)
                size = fp.tell()
                fp.seek(max(0, size - max_bytes))
                data = fp.read()
            text = data.decode(errors="replace")
            return "\n".join(text.splitlines()[-max_lines:])
        except OSError:
            return "(log file unreadable)"

    @staticmethod
    def _parse_output_directory(output_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Parse output directory: output_path/model_name/YYYYMMDD_HHMMSS_results.json

        Returns:
            {model_name: {"results": path, "samples": [paths]}}
        """
        output_dir = Path(output_path)
        if not output_dir.exists():
            return {}

        result = {}

        for model_dir in output_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # Group files by timestamp
            timestamps = defaultdict(lambda: {"results": None, "samples": []})

            for file in model_dir.glob("*_results.json"):
                timestamp = file.name.split("_results.json")[0]
                timestamps[timestamp]["results"] = str(file)

            for file in model_dir.glob("*_samples_*.jsonl"):
                # Extract timestamp (everything before first _samples_)
                timestamp = file.name.split("_samples_")[0]
                timestamps[timestamp]["samples"].append(str(file))

            if not timestamps:
                continue

            # Use latest timestamp
            sorted_ts = sorted(timestamps.keys(), reverse=True)
            if len(sorted_ts) > 1:
                logger.warning(f"Multiple timestamps for '{model_name}': {sorted_ts}. Using latest.")

            result[model_name] = timestamps[sorted_ts[0]]

        return result
