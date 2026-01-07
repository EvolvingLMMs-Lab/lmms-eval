"""
Job Scheduler for LMMS-Eval Server.

This module provides a thread-safe job scheduler that manages evaluation jobs
with queue-based execution. Jobs are processed sequentially to ensure proper
GPU resource management.
"""

import asyncio
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

# =============================================================================
# Enums and Models
# =============================================================================


class JobStatus(str, Enum):
    """Status of an evaluation job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluateRequest(BaseModel):
    """Request model for evaluation endpoint."""

    model: str = Field(..., description="Model name or path")
    tasks: List[str] = Field(..., description="List of task names to evaluate")
    model_args: Optional[Dict[str, Any]] = Field(default=None, description="Model arguments")
    num_fewshot: Optional[int] = Field(default=None, description="Number of few-shot examples")
    batch_size: Optional[Union[int, str]] = Field(default=None, description="Batch size")
    device: Optional[str] = Field(default=None, description="Device to run on")
    limit: Optional[Union[int, float]] = Field(default=None, description="Limit number of examples")
    gen_kwargs: Optional[str] = Field(default=None, description="Generation kwargs")
    log_samples: bool = Field(default=True, description="Whether to log samples")
    predict_only: bool = Field(default=False, description="Only generate predictions")
    num_gpus: int = Field(default=1, description="Number of GPUs to use")
    output_dir: Optional[str] = Field(default=None, description="Output directory for results")


class JobInfo(BaseModel):
    """Information about a job."""

    job_id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    request: EvaluateRequest
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    position_in_queue: Optional[int] = None


class JobSubmitResponse(BaseModel):
    """Response when submitting a job."""

    job_id: str
    status: JobStatus
    position_in_queue: int
    message: str


class QueueStatusResponse(BaseModel):
    """Response for queue status."""

    queue_size: int
    running_job: Optional[str] = None
    queued_jobs: List[str]
    completed_jobs: int
    failed_jobs: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    queue_size: int


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
        Run evaluation in a subprocess using accelerate/torchrun.

        This allows GPU-based evaluation to run in a separate process
        while the server remains responsive.
        """
        output_path = config.get("output_dir") or tempfile.mkdtemp(prefix=self._temp_dir_prefix)

        # Build command
        num_gpus = config.get("num_gpus", 1)
        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            str(num_gpus),
            "-m",
            "lmms_eval",
            "--model",
            config["model"],
            "--tasks",
            ",".join(config["tasks"]),
            "--output_path",
            output_path,
        ]

        # Add optional arguments
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

        # Run subprocess with streaming output
        logger.info(f"[EVAL] Launching: {' '.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            logger.info(f"[EVAL] {line.decode().rstrip()}")

        await proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"Evaluation failed with return code {proc.returncode}")

        return self._parse_output_directory(output_path)

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
