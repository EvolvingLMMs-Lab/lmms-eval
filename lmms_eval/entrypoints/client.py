"""
Client for interacting with the LMMS-Eval HTTP server.

Example usage:
    >>> from lmms_eval.entrypoints import EvalClient
    >>> client = EvalClient("http://localhost:8000")
    >>>
    >>> # Submit an evaluation job
    >>> job = client.evaluate(
    ...     model="qwen_vl",
    ...     tasks=["mmmu_val"],
    ...     model_args={"pretrained": "Qwen/Qwen2-VL-7B-Instruct"},
    ... )
    >>> print(f"Job submitted: {job['job_id']}")
    >>>
    >>> # Wait for results
    >>> result = client.wait_for_job(job['job_id'])
    >>> print(result)
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import httpx

# Type aliases for context manager exit args
ExcInfo = tuple[type[BaseException], BaseException, Any] | tuple[None, None, None]


def _process_job_status(job: Dict[str, Any], job_id: str, verbose: bool) -> tuple[bool, Dict[str, Any] | None]:
    """
    Process job status and handle terminal states.

    Returns:
        Tuple of (should_continue, result_or_none).
        If should_continue is False, result_or_none contains the job dict.
        Raises RuntimeError if job failed.
    """
    status = job.get("status")

    if verbose:
        if status == "queued":
            pos = job.get("position_in_queue", "?")
            print(f"Job {job_id[:8]}... queued (position: {pos})")
        elif status == "running":
            print(f"Job {job_id[:8]}... running")

    if status == "completed":
        if verbose:
            print(f"Job {job_id[:8]}... completed!")
        return False, job

    if status == "failed":
        error = job.get("error", "Unknown error")
        raise RuntimeError(f"Job failed: {error}")

    return True, None


class EvalClient:
    """
    Python client for the LMMS-Eval HTTP server.

    Provides a convenient interface for submitting evaluation jobs,
    checking status, and retrieving results.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: Optional[float] = None,
    ):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the evaluation server.
            timeout: Request timeout in seconds. None for no timeout.
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def __enter__(self) -> "EvalClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __del__(self):
        """Ensure client is closed on garbage collection."""
        try:
            self.client.close()
        except Exception:
            pass

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request and return JSON response."""
        url = f"{self.base_url}{endpoint}"
        response = self.client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Health & Info
    # =========================================================================

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        return self._request("GET", "/health")

    def is_healthy(self) -> bool:
        """Return True if server is healthy."""
        try:
            health = self.health()
            return health.get("status") == "healthy"
        except Exception:
            return False

    def list_tasks(self) -> List[str]:
        """List available evaluation tasks."""
        response = self._request("GET", "/tasks")
        return response.get("tasks", [])

    def list_models(self) -> List[str]:
        """List available model types."""
        response = self._request("GET", "/models")
        return response.get("models", [])

    # =========================================================================
    # Evaluation Jobs
    # =========================================================================

    def evaluate(
        self,
        model: str,
        tasks: List[str],
        model_args: Optional[Dict[str, Any]] = None,
        num_fewshot: Optional[int] = None,
        batch_size: Optional[Union[int, str]] = None,
        device: Optional[str] = None,
        limit: Optional[Union[int, float]] = None,
        gen_kwargs: Optional[str] = None,
        log_samples: bool = True,
        predict_only: bool = False,
        num_gpus: int = 1,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit an evaluation job.

        Args:
            model: Model name (e.g., "qwen_vl", "llava")
            tasks: List of task names to evaluate
            model_args: Model-specific arguments
            num_fewshot: Number of few-shot examples
            batch_size: Batch size for evaluation
            device: Device to run on (e.g., "cuda:0")
            limit: Limit number of examples (for testing)
            gen_kwargs: Generation kwargs string
            log_samples: Whether to log individual samples
            predict_only: Only generate predictions, skip metrics
            num_gpus: Number of GPUs to use
            output_dir: Output directory for results

        Returns:
            Dict with job_id, status, position_in_queue, message
        """
        payload = {
            "model": model,
            "tasks": tasks,
            "model_args": model_args,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "device": device,
            "limit": limit,
            "gen_kwargs": gen_kwargs,
            "log_samples": log_samples,
            "predict_only": predict_only,
            "num_gpus": num_gpus,
            "output_dir": output_dir,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        return self._request("POST", "/evaluate", json=payload)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status and results.

        Args:
            job_id: The job ID returned from evaluate()

        Returns:
            Dict with job info including status, result, error, etc.
        """
        return self._request("GET", f"/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a queued job.

        Args:
            job_id: The job ID to cancel

        Returns:
            Dict with cancellation message
        """
        return self._request("DELETE", f"/jobs/{job_id}")

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete and return results.

        Args:
            job_id: The job ID to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None for no timeout)
            verbose: Print status updates

        Returns:
            Dict with job info including results

        Raises:
            TimeoutError: If timeout is reached
            RuntimeError: If job fails
        """
        start_time = time.time()

        while True:
            job = self.get_job(job_id)
            should_continue, result = _process_job_status(job, job_id, verbose)
            if not should_continue:
                return result  # type: ignore[return-value]

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for job {job_id}")

            time.sleep(poll_interval)

    # =========================================================================
    # Queue Management
    # =========================================================================

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get queue status.

        Returns:
            Dict with queue_size, running_job, queued_jobs, etc.
        """
        return self._request("GET", "/queue")


class AsyncEvalClient:
    """
    Async version of the LMMS-Eval client.

    Example:
        >>> async with AsyncEvalClient() as client:
        ...     job = await client.evaluate(model="qwen_vl", tasks=["mmmu_val"])
        ...     result = await client.wait_for_job(job["job_id"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: Optional[float] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> "AsyncEvalClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    async def close(self):
        await self.client.aclose()

    def __del__(self):
        """Ensure client is closed on garbage collection."""
        try:
            # Note: This is sync cleanup for async client - best effort only
            if hasattr(self, "client") and not self.client.is_closed:
                import warnings

                warnings.warn(
                    "AsyncEvalClient was not properly closed. Use 'async with' or call 'await client.close()'.",
                    ResourceWarning,
                    stacklevel=2,
                )
        except Exception:
            pass

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        response = await self.client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/health")

    async def evaluate(
        self,
        model: str,
        tasks: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        payload = {"model": model, "tasks": tasks, **kwargs}
        payload = {k: v for k, v in payload.items() if v is not None}
        return await self._request("POST", "/evaluate", json=payload)

    async def get_job(self, job_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/jobs/{job_id}")

    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/jobs/{job_id}")

    async def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        start_time = time.time()

        while True:
            job = await self.get_job(job_id)
            should_continue, result = _process_job_status(job, job_id, verbose)
            if not should_continue:
                return result  # type: ignore[return-value]

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for job {job_id}")

            await asyncio.sleep(poll_interval)

    async def get_queue_status(self) -> Dict[str, Any]:
        return await self._request("GET", "/queue")
