import os
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Optional


def with_temp_dir(test_func):
    """Decorator that creates a temporary directory for tests and cleans up after."""

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Pass the temporary directory to the test function
            test_func(*args, temp_dir=temp_dir, **kwargs)
        finally:
            # Clean up the directory after the test
            shutil.rmtree(temp_dir, ignore_errors=True)

    return wrapper


def get_available_gpus():
    """Get the number of available GPUs."""
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def get_gpu_count():
    """
    Get the GPU count for testing.

    First checks the TEST_GPU_COUNT environment variable (set by run_cicd.py),
    then falls back to detecting available GPUs via CUDA_VISIBLE_DEVICES or torch.

    Returns:
        Number of GPUs to use for testing
    """
    # Check for explicit test GPU count setting
    test_gpu_count = os.environ.get("TEST_GPU_COUNT")
    if test_gpu_count:
        return int(test_gpu_count)

    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        # Count the number of GPUs in CUDA_VISIBLE_DEVICES
        gpus = [g.strip() for g in cuda_visible.split(",") if g.strip()]
        if gpus:
            return len(gpus)

    # Fall back to detecting available GPUs
    return get_available_gpus() or 1


def find_free_port():
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def wait_for_server(host: str, port: int, timeout: float = 60.0) -> bool:
    """
    Wait for the server to become available.

    Args:
        host: Server host
        port: Server port
        timeout: Maximum seconds to wait

    Returns:
        True if server is available, False if timeout
    """
    import httpx

    start_time = time.time()
    url = f"http://{host}:{port}/health"

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            pass
        time.sleep(1.0)

    return False


class ServerProcess:
    """
    Manages the evaluation server process lifecycle.

    Usage:
        server = ServerProcess(port=8000)
        server.start()
        try:
            # run tests
        finally:
            server.stop()
    """

    def __init__(self, host: str = "localhost", port: Optional[int] = None):
        self.host = host
        self.port = port or find_free_port()
        self.process: Optional[subprocess.Popen] = None
        self._output_lines = []
        self._log_thread: Optional[threading.Thread] = None
        self._stop_logging = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _stream_logs(self):
        """Background thread to stream server logs."""
        while not self._stop_logging.is_set():
            if self.process is None or self.process.stdout is None:
                break

            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.rstrip()
                    self._output_lines.append(line)
                    print(f"[SERVER] {line}")
                elif self.process.poll() is not None:
                    # Process has ended
                    break
            except Exception:
                break

    def start(self, timeout: float = 60.0) -> bool:
        """
        Start the evaluation server.

        Args:
            timeout: Maximum seconds to wait for server to be ready

        Returns:
            True if server started successfully
        """
        if self.process is not None:
            raise RuntimeError("Server already started")

        # Launch server process
        cmd = [
            "python",
            "-m",
            "lmms_eval.launch_server",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        print(f"[SERVER] Launching server: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        # Start background thread to stream logs
        self._stop_logging.clear()
        self._log_thread = threading.Thread(target=self._stream_logs, daemon=True)
        self._log_thread.start()

        # Wait for server to be ready
        print(f"[SERVER] Waiting for server at {self.url}...")
        if wait_for_server(self.host, self.port, timeout=timeout):
            print(f"[SERVER] Server is ready at {self.url}")
            return True
        else:
            # Server didn't start - clean up
            self.stop()
            raise RuntimeError(f"Server failed to start within {timeout} seconds")

    def stop(self):
        """Stop the evaluation server and clean up."""
        if self.process is None:
            return

        print("[SERVER] Stopping server...")

        # Signal the log thread to stop
        self._stop_logging.set()

        try:
            # Kill the entire process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # Process already dead

        # Wait for process to terminate
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()

        # Wait for log thread to finish
        if self._log_thread is not None:
            self._log_thread.join(timeout=2)
            self._log_thread = None

        self.process = None
        print("[SERVER] Server stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


@contextmanager
def managed_server(host: str = "localhost", port: Optional[int] = None):
    """
    Context manager for running a server during tests.

    Usage:
        with managed_server() as server:
            client = EvalClient(server.url)
            # run tests
    """
    server = ServerProcess(host=host, port=port)
    try:
        server.start()
        yield server
    finally:
        server.stop()


def with_server(test_func):
    """
    Decorator that starts a server before the test and stops it after.

    The test function receives a 'server' keyword argument with the ServerProcess instance.
    """

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        with managed_server() as server:
            test_func(*args, server=server, **kwargs)

    return wrapper


class TestResult:
    """Container for test execution results."""

    def __init__(self, success: bool, job_result: dict = None, error: str = None):
        self.success = success
        self.job_result = job_result
        self.error = error


def run_evaluation_test(
    server_url: str,
    model: str,
    tasks: list,
    model_args: dict,
    batch_size: int = 4,
    limit: int = 4,
    num_gpus: int = 1,
    timeout: float = 600.0,
) -> TestResult:
    """
    Helper function to run an evaluation test.

    Args:
        server_url: URL of the evaluation server
        model: Model name
        tasks: List of task names
        model_args: Model arguments
        batch_size: Batch size
        limit: Number of examples to evaluate
        num_gpus: Number of GPUs to use
        timeout: Maximum seconds to wait for job completion

    Returns:
        TestResult with success status and results
    """
    from lmms_eval.entrypoints import EvalClient

    with tempfile.TemporaryDirectory() as output_dir:
        client = EvalClient(server_url)

        try:
            # Submit job
            print(f"[TEST] Submitting evaluation job: {model} on {tasks}")
            job = client.evaluate(
                model=model,
                tasks=tasks,
                model_args=model_args,
                batch_size=batch_size,
                limit=limit,
                log_samples=True,
                num_gpus=num_gpus,
                output_dir=output_dir,
            )
            print(f"[TEST] Job submitted: {job['job_id']}")

            # Wait for completion
            result = client.wait_for_job(
                job["job_id"],
                poll_interval=5.0,
                timeout=timeout,
                verbose=True,
            )

            return TestResult(
                success=result.get("status") == "completed",
                job_result=result,
            )

        except Exception as e:
            return TestResult(success=False, error=str(e))
        finally:
            client.close()
