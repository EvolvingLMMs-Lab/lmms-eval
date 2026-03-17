"""Slurm-aware progress reporting.

In interactive terminals, delegates to tqdm. In batch jobs (Slurm, piped output),
emits periodic structured log lines instead of carriage-return progress bars.

Usage (drop-in tqdm replacement):
    from lmms_eval.models.model_utils.progress import make_progress

    pbar = make_progress(total=len(requests), desc="Model Responding", disable=(rank != 0))
    for batch in batches:
        ...
        pbar.update(len(batch))
    pbar.close()
"""

import os
import sys
import time

from loguru import logger as eval_logger
from tqdm import tqdm


def _is_batch_mode() -> bool:
    """Detect non-interactive execution (Slurm, piped output, etc.)."""
    if os.environ.get("SLURM_JOB_ID"):
        return True
    if not sys.stderr.isatty():
        return True
    return False


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 0 or seconds != seconds:  # negative or NaN
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    return f"{h}h{m:02d}m"


class SlurmProgress:
    """Structured log-line progress for batch jobs.

    Instead of tqdm's carriage-return bar, emits periodic log lines like:
        [rank 0] Model Responding: 104/2837 (3.7%) | 1.52 it/s | ETA 30m02s
    """

    def __init__(
        self,
        total: int,
        desc: str = "Progress",
        disable: bool = False,
        log_interval: float = 30.0,
    ):
        self.total = total
        self.desc = desc
        self.disable = disable
        self.log_interval = log_interval
        self.n = 0
        self._start_time = time.monotonic()
        self._last_log_time = 0.0  # force first log after first update
        self._rank = int(os.environ.get("RANK", os.environ.get("SLURM_ARRAY_TASK_ID", 0)))

    def update(self, n: int = 1) -> None:
        self.n += n
        if self.disable:
            return
        now = time.monotonic()
        is_final = self.n >= self.total
        if not is_final and (now - self._last_log_time) < self.log_interval:
            return
        self._last_log_time = now
        elapsed = now - self._start_time
        speed = self.n / elapsed if elapsed > 0 else 0
        eta = (self.total - self.n) / speed if speed > 0 else 0
        pct = 100 * self.n / self.total if self.total > 0 else 0
        eval_logger.info(f"[rank {self._rank}] {self.desc}: {self.n}/{self.total} ({pct:.1f}%) | " f"{speed:.2f} it/s | elapsed {_format_time(elapsed)} | ETA {_format_time(eta)}")

    def close(self) -> None:
        if self.disable:
            return
        elapsed = time.monotonic() - self._start_time
        speed = self.n / elapsed if elapsed > 0 else 0
        eval_logger.info(f"[rank {self._rank}] {self.desc}: done {self.n}/{self.total} " f"in {_format_time(elapsed)} ({speed:.2f} it/s)")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def make_progress(
    total: int,
    desc: str = "Progress",
    disable: bool = False,
    log_interval: float = 30.0,
) -> "tqdm | SlurmProgress":
    """Create a progress tracker — tqdm for interactive, SlurmProgress for batch.

    Args:
        total: Total number of items.
        desc: Description shown in progress output.
        disable: If True, suppress all output (e.g., non-rank-0 workers).
        log_interval: Seconds between log lines in batch mode (default 30s).
    """
    if _is_batch_mode():
        return SlurmProgress(total=total, desc=desc, disable=disable, log_interval=log_interval)
    return tqdm(total=total, desc=desc, disable=disable)
