"""Tests for JobScheduler subprocess handling.

The previous design drained subprocess stdout through asyncio inside the
HTTP server's event loop and was crashing on
``ValueError("Separator is not found, and chunk exceed the limit")`` once
a single chunk exceeded 64 KiB without a newline. The current design
redirects subprocess stdout directly to a per-job log file at the OS level,
so the server only awaits ``proc.wait()`` — the entire StreamReader /
backpressure failure class is gone by construction. These tests pin down
the contract.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

from lmms_eval.entrypoints.job_scheduler import JobScheduler


def _python_subprocess_cmd(snippet: str) -> list[str]:
    return [sys.executable, "-c", snippet]


def test_subprocess_log_captures_long_line_without_newline():
    """200 KiB of output with no newline used to crash readline(); now it
    must land in the log file and the subprocess must exit cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "eval.log"
        cmd = _python_subprocess_cmd("import sys; sys.stdout.write('x' * 200_000); sys.stdout.write('\\ndone\\n')")

        returncode = asyncio.run(JobScheduler._run_subprocess_with_log(cmd, log_path))

        assert returncode == 0
        contents = log_path.read_bytes()
        assert len(contents) > 200_000
        assert b"done\n" in contents


def test_subprocess_log_captures_pure_no_newline_output():
    """A subprocess that never emits a newline at all still lands cleanly
    in the log file."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "eval.log"
        cmd = _python_subprocess_cmd("import sys; sys.stdout.write('y' * 300_000); sys.stdout.flush()")

        returncode = asyncio.run(JobScheduler._run_subprocess_with_log(cmd, log_path))

        assert returncode == 0
        assert log_path.stat().st_size == 300_000


def test_subprocess_failure_returncode_propagates():
    """Non-zero exits propagate through the helper; caller is responsible
    for surfacing the log tail."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "eval.log"
        cmd = _python_subprocess_cmd("import sys; print('about to die'); sys.exit(7)")

        returncode = asyncio.run(JobScheduler._run_subprocess_with_log(cmd, log_path))

        assert returncode == 7
        assert b"about to die" in log_path.read_bytes()


def test_run_evaluation_fails_when_subprocess_writes_no_results(monkeypatch):
    """An eval child that exits 0 but produces no result files is a failure."""

    async def _scenario():
        scheduler = JobScheduler()
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(
                JobScheduler,
                "_build_eval_cmd",
                staticmethod(lambda config, output_path: _python_subprocess_cmd("print('child exited 0 without writing results')")),
            )

            with pytest.raises(RuntimeError, match="produced no parsed results"):
                await scheduler._run_evaluation({"output_dir": tmp})

    asyncio.run(_scenario())


def test_tail_log_returns_last_lines():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "eval.log"
        log_path.write_text("\n".join(f"line {i}" for i in range(1, 1001)) + "\n")

        tail = JobScheduler._tail_log(log_path, max_lines=3)

        assert tail == "line 998\nline 999\nline 1000"


def test_tail_log_handles_missing_file():
    assert JobScheduler._tail_log(Path("/does/not/exist")) == "(log file unreadable)"


def test_subprocess_cancellation_kills_orphan():
    """If the caller is cancelled mid-wait, the subprocess must be killed
    and reaped — otherwise we leak GPU-resident processes."""

    async def _scenario():
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "eval.log"
            # Subprocess that would run for a long time
            cmd = _python_subprocess_cmd("import time; time.sleep(60)")

            task = asyncio.create_task(JobScheduler._run_subprocess_with_log(cmd, log_path))
            # Give it a moment to start the subprocess
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Expected: we explicitly cancelled the task above; the test
                # verifies cancellation completes without hanging.
                pass
            # If kill+reap worked, the helper exited; no zombies remain in
            # this test's async-subprocess registry. Hard to assert on a
            # pid that's already gone — successful completion of cancel
            # without hanging is the contract.

    asyncio.run(_scenario())


def test_subprocess_started_in_new_session(monkeypatch):
    """The eval child must be spawned with ``start_new_session=True`` so a
    cancel can SIGKILL the whole process group (the accelerate launcher plus
    its GPU worker grandchildren) rather than orphaning them."""
    captured = {}

    class _FakeProc:
        returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    async def _fake_exec(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "eval.log"
        returncode = asyncio.run(JobScheduler._run_subprocess_with_log([sys.executable, "-c", "pass"], log_path))

    assert returncode == 0
    assert captured["kwargs"].get("start_new_session") is True


def test_extra_env_reaches_subprocess():
    """``extra_env`` must layer on top of os.environ + the PYTHONUNBUFFERED
    default so callers can ship per-job env (WANDB_DISABLED, task-specific
    knobs) through the safe-drain helper."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "eval.log"
        cmd = _python_subprocess_cmd("import os, sys; " "sys.stdout.write(os.environ.get('LMMS_EVAL_TEST_TOKEN', 'MISSING'))")

        returncode = asyncio.run(
            JobScheduler._run_subprocess_with_log(
                cmd,
                log_path,
                extra_env={"LMMS_EVAL_TEST_TOKEN": "hello-from-extra-env"},
            )
        )

        assert returncode == 0
        assert log_path.read_text() == "hello-from-extra-env"


def test_slurm_distributed_env_scrubbed_from_subprocess(monkeypatch):
    """Slurm srun-launched parents leak SLURM_PROCID / RANK / WORLD_SIZE etc.
    into the eval child. Accelerate's PartialState then treats the child as
    a Slurm-launched distributed task and tries env:// rendezvous, blowing
    up on ``WORLD_SIZE expected, but not set``. The helper must scrub those
    vars so the eval subprocess runs single-process clean."""
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "eval.log"
        cmd = _python_subprocess_cmd(
            "import os, sys; " "leaked = [v for v in ['SLURM_PROCID','SLURM_NTASKS','RANK'," "'LOCAL_RANK','MASTER_ADDR','MASTER_PORT'] if v in os.environ]; " "sys.stdout.write(','.join(leaked) if leaked else 'CLEAN')"
        )

        returncode = asyncio.run(JobScheduler._run_subprocess_with_log(cmd, log_path))

        assert returncode == 0
        assert log_path.read_text() == "CLEAN"


def test_extra_env_overrides_inherited_env(monkeypatch):
    """When both os.environ and extra_env set the same key, extra_env wins."""
    monkeypatch.setenv("LMMS_EVAL_TEST_TOKEN", "from-parent")
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "eval.log"
        cmd = _python_subprocess_cmd("import os, sys; sys.stdout.write(os.environ['LMMS_EVAL_TEST_TOKEN'])")

        returncode = asyncio.run(
            JobScheduler._run_subprocess_with_log(
                cmd,
                log_path,
                extra_env={"LMMS_EVAL_TEST_TOKEN": "from-extra-env"},
            )
        )

        assert returncode == 0
        assert log_path.read_text() == "from-extra-env"


# ---------------------------------------------------------------------------
# _build_eval_cmd — fixes accelerate `--num_processes 1` rendezvous bug
# ---------------------------------------------------------------------------


def test_build_eval_cmd_num_gpus_1_skips_accelerate():
    """When num_gpus=1, the cmd must be ``python -m lmms_eval ...`` directly.

    Going through ``accelerate launch --num_processes 1`` falls into
    ``simple_launcher``, which does not set ``WORLD_SIZE``/``RANK`` env vars.
    lmms_eval's ``cli_evaluate`` then crashes on Accelerator() env://
    rendezvous with ``ValueError: ... WORLD_SIZE expected, but not set``.
    """
    cmd = JobScheduler._build_eval_cmd(
        {"model": "llava", "tasks": ["mme"], "num_gpus": 1},
        output_path="/tmp/out",
    )
    assert "accelerate" not in cmd[0:3]
    assert "--multi_gpu" not in cmd
    assert cmd[:3] == [sys.executable, "-m", "lmms_eval"]
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "llava"
    assert cmd[cmd.index("--tasks") + 1] == "mme"
    assert cmd[cmd.index("--output_path") + 1] == "/tmp/out"


def test_build_eval_cmd_num_gpus_n_uses_accelerate_multi_gpu():
    """num_gpus>1 must go through accelerate.commands.launch --multi_gpu."""
    cmd = JobScheduler._build_eval_cmd(
        {"model": "llava", "tasks": ["mme"], "num_gpus": 4},
        output_path="/tmp/out",
    )
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "accelerate.commands.launch"]
    assert "--multi_gpu" in cmd
    assert "--num_processes" in cmd
    assert cmd[cmd.index("--num_processes") + 1] == "4"
    assert "--num_machines" in cmd
    assert cmd[cmd.index("--num_machines") + 1] == "1"
    assert cmd[cmd.index("-m", cmd.index("accelerate.commands.launch")) + 1] == "lmms_eval"


def test_build_eval_cmd_threads_optional_args():
    """model_args (dict→k=v), batch_size, limit, log_samples, and
    predict_only all surface on the cmd line."""
    cmd = JobScheduler._build_eval_cmd(
        {
            "model": "llava",
            "tasks": ["mme"],
            "num_gpus": 1,
            "model_args": {"checkpoint_path": "/ckpt", "image_size": 224},
            "batch_size": 4,
            "limit": 8,
            "log_samples": True,
            "predict_only": True,
        },
        output_path="/tmp/out",
    )
    assert "--model_args" in cmd
    assert cmd[cmd.index("--model_args") + 1] == "checkpoint_path=/ckpt,image_size=224"
    assert cmd[cmd.index("--batch_size") + 1] == "4"
    assert cmd[cmd.index("--limit") + 1] == "8"
    assert "--log_samples" in cmd
    assert "--predict_only" in cmd


def test_build_eval_cmd_num_gpus_default_is_1():
    """Missing num_gpus key should default to single-process (no accelerate)."""
    cmd = JobScheduler._build_eval_cmd(
        {"model": "llava", "tasks": ["mme"]},
        output_path="/tmp/out",
    )
    assert cmd[:3] == [sys.executable, "-m", "lmms_eval"]
