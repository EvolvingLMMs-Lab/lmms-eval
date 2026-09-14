"""Strict launcher contracts exercised through the actual upstream CLI boundary."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from lmms_eval import __main__ as cli
from lmms_eval.models.model_utils.grt import worker


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    """Replace hardware/evaluation only; keep cli_evaluate and its parser real."""
    for name in (*worker._PROCESS_SIZE_ENV, *worker._PROCESS_RANK_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    tensor_runtime = mock.MagicMock()
    tensor_runtime.cuda.is_available.return_value = True
    tensor_runtime.cuda.device_count.return_value = 1
    tensor_runtime.cuda.get_device_name.return_value = "mock-single-gpu"
    tensor_runtime.cuda.get_device_properties.return_value.uuid = "mock-single-uuid"
    tensor_runtime.version.cuda = "mock-cuda"
    tensor_runtime.distributed.is_available.return_value = True
    tensor_runtime.distributed.is_initialized.return_value = False
    tensor_runtime.distributed.get_world_size.return_value = 1
    monkeypatch.setitem(sys.modules, "torch", tensor_runtime)
    # The imported CLI keeps the real torch module. Its Accelerator is replaced
    # to avoid initializing GPUs; actual argument parsing and exception handling
    # still execute below.
    accelerator = mock.MagicMock()
    accelerator.is_main_process = True
    monkeypatch.setattr(cli, "Accelerator", mock.Mock(return_value=accelerator))
    monkeypatch.setattr(cli, "eval_logger", mock.Mock())
    monkeypatch.setattr(worker, "version", lambda _: "mock-version")
    output = tmp_path / "new-run"
    argv = ["grt-worker", "--model", "grt_llava_hf", "--tasks", "dive_bench_educational_high_fps", "--output_path", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate = mock.Mock(return_value=(None, None))
    monkeypatch.setattr(cli, "cli_evaluate_single", evaluate)
    return SimpleNamespace(torch=tensor_runtime, output=output, argv=argv, evaluate=evaluate, accelerator=accelerator)


@pytest.mark.parametrize("name", worker._PROCESS_SIZE_ENV)
@pytest.mark.parametrize("value", ["2", "0", "invalid"])
def test_worker_rejects_distributed_size_before_evaluation(boundary, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="single process"):
        worker.main()
    boundary.evaluate.assert_not_called()
    boundary.torch.cuda.is_available.assert_not_called()
    assert not boundary.output.exists()


@pytest.mark.parametrize("name", worker._PROCESS_RANK_ENV)
def test_worker_rejects_nonzero_rank_before_evaluation(boundary, monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(RuntimeError, match="single process"):
        worker.main()
    boundary.evaluate.assert_not_called()
    assert not boundary.output.exists()


def test_worker_rejects_initialized_multiprocess_group(boundary):
    boundary.torch.distributed.is_initialized.return_value = True
    boundary.torch.distributed.get_world_size.return_value = 2
    with pytest.raises(RuntimeError, match="initialized distributed"):
        worker.main()
    boundary.evaluate.assert_not_called()
    assert not boundary.output.exists()


@pytest.mark.parametrize("extra", [["--config", "override.yaml"], ["--config=override.yaml"]])
def test_worker_rejects_config_environment_overrides(boundary, extra):
    boundary.argv.extend(extra)
    with pytest.raises(RuntimeError, match="explicit profile flags"):
        worker.main()
    boundary.evaluate.assert_not_called()
    assert not boundary.output.exists()


@pytest.mark.parametrize("verbosity", [None, "INFO", "WARNING", "DEBUG"])
def test_actual_upstream_cli_propagates_evaluation_failure(boundary, verbosity):
    """Regression: upstream INFO otherwise catches this and returns exit zero."""
    if verbosity is not None:
        boundary.argv.extend(["--verbosity", verbosity])
    boundary.evaluate.side_effect = RuntimeError("simulated CUDA determinism or data failure")
    with pytest.raises(RuntimeError, match="simulated CUDA determinism or data failure"):
        worker.main()
    boundary.evaluate.assert_called_once()
    assert boundary.evaluate.call_args.args[0].verbosity == "DEBUG"
    boundary.accelerator.state.destroy_process_group.assert_called_once()
    assert not boundary.output.exists()


def test_single_process_uses_native_cli_and_preserves_explicit_flags(boundary, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "-1")
    boundary.argv.extend(["--seed", "0,1234,1234,1234", "--gen_kwargs", "max_new_tokens=128,temperature=0"])
    worker.main()
    boundary.evaluate.assert_called_once()
    args = boundary.evaluate.call_args.args[0]
    assert args.model == "grt_llava_hf"
    assert args.tasks == "dive_bench_educational_high_fps"
    assert args.output_path == str(boundary.output)
    assert args.seed == [0, 1234, 1234, 1234]
    assert args.gen_kwargs == "max_new_tokens=128,temperature=0"
    assert args.verbosity == "DEBUG"
    assert not boundary.output.exists()
