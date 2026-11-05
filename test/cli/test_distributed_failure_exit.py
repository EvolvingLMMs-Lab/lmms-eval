"""A failed rank in a distributed run must exit nonzero without process-group teardown."""

import sys

import pytest

import lmms_eval.__main__ as cli


class _Exited(BaseException):
    def __init__(self, code, destroyed):
        self.code = code
        self.destroyed = destroyed


class _FakeAccelerator:
    destroyed = False

    def __init__(self, *args, **kwargs):
        self.is_main_process = True
        self.state = self

    def destroy_process_group(self):
        _FakeAccelerator.destroyed = True


def test_failed_rank_exits_without_destroying_the_process_group(monkeypatch):
    def fail(args):
        raise RuntimeError("rank failed")

    def exit_now(code):
        # os._exit never runs finally blocks; record teardown state at the exit call.
        raise _Exited(code, _FakeAccelerator.destroyed)

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(sys, "argv", ["lmms_eval", "--model", "dummy", "--tasks", "dummy"])
    monkeypatch.setattr(cli, "Accelerator", _FakeAccelerator)
    monkeypatch.setattr(cli, "cli_evaluate_single", fail)
    monkeypatch.setattr(cli.os, "_exit", exit_now)
    _FakeAccelerator.destroyed = False

    with pytest.raises(_Exited) as exited:
        cli.cli_evaluate()

    assert exited.value.code == 1
    assert not exited.value.destroyed
