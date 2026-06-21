from __future__ import annotations

from lmms_eval.agentic.loop.session import LoopSession
from lmms_eval.agentic.model_server import ModelServer, RolloutJob
from lmms_eval.agentic.types import (
    AgentInput,
    AgentOutput,
    ContentBlock,
    EnvState,
    EpisodeResult,
)


class _BatchServer(ModelServer):
    def __init__(self, max_parallel_rollouts=2):
        self._max_parallel_rollouts = max_parallel_rollouts
        self.batches = []

    def generate(self, request):
        return self.generate_batch([request])[0]

    def generate_batch(self, requests):
        self.batches.append([request.first_text() for request in requests])
        return [AgentOutput(content=[ContentBlock.text(f"out:{request.first_text()}")]) for request in requests]


class _Session(LoopSession):
    def __init__(self, name, max_steps=2):
        self.name = name
        self.max_steps = max_steps
        self.step_idx = 0
        self.outputs = []

    @property
    def done(self):
        return self.step_idx >= self.max_steps

    def next_request(self):
        if self.done:
            return None
        return AgentInput(content=[ContentBlock.text(f"{self.name}:{self.step_idx}")])

    def apply_model_output(self, raw_output):
        self.outputs.append(raw_output.first_text())
        self.step_idx += 1

    def result(self):
        return EpisodeResult(
            final_state=EnvState(env_id=self.name, step_idx=self.step_idx, observation={}, terminal=True),
            metadata={"outputs": self.outputs},
        )


def test_model_server_schedules_loop_sessions_through_generate_batch():
    server = _BatchServer(max_parallel_rollouts=2)
    jobs = [RolloutJob(index=idx, make_session=lambda _server, idx=idx: _Session(f"job{idx}"), run_serial=lambda _server: None) for idx in range(3)]

    results = server.run_rollouts(jobs)

    assert server.batches == [
        ["job0:0", "job1:0"],
        ["job0:1", "job1:1"],
        ["job2:0"],
        ["job2:1"],
    ]
    assert [result.final_state.env_id for result in results] == ["job0", "job1", "job2"]
