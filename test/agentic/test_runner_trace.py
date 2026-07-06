from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmms_eval.agentic.loop.runner import _episode_to_json, _write_episode_artifacts
from lmms_eval.agentic.types import (
    AgentInput,
    AgentOutput,
    ContentBlock,
    EnvState,
    EpisodeResult,
    EpisodeStep,
    GameAction,
    ParsedAction,
    StepResult,
)


class _TensorLike:
    def __init__(self, shape=(2, 3), dtype="float32"):
        self.shape = shape
        self.dtype = dtype


def _episode() -> EpisodeResult:
    state_before = EnvState(
        env_id="trace-env",
        step_idx=0,
        observation={"position": [0, 0], "frame": b"abc"},
        active_agent_ids=["agent"],
        metadata={"seed": 7},
    )
    state_after = EnvState(
        env_id="trace-env",
        step_idx=1,
        observation={"position": [1, 0]},
        terminal=True,
        active_agent_ids=[],
        metadata={"success": True, "metrics": {"score": 1.0}},
    )
    request = AgentInput(
        content=[
            ContentBlock.text("Observation: ..."),
            ContentBlock(type="state_features", data=state_before.observation, metadata={"agent_id": "agent"}),
        ],
        generation_kwargs={"temperature": 0},
        metadata={"agent_id": "agent"},
    )
    raw_output = AgentOutput(content=[ContentBlock.text("<think>left</think>\nMOVE_FORWARD")], metadata={"backend": "test"})
    output = AgentOutput(content=[ContentBlock.text("MOVE_FORWARD")], metadata={"raw_text": raw_output.first_text()})
    parsed_action = ParsedAction(action=GameAction(type="MOVE_FORWARD", data={"speed": 1}, agent_id="agent"), metadata={"source": "text"})
    step_result = StepResult(state=state_after, reward=1.0, done=True, info={"action": "MOVE_FORWARD"})

    return EpisodeResult(
        final_state=state_after,
        steps=[
            EpisodeStep(
                state=state_before,
                request=request,
                raw_output=raw_output,
                output=output,
                parsed_action=parsed_action,
                result=step_result,
            )
        ],
        success=True,
        metrics={"score": 1.0},
        metadata={"max_steps": 6},
    )


def test_episode_json_basic_keeps_compact_step_shape():
    payload = json.loads(_episode_to_json(_episode()))
    step = payload["steps"][0]

    assert "agentic_trace_mode" not in payload
    assert step["raw_model_output"] == "<think>left</think>\nMOVE_FORWARD"
    assert step["model_output"] == "MOVE_FORWARD"
    assert step["action"]["type"] == "MOVE_FORWARD"
    assert "request" not in step
    assert "state_before" not in step


def test_episode_json_full_records_rollout_io_and_state():
    payload = json.loads(_episode_to_json(_episode(), trace_mode="full"))
    step = payload["steps"][0]

    assert payload["agentic_trace_mode"] == "full"
    assert payload["rollout"]["step_count"] == 1
    assert step["state_before"]["active_agent_ids"] == ["agent"]
    assert step["state_before"]["observation"]["frame"] == {"type": "bytes", "length": 3}
    assert step["request"]["first_text"] == "Observation: ..."
    assert step["raw_output"]["first_text"] == "<think>left</think>\nMOVE_FORWARD"
    assert step["output"]["first_text"] == "MOVE_FORWARD"
    assert step["parsed_action"]["action"]["type"] == "MOVE_FORWARD"
    assert step["result"]["state_after"]["step_idx"] == 1


def test_episode_json_summarizes_arbitrary_parser_payloads():
    state_before = EnvState(env_id="trace-env", step_idx=0, observation={})
    state_after = EnvState(env_id="trace-env", step_idx=1, observation={}, terminal=True)
    tensor = _TensorLike()
    episode = EpisodeResult(
        final_state=state_after,
        steps=[
            EpisodeStep(
                state=state_before,
                request=tensor,
                raw_output={"policy_logits": tensor},
                output={"action_scores": tensor},
                parsed_action=ParsedAction(action={"agent": tensor}),
                result=StepResult(state=state_after, reward=0.0, done=True),
            )
        ],
    )

    compact_step = json.loads(_episode_to_json(episode))["steps"][0]
    assert compact_step["raw_model_output"]["policy_logits"]["shape"] == [2, 3]
    assert compact_step["model_output"]["action_scores"]["dtype"] == "float32"
    assert compact_step["action"]["agent"]["type"].endswith("_TensorLike")

    full_step = json.loads(_episode_to_json(episode, trace_mode="full"))["steps"][0]
    assert full_step["request"]["value"]["shape"] == [2, 3]
    assert full_step["raw_output"]["value"]["policy_logits"]["dtype"] == "float32"


def test_episode_artifacts_write_human_readable_summary_and_video(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    frame0 = np.zeros((12, 16, 3), dtype=np.uint8)
    frame1 = np.full((12, 16, 3), 255, dtype=np.uint8)
    episode = _episode()
    episode.steps[0].state.observation["screen_buffer"] = frame0
    episode.final_state.observation["screen_buffer"] = frame1

    artifacts = _write_episode_artifacts(episode, output_path=str(tmp_path), task_name="trace-env", doc_id=3)

    assert set(artifacts) == {"summary", "actions", "video"}
    assert "MOVE_FORWARD" in Path(artifacts["summary"]).read_text(encoding="utf-8")
    assert json.loads(Path(artifacts["actions"]).read_text(encoding="utf-8").splitlines()[0])["action"] == "MOVE_FORWARD"
    assert Path(artifacts["video"]).stat().st_size > 0
