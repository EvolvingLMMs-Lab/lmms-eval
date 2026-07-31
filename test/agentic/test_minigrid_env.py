"""MiniGridEnvManager against a scripted fake gymnasium (no simulator install)."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from lmms_eval.agentic import EnvManager, GameAction
from lmms_eval.agentic.components import resolve
from lmms_eval.agentic.envs.minigrid import MiniGridEnvManager
from lmms_eval.agentic.runner import run_generate_until_game
from lmms_eval.api.instance import Instance


class _FakeSimEnv:
    """Reaches the goal after ``goal_after`` FORWARD steps."""

    def __init__(self, goal_after: int = 2) -> None:
        self.goal_after = goal_after
        self.forward_count = 0
        self.step_calls: list[int] = []
        self.reset_seed = None
        self.closed = False
        self.unwrapped = SimpleNamespace(carrying=None)

    def reset(self, seed=None):
        self.reset_seed = seed
        return {"mission": "get to the green goal square", "direction": 0}, {}

    def step(self, index):
        self.step_calls.append(index)
        terminated, reward = False, 0.0
        if index == 2:
            self.forward_count += 1
            if self.forward_count >= self.goal_after:
                terminated, reward = True, 0.95
        return {"mission": "get to the green goal square", "direction": 1}, reward, terminated, False, {}

    def render(self):
        return "frame"

    def close(self):
        self.closed = True


@pytest.fixture
def fake_gym(monkeypatch):
    made = {}

    def make(env_id, **kwargs):
        made["env_id"] = env_id
        made["kwargs"] = kwargs
        made["env"] = _FakeSimEnv()
        return made["env"]

    gymnasium = types.ModuleType("gymnasium")
    gymnasium.make = make
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "minigrid", types.ModuleType("minigrid"))
    return made


def test_reset_emits_reserved_observation_keys(fake_gym):
    manager = MiniGridEnvManager()

    state = manager.reset({"env_id": "MiniGrid-DoorKey-6x6-v0", "seed": 5})

    assert fake_gym["env_id"] == "MiniGrid-DoorKey-6x6-v0"
    assert fake_gym["kwargs"] == {"render_mode": "rgb_array"}
    assert fake_gym["env"].reset_seed == 5
    assert state.env_id == "MiniGrid-DoorKey-6x6-v0"
    assert state.observation["text"] == "get to the green goal square"
    assert state.observation["images"] == ["frame"]
    assert state.observation["variables"]["facing"] == "east"


def test_forward_steps_reach_goal_and_success(fake_gym):
    manager = MiniGridEnvManager()
    manager.reset({})

    manager.step(GameAction(type="FORWARD"))
    result = manager.step(GameAction(type="forward"))

    assert result.done is True
    assert result.state.metadata["success"] is True
    assert result.state.metadata["metrics"] == {"steps": 2.0, "invalid_actions": 0.0, "reward": 0.95}


def test_invalid_action_burns_the_turn_without_stepping_the_simulator(fake_gym):
    manager = MiniGridEnvManager()
    manager.reset({})

    result = manager.step(GameAction(type="parse_error", data="no valid action found"))

    assert fake_gym["env"].step_calls == []
    assert result.state.step_idx == 1
    assert result.state.terminal is False
    assert result.info["invalid_action"] is True
    assert result.state.metadata["metrics"]["invalid_actions"] == 1.0


def test_action_subset_limits_spec_and_step(fake_gym):
    manager = MiniGridEnvManager(actions=["LEFT", "RIGHT", "FORWARD"])
    manager.reset({})

    assert manager.action_spec().action_names() == ["LEFT", "RIGHT", "FORWARD"]
    result = manager.step(GameAction(type="PICKUP"))
    assert result.info["invalid_action"] is True


def test_unknown_action_subset_raises():
    with pytest.raises(ValueError, match="Unknown MiniGrid actions"):
        MiniGridEnvManager(actions=["FLY"])


def test_action_spec_is_discrete_with_aliases():
    spec = MiniGridEnvManager().action_spec()

    assert spec.kind == "discrete"
    assert "FORWARD" in spec.action_names()
    assert spec.alias_map()["TURN_LEFT"] == "LEFT"


def test_registry_builds_minigrid_without_simulator_installed():
    manager = resolve("env_manager", "minigrid", expected=EnvManager, doc={}, lmms_eval_specific_kwargs=None)

    assert isinstance(manager, MiniGridEnvManager)


def test_runner_end_to_end_with_registry_env_and_default_parsers(fake_gym):
    doc = {"env_id": "MiniGrid-Empty-6x6-v0", "seed": 1, "instruction": "Reach the goal."}
    lm = SimpleNamespace(task_dict={"minigrid": {"test": {0: doc}}})
    arguments = (
        "prompt",
        {"max_new_tokens": 16, "max_game_steps": 8},
        lambda doc: [],
        "minigrid",  # registry name straight from YAML
        None,  # observation_parser omitted -> TemplateObservationParser
        None,  # action_parser omitted -> built from env.action_spec()
        {"pre_prompt": ""},
        0,
        "minigrid",
        "test",
    )
    instance = Instance(request_type="generate_until_game", arguments=arguments, idx=0, metadata={"task": "minigrid", "doc_id": 0, "repeats": 1})
    cli_args = SimpleNamespace(
        agentic_model_server="debug",
        agentic_model_server_args="action=FORWARD",
        agentic_output_parser=None,
        agentic_output_parser_args="",
        agentic_max_parallel_rollouts=None,
        agentic_episode_retries=None,
        output_path=None,
    )

    [response] = run_generate_until_game(lm, [instance], cli_args=cli_args)

    payload = json.loads(response)
    assert payload["success"] is True
    assert payload["metrics"]["steps"] == 2.0
    assert [step["action"]["type"] for step in payload["steps"]] == ["FORWARD", "FORWARD"]


def test_task_process_results_maps_episode_and_error_payloads():
    from lmms_eval.tasks.minigrid_agentic import utils as task_utils

    ok = json.dumps({"success": True, "metrics": {"steps": 5.0, "invalid_actions": 1.0, "reward": 0.9}})
    err = json.dumps({"success": False, "metrics": {"env_error": 1.0}})

    ok_metrics = task_utils.minigrid_process_results({}, [ok])
    err_metrics = task_utils.minigrid_process_results({}, [err])

    assert ok_metrics == {"minigrid_success": 1.0, "minigrid_steps": 5.0, "minigrid_invalid_actions": 1.0, "minigrid_env_error": 0.0}
    assert err_metrics["minigrid_success"] == 0.0
    assert err_metrics["minigrid_env_error"] == 1.0
