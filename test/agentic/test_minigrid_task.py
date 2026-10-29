import json
from types import SimpleNamespace

import numpy as np
import pytest

from lmms_eval.protocol import ChatMessages
from lmms_eval.tasks.minigrid_agentic import env as env_module
from lmms_eval.tasks.minigrid_agentic import utils
from lmms_eval.tasks.minigrid_agentic.parsers import parse_action


class GridSimulator:
    def __init__(self):
        self.actions = []
        self.closed = 0
        self.unwrapped = self
        self.carrying = None

    def reset(self, seed=None):
        self.seed = seed
        return {"mission": "Reach the goal", "direction": 0}, {}

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def step(self, action):
        self.actions.append(action)
        won = len(self.actions) == 2
        return {"mission": "Reach the goal", "direction": 1}, float(won), won, False, {}

    def close(self):
        self.closed += 1


@pytest.fixture
def simulator(monkeypatch):
    simulator = GridSimulator()
    gym = SimpleNamespace(make=lambda *_args, **_kwargs: simulator)
    monkeypatch.setattr(env_module, "optional_import", lambda name: (gym if name == "gymnasium" else object(), True))
    return simulator


def test_request_construction_does_not_start_simulator(monkeypatch):
    def unexpected_env():
        raise AssertionError("Request construction started a simulator")

    monkeypatch.setattr(utils, "MiniGridEnvManager", unexpected_env)
    messages = utils.minigrid_doc_to_messages({"instruction": "Go to the goal"})
    assert "Go to the goal" in messages[0]["content"][0]["text"]
    assert "Available actions" in messages[0]["content"][0]["text"]


def test_episode_images_use_standard_chat_protocol_and_terminal_summary(simulator):
    doc = {"instruction": "Go to the goal", "seed": 7}
    messages, terminal, outputs, info = utils.minigrid_doc_to_messages(doc, round_idx=0)
    assert not terminal
    assert simulator.seed == 7
    encoded = ChatMessages(messages=messages).to_openai_messages()
    assert encoded[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    json.dumps(info["state"])

    messages, terminal, outputs, info = utils.minigrid_doc_to_messages(doc, round_idx=1, previous_output=["FORWARD"], previous_round_info=info)
    assert not terminal
    assert simulator.actions == [2]

    messages, terminal, outputs, info = utils.minigrid_doc_to_messages(doc, round_idx=2, previous_output=["FORWARD", "<think>choose</think> FORWARD"], previous_round_info=info)
    assert terminal and messages == []
    payload = json.loads(outputs[-1])
    assert payload["success"] is True
    assert payload["metrics"]["steps"] == 2
    assert utils.minigrid_process_results(doc, [outputs[-1]])["minigrid_success"] == 1
    info["close"]()
    assert simulator.closed == 1


def test_invalid_actions_and_budget_finalization(simulator):
    doc = {"instruction": "Go to the goal"}
    _, _, _, info = utils.minigrid_doc_to_messages(doc, round_idx=0)
    _, terminal, _, info = utils.minigrid_doc_to_messages(doc, round_idx=1, previous_output=["nonsense"], previous_round_info=info)
    assert not terminal
    payload = json.loads(info["finalize"]())
    assert payload["success"] is False
    assert payload["metrics"]["steps"] == 1
    assert payload["metrics"]["invalid_actions"] == 1
    assert simulator.actions == []
    info["close"]()


def test_simulator_failure_is_a_failed_episode_with_cleanup(simulator, monkeypatch):
    def fail_step(_action):
        raise RuntimeError("simulator disconnected")

    monkeypatch.setattr(simulator, "step", fail_step)
    doc = {}
    _, _, _, info = utils.minigrid_doc_to_messages(doc, round_idx=0)
    _, terminal, outputs, info = utils.minigrid_doc_to_messages(doc, round_idx=1, previous_output=["FORWARD"], previous_round_info=info)
    assert terminal
    metrics = utils.minigrid_process_results(doc, [outputs[-1]])
    assert metrics["minigrid_env_error"] == 1
    assert metrics["minigrid_success"] == 0
    info["close"]()
    assert simulator.closed == 1


@pytest.mark.parametrize("text, expected", [('```json\n{"action": "TURN_LEFT"}\n```', "LEFT"), ("reasoning </think> FORWARD", "FORWARD"), ("USE", "TOGGLE"), ("no valid move", None)])
def test_action_decoding(text, expected):
    assert parse_action(text) == expected
