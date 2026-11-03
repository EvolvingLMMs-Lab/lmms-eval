"""Task-level ViZDoom behavior without a model server."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "lmms_eval/tasks/vizdoom_agentic/utils.py"
_SPEC = importlib.util.spec_from_file_location("vizdoom_task_utils", _PATH)
utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(utils)
parsers = utils._sibling_module("parsers")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("MOVE_LEFT + ATTACK", {"buttons": {"MOVE_LEFT": 1.0, "ATTACK": 1.0}}),
        ('{"buttons": {"ATTACK": 1, "MOVE_RIGHT": 0.5}, "tics": 4}', {"buttons": {"ATTACK": 1.0, "MOVE_RIGHT": 0.5}, "tics": 4}),
        ("<think>MOVE_LEFT</think><tool_call><function=press_buttons><parameter=buttons>ATTACK</parameter><parameter=tics>5</parameter></function></tool_call>", {"buttons": {"ATTACK": 1.0}, "tics": 5}),
        ("press_buttons(MOVE_RIGHT, tics=5)", {"buttons": {"MOVE_RIGHT": 1.0}, "tics": 5}),
        ("fire", {"buttons": {"ATTACK": 1.0}}),
        ("noop()", {"buttons": {}}),
        ("submit()", {"submit": True}),
        ("gibberish", {"buttons": None}),
        ('{"buttons": ["NOT_AVAILABLE"]}', {"buttons": None}),
    ],
)
def test_action_formats(text, expected):
    assert parsers.parse_vizdoom_action(text, ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]) == expected


def test_observation_is_human_view_with_ordered_pil_frames():
    np = pytest.importorskip("numpy")
    observation = {"instruction": "win", "available_buttons": ["ATTACK"], "decision_tics": 5, "frames": [np.full((4, 4, 3), i, dtype=np.uint8) for i in (1, 2)], "game_variables": {"HEALTH": 100}}
    [message] = parsers.vizdoom_messages(observation)
    assert "HEALTH" not in message["content"][0]["text"]
    assert [part["url"].getpixel((0, 0)) for part in message["content"][1:]] == [(1, 1, 1), (2, 2, 2)]


def test_building_requests_does_not_start_simulator(monkeypatch):
    monkeypatch.setattr(utils, "vizdoom_env_manager", lambda **kwargs: pytest.fail("created simulator"))
    assert utils.vizdoom_doc_to_messages({"instruction": "win"})[0]["content"][0]["text"] == "win"


def test_callback_steps_and_returns_terminal_metric_json(monkeypatch):
    class FakeEnv:
        buttons = ["ATTACK"]
        closed = False
        steps = 0

        def reset(self, doc):
            return {"available_buttons": self.buttons, "terminal": False, "metrics": self.metrics()}

        def step(self, action):
            assert action == {"buttons": {"ATTACK": 1.0}}
            self.steps += 1
            return {"terminal": True, "metrics": self.metrics()}

        def metrics(self):
            return {"vizdoom_success": 1.0, "vizdoom_steps": self.steps, "vizdoom_invalid_actions": 0.0}

        def close(self):
            self.closed = True

    env = FakeEnv()
    monkeypatch.setattr(utils, "vizdoom_env_manager", lambda **kwargs: env)
    _, terminal, _, info = utils.vizdoom_doc_to_messages({}, round_idx=0)
    assert not terminal
    _, terminal, outputs, info = utils.vizdoom_doc_to_messages({}, round_idx=1, previous_output=["ATTACK"], previous_round_info=info)
    assert terminal
    assert json.loads(outputs[-1])["metrics"]["vizdoom_steps"] == 1
    info["close"]()
    assert env.closed


def test_reset_failure_closes_created_env(monkeypatch):
    class BrokenEnv:
        closed = False

        def reset(self, doc):
            raise RuntimeError("broken reset")

        def close(self):
            self.closed = True

    env = BrokenEnv()
    monkeypatch.setattr(utils, "vizdoom_env_manager", lambda **kwargs: env)
    with pytest.raises(RuntimeError, match="broken reset"):
        utils.vizdoom_doc_to_messages({}, round_idx=0)
    assert env.closed
