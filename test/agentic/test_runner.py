from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lmms_eval.agentic import ActionDef, ActionSpec
from lmms_eval.agentic.runner import run_generate_until_game
from lmms_eval.api.instance import Instance

from .conftest import ScriptedEnv, TextObservationParser, UppercaseActionParser


def _cli_args(**overrides):
    defaults = {
        "agentic_model_server": "debug",
        "agentic_model_server_args": "action=attack",
        "agentic_output_parser": None,
        "agentic_output_parser_args": "",
        "agentic_max_parallel_rollouts": None,
        "agentic_episode_retries": None,
        "output_path": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _game_instance(doc_id=0, episode_len=2, max_game_steps=8, task="game", split="test"):
    arguments = (
        "prompt",
        {"max_new_tokens": 16, "max_game_steps": max_game_steps},
        lambda doc: [],
        lambda doc=None, lmms_eval_specific_kwargs=None: ScriptedEnv(episode_len=episode_len),
        lambda doc=None, lmms_eval_specific_kwargs=None: TextObservationParser(),
        lambda doc=None, lmms_eval_specific_kwargs=None: UppercaseActionParser(),
        {"pre_prompt": ""},
        doc_id,
        task,
        split,
    )
    return Instance(request_type="generate_until_game", arguments=arguments, idx=0, metadata={"task": task, "doc_id": doc_id, "repeats": 1})


def _lm(docs, task="game", split="test"):
    return SimpleNamespace(task_dict={task: {split: docs}})


def test_runner_returns_parseable_episode_json():
    lm = _lm({0: {"instruction": "win"}})
    [resp] = run_generate_until_game(lm, [_game_instance(doc_id=0)], cli_args=_cli_args())

    payload = json.loads(resp)
    assert payload["success"] is True
    assert payload["metrics"] == {"scripted_steps": 2.0}
    assert payload["final_state"]["terminal"] is True
    assert [step["action"]["type"] for step in payload["steps"]] == ["ATTACK", "ATTACK"]
    assert payload["steps"][0]["raw_model_output"] == "attack"
    assert payload["metadata"]["max_steps"] == 8


def test_runner_preserves_request_order_with_parallel_rollouts():
    docs = {i: {"instruction": f"doc-{i}"} for i in range(6)}
    lm = _lm(docs)
    requests = [_game_instance(doc_id=i, episode_len=1 + (i % 3)) for i in range(6)]

    resps = run_generate_until_game(lm, requests, cli_args=_cli_args(agentic_max_parallel_rollouts=4))

    assert len(resps) == 6
    for i, resp in enumerate(resps):
        assert len(json.loads(resp)["steps"]) == 1 + (i % 3)


def test_runner_writes_artifacts_when_output_path_is_set(tmp_path):
    lm = _lm({0: {"instruction": "win"}})
    [resp] = run_generate_until_game(lm, [_game_instance(doc_id=0)], cli_args=_cli_args(output_path=str(tmp_path)))

    payload = json.loads(resp)
    artifacts = payload["metadata"]["artifacts"]
    summary = tmp_path / "agentic_artifacts"
    assert summary.exists()
    assert "summary" in artifacts and "actions" in artifacts
    actions_rows = [json.loads(line) for line in open(artifacts["actions"], encoding="utf-8")]
    assert [row["requested_action"] for row in actions_rows] == ["ATTACK", "ATTACK"]


def test_runner_rejects_wrong_arity():
    lm = _lm({0: {"instruction": "win"}})
    bad = Instance(request_type="generate_until_game", arguments=("ctx", {}, None), idx=0, metadata={"task": "game", "doc_id": 0, "repeats": 1})

    with pytest.raises(ValueError, match="10-element"):
        run_generate_until_game(lm, [bad], cli_args=_cli_args())


def test_runner_pops_loop_keys_from_generation_kwargs():
    lm = _lm({0: {"instruction": "win"}})
    inst = _game_instance(doc_id=0, episode_len=1)
    args = list(inst.arguments)
    args[1] = {"max_new_tokens": 16, "max_game_steps": 4, "game_seed": 7, "multiturn": True, "history_turns": 2}
    inst.arguments = tuple(args)

    [resp] = run_generate_until_game(lm, [inst], cli_args=_cli_args())

    payload = json.loads(resp)
    assert payload["metadata"]["multiturn"] is True
    assert payload["metadata"]["history_turns"] == 2
    # request-level generation kwargs no longer carry loop-only keys
    assert payload["metadata"]["max_steps"] == 4


# ---------------------------------------------------------------------------
# Default parsers from env.action_spec() + episode error semantics
# ---------------------------------------------------------------------------


class SpecEnv(ScriptedEnv):
    """ScriptedEnv that declares its action space."""

    def action_spec(self):
        return ActionSpec(kind="discrete", actions=[ActionDef(name="ATTACK"), ActionDef(name="NOOP")])


class CrashingEnv(ScriptedEnv):
    def reset(self, doc, seed=None):
        raise RuntimeError("simulator exploded")


def _instance_with(env_factory, observation_parser, action_parser, doc_id=0, max_game_steps=8):
    arguments = (
        "prompt",
        {"max_new_tokens": 16, "max_game_steps": max_game_steps},
        lambda doc: [],
        env_factory,
        observation_parser,
        action_parser,
        {"pre_prompt": ""},
        doc_id,
        "game",
        "test",
    )
    return Instance(request_type="generate_until_game", arguments=arguments, idx=0, metadata={"task": "game", "doc_id": doc_id, "repeats": 1})


def test_runner_defaults_parsers_from_env_action_spec():
    lm = _lm({0: {"instruction": "win"}})
    inst = _instance_with(lambda doc=None, lmms_eval_specific_kwargs=None: SpecEnv(episode_len=2), None, None)

    [resp] = run_generate_until_game(lm, [inst], cli_args=_cli_args())

    payload = json.loads(resp)
    assert payload["success"] is True
    assert [step["action"]["type"] for step in payload["steps"]] == ["ATTACK", "ATTACK"]


def test_runner_fails_loud_when_action_parser_and_spec_are_both_missing():
    lm = _lm({0: {"instruction": "win"}})
    inst = _instance_with(lambda doc=None, lmms_eval_specific_kwargs=None: ScriptedEnv(episode_len=2), None, None)

    with pytest.raises(ValueError, match="action_spec"):
        run_generate_until_game(lm, [inst], cli_args=_cli_args())


def test_runner_records_env_error_instead_of_raising():
    lm = _lm({0: {"instruction": "win"}})
    inst = _instance_with(
        lambda doc=None, lmms_eval_specific_kwargs=None: CrashingEnv(episode_len=2),
        lambda doc=None, lmms_eval_specific_kwargs=None: TextObservationParser(),
        lambda doc=None, lmms_eval_specific_kwargs=None: UppercaseActionParser(),
    )

    [resp] = run_generate_until_game(lm, [inst], cli_args=_cli_args())

    payload = json.loads(resp)
    assert payload["success"] is False
    assert payload["metrics"] == {"env_error": 1.0}
    assert "simulator exploded" in payload["metadata"]["error"]
    assert payload["metadata"]["error_type"] == "RuntimeError"
    assert payload["metadata"]["attempts"] == 1
    assert payload["steps"] == []


def test_runner_retries_flaky_episodes():
    lm = _lm({0: {"instruction": "win"}})
    calls = {"n": 0}

    def flaky_factory(doc=None, lmms_eval_specific_kwargs=None):
        calls["n"] += 1
        env_cls = CrashingEnv if calls["n"] == 1 else ScriptedEnv
        return env_cls(episode_len=2)

    inst = _instance_with(
        flaky_factory,
        lambda doc=None, lmms_eval_specific_kwargs=None: TextObservationParser(),
        lambda doc=None, lmms_eval_specific_kwargs=None: UppercaseActionParser(),
    )

    [resp] = run_generate_until_game(lm, [inst], cli_args=_cli_args(agentic_episode_retries=1))

    payload = json.loads(resp)
    assert calls["n"] == 2
    assert payload["success"] is True
    assert "env_error" not in payload["metrics"]
