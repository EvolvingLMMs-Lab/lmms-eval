import json
from typing import Any

from lmms_eval.api.agentic import AgentInput, ContentBlock, EnvState, GameAction, GameEnv, ObservationParser, StepResult, register_game_env, register_observation_parser
from lmms_eval.tasks.vizdoom_agentic.env import VizDoomEnv

ACTION_NAMES = {
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
    "USE",
    "NOOP",
}
FACING_ORDER = ["N", "E", "S", "W"]
MOVE_DELTAS = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}


def vizdoom_doc_to_visual(doc):
    return []


def vizdoom_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['instruction']}\nChoose actions from: {', '.join(sorted(ACTION_NAMES))}. Reach the goal tile.{post_prompt}"


def vizdoom_native_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['instruction']}\nUse the VizDoom visual input and state to choose the next action.{post_prompt}"


def vizdoom_doc_to_target(doc):
    return "reach_goal"


def vizdoom_native_doc_to_target(doc):
    return "maximize_reward"


class VizDoomGridEnv(GameEnv):
    """Small dependency-free environment with VizDoom-like action names."""

    def __init__(self) -> None:
        self.doc: dict[str, Any] = {}
        self.env_id = "vizdoom-grid"
        self.position = [0, 0]
        self.goal = [0, 0]
        self.facing = "E"
        self.step_idx = 0
        self.terminal = False
        self.success = False
        self.invalid_actions = 0

    def reset(self, doc: Any, seed: int | None = None) -> EnvState:
        self.doc = dict(doc)
        self.env_id = str(self.doc.get("id", "vizdoom-grid"))
        self.position = list(self.doc.get("start", [1, 1]))
        self.goal = list(self.doc.get("goal", [1, 1]))
        self.facing = str(self.doc.get("facing", "E"))
        self.step_idx = 0
        self.terminal = False
        self.success = False
        self.invalid_actions = 0
        return self._state()

    def step(self, action: GameAction | dict[str, GameAction]) -> StepResult:
        if isinstance(action, dict):
            action = next(iter(action.values()))

        reward = -0.01
        info: dict[str, Any] = {"action": action.type}

        if self.terminal:
            return StepResult(state=self._state(), reward=0.0, done=True, info={"already_terminal": True})

        action_type = action.type.upper()
        if action_type == "PARSE_ERROR":
            self.invalid_actions += 1
            info["error"] = action.data
        elif action_type == "SUBMIT":
            self.terminal = True
            self.success = self.position == self.goal
            reward = 1.0 if self.success else -0.1
        elif action_type == "TURN_LEFT":
            self.facing = FACING_ORDER[(FACING_ORDER.index(self.facing) - 1) % len(FACING_ORDER)]
        elif action_type == "TURN_RIGHT":
            self.facing = FACING_ORDER[(FACING_ORDER.index(self.facing) + 1) % len(FACING_ORDER)]
        elif action_type == "MOVE_FORWARD":
            self._move_forward(info)
        elif action_type in {"ATTACK", "USE", "NOOP"}:
            pass
        else:
            self.invalid_actions += 1
            info["error"] = f"unknown action: {action.type}"

        self.step_idx += 1
        if self.position == self.goal:
            self.success = True
            self.terminal = True
            reward = 1.0

        return StepResult(state=self._state(), reward=reward, done=self.terminal, info=info)

    def _move_forward(self, info: dict[str, Any]) -> None:
        dx, dy = MOVE_DELTAS.get(self.facing, (0, 0))
        next_position = [self.position[0] + dx, self.position[1] + dy]
        if self._cell(next_position) == "#":
            self.invalid_actions += 1
            info["blocked"] = True
            return
        self.position = next_position

    def _cell(self, position: list[int]) -> str:
        grid = self.doc.get("map", [])
        x, y = position
        if y < 0 or y >= len(grid):
            return "#"
        row = grid[y]
        if x < 0 or x >= len(row):
            return "#"
        return row[x]

    def _state(self) -> EnvState:
        observation = {
            "instruction": self.doc.get("instruction", ""),
            "map": self.doc.get("map", []),
            "position": self.position,
            "goal": self.goal,
            "facing": self.facing,
        }
        metrics = {
            "vizdoom_success": 1.0 if self.success else 0.0,
            "vizdoom_steps": float(self.step_idx),
            "vizdoom_invalid_actions": float(self.invalid_actions),
        }
        return EnvState(
            env_id=self.env_id,
            step_idx=self.step_idx,
            observation=observation,
            active_agent_ids=[] if self.terminal else ["agent"],
            terminal=self.terminal,
            metadata={"success": self.success, "metrics": metrics},
        )


class VizDoomObservationParser(ObservationParser):
    def parse(self, state: EnvState, agent_id: str | None = None) -> AgentInput:
        observation_json = json.dumps(state.observation, ensure_ascii=False)
        prompt = f"{state.observation.get('instruction', '')}\n" f"Observation: {observation_json}\n" "Respond with exactly one action name. Valid actions: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, ATTACK, USE, NOOP, SUBMIT."
        return AgentInput(
            content=[
                ContentBlock.text(prompt),
                ContentBlock(type="state_features", data=state.observation, metadata={"agent_id": agent_id}),
            ],
            metadata={"env_id": state.env_id, "step_idx": state.step_idx, "agent_id": agent_id},
        )


register_game_env("vizdoom_grid", VizDoomGridEnv, replace=True)
register_game_env("vizdoom_native", VizDoomEnv, replace=True)
register_observation_parser("vizdoom_text", VizDoomObservationParser, replace=True)


def vizdoom_game_env(doc, lmms_eval_specific_kwargs=None):
    return VizDoomGridEnv()


def vizdoom_observation_parser(doc, lmms_eval_specific_kwargs=None):
    return VizDoomObservationParser()


def vizdoom_process_results(doc, results):
    raw = results[0] if results else "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"vizdoom_success": 0.0, "vizdoom_steps": 0.0, "vizdoom_invalid_actions": 1.0}

    metrics = payload.get("metrics", {})
    return {
        "vizdoom_success": float(metrics.get("vizdoom_success", 1.0 if payload.get("success") else 0.0)),
        "vizdoom_steps": float(metrics.get("vizdoom_steps", 0.0)),
        "vizdoom_invalid_actions": float(metrics.get("vizdoom_invalid_actions", 0.0)),
    }


def vizdoom_aggregate_success(results):
    return sum(results) / len(results) if results else 0.0


def vizdoom_aggregate_steps(results):
    return sum(results) / len(results) if results else 0.0


def vizdoom_aggregate_invalid_actions(results):
    return sum(results) / len(results) if results else 0.0
