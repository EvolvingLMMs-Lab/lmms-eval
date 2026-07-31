"""Shared scripted components for agentic loop tests (no real env or backend)."""

from __future__ import annotations

from typing import Any

import pytest

from lmms_eval.agentic import (
    AgentInput,
    AgentOutput,
    ContentBlock,
    EnvManager,
    EnvState,
    GameAction,
    ParsedAction,
    StepResult,
)
from lmms_eval.agentic.parsers import ActionParser, ObservationParser, ParserContext


class ScriptedEnv(EnvManager):
    """Terminates after ``episode_len`` steps and records every action."""

    def __init__(self, episode_len: int = 3, doc: Any = None) -> None:
        self.episode_len = episode_len
        self.doc = doc
        self.step_idx = 0
        self.actions: list[Any] = []
        self.closed = False
        self.reset_seed: int | None = None

    def reset(self, doc: Any, seed: int | None = None) -> EnvState:
        self.doc = doc
        self.reset_seed = seed
        self.step_idx = 0
        return self.get_state()

    def step(self, action: GameAction | dict[str, GameAction]) -> StepResult:
        self.actions.append(action)
        self.step_idx += 1
        state = self.get_state()
        return StepResult(state=state, reward=1.0, done=state.terminal, info={"total_reward": float(self.step_idx)})

    def get_state(self) -> EnvState:
        terminal = self.step_idx >= self.episode_len
        return EnvState(
            env_id="scripted",
            step_idx=self.step_idx,
            observation={"text": f"obs-{self.step_idx}"},
            terminal=terminal,
            metadata={"success": terminal, "metrics": {"scripted_steps": float(self.step_idx)}},
        )

    def close(self) -> None:
        self.closed = True


class TextObservationParser(ObservationParser):
    def parse(self, state: EnvState, ctx: ParserContext) -> AgentInput:
        return AgentInput(content=[ContentBlock.text(str(state.observation["text"]))])


class UppercaseActionParser(ActionParser):
    def parse(self, output: AgentOutput, ctx: ParserContext) -> ParsedAction:
        text = (output.first_text() or "").strip()
        if not text:
            return ParsedAction(error="empty output")
        return ParsedAction(action=GameAction(type=text.upper(), agent_id=ctx.agent_id))


@pytest.fixture
def scripted_components():
    return {
        "env": ScriptedEnv(episode_len=3),
        "observation_parser": TextObservationParser(),
        "action_parser": UppercaseActionParser(),
    }
