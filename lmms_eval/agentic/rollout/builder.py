from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lmms_eval.agentic.factory import DEFAULT_AGENTIC_FACTORY, AgenticFactory
from lmms_eval.agentic.rollout.protocol import RolloutEpisodeSpec


@dataclass(slots=True)
class EpisodeComponents:
    model_server: Any
    env_manager: Any
    observation_parser: Any
    model_output_parser: Any
    action_parser: Any


class EpisodeComponentBuilder:
    """Encapsulates lmms-eval component construction for agentic rollouts."""

    def __init__(
        self,
        factory: AgenticFactory | None = None,
        model_server: Any | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.factory = factory or DEFAULT_AGENTIC_FACTORY
        self.model_server_spec = model_server
        self.generation_kwargs = dict(generation_kwargs or {})

    def build(self, spec: RolloutEpisodeSpec) -> EpisodeComponents:
        lmms_kwargs = spec.lmms_eval_specific_kwargs
        return EpisodeComponents(
            model_server=self.factory.build_model_server(
                spec.model_server if spec.model_server is not None else self.model_server_spec,
                generation_kwargs={**self.generation_kwargs, **spec.generation_kwargs},
            ),
            env_manager=self.factory.build_env_manager(
                spec.game_env,
                doc=spec.doc,
                lmms_eval_specific_kwargs=lmms_kwargs,
            ),
            observation_parser=self.factory.build_observation_parser(
                spec.observation_parser,
                doc=spec.doc,
                lmms_eval_specific_kwargs=lmms_kwargs,
            ),
            model_output_parser=self.factory.build_model_output_parser(
                spec.model_output_parser,
                doc=spec.doc,
                lmms_eval_specific_kwargs=lmms_kwargs,
            ),
            action_parser=self.factory.build_action_parser(
                spec.action_parser,
                doc=spec.doc,
                lmms_eval_specific_kwargs=lmms_kwargs,
            ),
        )
