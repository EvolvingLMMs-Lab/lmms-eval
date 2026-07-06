from __future__ import annotations

import asyncio
from typing import Any

from lmms_eval.agentic.factory import AgenticFactory
from lmms_eval.agentic.rollout.builder import EpisodeComponentBuilder
from lmms_eval.agentic.rollout.protocol import RolloutEpisodeSpec
from lmms_eval.agentic.types import EpisodeResult


class SyncEpisodeRolloutWorker:
    """Facade for lmms-eval's synchronous atomic episode loop."""

    def __init__(
        self,
        model_server: Any | None = None,
        factory: AgenticFactory | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        component_builder: EpisodeComponentBuilder | None = None,
    ) -> None:
        self.component_builder = component_builder or EpisodeComponentBuilder(
            factory=factory,
            model_server=model_server,
            generation_kwargs=generation_kwargs,
        )

    def run_episode(self, spec: RolloutEpisodeSpec) -> EpisodeResult:
        components = self.component_builder.build(spec)
        worker = self.component_builder.factory.build_loop_worker(
            spec.loop_worker,
            model_server=components.model_server,
            env_manager=components.env_manager,
            observation_parser=components.observation_parser,
            model_output_parser=components.model_output_parser,
            action_parser=components.action_parser,
            max_steps=spec.max_steps,
            generation_kwargs=spec.generation_kwargs,
            request_metadata=spec.request_metadata,
        )
        return worker.run(spec.doc, seed=spec.seed, agent_id=spec.agent_id)

    async def run_episode_async(self, spec: RolloutEpisodeSpec) -> EpisodeResult:
        return await asyncio.to_thread(self.run_episode, spec)
