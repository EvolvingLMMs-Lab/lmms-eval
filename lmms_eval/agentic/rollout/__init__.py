"""Agentic rollout wrappers over lmms-eval's existing loop stack."""

from lmms_eval.agentic.rollout.protocol import RolloutEpisodeSpec

__all__ = [
    "EpisodeComponentBuilder",
    "EpisodeComponents",
    "RolloutEpisodeSpec",
    "SyncEpisodeRolloutWorker",
]


def __getattr__(name):
    if name in {"EpisodeComponentBuilder", "EpisodeComponents"}:
        from lmms_eval.agentic.rollout.builder import (
            EpisodeComponentBuilder,
            EpisodeComponents,
        )

        return {
            "EpisodeComponentBuilder": EpisodeComponentBuilder,
            "EpisodeComponents": EpisodeComponents,
        }[name]
    if name == "SyncEpisodeRolloutWorker":
        from lmms_eval.agentic.rollout.worker import SyncEpisodeRolloutWorker

        return SyncEpisodeRolloutWorker
    raise AttributeError(name)
