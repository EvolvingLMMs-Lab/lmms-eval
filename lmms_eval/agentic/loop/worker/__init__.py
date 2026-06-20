from lmms_eval.agentic.loop.worker.base import LoopWorker
from lmms_eval.agentic.loop.worker.simple import SimpleLoopWorker

SingleAgentLoopWorker = SimpleLoopWorker

__all__ = [
    "LoopWorker",
    "SimpleLoopWorker",
    "SingleAgentLoopWorker",
]
