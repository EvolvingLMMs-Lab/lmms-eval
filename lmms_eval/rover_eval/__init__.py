"""
ROVER Evaluation Module

Evaluates Interleaved Reasoning and Reasoning Alignment for vision-language models.

Metrics:
- reasoning_process: Evaluates the quality of reasoning text (think_output)
- reasoning_visual: Evaluates the quality of generated images
- reasoning_alignment: Evaluates alignment between reasoning text and generated images
"""

from .evaluator import ROVEREvaluator, evaluate_single_sample, evaluate_batch
from .api import get_gpt4o_client, call_gpt4o_with_images

__all__ = [
    "ROVEREvaluator",
    "evaluate_single_sample",
    "evaluate_batch",
    "get_gpt4o_client",
    "call_gpt4o_with_images",
]
