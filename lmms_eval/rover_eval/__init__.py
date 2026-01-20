"""
ROVER Evaluation Module - Visual CoT Evaluator

专门用于评测 Visual Chain-of-Thought 任务

Metrics:
- RA (Reasoning-to-Visual Alignment): 生成图像是否符合 generation_prompt
- AL (Answer-to-Visual Alignment): 最终答案是否与生成图像一致

Supports 7 Task Categories:
1. real_world - 真实应用
2. mathematical - 数学推理
3. stem - 科学技术工程数学
4. puzzles - 谜题游戏
5. chart_table - 图表推理
6. spatial - 空间智能
7. perception - 感知推理
"""

from .visual_cot_evaluator import (
    VisualCoTEvaluator,
    evaluate_from_json,
    evaluate_batch_from_jsons,
)

from .api import get_gpt4o_client, call_gpt4o_with_images

__all__ = [
    "VisualCoTEvaluator",
    "evaluate_from_json",
    "evaluate_batch_from_jsons",
    "get_gpt4o_client",
    "call_gpt4o_with_images",
]
