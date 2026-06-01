"""OmniSpatial task for lmms-eval.

A 1,533-item single-image MCQ on comprehensive spatial reasoning, with
four task types (Complex_Logic, Dynamic_Reasoning, Perspective_Taking,
Spatial_Interaction) and ten sub-task types.

Each item carries 2+ options with a 0-based ``answer`` index. The
default scoring mode (``eval_type: direct``) does deterministic
letter-match — the question prompt asks the model to reply with a
single letter — so no LLM judge is required.

Dataset: nv-njb/OmniSpatial-Test — a bundled re-host of the test split
of qizekun/OmniSpatial (the canonical release ships a 1.66 GB zip
that doesn't load directly via load_dataset).

Reference: https://huggingface.co/datasets/qizekun/OmniSpatial
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from PIL import Image

# ---------------------------------------------------------------------------
# Prompting (matches the default direct-MCQ mode of the upstream OmniSpatial
# reference implementation)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are a spatial-reasoning assistant.

Task
-----
You will receive
1. **Image** - a single RGB frame depicting a scene.
2. **Multi-View Image** - a RGB frame depicting a scene from 6 novel views.
3. **Question** - a natural-language query about spatial relationships between objects in the image.
4. **Options** - >=2 answer candidates, each tagged by a capital letter (A, B, C, D...).

Based on the image and question, provide your answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option---never refuse or reply "insufficient information."
"""

DIRECT_FORMAT = "\nNote: You only need to respond with A, B, C, or D without providing any additional information.\n"


def _strip_think(text: str) -> str:
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</thought>", "", text, flags=re.DOTALL)
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def omni_spatial_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    return [doc["image"].convert("RGB")]


def omni_spatial_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None) -> str:
    prompt = DEFAULT_SYSTEM_PROMPT + DIRECT_FORMAT + "\n" + doc["question"]
    for i, opt in enumerate(doc["options"]):
        prompt += f"\n{chr(65 + i)}. {opt}"
    return prompt


def omni_spatial_doc_to_answer(doc: Dict[str, Any]) -> str:
    return chr(65 + int(doc["answer"]))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def omni_spatial_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    raw = results[0] if results else ""
    response = _strip_think(raw)
    pred_letter = response.strip().upper()[:1] if response else ""
    gt_letter = chr(65 + int(doc["answer"]))
    score = int(pred_letter == gt_letter)
    item = {
        "score": score,
        "task_type": str(doc.get("task_type", "")),
        "sub_task_type": str(doc.get("sub_task_type", "")),
    }
    return {
        "omni_spatial_score": item,
        "complex_logic": item,
        "dynamic_reasoning": item,
        "perspective_taking": item,
        "spatial_interaction": item,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def omni_spatial_aggregate_overall(results: List[Dict[str, Any]]) -> float:
    return _mean([r["score"] for r in results])


def _by_task(results, target):
    return _mean([r["score"] for r in results if r["task_type"] == target])


def omni_spatial_aggregate_complex_logic(results):
    return _by_task(results, "Complex_Logic")


def omni_spatial_aggregate_dynamic_reasoning(results):
    return _by_task(results, "Dynamic_Reasoning")


def omni_spatial_aggregate_perspective_taking(results):
    return _by_task(results, "Perspective_Taking")


def omni_spatial_aggregate_spatial_interaction(results):
    return _by_task(results, "Spatial_Interaction")
