"""
BabyVision Task Utilities
Evaluation for Fine-grained Discrimination and Visual Tracking tasks.
Uses Azure TRAPI GPT-4o for LLM Judge evaluation.
"""

import os
import re
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger
from PIL import Image

from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI


# ============================================================================
# Azure TRAPI Client for LLM Judge
# ============================================================================

_JUDGE_CLIENT = None
_JUDGE_DEPLOYMENT = None


def get_judge_client():
    """Get or create Azure OpenAI client for LLM Judge."""
    global _JUDGE_CLIENT, _JUDGE_DEPLOYMENT

    if _JUDGE_CLIENT is None:
        scope = os.getenv("TRAPI_SCOPE", "api://trapi/.default")
        api_version = os.getenv("TRAPI_API_VERSION", "2024-10-21")
        _JUDGE_DEPLOYMENT = os.getenv("TRAPI_DEPLOYMENT", "gpt-4o_2024-11-20")
        instance = os.getenv("TRAPI_INSTANCE", "gcr/shared")
        endpoint = f"https://trapi.research.microsoft.com/{instance}"

        chained = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        )
        credential_provider = get_bearer_token_provider(chained, scope)

        _JUDGE_CLIENT = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=credential_provider,
            api_version=api_version,
        )

    return _JUDGE_CLIENT, _JUDGE_DEPLOYMENT


LLM_JUDGE_PROMPT = """You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Important:**
* Think through your decision step-by-step **internally** before responding.
* In your final output, return **only** True or False, with no extra text or explanation.

**Input:**

Question: {question}
Ground Truth Answer: {groundtruth}
Model Output: {modeloutput}
"""


def call_judge(question: str, groundtruth: str, modeloutput: str) -> bool:
    """Call LLM Judge to evaluate answer correctness."""
    client, deployment = get_judge_client()

    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        groundtruth=groundtruth,
        modeloutput=modeloutput,
    )

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0,
        )
        response_text = resp.choices[0].message.content.strip().lower()
        return "true" in response_text
    except Exception as e:
        eval_logger.error(f"[LLM Judge Error] {e}")
        return False


# ============================================================================
# Document Processing Functions
# ============================================================================

def format_choices(choices: List[str]) -> str:
    """Format multiple choice options as (A), (B), (C), etc."""
    if not choices:
        return ""
    formatted = ""
    for idx, choice in enumerate(choices):
        formatted += f"({chr(65 + idx)}) {choice}\n"
    return formatted.strip()


def doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual input from document."""
    if "image" in doc and doc["image"]:
        img = doc["image"]
        if isinstance(img, Image.Image):
            return [img.convert("RGB")]
    return []


def doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Format question with choices if applicable."""
    question = doc["question"].strip()

    # Add choices for multiple choice questions
    if doc["ansType"] == "choice" and doc.get("options"):
        question = question + "\nChoices:\n" + format_choices(doc["options"])

    # Add pre_prompt and post_prompt
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            question = lmms_eval_specific_kwargs["pre_prompt"] + question
        if lmms_eval_specific_kwargs.get("post_prompt"):
            question = question + lmms_eval_specific_kwargs["post_prompt"]

    return question


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} format."""
    if not text:
        return None

    # Match \boxed{...} with nested braces support
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{.*\})*\})*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # Try backtick format
    pattern_backtick = r"`([^`]+)`"
    matches_bt = re.findall(pattern_backtick, text)
    if matches_bt:
        return matches_bt[-1].strip()

    return None


def process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """Process results using Azure TRAPI LLM Judge for evaluation."""
    pred_text = results[0] if results else ""

    # Get ground truth
    if doc["ansType"] == "choice":
        gt_answer = chr(65 + int(doc["choiceAns"])) if doc["choiceAns"] is not None else ""
    else:
        gt_answer = str(doc.get("blankAns", ""))

    # Extract predicted answer from boxed format
    pred_answer = extract_boxed_answer(pred_text)
    if pred_answer is None:
        # Try to find letter answer for choice questions
        if doc["ansType"] == "choice":
            letter_match = re.search(r"\b([A-D])\b", pred_text)
            if letter_match:
                pred_answer = letter_match.group(1)
            else:
                pred_answer = pred_text.strip()[:200]
        else:
            pred_answer = pred_text.strip()[:200]

    # Use LLM Judge for evaluation
    is_correct = call_judge(
        question=doc["question"],
        groundtruth=gt_answer,
        modeloutput=pred_answer,
    )
    score = 1.0 if is_correct else 0.0

    task_type = doc.get("type", "unknown")
    subtype = doc.get("subtype", "unknown")

    return {
        task_type: {
            "task_id": doc.get("taskId"),
            "subtype": subtype,
            "score": score,
            "gt": gt_answer,
            "pred": pred_answer,
        },
        "accuracy": {
            "task_id": doc.get("taskId"),
            "subtype": subtype,
            "score": score,
        },
    }


def aggregate_results(results: List[Dict]) -> float:
    """Aggregate results to compute accuracy."""
    subtype_scores = defaultdict(list)

    for result in results:
        score = result["score"]
        subtype = result["subtype"]
        subtype_scores[subtype].append(score)

    # Log subtype accuracies
    for subtype, scores in sorted(subtype_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0.0
        eval_logger.info(f"  {subtype}: {avg:.4f} ({sum(scores):.0f}/{len(scores)})")

    # Overall accuracy
    all_scores = [s for scores in subtype_scores.values() for s in scores]
    return sum(all_scores) / len(all_scores) if all_scores else 0.0


# ============================================================================
# Dataset filtering functions (used by process_docs)
# ============================================================================

def process_docs(dataset, task_type: str):
    """Filter dataset by task type."""
    return dataset.filter(lambda x: x["type"] == task_type)


process_fine_grained = partial(process_docs, task_type="Fine-grained Discrimination")
process_visual_tracking = partial(process_docs, task_type="Visual Tracking")


# ============================================================================
# Visual CoT Versions
# ============================================================================

# Fine-grained Discrimination: 需要辨别细微差异，生成图突出细节
FINE_GRAINED_GEN_PROMPT = (
    "This image contains subtle visual details that need careful examination. "
    "Your task: Generate a visualization that highlights and emphasizes the fine-grained details, "
    "subtle differences, and distinguishing features in the image. "
    "Make the key discriminative elements more prominent and easier to identify."
)

# Visual Tracking: 需要追踪物体，生成图突出物体轨迹/位置
VISUAL_TRACKING_GEN_PROMPT = (
    "This image involves tracking objects or their movements. "
    "Your task: Generate a visualization that highlights the objects of interest, "
    "their positions, trajectories, or movement patterns. "
    "Make it easier to follow and track the relevant visual elements."
)


def doc_to_text_visual_cot(
    doc: Dict,
    lmms_eval_specific_kwargs: Dict = None,
    gen_prompt: str = "",
) -> str:
    """Format Visual CoT prompt with generation prompt and question."""
    question = doc["question"].strip()

    # Add choices for multiple choice questions
    if doc["ansType"] == "choice" and doc.get("options"):
        question = question + "\nChoices:\n" + format_choices(doc["options"])

    # Add auxiliary image notice
    question = (
        "In addition to the original image, you are also given an auxiliary "
        "visualization image that highlights key visual elements.\n\n" + question
    )

    # Add pre_prompt and post_prompt
    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("pre_prompt"):
            question = lmms_eval_specific_kwargs["pre_prompt"] + question
        if lmms_eval_specific_kwargs.get("post_prompt"):
            question = question + lmms_eval_specific_kwargs["post_prompt"]

    return f"[GEN_PROMPT]{gen_prompt}[/GEN_PROMPT][QUESTION]{question}[/QUESTION]"


def doc_to_text_fine_grained_cot(
    doc: Dict, lmms_eval_specific_kwargs: Dict = None
) -> str:
    """Visual CoT prompt for Fine-grained Discrimination task."""
    return doc_to_text_visual_cot(
        doc, lmms_eval_specific_kwargs, FINE_GRAINED_GEN_PROMPT
    )


def doc_to_text_visual_tracking_cot(
    doc: Dict, lmms_eval_specific_kwargs: Dict = None
) -> str:
    """Visual CoT prompt for Visual Tracking task."""
    return doc_to_text_visual_cot(
        doc, lmms_eval_specific_kwargs, VISUAL_TRACKING_GEN_PROMPT
    )
