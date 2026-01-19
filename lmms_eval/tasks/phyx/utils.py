import base64
import io
import os
import re
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from loguru import logger as eval_logger

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


def load_phyx_config():
    with open(Path(__file__).parent / "phyx.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for line in raw_data:
            if "!function" not in line:
                safe_data.append(line)
        return yaml.safe_load("".join(safe_data))


config = load_phyx_config()


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def phyx_doc_to_visual(doc):
    image = decode_base64_to_image(doc["image"])
    return [image]


def phyx_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    query_prompt = doc["question"]
    return query_prompt


def extract_boxed_answer(text: str):
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


def phyx_process_results_mc(doc, results):
    """Process multiple choice results using Azure TRAPI LLM Judge."""
    prediction = results[0].strip() if results else ""
    
    # Extract answer from prediction
    pred_answer = extract_boxed_answer(prediction)
    if pred_answer is None:
        # Try to find letter answer
        letter_match = re.search(r"\b([A-Z])\b", prediction)
        if letter_match:
            pred_answer = letter_match.group(1)
        else:
            pred_answer = prediction[:200]
    
    # Get ground truth
    gt_answer = str(doc["answer"])
    
    # Use LLM Judge for evaluation
    is_correct = call_judge(
        question=doc["question"],
        groundtruth=gt_answer,
        modeloutput=pred_answer,
    )
    
    eval_result = {
        "index": doc["index"],
        "true_false": is_correct,
        "category": doc["category"],
        "answer": doc["answer"],
    }

    return {
        "eval_results": eval_result,
    }


def phyx_process_results(doc, results):
    """Process results using Azure TRAPI LLM Judge."""
    prediction = results[0].strip() if results else ""
    
    # Extract answer from prediction
    pred_answer = extract_boxed_answer(prediction)
    if pred_answer is None:
        pred_answer = prediction[:200]
    
    # Get ground truth
    gt_answer = str(doc["answer"])
    
    # Use LLM Judge for evaluation
    is_correct = call_judge(
        question=doc["question"],
        groundtruth=gt_answer,
        modeloutput=pred_answer,
    )
    
    eval_result = {
        "index": doc["index"],
        "true_false": is_correct,
        "category": doc["category"],
        "answer": doc["answer"],
    }

    return {
        "eval_results": eval_result,
    }


def phyx_aggregate_results(results):
    hit = [x["true_false"] for x in results]
    Overall_acc = np.mean(hit)
    return Overall_acc


# ============================================================================
# Visual CoT Versions
# ============================================================================

# Optics: 根据题目画光路图
OPTICS_GEN_PROMPT = (
    "Based on this optics problem, draw a light ray diagram that helps solve the problem. "
    "Show the paths of light rays, including incident rays, reflected rays, refracted rays, "
    "and any relevant angles or focal points as needed by the problem."
)

# Mechanics: 根据题目画受力分析图
MECHANICS_GEN_PROMPT = (
    "Based on this mechanics problem, draw a free body diagram (force analysis diagram) "
    "that helps solve the problem. "
    "Show all the forces acting on the object(s), including gravity, normal force, friction, "
    "tension, applied forces, etc., with arrows indicating direction and relative magnitude."
)


def phyx_doc_to_text_optics_cot(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for PhyX Optics task."""
    question = (
        "In addition to the original image, you are also given an auxiliary "
        "light ray diagram to help you solve the problem.\n\n"
        + doc["question"]
    )
    return f"[GEN_PROMPT]{OPTICS_GEN_PROMPT}[/GEN_PROMPT][QUESTION]{question}[/QUESTION]"


def phyx_doc_to_text_mechanics_cot(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for PhyX Mechanics task."""
    question = (
        "In addition to the original image, you are also given an auxiliary "
        "free body diagram (force analysis diagram) to help you solve the problem.\n\n"
        + doc["question"]
    )
    return f"[GEN_PROMPT]{MECHANICS_GEN_PROMPT}[/GEN_PROMPT][QUESTION]{question}[/QUESTION]"
