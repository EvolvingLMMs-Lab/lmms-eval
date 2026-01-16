"""
Geometry3K Task Utilities
Evaluation for plane geometry problems from the Geometry3K dataset.
"""

import json
import time
from typing import Any, Dict, List, Optional

from PIL import Image

# Azure TRAPI client for Judge evaluation (reuse from auxsolidmath)
_JUDGE_CLIENT = None
_JUDGE_DEPLOYMENT = None


def _get_judge_client():
    """Get or create Azure OpenAI client for Judge evaluation (singleton)"""
    global _JUDGE_CLIENT, _JUDGE_DEPLOYMENT
    if _JUDGE_CLIENT is None:
        import os
        from azure.identity import (
            AzureCliCredential,
            ChainedTokenCredential,
            ManagedIdentityCredential,
            get_bearer_token_provider,
        )
        from openai import AzureOpenAI

        scope = os.getenv("TRAPI_SCOPE", "api://trapi/.default")
        api_version = os.getenv("TRAPI_API_VERSION", "2024-10-21")
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
        # Use GPT-5.1 as Judge (stronger than GPT-4o)
        _JUDGE_DEPLOYMENT = os.getenv("JUDGE_DEPLOYMENT", "gpt-5.1_2025-11-13")
    return _JUDGE_CLIENT, _JUDGE_DEPLOYMENT


def _find_first_json_substring(text: str) -> Optional[str]:
    """Extract first JSON object from text"""
    if not text:
        return None
    start_index = text.find("{")
    if start_index == -1:
        return None

    brace_depth, in_string, is_escaped = 0, False, False
    for i in range(start_index, len(text)):
        char = text[i]
        if char == '"' and not is_escaped:
            in_string = not in_string
        if not in_string:
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    return text[start_index : i + 1]
        is_escaped = char == "\\" and not is_escaped
    return None


def geometry3k_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual input for geometry3k task"""
    images = doc.get("images", [])
    if images:
        # images is a list, return the first image
        if isinstance(images, list) and len(images) > 0:
            img = images[0]
            if isinstance(img, Image.Image):
                return [img]
    return []


def geometry3k_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """Get text prompt for geometry3k task"""
    problem = doc.get("problem", "")

    # Problem already contains <image> tag, so we use it directly
    return f"""{problem}

Instructions:
1. Carefully analyze the geometry diagram shown above.
2. Read the problem statement and identify what needs to be found.
3. Show your step-by-step solution with clear reasoning.
4. Include all intermediate calculations.
5. State the final answer clearly at the end.

Please solve this problem step by step."""


def geometry3k_doc_to_target(doc: Dict) -> str:
    """Get target answer for geometry3k task"""
    return doc.get("answer", "")


def geometry3k_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """
    Process geometry3k results with LLM Judge evaluation using Azure TRAPI.

    Uses GPT-5.1 as Judge (configurable via JUDGE_DEPLOYMENT env var).
    """
    result_text = results[0] if results else ""

    # Default scores
    accuracy = 0.0

    # Get Judge client
    try:
        judge_client, judge_deployment = _get_judge_client()
    except Exception as e:
        print(f"Warning: Failed to initialize Judge client: {e}")
        return {
            "geometry3k_accuracy": accuracy,
        }

    # === Answer Evaluation ===
    problem = doc.get("problem", "")
    gt_answer = doc.get("answer", "")

    if problem and gt_answer and result_text:
        judge_system = """You are a rigorous grader for plane geometry problems.
Given: problem statement (text with diagram), ground-truth answer, and a candidate solution text.

Evaluate if the candidate's final answer is mathematically equivalent to the ground-truth answer.

Important considerations:
- Numeric answers: Check if values match (allow minor rounding differences, e.g., 9.2 vs 9.19)
- Algebraic expressions: Check mathematical equivalence (e.g., "2√221" = "2*sqrt(221)")
- Fractions: Check equivalence (e.g., "1/2" = "0.5")
- LaTeX formatting: Ignore formatting differences (e.g., "\frac{1}{2}" = "1/2")
- Angle measures: Check numeric equivalence
- Units: Ignore unit differences if values match

Only mark as correct (answer_correct=1) if the final answer is mathematically equivalent.
Mark as incorrect (answer_correct=0) if:
- The answer is wrong
- No clear final answer is provided
- The answer is ambiguous

Output MUST be a compact JSON: {"answer_correct":0|1,"reason":"<short explanation>"}"""

        judge_user = f"""Problem:
{problem}

Ground truth answer:
{gt_answer}

Candidate solution:
{result_text}

Evaluate if the candidate's final answer matches the ground truth. Output JSON only."""

        for attempt in range(3):
            try:
                response = judge_client.chat.completions.create(
                    model=judge_deployment,
                    messages=[
                        {"role": "system", "content": judge_system},
                        {"role": "user", "content": judge_user},
                    ],
                    temperature=0.0,
                    max_completion_tokens=512,
                )
                response_text = response.choices[0].message.content
                result_json = _find_first_json_substring(response_text)
                if result_json:
                    data = json.loads(result_json)
                    accuracy = 1.0 if int(data.get("answer_correct", 0)) == 1 else 0.0
                break
            except Exception as e:
                print(f"Judge attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(5)

    return {
        "geometry3k_accuracy": accuracy,
    }


def geometry3k_aggregate(results: List[Optional[float]]) -> float:
    """Aggregate results"""
    vals = [v for v in results if v is not None]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)
