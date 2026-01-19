"""
Geometry3K Task Utilities
Evaluation for plane geometry problems from the Geometry3K dataset.
"""

import json
import time
from typing import Any, Dict, List, Optional

from PIL import Image

# Azure OpenAI client for Judge evaluation
_JUDGE_CLIENT = None
_JUDGE_DEPLOYMENT = None


def _get_judge_client():
    """Get or create Azure OpenAI client for Judge evaluation (singleton)"""
    global _JUDGE_CLIENT, _JUDGE_DEPLOYMENT
    if _JUDGE_CLIENT is None:
        import os

        from azure.identity import AzureCliCredential, get_bearer_token_provider
        from openai import AzureOpenAI

        endpoint = os.getenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://mcg-vision-flow-oai-eus2.openai.azure.com/",
        )
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        # Try API key first, fall back to Azure CLI credential
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if api_key:
            _JUDGE_CLIENT = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        else:
            # Use Azure CLI credential
            scope = os.getenv(
                "AZURE_OPENAI_SCOPE", "https://cognitiveservices.azure.com/.default"
            )
            token_provider = get_bearer_token_provider(AzureCliCredential(), scope)
            _JUDGE_CLIENT = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
            )
        _JUDGE_DEPLOYMENT = os.getenv("JUDGE_DEPLOYMENT", "gpt-5.1")
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


def geometry3k_doc_to_text_visual_cot(
    doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None
) -> str:
    """
    Get two-stage Visual Chain-of-Thought prompt for geometry3k task.

    Stage 1: Analyze the problem and draw auxiliary lines on the original diagram
    Stage 2: Solve the problem using both original diagram and the auxiliary diagram
    """
    problem = doc.get("problem", "")

    # Stage 1: Analyze problem and generate diagram with auxiliary constructions
    generation_prompt = f"""You are given a geometry problem with a diagram. Analyze the problem and create an enhanced version of the SAME diagram with auxiliary constructions added.

Problem: {problem}

Instructions:
1. KEEP all original elements exactly as they are (all points, lines, labels, and measurements)
2. Analyze what auxiliary constructions would help solve this problem
3. ADD auxiliary lines in a different color (e.g., red or dashed lines):
   - Perpendicular lines from center to chords
   - Extended lines if needed
   - Angle bisectors, midpoints, or other helpful constructions
4. Label any new points you add (use letters not already in the diagram)
5. The final diagram should look like the original with extra auxiliary lines drawn on top

Generate an enhanced diagram that preserves the original and adds helpful auxiliary constructions."""

    # Stage 2: Solve using both original and auxiliary diagram
    question_prompt = f"""{problem}

You are given TWO images:
1) ORIGINAL DIAGRAM: The geometry problem as given
2) AUXILIARY DIAGRAM: The same diagram with auxiliary constructions (extra lines) added to help solve the problem

Instructions:
1. Look at the auxiliary diagram to see what constructions were added
2. Use these auxiliary lines to identify key geometric relationships (perpendiculars, congruent segments, etc.)
3. Apply relevant theorems (Pythagorean theorem, chord properties, etc.)
4. Show your step-by-step solution with clear calculations
5. State the final numerical answer

Solve this problem step by step."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"


def geometry3k_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """
    Process geometry3k results with LLM Judge evaluation using Azure TRAPI.

    Uses GPT-5.1 as Judge (configurable via JUDGE_DEPLOYMENT env var).
    """
    result_text = results[0] if results else ""

    # Truncate to last 50000 characters to avoid token limit issues
    # Keep the end since answers are usually at the end
    MAX_RESULT_CHARS = 50000
    if len(result_text) > MAX_RESULT_CHARS:
        result_text = "...[truncated]...\n" + result_text[-MAX_RESULT_CHARS:]

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
