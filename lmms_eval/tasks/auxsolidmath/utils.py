"""
AuxSolidMath Task Utilities
Evaluation for solid geometry problems with auxiliary line construction.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

from PIL import Image

# Azure TRAPI client for Judge evaluation (reuse from uni_mmmu)
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


def auxsolidmath_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual input for auxsolidmath task (original diagram)"""
    original_image = doc.get("original_image")
    if original_image is not None:
        if isinstance(original_image, Image.Image):
            return [original_image]
    return []


def auxsolidmath_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """Get text prompt for auxsolidmath task"""
    question = doc.get("question", "")
    return f"""You are given a solid geometry problem with a 3D diagram.

Problem: {question}

Instructions:
1. First, carefully analyze the 3D diagram and identify what auxiliary lines (辅助线) need to be drawn to solve this problem. Common auxiliary constructions in solid geometry include:
   - Connecting points to form line segments
   - Drawing perpendiculars from a point to a plane or line
   - Finding midpoints and connecting them
   - Extending lines to find intersections
   - Drawing parallel lines through specific points
   - Constructing cross-sections

2. Clearly state which auxiliary lines you will draw and why they are helpful. For example:
   - "Connect point A to point B to form segment AB"
   - "Draw a perpendicular from point P to plane ABC, with foot H"
   - "Take the midpoint M of edge AB, connect M to C"
   - "Extend line DE to intersect plane ABC at point F"

3. After describing the auxiliary lines, provide a step-by-step solution using these auxiliary constructions.

4. Show all intermediate calculations and reasoning, including:
   - Distance calculations
   - Angle calculations
   - Volume/area calculations if needed

5. State the final answer clearly.

Please think step by step, starting with the auxiliary line construction."""


def auxsolidmath_doc_to_target(doc: Dict) -> str:
    """Get target answer for auxsolidmath task"""
    return doc.get("answer", "")


def auxsolidmath_doc_to_text_visual_cot(
    doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None
) -> str:
    """
    Get two-stage Visual Chain-of-Thought prompt for auxsolidmath task.

    Stage 1: Analyze the problem and draw auxiliary lines on the 3D diagram
    Stage 2: Solve the problem using both original diagram and the auxiliary diagram
    """
    question = doc.get("question", "")

    # Stage 1: Analyze problem and generate diagram with auxiliary constructions
    generation_prompt = f"""You are given a solid geometry problem with a 3D diagram. Analyze the problem and create an enhanced version of the SAME diagram with auxiliary constructions added.

Problem: {question}

Instructions:
1. KEEP all original elements exactly as they are (all points, edges, faces, labels)
2. Analyze what auxiliary constructions would help solve this problem
3. ADD auxiliary lines in a different color (e.g., red or dashed lines). Common auxiliary constructions include:
   - Perpendiculars from a point to a plane or line
   - Line segments connecting specific points
   - Midpoints of edges with connections
   - Extended lines to find intersections
   - Parallel lines through specific points
   - Cross-sections of the solid
4. Label any new points you add (use letters not already in the diagram)
5. The final diagram should look like the original 3D figure with extra auxiliary lines drawn on top

Generate an enhanced 3D diagram that preserves the original and adds helpful auxiliary constructions."""

    # Stage 2: Solve using both original and auxiliary diagram
    question_prompt = f"""You are given a solid geometry problem.

Problem: {question}

You are given TWO images:
1) ORIGINAL DIAGRAM: The 3D solid geometry figure as given
2) AUXILIARY DIAGRAM: The same figure with auxiliary constructions (extra lines) added to help solve the problem

Instructions:
1. Look at the auxiliary diagram to see what constructions were added (perpendiculars, connecting segments, midpoints, etc.)
2. Identify the geometric relationships established by these auxiliary lines
3. Use these constructions to set up your solution approach
4. Apply relevant theorems (Pythagorean theorem in 3D, properties of perpendiculars, volume formulas, etc.)
5. Show your step-by-step solution with clear calculations
6. State the final numerical answer

Solve this problem step by step using the auxiliary constructions."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"


def auxsolidmath_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """
    Process auxsolidmath results with LLM Judge evaluation using Azure TRAPI.

    Uses GPT-5.1 as Judge (configurable via JUDGE_DEPLOYMENT env var).
    """
    result_text = results[0] if results else ""

    # Truncate to last 50000 characters to avoid token limit issues
    # Keep the end since answers are usually at the end
    MAX_RESULT_CHARS = 50000
    if len(result_text) > MAX_RESULT_CHARS:
        result_text = "...[truncated]...\n" + result_text[-MAX_RESULT_CHARS:]

    # Default scores
    text_acc = 0.0

    # Get Judge client
    try:
        judge_client, judge_deployment = _get_judge_client()
    except Exception as e:
        print(f"Warning: Failed to initialize Judge client: {e}")
        return {
            "auxsolidmath_text_acc": text_acc,
        }

    # === Text Evaluation ===
    question = doc.get("question", "")
    gt_answer = doc.get("answer", "")
    gt_auxiliary = doc.get("auxiliary_line_description", "")

    if question and gt_answer and result_text:
        text_system = """You are a rigorous grader for solid geometry reasoning.
Given: problem statement (text), ground-truth answer, ground-truth auxiliary line description, and a candidate solution text.

Decide two things:
  (i) is the reasoning rigorous (no major logical gaps or false claims),
  (ii) is the final conclusion correct.

For CALCULATION problems: conclusion correctness means the final NUMERIC result matches the ground-truth
(ignore formatting/units; radicals/π are okay if numerically equivalent).

For PROVING problems: conclusion correctness means the claim is indeed established (may differ in steps but must be valid).

Only if (rigorous AND correct) → text_ok=1, else 0.

Output MUST be a compact JSON: {"reasoning_rigorous":0|1,"conclusion_correct":0|1,"text_ok":0|1,"text_reason":"<short>"}"""

        text_user = f"""Problem:
{question}

Ground truth auxiliary lines:
{gt_auxiliary}

Ground truth answer:
{gt_answer}

Candidate solution:
{result_text}

Evaluate the candidate. Output JSON only."""

        for attempt in range(3):
            try:
                response = judge_client.chat.completions.create(
                    model=judge_deployment,
                    messages=[
                        {"role": "system", "content": text_system},
                        {"role": "user", "content": text_user},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                response_text = response.choices[0].message.content
                result_json = _find_first_json_substring(response_text)
                if result_json:
                    data = json.loads(result_json)
                    text_acc = 1.0 if int(data.get("text_ok", 0)) == 1 else 0.0
                break
            except Exception as e:
                print(f"Text Judge attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(5)

    return {
        "auxsolidmath_text_acc": text_acc,
    }


def auxsolidmath_aggregate(results: List[Optional[float]]) -> float:
    """Aggregate results"""
    vals = [v for v in results if v is not None]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)
