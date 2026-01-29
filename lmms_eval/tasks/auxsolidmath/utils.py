"""
AuxSolidMath Task Utilities for lmms-eval.

Evaluation functions for solid geometry problems with auxiliary line construction.
Paper: https://arxiv.org/abs/2510.11020
Dataset: https://huggingface.co/datasets/shasha/AuxSolidMath
"""

import re
from typing import Any, Dict, List, Optional

from PIL import Image


def _extract_answer_from_response(response: str) -> Optional[str]:
    """Extract the final answer from model response."""
    if not response:
        return None

    patterns = [
        r"(?:final answer|answer is|the answer is|答案是|答案为|结果是|结果为)[:\s]*([^\n.,]+)",
        r"\\boxed\{([^}]+)\}",
        r"=\s*([^\n.,=]+)$",
        r"(?:therefore|hence|thus|so)[,:\s]*(?:the answer is)?[:\s]*([^\n.,]+)",
    ]

    response_lower = response.lower()
    lines = response.strip().split("\n")

    for pattern in patterns:
        match = re.search(pattern, response_lower, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    if lines:
        last_line = lines[-1].strip()
        if len(last_line) < 50:
            return last_line

    return None


def _normalize_math_answer(answer: str) -> str:
    """Normalize mathematical answer for comparison."""
    if not answer:
        return ""

    answer = answer.strip()
    answer = re.sub(r"\s+", "", answer)
    answer = answer.replace("\\sqrt", "sqrt")
    answer = answer.replace("\\frac", "frac")
    answer = answer.replace("\\pi", "pi")
    answer = answer.replace("\\arccos", "arccos")
    answer = answer.replace("\\arcsin", "arcsin")
    answer = answer.replace("\\arctan", "arctan")
    answer = answer.replace("\\cos", "cos")
    answer = answer.replace("\\sin", "sin")
    answer = answer.replace("\\tan", "tan")
    answer = re.sub(r"[°度]", "", answer)
    answer = answer.lower()

    return answer


def _compare_math_answers(pred: str, gt: str, tolerance: float = 0.01) -> bool:
    """Compare two mathematical answers with tolerance for numerical values."""
    if not pred or not gt:
        return False

    pred_norm = _normalize_math_answer(pred)
    gt_norm = _normalize_math_answer(gt)

    if pred_norm == gt_norm:
        return True

    try:
        pred_val = float(eval(pred_norm.replace("sqrt", "**0.5").replace("pi", "3.14159265358979")))
        gt_val = float(eval(gt_norm.replace("sqrt", "**0.5").replace("pi", "3.14159265358979")))
        if abs(pred_val - gt_val) < tolerance:
            return True
        if gt_val != 0 and abs((pred_val - gt_val) / gt_val) < tolerance:
            return True
    except (SyntaxError, NameError, ValueError, ZeroDivisionError):
        pass

    return pred_norm in gt_norm or gt_norm in pred_norm


def auxsolidmath_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    """Extract visual input (original diagram) from document."""
    original_image = doc.get("original_image")
    if original_image is not None:
        if isinstance(original_image, Image.Image):
            return [original_image.convert("RGB")]
    return []


def auxsolidmath_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """Build prompt for solid geometry problem."""
    question = doc.get("question", "")
    return f"""You are given a solid geometry problem with a 3D diagram.

Problem: {question}

Instructions:
1. Carefully analyze the 3D diagram and identify what auxiliary lines need to be drawn.
   Common auxiliary constructions include:
   - Connecting points to form line segments
   - Drawing perpendiculars from a point to a plane or line
   - Finding midpoints and connecting them
   - Extending lines to find intersections
   - Drawing parallel lines through specific points
   - Constructing cross-sections

2. Clearly state which auxiliary lines you will draw and why.

3. Provide a step-by-step solution using these auxiliary constructions.

4. Show all intermediate calculations including:
   - Distance calculations
   - Angle calculations
   - Volume/area calculations if needed

5. State the final answer clearly.

Please think step by step, starting with the auxiliary line construction."""


def auxsolidmath_doc_to_target(doc: Dict[str, Any]) -> str:
    """Get target answer from document."""
    return doc.get("answer", "")


def auxsolidmath_process_results(
    doc: Dict[str, Any],
    results: List[str],
) -> Dict[str, Any]:
    """Process results and compute accuracy using string matching."""
    response = results[0] if results else ""
    gt_answer = doc.get("answer", "")

    extracted = _extract_answer_from_response(response)
    is_correct = _compare_math_answers(extracted, gt_answer) if extracted and gt_answer else False

    return {
        "auxsolidmath_accuracy": 1.0 if is_correct else 0.0,
        "submission": {
            "question_id": doc.get("id", ""),
            "question": doc.get("question", ""),
            "gt_answer": gt_answer,
            "gt_auxiliary": doc.get("auxiliary_line_description", ""),
            "prediction": response,
            "extracted_answer": extracted,
            "is_correct": is_correct,
        },
    }


def auxsolidmath_aggregate_accuracy(results: List[float]) -> float:
    """Aggregate accuracy scores."""
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return 0.0
    return sum(valid_results) / len(valid_results)
