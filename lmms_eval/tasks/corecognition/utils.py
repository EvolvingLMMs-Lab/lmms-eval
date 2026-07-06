"""Utility functions for CoreCognition benchmark.

Implements hybrid answer matching: template matching first (MCQ and YORN),
with optional LLM judge fallback when enabled via task config (see stare/utils.py).
"""

import logging
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

eval_logger = logging.getLogger("lmms-eval")

# Load task config for use_lmms_judge: merge default template with corecognition.yaml (strip !function lines so yaml.safe_load works)
_default_template_path = Path(__file__).parent / "_default_template_yaml"
_corecognition_config_path = Path(__file__).parent / "corecognition.yaml"


def _load_yaml_stripped(path: Path) -> dict:
    with open(path, "r") as f:
        raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
    return yaml.safe_load("".join(safe_data)) or {}


_corecognition_config = _load_yaml_stripped(_default_template_path)
_corecognition_config.update(_load_yaml_stripped(_corecognition_config_path))

_use_lmms_judge = _corecognition_config.get("metadata", {}).get("use_lmms_judge", False)

# Lazy pipeline singleton for LLM judge
_pipeline = None
_pipeline_lock = threading.Lock()


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:  # double-check
                from lmms_eval.verifiers import VerificationPipeline
                from lmms_eval.verifiers.extractors import StripReasoningExtractor
                from lmms_eval.verifiers.openai import OpenAIVerifier

                eval_logger.info("Using LMMS judge server for CoreCognition task.")
                API_TYPE = os.getenv("API_TYPE", "openai").lower()
                DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME") or os.getenv("OPENAI_API_MODEL", "gpt-4o")

                def _build_prompt(question, prediction, ground_truth, **kwargs):
                    return CORECOGNITION_JUDGE_PROMPT.format(
                        response=prediction,
                        answer=ground_truth,
                    )

                _pipeline = VerificationPipeline(
                    extractors=[StripReasoningExtractor()],
                    verifier=OpenAIVerifier(
                        model=DEPLOYMENT_NAME,
                        api_type=API_TYPE,
                        custom_prompt=_build_prompt,
                        response_format="binary",
                    ),
                )
    return _pipeline


# Judge prompt for binary correct/incorrect (same style as stare create_test_prompt)
CORECOGNITION_JUDGE_PROMPT = """You are judging whether a model's response matches the correct answer for a single-choice or yes/no question.
Consider the response and the correct answer. If the response indicates the same choice as the answer (possibly with extra wording), output Correct.
Otherwise output Incorrect. Output only one word: Correct or Incorrect.

Response: {response}
Answer: {answer}
Correct_or_not:"""


# Answer options for template matching
OPTIONS_MCQ = ["A", "B", "C", "D", "E", "F"]
OPTIONS_YORN = ["YES", "NO"]


# ============================================================================
# Template Matching (MCQ and YORN)
# ============================================================================


def _rm_model_special(pred: str) -> str:
    """Remove model special tokens from the prediction."""
    pred = str(pred).strip()
    if ">\n\n" in pred:
        pred = pred.split(">\n\n")[-1]
    if "**\n\n" in pred:
        pred = pred.split("**\n\n")[-1]
    pred = pred.replace(r"\[ \boxed{", "")
    pred = pred.replace("} \\]", "")
    pred = pred.replace("<|end_of_sentence|>", "")
    pred = pred.replace("</s>", "")
    pred = pred.replace("<CONCLUSION>", "")
    pred = pred.replace("</CONCLUSION>", "")
    pred = pred.replace("Falcon: ", "")
    return pred.strip()


def _template_match(pred: str, question_type: str) -> str:
    """Template matching for answer extraction (MCQ and YORN).
    Returns extracted option or 'Fail' if no valid match.
    """
    pred = _rm_model_special(pred)
    valid_options = OPTIONS_YORN if question_type == "YORN" else OPTIONS_MCQ

    if len(pred.split()) >= 2:
        patterns = [
            r"^(yes|no|\w)(,|\.|\;| |\n|\*)+",
            r"[\n\*\{]+(yes|no|\w)(,|\.|\;| |\n|\*|\})+",
            r"(yes|no|\w) is the correct answer",
            r"answer is[\:\;\*\n ]*(yes|no|\w)",
            r"answer[\:\;\*\n ]*(yes|no|\w)",
            r"choice is[\:\;\*\n ]*(yes|no|\w)",
            r"choice[\:\;\*\n ]*(yes|no|\w)",
            r"option is[\:\;\*\n ]*(yes|no|\w)",
            r"Assistant[\:\;\*\n ]*(yes|no|\w)",
        ]
        for pattern in patterns:
            match = re.search(pattern, pred, re.IGNORECASE)
            if match:
                res = match.group(1).upper()
                if res in valid_options:
                    return res
    else:
        first = re.split(r",|\.| |\:|\;|\n", pred)[0].upper() if pred else ""
        if first in valid_options:
            return first

    return "Fail"


def corecognition_doc_to_visual(doc: dict[str, Any]) -> list:
    """Extract image from document.
    Args:
        doc: Document containing images field
    Returns:
        List containing the RGB image
    """
    img = doc.get("images") or (doc.get("image_paths") or [None])[0]
    if img is None:
        return []
    if hasattr(img, "convert"):
        return [img.convert("RGB")]
    return [img]


def corecognition_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, str]] = None) -> str:
    """Format question text with optional prompt additions.
    Args:
        doc: Document containing prompt field
        lmms_eval_specific_kwargs: Optional pre/post prompts
    Returns:
        Formatted question string
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    prompt = doc.get("prompt") or doc.get("question")

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt")

    return f"{pre_prompt or ''}{prompt or ''}{post_prompt or ''}"


def process_docs_stage_sensorimotor(dataset):
    """Filter dataset to only include Stage Sensorimotor samples."""
    return dataset.filter(lambda x: x["stage"] == "Stage Sensorimotor")


def process_docs_stage_concrete_operational(dataset):
    """Filter dataset to only include Stage Concrete Operational samples."""
    return dataset.filter(lambda x: x["stage"] == "Stage Concrete Operational")


def process_docs_stage_formal_operational(dataset):
    """Filter dataset to only include Stage Formal Operational samples."""
    return dataset.filter(lambda x: x["stage"] == "Stage Formal Operational")


def corecognition_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    """Process model results and compute accuracy.
    Uses template matching (MCQ/YORN) first; when template match fails and
    use_lmms_judge is True, calls LLM judge (reference: stare/utils.py).
    Args:
        doc: Document containing ground truth answer and type (MC/TF)
        results: List containing model prediction
    Returns:
        Dictionary with accuracy metric
    """
    pred = results[0] if results else ""
    if not isinstance(pred, str):
        pred = str(pred)
    ground_truth = str(doc["answer"]).strip()
    concept = doc.get("concept", "unknown")
    # MC -> MCQ, TF -> YORN (yes/no)
    qtype = (doc.get("type") or "MC").strip().upper()
    question_type = "YORN" if qtype == "TF" else "MCQ"

    matched = _template_match(pred, question_type)
    if matched != "Fail":
        gt_normalized = ground_truth.upper().strip()
        is_correct = matched == gt_normalized
    else:
        # Template match failed: try LLM judge if enabled, else direct comparison
        if _use_lmms_judge:
            try:
                pipeline = _get_pipeline()
                result = pipeline(question=pred, prediction=pred, ground_truth=ground_truth)
                if result.metadata.get("judge_failed"):
                    # OpenAIVerifier swallows judge-call errors and flags them
                    # here; fall back to direct comparison rather than counting
                    # an infra failure as a wrong answer.
                    pred_normalized = _rm_model_special(pred).upper().strip()
                    gt_normalized = ground_truth.upper().strip()
                    is_correct = pred_normalized == gt_normalized
                else:
                    is_correct = result.is_correct
            except Exception as e:
                # Pipeline construction (import/config) failed.
                eval_logger.debug("CoreCognition LLM judge failed, falling back to direct comparison: %s", e)
                pred_normalized = _rm_model_special(pred).upper().strip()
                gt_normalized = ground_truth.upper().strip()
                is_correct = pred_normalized == gt_normalized
        else:
            pred_normalized = _rm_model_special(pred).upper().strip()
            gt_normalized = ground_truth.upper().strip()
            is_correct = pred_normalized == gt_normalized

    return {
        "accuracy": float(is_correct),
        "accuracy_by_concept": {"concept": concept, "correct": is_correct},
    }


def _extract_answer(pred: str) -> str:
    """Extract answer from model prediction with aggressive cleanup.
    Args:
        pred: Raw model prediction
    Returns:
        Extracted answer in uppercase
    """
    pred = pred.strip()

    patterns = [
        r"^(yes|no|[a-d])(\.|\,|\;| |\n|\*)",
        r"[\n\*]+(yes|no|[a-d])(\.|\,|\;| |\n|\*)",
        r"(yes|no|[a-d]) is the correct answer",
        r"answer is[\:\;\*\n ]*(yes|no|[a-d])",
        r"answer[\:\;\*\n ]*(yes|no|[a-d])",
        r"option is[\:\;\*\n ]*(yes|no|[a-d])",
        r"choice is[\:\;\*\n ]*(yes|no|[a-d])",
    ]

    for pattern in patterns:
        match = re.search(pattern, pred, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    cleaned = re.split(r"[,\.\:\;\n\s]+", pred)[0].strip()
    if cleaned:
        return cleaned.upper()

    return pred.upper()


def corecognition_aggregate_by_concept(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate results by concept.
    Args:
        results: List of result dictionaries with concept and correctness
    Returns:
        Dictionary mapping concept names to accuracy scores
    """
    concept_correct: dict[str, int] = defaultdict(int)
    concept_total: dict[str, int] = defaultdict(int)

    for result in results:
        concept = result["concept"]
        correct = result["correct"]

        concept_total[concept] += 1
        if correct:
            concept_correct[concept] += 1

    concept_accuracy = {}
    for concept in concept_total:
        accuracy = concept_correct[concept] / concept_total[concept]
        concept_accuracy[concept] = accuracy

    total_correct = sum(concept_correct.values())
    total = sum(concept_total.values())
    concept_accuracy["overall"] = total_correct / total if total > 0 else 0.0

    return concept_accuracy
