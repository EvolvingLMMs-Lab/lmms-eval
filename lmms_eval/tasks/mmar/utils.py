"""
Utility functions for the MMAR (Massive Multi-disciplinary Audio Reasoning) benchmark.

MMAR evaluates deep reasoning capabilities of Audio-Language Models across
1,000 meticulously curated audio-question-answer triplets spanning:
- 4 reasoning layers: Signal, Perception, Semantic, Cultural
- 7 audio modalities: Sound, Music, Speech, and their combinations
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger

DEFAULT_PRE_PROMPT = ""
DEFAULT_POST_PROMPT = "The best answer is:"


def mmar_doc_to_audio(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract audio from the document.

    The dataset has an 'audio' column with the Audio feature that contains
    'array' (waveform) and 'sampling_rate'.

    Args:
        doc: Single dataset example.

    Returns:
        List containing the audio dict with 'array' and 'sampling_rate'.
    """
    audio = doc.get("audio")
    if audio is None:
        eval_logger.warning(f"No audio found for sample {doc.get('id', 'unknown')}")
        return []
    return [audio]


def mmar_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Build the prompt for MMAR audio reasoning.

    Args:
        doc: Single dataset example with 'question' and 'choices'.
        lmms_eval_specific_kwargs: Model-specific prompt configuration.

    Returns:
        Formatted prompt string.
    """
    pre_prompt = DEFAULT_PRE_PROMPT
    post_prompt = DEFAULT_POST_PROMPT
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", pre_prompt)
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", post_prompt)

    # Build the instruction
    instruction = "Listen to the audio and answer the following multiple-choice question. " "Respond with only the letter (A, B, C, or D) of the correct option.\n"

    # Get question and choices
    question = doc["question"]
    choices = doc["choices"]  # Already formatted as ["A. choice1", "B. choice2", ...]

    # Build prompt
    prompt_parts = [
        pre_prompt,
        instruction,
        question,
        "\n",
        "\n".join(choices),
        "\n",
        post_prompt,
    ]
    return "".join(prompt_parts).strip()


def mmar_doc_to_target(doc: Dict[str, Any]) -> str:
    """
    Extract the ground truth answer letter.

    Args:
        doc: Single dataset example.

    Returns:
        Answer letter (A, B, C, or D).
    """
    return str(doc["answer"]).strip().upper()


def parse_multi_choice_response(response: Optional[str], all_choices: List[str]) -> str:
    """
    Parse the model's response to extract the answer letter.

    Args:
        response: Raw model output string.
        all_choices: Valid choice labels, e.g., ["A", "B", "C", "D"].

    Returns:
        Parsed answer letter (uppercased). Falls back to first choice if no match.
    """
    response = response or ""

    # Remove common answer prefixes
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
        "Based on the audio",
        "Correct answer",
        "\u261e",
        "<|im_end|>",
    ]
    for prefix in answer_prefixes:
        response = response.replace(prefix, "")

    response = response.strip()
    # Remove punctuation and special characters
    response = re.sub(r"[.,:!\"'`;\\/?`~@#\$%\^&\*\(\)\[\]\{\}\\|<>\n]", " ", response)
    tokens = response.split()

    # Find first valid choice letter
    for token in tokens:
        token_upper = token.upper()
        if token_upper in all_choices:
            return token_upper

    # Fallback to first choice
    return all_choices[0] if all_choices else "A"


def mmar_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Process model results and compare with ground truth.

    Args:
        doc: Single dataset example.
        results: List containing the model's response.

    Returns:
        Dict with metric results including score and metadata.
    """
    pred = results[0] if results else ""
    all_choices = ["A", "B", "C", "D"]

    # Parse model response
    parsed_answer = parse_multi_choice_response(pred, all_choices)
    gt_answer = mmar_doc_to_target(doc)

    # Calculate score
    score = 1.0 if parsed_answer == gt_answer else 0.0

    return {
        "mmar_accuracy": {
            "id": doc.get("id", ""),
            "category": doc.get("category", ""),
            "subcategory": doc.get("subcategory", ""),
            "modality": doc.get("modality", ""),
            "prediction": parsed_answer,
            "ground_truth": gt_answer,
            "raw_response": pred[:200] if pred else "",  # Truncate for logging
            "score": score,
        }
    }


def mmar_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """
    Aggregate results and compute overall accuracy.

    Also logs per-category and per-modality breakdowns.

    Args:
        results: List of result dicts from process_results.

    Returns:
        Overall accuracy as a percentage (0-100).
    """
    if not results:
        eval_logger.warning("No results to aggregate")
        return 0.0

    # Aggregate by category
    category_scores = defaultdict(list)
    modality_scores = defaultdict(list)
    total_score = 0.0

    for result in results:
        score = float(result.get("score", 0.0))
        category = result.get("category", "Unknown")
        modality = result.get("modality", "Unknown")

        category_scores[category].append(score)
        modality_scores[modality].append(score)
        total_score += score

    # Calculate overall accuracy
    overall_accuracy = (total_score / len(results)) * 100.0

    # Log category breakdown
    eval_logger.info("=" * 50)
    eval_logger.info("MMAR Results by Category:")
    for category in sorted(category_scores.keys()):
        scores = category_scores[category]
        cat_accuracy = (sum(scores) / len(scores)) * 100.0
        eval_logger.info(f"  {category}: {cat_accuracy:.2f}% ({len(scores)} samples)")

    # Log modality breakdown
    eval_logger.info("-" * 50)
    eval_logger.info("MMAR Results by Modality:")
    for modality in sorted(modality_scores.keys()):
        scores = modality_scores[modality]
        mod_accuracy = (sum(scores) / len(scores)) * 100.0
        eval_logger.info(f"  {modality}: {mod_accuracy:.2f}% ({len(scores)} samples)")

    eval_logger.info("=" * 50)
    eval_logger.info(f"MMAR Overall Accuracy: {overall_accuracy:.2f}% ({len(results)} samples)")
    eval_logger.info("=" * 50)

    return overall_accuracy
