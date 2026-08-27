"""Geometry3K task utilities.

Evaluation for geometry problem solving from the Geometry3K dataset.
Paper: Inter-GPS (ACL 2021) - https://aclanthology.org/2021.acl-long.528/
"""

import re
from typing import Any

from PIL import Image


def geometry3k_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """Extract visual input from a Geometry3K document.

    Args:
        doc: Document containing 'images' field with PIL images.

    Returns:
        List containing the first image converted to RGB format.
    """
    images = doc.get("images", [])
    if images and len(images) > 0:
        img = images[0]
        if isinstance(img, Image.Image):
            return [img.convert("RGB")]
    return []


def geometry3k_doc_to_text(
    doc: dict[str, Any],
    lmms_eval_specific_kwargs: dict[str, Any] | None = None,
) -> str:
    """Generate text prompt from a Geometry3K document.

    Args:
        doc: Document containing 'problem' and 'choices' fields.
        lmms_eval_specific_kwargs: Optional kwargs with 'post_prompt' for
            customizing the response format instruction.

    Returns:
        Formatted prompt string with question and multiple choice options.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    problem = doc.get("problem", "")
    choices = doc.get("choices", [])

    # Build multiple choice options string
    options = ["A", "B", "C", "D"]
    choices_str = "\n".join(f"{opt}. {choice}" for opt, choice in zip(options, choices))

    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "\nAnswer with the option's letter from the given choices directly.",
    )

    return f"{problem}\n{choices_str}{post_prompt}"


def _extract_answer_letter(response: str, choices: list[str]) -> str:
    """Extract answer letter from model response.

    Tries multiple patterns to extract the answer letter (A/B/C/D)
    from the model's response.

    Args:
        response: Model's response string.
        choices: List of choice texts for content matching fallback.

    Returns:
        Extracted answer letter (A/B/C/D) or empty string if not found.
    """
    response = response.strip()
    all_choices = ["A", "B", "C", "D"]

    # Pattern 1: Answer: (A) or Answer: A
    patterns = [
        r"[Aa]nswer\s*[:=]\s*\(?([A-Da-d])\)?",
        r"\b([A-Da-d])\s*[.)\]]",  # A. or A) or A]
        r"^\s*\(?([A-Da-d])\)?$",  # Just the letter
        r"\b([A-Da-d])\b",  # Any standalone letter
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            # Return the last match (usually the final answer)
            for match in reversed(matches):
                letter = match.upper()
                if letter in all_choices:
                    return letter

    # Fallback: Check if response contains choice content
    if choices:
        for idx, choice in enumerate(choices):
            if choice.lower() in response.lower():
                return all_choices[idx]

    return ""


def geometry3k_process_results(
    doc: dict[str, Any],
    results: list[str],
) -> dict[str, float]:
    """Process Geometry3K results and compute accuracy.

    Args:
        doc: Document containing 'ground_truth' (A/B/C/D) and 'choices'.
        results: List containing model's response string.

    Returns:
        Dictionary with 'exact_match' score (1.0 for correct, 0.0 otherwise).
    """
    response = results[0] if results else ""
    choices = doc.get("choices", [])

    pred = _extract_answer_letter(response, choices)
    target = doc.get("ground_truth", "").strip().upper()

    if pred and pred == target:
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}
