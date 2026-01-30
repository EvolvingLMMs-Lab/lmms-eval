"""
MMSU (Massive Multi-task Spoken Language Understanding) benchmark utilities.

This module provides evaluation functions for the MMSU benchmark, which
tests spoken language understanding and reasoning across 47 tasks.
"""

from collections import defaultdict

import numpy as np
from loguru import logger as eval_logger


def mmsu_doc_to_audio(doc):
    """Extract audio from document.

    Args:
        doc: Document dictionary containing audio data.

    Returns:
        List containing the audio data.
    """
    audio_file = doc.get("audio")

    if audio_file:
        if str(type(audio_file).__name__) == "AudioDecoder":
            try:
                if hasattr(audio_file, "get_all_samples"):
                    decoded_audio = audio_file.get_all_samples()

                    if hasattr(decoded_audio, "samples"):
                        audio_array = decoded_audio.samples
                    elif hasattr(decoded_audio, "array"):
                        audio_array = decoded_audio.array
                    elif hasattr(decoded_audio, "data"):
                        audio_array = decoded_audio.data
                    else:
                        audio_array = decoded_audio

                    if hasattr(audio_array, "cpu") and hasattr(audio_array, "numpy"):
                        audio_array = audio_array.cpu().numpy()
                    elif hasattr(audio_array, "detach"):
                        audio_array = audio_array.detach().cpu().numpy()
                    elif str(type(audio_array).__name__) == "Tensor":
                        try:
                            audio_array = audio_array.cpu().numpy()
                        except Exception:
                            try:
                                audio_array = audio_array.detach().cpu().numpy()
                            except Exception:
                                audio_array = np.array(audio_array)

                    sampling_rate = 16000
                    if hasattr(decoded_audio, "sample_rate"):
                        sampling_rate = decoded_audio.sample_rate
                    elif hasattr(decoded_audio, "sampling_rate"):
                        sampling_rate = decoded_audio.sampling_rate
                    elif hasattr(audio_file, "metadata") and audio_file.metadata:
                        if hasattr(audio_file.metadata, "sample_rate"):
                            sampling_rate = audio_file.metadata.sample_rate
                        elif (
                            isinstance(audio_file.metadata, dict)
                            and "sample_rate" in audio_file.metadata
                        ):
                            sampling_rate = audio_file.metadata["sample_rate"]
                    elif (
                        hasattr(audio_file, "_desired_sample_rate")
                        and audio_file._desired_sample_rate
                    ):
                        sampling_rate = audio_file._desired_sample_rate

                    audio_dict = {"array": audio_array, "sampling_rate": sampling_rate}
                    return [audio_dict]
                elif hasattr(audio_file, "decode"):
                    decoded_audio = audio_file.decode()
                    if isinstance(decoded_audio, dict):
                        return [decoded_audio]
                    elif hasattr(decoded_audio, "array") and hasattr(
                        decoded_audio, "sampling_rate"
                    ):
                        return [
                            {
                                "array": decoded_audio.array,
                                "sampling_rate": decoded_audio.sampling_rate,
                            }
                        ]
                elif hasattr(audio_file, "__call__"):
                    decoded_audio = audio_file()
                    if isinstance(decoded_audio, dict):
                        return [decoded_audio]
                    elif hasattr(decoded_audio, "array") and hasattr(
                        decoded_audio, "sampling_rate"
                    ):
                        return [
                            {
                                "array": decoded_audio.array,
                                "sampling_rate": decoded_audio.sampling_rate,
                            }
                        ]
                else:
                    if hasattr(audio_file, "array") and hasattr(
                        audio_file, "sampling_rate"
                    ):
                        return [
                            {
                                "array": audio_file.array,
                                "sampling_rate": audio_file.sampling_rate,
                            }
                        ]
                    else:
                        return []
            except Exception as e:
                eval_logger.error(f"Error converting AudioDecoder object: {e}")
                return []
        elif hasattr(audio_file, "array") and hasattr(audio_file, "sampling_rate"):
            try:
                return [
                    {"array": audio_file.array, "sampling_rate": audio_file.sampling_rate}
                ]
            except Exception as e:
                eval_logger.error(f"Error converting audio object: {e}")
                return []
        elif (
            isinstance(audio_file, dict)
            and "array" in audio_file
            and "sampling_rate" in audio_file
        ):
            return [audio_file]
        else:
            return [audio_file]
    else:
        eval_logger.warning(
            f"No audio file found in document. Available keys: {list(doc.keys())}"
        )
        return []


def mmsu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Generate prompt for the audio model.

    Constructs a multiple-choice question prompt from the document.

    Args:
        doc: Document dictionary with question and choices.
        lmms_eval_specific_kwargs: Optional kwargs with pre/post prompts.

    Returns:
        Formatted prompt string.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc.get("question", "")
    choice_a = doc.get("choice_a", "")
    choice_b = doc.get("choice_b", "")
    choice_c = doc.get("choice_c", "")
    choice_d = doc.get("choice_d", "")

    prompt = f"""{pre_prompt}Listen to the audio and answer the following question.

Question: {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer with the letter (A, B, C, or D) only.{post_prompt}"""

    return prompt


def extract_answer(response):
    """Extract the answer letter from model response.

    Uses multiple patterns to extract A/B/C/D from various response formats.

    Args:
        response: Model response string.

    Returns:
        Extracted answer letter (A/B/C/D) or None if not found.
    """
    if not response:
        return None

    response = response.lower().strip()

    # Handle common prefixes
    if response.startswith("<1>") or response.startswith("<2>"):
        response = response[3:].strip()

    # Common answer patterns
    patterns = [
        "the answer is [CHOICE]",
        "the correct answer is [CHOICE]",
        "answer: [CHOICE]",
        "answer is [CHOICE]",
        "[CHOICE] is the answer",
        "[CHOICE] is correct",
        "would choose [CHOICE]",
        "would select [CHOICE]",
        "option [CHOICE]",
        "choice [CHOICE]",
        "is [CHOICE].",
        "is [CHOICE],",
        "is [CHOICE]:",
        "is [CHOICE])",
        "is ([CHOICE])",
        "**[CHOICE]**",
        "**[CHOICE].",
        "**[CHOICE])",
        '"[CHOICE]"',
        "'[CHOICE]'",
        "([CHOICE])",
        " [CHOICE].",
        " [CHOICE],",
        " [CHOICE]:",
        " [CHOICE])",
    ]

    for pattern in patterns:
        for choice in ["a", "b", "c", "d"]:
            if pattern.replace("[CHOICE]", choice) in response:
                return choice.upper()

    # Check if response is just a single letter
    for choice in ["a", "b", "c", "d"]:
        if response == choice:
            return choice.upper()
        # Check if response starts with the letter followed by punctuation
        for punc in [".", ",", ":", ")", " ", "\n"]:
            if response.startswith(choice + punc):
                return choice.upper()

    # Check for letter at the end of response
    response_clean = response.rstrip(".,!?;: \n")
    if response_clean and response_clean[-1] in ["a", "b", "c", "d"]:
        # Make sure it's not part of a word
        if len(response_clean) == 1 or not response_clean[-2].isalpha():
            return response_clean[-1].upper()

    return None


def mmsu_process_results(doc, results):
    """Process evaluation results for a single sample.

    Args:
        doc: Document dictionary with ground truth.
        results: List of model predictions.

    Returns:
        Dictionary with accuracy score and metadata.
    """
    response = results[0] if results else ""
    if response is None:
        response = ""

    ground_truth = doc.get("answer", "A")
    pred = extract_answer(response)

    if pred is None:
        pred = "A"  # Default fallback

    score = 1.0 if pred == ground_truth else 0.0

    return {
        "accuracy": {
            "score": score,
            "category": doc.get("category", "Unknown"),
            "sub_category": doc.get("sub_category", "Unknown"),
            "task_name": doc.get("task_name", "Unknown"),
        },
    }


def mmsu_aggregate_results(results):
    """Aggregate accuracy results across all samples.

    Computes overall accuracy and per-category breakdowns.

    Args:
        results: List of result dictionaries from process_results.

    Returns:
        Overall accuracy as a float (0-100).
    """
    if not results:
        return 0.0

    total_correct = 0
    total_count = len(results)

    # Category-wise tracking
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    # Sub-category tracking
    subcategory_correct = defaultdict(int)
    subcategory_total = defaultdict(int)

    # Task-wise tracking
    task_correct = defaultdict(int)
    task_total = defaultdict(int)

    for result in results:
        score = result.get("score", 0)
        category = result.get("category", "Unknown")
        sub_category = result.get("sub_category", "Unknown")
        task_name = result.get("task_name", "Unknown")

        total_correct += score

        category_correct[category] += score
        category_total[category] += 1

        subcategory_correct[sub_category] += score
        subcategory_total[sub_category] += 1

        task_correct[task_name] += score
        task_total[task_name] += 1

    overall_accuracy = (total_correct / total_count) * 100 if total_count > 0 else 0.0

    # Log detailed results
    eval_logger.info("=" * 60)
    eval_logger.info(f"MMSU Overall Accuracy: {overall_accuracy:.2f}%")
    eval_logger.info(f"Total samples: {total_count}")
    eval_logger.info("=" * 60)

    eval_logger.info("Category-wise Accuracy:")
    for cat in sorted(category_total.keys()):
        acc = (
            (category_correct[cat] / category_total[cat]) * 100
            if category_total[cat] > 0
            else 0.0
        )
        eval_logger.info(f"  {cat}: {acc:.2f}% ({int(category_correct[cat])}/{category_total[cat]})")

    eval_logger.info("-" * 60)
    eval_logger.info("Sub-category-wise Accuracy:")
    for subcat in sorted(subcategory_total.keys()):
        acc = (
            (subcategory_correct[subcat] / subcategory_total[subcat]) * 100
            if subcategory_total[subcat] > 0
            else 0.0
        )
        eval_logger.info(
            f"  {subcat}: {acc:.2f}% ({int(subcategory_correct[subcat])}/{subcategory_total[subcat]})"
        )

    eval_logger.info("=" * 60)

    return overall_accuracy


