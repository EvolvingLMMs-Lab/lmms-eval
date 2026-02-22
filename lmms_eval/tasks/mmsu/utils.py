"""
MMSU (Massive Multi-task Spoken Language Understanding) benchmark utilities.

This module provides evaluation functions for the MMSU benchmark, which
tests spoken language understanding and reasoning across 47 tasks.
"""

import random
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
                        elif isinstance(audio_file.metadata, dict) and "sample_rate" in audio_file.metadata:
                            sampling_rate = audio_file.metadata["sample_rate"]
                    elif hasattr(audio_file, "_desired_sample_rate") and audio_file._desired_sample_rate:
                        sampling_rate = audio_file._desired_sample_rate

                    audio_dict = {"array": audio_array, "sampling_rate": sampling_rate}
                    return [audio_dict]
                elif hasattr(audio_file, "decode"):
                    decoded_audio = audio_file.decode()
                    if isinstance(decoded_audio, dict):
                        return [decoded_audio]
                    elif hasattr(decoded_audio, "array") and hasattr(decoded_audio, "sampling_rate"):
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
                    elif hasattr(decoded_audio, "array") and hasattr(decoded_audio, "sampling_rate"):
                        return [
                            {
                                "array": decoded_audio.array,
                                "sampling_rate": decoded_audio.sampling_rate,
                            }
                        ]
                else:
                    if hasattr(audio_file, "array") and hasattr(audio_file, "sampling_rate"):
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
                return [{"array": audio_file.array, "sampling_rate": audio_file.sampling_rate}]
            except Exception as e:
                eval_logger.error(f"Error converting audio object: {e}")
                return []
        elif isinstance(audio_file, dict) and "array" in audio_file and "sampling_rate" in audio_file:
            return [audio_file]
        else:
            return [audio_file]
    else:
        eval_logger.warning(f"No audio file found in document. Available keys: {list(doc.keys())}")
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


def get_multi_choice_info(options):
    """Get index to answer mapping and all choices.

    Adapted from MMMU:
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54

    Args:
        options: List of option strings.

    Returns:
        Tuple of (index2ans dict, all_choices list).
    """
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """Parse the prediction from the generated response.

    Return the predicted index e.g., A, B, C, D.
    Adapted from MMMU:
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10

    Args:
        response: Model response string.
        all_choices: List of valid choices (e.g., ['A', 'B', 'C', 'D']).
        index2ans: Dict mapping choice to answer text.

    Returns:
        Predicted choice letter.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than
    # 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


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

    # Build options list from document
    options = [
        doc.get("choice_a", ""),
        doc.get("choice_b", ""),
        doc.get("choice_c", ""),
        doc.get("choice_d", ""),
    ]

    # Get choice info using MMMU pattern
    index2ans, all_choices = get_multi_choice_info(options)

    # Parse response using MMMU pattern
    pred = parse_multi_choice_response(response, all_choices, index2ans)

    ground_truth = doc.get("answer", "A")
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
        acc = (category_correct[cat] / category_total[cat]) * 100 if category_total[cat] > 0 else 0.0
        eval_logger.info(f"  {cat}: {acc:.2f}% ({int(category_correct[cat])}/{category_total[cat]})")

    eval_logger.info("-" * 60)
    eval_logger.info("Sub-category-wise Accuracy:")
    for subcat in sorted(subcategory_total.keys()):
        acc = (subcategory_correct[subcat] / subcategory_total[subcat]) * 100 if subcategory_total[subcat] > 0 else 0.0
        eval_logger.info(f"  {subcat}: {acc:.2f}% ({int(subcategory_correct[subcat])}/{subcategory_total[subcat]})")

    eval_logger.info("=" * 60)

    return overall_accuracy
